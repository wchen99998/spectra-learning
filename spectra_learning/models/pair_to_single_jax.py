"""Experimental direct pair-value readout for a compact joint encoder.

Pool before projecting: the expensive pair-width -> message-width projection
acts on N rows, rather than N**2 edges. This module returns a residual update;
it does not change the existing single attention or any training defaults.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx

from spectra_learning.models.common_jax import Linear, RMSNorm


class PairToSingleReadout(nnx.Module):
    """Multi-head row pooling of pair *values*, with a dense mean path.

    Each head averages its learned attention with uniform valid-edge weights.
    Thus sharp routing cannot turn off the entire dense value path. Heads use
    separate low-rank projections after pooling. A small, nonzero channel scale
    permits gradients into pairs from the first step, including for equal singles.

    Inputs must share the same compact token axis (including CLS if present).
    Invalid pairs must contain finite padding values. Pair direction is retained:
    row i summarizes outgoing relations z[i, j], without symmetrization.
    """

    def __init__(
        self,
        single_dim: int,
        pair_dim: int,
        *,
        num_heads: int = 4,
        head_dim: int = 32,
        residual_scale: float = 0.1,
        norm_eps: float = 1e-5,
        compute_dtype: object = jnp.float32,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        if min(single_dim, pair_dim, num_heads, head_dim) <= 0:
            raise ValueError("readout dimensions and head counts must be positive")
        rngs = nnx.Rngs(0) if rngs is None else rngs
        self.compute_dtype = compute_dtype
        self.precision = (jax.lax.Precision.DEFAULT if compute_dtype == jnp.bfloat16
                          else jax.lax.Precision.HIGHEST)
        self.pair_norm = RMSNorm(pair_dim, eps=norm_eps, affine=False)
        self.score = Linear(pair_dim, num_heads, bias=False,
                            compute_dtype=compute_dtype, rngs=rngs)
        self.head_weight = nnx.Param(
            rngs.params.normal((num_heads, pair_dim, head_dim), dtype=jnp.float32)
            / math.sqrt(pair_dim)
        )
        self.out = Linear(num_heads * head_dim, single_dim, bias=False,
                          compute_dtype=compute_dtype, rngs=rngs)
        self.output_norm = RMSNorm(single_dim, eps=norm_eps, affine=False)
        self.scale = nnx.Param(jnp.full((single_dim,), residual_scale, jnp.float32))

    def pool(self, pair: jax.Array, token_mask: jax.Array) -> jax.Array:
        normalized = self.pair_norm(pair).astype(self.compute_dtype)
        key_mask = token_mask[:, None, :, None]
        scores = self.score(normalized).astype(jnp.float32)
        scores = jnp.where(key_mask, scores, jnp.finfo(jnp.float32).min)
        attention = jax.nn.softmax(scores, axis=2) * key_mask
        uniform = key_mask / jnp.maximum(key_mask.sum(axis=2, keepdims=True), 1)
        weights = (0.5 * (attention + uniform)).astype(self.compute_dtype)
        pooled = jnp.einsum(
            "bijh,bijc->bihc", weights, normalized,
            precision=self.precision,
            preferred_element_type=jnp.float32,
        ).astype(self.compute_dtype)
        return pooled * token_mask[:, :, None, None]

    def __call__(self, pair: jax.Array, token_mask: jax.Array) -> jax.Array:
        return self.project(self.pool(pair, token_mask), token_mask)

    def project(self, pooled: jax.Array, token_mask: jax.Array) -> jax.Array:
        """Project pooled [batch, token, head, pair-channel] messages."""
        messages = jnp.einsum(
            "bihc,hcr->bihr", pooled, self.head_weight[...].astype(self.compute_dtype),
            precision=self.precision,
            preferred_element_type=jnp.float32,
        ).astype(self.compute_dtype)
        messages = messages.reshape(*messages.shape[:2], -1)
        update = self.output_norm(self.out(messages))
        return update * self.scale[...].astype(update.dtype) * token_mask[..., None]


class PairSummaryToSingleReadout(nnx.Module):
    """Reduce or gather pairs before normalization and low-rank projection.

    ``mean`` reads each valid outgoing row once and normalizes its mean.
    ``cls`` reads only the updated outgoing/incoming CLS edges of each token.
    Both normalize O(N) vectors, with no normalized N-by-N pair activation.
    The CLS variant requires one valid CLS position per nonempty spectrum;
    spectra without a valid CLS produce a zero update. Padding must be finite.
    """

    def __init__(
        self,
        single_dim: int,
        pair_dim: int,
        *,
        summary: str = "mean",
        message_dim: int = 128,
        residual_scale: float = 0.1,
        norm_eps: float = 1e-5,
        compute_dtype: object = jnp.float32,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        if summary not in {"mean", "cls"}:
            raise ValueError("summary must be 'mean' or 'cls'")
        if min(single_dim, pair_dim, message_dim) <= 0:
            raise ValueError("readout dimensions must be positive")
        rngs = nnx.Rngs(0) if rngs is None else rngs
        self.summary = summary
        self.compute_dtype = compute_dtype
        self.pair_norm = RMSNorm(pair_dim, eps=norm_eps, affine=False)
        self.value = Linear(pair_dim * (2 if summary == "cls" else 1), message_dim,
                            bias=False, compute_dtype=compute_dtype, rngs=rngs)
        self.out = Linear(message_dim, single_dim, bias=False,
                          compute_dtype=compute_dtype, rngs=rngs)
        self.output_norm = RMSNorm(single_dim, eps=norm_eps, affine=False)
        self.scale = nnx.Param(jnp.full((single_dim,), residual_scale, jnp.float32))

    def __call__(
        self,
        pair: jax.Array,
        token_mask: jax.Array,
        cls_mask: jax.Array | None = None,
    ) -> jax.Array:
        if self.summary == "mean":
            # A reduction allows XLA to fuse its broadcast/cast backward pass;
            # a GEMM formulation can materialize an N-by-N FP32 gradient.
            pooled = jnp.sum(jnp.where(token_mask[:, None, :, None],
                                      pair.astype(self.compute_dtype), 0),
                             axis=2, dtype=jnp.float32)
            pooled /= jnp.maximum(token_mask.sum(1)[:, None, None], 1)
            values = self.pair_norm(pooled)
            output_mask = token_mask
        else:
            if cls_mask is None:
                raise ValueError("CLS readout requires an explicit CLS mask")
            valid_cls = cls_mask & token_mask
            index = jnp.argmax(valid_cls, axis=1)
            batch = jnp.arange(pair.shape[0])
            # Gather first: normalization never materializes an N-by-N tensor.
            outgoing = self.pair_norm(pair[batch, :, index, :])
            incoming = self.pair_norm(pair[batch, index, :, :])
            values = jnp.concatenate((outgoing, incoming), axis=-1)
            output_mask = token_mask & valid_cls.any(1)[:, None]
        update = self.output_norm(self.out(self.value(values)))
        return update * self.scale[...].astype(update.dtype) * output_mask[..., None]
