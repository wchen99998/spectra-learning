"""Pair->single residual bridges for the joint fastmixer encoder.

The attention-logit bias passes one scalar per edge and head from pairs to
singles, so pair content reaches singles only through attention routing. In
``training_mode="dual"`` every fastmixer block adds two direct value paths
after its own updates (docs/pair_single_dual_20261001.md):

* mean:  u_i = g_m * RMSNorm(W_o W_v RMSNorm(mean_j z_ij))
* gated: m_i = mean_j sigmoid(G zn_ij) * (V sn_j) + P mean_j zn_ij,
         u_i = g_g * RMSNorm(W m_i),  with zn = RMSNorm(z), sn = RMSNorm(s)

The per-channel scales g start at 0.1, so the pair stream receives gradients
from singles at the first step. Inputs share one compact token axis; invalid
tokens are excluded from every mean and receive a zero update.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from spectra_learning.models.common_jax import Linear, RMSNorm

Array = jax.Array


def _masked_row_mean(values: Array, token_mask: Array, dtype: object) -> Array:
    # A reduction lets XLA fuse the broadcast/cast backward pass; a GEMM
    # formulation can materialize an N-by-N FP32 gradient.
    total = jnp.sum(
        jnp.where(token_mask[:, None, :, None], values.astype(dtype), 0),
        axis=2,
        dtype=jnp.float32,
    )
    return total / jnp.maximum(token_mask.sum(1), 1)[:, None, None]


class PairMeanMessage(nnx.Module):
    """Normalized mean of each token's outgoing pair row, low-rank projected."""

    def __init__(
        self,
        single_dim: int,
        pair_dim: int,
        *,
        rank: int,
        residual_scale: float,
        norm_eps: float,
        compute_dtype: object,
        rngs: nnx.Rngs,
    ) -> None:
        self.compute_dtype = compute_dtype
        self.pair_norm = RMSNorm(pair_dim, eps=norm_eps, affine=False)
        self.value = Linear(pair_dim, rank, bias=False,
                            compute_dtype=compute_dtype, rngs=rngs)
        self.out = Linear(rank, single_dim, bias=False,
                          compute_dtype=compute_dtype, rngs=rngs)
        self.output_norm = RMSNorm(single_dim, eps=norm_eps, affine=False)
        self.scale = nnx.Param(jnp.full((single_dim,), residual_scale, jnp.float32))

    def __call__(self, pair: Array, token_mask: Array) -> Array:
        pooled = self.pair_norm(_masked_row_mean(pair, token_mask, self.compute_dtype))
        update = self.output_norm(self.out(self.value(pooled)))
        return update * self.scale[...].astype(update.dtype) * token_mask[..., None]


class EdgeGatedMessage(nnx.Module):
    """Channel-wise pair-gated neighbor messages plus mean pair values.

    Each edge gates ``rank`` message channels rather than one attention
    logit, and the second term carries pair content even when neighbor
    singles are identical.
    """

    def __init__(
        self,
        single_dim: int,
        pair_dim: int,
        *,
        rank: int,
        residual_scale: float,
        norm_eps: float,
        compute_dtype: object,
        rngs: nnx.Rngs,
    ) -> None:
        self.compute_dtype = compute_dtype
        self.pair_norm = RMSNorm(pair_dim, eps=norm_eps, affine=False)
        self.single_norm = RMSNorm(single_dim, eps=norm_eps, affine=False)
        self.gate = Linear(pair_dim, rank, compute_dtype=compute_dtype, rngs=rngs)
        self.value = Linear(single_dim, rank, bias=False,
                            compute_dtype=compute_dtype, rngs=rngs)
        self.pair_value = Linear(pair_dim, rank, bias=False,
                                 compute_dtype=compute_dtype, rngs=rngs)
        self.out = Linear(rank, single_dim, bias=False,
                          compute_dtype=compute_dtype, rngs=rngs)
        self.output_norm = RMSNorm(single_dim, eps=norm_eps, affine=False)
        self.scale = nnx.Param(jnp.full((single_dim,), residual_scale, jnp.float32))

    def __call__(self, single: Array, pair: Array, token_mask: Array) -> Array:
        keys = token_mask[:, None, :, None]
        count = jnp.maximum(token_mask.sum(1), 1)[:, None, None].astype(jnp.float32)
        normalized = self.pair_norm(pair).astype(self.compute_dtype)
        gates = jax.nn.sigmoid(self.gate(normalized).astype(jnp.float32))
        values = self.value(self.single_norm(single)).astype(jnp.float32)
        gated = jnp.sum(jnp.where(keys, gates * values[:, None], 0), axis=2) / count
        mean_pair = jnp.sum(
            jnp.where(keys, normalized, 0), axis=2, dtype=jnp.float32,
        ) / count
        message = gated + self.pair_value(
            mean_pair.astype(self.compute_dtype)
        ).astype(jnp.float32)
        update = self.output_norm(self.out(message.astype(self.compute_dtype)))
        return update * self.scale[...].astype(update.dtype) * token_mask[..., None]


class PairToSingleBridge(nnx.Module):
    """Sum of the mean and edge-gated pair->single residual updates."""

    def __init__(
        self,
        single_dim: int,
        pair_dim: int,
        *,
        rank: int = 64,
        residual_scale: float = 0.1,
        norm_eps: float = 1e-5,
        compute_dtype: object = jnp.float32,
        rngs: nnx.Rngs,
    ) -> None:
        kwargs = dict(rank=rank, residual_scale=residual_scale, norm_eps=norm_eps,
                      compute_dtype=compute_dtype, rngs=rngs)
        self.mean = PairMeanMessage(single_dim, pair_dim, **kwargs)
        self.gated = EdgeGatedMessage(single_dim, pair_dim, **kwargs)

    def __call__(self, single: Array, pair: Array, token_mask: Array) -> Array:
        update = self.mean(pair, token_mask) + self.gated(single, pair, token_mask)
        return update.astype(single.dtype)
