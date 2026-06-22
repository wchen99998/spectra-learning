from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import torch
from flax import nnx

from spectra_learning.models.common_jax import Array, LayerNorm, Linear
from spectra_learning.models.pairmixer_jax import PairMixerBlock


class InducedPairState(NamedTuple):
    inducing: Array
    pair: Array
    assignment: Array


class TokenInducingAssignment(nnx.Module):
    def __init__(
        self,
        dim: int,
        *,
        compute_dtype: object = jnp.float32,
    ) -> None:
        self.dim = dim
        self.compute_dtype = compute_dtype
        self.matmul_precision = (
            jax.lax.Precision.DEFAULT if compute_dtype == jnp.bfloat16 else None
        )
        self.wq = Linear(dim, dim, bias=False, compute_dtype=compute_dtype)
        self.wk = Linear(dim, dim, bias=False, compute_dtype=compute_dtype)

    def __call__(self, token: Array, inducing: Array) -> Array:
        q = self.wq(token)
        k = self.wk(inducing)
        scores = jnp.einsum(
            "bid,bad->bia",
            q,
            k,
            precision=self.matmul_precision,
        ) / math.sqrt(self.dim)
        return jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(q.dtype)

    def load_torch_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
    ) -> None:
        self.wq.load_torch_state_dict(state_dict, f"{prefix}.wq")
        self.wk.load_torch_state_dict(state_dict, f"{prefix}.wk")


class CrossAttentionWithWeights(nnx.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        compute_dtype: object = jnp.float32,
    ) -> None:
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.compute_dtype = compute_dtype
        self.matmul_precision = (
            jax.lax.Precision.DEFAULT if compute_dtype == jnp.bfloat16 else None
        )
        self.wq = Linear(dim, dim, bias=False, compute_dtype=compute_dtype)
        self.wkv = Linear(dim, 2 * dim, bias=False, compute_dtype=compute_dtype)
        self.wo = Linear(dim, dim, bias=False, compute_dtype=compute_dtype)

    def __call__(
        self,
        query: Array,
        memory: Array,
        *,
        memory_mask: Array | None = None,
    ) -> tuple[Array, Array]:
        batch_size, query_len, _ = query.shape
        memory_len = memory.shape[1]
        q = self.wq(query).reshape(
            batch_size,
            query_len,
            self.num_heads,
            self.head_dim,
        )
        kv = self.wkv(memory).reshape(
            batch_size,
            memory_len,
            2,
            self.num_heads,
            self.head_dim,
        )
        k, v = jnp.moveaxis(kv, 2, 0)
        q = jnp.swapaxes(q, 1, 2)
        k = jnp.swapaxes(k, 1, 2)
        v = jnp.swapaxes(v, 1, 2)
        scores = jnp.einsum(
            "bhid,bhjd->bhij",
            q,
            k,
            precision=self.matmul_precision,
        ) / math.sqrt(self.head_dim)
        if memory_mask is not None:
            scores = jnp.where(
                memory_mask[:, None, None, :],
                scores,
                jnp.asarray(-jnp.inf, dtype=scores.dtype),
            )
        attn = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(v.dtype)
        out = jnp.einsum(
            "bhij,bhjd->bhid",
            attn,
            v,
            precision=self.matmul_precision,
        )
        out = jnp.swapaxes(out, 1, 2).reshape(batch_size, query_len, self.dim)
        return self.wo(out), jnp.mean(attn, axis=1)

    def load_torch_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
    ) -> None:
        self.wq.load_torch_state_dict(state_dict, f"{prefix}.wq")
        self.wkv.load_torch_state_dict(state_dict, f"{prefix}.wkv")
        self.wo.load_torch_state_dict(state_dict, f"{prefix}.wo")


class InducedPairBlock(nnx.Module):
    def __init__(
        self,
        *,
        single_dim: int,
        pair_dim: int,
        num_heads: int,
        attention_mlp_multiple: float,
        norm_eps: float,
        dropout: float,
        use_pair_bias_attention: bool = False,
        compute_dtype: object = jnp.float32,
    ) -> None:
        del dropout
        self.inducing_aggregate_norm = LayerNorm(single_dim, eps=norm_eps)
        self.single_aggregate_norm = LayerNorm(single_dim, eps=norm_eps)
        self.inducing_aggregate = CrossAttentionWithWeights(
            single_dim,
            num_heads,
            compute_dtype=compute_dtype,
        )
        self.latent_pair_mixer = PairMixerBlock(
            single_dim=single_dim,
            pair_dim=pair_dim,
            num_heads=num_heads,
            attention_mlp_multiple=attention_mlp_multiple,
            norm_eps=norm_eps,
            dropout=0.0,
            use_pair_bias_attention=use_pair_bias_attention,
            compute_dtype=compute_dtype,
        )
        self.single_update_norm = LayerNorm(single_dim, eps=norm_eps)
        self.inducing_update_norm = LayerNorm(single_dim, eps=norm_eps)
        self.single_update = CrossAttentionWithWeights(
            single_dim,
            num_heads,
            compute_dtype=compute_dtype,
        )

    def __call__(
        self,
        single: Array,
        state: InducedPairState,
        peak_mask: Array,
        token_mask: Array,
        *,
        deterministic: bool = True,
    ) -> tuple[Array, InducedPairState]:
        del peak_mask, deterministic
        inducing_update, _ = self.inducing_aggregate(
            self.inducing_aggregate_norm(state.inducing),
            self.single_aggregate_norm(single),
            memory_mask=token_mask,
        )
        inducing = state.inducing + inducing_update
        inducing_mask = jnp.ones(inducing.shape[:2], dtype=jnp.bool_)
        inducing, pair = self.latent_pair_mixer(
            inducing,
            state.pair,
            inducing_mask,
            inducing_mask,
        )
        single_update, assignment = self.single_update(
            self.single_update_norm(single),
            self.inducing_update_norm(inducing),
        )
        single = single + single_update
        assignment = assignment * token_mask[..., None].astype(assignment.dtype)
        return single, InducedPairState(inducing, pair, assignment)

    def load_torch_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
    ) -> None:
        self.inducing_aggregate_norm.load_torch_state_dict(
            state_dict,
            f"{prefix}.inducing_aggregate_norm",
        )
        self.single_aggregate_norm.load_torch_state_dict(
            state_dict,
            f"{prefix}.single_aggregate_norm",
        )
        self.inducing_aggregate.load_torch_state_dict(
            state_dict,
            f"{prefix}.inducing_aggregate",
        )
        self.latent_pair_mixer.load_torch_state_dict(
            state_dict,
            f"{prefix}.latent_pair_mixer",
        )
        self.single_update_norm.load_torch_state_dict(
            state_dict,
            f"{prefix}.single_update_norm",
        )
        self.inducing_update_norm.load_torch_state_dict(
            state_dict,
            f"{prefix}.inducing_update_norm",
        )
        self.single_update.load_torch_state_dict(state_dict, f"{prefix}.single_update")


def mask_induced_pair_assignment(state: InducedPairState, token_mask: Array) -> InducedPairState:
    return InducedPairState(
        state.inducing,
        state.pair,
        state.assignment * token_mask[..., None].astype(state.assignment.dtype),
    )


def induced_pair_batch_slice(state: InducedPairState, index) -> InducedPairState:
    return InducedPairState(
        state.inducing[index],
        state.pair[index],
        state.assignment[index],
    )


def induced_pair_slice_tokens(state: InducedPairState, token_count: int) -> InducedPairState:
    return InducedPairState(
        state.inducing,
        state.pair,
        state.assignment[..., :token_count, :],
    )


def induced_pair_expand_views(state: InducedPairState, num_views: int) -> InducedPairState:
    return InducedPairState(
        jnp.broadcast_to(
            state.inducing[:, None],
            (state.inducing.shape[0], num_views, *state.inducing.shape[1:]),
        ),
        jnp.broadcast_to(
            state.pair[:, None],
            (state.pair.shape[0], num_views, *state.pair.shape[1:]),
        ),
        jnp.broadcast_to(
            state.assignment[:, None],
            (state.assignment.shape[0], num_views, *state.assignment.shape[1:]),
        ),
    )


def induced_pair_flatten_views(state: InducedPairState) -> InducedPairState:
    batch_size, num_views = state.inducing.shape[:2]
    return InducedPairState(
        state.inducing.reshape(batch_size * num_views, *state.inducing.shape[2:]),
        state.pair.reshape(batch_size * num_views, *state.pair.shape[2:]),
        state.assignment.reshape(batch_size * num_views, *state.assignment.shape[2:]),
    )


def induced_pair_unflatten_views(
    state: InducedPairState,
    batch_size: int,
    num_views: int,
) -> InducedPairState:
    return InducedPairState(
        state.inducing.reshape(batch_size, num_views, *state.inducing.shape[1:]),
        state.pair.reshape(batch_size, num_views, *state.pair.shape[1:]),
        state.assignment.reshape(batch_size, num_views, *state.assignment.shape[1:]),
    )


def induced_pair_to_dense_pair(state: InducedPairState) -> Array:
    left = jnp.einsum("...ia,...abd->...ibd", state.assignment, state.pair)
    return jnp.einsum("...ibd,...jb->...ijd", left, state.assignment)


def induced_pair_distogram_logits(state: InducedPairState, head) -> Array:
    sym_pair = state.pair + jnp.swapaxes(state.pair, -3, -2)
    latent_logits = head(sym_pair)
    left = jnp.einsum("...ia,...abk->...ibk", state.assignment, latent_logits)
    return jnp.einsum("...ibk,...jb->...ijk", left, state.assignment)
