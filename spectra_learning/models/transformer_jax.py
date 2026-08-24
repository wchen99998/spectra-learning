from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import torch
from flax import nnx

from spectra_learning.models.common_jax import (
    Array,
    LayerNorm,
    Linear,
    RMSNorm,
    scaled_dot_product_attention,
    silu,
)


def _xavier_normal(rngs: nnx.Rngs, shape: tuple[int, int]) -> Array:
    out_features, in_features = shape
    return rngs.params.normal(shape, dtype=jnp.float32) * math.sqrt(
        2.0 / (in_features + out_features)
    )


class Attention(nnx.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        *,
        n_kv_heads: int | None = None,
        compute_dtype: object = jnp.float32,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        rngs = nnx.Rngs(0) if rngs is None else rngs
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.head_dim = dim // n_heads
        self.q_size = self.n_heads * self.head_dim
        self.kv_size = self.n_kv_heads * self.head_dim
        self.wqkv = Linear(
            dim,
            self.q_size + 2 * self.kv_size,
            bias=False,
            compute_dtype=compute_dtype,
            init="zeros",
            rngs=rngs,
        )
        self.wo = Linear(dim, dim, bias=False, compute_dtype=compute_dtype, rngs=rngs)
        self.wqkv.weight[: self.q_size] = _xavier_normal(
            rngs,
            (self.q_size, dim),
        )
        self.wqkv.weight[self.q_size : self.q_size + self.kv_size] = _xavier_normal(
            rngs,
            (self.kv_size, dim),
        )
        self.wqkv.weight[self.q_size + self.kv_size :] = _xavier_normal(
            rngs,
            (self.kv_size, dim),
        )

    def __call__(self, x: Array, *, attn_mask: Array | None = None) -> Array:
        bsz, seqlen, _ = x.shape
        qkv = self.wqkv(x)
        xq, xk, xv = jnp.split(
            qkv,
            [self.q_size, self.q_size + self.kv_size],
            axis=-1,
        )
        xq = xq.reshape(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.reshape(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.reshape(bsz, seqlen, self.n_kv_heads, self.head_dim)
        q = jnp.swapaxes(xq.astype(xv.dtype), 1, 2)
        k = jnp.swapaxes(xk.astype(xv.dtype), 1, 2)
        v = jnp.swapaxes(xv, 1, 2)
        if self.n_kv_heads != self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k = jnp.repeat(k, rep, axis=1)
            v = jnp.repeat(v, rep, axis=1)
        attn = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn = jnp.swapaxes(attn, 1, 2).reshape(bsz, seqlen, self.dim)
        return self.wo(attn)

    def load_torch_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
    ) -> None:
        self.wqkv.load_torch_state_dict(state_dict, f"{prefix}.wqkv")
        self.wo.load_torch_state_dict(state_dict, f"{prefix}.wo")


class CrossAttention(nnx.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        *,
        max_sequence_length: int,
        rope_base: float = 10_000.0,
        norm_eps: float = 1e-5,
        compute_dtype: object = jnp.float32,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        rngs = nnx.Rngs(0) if rngs is None else rngs
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert self.head_dim % 2 == 0
        self.wq = Linear(
            dim,
            dim,
            bias=False,
            compute_dtype=compute_dtype,
            rngs=rngs,
        )
        self.wkv = Linear(
            dim,
            2 * dim,
            bias=False,
            compute_dtype=compute_dtype,
            rngs=rngs,
        )
        self.wo = Linear(
            dim,
            dim,
            bias=False,
            compute_dtype=compute_dtype,
            rngs=rngs,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=norm_eps, affine=False)
        self.k_norm = RMSNorm(self.head_dim, eps=norm_eps, affine=False)
        inv_freq = 1.0 / (
            rope_base
            ** (
                jnp.arange(0, self.head_dim, 2, dtype=jnp.float32)
                / self.head_dim
            )
        )
        frequencies = (
            jnp.arange(max_sequence_length, dtype=jnp.float32)[:, None]
            * inv_freq[None, :]
        )
        self.rope_cos = jnp.cos(frequencies)
        self.rope_sin = jnp.sin(frequencies)

    def _apply_rope(self, tensor: Array, positions: Array) -> Array:
        cos = self.rope_cos[positions][:, None].astype(tensor.dtype)
        sin = self.rope_sin[positions][:, None].astype(tensor.dtype)
        even = tensor[..., 0::2]
        odd = tensor[..., 1::2]
        return jnp.stack(
            (even * cos - odd * sin, even * sin + odd * cos),
            axis=-1,
        ).reshape(tensor.shape)

    def __call__(
        self,
        query: Array,
        memory: Array,
        query_positions: Array,
        memory_positions: Array,
        memory_mask: Array,
    ) -> Array:
        batch_size, target_len, _ = query.shape
        memory_len = memory.shape[1]
        q = self.wq(query).reshape(
            batch_size,
            target_len,
            self.n_heads,
            self.head_dim,
        )
        k, v = jnp.split(self.wkv(memory), 2, axis=-1)
        k = k.reshape(batch_size, memory_len, self.n_heads, self.head_dim)
        v = v.reshape(batch_size, memory_len, self.n_heads, self.head_dim)
        q = self.q_norm(jnp.swapaxes(q, 1, 2))
        k = self.k_norm(jnp.swapaxes(k, 1, 2))
        q = self._apply_rope(q, query_positions)
        k = self._apply_rope(k, memory_positions)
        v = jnp.swapaxes(v, 1, 2)
        attended = scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=memory_mask[:, None, None, :],
        )
        attended = jnp.swapaxes(attended, 1, 2).reshape(
            batch_size,
            target_len,
            self.dim,
        )
        return self.wo(attended)

    def load_torch_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
    ) -> None:
        self.wq.load_torch_state_dict(state_dict, f"{prefix}.wq")
        self.wkv.load_torch_state_dict(state_dict, f"{prefix}.wkv")
        self.wo.load_torch_state_dict(state_dict, f"{prefix}.wo")


class FeedForward(nnx.Module):
    def __init__(
        self,
        dim: int,
        *,
        hidden_dim: int | None = None,
        compute_dtype: object = jnp.float32,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        rngs = nnx.Rngs(0) if rngs is None else rngs
        hidden_dim = hidden_dim or int((4 * dim) * 2 / 3)
        hidden_dim = 4 * math.ceil(hidden_dim / 4)
        self.w1 = Linear(
            dim,
            hidden_dim,
            bias=False,
            compute_dtype=compute_dtype,
            init="trunc_normal_fan_in",
            rngs=rngs,
        )
        self.w2 = Linear(
            hidden_dim,
            dim,
            bias=False,
            compute_dtype=compute_dtype,
            init="trunc_normal_fan_in",
            rngs=rngs,
        )

    def __call__(self, x: Array) -> Array:
        return self.w2(silu(self.w1(x)))

    def load_torch_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
    ) -> None:
        self.w1.load_torch_state_dict(state_dict, f"{prefix}.w1")
        self.w2.load_torch_state_dict(state_dict, f"{prefix}.w2")


class SwiGLUFeedForward(nnx.Module):
    def __init__(
        self,
        dim: int,
        *,
        hidden_dim: int | None = None,
        compute_dtype: object = jnp.float32,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        rngs = nnx.Rngs(0) if rngs is None else rngs
        hidden_dim = hidden_dim or 4 * dim
        hidden_dim = 4 * math.ceil(hidden_dim / 4)
        self.fc1 = Linear(
            dim,
            hidden_dim,
            bias=False,
            compute_dtype=compute_dtype,
            init="trunc_normal_fan_in",
            rngs=rngs,
        )
        self.fc2 = Linear(
            dim,
            hidden_dim,
            bias=False,
            compute_dtype=compute_dtype,
            init="trunc_normal_fan_in",
            rngs=rngs,
        )
        self.fc3 = Linear(
            hidden_dim,
            dim,
            bias=False,
            compute_dtype=compute_dtype,
            init="zeros",
            rngs=rngs,
        )

    def __call__(self, x: Array) -> Array:
        return self.fc3(silu(self.fc1(x)) * self.fc2(x))

    def load_torch_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
    ) -> None:
        self.fc1.load_torch_state_dict(state_dict, f"{prefix}.fc1")
        self.fc2.load_torch_state_dict(state_dict, f"{prefix}.fc2")
        self.fc3.load_torch_state_dict(state_dict, f"{prefix}.fc3")


class CrossAttentionBlock(nnx.Module):
    def __init__(
        self,
        *,
        dim: int,
        n_heads: int,
        norm_eps: float,
        hidden_dim: int,
        max_sequence_length: int,
        compute_dtype: object = jnp.float32,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        rngs = nnx.Rngs(0) if rngs is None else rngs
        self.attention = CrossAttention(
            dim,
            n_heads,
            max_sequence_length=max_sequence_length,
            norm_eps=norm_eps,
            compute_dtype=compute_dtype,
            rngs=rngs,
        )
        self.feed_forward = SwiGLUFeedForward(
            dim,
            hidden_dim=hidden_dim,
            compute_dtype=compute_dtype,
            rngs=rngs,
        )
        self.query_norm = RMSNorm(dim, eps=norm_eps)
        self.memory_norm = RMSNorm(dim, eps=norm_eps)
        self.attention_post_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_post_norm = RMSNorm(dim, eps=norm_eps)

    def __call__(
        self,
        query: Array,
        memory: Array,
        query_positions: Array,
        memory_positions: Array,
        query_mask: Array,
        memory_mask: Array,
    ) -> Array:
        attention_update = self.attention(
            self.query_norm(query),
            self.memory_norm(memory),
            query_positions,
            memory_positions,
            memory_mask,
        )
        query = query + self.attention_post_norm(attention_update)
        ffn_update = self.feed_forward(self.ffn_norm(query))
        query = query + self.ffn_post_norm(ffn_update)
        return query * query_mask[..., None].astype(query.dtype)

    def load_torch_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
    ) -> None:
        self.attention.load_torch_state_dict(state_dict, f"{prefix}.attention")
        self.feed_forward.load_torch_state_dict(
            state_dict,
            f"{prefix}.feed_forward",
        )
        self.query_norm.load_torch_state_dict(state_dict, f"{prefix}.query_norm")
        self.memory_norm.load_torch_state_dict(state_dict, f"{prefix}.memory_norm")
        self.attention_post_norm.load_torch_state_dict(
            state_dict,
            f"{prefix}.attention_post_norm",
        )
        self.ffn_norm.load_torch_state_dict(state_dict, f"{prefix}.ffn_norm")
        self.ffn_post_norm.load_torch_state_dict(
            state_dict,
            f"{prefix}.ffn_post_norm",
        )


class TransformerBlock(nnx.Module):
    def __init__(
        self,
        *,
        dim: int,
        n_heads: int,
        n_kv_heads: int | None,
        norm_eps: float,
        hidden_dim: int | None,
        compute_dtype: object = jnp.float32,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        rngs = nnx.Rngs(0) if rngs is None else rngs
        self.attention = Attention(
            dim,
            n_heads,
            n_kv_heads=n_kv_heads,
            compute_dtype=compute_dtype,
            rngs=rngs,
        )
        self.feed_forward = FeedForward(
            dim,
            hidden_dim=hidden_dim,
            compute_dtype=compute_dtype,
            rngs=rngs,
        )
        self.attention_norm = LayerNorm(dim, eps=norm_eps)
        self.ffn_norm = LayerNorm(dim, eps=norm_eps)

    def __call__(self, x: Array, *, attn_mask: Array | None = None) -> Array:
        h = x + self.attention(self.attention_norm(x), attn_mask=attn_mask)
        return h + self.feed_forward(self.ffn_norm(h))

    def load_torch_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
    ) -> None:
        self.attention.load_torch_state_dict(state_dict, f"{prefix}.attention")
        self.feed_forward.load_torch_state_dict(state_dict, f"{prefix}.feed_forward")
        self.attention_norm.load_torch_state_dict(state_dict, f"{prefix}.attention_norm")
        self.ffn_norm.load_torch_state_dict(state_dict, f"{prefix}.ffn_norm")
