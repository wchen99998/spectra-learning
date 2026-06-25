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
