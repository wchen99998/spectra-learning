"""PyTorch implementation of the transformer blocks used by the MAE BERT model."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def precompute_freqs_cis(
    dim: int,
    end: int,
    *,
    theta: float = 10000.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = torch.arange(0, dim, 2, device=device, dtype=torch.float32)
    freqs = 1.0 / (theta ** (freqs / dim))
    t = torch.arange(end, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return freqs.cos().to(dtype=dtype), freqs.sin().to(dtype=dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    orig_shape = x.shape
    x_pairs = x.reshape(*orig_shape[:-1], -1, 2)
    x1 = x_pairs[..., 0]
    x2 = x_pairs[..., 1]
    rotated = torch.stack((-x2, x1), dim=-1)
    return rotated.reshape(orig_shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = freqs_cos[None, :, None, :].repeat_interleave(2, dim=-1)
    sin = freqs_sin[None, :, None, :].repeat_interleave(2, dim=-1)

    q_rot = _rotate_half(xq)
    k_rot = _rotate_half(xk)

    xq_out = (xq * cos) + (q_rot * sin)
    xk_out = (xk * cos) + (k_rot * sin)
    return xq_out, xk_out


@dataclass
class ModelArgs:
    dim: int = 288
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: int | None = None
    output_channels: int = 1024
    hidden_dim: int | None = None
    multiple_of: int = 32
    norm_eps: float = 1e-5
    w_init_scale: float = 1.0
    depth_scaled_init: bool = False
    mlp_type: str = "swiglu"
    causal: bool = False
    rope_theta: float = 10000.0
    use_rotary_embeddings: bool = True
    input_dim: int | None = None


_ACTIVATIONS: dict[str, nn.Module] = {
    "swiglu": nn.SiLU(),
    "geglu": nn.GELU(),
    "glu": nn.Sigmoid(),
    "swish": nn.SiLU(),
}


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        *,
        n_kv_heads: int | None = None,
        causal: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.head_dim = self.dim // self.n_heads
        self.causal = causal

        out_features = (self.n_heads + 2 * self.n_kv_heads) * self.head_dim
        self.wqkv = nn.Linear(self.dim, out_features, bias=False)
        self.wo = nn.Linear(self.dim, self.dim, bias=False)

        nn.init.xavier_normal_(self.wqkv.weight)
        nn.init.xavier_normal_(self.wo.weight)

    def forward(
        self,
        x: torch.Tensor,
        *,
        freqs_cos: torch.Tensor | None = None,
        freqs_sin: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del attention_bias  # Kept for API parity with the JAX module.
        bsz, seqlen, _ = x.shape
        qkv = self.wqkv(x)

        q_size = self.n_heads * self.head_dim
        kv_size = self.n_kv_heads * self.head_dim

        xq, xk, xv = torch.split(qkv, [q_size, kv_size, kv_size], dim=-1)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        if freqs_cos is not None and freqs_sin is not None:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        q = xq.transpose(1, 2)
        k = xk.transpose(1, 2)
        v = xv.transpose(1, 2)

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=self.causal,
            enable_gqa=self.n_kv_heads != self.n_heads,
        )
        attn = attn.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        return self.wo(attn)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        multiple_of: int,
        hidden_dim: int | None = None,
        mlp_type: str = "swiglu",
        w_init_scale: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.mlp_type = mlp_type

        hidden_dim = hidden_dim or int((4 * dim) * 2 / 3)
        hidden_dim = multiple_of * math.ceil(hidden_dim / multiple_of)
        self.hidden_dim = hidden_dim

        self.activation = _ACTIVATIONS[mlp_type]
        self.uses_gating = mlp_type in {"swiglu", "geglu", "glu"}

        if self.uses_gating:
            self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=False)
        else:
            self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

        nn.init.trunc_normal_(self.w2.weight, std=w_init_scale / math.sqrt(dim))
        if self.uses_gating:
            nn.init.trunc_normal_(self.w12.weight, std=w_init_scale / math.sqrt(dim))
        else:
            nn.init.trunc_normal_(self.w1.weight, std=w_init_scale / math.sqrt(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.uses_gating:
            w1, w3 = self.w12(x).chunk(2, dim=-1)
            hidden = self.activation(w1) * w3
        else:
            hidden = self.activation(self.w1(x))
        return self.w2(hidden)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        n_heads: int,
        n_kv_heads: int | None,
        causal: bool,
        norm_eps: float,
        mlp_type: str,
        multiple_of: int,
        hidden_dim: int | None,
        w_init_scale: float,
        use_rotary_embeddings: bool,
    ):
        super().__init__()
        self.use_rotary_embeddings = use_rotary_embeddings
        self.attention = Attention(dim, n_heads, n_kv_heads=n_kv_heads, causal=causal)
        self.feed_forward = FeedForward(
            dim,
            multiple_of=multiple_of,
            hidden_dim=hidden_dim,
            mlp_type=mlp_type,
            w_init_scale=w_init_scale,
        )
        self.attention_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = nn.RMSNorm(dim, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        *,
        freqs_cos: torch.Tensor | None,
        freqs_sin: torch.Tensor | None,
        attention_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.use_rotary_embeddings:
            freqs_cos = None
            freqs_sin = None

        h = x + self.attention(
            self.attention_norm(x),
            freqs_cos=freqs_cos,
            freqs_sin=freqs_sin,
            attention_bias=attention_bias,
        )
        return h + self.feed_forward(self.ffn_norm(h))
