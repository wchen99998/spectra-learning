from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)


def create_visible_block_mask(visible_mask: torch.Tensor) -> BlockMask:
    """Create a BlockMask from a [B, N] boolean visibility mask."""
    B, N = visible_mask.shape

    def mask_mod(b, h, q_idx, kv_idx):
        return visible_mask[b, q_idx] & visible_mask[b, kv_idx]

    return create_block_mask(
        mask_mod, B=B, H=None, Q_LEN=N, KV_LEN=N, device=visible_mask.device
    )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    rotated = torch.empty_like(x)
    rotated[..., ::2] = -x[..., 1::2]
    rotated[..., 1::2] = x[..., ::2]
    return rotated


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_rot = _rotate_half(xq)
    k_rot = _rotate_half(xk)

    xq_out = (xq * freqs_cos) + (q_rot * freqs_sin)
    xk_out = (xk * freqs_cos) + (k_rot * freqs_sin)
    return xq_out, xk_out


_ACTIVATIONS: dict[str, nn.Module] = {
    "swiglu": nn.SiLU(),
    "geglu": nn.GELU(),
    "glu": nn.Sigmoid(),
    "swish": nn.SiLU(),
}


def _build_norm(dim: int, eps: float, norm_type: str) -> nn.Module:
    kind = str(norm_type).lower()
    if kind == "rmsnorm":
        return nn.RMSNorm(dim, eps=eps)
    if kind == "layernorm":
        return nn.LayerNorm(dim, eps=eps)
    raise ValueError(f"Unsupported norm_type: {norm_type}")


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        *,
        n_kv_heads: int | None = None,
        causal: bool = False,
        qk_norm: bool = False,
        norm_type: str = "rmsnorm",
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.head_dim = self.dim // self.n_heads
        self.causal = causal
        self.qk_norm = qk_norm

        out_features = (self.n_heads + 2 * self.n_kv_heads) * self.head_dim
        self.wqkv = nn.Linear(self.dim, out_features, bias=False)
        self.wo = nn.Linear(self.dim, self.dim, bias=False)

        if qk_norm:
            self.q_norm = _build_norm(self.head_dim, eps=1e-5, norm_type=norm_type)
            self.k_norm = _build_norm(self.head_dim, eps=1e-5, norm_type=norm_type)

        nn.init.xavier_normal_(self.wqkv.weight)
        nn.init.xavier_normal_(self.wo.weight)

    def forward(
        self,
        x: torch.Tensor,
        *,
        freqs_cos: torch.Tensor | None = None,
        freqs_sin: torch.Tensor | None = None,
        block_mask: BlockMask | None = None,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        qkv = self.wqkv(x)

        q_size = self.n_heads * self.head_dim
        kv_size = self.n_kv_heads * self.head_dim

        xq, xk, xv = torch.split(qkv, [q_size, kv_size, kv_size], dim=-1)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        if self.qk_norm:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        if freqs_cos is not None and freqs_sin is not None:
            xq, xk = apply_rotary_emb(
                xq,
                xk,
                freqs_cos,
                freqs_sin,
            )

        xq = xq.to(dtype=xv.dtype)
        xk = xk.to(dtype=xv.dtype)

        q = xq.transpose(1, 2)
        k = xk.transpose(1, 2)
        v = xv.transpose(1, 2)

        if self.n_kv_heads != self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        attn = flex_attention(q, k, v, block_mask=block_mask)
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
        qk_norm: bool = False,
        post_norm: bool = False,
        norm_type: str = "rmsnorm",
    ):
        super().__init__()
        self.use_rotary_embeddings = use_rotary_embeddings
        self.post_norm = post_norm
        self.attention = Attention(
            dim,
            n_heads,
            n_kv_heads=n_kv_heads,
            causal=causal,
            qk_norm=qk_norm,
            norm_type=norm_type,
        )
        self.feed_forward = FeedForward(
            dim,
            multiple_of=multiple_of,
            hidden_dim=hidden_dim,
            mlp_type=mlp_type,
            w_init_scale=w_init_scale,
        )
        self.attention_norm = _build_norm(dim, eps=norm_eps, norm_type=norm_type)
        self.ffn_norm = _build_norm(dim, eps=norm_eps, norm_type=norm_type)
        if post_norm:
            self.post_attn_norm = _build_norm(dim, eps=norm_eps, norm_type=norm_type)
            self.post_ffn_norm = _build_norm(dim, eps=norm_eps, norm_type=norm_type)

    def forward(
        self,
        x: torch.Tensor,
        *,
        freqs_cos: torch.Tensor | None,
        freqs_sin: torch.Tensor | None,
        block_mask: BlockMask | None = None,
    ) -> torch.Tensor:
        if not self.use_rotary_embeddings:
            freqs_cos = None
            freqs_sin = None

        h = x + self.attention(
            self.attention_norm(x),
            freqs_cos=freqs_cos,
            freqs_sin=freqs_sin,
            block_mask=block_mask,
        )
        if self.post_norm:
            h = self.post_attn_norm(h)
        out = h + self.feed_forward(self.ffn_norm(h))
        if self.post_norm:
            out = self.post_ffn_norm(out)
        return out
