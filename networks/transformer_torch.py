"""PyTorch implementation of the transformer blocks used by the MAE BERT model."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

compiled_flex_attention_cuda = torch.compile(flex_attention)
compiled_flex_attention_cpu = torch.compile(flex_attention, backend="eager")


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
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
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

        if self.n_kv_heads != self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k = k.unsqueeze(2).expand(-1, -1, rep, -1, -1).reshape(bsz, self.n_heads, seqlen, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, rep, -1, -1).reshape(bsz, self.n_heads, seqlen, self.head_dim)

        # attention_mask plumbing is in place but currently unused:
        # flex_attention score_mod with fullgraph=True compilation is not yet
        # stable in PyTorch 2.10. Padding is handled via zero-embedding +
        # masked pooling instead.
        del attention_mask
        if q.is_cuda:
            attn = compiled_flex_attention_cuda(q, k, v)
        else:
            attn = compiled_flex_attention_cpu(q, k, v)
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
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.use_rotary_embeddings:
            freqs_cos = None
            freqs_sin = None

        h = x + self.attention(
            self.attention_norm(x),
            freqs_cos=freqs_cos,
            freqs_sin=freqs_sin,
            attention_mask=attention_mask,
        )
        return h + self.feed_forward(self.ffn_norm(h))
