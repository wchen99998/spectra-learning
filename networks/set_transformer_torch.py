"""Set Transformer modules using flex_attention."""

import math

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

from networks.transformer_torch import FeedForward


def _build_norm(dim: int, eps: float, norm_type: str) -> nn.Module:
    kind = str(norm_type).lower()
    if kind == "rmsnorm":
        return nn.RMSNorm(dim, eps=eps)
    if kind == "layernorm":
        return nn.LayerNorm(dim, eps=eps)
    raise ValueError(f"Unsupported norm_type: {norm_type}")


def create_kv_padding_block_mask(valid_kv_mask: torch.Tensor, q_len: int) -> BlockMask:
    """Block mask that masks out invalid KV positions."""
    B, KV = valid_kv_mask.shape

    def mask_mod(b, h, q_idx, kv_idx):
        return valid_kv_mask[b, kv_idx]

    return create_block_mask(mask_mod, B=B, H=None, Q_LEN=q_len, KV_LEN=KV, device=valid_kv_mask.device)


def create_q_padding_block_mask(valid_q_mask: torch.Tensor, kv_len: int) -> BlockMask:
    """Block mask that masks out invalid Q positions."""
    B, Q = valid_q_mask.shape

    def mask_mod(b, h, q_idx, kv_idx):
        return valid_q_mask[b, q_idx]

    return create_block_mask(mask_mod, B=B, H=None, Q_LEN=Q, KV_LEN=kv_len, device=valid_q_mask.device)


class CrossAttention(nn.Module):
    """Multi-head cross-attention with optional grouped-query attention."""

    def __init__(self, dim: int, n_heads: int, *, n_kv_heads: int | None = None):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(dim, 2 * self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        nn.init.xavier_normal_(self.wq.weight)
        nn.init.xavier_normal_(self.wkv.weight)
        nn.init.xavier_normal_(self.wo.weight)

    def forward(
        self, q_in: torch.Tensor, kv_in: torch.Tensor, *, block_mask: BlockMask | None = None
    ) -> torch.Tensor:
        bsz, q_len, _ = q_in.shape
        _, kv_len, _ = kv_in.shape

        xq = self.wq(q_in).view(bsz, q_len, self.n_heads, self.head_dim)
        kv = self.wkv(kv_in)
        xk, xv = kv.split(self.n_kv_heads * self.head_dim, dim=-1)
        xk = xk.view(bsz, kv_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, kv_len, self.n_kv_heads, self.head_dim)

        # (bsz, heads, seq, head_dim)
        q = xq.transpose(1, 2)
        k = xk.transpose(1, 2)
        v = xv.transpose(1, 2)

        if self.n_kv_heads != self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k = k.unsqueeze(2).expand(-1, -1, rep, -1, -1).reshape(bsz, self.n_heads, kv_len, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, rep, -1, -1).reshape(bsz, self.n_heads, kv_len, self.head_dim)

        attn = flex_attention(q, k, v, block_mask=block_mask)
        attn = attn.transpose(1, 2).reshape(bsz, q_len, -1)
        return self.wo(attn)


class MAB(nn.Module):
    """Multihead Attention Block with pre-norm cross-attention and FFN."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        *,
        n_kv_heads: int | None = None,
        attention_mlp_multiple: float = 4.0,
        norm_eps: float = 1e-5,
        norm_type: str = "rmsnorm",
    ):
        super().__init__()
        self.cross_attn = CrossAttention(dim, n_heads, n_kv_heads=n_kv_heads)
        self.feed_forward = FeedForward(
            dim,
            multiple_of=4,
            hidden_dim=int(math.ceil(dim * attention_mlp_multiple)),
            mlp_type="swish",
            w_init_scale=1.0,
        )
        self.q_norm = _build_norm(dim, eps=norm_eps, norm_type=norm_type)
        self.kv_norm = _build_norm(dim, eps=norm_eps, norm_type=norm_type)
        self.ffn_norm = _build_norm(dim, eps=norm_eps, norm_type=norm_type)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, *, block_mask: BlockMask | None = None
    ) -> torch.Tensor:
        h = x + self.cross_attn(self.q_norm(x), self.kv_norm(y), block_mask=block_mask)
        return h + self.feed_forward(self.ffn_norm(h))


class ISAB(nn.Module):
    """Induced Set Attention Block with learned inducing points."""

    def __init__(
        self,
        dim: int,
        num_inducing_points: int,
        n_heads: int,
        *,
        n_kv_heads: int | None = None,
        attention_mlp_multiple: float = 4.0,
        norm_eps: float = 1e-5,
        norm_type: str = "rmsnorm",
    ):
        super().__init__()
        self.inducing_points = nn.Parameter(torch.empty(num_inducing_points, dim))
        nn.init.xavier_normal_(self.inducing_points)
        self.mab1 = MAB(
            dim,
            n_heads,
            n_kv_heads=n_kv_heads,
            attention_mlp_multiple=attention_mlp_multiple,
            norm_eps=norm_eps,
            norm_type=norm_type,
        )
        self.mab2 = MAB(
            dim,
            n_heads,
            n_kv_heads=n_kv_heads,
            attention_mlp_multiple=attention_mlp_multiple,
            norm_eps=norm_eps,
            norm_type=norm_type,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        kv_block_mask: BlockMask | None = None,
        q_block_mask: BlockMask | None = None,
    ) -> torch.Tensor:
        B = x.shape[0]
        I = self.inducing_points.unsqueeze(0).expand(B, -1, -1)  # [B, m, D]
        H = self.mab1(I, x, block_mask=kv_block_mask)  # inducing attends to set
        return self.mab2(x, H, block_mask=q_block_mask)  # set attends to inducing
