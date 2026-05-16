import math
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from spectra_learning.models.pooling import CovariancePool
from spectra_learning.models.transformer import Attention, FeedForward, _build_norm


class CrossAttention(nn.Module):
    """Cross-attention: Q from prediction queries, KV from source embeddings."""

    def __init__(self, dim: int, n_heads: int, *, n_kv_heads: int | None = None,
                 qk_norm: bool = False, norm_type: str = "rmsnorm",
                 norm_eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.head_dim = self.dim // self.n_heads
        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(self.dim, 2 * self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.dim, self.dim, bias=False)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = _build_norm(self.head_dim, eps=norm_eps, norm_type=norm_type)
            self.k_norm = _build_norm(self.head_dim, eps=norm_eps, norm_type=norm_type)
        nn.init.xavier_normal_(self.wq.weight)
        nn.init.xavier_normal_(self.wkv.weight)
        nn.init.xavier_normal_(self.wo.weight)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, *,
                memory_mask: torch.Tensor | None = None) -> torch.Tensor:
        bsz, tgt_len, _ = x.shape
        mem_len = memory.shape[1]
        xq = self.wq(x).view(bsz, tgt_len, self.n_heads, self.head_dim)
        kv = self.wkv(memory)
        xk, xv = kv.split(
            [self.n_kv_heads * self.head_dim, self.n_kv_heads * self.head_dim], dim=-1)
        xk = xk.view(bsz, mem_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, mem_len, self.n_kv_heads, self.head_dim)
        if self.qk_norm:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)
        q = xq.transpose(1, 2)  # [B, H, T, D]
        k = xk.transpose(1, 2)  # [B, H, S, D]
        v = xv.transpose(1, 2)
        attn_mask = None
        if memory_mask is not None:
            # memory_mask: [B, S] -> [B, 1, 1, S]
            attn_mask = memory_mask[:, None, None, :].to(dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(attn_mask == 0, float("-inf")).masked_fill(attn_mask == 1, 0.0)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn = attn.transpose(1, 2).contiguous().view(bsz, tgt_len, self.dim)
        return self.wo(attn)


class TemporalDecoderBlock(nn.Module):
    """Decoder block: self-attention + cross-attention + FFN."""

    def __init__(self, *, dim: int, n_heads: int, n_kv_heads: int | None,
                 norm_eps: float, hidden_dim: int | None,
                 qk_norm: bool = False, norm_type: str = "rmsnorm",
                 dropout: float = 0.0):
        super().__init__()
        self.attention = Attention(
            dim, n_heads, n_kv_heads=n_kv_heads,
            qk_norm=qk_norm, norm_type=norm_type, norm_eps=norm_eps,
        )
        self.cross_attn = CrossAttention(dim, n_heads, n_kv_heads=n_kv_heads,
                                          qk_norm=qk_norm, norm_type=norm_type,
                                          norm_eps=norm_eps)
        self.feed_forward = FeedForward(dim, hidden_dim=hidden_dim)
        self.attention_norm = _build_norm(dim, eps=norm_eps, norm_type=norm_type)
        self.cross_attn_norm = _build_norm(dim, eps=norm_eps, norm_type=norm_type)
        self.ffn_norm = _build_norm(dim, eps=norm_eps, norm_type=norm_type)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, memory: torch.Tensor, *,
                memory_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = x + self.drop(self.attention(self.attention_norm(x)))
        h = h + self.drop(
            self.cross_attn(self.cross_attn_norm(h), memory, memory_mask=memory_mask)
        )
        return h + self.drop(self.feed_forward(self.ffn_norm(h)))


class CrossAttentionDecoderBlock(nn.Module):
    """CrossMAE-style decoder block: cross-attention + FFN."""

    def __init__(self, *, dim: int, n_heads: int, n_kv_heads: int | None,
                 norm_eps: float, hidden_dim: int | None,
                 qk_norm: bool = False, norm_type: str = "rmsnorm",
                 dropout: float = 0.0):
        super().__init__()
        self.cross_attn = CrossAttention(dim, n_heads, n_kv_heads=n_kv_heads,
                                          qk_norm=qk_norm, norm_type=norm_type,
                                          norm_eps=norm_eps)
        self.feed_forward = FeedForward(dim, hidden_dim=hidden_dim)
        self.cross_attn_norm = _build_norm(dim, eps=norm_eps, norm_type=norm_type)
        self.ffn_norm = _build_norm(dim, eps=norm_eps, norm_type=norm_type)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, memory: torch.Tensor, *,
                memory_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = x + self.drop(
            self.cross_attn(self.cross_attn_norm(x), memory, memory_mask=memory_mask)
        )
        return h + self.drop(self.feed_forward(self.ffn_norm(h)))


def _apply_temporal_depth_scaled_init(blocks: nn.ModuleList, num_layers: int) -> None:
    """Depth-scaled init for temporal decoder blocks.

    Each block has 3 residual sub-layers (self-attn, cross-attn, FFN),
    so scale by 1/sqrt(3*num_layers).
    """
    if num_layers <= 0:
        return
    scale = 1.0 / math.sqrt(3.0 * num_layers)
    for module in blocks:
        block = cast(TemporalDecoderBlock, module)
        block.attention.wo.weight.data.mul_(scale)
        block.cross_attn.wo.weight.data.mul_(scale)
        block.feed_forward.w2.weight.data.mul_(scale)


def _apply_cross_attention_depth_scaled_init(
    blocks: nn.ModuleList,
    num_layers: int,
) -> None:
    """Depth-scaled init for cross-attention decoder blocks."""
    if num_layers <= 0:
        return
    scale = 1.0 / math.sqrt(2.0 * num_layers)
    for module in blocks:
        block = cast(CrossAttentionDecoderBlock, module)
        block.cross_attn.wo.weight.data.mul_(scale)
        block.feed_forward.w2.weight.data.mul_(scale)


def _build_temporal_decoder_blocks(*, dim: int, num_layers: int, num_heads: int,
                                    num_kv_heads: int | None, attention_mlp_multiple: float,
                                    norm_eps: float = 1e-5, qk_norm: bool = False,
                                    norm_type: str = "rmsnorm",
                                    dropout: float = 0.0) -> nn.ModuleList:
    block_kwargs = dict(
        dim=dim, n_heads=num_heads,
        n_kv_heads=num_heads if num_kv_heads is None else num_kv_heads,
        norm_eps=norm_eps, hidden_dim=math.ceil(dim * attention_mlp_multiple),
        qk_norm=qk_norm, norm_type=norm_type,
        dropout=dropout,
    )
    blocks = nn.ModuleList([TemporalDecoderBlock(**block_kwargs) for _ in range(num_layers)])
    _apply_temporal_depth_scaled_init(blocks, num_layers)
    return blocks


def _build_cross_attention_decoder_blocks(*, dim: int, num_layers: int,
                                          num_heads: int, num_kv_heads: int | None,
                                          attention_mlp_multiple: float,
                                          norm_eps: float = 1e-5,
                                          qk_norm: bool = False,
                                          norm_type: str = "rmsnorm",
                                          dropout: float = 0.0) -> nn.ModuleList:
    block_kwargs = dict(
        dim=dim, n_heads=num_heads,
        n_kv_heads=num_heads if num_kv_heads is None else num_kv_heads,
        norm_eps=norm_eps, hidden_dim=math.ceil(dim * attention_mlp_multiple),
        qk_norm=qk_norm, norm_type=norm_type,
        dropout=dropout,
    )
    blocks = nn.ModuleList(
        [CrossAttentionDecoderBlock(**block_kwargs) for _ in range(num_layers)]
    )
    _apply_cross_attention_depth_scaled_init(blocks, num_layers)
    return blocks
