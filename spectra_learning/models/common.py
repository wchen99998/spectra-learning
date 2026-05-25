import math
from contextlib import nullcontext
from typing import cast

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float
from torch import Tensor, nn

from spectra_learning.models.transformer import TransformerBlock


def _active_autocast_context(device_type: str):
    if torch.is_autocast_enabled(device_type):
        return torch.autocast(
            device_type=device_type,
            dtype=torch.get_autocast_dtype(device_type),
        )
    return nullcontext()


def _apply_depth_scaled_init(blocks: nn.ModuleList, num_layers: int) -> None:
    """Scale residual output projections by 1/sqrt(2*num_layers) (GPT-2 style).

    In pre-norm transformers each residual addition contributes ~unit variance,
    so after 2*L sub-layers the activation norm grows by sqrt(2*L).  Scaling
    the output projections (wo in attention, w2 in FFN) keeps the total
    variance growth O(1) regardless of depth.
    """
    if num_layers <= 0:
        return
    scale = 1.0 / math.sqrt(2.0 * num_layers)
    for module in blocks:
        block = cast(TransformerBlock, module)
        block.attention.wo.weight.data.mul_(scale)
        block.feed_forward.w2.weight.data.mul_(scale)


def _build_non_causal_blocks(
    *,
    dim: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int | None,
    attention_mlp_multiple: float,
    norm_eps: float = 1e-5,
    dropout: float = 0.0,
) -> nn.ModuleList:
    block_kwargs = dict(
        dim=dim,
        n_heads=num_heads,
        n_kv_heads=num_heads if num_kv_heads is None else num_kv_heads,
        norm_eps=norm_eps,
        hidden_dim=math.ceil(dim * attention_mlp_multiple),
        dropout=dropout,
    )
    blocks = nn.ModuleList(
        [TransformerBlock(**block_kwargs) for _ in range(num_layers)]
    )
    _apply_depth_scaled_init(blocks, num_layers)
    return blocks


def _build_sincos_position_table(
    num_positions: int,
    dim: int,
) -> Float[Tensor, "positions dim"]:
    half_dim = dim // 2
    positions = torch.arange(num_positions, dtype=torch.float32).unsqueeze(1)
    if half_dim == 0:
        return torch.zeros(num_positions, dim, dtype=torch.float32)
    scales = torch.exp(
        -math.log(10000.0) * torch.arange(half_dim, dtype=torch.float32) / half_dim
    )
    angles = positions * scales.unsqueeze(0)
    table = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if dim % 2 == 1:
        table = F.pad(table, (0, 1))
    return table


def _build_frozen_position_embedding(num_positions: int, dim: int) -> nn.Embedding:
    embedding = nn.Embedding(num_positions, dim)
    with torch.no_grad():
        embedding.weight.copy_(_build_sincos_position_table(num_positions, dim))
    embedding.weight.requires_grad_(False)
    return embedding


def _build_2d_sincos_position_table(
    num_positions: int,
    dim: int,
) -> Float[Tensor, "positions positions dim"]:
    row_dim = dim // 2
    col_dim = dim - row_dim
    row = _build_sincos_position_table(num_positions, row_dim)
    col = _build_sincos_position_table(num_positions, col_dim)
    row = row[:, None, :].expand(num_positions, num_positions, row_dim)
    col = col[None, :, :].expand(num_positions, num_positions, col_dim)
    return torch.cat([row, col], dim=-1)


def _build_frozen_2d_position_embedding(
    num_positions: int,
    dim: int,
) -> nn.Embedding:
    embedding = nn.Embedding(num_positions * num_positions, dim)
    with torch.no_grad():
        embedding.weight.copy_(
            _build_2d_sincos_position_table(num_positions, dim).reshape(
                num_positions * num_positions,
                dim,
            )
        )
    embedding.weight.requires_grad_(False)
    return embedding


def _merge_visible_mask(
    valid_mask: Bool[Tensor, "batch peaks"] | None,
    visible_mask: Bool[Tensor, "batch peaks"] | None,
) -> Bool[Tensor, "batch peaks"] | None:
    if visible_mask is not None and valid_mask is not None:
        return visible_mask & valid_mask
    return visible_mask if visible_mask is not None else valid_mask


def _masked_mean_pool(
    embeddings: Float[Tensor, "batch peaks dim"],
    valid_mask: Bool[Tensor, "batch peaks"],
) -> Float[Tensor, "batch dim"]:
    mask = valid_mask.unsqueeze(-1).to(dtype=embeddings.dtype)
    return (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
