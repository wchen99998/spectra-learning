import math
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float
from torch import Tensor, nn

def _active_autocast_context(device_type: str):
    if torch.is_autocast_enabled(device_type):
        return torch.autocast(
            device_type=device_type,
            dtype=torch.get_autocast_dtype(device_type),
        )
    return nullcontext()


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


def _merge_visible_mask(
    valid_mask: Bool[Tensor, "batch peaks"] | None,
    visible_mask: Bool[Tensor, "batch peaks"] | None,
) -> Bool[Tensor, "batch peaks"] | None:
    if visible_mask is not None and valid_mask is not None:
        return visible_mask & valid_mask
    return visible_mask if visible_mask is not None else valid_mask
