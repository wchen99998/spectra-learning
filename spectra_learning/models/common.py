from contextlib import nullcontext

import torch
from jaxtyping import Bool
from torch import Tensor


def _active_autocast_context(device_type: str):
    if torch.is_autocast_enabled(device_type):
        return torch.autocast(
            device_type=device_type,
            dtype=torch.get_autocast_dtype(device_type),
        )
    return nullcontext()


def _merge_visible_mask(
    valid_mask: Bool[Tensor, "batch peaks"] | None,
    visible_mask: Bool[Tensor, "batch peaks"] | None,
) -> Bool[Tensor, "batch peaks"] | None:
    if visible_mask is not None and valid_mask is not None:
        return visible_mask & valid_mask
    return visible_mask if visible_mask is not None else valid_mask
