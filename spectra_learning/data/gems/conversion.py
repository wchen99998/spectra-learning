from typing import Any

import numpy as np
import torch

from spectra_learning.data.spectra import PRECURSOR_TOKEN_INTENSITY


def numpy_batch_to_torch(batch: dict[str, Any]) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            if value.dtype == object:
                return [convert(item) for item in value.tolist()]
            if value.dtype.kind in {"U", "S"}:
                return value.tolist()
            if not value.flags.c_contiguous or not value.flags.writeable:
                value = value.copy()
            return torch.from_numpy(value)
        if isinstance(value, list):
            return [convert(item) for item in value]
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return value

    return {key: convert(value) for key, value in batch.items()}


def _prepend_precursor_token_torch(
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    batch_size = int(batch["peak_mz"].shape[0])
    device = batch["peak_intensity"].device
    result: dict[str, torch.Tensor] = {
        "peak_mz": torch.cat([batch["precursor_mz"].unsqueeze(1), batch["peak_mz"]], dim=1),
        "peak_intensity": torch.cat(
            [
                torch.full(
                    (batch_size, 1),
                    PRECURSOR_TOKEN_INTENSITY,
                    dtype=batch["peak_intensity"].dtype,
                    device=device,
                ),
                batch["peak_intensity"],
            ],
            dim=1,
        ),
        "peak_valid_mask": torch.cat(
            [
                torch.ones((batch_size, 1), dtype=torch.bool, device=device),
                batch["peak_valid_mask"],
            ],
            dim=1,
        ),
    }
    for key in batch:
        if key not in result and key != "precursor_mz":
            result[key] = batch[key]
    if "context_mask" in batch:
        result["context_mask"] = torch.cat(
            [
                torch.ones((batch_size, 1), dtype=torch.bool, device=device),
                batch["context_mask"],
            ],
            dim=1,
        )
    if "target_masks" in batch:
        num_targets = int(batch["target_masks"].shape[1])
        result["target_masks"] = torch.cat(
            [
                torch.zeros((batch_size, num_targets, 1), dtype=torch.bool, device=device),
                batch["target_masks"],
            ],
            dim=2,
        )
    return result
