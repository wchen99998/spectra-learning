from typing import Any

import numpy as np
import torch


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
