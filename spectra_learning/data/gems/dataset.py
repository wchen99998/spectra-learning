from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class GemsMemmapDataset(Dataset):
    def __init__(
        self,
        shard_entries: list[dict[str, Any]],
        *,
        return_numpy: bool = False,
    ) -> None:
        self._shard_entries = [
            {"dir": Path(entry["dir"]), "length": int(entry["length"])}
            for entry in shard_entries
        ]
        self._return_numpy = return_numpy
        lengths = np.asarray(
            [entry["length"] for entry in self._shard_entries],
            dtype=np.int64,
        )
        self._starts = np.zeros(len(lengths) + 1, dtype=np.int64)
        np.cumsum(lengths, out=self._starts[1:])
        self._arrays: list[dict[str, np.ndarray]] | None = None

    def __len__(self) -> int:
        return int(self._starts[-1])

    def _ensure_arrays(self) -> list[dict[str, np.ndarray]]:
        if self._arrays is not None:
            return self._arrays
        arrays: list[dict[str, np.ndarray]] = []
        for entry in self._shard_entries:
            shard_dir = entry["dir"]
            arrays.append(
                {
                    "spectra": np.load(shard_dir / "spectra.npy", mmap_mode="r"),
                    "precursor_mz_raw": np.load(
                        shard_dir / "precursor_mz_raw.npy",
                        mmap_mode="r",
                    ),
                }
            )
        self._arrays = arrays
        return arrays

    def __getitem__(self, index: int) -> dict[str, Any]:
        arrays_by_shard = self._ensure_arrays()
        index = index
        shard_idx = int(np.searchsorted(self._starts, index, side="right") - 1)
        local_idx = index - int(self._starts[shard_idx])
        arrays = arrays_by_shard[shard_idx]
        if self._return_numpy:
            return {
                "spectra": arrays["spectra"][local_idx].copy(),
                "precursor_mz_raw": np.float32(arrays["precursor_mz_raw"][local_idx]),
            }
        return {
            "spectra": torch.from_numpy(arrays["spectra"][local_idx].copy()),
            "precursor_mz_raw": torch.tensor(
                float(arrays["precursor_mz_raw"][local_idx]),
                dtype=torch.float32,
            ),
        }
