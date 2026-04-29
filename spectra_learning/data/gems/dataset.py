from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class GemsMemmapDataset(Dataset):
    def __init__(self, shard_entries: list[dict[str, Any]]) -> None:
        self._shard_entries = [
            {"dir": Path(entry["dir"]), "length": int(entry["length"])}
            for entry in shard_entries
        ]
        lengths = np.asarray(
            [entry["length"] for entry in self._shard_entries],
            dtype=np.int64,
        )
        self._starts = np.zeros(len(lengths) + 1, dtype=np.int64)
        np.cumsum(lengths, out=self._starts[1:])
        self._arrays: list[dict[str, np.ndarray]] | None = None

    def __len__(self) -> int:
        return int(self._starts[-1])

    def _ensure_arrays(self) -> None:
        if self._arrays is not None:
            return
        self._arrays = []
        for entry in self._shard_entries:
            shard_dir = entry["dir"]
            self._arrays.append(
                {
                    "spectra": np.load(shard_dir / "spectra.npy", mmap_mode="r"),
                    "precursor_mz_raw": np.load(
                        shard_dir / "precursor_mz_raw.npy",
                        mmap_mode="r",
                    ),
                }
            )

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        self._ensure_arrays()
        idx = int(idx)
        shard_idx = int(np.searchsorted(self._starts, idx, side="right") - 1)
        local_idx = idx - int(self._starts[shard_idx])
        arrays = self._arrays[shard_idx]
        return {
            "spectra": torch.from_numpy(arrays["spectra"][local_idx].copy()),
            "precursor_mz_raw": torch.tensor(
                float(arrays["precursor_mz_raw"][local_idx]),
                dtype=torch.float32,
            ),
        }
