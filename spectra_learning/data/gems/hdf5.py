from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

GEMS_SPECTRUM_METADATA_DATASETS = {
    "collision_energy": "collision_energy",
    "charge": "charge",
}


class GemsHdf5ShardDataset:
    def __init__(
        self,
        manifest_path: Path,
        *,
        spectrum_dataset: str,
        precursor_dataset: str,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.spectrum_dataset = spectrum_dataset
        self.precursor_dataset = precursor_dataset
        self.paths = self._read_manifest_paths()
        self.files: list[h5py.File] | None = None
        self.spectra: list[h5py.Dataset] | None = None
        self.precursors: list[h5py.Dataset] | None = None
        self.metadata: dict[str, list[h5py.Dataset]] | None = None
        self.starts: list[int] = []
        self.stops: list[int] = []
        self.infos: list[dict[str, Any]] = []
        self._inspect_shards()

    def _read_manifest_paths(self) -> list[str]:
        manifest = json.loads(self.manifest_path.read_text())
        paths = []
        for shard in manifest["shards"]:
            path = Path(shard["path"])
            if not path.is_absolute():
                path = self.manifest_path.parent / path
            paths.append(str(path))
        return paths

    def _inspect_shards(self) -> None:
        offset = 0
        for path in self.paths:
            with h5py.File(path, "r") as file:
                spectrum = file[self.spectrum_dataset]
                length = int(spectrum.shape[0])
                chunk = spectrum.chunks or (1,)
                self.starts.append(offset)
                self.stops.append(offset + length)
                self.infos.append(
                    {
                        "length": length,
                        "spectrum_chunk": tuple(int(value) for value in chunk),
                    }
                )
                offset += length
        self.length = offset

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["files"] = None
        state["spectra"] = None
        state["precursors"] = None
        state["metadata"] = None
        return state

    def _ensure_open(self) -> None:
        if self.files is None:
            self.files = [h5py.File(path, "r") for path in self.paths]
            self.spectra = [file[self.spectrum_dataset] for file in self.files]
            self.precursors = [file[self.precursor_dataset] for file in self.files]
            self.metadata = {
                key: [file[dataset] for file in self.files]
                for key, dataset in GEMS_SPECTRUM_METADATA_DATASETS.items()
            }

    def close(self) -> None:
        if self.files is not None:
            for file in self.files:
                file.close()
        self.files = None
        self.spectra = None
        self.precursors = None
        self.metadata = None

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.__getitems__([index])[0]

    def __getitems__(self, indices: list[int]) -> list[dict[str, Any]]:
        spectra, precursor, metadata = self.read_raw_batch(indices)
        return [
            {
                "spectra": spectra[position],
                "precursor_mz_raw": precursor[position],
                **{
                    key: values[position]
                    for key, values in metadata.items()
                },
                "index": int(index),
            }
            for position, index in enumerate(indices)
        ]

    def read_raw_batch(
        self,
        indices: list[int],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        self._ensure_open()
        spectra_datasets = self.spectra
        precursor_datasets = self.precursors
        metadata_datasets = self.metadata
        assert spectra_datasets is not None
        assert precursor_datasets is not None
        assert metadata_datasets is not None

        indices_np = np.asarray(indices, dtype=np.int64)
        spectra_out = np.empty(
            (len(indices_np), *spectra_datasets[0].shape[1:]),
            dtype=np.float32,
        )
        precursor_out = np.empty((len(indices_np),), dtype=np.float32)
        metadata_out = {
            key: np.empty((len(indices_np),), dtype=np.float32)
            for key in metadata_datasets
        }
        order = np.argsort(indices_np, kind="stable")
        sorted_indices = indices_np[order]
        shard_ids = np.searchsorted(np.asarray(self.stops), sorted_indices, side="right")

        position = 0
        while position < len(sorted_indices):
            shard = int(shard_ids[position])
            shard_start = self.starts[shard]
            next_position = position + 1
            while (
                next_position < len(sorted_indices)
                and int(shard_ids[next_position]) == shard
            ):
                next_position += 1

            local_pairs = [
                (int(order[item]), int(sorted_indices[item] - shard_start))
                for item in range(position, next_position)
            ]
            self._read_shard_pairs(
                spectra_datasets[shard],
                precursor_datasets[shard],
                {key: datasets[shard] for key, datasets in metadata_datasets.items()},
                local_pairs,
                spectra_out,
                precursor_out,
                metadata_out,
            )
            position = next_position

        return spectra_out, precursor_out, metadata_out

    def _read_shard_pairs(
        self,
        spectra_dataset: h5py.Dataset,
        precursor_dataset: h5py.Dataset,
        metadata_datasets: dict[str, h5py.Dataset],
        local_pairs: list[tuple[int, int]],
        spectra_out: np.ndarray,
        precursor_out: np.ndarray,
        metadata_out: dict[str, np.ndarray],
    ) -> None:
        local_pairs.sort(key=lambda item: item[1])
        run_position = 0
        while run_position < len(local_pairs):
            pair_start = run_position
            run_start = local_pairs[run_position][1]
            previous = run_start
            run_position += 1
            while (
                run_position < len(local_pairs)
                and local_pairs[run_position][1] == previous + 1
            ):
                previous = local_pairs[run_position][1]
                run_position += 1

            spectra = spectra_dataset[run_start : previous + 1].astype(
                np.float32,
                copy=False,
            )
            precursor = precursor_dataset[run_start : previous + 1]
            metadata = {
                key: dataset[run_start : previous + 1].astype(np.float32, copy=False)
                for key, dataset in metadata_datasets.items()
            }
            for local_position, (output_position, _) in enumerate(
                local_pairs[pair_start:run_position]
            ):
                spectra_out[output_position] = spectra[local_position]
                precursor_out[output_position] = precursor[local_position]
                for key, values in metadata.items():
                    metadata_out[key][output_position] = values[local_position]
