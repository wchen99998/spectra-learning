from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np

from spectra_learning.data.spectra import NUM_PEAKS_INPUT

GEMS_SPECTRUM_METADATA_DATASETS = {
    "collision_energy": "collision_energy",
    "charge": "charge",
}
GEMS_SPLIT_VERSION = "global_chunk_modulo_v1"
GEMS_SPLIT_CHUNK_ROWS = 256
GEMS_SPLIT_MODULUS = 20
GEMS_SPLIT_SEED = 42
GEMS_VALIDATION_REMAINDER = 0
GEMS_REQUIRED_MS_LEVEL = 2
GEMS_ELIGIBILITY_VERSION = "bounded_precursor_rt_ms2_v3"
GEMS_ELIGIBILITY_SCAN_ROWS = 1 << 20
_GEMS_SPLIT_CHUNK_BYTES = GEMS_SPLIT_CHUNK_ROWS // 8
_BYTE_POPCOUNT = np.unpackbits(
    np.arange(256, dtype=np.uint8)[:, None],
    axis=1,
).sum(axis=1)


@dataclass(frozen=True)
class GemsHdf5Eligibility:
    packed: np.ndarray
    chunk_counts: np.ndarray
    eligible_count: int


class GemsHdf5ShardDataset:
    def __init__(
        self,
        manifest_path: Path,
        *,
        spectrum_dataset: str,
        precursor_dataset: str,
        retention_time_dataset: str,
        ms_level_dataset: str,
        min_precursor_mz: float,
        max_precursor_mz: float,
        split: Literal["train", "validation"],
        eligibility: GemsHdf5Eligibility | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.spectrum_dataset = spectrum_dataset
        self.precursor_dataset = precursor_dataset
        self.retention_time_dataset = retention_time_dataset
        self.ms_level_dataset = ms_level_dataset
        self.min_precursor_mz = min_precursor_mz
        self.max_precursor_mz = max_precursor_mz
        self.split = split
        self.paths = self._read_manifest_paths()
        self.files: list[h5py.File] | None = None
        self.spectra: list[h5py.Dataset] | None = None
        self.precursors: list[h5py.Dataset] | None = None
        self.metadata: dict[str, list[h5py.Dataset]] | None = None
        self.starts: list[int] = []
        self.stops: list[int] = []
        self.infos: list[dict[str, Any]] = []
        self._inspect_shards()
        self.eligibility = eligibility or self._scan_eligibility()
        self.split_source_length = self._split_length()
        chunk_counts = self._split_chunk_counts()
        self._eligible_chunk_offsets = np.concatenate(
            (
                np.zeros(1, dtype=np.int64),
                np.cumsum(chunk_counts, dtype=np.int64),
            )
        )
        self.length = int(self._eligible_chunk_offsets[-1])

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
                trailing_shape = tuple(int(value) for value in spectrum.shape[1:])
                if trailing_shape != (2, NUM_PEAKS_INPUT):
                    raise ValueError(
                        f"Invalid GeMS spectrum shape in {path}: expected "
                        f"(N, 2, {NUM_PEAKS_INPUT}), got {tuple(spectrum.shape)}"
                    )
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
        self.source_length = offset

    def _scan_eligibility(self) -> GemsHdf5Eligibility:
        packed = bytearray()
        pending = np.empty(0, dtype=np.bool_)
        for path in self.paths:
            with h5py.File(path, "r") as file:
                precursor = file[self.precursor_dataset]
                retention_time = file[self.retention_time_dataset]
                ms_level = file[self.ms_level_dataset]
                for start in range(0, len(precursor), GEMS_ELIGIBILITY_SCAN_ROWS):
                    stop = min(start + GEMS_ELIGIBILITY_SCAN_ROWS, len(precursor))
                    precursor_rows = precursor[start:stop]
                    retention_time_rows = retention_time[start:stop]
                    ms_level_rows = ms_level[start:stop]
                    valid = (
                        np.isfinite(ms_level_rows)
                        & (ms_level_rows == GEMS_REQUIRED_MS_LEVEL)
                        & np.isfinite(precursor_rows)
                        & (precursor_rows >= self.min_precursor_mz)
                        & (precursor_rows <= self.max_precursor_mz)
                        & np.isfinite(retention_time_rows)
                        & (retention_time_rows > 0)
                    )
                    pending = self._append_packed(packed, pending, valid)
        if len(pending):
            packed.extend(np.packbits(pending).tobytes())

        packed_array = np.frombuffer(packed, dtype=np.uint8)
        padded = np.pad(
            packed_array,
            (0, -len(packed_array) % _GEMS_SPLIT_CHUNK_BYTES),
        )
        chunk_counts = _BYTE_POPCOUNT[padded].reshape(
            -1,
            _GEMS_SPLIT_CHUNK_BYTES,
        ).sum(axis=1, dtype=np.uint16)
        eligible_count = int(chunk_counts.sum(dtype=np.int64))
        return GemsHdf5Eligibility(
            packed=packed_array,
            chunk_counts=chunk_counts,
            eligible_count=eligible_count,
        )

    @staticmethod
    def _append_packed(
        packed: bytearray,
        pending: np.ndarray,
        values: np.ndarray,
    ) -> np.ndarray:
        if len(pending):
            needed = 8 - len(pending)
            pending = np.concatenate((pending, values[:needed]))
            values = values[needed:]
            if len(pending) == 8:
                packed.extend(np.packbits(pending).tobytes())
                pending = np.empty(0, dtype=np.bool_)
            else:
                return pending
        packed_rows = len(values) - len(values) % 8
        packed.extend(np.packbits(values[:packed_rows]).tobytes())
        return values[packed_rows:].copy()

    @property
    def _validation_chunk_remainder(self) -> int:
        return (GEMS_VALIDATION_REMAINDER - GEMS_SPLIT_SEED) % GEMS_SPLIT_MODULUS

    def _is_validation_chunk(self, chunk: int) -> bool:
        return (
            chunk + GEMS_SPLIT_SEED
        ) % GEMS_SPLIT_MODULUS == GEMS_VALIDATION_REMAINDER

    def _split_length(self) -> int:
        full_chunks, tail_rows = divmod(self.source_length, GEMS_SPLIT_CHUNK_ROWS)
        first_validation_chunk = self._validation_chunk_remainder
        validation_full_chunks = (
            0
            if full_chunks <= first_validation_chunk
            else 1
            + (full_chunks - first_validation_chunk - 1) // GEMS_SPLIT_MODULUS
        )
        selected_full_chunks = (
            full_chunks - validation_full_chunks
            if self.split == "train"
            else validation_full_chunks
        )
        tail_is_validation = self._is_validation_chunk(full_chunks)
        includes_tail = (
            tail_is_validation
            if self.split == "validation"
            else not tail_is_validation
        )
        return (
            selected_full_chunks * GEMS_SPLIT_CHUNK_ROWS
            + (tail_rows if includes_tail else 0)
        )

    def _split_chunk_counts(self) -> np.ndarray:
        counts = self.eligibility.chunk_counts
        if self.split == "validation":
            return counts[self._validation_chunk_remainder :: GEMS_SPLIT_MODULUS]
        selected = np.ones(len(counts), dtype=np.bool_)
        selected[self._validation_chunk_remainder :: GEMS_SPLIT_MODULUS] = False
        return counts[selected]

    def _source_chunks(self, selected_chunks: np.ndarray) -> np.ndarray:
        if self.split == "validation":
            return (
                self._validation_chunk_remainder
                + selected_chunks * GEMS_SPLIT_MODULUS
            )
        groups, positions = divmod(selected_chunks, GEMS_SPLIT_MODULUS - 1)
        source_chunks = groups * GEMS_SPLIT_MODULUS + positions
        source_chunks += positions >= self._validation_chunk_remainder
        return source_chunks

    def source_indices(self, indices: list[int] | np.ndarray) -> np.ndarray:
        indices_np = np.asarray(indices, dtype=np.int64)
        selected_chunks = np.searchsorted(
            self._eligible_chunk_offsets,
            indices_np,
            side="right",
        ) - 1
        source_chunks = self._source_chunks(selected_chunks)
        ranks_in_chunk = (
            indices_np - self._eligible_chunk_offsets[selected_chunks]
        )
        source_indices = np.empty_like(indices_np)
        for source_chunk in np.unique(source_chunks):
            positions = np.flatnonzero(source_chunks == source_chunk)
            byte_start = int(source_chunk) * _GEMS_SPLIT_CHUNK_BYTES
            valid_offsets = np.flatnonzero(
                np.unpackbits(
                    self.eligibility.packed[
                        byte_start : byte_start + _GEMS_SPLIT_CHUNK_BYTES
                    ]
                )
            )
            source_indices[positions] = (
                int(source_chunk) * GEMS_SPLIT_CHUNK_ROWS
                + valid_offsets[ranks_in_chunk[positions]]
            )
        return source_indices

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
        source_indices = self.source_indices(indices)
        spectra, precursor, metadata = self.read_raw_batch(indices)
        return [
            {
                "spectra": spectra[position],
                "precursor_mz_raw": precursor[position],
                **{
                    key: values[position]
                    for key, values in metadata.items()
                },
                "index": int(source_indices[position]),
            }
            for position in range(len(indices))
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

        indices_np = self.source_indices(indices)
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
