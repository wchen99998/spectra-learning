from __future__ import annotations

import json
from collections import OrderedDict
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np

from spectra_learning.data.gems.artifacts import (
    GemsHdf5Shard,
)
from spectra_learning.data.spectra import NUM_PEAKS_INPUT
from spectra_learning.data.gems.randomness import data_seed

GEMS_SPECTRUM_METADATA_DATASETS = {
    "collision_energy": "collision_energy",
    "charge": "charge",
}


def spectrum_metadata_datasets(manifest: dict[str, Any]) -> dict[str, str]:
    metadata = manifest.get("spectrum_metadata")
    if metadata is None:
        return dict(GEMS_SPECTRUM_METADATA_DATASETS)
    return {
        key: dataset
        for key, dataset in metadata["columns"].items()
        if key != "precursor_mz"
    }


GEMS_SPLIT_CHUNK_ROWS = 256
GEMS_ELIGIBILITY_SCAN_ROWS = 1 << 20
_GEMS_SPLIT_CHUNK_BYTES = GEMS_SPLIT_CHUNK_ROWS // 8
_BYTE_POPCOUNT = np.unpackbits(
    np.arange(256, dtype=np.uint8)[:, None],
    axis=1,
).sum(axis=1)


@dataclass
class MassiveV2ShardState:
    spec: GemsHdf5Shard
    path: str
    logical_start: int
    logical_stop: int
    eligibility_packed: np.ndarray | None = None
    eligibility_offsets: np.ndarray | None = None


class MassiveV2Hdf5ShardDataset:
    def __init__(
        self,
        manifest_path: Path,
        shards: tuple[GemsHdf5Shard, ...],
        *,
        spectrum_dataset: str,
        precursor_dataset: str,
        split: Literal["train", "validation"],
        metadata_keys: tuple[str, ...] | None = None,
        cache_blocks: int = 2048,
        rows_per_block: int | None = None,
    ) -> None:
        if cache_blocks < 1:
            raise ValueError("cache_blocks must be positive")
        self.cache_blocks = cache_blocks
        self.rows_per_block = rows_per_block
        self._raw_blocks = OrderedDict()
        self._shard_seeds = [data_seed(shard.path, shard.sha256) for shard in shards]
        self.manifest_path = Path(manifest_path)
        self.spectrum_dataset = spectrum_dataset
        self.precursor_dataset = precursor_dataset
        self.split = split
        manifest = json.loads(self.manifest_path.read_text())
        available_metadata = spectrum_metadata_datasets(manifest)
        self.metadata_datasets = (
            available_metadata
            if metadata_keys is None
            else {key: available_metadata[key] for key in metadata_keys}
        )
        missing = [
            str(self.manifest_path.parent / spec.path)
            for spec in shards
            if not (self.manifest_path.parent / spec.path).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"GeMS shards are not staged: {missing}. "
                "Run python -m spectra_learning.data.gems.materialize first."
            )
        self.files: dict[int, h5py.File] = {}
        self.spectra: dict[int, h5py.Dataset] = {}
        self.precursors: dict[int, h5py.Dataset] = {}
        self.metadata: dict[str, dict[int, h5py.Dataset]] = {
            key: {} for key in self.metadata_datasets
        }
        self.states: list[MassiveV2ShardState] = []
        self.infos: list[dict[str, Any]] = []
        logical_offset = 0
        for spec in shards:
            path = self.manifest_path.parent / spec.path
            self.states.append(
                MassiveV2ShardState(
                    spec=spec,
                    path=str(path),
                    logical_start=logical_offset,
                    logical_stop=logical_offset + spec.eligible_rows,
                )
            )
            self.infos.append(
                {
                    "length": spec.eligible_rows,
                    "source_rows": spec.rows,
                    "spectrum_chunk": (
                        spec.chunk_rows,
                        2,
                        NUM_PEAKS_INPUT,
                    ),
                }
            )
            logical_offset += spec.eligible_rows
        self.paths = [state.path for state in self.states]
        self.starts = [state.logical_start for state in self.states]
        self.stops = [state.logical_stop for state in self.states]
        self.source_length = sum(state.spec.rows for state in self.states)
        self.length = logical_offset

    @property
    def segments(self) -> list[tuple[int, int, int]]:
        return [
            (
                state.logical_start,
                state.logical_stop - state.logical_start,
                state.spec.chunk_rows,
            )
            for state in self.states
        ]

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_raw_blocks"] = OrderedDict()
        state["files"] = {}
        state["spectra"] = {}
        state["precursors"] = {}
        state["metadata"] = {key: {} for key in self.metadata_datasets}
        return state

    def _ensure_shard_open(self, shard_id: int) -> None:
        if shard_id in self.files:
            return
        state = self.states[shard_id]
        path = state.path
        with ExitStack() as resources:
            file = resources.enter_context(
                h5py.File(path, "r", rdcc_nbytes=32 * 1024 * 1024)
            )
            spectrum = file[self.spectrum_dataset]
            if tuple(spectrum.shape) != (
                state.spec.rows,
                2,
                NUM_PEAKS_INPUT,
            ):
                raise ValueError(
                    f"Invalid GeMS spectrum shape in {path}: expected "
                    f"({state.spec.rows}, 2, {NUM_PEAKS_INPUT}), "
                    f"got {tuple(spectrum.shape)}"
                )
            if spectrum.dtype != np.dtype(np.float32):
                raise ValueError(
                    f"Invalid GeMS spectrum dtype in {path}: expected "
                    f"float32, got {spectrum.dtype}"
                )
            eligible = file["training_eligible"][:].astype(np.bool_, copy=False)
            packed = np.packbits(eligible)
            padded = np.pad(
                packed,
                (0, -len(packed) % _GEMS_SPLIT_CHUNK_BYTES),
            )
            counts = (
                _BYTE_POPCOUNT[padded]
                .reshape(
                    -1,
                    _GEMS_SPLIT_CHUNK_BYTES,
                )
                .sum(axis=1, dtype=np.uint16)
            )
            if int(counts.sum(dtype=np.int64)) != state.spec.eligible_rows:
                raise ValueError(
                    f"GeMS eligible row count mismatch for {path}: "
                    f"manifest={state.spec.eligible_rows}, "
                    f"file={int(counts.sum(dtype=np.int64))}"
                )
            state.eligibility_packed = packed
            state.eligibility_offsets = np.concatenate(
                (
                    np.zeros(1, dtype=np.int64),
                    np.cumsum(counts, dtype=np.int64),
                )
            )
            precursor = file[self.precursor_dataset]
            metadata = {key: file[name] for key, name in self.metadata_datasets.items()}
            self.files[shard_id] = file
            self.spectra[shard_id] = spectrum
            self.precursors[shard_id] = precursor
            for key, dataset in metadata.items():
                self.metadata[key][shard_id] = dataset
            resources.pop_all()

    def close(self) -> None:
        self._raw_blocks.clear()
        for file in self.files.values():
            file.close()
        self.files = {}
        self.spectra = {}
        self.precursors = {}
        self.metadata = {key: {} for key in self.metadata_datasets}

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, indices: int | list[int]) -> dict[str, Any]:
        if isinstance(indices, int):
            return {key: value[0] for key, value in self[[indices]].items()}
        spectra, precursor, metadata = self.read_raw_batch(indices)
        shard_ids, physical_rows = self._physical_rows(
            np.asarray(indices, dtype=np.int64)
        )
        sample_ids = np.asarray(
            [
                data_seed(self._shard_seeds[int(shard)], int(row))
                for shard, row in zip(shard_ids, physical_rows, strict=True)
            ],
            dtype=np.uint64,
        )
        return {
            "spectra": spectra,
            "precursor_mz_raw": precursor,
            **metadata,
            "index": np.asarray(indices, dtype=np.int64),
            "sample_id": sample_ids,
        }

    def _physical_rows(
        self,
        indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        stops = np.asarray(self.stops)
        shard_ids = np.searchsorted(stops, indices, side="right")
        physical_rows = np.empty_like(indices)
        for shard_id in np.unique(shard_ids):
            self._ensure_shard_open(int(shard_id))
            positions = np.flatnonzero(shard_ids == shard_id)
            state = self.states[int(shard_id)]
            assert state.eligibility_offsets is not None
            assert state.eligibility_packed is not None
            local_indices = indices[positions] - state.logical_start
            chunks = (
                np.searchsorted(
                    state.eligibility_offsets,
                    local_indices,
                    side="right",
                )
                - 1
            )
            ranks = local_indices - state.eligibility_offsets[chunks]
            for chunk in np.unique(chunks):
                chunk_positions = np.flatnonzero(chunks == chunk)
                byte_start = int(chunk) * _GEMS_SPLIT_CHUNK_BYTES
                valid_offsets = np.flatnonzero(
                    np.unpackbits(
                        state.eligibility_packed[
                            byte_start : byte_start + _GEMS_SPLIT_CHUNK_BYTES
                        ]
                    )
                )
                physical_rows[positions[chunk_positions]] = (
                    int(chunk) * GEMS_SPLIT_CHUNK_ROWS
                    + valid_offsets[ranks[chunk_positions]]
                )
        return shard_ids, physical_rows

    def read_raw_batch(
        self, indices: list[int]
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        indices_np = np.asarray(indices, dtype=np.int64)
        spectra = np.empty((len(indices), 2, NUM_PEAKS_INPUT), dtype=np.float32)
        precursor = np.empty(len(indices), dtype=np.float32)
        metadata = {
            key: np.empty(len(indices), dtype=np.float32) for key in self.metadata
        }
        shard_ids = np.searchsorted(self.stops, indices_np, side="right")
        for shard_id in np.unique(shard_ids):
            state = self.states[int(shard_id)]
            positions = np.flatnonzero(shard_ids == shard_id)
            local = indices_np[positions] - state.logical_start
            block_rows = max(
                state.spec.chunk_rows, self.rows_per_block or state.spec.chunk_rows
            )
            block_ids = local // block_rows
            for block_id in np.unique(block_ids):
                key = (int(shard_id), int(block_id))
                if key not in self._raw_blocks:
                    start = state.logical_start + int(block_id) * block_rows
                    stop = min(start + block_rows, state.logical_stop)
                    self._raw_blocks[key] = self._read_raw_rows(
                        list(range(start, stop))
                    )
                    if len(self._raw_blocks) > self.cache_blocks:
                        self._raw_blocks.popitem(last=False)
                self._raw_blocks.move_to_end(key)
                block_spectra, block_precursor, block_metadata = self._raw_blocks[key]
                selected = np.flatnonzero(block_ids == block_id)
                output = positions[selected]
                offsets = local[selected] % block_rows
                spectra[output] = block_spectra[offsets]
                precursor[output] = block_precursor[offsets]
                for name in metadata:
                    metadata[name][output] = block_metadata[name][offsets]
        return spectra, precursor, metadata

    def _read_raw_rows(
        self,
        indices: list[int],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        indices_np = np.asarray(indices, dtype=np.int64)
        shard_ids, physical_rows = self._physical_rows(indices_np)
        spectra_out = np.empty(
            (len(indices_np), 2, NUM_PEAKS_INPUT),
            dtype=np.float32,
        )
        precursor_out = np.empty((len(indices_np),), dtype=np.float32)
        metadata_out = {
            key: np.empty((len(indices_np),), dtype=np.float32) for key in self.metadata
        }
        for shard_id in np.unique(shard_ids):
            positions = np.flatnonzero(shard_ids == shard_id)
            rows, inverse = np.unique(physical_rows[positions], return_inverse=True)
            spectra_out[positions] = _read_rows(self.spectra[int(shard_id)], rows)[
                inverse
            ]
            precursor_out[positions] = _read_rows(self.precursors[int(shard_id)], rows)[
                inverse
            ]
            for key, datasets in self.metadata.items():
                metadata_out[key][positions] = _read_rows(
                    datasets[int(shard_id)], rows
                )[inverse]
        return spectra_out, precursor_out, metadata_out

    def grouped_logical_ranges(
        self,
        shard_id: int,
        *,
        minimum_size: int,
        scan_rows: int = GEMS_ELIGIBILITY_SCAN_ROWS,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._ensure_shard_open(shard_id)
        file = self.files[shard_id]
        state = self.states[shard_id]
        return _grouped_eligible_ranges(
            file,
            logical_start=state.logical_start,
            minimum_size=minimum_size,
            scan_rows=scan_rows,
        )


def _read_rows(dataset: h5py.Dataset, rows: np.ndarray) -> np.ndarray:
    """Read sorted unique rows without changing the sampler's output order."""
    chunk_rows = dataset.chunks[0] if dataset.chunks else 1
    chunks = np.unique(rows // chunk_rows)
    if int(chunks[-1] - chunks[0] + 1) == len(chunks):
        start = int(rows[0])
        return dataset[start : int(rows[-1]) + 1][rows - start]
    return dataset[rows]


def _grouped_eligible_ranges(
    file: h5py.File,
    *,
    logical_start: int,
    minimum_size: int,
    scan_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    group_id = file["group_id"]
    global_group_id = file["global_group_id"]
    massive_id = file["massive_id"]
    training_eligible = file["training_eligible"]
    start_parts: list[np.ndarray] = []
    count_parts: list[np.ndarray] = []
    eligible_before = 0
    pending_key: tuple[bytes, int] | None = None
    pending_start = 0
    pending_count = 0

    def append_pending() -> None:
        nonlocal pending_key, pending_start, pending_count
        if pending_key is not None and pending_count >= minimum_size:
            start_parts.append(np.asarray([pending_start], dtype=np.int64))
            count_parts.append(np.asarray([pending_count], dtype=np.int64))
        pending_key = None
        pending_count = 0

    for physical_start in range(0, len(group_id), scan_rows):
        physical_stop = min(physical_start + scan_rows, len(group_id))
        local_group_id = group_id[physical_start:physical_stop]
        local_global_group_id = global_group_id[physical_start:physical_stop]
        local_massive_id = massive_id[physical_start:physical_stop]
        eligible = training_eligible[physical_start:physical_stop].astype(
            np.int64,
            copy=False,
        )
        assigned = local_group_id >= 0
        changes = np.ones(len(assigned), dtype=np.bool_)
        changes[1:] = (assigned[1:] != assigned[:-1]) | (
            assigned[1:]
            & (
                (local_global_group_id[1:] != local_global_group_id[:-1])
                | (local_massive_id[1:] != local_massive_id[:-1])
            )
        )
        run_starts = np.flatnonzero(changes)
        run_stops = np.concatenate(
            (run_starts[1:], np.asarray([len(assigned)], dtype=np.int64))
        )
        eligible_prefix = np.concatenate(
            (np.zeros(1, dtype=np.int64), np.cumsum(eligible, dtype=np.int64))
        )
        run_counts = eligible_prefix[run_stops] - eligible_prefix[run_starts]
        run_logical_starts = (
            logical_start + eligible_before + eligible_prefix[run_starts]
        )
        run_assigned = assigned[run_starts]

        first_key = (
            bytes(local_massive_id[run_starts[0]]),
            int(local_global_group_id[run_starts[0]]),
        )
        first_run = 0
        if pending_key is not None:
            if run_assigned[0] and first_key == pending_key:
                pending_count += int(run_counts[0])
                if len(run_starts) == 1:
                    eligible_before += int(eligible_prefix[-1])
                    continue
                append_pending()
                first_run = 1
            else:
                append_pending()

        last_run = len(run_starts)
        if run_assigned[-1]:
            last_run -= 1
            pending_key = (
                bytes(local_massive_id[run_starts[-1]]),
                int(local_global_group_id[run_starts[-1]]),
            )
            pending_start = int(run_logical_starts[-1])
            pending_count = int(run_counts[-1])

        complete = np.arange(first_run, last_run)
        complete = complete[
            run_assigned[complete] & (run_counts[complete] >= minimum_size)
        ]
        if len(complete):
            start_parts.append(run_logical_starts[complete].astype(np.int64))
            count_parts.append(run_counts[complete].astype(np.int64))
        eligible_before += int(eligible_prefix[-1])

    append_pending()
    if not start_parts:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    return np.concatenate(start_parts), np.concatenate(count_parts)
