from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np

from spectra_learning.data.gems.artifacts import (
    GemsHdf5Shard,
    materialize_gems_hdf5_shard,
    prefetch_gems_hdf5_shard,
)
from spectra_learning.data.spectra import NUM_PEAKS_INPUT

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
        repo_id: str,
        revision: str,
        spectrum_dataset: str,
        precursor_dataset: str,
        split: Literal["train", "validation"],
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.repo_id = repo_id
        self.revision = revision
        self.spectrum_dataset = spectrum_dataset
        self.precursor_dataset = precursor_dataset
        self.split = split
        manifest = json.loads(self.manifest_path.read_text())
        self.metadata_datasets = spectrum_metadata_datasets(manifest)
        self.files: dict[int, h5py.File] = {}
        self.spectra: dict[int, h5py.Dataset] = {}
        self.precursors: dict[int, h5py.Dataset] = {}
        self.metadata: dict[str, dict[int, h5py.Dataset]] = {
            key: {} for key in self.metadata_datasets
        }
        self.shard_order: tuple[int, ...] = tuple(range(len(shards)))
        self.shard_positions = {
            shard_id: shard_id for shard_id in range(len(shards))
        }
        self.prefetch_thread: threading.Thread | None = None
        self.prefetch_error: BaseException | None = None
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
        state["files"] = {}
        state["spectra"] = {}
        state["precursors"] = {}
        state["metadata"] = {
            key: {} for key in self.metadata_datasets
        }
        state["prefetch_thread"] = None
        state["prefetch_error"] = None
        return state

    def set_shard_order(self, shard_order: list[int]) -> None:
        self.shard_order = tuple(shard_order)
        self.shard_positions = {
            shard_id: position
            for position, shard_id in enumerate(self.shard_order)
        }

    def prefetch_first_shard(self) -> None:
        self.prefetch_first_shards(1)

    def prefetch_first_shards(self, count: int) -> None:
        self._start_prefetch(self.shard_order[:count])

    def wait_for_prefetch(self) -> None:
        if self.prefetch_thread is not None:
            self.prefetch_thread.join()
        if self.prefetch_error is not None:
            raise RuntimeError("GeMS shard prefetch failed") from self.prefetch_error

    def _start_prefetch(self, shard_ids: int | tuple[int, ...]) -> None:
        if isinstance(shard_ids, int):
            shard_ids = (shard_ids,)
        pending = tuple(
            shard_id
            for shard_id in shard_ids
            if not Path(self.states[shard_id].path).exists()
        )
        if not pending:
            return
        if self.prefetch_thread is not None and self.prefetch_thread.is_alive():
            return
        self.prefetch_error = None
        self.prefetch_thread = threading.Thread(
            target=self._prefetch_shards,
            args=(pending,),
            daemon=True,
        )
        self.prefetch_thread.start()

    def _prefetch_shards(self, shard_ids: tuple[int, ...]) -> None:
        try:
            for shard_id in shard_ids:
                prefetch_gems_hdf5_shard(
                    repo_id=self.repo_id,
                    revision=self.revision,
                    manifest_path=self.manifest_path,
                    shard_path=self.states[shard_id].spec.path,
                )
        except BaseException as error:
            self.prefetch_error = error

    def _prefetch_next_shard(self, shard_id: int) -> None:
        position = self.shard_positions[shard_id]
        if position + 1 < len(self.shard_order):
            self._start_prefetch(self.shard_order[position + 1])

    def _ensure_shard_open(self, shard_id: int) -> None:
        if shard_id in self.files:
            return
        state = self.states[shard_id]
        path = materialize_gems_hdf5_shard(
            repo_id=self.repo_id,
            revision=self.revision,
            manifest_path=self.manifest_path,
            shard_path=state.spec.path,
        )
        file = h5py.File(path, "r", rdcc_nbytes=32 * 1024 * 1024)
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
        counts = _BYTE_POPCOUNT[padded].reshape(
            -1,
            _GEMS_SPLIT_CHUNK_BYTES,
        ).sum(axis=1, dtype=np.uint16)
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
        self.files[shard_id] = file
        self.spectra[shard_id] = spectrum
        self.precursors[shard_id] = file[self.precursor_dataset]
        for key, dataset in self.metadata_datasets.items():
            self.metadata[key][shard_id] = file[dataset]
        self._prefetch_next_shard(shard_id)

    def close(self) -> None:
        for file in self.files.values():
            file.close()
        self.files = {}
        self.spectra = {}
        self.precursors = {}
        self.metadata = {
            key: {} for key in self.metadata_datasets
        }

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
            chunks = np.searchsorted(
                state.eligibility_offsets,
                local_indices,
                side="right",
            ) - 1
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
            key: np.empty((len(indices_np),), dtype=np.float32)
            for key in self.metadata
        }
        for shard_id in np.unique(shard_ids):
            positions = np.flatnonzero(shard_ids == shard_id)
            pairs = [
                (int(position), int(physical_rows[position]))
                for position in positions
            ]
            self._read_shard_pairs(
                self.spectra[int(shard_id)],
                self.precursors[int(shard_id)],
                {
                    key: datasets[int(shard_id)]
                    for key, datasets in self.metadata.items()
                },
                pairs,
                spectra_out,
                precursor_out,
                metadata_out,
            )
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

    @staticmethod
    def _read_shard_pairs(
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
            spectra = spectra_dataset[run_start : previous + 1]
            precursor = precursor_dataset[run_start : previous + 1]
            metadata = {
                key: dataset[run_start : previous + 1].astype(
                    np.float32,
                    copy=False,
                )
                for key, dataset in metadata_datasets.items()
            }
            for local_position, (output_position, _) in enumerate(
                local_pairs[pair_start:run_position]
            ):
                spectra_out[output_position] = spectra[local_position]
                precursor_out[output_position] = precursor[local_position]
                for key, values in metadata.items():
                    metadata_out[key][output_position] = values[local_position]


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
        changes[1:] = (
            (assigned[1:] != assigned[:-1])
            | (
                assigned[1:]
                & (
                    (local_global_group_id[1:] != local_global_group_id[:-1])
                    | (local_massive_id[1:] != local_massive_id[:-1])
                )
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
