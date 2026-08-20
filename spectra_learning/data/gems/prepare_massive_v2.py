from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from huggingface_hub import HfApi, hf_hub_download

from spectra_learning.data.gems.artifacts import MASSIVE_V2_HDF5_FORMAT
from spectra_learning.data.gems.eligibility import (
    MASSIVE_V2_REQUIRED_MS_LEVEL as REQUIRED_MS_LEVEL,
    massive_v2_eligibility_contract,
    massive_v2_training_eligibility_numpy,
)
from spectra_learning.models.spectrum_metadata import (
    INSTRUMENT_FAMILIES,
    MASSIVE_V2_ACQUISITION_SCHEMA,
    acquisition_type_id,
    instrument_family_id,
)

SOURCE_REPO_ID = "novogaia/massive-v2"
SOURCE_REVISION = "10c48d8184119829c48651b8a40ea5e0b9015687"
SOURCE_SUFFIX = "_t0.95_l0.80_grouped.hdf5"
CONVERSION_VERSION = "massive_v2_ms2_acquisition_metadata_v1"
DEFAULT_WORK_DIR = Path("/mnt/tg-go-nvme/massive-v2-conversion-v3")
TARGET_ROWS_PER_SHARD = 2_097_152
SPECTRUM_CHUNK_ROWS = 256
SCALAR_CHUNK_ROWS = 4096
SPLIT_MODULUS = 20
SPLIT_SEED = 42
VALIDATION_REMAINDER = 0
FINAL_DATASETS = (
    "spectrum",
    "training_eligible",
    "MS level",
    "RT",
    "precursor_mz",
    "collision_energy",
    "charge",
    "massive_id",
    "file_id",
    "group_id",
    "global_group_id",
    "unique_spectrum_id",
)
ACQUISITION_DATASETS = (
    "precursor_mz_present",
    "collision_energy_present",
    "charge_present",
    "polarity_id",
    "acquisition_type_id",
    "isolation_window_lower_offset",
    "isolation_window_upper_offset",
    "isolation_window_present",
    "instrument_family_id",
    "mass_accuracy",
    "mass_accuracy_present",
    "retention_time_fraction",
    "retention_time_present",
    "precursor_intensity_zscore",
    "precursor_intensity_present",
)
ACQUISITION_SOURCE_DATASETS = (
    "acquisition_type",
    "positive polarity",
    "window lo",
    "window uo",
    "instrument accuracy est.",
    "precursor intensity",
)
SOURCE_DATASETS = tuple(name for name in FINAL_DATASETS if name != "training_eligible")
STRING_DATASETS = {"massive_id", "unique_spectrum_id"}

logger = logging.getLogger(__name__)


def _json_write(path: Path, value: Any) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp_path.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _splitmix64(values: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        values = values.astype(np.uint64, copy=False) + np.uint64(
            0x9E3779B97F4A7C15
        )
        values = (values ^ (values >> np.uint64(30))) * np.uint64(
            0xBF58476D1CE4E5B9
        )
        values = (values ^ (values >> np.uint64(27))) * np.uint64(
            0x94D049BB133111EB
        )
        return values ^ (values >> np.uint64(31))


def _project_hash(massive_id: bytes) -> np.uint64:
    digest = hashlib.blake2b(
        massive_id,
        digest_size=8,
        person=b"sl-msv2-v1",
    ).digest()
    return np.uint64(int.from_bytes(digest, "little"))


def _validation_mask(
    massive_id: bytes,
    source_rows: np.ndarray,
    group_ids: np.ndarray,
    global_group_ids: np.ndarray,
) -> np.ndarray:
    assigned = group_ids >= 0
    keys = source_rows.astype(np.uint64, copy=True)
    keys[assigned] = global_group_ids[assigned].astype(
        np.uint64,
        copy=False,
    )
    type_salt = np.where(
        assigned,
        np.uint64(0x47524F5550),
        np.uint64(0x535045435452554D),
    )
    values = keys ^ _project_hash(massive_id) ^ type_salt ^ np.uint64(
        SPLIT_SEED
    )
    return (
        _splitmix64(values) % np.uint64(SPLIT_MODULUS)
        == VALIDATION_REMAINDER
    )


def source_files(api: HfApi) -> list[str]:
    return sorted(
        filename
        for filename in api.list_repo_files(
            SOURCE_REPO_ID,
            repo_type="dataset",
            revision=SOURCE_REVISION,
        )
        if filename.endswith(SOURCE_SUFFIX)
    )


def download_source(source_dir: Path, filename: str) -> Path:
    local_path = source_dir / filename
    if local_path.exists():
        return local_path
    return Path(
        hf_hub_download(
            SOURCE_REPO_ID,
            filename,
            repo_type="dataset",
            revision=SOURCE_REVISION,
            local_dir=source_dir,
        )
    )


def _run_normalization_stats(
    file_ids: np.ndarray,
    retention_time: np.ndarray,
    precursor_intensity: np.ndarray,
    run_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rt_max = np.zeros(run_count, dtype=np.float32)
    positive_rt = np.isfinite(retention_time) & (retention_time > 0)
    np.maximum.at(
        rt_max,
        file_ids[positive_rt],
        retention_time[positive_rt],
    )

    positive_intensity = np.isfinite(precursor_intensity) & (
        precursor_intensity > 0
    )
    intensity_file_ids = file_ids[positive_intensity]
    log_intensity = np.log1p(
        precursor_intensity[positive_intensity]
    ).astype(np.float64)
    counts = np.bincount(intensity_file_ids, minlength=run_count)
    sums = np.bincount(
        intensity_file_ids,
        weights=log_intensity,
        minlength=run_count,
    )
    squared_sums = np.bincount(
        intensity_file_ids,
        weights=log_intensity * log_intensity,
        minlength=run_count,
    )
    nonempty = counts > 0
    mean = np.zeros(run_count, dtype=np.float32)
    std = np.zeros(run_count, dtype=np.float32)
    mean[nonempty] = (sums[nonempty] / counts[nonempty]).astype(np.float32)
    variance = np.zeros(run_count, dtype=np.float64)
    variance[nonempty] = (
        squared_sums[nonempty] / counts[nonempty]
        - np.square(sums[nonempty] / counts[nonempty])
    )
    std[nonempty] = np.sqrt(np.maximum(variance[nonempty], 0.0)).astype(
        np.float32
    )
    return rt_max, mean, std


def _read_source_arrays(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    with h5py.File(path, "r") as file:
        missing = sorted(set(SOURCE_DATASETS) - set(file))
        if missing:
            raise ValueError(f"Missing source datasets in {path}: {missing}")
        source_levels = file["MS level"][:]
        selected = source_levels == REQUIRED_MS_LEVEL
        source_rows = np.flatnonzero(selected).astype(np.int64)
        arrays = {name: file[name][:][selected] for name in SOURCE_DATASETS}
        if all(name in file for name in ACQUISITION_SOURCE_DATASETS):
            arrays.update(
                {
                    name: file[name][:][selected]
                    for name in ACQUISITION_SOURCE_DATASETS
                }
            )
            instrument_names = file["metadata/instrument name"][:]
            arrays["instrument_name"] = instrument_names[
                arrays["file_id"].astype(np.int64)
            ]
            full_file_ids = file["file_id"][:].astype(np.int64, copy=False)
            full_rt = file["RT"][:].astype(np.float32, copy=False)
            full_intensity = file["precursor intensity"][:].astype(
                np.float32,
                copy=False,
            )
            (
                run_rt_max,
                run_log_intensity_mean,
                run_log_intensity_std,
            ) = _run_normalization_stats(
                full_file_ids,
                full_rt,
                full_intensity,
                len(instrument_names),
            )
            selected_file_ids = arrays["file_id"].astype(np.int64, copy=False)
            arrays["run_rt_max"] = run_rt_max[selected_file_ids]
            arrays["run_log_intensity_mean"] = run_log_intensity_mean[
                selected_file_ids
            ]
            arrays["run_log_intensity_std"] = run_log_intensity_std[
                selected_file_ids
            ]
    arrays["spectrum"] = arrays["spectrum"].astype(np.float32)
    arrays["MS level"] = arrays["MS level"].astype(np.int8, copy=False)
    return arrays, source_rows


def _present(values: np.ndarray, *, positive: bool = False) -> np.ndarray:
    present = np.isfinite(values)
    if positive:
        present &= values > 0
    return present


def _categorical_ids(values: np.ndarray, mapper: Any) -> np.ndarray:
    unique, inverse = np.unique(values, return_inverse=True)
    ids = np.asarray(
        [mapper(bytes(value).decode()) for value in unique],
        dtype=np.int8,
    )
    return ids[inverse]


def _add_acquisition_metadata(arrays: dict[str, np.ndarray]) -> None:
    precursor = arrays["precursor_mz"].astype(np.float32, copy=False)
    collision = arrays["collision_energy"].astype(np.float32, copy=False)
    charge = arrays["charge"].astype(np.float32, copy=False)
    accuracy = arrays.pop("instrument accuracy est.").astype(np.float32, copy=False)
    intensity = arrays.pop("precursor intensity").astype(np.float32, copy=False)
    retention_time = arrays["RT"].astype(np.float32, copy=False)
    lower = arrays.pop("window lo").astype(np.float32, copy=False)
    upper = arrays.pop("window uo").astype(np.float32, copy=False)
    run_rt_max = arrays.pop("run_rt_max")
    run_log_intensity_mean = arrays.pop("run_log_intensity_mean")
    run_log_intensity_std = arrays.pop("run_log_intensity_std")

    arrays["precursor_mz_present"] = _present(precursor, positive=True)
    arrays["collision_energy_present"] = _present(collision, positive=True)
    arrays["charge_present"] = _present(charge, positive=True)
    raw_polarity = arrays.pop("positive polarity")
    arrays["polarity_id"] = np.where(
        raw_polarity == 1,
        1,
        np.where(raw_polarity == 0, 2, 0),
    ).astype(np.int8)
    arrays["acquisition_type_id"] = _categorical_ids(
        arrays.pop("acquisition_type"),
        acquisition_type_id,
    )
    isolation_present = _present(lower) & _present(upper) & (
        (lower > 0) | (upper > 0)
    )
    isolation_scale = np.float32(np.log(101.0))
    arrays["isolation_window_lower_offset"] = np.where(
        isolation_present,
        np.log1p(np.clip(lower, 0.0, 100.0)) / isolation_scale,
        0.0,
    ).astype(np.float32)
    arrays["isolation_window_upper_offset"] = np.where(
        isolation_present,
        np.log1p(np.clip(upper, 0.0, 100.0)) / isolation_scale,
        0.0,
    ).astype(np.float32)
    arrays["isolation_window_present"] = isolation_present
    arrays["instrument_family_id"] = _categorical_ids(
        arrays.pop("instrument_name"),
        instrument_family_id,
    )
    accuracy_present = _present(accuracy, positive=True)
    arrays["mass_accuracy"] = np.where(
        accuracy_present,
        (np.log10(np.clip(accuracy, 1e-6, 1e-1)) + 6.0) / 5.0,
        0.0,
    ).astype(np.float32)
    arrays["mass_accuracy_present"] = accuracy_present

    rt_fraction = np.zeros_like(retention_time)
    positive_rt = _present(retention_time, positive=True) & (run_rt_max > 0)
    rt_fraction[positive_rt] = (
        retention_time[positive_rt] / run_rt_max[positive_rt]
    )
    arrays["retention_time_fraction"] = rt_fraction
    arrays["retention_time_present"] = _present(retention_time, positive=True)
    intensity_zscore = np.zeros_like(intensity)
    positive_intensity = (
        _present(intensity, positive=True) & (run_log_intensity_std > 0)
    )
    intensity_zscore[positive_intensity] = np.clip(
        (
            np.log1p(intensity[positive_intensity])
            - run_log_intensity_mean[positive_intensity]
        )
        / run_log_intensity_std[positive_intensity],
        -5.0,
        5.0,
    ) / 5.0
    arrays["precursor_intensity_zscore"] = intensity_zscore
    arrays["precursor_intensity_present"] = _present(intensity, positive=True)


def _write_array_dataset(
    file: h5py.File,
    name: str,
    values: np.ndarray,
    *,
    resizable: bool,
) -> h5py.Dataset:
    trailing_shape = values.shape[1:]
    chunk_rows = (
        SPECTRUM_CHUNK_ROWS if name == "spectrum" else SCALAR_CHUNK_ROWS
    )
    leading_chunk = (
        chunk_rows
        if resizable
        else min(chunk_rows, max(1, len(values)))
    )
    chunks = (leading_chunk, *trailing_shape)
    maxshape = (None, *trailing_shape) if resizable else values.shape
    dtype = h5py.special_dtype(vlen=bytes) if name in STRING_DATASETS else values.dtype
    return file.create_dataset(
        name,
        data=values,
        dtype=dtype,
        maxshape=maxshape,
        chunks=chunks,
        compression="gzip",
        compression_opts=1,
    )


def prepare_source_arrays(
    source_path: Path,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    arrays, source_rows = _read_source_arrays(source_path)
    has_acquisition_metadata = "acquisition_type" in arrays
    if has_acquisition_metadata:
        _add_acquisition_metadata(arrays)
    precursor = arrays["precursor_mz"]
    retention_time = arrays["RT"]
    eligible = massive_v2_training_eligibility_numpy(
        arrays["spectrum"],
        precursor,
        retention_time,
    )
    arrays["training_eligible"] = eligible
    validation = np.zeros(len(source_rows), dtype=bool)
    for massive_id in np.unique(arrays["massive_id"]):
        project_rows = arrays["massive_id"] == massive_id
        validation[project_rows] = _validation_mask(
            bytes(massive_id),
            source_rows[project_rows],
            arrays["group_id"][project_rows],
            arrays["global_group_id"][project_rows],
        )
    assigned = arrays["group_id"] >= 0
    assigned_groups = sum(
        len(
            np.unique(
                arrays["global_group_id"][
                    assigned & (arrays["massive_id"] == massive_id)
                ]
            )
        )
        for massive_id in np.unique(arrays["massive_id"][assigned])
    )
    stats = {
        "rows": len(source_rows),
        "eligible_rows": int(eligible.sum()),
        "train_rows": int((~validation).sum()),
        "train_eligible_rows": int((eligible & ~validation).sum()),
        "validation_rows": int(validation.sum()),
        "validation_eligible_rows": int((eligible & validation).sum()),
        "assigned_rows": int(assigned.sum()),
        "assigned_groups": int(assigned_groups),
        "metadata_schema": MASSIVE_V2_ACQUISITION_SCHEMA if has_acquisition_metadata else "",
    }
    return arrays, validation, stats


@dataclass
class WriterState:
    shard_index: int
    rows: int
    eligible_rows: int
    completed: list[dict[str, Any]]


class SplitShardWriter:
    def __init__(
        self,
        output_dir: Path,
        split: str,
        state: WriterState | None = None,
        *,
        target_rows: int = TARGET_ROWS_PER_SHARD,
    ) -> None:
        self.output_dir = output_dir / split
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.target_rows = target_rows
        self.state = state or WriterState(0, 0, 0, [])
        self.file: h5py.File | None = None
        self.datasets: dict[str, h5py.Dataset] = {}
        self._recover()

    def _partial_path(self) -> Path:
        return self.output_dir / f".partial_{self.state.shard_index:05d}.hdf5"

    def _final_path(self) -> Path:
        return self.output_dir / f"shard_{self.state.shard_index:05d}.hdf5"

    def _recover(self) -> None:
        expected_partial = self._partial_path()
        for path in self.output_dir.glob(".partial_*.hdf5"):
            if not self.state.rows or path != expected_partial:
                path.unlink()
        if self.state.rows and expected_partial.exists():
            self._open_partial()
            if any(
                dataset.shape[0] != self.state.rows
                for dataset in self.datasets.values()
            ):
                self._rewrite_partial(self.state.rows)
            self.state.eligible_rows = int(
                self.datasets["training_eligible"][:].sum()
            )
            return
        for path in self.output_dir.glob("shard_*.hdf5"):
            index = int(path.stem.rsplit("_", 1)[1])
            if index < self.state.shard_index:
                continue
            if index == self.state.shard_index and self.state.rows:
                path.replace(self._partial_path())
                self._open_partial()
                self._resize(self.state.rows)
                return
            path.unlink()
        if self.state.rows:
            self._open_partial()

    def _rewrite_partial(self, rows: int) -> None:
        assert self.file is not None
        path = self._partial_path()
        recovery_path = path.with_suffix(".recovery.hdf5")
        with h5py.File(recovery_path, "w") as recovered:
            for name, dataset in self.datasets.items():
                _write_array_dataset(
                    recovered,
                    name,
                    dataset[:rows],
                    resizable=True,
                )
        self.file.close()
        recovery_path.replace(path)
        self._open_partial()

    def _open_partial(self) -> None:
        path = self._partial_path()
        self.file = h5py.File(path, "a")
        self.datasets = {name: dataset for name, dataset in self.file.items()}

    def _ensure_file(self, arrays: dict[str, np.ndarray]) -> None:
        if self.file is not None:
            return
        self.file = h5py.File(self._partial_path(), "w")
        self.datasets = {}
        for name in arrays:
            values = arrays[name][:0]
            self.datasets[name] = _write_array_dataset(
                self.file,
                name,
                values,
                resizable=True,
            )

    def _resize(self, rows: int) -> None:
        for dataset in self.datasets.values():
            dataset.resize(rows, axis=0)
        self.state.rows = rows
        self.state.eligible_rows = int(
            self.datasets["training_eligible"][:].sum()
        )
        assert self.file is not None
        self.file.flush()

    def append(
        self,
        arrays: dict[str, np.ndarray],
        indices: np.ndarray,
        *,
        atomic_boundaries: np.ndarray | None = None,
    ) -> None:
        position = 0
        while position < len(indices):
            self._ensure_file(arrays)
            remaining = self.target_rows - self.state.rows
            stop = min(len(indices), position + remaining)
            if atomic_boundaries is not None and stop < len(indices):
                boundary_position = np.searchsorted(
                    atomic_boundaries,
                    stop,
                    side="right",
                ) - 1
                stop = int(atomic_boundaries[boundary_position])
                if stop == position:
                    if self.state.rows:
                        self._finish_shard()
                        continue
                    stop = int(
                        atomic_boundaries[
                            np.searchsorted(
                                atomic_boundaries,
                                position,
                                side="right",
                            )
                        ]
                    )
            selected = indices[position:stop]
            old_rows = self.state.rows
            new_rows = old_rows + len(selected)
            for name, dataset in self.datasets.items():
                dataset.resize(new_rows, axis=0)
                dataset[old_rows:new_rows] = arrays[name][selected]
            self.state.rows = new_rows
            self.state.eligible_rows += int(
                arrays["training_eligible"][selected].sum()
            )
            position = stop
            if self.state.rows >= self.target_rows:
                self._finish_shard()

    def _finish_shard(self) -> None:
        if self.file is None or self.state.rows == 0:
            return
        path = self._partial_path()
        self.file.close()
        self.file = None
        self.datasets = {}
        final_path = self._final_path()
        path.replace(final_path)
        self.state.completed.append(
            {
                "path": f"{self.split}/{final_path.name}",
                "rows": self.state.rows,
                "eligible_rows": self.state.eligible_rows,
                "bytes": final_path.stat().st_size,
                "sha256": _sha256(final_path),
                "chunk_rows": SPECTRUM_CHUNK_ROWS,
            }
        )
        self.state.shard_index += 1
        self.state.rows = 0
        self.state.eligible_rows = 0

    def flush(self) -> None:
        if self.file is not None:
            self.file.flush()

    def finish(self) -> None:
        self._finish_shard()


def _writer_state(value: dict[str, Any] | None) -> WriterState | None:
    return None if value is None else WriterState(**value)


def _state_payload(
    next_project_index: int,
    train_writer: SplitShardWriter,
    validation_writer: SplitShardWriter,
) -> dict[str, Any]:
    return {
        "conversion_version": CONVERSION_VERSION,
        "next_project_index": next_project_index,
        "train": vars(train_writer.state),
        "validation": vars(validation_writer.state),
    }


def _append_project_split(
    writer: SplitShardWriter,
    arrays: dict[str, np.ndarray],
    selected: np.ndarray,
) -> None:
    group_ids = arrays["group_id"]
    global_group_ids = arrays["global_group_id"]
    assigned = selected[group_ids[selected] >= 0]
    if len(assigned):
        massive_ids = arrays["massive_id"]
        order = np.lexsort(
            (
                assigned,
                global_group_ids[assigned],
                massive_ids[assigned],
            )
        )
        assigned = assigned[order]
        sorted_groups = global_group_ids[assigned]
        sorted_massive_ids = massive_ids[assigned]
        boundaries = np.concatenate(
            (
                np.zeros(1, dtype=np.int64),
                np.flatnonzero(
                    (sorted_massive_ids[1:] != sorted_massive_ids[:-1])
                    | (sorted_groups[1:] != sorted_groups[:-1])
                )
                + 1,
                np.asarray([len(assigned)], dtype=np.int64),
            )
        )
        writer.append(
            arrays,
            assigned,
            atomic_boundaries=boundaries,
        )
    unassigned = selected[group_ids[selected] < 0]
    if len(unassigned):
        writer.append(arrays, unassigned)


def convert_worker(
    work_dir: Path,
    output_dir: Path,
    filenames: list[str],
    worker_index: int,
    target_rows: int,
) -> dict[str, Any]:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    worker_name = f"worker_{worker_index:02d}"
    worker_dir = work_dir / "workers" / worker_name
    worker_dir.mkdir(parents=True, exist_ok=True)
    result_path = worker_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result.get("conversion_version") != CONVERSION_VERSION:
            raise ValueError(
                f"Stale conversion result in {result_path}; use a new work dir"
            )
        return result
    source_dir = worker_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    state_path = worker_dir / "state.json"
    stats_path = worker_dir / "stats.json"
    saved = json.loads(state_path.read_text()) if state_path.exists() else {}
    if saved and saved.get("conversion_version") != CONVERSION_VERSION:
        raise ValueError(
            f"Stale conversion state in {state_path}; use a new work dir"
        )
    stats_by_file = {
        item["source_file"]: item
        for item in (
            json.loads(stats_path.read_text())
            if stats_path.exists()
            else []
        )
    }
    train_writer = SplitShardWriter(
        output_dir,
        f"train/{worker_name}",
        _writer_state(saved.get("train")),
        target_rows=target_rows,
    )
    validation_writer = SplitShardWriter(
        output_dir,
        f"validation/{worker_name}",
        _writer_state(saved.get("validation")),
        target_rows=target_rows,
    )
    start = int(saved.get("next_project_index", 0))

    for project_index in range(start, len(filenames)):
        filename = filenames[project_index]
        logger.info(
            "Worker %02d converting project %d/%d: %s",
            worker_index,
            project_index + 1,
            len(filenames),
            filename,
        )
        source_path = download_source(source_dir, filename)
        arrays, validation, stats = prepare_source_arrays(source_path)
        stats_by_file[filename] = {
            "source_file": filename,
            **stats,
        }
        _append_project_split(
            train_writer,
            arrays,
            np.flatnonzero(~validation),
        )
        _append_project_split(
            validation_writer,
            arrays,
            np.flatnonzero(validation),
        )
        train_writer.flush()
        validation_writer.flush()
        _json_write(
            stats_path,
            [
                stats_by_file[name]
                for name in filenames
                if name in stats_by_file
            ],
        )
        _json_write(
            state_path,
            _state_payload(
                project_index + 1,
                train_writer,
                validation_writer,
            ),
        )
        source_path.unlink()

    train_writer.finish()
    validation_writer.finish()
    result = {
        "conversion_version": CONVERSION_VERSION,
        "stats": [stats_by_file[name] for name in filenames],
        "train_shards": train_writer.state.completed,
        "validation_shards": validation_writer.state.completed,
    }
    _json_write(result_path, result)
    return result


def _shard_record(output_dir: Path, relative_path: str) -> dict[str, Any]:
    path = output_dir / relative_path
    with h5py.File(path, "r") as file:
        rows = len(file["spectrum"])
        eligible_rows = int(file["training_eligible"][:].sum())
        chunk_rows = int(file["spectrum"].chunks[0])
    return {
        "path": relative_path,
        "rows": rows,
        "eligible_rows": eligible_rows,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "chunk_rows": chunk_rows,
    }


def _regrouping_projects(
    output_dir: Path,
    shards: list[dict[str, Any]],
) -> tuple[set[bytes], dict[bytes, set[str]]]:
    last_group: dict[bytes, int] = {}
    project_paths: dict[bytes, set[str]] = {}
    regroup = set()
    for shard in shards:
        path = output_dir / shard["path"]
        with h5py.File(path, "r") as file:
            massive_ids = file["massive_id"][:]
            group_ids = file["group_id"][:]
            global_ids = file["global_group_id"][:]
        assigned = group_ids >= 0
        for massive_id in np.unique(massive_ids[assigned]):
            project_key = bytes(massive_id)
            project_paths.setdefault(project_key, set()).add(
                str(shard["path"])
            )
            groups = global_ids[assigned & (massive_ids == massive_id)]
            previous = last_group.get(project_key)
            if (
                np.any(np.diff(groups) < 0)
                or previous is not None
                and int(groups[0]) <= previous
            ):
                regroup.add(project_key)
            last_group[project_key] = max(
                int(groups[-1]),
                previous if previous is not None else int(groups[-1]),
            )
    return regroup, project_paths


def _collect_project_arrays(
    backup_dir: Path,
    relative_paths: list[str],
    project: bytes,
) -> dict[str, np.ndarray]:
    pieces: dict[str, list[np.ndarray]] = {
        name: [] for name in FINAL_DATASETS
    }
    for relative_path in relative_paths:
        with h5py.File(backup_dir / relative_path, "r") as file:
            massive_ids = file["massive_id"][:]
            group_ids = file["group_id"][:]
            selected = (massive_ids == project) & (group_ids >= 0)
            if np.any(selected):
                for name in FINAL_DATASETS:
                    pieces[name].append(file[name][:][selected])
    arrays = {
        name: np.concatenate(values)
        for name, values in pieces.items()
    }
    order = np.argsort(arrays["global_group_id"], kind="stable")
    return {
        name: values[order]
        for name, values in arrays.items()
    }


def _rewrite_regrouped_source_shard(
    backup_path: Path,
    output_path: Path,
    projects: np.ndarray,
) -> dict[str, Any] | None:
    with h5py.File(backup_path, "r") as source:
        massive_ids = source["massive_id"][:]
        group_ids = source["group_id"][:]
        keep = ~(
            (group_ids >= 0)
            & np.isin(massive_ids, projects)
        )
        arrays = {
            name: source[name][:][keep]
            for name in FINAL_DATASETS
        }
    if not np.any(keep):
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".regrouping.hdf5")
    with h5py.File(temporary_path, "w") as output:
        for name, values in arrays.items():
            _write_array_dataset(
                output,
                name,
                values,
                resizable=True,
            )
    temporary_path.replace(output_path)
    return _shard_record(
        output_path.parents[2],
        str(output_path.relative_to(output_path.parents[2])),
    )


def regroup_cross_shard_projects(
    work_dir: Path,
    output_dir: Path,
    split: str,
    shards: list[dict[str, Any]],
    *,
    target_rows: int,
) -> list[dict[str, Any]]:
    regroup_dir = work_dir / "regrouping" / split
    regroup_dir.mkdir(parents=True, exist_ok=True)
    result_path = regroup_dir / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    plan_path = regroup_dir / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
    else:
        projects, project_paths = _regrouping_projects(
            output_dir,
            shards,
        )
        if not projects:
            _json_write(result_path, shards)
            return shards
        plan = {
            "projects": sorted(
                project.decode()
                for project in projects
            ),
            "project_paths": {
                project.decode(): sorted(project_paths[project])
                for project in projects
            },
            "affected_paths": sorted(
                {
                    path
                    for project in projects
                    for path in project_paths[project]
                }
            ),
            "shards": shards,
        }
        _json_write(plan_path, plan)

    backup_dir = regroup_dir / "originals"
    backup_ready = regroup_dir / "backups_ready"
    if not backup_ready.exists():
        for relative_path in plan["affected_paths"]:
            backup_path = backup_dir / relative_path
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            if not backup_path.exists():
                os.link(output_dir / relative_path, backup_path)
        backup_ready.touch()

    for relative_path in plan["affected_paths"]:
        (output_dir / relative_path).unlink(missing_ok=True)
    repair_output = output_dir / split / "regrouped"
    if repair_output.exists():
        shutil.rmtree(repair_output)

    projects = np.asarray(
        [project.encode() for project in plan["projects"]],
    )
    rewritten: dict[str, dict[str, Any]] = {}
    for relative_path in plan["affected_paths"]:
        record = _rewrite_regrouped_source_shard(
            backup_dir / relative_path,
            output_dir / relative_path,
            projects,
        )
        if record is not None:
            rewritten[relative_path] = record

    writer = SplitShardWriter(
        output_dir,
        f"{split}/regrouped",
        target_rows=target_rows,
    )
    for project in plan["projects"]:
        arrays = _collect_project_arrays(
            backup_dir,
            plan["project_paths"][project],
            project.encode(),
        )
        _append_project_split(
            writer,
            arrays,
            np.arange(len(arrays["spectrum"]), dtype=np.int64),
        )
    writer.finish()

    updated = [
        (
            rewritten[str(shard["path"])]
            if str(shard["path"]) in rewritten
            else shard
        )
        for shard in plan["shards"]
        if (
            str(shard["path"]) not in plan["affected_paths"]
            or str(shard["path"]) in rewritten
        )
    ]
    updated.extend(writer.state.completed)
    _json_write(result_path, updated)
    return updated


def remove_regrouping_backups(work_dir: Path) -> None:
    for split in ("train", "validation"):
        backup_dir = work_dir / "regrouping" / split / "originals"
        if backup_dir.exists():
            shutil.rmtree(backup_dir)


def convert_streaming(
    work_dir: Path,
    *,
    limit: int | None = None,
    target_rows: int = TARGET_ROWS_PER_SHARD,
    workers: int = 1,
) -> Path:
    filenames = source_files(HfApi())
    if limit is not None:
        filenames = filenames[:limit]
    output_dir = work_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["conversion"].get("version") != CONVERSION_VERSION:
            raise ValueError(
                f"Stale conversion artifact in {manifest_path}; "
                "use a new work dir"
            )
        return manifest_path
    assignments = [filenames[index::workers] for index in range(workers)]
    if workers == 1:
        results = [
            convert_worker(
                work_dir,
                output_dir,
                assignments[0],
                0,
                target_rows,
            )
        ]
    else:
        context = multiprocessing.get_context("spawn")
        processes = [
            context.Process(
                target=convert_worker,
                args=(
                    work_dir,
                    output_dir,
                    assignment,
                    worker_index,
                    target_rows,
                ),
            )
            for worker_index, assignment in enumerate(assignments)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join()
            if process.exitcode:
                raise RuntimeError(
                    f"Conversion worker {process.name} exited "
                    f"with code {process.exitcode}"
                )
        results = [
            json.loads(
                (
                    work_dir
                    / "workers"
                    / f"worker_{worker_index:02d}"
                    / "result.json"
                ).read_text()
            )
            for worker_index in range(workers)
        ]
    stats_by_file = {
        item["source_file"]: item
        for result in results
        for item in result["stats"]
    }
    train_shards = [
        shard
        for result in results
        for shard in result["train_shards"]
    ]
    validation_shards = [
        shard
        for result in results
        for shard in result["validation_shards"]
    ]
    train_shards = regroup_cross_shard_projects(
        work_dir,
        output_dir,
        "train",
        train_shards,
        target_rows=target_rows,
    )
    validation_shards = regroup_cross_shard_projects(
        work_dir,
        output_dir,
        "validation",
        validation_shards,
        target_rows=target_rows,
    )
    manifest = build_manifest(
        filenames,
        [stats_by_file[name] for name in filenames],
        train_shards,
        validation_shards,
        target_rows=target_rows,
    )
    _json_write(manifest_path, manifest)
    (output_dir / "README.md").write_text(dataset_card(manifest))
    return manifest_path


def build_manifest(
    filenames: list[str],
    prepared_stats: list[dict[str, Any]],
    train_shards: list[dict[str, Any]],
    validation_shards: list[dict[str, Any]],
    *,
    target_rows: int,
) -> dict[str, Any]:
    total = lambda key: sum(int(item[key]) for item in prepared_stats)
    metadata_schema = prepared_stats[0].get("metadata_schema", "")
    row_aligned_datasets = list(FINAL_DATASETS)
    metadata: dict[str, Any] | None = None
    if metadata_schema:
        row_aligned_datasets.extend(ACQUISITION_DATASETS)
        metadata = {
            "schema": MASSIVE_V2_ACQUISITION_SCHEMA,
            "condition_dim": 28,
            "columns": {
                "precursor_mz": "precursor_mz",
                **{name: name for name in ACQUISITION_DATASETS},
                "collision_energy": "collision_energy",
                "charge": "charge",
            },
            "categorical_vocabularies": {
                "polarity": ["unknown", "positive", "negative"],
                "acquisition_type": ["unknown", "dda", "dia"],
                "instrument_family": list(INSTRUMENT_FAMILIES),
            },
            "normalization": {
                "precursor_mz": "value / 1000",
                "collision_energy": "value / 100",
                "charge": "value / 21",
                "isolation_offsets": "log1p(clip(value, 0, 100)) / log(101)",
                "mass_accuracy": "(log10(clip(value, 1e-6, 1e-1)) + 6) / 5",
                "retention_time": "value / max_positive_value_within_file",
                "precursor_intensity": "clip(zscore(log1p(value)), -5, 5) / 5 within file",
            },
        }
    manifest = {
        "format": MASSIVE_V2_HDF5_FORMAT,
        "source": {
            "repo_id": SOURCE_REPO_ID,
            "revision": SOURCE_REVISION,
            "filename_suffix": SOURCE_SUFFIX,
            "file_count": len(filenames),
        },
        "conversion": {
            "version": CONVERSION_VERSION,
            "exact_ms_level": REQUIRED_MS_LEVEL,
            "target_rows_per_shard": target_rows,
            "spectrum_dtype": "float32",
            "compression": "gzip",
            "compression_level": 1,
        },
        "datasets": {
            "spectrum": "spectrum",
            "precursor_mz": "precursor_mz",
            "retention_time": "RT",
            "ms_level": "MS level",
        },
        "row_aligned_datasets": row_aligned_datasets,
        "eligibility": massive_v2_eligibility_contract(),
        "split": {
            "version": "entity_hash_v1",
            "algorithm": "splitmix64",
            "seed": SPLIT_SEED,
            "modulus": SPLIT_MODULUS,
            "validation_remainder": VALIDATION_REMAINDER,
            "assigned_entity_key": ["massive_id", "global_group_id"],
            "unassigned_entity_key": ["massive_id", "source_row_index"],
        },
        "grouping": {
            "assigned_rule": "group_id >= 0",
            "corpus_entity_key": ["massive_id", "global_group_id"],
            "raw_global_group_id_formula": "file_id * 2**32 + group_id",
            "assigned_rows": total("assigned_rows"),
            "assigned_group_occurrences_within_source_files": total(
                "assigned_groups"
            ),
        },
        "rows": total("rows"),
        "eligible_rows": total("eligible_rows"),
        "splits": {
            "train": {
                "rows": total("train_rows"),
                "eligible_rows": total("train_eligible_rows"),
                "shards": train_shards,
            },
            "validation": {
                "rows": total("validation_rows"),
                "eligible_rows": total("validation_eligible_rows"),
                "shards": validation_shards,
            },
        },
    }
    if metadata is not None:
        manifest["spectrum_metadata"] = metadata
    return manifest


def dataset_card(manifest: dict[str, Any]) -> str:
    return f"""---
license: other
task_categories:
- feature-extraction
---

# MassIVE v2 exact-MS2 training shards

This dataset is a training-oriented repack of
[`{SOURCE_REPO_ID}`](https://huggingface.co/datasets/{SOURCE_REPO_ID}) at
revision `{SOURCE_REVISION}`. It includes only source files ending in
`{SOURCE_SUFFIX}` and retains every row whose `MS level` is exactly 2.

- Rows: {manifest['rows']:,}
- Eligible training rows: {manifest['eligible_rows']:,}
- Train shards: {len(manifest['splits']['train']['shards']):,}
- Validation shards: {len(manifest['splits']['validation']['shards']):,}

The `training_eligible` column records the canonical precursor, retention-time,
and usable-spectrum policy documented in `manifest.json`. Every eligible row
has enough peaks for distinct context and target masks. Assigned same-entity
groups are kept within one split and one shard. The collision-safe corpus
grouping key is `(massive_id, global_group_id)`; `global_group_id` alone is
only unique within one MassIVE project.
"""


def _validate_artifact_shard(
    args: tuple[Path, str, dict[str, Any], list[str]],
) -> tuple[str, int, int, dict[bytes, tuple[int, int]]]:
    artifact_dir, split, shard, row_aligned_datasets = args
    path = artifact_dir / shard["path"]
    if path.stat().st_size != shard["bytes"]:
        raise ValueError(f"Shard size mismatch: {path}")
    if _sha256(path) != shard["sha256"]:
        raise ValueError(f"Shard checksum mismatch: {path}")
    with h5py.File(path, "r") as file:
        lengths = {len(file[name]) for name in row_aligned_datasets}
        if lengths != {int(shard["rows"])}:
            raise ValueError(f"Row alignment mismatch: {path}")
        spectrum = file["spectrum"]
        if spectrum.dtype != np.dtype(np.float32) or tuple(
            spectrum.shape[1:]
        ) != (2, 128):
            raise ValueError(f"Spectrum contract mismatch: {path}")
        levels = file["MS level"][:]
        if np.any(levels != REQUIRED_MS_LEVEL):
            raise ValueError(f"Non-MS2 row in {path}")
        eligible = file["training_eligible"][:]
        for start in range(0, len(eligible), SCALAR_CHUNK_ROWS):
            stop = min(start + SCALAR_CHUNK_ROWS, len(eligible))
            expected_eligible = massive_v2_training_eligibility_numpy(
                spectrum[start:stop],
                file["precursor_mz"][start:stop],
                file["RT"][start:stop],
            )
            if not np.array_equal(
                eligible[start:stop],
                expected_eligible,
            ):
                raise ValueError(f"Eligibility mask mismatch: {path}")
        eligible_rows = int(eligible.sum())
        if eligible_rows != int(shard["eligible_rows"]):
            raise ValueError(f"Eligibility mismatch: {path}")
        massive_ids = file["massive_id"][:]
        file_ids = file["file_id"][:].astype(np.int64)
        group_ids = file["group_id"][:]
        global_ids = file["global_group_id"][:]
        assigned = group_ids >= 0
        expected_global_ids = (
            file_ids[assigned] * np.int64(1 << 32)
            + group_ids[assigned].astype(np.int64)
        )
        if not np.array_equal(
            global_ids[assigned],
            expected_global_ids,
        ):
            raise ValueError(f"Invalid global_group_id in {path}")
        project_bounds: dict[bytes, tuple[int, int]] = {}
        for massive_id in np.unique(massive_ids[assigned]):
            project_rows = assigned & (massive_ids == massive_id)
            project_groups = global_ids[project_rows]
            if np.any(np.diff(project_groups) < 0):
                raise ValueError(f"Groups are not contiguous in {path}")
            project_key = bytes(massive_id)
            project_bounds[project_key] = (
                int(project_groups[0]),
                int(project_groups[-1]),
            )
            expected_validation = _validation_mask(
                project_key,
                np.zeros(len(project_groups), dtype=np.int64),
                np.ones(len(project_groups), dtype=np.int32),
                project_groups,
            )
            if np.any(expected_validation != (split == "validation")):
                raise ValueError(
                    f"Group assigned to the wrong split in {path}"
                )
    return str(shard["path"]), int(shard["rows"]), eligible_rows, project_bounds


def validate_artifact(manifest_path: Path, *, workers: int = 1) -> None:
    manifest = json.loads(manifest_path.read_text())
    if manifest["format"] != MASSIVE_V2_HDF5_FORMAT:
        raise ValueError(f"Unknown artifact format in {manifest_path}")
    expected_eligibility = massive_v2_eligibility_contract()
    if manifest["eligibility"] != expected_eligibility:
        raise ValueError(
            "Artifact eligibility contract mismatch: "
            f"expected {expected_eligibility}, got {manifest['eligibility']}"
        )
    tasks: list[tuple[Path, str, dict[str, Any], list[str]]] = []
    seen_paths: set[str] = set()
    for split in ("train", "validation"):
        for shard in manifest["splits"][split]["shards"]:
            if shard["path"] in seen_paths:
                raise ValueError(f"Duplicate shard path: {shard['path']}")
            seen_paths.add(shard["path"])
            tasks.append(
                (
                    manifest_path.parent,
                    split,
                    shard,
                    manifest.get("row_aligned_datasets", list(FINAL_DATASETS)),
                )
            )
    if workers == 1:
        results = [_validate_artifact_shard(task) for task in tasks]
    else:
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            results = pool.map(
                _validate_artifact_shard,
                tasks,
                chunksize=1,
            )

    result_index = 0
    total_rows = 0
    total_eligible = 0
    last_group_by_split: dict[str, dict[bytes, int]] = {
        "train": {},
        "validation": {},
    }
    for split in ("train", "validation"):
        split_rows = 0
        split_eligible = 0
        for shard in manifest["splits"][split]["shards"]:
            path, rows, eligible_rows, project_bounds = results[result_index]
            result_index += 1
            for project_key, (first_group, last_group) in project_bounds.items():
                previous = last_group_by_split[split].get(project_key)
                if previous is not None and first_group <= previous:
                    raise ValueError(
                        f"Group crosses shards in {manifest_path.parent / path}: "
                        f"{project_key!r}/{first_group}"
                    )
                last_group_by_split[split][project_key] = last_group
            split_rows += rows
            split_eligible += eligible_rows
        if split_rows != int(manifest["splits"][split]["rows"]):
            raise ValueError(f"{split} row total mismatch")
        if split_eligible != int(
            manifest["splits"][split]["eligible_rows"]
        ):
            raise ValueError(f"{split} eligibility total mismatch")
        total_rows += split_rows
        total_eligible += split_eligible
    if total_rows != int(manifest["rows"]):
        raise ValueError("Artifact row total mismatch")
    if total_eligible != int(manifest["eligible_rows"]):
        raise ValueError("Artifact eligibility total mismatch")


def validate_replacement_totals(
    candidate_manifest_path: Path,
    baseline_manifest_path: Path,
) -> None:
    candidate = json.loads(candidate_manifest_path.read_text())
    baseline = json.loads(baseline_manifest_path.read_text())
    keys = (
        ("rows",),
        ("eligible_rows",),
        ("splits", "train", "rows"),
        ("splits", "train", "eligible_rows"),
        ("splits", "validation", "rows"),
        ("splits", "validation", "eligible_rows"),
    )
    candidate_totals = {
        "/".join(key): int(candidate[key[0]])
        if len(key) == 1
        else int(candidate[key[0]][key[1]][key[2]])
        for key in keys
    }
    baseline_totals = {
        "/".join(key): int(baseline[key[0]])
        if len(key) == 1
        else int(baseline[key[0]][key[1]][key[2]])
        for key in keys
    }
    if candidate_totals != baseline_totals:
        raise ValueError(
            "Replacement artifact totals differ from the immutable baseline: "
            f"candidate={candidate_totals}, baseline={baseline_totals}"
        )


def upload_artifact(output_dir: Path, repo_id: str) -> str:
    api = HfApi()
    api.create_repo(
        repo_id,
        repo_type="dataset",
        private=False,
        exist_ok=True,
    )
    local_shards = {
        str(path.relative_to(output_dir))
        for split in ("train", "validation")
        for path in (output_dir / split).glob("*.hdf5")
    }
    stale_shards = [
        path
        for path in api.list_repo_files(repo_id, repo_type="dataset")
        if path.endswith(".hdf5")
        and path.split("/", 1)[0] in {"train", "validation"}
        and path not in local_shards
    ]
    if stale_shards:
        api.delete_files(
            repo_id,
            stale_shards,
            repo_type="dataset",
            commit_message="Remove superseded MassIVE v2 shards",
        )
    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=output_dir,
        allow_patterns=[
            "README.md",
            "manifest.json",
            "train/*.hdf5",
            "validation/*.hdf5",
        ],
        num_workers=8,
    )
    return str(api.dataset_info(repo_id).sha)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build rank-friendly exact-MS2 shards from MassIVE v2."
    )
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--target-rows-per-shard",
        type=int,
        default=TARGET_ROWS_PER_SHARD,
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--validate-only",
        type=Path,
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args()
    if args.validate_only is not None:
        validate_artifact(args.validate_only, workers=args.workers)
        return
    args.work_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = convert_streaming(
        args.work_dir,
        limit=args.limit,
        target_rows=args.target_rows_per_shard,
        workers=args.workers,
    )
    validate_artifact(manifest_path, workers=args.workers)
    remove_regrouping_backups(args.work_dir)


if __name__ == "__main__":
    main()
