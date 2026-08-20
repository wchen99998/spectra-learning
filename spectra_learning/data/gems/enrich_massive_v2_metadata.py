from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from huggingface_hub import HfApi, hf_hub_download

from spectra_learning.data.gems.prepare_massive_v2 import (
    ACQUISITION_DATASETS,
    CONVERSION_VERSION,
    SCALAR_CHUNK_ROWS,
    _add_acquisition_metadata,
    _json_write,
    _run_normalization_stats,
    _validation_mask,
    dataset_card,
    download_source,
    source_files,
    upload_artifact,
    validate_artifact,
    validate_replacement_totals,
)
from spectra_learning.models.spectrum_metadata import (
    INSTRUMENT_FAMILIES,
    MASSIVE_V2_ACQUISITION_SCHEMA,
)

DEFAULT_ARTIFACT_DIR = Path(
    "/mnt/data/massive-v2-ms2-usable-peaks-v5/artifact"
)
DEFAULT_WORK_DIR = Path("/mnt/data/massive-v2-metadata-enrichment-v1")
DEFAULT_DESTINATION_REPO_ID = (
    "novogaia/massive-v2-ms2-t095-l080-sharded-10gb"
)
BASELINE_REVISION = "4de48add4e687f6ea561dc6ec74f8984ad8aebe0"
INDEX_ROWS = 1 << 20
WRITE_ROWS = 1 << 20

logger = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_project_metadata(
    source_path: Path,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    with h5py.File(source_path, "r") as file:
        levels = file["MS level"][:]
        selected = levels == 2
        source_rows = np.flatnonzero(selected).astype(np.int64)
        selected_names = (
            "RT",
            "precursor_mz",
            "collision_energy",
            "charge",
            "massive_id",
            "file_id",
            "group_id",
            "global_group_id",
            "unique_spectrum_id",
            "acquisition_type",
            "positive polarity",
            "window lo",
            "window uo",
            "instrument accuracy est.",
        )
        arrays = {name: file[name][:][selected] for name in selected_names}
        instrument_names = file["metadata/instrument name"][:]
        selected_file_ids = arrays["file_id"].astype(np.int64, copy=False)
        arrays["instrument_name"] = instrument_names[selected_file_ids]

        full_file_ids = file["file_id"][:].astype(np.int64, copy=False)
        full_rt = file["RT"][:].astype(np.float32, copy=False)
        run_count = len(instrument_names)
        rt_max = _run_normalization_stats(
            full_file_ids,
            full_rt,
            run_count,
        )
        arrays["run_rt_max"] = rt_max[selected_file_ids]

    _add_acquisition_metadata(arrays)
    validation = np.zeros(len(source_rows), dtype=np.bool_)
    for massive_id in np.unique(arrays["massive_id"]):
        project = arrays["massive_id"] == massive_id
        validation[project] = _validation_mask(
            bytes(massive_id),
            source_rows[project],
            arrays["group_id"][project],
            arrays["global_group_id"][project],
        )
    return arrays, validation


def ordered_project_indices(
    arrays: dict[str, np.ndarray],
    validation: np.ndarray,
    split: str,
) -> tuple[np.ndarray, int]:
    selected = np.flatnonzero(validation if split == "validation" else ~validation)
    assigned = selected[arrays["group_id"][selected] >= 0]
    order = np.lexsort(
        (
            assigned,
            arrays["global_group_id"][assigned],
            arrays["massive_id"][assigned],
        )
    )
    assigned = assigned[order]
    unassigned = selected[arrays["group_id"][selected] < 0]
    return np.concatenate((assigned, unassigned)), len(assigned)


def index_project_runs(
    manifest_path: Path,
    index_path: Path,
) -> dict[str, list[dict[str, Any]]]:
    if index_path.exists():
        return json.loads(index_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    runs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for split in ("train", "validation"):
        for shard in manifest["splits"][split]["shards"]:
            relative_path = str(shard["path"])
            path = manifest_path.parent / relative_path
            logger.info("Indexing %s", relative_path)
            with h5py.File(path, "r") as file:
                dataset = file["massive_id"]
                run_project: bytes | None = None
                run_start = 0
                for start in range(0, len(dataset), INDEX_ROWS):
                    stop = min(start + INDEX_ROWS, len(dataset))
                    values = dataset[start:stop]
                    boundaries = np.flatnonzero(values[1:] != values[:-1]) + 1
                    if run_project is None:
                        run_project = bytes(values[0])
                    elif bytes(values[0]) != run_project:
                        runs[run_project.decode()].append(
                            {
                                "split": split,
                                "path": relative_path,
                                "start": run_start,
                                "stop": start,
                            }
                        )
                        run_project = bytes(values[0])
                        run_start = start
                    for boundary in boundaries:
                        absolute = start + int(boundary)
                        runs[run_project.decode()].append(
                            {
                                "split": split,
                                "path": relative_path,
                                "start": run_start,
                                "stop": absolute,
                            }
                        )
                        run_project = bytes(values[boundary])
                        run_start = absolute
                if run_project is not None:
                    runs[run_project.decode()].append(
                        {
                            "split": split,
                            "path": relative_path,
                            "start": run_start,
                            "stop": len(dataset),
                        }
                    )
    value = dict(sorted(runs.items()))
    _json_write(index_path, value)
    return value


def _ensure_metadata_datasets(file: h5py.File, rows: int) -> None:
    dtypes = {
        "precursor_mz_present": np.bool_,
        "collision_energy_present": np.bool_,
        "charge_present": np.bool_,
        "polarity_id": np.int8,
        "acquisition_type_id": np.int8,
        "isolation_window_lower_offset": np.float32,
        "isolation_window_upper_offset": np.float32,
        "isolation_window_present": np.bool_,
        "instrument_family_id": np.int8,
        "mass_accuracy": np.float32,
        "mass_accuracy_present": np.bool_,
        "retention_time_fraction": np.float32,
        "retention_time_present": np.bool_,
    }
    for name in ACQUISITION_DATASETS:
        if name not in file:
            file.create_dataset(
                name,
                shape=(rows,),
                dtype=dtypes[name],
                chunks=(min(SCALAR_CHUNK_ROWS, rows),),
                compression="gzip",
                compression_opts=1,
            )


def _take_interval_rows(
    intervals: list[list[int]],
    ordered_ids: np.ndarray,
    first_id: bytes,
    length: int,
) -> np.ndarray:
    interval_index = next(
        index
        for index, (start, stop) in enumerate(intervals)
        if start < stop and bytes(ordered_ids[start]) == first_id
    )
    pieces = []
    remaining = length
    while remaining:
        start, stop = intervals[interval_index]
        take = min(remaining, stop - start)
        pieces.append(np.arange(start, start + take, dtype=np.int64))
        intervals[interval_index][0] += take
        remaining -= take
        if remaining:
            interval_index += 1
    return np.concatenate(pieces)


def _project_rt_max(
    artifact_dir: Path,
    runs: list[dict[str, Any]],
) -> np.ndarray:
    max_file_id = 0
    pieces = []
    for run in runs:
        with h5py.File(artifact_dir / run["path"], "r") as file:
            rows = slice(int(run["start"]), int(run["stop"]))
            file_ids = file["file_id"][rows].astype(np.int64, copy=False)
            retention_time = file["RT"][rows].astype(np.float32, copy=False)
        pieces.append((file_ids, retention_time))
        max_file_id = max(max_file_id, int(file_ids.max(initial=0)))
    rt_max = np.zeros(max_file_id + 1, dtype=np.float32)
    for file_ids, retention_time in pieces:
        present = np.isfinite(retention_time) & (retention_time > 0)
        np.maximum.at(rt_max, file_ids[present], retention_time[present])
    return rt_max


def _write_project_by_id(
    artifact_dir: Path,
    runs: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
) -> None:
    source_ids = arrays["unique_spectrum_id"]
    id_length = max(len(value) for value in source_ids)
    fixed_source_ids = source_ids.astype(f"S{id_length}")
    order = np.argsort(fixed_source_ids)
    sorted_ids = fixed_source_ids[order]
    rt_max = _project_rt_max(artifact_dir, runs)
    missing_rows = 0
    for run in runs:
        path = artifact_dir / run["path"]
        run_start = int(run["start"])
        run_stop = int(run["stop"])
        with h5py.File(path, "r+") as file:
            _ensure_metadata_datasets(file, len(file["spectrum"]))
            for start in range(run_start, run_stop, WRITE_ROWS):
                stop = min(start + WRITE_ROWS, run_stop)
                destination = slice(start, stop)
                destination_ids = file["unique_spectrum_id"][destination].astype(
                    f"S{id_length}"
                )
                positions = np.searchsorted(sorted_ids, destination_ids)
                bounded = positions < len(sorted_ids)
                matched = np.zeros(len(positions), dtype=np.bool_)
                matched[bounded] = (
                    sorted_ids[positions[bounded]] == destination_ids[bounded]
                )
                source_rows = np.zeros(len(positions), dtype=np.int64)
                source_rows[matched] = order[positions[matched]]
                missing = ~matched
                missing_rows += int(missing.sum())

                values = {
                    name: arrays[name][source_rows].copy()
                    for name in ACQUISITION_DATASETS
                }
                if missing.any():
                    precursor = file["precursor_mz"][destination]
                    collision = file["collision_energy"][destination]
                    charge = file["charge"][destination]
                    retention_time = file["RT"][destination]
                    file_ids = file["file_id"][destination].astype(
                        np.int64,
                        copy=False,
                    )
                    values["precursor_mz_present"][missing] = (
                        np.isfinite(precursor[missing]) & (precursor[missing] > 0)
                    )
                    values["collision_energy_present"][missing] = (
                        np.isfinite(collision[missing]) & (collision[missing] > 0)
                    )
                    values["charge_present"][missing] = (
                        np.isfinite(charge[missing]) & (charge[missing] > 0)
                    )
                    for name in (
                        "polarity_id",
                        "acquisition_type_id",
                        "isolation_window_lower_offset",
                        "isolation_window_upper_offset",
                        "isolation_window_present",
                        "instrument_family_id",
                        "mass_accuracy",
                        "mass_accuracy_present",
                    ):
                        values[name][missing] = 0
                    rt_present = np.isfinite(retention_time) & (
                        retention_time > 0
                    )
                    values["retention_time_present"][missing] = rt_present[
                        missing
                    ]
                    values["retention_time_fraction"][missing] = 0
                    normalized = rt_present & (rt_max[file_ids] > 0)
                    values["retention_time_fraction"][missing & normalized] = (
                        retention_time[missing & normalized]
                        / rt_max[file_ids[missing & normalized]]
                    )
                for name in ACQUISITION_DATASETS:
                    file[name][destination] = values[name]
            file.flush()
    logger.warning(
        "Used explicit unknown metadata for %d legacy rows absent from source",
        missing_rows,
    )


def write_project(
    artifact_dir: Path,
    runs: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    validation: np.ndarray,
) -> None:
    by_split = {
        split: [run for run in runs if run["split"] == split]
        for split in ("train", "validation")
    }
    try:
        for split, split_runs in by_split.items():
            if not split_runs:
                continue
            ordered, assigned_rows = ordered_project_indices(
                arrays,
                validation,
                split,
            )
            ordered_ids = arrays["unique_spectrum_id"][ordered]
            intervals = [[0, assigned_rows], [assigned_rows, len(ordered)]]
            for run in split_runs:
                path = artifact_dir / run["path"]
                start = int(run["start"])
                stop = int(run["stop"])
                with h5py.File(path, "r+") as file:
                    _ensure_metadata_datasets(file, len(file["spectrum"]))
                    first_id = bytes(file["unique_spectrum_id"][start])
                    positions = _take_interval_rows(
                        intervals,
                        ordered_ids,
                        first_id,
                        stop - start,
                    )
                    source_rows = ordered[positions]
                    for offset in range(0, stop - start, WRITE_ROWS):
                        destination = slice(
                            start + offset,
                            min(start + offset + WRITE_ROWS, stop),
                        )
                        source = source_rows[offset : offset + WRITE_ROWS]
                        destination_ids = file["unique_spectrum_id"][destination]
                        if not np.array_equal(
                            destination_ids,
                            arrays["unique_spectrum_id"][source],
                        ):
                            raise ValueError
                        for name in ACQUISITION_DATASETS:
                            file[name][destination] = arrays[name][source]
                    file.flush()
            if any(start != stop for start, stop in intervals):
                raise ValueError
    except (StopIteration, ValueError):
        _write_project_by_id(artifact_dir, runs, arrays)


def metadata_manifest() -> dict[str, Any]:
    return {
        "schema": MASSIVE_V2_ACQUISITION_SCHEMA,
        "condition_dim": 26,
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
        },
    }


def finalize_manifest(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    manifest["conversion"]["version"] = CONVERSION_VERSION
    manifest["row_aligned_datasets"] = list(
        dict.fromkeys([*manifest["row_aligned_datasets"], *ACQUISITION_DATASETS])
    )
    manifest["spectrum_metadata"] = metadata_manifest()
    for split in ("train", "validation"):
        for shard in manifest["splits"][split]["shards"]:
            path = manifest_path.parent / shard["path"]
            shard["bytes"] = path.stat().st_size
            shard["sha256"] = _sha256(path)
    _json_write(manifest_path, manifest)
    (manifest_path.parent / "README.md").write_text(dataset_card(manifest))


def enrich_artifact(
    artifact_dir: Path,
    work_dir: Path,
    *,
    limit: int | None = None,
    download_workers: int = 3,
) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = artifact_dir / "manifest.json"
    runs = index_project_runs(manifest_path, work_dir / "project_runs.json")
    filenames = source_files(HfApi())
    completed_path = work_dir / "completed.json"
    completed = set(
        json.loads(completed_path.read_text()) if completed_path.exists() else []
    )
    source_dir = work_dir / "source"
    source_dir.mkdir(exist_ok=True)
    pending = [name for name in filenames if name not in completed]
    if limit is not None:
        pending = pending[:limit]
    zero_row_files = [
        filename
        for filename in pending
        if filename.split("_", 1)[0] not in runs
    ]
    completed.update(zero_row_files)
    if zero_row_files:
        _json_write(completed_path, sorted(completed))
    pending = [filename for filename in pending if filename not in zero_row_files]

    with ThreadPoolExecutor(max_workers=download_workers) as pool:
        downloads = {
            pool.submit(download_source, source_dir, filename): filename
            for filename in pending[:download_workers]
        }
        next_index = download_workers
        processed = 0
        while downloads:
            future = next(as_completed(downloads))
            filename = downloads.pop(future)
            source_path = future.result()
            if next_index < len(pending):
                next_filename = pending[next_index]
                downloads[
                    pool.submit(download_source, source_dir, next_filename)
                ] = next_filename
                next_index += 1
            processed += 1
            project = filename.split("_", 1)[0]
            logger.info(
                "Enriching %d/%d: %s",
                processed,
                len(pending),
                filename,
            )
            arrays, validation = read_project_metadata(source_path)
            write_project(artifact_dir, runs[project], arrays, validation)
            source_path.unlink()
            completed.add(filename)
            _json_write(completed_path, sorted(completed))
    if len(completed) == len(filenames):
        finalize_manifest(manifest_path)
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append acquisition metadata to the existing MassIVE v2 artifact."
    )
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--download-workers", type=int, default=3)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument(
        "--destination-repo-id",
        default=DEFAULT_DESTINATION_REPO_ID,
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    args = parse_args()
    manifest_path = enrich_artifact(
        args.artifact_dir,
        args.work_dir,
        limit=args.limit,
        download_workers=args.download_workers,
    )
    if args.validate:
        validate_artifact(manifest_path)
    if args.upload:
        baseline = Path(
            hf_hub_download(
                args.destination_repo_id,
                "manifest.json",
                repo_type="dataset",
                revision=BASELINE_REVISION,
            )
        )
        validate_replacement_totals(manifest_path, baseline)
        print(upload_artifact(args.artifact_dir, args.destination_repo_id))


if __name__ == "__main__":
    main()
