import json
import logging
import math
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from utils.gems_tfrecords import (
    CANONICAL_MAX_PRECURSOR_MZ,
    CANONICAL_NUM_SHARDS,
    CANONICAL_SPLIT_SEED,
    CANONICAL_VALIDATION_FRACTION,
    load_gems_arrays,
)

log = logging.getLogger(__name__)

METADATA_FILENAME = "metadata.json"
GEMS_NATIVE_METADATA_VERSION = 1
_NUM_PEAKS_INPUT = 128
_PEAK_MZ_MIN = 20.0
_PEAK_MZ_MAX = 1000.0
_DEFAULT_MIN_PEAK_INTENSITY = 1e-4
_DEFAULT_NUM_PEAKS = 64


def _preprocess_spectra(
    spectra: np.ndarray,
    precursor: np.ndarray,
    *,
    num_peaks: int,
    min_peak_intensity: float,
    peak_ordering: str,
    max_precursor_mz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mz = spectra[:, 0, :].astype(np.float32, copy=False)
    intensity = spectra[:, 1, :].astype(np.float32, copy=False)

    keep = (
        (mz >= _PEAK_MZ_MIN)
        & (mz <= _PEAK_MZ_MAX)
        & (intensity >= float(min_peak_intensity))
    )
    mz = np.where(keep, mz, 0.0)
    intensity = np.where(keep, intensity, 0.0)

    if mz.shape[1] > num_peaks:
        topk_idx = np.argpartition(-intensity, num_peaks, axis=1)[:, :num_peaks]
        rows = np.arange(mz.shape[0])[:, None]
        mz = mz[rows, topk_idx]
        intensity = intensity[rows, topk_idx]
        sort_within = np.argsort(-intensity, axis=1, kind="stable")
        mz = np.take_along_axis(mz, sort_within, axis=1)
        intensity = np.take_along_axis(intensity, sort_within, axis=1)
    elif mz.shape[1] < num_peaks:
        pad = num_peaks - mz.shape[1]
        mz = np.pad(mz, ((0, 0), (0, pad)))
        intensity = np.pad(intensity, ((0, 0), (0, pad)))

    max_intensity = np.maximum(intensity.max(axis=1, keepdims=True), 1e-8)
    intensity = intensity / max_intensity
    valid = intensity > 0

    if peak_ordering == "mz":
        sort_key = np.where(valid, mz, np.inf)
        order = np.argsort(sort_key, axis=1, kind="stable")
    else:
        sort_key = np.where(valid, intensity, -np.inf)
        order = np.argsort(-sort_key, axis=1, kind="stable")

    mz = np.take_along_axis(mz, order, axis=1)
    intensity = np.take_along_axis(intensity, order, axis=1)
    valid = np.take_along_axis(valid, order, axis=1)

    mz = np.where(valid, mz, 0.0)
    intensity = np.where(valid, intensity, 0.0)
    precursor = (
        np.clip(precursor, 0.0, float(max_precursor_mz)).astype(np.float32)
        / float(max_precursor_mz)
    )

    return (
        (mz / _PEAK_MZ_MAX).astype(np.float32),
        intensity.astype(np.float32),
        valid,
        precursor,
    )


def _write_native_shard(
    output_path_str: str,
    *,
    shard_id: int,
    num_shards: int,
    spectra: np.ndarray,
    precursor: np.ndarray,
    num_peaks: int,
    min_peak_intensity: float,
    peak_ordering: str,
    max_precursor_mz: float,
) -> tuple[str, int]:
    output_path = Path(output_path_str)
    shard_name = f"shard-{shard_id:05d}-of-{num_shards:05d}"
    shard_dir = output_path / shard_name
    shard_dir.mkdir(parents=True, exist_ok=True)

    peak_mz, peak_intensity, peak_valid_mask, precursor_mz = _preprocess_spectra(
        spectra,
        precursor,
        num_peaks=num_peaks,
        min_peak_intensity=min_peak_intensity,
        peak_ordering=peak_ordering,
        max_precursor_mz=max_precursor_mz,
    )
    np.save(shard_dir / "peak_mz.npy", peak_mz)
    np.save(shard_dir / "peak_intensity.npy", peak_intensity)
    np.save(shard_dir / "peak_valid_mask.npy", peak_valid_mask)
    np.save(shard_dir / "precursor_mz.npy", precursor_mz)
    return shard_name, len(spectra)


def write_gems_native_shards(
    spectra: np.ndarray,
    precursor: np.ndarray,
    output_path: Path,
    *,
    num_shards: int,
    desc: str,
    num_peaks: int,
    min_peak_intensity: float,
    peak_ordering: str,
    max_precursor_mz: float,
    num_workers: int = 1,
) -> tuple[list[str], list[int]]:
    n = len(spectra)
    num_shards = max(1, min(int(num_shards), n))
    shard_size = math.ceil(n / num_shards)
    output_path.mkdir(parents=True, exist_ok=True)
    jobs = [
        (
            sid,
            spectra[start:end],
            precursor[start:end],
        )
        for sid in range(num_shards)
        if (start := sid * shard_size) < (end := min((sid + 1) * shard_size, n))
    ]
    output_path_str = str(output_path)

    def _call(sid, sp, pc):
        return _write_native_shard(
            output_path_str,
            shard_id=sid,
            num_shards=num_shards,
            spectra=sp,
            precursor=pc,
            num_peaks=num_peaks,
            min_peak_intensity=min_peak_intensity,
            peak_ordering=peak_ordering,
            max_precursor_mz=max_precursor_mz,
        )

    if num_workers == 1:
        results = [_call(*job) for job in jobs]
    else:
        worker_count = min(num_workers, len(jobs))
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=ctx) as pool:
            futures = [
                pool.submit(
                    _write_native_shard,
                    output_path_str,
                    shard_id=sid,
                    num_shards=num_shards,
                    spectra=sp,
                    precursor=pc,
                    num_peaks=num_peaks,
                    min_peak_intensity=min_peak_intensity,
                    peak_ordering=peak_ordering,
                    max_precursor_mz=max_precursor_mz,
                )
                for sid, sp, pc in jobs
            ]
            results = [f.result() for f in tqdm(futures, desc=f"{desc} shards")]
    return [name for name, _ in results], [length for _, length in results]


def build_gems_native_artifact(
    *,
    hdf5_path: Path,
    output_dir: Path,
    num_peaks: int = _DEFAULT_NUM_PEAKS,
    min_peak_intensity: float = _DEFAULT_MIN_PEAK_INTENSITY,
    peak_ordering: str = "mz",
    max_precursor_mz: float = CANONICAL_MAX_PRECURSOR_MZ,
    num_shards: int = CANONICAL_NUM_SHARDS,
    num_workers: int | None = None,
    source_path: str | None = None,
    source_url: str | None = None,
) -> dict[str, Any]:
    log.info("Loading GeMS data from %s", hdf5_path)
    spectra, retention, precursor = load_gems_arrays(hdf5_path)
    mask = (
        np.isfinite(retention)
        & (retention > 0.0)
        & np.isfinite(precursor)
        & (precursor <= float(max_precursor_mz))
    )
    spectra = spectra[mask]
    precursor = precursor[mask]
    n = len(spectra)
    log.info("Valid GeMS spectra: %d", n)

    perm = np.random.default_rng(CANONICAL_SPLIT_SEED).permutation(n)
    train_size = int(n * (1.0 - CANONICAL_VALIDATION_FRACTION))
    train_idx = perm[:train_size]
    val_idx = perm[train_size:]

    output_dir.mkdir(parents=True, exist_ok=True)
    train_shards, train_lengths = write_gems_native_shards(
        spectra[train_idx],
        precursor[train_idx],
        output_dir / "train",
        num_shards=num_shards,
        desc="Train",
        num_peaks=int(num_peaks),
        min_peak_intensity=float(min_peak_intensity),
        peak_ordering=str(peak_ordering),
        max_precursor_mz=float(max_precursor_mz),
        num_workers=_resolve_num_workers(num_workers),
    )
    val_shards, val_lengths = write_gems_native_shards(
        spectra[val_idx],
        precursor[val_idx],
        output_dir / "validation",
        num_shards=max(1, int(num_shards) // 4),
        desc="Validation",
        num_peaks=int(num_peaks),
        min_peak_intensity=float(min_peak_intensity),
        peak_ordering=str(peak_ordering),
        max_precursor_mz=float(max_precursor_mz),
        num_workers=_resolve_num_workers(num_workers),
    )
    metadata = {
        "gems_native_metadata_version": GEMS_NATIVE_METADATA_VERSION,
        "num_peaks_input": _NUM_PEAKS_INPUT,
        "num_peaks": int(num_peaks),
        "peak_ordering": str(peak_ordering),
        "min_peak_intensity": float(min_peak_intensity),
        "max_precursor_mz": float(max_precursor_mz),
        "train_shards": train_shards,
        "train_lengths": train_lengths,
        "validation_shards": val_shards,
        "validation_lengths": val_lengths,
        "train_size": int(train_size),
        "validation_size": int(n - train_size),
        "validation_fraction": CANONICAL_VALIDATION_FRACTION,
        "split_seed": CANONICAL_SPLIT_SEED,
        "num_shards": int(num_shards),
        "source_hdf5_path": source_path or str(hdf5_path),
        "source_url": source_url,
    }
    with (output_dir / METADATA_FILENAME).open("w") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata


def _resolve_num_workers(num_workers: int | None) -> int:
    if num_workers is None:
        return min(CANONICAL_NUM_SHARDS, os.cpu_count() or 1)
    return max(1, int(num_workers))


def load_gems_native_metadata(artifact_dir: Path) -> dict[str, Any]:
    with (artifact_dir / METADATA_FILENAME).open() as handle:
        return json.load(handle)


def validate_gems_native_artifact(artifact_dir: Path, metadata: dict[str, Any]) -> None:
    if int(metadata["gems_native_metadata_version"]) != GEMS_NATIVE_METADATA_VERSION:
        raise ValueError(
            f"Expected GeMS native metadata version {GEMS_NATIVE_METADATA_VERSION}, got {metadata['gems_native_metadata_version']}"
        )
    for split, key in [("train", "train_shards"), ("validation", "validation_shards")]:
        for name in metadata[key]:
            shard_dir = artifact_dir / split / name
            for filename in (
                "peak_mz.npy",
                "peak_intensity.npy",
                "peak_valid_mask.npy",
                "precursor_mz.npy",
            ):
                path = shard_dir / filename
                if not path.exists():
                    raise FileNotFoundError(path)
