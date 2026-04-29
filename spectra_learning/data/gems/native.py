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

from spectra_learning.data.spectra import (
    DEFAULT_MAX_PRECURSOR_MZ,
    NUM_PEAKS_INPUT,
)
from spectra_learning.data.gems.arrays import (
    CANONICAL_NUM_SHARDS,
    CANONICAL_SPLIT_SEED,
    CANONICAL_VALIDATION_FRACTION,
    load_gems_arrays,
)

log = logging.getLogger(__name__)

METADATA_FILENAME = "metadata.json"
GEMS_NATIVE_METADATA_VERSION = 2


def _write_native_shard(
    output_path_str: str,
    *,
    shard_id: int,
    num_shards: int,
    spectra: np.ndarray,
    precursor: np.ndarray,
) -> tuple[str, int]:
    output_path = Path(output_path_str)
    shard_name = f"shard-{shard_id:05d}-of-{num_shards:05d}"
    shard_dir = output_path / shard_name
    shard_dir.mkdir(parents=True, exist_ok=True)
    np.save(shard_dir / "spectra.npy", spectra.astype(np.float32, copy=False))
    np.save(
        shard_dir / "precursor_mz_raw.npy",
        precursor.astype(np.float32, copy=False),
    )
    return shard_name, len(spectra)


def write_gems_native_shards(
    spectra: np.ndarray,
    precursor: np.ndarray,
    output_path: Path,
    *,
    num_shards: int,
    desc: str,
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
                )
                for sid, sp, pc in jobs
            ]
            results = [f.result() for f in tqdm(futures, desc=f"{desc} shards")]
    return [name for name, _ in results], [length for _, length in results]


def build_gems_native_artifact(
    *,
    hdf5_path: Path,
    output_dir: Path,
    max_precursor_mz: float = DEFAULT_MAX_PRECURSOR_MZ,
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
        num_workers=_resolve_num_workers(num_workers),
    )
    val_shards, val_lengths = write_gems_native_shards(
        spectra[val_idx],
        precursor[val_idx],
        output_dir / "validation",
        num_shards=max(1, int(num_shards) // 4),
        desc="Validation",
        num_workers=_resolve_num_workers(num_workers),
    )
    metadata = {
        "gems_native_metadata_version": GEMS_NATIVE_METADATA_VERSION,
        "num_peaks_input": NUM_PEAKS_INPUT,
        "artifact_format": "raw_peaklist_v1",
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
            for filename in ("spectra.npy", "precursor_mz_raw.npy"):
                path = shard_dir / filename
                if not path.exists():
                    raise FileNotFoundError(path)
