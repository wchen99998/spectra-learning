from __future__ import annotations

import multiprocessing as mp
import json
import logging
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tqdm import tqdm

log = logging.getLogger(__name__)

METADATA_FILENAME = "metadata.json"
GEMS_METADATA_VERSION = 1
CANONICAL_VALIDATION_FRACTION = 0.05
CANONICAL_SPLIT_SEED = 42
CANONICAL_NUM_SHARDS = 4
CANONICAL_MAX_PRECURSOR_MZ = 1000.0
_NUM_PEAKS_INPUT = 128


def load_gems_arrays(hdf5_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import h5py

    with h5py.File(hdf5_path, "r") as f:
        spectra = f["spectrum"][:]
        retention = np.asarray(f["RT"], dtype=np.float32)
        precursor = np.asarray(f["precursor_mz"], dtype=np.float32)
    return spectra, retention, precursor


def _write_shard(
    output_path_str: str,
    *,
    shard_id: int,
    num_shards: int,
    spectra: np.ndarray,
    retention: np.ndarray,
    precursor: np.ndarray,
) -> tuple[str, int]:
    output_path = Path(output_path_str)
    shard_file = output_path / f"shard-{shard_id:05d}-of-{num_shards:05d}.tfrecord"
    options = tf.io.TFRecordOptions(compression_type="GZIP")

    def _float_feat(v):
        return tf.train.Feature(float_list=tf.train.FloatList(value=v))

    with tf.io.TFRecordWriter(str(shard_file), options=options) as writer:  # type: ignore[attr-defined]
        for i in range(len(spectra)):
            feat = {
                "mz": _float_feat(spectra[i, 0].astype(np.float32)),
                "intensity": _float_feat(spectra[i, 1].astype(np.float32)),
                "rt": _float_feat([retention[i]]),
                "precursor_mz": _float_feat([precursor[i]]),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feat))
            writer.write(example.SerializeToString())

    return shard_file.name, len(spectra)


def write_peaklist_tfrecords(
    spectra: np.ndarray,
    retention: np.ndarray,
    precursor: np.ndarray,
    output_path: Path,
    *,
    num_shards: int,
    desc: str,
    num_workers: int = 1,
) -> tuple[list[str], list[int]]:
    n = len(spectra)
    num_shards = max(1, min(num_shards, n))
    shard_size = math.ceil(n / num_shards)

    output_path.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    for shard_id in range(num_shards):
        start = shard_id * shard_size
        end = min(start + shard_size, n)
        if start < end:
            jobs.append(
                (
                    shard_id,
                    spectra[start:end],
                    retention[start:end],
                    precursor[start:end],
                )
            )

    if num_workers == 1:
        results = []
        for shard_id, shard_spectra, shard_retention, shard_precursor in jobs:
            results.append(
                _write_shard(
                    str(output_path),
                    shard_id=shard_id,
                    num_shards=num_shards,
                    spectra=shard_spectra,
                    retention=shard_retention,
                    precursor=shard_precursor,
                )
            )
        return [name for name, _ in results], [length for _, length in results]

    worker_count = min(num_workers, len(jobs))
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=ctx,
    ) as pool:
        futures = [
            pool.submit(
                _write_shard,
                str(output_path),
                shard_id=shard_id,
                num_shards=num_shards,
                spectra=shard_spectra,
                retention=shard_retention,
                precursor=shard_precursor,
            )
            for shard_id, shard_spectra, shard_retention, shard_precursor in jobs
        ]
        results = [future.result() for future in tqdm(futures, desc=f"{desc} shards")]
    return [name for name, _ in results], [length for _, length in results]


def build_gems_tfrecord_artifact(
    *,
    hdf5_path: Path,
    output_dir: Path,
    max_precursor_mz: float = CANONICAL_MAX_PRECURSOR_MZ,
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
    retention = retention[mask]
    precursor = precursor[mask]

    n = len(spectra)
    log.info("Valid GeMS spectra: %d", n)

    rng = np.random.default_rng(CANONICAL_SPLIT_SEED)
    perm = rng.permutation(n)
    train_size = int(n * (1.0 - CANONICAL_VALIDATION_FRACTION))
    train_idx = perm[:train_size]
    val_idx = perm[train_size:]

    output_dir.mkdir(parents=True, exist_ok=True)
    train_files, train_lengths = write_peaklist_tfrecords(
        spectra[train_idx],
        retention[train_idx],
        precursor[train_idx],
        output_dir / "train",
        num_shards=CANONICAL_NUM_SHARDS,
        desc="Train",
        num_workers=_resolve_num_workers(num_workers),
    )
    val_files, val_lengths = write_peaklist_tfrecords(
        spectra[val_idx],
        retention[val_idx],
        precursor[val_idx],
        output_dir / "validation",
        num_shards=max(1, CANONICAL_NUM_SHARDS // 4),
        desc="Validation",
        num_workers=_resolve_num_workers(num_workers),
    )

    metadata = {
        "gems_metadata_version": GEMS_METADATA_VERSION,
        "num_peaks_input": _NUM_PEAKS_INPUT,
        "train_files": train_files,
        "train_lengths": train_lengths,
        "validation_files": val_files,
        "validation_lengths": val_lengths,
        "train_size": int(train_size),
        "validation_size": int(n - train_size),
        "validation_fraction": CANONICAL_VALIDATION_FRACTION,
        "split_seed": CANONICAL_SPLIT_SEED,
        "num_shards": CANONICAL_NUM_SHARDS,
        "max_precursor_mz": float(max_precursor_mz),
        "source_hdf5_path": source_path or str(hdf5_path),
        "source_url": source_url,
    }
    with (output_dir / METADATA_FILENAME).open("w") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def _resolve_num_workers(num_workers: int | None) -> int:
    if num_workers is None:
        return min(CANONICAL_NUM_SHARDS, os.cpu_count() or 1)
    return max(1, int(num_workers))


def load_gems_metadata(artifact_dir: Path) -> dict[str, Any]:
    with (artifact_dir / METADATA_FILENAME).open() as f:
        return json.load(f)


def validate_gems_artifact(artifact_dir: Path, metadata: dict[str, Any]) -> None:
    version = int(metadata["gems_metadata_version"])
    if version != GEMS_METADATA_VERSION:
        raise ValueError(
            f"Expected GeMS metadata version {GEMS_METADATA_VERSION}, got {version}"
        )
    for split, key in [("train", "train_files"), ("validation", "validation_files")]:
        for name in metadata[key]:
            path = artifact_dir / split / name
            if not path.exists():
                raise FileNotFoundError(path)
