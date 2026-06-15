import argparse
import json
import logging
import math
import multiprocessing as mp
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlretrieve

import numpy as np
from huggingface_hub import HfApi, hf_hub_download
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
GEMS_SOURCE_REPO_ID = "roman-bushuiev/GeMS"
GEMS_A10_SOURCE_FILENAME = "data/GeMS_A/GeMS_A10.hdf5"
GEMS_B_SOURCE_FILENAME = "data/GeMS_B/GeMS_B.hdf5"


def download_gems_source_url(source_url: str, work_dir: Path) -> Path:
    parsed = urlparse(source_url)
    filename = Path(parsed.path).name or "source.hdf5"
    download_path = work_dir / "source" / filename
    download_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading source HDF5 from %s", source_url)
    urlretrieve(source_url, download_path)
    return download_path


def download_gems_hf_source(
    *,
    repo_id: str,
    filename: str,
    revision: str,
    work_dir: Path,
) -> Path:
    log.info("Downloading source HDF5 from %s@%s:%s", repo_id, revision, filename)
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            revision=revision,
            local_dir=work_dir / "source",
        )
    )


def gems_hf_source_url(*, repo_id: str, filename: str, revision: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{filename}"


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
    num_shards = max(1, min(num_shards, n))
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
        & (precursor <= max_precursor_mz)
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
        num_shards=max(1, num_shards // 4),
        desc="Validation",
        num_workers=_resolve_num_workers(num_workers),
    )
    metadata = {
        "gems_native_metadata_version": GEMS_NATIVE_METADATA_VERSION,
        "num_peaks_input": NUM_PEAKS_INPUT,
        "artifact_format": "raw_peaklist_v1",
        "max_precursor_mz": max_precursor_mz,
        "train_shards": train_shards,
        "train_lengths": train_lengths,
        "validation_shards": val_shards,
        "validation_lengths": val_lengths,
        "train_size": train_size,
        "validation_size": n - train_size,
        "validation_fraction": CANONICAL_VALIDATION_FRACTION,
        "split_seed": CANONICAL_SPLIT_SEED,
        "num_shards": num_shards,
        "source_hdf5_path": source_path or "",
        "source_url": source_url,
    }
    with (output_dir / METADATA_FILENAME).open("w") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata


def prepare_gems_native_dataset(
    *,
    work_dir: Path,
    hf_repo_id: str,
    hf_subdir: str = "",
    source_hdf5_path: Path | None = None,
    source_url: str | None = None,
    source_hf_filename: str | None = None,
    source_gems_a10: bool = False,
    source_gems_b: bool = False,
    source_hf_repo_id: str = GEMS_SOURCE_REPO_ID,
    source_hf_revision: str = "main",
    hf_revision: str = "main",
    max_precursor_mz: float = DEFAULT_MAX_PRECURSOR_MZ,
    num_shards: int = CANONICAL_NUM_SHARDS,
    num_workers: int = 0,
    upload_num_workers: int = 8,
    upload: bool = True,
    include_raw: bool = False,
) -> dict[str, Any]:
    work_dir = work_dir.expanduser().resolve()
    artifact_root = work_dir / "artifact"
    artifact_dir = artifact_root / hf_subdir.strip("/") if hf_subdir else artifact_root
    if artifact_root.exists():
        shutil.rmtree(artifact_root)

    if source_hdf5_path is not None:
        hdf5_path = source_hdf5_path.expanduser().resolve()
        source_path = str(hdf5_path)
        resolved_source_url = source_url
    elif source_url is not None:
        hdf5_path = download_gems_source_url(source_url, work_dir)
        source_path = None
        resolved_source_url = source_url
    else:
        if source_gems_a10:
            filename = GEMS_A10_SOURCE_FILENAME
        elif source_gems_b:
            filename = GEMS_B_SOURCE_FILENAME
        else:
            filename = source_hf_filename
        if filename is None:
            raise ValueError(
                "source_hdf5_path, source_url, source_hf_filename, or "
                "source_gems_a10/source_gems_b is required"
            )
        hdf5_path = download_gems_hf_source(
            repo_id=source_hf_repo_id,
            filename=filename,
            revision=source_hf_revision,
            work_dir=work_dir,
        )
        source_path = None
        resolved_source_url = gems_hf_source_url(
            repo_id=source_hf_repo_id,
            filename=filename,
            revision=source_hf_revision,
        )

    metadata = build_gems_native_artifact(
        hdf5_path=hdf5_path,
        output_dir=artifact_dir,
        max_precursor_mz=max_precursor_mz,
        num_shards=num_shards,
        num_workers=None if num_workers <= 0 else num_workers,
        source_path=source_path,
        source_url=resolved_source_url,
    )
    metadata["hf_subdir"] = hf_subdir.strip("/")
    metadata["raw_hdf5_path"] = ""
    if include_raw:
        raw_path = artifact_dir / "raw" / hdf5_path.name
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(hdf5_path, raw_path)
        metadata["raw_hdf5_path"] = str(raw_path.relative_to(artifact_dir))
    with (artifact_dir / METADATA_FILENAME).open("w") as handle:
        json.dump(metadata, handle, indent=2)

    if upload:
        api = HfApi()
        api.create_repo(hf_repo_id, repo_type="dataset", exist_ok=True)
        api.upload_large_folder(
            repo_id=hf_repo_id,
            folder_path=artifact_root,
            repo_type="dataset",
            revision=hf_revision,
            num_workers=upload_num_workers,
        )
    metadata["artifact_dir"] = str(artifact_dir)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and upload GeMS native PyTorch shards."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--source-hdf5-path", type=Path)
    source_group.add_argument("--source-url")
    source_group.add_argument("--source-hf-filename")
    source_group.add_argument("--source-gems-a10", action="store_true")
    source_group.add_argument("--source-gems-b", action="store_true")
    parser.add_argument("--source-hf-repo-id", default=GEMS_SOURCE_REPO_ID)
    parser.add_argument("--source-hf-revision", default="main")
    parser.add_argument("--hf-repo-id", required=True)
    parser.add_argument("--hf-subdir", default="")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--hf-revision", default="main")
    parser.add_argument("--max-precursor-mz", type=float, default=1000.0)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--upload-num-workers", type=int, default=8)
    parser.add_argument("--include-raw", action="store_true")
    args = parser.parse_args()
    prepare_gems_native_dataset(
        work_dir=args.work_dir,
        hf_repo_id=args.hf_repo_id,
        hf_subdir=args.hf_subdir,
        source_hdf5_path=args.source_hdf5_path,
        source_url=args.source_url,
        source_hf_filename=args.source_hf_filename,
        source_gems_a10=args.source_gems_a10,
        source_gems_b=args.source_gems_b,
        source_hf_repo_id=args.source_hf_repo_id,
        source_hf_revision=args.source_hf_revision,
        hf_revision=args.hf_revision,
        max_precursor_mz=args.max_precursor_mz,
        num_shards=args.num_shards,
        num_workers=args.num_workers,
        upload_num_workers=args.upload_num_workers,
        include_raw=args.include_raw,
    )


def _resolve_num_workers(num_workers: int | None) -> int:
    if num_workers is None:
        return min(CANONICAL_NUM_SHARDS, os.cpu_count() or 1)
    return max(1, num_workers)


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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
