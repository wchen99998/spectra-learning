import logging
import shutil
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve

import torch
from huggingface_hub import snapshot_download

from spectra_learning.data.gems.settings import GEMS_METADATA_FILENAME
from spectra_learning.data.gems.native import (
    build_gems_native_artifact,
    load_gems_native_metadata,
    validate_gems_native_artifact,
)
from spectra_learning.data.spectra import NUM_PEAKS_INPUT

logger = logging.getLogger(__name__)


def _validate_gems_native_metadata(
    metadata: dict,
    *,
    max_precursor_mz: float,
) -> None:
    expected = {
        "num_peaks_input": int(NUM_PEAKS_INPUT),
        "max_precursor_mz": max_precursor_mz,
        "artifact_format": "raw_peaklist_v1",
    }
    actual = {
        "num_peaks_input": int(metadata["num_peaks_input"]),
        "max_precursor_mz": float(metadata["max_precursor_mz"]),
        "artifact_format": str(metadata.get("artifact_format", "")),
    }
    if actual != expected:
        raise ValueError(
            f"GeMS native artifact preprocessing mismatch: expected {expected}, got {actual}"
        )


def _gems_native_artifact_dir_name(*, max_precursor_mz: float) -> str:
    value = str(max_precursor_mz).replace(".", "p").replace("-", "m")
    return f"gems_native_raw_pmax{value}"


def _coordinate_distributed_io(distributed_world_size: int) -> bool:
    return (
        distributed_world_size > 1
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    )


def _download_gems_source_hdf5(source_url: str, output_dir: Path) -> Path:
    filename = Path(urlparse(source_url).path).name or "source.hdf5"
    source_dir = output_dir / "gems_source"
    source_dir.mkdir(parents=True, exist_ok=True)
    download_path = source_dir / filename
    if not download_path.exists():
        logger.info("Downloading GeMS source HDF5 from %s", source_url)
        urlretrieve(source_url, download_path)
    return download_path


def _download_gems_raw_hdf5(
    *,
    base_artifact_dir: Path,
    repo_id: str,
    revision: str,
    repo_subdir: str,
    raw_hdf5_path: str,
) -> Path:
    repo_subdir = repo_subdir.strip("/")
    prefix = f"{repo_subdir}/" if repo_subdir else ""
    local_dir = base_artifact_dir.parent if repo_subdir else base_artifact_dir
    hdf5_path = base_artifact_dir / raw_hdf5_path
    if not hdf5_path.exists():
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=local_dir,
            allow_patterns=[f"{prefix}{raw_hdf5_path}"],
        )
    return hdf5_path


def ensure_base_gems_artifact(
    *,
    gems_base_dir: Path,
    repo_id: str,
    revision: str,
    repo_subdir: str = "",
    distributed_world_size: int = 1,
    distributed_rank: int = 0,
) -> tuple[Path, dict]:
    repo_subdir = repo_subdir.strip("/")
    artifact_dir = gems_base_dir / repo_subdir if repo_subdir else gems_base_dir
    download_dir = gems_base_dir if repo_subdir else artifact_dir
    prefix = f"{repo_subdir}/" if repo_subdir else ""
    coordinated = _coordinate_distributed_io(distributed_world_size)
    if coordinated and distributed_rank != 0:
        torch.distributed.barrier()
        metadata = load_gems_native_metadata(artifact_dir)
        validate_gems_native_artifact(artifact_dir, metadata)
        return artifact_dir, metadata

    metadata_path = artifact_dir / GEMS_METADATA_FILENAME
    should_download = True
    if metadata_path.exists():
        existing_metadata = load_gems_native_metadata(artifact_dir)
        should_download = "gems_native_metadata_version" not in existing_metadata
        if should_download:
            shutil.rmtree(artifact_dir)
    if should_download:
        logger.info("Downloading GeMS native artifact from %s@%s", repo_id, revision)
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=download_dir,
            allow_patterns=[
                f"{prefix}{GEMS_METADATA_FILENAME}",
                f"{prefix}train/*",
                f"{prefix}validation/*",
            ],
        )
    if coordinated:
        torch.distributed.barrier()
    metadata = load_gems_native_metadata(artifact_dir)
    validate_gems_native_artifact(artifact_dir, metadata)
    return artifact_dir, metadata


def ensure_custom_gems_artifact(
    *,
    output_dir: Path,
    base_artifact_dir: Path,
    base_metadata: dict,
    repo_id: str,
    revision: str,
    repo_subdir: str,
    max_precursor_mz: float,
    source_hdf5_path: str,
    source_url: str,
    distributed_world_size: int = 1,
    distributed_rank: int = 0,
) -> Path:
    variant_dir = output_dir / "gems_variants" / _gems_native_artifact_dir_name(
        max_precursor_mz=max_precursor_mz,
    )
    coordinated = _coordinate_distributed_io(distributed_world_size)
    if coordinated and distributed_rank != 0:
        torch.distributed.barrier()
        metadata = load_gems_native_metadata(variant_dir)
        validate_gems_native_artifact(variant_dir, metadata)
        _validate_gems_native_metadata(metadata, max_precursor_mz=max_precursor_mz)
        return variant_dir

    metadata_path = variant_dir / GEMS_METADATA_FILENAME
    if metadata_path.exists():
        metadata = load_gems_native_metadata(variant_dir)
        validate_gems_native_artifact(variant_dir, metadata)
        _validate_gems_native_metadata(metadata, max_precursor_mz=max_precursor_mz)
        if coordinated:
            torch.distributed.barrier()
        return variant_dir
    source_path = source_hdf5_path or str(base_metadata.get("source_hdf5_path", "")).strip()
    resolved_source_url = source_url or str(base_metadata.get("source_url", "")).strip()
    raw_hdf5_path = str(base_metadata.get("raw_hdf5_path", "")).strip()
    if source_path:
        hdf5_path = Path(source_path).expanduser().resolve()
    elif raw_hdf5_path:
        hdf5_path = _download_gems_raw_hdf5(
            base_artifact_dir=base_artifact_dir,
            repo_id=repo_id,
            revision=revision,
            repo_subdir=repo_subdir,
            raw_hdf5_path=raw_hdf5_path,
        )
    elif resolved_source_url:
        hdf5_path = _download_gems_source_hdf5(resolved_source_url, output_dir)
    else:
        raise FileNotFoundError(
            "Need source HDF5 or source URL to build a custom GeMS native artifact."
        )
    build_gems_native_artifact(
        hdf5_path=hdf5_path,
        output_dir=variant_dir,
        max_precursor_mz=max_precursor_mz,
        num_workers=1,
        source_path=str(hdf5_path),
        source_url=resolved_source_url or None,
    )
    if coordinated:
        torch.distributed.barrier()
    return variant_dir


def resolve_gems_artifact(
    *,
    output_dir: Path,
    gems_base_dir: Path,
    repo_id: str,
    revision: str,
    max_precursor_mz: float,
    source_hdf5_path: str,
    source_url: str,
    repo_subdir: str = "",
    distributed_world_size: int = 1,
    distributed_rank: int = 0,
) -> tuple[Path, dict]:
    base_artifact_dir, base_metadata = ensure_base_gems_artifact(
        gems_base_dir=gems_base_dir,
        repo_id=repo_id,
        revision=revision,
        repo_subdir=repo_subdir,
        distributed_world_size=distributed_world_size,
        distributed_rank=distributed_rank,
    )
    try:
        _validate_gems_native_metadata(
            base_metadata,
            max_precursor_mz=max_precursor_mz,
        )
        return base_artifact_dir, base_metadata
    except ValueError:
        variant_dir = ensure_custom_gems_artifact(
            output_dir=output_dir,
            base_artifact_dir=base_artifact_dir,
            base_metadata=base_metadata,
            repo_id=repo_id,
            revision=revision,
            repo_subdir=repo_subdir,
            max_precursor_mz=max_precursor_mz,
            source_hdf5_path=source_hdf5_path,
            source_url=source_url,
            distributed_world_size=distributed_world_size,
            distributed_rank=distributed_rank,
        )
        metadata = load_gems_native_metadata(variant_dir)
        validate_gems_native_artifact(variant_dir, metadata)
        _validate_gems_native_metadata(metadata, max_precursor_mz=max_precursor_mz)
        return variant_dir, metadata
