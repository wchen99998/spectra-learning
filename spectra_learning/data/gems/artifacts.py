import logging
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from spectra_learning.data.gems.settings import GEMS_METADATA_FILENAME
from spectra_learning.data.gems.native import (
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


def _coordinate_distributed_io(distributed_world_size: int) -> bool:
    return (
        distributed_world_size > 1
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    )


def _raise_stale_gems_artifact(artifact_dir: Path, reason: str) -> None:
    raise ValueError(
        f"Invalid GeMS native artifact in {artifact_dir}: {reason}. "
        "Delete the artifact directory and download or rebuild it with "
        "prepare_gems_native_dataset."
    )


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

    if (artifact_dir / GEMS_METADATA_FILENAME).exists():
        metadata = load_gems_native_metadata(artifact_dir)
        if "gems_native_metadata_version" not in metadata:
            _raise_stale_gems_artifact(
                artifact_dir,
                "metadata is not a GeMS native artifact",
            )
    else:
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


def resolve_gems_artifact(
    *,
    gems_base_dir: Path,
    repo_id: str,
    revision: str,
    max_precursor_mz: float,
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
    _validate_gems_native_metadata(
        base_metadata,
        max_precursor_mz=max_precursor_mz,
    )
    return base_artifact_dir, base_metadata
