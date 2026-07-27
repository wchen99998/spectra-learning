import logging
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

HDF5_SHARD_PATTERNS = ["*.hdf5", "*.h5"]


def _repo_cache_name(repo_id: str, revision: str) -> str:
    return "--".join(
        (repo_id.replace("/", "--"), revision.replace("/", "--"))
    )


def resolve_gems_hdf5_manifest(
    *,
    gems_base_dir: Path,
    repo_id: str,
    revision: str,
    manifest_filename: str,
    distributed_world_size: int = 1,
    distributed_rank: int = 0,
    distributed_local_rank: int | None = None,
) -> Path:
    distributed_local_rank = (
        distributed_rank if distributed_local_rank is None else distributed_local_rank
    )
    artifact_dir = gems_base_dir / _repo_cache_name(repo_id, revision)
    manifest_path = artifact_dir / manifest_filename
    coordinated = _coordinate_distributed_io(distributed_world_size)
    if coordinated and distributed_local_rank != 0:
        torch.distributed.barrier()
        return manifest_path

    if not manifest_path.exists():
        logger.info("Downloading GeMS HDF5 shards from %s@%s", repo_id, revision)
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=artifact_dir,
            allow_patterns=[manifest_filename, *HDF5_SHARD_PATTERNS],
        )
    if coordinated:
        torch.distributed.barrier()
    return manifest_path


def _coordinate_distributed_io(distributed_world_size: int) -> bool:
    return (
        distributed_world_size > 1
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    )
