from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from ml_collections import config_dict

from spectra_learning.data.gems.artifacts import (
    _repo_cache_name,
    resolve_gems_hdf5_artifact,
)
from spectra_learning.data.gems.settings import GemsDataConfig


def materialize_gems_hdf5_shard(
    *,
    repo_id: str,
    revision: str,
    manifest_path: Path,
    shard_path: str,
) -> Path:
    path = manifest_path.parent / shard_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".download.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not path.is_file():
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                local_dir=manifest_path.parent,
                allow_patterns=[shard_path],
            )
    return path


def materialize_rank_shards(
    config: config_dict.ConfigDict,
    *,
    world_size: int,
    rank: int,
) -> tuple[Path, ...]:
    if not 0 <= rank < world_size:
        raise ValueError("rank must be between zero and world_size - 1")
    # Match grouped training's spectrum budget rather than its group count.
    if str(config.get("training_task", "pretrain")) == "grouped_jepa":
        config = config_dict.ConfigDict(config.to_dict())
        config.batch_size = int(config.group_jepa_groups_per_batch) * int(
            config.group_jepa_spectra_per_group
        )
        config.training_max_steps = None
    data_config = GemsDataConfig.from_config(config)
    manifest_path = (
        data_config.artifact_dir
        / "gems"
        / _repo_cache_name(
            data_config.gems_hdf5_repo_id, data_config.gems_hdf5_revision
        )
        / data_config.gems_hdf5_manifest
    )
    materialize_gems_hdf5_shard(
        repo_id=data_config.gems_hdf5_repo_id,
        revision=data_config.gems_hdf5_revision,
        manifest_path=manifest_path,
        shard_path=data_config.gems_hdf5_manifest,
    )
    artifact = resolve_gems_hdf5_artifact(
        gems_base_dir=data_config.artifact_dir / "gems",
        repo_id=data_config.gems_hdf5_repo_id,
        revision=data_config.gems_hdf5_revision,
        manifest_filename=data_config.gems_hdf5_manifest,
        distributed_world_size=world_size,
        seed=int(config.seed),
        global_batch_size=data_config.batch_size,
        gradient_accumulation_steps=data_config.gradient_accumulation_steps,
        rows_per_block=data_config.gems_hdf5_rows_per_block,
        drop_remainder=data_config.drop_remainder,
        training_max_steps=data_config.training_max_steps,
        val_num_steps=data_config.val_num_steps,
    )
    assigned = {
        shard.path: shard
        for shard in (
            *artifact.train_shards(rank),
            *artifact.validation_shards(rank),
        )
    }
    total_bytes = sum(shard.bytes for shard in assigned.values())
    logging.info(
        "Materializing %d GeMS shards (%.2f GB) for rank %d/%d",
        len(assigned),
        total_bytes / 1_000_000_000,
        rank,
        world_size,
    )
    paths = []
    for index, shard in enumerate(assigned.values(), start=1):
        logging.info(
            "Materializing GeMS shard %d/%d: %s",
            index,
            len(assigned),
            shard.path,
        )
        paths.append(
            materialize_gems_hdf5_shard(
                repo_id=artifact.repo_id,
                revision=artifact.revision,
                manifest_path=artifact.manifest_path,
                shard_path=shard.path,
            )
        )
    logging.info("Materialized all GeMS shards for rank %d", rank)
    return tuple(paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize this process's planned GeMS shards."
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--overrides-json", default="{}")
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    from spectra_learning.config import load_config

    config = (
        load_config(args.config, json.loads(args.overrides_json))
        if args.config is not None
        else config_dict.ConfigDict(json.loads(os.environ["SPECTRA_CONFIG_JSON"]))
    )
    materialize_rank_shards(
        config,
        world_size=args.world_size,
        rank=args.rank,
    )


if __name__ == "__main__":
    main()
