from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from ml_collections import config_dict

from spectra_learning.data.gems.artifacts import (
    MASSIVE_V2_HDF5_FORMAT,
    materialize_gems_hdf5_shard,
    resolve_gems_hdf5_artifact,
)
from spectra_learning.data.gems.settings import GemsDataConfig


def materialize_rank_shards(
    config: config_dict.ConfigDict,
    *,
    world_size: int,
    rank: int,
) -> tuple[Path, ...]:
    data_config = GemsDataConfig.from_config(config)
    artifact = resolve_gems_hdf5_artifact(
        gems_base_dir=data_config.artifact_dir / "gems",
        repo_id=data_config.gems_hdf5_repo_id,
        revision=data_config.gems_hdf5_revision,
        manifest_filename=data_config.gems_hdf5_manifest,
        distributed_world_size=world_size,
        distributed_rank=rank,
        distributed_local_rank=0,
        seed=int(config.seed),
        global_batch_size=data_config.batch_size,
        gradient_accumulation_steps=data_config.gradient_accumulation_steps,
        rows_per_block=data_config.gems_hdf5_rows_per_block,
        drop_remainder=data_config.drop_remainder,
        training_max_steps=data_config.training_max_steps,
        val_num_steps=data_config.val_num_steps,
    )
    if artifact.format != MASSIVE_V2_HDF5_FORMAT:
        logging.info(
            "Materialized legacy GeMS artifact at %s",
            artifact.manifest_path.parent,
        )
        return ()

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
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    config = config_dict.ConfigDict(
        json.loads(os.environ["SPECTRA_CONFIG_JSON"])
    )
    materialize_rank_shards(
        config,
        world_size=args.world_size,
        rank=args.rank,
    )


if __name__ == "__main__":
    main()
