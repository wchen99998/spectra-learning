from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spectra_learning.data.gems.randomness import INPUT_RANDOMNESS_VERSION

MASSIVE_V2_HDF5_FORMAT = "spectra_learning.massive_v2_ms2_shards.v1"


@dataclass(frozen=True)
class GemsHdf5Shard:
    path: str
    rows: int
    eligible_rows: int
    bytes: int
    sha256: str
    chunk_rows: int


@dataclass(frozen=True)
class ResolvedGemsHdf5Artifact:
    format: str
    manifest_path: Path
    manifest: dict[str, Any]
    repo_id: str
    revision: str
    train_assignments: tuple[tuple[GemsHdf5Shard, ...], ...] = ()
    validation_assignments: tuple[tuple[GemsHdf5Shard, ...], ...] = ()
    plan_sha256: str = ""

    def train_shards(self, rank: int) -> tuple[GemsHdf5Shard, ...]:
        return self.train_assignments[rank]

    def validation_shards(self, rank: int) -> tuple[GemsHdf5Shard, ...]:
        return self.validation_assignments[rank]

    def validation_sampler_partition(self, rank: int) -> tuple[int, int]:
        assignment = self.validation_assignments[rank]
        peers = [
            peer_rank
            for peer_rank, peer_assignment in enumerate(self.validation_assignments)
            if peer_assignment == assignment
        ]
        return len(peers), peers.index(rank)


def _repo_cache_name(repo_id: str, revision: str) -> str:
    return "--".join((repo_id.replace("/", "--"), revision.replace("/", "--")))


def _manifest_shards(
    manifest: dict[str, Any],
    split: str,
) -> tuple[GemsHdf5Shard, ...]:
    return tuple(
        GemsHdf5Shard(
            path=str(shard["path"]),
            rows=int(shard["rows"]),
            eligible_rows=int(shard["eligible_rows"]),
            bytes=int(shard["bytes"]),
            sha256=str(shard["sha256"]),
            chunk_rows=int(shard["chunk_rows"]),
        )
        for shard in manifest["splits"][split]["shards"]
    )


def _seeded_shard_order(
    shards: tuple[GemsHdf5Shard, ...],
    *,
    seed: int,
    split: str,
) -> list[GemsHdf5Shard]:
    def order_key(shard: GemsHdf5Shard) -> bytes:
        value = f"{seed}\0{split}\0{shard.path}".encode()
        return hashlib.blake2b(value, digest_size=16).digest()

    return sorted(shards, key=order_key)


def _batch_count(
    rows: int,
    *,
    batch_size: int,
    chunk_rows: int,
    rows_per_block: int,
    drop_last: bool,
) -> int:
    del chunk_rows, rows_per_block
    return rows // batch_size if drop_last else math.ceil(rows / batch_size)


def _assign_shards(
    shards: tuple[GemsHdf5Shard, ...],
    *,
    world_size: int,
    seed: int,
    split: str,
) -> tuple[tuple[GemsHdf5Shard, ...], ...]:
    # Every training shard participates, including in short runs. Host ownership
    # depends on corpus/seed/topology, never on optimizer batching or run length.
    assignments: list[list[GemsHdf5Shard]] = [[] for _ in range(world_size)]
    eligible_rows = [0] * world_size
    for shard in _seeded_shard_order(shards, seed=seed, split=split):
        rank = min(range(world_size), key=lambda item: (eligible_rows[item], item))
        assignments[rank].append(shard)
        eligible_rows[rank] += shard.eligible_rows
    return tuple(tuple(shards) for shards in assignments)


def _assign_validation_shards(
    shards: tuple[GemsHdf5Shard, ...],
    *,
    world_size: int,
    seed: int,
    batch_size: int,
    rows_per_block: int,
    required_batches: int,
) -> tuple[tuple[GemsHdf5Shard, ...], ...]:
    ordered = _seeded_shard_order(shards, seed=seed, split="validation")
    capacities = [
        _batch_count(
            shard.eligible_rows,
            batch_size=batch_size,
            chunk_rows=shard.chunk_rows,
            rows_per_block=rows_per_block,
            drop_last=True,
        )
        // required_batches
        for shard in ordered
    ]
    if sum(capacities) < world_size:
        raise ValueError(
            "MassIVE v2 validation split cannot provide "
            f"{required_batches} full batches to every process"
        )

    assignments: list[tuple[GemsHdf5Shard, ...]] = []
    group_sizes = [0] * len(ordered)
    while len(assignments) < world_size:
        for shard_index, shard in enumerate(ordered):
            if group_sizes[shard_index] == capacities[shard_index]:
                continue
            assignments.append((shard,))
            group_sizes[shard_index] += 1
            if len(assignments) == world_size:
                break
    return tuple(assignments)


def plan_massive_v2_shards(
    manifest: dict[str, Any],
    *,
    world_size: int,
    seed: int,
    global_batch_size: int,
    gradient_accumulation_steps: int,
    rows_per_block: int,
    drop_remainder: bool,
    training_max_steps: int | None,
    val_num_steps: int,
) -> tuple[
    tuple[tuple[GemsHdf5Shard, ...], ...],
    tuple[tuple[GemsHdf5Shard, ...], ...],
    str,
]:
    denominator = world_size * gradient_accumulation_steps
    assert global_batch_size % denominator == 0
    local_batch_size = global_batch_size // denominator
    train_shards = tuple(
        shard for shard in _manifest_shards(manifest, "train") if shard.eligible_rows
    )
    validation_shards = tuple(
        shard
        for shard in _manifest_shards(manifest, "validation")
        if shard.eligible_rows
    )
    train_assignments = _assign_shards(
        train_shards,
        world_size=world_size,
        seed=seed,
        split="train",
    )
    validation_assignments = _assign_validation_shards(
        validation_shards,
        world_size=world_size,
        seed=42,
        batch_size=local_batch_size,
        rows_per_block=rows_per_block,
        required_batches=val_num_steps,
    )
    plan_payload = {
        "format": MASSIVE_V2_HDF5_FORMAT,
        "world_size": world_size,
        "seed": seed,
        "input_randomness_version": INPUT_RANDOMNESS_VERSION,
        "validation_batch_partition": "shard_peer_stride_v1",
        "train": [
            [shard.path for shard in assignment] for assignment in train_assignments
        ],
        "validation": [
            [shard.path for shard in assignment]
            for assignment in validation_assignments
        ],
    }
    plan_sha256 = hashlib.sha256(
        json.dumps(plan_payload, sort_keys=True).encode()
    ).hexdigest()
    return train_assignments, validation_assignments, plan_sha256


def resolve_gems_hdf5_artifact(
    *,
    gems_base_dir: Path,
    repo_id: str,
    revision: str,
    manifest_filename: str,
    distributed_world_size: int,
    seed: int,
    global_batch_size: int,
    gradient_accumulation_steps: int,
    rows_per_block: int,
    drop_remainder: bool,
    training_max_steps: int | None,
    val_num_steps: int,
) -> ResolvedGemsHdf5Artifact:
    artifact_dir = gems_base_dir / _repo_cache_name(repo_id, revision)
    manifest_path = artifact_dir / manifest_filename
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"GeMS manifest is not staged: {manifest_path}. "
            "Run python -m spectra_learning.data.gems.materialize first."
        )
    manifest = json.loads(manifest_path.read_text())
    artifact_format = str(manifest["format"])
    if artifact_format != MASSIVE_V2_HDF5_FORMAT:
        raise ValueError(f"Unknown GeMS HDF5 manifest format: {artifact_format}")

    train, validation, plan_sha256 = plan_massive_v2_shards(
        manifest,
        world_size=distributed_world_size,
        seed=seed,
        global_batch_size=global_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        rows_per_block=rows_per_block,
        drop_remainder=drop_remainder,
        training_max_steps=training_max_steps,
        val_num_steps=val_num_steps,
    )
    return ResolvedGemsHdf5Artifact(
        format=artifact_format,
        manifest_path=manifest_path,
        manifest=manifest,
        repo_id=repo_id,
        revision=revision,
        train_assignments=train,
        validation_assignments=validation,
        plan_sha256=plan_sha256,
    )
