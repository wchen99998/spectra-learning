import hashlib
import json
import shutil
import threading
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import pytest
from ml_collections import config_dict

import spectra_learning.data.gems as gems
import spectra_learning.data.gems.artifacts as gems_artifacts
import spectra_learning.data.gems.prepare_massive_v2 as converter
import spectra_learning.data.gems.repack_massive_v2 as repacker
from spectra_learning.data.gems.artifacts import MASSIVE_V2_HDF5_FORMAT
from spectra_learning.data.gems.prepare_massive_v2 import (
    FINAL_DATASETS,
    validate_artifact,
)


def _write_shard(path: Path, rows: int, eligible: np.ndarray) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    spectrum = np.zeros((rows, 2, 128), dtype=np.float32)
    spectrum[:, 0, :2] = np.arange(rows)[:, None] + (100.0, 101.0)
    spectrum[:, 1, :2] = (1.0, 0.5)
    with h5py.File(path, "w") as file:
        file.create_dataset("spectrum", data=spectrum, chunks=(2, 2, 128))
        file.create_dataset(
            "training_eligible",
            data=eligible,
            chunks=(2,),
        )
        file.create_dataset(
            "MS level",
            data=np.full(rows, 2, dtype=np.int8),
        )
        file.create_dataset("RT", data=np.ones(rows, dtype=np.float32))
        file.create_dataset(
            "precursor_mz",
            data=np.arange(rows, dtype=np.float32) + 100.0,
        )
        file.create_dataset(
            "collision_energy",
            data=np.full(rows, 25.0, dtype=np.float32),
        )
        file.create_dataset("charge", data=np.ones(rows, dtype=np.int8))
        file.create_dataset(
            "massive_id",
            data=np.full(rows, b"MSV000000001", dtype="S12"),
        )
        file.create_dataset("file_id", data=np.zeros(rows, dtype=np.int32))
        file.create_dataset("group_id", data=np.arange(rows, dtype=np.int32))
        file.create_dataset(
            "global_group_id",
            data=np.arange(rows, dtype=np.int64),
        )
        file.create_dataset(
            "unique_spectrum_id",
            data=np.asarray(
                [f"MSV000000001_ms2_{row}".encode() for row in range(rows)]
            ),
        )
    return {
        "path": str(path.relative_to(path.parents[1])),
        "rows": rows,
        "eligible_rows": int(eligible.sum()),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "chunk_rows": 2,
    }


def _write_artifact(root: Path) -> dict:
    train = [
        _write_shard(
            root / "train" / f"shard_{index:05d}.hdf5",
            4,
            np.asarray([True, False, True, True]),
        )
        for index in range(4)
    ]
    validation = [
        _write_shard(
            root / "validation" / f"shard_{index:05d}.hdf5",
            2,
            np.asarray([True, True]),
        )
        for index in range(2)
    ]
    manifest = {
        "format": MASSIVE_V2_HDF5_FORMAT,
        "source": {
            "repo_id": "novogaia/massive-v2",
            "revision": "source-revision",
            "filename_suffix": "_t0.95_l0.80_grouped.hdf5",
            "file_count": 2,
        },
        "datasets": {
            "spectrum": "spectrum",
            "precursor_mz": "precursor_mz",
            "retention_time": "RT",
            "ms_level": "MS level",
        },
        "eligibility": {
            "version": "bounded_precursor_rt_ms2_v4",
            "ms_level": 2,
            "min_precursor_mz": 1.0,
            "max_precursor_mz": 1000.0,
            "min_retention_time_exclusive": 0.0,
            "requires_finite_precursor_mz": True,
            "requires_finite_retention_time": True,
        },
        "split": {
            "version": "entity_hash_v1",
            "algorithm": "splitmix64",
            "seed": 42,
            "modulus": 20,
            "validation_remainder": 0,
            "assigned_entity_key": ["massive_id", "global_group_id"],
            "unassigned_entity_key": [
                "massive_id",
                "source_row_index",
            ],
        },
        "grouping": {
            "assigned_rule": "group_id >= 0",
            "corpus_entity_key": ["massive_id", "global_group_id"],
            "raw_global_group_id_formula": (
                "file_id * 2**32 + group_id"
            ),
            "assigned_rows": 20,
            "assigned_group_occurrences_within_source_files": 20,
        },
        "splits": {
            "train": {
                "rows": 16,
                "eligible_rows": 12,
                "shards": train,
            },
            "validation": {
                "rows": 4,
                "eligible_rows": 4,
                "shards": validation,
            },
        },
        "rows": 20,
        "eligible_rows": 16,
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def _planning_manifest(
    *,
    train_shard_count: int = 64,
    validation_shard_count: int = 3,
    eligible_rows_per_shard: int = 5_000_000,
) -> dict:
    def shards(split: str, count: int) -> list[dict]:
        return [
            {
                "path": f"{split}/shard_{index:05d}.hdf5",
                "rows": eligible_rows_per_shard,
                "eligible_rows": eligible_rows_per_shard,
                "bytes": 1,
                "sha256": str(index),
                "chunk_rows": 256,
            }
            for index in range(count)
        ]

    return {
        "format": MASSIVE_V2_HDF5_FORMAT,
        "splits": {
            "train": {
                "eligible_rows": train_shard_count * eligible_rows_per_shard,
                "shards": shards("train", train_shard_count),
            },
            "validation": {
                "eligible_rows": (
                    validation_shard_count * eligible_rows_per_shard
                ),
                "shards": shards("validation", validation_shard_count),
            },
        },
    }


def _config(tmp_path: Path) -> config_dict.ConfigDict:
    config = config_dict.ConfigDict()
    config.artifact_dir = str(tmp_path / "cache")
    config.gems_hdf5_repo_id = "unit/massive-v2"
    config.gems_hdf5_revision = "unit-revision"
    config.gems_hdf5_manifest = "manifest.json"
    config.batch_size = 4
    config.gradient_accumulation_steps = 1
    config.drop_remainder = True
    config.training_max_steps = 1
    config.val_num_steps = 1
    config.min_precursor_mz = 1.0
    config.max_precursor_mz = 1000.0
    config.min_peak_intensity = 1e-4
    config.peak_drop_min_intensity = 1e-4
    config.precursor_peak_exclusion_window_da = 0.0
    config.peak_ordering = "mz"
    config.num_peaks = 4
    config.jepa_num_target_blocks = 1
    config.jepa_context_fraction = 0.5
    config.jepa_target_fraction = 0.5
    config.jepa_block_min_len = 1
    config.dataloader_num_workers = 0
    return config


def test_multihost_plan_downloads_only_each_ranks_budgeted_shards(
    tmp_path: Path,
) -> None:
    remote = tmp_path / "remote"
    _write_artifact(remote)
    config = _config(tmp_path)
    config.training_max_steps = 2
    calls: list[list[str]] = []

    def snapshot_download(*, local_dir: str | Path, allow_patterns, **_kwargs):
        local = Path(local_dir)
        local.mkdir(parents=True, exist_ok=True)
        calls.append(list(allow_patterns))
        for pattern in allow_patterns:
            source = remote / pattern
            if source.is_file():
                target = local / pattern
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        return str(local)

    with mock.patch.object(
        gems_artifacts,
        "snapshot_download",
        side_effect=snapshot_download,
    ):
        rank0 = gems.GemsDataModule(
            config,
            seed=7,
            distributed_world_size=2,
            distributed_rank=0,
            distributed_local_rank=0,
        )
        rank1 = gems.GemsDataModule(
            config,
            seed=7,
            distributed_world_size=2,
            distributed_rank=1,
            distributed_local_rank=0,
        )
        rank0_train = set(rank0.artifact.train_shards(0))
        rank1_train = set(rank1.artifact.train_shards(1))
        assert rank0_train.isdisjoint(rank1_train)
        assert len(rank0_train) == len(rank1_train) == 2
        assert len(rank0.artifact.validation_shards(0)) == 1
        assert len(rank1.artifact.validation_shards(1)) == 1
        rank0._datasets["train"].wait_for_prefetch()
        rank1._datasets["train"].wait_for_prefetch()
        assert calls[0] == ["manifest.json"]
        assert {call[0] for call in calls[1:]} == {
            rank0._datasets["train"].states[
                rank0._datasets["train"].shard_order[0]
            ].spec.path,
            rank1._datasets["train"].states[
                rank1._datasets["train"].shard_order[0]
            ].spec.path,
        }
        assert not any(
            Path(path).exists()
            for path in (
                *rank0.gems_validation_shards,
                *rank1.gems_validation_shards,
            )
        )

        batch = next(iter(rank0.train_loader_for_epoch(0)))
        assert batch["peak_mz"].shape == (2, 4)
        rank0._datasets["train"].wait_for_prefetch()
        assert all(
            Path(path).exists() for path in rank0.gems_train_shards
        )
        rank0_validation = rank0.val_loader_for_eval(augment=False)
        rank1_validation = rank1.val_loader_for_eval(augment=False)
        assert rank0_validation.batch_sampler.seed == 42
        assert rank1_validation.batch_sampler.seed == 42
        assert rank0_validation.batch_sampler.partition_batches
        assert rank1_validation.batch_sampler.partition_batches
        next(iter(rank0_validation))
        assert calls[-1] == [
            rank0.artifact.validation_shards(0)[0].path
        ]
        next(iter(rank1.train_loader_for_epoch(0)))
        rank1._datasets["train"].wait_for_prefetch()
        assert all(
            Path(path).exists() for path in rank1.gems_train_shards
        )
        assert len(calls) == 6
        assert rank0.info["gems_shard_plan_sha256"] == (
            rank1.info["gems_shard_plan_sha256"]
        )
        assert "planned_download_bytes" in rank0.info


@pytest.mark.parametrize("world_size", (1, 2, 4, 8, 16, 32, 64))
def test_validation_plan_supplies_every_process_for_any_v6e_topology(
    world_size: int,
) -> None:
    _, validation, _ = gems_artifacts.plan_massive_v2_shards(
        _planning_manifest(),
        world_size=world_size,
        seed=66,
        global_batch_size=4_096,
        gradient_accumulation_steps=2,
        rows_per_block=0,
        drop_remainder=True,
        training_max_steps=1,
        val_num_steps=500,
    )

    assert len(validation) == world_size
    assert all(len(assignment) == 1 for assignment in validation)
    assert len({assignment[0].path for assignment in validation}) == min(
        world_size,
        3,
    )
    artifact = gems_artifacts.ResolvedGemsHdf5Artifact(
        format=MASSIVE_V2_HDF5_FORMAT,
        manifest_path=Path("manifest.json"),
        manifest={},
        repo_id="unit/massive-v2",
        revision="unit-revision",
        validation_assignments=validation,
    )
    partitions = [
        artifact.validation_sampler_partition(rank)
        for rank in range(world_size)
    ]
    for assignment in set(validation):
        peers = [
            rank
            for rank, rank_assignment in enumerate(validation)
            if rank_assignment == assignment
        ]
        assert [partitions[rank] for rank in peers] == [
            (len(peers), peer_index)
            for peer_index in range(len(peers))
        ]
        shard = assignment[0]
        local_batch_size = 4_096 // (world_size * 2)
        assert gems_artifacts._batch_count(
            shard.eligible_rows,
            batch_size=local_batch_size,
            chunk_rows=shard.chunk_rows,
            rows_per_block=0,
            drop_last=True,
        ) >= len(peers) * 500


def test_validation_plan_rejects_too_few_full_batches() -> None:
    manifest = _planning_manifest()
    for shard in manifest["splits"]["validation"]["shards"]:
        shard["rows"] = 128
        shard["eligible_rows"] = 128
    manifest["splits"]["validation"]["eligible_rows"] = 3 * 128
    with pytest.raises(
        ValueError,
        match="validation split cannot provide 500 full batches",
    ):
        gems_artifacts.plan_massive_v2_shards(
            manifest,
            world_size=16,
            seed=66,
            global_batch_size=4_096,
            gradient_accumulation_steps=2,
            rows_per_block=0,
            drop_remainder=True,
            training_max_steps=1,
            val_num_steps=500,
        )


def test_initial_shard_prefetch_does_not_block_datamodule_init(
    tmp_path: Path,
) -> None:
    remote = tmp_path / "remote"
    _write_artifact(remote)
    config = _config(tmp_path)
    config.batch_size = 2
    prefetch_started = threading.Event()
    release_prefetch = threading.Event()

    def snapshot_download(*, local_dir: str | Path, allow_patterns, **_kwargs):
        local = Path(local_dir)
        local.mkdir(parents=True, exist_ok=True)
        for pattern in allow_patterns:
            if pattern.endswith(".hdf5"):
                prefetch_started.set()
                release_prefetch.wait()
            source = remote / pattern
            if source.is_file():
                target = local / pattern
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        return str(local)

    with mock.patch.object(
        gems_artifacts,
        "snapshot_download",
        side_effect=snapshot_download,
    ):
        datamodule = gems.GemsDataModule(config, seed=7)
        assert prefetch_started.wait(timeout=1.0)
        dataset = datamodule._datasets["train"]
        assert dataset.prefetch_thread is not None
        assert dataset.prefetch_thread.is_alive()
        assert not list(Path(config.artifact_dir).rglob("*.hdf5"))
        release_prefetch.set()
        dataset.wait_for_prefetch()
        assert len(list(Path(config.artifact_dir).rglob("*.hdf5"))) == 1


def test_repacker_uses_soft_byte_target_and_preserves_metadata(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    manifest = _write_artifact(source)
    manifest["conversion"] = {"target_rows_per_shard": 4}
    manifest["row_aligned_datasets"] = list(FINAL_DATASETS)
    source_manifest = source / "manifest.json"
    source_manifest.write_text(json.dumps(manifest))

    groups = repacker.plan_shard_groups(
        [
            {"bytes": size}
            for size in (4, 5, 6, 5, 4, 6)
        ],
        target_bytes=10,
    )
    assert [sum(item["bytes"] for item in group) for group in groups] == [
        9,
        11,
        10,
    ]

    repacked_manifest = repacker.repack_artifact(
        source_manifest,
        tmp_path / "repacked",
        target_bytes=1_000_000_000,
        workers=1,
    )
    repacked = json.loads(repacked_manifest.read_text())
    assert "target_rows_per_shard" not in repacked["conversion"]
    assert repacked["conversion"]["target_shard_bytes"] == 1_000_000_000
    assert len(repacked["splits"]["train"]["shards"]) == 1
    assert len(repacked["splits"]["validation"]["shards"]) == 1
    train_path = (
        repacked_manifest.parent
        / repacked["splits"]["train"]["shards"][0]["path"]
    )
    with h5py.File(train_path) as file:
        assert len(file["spectrum"]) == 16
        assert len(file["global_group_id"]) == 16
        assert set(file) == set(FINAL_DATASETS)


def test_massive_v2_contract_mismatch_fails_fast(tmp_path: Path) -> None:
    artifact = (
        tmp_path
        / "cache"
        / "gems"
        / "unit--massive-v2--unit-revision"
    )
    manifest = _write_artifact(artifact)
    manifest["eligibility"]["max_precursor_mz"] = 2000.0
    (artifact / "manifest.json").write_text(json.dumps(manifest))
    config = _config(tmp_path)
    config.batch_size = 2

    try:
        gems.GemsDataModule(config, seed=7)
    except ValueError as error:
        assert "eligibility contract mismatch" in str(error)
    else:
        raise AssertionError("Expected a v2 eligibility contract mismatch")


def test_converter_keeps_exact_ms2_and_group_metadata_atomic(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.hdf5"
    rows = 9
    spectrum = np.zeros((rows, 2, 128), dtype=np.float64)
    spectrum[:, 0, :2] = (100.0, 101.0)
    spectrum[:, 1, :2] = (1.0, 0.5)
    group_id = np.asarray([1, 1, 2, 2, -1, -1, 3, 3, 9], dtype=np.int32)
    global_group_id = np.where(
        group_id >= 0,
        group_id.astype(np.int64),
        -1,
    )
    with h5py.File(source, "w") as file:
        file.create_dataset("spectrum", data=spectrum)
        file.create_dataset(
            "MS level",
            data=np.asarray([2] * 8 + [1], dtype=np.int8),
        )
        file.create_dataset("RT", data=np.ones(rows, dtype=np.float32))
        file.create_dataset(
            "precursor_mz",
            data=np.arange(rows, dtype=np.float32) + 100.0,
        )
        file.create_dataset(
            "collision_energy",
            data=np.full(rows, 25.0, dtype=np.float32),
        )
        file.create_dataset("charge", data=np.ones(rows, dtype=np.int8))
        file.create_dataset(
            "massive_id",
            data=np.full(rows, b"MSV000000001", dtype="S12"),
        )
        file.create_dataset("file_id", data=np.zeros(rows, dtype=np.int32))
        file.create_dataset("group_id", data=group_id)
        file.create_dataset("global_group_id", data=global_group_id)
        file.create_dataset(
            "unique_spectrum_id",
            data=np.asarray(
                [f"MSV000000001_ms2_{row}".encode() for row in range(rows)]
            ),
        )

    filename = "MSV000000001_t0.95_l0.80_grouped.hdf5"
    work_dir = tmp_path / "work"
    with (
        mock.patch.object(
            converter,
            "source_files",
            return_value=[filename],
        ),
        mock.patch.object(
            converter,
            "hf_hub_download",
            return_value=str(source),
        ),
    ):
        manifest_path = converter.convert_streaming(
            work_dir,
            target_rows=3,
        )
    validate_artifact(manifest_path)
    manifest = json.loads(manifest_path.read_text())

    assert manifest["rows"] == 8
    locations: dict[tuple[bytes, int], set[str]] = {}
    for split in ("train", "validation"):
        for shard in manifest["splits"][split]["shards"]:
            with h5py.File(manifest_path.parent / shard["path"]) as file:
                assert set(file) == set(FINAL_DATASETS)
                assert file["spectrum"].dtype == np.dtype(np.float32)
                for massive_id, global_id in zip(
                    file["massive_id"][:],
                    file["global_group_id"][:],
                    strict=True,
                ):
                    if global_id >= 0:
                        locations.setdefault(
                            (bytes(massive_id), int(global_id)),
                            set(),
                        ).add(f"{split}/{shard['path']}")
    assert all(len(group_locations) == 1 for group_locations in locations.values())


def test_converter_accepts_source_file_without_ms2_rows(
    tmp_path: Path,
) -> None:
    source = tmp_path / "ms1_only.hdf5"
    with h5py.File(source, "w") as file:
        file.create_dataset(
            "spectrum",
            data=np.zeros((1, 2, 128), dtype=np.float32),
        )
        file.create_dataset("MS level", data=np.ones(1, dtype=np.int8))
        file.create_dataset("RT", data=np.ones(1, dtype=np.float32))
        file.create_dataset(
            "precursor_mz",
            data=np.ones(1, dtype=np.float32),
        )
        file.create_dataset(
            "collision_energy",
            data=np.ones(1, dtype=np.float32),
        )
        file.create_dataset("charge", data=np.ones(1, dtype=np.int8))
        file.create_dataset("massive_id", data=np.asarray([b"MSV000000001"]))
        file.create_dataset("file_id", data=np.zeros(1, dtype=np.int32))
        file.create_dataset("group_id", data=np.zeros(1, dtype=np.int32))
        file.create_dataset(
            "global_group_id",
            data=np.zeros(1, dtype=np.int64),
        )
        file.create_dataset(
            "unique_spectrum_id",
            data=np.asarray([b"MSV000000001_ms1_0"]),
        )

    arrays, validation, stats = converter.prepare_source_arrays(source)

    assert all(len(values) == 0 for values in arrays.values())
    assert validation.shape == (0,)
    assert stats["rows"] == 0


def test_converter_splits_groups_by_massive_id(tmp_path: Path) -> None:
    source = tmp_path / "mixed_projects.hdf5"
    rows = 4
    with h5py.File(source, "w") as file:
        file.create_dataset(
            "spectrum",
            data=np.zeros((rows, 2, 128), dtype=np.float32),
        )
        file.create_dataset(
            "MS level",
            data=np.full(rows, 2, dtype=np.int8),
        )
        file.create_dataset("RT", data=np.ones(rows, dtype=np.float32))
        file.create_dataset(
            "precursor_mz",
            data=np.full(rows, 100.0, dtype=np.float32),
        )
        file.create_dataset(
            "collision_energy",
            data=np.ones(rows, dtype=np.float32),
        )
        file.create_dataset("charge", data=np.ones(rows, dtype=np.int8))
        file.create_dataset(
            "massive_id",
            data=np.asarray([b"MSV1", b"MSV1", b"MSV2", b"MSV2"]),
        )
        file.create_dataset(
            "file_id",
            data=np.ones(rows, dtype=np.int32),
        )
        file.create_dataset(
            "group_id",
            data=np.ones(rows, dtype=np.int32),
        )
        file.create_dataset(
            "global_group_id",
            data=np.full(rows, (1 << 32) + 1, dtype=np.int64),
        )
        file.create_dataset(
            "unique_spectrum_id",
            data=np.asarray([b"a", b"b", b"c", b"d"]),
        )

    arrays, validation, stats = converter.prepare_source_arrays(source)

    assert len(arrays["spectrum"]) == rows
    assert validation[:2].tolist() == [validation[0]] * 2
    assert validation[2:].tolist() == [validation[2]] * 2
    assert stats["assigned_groups"] == 2


def test_converter_regroups_entities_repeated_across_source_shards(
    tmp_path: Path,
) -> None:
    work_dir = tmp_path / "work"
    output_dir = work_dir / "output"
    shards = []
    for worker in ("worker_00", "worker_01"):
        path = output_dir / "train" / worker / "shard_00000.hdf5"
        record = _write_shard(
            path,
            4,
            np.ones(4, dtype=bool),
        )
        record["path"] = str(path.relative_to(output_dir))
        shards.append(record)

    regrouped = converter.regroup_cross_shard_projects(
        work_dir,
        output_dir,
        "train",
        shards,
        target_rows=3,
    )

    assert sum(record["rows"] for record in regrouped) == 8
    locations: dict[tuple[bytes, int], set[str]] = {}
    for record in regrouped:
        with h5py.File(output_dir / record["path"], "r") as file:
            for massive_id, global_id in zip(
                file["massive_id"][:],
                file["global_group_id"][:],
                strict=True,
            ):
                locations.setdefault(
                    (bytes(massive_id), int(global_id)),
                    set(),
                ).add(record["path"])
    assert all(len(paths) == 1 for paths in locations.values())


def test_converter_regroups_project_restart_within_one_shard(
    tmp_path: Path,
) -> None:
    work_dir = tmp_path / "work"
    output_dir = work_dir / "output"
    path = output_dir / "validation" / "worker_00" / "shard_00000.hdf5"
    record = _write_shard(path, 4, np.ones(4, dtype=bool))
    record["path"] = str(path.relative_to(output_dir))
    with h5py.File(path, "r+") as file:
        file["group_id"][:] = np.asarray([0, 1, 0, 2])
        file["global_group_id"][:] = np.asarray([0, 1, 0, 2])

    regrouped = converter.regroup_cross_shard_projects(
        work_dir,
        output_dir,
        "validation",
        [record],
        target_rows=16,
    )

    assert [item["path"] for item in regrouped] == [
        "validation/regrouped/shard_00000.hdf5"
    ]
    with h5py.File(output_dir / regrouped[0]["path"], "r") as file:
        assert np.all(np.diff(file["global_group_id"][:]) >= 0)
