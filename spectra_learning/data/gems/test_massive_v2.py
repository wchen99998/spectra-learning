import hashlib
import json
import shutil
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import pytest
from ml_collections import config_dict

import spectra_learning.data.gems as gems
import spectra_learning.data.gems.artifacts as gems_artifacts
import spectra_learning.data.gems.materialize as materializer
import spectra_learning.data.gems.prepare_massive_v2 as converter
import spectra_learning.data.gems.repair_massive_v2_eligibility as repairer
import spectra_learning.data.gems.repack_massive_v2 as repacker
from spectra_learning.data.gems.artifacts import MASSIVE_V2_HDF5_FORMAT
from spectra_learning.data.gems.eligibility import (
    massive_v2_eligibility_contract,
)
from spectra_learning.data.gems.prepare_massive_v2 import (
    FINAL_DATASETS,
    validate_artifact,
)


def _write_shard(path: Path, rows: int, eligible: np.ndarray) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    spectrum = np.zeros((rows, 2, 128), dtype=np.float32)
    spectrum[:, 0, :2] = np.arange(rows)[:, None] + (100.0, 101.0)
    spectrum[:, 1, :2] = (1.0, 0.5)
    spectrum[~eligible] = 0.0
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
        "eligibility": massive_v2_eligibility_contract(),
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
            "raw_global_group_id_formula": ("file_id * 2**32 + group_id"),
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
                "eligible_rows": (validation_shard_count * eligible_rows_per_shard),
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
    config.seed = 7
    return config


def test_multihost_staged_loaders_preserve_assignment_and_validation(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config.training_max_steps = 2
    _write_artifact(
        Path(config.artifact_dir) / "gems" / "unit--massive-v2--unit-revision"
    )
    with mock.patch.object(materializer, "snapshot_download") as download:
        modules = [
            gems.GemsDataModule(
                config, seed=7, distributed_world_size=2, distributed_rank=rank
            )
            for rank in range(2)
        ]
        rank0, rank1 = modules
        assert set(rank0.artifact.train_shards(0)).isdisjoint(
            rank1.artifact.train_shards(1)
        )
        assert (
            rank0.info["gems_shard_plan_sha256"] == rank1.info["gems_shard_plan_sha256"]
        )
        for module in modules:
            assert next(iter(module.train_loader_for_epoch(0)))["peak_mz"].shape == (
                2,
                4,
            )
            validation = module.val_loader_for_eval(augment=False)
            assert validation.sampler.seed == 42
            assert validation.sampler.partition_batches
            next(iter(validation))
            for dataset in module._datasets.values():
                dataset.close()
        download.assert_not_called()


def test_materialize_rank_shards_downloads_complete_plan(
    tmp_path: Path,
) -> None:
    remote = tmp_path / "remote"
    _write_artifact(remote)
    config = _config(tmp_path)
    config.training_max_steps = 2
    calls = []

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
        materializer,
        "snapshot_download",
        side_effect=snapshot_download,
    ):
        paths = materializer.materialize_rank_shards(
            config,
            world_size=2,
            rank=0,
        )

    assert len(paths) == 3
    assert all(path.is_file() for path in paths)
    assert calls[0] == ["manifest.json"]
    assert {call[0] for call in calls[1:]} == {
        str(path.relative_to(paths[0].parents[1])) for path in paths
    }


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
        artifact.validation_sampler_partition(rank) for rank in range(world_size)
    ]
    for assignment in set(validation):
        peers = [
            rank
            for rank, rank_assignment in enumerate(validation)
            if rank_assignment == assignment
        ]
        assert [partitions[rank] for rank in peers] == [
            (len(peers), peer_index) for peer_index in range(len(peers))
        ]
        shard = assignment[0]
        local_batch_size = 4_096 // (world_size * 2)
        assert (
            gems_artifacts._batch_count(
                shard.eligible_rows,
                batch_size=local_batch_size,
                chunk_rows=shard.chunk_rows,
                rows_per_block=0,
                drop_last=True,
            )
            >= len(peers) * 500
        )


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


def test_training_requires_staged_manifest_and_shards(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.batch_size = 2
    with mock.patch.object(materializer, "snapshot_download") as download:
        with pytest.raises(FileNotFoundError, match="manifest is not staged"):
            gems.GemsDataModule(config, seed=7)
        root = Path(config.artifact_dir) / "gems" / "unit--massive-v2--unit-revision"
        manifest = _write_artifact(root)
        for shard in manifest["splits"]["train"]["shards"]:
            (root / shard["path"]).unlink()
        with pytest.raises(FileNotFoundError, match="shards are not staged"):
            gems.GemsDataModule(config, seed=7)
        download.assert_not_called()


def test_batched_reader_preserves_rows_metadata_and_duplicates(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.batch_size = 2
    config.training_max_steps = None
    root = Path(config.artifact_dir) / "gems" / "unit--massive-v2--unit-revision"
    _write_artifact(root)
    module = gems.GemsDataModule(config, seed=7)
    dataset = module._datasets["train"]
    expected = {
        key: [] for key in ("spectra", "precursor_mz_raw", "collision_energy", "charge")
    }
    for shard in module.artifact.train_shards(0):
        with h5py.File(root / shard.path, "r") as file:
            eligible = file["training_eligible"][:]
            for key, name in (
                ("spectra", "spectrum"),
                ("precursor_mz_raw", "precursor_mz"),
                ("collision_energy", "collision_energy"),
                ("charge", "charge"),
            ):
                expected[key].append(file[name][:][eligible])
    indices = [7, 0, 3, 7, 2, 11]
    try:
        batch = dataset[indices]
        for key, parts in expected.items():
            np.testing.assert_array_equal(batch[key], np.concatenate(parts)[indices])
        np.testing.assert_array_equal(batch["index"], indices)
        empty = dataset[[]]
        assert empty["spectra"].shape == (0, 2, 128)
    finally:
        dataset.close()


@pytest.mark.parametrize("fault", ["shape", "dtype", "eligibility", "metadata"])
def test_invalid_shard_closes_file(tmp_path: Path, monkeypatch, fault: str) -> None:
    config = _config(tmp_path)
    config.batch_size = 2
    root = Path(config.artifact_dir) / "gems" / "unit--massive-v2--unit-revision"
    _write_artifact(root)
    module = gems.GemsDataModule(config, seed=7)
    dataset = module._datasets["train"]
    with h5py.File(dataset.paths[0], "a") as file:
        if fault in {"shape", "dtype"}:
            del file["spectrum"]
            file.create_dataset(
                "spectrum",
                shape=(4, 128, 2) if fault == "shape" else (4, 2, 128),
                dtype=np.float64 if fault == "dtype" else np.float32,
            )
        elif fault == "eligibility":
            file["training_eligible"][:] = False
        else:
            del file["charge"]
    opened = []
    original = h5py.File

    def open_file(*args, **kwargs):
        file = original(*args, **kwargs)
        opened.append(file)
        return file

    monkeypatch.setattr(h5py, "File", open_file)
    with pytest.raises(KeyError if fault == "metadata" else ValueError):
        dataset[[0]]
    assert opened and not opened[0].id.valid
    assert not dataset.files


@pytest.mark.parametrize("task", ["pretrain", "sjepa", "grouped_jepa", "ar_spectra"])
@pytest.mark.parametrize("workers", [0, 1])
def test_all_tasks_use_batched_loaders(tmp_path: Path, task: str, workers: int) -> None:
    from spectra_learning.data.ar_spectra import SpectraARGemsDataModule
    from spectra_learning.data.gems.grouped import GroupedGemsDataModule
    from spectra_learning.training.pretrain_jax import _shutdown_torch_loader_iterator

    config = _config(tmp_path)
    config.batch_size = 2
    config.training_max_steps = None
    config.training_task = task
    config.training_mode = "sjepa" if task == "sjepa" else "mae"
    config.dataloader_num_workers = workers
    config.dataloader_multiprocessing_context = "spawn"
    config.dataloader_output_format = "numpy"
    config.dataloader_pin_memory = False
    config.dataloader_persistent_workers = False
    root = Path(config.artifact_dir) / "gems" / "unit--massive-v2--unit-revision"
    manifest = _write_artifact(root)
    module_type = gems.GemsDataModule
    if task == "grouped_jepa":
        config.group_jepa_groups_per_batch = 1
        config.group_jepa_spectra_per_group = 2
        config.group_jepa_teacher_spectra_per_group = 1
        for split in manifest["splits"].values():
            for shard in split["shards"]:
                with h5py.File(root / shard["path"], "a") as file:
                    file["global_group_id"][:] = 0
                    file["group_id"][:] = 0
        module_type = GroupedGemsDataModule
    elif task == "ar_spectra":
        # AR has no JEPA masks, regardless of unused masking defaults.
        config.jepa_num_target_blocks = 10
        module_type = SpectraARGemsDataModule
    module = module_type(config, seed=7)
    try:
        for loader in (
            module.train_loader_for_epoch(0),
            module.val_loader_for_eval(augment=False),
        ):
            iterator = iter(loader)
            try:
                batch = next(iterator)
                assert all(isinstance(value, np.ndarray) for value in batch.values())
                if task == "ar_spectra":
                    assert batch["input_token_ids"].shape[0] == 2
                elif task == "grouped_jepa":
                    assert batch["peak_mz"].shape == (1, 2, 4)
                else:
                    assert batch["peak_mz"].shape == (2, 4)
            finally:
                _shutdown_torch_loader_iterator(iterator)
    finally:
        for dataset in module._datasets.values():
            dataset.close()


def test_workers_are_per_process_and_resume_preserves_batch_suffix(
    tmp_path: Path,
) -> None:
    import torch

    config = _config(tmp_path)
    config.training_max_steps = None
    config.dataloader_num_workers = 4
    root = Path(config.artifact_dir) / "gems" / "unit--massive-v2--unit-revision"
    manifest = _write_artifact(root)
    manifest["splits"]["train"]["shards"] = [
        _write_shard(root / shard["path"], 4, np.asarray([True, False, True, False]))
        for shard in manifest["splits"]["train"]["shards"]
    ]
    manifest["splits"]["train"]["eligible_rows"] = 8
    manifest["eligible_rows"] = 12
    (root / "manifest.json").write_text(json.dumps(manifest))
    module = gems.GemsDataModule(
        config, seed=7, distributed_world_size=2, distributed_rank=0
    )
    assert module.dataloader_num_workers == 4
    module.dataloader_num_workers = 0
    full = list(module.train_loader_for_epoch(0).sampler)
    resumed = list(module.train_loader_for_epoch(0, start_batch=1).sampler)
    assert resumed == full[1:]
    batch = next(iter(module.train_loader_for_epoch(0)))
    assert all(isinstance(value, torch.Tensor) for value in batch.values())
    module.set_gradient_accumulation_steps(2)
    resumed = list(module.train_loader_for_epoch(0, start_batch=1).sampler)
    assert [index for batch in resumed for index in batch] == [
        index for batch in full[1:] for index in batch
    ]
    for dataset in module._datasets.values():
        dataset.close()


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
        [{"bytes": size} for size in (4, 5, 6, 5, 4, 6)],
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
        repacked_manifest.parent / repacked["splits"]["train"]["shards"][0]["path"]
    )
    with h5py.File(train_path) as file:
        assert len(file["spectrum"]) == 16
        assert len(file["global_group_id"]) == 16
        assert set(file) == set(FINAL_DATASETS)


def test_massive_v2_contract_mismatch_fails_fast(tmp_path: Path) -> None:
    artifact = tmp_path / "cache" / "gems" / "unit--massive-v2--unit-revision"
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


def test_repair_recomputes_training_eligibility_from_spectra(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    manifest = _write_artifact(artifact)
    manifest["conversion"] = {"exact_ms_level": 2}
    for split in ("train", "validation"):
        for shard in manifest["splits"][split]["shards"]:
            path = artifact / shard["path"]
            with h5py.File(path, "r+") as file:
                file["training_eligible"][:] = True
            shard["eligible_rows"] = shard["rows"]
            shard["bytes"] = path.stat().st_size
            shard["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest["splits"][split]["eligible_rows"] = manifest["splits"][split]["rows"]
    manifest["eligible_rows"] = manifest["rows"]
    manifest_path = artifact / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    repairer.repair_local_artifact(
        manifest_path,
        workers=1,
        source_revision="old-revision",
    )

    repaired = json.loads(manifest_path.read_text())
    assert repaired["eligibility"] == massive_v2_eligibility_contract()
    assert repaired["eligible_rows"] == 16
    assert repaired["conversion"]["eligibility_repaired_from_revision"] == (
        "old-revision"
    )
    with h5py.File(artifact / "train/shard_00000.hdf5") as file:
        assert file["training_eligible"][:].tolist() == [
            True,
            False,
            True,
            True,
        ]


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
    validate_artifact(manifest_path, workers=2)
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


def test_converter_excludes_rows_without_two_usable_peaks(
    tmp_path: Path,
) -> None:
    source = tmp_path / "spectrum_eligibility.hdf5"
    rows = 4
    spectrum = np.zeros((rows, 2, 128), dtype=np.float32)
    spectrum[0, :, :2] = ((100.0, 200.0), (1.0, 0.5))
    spectrum[1, :, :1] = ((100.0,), (1.0,))
    spectrum[2, :, :2] = ((100.0, 200.0), (1.0, 5e-5))
    spectrum[3, :, :2] = ((19.0, 200.0), (1.0, 0.5))
    with h5py.File(source, "w") as file:
        file.create_dataset("spectrum", data=spectrum)
        file.create_dataset(
            "MS level",
            data=np.full(rows, 2, dtype=np.int8),
        )
        file.create_dataset("RT", data=np.ones(rows, dtype=np.float32))
        file.create_dataset(
            "precursor_mz",
            data=np.full(rows, 500.0, dtype=np.float32),
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
        file.create_dataset(
            "group_id",
            data=np.full(rows, -1, dtype=np.int32),
        )
        file.create_dataset(
            "global_group_id",
            data=np.full(rows, -1, dtype=np.int64),
        )
        file.create_dataset(
            "unique_spectrum_id",
            data=np.asarray([b"a", b"b", b"c", b"d"]),
        )

    arrays, _, stats = converter.prepare_source_arrays(source)

    assert arrays["training_eligible"].tolist() == [True, False, False, False]
    assert stats["eligible_rows"] == 1


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


@pytest.mark.parametrize("task", ["pretrain", "sjepa", "ar_spectra"])
@pytest.mark.parametrize("seed", [0, 7, 19])
def test_columnar_collation_preserves_sample_adapter_randomness(tmp_path, task, seed):
    import torch
    from spectra_learning.data.ar_spectra import SpectraARGemsDataModule

    config = _config(tmp_path)
    config.batch_size = 2
    config.training_task = task
    config.training_mode = "sjepa" if task == "sjepa" else "mae"
    root = Path(config.artifact_dir) / "gems" / "unit--massive-v2--unit-revision"
    _write_artifact(root)
    module_type = (
        SpectraARGemsDataModule if task == "ar_spectra" else gems.GemsDataModule
    )
    module = module_type(config, seed=7)
    dataset = module._datasets["train"]
    try:
        raw = dataset[[1, 0, 1, 2]]
        samples = [{key: value[i] for key, value in raw.items()} for i in range(4)]
        collator = module._collator(augment=True, metadata_dropout=True)
        torch.manual_seed(seed)
        columnar = collator(raw)
        after_columnar = torch.get_rng_state()
        torch.manual_seed(seed)
        adapted = collator(samples)
        assert torch.equal(after_columnar, torch.get_rng_state())
        assert columnar.keys() == adapted.keys()
        for key in columnar:
            np.testing.assert_array_equal(columnar[key], adapted[key])
    finally:
        for dataset in module._datasets.values():
            dataset.close()


def test_torch_persistent_workers_across_epochs(tmp_path):
    import torch
    from spectra_learning.training.pretrain_jax import _shutdown_torch_loader_iterator

    config = _config(tmp_path)
    config.batch_size = 2
    config.training_max_steps = None
    config.dataloader_num_workers = 1
    config.dataloader_multiprocessing_context = "spawn"
    config.dataloader_persistent_workers = True
    root = Path(config.artifact_dir) / "gems" / "unit--massive-v2--unit-revision"
    _write_artifact(root)
    module = gems.GemsDataModule(config, seed=7)
    try:
        for epoch in (0, 1):
            loader = module.train_loader_for_epoch(epoch)
            assert loader.persistent_workers
            iterator = iter(loader)
            try:
                batches = list(iterator)
                assert len(batches) == module.train_steps
                assert all(
                    isinstance(value, torch.Tensor) for value in batches[0].values()
                )
            finally:
                _shutdown_torch_loader_iterator(iterator)
    finally:
        for dataset in module._datasets.values():
            dataset.close()
