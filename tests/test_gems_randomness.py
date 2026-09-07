from itertools import islice

import numpy as np
import pytest

from spectra_learning.data.gems.sampling import (
    ChunkedDistributedBatchSampler,
    OffsetBatchSampler,
)


def sampler(
    *,
    rows=4099,
    batch_size=64,
    rank=0,
    world_size=1,
    partition_batches=False,
    seed=67,
    drop_last=False,
    window=8,
):
    return ChunkedDistributedBatchSampler(
        [(0, rows, 256)],
        batch_size=batch_size,
        rows_per_block=None,
        shuffle=True,
        seed=seed,
        drop_last=drop_last,
        world_size=world_size,
        rank=rank,
        partition_batches=partition_batches,
        shuffle_window_blocks=window,
    )


def flatten(batches):
    return [row for batch in batches for row in batch]


@pytest.mark.parametrize("batch_size", [32, 64, 128, 512])
@pytest.mark.parametrize("drop_last", [False, True])
def test_shuffle_drains_windows_and_only_drops_final_tail(batch_size, drop_last):
    sample = sampler(batch_size=batch_size, drop_last=drop_last)
    batches = list(sample)
    rows = flatten(batches)
    assert len(batches) == len(sample)
    assert len(rows) == len(set(rows))
    assert len(rows) == (4099 // batch_size * batch_size if drop_last else 4099)
    if not drop_last:
        assert sorted(rows) == list(range(4099))


def test_accumulation_and_resume_do_not_change_the_sample_stream():
    expected = flatten(sampler(batch_size=128))
    for size in (32, 64, 256):
        source = sampler(batch_size=size)
        assert flatten(source) == expected
        resumed = OffsetBatchSampler(
            source, start_index=1024, batch_size=size, drop_last=False
        )
        assert flatten(resumed) == expected[1024:]


@pytest.mark.parametrize("partition_batches", [False, True])
@pytest.mark.parametrize("world_size", [2, 8, 16])
def test_partitions_are_disjoint_and_cover_the_source(partition_batches, world_size):
    batches = [
        list(
            sampler(
                rank=rank, world_size=world_size, partition_batches=partition_batches
            )
        )
        for rank in range(world_size)
    ]
    rows = flatten(batch for host in batches for batch in host)
    assert sorted(rows) == list(range(4099))


def test_seed_and_epoch_change_permutations_without_changing_coverage():
    source = sampler()
    first = flatten(source)
    assert first == flatten(source)
    source.set_epoch(1)
    second = flatten(source)
    assert first != second
    assert sorted(first) == sorted(second)
    assert first != flatten(sampler(seed=68))


@pytest.mark.parametrize("seed", [66, 67, 68])
@pytest.mark.parametrize("epoch", [0, 1])
def test_early_prefix_has_no_lane_position_bias(seed, epoch):
    source = sampler(rows=32768 * 128, batch_size=128, window=2048, seed=seed)
    source.set_epoch(epoch)
    rows = np.asarray(flatten(islice(source, 1024)))
    counts = np.bincount((rows % 128) // 2, minlength=64)
    # Former lane 3 was ~9x expected; seven lanes were never visited.
    expected = len(rows) / 64
    assert counts.min() > expected * 0.85
    assert counts.max() < expected * 1.15
    assert len(np.unique(rows)) == len(rows)


def test_windows_bound_storage_locality_without_batch_striping():
    source = sampler(rows=256 * 128, batch_size=64, window=8)
    rows = np.asarray(flatten(islice(source, 32)))
    assert len(np.unique(rows // 256)) == 8
    assert len(np.unique(rows)) == 8 * 256
    # Individual rows are shuffled: contiguous row pairs aren't forced together.
    batches = rows.reshape(-1, 64)
    paired = sum(sum((int(row) ^ 1) in batch for row in batch) for batch in batches)
    assert paired < rows.size // 4


@pytest.mark.parametrize("task", ["pretrain", "sjepa"])
def test_loader_randomness_survives_workers_accumulation_resume_and_validation(
    tmp_path, task
):
    import h5py
    import torch
    from pathlib import Path
    from spectra_learning.data.gems.test_massive_v2 import _config, _write_artifact
    from spectra_learning.data.gems.datamodule import GemsDataModule
    from spectra_learning.training.pretrain_jax import _shutdown_torch_loader_iterator

    config = _config(tmp_path)
    config.batch_size = 2
    config.training_max_steps = None
    config.training_task = task
    config.training_mode = "sjepa" if task == "sjepa" else "mae"
    config.num_peaks = 32
    config.dataloader_output_format = "numpy"
    config.dataloader_multiprocessing_context = "spawn"
    root = Path(config.artifact_dir) / "gems" / "unit--massive-v2--unit-revision"
    manifest = _write_artifact(root)
    for split in manifest["splits"].values():
        for shard in split["shards"]:
            with h5py.File(root / shard["path"], "a") as file:
                values = np.zeros(file["spectrum"].shape, dtype=np.float32)
                eligible = file["training_eligible"][:]
                values[eligible, 0, :32] = np.arange(100, 132)
                values[eligible, 1, :32] = 1.0
                file["spectrum"][:] = values

    def consume(module, epoch=0, start_batch=0, validation=False):
        loader = (
            module.val_loader_for_eval(augment=True)
            if validation
            else module.train_loader_for_epoch(epoch, start_batch=start_batch)
        )
        iterator = iter(loader)
        try:
            batches = list(iterator)
            return {
                key: np.concatenate([batch[key] for batch in batches])
                for key in batches[0]
            }
        finally:
            _shutdown_torch_loader_iterator(iterator)

    reference = None
    for workers, accumulation in [(0, 1), (1, 1), (2, 2)]:
        config.dataloader_num_workers = workers
        config.gradient_accumulation_steps = accumulation
        module = GemsDataModule(config, seed=7)
        try:
            torch.manual_seed(999 + workers)
            values = consume(module)
            if reference is None:
                reference = values
            else:
                for key in reference:
                    np.testing.assert_array_equal(values[key], reference[key])
            first_val = consume(module, validation=True)
            torch.rand(1000)
            second_val = consume(module, validation=True)
            for key in first_val:
                np.testing.assert_array_equal(first_val[key], second_val[key])
            resumed = consume(module, start_batch=1)
            for key in reference:
                np.testing.assert_array_equal(
                    resumed[key], reference[key][config.batch_size :]
                )
            later = consume(module, epoch=1)
            mask_key = (
                "context_mask" if task == "pretrain" else "sjepa_global_view_masks"
            )
            assert not np.array_equal(later[mask_key], reference[mask_key])
            # Same spectra with different stable identities must not share masks.
            assert len(np.unique(values[mask_key].reshape(12, -1), axis=0)) > 8
        finally:
            for dataset in module._datasets.values():
                dataset.close()


def test_augmentation_streams_are_independent_and_batch_order_invariant(tmp_path):
    import torch
    from pathlib import Path
    from spectra_learning.data.gems.test_massive_v2 import _config, _write_artifact
    from spectra_learning.data.gems.datamodule import GemsDataModule
    from spectra_learning.data.gems.collate import ACQUISITION_METADATA_KEYS
    from spectra_learning.models.spectrum_metadata import MASSIVE_V2_ACQUISITION_SCHEMA

    config = _config(tmp_path)
    config.batch_size = 2
    config.num_peaks = 32
    config.dataloader_output_format = "numpy"
    root = Path(config.artifact_dir) / "gems" / "unit--massive-v2--unit-revision"
    _write_artifact(root)
    module = GemsDataModule(config, seed=7)
    try:
        raw = module._datasets["train"][[0, 1, 2, 3]]
        raw["spectra"][:, 0, :32] = np.arange(100, 132)
        raw["spectra"][:, 1, :32] = 1.0
        for key in ACQUISITION_METADATA_KEYS:
            raw[key] = np.ones(4, dtype=np.float32)
        collator = module._collator(augment=True, epoch=3)
        collator.spectrum_metadata_schema = MASSIVE_V2_ACQUISITION_SCHEMA
        collator.spectrum_metadata_dropout_probability = 0.5
        before = torch.get_rng_state()
        first = collator(raw)
        assert torch.equal(before, torch.get_rng_state())
        order = np.array([2, 0, 3, 1])
        shuffled = collator({key: value[order] for key, value in raw.items()})
        split = [
            collator({key: value[i : i + 1] for key, value in raw.items()})
            for i in range(4)
        ]
        for key in first:
            np.testing.assert_array_equal(shuffled[key], first[key][order])
            np.testing.assert_array_equal(
                np.concatenate([row[key] for row in split]), first[key]
            )
        collator.spectrum_metadata_dropout_probability = 0.0
        no_dropout = collator(raw)
        np.testing.assert_array_equal(no_dropout["context_mask"], first["context_mask"])
        assert not np.array_equal(
            no_dropout["spectrum_metadata"], first["spectrum_metadata"]
        )
    finally:
        for dataset in module._datasets.values():
            dataset.close()


def test_group_sampling_is_batch_size_independent_and_replays():
    from spectra_learning.data.gems.grouped import GroupedBatchSampler

    ranges = ((np.arange(0, 800, 10), np.full(80, 10)),)

    def make(size):
        return GroupedBatchSampler(
            ranges,
            groups_per_batch=size,
            spectra_per_group=4,
            seed=67,
            epoch=2,
            shuffle=True,
            drop_last=False,
        )

    first = flatten(make(8))
    assert first == flatten(make(3))
    assert first == flatten(make(8))
    assert len(first) == len(set(first)) == 320
    groups = np.asarray(first).reshape(-1, 4) // 10
    assert np.all(groups == groups[:, :1])
    assert len(np.unique(groups[:, 0])) == 80


def test_complete_training_assignment_does_not_depend_on_run_length_or_accumulation(
    tmp_path,
):
    from pathlib import Path
    from spectra_learning.data.gems.test_massive_v2 import _config, _write_artifact
    from spectra_learning.data.gems.datamodule import GemsDataModule

    config = _config(tmp_path)
    config.batch_size = 2
    root = Path(config.artifact_dir) / "gems" / "unit--massive-v2--unit-revision"
    manifest = _write_artifact(root)
    reference = None
    for steps, accumulation in [(1, 1), (100, 1), (100, 2)]:
        config.training_max_steps = steps
        config.gradient_accumulation_steps = accumulation
        module = GemsDataModule(config, seed=7)
        assigned = tuple(shard.path for shard in module.artifact.train_shards(0))
        assert set(assigned) == {
            shard["path"] for shard in manifest["splits"]["train"]["shards"]
        }
        if reference is None:
            reference = assigned
        assert assigned == reference


def test_raw_block_cache_is_bounded_reused_and_not_pickled(tmp_path, monkeypatch):
    from pathlib import Path
    from spectra_learning.data.gems.test_massive_v2 import _config, _write_artifact
    from spectra_learning.data.gems.datamodule import GemsDataModule

    config = _config(tmp_path)
    config.batch_size = 2
    config.gems_hdf5_shuffle_window_blocks = 1
    root = Path(config.artifact_dir) / "gems" / "unit--massive-v2--unit-revision"
    _write_artifact(root)
    module = GemsDataModule(config, seed=7)
    dataset = module._datasets["train"]
    reads = []
    original = dataset._read_raw_rows

    def read(indices):
        reads.append(indices)
        return original(indices)

    monkeypatch.setattr(dataset, "_read_raw_rows", read)
    try:
        first = dataset.read_raw_batch([0, 1])
        repeated = dataset.read_raw_batch([1, 0, 1])
        assert len(reads) == 1
        np.testing.assert_array_equal(repeated[0], first[0][[1, 0, 1]])
        dataset.read_raw_batch([2, 3, 6])
        assert len(dataset._raw_blocks) == 1
        assert not dataset.__getstate__()["_raw_blocks"]
        dataset.close()
        assert not dataset._raw_blocks
        assert not dataset.files
    finally:
        dataset.close()


def test_loader_does_not_consume_model_rng_and_has_independent_worker_seeds(tmp_path):
    import torch
    from pathlib import Path
    from spectra_learning.data.gems.test_massive_v2 import _config, _write_artifact
    from spectra_learning.data.gems.datamodule import GemsDataModule

    config = _config(tmp_path)
    config.batch_size = 2
    root = Path(config.artifact_dir) / "gems" / "unit--massive-v2--unit-revision"
    _write_artifact(root)
    seeds = []
    for rank in (0, 1):
        module = GemsDataModule(
            config, seed=7, distributed_world_size=2, distributed_rank=rank
        )
        before = torch.get_rng_state()
        loader = module.train_loader_for_epoch(0)
        seeds.append(loader.generator.initial_seed())
        next(iter(loader))
        assert torch.equal(before, torch.get_rng_state())
        assert (
            loader.generator.initial_seed()
            != module.train_loader_for_epoch(1).generator.initial_seed()
        )
        for dataset in module._datasets.values():
            dataset.close()
    assert seeds[0] != seeds[1]


def test_randomness_contract_is_versioned_and_old_knobs_are_rejected(tmp_path):
    from pathlib import Path
    from spectra_learning.data.gems.test_massive_v2 import _config, _write_artifact
    from spectra_learning.data.gems.datamodule import GemsDataModule
    from spectra_learning.data.gems.settings import GemsDataConfig
    from spectra_learning.training.pretrain import pretrain_checkpoint_dataset_metadata

    config = _config(tmp_path)
    config.batch_size = 2
    root = Path(config.artifact_dir) / "gems" / "unit--massive-v2--unit-revision"
    _write_artifact(root)
    module = GemsDataModule(config, seed=7)
    contract = pretrain_checkpoint_dataset_metadata(module)["input_contract"]
    assert contract == {
        "version": 2,
        "seed": 7,
        "shuffle_seed": None,
        "window_blocks": 2048,
        "rows_per_block": 0,
    }
    config.gems_hdf5_mix_pool_blocks = 2048
    with pytest.raises(ValueError, match="Removed input settings"):
        GemsDataConfig.from_config(config)


def test_sjepa_views_have_separate_keys_and_validation_is_seed_independent(tmp_path):
    from pathlib import Path
    from spectra_learning.data.gems.test_massive_v2 import _config, _write_artifact
    from spectra_learning.data.gems.datamodule import GemsDataModule

    config = _config(tmp_path)
    config.batch_size = 2
    config.training_task = config.training_mode = "sjepa"
    config.num_peaks = 32
    config.dataloader_output_format = "numpy"
    root = Path(config.artifact_dir) / "gems" / "unit--massive-v2--unit-revision"
    _write_artifact(root)
    module = GemsDataModule(config, seed=7)
    try:
        raw = module._datasets["train"][[0, 1, 2, 3]]
        raw["spectra"][:, 0, :32] = np.arange(100, 132)
        raw["spectra"][:, 1, :32] = 1.0
        collator = module._collator(augment=True, epoch=2)
        first = collator(raw)
        collator.num_global_views += 1
        extra = collator(raw)
        np.testing.assert_array_equal(
            extra["sjepa_global_view_masks"][:, :-1], first["sjepa_global_view_masks"]
        )
        np.testing.assert_array_equal(
            extra["sjepa_local_view_masks"], first["sjepa_local_view_masks"]
        )
        assert not np.array_equal(
            extra["sjepa_global_view_masks"][:, 0],
            extra["sjepa_global_view_masks"][:, 1],
        )
        first_val = module._collator(augment=True, split="validation")(raw)
        module.seed = 999
        second_val = module._collator(augment=True, split="validation")(raw)
        for key in first_val:
            np.testing.assert_array_equal(first_val[key], second_val[key])
    finally:
        for dataset in module._datasets.values():
            dataset.close()


@pytest.mark.parametrize("partitions", [2, 3, 8])
def test_group_partition_lengths_and_spectra_match_the_shared_stream(partitions):
    from spectra_learning.data.gems.grouped import GroupedBatchSampler

    ranges = tuple(
        (np.arange(shard * 1000, shard * 1000 + count * 10, 10), np.full(count, 10))
        for shard, count in enumerate([3, 5, 7])
    )
    common = dict(
        groups_per_batch=2,
        spectra_per_group=3,
        seed=7,
        epoch=1,
        shuffle=True,
        drop_last=False,
    )
    reference = flatten(GroupedBatchSampler(ranges, **common))
    combined = []
    for rank in range(partitions):
        source = GroupedBatchSampler(
            ranges, **common, partition_count=partitions, partition_index=rank
        )
        batches = list(source)
        assert len(batches) == len(source)
        combined.extend(flatten(batches))
    assert sorted(combined) == sorted(reference)
    assert len(set(combined)) == len(combined)
    assert list(GroupedBatchSampler(ranges, **common, max_batches=0)) == []
