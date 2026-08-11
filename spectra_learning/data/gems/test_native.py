import hashlib
import json
import tempfile
import unittest
from itertools import combinations, islice
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import torch
from ml_collections import config_dict

import spectra_learning.data.gems as gems
import spectra_learning.data.gems.artifacts as gems_artifacts
import spectra_learning.data.massspec_probe as massspec_probe_data
from spectra_learning.data.gems.hdf5 import (
    GEMS_SPLIT_CHUNK_ROWS,
    GEMS_SPLIT_MODULUS,
)
from spectra_learning.data.gems.sampling import ChunkedDistributedBatchSampler


def _write_fake_hdf5_shards(
    root: Path,
    lengths: list[int],
    *,
    precursor_mz: np.ndarray | None = None,
    retention_time: np.ndarray | None = None,
    ms_level: np.ndarray | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "fdataloader.shards.v1",
        "shards": [],
    }
    start = 0
    for shard_idx, length in enumerate(lengths):
        shard_name = f"shard_{shard_idx:05d}.hdf5"
        shard_path = root / shard_name
        row_indices = np.arange(start, start + length, dtype=np.float32)
        precursor = (
            row_indices + 1.0
            if precursor_mz is None
            else precursor_mz[start : start + length]
        )
        shard_retention_time = (
            np.ones(length, dtype=np.float32)
            if retention_time is None
            else retention_time[start : start + length]
        )
        shard_ms_level = (
            np.full(length, 2, dtype=np.int8)
            if ms_level is None
            else ms_level[start : start + length]
        )
        collision_energy = 10.0 + (row_indices % 90.0)
        charge = 1.0 + (row_indices % 4.0)
        spectra = np.zeros((length, 2, 128), dtype=np.float64)
        spectra[:, 0, 0] = precursor + 100.0
        spectra[:, 1, 0] = 1.0
        spectra[:, 0, 1] = precursor + 101.0
        spectra[:, 1, 1] = 0.5
        with h5py.File(shard_path, "w") as f:
            f.create_dataset("spectrum", data=spectra, chunks=(1, 2, 128))
            f.create_dataset("precursor_mz", data=precursor, chunks=(1,))
            f.create_dataset("RT", data=shard_retention_time, chunks=(1,))
            f.create_dataset("MS level", data=shard_ms_level, chunks=(1,))
            f.create_dataset("collision_energy", data=collision_energy, chunks=(1,))
            f.create_dataset("charge", data=charge, chunks=(1,))
        manifest["shards"].append({"path": shard_name, "rows": length})
        start += length
    manifest_path = root / "fdataloader_shards.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path


class GemsSamplingTests(unittest.TestCase):
    def test_chunked_sampler_partitions_shard_blocks_by_rank(self):
        segments = [(0, 5, 2), (5, 4, 2)]
        rank0 = list(
            ChunkedDistributedBatchSampler(
                segments,
                batch_size=2,
                rows_per_block=4,
                shuffle=False,
                seed=123,
                drop_last=False,
                world_size=2,
                rank=0,
            )
        )
        rank1 = list(
            ChunkedDistributedBatchSampler(
                segments,
                batch_size=2,
                rows_per_block=4,
                shuffle=False,
                seed=123,
                drop_last=False,
                world_size=2,
                rank=1,
            )
        )

        self.assertEqual(rank0, [[0, 1], [2, 3], [5, 6], [7, 8]])
        self.assertEqual(rank1, [[4]])
        combined = [index for batch in rank0 + rank1 for index in batch]
        self.assertEqual(sorted(combined), list(range(9)))

    def test_shuffled_sampler_length_matches_every_rank_and_epoch(self):
        segments = [(0, 1025, 256)]
        for epoch in range(4):
            for rank in range(2):
                sampler = ChunkedDistributedBatchSampler(
                    segments,
                    batch_size=512,
                    rows_per_block=512,
                    shuffle=True,
                    seed=42,
                    drop_last=True,
                    world_size=2,
                    rank=rank,
                )
                sampler.set_epoch(epoch)
                self.assertEqual(len(list(sampler)), len(sampler))

    def test_shuffled_segments_are_consumed_one_shard_at_a_time(self):
        sampler = ChunkedDistributedBatchSampler(
            [(0, 8, 2), (8, 8, 2), (16, 8, 2)],
            batch_size=2,
            rows_per_block=4,
            shuffle=True,
            seed=42,
            drop_last=True,
            world_size=1,
            rank=0,
            shuffle_segments=True,
        )
        shard_order = [batch[0] // 8 for batch in sampler]
        transitions = [
            shard
            for position, shard in enumerate(shard_order)
            if position == 0 or shard != shard_order[position - 1]
        ]
        self.assertEqual(len(transitions), 3)
        self.assertEqual(set(transitions), {0, 1, 2})

    def test_batch_partition_has_no_overlap_between_shard_peers(self):
        batch_size = 4
        required_batches = 5
        for group_size in (1, 2, 5, 6, 22):
            rows = group_size * required_batches * batch_size + 17
            rank_rows = []
            for rank in range(group_size):
                sampler = ChunkedDistributedBatchSampler(
                    [(0, rows, 2)],
                    batch_size=batch_size,
                    rows_per_block=16,
                    shuffle=True,
                    seed=42,
                    drop_last=True,
                    world_size=group_size,
                    rank=rank,
                    shuffle_segments=True,
                    partition_batches=True,
                )
                batches = list(islice(sampler, required_batches))
                self.assertGreaterEqual(
                    sampler.full_batch_count,
                    required_batches,
                )
                self.assertEqual(len(batches), required_batches)
                rank_rows.append(
                    {row for batch in batches for row in batch}
                )

            for left, right in combinations(rank_rows, 2):
                self.assertTrue(left.isdisjoint(right))
            self.assertEqual(
                len(set().union(*rank_rows)),
                group_size * required_batches * batch_size,
            )

    def test_batch_partition_covers_every_row_once(self):
        rank_batches = []
        for rank in range(3):
            sampler = ChunkedDistributedBatchSampler(
                [(0, 5, 2), (5, 4, 2)],
                batch_size=2,
                rows_per_block=4,
                shuffle=True,
                seed=42,
                drop_last=False,
                world_size=3,
                rank=rank,
                shuffle_segments=True,
                partition_batches=True,
            )
            batches = list(sampler)
            self.assertEqual(len(batches), len(sampler))
            rank_batches.append(batches)

        rows = [
            row
            for batches in rank_batches
            for batch in batches
            for row in batch
        ]
        self.assertEqual(sorted(rows), list(range(9)))


class GeMSRuntimeDownloadTests(unittest.TestCase):
    def _make_config(self, tmp_path: Path) -> config_dict.ConfigDict:
        cfg = config_dict.ConfigDict()
        cfg.artifact_dir = str(tmp_path / "cache")
        cfg.gems_hdf5_repo_id = "unit/hdf5-gems"
        cfg.gems_hdf5_revision = "unit-test"
        cfg.gems_hdf5_manifest = "fdataloader_shards.json"
        cfg.batch_size = 2
        cfg.shuffle_buffer = 4
        cfg.drop_remainder = False
        cfg.max_precursor_mz = 1000.0
        cfg.min_peak_intensity = 1e-4
        cfg.peak_drop_min_intensity = 1e-4
        cfg.precursor_peak_exclusion_window_da = 0.0
        cfg.peak_ordering = "mz"
        cfg.num_peaks = 64
        cfg.jepa_num_target_blocks = 1
        cfg.jepa_context_fraction = 0.5
        cfg.jepa_target_fraction = 0.5
        cfg.jepa_block_min_len = 1
        cfg.dataloader_num_workers = 0
        return cfg

    def _artifact_dir(self, cfg: config_dict.ConfigDict) -> Path:
        return (
            Path(cfg.artifact_dir)
            / "gems"
            / "unit--hdf5-gems--unit-test"
        )

    def test_datamodule_downloads_hdf5_shards_and_builds_train_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = self._make_config(tmp_path)

            def fake_snapshot_download(*, local_dir, **kwargs):
                _write_fake_hdf5_shards(Path(local_dir), [3, 2])
                return str(local_dir)

            with (
                mock.patch.object(gems_artifacts, "snapshot_download",
                    side_effect=fake_snapshot_download,
                ) as download_mock,
            ):
                datamodule = gems.GemsDataModule(cfg, seed=42)
                batch = next(iter(datamodule.train_loader_for_epoch(0)))

            self.assertEqual(datamodule.info["train_size"], 5)
            self.assertEqual(datamodule.info["validation_size"], 0)
            self.assertEqual(datamodule.info["num_peaks_input"], 128)
            self.assertEqual(datamodule.info["num_peaks"], 64)
            self.assertEqual(
                datamodule.info["gems_hdf5_revision"],
                "unit-test",
            )
            self.assertEqual(
                datamodule.info["gems_manifest_sha256"],
                hashlib.sha256(datamodule.gems_manifest.read_bytes()).hexdigest(),
            )
            self.assertNotIn("massspec_train_size", datamodule.info)
            self.assertTrue(
                all(Path(path).exists() for path in datamodule.gems_train_shards)
            )
            self.assertIn("peak_mz", batch)
            self.assertIn("context_mask", batch)
            self.assertIn("target_masks", batch)
            self.assertEqual(tuple(batch["peak_mz"].shape), (2, 64))
            self.assertEqual(tuple(batch["target_masks"].shape), (2, 1, 64))
            _, kwargs = download_mock.call_args
            self.assertEqual(kwargs["repo_id"], "unit/hdf5-gems")
            self.assertEqual(kwargs["revision"], "unit-test")
            self.assertEqual(kwargs["repo_type"], "dataset")
            self.assertEqual(
                kwargs["allow_patterns"],
                ["fdataloader_shards.json"],
            )

    def test_datamodule_uses_local_hdf5_cache_without_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = self._make_config(tmp_path)
            _write_fake_hdf5_shards(self._artifact_dir(cfg), [3, 2])

            with mock.patch.object(
                gems_artifacts,
                "snapshot_download",
            ) as download_mock:
                datamodule = gems.GemsDataModule(cfg, seed=42)
                batch = next(iter(datamodule.train_loader_for_epoch(0)))

            self.assertIn("peak_mz", batch)
            download_mock.assert_not_called()
            self.assertEqual(datamodule.gems_dir, self._artifact_dir(cfg))

    def test_datamodule_rejects_noncanonical_spectrum_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = self._make_config(tmp_path)
            artifact_dir = self._artifact_dir(cfg)
            _write_fake_hdf5_shards(artifact_dir, [3])
            with h5py.File(artifact_dir / "shard_00000.hdf5", "a") as file:
                del file["spectrum"]
                file.create_dataset(
                    "spectrum",
                    data=np.zeros((3, 128, 2), dtype=np.float32),
                )

            with self.assertRaisesRegex(
                ValueError,
                "Invalid GeMS spectrum shape",
            ):
                gems.GemsDataModule(cfg, seed=42)

    def test_train_validation_views_are_disjoint_and_cover_the_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = self._make_config(tmp_path)
            source_size = GEMS_SPLIT_CHUNK_ROWS * GEMS_SPLIT_MODULUS + 80
            cfg.max_precursor_mz = float(source_size)
            _write_fake_hdf5_shards(
                self._artifact_dir(cfg),
                [source_size // 2, source_size - source_size // 2],
            )

            datamodule = gems.GemsDataModule(cfg, seed=42)
            train = datamodule._get_dataset("train")
            validation = datamodule._get_dataset("validation")
            train_indices = set(
                train.source_indices(np.arange(len(train))).tolist()
            )
            validation_indices = set(
                validation.source_indices(np.arange(len(validation))).tolist()
            )

            self.assertFalse(train_indices & validation_indices)
            self.assertEqual(
                train_indices | validation_indices,
                set(range(source_size)),
            )
            self.assertEqual(len(validation), GEMS_SPLIT_CHUNK_ROWS)
            self.assertEqual(
                datamodule.info["train_size"] + datamodule.info["validation_size"],
                source_size,
            )
            self.assertEqual(
                datamodule.info["gems_split"]["version"],
                "global_chunk_modulo_v1",
            )

    def test_train_and_validation_exclude_ineligible_source_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = self._make_config(tmp_path)
            source_size = GEMS_SPLIT_CHUNK_ROWS * GEMS_SPLIT_MODULUS
            precursor = np.full(source_size, 100.0, dtype=np.float32)
            retention_time = np.ones(source_size, dtype=np.float32)
            ms_level = np.full(source_size, 2.0, dtype=np.float32)
            invalid = {
                1,
                2,
                3,
                4,
                5,
                GEMS_SPLIT_CHUNK_ROWS * 18 + 1,
                GEMS_SPLIT_CHUNK_ROWS * 18 + 2,
                GEMS_SPLIT_CHUNK_ROWS * 18 + 3,
                GEMS_SPLIT_CHUNK_ROWS * 18 + 4,
            }
            precursor[1] = np.nan
            precursor[2] = cfg.max_precursor_mz + 1
            retention_time[3] = 0
            ms_level[4] = 3
            precursor[5] = 0.5
            precursor[GEMS_SPLIT_CHUNK_ROWS * 18 + 1] = np.inf
            retention_time[GEMS_SPLIT_CHUNK_ROWS * 18 + 2] = np.nan
            ms_level[GEMS_SPLIT_CHUNK_ROWS * 18 + 3] = np.nan
            precursor[GEMS_SPLIT_CHUNK_ROWS * 18 + 4] = 0.0
            _write_fake_hdf5_shards(
                self._artifact_dir(cfg),
                [source_size // 2, source_size - source_size // 2],
                precursor_mz=precursor,
                retention_time=retention_time,
                ms_level=ms_level,
            )

            datamodule = gems.GemsDataModule(cfg, seed=42)
            train = datamodule._get_dataset("train")
            validation = datamodule._get_dataset("validation")
            train_indices = set(
                train.source_indices(np.arange(len(train))).tolist()
            )
            validation_indices = set(
                validation.source_indices(np.arange(len(validation))).tolist()
            )

            self.assertFalse(train_indices & validation_indices)
            self.assertEqual(
                train_indices | validation_indices,
                set(range(source_size)) - invalid,
            )
            self.assertEqual(
                datamodule.info["gems_eligibility"],
                {
                    "version": "bounded_precursor_rt_ms2_v3",
                    "rule": (
                        "isfinite(ms_level) and ms_level == 2 and "
                        "isfinite(precursor_mz) and min_precursor_mz <= "
                        "precursor_mz and precursor_mz <= "
                        "max_precursor_mz and isfinite(retention_time) and "
                        "retention_time > 0"
                    ),
                    "ms_level_dataset": "MS level",
                    "required_ms_level": 2,
                    "precursor_dataset": "precursor_mz",
                    "spectrum_dataset": "spectrum",
                    "spectrum_trailing_shape": [2, 128],
                    "retention_time_dataset": "RT",
                    "min_precursor_mz": 1.0,
                    "max_precursor_mz": 1000.0,
                    "source_count": source_size,
                    "eligible_count": source_size - len(invalid),
                    "excluded_count": len(invalid),
                    "train_source_count": source_size - GEMS_SPLIT_CHUNK_ROWS,
                    "train_eligible_count": (
                        source_size - GEMS_SPLIT_CHUNK_ROWS - 5
                    ),
                    "train_excluded_count": 5,
                    "validation_source_count": GEMS_SPLIT_CHUNK_ROWS,
                    "validation_eligible_count": GEMS_SPLIT_CHUNK_ROWS - 4,
                    "validation_excluded_count": 4,
                },
            )

    def test_validation_sampler_is_deterministic_and_not_source_ordered(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = self._make_config(tmp_path)
            cfg.batch_size = 16
            source_size = GEMS_SPLIT_CHUNK_ROWS * GEMS_SPLIT_MODULUS
            _write_fake_hdf5_shards(
                self._artifact_dir(cfg),
                [source_size],
                precursor_mz=np.full(source_size, 100.0, dtype=np.float32),
            )

            datamodule = gems.GemsDataModule(cfg, seed=999)
            first = next(
                iter(datamodule.val_loader_for_eval(augment=False).batch_sampler)
            )
            second = next(
                iter(datamodule.val_loader_for_eval(augment=False).batch_sampler)
            )

            self.assertEqual(first, second)
            self.assertNotEqual(first, list(range(len(first))))

    def test_datamodule_keeps_num_peaks(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = self._make_config(tmp_path)
            cfg.num_peaks = 4

            def fake_snapshot_download(*, local_dir, **kwargs):
                _write_fake_hdf5_shards(Path(local_dir), [3, 2])
                return str(local_dir)

            with mock.patch.object(gems_artifacts, "snapshot_download",
                side_effect=fake_snapshot_download,
            ):
                datamodule = gems.GemsDataModule(cfg, seed=42)
                batch = next(iter(datamodule.train_loader_for_epoch(0)))

        self.assertEqual(cfg.num_peaks, 4)
        self.assertEqual(datamodule.info["num_peaks"], 4)
        self.assertEqual(tuple(batch["peak_mz"].shape), (2, 4))
        self.assertEqual(tuple(batch["target_masks"].shape), (2, 1, 4))

    def test_datamodule_rank_one_waits_for_hdf5_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = self._make_config(tmp_path)
            download_calls = []

            def fake_snapshot_download(**kwargs):
                download_calls.append(kwargs)
                return str(kwargs["local_dir"])

            def fake_barrier():
                _write_fake_hdf5_shards(self._artifact_dir(cfg), [3, 2])

            with (
                mock.patch.object(
                    gems_artifacts,
                    "snapshot_download",
                    side_effect=fake_snapshot_download,
                ),
                mock.patch.object(
                    gems_artifacts.torch.distributed,
                    "is_available",
                    return_value=True,
                ),
                mock.patch.object(
                    gems_artifacts.torch.distributed,
                    "is_initialized",
                    return_value=True,
                ),
                mock.patch.object(
                    gems_artifacts.torch.distributed,
                    "barrier",
                    side_effect=fake_barrier,
                ) as barrier_mock,
            ):
                datamodule = gems.GemsDataModule(
                    cfg,
                    seed=42,
                    distributed_world_size=2,
                    distributed_rank=1,
                )

        self.assertEqual(download_calls, [])
        barrier_mock.assert_called_once()
        self.assertEqual(datamodule.info["train_size"], 5)

    def test_datamodule_local_rank_zero_downloads_on_nonzero_global_rank(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = self._make_config(tmp_path)
            cfg.batch_size = 4
            download_calls = []

            def fake_snapshot_download(**kwargs):
                download_calls.append(kwargs)
                _write_fake_hdf5_shards(Path(kwargs["local_dir"]), [3, 2])
                return str(kwargs["local_dir"])

            with (
                mock.patch.object(
                    gems_artifacts,
                    "snapshot_download",
                    side_effect=fake_snapshot_download,
                ),
                mock.patch.object(
                    gems_artifacts.torch.distributed,
                    "is_available",
                    return_value=True,
                ),
                mock.patch.object(
                    gems_artifacts.torch.distributed,
                    "is_initialized",
                    return_value=True,
                ),
                mock.patch.object(
                    gems_artifacts.torch.distributed,
                    "barrier",
                    return_value=None,
                ) as barrier_mock,
            ):
                datamodule = gems.GemsDataModule(
                    cfg,
                    seed=42,
                    distributed_world_size=4,
                    distributed_rank=2,
                    distributed_local_rank=0,
                )

        self.assertEqual(len(download_calls), 1)
        barrier_mock.assert_called_once()
        self.assertEqual(datamodule.info["train_size"], 5)

    def test_hdf5_loader_respects_persistent_workers_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = self._make_config(tmp_path)
            cfg.dataloader_num_workers = 1
            cfg.dataloader_persistent_workers = True
            cfg.dataloader_prefetch_factor = 2

            def fake_snapshot_download(*, local_dir, **kwargs):
                _write_fake_hdf5_shards(Path(local_dir), [3, 2])
                return str(local_dir)

            with mock.patch.object(gems_artifacts, "snapshot_download",
                side_effect=fake_snapshot_download,
            ):
                datamodule = gems.GemsDataModule(cfg, seed=42)
                loader = datamodule.train_loader_for_epoch(0)

            self.assertEqual(loader.num_workers, 1)
            self.assertTrue(loader.persistent_workers)

    def test_hdf5_loader_reaches_second_epoch_with_persistent_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = self._make_config(tmp_path)
            cfg.batch_size = 1
            cfg.dataloader_num_workers = 1
            cfg.dataloader_persistent_workers = True
            cfg.dataloader_prefetch_factor = 2

            def fake_snapshot_download(*, local_dir, **kwargs):
                _write_fake_hdf5_shards(Path(local_dir), [3, 2])
                return str(local_dir)

            with mock.patch.object(gems_artifacts, "snapshot_download",
                side_effect=fake_snapshot_download,
            ):
                datamodule = gems.GemsDataModule(cfg, seed=42)
                loader0 = datamodule.train_loader_for_epoch(0)
                epoch0_batches = list(loader0)
                del loader0
                loader1 = datamodule.train_loader_for_epoch(1)
                batch1 = next(iter(loader1))

            self.assertEqual(len(epoch0_batches), datamodule.train_steps)
            self.assertIn("peak_mz", batch1)
            self.assertIn("context_mask", batch1)
            self.assertEqual(tuple(batch1["peak_mz"].shape), (1, 64))

    def test_train_loader_shuffles_each_epoch_without_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = self._make_config(tmp_path)
            cfg.batch_size = 1
            _write_fake_hdf5_shards(self._artifact_dir(cfg), [5, 4])

            datamodule = gems.GemsDataModule(cfg, seed=42)
            train_dataset = datamodule._get_dataset("train")
            expected_ids = sorted(
                float(train_dataset[idx]["precursor_mz_raw"])
                for idx in range(len(train_dataset))
            )
            epoch0_ids = [
                round(float(batch["precursor_mz"][0]) * 1000.0, 6)
                for batch in datamodule.train_loader_for_epoch(0)
            ]
            epoch1_ids = [
                round(float(batch["precursor_mz"][0]) * 1000.0, 6)
                for batch in datamodule.train_loader_for_epoch(1)
            ]

        self.assertEqual(sorted(epoch0_ids), expected_ids)
        self.assertEqual(sorted(epoch1_ids), expected_ids)
        self.assertEqual(len(set(epoch0_ids)), len(expected_ids))
        self.assertEqual(len(set(epoch1_ids)), len(expected_ids))
        self.assertNotEqual(epoch0_ids, epoch1_ids)

    def test_train_loader_start_batch_matches_epoch_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = self._make_config(tmp_path)
            cfg.batch_size = 2
            _write_fake_hdf5_shards(self._artifact_dir(cfg), [5, 4])

            datamodule = gems.GemsDataModule(cfg, seed=42)
            full_ids = [
                round(float(value) * 1000.0, 6)
                for batch in datamodule.train_loader_for_epoch(0)
                for value in batch["precursor_mz"]
            ]
            offset_loader = datamodule.train_loader_for_epoch(0, start_batch=2)
            offset_ids = [
                round(float(value) * 1000.0, 6)
                for batch in offset_loader
                for value in batch["precursor_mz"]
            ]

            self.assertEqual(offset_ids, full_ids[4:])
            self.assertEqual(len(offset_loader), datamodule.train_steps - 2)

    def test_accumulation_change_preserves_global_batch_offset(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = self._make_config(tmp_path)
            cfg.batch_size = 8
            cfg.gradient_accumulation_steps = 1
            cfg.gems_hdf5_rows_per_block = 8
            cfg.drop_remainder = True
            _write_fake_hdf5_shards(self._artifact_dir(cfg), [16])

            datamodule = gems.GemsDataModule(cfg, seed=42)
            full_ids = [
                round(float(value) * 1000.0, 6)
                for batch in datamodule.train_loader_for_epoch(0)
                for value in batch["precursor_mz"]
            ]
            datamodule.set_gradient_accumulation_steps(2)
            offset_loader = datamodule.train_loader_for_epoch(0, start_batch=1)
            offset_ids = [
                round(float(value) * 1000.0, 6)
                for batch in offset_loader
                for value in batch["precursor_mz"]
            ]

            self.assertEqual(datamodule.global_batch_size, 8)
            self.assertEqual(datamodule.batch_size, 4)
            self.assertEqual(datamodule.train_steps, 2)
            self.assertEqual(len(offset_loader), 2)
            self.assertEqual(offset_ids, full_ids[8:])

    def test_distributed_train_loader_splits_fixed_global_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = self._make_config(tmp_path)
            cfg.batch_size = 4
            cfg.drop_remainder = True
            cfg.dataloader_num_workers = 4
            _write_fake_hdf5_shards(self._artifact_dir(cfg), [8])

            rank_modules = [
                gems.GemsDataModule(
                    cfg,
                    seed=42,
                    distributed_world_size=2,
                    distributed_rank=rank,
                )
                for rank in range(2)
            ]
            rank_batches = [
                [
                    [round(float(value) * 1000.0, 6) for value in batch["precursor_mz"]]
                    for batch in datamodule.train_loader_for_epoch(0)
                ]
                for datamodule in rank_modules
            ]
            offset_rank_batches = [
                [
                    [round(float(value) * 1000.0, 6) for value in batch["precursor_mz"]]
                    for batch in datamodule.train_loader_for_epoch(0, start_batch=1)
                ]
                for datamodule in rank_modules
            ]

        for datamodule in rank_modules:
            self.assertEqual(datamodule.global_batch_size, 4)
            self.assertEqual(datamodule.batch_size, 2)
            self.assertEqual(datamodule.train_steps, 2)
            self.assertEqual(datamodule.dataloader_num_workers, 2)
        self.assertEqual([len(batches) for batches in rank_batches], [2, 2])
        for step in range(2):
            self.assertEqual(len(rank_batches[0][step]), 2)
            self.assertEqual(len(rank_batches[1][step]), 2)
            self.assertFalse(set(rank_batches[0][step]) & set(rank_batches[1][step]))
        self.assertEqual(
            sorted(value for batches in rank_batches for batch in batches for value in batch),
            list(range(1, 9)),
        )
        self.assertEqual([len(batches) for batches in offset_rank_batches], [1, 1])
        self.assertEqual(
            sorted(
                value
                for batches in offset_rank_batches
                for batch in batches
                for value in batch
            ),
            sorted(value for batches in rank_batches for value in batches[1]),
        )

    def test_distributed_train_loader_truncates_uneven_tail_without_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = self._make_config(tmp_path)
            cfg.batch_size = 4
            cfg.drop_remainder = False
            cfg.dataloader_num_workers = 0
            _write_fake_hdf5_shards(self._artifact_dir(cfg), [9])

            rank_modules = [
                gems.GemsDataModule(
                    cfg,
                    seed=123,
                    distributed_world_size=2,
                    distributed_rank=rank,
                )
                for rank in range(2)
            ]
            rank_batches = [
                [
                    [round(float(value) * 1000.0, 6) for value in batch["precursor_mz"]]
                    for batch in datamodule.train_loader_for_epoch(0)
                ]
                for datamodule in rank_modules
            ]
            offset_rank_batches = [
                [
                    [round(float(value) * 1000.0, 6) for value in batch["precursor_mz"]]
                    for batch in datamodule.train_loader_for_epoch(0, start_batch=2)
                ]
                for datamodule in rank_modules
            ]

        for datamodule in rank_modules:
            self.assertEqual(datamodule.global_batch_size, 4)
            self.assertEqual(datamodule.batch_size, 2)
            self.assertEqual(datamodule.train_steps, 2)
        self.assertEqual([len(batches) for batches in rank_batches], [2, 2])
        self.assertEqual(
            [[len(batch) for batch in batches] for batches in rank_batches],
            [[2, 2], [2, 2]],
        )
        combined = [
            value
            for batches in rank_batches
            for batch in batches
            for value in batch
        ]
        self.assertEqual(len(combined), 8)
        self.assertEqual(len(set(combined)), 8)
        self.assertLessEqual(set(combined), set(range(9)))
        self.assertEqual([len(batches) for batches in offset_rank_batches], [0, 0])


class MassSpecPreprocessTests(unittest.TestCase):
    def test_probe_collator_applies_peak_and_precursor_window_filters(self):
        collator = massspec_probe_data._ProbeBatchCollator(
            num_peaks=8,
            max_precursor_mz=1000.0,
            min_peak_intensity=1e-4,
            peak_drop_min_intensity=0.01,
            peak_ordering="mz",
            precursor_peak_exclusion_window_da=5.0,
        )
        spectra = torch.zeros((2, 128), dtype=torch.float32)
        spectra[0, :5] = torch.tensor([90.0, 95.0, 97.0, 98.5, 150.0])
        spectra[1, :4] = torch.tensor([50.0, 60.0, 194.0, 210.0])
        intensity = torch.zeros((2, 128), dtype=torch.float32)
        intensity[0, :5] = torch.tensor([1.0, 0.009, 0.8, 0.7, 0.5])
        intensity[1, :4] = torch.tensor([0.02, 0.5, 0.8, 0.9])
        batch = collator(
            [
                {
                    "spectra": torch.stack([spectra[0], intensity[0]], dim=0),
                    "precursor_mz_raw": 100.0,
                    "fingerprint": torch.zeros(1024, dtype=torch.int32),
                    "smiles": "CCO",
                    "adduct_id": 0,
                    "instrument_type_id": 0,
                    "collision_energy": 0.0,
                    "collision_energy_present": 0,
                    "charge": 1.0,
                    "probe_valid_mol": True,
                    "probe_maccs": torch.zeros(166, dtype=torch.int32),
                    "probe_morgan": torch.zeros(4096, dtype=torch.int32),
                    "probe_mol_weight": 0.0,
                    "probe_logp": 0.0,
                    "probe_num_heavy_atoms": 0.0,
                    "probe_num_rings": 0.0,
                },
                {
                    "spectra": torch.stack([spectra[1], intensity[1]], dim=0),
                    "precursor_mz_raw": 200.0,
                    "fingerprint": torch.zeros(1024, dtype=torch.int32),
                    "smiles": "CCN",
                    "adduct_id": 0,
                    "instrument_type_id": 0,
                    "collision_energy": 0.0,
                    "collision_energy_present": 0,
                    "charge": 1.0,
                    "probe_valid_mol": True,
                    "probe_maccs": torch.zeros(166, dtype=torch.int32),
                    "probe_morgan": torch.zeros(4096, dtype=torch.int32),
                    "probe_mol_weight": 0.0,
                    "probe_logp": 0.0,
                    "probe_num_heavy_atoms": 0.0,
                    "probe_num_rings": 0.0,
                },
            ]
        )

        peak_mz = batch["peak_mz"] * 1000.0
        valid = batch["peak_valid_mask"]
        precursor = batch["precursor_mz"] * 1000.0

        self.assertEqual(int(valid[0].sum().item()), 1)
        self.assertTrue(torch.allclose(peak_mz[0, :1], torch.tensor([90.0])))
        self.assertEqual(int(valid[1].sum().item()), 3)
        self.assertTrue(
            torch.allclose(peak_mz[1, :3], torch.tensor([50.0, 60.0, 194.0]))
        )
        self.assertEqual(
            int(
                (
                    valid
                    & (peak_mz > (precursor.unsqueeze(1) - 5.0))
                ).sum().item()
            ),
            0,
        )
        self.assertGreaterEqual(
            float(batch["peak_intensity"][valid].min().item()),
            0.01 - 1e-6,
        )

    def test_probe_collator_keeps_num_peaks(self):
        collator = massspec_probe_data._ProbeBatchCollator(
            num_peaks=4,
            max_precursor_mz=1000.0,
            min_peak_intensity=1e-4,
            peak_drop_min_intensity=1e-4,
            peak_ordering="mz",
            precursor_peak_exclusion_window_da=0.0,
        )
        spectra = torch.zeros((2, 128), dtype=torch.float32)
        spectra[0, :6] = torch.tensor([90.0, 100.0, 110.0, 120.0, 130.0, 140.0])
        spectra[1, :6] = torch.tensor([190.0, 200.0, 210.0, 220.0, 230.0, 240.0])
        intensity = torch.zeros((2, 128), dtype=torch.float32)
        intensity[:, :6] = torch.tensor([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
        samples = []
        for idx, precursor_mz in enumerate((150.0, 250.0)):
            samples.append(
                {
                    "spectra": torch.stack([spectra[idx], intensity[idx]], dim=0),
                    "precursor_mz_raw": precursor_mz,
                    "fingerprint": torch.zeros(1024, dtype=torch.int32),
                    "smiles": "CCO",
                    "adduct_id": 0,
                    "instrument_type_id": 0,
                    "collision_energy": 0.0,
                    "collision_energy_present": 0,
                    "charge": 1.0,
                    "probe_valid_mol": True,
                    "probe_maccs": torch.zeros(166, dtype=torch.int32),
                    "probe_morgan": torch.zeros(4096, dtype=torch.int32),
                    "probe_mol_weight": 0.0,
                    "probe_logp": 0.0,
                    "probe_num_heavy_atoms": 0.0,
                    "probe_num_rings": 0.0,
                }
            )

        batch = collator(samples)

        self.assertEqual(tuple(batch["peak_mz"].shape), (2, 4))
        self.assertEqual(batch["peak_valid_mask"].sum().item(), 8)


if __name__ == "__main__":
    unittest.main()
