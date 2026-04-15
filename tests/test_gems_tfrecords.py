import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import tensorflow as tf
from ml_collections import config_dict
from torch.utils.data import DataLoader

import input_pipeline
import utils.massspec_probe_data as massspec_probe_data
from scripts.benchmark_linear_probe import preprocess_dreams_spectra
from scripts.prepare_gems_native import main as prepare_gems_main
from utils.gems_native import (
    GEMS_NATIVE_METADATA_VERSION,
    build_gems_native_artifact,
)
from utils.gems_tfrecords import (
    CANONICAL_NUM_SHARDS,
)


def _write_fake_gems_hdf5(path: Path) -> None:
    spectra = np.zeros((4, 2, 128), dtype=np.float32)
    spectra[0, 0, :4] = [100.0, 120.0, 140.0, 160.0]
    spectra[0, 1, :4] = [1.0, 0.8, 0.6, 0.4]
    spectra[1, 0, :3] = [200.0, 220.0, 240.0]
    spectra[1, 1, :3] = [0.9, 0.7, 0.5]
    spectra[2, 0, :2] = [300.0, 320.0]
    spectra[2, 1, :2] = [0.6, 0.3]
    spectra[3, 0, :5] = [400.0, 420.0, 440.0, 460.0, 480.0]
    spectra[3, 1, :5] = [1.0, 0.9, 0.8, 0.7, 0.6]
    retention = np.asarray([10.0, 20.0, -1.0, 40.0], dtype=np.float32)
    precursor = np.asarray([500.0, 600.0, 700.0, 800.0], dtype=np.float32)

    with h5py.File(path, "w") as f:
        f.create_dataset("spectrum", data=spectra)
        f.create_dataset("RT", data=retention)
        f.create_dataset("precursor_mz", data=precursor)


def _write_fake_native_shards(root: Path, lengths: list[int], num_peaks: int = 4) -> list[dict[str, int | str]]:
    entries: list[dict[str, int | str]] = []
    start = 0
    for shard_idx, length in enumerate(lengths):
        shard_dir = root / f"shard-{shard_idx:05d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        peak_mz = np.zeros((length, num_peaks), dtype=np.float32)
        peak_intensity = np.zeros((length, num_peaks), dtype=np.float32)
        peak_valid_mask = np.ones((length, num_peaks), dtype=bool)
        precursor_mz = np.arange(start, start + length, dtype=np.float32)
        peak_mz[:, 0] = precursor_mz + 1000.0
        peak_intensity[:, 0] = 1.0
        np.save(shard_dir / "peak_mz.npy", peak_mz)
        np.save(shard_dir / "peak_intensity.npy", peak_intensity)
        np.save(shard_dir / "peak_valid_mask.npy", peak_valid_mask)
        np.save(shard_dir / "precursor_mz.npy", precursor_mz)
        entries.append({"dir": str(shard_dir), "length": int(length)})
        start += length
    return entries


class GeMSNativeArtifactTests(unittest.TestCase):
    def test_build_gems_native_artifact_writes_expected_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            hdf5_path = tmp_path / "GeMS_A.hdf5"
            artifact_dir = tmp_path / "artifact"
            _write_fake_gems_hdf5(hdf5_path)

            metadata = build_gems_native_artifact(
                hdf5_path=hdf5_path,
                output_dir=artifact_dir,
                num_workers=1,
                source_path=str(hdf5_path),
            )

            self.assertEqual(
                metadata["gems_native_metadata_version"],
                GEMS_NATIVE_METADATA_VERSION,
            )
            self.assertEqual(metadata["num_shards"], CANONICAL_NUM_SHARDS)
            self.assertEqual(metadata["train_size"] + metadata["validation_size"], 3)
            self.assertTrue((artifact_dir / "metadata.json").exists())
            for name in metadata["train_shards"]:
                self.assertTrue((artifact_dir / "train" / name).exists())
            for name in metadata["validation_shards"]:
                self.assertTrue((artifact_dir / "validation" / name).exists())

            shard_dir = artifact_dir / "train" / metadata["train_shards"][0]
            peak_mz = np.load(shard_dir / "peak_mz.npy")
            peak_intensity = np.load(shard_dir / "peak_intensity.npy")
            peak_valid_mask = np.load(shard_dir / "peak_valid_mask.npy")
            precursor_mz = np.load(shard_dir / "precursor_mz.npy")

            self.assertEqual(tuple(peak_mz.shape), (1, 64))
            self.assertEqual(peak_intensity.shape, peak_mz.shape)
            self.assertEqual(peak_valid_mask.shape, peak_mz.shape)
            self.assertEqual(tuple(precursor_mz.shape), (1,))

    def test_build_gems_native_artifact_filters_large_precursor(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            hdf5_path = tmp_path / "GeMS_A.hdf5"
            artifact_dir = tmp_path / "artifact"
            _write_fake_gems_hdf5(hdf5_path)

            metadata = build_gems_native_artifact(
                hdf5_path=hdf5_path,
                output_dir=artifact_dir,
                max_precursor_mz=650.0,
                num_workers=1,
                source_path=str(hdf5_path),
            )

            self.assertEqual(metadata["train_size"] + metadata["validation_size"], 2)
            self.assertEqual(metadata["max_precursor_mz"], 650.0)

    def test_prepare_gems_native_script_builds_and_uploads(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            hdf5_path = tmp_path / "GeMS_A.hdf5"
            _write_fake_gems_hdf5(hdf5_path)

            with (
                mock.patch("scripts.prepare_gems_native.HfApi") as api_cls,
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "prepare_gems_native.py",
                        "--source-hdf5-path",
                        str(hdf5_path),
                        "--hf-repo-id",
                        "cjim8889/test-gems-native",
                        "--work-dir",
                        str(tmp_path / "work"),
                        "--hf-revision",
                        "main",
                        "--num-peaks",
                        "64",
                        "--num-workers",
                        "1",
                    ],
                ),
            ):
                api = api_cls.return_value
                prepare_gems_main()

            artifact_dir = tmp_path / "work" / "artifact"
            self.assertTrue((artifact_dir / "metadata.json").exists())
            api.create_repo.assert_called_once_with(
                "cjim8889/test-gems-native",
                repo_type="dataset",
                exist_ok=True,
            )
            api.upload_large_folder.assert_called_once()
            _, kwargs = api.upload_large_folder.call_args
            self.assertEqual(kwargs["repo_id"], "cjim8889/test-gems-native")
            self.assertEqual(Path(kwargs["folder_path"]), artifact_dir)
            self.assertEqual(kwargs["repo_type"], "dataset")
            self.assertEqual(kwargs["revision"], "main")

    def test_build_gems_native_artifact_supports_parallel_shard_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            hdf5_path = tmp_path / "GeMS_A.hdf5"
            artifact_dir = tmp_path / "artifact"
            _write_fake_gems_hdf5(hdf5_path)

            metadata = build_gems_native_artifact(
                hdf5_path=hdf5_path,
                output_dir=artifact_dir,
                num_workers=2,
                source_path=str(hdf5_path),
            )

            self.assertEqual(metadata["train_size"] + metadata["validation_size"], 3)
            for name in metadata["train_shards"]:
                self.assertTrue((artifact_dir / "train" / name).exists())


class GeMSRuntimeDownloadTests(unittest.TestCase):
    def _make_config(self, tmp_path: Path) -> config_dict.ConfigDict:
        cfg = config_dict.ConfigDict()
        cfg.tfrecord_dir = str(tmp_path / "cache")
        cfg.gems_native_repo_id = "cjim8889/gems-a-native"
        cfg.gems_native_revision = "unit-test"
        cfg.batch_size = 2
        cfg.shuffle_buffer = 4
        cfg.tfrecord_buffer_size = 1024
        cfg.drop_remainder = False
        cfg.max_precursor_mz = 1000.0
        cfg.min_peak_intensity = 1e-4
        cfg.peak_ordering = "mz"
        cfg.num_peaks = 64
        cfg.jepa_num_target_blocks = 1
        cfg.jepa_context_fraction = 0.5
        cfg.jepa_target_fraction = 0.5
        cfg.jepa_block_min_len = 1
        cfg.augmentation_mz_jitter_std = 0.0
        cfg.augmentation_intensity_jitter_std = 0.0
        return cfg

    def _build_native_artifact(
        self,
        *,
        source_hdf5: Path,
        output_dir: Path,
        cfg: config_dict.ConfigDict,
    ) -> None:
        build_gems_native_artifact(
            hdf5_path=source_hdf5,
            output_dir=output_dir,
            num_peaks=int(cfg.num_peaks),
            min_peak_intensity=float(cfg.min_peak_intensity),
            peak_ordering=str(cfg.peak_ordering),
            max_precursor_mz=float(cfg.max_precursor_mz),
            num_workers=1,
            source_path=str(source_hdf5),
        )

    def test_datamodule_downloads_gems_artifact_and_builds_train_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_hdf5 = tmp_path / "GeMS_A.hdf5"
            _write_fake_gems_hdf5(source_hdf5)
            cfg = self._make_config(tmp_path)

            def fake_snapshot_download(*, local_dir, **kwargs):
                self._build_native_artifact(
                    source_hdf5=source_hdf5,
                    output_dir=Path(local_dir),
                    cfg=cfg,
                )
                return str(local_dir)

            with (
                mock.patch.object(
                    input_pipeline,
                    "snapshot_download",
                    side_effect=fake_snapshot_download,
                ) as download_mock,
            ):
                datamodule = input_pipeline.TfLightningDataModule(cfg, seed=42)
                batch = next(iter(datamodule.train_loader_for_epoch(0)))

            self.assertEqual(datamodule.info["train_size"], 2)
            self.assertEqual(datamodule.info["validation_size"], 1)
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
            self.assertEqual(kwargs["repo_id"], "cjim8889/gems-a-native")
            self.assertEqual(kwargs["revision"], "unit-test")
            self.assertEqual(kwargs["repo_type"], "dataset")

    def test_datamodule_uses_local_gems_cache_without_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_hdf5 = tmp_path / "GeMS_A.hdf5"
            _write_fake_gems_hdf5(source_hdf5)
            cfg = self._make_config(tmp_path)

            artifact_dir = Path(cfg.tfrecord_dir) / "gems"
            self._build_native_artifact(
                source_hdf5=source_hdf5,
                output_dir=artifact_dir,
                cfg=cfg,
            )

            with mock.patch.object(input_pipeline, "snapshot_download") as download_mock:
                datamodule = input_pipeline.TfLightningDataModule(cfg, seed=42)
                batch = next(iter(datamodule.train_loader_for_epoch(0)))

            self.assertIn("peak_mz", batch)
            download_mock.assert_not_called()

    def test_datamodule_replaces_legacy_gems_cache_with_native_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_hdf5 = tmp_path / "GeMS_A.hdf5"
            _write_fake_gems_hdf5(source_hdf5)
            cfg = self._make_config(tmp_path)

            artifact_dir = Path(cfg.tfrecord_dir) / "gems"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            (artifact_dir / "metadata.json").write_text(
                json.dumps({"gems_metadata_version": 1})
            )

            def fake_snapshot_download(*, local_dir, **kwargs):
                self._build_native_artifact(
                    source_hdf5=source_hdf5,
                    output_dir=Path(local_dir),
                    cfg=cfg,
                )
                return str(local_dir)

            with mock.patch.object(
                input_pipeline,
                "snapshot_download",
                side_effect=fake_snapshot_download,
            ) as download_mock:
                datamodule = input_pipeline.TfLightningDataModule(cfg, seed=42)
                batch = next(iter(datamodule.train_loader_for_epoch(0)))

            self.assertEqual(datamodule.info["train_size"], 2)
            self.assertIn("peak_mz", batch)
            download_mock.assert_called_once()

    def test_native_loader_respects_persistent_workers_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_hdf5 = tmp_path / "GeMS_A.hdf5"
            _write_fake_gems_hdf5(source_hdf5)
            cfg = self._make_config(tmp_path)
            cfg.dataloader_num_workers = 1
            cfg.dataloader_persistent_workers = True
            cfg.dataloader_prefetch_factor = 2

            def fake_snapshot_download(*, local_dir, **kwargs):
                self._build_native_artifact(
                    source_hdf5=source_hdf5,
                    output_dir=Path(local_dir),
                    cfg=cfg,
                )
                return str(local_dir)

            with mock.patch.object(
                input_pipeline,
                "snapshot_download",
                side_effect=fake_snapshot_download,
            ):
                datamodule = input_pipeline.TfLightningDataModule(cfg, seed=42)
                loader = datamodule.train_loader_for_epoch(0)

            self.assertEqual(loader.num_workers, 1)
            self.assertTrue(loader.persistent_workers)

    def test_native_loader_reaches_second_epoch_with_persistent_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_hdf5 = tmp_path / "GeMS_A.hdf5"
            _write_fake_gems_hdf5(source_hdf5)
            cfg = self._make_config(tmp_path)
            cfg.batch_size = 1
            cfg.dataloader_num_workers = 1
            cfg.dataloader_persistent_workers = True
            cfg.dataloader_prefetch_factor = 2

            def fake_snapshot_download(*, local_dir, **kwargs):
                self._build_native_artifact(
                    source_hdf5=source_hdf5,
                    output_dir=Path(local_dir),
                    cfg=cfg,
                )
                return str(local_dir)

            with mock.patch.object(
                input_pipeline,
                "snapshot_download",
                side_effect=fake_snapshot_download,
            ):
                datamodule = input_pipeline.TfLightningDataModule(cfg, seed=42)
                loader0 = datamodule.train_loader_for_epoch(0)
                epoch0_batches = list(loader0)
                del loader0
                loader1 = datamodule.train_loader_for_epoch(1)
                batch1 = next(iter(loader1))

            self.assertEqual(len(epoch0_batches), datamodule.train_steps)
            self.assertIn("peak_mz", batch1)
            self.assertIn("context_mask", batch1)
            self.assertEqual(tuple(batch1["peak_mz"].shape), (1, 64))

    def test_train_loader_shuffles_without_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = self._make_config(tmp_path)
            cfg.batch_size = 1
            cfg.dataloader_num_workers = 0

            artifact_dir = Path(cfg.tfrecord_dir) / "gems"
            train_entries = _write_fake_native_shards(
                artifact_dir / "train",
                [5, 4],
                num_peaks=int(cfg.num_peaks),
            )
            val_entries = _write_fake_native_shards(
                artifact_dir / "validation",
                [3],
                num_peaks=int(cfg.num_peaks),
            )
            metadata = {
                "gems_native_metadata_version": GEMS_NATIVE_METADATA_VERSION,
                "num_peaks_input": 128,
                "num_peaks": int(cfg.num_peaks),
                "peak_ordering": str(cfg.peak_ordering),
                "min_peak_intensity": float(cfg.min_peak_intensity),
                "max_precursor_mz": float(cfg.max_precursor_mz),
                "train_shards": [Path(entry["dir"]).name for entry in train_entries],
                "train_lengths": [int(entry["length"]) for entry in train_entries],
                "validation_shards": [
                    Path(entry["dir"]).name for entry in val_entries
                ],
                "validation_lengths": [int(entry["length"]) for entry in val_entries],
                "train_size": 9,
                "validation_size": 3,
                "validation_fraction": 0.25,
                "split_seed": 42,
                "num_shards": len(train_entries),
                "source_hdf5_path": "unit-test",
                "source_url": None,
            }
            artifact_dir.mkdir(parents=True, exist_ok=True)
            (artifact_dir / "metadata.json").write_text(json.dumps(metadata))

            datamodule = input_pipeline.TfLightningDataModule(cfg, seed=42)
            train_dataset = datamodule._get_dataset("train")
            expected_ids = sorted(
                float(train_dataset[idx]["precursor_mz"])
                for idx in range(len(train_dataset))
            )
            epoch0_ids = [
                float(batch["precursor_mz"][0])
                for batch in datamodule.train_loader_for_epoch(0)
            ]
            epoch1_ids = [
                float(batch["precursor_mz"][0])
                for batch in datamodule.train_loader_for_epoch(1)
            ]

        self.assertEqual(sorted(epoch0_ids), expected_ids)
        self.assertEqual(sorted(epoch1_ids), expected_ids)
        self.assertEqual(len(set(epoch0_ids)), len(expected_ids))
        self.assertEqual(len(set(epoch1_ids)), len(expected_ids))

    def test_memmap_loader_multi_worker_covers_each_sample_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            entries = _write_fake_native_shards(tmp_path / "train", [3, 2, 4])
            dataset = input_pipeline._GemsMemmapDataset(entries)
            loader = DataLoader(
                dataset,
                batch_size=2,
                shuffle=False,
                num_workers=4,
                persistent_workers=True,
                prefetch_factor=2,
                collate_fn=input_pipeline._GemsBatchCollator(
                    augment=False,
                    num_target_blocks=1,
                    context_fraction=0.5,
                    target_fraction=0.5,
                    block_min_len=1,
                    mz_jitter_std=0.0,
                    intensity_jitter_std=0.0,
                    use_precursor_token=False,
                ),
            )

            ids = []
            for batch in loader:
                ids.extend(int(v) for v in batch["precursor_mz"].tolist())

        self.assertEqual(ids, list(range(9)))

    def test_memmap_loader_persistent_workers_repeat_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            entries = _write_fake_native_shards(tmp_path / "train", [3, 2, 4])
            dataset = input_pipeline._GemsMemmapDataset(entries)
            loader = DataLoader(
                dataset,
                batch_size=2,
                shuffle=False,
                num_workers=4,
                persistent_workers=True,
                prefetch_factor=2,
                collate_fn=input_pipeline._GemsBatchCollator(
                    augment=False,
                    num_target_blocks=1,
                    context_fraction=0.5,
                    target_fraction=0.5,
                    block_min_len=1,
                    mz_jitter_std=0.0,
                    intensity_jitter_std=0.0,
                    use_precursor_token=False,
                ),
            )

            first_pass = []
            second_pass = []
            for batch in loader:
                first_pass.extend(int(v) for v in batch["precursor_mz"].tolist())
            for batch in loader:
                second_pass.extend(int(v) for v in batch["precursor_mz"].tolist())

        self.assertEqual(first_pass, list(range(9)))
        self.assertEqual(second_pass, list(range(9)))

    def test_missing_gems_repo_id_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._make_config(Path(tmp))
            cfg.gems_native_repo_id = ""
            with self.assertRaisesRegex(ValueError, "gems_native_repo_id"):
                input_pipeline.TfLightningDataModule(cfg, seed=42)


class MassSpecPreprocessTests(unittest.TestCase):
    def test_process_massspec_probe_filters_large_precursor(self):
        spectra = np.zeros((4, 2, 128), dtype=np.float32)
        precursor = np.asarray([500.0, 1200.0, 750.0, 900.0], dtype=np.float32)
        retention = np.ones(4, dtype=np.float32)
        fold = np.asarray(["train", "train", "val", "test"], dtype=object)
        smiles = np.asarray(["CCO", "CCC", "CCN", "CCCl"], dtype=object)
        adduct = np.asarray(["[M+H]+"] * 4, dtype=object)
        instrument = np.asarray(["Orbitrap"] * 4, dtype=object)
        collision_energy = np.asarray([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        collision_energy_present = np.asarray([1, 1, 1, 1], dtype=np.int32)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_tsv = tmp_path / "MassSpecGym.tsv"
            fake_tsv.write_text("unused\n")

            with (
                mock.patch.object(
                    massspec_probe_data, "_download_hf_file", return_value=fake_tsv
                ),
                mock.patch.object(
                    massspec_probe_data,
                    "_load_massspec_tsv",
                    return_value={
                        "spectra": spectra,
                        "retention": retention,
                        "precursor": precursor,
                        "fold": fold,
                        "smiles": smiles,
                        "adduct": adduct,
                        "instrument_type": instrument,
                        "collision_energy": collision_energy,
                        "collision_energy_present": collision_energy_present,
                    },
                ),
            ):
                metadata = massspec_probe_data.ensure_massspec_probe_prepared(
                    tmp_path / "massspec_probe",
                    max_precursor_mz=1000.0,
                    num_shards=4,
                )

            record_path = (
                tmp_path / "massspec_probe" / "train" / metadata["train_files"][0]
            )
            dataset = tf.data.TFRecordDataset(
                [str(record_path)],
                compression_type="GZIP",
            )
            example = next(dataset.as_numpy_iterator())
            parsed = tf.io.parse_single_example(
                example,
                {
                    "probe_mol_weight": tf.io.FixedLenFeature([1], tf.float32),
                    "probe_logp": tf.io.FixedLenFeature([1], tf.float32),
                    "probe_num_heavy_atoms": tf.io.FixedLenFeature([1], tf.float32),
                    "probe_num_rings": tf.io.FixedLenFeature([1], tf.float32),
                    "probe_fg_hydroxyl": tf.io.FixedLenFeature([1], tf.int64),
                    "probe_valid_mol": tf.io.FixedLenFeature([1], tf.int64),
                },
            )

        self.assertEqual(metadata["train_size"], 1)
        self.assertEqual(metadata["val_size"], 1)
        self.assertEqual(metadata["test_size"], 1)
        self.assertEqual(metadata["max_precursor_mz"], 1000.0)
        self.assertEqual(parsed["probe_mol_weight"].shape[0], 1)
        self.assertEqual(parsed["probe_logp"].shape[0], 1)
        self.assertEqual(parsed["probe_num_heavy_atoms"].shape[0], 1)
        self.assertEqual(parsed["probe_num_rings"].shape[0], 1)
        self.assertEqual(parsed["probe_fg_hydroxyl"].shape[0], 1)
        self.assertEqual(parsed["probe_valid_mol"].shape[0], 1)

    def test_benchmark_preprocess_matches_input_pipeline_without_precursor_window(self):
        spectrum = np.zeros((1, 2, 128), dtype=np.float32)
        spectrum[0, 0, :4] = [99.0, 100.5, 101.5, 1020.0]
        spectrum[0, 1, :4] = [0.8, 5e-4, 0.7, 0.9]
        precursor_mz = np.asarray([101.0], dtype=np.float32)

        mz = spectrum[0, 0].tolist()
        intensity = spectrum[0, 1].tolist()
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "mz": tf.train.Feature(
                        float_list=tf.train.FloatList(value=mz)
                    ),
                    "intensity": tf.train.Feature(
                        float_list=tf.train.FloatList(value=intensity)
                    ),
                    "rt": tf.train.Feature(
                        float_list=tf.train.FloatList(value=[0.0])
                    ),
                    "precursor_mz": tf.train.Feature(
                        float_list=tf.train.FloatList(value=[101.0])
                    ),
                }
            )
        )
        transform = input_pipeline._batched_parse_and_transform(
            max_precursor_mz=1000.0,
            min_peak_intensity=1e-4,
            num_peaks=60,
            peak_ordering="mz",
        )
        tf_batch = transform(tf.constant([example.SerializeToString()]))
        benchmark = preprocess_dreams_spectra(
            spectrum,
            precursor_mz,
            num_peaks=60,
            min_intensity=1e-4,
        )

        np.testing.assert_allclose(
            benchmark["peak_mz"], tf_batch["peak_mz"].numpy(), atol=1e-6
        )
        np.testing.assert_allclose(
            benchmark["peak_intensity"],
            tf_batch["peak_intensity"].numpy(),
            atol=1e-6,
        )
        np.testing.assert_array_equal(
            benchmark["peak_valid_mask"], tf_batch["peak_valid_mask"].numpy()
        )
        np.testing.assert_allclose(
            benchmark["precursor_mz"], tf_batch["precursor_mz"].numpy(), atol=1e-6
        )

if __name__ == "__main__":
    unittest.main()
