import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import tensorflow as tf
from ml_collections import config_dict

import input_pipeline
import utils.massspec_probe_data as massspec_probe_data
from scripts.prepare_gems_tfrecords import main as prepare_gems_main
from utils.gems_tfrecords import (
    CANONICAL_NUM_SHARDS,
    GEMS_METADATA_VERSION,
    build_gems_tfrecord_artifact,
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


class GeMSTFRecordArtifactTests(unittest.TestCase):
    def test_build_gems_tfrecord_artifact_writes_expected_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            hdf5_path = tmp_path / "GeMS_A.hdf5"
            artifact_dir = tmp_path / "artifact"
            _write_fake_gems_hdf5(hdf5_path)

            metadata = build_gems_tfrecord_artifact(
                hdf5_path=hdf5_path,
                output_dir=artifact_dir,
                num_workers=1,
                source_path=str(hdf5_path),
            )

            self.assertEqual(metadata["gems_metadata_version"], GEMS_METADATA_VERSION)
            self.assertEqual(metadata["num_shards"], CANONICAL_NUM_SHARDS)
            self.assertEqual(metadata["train_size"] + metadata["validation_size"], 3)
            self.assertTrue((artifact_dir / "metadata.json").exists())
            for name in metadata["train_files"]:
                self.assertTrue((artifact_dir / "train" / name).exists())
            for name in metadata["validation_files"]:
                self.assertTrue((artifact_dir / "validation" / name).exists())

            record_path = artifact_dir / "train" / metadata["train_files"][0]
            dataset = tf.data.TFRecordDataset(
                [str(record_path)],
                compression_type="GZIP",
            )
            example = next(dataset.as_numpy_iterator())
            parsed = tf.io.parse_single_example(
                example,
                {
                    "mz": tf.io.FixedLenFeature([128], tf.float32),
                    "intensity": tf.io.FixedLenFeature([128], tf.float32),
                    "rt": tf.io.FixedLenFeature([1], tf.float32),
                    "precursor_mz": tf.io.FixedLenFeature([1], tf.float32),
                },
            )
            self.assertEqual(parsed["mz"].shape[0], 128)
            self.assertEqual(parsed["intensity"].shape[0], 128)
            self.assertEqual(parsed["rt"].shape[0], 1)
            self.assertEqual(parsed["precursor_mz"].shape[0], 1)

    def test_build_gems_tfrecord_artifact_filters_large_precursor(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            hdf5_path = tmp_path / "GeMS_A.hdf5"
            artifact_dir = tmp_path / "artifact"
            _write_fake_gems_hdf5(hdf5_path)

            metadata = build_gems_tfrecord_artifact(
                hdf5_path=hdf5_path,
                output_dir=artifact_dir,
                max_precursor_mz=650.0,
                num_workers=1,
                source_path=str(hdf5_path),
            )

            self.assertEqual(metadata["train_size"] + metadata["validation_size"], 2)
            self.assertEqual(metadata["max_precursor_mz"], 650.0)

    def test_prepare_gems_script_builds_and_uploads(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            hdf5_path = tmp_path / "GeMS_A.hdf5"
            _write_fake_gems_hdf5(hdf5_path)

            with (
                mock.patch("scripts.prepare_gems_tfrecords.HfApi") as api_cls,
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "prepare_gems_tfrecords.py",
                        "--source-hdf5-path",
                        str(hdf5_path),
                        "--hf-repo-id",
                        "cjim8889/test-gems-tfrecords",
                        "--work-dir",
                        str(tmp_path / "work"),
                        "--hf-revision",
                        "main",
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
                "cjim8889/test-gems-tfrecords",
                repo_type="dataset",
                exist_ok=True,
            )
            api.upload_large_folder.assert_called_once()
            _, kwargs = api.upload_large_folder.call_args
            self.assertEqual(kwargs["repo_id"], "cjim8889/test-gems-tfrecords")
            self.assertEqual(Path(kwargs["folder_path"]), artifact_dir)
            self.assertEqual(kwargs["repo_type"], "dataset")
            self.assertEqual(kwargs["revision"], "main")


class GeMSRuntimeDownloadTests(unittest.TestCase):
    def _make_config(self, tmp_path: Path) -> config_dict.ConfigDict:
        cfg = config_dict.ConfigDict()
        cfg.tfrecord_dir = str(tmp_path / "cache")
        cfg.gems_tfrecord_repo_id = "cjim8889/gems-a-tfrecords"
        cfg.gems_tfrecord_revision = "unit-test"
        cfg.batch_size = 2
        cfg.shuffle_buffer = 4
        cfg.tfrecord_buffer_size = 1024
        cfg.drop_remainder = False
        cfg.max_precursor_mz = 1000.0
        cfg.min_peak_intensity = 0.0
        cfg.peak_ordering = "mz"
        cfg.jepa_num_target_blocks = 1
        cfg.jepa_context_fraction = 0.5
        cfg.jepa_target_fraction = 0.5
        cfg.jepa_block_min_len = 1
        cfg.sigreg_mz_jitter_std = 0.0
        cfg.sigreg_intensity_jitter_std = 0.0
        return cfg

    def test_datamodule_downloads_gems_artifact_and_builds_train_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_hdf5 = tmp_path / "GeMS_A.hdf5"
            _write_fake_gems_hdf5(source_hdf5)
            cfg = self._make_config(tmp_path)

            def fake_snapshot_download(*, local_dir, **kwargs):
                build_gems_tfrecord_artifact(
                    hdf5_path=source_hdf5,
                    output_dir=Path(local_dir),
                    num_workers=1,
                    source_path=str(source_hdf5),
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
                batch = next(
                    datamodule._build_dataset_for_files(
                        datamodule.gems_train_files,
                        seed=42,
                        shuffle=True,
                        drop_remainder=datamodule.drop_remainder,
                    ).as_numpy_iterator()
                )

            self.assertEqual(datamodule.info["train_size"], 2)
            self.assertEqual(datamodule.info["validation_size"], 1)
            self.assertNotIn("massspec_train_size", datamodule.info)
            self.assertTrue(
                all(Path(path).exists() for path in datamodule.gems_train_files)
            )
            self.assertIn("peak_mz", batch)
            self.assertIn("context_mask", batch)
            self.assertIn("target_masks", batch)
            _, kwargs = download_mock.call_args
            self.assertEqual(kwargs["repo_id"], "cjim8889/gems-a-tfrecords")
            self.assertEqual(kwargs["revision"], "unit-test")
            self.assertEqual(kwargs["repo_type"], "dataset")

    def test_missing_gems_repo_id_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._make_config(Path(tmp))
            cfg.gems_tfrecord_repo_id = ""
            with self.assertRaisesRegex(ValueError, "gems_tfrecord_repo_id"):
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

    def test_build_gems_tfrecord_artifact_supports_parallel_shard_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            hdf5_path = tmp_path / "GeMS_A.hdf5"
            artifact_dir = tmp_path / "artifact"
            _write_fake_gems_hdf5(hdf5_path)

            metadata = build_gems_tfrecord_artifact(
                hdf5_path=hdf5_path,
                output_dir=artifact_dir,
                num_workers=2,
                source_path=str(hdf5_path),
            )

            self.assertEqual(metadata["train_size"] + metadata["validation_size"], 3)
            for name in metadata["train_files"]:
                self.assertTrue((artifact_dir / "train" / name).exists())


if __name__ == "__main__":
    unittest.main()
