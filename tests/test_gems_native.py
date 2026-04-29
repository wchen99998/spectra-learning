import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import torch
from ml_collections import config_dict
from torch.utils.data import DataLoader

import spectra_learning.data.gems as gems
import spectra_learning.data.gems.artifacts as gems_artifacts
import spectra_learning.probes.massspec.data as massspec_probe_data
from scripts.benchmark_linear_probe import preprocess_dreams_spectra
from scripts.prepare_gems_native import main as prepare_gems_main
from spectra_learning.data.gems.native import (
    GEMS_NATIVE_METADATA_VERSION,
    build_gems_native_artifact,
)
from spectra_learning.data.gems.arrays import (
    CANONICAL_NUM_SHARDS,
)
from spectra_learning.probes.massspec.targets import (
    build_maccs_targets_for_rows,
    build_morgan_targets_for_rows,
    build_probe_targets_for_rows,
)
from spectra_learning.data.spectra import preprocess_peak_batch_numpy


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


def _write_fake_nist_hdf5(path: Path) -> dict[str, np.ndarray]:
    smiles = np.asarray(["CCO", "CCN", "c1ccccc1O", "CCC"], dtype=object)
    adduct = np.asarray(["[M+H]+", "[M+Na]+", "[M+H]+", "[M+K]+"], dtype=object)
    precursor = np.asarray([111.0, 222.0, 333.0, 1500.0], dtype=np.float32)
    dreams = np.asarray(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ],
        dtype=np.float32,
    )
    spectra = np.zeros((4, 2, 128), dtype=np.float32)
    spectra[0, 0, :3] = [10.0, 11.0, 12.0]
    spectra[0, 1, :3] = [2.0, 6.0, 3.0]
    spectra[1, 0, :3] = [20.0, 21.0, 22.0]
    spectra[1, 1, :3] = [7.0, 1.0, 2.0]
    spectra[2, 0, :3] = [30.0, 31.0, 32.0]
    spectra[2, 1, :3] = [4.0, 8.0, 2.0]
    spectra[3, 0, :3] = [40.0, 41.0, 42.0]
    spectra[3, 1, :3] = [5.0, 9.0, 1.0]

    with h5py.File(path, "w") as f:
        f.create_dataset("spectrum", data=spectra)
        f.create_dataset("precursor_mz", data=precursor)
        f.create_dataset("smiles", data=smiles, dtype=h5py.string_dtype("utf-8"))
        f.create_dataset("adduct", data=adduct, dtype=h5py.string_dtype("utf-8"))
        f.create_dataset("DreaMS_embedding", data=dreams)

    return {
        "spectra": spectra,
        "precursor": precursor,
        "smiles": smiles.astype(str),
        "adduct": adduct.astype(str),
        "dreams": dreams,
    }


def _write_fake_native_shards(root: Path, lengths: list[int], num_peaks: int = 4) -> list[dict[str, int | str]]:
    entries: list[dict[str, int | str]] = []
    start = 0
    for shard_idx, length in enumerate(lengths):
        shard_dir = root / f"shard-{shard_idx:05d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        spectra = np.zeros((length, 2, 128), dtype=np.float32)
        precursor_mz_raw = np.arange(start, start + length, dtype=np.float32)
        spectra[:, 0, 0] = precursor_mz_raw + 1000.0
        spectra[:, 1, 0] = 1.0
        np.save(shard_dir / "spectra.npy", spectra)
        np.save(shard_dir / "precursor_mz_raw.npy", precursor_mz_raw)
        entries.append({"dir": str(shard_dir), "length": int(length)})
        start += length
    return entries


def _write_fake_nist_full_probe_artifact(
    root: Path,
    *,
    max_precursor_mz: float = 1000.0,
) -> dict[str, object]:
    split_lengths = {
        "train": [8],
        "val": [4],
        "test": [2],
    }
    metadata: dict[str, object] = {
        "metadata_version": massspec_probe_data.NIST_FULL_METADATA_VERSION,
        "artifact_format": massspec_probe_data.NIST_FULL_ARTIFACT_FORMAT,
        "max_precursor_mz": max_precursor_mz,
        "adduct_vocab": {"unknown": 0},
        "instrument_type_vocab": {"unknown": 0},
        "dreams_dim": 0,
        "probe_maccs_bits": 166,
    }
    for split_name, lengths in split_lengths.items():
        shard_names = []
        split_dir = root / split_name
        for shard_idx in range(len(lengths)):
            shard_name = f"shard-{shard_idx:05d}-of-{len(lengths):05d}"
            (split_dir / shard_name).mkdir(parents=True, exist_ok=True)
            shard_names.append(shard_name)
        metadata[f"{split_name}_files"] = shard_names
        metadata[f"{split_name}_lengths"] = lengths
        metadata[f"{split_name}_size"] = sum(lengths)
    root.mkdir(parents=True, exist_ok=True)
    (root / "metadata.json").write_text(json.dumps(metadata))
    return metadata


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
            spectra = np.load(shard_dir / "spectra.npy")
            precursor_mz_raw = np.load(shard_dir / "precursor_mz_raw.npy")

            self.assertEqual(tuple(spectra.shape), (1, 2, 128))
            self.assertEqual(tuple(precursor_mz_raw.shape), (1,))

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
        cfg.artifact_dir = str(tmp_path / "cache")
        cfg.gems_native_repo_id = "cjim8889/gems-a-native"
        cfg.gems_native_revision = "unit-test"
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
                mock.patch.object(gems_artifacts, "snapshot_download",
                    side_effect=fake_snapshot_download,
                ) as download_mock,
            ):
                datamodule = gems.GemsNativeDataModule(cfg, seed=42)
                batch = next(iter(datamodule.train_loader_for_epoch(0)))

            self.assertEqual(datamodule.info["train_size"], 2)
            self.assertEqual(datamodule.info["validation_size"], 1)
            self.assertEqual(datamodule.info["num_peaks"], 64)
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

    def test_datamodule_separates_real_peaks_from_precursor_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_hdf5 = tmp_path / "GeMS_A.hdf5"
            _write_fake_gems_hdf5(source_hdf5)
            cfg = self._make_config(tmp_path)
            cfg.num_peaks = 4
            cfg.use_precursor_token = True
            cfg.dataloader_num_workers = 0

            def fake_snapshot_download(*, local_dir, **kwargs):
                self._build_native_artifact(
                    source_hdf5=source_hdf5,
                    output_dir=Path(local_dir),
                    cfg=cfg,
                )
                return str(local_dir)

            with mock.patch.object(gems_artifacts, "snapshot_download",
                side_effect=fake_snapshot_download,
            ):
                datamodule = gems.GemsNativeDataModule(cfg, seed=42)
                batch = next(iter(datamodule.train_loader_for_epoch(0)))

        self.assertEqual(cfg.num_peaks, 4)
        self.assertEqual(datamodule.info["num_peaks"], 4)
        self.assertEqual(tuple(batch["peak_mz"].shape), (2, 5))
        self.assertEqual(tuple(batch["target_masks"].shape), (2, 1, 5))

    def test_datamodule_uses_local_gems_cache_without_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_hdf5 = tmp_path / "GeMS_A.hdf5"
            _write_fake_gems_hdf5(source_hdf5)
            cfg = self._make_config(tmp_path)

            artifact_dir = Path(cfg.artifact_dir) / "gems"
            self._build_native_artifact(
                source_hdf5=source_hdf5,
                output_dir=artifact_dir,
                cfg=cfg,
            )

            with mock.patch.object(gems_artifacts, "snapshot_download") as download_mock:
                datamodule = gems.GemsNativeDataModule(cfg, seed=42)
                batch = next(iter(datamodule.train_loader_for_epoch(0)))

            self.assertIn("peak_mz", batch)
            download_mock.assert_not_called()

    def test_datamodule_replaces_legacy_gems_cache_with_native_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_hdf5 = tmp_path / "GeMS_A.hdf5"
            _write_fake_gems_hdf5(source_hdf5)
            cfg = self._make_config(tmp_path)

            artifact_dir = Path(cfg.artifact_dir) / "gems"
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

            with mock.patch.object(gems_artifacts, "snapshot_download",
                side_effect=fake_snapshot_download,
            ) as download_mock:
                datamodule = gems.GemsNativeDataModule(cfg, seed=42)
                batch = next(iter(datamodule.train_loader_for_epoch(0)))

            self.assertEqual(datamodule.info["train_size"], 2)
            self.assertIn("peak_mz", batch)
            download_mock.assert_called_once()

    def test_datamodule_reuses_same_raw_artifact_for_peak_filter_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_hdf5 = tmp_path / "GeMS_A.hdf5"
            _write_fake_gems_hdf5(source_hdf5)
            cfg = self._make_config(tmp_path)
            cfg.peak_drop_min_intensity = 1e-3
            cfg.precursor_peak_exclusion_window_da = 5.0

            def fake_snapshot_download(*, local_dir, **kwargs):
                self._build_native_artifact(
                    source_hdf5=source_hdf5,
                    output_dir=Path(local_dir),
                    cfg=cfg,
                )
                return str(local_dir)

            with mock.patch.object(gems_artifacts, "snapshot_download",
                side_effect=fake_snapshot_download,
            ):
                datamodule = gems.GemsNativeDataModule(cfg, seed=42)

            self.assertNotIn("gems_variants", str(datamodule.gems_dir))
            self.assertEqual(datamodule.gems_dir, Path(cfg.artifact_dir) / "gems")

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

            with mock.patch.object(gems_artifacts, "snapshot_download",
                side_effect=fake_snapshot_download,
            ):
                datamodule = gems.GemsNativeDataModule(cfg, seed=42)
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

            with mock.patch.object(gems_artifacts, "snapshot_download",
                side_effect=fake_snapshot_download,
            ):
                datamodule = gems.GemsNativeDataModule(cfg, seed=42)
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

            artifact_dir = Path(cfg.artifact_dir) / "gems"
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
                "artifact_format": "raw_peaklist_v1",
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

            datamodule = gems.GemsNativeDataModule(cfg, seed=42)
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

    def test_memmap_loader_multi_worker_covers_each_sample_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            entries = _write_fake_native_shards(tmp_path / "train", [3, 2, 4])
            dataset = gems.GemsMemmapDataset(entries)
            loader = DataLoader(
                dataset,
                batch_size=2,
                shuffle=False,
                num_workers=4,
                persistent_workers=True,
                prefetch_factor=2,
                collate_fn=gems.GemsBatchCollator(
                    augment=False,
                    num_target_blocks=1,
                    context_fraction=0.5,
                    target_fraction=0.5,
                    block_min_len=1,
                    use_precursor_token=False,
                    num_peaks=4,
                    max_precursor_mz=1000.0,
                    min_peak_intensity=1e-4,
                    peak_drop_min_intensity=1e-4,
                    peak_ordering="mz",
                    precursor_peak_exclusion_window_da=0.0,
                ),
            )

            ids = []
            for batch in loader:
                ids.extend(int(round(float(v) * 1000.0)) for v in batch["precursor_mz"].tolist())

        self.assertEqual(ids, list(range(9)))

    def test_memmap_loader_persistent_workers_repeat_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            entries = _write_fake_native_shards(tmp_path / "train", [3, 2, 4])
            dataset = gems.GemsMemmapDataset(entries)
            loader = DataLoader(
                dataset,
                batch_size=2,
                shuffle=False,
                num_workers=4,
                persistent_workers=True,
                prefetch_factor=2,
                collate_fn=gems.GemsBatchCollator(
                    augment=False,
                    num_target_blocks=1,
                    context_fraction=0.5,
                    target_fraction=0.5,
                    block_min_len=1,
                    use_precursor_token=False,
                    num_peaks=4,
                    max_precursor_mz=1000.0,
                    min_peak_intensity=1e-4,
                    peak_drop_min_intensity=1e-4,
                    peak_ordering="mz",
                    precursor_peak_exclusion_window_da=0.0,
                ),
            )

            first_pass = []
            second_pass = []
            for batch in loader:
                first_pass.extend(int(round(float(v) * 1000.0)) for v in batch["precursor_mz"].tolist())
            for batch in loader:
                second_pass.extend(int(round(float(v) * 1000.0)) for v in batch["precursor_mz"].tolist())

        self.assertEqual(first_pass, list(range(9)))
        self.assertEqual(second_pass, list(range(9)))

    def test_missing_gems_repo_id_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._make_config(Path(tmp))
            cfg.gems_native_repo_id = ""
            with self.assertRaisesRegex(ValueError, "gems_native_repo_id"):
                gems.GemsNativeDataModule(cfg, seed=42)


class MassSpecPreprocessTests(unittest.TestCase):
    def test_probe_data_uses_dedicated_msg_probe_batch_size(self):
        cfg = config_dict.ConfigDict()
        cfg.artifact_dir = "/tmp/probe-cache"
        cfg.probe_dataset = "massspec"
        cfg.batch_size = 2048
        cfg.msg_probe_batch_size = 256
        cfg.shuffle_buffer = 4
        cfg.max_precursor_mz = 1000.0
        cfg.min_peak_intensity = 1e-4
        cfg.peak_ordering = "mz"
        cfg.num_peaks = 60

        metadata = {
            "train_size": 8,
            "val_size": 4,
            "test_size": 2,
            "metadata_version": massspec_probe_data.MASSSPEC_METADATA_VERSION,
            "adduct_vocab": {"unknown": 0},
            "instrument_type_vocab": {"unknown": 0},
            "train_files": ["shard-00000-of-00001"],
            "train_lengths": [8],
            "val_files": ["shard-00000-of-00001"],
            "val_lengths": [4],
            "test_files": ["shard-00000-of-00001"],
            "test_lengths": [2],
            "dreams_dim": 0,
        }

        with mock.patch.object(
            massspec_probe_data,
            "ensure_massspec_probe_prepared",
            return_value=metadata,
        ):
            probe_data = massspec_probe_data.MassSpecProbeData.from_config(cfg)

        self.assertEqual(probe_data.batch_size, 256)

    def test_probe_data_supports_nist_full_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = config_dict.ConfigDict()
            cfg.artifact_dir = str(tmp_path / "probe-cache")
            cfg.probe_dataset = "nist-full"
            cfg.batch_size = 128
            cfg.max_precursor_mz = 1000.0
            cfg.min_peak_intensity = 1e-4
            cfg.peak_ordering = "mz"
            cfg.num_peaks = 60
            cfg.nist_full_probe_repo_id = "owner/nist-full"
            cfg.nist_full_probe_revision = "unit-test"

            def fake_snapshot_download(*, local_dir, **kwargs):
                _write_fake_nist_full_probe_artifact(Path(local_dir))
                return str(local_dir)

            with mock.patch.object(
                massspec_probe_data,
                "snapshot_download",
                side_effect=fake_snapshot_download,
            ) as download_mock:
                probe_data = massspec_probe_data.MassSpecProbeData.from_config(cfg)

        self.assertEqual(probe_data.info["massspec_train_size"], 8)
        self.assertEqual(
            probe_data.train_files,
            [str(tmp_path / "probe-cache" / "nist_full_probe" / "train" / "shard-00000-of-00001")],
        )
        _, kwargs = download_mock.call_args
        self.assertEqual(kwargs["repo_id"], "owner/nist-full")
        self.assertEqual(kwargs["revision"], "unit-test")
        self.assertEqual(kwargs["repo_type"], "dataset")
        self.assertEqual(
            kwargs["allow_patterns"],
            [
                "metadata.json",
                massspec_probe_data.NIST_FULL_PAIRWISE_ALIGNMENT_FILENAME,
                "train/*",
                "val/*",
                "test/*",
            ],
        )

    def test_probe_data_uses_local_nist_full_artifact_without_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            artifact_root = tmp_path / "probe-cache" / "nist_full_probe"
            _write_fake_nist_full_probe_artifact(artifact_root)

            cfg = config_dict.ConfigDict()
            cfg.artifact_dir = str(tmp_path / "probe-cache")
            cfg.probe_dataset = "nist-full"
            cfg.batch_size = 128
            cfg.max_precursor_mz = 1000.0
            cfg.min_peak_intensity = 1e-4
            cfg.peak_ordering = "mz"
            cfg.num_peaks = 60
            cfg.nist_full_probe_repo_id = "owner/nist-full"

            with mock.patch.object(
                massspec_probe_data,
                "snapshot_download",
            ) as download_mock:
                probe_data = massspec_probe_data.MassSpecProbeData.from_config(cfg)

        self.assertEqual(probe_data.info["massspec_test_size"], 2)
        download_mock.assert_not_called()

    def test_nist_full_artifact_preserves_row_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            hdf5_path = tmp_path / "nist_full.hdf5"
            raw = _write_fake_nist_hdf5(hdf5_path)
            artifact_dir = tmp_path / "artifact"

            metadata = massspec_probe_data.build_nist_full_probe_artifact(
                hdf5_path,
                artifact_dir,
                max_precursor_mz=1000.0,
                num_shards=4,
            )
            self.assertEqual(metadata["instrument_type_vocab"], {"unknown": 0})

            kept_mask = raw["precursor"] <= 1000.0
            kept_smiles = raw["smiles"][kept_mask]
            kept_adduct = raw["adduct"][kept_mask]
            kept_precursor = raw["precursor"][kept_mask]
            kept_dreams = raw["dreams"][kept_mask]
            kept_spectra = massspec_probe_data._normalize_spectra_intensity(
                raw["spectra"][kept_mask].copy()
            )
            fingerprints = massspec_probe_data._compute_morgan_fingerprints(kept_smiles)
            probe_props, _, probe_valid = build_probe_targets_for_rows(kept_smiles)
            probe_maccs, probe_maccs_valid = build_maccs_targets_for_rows(kept_smiles)
            probe_morgan, probe_morgan_valid = build_morgan_targets_for_rows(kept_smiles)
            probe_valid &= probe_maccs_valid & probe_morgan_valid
            expected_by_smiles = {
                str(smiles): {
                    "spectra": kept_spectra[row_idx],
                    "precursor": float(kept_precursor[row_idx]),
                    "dreams_embedding": kept_dreams[row_idx],
                    "fingerprint": fingerprints[row_idx],
                    "probe_maccs": probe_maccs[row_idx],
                    "probe_morgan": probe_morgan[row_idx],
                    "probe_valid_mol": bool(probe_valid[row_idx]),
                    "adduct": str(kept_adduct[row_idx]),
                    **{
                        name: float(values[row_idx])
                        for name, values in probe_props.items()
                    },
                }
                for row_idx, smiles in enumerate(kept_smiles.tolist())
            }
            shard_entries = []
            for split_name in ("train", "val", "test"):
                shard_entries.extend(
                    {
                        "dir": artifact_dir / split_name / shard_name,
                        "length": int(shard_length),
                    }
                    for shard_name, shard_length in zip(
                        metadata[f"{split_name}_files"],
                        metadata[f"{split_name}_lengths"],
                        strict=True,
                    )
                )
            dataset = massspec_probe_data._ProbeMemmapDataset(shard_entries)

            seen_smiles: list[str] = []
            for row_idx in range(len(dataset)):
                sample = dataset[row_idx]
                smiles = str(sample["smiles"])
                seen_smiles.append(smiles)
                expected = expected_by_smiles[smiles]
                np.testing.assert_allclose(
                    sample["spectra"].numpy(),
                    expected["spectra"],
                    atol=1e-6,
                )
                self.assertEqual(float(sample["precursor_mz_raw"]), expected["precursor"])
                np.testing.assert_array_equal(
                    sample["dreams_embedding"].numpy(),
                    expected["dreams_embedding"],
                )
                np.testing.assert_array_equal(
                    sample["fingerprint"].numpy(),
                    expected["fingerprint"],
                )
                np.testing.assert_array_equal(
                    sample["probe_maccs"].numpy(),
                    expected["probe_maccs"],
                )
                np.testing.assert_array_equal(
                    sample["probe_morgan"].numpy(),
                    expected["probe_morgan"],
                )
                self.assertEqual(
                    bool(sample["probe_valid_mol"]),
                    expected["probe_valid_mol"],
                )
                self.assertEqual(
                    int(sample["adduct_id"]),
                    metadata["adduct_vocab"][expected["adduct"]],
                )
                self.assertEqual(int(sample["instrument_type_id"]), 0)
                for name in massspec_probe_data.REGRESSION_TARGET_KEYS:
                    self.assertAlmostEqual(
                        float(sample[f"probe_{name}"]),
                        expected[name],
                        places=5,
                    )

        self.assertCountEqual(seen_smiles, kept_smiles.tolist())
        self.assertNotIn("CCC", seen_smiles)

    def test_balanced_morgan_pair_sampler_writes_fixed_pair_payload(self):
        payload = massspec_probe_data._sample_balanced_morgan_pairs(
            np.asarray(["CC", "CCC", "CCCC", "CCO", "CCCO", "CCCCO"], dtype=str),
            np.ones(6, dtype=bool),
            num_pairs=4,
            bin_size=0.5,
            seed=1,
        )

        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload["tanimoto"].shape, (4,))
        self.assertEqual(payload["bin_counts"].tolist(), [2, 2])
        self.assertEqual(
            payload["left_endpoint"].shape,
            payload["right_endpoint"].shape,
        )

    def test_probe_collator_applies_peak_and_precursor_window_filters(self):
        collator = massspec_probe_data._ProbeBatchCollator(
            num_peaks=8,
            max_precursor_mz=1000.0,
            min_peak_intensity=1e-4,
            peak_drop_min_intensity=0.01,
            peak_ordering="mz",
            use_precursor_token=False,
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

    def test_probe_collator_separates_real_peaks_from_precursor_token(self):
        collator = massspec_probe_data._ProbeBatchCollator(
            num_peaks=4,
            max_precursor_mz=1000.0,
            min_peak_intensity=1e-4,
            peak_drop_min_intensity=1e-4,
            peak_ordering="mz",
            use_precursor_token=True,
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

        self.assertEqual(tuple(batch["peak_mz"].shape), (2, 5))
        self.assertTrue(
            torch.allclose(batch["peak_mz"][:, 0], torch.tensor([0.15, 0.25]))
        )
        self.assertTrue(torch.all(batch["peak_valid_mask"][:, 0]))
        self.assertEqual(batch["peak_valid_mask"][:, 1:].sum().item(), 8)

    def test_process_massspec_probe_filters_large_precursor(self):
        spectra = np.zeros((4, 2, 128), dtype=np.float32)
        precursor = np.asarray([500.0, 1200.0, 750.0, 900.0], dtype=np.float32)
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
            shard_dir = (
                tmp_path / "massspec_probe" / "train" / metadata["train_files"][0]
            )
            probe_mol_weight = np.load(shard_dir / "probe_mol_weight.npy")
            probe_logp = np.load(shard_dir / "probe_logp.npy")
            probe_num_heavy_atoms = np.load(shard_dir / "probe_num_heavy_atoms.npy")
            probe_num_rings = np.load(shard_dir / "probe_num_rings.npy")
            probe_maccs = np.load(shard_dir / "probe_maccs.npy")
            probe_morgan = np.load(shard_dir / "probe_morgan.npy")
            probe_valid_mol = np.load(shard_dir / "probe_valid_mol.npy")

        self.assertEqual(metadata["train_size"], 1)
        self.assertEqual(metadata["val_size"], 1)
        self.assertEqual(metadata["test_size"], 1)
        self.assertEqual(metadata["max_precursor_mz"], 1000.0)
        self.assertEqual(metadata["probe_maccs_bits"], 166)
        self.assertEqual(metadata["probe_morgan_bits"], 4096)
        self.assertEqual(metadata["probe_morgan_radius"], 2)
        self.assertEqual(probe_mol_weight.shape, (1,))
        self.assertEqual(probe_logp.shape, (1,))
        self.assertEqual(probe_num_heavy_atoms.shape, (1,))
        self.assertEqual(probe_num_rings.shape, (1,))
        self.assertEqual(probe_maccs.shape, (1, 166))
        self.assertEqual(probe_morgan.shape, (1, 4096))
        self.assertEqual(probe_valid_mol.shape, (1,))

    def test_benchmark_preprocess_matches_input_pipeline_without_precursor_window(self):
        spectrum = np.zeros((1, 2, 128), dtype=np.float32)
        spectrum[0, 0, :4] = [99.0, 100.5, 101.5, 1020.0]
        spectrum[0, 1, :4] = [0.8, 5e-4, 0.7, 0.9]
        precursor_mz = np.asarray([101.0], dtype=np.float32)

        native = preprocess_peak_batch_numpy(
            spectrum,
            precursor_mz,
            max_precursor_mz=1000.0,
            num_peaks=60,
            peak_drop_min_intensity=1e-4,
            peak_ordering="mz",
        )
        benchmark = preprocess_dreams_spectra(
            spectrum,
            precursor_mz,
            num_peaks=60,
            min_intensity=1e-4,
        )

        np.testing.assert_allclose(
            benchmark["peak_mz"], native["peak_mz"], atol=1e-6
        )
        np.testing.assert_allclose(
            benchmark["peak_intensity"],
            native["peak_intensity"],
            atol=1e-6,
        )
        np.testing.assert_array_equal(benchmark["peak_valid_mask"], native["peak_valid_mask"])
        np.testing.assert_allclose(
            benchmark["precursor_mz"], native["precursor_mz"], atol=1e-6
        )

if __name__ == "__main__":
    unittest.main()
