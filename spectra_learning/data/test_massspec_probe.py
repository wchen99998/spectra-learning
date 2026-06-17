import tempfile
import json
import shutil
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from ml_collections import config_dict

from spectra_learning.data import murcko as murcko_data
from spectra_learning.data.test_murcko import _write_fake_nist_murcko_probe_artifact
import spectra_learning.data.massspec_probe as massspec_probe_data


def _write_fake_mcebio_murcko_probe_artifact(
    root: Path,
    *,
    include_dreams: bool = False,
) -> None:
    tmp_root = root.parent / f"{root.name}_tmp"
    if root.exists():
        shutil.rmtree(root)
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    metadata = _write_fake_nist_murcko_probe_artifact(
        tmp_root,
        include_dreams=include_dreams,
    )
    root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tmp_root / "test.parquet", root / "all.parquet")
    shutil.copytree(tmp_root / "auxiliary" / "morgan", root / "auxiliary" / "morgan")
    if include_dreams:
        dreams_dir = root / "auxiliary" / "dreams"
        dreams_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            tmp_root / "auxiliary" / "dreams" / "test-part-00000.npz",
            dreams_dir / "all-part-00000.npz",
        )
    all_metadata = {
        key: value
        for key, value in metadata.items()
        if not key.startswith(("train_", "val_", "test_"))
    }
    all_metadata.update(
        {
            "all_files": ["all.parquet"],
            "all_lengths": metadata["test_lengths"],
            "all_size": metadata["test_size"],
            "all_positive": metadata["test_positive"],
            "all_sulfur_positive": metadata["test_sulfur_positive"],
            "morgan_auxiliary_files": {
                "all": ["auxiliary/morgan/all-part-00000.npz"]
            },
            "morgan_auxiliary_lengths": {"all": metadata["test_lengths"]},
        }
    )
    shutil.copy2(
        tmp_root / "auxiliary" / "morgan" / "test-part-00000.npz",
        root / "auxiliary" / "morgan" / "all-part-00000.npz",
    )
    (root / "auxiliary" / "morgan" / "test-part-00000.npz").unlink()
    if include_dreams:
        all_metadata["dreams_auxiliary_files"] = {
            "all": ["auxiliary/dreams/all-part-00000.npz"]
        }
        all_metadata["dreams_auxiliary_lengths"] = {
            "all": metadata["test_lengths"]
        }
    (root / "metadata.json").write_text(json.dumps(all_metadata))
    shutil.rmtree(tmp_root)


def _write_fake_combined_murcko_probe_artifacts(
    root: Path,
    *,
    include_dreams: bool = False,
) -> None:
    _write_fake_nist_murcko_probe_artifact(
        root / murcko_data.NIST_MURCKO_PREPARED_SUBDIR,
        include_dreams=include_dreams,
    )
    _write_fake_mcebio_murcko_probe_artifact(
        root / murcko_data.MCEBIO_MURCKO_PREPARED_SUBDIR,
        include_dreams=include_dreams,
    )


class MassSpecProbeMurckoDataTests(unittest.TestCase):
    def test_probe_data_uses_dedicated_msg_probe_batch_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_fake_combined_murcko_probe_artifacts(tmp_path / "probe-cache")
            cfg = config_dict.ConfigDict()
            cfg.artifact_dir = str(tmp_path / "probe-cache")
            cfg.batch_size = 2048
            cfg.msg_probe_batch_size = 256
            cfg.shuffle_buffer = 4
            cfg.max_precursor_mz = 1000.0
            cfg.min_peak_intensity = 1e-4
            cfg.peak_ordering = "mz"
            cfg.num_peaks = 4

            probe_data = massspec_probe_data.MassSpecProbeData.from_config(cfg)

        self.assertEqual(probe_data.batch_size, 256)

    def test_probe_data_supports_nist_murcko_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = config_dict.ConfigDict()
            cfg.artifact_dir = str(tmp_path / "probe-cache")
            cfg.batch_size = 2
            cfg.max_precursor_mz = 1000.0
            cfg.min_peak_intensity = 1e-4
            cfg.peak_ordering = "mz"
            cfg.num_peaks = 4
            cfg.nist_murcko_probe_repo_id = "owner/nist-murcko"
            cfg.nist_murcko_probe_revision = "unit-test"

            def fake_snapshot_download(*, local_dir, **kwargs):
                _write_fake_combined_murcko_probe_artifacts(Path(local_dir))
                return str(local_dir)

            with mock.patch.object(
                murcko_data,
                "snapshot_download",
                side_effect=fake_snapshot_download,
            ) as download_mock:
                probe_data = massspec_probe_data.MassSpecProbeData.from_config(cfg)

        self.assertEqual(probe_data.info["massspec_train_size"], 2)
        self.assertEqual(probe_data.info["massspec_val_size"], 1)
        self.assertEqual(probe_data.info["massspec_test_size"], 1)
        self.assertEqual(probe_data.info["massspec_mcebio_test_size"], 1)
        self.assertEqual(probe_data.dreams_dim, 0)
        self.assertEqual(probe_data.storage_format, "parquet")
        self.assertFalse(probe_data.info["pairwise_alignment_available"])
        self.assertEqual(
            probe_data.train_files,
            [
                str(
                    tmp_path
                    / "probe-cache"
                    / "nist_murcko_probe"
                    / "train.parquet"
                )
            ],
        )
        self.assertEqual(
            probe_data.test_files,
            [
                str(
                    tmp_path
                    / "probe-cache"
                    / "nist_murcko_probe"
                    / "test.parquet"
                )
            ],
        )
        self.assertEqual(
            probe_data.mcebio_test_files,
            [
                str(
                    tmp_path
                    / "probe-cache"
                    / "mcebio_murcko_probe"
                    / "all.parquet"
                )
            ],
        )
        _, kwargs = download_mock.call_args
        self.assertEqual(kwargs["repo_id"], "owner/nist-murcko")
        self.assertEqual(kwargs["revision"], "unit-test")
        self.assertEqual(kwargs["repo_type"], "dataset")
        self.assertEqual(
            kwargs["allow_patterns"],
            [
                "nist_murcko_probe/metadata.json",
                "nist_murcko_probe/train.parquet",
                "nist_murcko_probe/val.parquet",
                "nist_murcko_probe/test.parquet",
            ],
        )

    def test_probe_data_downloads_nist_murcko_dreams_auxiliary_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = config_dict.ConfigDict()
            cfg.artifact_dir = str(tmp_path / "probe-cache")
            cfg.batch_size = 2
            cfg.max_precursor_mz = 1000.0
            cfg.min_peak_intensity = 1e-4
            cfg.peak_ordering = "mz"
            cfg.num_peaks = 4
            cfg.nist_murcko_probe_repo_id = "owner/nist-murcko"
            cfg.nist_murcko_probe_revision = "unit-test"
            cfg.nist_murcko_probe_include_dreams_auxiliary = True

            def fake_snapshot_download(*, local_dir, **kwargs):
                _write_fake_combined_murcko_probe_artifacts(
                    Path(local_dir),
                    include_dreams=True,
                )
                return str(local_dir)

            with mock.patch.object(
                murcko_data,
                "snapshot_download",
                side_effect=fake_snapshot_download,
            ) as download_mock:
                probe_data = massspec_probe_data.MassSpecProbeData.from_config(cfg)

        self.assertEqual(probe_data.dreams_dim, 2)
        self.assertTrue(probe_data.info["dreams_auxiliary_available"])
        _, kwargs = download_mock.call_args
        self.assertEqual(
            kwargs["allow_patterns"],
            [
                "nist_murcko_probe/metadata.json",
                "nist_murcko_probe/train.parquet",
                "nist_murcko_probe/val.parquet",
                "nist_murcko_probe/test.parquet",
                "nist_murcko_probe/auxiliary/dreams/*",
            ],
        )

    def test_probe_data_loads_nist_murcko_dreams_auxiliary_row_aligned(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_fake_combined_murcko_probe_artifacts(
                tmp_path / "probe-cache",
                include_dreams=True,
            )

            cfg = config_dict.ConfigDict()
            cfg.artifact_dir = str(tmp_path / "probe-cache")
            cfg.batch_size = 2
            cfg.max_precursor_mz = 1000.0
            cfg.min_peak_intensity = 1e-4
            cfg.peak_ordering = "mz"
            cfg.num_peaks = 4
            cfg.nist_murcko_probe_include_dreams_auxiliary = True

            probe_data = massspec_probe_data.MassSpecProbeData.from_config(cfg)
            batch = next(
                iter(
                    probe_data.build_dataset(
                        "massspec_train",
                        seed=0,
                        peak_ordering="mz",
                        shuffle=False,
                        drop_remainder=False,
                    )
                )
            )

        self.assertEqual(probe_data.dreams_dim, 2)
        self.assertTrue(probe_data.info["dreams_auxiliary_available"])
        self.assertTrue(
            torch.allclose(
                batch["dreams_embedding"],
                torch.tensor([[10.0, 10.5], [11.0, 11.5]]),
            )
        )
        self.assertTrue(
            torch.equal(
                batch["dreams_embedding_valid"],
                torch.tensor([True, False]),
            )
        )

    def test_probe_data_uses_local_nist_murcko_artifact_without_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_fake_combined_murcko_probe_artifacts(tmp_path / "probe-cache")

            cfg = config_dict.ConfigDict()
            cfg.artifact_dir = str(tmp_path / "probe-cache")
            cfg.batch_size = 2
            cfg.max_precursor_mz = 1000.0
            cfg.min_peak_intensity = 1e-4
            cfg.peak_ordering = "mz"
            cfg.num_peaks = 4
            cfg.nist_murcko_probe_repo_id = "owner/nist-murcko"

            with mock.patch.object(
                murcko_data,
                "snapshot_download",
            ) as download_mock:
                probe_data = massspec_probe_data.MassSpecProbeData.from_config(cfg)

        self.assertEqual(probe_data.info["massspec_train_size"], 2)
        self.assertEqual(probe_data.dreams_dim, 0)
        download_mock.assert_not_called()

    def test_probe_data_rank_one_waits_for_nist_murcko_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = config_dict.ConfigDict()
            cfg.artifact_dir = str(tmp_path / "probe-cache")
            cfg.batch_size = 2
            cfg.max_precursor_mz = 1000.0
            cfg.min_peak_intensity = 1e-4
            cfg.peak_ordering = "mz"
            cfg.num_peaks = 4

            download_calls = []

            def fake_snapshot_download(**kwargs):
                download_calls.append(kwargs)
                return str(kwargs["local_dir"])

            def fake_barrier():
                _write_fake_combined_murcko_probe_artifacts(
                    tmp_path / "probe-cache"
                )

            with (
                mock.patch.object(
                    murcko_data,
                    "snapshot_download",
                    side_effect=fake_snapshot_download,
                ),
                mock.patch.object(
                    murcko_data.torch.distributed,
                    "is_available",
                    return_value=True,
                ),
                mock.patch.object(
                    murcko_data.torch.distributed,
                    "is_initialized",
                    return_value=True,
                ),
                mock.patch.object(
                    murcko_data.torch.distributed,
                    "barrier",
                    side_effect=fake_barrier,
                ) as barrier_mock,
            ):
                probe_data = massspec_probe_data.MassSpecProbeData.from_config(
                    cfg,
                    distributed_world_size=2,
                    distributed_rank=1,
                )

        self.assertEqual(download_calls, [])
        self.assertEqual(barrier_mock.call_count, 2)
        self.assertEqual(probe_data.info["massspec_train_size"], 2)

    def test_nist_full_download_rejects_invalid_metadata_without_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_dir = tmp_path / "nist-full"
            output_dir.mkdir()
            (output_dir / "metadata.json").write_text(
                json.dumps({"metadata_version": 0, "max_precursor_mz": 1000.0})
            )

            with mock.patch.object(
                massspec_probe_data,
                "snapshot_download",
            ) as download_mock:
                with self.assertRaisesRegex(ValueError, "Delete the artifact directory"):
                    massspec_probe_data.ensure_nist_full_probe_downloaded(
                        output_dir,
                        max_precursor_mz=1000.0,
                        repo_id="owner/nist-full",
                    )

            download_mock.assert_not_called()

    def test_probe_dataset_can_return_jax_batches(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_fake_combined_murcko_probe_artifacts(tmp_path / "probe-cache")

            cfg = config_dict.ConfigDict()
            cfg.artifact_dir = str(tmp_path / "probe-cache")
            cfg.batch_size = 2
            cfg.max_precursor_mz = 1000.0
            cfg.min_peak_intensity = 1e-4
            cfg.peak_ordering = "mz"
            cfg.num_peaks = 4

            probe_data = massspec_probe_data.MassSpecProbeData.from_config(cfg)
            batch = next(
                iter(
                    probe_data.build_dataset(
                        "massspec_train",
                        seed=0,
                        peak_ordering="mz",
                        shuffle=False,
                        drop_remainder=False,
                        output_format="jax",
                    )
                )
            )

        import jax

        self.assertIsInstance(batch["peak_mz"], jax.Array)
        self.assertIsInstance(batch["probe_maccs"], jax.Array)
        self.assertIsInstance(batch["probe_fluorine"], jax.Array)
        self.assertIsInstance(batch["probe_sulfur"], jax.Array)
        self.assertEqual(tuple(batch["peak_mz"].shape), (2, 4))
        self.assertEqual(batch["smiles"], ["CCO", "CC(F)O"])

    def test_probe_dataset_exposes_mcebio_as_separate_test_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_fake_combined_murcko_probe_artifacts(tmp_path / "probe-cache")

            cfg = config_dict.ConfigDict()
            cfg.artifact_dir = str(tmp_path / "probe-cache")
            cfg.batch_size = 2
            cfg.max_precursor_mz = 1000.0
            cfg.min_peak_intensity = 1e-4
            cfg.peak_ordering = "mz"
            cfg.num_peaks = 4

            probe_data = massspec_probe_data.MassSpecProbeData.from_config(cfg)
            nist_test = next(
                iter(
                    probe_data.build_dataset(
                        "massspec_test",
                        seed=0,
                        peak_ordering="mz",
                        shuffle=False,
                        drop_remainder=False,
                    )
                )
            )
            mcebio_test = next(
                iter(
                    probe_data.build_dataset(
                        "massspec_mcebio_test",
                        seed=0,
                        peak_ordering="mz",
                        shuffle=False,
                        drop_remainder=False,
                    )
                )
            )

        self.assertEqual(probe_data.info["massspec_test_size"], 1)
        self.assertEqual(probe_data.info["massspec_mcebio_test_size"], 1)
        self.assertEqual(nist_test["smiles"], ["c1ccccc1"])
        self.assertEqual(mcebio_test["smiles"], ["c1ccccc1"])
        self.assertIn("mcebio_murcko_probe/all.parquet", probe_data.mcebio_test_files[0])
        self.assertIn("nist_murcko_probe/test.parquet", probe_data.test_files[0])

    def test_probe_dataset_distributed_eval_does_not_pad_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_fake_combined_murcko_probe_artifacts(tmp_path / "probe-cache")

            cfg = config_dict.ConfigDict()
            cfg.artifact_dir = str(tmp_path / "probe-cache")
            cfg.batch_size = 2
            cfg.max_precursor_mz = 1000.0
            cfg.min_peak_intensity = 1e-4
            cfg.peak_ordering = "mz"
            cfg.num_peaks = 4

            probe_data = massspec_probe_data.MassSpecProbeData.from_config(cfg)
            rank0 = list(
                probe_data.build_dataset(
                    "massspec_test",
                    seed=0,
                    peak_ordering="mz",
                    shuffle=False,
                    drop_remainder=False,
                    distributed_world_size=2,
                    distributed_rank=0,
                )
            )
            rank1 = list(
                probe_data.build_dataset(
                    "massspec_test",
                    seed=0,
                    peak_ordering="mz",
                    shuffle=False,
                    drop_remainder=False,
                    distributed_world_size=2,
                    distributed_rank=1,
                )
            )
            padded_rank1 = list(
                probe_data.build_dataset(
                    "massspec_test",
                    seed=0,
                    peak_ordering="mz",
                    shuffle=False,
                    drop_remainder=False,
                    distributed_world_size=2,
                    distributed_rank=1,
                    pad_distributed=True,
                )
            )

        self.assertEqual([batch["smiles"] for batch in rank0], [["c1ccccc1"]])
        self.assertEqual(rank1, [])
        self.assertEqual([batch["smiles"] for batch in padded_rank1], [["c1ccccc1"]])

    def test_indexed_probe_dataset_can_return_jax_batches(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_fake_combined_murcko_probe_artifacts(tmp_path / "probe-cache")

            cfg = config_dict.ConfigDict()
            cfg.artifact_dir = str(tmp_path / "probe-cache")
            cfg.batch_size = 2
            cfg.max_precursor_mz = 1000.0
            cfg.min_peak_intensity = 1e-4
            cfg.peak_ordering = "mz"
            cfg.num_peaks = 4

            probe_data = massspec_probe_data.MassSpecProbeData.from_config(cfg)
            batch = next(
                iter(
                    probe_data.build_indexed_dataset(
                        "massspec_train",
                        np.asarray([1, 0], dtype=np.int64),
                        peak_ordering="mz",
                        drop_remainder=False,
                        output_format="jax",
                    )
                )
            )

        import jax

        self.assertIsInstance(batch["peak_mz"], jax.Array)
        self.assertIsInstance(batch["probe_maccs"], jax.Array)
        self.assertIsInstance(batch["probe_fluorine"], jax.Array)
        self.assertIsInstance(batch["probe_sulfur"], jax.Array)
        self.assertEqual(tuple(batch["peak_mz"].shape), (2, 4))
        self.assertEqual(batch["smiles"], ["CC(F)O", "CCO"])
