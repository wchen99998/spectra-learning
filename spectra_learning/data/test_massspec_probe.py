import tempfile
import json
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from ml_collections import config_dict

from spectra_learning.data import murcko as murcko_data
from spectra_learning.data.test_murcko import _write_fake_nist_murcko_probe_artifact
import spectra_learning.data.massspec_probe as massspec_probe_data


def test_precursor_charge_uses_metadata_and_canonical_default() -> None:
    assert murcko_data.precursor_charge_from_metadata_json('{"charge": "2+"}') == 2.0
    assert murcko_data.precursor_charge_from_metadata_json('{"CHARGE": "0+"}') == 1.0
    assert murcko_data.precursor_charge_from_metadata_json("{}") == 1.0


def _write_fake_cached_murcko_probe_artifacts(
    artifact_root: Path,
    *,
    nist_repo_id: str = murcko_data.NIST_MURCKO_HF_REPO,
    nist_revision: str = murcko_data.NIST_MURCKO_HF_REVISION,
    include_dreams: bool = False,
) -> None:
    nist_cache = massspec_probe_data.massspec_source_cache_dir(
        artifact_root,
        nist_repo_id,
        nist_revision,
    )
    _write_fake_nist_murcko_probe_artifact(
        nist_cache / murcko_data.NIST_MURCKO_PREPARED_SUBDIR,
        include_dreams=include_dreams,
    )


class MassSpecProbeMurckoDataTests(unittest.TestCase):
    def test_probe_data_rejects_nist_artifact_contract_mismatch(self):
        invalid_values = {
            "metadata_version": 1,
            "artifact_format": "legacy",
            "min_precursor_mz": 0.0,
            "max_precursor_mz": 999.0,
            "num_peaks_input": 64,
        }
        for key, value in invalid_values.items():
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                artifact_root = Path(tmp) / "probe-cache"
                _write_fake_cached_murcko_probe_artifacts(artifact_root)
                cache = massspec_probe_data.massspec_source_cache_dir(
                    artifact_root,
                    murcko_data.NIST_MURCKO_HF_REPO,
                    murcko_data.NIST_MURCKO_HF_REVISION,
                )
                metadata_path = (
                    cache
                    / murcko_data.NIST_MURCKO_PREPARED_SUBDIR
                    / "metadata.json"
                )
                metadata = json.loads(metadata_path.read_text())
                metadata[key] = value
                metadata_path.write_text(json.dumps(metadata))

                with self.assertRaisesRegex(ValueError, f"{key} mismatch"):
                    massspec_probe_data.MassSpecProbeData.from_config(
                        config_dict.ConfigDict(
                            {"artifact_dir": str(artifact_root)}
                        )
                    )

    def test_probe_data_rejects_peak_ordering_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_fake_cached_murcko_probe_artifacts(tmp_path / "probe-cache")
            cfg = config_dict.ConfigDict(
                {
                    "artifact_dir": str(tmp_path / "probe-cache"),
                    "batch_size": 2,
                    "peak_ordering": "mz",
                    "num_peaks": 4,
                }
            )
            probe_data = massspec_probe_data.MassSpecProbeData.from_config(cfg)

            with self.assertRaisesRegex(
                ValueError,
                "must match the configured ordering",
            ):
                probe_data.build_dataset(
                    "massspec_train",
                    peak_ordering="intensity",
                )

    def test_probe_data_uses_dedicated_msg_probe_batch_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_fake_cached_murcko_probe_artifacts(tmp_path / "probe-cache")
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
                _write_fake_nist_murcko_probe_artifact(
                    Path(local_dir) / murcko_data.NIST_MURCKO_PREPARED_SUBDIR
                )
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
        self.assertEqual(probe_data.dreams_dim, 0)
        self.assertFalse(probe_data.info["pairwise_alignment_available"])
        nist_cache = massspec_probe_data.massspec_source_cache_dir(
            tmp_path / "probe-cache",
            "owner/nist-murcko",
            "unit-test",
        )
        self.assertEqual(
            probe_data.train_files,
            [
                str(
                    nist_cache
                    / murcko_data.NIST_MURCKO_PREPARED_SUBDIR
                    / "train.parquet"
                )
            ],
        )
        self.assertEqual(
            probe_data.test_files,
            [
                str(
                    nist_cache
                    / murcko_data.NIST_MURCKO_PREPARED_SUBDIR
                    / "test.parquet"
                )
            ],
        )
        _, kwargs = download_mock.call_args_list[0]
        self.assertEqual(kwargs["repo_id"], "owner/nist-murcko")
        self.assertEqual(kwargs["revision"], "unit-test")
        self.assertEqual(kwargs["repo_type"], "dataset")
        self.assertEqual(
            kwargs["allow_patterns"],
            [
                f"{murcko_data.NIST_MURCKO_PREPARED_SUBDIR}/metadata.json",
                f"{murcko_data.NIST_MURCKO_PREPARED_SUBDIR}/train.parquet",
                f"{murcko_data.NIST_MURCKO_PREPARED_SUBDIR}/val.parquet",
                f"{murcko_data.NIST_MURCKO_PREPARED_SUBDIR}/test.parquet",
            ],
        )
        self.assertEqual(probe_data.info["massspec_nist_cache_dir"], str(nist_cache))

    def test_probe_data_revision_change_uses_a_new_cache_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_root = Path(tmp) / "probe-cache"
            cfg = config_dict.ConfigDict(
                {
                    "artifact_dir": str(artifact_root),
                    "nist_murcko_probe_repo_id": "owner/nist-murcko",
                    "nist_murcko_probe_revision": "revision-a",
                }
            )

            def fake_snapshot_download(*, local_dir, **_kwargs):
                _write_fake_nist_murcko_probe_artifact(
                    Path(local_dir) / murcko_data.NIST_MURCKO_PREPARED_SUBDIR
                )
                return str(local_dir)

            with mock.patch.object(
                murcko_data,
                "snapshot_download",
                side_effect=fake_snapshot_download,
            ) as download_mock:
                first = massspec_probe_data.MassSpecProbeData.from_config(cfg)
                cfg.nist_murcko_probe_revision = "revision-b"
                second = massspec_probe_data.MassSpecProbeData.from_config(cfg)

        self.assertEqual(download_mock.call_count, 2)
        self.assertNotEqual(
            first.info["massspec_nist_source_dir"],
            second.info["massspec_nist_source_dir"],
        )
        self.assertIn("revision-a", first.info["massspec_nist_source_dir"])
        self.assertIn("revision-b", second.info["massspec_nist_source_dir"])
        self.assertEqual(first.peak_ordering, "mz")

    def test_probe_data_supports_canonical_nist_probe_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = config_dict.ConfigDict()
            cfg.artifact_dir = str(tmp_path / "probe-cache")
            cfg.batch_size = 2
            cfg.max_precursor_mz = 1000.0
            cfg.min_peak_intensity = 1e-4
            cfg.peak_ordering = "mz"
            cfg.num_peaks = 4
            cfg.nist_murcko_probe_repo_id = murcko_data.NIST_MURCKO_HF_REPO
            cfg.nist_murcko_probe_revision = "new-nist-rev"
            cfg.nist_murcko_probe_hf_subdir = (
                murcko_data.NIST_MURCKO_PREPARED_SUBDIR
            )

            download_calls = []

            def fake_snapshot_download(
                *,
                local_dir,
                repo_id,
                revision,
                allow_patterns,
                **kwargs,
            ):
                download_calls.append(
                    {
                        "repo_id": repo_id,
                        "revision": revision,
                        "allow_patterns": list(allow_patterns),
                    }
                )
                _write_fake_nist_murcko_probe_artifact(
                    Path(local_dir) / murcko_data.NIST_MURCKO_PREPARED_SUBDIR
                )
                return str(local_dir)

            with mock.patch.object(
                murcko_data,
                "snapshot_download",
                side_effect=fake_snapshot_download,
            ):
                probe_data = massspec_probe_data.MassSpecProbeData.from_config(cfg)

        self.assertEqual(probe_data.info["massspec_train_size"], 2)
        self.assertEqual(
            probe_data.info["massspec_nist_repo_id"],
            murcko_data.NIST_MURCKO_HF_REPO,
        )
        self.assertEqual(
            probe_data.info["massspec_nist_subdir"],
            murcko_data.NIST_MURCKO_PREPARED_SUBDIR,
        )
        nist_cache = massspec_probe_data.massspec_source_cache_dir(
            tmp_path / "probe-cache",
            murcko_data.NIST_MURCKO_HF_REPO,
            "new-nist-rev",
        )
        self.assertEqual(
            probe_data.train_files,
            [
                str(
                    nist_cache
                    / murcko_data.NIST_MURCKO_PREPARED_SUBDIR
                    / "train.parquet"
                )
            ],
        )
        self.assertEqual(
            download_calls,
            [
                {
                    "repo_id": murcko_data.NIST_MURCKO_HF_REPO,
                    "revision": "new-nist-rev",
                    "allow_patterns": [
                        f"{murcko_data.NIST_MURCKO_PREPARED_SUBDIR}/metadata.json",
                        f"{murcko_data.NIST_MURCKO_PREPARED_SUBDIR}/train.parquet",
                        f"{murcko_data.NIST_MURCKO_PREPARED_SUBDIR}/val.parquet",
                        f"{murcko_data.NIST_MURCKO_PREPARED_SUBDIR}/test.parquet",
                    ],
                },
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
                _write_fake_nist_murcko_probe_artifact(
                    Path(local_dir) / murcko_data.NIST_MURCKO_PREPARED_SUBDIR,
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
        _, kwargs = download_mock.call_args_list[0]
        self.assertEqual(
            kwargs["allow_patterns"],
            [
                f"{murcko_data.NIST_MURCKO_PREPARED_SUBDIR}/metadata.json",
                f"{murcko_data.NIST_MURCKO_PREPARED_SUBDIR}/train.parquet",
                f"{murcko_data.NIST_MURCKO_PREPARED_SUBDIR}/val.parquet",
                f"{murcko_data.NIST_MURCKO_PREPARED_SUBDIR}/test.parquet",
                f"{murcko_data.NIST_MURCKO_PREPARED_SUBDIR}/auxiliary/dreams/*",
            ],
        )

    def test_probe_data_loads_nist_murcko_dreams_auxiliary_row_aligned(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_fake_cached_murcko_probe_artifacts(
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
            _write_fake_cached_murcko_probe_artifacts(
                tmp_path / "probe-cache",
                nist_repo_id="owner/nist-murcko",
            )

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
                _write_fake_cached_murcko_probe_artifacts(
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
        self.assertEqual(barrier_mock.call_count, 1)
        self.assertEqual(probe_data.info["massspec_train_size"], 2)

    def test_nist_murcko_download_rejects_non_parquet_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "nist-murcko"
            metadata = _write_fake_nist_murcko_probe_artifact(output_dir)
            metadata["storage_format"] = "native"
            (output_dir / "metadata.json").write_text(json.dumps(metadata))

            with mock.patch.object(
                murcko_data,
                "snapshot_download",
            ) as download_mock:
                with self.assertRaisesRegex(ValueError, "storage_format must be parquet"):
                    murcko_data.ensure_nist_murcko_probe_downloaded(
                        output_dir,
                        max_precursor_mz=1000.0,
                    )

            download_mock.assert_not_called()

    def test_probe_dataset_can_return_jax_batches(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_fake_cached_murcko_probe_artifacts(tmp_path / "probe-cache")

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

    def test_probe_dataset_distributed_eval_does_not_pad_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_fake_cached_murcko_probe_artifacts(tmp_path / "probe-cache")

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
            _write_fake_cached_murcko_probe_artifacts(tmp_path / "probe-cache")

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
