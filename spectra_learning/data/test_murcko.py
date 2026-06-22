from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from spectra_learning.data import murcko


def _write_split(path: Path, labels: list[bool]) -> None:
    table = pa.table(
        {
            "dreams_embedding": pa.array(
                [[float(i), float(i + 1)] for i in range(len(labels))],
                type=pa.list_(pa.float32()),
            ),
            "spectrum_mz": pa.array(
                [[10.0, 50.0, 25.0, 900.0] for _ in labels],
                type=pa.list_(pa.float32()),
            ),
            "spectrum_intensity": pa.array(
                [[1.0, 0.5, 0.25, 0.1] for _ in labels],
                type=pa.list_(pa.float32()),
            ),
            "precursor_mz": pa.array([100.0 for _ in labels], type=pa.float64()),
            "has_fluorine": pa.array(labels, type=pa.bool_()),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _write_dreams_auxiliary(root: Path, split: str, length: int, base: float) -> None:
    auxiliary_dir = root / "auxiliary" / "dreams"
    auxiliary_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        auxiliary_dir / f"{split}-part-00000.npz",
        spectrum_index=np.arange(length, dtype=np.int64),
        dreams_embedding=np.asarray(
            [[base + float(i), base + float(i) + 0.5] for i in range(length)],
            dtype=np.float32,
        ),
        dreams_embedding_valid=np.asarray(
            [i % 2 == 0 for i in range(length)],
            dtype=bool,
        ),
    )


def _write_fake_nist_murcko_probe_artifact(
    root: Path,
    *,
    max_precursor_mz: float = 1000.0,
    include_dreams: bool = False,
) -> dict[str, object]:
    rows_by_split = {
        "train": [("CCO", 111.0, False, False), ("CC(F)O", 222.0, True, False)],
        "val": [("CCS", 333.0, False, True)],
        "test": [("c1ccccc1", 444.0, False, False)],
    }
    root.mkdir(parents=True, exist_ok=True)
    morgan_files: dict[str, list[str]] = {}
    morgan_lengths: dict[str, list[int]] = {}
    dreams_files: dict[str, list[str]] = {}
    dreams_lengths: dict[str, list[int]] = {}
    for split_name, rows in rows_by_split.items():
        table = pa.table(
            {
                "spectrum_index": pa.array(list(range(len(rows))), type=pa.int64()),
                "fold": pa.array([split_name] * len(rows), type=pa.string()),
                "precursor_mz": pa.array([precursor for _, precursor, _, _ in rows], type=pa.float32()),
                "num_peaks": pa.array([2] * len(rows), type=pa.int32()),
                "spectrum_mz": pa.array(
                    [[100.0 + idx, 101.0 + idx] for idx, _ in enumerate(rows)],
                    type=pa.list_(pa.float32()),
                ),
                "spectrum_intensity": pa.array(
                    [[10.0, 5.0] for _ in rows],
                    type=pa.list_(pa.float32()),
                ),
                "smiles": pa.array([smiles for smiles, _, _, _ in rows], type=pa.string()),
                "canonical_smiles": pa.array([smiles for smiles, _, _, _ in rows], type=pa.string()),
                "adduct": pa.array(["[M+H]+"] * len(rows), type=pa.string()),
                "instrument_type": pa.array(["Q-TOF"] * len(rows), type=pa.string()),
                "collision_energy": pa.array([10.0] * len(rows), type=pa.float32()),
                "collision_energy_present": pa.array([1] * len(rows), type=pa.int32()),
                "has_fluorine": pa.array([has_f for _, _, has_f, _ in rows], type=pa.bool_()),
                "has_sulfur": pa.array([has_s for _, _, _, has_s in rows], type=pa.bool_()),
                "mol_weight": pa.array([10.0 + idx for idx, _ in enumerate(rows)], type=pa.float32()),
                "logp": pa.array([0.1 + idx for idx, _ in enumerate(rows)], type=pa.float32()),
                "num_heavy_atoms": pa.array([3.0] * len(rows), type=pa.float32()),
                "num_rings": pa.array([0.0] * len(rows), type=pa.float32()),
                "tpsa": pa.array([1.0] * len(rows), type=pa.float32()),
                "num_hbd": pa.array([0.0] * len(rows), type=pa.float32()),
                "num_hba": pa.array([1.0] * len(rows), type=pa.float32()),
                "num_rotatable_bonds": pa.array([1.0] * len(rows), type=pa.float32()),
                "fraction_csp3": pa.array([1.0] * len(rows), type=pa.float32()),
                "formal_charge": pa.array([0.0] * len(rows), type=pa.float32()),
                "num_aromatic_rings": pa.array([0.0] * len(rows), type=pa.float32()),
                "maccs_166": pa.FixedSizeListArray.from_arrays(
                    pa.array(np.zeros(len(rows) * 166, dtype=np.int8), type=pa.int8()),
                    166,
                ),
                "murcko_hist_key": pa.array(["{}"] * len(rows), type=pa.string()),
                "murcko_hist_json": pa.array(["{}"] * len(rows), type=pa.string()),
                "metadata_json": pa.array(["{}"] * len(rows), type=pa.string()),
            }
        )
        pq.write_table(table, root / f"{split_name}.parquet")
        morgan_dir = root / "auxiliary" / "morgan"
        morgan_dir.mkdir(parents=True, exist_ok=True)
        morgan_name = f"{split_name}-part-00000.npz"
        np.savez_compressed(
            morgan_dir / morgan_name,
            spectrum_index=np.arange(len(rows), dtype=np.int64),
            morgan=np.zeros((len(rows), 4096), dtype=np.int8),
        )
        morgan_files[split_name] = [f"auxiliary/morgan/{morgan_name}"]
        morgan_lengths[split_name] = [len(rows)]
        if include_dreams:
            dreams_dir = root / "auxiliary" / "dreams"
            dreams_dir.mkdir(parents=True, exist_ok=True)
            dreams_name = f"{split_name}-part-00000.npz"
            base = {"train": 10.0, "val": 20.0, "test": 30.0}[split_name]
            np.savez_compressed(
                dreams_dir / dreams_name,
                spectrum_index=np.arange(len(rows), dtype=np.int64),
                dreams_embedding=np.asarray(
                    [
                        [base + row_idx, base + row_idx + 0.5]
                        for row_idx in range(len(rows))
                    ],
                    dtype=np.float32,
                ),
                dreams_embedding_valid=np.asarray(
                    [row_idx % 2 == 0 for row_idx in range(len(rows))],
                    dtype=bool,
                ),
            )
            dreams_files[split_name] = [f"auxiliary/dreams/{dreams_name}"]
            dreams_lengths[split_name] = [len(rows)]
    metadata: dict[str, object] = {
        "metadata_version": murcko.NIST_MURCKO_METADATA_VERSION,
        "artifact_format": murcko.NIST_MURCKO_ARTIFACT_FORMAT,
        "storage_format": "parquet",
        "max_precursor_mz": max_precursor_mz,
        "adduct_vocab": {"[M+H]+": 0},
        "instrument_type_vocab": {"Q-TOF": 0},
        "dreams_dim": 2 if include_dreams else 0,
        "probe_maccs_bits": 166,
        "probe_morgan_bits": 4096,
        "probe_morgan_radius": 2,
        "morgan_auxiliary_available": True,
        "morgan_auxiliary_files": morgan_files,
        "morgan_auxiliary_lengths": morgan_lengths,
        "dreams_auxiliary_available": include_dreams,
        "dreams_auxiliary_files": dreams_files,
        "dreams_auxiliary_lengths": dreams_lengths,
        "pairwise_alignment_available": False,
        "pairwise_alignment_num_pairs": 0,
        "pairwise_alignment_num_endpoints": 0,
    }
    for split_name, rows in rows_by_split.items():
        metadata[f"{split_name}_files"] = [f"{split_name}.parquet"]
        metadata[f"{split_name}_lengths"] = [len(rows)]
        metadata[f"{split_name}_size"] = len(rows)
        metadata[f"{split_name}_positive"] = sum(1 for _, _, has_f, _ in rows if has_f)
        metadata[f"{split_name}_sulfur_positive"] = sum(
            1 for _, _, _, has_s in rows if has_s
        )
    (root / "metadata.json").write_text(json.dumps(metadata))
    return metadata


def _mgf_block(
    title: str,
    *,
    pepmass: float,
    smiles: str,
    adduct: str,
    peaks: list[tuple[float, float]],
) -> list[str]:
    lines = [
        "BEGIN IONS",
        f"TITLE={title}",
        f"PEPMASS={pepmass}",
        f"SMILES={smiles}",
        f"PRECURSORTYPE={adduct}",
        "INSTRUMENTTYPE=Q-TOF",
        "COLLISIONENERGY=20",
    ]
    lines.extend(f"{mz} {intensity}" for mz, intensity in peaks)
    lines.append("END IONS")
    return lines


def _read_murcko_artifact_rows(
    artifact_dir: Path,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name in metadata.get("splits", ("train", "val", "test")):
        for filename in metadata[f"{split_name}_files"]:
            table = pq.read_table(artifact_dir / filename)
            rows.extend(table.to_pylist())
    return rows


def _fluorine_data(metadata: dict[str, Any], root: Path) -> murcko.MurckoFluorineData:
    return murcko.MurckoFluorineData(
        metadata=metadata,
        root=root,
        batch_size=2,
        num_peaks=2,
        max_precursor_mz=1000.0,
        min_peak_intensity=1e-4,
        peak_drop_min_intensity=1e-4,
        peak_filtering="top_intensity",
        grouped_peak_shoulder_da=0.05,
        grouped_peak_isotope_charges=(1, 2, 3),
        peak_ordering="mz",
        precursor_peak_exclusion_window_da=0.0,
    )


def test_ensure_nist_murcko_downloads_expected_patterns(monkeypatch, tmp_path: Path):
    def fake_snapshot_download(*, local_dir, **kwargs):
        _write_fake_nist_murcko_probe_artifact(
            Path(local_dir) / murcko.NIST_MURCKO_PREPARED_SUBDIR
        )
        return str(local_dir)

    monkeypatch.setattr(murcko, "snapshot_download", fake_snapshot_download)

    metadata = murcko.ensure_nist_murcko_probe_downloaded(
        tmp_path / "probe-cache" / "nist_murcko_probe",
        max_precursor_mz=1000.0,
        repo_id="owner/nist-murcko",
        revision="unit-test",
    )

    assert metadata["train_size"] == 2


def test_ensure_nist_murcko_downloads_dreams_auxiliary(monkeypatch, tmp_path: Path):
    calls = []

    def fake_snapshot_download(*, local_dir, **kwargs):
        calls.append(kwargs)
        _write_fake_nist_murcko_probe_artifact(
            Path(local_dir) / murcko.NIST_MURCKO_PREPARED_SUBDIR,
            include_dreams=True,
        )
        return str(local_dir)

    monkeypatch.setattr(murcko, "snapshot_download", fake_snapshot_download)

    metadata = murcko.ensure_nist_murcko_probe_downloaded(
        tmp_path / "probe-cache" / "nist_murcko_probe",
        max_precursor_mz=1000.0,
        repo_id="owner/nist-murcko",
        revision="unit-test",
        include_dreams=True,
    )

    assert metadata["dreams_auxiliary_available"]
    assert calls[0]["allow_patterns"] == [
        "nist_murcko_probe/metadata.json",
        "nist_murcko_probe/train.parquet",
        "nist_murcko_probe/val.parquet",
        "nist_murcko_probe/test.parquet",
        "nist_murcko_probe/auxiliary/dreams/*",
    ]


def test_ensure_nist_murcko_rejects_invalid_metadata_without_download(
    monkeypatch,
    tmp_path: Path,
):
    output_dir = tmp_path / "probe-cache" / "nist_murcko_probe"
    output_dir.mkdir(parents=True)
    (output_dir / "metadata.json").write_text(
        json.dumps({"metadata_version": 0, "max_precursor_mz": 1000.0})
    )
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(murcko, "snapshot_download", fake_snapshot_download)

    with pytest.raises(ValueError, match="Delete the artifact directory"):
        murcko.ensure_nist_murcko_probe_downloaded(
            output_dir,
            max_precursor_mz=1000.0,
            repo_id="owner/nist-murcko",
            revision="unit-test",
        )

    assert calls == []


def test_murcko_fluorine_rejects_partial_metadata_without_download(
    monkeypatch,
    tmp_path: Path,
):
    train_dir = tmp_path / "cache" / "nist_murcko_probe"
    train_dir.mkdir(parents=True)
    (train_dir / "metadata.json").write_text(
        json.dumps(
            {
                "train_files": ["missing-train.parquet"],
                "val_files": ["missing-val.parquet"],
            }
        )
    )
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(murcko, "snapshot_download", fake_snapshot_download)

    with pytest.raises(ValueError, match="Delete the artifact directory"):
        murcko.ensure_murcko_fluorine_data_downloaded(tmp_path / "cache")

    assert calls == []


def test_murcko_fluorine_cache_and_loader_use_shared_peak_preprocessing(
    monkeypatch,
    tmp_path: Path,
):
    source_root = tmp_path / "source_repo"
    nist_source = source_root / "nist_murcko_probe"
    mcebio_source = source_root / "mcebio_murcko_probe"
    _write_split(nist_source / "train.parquet", [True, False, False, True])
    _write_split(nist_source / "val.parquet", [True, False])
    _write_split(mcebio_source / "all.parquet", [False, True])
    (nist_source / "metadata.json").write_text(
        """
        {
          "train_files": ["train.parquet"],
          "train_lengths": [4],
          "train_size": 4,
          "train_positive": 2,
          "val_files": ["val.parquet"],
          "val_lengths": [2],
          "val_size": 2,
          "val_positive": 1,
          "dreams_dim": 2,
          "dreams_auxiliary_available": true,
          "dreams_auxiliary_files": {
            "train": ["auxiliary/dreams/train-part-00000.npz"],
            "val": ["auxiliary/dreams/val-part-00000.npz"]
          },
          "dreams_auxiliary_lengths": {
            "train": [4],
            "val": [2]
          },
          "adduct_vocab": {"[M+H]+": 0},
          "instrument_type_vocab": {"Q-TOF": 0}
        }
        """
    )
    (mcebio_source / "metadata.json").write_text(
        """
        {
          "all_files": ["all.parquet"],
          "all_lengths": [2],
          "all_size": 2,
          "all_positive": 1,
          "dreams_dim": 2,
          "dreams_auxiliary_available": true,
          "dreams_auxiliary_files": {
            "all": ["auxiliary/dreams/all-part-00000.npz"]
          },
          "dreams_auxiliary_lengths": {
            "all": [2]
          },
          "adduct_vocab": {"[M+H-H2O]+": 0},
          "instrument_type_vocab": {"Orbitrap": 0}
        }
        """
    )

    def fake_snapshot_download(**kwargs):
        local_dir = Path(kwargs["local_dir"])
        shutil.copytree(source_root, local_dir, dirs_exist_ok=True)
        return str(local_dir)

    monkeypatch.setattr(murcko, "snapshot_download", fake_snapshot_download)
    cache_dir = tmp_path / "cache"
    metadata = murcko.ensure_murcko_fluorine_data_downloaded(
        cache_dir,
        repo_id="unit/repo",
        revision="main",
        train_subdir="nist_murcko_probe",
        test_subdir="mcebio_murcko_probe",
    )

    assert metadata["train_size"] == 4
    assert metadata["train_positive"] == 2
    assert metadata["test_size"] == 2
    assert metadata["test_positive"] == 1
    assert metadata["adduct_vocab"] == {"[M+H-H2O]+": 0, "[M+H]+": 1}
    assert metadata["instrument_type_vocab"] == {"Orbitrap": 0, "Q-TOF": 1}
    assert not metadata["dreams_auxiliary_available"]
    assert "train_dreams_files" not in metadata
    assert "test_dreams_files" not in metadata
    assert (cache_dir / metadata["train_files"][0]).exists()
    assert (cache_dir / metadata["test_files"][0]).exists()

    data = _fluorine_data(metadata, cache_dir)
    batch = next(
        iter(
            murcko.build_murcko_fluorine_loader(
                data,
                "train",
                shuffle=False,
                seed=0,
                max_samples=None,
            )
        )
    )

    assert torch.allclose(batch["peak_mz"][0], torch.tensor([0.025, 0.05]))
    assert torch.allclose(batch["peak_intensity"][0], torch.tensor([0.5, 1.0]))
    assert torch.equal(batch["peak_valid_mask"][0], torch.tensor([True, True]))
    assert torch.allclose(batch["label"], torch.tensor([1.0, 0.0]))

    distributed_batch = next(
        iter(
            murcko.build_murcko_fluorine_loader(
                data,
                "train",
                shuffle=False,
                seed=0,
                max_samples=None,
                distributed_world_size=2,
                distributed_rank=1,
            )
        )
    )
    assert tuple(distributed_batch["label"].shape) == (1,)


def test_murcko_fluorine_loader_reads_dreams_auxiliary(monkeypatch, tmp_path: Path):
    source_root = tmp_path / "source_repo"
    nist_source = source_root / "nist_murcko_probe"
    mcebio_source = source_root / "mcebio_murcko_probe"
    _write_split(nist_source / "train.parquet", [True, False])
    _write_split(nist_source / "val.parquet", [False])
    _write_split(mcebio_source / "all.parquet", [False, True])
    _write_dreams_auxiliary(nist_source, "train", 2, 100.0)
    _write_dreams_auxiliary(nist_source, "val", 1, 200.0)
    _write_dreams_auxiliary(mcebio_source, "all", 2, 300.0)
    (nist_source / "metadata.json").write_text(
        """
        {
          "train_files": ["train.parquet"],
          "train_lengths": [2],
          "train_size": 2,
          "train_positive": 1,
          "val_files": ["val.parquet"],
          "val_lengths": [1],
          "val_size": 1,
          "val_positive": 0,
          "dreams_dim": 2,
          "dreams_auxiliary_available": true,
          "dreams_auxiliary_files": {
            "train": ["auxiliary/dreams/train-part-00000.npz"],
            "val": ["auxiliary/dreams/val-part-00000.npz"]
          },
          "dreams_auxiliary_lengths": {
            "train": [2],
            "val": [1]
          }
        }
        """
    )
    (mcebio_source / "metadata.json").write_text(
        """
        {
          "all_files": ["all.parquet"],
          "all_lengths": [2],
          "all_size": 2,
          "all_positive": 1,
          "dreams_dim": 2,
          "dreams_auxiliary_available": true,
          "dreams_auxiliary_files": {
            "all": ["auxiliary/dreams/all-part-00000.npz"]
          },
          "dreams_auxiliary_lengths": {
            "all": [2]
          }
        }
        """
    )

    def fake_snapshot_download(**kwargs):
        local_dir = Path(kwargs["local_dir"])
        shutil.copytree(source_root, local_dir, dirs_exist_ok=True)
        return str(local_dir)

    monkeypatch.setattr(murcko, "snapshot_download", fake_snapshot_download)
    cache_dir = tmp_path / "cache"
    metadata = murcko.ensure_murcko_fluorine_data_downloaded(
        cache_dir,
        repo_id="unit/repo",
        revision="main",
        train_subdir="nist_murcko_probe",
        test_subdir="mcebio_murcko_probe",
        include_dreams=True,
    )
    data = _fluorine_data(metadata, cache_dir)

    train_batch = next(
        iter(
            murcko.build_murcko_fluorine_loader(
                data,
                "train",
                shuffle=False,
                seed=0,
                max_samples=None,
            )
        )
    )
    test_batch = next(
        iter(
            murcko.build_murcko_fluorine_loader(
                data,
                "test",
                shuffle=False,
                seed=0,
                max_samples=None,
                dreams_only=True,
            )
        )
    )

    assert metadata["dreams_auxiliary_available"]
    assert metadata["test_dreams_files"] == [
        "mcebio_murcko_probe/auxiliary/dreams/all-part-00000.npz"
    ]
    assert torch.allclose(
        train_batch["dreams_embedding"],
        torch.tensor([[100.0, 100.5], [101.0, 101.5]]),
    )
    assert torch.equal(
        train_batch["dreams_embedding_valid"],
        torch.tensor([True, False]),
    )
    assert torch.allclose(
        test_batch["dreams_embedding"],
        torch.tensor([[300.0, 300.5], [301.0, 301.5]]),
    )


def test_murcko_rank_one_waits_for_download(monkeypatch, tmp_path: Path):
    source_root = tmp_path / "source_repo"
    nist_source = source_root / "nist_murcko_probe"
    mcebio_source = source_root / "mcebio_murcko_probe"
    _write_split(nist_source / "train.parquet", [True, False])
    _write_split(nist_source / "val.parquet", [False])
    _write_split(mcebio_source / "all.parquet", [False, True])
    (nist_source / "metadata.json").write_text(
        '{"train_files":["train.parquet"],"train_lengths":[2],"train_size":2,'
        '"val_files":["val.parquet"],"val_lengths":[1],"val_size":1}'
    )
    (mcebio_source / "metadata.json").write_text(
        '{"all_files":["all.parquet"],"all_lengths":[2],"all_size":2}'
    )

    rank1_cache = tmp_path / "rank1_cache"
    download_calls = []

    def rank1_snapshot_download(**kwargs):
        download_calls.append(kwargs)
        return str(kwargs["local_dir"])

    def rank1_barrier():
        shutil.copytree(source_root, rank1_cache, dirs_exist_ok=True)

    monkeypatch.setattr(murcko, "snapshot_download", rank1_snapshot_download)
    monkeypatch.setattr(murcko.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(murcko.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(murcko.torch.distributed, "barrier", rank1_barrier)

    metadata = murcko.ensure_murcko_fluorine_data_downloaded(
        rank1_cache,
        repo_id="unit/repo",
        revision="main",
        train_subdir="nist_murcko_probe",
        test_subdir="mcebio_murcko_probe",
        distributed_world_size=2,
        distributed_rank=1,
    )

    assert download_calls == []
    assert metadata["test_size"] == 2


def test_murcko_local_rank_zero_downloads_on_nonzero_global_rank(
    monkeypatch,
    tmp_path: Path,
):
    source_root = tmp_path / "source_repo"
    nist_source = source_root / "nist_murcko_probe"
    mcebio_source = source_root / "mcebio_murcko_probe"
    _write_split(nist_source / "train.parquet", [True, False])
    _write_split(nist_source / "val.parquet", [False])
    _write_split(mcebio_source / "all.parquet", [False, True])
    (nist_source / "metadata.json").write_text(
        '{"train_files":["train.parquet"],"train_lengths":[2],"train_size":2,'
        '"val_files":["val.parquet"],"val_lengths":[1],"val_size":1}'
    )
    (mcebio_source / "metadata.json").write_text(
        '{"all_files":["all.parquet"],"all_lengths":[2],"all_size":2}'
    )

    rank2_cache = tmp_path / "rank2_cache"
    download_calls = []

    def rank2_snapshot_download(**kwargs):
        download_calls.append(kwargs)
        shutil.copytree(source_root, rank2_cache, dirs_exist_ok=True)
        return str(kwargs["local_dir"])

    monkeypatch.setattr(murcko, "snapshot_download", rank2_snapshot_download)
    monkeypatch.setattr(murcko.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(murcko.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(murcko.torch.distributed, "barrier", lambda: None)

    metadata = murcko.ensure_murcko_fluorine_data_downloaded(
        rank2_cache,
        repo_id="unit/repo",
        revision="main",
        train_subdir="nist_murcko_probe",
        test_subdir="mcebio_murcko_probe",
        distributed_world_size=4,
        distributed_rank=2,
        distributed_local_rank=0,
    )

    assert len(download_calls) == 1
    assert metadata["test_size"] == 2


def test_build_nist_murcko_artifact_from_mgf(tmp_path: Path):
    mgf_path = tmp_path / "source.mgf"
    mgf_path.write_text(
        "\n".join(
            [
                "BEGIN IONS",
                "TITLE=fluoro",
                "PEPMASS=111.0",
                "SMILES=CC(F)O",
                "PRECURSORTYPE=[M+H]+",
                "INSTRUMENTTYPE=Q-TOF",
                "COLLISIONENERGY=10 eV",
                "10 100",
                "20 50",
                "END IONS",
                "BEGIN IONS",
                "TITLE=sulfur",
                "PEPMASS=222.0",
                "SMILES=CCS",
                "PRECURSORTYPE=[M+H]+",
                "INSTRUMENTTYPE=Q-TOF",
                "COLLISIONENERGY=20",
                "11 100",
                "21 50",
                "END IONS",
                "BEGIN IONS",
                "TITLE=ring",
                "PEPMASS=333.0",
                "SMILES=c1ccccc1O",
                "PRECURSORTYPE=[M+Na]+",
                "INSTRUMENTTYPE=Orbitrap",
                "COLLISIONENERGY=30",
                "12 100",
                "22 50",
                "END IONS",
            ]
        )
    )
    artifact_dir = tmp_path / "artifact"

    metadata = murcko.build_murcko_mgf_dataset(
        mgf_path=mgf_path,
        output_dir=artifact_dir,
        source_uri="source.mgf",
        val_frac=0.2,
        test_frac=0.2,
        seed=1,
        min_precursor_mz=1.0,
        max_precursor_mz=1000.0,
        num_peaks_input=128,
        num_workers=1,
        batch_size=2,
        parquet_batch_size=2,
    )

    assert metadata["artifact_format"] == "nist_murcko_parquet_v2"
    assert metadata["train_size"] + metadata["val_size"] + metadata["test_size"] == 3
    assert metadata["dreams_dim"] == 0
    split_name = next(
        split for split in ("train", "val", "test") if metadata[f"{split}_files"]
    )
    parquet_path = artifact_dir / metadata[f"{split_name}_files"][0]
    assert parquet_path.exists()
    table = pq.read_table(parquet_path)
    assert "has_sulfur" in table.column_names
    assert "maccs_166" in table.column_names
    assert (artifact_dir / metadata["morgan_auxiliary_files"][split_name][0]).exists()


def test_build_nist_murcko_artifact_can_filter_to_proton_adduct(tmp_path: Path):
    mgf_path = tmp_path / "source.mgf"
    mgf_path.write_text(
        "\n".join(
            _mgf_block(
                "protonated",
                pepmass=111.0,
                smiles="CCO",
                adduct="[M+H]+",
                peaks=[(10.0, 100.0), (20.0, 50.0)],
            )
            + _mgf_block(
                "sodium",
                pepmass=222.0,
                smiles="CCN",
                adduct="[M+Na]+",
                peaks=[(11.0, 100.0), (21.0, 50.0)],
            )
            + _mgf_block(
                "protonated-ring",
                pepmass=333.0,
                smiles="c1ccccc1O",
                adduct="[M+H]+",
                peaks=[(12.0, 100.0), (22.0, 50.0)],
            )
        )
    )

    metadata = murcko.build_murcko_mgf_dataset(
        mgf_path=mgf_path,
        output_dir=tmp_path / "artifact",
        source_uri="source.mgf",
        val_frac=0.2,
        test_frac=0.2,
        seed=1,
        min_precursor_mz=1.0,
        max_precursor_mz=1000.0,
        num_peaks_input=128,
        num_workers=1,
        batch_size=2,
        parquet_batch_size=2,
        allowed_adducts=("[M+H]+",),
    )

    rows = _read_murcko_artifact_rows(tmp_path / "artifact", metadata)
    assert metadata["train_size"] + metadata["val_size"] + metadata["test_size"] == 2
    assert metadata["adduct_vocab"] == {"[M+H]+": 0}
    assert {row["adduct"] for row in rows} == {"[M+H]+"}
    assert metadata["allowed_adducts"] == ["[M+H]+"]


def test_spectral_lsh_removes_redundant_spectra_and_preserves_smiles(tmp_path: Path):
    mgf_path = tmp_path / "source.mgf"
    duplicate_peaks = [(10.0, 100.0), (20.0, 50.0), (30.0, 25.0)]
    mgf_path.write_text(
        "\n".join(
            _mgf_block(
                "ethanol-a",
                pepmass=111.0,
                smiles="CCO",
                adduct="[M+H]+",
                peaks=duplicate_peaks,
            )
            + _mgf_block(
                "ethanol-b",
                pepmass=112.0,
                smiles="CCO",
                adduct="[M+H]+",
                peaks=duplicate_peaks,
            )
            + _mgf_block(
                "ethylamine",
                pepmass=113.0,
                smiles="CCN",
                adduct="[M+H]+",
                peaks=[(40.0, 100.0), (50.0, 60.0)],
            )
            + _mgf_block(
                "propane",
                pepmass=114.0,
                smiles="CCC",
                adduct="[M+H]+",
                peaks=[(60.0, 100.0), (70.0, 60.0)],
            )
        )
    )
    artifact_dir = tmp_path / "artifact"

    metadata = murcko.build_murcko_mgf_dataset(
        mgf_path=mgf_path,
        output_dir=artifact_dir,
        source_uri="source.mgf",
        val_frac=0.0,
        test_frac=0.0,
        seed=1,
        min_precursor_mz=1.0,
        max_precursor_mz=1000.0,
        num_peaks_input=128,
        num_workers=1,
        batch_size=2,
        parquet_batch_size=2,
        single_split="train",
        allowed_adducts=("[M+H]+",),
        split_size_caps={"train": 4},
        spectral_lsh_threshold=0.90,
    )

    rows = _read_murcko_artifact_rows(artifact_dir, metadata)
    assert metadata["train_size"] == 3
    assert metadata["pre_lsh_split_sizes"]["train"] == 4
    assert metadata["lsh_removed_by_split"]["train"] == 1
    assert metadata["unique_smiles_by_split"]["train"] == 3
    assert {row["canonical_smiles"] for row in rows} == {"CCO", "CCN", "CCC"}


def test_spectral_lsh_reserves_capacity_for_unseen_smiles(tmp_path: Path):
    mgf_path = tmp_path / "source.mgf"
    mgf_path.write_text(
        "\n".join(
            _mgf_block(
                "ethanol-a",
                pepmass=111.0,
                smiles="CCO",
                adduct="[M+H]+",
                peaks=[(10.0, 100.0), (20.0, 50.0)],
            )
            + _mgf_block(
                "ethanol-b",
                pepmass=112.0,
                smiles="CCO",
                adduct="[M+H]+",
                peaks=[(80.0, 100.0), (90.0, 50.0)],
            )
            + _mgf_block(
                "ethylamine",
                pepmass=113.0,
                smiles="CCN",
                adduct="[M+H]+",
                peaks=[(40.0, 100.0), (50.0, 60.0)],
            )
            + _mgf_block(
                "propane",
                pepmass=114.0,
                smiles="CCC",
                adduct="[M+H]+",
                peaks=[(60.0, 100.0), (70.0, 60.0)],
            )
        )
    )
    artifact_dir = tmp_path / "artifact"

    metadata = murcko.build_murcko_mgf_dataset(
        mgf_path=mgf_path,
        output_dir=artifact_dir,
        source_uri="source.mgf",
        val_frac=0.0,
        test_frac=0.0,
        seed=1,
        min_precursor_mz=1.0,
        max_precursor_mz=1000.0,
        num_peaks_input=128,
        num_workers=1,
        batch_size=2,
        parquet_batch_size=2,
        single_split="train",
        allowed_adducts=("[M+H]+",),
        split_size_caps={"train": 3},
        spectral_lsh_threshold=0.90,
    )

    rows = _read_murcko_artifact_rows(artifact_dir, metadata)
    assert metadata["train_size"] == 3
    assert metadata["lsh_removed_by_split"]["train"] == 1
    assert {row["canonical_smiles"] for row in rows} == {"CCO", "CCN", "CCC"}


def test_spectral_lsh_rejects_cap_below_unique_smiles_count(tmp_path: Path):
    mgf_path = tmp_path / "source.mgf"
    mgf_path.write_text(
        "\n".join(
            _mgf_block(
                "ethanol",
                pepmass=111.0,
                smiles="CCO",
                adduct="[M+H]+",
                peaks=[(10.0, 100.0), (20.0, 50.0)],
            )
            + _mgf_block(
                "ethylamine",
                pepmass=112.0,
                smiles="CCN",
                adduct="[M+H]+",
                peaks=[(30.0, 100.0), (40.0, 50.0)],
            )
            + _mgf_block(
                "propane",
                pepmass=113.0,
                smiles="CCC",
                adduct="[M+H]+",
                peaks=[(50.0, 100.0), (60.0, 50.0)],
            )
        )
    )

    with pytest.raises(ValueError, match="smaller than 3 unique canonical SMILES"):
        murcko.build_murcko_mgf_dataset(
            mgf_path=mgf_path,
            output_dir=tmp_path / "artifact",
            source_uri="source.mgf",
            val_frac=0.0,
            test_frac=0.0,
            seed=1,
            min_precursor_mz=1.0,
            max_precursor_mz=1000.0,
            num_peaks_input=128,
            num_workers=1,
            batch_size=2,
            parquet_batch_size=2,
            single_split="train",
            allowed_adducts=("[M+H]+",),
            split_size_caps={"train": 2},
            spectral_lsh_threshold=0.90,
        )


def test_build_mcebio_murcko_artifact_writes_standalone_all_split(tmp_path: Path):
    mgf_path = tmp_path / "source.mgf"
    mgf_path.write_text(
        "\n".join(
            [
                "BEGIN IONS",
                "TITLE=fluoro",
                "PEPMASS=111.0",
                "SMILES=CC(F)O",
                "ADDUCT=[M+H]+",
                "INSTRUMENT_TYPE=Q-TOF",
                "Collision energy=12.0",
                "10 100",
                "20 50",
                "END IONS",
                "BEGIN IONS",
                "TITLE=plain",
                "PEPMASS=222.0",
                "SMILES=CCO",
                "ADDUCT=[M+H]+",
                "INSTRUMENT_TYPE=Orbitrap",
                "Collision energy=24.0",
                "11 100",
                "21 50",
                "END IONS",
            ]
        )
    )
    artifact_dir = tmp_path / "artifact"

    metadata = murcko.build_murcko_mgf_dataset(
        mgf_path=mgf_path,
        output_dir=artifact_dir,
        source_uri="source.mgf",
        val_frac=0.2,
        test_frac=0.2,
        seed=1,
        min_precursor_mz=1.0,
        max_precursor_mz=1000.0,
        num_peaks_input=128,
        num_workers=1,
        batch_size=2,
        parquet_batch_size=2,
        single_split="all",
    )

    assert metadata["splits"] == ["all"]
    assert metadata["all_size"] == 2
    assert metadata["all_positive"] == 1
    assert metadata["all_files"] == ["all.parquet"]
    assert metadata["adduct_vocab"] == {"[M+H]+": 0}
    assert metadata["allowed_adducts"] is None
    assert not metadata["spectral_lsh_enabled"]
    assert (artifact_dir / "all.parquet").exists()

    rows = _read_murcko_artifact_rows(artifact_dir, metadata)
    assert {row["fold"] for row in rows} == {"all"}
    assert {row["instrument_type"] for row in rows} == {"Q-TOF", "Orbitrap"}
    assert [row["collision_energy"] for row in rows] == [12.0, 24.0]


def test_prepare_murcko_mgf_collection_builds_nist_and_mcebio(tmp_path: Path):
    nist_mgf = tmp_path / "nist.mgf"
    nist_mgf.write_text(
        "\n".join(
            _mgf_block(
                "ethanol",
                pepmass=111.0,
                smiles="CCO",
                adduct="[M+H]+",
                peaks=[(10.0, 100.0), (20.0, 50.0)],
            )
            + _mgf_block(
                "fluoro",
                pepmass=112.0,
                smiles="CC(F)O",
                adduct="[M+H]+",
                peaks=[(30.0, 100.0), (40.0, 50.0)],
            )
            + _mgf_block(
                "phenol",
                pepmass=113.0,
                smiles="c1ccccc1O",
                adduct="[M+H]+",
                peaks=[(50.0, 100.0), (60.0, 50.0)],
            )
            + _mgf_block(
                "propane",
                pepmass=114.0,
                smiles="CCC",
                adduct="[M+H]+",
                peaks=[(70.0, 100.0), (80.0, 50.0)],
            )
        )
    )
    mcebio_mgf = tmp_path / "mcebio.mgf"
    mcebio_mgf.write_text(
        "\n".join(
            _mgf_block(
                "mcebio-fluoro",
                pepmass=211.0,
                smiles="CC(F)N",
                adduct="[M+H]+",
                peaks=[(11.0, 100.0), (21.0, 50.0)],
            )
            + _mgf_block(
                "mcebio-plain",
                pepmass=212.0,
                smiles="CCN",
                adduct="[M+H]+",
                peaks=[(31.0, 100.0), (41.0, 50.0)],
            )
        )
    )

    metadata = murcko.prepare_murcko_mgf_collection(
        nist_mgf=str(nist_mgf),
        mcebio_mgf=str(mcebio_mgf),
        work_dir=tmp_path / "work",
        upload=False,
        val_frac=0.25,
        test_frac=0.25,
        num_workers=1,
        batch_size=2,
        parquet_batch_size=2,
        nist_split_size_caps={"train": 10, "val": 10, "test": 10},
    )
    artifact_dir = Path(metadata["artifact_dir"])

    assert metadata["datasets"]["nist"]["subdir"] == murcko.NIST_MURCKO_PREPARED_SUBDIR
    assert metadata["datasets"]["mcebio"]["subdir"] == murcko.MCEBIO_MURCKO_PREPARED_SUBDIR
    assert metadata["datasets"]["mcebio"]["splits"] == ["all"]
    assert (artifact_dir / murcko.NIST_MURCKO_PREPARED_SUBDIR / "metadata.json").exists()
    assert (
        artifact_dir / murcko.MCEBIO_MURCKO_PREPARED_SUBDIR / "metadata.json"
    ).exists()
    assert (artifact_dir / "raw" / nist_mgf.name).exists()
    assert (artifact_dir / "raw" / mcebio_mgf.name).exists()


def test_prepare_murcko_mgf_collection_builds_dreams_before_upload(
    monkeypatch,
    tmp_path: Path,
):
    from spectra_learning.data import murcko_dreams

    nist_mgf = tmp_path / "nist.mgf"
    nist_mgf.write_text(
        "\n".join(
            _mgf_block(
                "ethanol",
                pepmass=111.0,
                smiles="CCO",
                adduct="[M+H]+",
                peaks=[(10.0, 100.0), (20.0, 50.0)],
            )
            + _mgf_block(
                "fluoro",
                pepmass=112.0,
                smiles="CC(F)O",
                adduct="[M+H]+",
                peaks=[(30.0, 100.0), (40.0, 50.0)],
            )
            + _mgf_block(
                "phenol",
                pepmass=113.0,
                smiles="c1ccccc1O",
                adduct="[M+H]+",
                peaks=[(50.0, 100.0), (60.0, 50.0)],
            )
            + _mgf_block(
                "propane",
                pepmass=114.0,
                smiles="CCC",
                adduct="[M+H]+",
                peaks=[(70.0, 100.0), (80.0, 50.0)],
            )
        )
    )
    mcebio_mgf = tmp_path / "mcebio.mgf"
    mcebio_mgf.write_text(
        "\n".join(
            _mgf_block(
                "mcebio-fluoro",
                pepmass=211.0,
                smiles="CC(F)N",
                adduct="[M+H]+",
                peaks=[(11.0, 100.0), (21.0, 50.0)],
            )
            + _mgf_block(
                "mcebio-plain",
                pepmass=212.0,
                smiles="CCN",
                adduct="[M+H]+",
                peaks=[(31.0, 100.0), (41.0, 50.0)],
            )
        )
    )
    events = []

    def fake_build_dreams_auxiliary(**kwargs):
        marker = (
            Path(kwargs["artifact_dir"])
            / murcko.NIST_MURCKO_PREPARED_SUBDIR
            / "auxiliary"
            / "dreams"
            / "train-part-00000.npz"
        )
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_bytes(b"fake dreams")
        events.append(("dreams", kwargs))
        return [marker]

    class FakeHfApi:
        def create_repo(self, *args, **kwargs):
            events.append(("create_repo", args, kwargs))

        def upload_large_folder(self, **kwargs):
            metadata = json.loads((Path(kwargs["folder_path"]) / "metadata.json").read_text())
            assert metadata["dreams_auxiliary_built"]
            assert metadata["dreams_auxiliary_paths"] == [
                "nist_murcko_probe/auxiliary/dreams/train-part-00000.npz"
            ]
            assert (
                Path(kwargs["folder_path"])
                / "nist_murcko_probe"
                / "auxiliary"
                / "dreams"
                / "train-part-00000.npz"
            ).exists()
            events.append(("upload", kwargs))

    monkeypatch.setattr(
        murcko_dreams,
        "build_murcko_dreams_auxiliary",
        fake_build_dreams_auxiliary,
    )
    monkeypatch.setattr(murcko, "HfApi", FakeHfApi)

    metadata = murcko.prepare_murcko_mgf_collection(
        nist_mgf=str(nist_mgf),
        mcebio_mgf=str(mcebio_mgf),
        work_dir=tmp_path / "work",
        upload=True,
        hf_repo_id="unit/repo",
        hf_revision="unit-test",
        build_dreams_auxiliary=True,
        dreams_root=tmp_path / "Dreams",
        dreams_checkpoint=tmp_path / "Dreams" / "embedding_model.ckpt",
        dreams_subdirs=[murcko.NIST_MURCKO_PREPARED_SUBDIR],
        dreams_device="cpu",
        val_frac=0.25,
        test_frac=0.25,
        num_workers=1,
        batch_size=2,
        parquet_batch_size=2,
        nist_split_size_caps={"train": 10, "val": 10, "test": 10},
    )

    assert [event[0] for event in events] == ["dreams", "create_repo", "upload"]
    assert metadata["dreams_auxiliary_built"]


def test_prepare_nist_disjoint_probe_retrieval_collection_writes_fixed_tasks(
    monkeypatch,
    tmp_path: Path,
):
    mgf_path = tmp_path / "nist.mgf"
    mgf_path.write_text(
        "\n".join(
            _mgf_block(
                "probe-phenol-a",
                pepmass=150.0,
                smiles="c1ccccc1O",
                adduct="[M+H]+",
                peaks=[(50.0, 100.0), (60.0, 50.0)],
            )
            + _mgf_block(
                "probe-phenol-b",
                pepmass=150.0005,
                smiles="Oc1ccccc1",
                adduct="[M+H]+",
                peaks=[(51.0, 100.0), (61.0, 50.0)],
            )
            + _mgf_block(
                "retrieval-ethanol-a",
                pepmass=100.0000,
                smiles="CCO",
                adduct="[M+H]+",
                peaks=[(10.0, 100.0), (20.0, 50.0)],
            )
            + _mgf_block(
                "retrieval-ethanol-b",
                pepmass=100.0005,
                smiles="OCC",
                adduct="[M+H]+",
                peaks=[(11.0, 100.0), (21.0, 50.0)],
            )
            + _mgf_block(
                "retrieval-ethylamine",
                pepmass=100.0008,
                smiles="CCN",
                adduct="[M+H]+",
                peaks=[(12.0, 100.0), (22.0, 50.0)],
            )
        )
    )

    def fake_mces_values(retrieval_rows, pairs, *, workers):
        return (
            np.asarray([2.0 for _ in range(len(pairs))], dtype=np.float32),
            np.asarray([0.01 for _ in range(len(pairs))], dtype=np.float32),
            np.zeros(len(pairs), dtype=np.int16),
        )

    monkeypatch.setattr(murcko, "_compute_mces_values", fake_mces_values)

    metadata = murcko.prepare_nist_disjoint_probe_retrieval_collection(
        nist_mgf=str(mgf_path),
        work_dir=tmp_path / "work",
        upload=False,
        online_probe_size=2,
        val_frac=0.2,
        test_frac=0.2,
        num_workers=1,
        batch_size=2,
        parquet_batch_size=2,
        online_split_size_caps=None,
        same_inchi_pairs_per_class=1,
        mces_pairs=1,
        mces_tanimoto_bin_size=1.0,
        mces_workers=1,
    )

    artifact_dir = Path(metadata["artifact_dir"])
    assert metadata["artifact_format"] == murcko.NIST_DISJOINT_PROBE_RETRIEVAL_ARTIFACT_FORMAT
    assert metadata["online_probe"]["selected_spectra_before_split_processing"] == 2
    assert metadata["retrieval_pool"]["all_size"] == 3
    assert metadata["disjointness"]["online_probe_retrieval_murcko_hist_overlap"] == 0

    same_pairs = pq.read_table(
        artifact_dir / murcko.NIST_10PPM_RETRIEVAL_SUBDIR / "pairs.parquet"
    ).to_pydict()
    assert same_pairs["label"] == [1, 0]
    assert same_pairs["left_row"][0] != same_pairs["right_row"][0]
    assert same_pairs["left_inchi14"][0] == same_pairs["right_inchi14"][0]
    assert same_pairs["left_inchi14"][1] != same_pairs["right_inchi14"][1]

    mces_pairs = pq.read_table(
        artifact_dir / murcko.NIST_MCES_RETRIEVAL_SUBDIR / "pairs.parquet"
    ).to_pydict()
    assert mces_pairs["mces"] == [2.0]
    assert mces_pairs["mces_le_2"] == [1]
    assert mces_pairs["mces_le_1"] == [0]


def test_murcko_fluorine_loader_can_return_jax_batches(tmp_path: Path):
    root = tmp_path / "cache"
    nist = root / "nist_murcko_probe"
    mcebio = root / "mcebio_murcko_probe"
    _write_split(nist / "train.parquet", [True, False])
    _write_split(nist / "val.parquet", [False])
    _write_split(mcebio / "all.parquet", [False, True])
    _write_dreams_auxiliary(nist, "train", 2, 100.0)
    _write_dreams_auxiliary(mcebio, "all", 2, 300.0)
    (nist / "metadata.json").write_text(
        """
        {
          "train_files": ["train.parquet"],
          "train_lengths": [2],
          "train_size": 2,
          "train_positive": 1,
          "val_files": ["val.parquet"],
          "val_lengths": [1],
          "val_size": 1,
          "val_positive": 0,
          "dreams_dim": 2,
          "dreams_auxiliary_available": true,
          "dreams_auxiliary_files": {
            "train": ["auxiliary/dreams/train-part-00000.npz"]
          },
          "dreams_auxiliary_lengths": {
            "train": [2]
          }
        }
        """
    )
    (mcebio / "metadata.json").write_text(
        """
        {
          "all_files": ["all.parquet"],
          "all_lengths": [2],
          "all_size": 2,
          "all_positive": 1,
          "dreams_dim": 2,
          "dreams_auxiliary_available": true,
          "dreams_auxiliary_files": {
            "all": ["auxiliary/dreams/all-part-00000.npz"]
          },
          "dreams_auxiliary_lengths": {
            "all": [2]
          }
        }
        """
    )
    metadata = murcko.ensure_murcko_fluorine_data_downloaded(
        root,
        repo_id="unit/repo",
        train_subdir="nist_murcko_probe",
        test_subdir="mcebio_murcko_probe",
        include_dreams=True,
    )
    data = _fluorine_data(metadata, root)

    batch = next(
        iter(
            murcko.build_murcko_fluorine_loader(
                data,
                "train",
                shuffle=False,
                seed=0,
                max_samples=None,
                output_format="jax",
            )
        )
    )
    dreams_batch = next(
        iter(
            murcko.build_murcko_fluorine_loader(
                data,
                "test",
                shuffle=False,
                seed=0,
                max_samples=None,
                dreams_only=True,
                output_format="jax",
            )
        )
    )
    import jax

    assert isinstance(batch["peak_mz"], jax.Array)
    assert isinstance(batch["label"], jax.Array)
    assert tuple(batch["peak_mz"].shape) == (2, 2)
    assert isinstance(dreams_batch["dreams_embedding"], jax.Array)
    assert tuple(dreams_batch["dreams_embedding"].shape) == (2, 2)
