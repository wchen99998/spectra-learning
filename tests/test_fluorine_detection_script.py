from __future__ import annotations

from pathlib import Path
import shutil

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from scripts import train_fluorine_detection as fluorine
from spectra_learning.probes.massspec import data as massspec_data


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


def test_fluorine_cache_and_collator_use_project_peak_preprocessing(monkeypatch, tmp_path: Path):
    source_root = tmp_path / "source_repo"
    nist_source = source_root / "nist_murcko_probe"
    mcebio_source = source_root / "mcebio_murcko_probe"
    _write_split(nist_source / "train.parquet", [True, False, False, True])
    _write_split(nist_source / "val.parquet", [True, False])
    _write_split(mcebio_source / "test.parquet", [False, True])
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
          "dreams_dim": 2
        }
        """
    )
    (mcebio_source / "metadata.json").write_text(
        """
        {
          "test_files": ["test.parquet"],
          "test_lengths": [2],
          "test_size": 2,
          "test_positive": 1,
          "dreams_dim": 2
        }
        """
    )

    def fake_snapshot_download(**kwargs):
        local_dir = Path(kwargs["local_dir"])
        shutil.copytree(source_root, local_dir, dirs_exist_ok=True)
        return str(local_dir)

    monkeypatch.setattr(massspec_data, "snapshot_download", fake_snapshot_download)
    cache_dir = tmp_path / "cache"
    metadata = massspec_data.ensure_murcko_fluorine_data_downloaded(
        cache_dir,
        repo_id="unit/repo",
        revision="main",
        train_subdir="nist_murcko_probe",
        test_subdir="mcebio_murcko_probe",
    )

    assert metadata["train_size"] == 4
    assert metadata["train_positive"] == 2
    assert metadata["dreams_dim"] == 2
    assert (cache_dir / metadata["train_files"][0]).exists()

    data = fluorine.FluorineData(
        metadata=metadata,
        root=cache_dir,
        batch_size=2,
        num_peaks=2,
        max_precursor_mz=1000.0,
        min_peak_intensity=1e-4,
        peak_drop_min_intensity=1e-4,
        peak_ordering="mz",
        precursor_peak_exclusion_window_da=0.0,
    )
    batch = next(
        iter(
            fluorine._make_loader(
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
    assert torch.allclose(
        batch["dreams_embedding"],
        torch.tensor([[0.0, 1.0], [1.0, 2.0]]),
    )

    rank_indices = []
    for rank in range(2):
        loader = massspec_data.build_murcko_fluorine_loader(
            data,
            "train",
            shuffle=False,
            seed=0,
            max_samples=None,
            distributed_world_size=2,
            distributed_rank=rank,
        )
        indices = []
        for rank_batch in loader:
            indices.extend(int(value) for value in rank_batch["row_idx"].tolist())
        rank_indices.append(indices)
    assert sorted(rank_indices[0] + rank_indices[1]) == [0, 1, 2, 3]
    assert not (set(rank_indices[0]) & set(rank_indices[1]))

    rank1_cache = tmp_path / "rank1_cache"
    download_calls = []

    def rank1_snapshot_download(**kwargs):
        download_calls.append(kwargs)
        return str(kwargs["local_dir"])

    def rank1_barrier():
        shutil.copytree(source_root, rank1_cache, dirs_exist_ok=True)

    monkeypatch.setattr(massspec_data, "snapshot_download", rank1_snapshot_download)
    monkeypatch.setattr(massspec_data.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(massspec_data.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(massspec_data.torch.distributed, "barrier", rank1_barrier)

    rank1_metadata = massspec_data.ensure_murcko_fluorine_data_downloaded(
        rank1_cache,
        repo_id="unit/repo",
        revision="main",
        train_subdir="nist_murcko_probe",
        test_subdir="mcebio_murcko_probe",
        distributed_world_size=2,
        distributed_rank=1,
    )
    assert download_calls == []
    assert rank1_metadata["test_size"] == 2


def test_focal_loss_keeps_positive_alpha_weighting():
    logits = torch.tensor([0.0, 0.0])
    targets = torch.tensor([1.0, 0.0])
    loss = fluorine.binary_focal_loss_with_logits(
        logits,
        targets,
        alpha=0.75,
        gamma=2.0,
    )
    expected = 0.5 * (0.75 + 0.25) * (0.5**2) * torch.log(torch.tensor(2.0))
    assert torch.allclose(loss, expected)
