from __future__ import annotations

from pathlib import Path
import shutil

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from scripts import train_fluorine_detection as fluorine


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
    source = tmp_path / "source_repo" / "fine_tuned"
    _write_split(source / "train.parquet", [True, False, False, True])
    _write_split(source / "val.parquet", [True, False])
    _write_split(source / "test.parquet", [False, True])
    (source / "metadata.json").write_text("{}")

    def fake_snapshot_download(**kwargs):
        local_dir = Path(kwargs["local_dir"])
        shutil.copytree(source.parent, local_dir, dirs_exist_ok=True)
        return str(local_dir)

    monkeypatch.setattr(fluorine, "snapshot_download", fake_snapshot_download)
    cache_dir = tmp_path / "cache"
    metadata = fluorine.ensure_fluorine_cache(
        cache_dir,
        repo_id="unit/repo",
        revision="main",
        subdir="fine_tuned",
        num_shards=2,
        parquet_batch_size=2,
    )

    assert metadata["train_size"] == 4
    assert metadata["train_positive"] == 2
    assert metadata["dreams_dim"] == 2
    assert (cache_dir / metadata["train_files"][0] / "spectra.npy").exists()

    data = fluorine.FluorineData(
        metadata=metadata,
        root=cache_dir,
        batch_size=2,
        num_peaks=2,
        max_precursor_mz=1000.0,
        min_peak_intensity=1e-4,
        peak_drop_min_intensity=1e-4,
        peak_ordering="mz",
        use_precursor_token=False,
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
