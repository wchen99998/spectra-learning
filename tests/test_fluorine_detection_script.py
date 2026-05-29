from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from spectra_learning.probes.massspec import data as massspec_data
from spectra_learning.probes.massspec import fluorine
from spectra_learning.models.lora import LoRALinear, lora_state_dict
from spectra_learning.training.storage import storage_exists, read_text


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
    assert metadata["test_size"] == 2
    assert metadata["test_positive"] == 1
    assert metadata["dreams_dim"] == 2
    assert (cache_dir / metadata["train_files"][0]).exists()
    assert (cache_dir / metadata["test_files"][0]).exists()

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


def test_fluorine_outputs_support_fsspec_prefix(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("SPECTRA_SCRATCH_DIR", str(tmp_path / "scratch"))
    output_prefix = "memory://fluorine-unit/run"
    data = fluorine.FluorineData(
        metadata={"test_size": 4, "test_positive": 2},
        root=tmp_path,
        batch_size=2,
        num_peaks=2,
        max_precursor_mz=1000.0,
        min_peak_intensity=1e-4,
        peak_drop_min_intensity=1e-4,
        peak_ordering="mz",
        precursor_peak_exclusion_window_da=0.0,
    )
    head_state = {
        "mode": "lora",
        "best_epoch": 1,
        "best_val": {"val/average_precision": 0.5, "val/roc_auc": 0.75},
        "hparams": {"hidden_dim": 8, "dropout": 0.1},
        "focal_alpha": 0.5,
        "focal_gamma": 2.0,
        "history": [
            {
                "epoch": 1,
                "train_loss": 0.1,
                "val": {"val/average_precision": 0.5, "val/roc_auc": 0.75},
            }
        ],
    }

    summary = fluorine.write_standard_fluorine_outputs(
        output_prefix=output_prefix,
        config_path=tmp_path / "config.py",
        checkpoint_path=tmp_path / "checkpoint.pt",
        head_state_path=tmp_path / "state.pt",
        data=data,
        targets=np.asarray([0, 1, 0, 1], dtype=np.float32),
        logits=np.asarray([-2.0, 1.5, -0.5, 2.0], dtype=np.float32),
        row_indices=np.arange(4),
        head_state=head_state,
    )
    all_curves = fluorine.write_all_pr_curve_comparison(
        output_prefix=output_prefix,
        curve_dirs=["memory://fluorine-unit"],
    )

    assert summary["metrics"]["average_precision"] > 0.0
    assert storage_exists("memory://fluorine-unit/run.summary.json")
    assert storage_exists("memory://fluorine-unit/run.pr_curve.csv")
    assert storage_exists("memory://fluorine-unit/run.pr_curve.png")
    assert storage_exists("memory://fluorine-unit/run.all_pr_curves.png")
    assert len(all_curves["curves"]) == 1
    saved = json.loads(read_text("memory://fluorine-unit/run.all_pr_curves.summary.json"))
    assert saved["curves"][0]["name"] == "run"


def test_autocast_dtype_resolves_from_config_and_cli():
    assert fluorine._resolve_autocast_dtype({}, None) == torch.bfloat16
    assert fluorine._resolve_autocast_dtype({"autocast_dtype": "fp16"}, None) == torch.float16
    assert fluorine._resolve_autocast_dtype({}, "none") is None


class _FakeFluorineBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.single_attention = torch.nn.Module()
        self.single_attention.wqkv = torch.nn.Linear(4, 12, bias=False)
        self.single_attention.wo = torch.nn.Linear(4, 4, bias=False)
        self.single_transition = torch.nn.Module()
        self.single_transition.w1 = torch.nn.Linear(4, 8, bias=False)
        self.single_transition.w2 = torch.nn.Linear(8, 4, bias=False)
        self.pair_transition = torch.nn.Module()
        self.pair_transition.w1 = torch.nn.Linear(4, 8, bias=False)
        self.pair_transition.w2 = torch.nn.Linear(8, 4, bias=False)
        self.embed = torch.nn.Linear(4, 4, bias=False)


class _FakeFluorineEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([_FakeFluorineBlock()])


class _FakeFluorineModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _FakeFluorineEncoder()


def test_cli_accepts_lora_mode(monkeypatch):
    from scripts import train_fluorine_detection

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_fluorine_detection.py",
            "--mode",
            "lora",
            "--autocast-dtype",
            "bf16",
        ],
    )

    args = train_fluorine_detection.parse_args()

    assert args.mode == "lora"
    assert args.epochs == 3
    assert args.patience == 2
    assert args.head_state == fluorine.default_state_path("lora")
    assert args.lora_rank == 8
    assert args.lora_alpha == 16.0
    assert args.lora_dropout == 0.0
    assert args.lora_learning_rate == 1e-4
    assert args.autocast_dtype == "bf16"


def test_lora_cached_state_injects_adapters_without_full_model_state(tmp_path: Path):
    source = _FakeFluorineModel()
    lora_config = fluorine._lora_config(rank=2, alpha=4.0, dropout=0.0)
    fluorine.apply_fluorine_lora(source.encoder, lora_config)
    lora_state = lora_state_dict(source.encoder)
    hparams = {
        "hidden_dim": 8,
        "dropout": 0.1,
        "lora_rank": 2,
        "lora_alpha": 4.0,
        "lora_dropout": 0.0,
        "lora_learning_rate": 1e-4,
        "head_learning_rate": 1e-4,
        "weight_decay": 0.01,
        "autocast_dtype": "bf16",
        "epochs": 3,
        "patience": 2,
    }
    config_path = tmp_path / "config.py"
    checkpoint_path = tmp_path / "checkpoint.pt"
    state_path = tmp_path / "lora_state.pt"
    torch.save(
        {
            "mode": "lora",
            "config_path": str(config_path),
            "checkpoint_path": str(checkpoint_path),
            "pooling": "covariance",
            "hparams": hparams,
            "lora_config": lora_config,
            "lora_state": lora_state,
            "max_train_samples": None,
            "max_val_samples": None,
        },
        state_path,
    )
    model = _FakeFluorineModel()

    loaded = fluorine.train_or_load_lora(
        state_path=state_path,
        model=model,
        config={},
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        cache_dir=tmp_path,
        device=torch.device("cpu"),
        batch_size=1,
        num_workers=0,
        seed=0,
        epochs=3,
        patience=2,
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
        lora_learning_rate=1e-4,
        head_learning_rate=1e-4,
        weight_decay=0.01,
        autocast_dtype=torch.bfloat16,
        hidden_dim=8,
        dropout=0.1,
        revision="main",
        max_train_samples=None,
        max_val_samples=None,
        max_test_samples=None,
        pooling="covariance",
        device_ids=None,
    )

    block = model.encoder.blocks[0]
    assert loaded["mode"] == "lora"
    assert "model_state" not in loaded
    assert set(loaded["lora_state"]) == set(lora_state)
    assert isinstance(block.single_attention.wqkv, LoRALinear)
    assert isinstance(block.single_attention.wo, LoRALinear)
    assert isinstance(block.single_transition.w1, LoRALinear)
    assert isinstance(block.single_transition.w2, LoRALinear)
    assert isinstance(block.pair_transition.w1, LoRALinear)
    assert isinstance(block.pair_transition.w2, LoRALinear)
    assert isinstance(block.embed, torch.nn.Linear)
