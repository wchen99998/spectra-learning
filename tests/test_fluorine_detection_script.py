from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from spectra_learning.probes.massspec import fluorine
from spectra_learning.models.lora import LoRALinear, lora_state_dict
from spectra_learning.training.storage import storage_exists, read_text


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
        peak_filtering="top_intensity",
        grouped_peak_shoulder_da=0.05,
        grouped_peak_isotope_charges=(1, 2, 3),
        peak_ordering="mz",
        precursor_peak_exclusion_window_da=0.0,
    )
    head_state = {
        "mode": "lora",
        "best_epoch": 1,
        "best_val": {"val/average_precision": 0.5, "val/roc_auc": 0.75},
        "test": None,
        "hparams": {"hidden_dim": 8, "dropout": 0.1},
        "pooling": "covariance",
        "pair_dim": 4,
        "device_ids": [],
        "focal_alpha": 0.5,
        "focal_gamma": 2.0,
        "finetune_cache_dir": str(tmp_path),
        "train_size": 8,
        "train_positive": 4,
        "val_size": 4,
        "val_positive": 2,
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


def test_lora_cached_state_requires_pooling_field(tmp_path: Path):
    config_path = tmp_path / "config.py"
    checkpoint_path = tmp_path / "checkpoint.pt"
    state_path = tmp_path / "lora_state.pt"
    torch.save(
        {
            "mode": "lora",
            "config_path": str(config_path),
            "checkpoint_path": str(checkpoint_path),
            "hparams": {},
            "max_train_samples": None,
            "max_val_samples": None,
        },
        state_path,
    )

    with pytest.raises(KeyError, match="pooling"):
        fluorine.train_or_load_lora(
            state_path=state_path,
            model=_FakeFluorineModel(),
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
