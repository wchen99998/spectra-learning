from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

from ml_collections import config_dict
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
        metadata={
            "nist_repo_id": "owner/nist",
            "nist_revision": "nist-sha",
            "nist_subdir": "nist",
            "nist_source_dir": str(tmp_path / "nist"),
            "peak_preprocessing": {"version": 1},
            "test_size": 4,
            "test_positive": 2,
        },
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
        "mode": "probe",
        "best_epoch": 1,
        "best_val": {"val/average_precision": 0.5, "val/roc_auc": 0.75},
        "test": None,
        "hparams": {"hidden_dim": 8, "dropout": 0.1},
        "pooling": "covariance",
        "pair_dim": 4,
        "focal_alpha": 0.5,
        "focal_gamma": 2.0,
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
    assert "metrics_prefixed" not in summary
    assert "device_ids" not in summary["head"]
    assert "finetune_cache_dir" not in summary["head"]
    assert storage_exists("memory://fluorine-unit/run.summary.json")
    assert storage_exists("memory://fluorine-unit/run.pr_curve.csv")
    assert storage_exists("memory://fluorine-unit/run.pr_curve.png")
    assert storage_exists("memory://fluorine-unit/run.all_pr_curves.png")
    assert len(all_curves["curves"]) == 1
    saved = json.loads(read_text("memory://fluorine-unit/run.all_pr_curves.summary.json"))
    assert saved["curves"][0]["name"] == "run"


def test_checkpoint_probe_data_uses_current_sources_and_checkpoint_preprocessing(
    monkeypatch,
    tmp_path: Path,
):
    cache_dir = tmp_path / "cache"
    captured = {}

    def fake_from_config(config, **_kwargs):
        captured["config"] = config
        info = {
            "massspec_metadata_version": 1,
            "massspec_nist_repo_id": config.nist_murcko_probe_repo_id,
            "massspec_nist_revision": config.nist_murcko_probe_revision,
            "massspec_nist_subdir": config.nist_murcko_probe_hf_subdir,
            "massspec_nist_source_dir": str(cache_dir / "nist-source"),
            "massspec_train_size": 2,
            "massspec_train_positive": 1,
            "massspec_val_size": 1,
            "massspec_val_positive": 0,
            "massspec_test_size": 1,
            "massspec_test_positive": 1,
            "massspec_peak_preprocessing": {"version": 1},
        }
        return SimpleNamespace(
            info=info,
            train_files=[str(cache_dir / "nist-source/train.parquet")],
            train_lengths=[2],
            val_files=[str(cache_dir / "nist-source/val.parquet")],
            val_lengths=[1],
            test_files=[str(cache_dir / "nist-source/test.parquet")],
            test_lengths=[1],
            num_peaks=config.num_peaks,
            max_precursor_mz=config.max_precursor_mz,
            min_peak_intensity=config.min_peak_intensity,
            peak_drop_min_intensity=config.peak_drop_min_intensity,
            peak_filtering=config.peak_filtering,
            grouped_peak_shoulder_da=config.grouped_peak_shoulder_da,
            grouped_peak_isotope_charges=tuple(
                config.grouped_peak_isotope_charges
            ),
            peak_ordering=config.peak_ordering,
            precursor_peak_exclusion_window_da=(
                config.precursor_peak_exclusion_window_da
            ),
        )

    monkeypatch.setattr(
        fluorine,
        "MassSpecProbeData",
        SimpleNamespace(from_config=fake_from_config),
    )
    checkpoint_config = config_dict.ConfigDict(
        {
            "nist_murcko_probe_repo_id": "owner/nist",
            "nist_murcko_probe_revision": "nist-sha",
            "nist_murcko_probe_hf_subdir": "nist",
            "num_peaks": 17,
            "max_precursor_mz": 900.0,
            "min_peak_intensity": 0.01,
            "peak_drop_min_intensity": 0.02,
            "peak_filtering": "top_intensity",
            "grouped_peak_shoulder_da": 0.03,
            "grouped_peak_isotope_charges": (1, 2),
            "peak_ordering": "mz",
            "precursor_peak_exclusion_window_da": 1.5,
        }
    )
    args = SimpleNamespace(
        batch_size=8,
    )

    data = fluorine._build_fluorine_probe_data(
        args=args,
        checkpoint_config=checkpoint_config,
        cache_dir=cache_dir,
    )

    assert captured["config"].artifact_dir == str(cache_dir.resolve())
    assert data.num_peaks == 17
    assert data.peak_ordering == "mz"
    assert data.metadata["nist_repo_id"] == fluorine.NIST_MURCKO_HF_REPO
    assert data.metadata["nist_revision"] == fluorine.NIST_MURCKO_HF_REVISION
    assert data.metadata["nist_subdir"] == fluorine.NIST_MURCKO_PREPARED_SUBDIR


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
        self.single_transition.fc1 = torch.nn.Linear(4, 8, bias=False)
        self.single_transition.fc2 = torch.nn.Linear(4, 8, bias=False)
        self.single_transition.fc3 = torch.nn.Linear(8, 4, bias=False)
        self.pair_transition = torch.nn.Module()
        self.pair_transition.fc1 = torch.nn.Linear(4, 8, bias=False)
        self.pair_transition.fc2 = torch.nn.Linear(4, 8, bias=False)
        self.pair_transition.fc3 = torch.nn.Linear(8, 4, bias=False)
        self.embed = torch.nn.Linear(4, 4, bias=False)


class _FakeFluorineEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([_FakeFluorineBlock()])


class _FakeFluorineModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _FakeFluorineEncoder()


class _TinyFluorineEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.single_attention = torch.nn.Module()
        self.single_attention.wo = torch.nn.Linear(2, 2, bias=False)

    def forward(
        self,
        peak_mz: torch.Tensor,
        peak_intensity: torch.Tensor,
        *,
        valid_mask: torch.Tensor,
        precursor_mz: torch.Tensor | None = None,
        spectrum_metadata: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.single_attention.wo(
            torch.stack((peak_mz, peak_intensity), dim=-1)
        )


class _TinyFluorineModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _TinyFluorineEncoder()


def _tiny_fluorine_batch() -> dict[str, torch.Tensor]:
    return {
        "peak_mz": torch.tensor([[0.1, 0.2], [0.8, 0.9]]),
        "peak_intensity": torch.tensor([[0.2, 0.3], [0.7, 0.8]]),
        "peak_valid_mask": torch.ones(2, 2, dtype=torch.bool),
        "label": torch.tensor([0.0, 1.0]),
    }


def _source_checkpoint_contract() -> dict[str, object]:
    return {
        "backend": "torch",
        "global_step": 7,
        "model_sha256": "model-sha",
        "covariance_pooler_sha256": None,
        "model_settings": {"model_dim": 2},
        "peak_preprocessing": {"version": 1},
        "data_provenance": {"gems_revision": "gems-sha"},
    }


def _adaptation_data_metadata() -> dict[str, object]:
    return {
        "train_size": 2,
        "train_positive": 1,
        "val_size": 2,
        "val_positive": 1,
        "data_provenance": {
            "massspec_nist_revision": "nist-sha",
        },
        "peak_preprocessing": {"version": 1},
    }


def test_torch_source_checkpoint_contract_tracks_model_content(tmp_path: Path):
    checkpoint = {
        "global_step": 7,
        "model": {"weight": torch.tensor([1.0, 2.0])},
        "covariance_pooler_checkpoint": None,
        "peak_preprocessing": {"version": 1},
        "data_provenance": {"gems_revision": "gems-sha"},
    }
    config = config_dict.ConfigDict({"model_dim": 2})

    baseline = fluorine._torch_source_checkpoint_contract(
        config=config,
        checkpoint_path=tmp_path / "checkpoint.pt",
        checkpoint=checkpoint,
    )
    checkpoint["model"]["weight"][0] = 3.0
    changed = fluorine._torch_source_checkpoint_contract(
        config=config,
        checkpoint_path=tmp_path / "checkpoint.pt",
        checkpoint=checkpoint,
    )

    assert baseline["model_sha256"] != changed["model_sha256"]


@pytest.mark.parametrize("changed_part", ["source_checkpoint", "evaluation_data"])
def test_cached_adaptation_state_rejects_stale_contract(
    tmp_path: Path,
    changed_part: str,
):
    state_path = tmp_path / "state.pt"
    current_contract = fluorine._fluorine_state_contract(
        source_checkpoint=_source_checkpoint_contract(),
        data=SimpleNamespace(metadata=_adaptation_data_metadata()),
    )
    torch.save(
        {
            "complete": True,
            "mode": "lora",
            "pooling": "covariance",
            "hparams": {},
            "max_train_samples": None,
            "max_val_samples": None,
            "state_contract": current_contract,
        },
        state_path,
    )
    changed_contract = json.loads(json.dumps(current_contract))
    if changed_part == "source_checkpoint":
        changed_contract["source_checkpoint"]["model_sha256"] = "changed"
    else:
        changed_contract["evaluation_data_provenance"][
            "massspec_nist_revision"
        ] = "changed"

    with pytest.raises(ValueError, match="training contract"):
        fluorine._load_cached_adaptation_state(
            state_path=state_path,
            device=torch.device("cpu"),
            mode="lora",
            pooling="covariance",
            requested_hparams={},
            max_train_samples=None,
            max_val_samples=None,
            state_contract=changed_contract,
        )


def _adaptation_trainer_kwargs(
    *,
    trainer_name: str,
    tmp_path: Path,
    model: torch.nn.Module,
    config: config_dict.ConfigDict,
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "state_path": tmp_path / f"{trainer_name}.pt",
        "model": model,
        "config": config,
        "config_path": tmp_path / "config.py",
        "checkpoint_path": tmp_path / "checkpoint.pt",
        "source_checkpoint_contract": _source_checkpoint_contract(),
        "cache_dir": tmp_path,
        "device": torch.device("cpu"),
        "batch_size": 2,
        "num_workers": 0,
        "seed": 7,
        "epochs": 1,
        "patience": 1,
        "weight_decay": 0.0,
        "autocast_dtype": None,
        "hidden_dim": 2,
        "dropout": 0.0,
        "max_train_samples": None,
        "max_val_samples": None,
        "max_test_samples": None,
        "pooling": "covariance",
        "eval_test_every_epoch": True,
    }
    if trainer_name == "train_or_load_finetuned":
        kwargs.update(
            model_learning_rate=1e-2,
            pooler_learning_rate=1e-2,
            head_learning_rate=1e-2,
            focal_alpha="auto",
            focal_gamma=2.0,
            select_metric="average_precision",
        )
    else:
        kwargs.update(
            lora_rank=1,
            lora_alpha=1.0,
            lora_dropout=0.0,
            lora_learning_rate=1e-2,
            head_learning_rate=1e-2,
        )
    return kwargs


def test_cli_accepts_lora_mode(monkeypatch):
    from scripts import train_fluorine_detection

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_fluorine_detection.py",
            "--mode",
            "lora",
            "--config",
            "configs/100m_pairmixer_dense_adamw.py",
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


class _StopAfterLoaderConstruction(Exception):
    pass


@pytest.mark.parametrize("trainer_name", ["train_or_load_finetuned", "train_or_load_lora"])
@pytest.mark.parametrize("eval_test_every_epoch", [False, True])
def test_adaptation_trainers_only_build_test_loader_for_per_epoch_evaluation(
    monkeypatch,
    tmp_path: Path,
    trainer_name: str,
    eval_test_every_epoch: bool,
):
    loader_splits = []

    monkeypatch.setattr(
        fluorine,
        "build_fluorine_data",
        lambda **_kwargs: SimpleNamespace(metadata=_adaptation_data_metadata()),
    )

    def fake_make_loader(_data, split, **_kwargs):
        loader_splits.append(split)
        return []

    def stop_after_loaders(**_kwargs):
        raise _StopAfterLoaderConstruction

    monkeypatch.setattr(fluorine, "_make_loader", fake_make_loader)
    monkeypatch.setattr(fluorine, "CovariancePool", stop_after_loaders)
    kwargs = {
        "state_path": tmp_path / f"{trainer_name}.pt",
        "model": object(),
        "config": config_dict.ConfigDict(
            {"model_dim": 4, "covariance_pooling_dim": 2}
        ),
        "config_path": tmp_path / "config.py",
        "checkpoint_path": tmp_path / "checkpoint.pt",
        "source_checkpoint_contract": _source_checkpoint_contract(),
        "cache_dir": tmp_path,
        "device": torch.device("cpu"),
        "batch_size": 2,
        "num_workers": 0,
        "seed": 0,
        "epochs": 1,
        "patience": 1,
        "weight_decay": 0.0,
        "autocast_dtype": None,
        "hidden_dim": 4,
        "dropout": 0.0,
        "max_train_samples": None,
        "max_val_samples": None,
        "max_test_samples": None,
        "pooling": "covariance",
        "eval_test_every_epoch": eval_test_every_epoch,
    }
    if trainer_name == "train_or_load_finetuned":
        kwargs.update(
            {
                "model_learning_rate": 1e-4,
                "pooler_learning_rate": 1e-4,
                "head_learning_rate": 1e-4,
                "focal_alpha": "auto",
                "focal_gamma": 2.0,
                "select_metric": "average_precision",
            }
        )
    else:
        kwargs.update(
            {
                "lora_rank": 2,
                "lora_alpha": 4.0,
                "lora_dropout": 0.0,
                "lora_learning_rate": 1e-4,
                "head_learning_rate": 1e-4,
            }
        )

    with pytest.raises(_StopAfterLoaderConstruction):
        getattr(fluorine, trainer_name)(**kwargs)

    assert loader_splits == (
        ["train", "val", "test"]
        if eval_test_every_epoch
        else ["train", "val"]
    )


@pytest.mark.parametrize(
    "trainer_name",
    ["train_or_load_finetuned", "train_or_load_lora"],
)
def test_adaptation_trainers_complete_one_epoch_and_reload_best_state(
    monkeypatch,
    tmp_path: Path,
    trainer_name: str,
):
    torch.manual_seed(0)
    model = _TinyFluorineModel()
    config = config_dict.ConfigDict(
        {"model_dim": 2, "covariance_pooling_dim": 1, "compile_mode": "none"}
    )
    data = SimpleNamespace(metadata=_adaptation_data_metadata())
    loader_calls = []

    def fake_make_loader(_data, split, **kwargs):
        loader_calls.append((split, kwargs))
        return [_tiny_fluorine_batch()]

    monkeypatch.setattr(fluorine, "build_fluorine_data", lambda **_kwargs: data)
    monkeypatch.setattr(fluorine, "_make_loader", fake_make_loader)
    if trainer_name == "train_or_load_finetuned":
        monkeypatch.setattr(fluorine, "load_torch_checkpoint", lambda *_args, **_kwargs: {})
        monkeypatch.setattr(
            fluorine,
            "load_resume_covariance_pooler_state",
            lambda *_args, **_kwargs: None,
        )

    kwargs = _adaptation_trainer_kwargs(
        trainer_name=trainer_name,
        tmp_path=tmp_path,
        model=model,
        config=config,
    )
    state = getattr(fluorine, trainer_name)(**kwargs)

    assert [(split, call["shuffle"], call["seed"]) for split, call in loader_calls] == [
        ("train", True, 7),
        ("val", False, 10_007),
        ("test", False, 20_007),
    ]
    state_path = kwargs["state_path"]
    assert isinstance(state_path, Path)
    saved_state = torch.load(state_path, weights_only=False)
    best_state = torch.load(
        state_path.with_name(f"{state_path.stem}.best.pt"),
        weights_only=False,
    )
    assert state["complete"] is True
    assert saved_state["complete"] is True
    assert best_state["complete"] is False
    assert saved_state.keys() == state.keys()
    assert best_state.keys() == state.keys()
    assert state["best_epoch"] == 1
    assert len(state["history"]) == 1
    assert state["best_val"] == state["history"][0]["val"]
    assert "test/average_precision" in state["history"][0]["test"]
    assert {
        key: tuple(value.shape) for key, value in state["pooler_state"].items()
    } == {"left_proj.weight": (1, 2), "right_proj.weight": (1, 2)}
    assert {
        key: tuple(value.shape) for key, value in state["classifier_state"].items()
    } == {
        "net.0.weight": (2, 1),
        "net.0.bias": (2,),
        "net.3.weight": (2, 2),
        "net.3.bias": (2,),
        "net.6.weight": (1, 2),
        "net.6.bias": (1,),
    }

    if trainer_name == "train_or_load_finetuned":
        assert "lora_state" not in state
        assert state["distributed_world_size"] == 1
        state_key = "model_state"
        expected_state = state[state_key]
        assert {key: tuple(value.shape) for key, value in expected_state.items()} == {
            "encoder.single_attention.wo.weight": (2, 2)
        }
        actual_state = model.state_dict()
    else:
        assert "model_state" not in state
        assert "distributed_world_size" not in state
        assert state["lora_config"]["applied_modules"] == ["single_attention.wo"]
        state_key = "lora_state"
        expected_state = state[state_key]
        assert {key: tuple(value.shape) for key, value in expected_state.items()} == {
            "single_attention.wo.lora_a.weight": (1, 2),
            "single_attention.wo.lora_b.weight": (2, 1),
        }
        actual_state = lora_state_dict(model.encoder)
    for key, value in expected_state.items():
        assert torch.equal(actual_state[key], value)
        assert torch.equal(saved_state[state_key][key], value)
        assert torch.equal(best_state[state_key][key], value)

    restored_model = _TinyFluorineModel()
    kwargs["model"] = restored_model
    monkeypatch.setattr(
        fluorine,
        "build_fluorine_data",
        lambda **_kwargs: data,
    )
    monkeypatch.setattr(
        fluorine,
        "_make_loader",
        lambda *_args, **_kwargs: pytest.fail("cached state rebuilt loaders"),
    )
    cached_state = getattr(fluorine, trainer_name)(**kwargs)

    assert cached_state["best_epoch"] == 1
    if trainer_name == "train_or_load_finetuned":
        restored_state = restored_model.state_dict()
    else:
        assert isinstance(restored_model.encoder.single_attention.wo, LoRALinear)
        restored_state = lora_state_dict(restored_model.encoder)
    for key, value in expected_state.items():
        assert torch.equal(restored_state[key], value)


def test_run_persists_single_canonical_final_test_evaluation(monkeypatch, tmp_path: Path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    head_state_path = tmp_path / "head.pt"
    output_prefix = tmp_path / "fluorine"
    config = config_dict.ConfigDict({"model_dim": 4, "autocast_dtype": "none"})
    model = torch.nn.Module()
    head_state = {
        "mode": "finetune",
        "complete": True,
        "test": None,
    }
    evaluation_calls = 0

    monkeypatch.setattr(
        fluorine,
        "init_distributed_from_env",
        lambda: SimpleNamespace(is_distributed=False, is_main=True),
    )
    monkeypatch.setattr(
        fluorine,
        "resolve_checkpoint_path",
        lambda _checkpoint, _workdir: checkpoint_path,
    )
    monkeypatch.setattr(
        fluorine,
        "_load_checkpoint_model",
        lambda _config_path, _checkpoint_path, _device: (
            config,
            model,
            _source_checkpoint_contract(),
        ),
    )
    monkeypatch.setattr(
        fluorine,
        "train_or_load_finetuned",
        lambda **_kwargs: head_state,
    )
    monkeypatch.setattr(
        fluorine,
        "build_fluorine_data",
        lambda **_kwargs: SimpleNamespace(
            metadata={"test_size": 2, "test_positive": 1}
        ),
    )

    def fake_evaluate(**_kwargs):
        nonlocal evaluation_calls
        evaluation_calls += 1
        return (
            np.asarray([0.0, 1.0]),
            np.asarray([-1.0, 1.0]),
            np.asarray([0, 1]),
        )

    def fake_write_outputs(**kwargs):
        assert kwargs["head_state"]["test"]["test/average_precision"] == 1.0
        persisted_state = torch.load(head_state_path, weights_only=False)
        assert persisted_state["test"]["test/average_precision"] == 1.0
        return {"mode": "finetune", "metrics": {"average_precision": 1.0}}

    monkeypatch.setattr(fluorine, "evaluate_fluorine_test_split", fake_evaluate)
    monkeypatch.setattr(fluorine, "write_standard_fluorine_outputs", fake_write_outputs)
    monkeypatch.setattr(
        fluorine,
        "write_all_pr_curve_comparison",
        lambda **_kwargs: {"curves": []},
    )
    args = SimpleNamespace(
        mode="finetune",
        seed=0,
        device="cpu",
        checkpoint=checkpoint_path,
        workdir=None,
        config=tmp_path / "config.py",
        head_state=head_state_path,
        output_prefix=output_prefix,
        autocast_dtype="none",
        epochs=1,
        patience=1,
        finetune_cache_dir=tmp_path,
        batch_size=2,
        num_workers=0,
        finetune_model_lr=1e-4,
        finetune_pooler_lr=1e-4,
        finetune_head_lr=1e-4,
        finetune_weight_decay=0.0,
        focal_alpha="auto",
        focal_gamma=2.0,
        hidden_dim=4,
        dropout=0.0,
        max_train_samples=None,
        max_val_samples=None,
        max_test_samples=None,
        pooling="covariance",
        select_metric="average_precision",
        eval_test_every_epoch=False,
        comparison_dir=None,
        previous_ours_prefix=None,
        output_json=None,
    )

    fluorine.run(args)

    saved_state = torch.load(head_state_path, weights_only=False)
    assert evaluation_calls == 1
    assert saved_state["test"]["test/average_precision"] == 1.0


def test_lora_cached_state_injects_adapters_without_full_model_state(
    monkeypatch,
    tmp_path: Path,
):
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
    data = SimpleNamespace(metadata=_adaptation_data_metadata())
    source_checkpoint_contract = _source_checkpoint_contract()
    state_contract = fluorine._fluorine_state_contract(
        source_checkpoint=source_checkpoint_contract,
        data=data,
    )
    monkeypatch.setattr(
        fluorine,
        "build_fluorine_data",
        lambda **_kwargs: data,
    )
    torch.save(
        {
            "mode": "lora",
            "complete": True,
            "config_path": str(config_path),
            "checkpoint_path": str(checkpoint_path),
            "pooling": "covariance",
            "hparams": hparams,
            "state_contract": state_contract,
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
        source_checkpoint_contract=source_checkpoint_contract,
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
        max_train_samples=None,
        max_val_samples=None,
        max_test_samples=None,
        pooling="covariance",
    )

    block = model.encoder.blocks[0]
    assert loaded["mode"] == "lora"
    assert "model_state" not in loaded
    assert set(loaded["lora_state"]) == set(lora_state)
    assert isinstance(block.single_attention.wqkv, LoRALinear)
    assert isinstance(block.single_attention.wo, LoRALinear)
    assert isinstance(block.single_transition.fc1, LoRALinear)
    assert isinstance(block.single_transition.fc2, LoRALinear)
    assert isinstance(block.single_transition.fc3, LoRALinear)
    assert isinstance(block.pair_transition.fc1, LoRALinear)
    assert isinstance(block.pair_transition.fc2, LoRALinear)
    assert isinstance(block.pair_transition.fc3, LoRALinear)
    assert isinstance(block.embed, torch.nn.Linear)


def test_lora_cached_state_requires_current_contract(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "config.py"
    checkpoint_path = tmp_path / "checkpoint.pt"
    state_path = tmp_path / "lora_state.pt"
    monkeypatch.setattr(
        fluorine,
        "build_fluorine_data",
        lambda **_kwargs: SimpleNamespace(metadata=_adaptation_data_metadata()),
    )
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

    with pytest.raises(ValueError, match="training contract"):
        fluorine.train_or_load_lora(
            state_path=state_path,
            model=_FakeFluorineModel(),
            config={},
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            source_checkpoint_contract=_source_checkpoint_contract(),
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
            max_train_samples=None,
            max_val_samples=None,
            max_test_samples=None,
            pooling="covariance",
        )
