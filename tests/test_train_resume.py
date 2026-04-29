import tempfile

import pytest
import torch
from ml_collections import config_dict

from models.model import PeakSetSIGReg
from train import (
    _build_optimizers,
    _is_predictor_parameter,
    _is_weight_decay_target,
    _load_resume_model_state,
    _save_checkpoint,
)
from utils.training import _build_wandb_init_kwargs, build_model_from_config


def _small_model(**overrides) -> PeakSetSIGReg:
    kwargs = dict(
        model_dim=64,
        encoder_num_layers=2,
        encoder_num_heads=4,
        encoder_num_kv_heads=4,
        attention_mlp_multiple=2.0,
        feature_mlp_hidden_dim=32,
        masked_token_loss_weight=1.0,
        masked_latent_predictor_num_layers=1,
        jepa_num_target_blocks=1,
        num_peaks=8,
    )
    kwargs.update(overrides)
    return PeakSetSIGReg(**kwargs)


def _optimizer_param_ids(optimizer: torch.optim.Optimizer) -> set[int]:
    return {
        id(param)
        for group in optimizer.param_groups
        for param in group["params"]
    }


def _optimizer_config(**overrides) -> config_dict.ConfigDict:
    cfg = config_dict.ConfigDict()
    cfg.learning_rate = 1e-3
    cfg.min_learning_rate = 1e-4
    cfg.warmup_steps = 0
    cfg.b2 = 0.999
    cfg.weight_decay = 0.01
    cfg.optimizer = "adamw"
    cfg.optimizer_capturable = False
    cfg.optimizer_fused = False
    cfg.update(overrides)
    return cfg


def test_save_checkpoint_persists_nested_scalar_optimizer_state():
    model = _small_model()
    matrix_param = next(
        param for param in model.parameters() if param.requires_grad and param.ndim >= 2
    )
    vector_param = next(
        param for param in model.parameters() if param.requires_grad and param.ndim == 1
    )
    optimizer = torch.optim.SGD([matrix_param], lr=0.1)
    optimizer.scalar_optimizer = torch.optim.AdamW([vector_param], lr=0.01)

    loss = matrix_param.sum() + vector_param.sum()
    loss.backward()
    optimizer.step()
    optimizer.scalar_optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    optimizer.scalar_optimizer.zero_grad(set_to_none=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/resume.pt"
        _save_checkpoint(
            path=path,
            model=model,
            optimizers=[optimizer],
            schedulers=[],
            global_step=12,
            epoch=1,
            loss=float(loss.detach()),
            wandb_run_id="wandb-run-123",
        )
        ckpt = torch.load(path, map_location="cpu", weights_only=True)

    saved_optimizer = ckpt["optimizers"][0]
    assert ckpt["wandb_run_id"] == "wandb-run-123"
    assert "state_dict" in saved_optimizer
    assert "scalar_optimizer_state" in saved_optimizer
    assert saved_optimizer["scalar_optimizer_state"]["state"]


def test_build_optimizers_uses_single_adamw_optimizer_by_default():
    model = _small_model()
    cfg = _optimizer_config()

    optimizers, schedulers = _build_optimizers(
        cfg,
        model,
        total_steps=10,
        device=torch.device("cpu"),
    )

    assert len(optimizers) == 1
    assert len(schedulers) == 1


def test_build_optimizers_applies_predictor_learning_rate_ratio():
    model = _small_model()
    cfg = _optimizer_config(predictor_learning_rate_ratio=3.0)

    optimizers, schedulers = _build_optimizers(
        cfg,
        model,
        total_steps=10,
        device=torch.device("cpu"),
    )

    assert len(optimizers) == 2
    assert len(schedulers) == 2
    assert all(
        float(group["lr"]) == pytest.approx(1e-3)
        for group in optimizers[0].param_groups
    )
    assert all(
        float(group["lr"]) == pytest.approx(3e-3)
        for group in optimizers[1].param_groups
    )
    assert schedulers[0].eta_min == pytest.approx(1e-4)
    assert schedulers[1].eta_min == pytest.approx(3e-4)

    predictor_param_ids = {
        id(param)
        for name, param in model.named_parameters()
        if param.requires_grad and _is_predictor_parameter(name)
    }
    base_param_ids = _optimizer_param_ids(optimizers[0])
    actual_predictor_param_ids = _optimizer_param_ids(optimizers[1])
    all_trainable_param_ids = {
        id(param) for param in model.parameters() if param.requires_grad
    }

    assert actual_predictor_param_ids == predictor_param_ids
    assert base_param_ids.isdisjoint(actual_predictor_param_ids)
    assert base_param_ids | actual_predictor_param_ids == all_trainable_param_ids


def test_load_resume_model_state_rejects_sigreg_checkpoint_drift():
    model = _small_model(representation_regularizer="sigreg", sigreg_lambda=0.02)
    resume_state = model.state_dict()
    for key in ("sigreg.t", "sigreg.phi", "sigreg.weights"):
        resume_state.pop(key)
    for key in tuple(resume_state):
        if key.startswith("target_projector."):
            resume_state.pop(key)
    resume_state["sigreg_lambda_target"] = torch.tensor(0.02)
    resume_state["sigreg_lambda_current"] = torch.tensor(0.02)
    resume_state["sigreg_lambda_step"] = torch.tensor(0)

    restored = _small_model(representation_regularizer="sigreg", sigreg_lambda=0.02)
    with pytest.raises(RuntimeError, match="Missing key"):
        _load_resume_model_state(restored, resume_state)


def test_load_resume_model_state_rejects_removed_cls_predictor_keys():
    model = _small_model()
    resume_state = model.state_dict()
    resume_state["cls_predictor.0.weight"] = torch.ones(model.model_dim)
    resume_state["cls_predictor.0.bias"] = torch.zeros(model.model_dim)
    resume_state["cls_predictor.1.weight"] = torch.randn(
        model.predictor_dim,
        model.model_dim,
    )
    resume_state["cls_predictor.3.weight"] = torch.randn(
        model.model_dim,
        model.predictor_dim,
    )

    restored = _small_model()
    with pytest.raises(RuntimeError, match="Unexpected key"):
        _load_resume_model_state(restored, resume_state)


def test_load_resume_model_state_rejects_removed_target_projector():
    model = _small_model()
    resume_state = model.state_dict()

    restored = _small_model(target_projector_dim=-1)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        _load_resume_model_state(restored, resume_state)


def test_load_resume_model_state_rejects_missing_ema_target_projector():
    model = _small_model()
    resume_state = model.state_dict()

    restored = _small_model(use_ema_teacher=True)
    with pytest.raises(RuntimeError, match="Missing key"):
        _load_resume_model_state(restored, resume_state)


def test_load_resume_model_state_rejects_removed_special_tokens():
    model = _small_model(
        encoder_num_register_tokens=2,
        predictor_num_register_tokens=2,
    )
    resume_state = model.state_dict()

    restored = _small_model(
        encoder_use_cls_token=False,
        encoder_num_register_tokens=0,
        predictor_num_register_tokens=0,
    )
    with pytest.raises(RuntimeError, match="Unexpected key"):
        _load_resume_model_state(restored, resume_state)


def test_build_wandb_init_kwargs_prefers_config_resume_id(monkeypatch):
    monkeypatch.delenv("WANDB_RESUME_ID", raising=False)
    cfg = config_dict.ConfigDict()
    cfg.wandb_kwargs = {"name": "fresh-run"}
    cfg.wandb_resume_id = "resume-123"

    kwargs = _build_wandb_init_kwargs(cfg)

    assert kwargs["id"] == "resume-123"
    assert kwargs["resume"] == "must"
    assert "name" not in kwargs


def test_is_weight_decay_target_matches_pretrain_expectation():
    model = _small_model()
    assert _is_weight_decay_target(
        "encoder.embedder.output_proj.weight",
        model.encoder.embedder.output_proj.weight,
    )
    assert _is_weight_decay_target(
        "encoder.embedder.fourier_ffn.0.weight",
        model.encoder.embedder.fourier_ffn[0].weight,
    )
    assert not _is_weight_decay_target(
        "encoder.embedder.mz_fourier.b",
        model.encoder.embedder.mz_fourier.b,
    )


def test_jepa_mae_mz_scale_follows_peak_mz_preprocessing_scale():
    cfg = config_dict.ConfigDict()
    cfg.model_dim = 32
    cfg.encoder_num_layers = 1
    cfg.encoder_num_heads = 4
    cfg.encoder_num_kv_heads = 4
    cfg.attention_mlp_multiple = 2.0
    cfg.feature_mlp_hidden_dim = 16
    cfg.num_peaks = 8
    cfg.peak_mz_max = 750.0
    cfg.encoder_fourier_input_scale = 1000.0
    cfg.max_precursor_mz = 2000.0
    cfg.jepa_mae_loss_weight = 1.0
    cfg.jepa_mae_mz_bin_size = 2.5

    model = build_model_from_config(cfg)

    assert model.jepa_mae_mz_max == 750.0
    assert model.jepa_mae_num_mz_bins == 300
