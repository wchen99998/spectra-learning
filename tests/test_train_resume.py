import tempfile

import torch
from ml_collections import config_dict

from models.model import PeakSetSIGReg
from train import (
    _build_optimizers,
    _is_weight_decay_target,
    _load_optimizer_states,
    _load_resume_model_state,
    _save_checkpoint,
)
from utils.training import _build_wandb_init_kwargs


def _small_model(**overrides) -> PeakSetSIGReg:
    kwargs = dict(
        model_dim=64,
        encoder_num_layers=2,
        encoder_num_heads=4,
        encoder_num_kv_heads=4,
        attention_mlp_multiple=2.0,
        feature_mlp_hidden_dim=32,
        masked_token_loss_weight=1.0,
        masked_token_loss_type="l2",
        masked_latent_predictor_num_layers=1,
        jepa_num_target_blocks=1,
        num_peaks=8,
        use_ema_teacher_target=False,
    )
    kwargs.update(overrides)
    return PeakSetSIGReg(**kwargs)


def _small_train_config(**overrides) -> config_dict.ConfigDict:
    cfg = config_dict.ConfigDict()
    cfg.learning_rate = 5e-4
    cfg.warmup_steps = 0
    cfg.min_learning_rate = 3e-5
    cfg.b2 = 0.95
    cfg.weight_decay = 0.05
    cfg.optimizer = "muon"
    cfg.optimizer_capturable = False
    cfg.optimizer_fused = False
    cfg.muon_lr = None
    cfg.adamw_lr = None
    cfg.muon_momentum = 0.95
    cfg.muon_nesterov = True
    cfg.muon_ns_steps = 5
    cfg.muon_weight_decay = None
    cfg.muon_adjust_lr_fn = "match_rms_adamw"
    for key, value in overrides.items():
        cfg[key] = value
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


def test_load_resume_model_state_allows_sigreg_checkpoint_compatibility():
    model = _small_model(representation_regularizer="sigreg", sigreg_lambda=0.02)
    resume_state = model.state_dict()
    for key in ("sigreg.t", "sigreg.phi", "sigreg.weights"):
        resume_state.pop(key)
    resume_state["sigreg_lambda_target"] = torch.tensor(0.02)
    resume_state["sigreg_lambda_current"] = torch.tensor(0.02)
    resume_state["sigreg_lambda_step"] = torch.tensor(0)

    restored = _small_model(representation_regularizer="sigreg", sigreg_lambda=0.02)
    _load_resume_model_state(restored, resume_state)


def test_build_optimizers_uses_torch_muon_and_adamw():
    model = _small_model()
    config = _small_train_config()

    optimizers, schedulers = _build_optimizers(
        config,
        model,
        total_steps=16,
        device=torch.device("cpu"),
    )

    assert len(optimizers) == 2
    assert len(schedulers) == 2
    assert isinstance(optimizers[0], torch.optim.Muon)
    assert isinstance(optimizers[1], torch.optim.AdamW)

    muon_params = {
        id(param)
        for group in optimizers[0].param_groups
        for param in group["params"]
    }
    adamw_params = {
        id(param)
        for group in optimizers[1].param_groups
        for param in group["params"]
    }
    all_trainable = {id(param) for param in model.parameters() if param.requires_grad}

    assert muon_params.isdisjoint(adamw_params)
    assert muon_params | adamw_params == all_trainable
    assert all(param.ndim == 2 for group in optimizers[0].param_groups for param in group["params"])


def test_load_optimizer_states_restores_legacy_nested_muon_format():
    model = _small_model()
    config = _small_train_config()
    optimizers, _ = _build_optimizers(
        config,
        model,
        total_steps=16,
        device=torch.device("cpu"),
    )
    for param in model.parameters():
        if param.requires_grad:
            param.grad = torch.ones_like(param)
    for opt in optimizers:
        opt.step()
        opt.zero_grad(set_to_none=True)

    legacy_state = [
        {
            "state_dict": optimizers[0].state_dict(),
            "scalar_optimizer_state": optimizers[1].state_dict(),
        }
    ]

    restored_model = _small_model()
    restored_optimizers, _ = _build_optimizers(
        config,
        restored_model,
        total_steps=16,
        device=torch.device("cpu"),
    )
    _load_optimizer_states(restored_optimizers, legacy_state)

    assert len(restored_optimizers[0].state) == len(optimizers[0].state)
    assert len(restored_optimizers[1].state) == len(optimizers[1].state)


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
