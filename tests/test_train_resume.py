import tempfile

import torch
from ml_collections import config_dict

from models.model import PeakSetSIGReg
from train import _is_weight_decay_target, _load_resume_model_state, _save_checkpoint
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
