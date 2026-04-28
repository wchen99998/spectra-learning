import importlib.util
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.loggers import CSVLogger
from ml_collections import config_dict

from models.model import PeakSetSIGReg
from utils.spectra_preprocessing import PEAK_MZ_MAX

def load_config(path: str | Path) -> config_dict.ConfigDict:
    path = Path(path)
    spec = importlib.util.spec_from_file_location("experiment_config", path)
    assert spec is not None, f"Could not load module spec from {path}"
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.get_config()


def parse_autocast_dtype(value: object) -> torch.dtype | None:
    name = str(value).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp32", "float32", "none"}:
        return None
    if name in {"fp16", "float16", "half"}:
        raise ValueError("autocast_dtype=fp16 requires GradScaler; use bf16 or fp32")
    raise ValueError(f"Unsupported autocast_dtype: {name}")


def build_model_from_config(config: config_dict.ConfigDict) -> PeakSetSIGReg:
    encoder_num_layers = int(
        config.get("encoder_num_layers", config.get("num_layers"))
    )
    encoder_num_heads = int(
        config.get("encoder_num_heads", config.get("num_heads"))
    )
    encoder_num_kv_heads = config.get(
        "encoder_num_kv_heads",
        config.get("num_kv_heads", None),
    )
    encoder_fourier_mlp_hidden_dim = config.get(
        "encoder_fourier_mlp_hidden_dim",
        None,
    )
    spectral_bias_clip = config.get("spectral_bias_clip", None)
    spectral_bias_clip = (
        None if spectral_bias_clip is None else float(spectral_bias_clip)
    )
    return PeakSetSIGReg(
        model_dim=int(config.model_dim),
        encoder_num_layers=encoder_num_layers,
        encoder_num_heads=encoder_num_heads,
        encoder_num_kv_heads=encoder_num_kv_heads,
        attention_mlp_multiple=float(config.attention_mlp_multiple),
        feature_mlp_hidden_dim=int(config.get("feature_mlp_hidden_dim", 128)),
        encoder_fourier_mlp_hidden_dim=(
            None
            if encoder_fourier_mlp_hidden_dim is None
            else int(encoder_fourier_mlp_hidden_dim)
        ),
        encoder_fourier_mlp_num_layers=int(
            config.get("encoder_fourier_mlp_num_layers", 2)
        ),
        encoder_fourier_strategy=str(
            config.get("encoder_fourier_strategy", "log_spaced")
        ),
        encoder_fourier_x_min=float(config.get("encoder_fourier_x_min", 3e-3)),
        encoder_fourier_x_max=float(config.get("encoder_fourier_x_max", 1000.0)),
        encoder_fourier_funcs=str(config.get("encoder_fourier_funcs", "both")),
        encoder_fourier_num_freqs=int(config.get("encoder_fourier_num_freqs", 256)),
        encoder_fourier_sigma=float(config.get("encoder_fourier_sigma", 10.0)),
        encoder_fourier_trainable=bool(
            config.get("encoder_fourier_trainable", False)
        ),
        encoder_fourier_input_scale=float(
            config.get(
                "encoder_fourier_input_scale",
                config.get("peak_mz_max", config.get("max_precursor_mz", PEAK_MZ_MAX)),
            )
        ),
        masked_token_loss_weight=float(config.get("masked_token_loss_weight", 0.0)),
        masked_token_loss_type=str(config.get("masked_token_loss_type", "l1")),
        jepa_mae_loss_weight=float(config.get("jepa_mae_loss_weight", 0.0)),
        jepa_mae_mz_bin_size=float(config.get("jepa_mae_mz_bin_size", 2.5)),
        jepa_mae_intensity_bin_size=float(
            config.get("jepa_mae_intensity_bin_size", 0.1)
        ),
        jepa_mae_mz_max=float(
            config.get(
                "jepa_mae_mz_max",
                config.get("peak_mz_max", PEAK_MZ_MAX),
            )
        ),
        jepa_mae_intensity_max=float(config.get("jepa_mae_intensity_max", 1.0)),
        jepa_target_normalization=str(
            config.get("jepa_target_normalization", "none")
        ),
        jepa_target_layers=config.get("jepa_target_layers", None),
        representation_regularizer=str(
            config.get("representation_regularizer", "none")
        ),
        masked_latent_predictor_num_layers=int(
            config.get("masked_latent_predictor_num_layers", 2)
        ),
        masked_latent_predictor_num_heads=int(
            config.get("masked_latent_predictor_num_heads", 8)
        ),
        sigreg_num_slices=int(config.get("sigreg_num_slices", 256)),
        sigreg_lambda=float(config.get("sigreg_lambda", 0.02)),
        sigreg_precursor_scale=float(config.get("sigreg_precursor_scale", 1.0)),
        jepa_num_target_blocks=int(config.get("jepa_num_target_blocks", 2)),
        jepa_context_fraction=float(config.get("jepa_context_fraction", 0.5)),
        jepa_target_fraction=float(config.get("jepa_target_fraction", 0.25)),
        encoder_qk_norm=bool(config.get("encoder_qk_norm", False)),
        norm_type=str(config.get("norm_type", "rmsnorm")),
        norm_eps=float(config.get("norm_eps", 1e-5)),
        encoder_use_position_embedding=bool(
            config.get("encoder_use_position_embedding", True)
        ),
        encoder_apply_final_norm=bool(config.get("encoder_apply_final_norm", True)),
        predictor_apply_final_norm=bool(
            config.get("predictor_apply_final_norm", True)
        ),
        encoder_use_cls_token=bool(config.get("encoder_use_cls_token", True)),
        use_precursor_token=bool(config.get("use_precursor_token", False)),
        spectral_bias_relative_kind=str(
            config.get("spectral_bias_relative_kind", "none")
        ),
        spectral_bias_use_precursor=bool(
            config.get("spectral_bias_use_precursor", False)
        ),
        spectral_bias_use_intensity=bool(
            config.get("spectral_bias_use_intensity", False)
        ),
        spectral_bias_num_freqs=int(
            config.get(
                "spectral_bias_num_freqs",
                config.get("encoder_fourier_num_freqs", 128),
            )
        ),
        spectral_bias_fourier_strategy=str(
            config.get(
                "spectral_bias_fourier_strategy",
                config.get("encoder_fourier_strategy", "log_spaced"),
            )
        ),
        spectral_bias_fourier_x_min=float(
            config.get(
                "spectral_bias_fourier_x_min",
                config.get("encoder_fourier_x_min", 3e-3),
            )
        ),
        spectral_bias_fourier_x_max=float(
            config.get(
                "spectral_bias_fourier_x_max",
                config.get("encoder_fourier_x_max", 1000.0),
            )
        ),
        spectral_bias_fourier_sigma=float(
            config.get(
                "spectral_bias_fourier_sigma",
                config.get("encoder_fourier_sigma", 10.0),
            )
        ),
        spectral_bias_fourier_trainable=bool(
            config.get("spectral_bias_fourier_trainable", False)
        ),
        spectral_bias_mass_scale=float(
            config.get(
                "spectral_bias_mass_scale",
                config.get("encoder_fourier_input_scale", PEAK_MZ_MAX),
            )
        ),
        spectral_bias_precursor_scale=float(
            config.get(
                "spectral_bias_precursor_scale",
                config.get("max_precursor_mz", PEAK_MZ_MAX),
            )
        ),
        spectral_bias_rbf_num_basis=int(
            config.get("spectral_bias_rbf_num_basis", 64)
        ),
        spectral_bias_rbf_delta_min=float(
            config.get(
                "spectral_bias_rbf_delta_min",
                -float(config.get("max_precursor_mz", PEAK_MZ_MAX)),
            )
        ),
        spectral_bias_rbf_delta_max=float(
            config.get(
                "spectral_bias_rbf_delta_max",
                float(config.get("max_precursor_mz", PEAK_MZ_MAX)),
            )
        ),
        spectral_bias_rbf_use_absolute_delta=bool(
            config.get("spectral_bias_rbf_use_absolute_delta", False)
        ),
        spectral_bias_intensity_hidden_dim=int(
            config.get("spectral_bias_intensity_hidden_dim", 16)
        ),
        spectral_bias_init_std=float(config.get("spectral_bias_init_std", 0.0)),
        spectral_bias_clip=spectral_bias_clip,
        num_peaks=int(config.get("num_peaks", 64)),
        encoder_num_register_tokens=int(
            config.get("encoder_num_register_tokens", 0)
        ),
        predictor_num_register_tokens=int(
            config.get("predictor_num_register_tokens", 0)
        ),
        temporal_predictor_num_layers=int(
            config.get("temporal_predictor_num_layers", 0)
        ),
        predictor_dim=config.get("predictor_dim", None),
        target_projector_dim=config.get("target_projector_dim", None),
        use_target_projector=bool(config.get("use_target_projector", True)),
        predictor_dropout=float(config.get("predictor_dropout", 0.0)),
        train_covariance_pooling=bool(config.get("train_covariance_pooling", False)),
        covariance_pooling_dim=int(
            config.get(
                "covariance_pooling_dim",
                config.get("msg_probe_covariance_dim", 32),
            )
        ),
        use_ema_teacher=bool(config.get("use_ema_teacher", False)),
        ema_teacher_momentum=float(config.get("ema_teacher_momentum", 0.996)),
        ema_teacher_momentum_mid=config.get("ema_teacher_momentum_mid", None),
        ema_teacher_momentum_final=config.get("ema_teacher_momentum_final", None),
        ema_teacher_schedule_peak_fraction=float(
            config.get("ema_teacher_schedule_peak_fraction", 0.35)
        ),
        ema_teacher_schedule=str(config.get("ema_teacher_schedule", "constant")),
    )


def collect_and_log_param_metrics(model: torch.nn.Module) -> dict[str, float]:
    by_module: dict[str, list[int]] = {}
    total = trainable = 0
    for name, param in model.named_parameters():
        numel = int(param.numel())
        module_name = name.split(".", 1)[0]
        counts = by_module.setdefault(module_name, [0, 0])
        total += numel
        counts[0] += numel
        if param.requires_grad:
            trainable += numel
            counts[1] += numel
    logging.info(
        "Model parameters: total=%s trainable=%s non_trainable=%s",
        f"{total:,}",
        f"{trainable:,}",
        f"{total - trainable:,}",
    )
    metrics: dict[str, float] = {
        "model/params_total": float(total),
        "model/params_trainable": float(trainable),
        "model/params_non_trainable": float(total - trainable),
    }
    for module_name in sorted(by_module):
        mod_total, mod_train = by_module[module_name]
        logging.info(
            "  [%s] total=%s trainable=%s",
            module_name,
            f"{mod_total:,}",
            f"{mod_train:,}",
        )
        metrics[f"model/params_total/{module_name}"] = float(mod_total)
        metrics[f"model/params_trainable/{module_name}"] = float(mod_train)
    return metrics


def auto_run_name(config: Any) -> str:
    """Generate a descriptive run name from key config hyperparameters."""
    dim = int(config.get("model_dim", 0))
    layers = int(config.get("encoder_num_layers", 0))
    heads = int(config.get("encoder_num_heads", 0))
    pred_layers = int(config.get("masked_latent_predictor_num_layers", 0))
    bs = int(config.get("batch_size", 0))
    opt = str(config.get("optimizer", "adamw")).lower()
    lr = float(config.get("learning_rate", 0))
    epochs = config.get("num_epochs", "?")

    # Estimate total params (build model transiently if needed)
    model = build_model_from_config(config)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    del model
    if total_params >= 1_000_000_000:
        param_str = f"{total_params / 1_000_000_000:.1f}B"
    else:
        param_str = f"{total_params / 1_000_000:.0f}M"

    wd = float(config.get("weight_decay", 0))
    min_lr = config.get("min_learning_rate", None)
    warmup = int(config.get("warmup_steps", 0))

    parts = [
        param_str,
        f"d{dim}",
        f"L{layers}",
        f"h{heads}",
        f"pred{pred_layers}",
        f"bs{bs}",
        opt,
        f"lr{lr:.0e}",
        f"wd{wd:.0e}",
        f"ep{epochs}",
    ]

    fourier_mlp_layers = int(config.get("encoder_fourier_mlp_num_layers", 2))
    fourier_mlp_hidden = config.get("encoder_fourier_mlp_hidden_dim", None)
    if fourier_mlp_layers != 2 or fourier_mlp_hidden is not None:
        hidden = (
            int(config.get("feature_mlp_hidden_dim", 128))
            if fourier_mlp_hidden is None
            else int(fourier_mlp_hidden)
        )
        parts.append(f"fmlp{fourier_mlp_layers}x{hidden}")

    if min_lr is not None:
        parts.append(f"minlr{float(min_lr):.0e}")
    if warmup > 0:
        parts.append(f"wu{warmup // 1000}k")
    if config.get("use_precursor_token", False):
        parts.append("prec")
    spectral_kind = str(config.get("spectral_bias_relative_kind", "none")).lower()
    spectral_precursor = bool(config.get("spectral_bias_use_precursor", False))
    spectral_intensity = bool(config.get("spectral_bias_use_intensity", False))
    if (
        spectral_kind not in ("", "none", "false", "off")
        or spectral_precursor
        or spectral_intensity
    ):
        parts.append(f"sb{spectral_kind or 'none'}")
        if spectral_precursor:
            parts.append("sbprec")
        if spectral_intensity:
            parts.append("sbint")
        parts.append(f"sbf{int(config.get('spectral_bias_num_freqs', 128))}")
    norm = str(config.get("norm_type", "")).lower()
    if norm and norm != "rmsnorm":
        parts.append(norm)
    if not bool(config.get("encoder_use_cls_token", True)):
        parts.append("no-cls")
    if int(config.get("encoder_num_register_tokens", 0)) == 0:
        parts.append("no-ereg")
    if int(config.get("predictor_num_register_tokens", 0)) == 0:
        parts.append("no-preg")
    target_layers = config.get("jepa_target_layers", None)
    if target_layers:
        parts.append(f"tgt{'_'.join(str(x) for x in target_layers)}")
    regularizer = str(config.get("representation_regularizer", "none")).lower()
    if regularizer and regularizer != "none":
        parts.append(regularizer)
        parts.append(f"lam{float(config.get('sigreg_lambda', 0.0)):.0e}")
    jepa_mae_loss_weight = float(config.get("jepa_mae_loss_weight", 0.0))
    if jepa_mae_loss_weight > 0:
        parts.append("jepamae")
        parts.append(f"maew{jepa_mae_loss_weight:.0e}")
        parts.append(f"mzbin{float(config.get('jepa_mae_mz_bin_size', 2.5)):g}")
        parts.append(
            f"intbin{float(config.get('jepa_mae_intensity_bin_size', 0.1)):g}"
        )
    if config.get("train_covariance_pooling", False):
        cov_dim = int(
            config.get(
                "covariance_pooling_dim",
                config.get("msg_probe_covariance_dim", 32),
            )
        )
        parts.append(f"covpool{cov_dim}")
    if config.get("use_ema_teacher", False):
        parts.append("ema")
        parts.append(str(config.get("ema_teacher_schedule", "constant")))
        parts.append(f"m{float(config.get('ema_teacher_momentum', 0.996)):.4f}")
        ema_mid = config.get("ema_teacher_momentum_mid", None)
        if ema_mid is not None:
            parts.append(f"mm{float(ema_mid):.4f}")
        ema_final = config.get("ema_teacher_momentum_final", None)
        if ema_final is not None:
            parts.append(f"mf{float(ema_final):.4f}")
        if str(config.get("ema_teacher_schedule", "")).lower() == "slow-fast-slow":
            peak_frac = float(config.get("ema_teacher_schedule_peak_fraction", 0.35))
            parts.append(f"peak{peak_frac:.2f}")
    if not bool(config.get("use_target_projector", True)):
        parts.append("no-tproj")
    else:
        target_projector_dim = config.get("target_projector_dim", None)
        if target_projector_dim is not None and int(target_projector_dim) != dim:
            parts.append(f"tproj{int(target_projector_dim)}")
    run_name_suffix = str(config.get("run_name_suffix", "")).strip()
    if run_name_suffix:
        parts.append(run_name_suffix)

    return "_".join(parts)


def _build_wandb_init_kwargs(config: Any | None) -> dict[str, Any]:
    if config is None:
        return {}
    wandb_kwargs = dict(config.get("wandb_kwargs", {}) or {})
    resume_id = str(config.get("wandb_resume_id", "") or "")
    if not resume_id:
        resume_id = os.environ.get("WANDB_RESUME_ID", "")
    if resume_id:
        wandb_kwargs.setdefault("id", resume_id)
        wandb_kwargs.setdefault("resume", "must")
        wandb_kwargs.pop("name", None)
        return wandb_kwargs
    if "name" not in wandb_kwargs:
        wandb_kwargs["name"] = auto_run_name(config)
    return wandb_kwargs


def _to_serialisable_config(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, Mapping):
        return {str(k): _to_serialisable_config(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_serialisable_config(v) for v in value]
    return str(value)


def _config_to_wandb_dict(config: Any | None) -> dict[str, Any]:
    if config is None:
        return {}
    if callable(getattr(config, "to_dict", None)):
        return dict(_to_serialisable_config(config.to_dict()))
    if isinstance(config, Mapping):
        return dict(_to_serialisable_config(config))
    return dict(_to_serialisable_config(vars(config)))


def build_logger(config: config_dict.ConfigDict, workdir: Path) -> pl.loggers.Logger:
    if config.get("enable_wandb", False):
        from lightning.pytorch.loggers import WandbLogger

        wandb_kwargs = _build_wandb_init_kwargs(config)
        logger = WandbLogger(
            project=config.get("wandb_project", "md4"),
            save_dir=str(workdir),
            log_model=False,
            **wandb_kwargs,
        )
        logger.log_hyperparams(_config_to_wandb_dict(config))
        return logger
    return CSVLogger(save_dir=str(workdir), name="csv_logs")


def load_pretrained_weights(
    model: PeakSetSIGReg,
    checkpoint_path: str,
) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    sd = ckpt.get("state_dict") or ckpt.get("model") or ckpt
    prefixed = {
        k.removeprefix("model."): v for k, v in sd.items() if k.startswith("model.")
    }
    sd = prefixed or sd
    for key in tuple(sd):
        if key.endswith(
            (
                "position_embedding.weight",
                "predictor_position_embedding.weight",
            )
        ):
            sd.pop(key)
    resize_prefixes = (
        "encoder.embedder.fourier_ffn.",
        "teacher_encoder.embedder.fourier_ffn.",
    )
    model_sd = model.state_dict()
    incompatible = [
        key
        for key, value in sd.items()
        if key.startswith(resize_prefixes)
        and key in model_sd
        and value.shape != model_sd[key].shape
    ]
    for key in incompatible:
        sd.pop(key)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    sync_missing_teacher = any(
        key.startswith(("teacher_encoder.", "teacher_target_projector."))
        for key in missing
    )
    allowed_missing_suffixes = (
        "position_embedding.weight",
        "predictor_position_embedding.weight",
        "cls_token",
        "register_tokens",
        "predictor_register_tokens",
        "temporal_query_token",
    )
    allowed_missing_prefixes = (
        "masked_latent_readout.",
        "target_projector.",
        "jepa_mae_mz_head.",
        "jepa_mae_intensity_head.",
        "sigreg.",
        "covariance_pooler.",
        "covariance_sigreg.",
        "teacher_encoder.",
        "teacher_target_projector.",
        "encoder.embedder.fourier_ffn.",
        "encoder.spectral_attn_biases.",
        "teacher_encoder.module.spectral_attn_biases.",
        "teacher_encoder.spectral_attn_biases.",
    )
    allowed_unexpected_prefixes = (
        "target_projector.",
        "jepa_mae_mz_head.",
        "jepa_mae_intensity_head.",
        "covariance_pooler.",
        "covariance_sigreg.",
        "teacher_encoder.",
        "teacher_target_projector.",
    )
    unexpected = [
        key for key in unexpected
        if not key.startswith(allowed_unexpected_prefixes)
        and not key.endswith(
            (
                "temporal_query_token",
                "cls_token",
                "register_tokens",
                "predictor_register_tokens",
                "covariance_sigreg.t",
                "covariance_sigreg.phi",
                "covariance_sigreg.weights",
                "sigreg_lambda_target",
                "sigreg_lambda_current",
                "sigreg_lambda_step",
            )
        )
    ]
    missing = [
        key for key in missing
        if not key.endswith(allowed_missing_suffixes)
        and not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
    ]
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint load mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    if incompatible:
        logging.warning("Skipped incompatible pretrained keys: %s", incompatible)
    if sync_missing_teacher:
        model.sync_ema_teacher()


def latest_ckpt_path(directory: Path) -> str | None:
    checkpoint_dir = directory / "checkpoints"
    root = checkpoint_dir if checkpoint_dir.exists() else directory
    ckpts = sorted(
        [*root.rglob("*.ckpt"), *root.rglob("*.pt")],
        key=lambda p: p.stat().st_mtime,
    )
    return str(ckpts[-1]) if ckpts else None
