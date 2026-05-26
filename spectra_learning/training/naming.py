from typing import Any

from spectra_learning.models.factory import build_model_from_config


def auto_run_name(config: Any) -> str:
    parts = [
        _model_size_part(config),
        f"d{int(config.get('model_dim', 0))}",
        f"L{int(config.get('encoder_num_layers', 0))}",
        f"h{int(config.get('encoder_num_heads', 0))}",
        f"pred{int(config.get('masked_latent_predictor_num_layers', 0))}",
        f"bs{int(config.get('batch_size', 0))}",
        str(config.get("optimizer", "adamw")).lower(),
        f"lr{float(config.get('learning_rate', 0)):.0e}",
        f"wd{float(config.get('weight_decay', 0)):.0e}",
        f"ep{config.get('num_epochs', '?')}",
    ]
    parts.extend(_training_mode_parts(config))
    parts.extend(_architecture_parts(config))
    parts.extend(_optimization_parts(config))
    parts.extend(_target_parts(config))
    parts.extend(_objective_parts(config))
    parts.extend(_ema_parts(config))
    suffix = str(config.get("run_name_suffix", "")).strip()
    if suffix:
        parts.append(suffix)
    return "_".join(parts)


def _model_size_part(config: Any) -> str:
    model = build_model_from_config(config)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    del model
    if total_params >= 1_000_000_000:
        return f"{total_params / 1_000_000_000:.1f}B"
    return f"{total_params / 1_000_000:.0f}M"


def _architecture_parts(config: Any) -> list[str]:
    parts: list[str] = []
    if not bool(config.get("encoder_use_fourier_features", True)):
        parts.append("no-fourier")
    fourier_mlp_layers = int(config.get("encoder_fourier_mlp_num_layers", 2))
    fourier_mlp_hidden = config.get("encoder_fourier_mlp_hidden_dim", None)
    if fourier_mlp_layers != 2 or fourier_mlp_hidden is not None:
        hidden = (
            int(config.get("feature_mlp_hidden_dim", 128))
            if fourier_mlp_hidden is None
            else int(fourier_mlp_hidden)
        )
        parts.append(f"fmlp{fourier_mlp_layers}x{hidden}")
    return parts


def _training_mode_parts(config: Any) -> list[str]:
    mode = str(config.get("training_mode", "jepa")).lower()
    return [] if mode == "jepa" else [mode]


def _optimization_parts(config: Any) -> list[str]:
    parts: list[str] = []
    min_lr = config.get("min_learning_rate", None)
    if min_lr is not None:
        parts.append(f"minlr{float(min_lr):.0e}")
    warmup = int(config.get("warmup_steps", 0))
    if warmup > 0:
        parts.append(f"wu{warmup // 1000}k")
    return parts


def _target_parts(config: Any) -> list[str]:
    parts: list[str] = []
    mode = str(config.get("training_mode", "jepa")).lower()
    target_layers = config.get("jepa_target_layers", None)
    if mode == "jepa" and target_layers:
        parts.append(f"tgt{'_'.join(str(x) for x in target_layers)}")
    target_projector_dim = config.get("target_projector_dim", None)
    model_dim = int(config.get("model_dim", 0))
    if target_projector_dim is not None and int(target_projector_dim) < 0:
        parts.append("no-tproj")
    elif target_projector_dim is not None and int(target_projector_dim) != model_dim:
        parts.append(f"tproj{int(target_projector_dim)}")
    return parts


def _objective_parts(config: Any) -> list[str]:
    parts: list[str] = []
    has_mz_bin_part = False
    if str(config.get("training_mode", "jepa")).lower() == "mae":
        parts.append(f"maew{float(config.get('mae_loss_weight', 1.0)):.0e}")
        parts.append(f"mzbin{float(config.get('jepa_mae_mz_bin_size', 2.5)):g}")
        has_mz_bin_part = True
        parts.append(
            f"intbin{float(config.get('jepa_mae_intensity_bin_size', 0.1)):g}"
        )
    elif (jepa_mae_loss_weight := float(config.get("jepa_mae_loss_weight", 0.0))) > 0:
        parts.append("jepamae")
        parts.append(f"maew{jepa_mae_loss_weight:.0e}")
        parts.append(f"mzbin{float(config.get('jepa_mae_mz_bin_size', 2.5)):g}")
        has_mz_bin_part = True
        parts.append(
            f"intbin{float(config.get('jepa_mae_intensity_bin_size', 0.1)):g}"
        )
    if (distogram_loss_weight := float(config.get("distogram_loss_weight", 0.0))) > 0:
        parts.append("disto")
        parts.append(f"distow{distogram_loss_weight:.0e}")
        if not has_mz_bin_part:
            parts.append(f"mzbin{float(config.get('jepa_mae_mz_bin_size', 2.5)):g}")
    return parts


def _ema_parts(config: Any) -> list[str]:
    if str(config.get("training_mode", "jepa")).lower() != "jepa":
        return []
    if not config.get("use_ema_teacher", False):
        return []
    parts = [
        "ema",
        str(config.get("ema_teacher_schedule", "constant")),
        f"mstart{float(config.get('ema_teacher_momentum_start', 0.996)):.4f}",
    ]
    ema_mid = config.get("ema_teacher_momentum_mid", None)
    if ema_mid is not None:
        parts.append(f"mmid{float(ema_mid):.4f}")
    ema_final = config.get("ema_teacher_momentum_final", None)
    if ema_final is not None:
        parts.append(f"mfinal{float(ema_final):.4f}")
    if str(config.get("ema_teacher_schedule", "")).lower() == "slow-fast-slow":
        peak_frac = float(config.get("ema_teacher_schedule_peak_fraction", 0.35))
        parts.append(f"peak{peak_frac:.2f}")
    return parts
