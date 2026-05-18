from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from typing import Any

from spectra_learning.data.spectra import PEAK_MZ_MAX


@dataclass(slots=True)
class PeakSetSIGRegSettings:
    training_mode: str = "jepa"
    model_dim: int = 768
    encoder_num_layers: int = 20
    encoder_num_heads: int = 12
    encoder_num_kv_heads: int | None = None
    attention_mlp_multiple: float = 4.0
    feature_mlp_hidden_dim: int = 128
    encoder_fourier_mlp_hidden_dim: int | None = None
    encoder_fourier_mlp_num_layers: int = 2
    encoder_fourier_strategy: str = "log_spaced"
    encoder_fourier_x_min: float = 3e-3
    encoder_fourier_x_max: float = 1000.0
    encoder_fourier_funcs: str = "both"
    encoder_fourier_num_freqs: int = 256
    encoder_fourier_sigma: float = 10.0
    encoder_fourier_trainable: bool = False
    encoder_fourier_input_scale: float = PEAK_MZ_MAX
    encoder_use_fourier_features: bool = True
    masked_token_loss_weight: float = 0.0
    mae_loss_weight: float = 1.0
    jepa_mae_loss_weight: float = 0.0
    jepa_mae_mz_bin_size: float = 2.5
    jepa_mae_intensity_bin_size: float = 0.1
    jepa_mae_mz_max: float = PEAK_MZ_MAX
    jepa_mae_intensity_max: float = 1.0
    jepa_target_normalization: str = "none"
    jepa_target_layers: list[int] | tuple[int, ...] | None = None
    masked_token_input_mode: str = "latent_token"
    masked_mz_sentinel: float = -1.0
    representation_regularizer: str = "none"
    masked_latent_predictor_num_layers: int = 2
    masked_latent_predictor_num_heads: int = 8
    sigreg_num_slices: int = 256
    sigreg_lambda: float = 0.02
    sigreg_precursor_scale: float = 1.0
    jepa_num_target_blocks: int = 2
    encoder_qk_norm: bool = False
    norm_type: str = "rmsnorm"
    norm_eps: float = 1e-5
    encoder_use_position_embedding: bool = True
    encoder_apply_final_norm: bool = True
    predictor_apply_final_norm: bool = True
    predictor_use_rope: bool = True
    encoder_use_cls_token: bool = True
    use_precursor_token: bool = False
    spectral_bias_relative_kind: str = "none"
    spectral_bias_use_precursor: bool = False
    spectral_bias_use_intensity: bool = False
    spectral_bias_num_freqs: int = 128
    spectral_bias_fourier_strategy: str = "log_spaced"
    spectral_bias_fourier_x_min: float = 3e-3
    spectral_bias_fourier_x_max: float = 1000.0
    spectral_bias_fourier_sigma: float = 10.0
    spectral_bias_fourier_trainable: bool = False
    spectral_bias_mass_scale: float = PEAK_MZ_MAX
    spectral_bias_precursor_scale: float = PEAK_MZ_MAX
    spectral_bias_rbf_num_basis: int = 64
    spectral_bias_rbf_delta_min: float = -PEAK_MZ_MAX
    spectral_bias_rbf_delta_max: float = PEAK_MZ_MAX
    spectral_bias_rbf_use_absolute_delta: bool = False
    spectral_bias_intensity_hidden_dim: int = 16
    spectral_bias_init_std: float = 0.0
    spectral_bias_clip: float | None = None
    num_peaks: int = 64
    temporal_predictor_num_layers: int = 0
    encoder_num_register_tokens: int = 0
    predictor_num_register_tokens: int = 0
    predictor_dim: int | None = None
    target_projector_dim: int | None = None
    predictor_dropout: float = 0.0
    covariance_pooling_dim: int = -1
    train_covariance_pooling: bool = True
    covariance_pooling_loss_weight: float = 1.0
    use_ema_teacher: bool = False
    ema_teacher_momentum_start: float = 0.996
    ema_teacher_momentum_mid: float | None = None
    ema_teacher_momentum_final: float | None = None
    ema_teacher_schedule_peak_fraction: float = 0.35
    ema_teacher_schedule: str = "constant"

    @classmethod
    def from_config(cls, config: Any) -> "PeakSetSIGRegSettings":
        values = _default_values(cls())
        _apply_derived_defaults(values, config)
        for name, cast in SETTING_CASTS.items():
            values[name] = cast(_config_get(config, name, values[name]))
        _apply_dependent_defaults(values, config)
        return cls(**values)

    @classmethod
    def create(
        cls,
        settings: "PeakSetSIGRegSettings | None" = None,
        **overrides: Any,
    ) -> "PeakSetSIGRegSettings":
        if settings is None:
            return cls(**overrides)
        if not overrides:
            return settings
        return replace(settings, **overrides)


def _default_values(settings: PeakSetSIGRegSettings) -> dict[str, Any]:
    return {field.name: getattr(settings, field.name) for field in fields(settings)}


def _apply_derived_defaults(values: dict[str, Any], config: Any) -> None:
    peak_mz_max = float(_config_get(config, "peak_mz_max", PEAK_MZ_MAX))
    precursor_mz_max = float(_config_get(config, "max_precursor_mz", PEAK_MZ_MAX))
    values["encoder_fourier_input_scale"] = peak_mz_max
    values["jepa_mae_mz_max"] = peak_mz_max
    values["spectral_bias_precursor_scale"] = precursor_mz_max
    values["spectral_bias_rbf_delta_min"] = -precursor_mz_max
    values["spectral_bias_rbf_delta_max"] = precursor_mz_max
    values["spectral_bias_num_freqs"] = _config_get(
        config,
        "encoder_fourier_num_freqs",
        values["spectral_bias_num_freqs"],
    )
    values["spectral_bias_fourier_strategy"] = _config_get(
        config,
        "encoder_fourier_strategy",
        values["spectral_bias_fourier_strategy"],
    )
    values["spectral_bias_fourier_x_min"] = _config_get(
        config,
        "encoder_fourier_x_min",
        values["spectral_bias_fourier_x_min"],
    )
    values["spectral_bias_fourier_x_max"] = _config_get(
        config,
        "encoder_fourier_x_max",
        values["spectral_bias_fourier_x_max"],
    )
    values["spectral_bias_fourier_sigma"] = _config_get(
        config,
        "encoder_fourier_sigma",
        values["spectral_bias_fourier_sigma"],
    )


def _apply_dependent_defaults(values: dict[str, Any], config: Any) -> None:
    if not _config_has(config, "spectral_bias_mass_scale"):
        values["spectral_bias_mass_scale"] = values["encoder_fourier_input_scale"]


def _config_get(config: Any, key: str, default: Any) -> Any:
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _config_has(config: Any, key: str) -> bool:
    if hasattr(config, "__contains__") and key in config:
        return True
    return hasattr(config, key)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _identity(value: Any) -> Any:
    return value


SETTING_CASTS: dict[str, Callable[[Any], Any]] = {
    "training_mode": str,
    "model_dim": int,
    "encoder_num_layers": int,
    "encoder_num_heads": int,
    "encoder_num_kv_heads": _optional_int,
    "attention_mlp_multiple": float,
    "feature_mlp_hidden_dim": int,
    "encoder_fourier_mlp_hidden_dim": _optional_int,
    "encoder_fourier_mlp_num_layers": int,
    "encoder_fourier_strategy": str,
    "encoder_fourier_x_min": float,
    "encoder_fourier_x_max": float,
    "encoder_fourier_funcs": str,
    "encoder_fourier_num_freqs": int,
    "encoder_fourier_sigma": float,
    "encoder_fourier_trainable": bool,
    "encoder_fourier_input_scale": float,
    "encoder_use_fourier_features": bool,
    "masked_token_loss_weight": float,
    "mae_loss_weight": float,
    "jepa_mae_loss_weight": float,
    "jepa_mae_mz_bin_size": float,
    "jepa_mae_intensity_bin_size": float,
    "jepa_mae_mz_max": float,
    "jepa_mae_intensity_max": float,
    "jepa_target_normalization": str,
    "jepa_target_layers": _identity,
    "masked_token_input_mode": str,
    "masked_mz_sentinel": float,
    "representation_regularizer": str,
    "masked_latent_predictor_num_layers": int,
    "masked_latent_predictor_num_heads": int,
    "sigreg_num_slices": int,
    "sigreg_lambda": float,
    "sigreg_precursor_scale": float,
    "jepa_num_target_blocks": int,
    "encoder_qk_norm": bool,
    "norm_type": str,
    "norm_eps": float,
    "encoder_use_position_embedding": bool,
    "encoder_apply_final_norm": bool,
    "predictor_apply_final_norm": bool,
    "predictor_use_rope": bool,
    "encoder_use_cls_token": bool,
    "use_precursor_token": bool,
    "spectral_bias_relative_kind": str,
    "spectral_bias_use_precursor": bool,
    "spectral_bias_use_intensity": bool,
    "spectral_bias_num_freqs": int,
    "spectral_bias_fourier_strategy": str,
    "spectral_bias_fourier_x_min": float,
    "spectral_bias_fourier_x_max": float,
    "spectral_bias_fourier_sigma": float,
    "spectral_bias_fourier_trainable": bool,
    "spectral_bias_mass_scale": float,
    "spectral_bias_precursor_scale": float,
    "spectral_bias_rbf_num_basis": int,
    "spectral_bias_rbf_delta_min": float,
    "spectral_bias_rbf_delta_max": float,
    "spectral_bias_rbf_use_absolute_delta": bool,
    "spectral_bias_intensity_hidden_dim": int,
    "spectral_bias_init_std": float,
    "spectral_bias_clip": _optional_float,
    "num_peaks": int,
    "temporal_predictor_num_layers": int,
    "encoder_num_register_tokens": int,
    "predictor_num_register_tokens": int,
    "predictor_dim": _optional_int,
    "target_projector_dim": _optional_int,
    "predictor_dropout": float,
    "covariance_pooling_dim": int,
    "train_covariance_pooling": bool,
    "covariance_pooling_loss_weight": float,
    "use_ema_teacher": bool,
    "ema_teacher_momentum_start": float,
    "ema_teacher_momentum_mid": _optional_float,
    "ema_teacher_momentum_final": _optional_float,
    "ema_teacher_schedule_peak_fraction": float,
    "ema_teacher_schedule": str,
}
