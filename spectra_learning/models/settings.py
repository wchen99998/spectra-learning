from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from typing import Any

from spectra_learning.data.spectra import PEAK_MZ_MAX


@dataclass(slots=True)
class PeakSetJEPASettings:
    training_mode: str = "jepa"
    model_dim: int = 768
    encoder_num_layers: int = 20
    encoder_num_heads: int = 12
    attention_mlp_multiple: float = 4.0
    feature_mlp_hidden_dim: int = 128
    encoder_fourier_mlp_hidden_dim: int | None = None
    encoder_fourier_mlp_num_layers: int = 2
    encoder_fourier_x_min: float = 3e-3
    encoder_fourier_x_max: float = 1000.0
    encoder_fourier_num_freqs: int = 256
    encoder_fourier_input_scale: float = PEAK_MZ_MAX
    encoder_use_fourier_features: bool = True
    masked_token_loss_weight: float = 0.0
    mae_loss_weight: float = 1.0
    jepa_mae_loss_weight: float = 0.0
    distogram_loss_weight: float = 0.0
    latent_pair_loss_weight: float = 0.0
    distogram_mz_max: float = PEAK_MZ_MAX
    distogram_loss_chunk_size: int = 4096
    jepa_mae_mz_bin_size: float = 2.5
    jepa_mae_intensity_bin_size: float = 0.1
    jepa_mae_mz_max: float = PEAK_MZ_MAX
    jepa_mae_intensity_max: float = 1.0
    jepa_target_normalization: str = "none"
    latent_pair_target_normalization: str = "layernorm"
    jepa_target_layers: list[int] | tuple[int, ...] | None = None
    masked_token_input_mode: str = "latent_token"
    masked_mz_sentinel: float = -1.0
    masked_latent_predictor_num_layers: int = 2
    masked_latent_predictor_num_heads: int = 8
    jepa_num_target_blocks: int = 2
    norm_eps: float = 1e-5
    encoder_use_position_embedding: bool = True
    encoder_apply_final_norm: bool = True
    encoder_apply_final_pair_norm: bool = False
    pairformer_pair_dim: int | None = None
    pairformer_pair_num_heads: int | None = None
    pairformer_pair_feature_hidden_dim: int = 128
    pairformer_dropout: float = 0.0
    pairmixer_use_pair_bias_attention: bool = False
    pairformer_mz_scale: float = PEAK_MZ_MAX
    pairformer_precursor_mz_scale: float = PEAK_MZ_MAX
    pairformer_use_fourier_features: bool = True
    pairformer_fourier_num_freqs: int = 16
    pairformer_fourier_x_min: float = 1e-2
    pairformer_fourier_x_max: float = PEAK_MZ_MAX
    pairformer_relative_fourier_x_min: float = 1e-3
    pairformer_relative_fourier_x_max: float = 1.0
    predictor_apply_final_norm: bool = True
    num_peaks: int = 64
    predictor_dim: int | None = None
    target_projector_dim: int | None = None
    predictor_dropout: float = 0.0
    use_ema_teacher: bool = False
    frozen_teacher_config_path: str | None = None
    ema_teacher_momentum_start: float = 0.996
    ema_teacher_momentum_mid: float | None = None
    ema_teacher_momentum_final: float | None = None
    ema_teacher_schedule_peak_fraction: float = 0.35
    ema_teacher_schedule: str = "constant"
    activation_checkpoint_mode: str = "none"
    activation_checkpoint_every_n_layers: int = 1
    activation_checkpoint_modules: tuple[str, ...] = ("encoder", "predictor")
    autocast_dtype: str = "none"

    @classmethod
    def from_config(cls, config: Any) -> "PeakSetJEPASettings":
        values = _default_values(cls())
        _apply_derived_defaults(values, config)
        for name, cast in SETTING_CASTS.items():
            values[name] = cast(_config_get(config, name, values[name]))
        return cls(**values)

    @classmethod
    def create(
        cls,
        settings: "PeakSetJEPASettings | None" = None,
        **overrides: Any,
    ) -> "PeakSetJEPASettings":
        if settings is None:
            return cls(**overrides)
        if not overrides:
            return settings
        return replace(settings, **overrides)


def _default_values(settings: PeakSetJEPASettings) -> dict[str, Any]:
    return {field.name: getattr(settings, field.name) for field in fields(settings)}


def _apply_derived_defaults(values: dict[str, Any], config: Any) -> None:
    peak_mz_max = float(_config_get(config, "peak_mz_max", PEAK_MZ_MAX))
    precursor_mz_max = float(_config_get(config, "max_precursor_mz", peak_mz_max))
    values["encoder_fourier_input_scale"] = peak_mz_max
    values["jepa_mae_mz_max"] = peak_mz_max
    values["distogram_mz_max"] = peak_mz_max
    values["pairformer_mz_scale"] = peak_mz_max
    values["pairformer_precursor_mz_scale"] = precursor_mz_max
    values["pairformer_fourier_x_max"] = peak_mz_max


def _config_get(config: Any, key: str, default: Any) -> Any:
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


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
    "attention_mlp_multiple": float,
    "feature_mlp_hidden_dim": int,
    "encoder_fourier_mlp_hidden_dim": _optional_int,
    "encoder_fourier_mlp_num_layers": int,
    "encoder_fourier_x_min": float,
    "encoder_fourier_x_max": float,
    "encoder_fourier_num_freqs": int,
    "encoder_fourier_input_scale": float,
    "encoder_use_fourier_features": bool,
    "masked_token_loss_weight": float,
    "mae_loss_weight": float,
    "jepa_mae_loss_weight": float,
    "distogram_loss_weight": float,
    "latent_pair_loss_weight": float,
    "distogram_mz_max": float,
    "distogram_loss_chunk_size": int,
    "jepa_mae_mz_bin_size": float,
    "jepa_mae_intensity_bin_size": float,
    "jepa_mae_mz_max": float,
    "jepa_mae_intensity_max": float,
    "jepa_target_normalization": str,
    "latent_pair_target_normalization": str,
    "jepa_target_layers": _identity,
    "masked_token_input_mode": str,
    "masked_mz_sentinel": float,
    "masked_latent_predictor_num_layers": int,
    "masked_latent_predictor_num_heads": int,
    "jepa_num_target_blocks": int,
    "norm_eps": float,
    "encoder_use_position_embedding": bool,
    "encoder_apply_final_norm": bool,
    "encoder_apply_final_pair_norm": bool,
    "pairformer_pair_dim": _optional_int,
    "pairformer_pair_num_heads": _optional_int,
    "pairformer_pair_feature_hidden_dim": int,
    "pairformer_dropout": float,
    "pairmixer_use_pair_bias_attention": bool,
    "pairformer_mz_scale": float,
    "pairformer_precursor_mz_scale": float,
    "pairformer_use_fourier_features": bool,
    "pairformer_fourier_num_freqs": int,
    "pairformer_fourier_x_min": float,
    "pairformer_fourier_x_max": float,
    "pairformer_relative_fourier_x_min": float,
    "pairformer_relative_fourier_x_max": float,
    "predictor_apply_final_norm": bool,
    "num_peaks": int,
    "predictor_dim": _optional_int,
    "target_projector_dim": _optional_int,
    "predictor_dropout": float,
    "use_ema_teacher": bool,
    "frozen_teacher_config_path": lambda value: None if value is None else str(value),
    "ema_teacher_momentum_start": float,
    "ema_teacher_momentum_mid": _optional_float,
    "ema_teacher_momentum_final": _optional_float,
    "ema_teacher_schedule_peak_fraction": float,
    "ema_teacher_schedule": str,
    "activation_checkpoint_mode": str,
    "activation_checkpoint_every_n_layers": int,
    "activation_checkpoint_modules": tuple,
    "autocast_dtype": str,
}
