from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import Any

from spectra_learning.config import load_config
from spectra_learning.data.spectra import DEFAULT_NUM_PEAKS, PEAK_MZ_MAX
from spectra_learning.models.fastmixer_capacity import (
    resolve_pairmixer_fast_encoder_max_visible_tokens,
    resolve_pairmixer_fast_max_visible_tokens,
)


REMOVED_SETTING_KEYS = (
    "encoder_discrete_mz_bin_size",
    "encoder_discrete_mz_coarse_bin_size",
    "encoder_discrete_mz_embedding_dim",
    "encoder_fourier_input_scale",
    "encoder_mz_scale",
    "encoder_use_fourier_features",
    "distogram_mz_max",
    "jepa_mae_mz_max",
    "mae_context_encoder_pack_tokens",
    "mae_context_encoder_pack_token_choices",
    "peak_mz_max",
    "pairmixer_encoder_projection_kernel",
    "pairmixer_fourier_x_max",
    "pairmixer_mz_scale",
    "pairmixer_precursor_mz_scale",
    "pairmixer_predictor_projection_kernel",
    "pairmixer_encoder_projection_kernel_schedule",
    "pairmixer_predictor_projection_kernel_schedule",
)


def ema_teacher_momentum_at(
    *,
    schedule: str,
    start: float,
    mid: float,
    final: float,
    peak_fraction: float,
    step: int,
    total_steps: int,
) -> float:
    if schedule == "constant":
        return start
    progress = min(1.0, max(0.0, float(step) / float(max(1, total_steps))))
    if schedule == "slow-fast-slow":
        peak = min(1.0, max(1e-6, peak_fraction))
        if progress <= peak:
            phase = progress / peak
            eased = 0.5 - 0.5 * math.cos(math.pi * phase)
            return start + eased * (mid - start)
        phase = (progress - peak) / max(1e-6, 1.0 - peak)
        eased = 0.5 - 0.5 * math.cos(math.pi * phase)
        return mid + eased * (final - mid)
    if schedule == "cosine":
        progress = 0.5 - 0.5 * math.cos(math.pi * progress)
    return start + progress * (final - start)


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
    encoder_fourier_x_max: float = PEAK_MZ_MAX
    encoder_fourier_num_freqs: int = 256
    encoder_mz_scale: float = PEAK_MZ_MAX
    encoder_mz_embedding: str = "fourier"
    encoder_mz_token_bin_size: float = 0.02
    encoder_mz_token_embedding_dim: int = 77
    masked_token_loss_weight: float = 0.0
    mae_loss_weight: float = 1.0
    jepa_mae_loss_weight: float = 0.0
    distogram_loss_weight: float = 0.0
    latent_pair_loss_weight: float = 0.0
    distogram_mz_max: float = PEAK_MZ_MAX
    jepa_mae_mz_bin_size: float = 2.5
    jepa_mae_intensity_bin_size: float = 0.1
    mae_intensity_loss_weight: float = 1.0
    jepa_mae_mz_max: float = PEAK_MZ_MAX
    jepa_mae_intensity_max: float = 1.0
    jepa_target_normalization: str = "none"
    latent_pair_target_normalization: str = "layernorm"
    masked_token_input_mode: str = "latent_token"
    masked_mz_sentinel: float = -1.0
    masked_latent_predictor_num_layers: int = 2
    masked_latent_predictor_num_heads: int = 8
    jepa_num_target_blocks: int = 2
    norm_eps: float = 1e-5
    encoder_use_position_embedding: bool = True
    encoder_apply_final_norm: bool = True
    encoder_apply_final_pair_norm: bool = False
    pairmixer_block_type: str = "dense"
    pairmixer_transition_type: str = "swiglu"
    pairmixer_pair_dim: int | None = None
    pairmixer_pair_feature_hidden_dim: int = 128
    pairmixer_dropout: float = 0.0
    pairmixer_use_pair_bias: bool = True
    pairmixer_mz_scale: float = PEAK_MZ_MAX
    pairmixer_precursor_mz_scale: float = PEAK_MZ_MAX
    pairmixer_use_fourier_features: bool = True
    pairmixer_fourier_num_freqs: int = 16
    pairmixer_fourier_x_min: float = 1e-2
    pairmixer_fourier_x_max: float = PEAK_MZ_MAX
    pairmixer_relative_fourier_x_min: float = 1e-3
    pairmixer_relative_fourier_x_max: float = 1.0
    pairmixer_fast_max_visible_tokens: int | None = None
    pairmixer_fast_encoder_max_visible_tokens: int | None = None
    predictor_apply_final_norm: bool = True
    num_peaks: int = DEFAULT_NUM_PEAKS
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
        for key in REMOVED_SETTING_KEYS:
            if key in config:
                raise ValueError(
                    f"{key} has been removed from PeakSetJEPASettings; remove it "
                    "from experiment configs."
                )
        values = _default_values(cls())
        _apply_derived_defaults(values, config)
        values = {
            name: config.get(name, default) for name, default in values.items()
        }
        values["pairmixer_fast_max_visible_tokens"] = (
            resolve_pairmixer_fast_max_visible_tokens(config)
        )
        values["pairmixer_fast_encoder_max_visible_tokens"] = (
            resolve_pairmixer_fast_encoder_max_visible_tokens(config)
        )
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


def load_frozen_teacher_settings(
    settings: PeakSetJEPASettings,
) -> PeakSetJEPASettings | None:
    if settings.training_mode.lower() != "mae_teacher_jepa":
        return None
    if settings.frozen_teacher_config_path is None:
        return None
    return PeakSetJEPASettings.from_config(
        load_config(settings.frozen_teacher_config_path)
    )


def _default_values(settings: PeakSetJEPASettings) -> dict[str, Any]:
    return {field.name: getattr(settings, field.name) for field in fields(settings)}


def _apply_derived_defaults(values: dict[str, Any], config: Any) -> None:
    precursor_mz_max = config.get("max_precursor_mz", PEAK_MZ_MAX)
    values["encoder_mz_scale"] = PEAK_MZ_MAX
    values["jepa_mae_mz_max"] = PEAK_MZ_MAX
    values["distogram_mz_max"] = PEAK_MZ_MAX
    values["pairmixer_mz_scale"] = PEAK_MZ_MAX
    values["pairmixer_precursor_mz_scale"] = precursor_mz_max
    values["pairmixer_fourier_x_max"] = PEAK_MZ_MAX
