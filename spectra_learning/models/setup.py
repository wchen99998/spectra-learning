from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING

import torch
from torch import nn

from spectra_learning.models.transformer import _build_norm
from spectra_learning.models.encoder import PeakSetEncoder
from spectra_learning.models.losses import SIGReg, SlotwiseSIGReg
from spectra_learning.models.settings import PeakSetSIGRegSettings
from spectra_learning.models.temporal import (
    _build_cross_attention_decoder_blocks,
    _build_temporal_decoder_blocks,
)

if TYPE_CHECKING:
    from spectra_learning.models.model import PeakSetSIGReg


SLOTWISE_REGULARIZERS = {
    "slot-sigreg-enc",
    "slot-sigreg-pred",
    "slot-sigreg-proj",
    "slot-sigreg-enc-pred",
}

SUPPORTED_REGULARIZERS = {
    "none",
    "",
    "sigreg-enc",
    "sigreg-pred",
    "sigreg-proj",
    "sigreg-enc-pred",
    *SLOTWISE_REGULARIZERS,
}

SUPPORTED_TRAINING_MODES = {"jepa", "mae", "mae_teacher_jepa"}
SUPPORTED_TARGET_NORMALIZATIONS = {"none", "zscore"}
SUPPORTED_EMA_SCHEDULES = {"constant", "linear", "cosine", "slow-fast-slow"}
SUPPORTED_MASKED_TOKEN_INPUT_MODES = {"latent_token", "mz_sentinel"}


def configure_peak_set_sigreg(model: PeakSetSIGReg, cfg: PeakSetSIGRegSettings) -> None:
    _configure_dimensions(model, cfg)
    _configure_targets(model, cfg)
    _configure_losses(model, cfg)
    _build_encoder(model, cfg)
    _build_teacher(model, cfg)
    _build_predictor(model, cfg)
    _build_target_projectors(model, cfg)
    _build_jepa_mae_heads(model)
    _build_regularizer(model, cfg)
    _build_temporal_predictor(model, cfg)


def _configure_dimensions(model: PeakSetSIGReg, cfg: PeakSetSIGRegSettings) -> None:
    model.training_mode = cfg.training_mode.lower()
    if model.training_mode not in SUPPORTED_TRAINING_MODES:
        raise ValueError("training_mode must be one of ('jepa', 'mae')")
    model.model_dim = cfg.model_dim
    model.predictor_dim = (
        cfg.predictor_dim if cfg.predictor_dim is not None else model.model_dim
    )
    model.encoder_num_layers = cfg.encoder_num_layers
    model.encoder_use_cls_token = cfg.encoder_use_cls_token
    model.use_precursor_token = cfg.use_precursor_token
    model.norm_type = cfg.norm_type.lower()
    model.norm_eps = cfg.norm_eps
    model.temporal_predictor_num_layers = cfg.temporal_predictor_num_layers
    model.predictor_num_register_tokens = cfg.predictor_num_register_tokens
    model.covariance_pooling_dim = cfg.covariance_pooling_dim
    model.train_covariance_pooling = (
        model.covariance_pooling_dim > 0 and cfg.train_covariance_pooling
    )
    model.covariance_pooling_loss_weight = cfg.covariance_pooling_loss_weight


def _configure_targets(model: PeakSetSIGReg, cfg: PeakSetSIGRegSettings) -> None:
    model.jepa_num_target_blocks = cfg.jepa_num_target_blocks
    if model.jepa_num_target_blocks < 1:
        raise ValueError("jepa_num_target_blocks must be >= 1")
    model.jepa_target_layers = (
        [model.encoder_num_layers]
        if cfg.jepa_target_layers is None
        else [layer_idx for layer_idx in cfg.jepa_target_layers]
    )
    if not model.jepa_target_layers:
        raise ValueError("jepa_target_layers must not be empty")
    if (
        min(model.jepa_target_layers) < 1
        or max(model.jepa_target_layers) > model.encoder_num_layers
    ):
        raise ValueError("jepa_target_layers must be within encoder depth")

    model.num_jepa_target_layers = len(model.jepa_target_layers)
    model.jepa_target_dim = model.num_jepa_target_layers * model.model_dim
    raw_target_projector_dim = (
        model.model_dim
        if cfg.target_projector_dim is None
        else cfg.target_projector_dim
    )
    model.use_target_projector = raw_target_projector_dim >= 0
    model.target_projector_dim = (
        raw_target_projector_dim if model.use_target_projector else model.jepa_target_dim
    )
    model.jepa_target_normalization = cfg.jepa_target_normalization.lower()
    if model.jepa_target_normalization not in SUPPORTED_TARGET_NORMALIZATIONS:
        raise ValueError("jepa_target_normalization must be one of ('none', 'zscore')")
    model.masked_token_input_mode = cfg.masked_token_input_mode.lower()
    if model.masked_token_input_mode not in SUPPORTED_MASKED_TOKEN_INPUT_MODES:
        raise ValueError(
            "masked_token_input_mode must be one of ('latent_token', 'mz_sentinel')"
        )
    model.masked_mz_sentinel = cfg.masked_mz_sentinel


def _configure_losses(model: PeakSetSIGReg, cfg: PeakSetSIGRegSettings) -> None:
    model.representation_regularizer = _canonical_regularizer(
        cfg.representation_regularizer
    )
    model.sigreg_lambda = cfg.sigreg_lambda
    model.sigreg_precursor_scale = cfg.sigreg_precursor_scale
    model.mae_loss_weight = cfg.mae_loss_weight
    model.masked_token_loss_weight = (
        0.0 if model.training_mode == "mae" else cfg.masked_token_loss_weight
    )
    model.jepa_mae_loss_weight = (
        0.0 if model.training_mode == "mae" else cfg.jepa_mae_loss_weight
    )
    model.jepa_mae_mz_bin_size = cfg.jepa_mae_mz_bin_size
    model.jepa_mae_intensity_bin_size = cfg.jepa_mae_intensity_bin_size
    model.jepa_mae_mz_max = cfg.jepa_mae_mz_max
    model.jepa_mae_intensity_max = cfg.jepa_mae_intensity_max
    model.jepa_mae_num_mz_bins = math.ceil(model.jepa_mae_mz_max / model.jepa_mae_mz_bin_size)
    model.jepa_mae_num_intensity_bins = math.ceil(model.jepa_mae_intensity_max / model.jepa_mae_intensity_bin_size)


def _build_encoder(model: PeakSetSIGReg, cfg: PeakSetSIGRegSettings) -> None:
    num_peak_tokens = cfg.num_peaks + int(model.use_precursor_token)
    model.num_peak_tokens = num_peak_tokens
    model.encoder = PeakSetEncoder(
        model_dim=model.model_dim,
        num_layers=model.encoder_num_layers,
        num_heads=cfg.encoder_num_heads,
        num_kv_heads=cfg.encoder_num_kv_heads,
        attention_mlp_multiple=cfg.attention_mlp_multiple,
        feature_mlp_hidden_dim=cfg.feature_mlp_hidden_dim,
        fourier_mlp_hidden_dim=cfg.encoder_fourier_mlp_hidden_dim,
        fourier_mlp_num_layers=cfg.encoder_fourier_mlp_num_layers,
        fourier_strategy=cfg.encoder_fourier_strategy,
        fourier_x_min=cfg.encoder_fourier_x_min,
        fourier_x_max=cfg.encoder_fourier_x_max,
        fourier_funcs=cfg.encoder_fourier_funcs,
        fourier_num_freqs=cfg.encoder_fourier_num_freqs,
        fourier_sigma=cfg.encoder_fourier_sigma,
        fourier_trainable=cfg.encoder_fourier_trainable,
        fourier_input_scale=cfg.encoder_fourier_input_scale,
        use_fourier_features=cfg.encoder_use_fourier_features,
        qk_norm=cfg.encoder_qk_norm,
        norm_type=model.norm_type,
        norm_eps=model.norm_eps,
        use_position_embedding=cfg.encoder_use_position_embedding,
        apply_final_norm=cfg.encoder_apply_final_norm,
        num_peaks=num_peak_tokens,
        use_cls_token=model.encoder_use_cls_token,
        num_register_tokens=cfg.encoder_num_register_tokens,
        use_precursor_token=model.use_precursor_token,
        spectral_bias_relative_kind=cfg.spectral_bias_relative_kind,
        spectral_bias_use_precursor=cfg.spectral_bias_use_precursor,
        spectral_bias_use_intensity=cfg.spectral_bias_use_intensity,
        spectral_bias_num_freqs=cfg.spectral_bias_num_freqs,
        spectral_bias_fourier_strategy=cfg.spectral_bias_fourier_strategy,
        spectral_bias_fourier_x_min=cfg.spectral_bias_fourier_x_min,
        spectral_bias_fourier_x_max=cfg.spectral_bias_fourier_x_max,
        spectral_bias_fourier_sigma=cfg.spectral_bias_fourier_sigma,
        spectral_bias_fourier_trainable=cfg.spectral_bias_fourier_trainable,
        spectral_bias_mass_scale=cfg.spectral_bias_mass_scale,
        spectral_bias_precursor_scale=cfg.spectral_bias_precursor_scale,
        spectral_bias_rbf_num_basis=cfg.spectral_bias_rbf_num_basis,
        spectral_bias_rbf_delta_min=cfg.spectral_bias_rbf_delta_min,
        spectral_bias_rbf_delta_max=cfg.spectral_bias_rbf_delta_max,
        spectral_bias_rbf_use_absolute_delta=cfg.spectral_bias_rbf_use_absolute_delta,
        spectral_bias_intensity_hidden_dim=cfg.spectral_bias_intensity_hidden_dim,
        spectral_bias_init_std=cfg.spectral_bias_init_std,
        spectral_bias_clip=cfg.spectral_bias_clip,
    )


def _build_teacher(model: PeakSetSIGReg, cfg: PeakSetSIGRegSettings) -> None:
    model.use_frozen_teacher = model.training_mode == "mae_teacher_jepa"
    model.use_ema_teacher = cfg.use_ema_teacher and model.training_mode == "jepa"
    model.ema_teacher_momentum_start = cfg.ema_teacher_momentum_start
    model.ema_teacher_momentum_mid = (
        cfg.ema_teacher_momentum_mid
        if cfg.ema_teacher_momentum_mid is not None
        else model.ema_teacher_momentum_start
    )
    model.ema_teacher_momentum_final = (
        cfg.ema_teacher_momentum_final
        if cfg.ema_teacher_momentum_final is not None
        else model.ema_teacher_momentum_start
    )
    model.ema_teacher_schedule_peak_fraction = cfg.ema_teacher_schedule_peak_fraction
    model.ema_teacher_schedule = cfg.ema_teacher_schedule.lower()
    if model.ema_teacher_schedule not in SUPPORTED_EMA_SCHEDULES:
        raise ValueError(
            "ema_teacher_schedule must be one of "
            "('constant', 'linear', 'cosine', 'slow-fast-slow')"
        )
    if model.use_ema_teacher or model.use_frozen_teacher:
        model.teacher_encoder = copy.deepcopy(model.encoder)
        model.teacher_encoder.requires_grad_(False)
    else:
        model.teacher_encoder = None


def _build_predictor(model: PeakSetSIGReg, cfg: PeakSetSIGRegSettings) -> None:
    if model.predictor_dim != model.model_dim:
        encoder_to_predictor_proj = nn.Linear(
            model.model_dim,
            model.predictor_dim,
            bias=False,
        )
        nn.init.xavier_normal_(encoder_to_predictor_proj.weight)
        model.encoder_to_predictor_proj = encoder_to_predictor_proj
    else:
        model.encoder_to_predictor_proj = nn.Identity()

    predictor_slot_embedding = nn.Embedding(
        model.num_peak_tokens,
        model.predictor_dim,
    )
    nn.init.trunc_normal_(predictor_slot_embedding.weight, std=0.02)
    model.predictor_slot_embedding = predictor_slot_embedding
    if model.predictor_num_register_tokens > 0:
        model.predictor_register_tokens = nn.Parameter(
            torch.empty(model.predictor_num_register_tokens, model.predictor_dim)
        )
        nn.init.trunc_normal_(model.predictor_register_tokens, std=0.02)
    else:
        model.predictor_register_tokens = None

    model.masked_latent_predictor = _build_cross_attention_decoder_blocks(
        dim=model.predictor_dim,
        num_layers=cfg.masked_latent_predictor_num_layers,
        num_heads=cfg.masked_latent_predictor_num_heads,
        num_kv_heads=None,
        attention_mlp_multiple=cfg.attention_mlp_multiple,
        norm_eps=model.norm_eps,
        qk_norm=cfg.encoder_qk_norm,
        norm_type=model.norm_type,
        dropout=cfg.predictor_dropout,
    )
    model.predictor_final_norm = (
        _build_norm(
            model.predictor_dim,
            eps=model.norm_eps,
            norm_type=model.norm_type,
            affine=False,
        )
        if cfg.predictor_apply_final_norm
        else nn.Identity()
    )
    masked_latent_readout = nn.Linear(
        model.predictor_dim,
        model.jepa_target_dim,
    )
    nn.init.xavier_normal_(masked_latent_readout.weight)
    nn.init.zeros_(masked_latent_readout.bias)
    model.masked_latent_readout = masked_latent_readout


def _build_target_projectors(model: PeakSetSIGReg, cfg: PeakSetSIGRegSettings) -> None:
    if model.use_target_projector:
        target_projector = nn.Sequential(
            nn.Linear(model.jepa_target_dim, model.jepa_target_dim),
            nn.GELU(),
            nn.Linear(model.jepa_target_dim, model.target_projector_dim),
        )
        for layer in target_projector:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        model.target_projector = target_projector
    else:
        model.target_projector = nn.Identity()

    if model.use_ema_teacher or model.use_frozen_teacher:
        model.teacher_target_projector = copy.deepcopy(model.target_projector)
        model.teacher_target_projector.requires_grad_(False)
    else:
        model.teacher_target_projector = None


def _build_jepa_mae_heads(model: PeakSetSIGReg) -> None:
    if model.jepa_mae_loss_weight <= 0 and model.training_mode != "mae":
        model.jepa_mae_mz_head = None
        model.jepa_mae_intensity_head = None
        return

    jepa_mae_mz_head = nn.Linear(
        model.target_projector_dim,
        model.jepa_mae_num_mz_bins,
    )
    jepa_mae_intensity_head = nn.Linear(
        model.target_projector_dim,
        model.jepa_mae_num_intensity_bins,
    )
    nn.init.xavier_normal_(jepa_mae_mz_head.weight)
    nn.init.zeros_(jepa_mae_mz_head.bias)
    nn.init.xavier_normal_(jepa_mae_intensity_head.weight)
    nn.init.zeros_(jepa_mae_intensity_head.bias)
    model.jepa_mae_mz_head = jepa_mae_mz_head
    model.jepa_mae_intensity_head = jepa_mae_intensity_head


def _build_regularizer(model: PeakSetSIGReg, cfg: PeakSetSIGRegSettings) -> None:
    sigreg_cls = (
        SlotwiseSIGReg
        if model.representation_regularizer in SLOTWISE_REGULARIZERS
        else SIGReg
    )
    model.sigreg = sigreg_cls(num_slices=cfg.sigreg_num_slices)


def _build_temporal_predictor(model: PeakSetSIGReg, cfg: PeakSetSIGRegSettings) -> None:
    if model.temporal_predictor_num_layers <= 0:
        return

    model.temporal_predictor = _build_temporal_decoder_blocks(
        dim=model.model_dim,
        num_layers=model.temporal_predictor_num_layers,
        num_heads=cfg.masked_latent_predictor_num_heads,
        num_kv_heads=None,
        attention_mlp_multiple=cfg.attention_mlp_multiple,
        norm_eps=model.norm_eps,
        qk_norm=cfg.encoder_qk_norm,
        norm_type=model.norm_type,
    )
    temporal_rt_proj = nn.Sequential(
        nn.Linear(1, model.model_dim),
        nn.SiLU(),
        nn.Linear(model.model_dim, model.model_dim),
    )
    for layer in temporal_rt_proj:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
    model.temporal_rt_proj = temporal_rt_proj
    model.temporal_query_token = nn.Parameter(torch.empty(model.model_dim))
    nn.init.trunc_normal_(model.temporal_query_token, std=0.02)
    temporal_slot_embedding = nn.Embedding(
        model.num_peak_tokens,
        model.model_dim,
    )
    nn.init.trunc_normal_(temporal_slot_embedding.weight, std=0.02)
    model.temporal_slot_embedding = temporal_slot_embedding


def _canonical_regularizer(value: str) -> str:
    regularizer = value.lower()
    aliases = {
        "sigreg": "sigreg-enc",
        "slog-sigreg-pred": "slot-sigreg-pred",
        "slog-sigreg-proj": "slot-sigreg-proj",
    }
    regularizer = aliases.get(regularizer, regularizer)
    if regularizer not in SUPPORTED_REGULARIZERS:
        raise ValueError(f"Unsupported regularizer: {regularizer!r}")
    return regularizer
