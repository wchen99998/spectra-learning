import copy
import math

import torch
from torch import nn

from spectra_learning.models.transformer import _build_norm
from spectra_learning.models.common import (
    _build_frozen_position_embedding,
    _build_non_causal_blocks,
)
from spectra_learning.models.encoder import PeakSetEncoder
from spectra_learning.models.losses import SIGReg, SlotwiseSIGReg
from spectra_learning.models.settings import PeakSetSIGRegSettings
from spectra_learning.models.temporal import CovariancePool, _build_temporal_decoder_blocks


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

SUPPORTED_TARGET_NORMALIZATIONS = {"none", "zscore"}
SUPPORTED_EMA_SCHEDULES = {"constant", "linear", "cosine", "slow-fast-slow"}


def configure_peak_set_sigreg(model: nn.Module, cfg: PeakSetSIGRegSettings) -> None:
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


def _configure_dimensions(model: nn.Module, cfg: PeakSetSIGRegSettings) -> None:
    model.model_dim = int(cfg.model_dim)
    model.predictor_dim = (
        int(cfg.predictor_dim) if cfg.predictor_dim is not None else model.model_dim
    )
    model.encoder_num_layers = int(cfg.encoder_num_layers)
    model.encoder_use_cls_token = bool(cfg.encoder_use_cls_token)
    model.use_precursor_token = bool(cfg.use_precursor_token)
    model.norm_type = str(cfg.norm_type).lower()
    model.norm_eps = float(cfg.norm_eps)
    model.temporal_predictor_num_layers = int(cfg.temporal_predictor_num_layers)
    model.predictor_num_register_tokens = int(cfg.predictor_num_register_tokens)
    model.covariance_pooling_dim = int(cfg.covariance_pooling_dim)
    model.train_covariance_pooling = model.covariance_pooling_dim > 0


def _configure_targets(model: nn.Module, cfg: PeakSetSIGRegSettings) -> None:
    model.jepa_num_target_blocks = int(cfg.jepa_num_target_blocks)
    if model.jepa_num_target_blocks < 1:
        raise ValueError("jepa_num_target_blocks must be >= 1")
    model.jepa_target_layers = (
        [model.encoder_num_layers]
        if cfg.jepa_target_layers is None
        else [int(layer_idx) for layer_idx in cfg.jepa_target_layers]
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
        else int(cfg.target_projector_dim)
    )
    model.use_target_projector = raw_target_projector_dim >= 0
    model.target_projector_dim = (
        raw_target_projector_dim if model.use_target_projector else model.jepa_target_dim
    )
    model.jepa_target_normalization = str(cfg.jepa_target_normalization).lower()
    if model.jepa_target_normalization not in SUPPORTED_TARGET_NORMALIZATIONS:
        raise ValueError("jepa_target_normalization must be one of ('none', 'zscore')")


def _configure_losses(model: nn.Module, cfg: PeakSetSIGRegSettings) -> None:
    model.representation_regularizer = _canonical_regularizer(
        cfg.representation_regularizer
    )
    model.sigreg_lambda = float(cfg.sigreg_lambda)
    model.sigreg_precursor_scale = float(cfg.sigreg_precursor_scale)
    model.masked_token_loss_weight = float(cfg.masked_token_loss_weight)
    model.jepa_mae_loss_weight = float(cfg.jepa_mae_loss_weight)
    model.jepa_mae_mz_bin_size = float(cfg.jepa_mae_mz_bin_size)
    model.jepa_mae_intensity_bin_size = float(cfg.jepa_mae_intensity_bin_size)
    model.jepa_mae_mz_max = float(cfg.jepa_mae_mz_max)
    model.jepa_mae_intensity_max = float(cfg.jepa_mae_intensity_max)
    model.jepa_mae_num_mz_bins = int(
        math.ceil(model.jepa_mae_mz_max / model.jepa_mae_mz_bin_size)
    )
    model.jepa_mae_num_intensity_bins = int(
        math.ceil(model.jepa_mae_intensity_max / model.jepa_mae_intensity_bin_size)
    )


def _build_encoder(model: nn.Module, cfg: PeakSetSIGRegSettings) -> None:
    num_peak_tokens = int(cfg.num_peaks) + int(model.use_precursor_token)
    model.num_peak_tokens = num_peak_tokens
    model.encoder = PeakSetEncoder(
        model_dim=model.model_dim,
        num_layers=model.encoder_num_layers,
        num_heads=int(cfg.encoder_num_heads),
        num_kv_heads=cfg.encoder_num_kv_heads,
        attention_mlp_multiple=float(cfg.attention_mlp_multiple),
        feature_mlp_hidden_dim=int(cfg.feature_mlp_hidden_dim),
        fourier_mlp_hidden_dim=cfg.encoder_fourier_mlp_hidden_dim,
        fourier_mlp_num_layers=int(cfg.encoder_fourier_mlp_num_layers),
        fourier_strategy=str(cfg.encoder_fourier_strategy),
        fourier_x_min=float(cfg.encoder_fourier_x_min),
        fourier_x_max=float(cfg.encoder_fourier_x_max),
        fourier_funcs=str(cfg.encoder_fourier_funcs),
        fourier_num_freqs=int(cfg.encoder_fourier_num_freqs),
        fourier_sigma=float(cfg.encoder_fourier_sigma),
        fourier_trainable=bool(cfg.encoder_fourier_trainable),
        fourier_input_scale=float(cfg.encoder_fourier_input_scale),
        qk_norm=bool(cfg.encoder_qk_norm),
        norm_type=model.norm_type,
        norm_eps=model.norm_eps,
        use_position_embedding=bool(cfg.encoder_use_position_embedding),
        apply_final_norm=bool(cfg.encoder_apply_final_norm),
        num_peaks=num_peak_tokens,
        use_cls_token=model.encoder_use_cls_token,
        num_register_tokens=int(cfg.encoder_num_register_tokens),
        use_precursor_token=model.use_precursor_token,
        spectral_bias_relative_kind=str(cfg.spectral_bias_relative_kind),
        spectral_bias_use_precursor=bool(cfg.spectral_bias_use_precursor),
        spectral_bias_use_intensity=bool(cfg.spectral_bias_use_intensity),
        spectral_bias_num_freqs=int(cfg.spectral_bias_num_freqs),
        spectral_bias_fourier_strategy=str(cfg.spectral_bias_fourier_strategy),
        spectral_bias_fourier_x_min=float(cfg.spectral_bias_fourier_x_min),
        spectral_bias_fourier_x_max=float(cfg.spectral_bias_fourier_x_max),
        spectral_bias_fourier_sigma=float(cfg.spectral_bias_fourier_sigma),
        spectral_bias_fourier_trainable=bool(cfg.spectral_bias_fourier_trainable),
        spectral_bias_mass_scale=float(cfg.spectral_bias_mass_scale),
        spectral_bias_precursor_scale=float(cfg.spectral_bias_precursor_scale),
        spectral_bias_rbf_num_basis=int(cfg.spectral_bias_rbf_num_basis),
        spectral_bias_rbf_delta_min=float(cfg.spectral_bias_rbf_delta_min),
        spectral_bias_rbf_delta_max=float(cfg.spectral_bias_rbf_delta_max),
        spectral_bias_rbf_use_absolute_delta=bool(
            cfg.spectral_bias_rbf_use_absolute_delta
        ),
        spectral_bias_intensity_hidden_dim=int(cfg.spectral_bias_intensity_hidden_dim),
        spectral_bias_init_std=float(cfg.spectral_bias_init_std),
        spectral_bias_clip=cfg.spectral_bias_clip,
    )


def _build_teacher(model: nn.Module, cfg: PeakSetSIGRegSettings) -> None:
    model.use_ema_teacher = bool(cfg.use_ema_teacher)
    model.ema_teacher_momentum_start = float(cfg.ema_teacher_momentum_start)
    model.ema_teacher_momentum_mid = (
        float(cfg.ema_teacher_momentum_mid)
        if cfg.ema_teacher_momentum_mid is not None
        else model.ema_teacher_momentum_start
    )
    model.ema_teacher_momentum_final = (
        float(cfg.ema_teacher_momentum_final)
        if cfg.ema_teacher_momentum_final is not None
        else model.ema_teacher_momentum_start
    )
    model.ema_teacher_schedule_peak_fraction = float(
        cfg.ema_teacher_schedule_peak_fraction
    )
    model.ema_teacher_schedule = str(cfg.ema_teacher_schedule).lower()
    if model.ema_teacher_schedule not in SUPPORTED_EMA_SCHEDULES:
        raise ValueError(
            "ema_teacher_schedule must be one of "
            "('constant', 'linear', 'cosine', 'slow-fast-slow')"
        )
    if model.use_ema_teacher:
        model.teacher_encoder = copy.deepcopy(model.encoder)
        model.teacher_encoder.requires_grad_(False)
    else:
        model.teacher_encoder = None


def _build_predictor(model: nn.Module, cfg: PeakSetSIGRegSettings) -> None:
    model.latent_mask_token = nn.Parameter(torch.empty(model.model_dim))
    nn.init.normal_(model.latent_mask_token, std=0.02)

    if model.predictor_dim != model.model_dim:
        model.encoder_to_predictor_proj = nn.Linear(
            model.model_dim,
            model.predictor_dim,
            bias=False,
        )
        nn.init.xavier_normal_(model.encoder_to_predictor_proj.weight)
    else:
        model.encoder_to_predictor_proj = nn.Identity()

    model.predictor_position_embedding = _build_frozen_position_embedding(
        model.num_peak_tokens,
        model.model_dim,
    )
    if model.predictor_num_register_tokens > 0:
        model.predictor_register_tokens = nn.Parameter(
            torch.empty(model.predictor_num_register_tokens, model.model_dim)
        )
        nn.init.trunc_normal_(model.predictor_register_tokens, std=0.02)
    else:
        model.predictor_register_tokens = None

    model.masked_latent_predictor = _build_non_causal_blocks(
        dim=model.predictor_dim,
        num_layers=int(cfg.masked_latent_predictor_num_layers),
        num_heads=int(cfg.masked_latent_predictor_num_heads),
        num_kv_heads=None,
        attention_mlp_multiple=float(cfg.attention_mlp_multiple),
        norm_eps=model.norm_eps,
        qk_norm=bool(cfg.encoder_qk_norm),
        norm_type=model.norm_type,
        dropout=float(cfg.predictor_dropout),
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
    model.masked_latent_readout = nn.Linear(
        model.predictor_dim,
        model.jepa_target_dim,
    )
    nn.init.xavier_normal_(model.masked_latent_readout.weight)
    nn.init.zeros_(model.masked_latent_readout.bias)


def _build_target_projectors(model: nn.Module, cfg: PeakSetSIGRegSettings) -> None:
    if model.use_target_projector:
        model.target_projector = nn.Sequential(
            nn.Linear(model.jepa_target_dim, model.jepa_target_dim),
            nn.GELU(),
            nn.Linear(model.jepa_target_dim, model.target_projector_dim),
        )
        for layer in model.target_projector:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
    else:
        model.target_projector = nn.Identity()

    if model.use_ema_teacher:
        model.teacher_target_projector = copy.deepcopy(model.target_projector)
        model.teacher_target_projector.requires_grad_(False)
    else:
        model.teacher_target_projector = None


def _build_jepa_mae_heads(model: nn.Module) -> None:
    if model.jepa_mae_loss_weight <= 0:
        model.jepa_mae_mz_head = None
        model.jepa_mae_intensity_head = None
        return

    model.jepa_mae_mz_head = nn.Linear(
        model.target_projector_dim,
        model.jepa_mae_num_mz_bins,
    )
    model.jepa_mae_intensity_head = nn.Linear(
        model.target_projector_dim,
        model.jepa_mae_num_intensity_bins,
    )
    nn.init.xavier_normal_(model.jepa_mae_mz_head.weight)
    nn.init.zeros_(model.jepa_mae_mz_head.bias)
    nn.init.xavier_normal_(model.jepa_mae_intensity_head.weight)
    nn.init.zeros_(model.jepa_mae_intensity_head.bias)


def _build_regularizer(model: nn.Module, cfg: PeakSetSIGRegSettings) -> None:
    sigreg_cls = (
        SlotwiseSIGReg
        if model.representation_regularizer in SLOTWISE_REGULARIZERS
        else SIGReg
    )
    model.sigreg = sigreg_cls(num_slices=int(cfg.sigreg_num_slices))
    if model.train_covariance_pooling:
        model.covariance_pooler = CovariancePool(
            input_dim=model.model_dim,
            compressed_dim=model.covariance_pooling_dim,
        )


def _build_temporal_predictor(model: nn.Module, cfg: PeakSetSIGRegSettings) -> None:
    if model.temporal_predictor_num_layers <= 0:
        return

    model.temporal_predictor = _build_temporal_decoder_blocks(
        dim=model.model_dim,
        num_layers=model.temporal_predictor_num_layers,
        num_heads=int(cfg.masked_latent_predictor_num_heads),
        num_kv_heads=None,
        attention_mlp_multiple=float(cfg.attention_mlp_multiple),
        norm_eps=model.norm_eps,
        qk_norm=bool(cfg.encoder_qk_norm),
        norm_type=model.norm_type,
    )
    model.temporal_rt_proj = nn.Sequential(
        nn.Linear(1, model.model_dim),
        nn.SiLU(),
        nn.Linear(model.model_dim, model.model_dim),
    )
    for layer in model.temporal_rt_proj:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
    model.temporal_query_token = nn.Parameter(torch.empty(model.model_dim))
    nn.init.trunc_normal_(model.temporal_query_token, std=0.02)


def _canonical_regularizer(value: str) -> str:
    regularizer = str(value).lower()
    aliases = {
        "sigreg": "sigreg-enc",
        "slog-sigreg-pred": "slot-sigreg-pred",
        "slog-sigreg-proj": "slot-sigreg-proj",
    }
    regularizer = aliases.get(regularizer, regularizer)
    if regularizer not in SUPPORTED_REGULARIZERS:
        raise ValueError(f"Unsupported regularizer: {regularizer!r}")
    return regularizer
