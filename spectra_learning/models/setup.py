from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING

import torch
from torch import nn

from spectra_learning.config import load_config
from spectra_learning.models.common import (
    _build_frozen_2d_position_embedding,
    _build_frozen_position_embedding,
)
from spectra_learning.models.transformer import _build_norm
from spectra_learning.models.encoder import PeakSetEncoder
from spectra_learning.models.pairformer import PairMixerBlock
from spectra_learning.models.peak_features import PeakFeatureEmbedder
from spectra_learning.models.settings import PeakSetJEPASettings

if TYPE_CHECKING:
    from spectra_learning.models.model import PeakSetJEPA


SUPPORTED_TRAINING_MODES = {"jepa", "mae", "mae_teacher_jepa", "contrastive"}
SUPPORTED_TARGET_NORMALIZATIONS = {"none", "zscore"}
SUPPORTED_LATENT_PAIR_TARGET_NORMALIZATIONS = {"none", "layernorm"}
SUPPORTED_EMA_SCHEDULES = {"constant", "linear", "cosine", "slow-fast-slow"}
SUPPORTED_MASKED_TOKEN_INPUT_MODES = {"latent_token", "mz_sentinel"}


def configure_peak_set_model(model: PeakSetJEPA, cfg: PeakSetJEPASettings) -> None:
    frozen_teacher_cfg = _load_frozen_teacher_settings(cfg)
    _configure_dimensions(model, cfg)
    _configure_targets(model, cfg, frozen_teacher_cfg)
    _configure_losses(model, cfg)
    _build_encoder(model, cfg)
    _build_teacher(model, cfg, frozen_teacher_cfg)
    _build_predictor(model, cfg)
    _build_target_projectors(model, cfg)
    _build_jepa_mae_heads(model)
    _build_distogram_head(model)


def _load_frozen_teacher_settings(
    cfg: PeakSetJEPASettings,
) -> PeakSetJEPASettings | None:
    if cfg.training_mode.lower() != "mae_teacher_jepa":
        return None
    if cfg.frozen_teacher_config_path is None:
        return None
    return PeakSetJEPASettings.from_config(load_config(cfg.frozen_teacher_config_path))


def _configure_dimensions(model: PeakSetJEPA, cfg: PeakSetJEPASettings) -> None:
    model.training_mode = cfg.training_mode.lower()
    if model.training_mode not in SUPPORTED_TRAINING_MODES:
        raise ValueError(
            "training_mode must be one of ('jepa', 'mae', 'mae_teacher_jepa', 'contrastive')"
        )
    model.model_dim = cfg.model_dim
    model.predictor_dim = (
        cfg.predictor_dim if cfg.predictor_dim is not None else model.model_dim
    )
    model.predictor_pair_dim = (
        cfg.pairformer_pair_dim if cfg.pairformer_pair_dim is not None else model.model_dim
    )
    model.encoder_num_layers = cfg.encoder_num_layers
    model.norm_eps = cfg.norm_eps


def _configure_targets(
    model: PeakSetJEPA,
    cfg: PeakSetJEPASettings,
    frozen_teacher_cfg: PeakSetJEPASettings | None,
) -> None:
    model.jepa_num_target_blocks = cfg.jepa_num_target_blocks
    if model.jepa_num_target_blocks < 1:
        raise ValueError("jepa_num_target_blocks must be >= 1")
    model.teacher_model_dim = (
        frozen_teacher_cfg.model_dim if frozen_teacher_cfg is not None else model.model_dim
    )
    model.teacher_encoder_num_layers = (
        frozen_teacher_cfg.encoder_num_layers
        if frozen_teacher_cfg is not None
        else model.encoder_num_layers
    )
    model.teacher_pair_dim = (
        _pair_dim(frozen_teacher_cfg)
        if frozen_teacher_cfg is not None
        else model.predictor_pair_dim
    )
    model.jepa_target_group_dim = model.teacher_model_dim
    if model.training_mode == "mae_teacher_jepa":
        model.jepa_target_layers = [model.teacher_encoder_num_layers]
    else:
        model.jepa_target_layers = (
            [model.teacher_encoder_num_layers]
            if cfg.jepa_target_layers is None
            else [layer_idx for layer_idx in cfg.jepa_target_layers]
        )
    if not model.jepa_target_layers:
        raise ValueError("jepa_target_layers must not be empty")
    if (
        min(model.jepa_target_layers) < 1
        or max(model.jepa_target_layers) > model.teacher_encoder_num_layers
    ):
        raise ValueError("jepa_target_layers must be within teacher encoder depth")

    model.num_jepa_target_layers = len(model.jepa_target_layers)
    model.jepa_target_dim = model.num_jepa_target_layers * model.teacher_model_dim
    raw_target_projector_dim = (
        model.teacher_model_dim
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


def _configure_losses(model: PeakSetJEPA, cfg: PeakSetJEPASettings) -> None:
    model.mae_loss_weight = cfg.mae_loss_weight
    model.masked_token_loss_weight = (
        0.0 if model.training_mode == "mae" else cfg.masked_token_loss_weight
    )
    model.jepa_mae_loss_weight = (
        0.0 if model.training_mode == "mae" else cfg.jepa_mae_loss_weight
    )
    model.distogram_loss_weight = cfg.distogram_loss_weight
    model.latent_pair_loss_weight = (
        0.0 if model.training_mode == "mae" else cfg.latent_pair_loss_weight
    )
    model.latent_pair_target_normalization = (
        cfg.latent_pair_target_normalization.lower()
    )
    if (
        model.latent_pair_target_normalization
        not in SUPPORTED_LATENT_PAIR_TARGET_NORMALIZATIONS
    ):
        raise ValueError(
            "latent_pair_target_normalization must be one of ('none', 'layernorm')"
        )
    model.distogram_mz_max = cfg.distogram_mz_max
    model.distogram_loss_chunk_size = cfg.distogram_loss_chunk_size
    model.jepa_mae_mz_bin_size = cfg.jepa_mae_mz_bin_size
    model.jepa_mae_intensity_bin_size = cfg.jepa_mae_intensity_bin_size
    model.jepa_mae_mz_max = cfg.jepa_mae_mz_max
    model.jepa_mae_intensity_max = cfg.jepa_mae_intensity_max
    model.jepa_mae_num_mz_bins = math.ceil(
        model.jepa_mae_mz_max / model.jepa_mae_mz_bin_size
    )
    model.distogram_num_bins = math.ceil(
        model.distogram_mz_max / model.jepa_mae_mz_bin_size
    )
    model.jepa_mae_num_intensity_bins = math.ceil(
        model.jepa_mae_intensity_max / model.jepa_mae_intensity_bin_size
    )


def _build_encoder(model: PeakSetJEPA, cfg: PeakSetJEPASettings) -> None:
    model.num_peak_tokens = _num_peak_tokens(cfg)
    model.encoder = _build_peak_set_encoder(cfg)


def _num_peak_tokens(cfg: PeakSetJEPASettings) -> int:
    return cfg.num_peaks


def _build_peak_set_encoder(cfg: PeakSetJEPASettings) -> PeakSetEncoder:
    return PeakSetEncoder(
        model_dim=cfg.model_dim,
        embedder=_build_peak_feature_embedder(cfg),
        num_layers=cfg.encoder_num_layers,
        num_heads=cfg.encoder_num_heads,
        attention_mlp_multiple=cfg.attention_mlp_multiple,
        norm_eps=cfg.norm_eps,
        use_position_embedding=cfg.encoder_use_position_embedding,
        apply_final_norm=cfg.encoder_apply_final_norm,
        apply_final_pair_norm=cfg.encoder_apply_final_pair_norm,
        num_peaks=_num_peak_tokens(cfg),
        pair_dim=cfg.pairformer_pair_dim,
        pair_feature_hidden_dim=cfg.pairformer_pair_feature_hidden_dim,
        pairformer_dropout=cfg.pairformer_dropout,
        pairmixer_triangle_mediator_rank=cfg.pairmixer_triangle_mediator_rank,
        pairmixer_use_commuted_low_rank_triangle=(
            cfg.pairmixer_use_commuted_low_rank_triangle
        ),
        pairformer_mz_scale=cfg.pairformer_mz_scale,
        pairformer_precursor_mz_scale=cfg.pairformer_precursor_mz_scale,
        pairformer_use_fourier_features=cfg.pairformer_use_fourier_features,
        pairformer_fourier_num_freqs=cfg.pairformer_fourier_num_freqs,
        pairformer_fourier_x_min=cfg.pairformer_fourier_x_min,
        pairformer_fourier_x_max=cfg.pairformer_fourier_x_max,
        pairformer_relative_fourier_x_min=cfg.pairformer_relative_fourier_x_min,
        pairformer_relative_fourier_x_max=cfg.pairformer_relative_fourier_x_max,
    )


def _pair_dim(cfg: PeakSetJEPASettings) -> int:
    return cfg.model_dim if cfg.pairformer_pair_dim is None else cfg.pairformer_pair_dim


def _build_peak_feature_embedder(cfg: PeakSetJEPASettings) -> PeakFeatureEmbedder:
    return PeakFeatureEmbedder(
        model_dim=cfg.model_dim,
        hidden_dim=cfg.feature_mlp_hidden_dim,
        fourier_mlp_hidden_dim=cfg.encoder_fourier_mlp_hidden_dim,
        fourier_mlp_num_layers=cfg.encoder_fourier_mlp_num_layers,
        fourier_x_min=cfg.encoder_fourier_x_min,
        fourier_x_max=cfg.encoder_fourier_x_max,
        fourier_num_freqs=cfg.encoder_fourier_num_freqs,
        fourier_input_scale=cfg.encoder_fourier_input_scale,
        use_fourier_features=cfg.encoder_use_fourier_features,
    )


def _build_teacher(
    model: PeakSetJEPA,
    cfg: PeakSetJEPASettings,
    frozen_teacher_cfg: PeakSetJEPASettings | None,
) -> None:
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
    if model.use_ema_teacher:
        model.teacher_encoder = copy.deepcopy(model.encoder)
        model.teacher_encoder.requires_grad_(False)
    elif model.use_frozen_teacher:
        model.teacher_encoder = _build_peak_set_encoder(frozen_teacher_cfg or cfg)
        model.teacher_encoder.requires_grad_(False)
    else:
        model.teacher_encoder = None


def _build_predictor(model: PeakSetJEPA, cfg: PeakSetJEPASettings) -> None:
    model.latent_mask_token = nn.Parameter(torch.empty(model.model_dim))
    nn.init.normal_(model.latent_mask_token, std=0.02)
    model.pair_mask_token = nn.Parameter(torch.empty(model.predictor_pair_dim))
    nn.init.normal_(model.pair_mask_token, std=0.02)

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

    model.num_predictor_input_tokens = model.num_peak_tokens + 1
    model.predictor_position_embedding = _build_frozen_position_embedding(
        model.num_predictor_input_tokens,
        model.model_dim,
    )
    model.predictor_pair_position_embedding = _build_frozen_2d_position_embedding(
        model.num_predictor_input_tokens,
        model.predictor_pair_dim,
    )

    model.masked_latent_predictor = nn.ModuleList(
        [
            PairMixerBlock(
                single_dim=model.predictor_dim,
                pair_dim=model.predictor_pair_dim,
                num_heads=cfg.masked_latent_predictor_num_heads,
                attention_mlp_multiple=cfg.attention_mlp_multiple,
                norm_eps=model.norm_eps,
                dropout=cfg.predictor_dropout,
                triangle_mediator_rank=cfg.pairmixer_triangle_mediator_rank,
                use_commuted_low_rank_triangle=(
                    cfg.pairmixer_use_commuted_low_rank_triangle
                ),
                max_mediator_tokens=model.num_predictor_input_tokens,
            )
            for _ in range(cfg.masked_latent_predictor_num_layers)
        ]
    )
    model.predictor_final_norm = (
        _build_norm(
            model.predictor_dim,
            eps=model.norm_eps,
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

    if (
        model.latent_pair_loss_weight > 0
        and model.predictor_pair_dim != model.teacher_pair_dim
    ):
        masked_pair_readout = nn.Linear(
            model.predictor_pair_dim,
            model.teacher_pair_dim,
        )
        nn.init.xavier_normal_(masked_pair_readout.weight)
        nn.init.zeros_(masked_pair_readout.bias)
        model.masked_pair_readout = masked_pair_readout
    else:
        model.masked_pair_readout = nn.Identity()


def _build_target_projectors(model: PeakSetJEPA, cfg: PeakSetJEPASettings) -> None:
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


def _build_jepa_mae_heads(model: PeakSetJEPA) -> None:
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


def _build_distogram_head(model: PeakSetJEPA) -> None:
    if model.distogram_loss_weight <= 0:
        model.distogram_head = None
        return

    distogram_head = nn.Linear(
        model.predictor_pair_dim,
        model.distogram_num_bins,
    )
    nn.init.xavier_normal_(distogram_head.weight)
    nn.init.zeros_(distogram_head.bias)
    model.distogram_head = distogram_head
