from __future__ import annotations

import copy
import math
from typing import Any, Literal, cast, overload

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch import nn

from spectra_learning.models.common import (
    _active_autocast_context,
    _build_frozen_2d_position_embedding,
    _build_frozen_position_embedding,
)
from spectra_learning.models.encoder import PeakSetEncoder
from spectra_learning.models.pairmixer import PairMixerBlock
from spectra_learning.models.peak_features import PeakFeatureEmbedder
from spectra_learning.models.settings import (
    PeakSetJEPASettings,
    ema_teacher_momentum_at as resolve_ema_teacher_momentum,
    load_frozen_teacher_settings,
)
from spectra_learning.models.spectrum_metadata import torch_spectrum_metadata_from_batch
from spectra_learning.models.transformer import _build_norm


def _zero_scalar_like(value: Tensor) -> Tensor:
    return value.reshape(-1)[0] * 0.0


def _cross_entropy_from_logits(
    logits: Float[Tensor, "... classes"],
    targets: Int[Tensor, "..."],
) -> Float[Tensor, "..."]:
    logits = logits.float()
    classes = torch.arange(logits.shape[-1], device=targets.device)
    target_one_hot = (classes == targets.unsqueeze(-1)).to(dtype=logits.dtype)
    return -(F.log_softmax(logits, dim=-1) * target_one_hot).sum(dim=-1)


class PeakSetJEPA(nn.Module):
    training_mode: str
    model_dim: int
    predictor_dim: int
    predictor_pair_dim: int
    pairmixer_block_type: str
    pairmixer_transition_type: str
    teacher_pair_dim: int
    encoder_num_layers: int
    norm_eps: float
    jepa_num_target_blocks: int
    jepa_target_dim: int
    use_target_projector: bool
    target_projector_dim: int
    jepa_target_normalization: str
    masked_token_input_mode: str
    masked_mz_sentinel: float
    mae_loss_weight: float
    masked_token_loss_weight: float
    jepa_mae_loss_weight: float
    distogram_loss_weight: float
    latent_pair_loss_weight: float
    latent_pair_target_normalization: str
    distogram_num_bins: int
    distogram_mz_max: float
    jepa_mae_mz_bin_size: float
    jepa_mae_intensity_bin_size: float
    mae_intensity_loss_weight: float
    jepa_mae_mz_max: float
    jepa_mae_intensity_max: float
    jepa_mae_num_mz_bins: int
    jepa_mae_num_intensity_bins: int
    use_frozen_teacher: bool
    use_ema_teacher: bool
    teacher_model_dim: int
    teacher_encoder_num_layers: int
    jepa_target_group_dim: int
    ema_teacher_momentum_start: float
    ema_teacher_momentum_mid: float
    ema_teacher_momentum_final: float
    ema_teacher_schedule_peak_fraction: float
    ema_teacher_schedule: str
    num_peak_tokens: int
    num_predictor_input_tokens: int

    encoder: PeakSetEncoder
    teacher_encoder: PeakSetEncoder | None
    encoder_to_predictor_proj: nn.Module
    latent_mask_token: nn.Parameter
    pair_mask_token: nn.Parameter
    masked_latent_predictor: nn.ModuleList
    predictor_final_norm: nn.Module
    masked_latent_readout: nn.Linear
    masked_pair_readout: nn.Module
    target_projector: nn.Module
    teacher_target_projector: nn.Module | None
    jepa_mae_mz_head: nn.Linear | None
    jepa_mae_intensity_head: nn.Linear | None
    distogram_head: nn.Linear | None

    def __init__(
        self,
        settings: PeakSetJEPASettings | None = None,
        **overrides: Any,
    ) -> None:
        super().__init__()
        cfg = PeakSetJEPASettings.create(settings, **overrides)
        frozen_teacher_cfg = load_frozen_teacher_settings(cfg)
        self._configure_dimensions(cfg)
        self._configure_targets(cfg, frozen_teacher_cfg)
        self._configure_losses(cfg)
        self._build_encoder(cfg)
        self._build_teacher(cfg, frozen_teacher_cfg)
        self._build_predictor(cfg)
        self._build_target_projectors()
        self._build_jepa_mae_heads()
        self._build_distogram_head()

    def _configure_dimensions(self, cfg: PeakSetJEPASettings) -> None:
        self.training_mode = cfg.training_mode.lower()
        if self.training_mode not in {"jepa", "mae", "mae_teacher_jepa", "contrastive"}:
            raise ValueError(
                "training_mode must be one of ('jepa', 'mae', 'mae_teacher_jepa', 'contrastive')"
            )
        self.model_dim = cfg.model_dim
        self.predictor_dim = (
            cfg.predictor_dim if cfg.predictor_dim is not None else self.model_dim
        )
        self.predictor_pair_dim = (
            cfg.pairmixer_pair_dim
            if cfg.pairmixer_pair_dim is not None
            else self.model_dim
        )
        self.pairmixer_block_type = cfg.pairmixer_block_type.lower()
        if self.pairmixer_block_type not in {
            "dense",
            "bi-dense",
            "fastmixer",
            "fastmixer-dense",
        }:
            raise ValueError(
                "pairmixer_block_type must be one of "
                "('dense', 'bi-dense', 'fastmixer', 'fastmixer-dense')"
            )
        self.pairmixer_transition_type = cfg.pairmixer_transition_type.lower()
        if self.pairmixer_transition_type not in {"swiglu", "feedforward"}:
            raise ValueError(
                "pairmixer_transition_type must be one of ('swiglu', 'feedforward')"
            )
        self.encoder_num_layers = cfg.encoder_num_layers
        self.norm_eps = cfg.norm_eps

    def _configure_targets(
        self,
        cfg: PeakSetJEPASettings,
        frozen_teacher_cfg: PeakSetJEPASettings | None,
    ) -> None:
        self.jepa_num_target_blocks = cfg.jepa_num_target_blocks
        if self.jepa_num_target_blocks < 1:
            raise ValueError("jepa_num_target_blocks must be >= 1")
        self.teacher_model_dim = (
            frozen_teacher_cfg.model_dim
            if frozen_teacher_cfg is not None
            else self.model_dim
        )
        self.teacher_encoder_num_layers = (
            frozen_teacher_cfg.encoder_num_layers
            if frozen_teacher_cfg is not None
            else self.encoder_num_layers
        )
        self.teacher_pair_dim = (
            (
                frozen_teacher_cfg.model_dim
                if frozen_teacher_cfg.pairmixer_pair_dim is None
                else frozen_teacher_cfg.pairmixer_pair_dim
            )
            if frozen_teacher_cfg is not None
            else self.predictor_pair_dim
        )
        self.jepa_target_group_dim = self.teacher_model_dim
        self.jepa_target_dim = self.teacher_model_dim
        raw_target_projector_dim = (
            self.teacher_model_dim
            if cfg.target_projector_dim is None
            else cfg.target_projector_dim
        )
        self.use_target_projector = raw_target_projector_dim >= 0
        self.target_projector_dim = (
            raw_target_projector_dim
            if self.use_target_projector
            else self.jepa_target_dim
        )
        self.jepa_target_normalization = cfg.jepa_target_normalization.lower()
        if self.jepa_target_normalization not in {"none", "zscore"}:
            raise ValueError("jepa_target_normalization must be one of ('none', 'zscore')")
        self.masked_token_input_mode = cfg.masked_token_input_mode.lower()
        if self.masked_token_input_mode not in {"latent_token", "mz_sentinel"}:
            raise ValueError(
                "masked_token_input_mode must be one of ('latent_token', 'mz_sentinel')"
            )
        self.masked_mz_sentinel = cfg.masked_mz_sentinel

    def _configure_losses(self, cfg: PeakSetJEPASettings) -> None:
        self.mae_loss_weight = cfg.mae_loss_weight
        self.masked_token_loss_weight = (
            0.0 if self.training_mode == "mae" else cfg.masked_token_loss_weight
        )
        self.jepa_mae_loss_weight = (
            0.0 if self.training_mode == "mae" else cfg.jepa_mae_loss_weight
        )
        self.distogram_loss_weight = cfg.distogram_loss_weight
        self.latent_pair_loss_weight = (
            0.0 if self.training_mode == "mae" else cfg.latent_pair_loss_weight
        )
        self.latent_pair_target_normalization = (
            cfg.latent_pair_target_normalization.lower()
        )
        if self.latent_pair_target_normalization not in {"none", "layernorm"}:
            raise ValueError(
                "latent_pair_target_normalization must be one of ('none', 'layernorm')"
            )
        self.distogram_mz_max = cfg.distogram_mz_max
        self.jepa_mae_mz_bin_size = cfg.jepa_mae_mz_bin_size
        self.jepa_mae_intensity_bin_size = cfg.jepa_mae_intensity_bin_size
        self.mae_intensity_loss_weight = cfg.mae_intensity_loss_weight
        self.jepa_mae_mz_max = cfg.jepa_mae_mz_max
        self.jepa_mae_intensity_max = cfg.jepa_mae_intensity_max
        self.jepa_mae_num_mz_bins = math.ceil(
            self.jepa_mae_mz_max / self.jepa_mae_mz_bin_size
        )
        self.distogram_num_bins = math.ceil(
            self.distogram_mz_max / self.jepa_mae_mz_bin_size
        )
        self.jepa_mae_num_intensity_bins = math.ceil(
            self.jepa_mae_intensity_max / self.jepa_mae_intensity_bin_size
        )

    def _build_encoder(self, cfg: PeakSetJEPASettings) -> None:
        self.num_peak_tokens = cfg.num_peaks
        self.encoder = self._build_peak_set_encoder(cfg)

    def _build_peak_set_encoder(self, cfg: PeakSetJEPASettings) -> PeakSetEncoder:
        return PeakSetEncoder(
            model_dim=cfg.model_dim,
            embedder=self._build_peak_feature_embedder(cfg),
            num_layers=cfg.encoder_num_layers,
            num_heads=cfg.encoder_num_heads,
            attention_mlp_multiple=cfg.attention_mlp_multiple,
            norm_eps=cfg.norm_eps,
            use_position_embedding=cfg.encoder_use_position_embedding,
            apply_final_norm=cfg.encoder_apply_final_norm,
            apply_final_pair_norm=cfg.encoder_apply_final_pair_norm,
            num_peaks=cfg.num_peaks,
            pairmixer_block_type=cfg.pairmixer_block_type.lower(),
            pairmixer_transition_type=cfg.pairmixer_transition_type.lower(),
            pair_dim=cfg.pairmixer_pair_dim,
            pair_feature_hidden_dim=cfg.pairmixer_pair_feature_hidden_dim,
            pairmixer_dropout=cfg.pairmixer_dropout,
            pairmixer_mz_scale=cfg.pairmixer_mz_scale,
            pairmixer_precursor_mz_scale=cfg.pairmixer_precursor_mz_scale,
            pairmixer_use_fourier_features=cfg.pairmixer_use_fourier_features,
            pairmixer_fourier_num_freqs=cfg.pairmixer_fourier_num_freqs,
            pairmixer_fourier_x_min=cfg.pairmixer_fourier_x_min,
            pairmixer_fourier_x_max=cfg.pairmixer_fourier_x_max,
            pairmixer_relative_fourier_x_min=cfg.pairmixer_relative_fourier_x_min,
            pairmixer_relative_fourier_x_max=cfg.pairmixer_relative_fourier_x_max,
        )

    @staticmethod
    def _build_peak_feature_embedder(cfg: PeakSetJEPASettings) -> PeakFeatureEmbedder:
        return PeakFeatureEmbedder(
            model_dim=cfg.model_dim,
            hidden_dim=cfg.feature_mlp_hidden_dim,
            fourier_mlp_hidden_dim=cfg.encoder_fourier_mlp_hidden_dim,
            fourier_mlp_num_layers=cfg.encoder_fourier_mlp_num_layers,
            fourier_x_min=cfg.encoder_fourier_x_min,
            fourier_x_max=cfg.encoder_fourier_x_max,
            fourier_num_freqs=cfg.encoder_fourier_num_freqs,
            mz_scale=cfg.encoder_mz_scale,
            mz_embedding=cfg.encoder_mz_embedding,
            discrete_bin_size=cfg.encoder_discrete_mz_bin_size,
            discrete_coarse_bin_size=cfg.encoder_discrete_mz_coarse_bin_size,
            discrete_embedding_dim=cfg.encoder_discrete_mz_embedding_dim,
        )

    def _build_teacher(
        self,
        cfg: PeakSetJEPASettings,
        frozen_teacher_cfg: PeakSetJEPASettings | None,
    ) -> None:
        self.use_frozen_teacher = self.training_mode == "mae_teacher_jepa"
        self.use_ema_teacher = cfg.use_ema_teacher and self.training_mode == "jepa"
        self.ema_teacher_momentum_start = cfg.ema_teacher_momentum_start
        self.ema_teacher_momentum_mid = (
            cfg.ema_teacher_momentum_mid
            if cfg.ema_teacher_momentum_mid is not None
            else self.ema_teacher_momentum_start
        )
        self.ema_teacher_momentum_final = (
            cfg.ema_teacher_momentum_final
            if cfg.ema_teacher_momentum_final is not None
            else self.ema_teacher_momentum_start
        )
        self.ema_teacher_schedule_peak_fraction = cfg.ema_teacher_schedule_peak_fraction
        self.ema_teacher_schedule = cfg.ema_teacher_schedule.lower()
        if self.ema_teacher_schedule not in {
            "constant",
            "linear",
            "cosine",
            "slow-fast-slow",
        }:
            raise ValueError(
                "ema_teacher_schedule must be one of "
                "('constant', 'linear', 'cosine', 'slow-fast-slow')"
            )
        if self.use_ema_teacher:
            self.teacher_encoder = copy.deepcopy(self.encoder)
            self.teacher_encoder.requires_grad_(False)
        elif self.use_frozen_teacher:
            self.teacher_encoder = self._build_peak_set_encoder(
                frozen_teacher_cfg or cfg
            )
            self.teacher_encoder.requires_grad_(False)
        else:
            self.teacher_encoder = None

    def _build_predictor(self, cfg: PeakSetJEPASettings) -> None:
        self.latent_mask_token = nn.Parameter(torch.empty(self.model_dim))
        nn.init.normal_(self.latent_mask_token, std=0.02)
        self.pair_mask_token = nn.Parameter(torch.empty(self.predictor_pair_dim))
        nn.init.normal_(self.pair_mask_token, std=0.02)

        if self.predictor_dim != self.model_dim:
            encoder_to_predictor_proj = nn.Linear(
                self.model_dim,
                self.predictor_dim,
                bias=False,
            )
            nn.init.xavier_normal_(encoder_to_predictor_proj.weight)
            self.encoder_to_predictor_proj = encoder_to_predictor_proj
        else:
            self.encoder_to_predictor_proj = nn.Identity()

        self.num_predictor_input_tokens = self.num_peak_tokens + 1
        self.predictor_position_embedding = _build_frozen_position_embedding(
            self.num_predictor_input_tokens,
            self.model_dim,
        )
        self.predictor_pair_position_embedding = _build_frozen_2d_position_embedding(
            self.num_predictor_input_tokens,
            self.predictor_pair_dim,
        )

        predictor_blocks = []
        for _ in range(cfg.masked_latent_predictor_num_layers):
            block = PairMixerBlock(
                single_dim=self.predictor_dim,
                pair_dim=self.predictor_pair_dim,
                num_heads=cfg.masked_latent_predictor_num_heads,
                attention_mlp_multiple=cfg.attention_mlp_multiple,
                norm_eps=self.norm_eps,
                dropout=cfg.predictor_dropout,
                use_single_to_pair_update=(
                    self.pairmixer_block_type in {"bi-dense", "fastmixer"}
                ),
                transition_type=self.pairmixer_transition_type,
            )
            predictor_blocks.append(block)
        self.masked_latent_predictor = nn.ModuleList(predictor_blocks)
        self.predictor_final_norm = (
            _build_norm(
                self.predictor_dim,
                eps=self.norm_eps,
                affine=False,
            )
            if cfg.predictor_apply_final_norm
            else nn.Identity()
        )
        masked_latent_readout = nn.Linear(
            self.predictor_dim,
            self.jepa_target_dim,
        )
        nn.init.xavier_normal_(masked_latent_readout.weight)
        nn.init.zeros_(masked_latent_readout.bias)
        self.masked_latent_readout = masked_latent_readout

        if (
            self.latent_pair_loss_weight > 0
            and self.predictor_pair_dim != self.teacher_pair_dim
        ):
            masked_pair_readout = nn.Linear(
                self.predictor_pair_dim,
                self.teacher_pair_dim,
            )
            nn.init.xavier_normal_(masked_pair_readout.weight)
            nn.init.zeros_(masked_pair_readout.bias)
            self.masked_pair_readout = masked_pair_readout
        else:
            self.masked_pair_readout = nn.Identity()

    def _build_target_projectors(self) -> None:
        if self.use_target_projector:
            target_projector = nn.Sequential(
                nn.Linear(self.jepa_target_dim, self.jepa_target_dim),
                nn.GELU(),
                nn.Linear(self.jepa_target_dim, self.target_projector_dim),
            )
            for layer in target_projector:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    nn.init.zeros_(layer.bias)
            self.target_projector = target_projector
        else:
            self.target_projector = nn.Identity()

        if self.use_ema_teacher or self.use_frozen_teacher:
            self.teacher_target_projector = copy.deepcopy(self.target_projector)
            self.teacher_target_projector.requires_grad_(False)
        else:
            self.teacher_target_projector = None

    def _build_jepa_mae_heads(self) -> None:
        if self.jepa_mae_loss_weight <= 0 and self.training_mode != "mae":
            self.jepa_mae_mz_head = None
            self.jepa_mae_intensity_head = None
            return

        jepa_mae_mz_head = nn.Linear(
            self.target_projector_dim,
            self.jepa_mae_num_mz_bins,
        )
        nn.init.xavier_normal_(jepa_mae_mz_head.weight)
        nn.init.zeros_(jepa_mae_mz_head.bias)
        self.jepa_mae_mz_head = jepa_mae_mz_head
        if self.mae_intensity_loss_weight <= 0.0:
            self.jepa_mae_intensity_head = None
            return

        jepa_mae_intensity_head = nn.Linear(
            self.target_projector_dim,
            self.jepa_mae_num_intensity_bins,
        )
        nn.init.xavier_normal_(jepa_mae_intensity_head.weight)
        nn.init.zeros_(jepa_mae_intensity_head.bias)
        self.jepa_mae_intensity_head = jepa_mae_intensity_head

    def _build_distogram_head(self) -> None:
        if self.distogram_loss_weight <= 0:
            self.distogram_head = None
            return

        distogram_head = nn.Linear(
            self.predictor_pair_dim,
            self.distogram_num_bins,
        )
        nn.init.xavier_normal_(distogram_head.weight)
        nn.init.zeros_(distogram_head.bias)
        self.distogram_head = distogram_head

    def ema_teacher_momentum_at(
        self,
        step: int,
        total_steps: int,
    ) -> float:
        return resolve_ema_teacher_momentum(
            schedule=self.ema_teacher_schedule,
            start=self.ema_teacher_momentum_start,
            mid=self.ema_teacher_momentum_mid,
            final=self.ema_teacher_momentum_final,
            peak_fraction=self.ema_teacher_schedule_peak_fraction,
            step=step,
            total_steps=total_steps,
        )

    @torch.no_grad()
    def sync_ema_teacher(self) -> None:
        if self.teacher_encoder is not None:
            self.teacher_encoder.load_state_dict(self.encoder.state_dict())
        if self.teacher_target_projector is not None:
            self.teacher_target_projector.load_state_dict(
                self.target_projector.state_dict()
            )

    @staticmethod
    @torch.no_grad()
    def _update_ema_module(
        teacher: nn.Module,
        student: nn.Module,
        momentum: float,
    ) -> None:
        teacher_params = list(teacher.parameters())
        if teacher_params:
            torch._foreach_lerp_(
                cast(list[torch.Tensor], teacher_params),
                cast(list[torch.Tensor], list(student.parameters())),
                1.0 - momentum,
            )
        teacher_float_buffers = []
        student_float_buffers = []
        for teacher_buffer, student_buffer in zip(
            teacher.buffers(),
            student.buffers(),
        ):
            if torch.is_floating_point(teacher_buffer):
                teacher_float_buffers.append(teacher_buffer)
                student_float_buffers.append(student_buffer)
            else:
                teacher_buffer.copy_(student_buffer)
        if teacher_float_buffers:
            torch._foreach_lerp_(
                teacher_float_buffers,
                student_float_buffers,
                1.0 - momentum,
            )

    @torch.no_grad()
    def update_ema_teacher(
        self,
        step: int,
        total_steps: int,
    ) -> float | None:
        if not self.use_ema_teacher or self.teacher_encoder is None:
            return None
        momentum = self.ema_teacher_momentum_at(step, total_steps)
        self._update_ema_module(self.teacher_encoder, self.encoder, momentum)
        if self.teacher_target_projector is not None:
            self._update_ema_module(
                self.teacher_target_projector,
                self.target_projector,
                momentum,
            )
        return momentum

    def _apply_group_target_normalization(
        self,
        x: Float[Tensor, "*batch dim"],
        group_dim: int,
    ) -> Float[Tensor, "*batch dim"]:
        if self.jepa_target_normalization == "none":
            return x
        orig_dtype = x.dtype
        # x: [..., groups * group_dim] -> [..., groups, group_dim]
        x = x.float().reshape(*x.shape[:-1], -1, group_dim)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        normalized = ((x - mean) / std).reshape(*x.shape[:-2], -1)
        return normalized.to(dtype=orig_dtype)

    def _apply_jepa_target_normalization(
        self,
        x: Float[Tensor, "batch peaks dim"],
    ) -> Float[Tensor, "batch peaks dim"]:
        return self._apply_group_target_normalization(x, self.jepa_target_group_dim)

    def _add_predictor_positions(
        self,
        x: Float[Tensor, "batch tokens dim"],
    ) -> Float[Tensor, "batch tokens dim"]:
        positions = torch.arange(x.shape[1], device=x.device)
        return x + self.predictor_position_embedding(positions).to(dtype=x.dtype)

    def _add_predictor_pair_positions(
        self,
        pair: Float[Tensor, "batch tokens tokens pair"],
    ) -> Float[Tensor, "batch tokens tokens pair"]:
        num_tokens = pair.shape[1]
        positions = torch.arange(num_tokens * num_tokens, device=pair.device)
        position_encoding = self.predictor_pair_position_embedding(positions)
        position_encoding = position_encoding.view(num_tokens, num_tokens, -1)
        return pair + position_encoding.to(dtype=pair.dtype)

    def predict_masked_latents(
        self,
        x: Float[Tensor, "batch tokens dim"],
        pair: Float[Tensor, "batch tokens tokens pair"],
        visible_mask: Bool[Tensor, "batch tokens"],
    ) -> Float[Tensor, "batch tokens dim"]:
        x, _pair = self._predict_masked_latents_and_pair(
            x,
            pair,
            visible_mask,
        )
        return x

    def _predict_masked_latents_and_pair(
        self,
        x: Float[Tensor, "batch tokens dim"],
        pair: Float[Tensor, "batch tokens tokens pair"],
        visible_mask: Bool[Tensor, "batch tokens"],
    ) -> tuple[
        Float[Tensor, "batch tokens dim"],
        Float[Tensor, "batch tokens tokens pair"],
    ]:
        x = self._add_predictor_positions(x)
        x = self.encoder_to_predictor_proj(x)
        pair = self._add_predictor_pair_positions(pair)
        if len(self.masked_latent_predictor) > 0:
            for block in self.masked_latent_predictor:
                x, pair = block(
                    x,
                    pair,
                    visible_mask,
                    visible_mask,
                )
        x = self.predictor_final_norm(x)
        pair_mask = visible_mask.unsqueeze(2) & visible_mask.unsqueeze(1)
        pair = pair * pair_mask.unsqueeze(-1).to(dtype=pair.dtype)
        return x, pair

    def project_targets(
        self,
        x: Float[Tensor, "*batch dim"],
    ) -> Float[Tensor, "*batch target_dim"]:
        return self.target_projector(x)

    def project_teacher_targets(
        self,
        x: Float[Tensor, "*batch dim"],
    ) -> Float[Tensor, "*batch target_dim"]:
        projector = (
            self.teacher_target_projector
            if self.teacher_target_projector is not None
            else self.target_projector
        )
        return projector(x)

    def predict_masked_target_features(
        self,
        x: Float[Tensor, "batch tokens dim"],
        pair: Float[Tensor, "batch tokens tokens pair"],
        visible_mask: Bool[Tensor, "batch tokens"],
    ) -> Float[Tensor, "batch tokens target_dim"]:
        return self.masked_latent_readout(
            self.predict_masked_latents(
                x,
                pair,
                visible_mask,
            )
        )

    def predict_masked_target_features_with_pair(
        self,
        x: Float[Tensor, "batch tokens dim"],
        pair: Float[Tensor, "batch tokens tokens pair"],
        visible_mask: Bool[Tensor, "batch tokens"],
    ) -> tuple[
        Float[Tensor, "batch tokens target_dim"],
        Float[Tensor, "batch tokens tokens pair"],
    ]:
        x, pair = self._predict_masked_latents_and_pair(
            x,
            pair,
            visible_mask,
        )
        return self.masked_latent_readout(x), pair

    def predict_masked_targets(
        self,
        x: Float[Tensor, "batch tokens dim"],
        pair: Float[Tensor, "batch tokens tokens pair"],
        visible_mask: Bool[Tensor, "batch tokens"],
    ) -> Float[Tensor, "batch tokens target_dim"]:
        return self.project_targets(
            self.predict_masked_target_features(
                x,
                pair,
                visible_mask,
            )
        )

    def _compute_jepa_teacher_target_features(
        self,
        peak_mz: Float[Tensor, "batch peaks"],
        peak_intensity: Float[Tensor, "batch peaks"],
        peak_valid_mask: Bool[Tensor, "batch peaks"],
        precursor_mz: Float[Tensor, "batch"] | None = None,
        spectrum_metadata: Float[Tensor, "batch metadata"] | None = None,
    ) -> Float[Tensor, "batch peaks target_dim"]:
        with _active_autocast_context(peak_mz.device.type):
            teacher_encoder = (
                self.teacher_encoder
                if self.teacher_encoder is not None
                else self.encoder
            )
            teacher_encoded, _ = teacher_encoder.forward_with_pair(
                peak_mz,
                peak_intensity,
                valid_mask=peak_valid_mask,
                visible_mask=peak_valid_mask,
                precursor_mz=precursor_mz,
                spectrum_metadata=spectrum_metadata,
            )
            return teacher_encoded[:, : peak_mz.shape[1]]

    def _compute_jepa_teacher_targets(
        self,
        peak_mz: Float[Tensor, "batch peaks"],
        peak_intensity: Float[Tensor, "batch peaks"],
        peak_valid_mask: Bool[Tensor, "batch peaks"],
        precursor_mz: Float[Tensor, "batch"] | None = None,
        spectrum_metadata: Float[Tensor, "batch metadata"] | None = None,
    ) -> Float[Tensor, "batch peaks target_dim"]:
        with torch.no_grad():
            teacher_target_features = self._compute_jepa_teacher_target_features(
                peak_mz,
                peak_intensity,
                peak_valid_mask,
                precursor_mz=precursor_mz,
                spectrum_metadata=spectrum_metadata,
            )
            return self.project_teacher_targets(
                self._apply_jepa_target_normalization(teacher_target_features)
            )

    def _context_encoder_inputs(
        self,
        peak_mz: Float[Tensor, "batch peaks"],
        peak_intensity: Float[Tensor, "batch peaks"],
        context_mask: Bool[Tensor, "batch peaks"],
        target_masks: Bool[Tensor, "batch views peaks"],
    ) -> tuple[
        Float[Tensor, "batch peaks"],
        Float[Tensor, "batch peaks"],
        Bool[Tensor, "batch peaks"],
    ]:
        if self.masked_token_input_mode != "mz_sentinel":
            return peak_mz, peak_intensity, context_mask
        target_union = target_masks.any(dim=1)
        masked_mz = torch.where(
            target_union,
            torch.full_like(peak_mz, self.masked_mz_sentinel),
            peak_mz,
        )
        return masked_mz, peak_intensity, context_mask | target_union

    def compute_teacher_targets(
        self,
        augmented_batch: dict[str, Tensor],
    ) -> Float[Tensor, "batch peaks target_dim"]:
        return self._compute_jepa_teacher_targets(
            augmented_batch["peak_mz"],
            augmented_batch["peak_intensity"],
            augmented_batch["peak_valid_mask"],
            precursor_mz=augmented_batch.get("precursor_mz", None),
            spectrum_metadata=torch_spectrum_metadata_from_batch(augmented_batch),
        )

    def _encode_augmented_teacher_and_context(
        self,
        peak_mz: Float[Tensor, "batch peaks"],
        peak_intensity: Float[Tensor, "batch peaks"],
        peak_valid_mask: Bool[Tensor, "batch peaks"],
        context_mask: Bool[Tensor, "batch peaks"],
        target_masks: Bool[Tensor, "batch views peaks"],
        precursor_mz: Float[Tensor, "batch"] | None = None,
        spectrum_metadata: Float[Tensor, "batch metadata"] | None = None,
    ) -> tuple[
        Float[Tensor, "batch peaks target_dim"],
        Float[Tensor, "batch tokens dim"],
        Float[Tensor, "batch peaks peaks pair"],
        Float[Tensor, "batch tokens dim"],
        Float[Tensor, "batch tokens tokens pair"],
    ]:
        batch_size = peak_mz.shape[0]
        context_mz, context_intensity, context_visible_mask = self._context_encoder_inputs(
            peak_mz,
            peak_intensity,
            context_mask,
            target_masks,
        )
        if self.teacher_encoder is not None:
            with torch.no_grad(), _active_autocast_context(peak_mz.device.type):
                teacher_encoded, teacher_pair = self.teacher_encoder.forward_with_pair(
                    peak_mz,
                    peak_intensity,
                    valid_mask=peak_valid_mask,
                    visible_mask=peak_valid_mask,
                    precursor_mz=precursor_mz,
                    spectrum_metadata=spectrum_metadata,
                )
            context_encoded, context_pair = self.encoder.forward_with_pair(
                context_mz,
                context_intensity,
                valid_mask=peak_valid_mask,
                visible_mask=context_visible_mask,
                precursor_mz=precursor_mz,
                spectrum_metadata=spectrum_metadata,
            )
            return (
                teacher_encoded[:, : peak_mz.shape[1]],
                teacher_encoded,
                teacher_pair[:, : peak_mz.shape[1], : peak_mz.shape[1]],
                context_encoded,
                context_pair,
            )
        encoded, pair = self.encoder.forward_with_pair(
            torch.cat([peak_mz, context_mz], dim=0),
            torch.cat([peak_intensity, context_intensity], dim=0),
            valid_mask=torch.cat([peak_valid_mask, peak_valid_mask], dim=0),
            visible_mask=torch.cat([peak_valid_mask, context_visible_mask], dim=0),
            precursor_mz=(
                None
                if precursor_mz is None
                else torch.cat([precursor_mz, precursor_mz], dim=0)
            ),
            spectrum_metadata=(
                None
                if spectrum_metadata is None
                else torch.cat([spectrum_metadata, spectrum_metadata], dim=0)
            ),
        )
        return (
            encoded[:batch_size, : peak_mz.shape[1]],
            encoded[:batch_size],
            pair[:batch_size, : peak_mz.shape[1], : peak_mz.shape[1]],
            encoded[batch_size:],
            pair[batch_size:],
        )

    def _compute_pooled_teacher_peak_targets(
        self,
        peak_mz: Float[Tensor, "batch peaks"],
        peak_intensity: Float[Tensor, "batch peaks"],
        peak_valid_mask: Bool[Tensor, "batch peaks"],
        visible_mask: Bool[Tensor, "batch peaks"] | None = None,
        precursor_mz: Float[Tensor, "batch"] | None = None,
        spectrum_metadata: Float[Tensor, "batch metadata"] | None = None,
    ) -> Float[Tensor, "batch dim"]:
        if visible_mask is None:
            visible_mask = peak_valid_mask
        with _active_autocast_context(peak_mz.device.type):
            teacher_encoder = (
                self.teacher_encoder
                if self.teacher_encoder is not None
                else self.encoder
            )
            teacher_encoded = teacher_encoder(
                peak_mz,
                peak_intensity,
                valid_mask=peak_valid_mask,
                visible_mask=visible_mask,
                precursor_mz=precursor_mz,
                spectrum_metadata=spectrum_metadata,
            )
        return self.pool(teacher_encoded, visible_mask)

    def _embedding_loss(
        self,
        prediction: Float[Tensor, "*batch dim"],
        target: Float[Tensor, "*batch dim"],
    ) -> Float[Tensor, "*batch"]:
        prediction = prediction.float()
        target = target.float()
        return (prediction - target).square().mean(dim=-1)

    def _jepa_mae_targets(
        self,
        peak_mz: Float[Tensor, "batch peaks"],
        peak_intensity: Float[Tensor, "batch peaks"],
    ) -> tuple[Int[Tensor, "batch peaks"], Int[Tensor, "batch peaks"]]:
        mz_target = torch.floor(
            peak_mz.float() * self.jepa_mae_mz_max / self.jepa_mae_mz_bin_size
        ).long()
        intensity_target = torch.floor(
            peak_intensity.float() / self.jepa_mae_intensity_bin_size
        ).long()
        return (
            mz_target.clamp(0, self.jepa_mae_num_mz_bins - 1),
            intensity_target.clamp(0, self.jepa_mae_num_intensity_bins - 1),
        )

    def _masked_ce_loss(
        self,
        logits: Float[Tensor, "... classes"],
        targets: Int[Tensor, "..."],
        valid_mask: Bool[Tensor, "..."],
    ) -> Float[Tensor, ""]:
        per_token = _cross_entropy_from_logits(logits, targets)
        weights = valid_mask.float()
        return (per_token * weights).sum() / weights.sum().clamp_min(1.0)

    def _jepa_mae_value_prediction_loss(
        self,
        predicted_latents: Float[Tensor, "batch views peaks dim"],
        peak_mz: Float[Tensor, "batch peaks"],
        peak_intensity: Float[Tensor, "batch peaks"],
        target_masks: Bool[Tensor, "batch views peaks"],
    ) -> tuple[
        Float[Tensor, ""],
        Float[Tensor, ""],
        Float[Tensor, ""],
        Float[Tensor, ""],
        Float[Tensor, ""],
    ]:
        mz_logits = cast(nn.Linear, self.jepa_mae_mz_head)(predicted_latents)
        mz_target, intensity_target = self._jepa_mae_targets(peak_mz, peak_intensity)
        # targets: [B, N] -> [B, K, N], matching predicted_latents/logits.
        view_shape = (mz_logits.shape[0], mz_logits.shape[1], mz_logits.shape[2])
        mz_target = mz_target.unsqueeze(1).expand(view_shape)
        intensity_target = intensity_target.unsqueeze(1).expand(view_shape)
        mz_loss = self._masked_ce_loss(mz_logits, mz_target, target_masks)
        target_weights = target_masks
        mz_accuracy = (
            (mz_logits.argmax(dim=-1) == mz_target).float() * target_weights.float()
        ).sum() / target_weights.float().sum().clamp_min(1.0)
        zero = mz_loss * 0.0
        if (
            self.masked_token_input_mode == "mz_sentinel"
            or self.mae_intensity_loss_weight <= 0.0
        ):
            return mz_loss, mz_loss, zero, mz_accuracy, zero

        intensity_logits = cast(nn.Linear, self.jepa_mae_intensity_head)(
            predicted_latents
        )
        intensity_loss = self._masked_ce_loss(
            intensity_logits,
            intensity_target,
            target_masks,
        )
        value_loss = mz_loss + intensity_loss * self.mae_intensity_loss_weight
        intensity_accuracy = (
            (intensity_logits.argmax(dim=-1) == intensity_target).float()
            * target_weights.float()
        ).sum() / target_weights.float().sum().clamp_min(1.0)
        return value_loss, mz_loss, intensity_loss, mz_accuracy, intensity_accuracy

    def _predict_augmented_targets(
        self,
        context_emb: Float[Tensor, "batch tokens dim"],
        context_pair: Float[Tensor, "batch tokens tokens pair"],
        context_mask: Bool[Tensor, "batch peaks"],
        target_masks: Bool[Tensor, "batch views peaks"],
    ) -> tuple[
        Float[Tensor, "batch views peaks target_dim"],
        Float[Tensor, "batch views peaks target_dim"],
    ]:
        predictor_features, predictor_output, _predictor_pair = (
            self._predict_augmented_target_outputs(
                context_emb,
                context_pair,
                context_mask,
                target_masks,
            )
        )
        return predictor_features, predictor_output

    def _predict_augmented_target_outputs(
        self,
        context_emb: Float[Tensor, "batch tokens dim"],
        context_pair: Float[Tensor, "batch tokens tokens pair"],
        context_mask: Bool[Tensor, "batch peaks"],
        target_masks: Bool[Tensor, "batch views peaks"],
    ) -> tuple[
        Float[Tensor, "batch views peaks target_dim"],
        Float[Tensor, "batch views peaks target_dim"],
        Float[Tensor, "batch views peaks peaks pair"],
    ]:
        batch_size, num_target_blocks, num_peaks = target_masks.shape
        context_mask_by_view = context_mask.unsqueeze(1)
        context_peak_emb = context_emb[:, :num_peaks]
        context_cls_emb = context_emb[:, num_peaks : num_peaks + 1]
        # predictor_input: [B, K, N + 1, D]
        predictor_input = (
            context_peak_emb.unsqueeze(1).expand(-1, num_target_blocks, -1, -1)
            * context_mask_by_view.unsqueeze(-1)
        )
        latent_mask_token = self.latent_mask_token.view(1, 1, 1, -1).to(context_emb)
        latent_mask_token = latent_mask_token + predictor_input[:, :, :1] * 0.0
        predictor_input = torch.where(
            target_masks.unsqueeze(-1),
            latent_mask_token,
            predictor_input,
        )
        predictor_input = torch.cat(
            [
                predictor_input,
                context_cls_emb.unsqueeze(1).expand(-1, num_target_blocks, -1, -1),
            ],
            dim=2,
        )
        predictor_visible_mask = context_mask_by_view | target_masks
        predictor_visible_mask = torch.cat(
            [
                predictor_visible_mask,
                torch.ones_like(predictor_visible_mask[:, :, :1]),
            ],
            dim=2,
        )
        predictor_pair = context_pair.unsqueeze(1).expand(
            -1,
            num_target_blocks,
            -1,
            -1,
            -1,
        )
        context_token_mask = torch.cat(
            [
                context_mask_by_view.expand(-1, num_target_blocks, -1),
                torch.ones_like(context_mask_by_view[:, :, :1]).expand(
                    -1,
                    num_target_blocks,
                    -1,
                ),
            ],
            dim=2,
        )
        context_pair_mask = (
            context_token_mask.unsqueeze(3) & context_token_mask.unsqueeze(2)
        )
        predictor_pair = predictor_pair * context_pair_mask.unsqueeze(-1).to(
            dtype=predictor_pair.dtype
        )
        target_token_mask = torch.cat(
            [
                target_masks,
                torch.zeros_like(target_masks[:, :, :1]),
            ],
            dim=2,
        )
        target_pair_mask = target_token_mask.unsqueeze(3) | target_token_mask.unsqueeze(2)
        pair_mask_token = self.pair_mask_token.view(1, 1, 1, 1, -1).to(
            context_pair
        )
        pair_mask_token = pair_mask_token + predictor_pair[:, :, :1, :1] * 0.0
        predictor_pair = torch.where(
            target_pair_mask.unsqueeze(-1),
            pair_mask_token,
            predictor_pair,
        )
        predictor_pair_mask = (
            predictor_visible_mask.unsqueeze(3) & predictor_visible_mask.unsqueeze(2)
        )
        predictor_pair = predictor_pair * predictor_pair_mask.unsqueeze(-1).to(
            dtype=predictor_pair.dtype
        )
        predictor_visible_mask = predictor_visible_mask.reshape(
            batch_size * num_target_blocks,
            num_peaks + 1,
        )
        # Flatten target views into the batch: [B, K, T, D] -> [B*K, T, D].
        flat_predictor_input = predictor_input.reshape(
            batch_size * num_target_blocks,
            predictor_input.shape[2],
            -1,
        )
        flat_predictor_pair = predictor_pair.reshape(
            batch_size * num_target_blocks,
            predictor_pair.shape[2],
            predictor_pair.shape[3],
            -1,
        )
        predictor_features, predictor_pair = (
            self.predict_masked_target_features_with_pair(
                flat_predictor_input,
                flat_predictor_pair,
                predictor_visible_mask,
            )
        )
        predictor_features = predictor_features.reshape(
            batch_size,
            num_target_blocks,
            predictor_input.shape[2],
            -1,
        )
        predictor_features = predictor_features[:, :, :num_peaks]
        predictor_pair = predictor_pair.reshape(
            batch_size,
            num_target_blocks,
            flat_predictor_pair.shape[1],
            flat_predictor_pair.shape[2],
            -1,
        )
        predictor_pair = predictor_pair[:, :, :num_peaks, :num_peaks]
        predictor_output = self.project_targets(predictor_features)
        return predictor_features, predictor_output, predictor_pair

    def _masked_prediction_loss(
        self,
        predictor_output: Float[Tensor, "batch views peaks target_dim"],
        teacher_targets: Float[Tensor, "batch peaks target_dim"],
        target_masks: Bool[Tensor, "batch views peaks"],
    ) -> Float[Tensor, ""]:
        per_token = self._embedding_loss(predictor_output, teacher_targets.unsqueeze(1))
        target_weights = target_masks.float()
        return (per_token * target_weights).sum() / target_weights.sum().clamp_min(1.0)

    def _distogram_targets(
        self,
        peak_mz: Float[Tensor, "batch peaks"],
    ) -> Int[Tensor, "batch peaks peaks"]:
        mz_da = peak_mz.float() * self.distogram_mz_max
        pair_distance = (mz_da.unsqueeze(2) - mz_da.unsqueeze(1)).abs()
        return torch.floor(pair_distance / self.jepa_mae_mz_bin_size).long().clamp(
            0,
            self.distogram_num_bins - 1,
        )

    def _target_pair_mask(
        self,
        target_masks: Bool[Tensor, "batch views peaks"],
        predictor_visible_masks: Bool[Tensor, "batch views peaks"],
    ) -> Bool[Tensor, "batch views peaks peaks"]:
        target_pair_mask = target_masks.unsqueeze(3) | target_masks.unsqueeze(2)
        visible_pair_mask = (
            predictor_visible_masks.unsqueeze(3) & predictor_visible_masks.unsqueeze(2)
        )
        diagonal = torch.eye(
            target_masks.shape[-1],
            dtype=torch.bool,
            device=target_masks.device,
        )
        return target_pair_mask & visible_pair_mask & ~diagonal.view(
            1,
            1,
            target_masks.shape[-1],
            target_masks.shape[-1],
        )

    def _distogram_pair_mask(
        self,
        target_masks: Bool[Tensor, "batch views peaks"],
        predictor_visible_masks: Bool[Tensor, "batch views peaks"],
    ) -> Bool[Tensor, "batch views peaks peaks"]:
        return self._target_pair_mask(target_masks, predictor_visible_masks)

    def _distogram_logits(
        self,
        predictor_pair: Float[Tensor, "batch views peaks peaks pair"],
    ) -> Float[Tensor, "batch views peaks peaks bins"]:
        sym_pair = predictor_pair + predictor_pair.transpose(2, 3)
        return cast(nn.Linear, self.distogram_head)(sym_pair)

    def _distogram_metrics(
        self,
        predictor_pair: Float[Tensor, "batch views peaks peaks pair"],
        peak_mz: Float[Tensor, "batch peaks"],
        target_masks: Bool[Tensor, "batch views peaks"],
        predictor_visible_masks: Bool[Tensor, "batch views peaks"],
        reference: Float[Tensor, "*batch dim"],
    ) -> tuple[Float[Tensor, ""], dict[str, Tensor]]:
        if self.distogram_loss_weight <= 0:
            return _zero_scalar_like(reference), {}
        pair_mask = self._distogram_pair_mask(target_masks, predictor_visible_masks)
        logits = self._distogram_logits(predictor_pair)
        targets = self._distogram_targets(peak_mz).unsqueeze(1).expand(
            logits.shape[0],
            logits.shape[1],
            logits.shape[2],
            logits.shape[3],
        )
        distogram_loss = self._masked_ce_loss(
            logits,
            targets,
            pair_mask,
        )
        term = distogram_loss.to(dtype=reference.dtype) * self.distogram_loss_weight
        return term, {
            "distogram_loss": distogram_loss.to(dtype=reference.dtype),
            "distogram_term": term,
        }

    def _latent_pair_metrics(
        self,
        predictor_pair: Float[Tensor, "batch views peaks peaks pair"],
        teacher_pair: Float[Tensor, "batch peaks peaks pair"],
        target_masks: Bool[Tensor, "batch views peaks"],
        predictor_visible_masks: Bool[Tensor, "batch views peaks"],
        reference: Float[Tensor, "*batch dim"],
    ) -> tuple[Float[Tensor, ""], dict[str, Tensor]]:
        if self.latent_pair_loss_weight <= 0:
            return _zero_scalar_like(reference), {}
        pair_mask = self._target_pair_mask(target_masks, predictor_visible_masks)
        predicted_pair = self.masked_pair_readout(predictor_pair)
        with torch.no_grad():
            teacher_pair_targets = teacher_pair.detach()
            if self.latent_pair_target_normalization == "layernorm":
                teacher_pair_targets = F.layer_norm(
                    teacher_pair_targets.float(),
                    (self.teacher_pair_dim,),
                )
            teacher_pair_targets = teacher_pair_targets.to(dtype=predicted_pair.dtype)
            teacher_pair_targets = teacher_pair_targets.unsqueeze(1).expand(
                predicted_pair.shape[0],
                predicted_pair.shape[1],
                predicted_pair.shape[2],
                predicted_pair.shape[3],
                predicted_pair.shape[4],
            )
        per_pair = self._embedding_loss(predicted_pair, teacher_pair_targets)
        pair_weights = pair_mask.float()
        latent_pair_loss = (
            per_pair * pair_weights
        ).sum() / pair_weights.sum().clamp_min(1.0)
        term = latent_pair_loss.to(dtype=reference.dtype) * self.latent_pair_loss_weight
        return term, {
            "latent_pair_loss": latent_pair_loss.to(dtype=reference.dtype),
            "latent_pair_term": term,
        }

    def _jepa_mae_metrics(
        self,
        predictor_output: Float[Tensor, "batch views peaks target_dim"],
        peak_mz: Float[Tensor, "batch peaks"],
        peak_intensity: Float[Tensor, "batch peaks"],
        target_masks: Bool[Tensor, "batch views peaks"],
        reference: Float[Tensor, "*batch dim"],
    ) -> tuple[Float[Tensor, ""], dict[str, Tensor]]:
        if self.jepa_mae_loss_weight <= 0:
            return _zero_scalar_like(reference), {}
        (
            value_loss,
            _mz_loss,
            _intensity_loss,
            _mz_accuracy,
            _intensity_accuracy,
        ) = self._jepa_mae_value_prediction_loss(
            predictor_output,
            peak_mz,
            peak_intensity,
            target_masks,
        )
        term = value_loss.to(dtype=reference.dtype) * self.jepa_mae_loss_weight
        return term, {
            "jepa_mae_loss": value_loss.to(dtype=reference.dtype),
            "jepa_mae_term": term,
        }

    def _mae_metrics(
        self,
        predictor_output: Float[Tensor, "batch views peaks target_dim"],
        peak_mz: Float[Tensor, "batch peaks"],
        peak_intensity: Float[Tensor, "batch peaks"],
        target_masks: Bool[Tensor, "batch views peaks"],
        reference: Float[Tensor, "*batch dim"],
    ) -> tuple[Float[Tensor, ""], dict[str, Tensor]]:
        (
            value_loss,
            mz_loss,
            intensity_loss,
            mz_accuracy,
            intensity_accuracy,
        ) = self._jepa_mae_value_prediction_loss(
            predictor_output,
            peak_mz,
            peak_intensity,
            target_masks,
        )
        term = value_loss.to(dtype=reference.dtype) * self.mae_loss_weight
        return term, {
            "mae_loss": value_loss.to(dtype=reference.dtype),
            "mae_term": term,
            "mae_mz_loss": mz_loss.to(dtype=reference.dtype),
            "mae_intensity_loss": intensity_loss.to(dtype=reference.dtype),
            "mae_mz_accuracy": mz_accuracy.to(dtype=reference.dtype),
            "mae_intensity_accuracy": intensity_accuracy.to(dtype=reference.dtype),
        }

    def pool(
        self,
        embeddings: Float[Tensor, "batch tokens dim"],
        valid_mask: Bool[Tensor, "batch peaks"],
    ) -> Float[Tensor, "batch dim"]:
        num_extra_tokens = embeddings.shape[1] - valid_mask.shape[1]
        if num_extra_tokens > 0:
            embeddings = embeddings[:, : valid_mask.shape[1]]
        mask = valid_mask.unsqueeze(-1).float()
        return (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    @staticmethod
    def _target_mask_metrics(
        target_masks: Bool[Tensor, "batch views peaks"],
        valid_peak_count: Float[Tensor, ""],
    ) -> dict[str, Tensor]:
        target_entries = target_masks.float().sum()
        target_union = target_masks.any(dim=1).float().sum()
        target_overlap_entries = target_entries - target_union
        per_view_denominator = valid_peak_count * max(target_masks.shape[1], 1)
        return {
            "target_fraction": target_entries / per_view_denominator,
            "target_fraction_per_view": target_entries / per_view_denominator,
            "target_union_fraction": target_union / valid_peak_count,
            "target_entry_fraction": target_entries / valid_peak_count,
            "target_overlap_entries": target_overlap_entries,
        }

    @overload
    def forward_augmented(
        self,
        augmented_batch: dict[str, Tensor],
        return_collapse_data: Literal[False] = False,
    ) -> dict[str, Tensor]: ...

    @overload
    def forward_augmented(
        self,
        augmented_batch: dict[str, Tensor],
        return_collapse_data: Literal[True],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]: ...

    def forward_augmented(
        self,
        augmented_batch: dict[str, Tensor],
        return_collapse_data: bool = False,
    ) -> dict[str, Tensor] | tuple[dict[str, Tensor], dict[str, Tensor]]:
        # augmented_batch tensors:
        # peak_mz/peak_intensity/context_mask/peak_valid_mask: [B, N]
        # target_masks: [B, K, N], precursor_mz: [B] when present.
        if self.training_mode == "mae":
            return self.forward_mae(
                augmented_batch,
                return_collapse_data=return_collapse_data,
            )

        peak_mz = augmented_batch["peak_mz"]
        peak_intensity = augmented_batch["peak_intensity"]
        peak_valid_mask = augmented_batch["peak_valid_mask"]
        precursor_mz = augmented_batch.get("precursor_mz", None)
        spectrum_metadata = torch_spectrum_metadata_from_batch(augmented_batch)
        context_mask = augmented_batch["context_mask"] & peak_valid_mask
        target_masks = augmented_batch["target_masks"] & peak_valid_mask.unsqueeze(1)
        (
            teacher_target_features,
            teacher_peak_emb,
            teacher_pair,
            context_emb,
            context_pair,
        ) = self._encode_augmented_teacher_and_context(
            peak_mz,
            peak_intensity,
            peak_valid_mask,
            context_mask,
            target_masks,
            precursor_mz=precursor_mz,
            spectrum_metadata=spectrum_metadata,
        )
        predictor_output_features, predictor_output, predictor_pair = (
            self._predict_augmented_target_outputs(
                context_emb,
                context_pair,
                context_mask,
                target_masks,
            )
        )
        teacher_target_features_normalized = self._apply_jepa_target_normalization(
            teacher_target_features.detach()
        )
        with torch.no_grad():
            teacher_targets = self.project_teacher_targets(
                teacher_target_features_normalized
            )

        masked_prediction_loss = self._masked_prediction_loss(
            predictor_output,
            teacher_targets,
            target_masks,
        )
        masked_prediction_term = masked_prediction_loss * self.masked_token_loss_weight
        jepa_mae_term, jepa_mae_metrics = self._jepa_mae_metrics(
            predictor_output,
            peak_mz,
            peak_intensity,
            target_masks,
            context_emb,
        )
        predictor_visible_masks = context_mask.unsqueeze(1) | target_masks
        distogram_term, distogram_metrics = self._distogram_metrics(
            predictor_pair,
            peak_mz,
            target_masks,
            predictor_visible_masks,
            predictor_output,
        )
        latent_pair_term, latent_pair_metrics = self._latent_pair_metrics(
            predictor_pair,
            teacher_pair,
            target_masks,
            predictor_visible_masks,
            predictor_output,
        )
        loss = (
            masked_prediction_term
            + jepa_mae_term
            + distogram_term
            + latent_pair_term
        )
        valid_peak_count = peak_valid_mask.float().sum().clamp_min(1.0)
        collapse_data: dict[str, Tensor] = {}
        if return_collapse_data:
            pooled_mean = self.pool(teacher_peak_emb, peak_valid_mask)
            collapse_data = {
                "teacher_peak_emb": teacher_peak_emb.detach(),
                "context_emb": context_emb.detach(),
                "context_mask": context_mask.detach(),
                "peak_valid_mask": peak_valid_mask.detach(),
                "target_masks": target_masks.detach(),
                "teacher_target_features": teacher_target_features.detach(),
                "teacher_target_features_normalized": (
                    teacher_target_features_normalized.detach()
                ),
                "teacher_targets": teacher_targets.detach(),
                "predictor_output_features": predictor_output_features.detach(),
                "predictor_output": predictor_output.detach(),
                "pooled_mean": pooled_mean.detach(),
            }
        metrics = {
            "loss": loss,
            "masked_prediction_loss": masked_prediction_loss,
            "masked_prediction_term": masked_prediction_term,
            "context_fraction": context_mask.float().sum() / valid_peak_count,
        }
        metrics.update(self._target_mask_metrics(target_masks, valid_peak_count))
        metrics.update(jepa_mae_metrics)
        metrics.update(distogram_metrics)
        metrics.update(latent_pair_metrics)
        if return_collapse_data:
            return metrics, collapse_data
        return metrics

    @overload
    def forward_mae(
        self,
        augmented_batch: dict[str, Tensor],
        return_collapse_data: Literal[False] = False,
    ) -> dict[str, Tensor]: ...

    @overload
    def forward_mae(
        self,
        augmented_batch: dict[str, Tensor],
        return_collapse_data: Literal[True],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]: ...

    def forward_mae(
        self,
        augmented_batch: dict[str, Tensor],
        return_collapse_data: bool = False,
    ) -> dict[str, Tensor] | tuple[dict[str, Tensor], dict[str, Tensor]]:
        # augmented_batch tensors:
        # peak_mz/peak_intensity/context_mask/peak_valid_mask: [B, N]
        # target_masks: [B, K, N], precursor_mz: [B] when present.
        peak_mz = augmented_batch["peak_mz"]
        peak_intensity = augmented_batch["peak_intensity"]
        peak_valid_mask = augmented_batch["peak_valid_mask"]
        precursor_mz = augmented_batch.get("precursor_mz", None)
        spectrum_metadata = torch_spectrum_metadata_from_batch(augmented_batch)
        context_mask = augmented_batch["context_mask"] & peak_valid_mask
        target_masks = augmented_batch["target_masks"] & peak_valid_mask.unsqueeze(1)

        context_mz, context_intensity, context_visible_mask = self._context_encoder_inputs(
            peak_mz,
            peak_intensity,
            context_mask,
            target_masks,
        )

        context_encoded, context_pair = self.encoder.forward_with_pair(
            context_mz,
            context_intensity,
            valid_mask=peak_valid_mask,
            visible_mask=context_visible_mask,
            precursor_mz=precursor_mz,
            spectrum_metadata=spectrum_metadata,
        )
        predictor_output_features, predictor_output, predictor_pair = (
            self._predict_augmented_target_outputs(
                context_encoded,
                context_pair,
                context_mask,
                target_masks,
            )
        )
        mae_term, mae_metrics = self._mae_metrics(
            predictor_output,
            peak_mz,
            peak_intensity,
            target_masks,
            context_encoded,
        )
        predictor_visible_masks = context_mask.unsqueeze(1) | target_masks
        distogram_term, distogram_metrics = self._distogram_metrics(
            predictor_pair,
            peak_mz,
            target_masks,
            predictor_visible_masks,
            predictor_output,
        )
        loss = mae_term + distogram_term
        valid_peak_count = peak_valid_mask.float().sum().clamp_min(1.0)
        metrics = {
            "loss": loss,
            "context_fraction": context_mask.float().sum() / valid_peak_count,
        }
        metrics.update(self._target_mask_metrics(target_masks, valid_peak_count))
        metrics.update(mae_metrics)
        metrics.update(distogram_metrics)
        if return_collapse_data:
            return metrics, {}
        return metrics

    def encode(
        self,
        batch: dict[str, Tensor],
    ) -> Float[Tensor, "batch dim"]:
        mz, intensity, valid = (
            batch["peak_mz"],
            batch["peak_intensity"],
            batch["peak_valid_mask"],
        )
        encoded = self.encoder(
            mz,
            intensity,
            valid_mask=valid,
            visible_mask=valid,
            precursor_mz=batch.get("precursor_mz", None),
            spectrum_metadata=torch_spectrum_metadata_from_batch(batch),
        )
        return self.pool(encoded, valid)

    @overload
    def forward(
        self,
        augmented_batch: dict[str, Tensor],
        return_collapse_data: Literal[False] = False,
    ) -> dict[str, Tensor]: ...

    @overload
    def forward(
        self,
        augmented_batch: dict[str, Tensor],
        return_collapse_data: Literal[True],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]: ...

    def forward(
        self,
        augmented_batch: dict[str, Tensor],
        return_collapse_data: bool = False,
    ) -> dict[str, Tensor] | tuple[dict[str, Tensor], dict[str, Tensor]]:
        # augmented_batch tensors:
        # peak_mz/peak_intensity/context_mask/peak_valid_mask: [B, N]
        # target_masks: [B, K, N], precursor_mz: [B] when present.
        if return_collapse_data:
            return self.forward_augmented(
                augmented_batch,
                return_collapse_data=True,
            )
        return self.forward_augmented(augmented_batch)
