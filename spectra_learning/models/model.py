from __future__ import annotations

from typing import Any, Literal, overload

import torch
from torch import nn

from spectra_learning.models.ema import EMATeacherMixin
from spectra_learning.models.encoder import PeakSetEncoder
from spectra_learning.models.forwards import ForwardMixin
from spectra_learning.models.losses import SIGReg
from spectra_learning.models.objectives import CovariancePooler, ObjectiveMixin
from spectra_learning.models.settings import PeakSetSIGRegSettings
from spectra_learning.models.setup import configure_peak_set_sigreg
from spectra_learning.models.targets import TargetProjectionMixin


class PeakSetSIGReg(
    EMATeacherMixin,
    TargetProjectionMixin,
    ObjectiveMixin,
    ForwardMixin,
    nn.Module,
):
    training_mode: str
    model_dim: int
    predictor_dim: int
    encoder_num_layers: int
    encoder_use_cls_token: bool
    use_precursor_token: bool
    norm_type: str
    norm_eps: float
    temporal_predictor_num_layers: int
    predictor_num_register_tokens: int
    covariance_pooling_dim: int
    train_covariance_pooling: bool
    covariance_pooling_loss_weight: float
    jepa_num_target_blocks: int
    jepa_target_layers: list[int]
    num_jepa_target_layers: int
    jepa_target_dim: int
    use_target_projector: bool
    target_projector_dim: int
    jepa_target_normalization: str
    masked_token_input_mode: str
    masked_mz_sentinel: float
    predictor_rope_input_scale: float
    representation_regularizer: str
    sigreg_lambda: float
    sigreg_precursor_scale: float
    mae_loss_weight: float
    masked_token_loss_weight: float
    jepa_mae_loss_weight: float
    jepa_mae_mz_bin_size: float
    jepa_mae_intensity_bin_size: float
    jepa_mae_mz_max: float
    jepa_mae_intensity_max: float
    jepa_mae_num_mz_bins: int
    jepa_mae_num_intensity_bins: int
    use_frozen_teacher: bool
    use_ema_teacher: bool
    ema_teacher_momentum_start: float
    ema_teacher_momentum_mid: float
    ema_teacher_momentum_final: float
    ema_teacher_schedule_peak_fraction: float
    ema_teacher_schedule: str
    num_peak_tokens: int

    encoder: PeakSetEncoder
    teacher_encoder: PeakSetEncoder | None
    encoder_to_predictor_proj: nn.Module
    predictor_mask_token: nn.Parameter
    predictor_intensity_embed: nn.Module
    predictor_register_tokens: nn.Parameter | None
    masked_latent_predictor: nn.ModuleList
    predictor_final_norm: nn.Module
    masked_latent_readout: nn.Linear
    target_projector: nn.Module
    teacher_target_projector: nn.Module | None
    jepa_mae_mz_head: nn.Linear | None
    jepa_mae_intensity_head: nn.Linear | None
    sigreg: SIGReg
    temporal_predictor: nn.ModuleList
    temporal_rt_proj: nn.Module
    temporal_query_token: nn.Parameter
    temporal_slot_embedding: nn.Embedding

    def __init__(
        self,
        settings: PeakSetSIGRegSettings | None = None,
        **overrides: Any,
    ) -> None:
        super().__init__()
        configure_peak_set_sigreg(
            self,
            PeakSetSIGRegSettings.create(settings, **overrides),
        )

    @overload
    def forward(
        self,
        augmented_batch: dict[str, torch.Tensor],
        return_collapse_data: Literal[False] = False,
        covariance_pooler: CovariancePooler | None = None,
    ) -> dict[str, torch.Tensor]: ...

    @overload
    def forward(
        self,
        augmented_batch: dict[str, torch.Tensor],
        return_collapse_data: Literal[True],
        covariance_pooler: CovariancePooler | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]: ...

    def forward(
        self,
        augmented_batch: dict[str, torch.Tensor],
        return_collapse_data: bool = False,
        covariance_pooler: CovariancePooler | None = None,
    ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if covariance_pooler is None:
            if return_collapse_data:
                return self.forward_augmented(
                    augmented_batch,
                    return_collapse_data=True,
                )
            return self.forward_augmented(augmented_batch)
        if return_collapse_data:
            return self.forward_augmented(
                augmented_batch,
                return_collapse_data=True,
                covariance_pooler=covariance_pooler,
            )
        return self.forward_augmented(
            augmented_batch,
            covariance_pooler=covariance_pooler,
        )
