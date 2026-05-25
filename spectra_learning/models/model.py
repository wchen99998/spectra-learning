from __future__ import annotations

from typing import Any, Literal, overload

import torch
from torch import Tensor
from torch import nn

from spectra_learning.models.ema import EMATeacherMixin
from spectra_learning.models.encoder import PeakSetEncoder
from spectra_learning.models.forwards import ForwardMixin
from spectra_learning.models.objectives import CovariancePooler, ObjectiveMixin
from spectra_learning.models.settings import PeakSetJEPASettings
from spectra_learning.models.setup import configure_peak_set_model
from spectra_learning.models.targets import TargetProjectionMixin


class PeakSetJEPA(
    EMATeacherMixin,
    TargetProjectionMixin,
    ObjectiveMixin,
    ForwardMixin,
    nn.Module,
):
    training_mode: str
    model_dim: int
    predictor_dim: int
    predictor_pair_dim: int
    encoder_num_layers: int
    norm_eps: float
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
    target_projector: nn.Module
    teacher_target_projector: nn.Module | None
    jepa_mae_mz_head: nn.Linear | None
    jepa_mae_intensity_head: nn.Linear | None

    def __init__(
        self,
        settings: PeakSetJEPASettings | None = None,
        **overrides: Any,
    ) -> None:
        super().__init__()
        configure_peak_set_model(
            self,
            PeakSetJEPASettings.create(settings, **overrides),
        )

    @overload
    def forward(
        self,
        augmented_batch: dict[str, Tensor],
        return_collapse_data: Literal[False] = False,
        covariance_pooler: CovariancePooler | None = None,
    ) -> dict[str, Tensor]: ...

    @overload
    def forward(
        self,
        augmented_batch: dict[str, Tensor],
        return_collapse_data: Literal[True],
        covariance_pooler: CovariancePooler | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]: ...

    def forward(
        self,
        augmented_batch: dict[str, Tensor],
        return_collapse_data: bool = False,
        covariance_pooler: CovariancePooler | None = None,
    ) -> dict[str, Tensor] | tuple[dict[str, Tensor], dict[str, Tensor]]:
        # augmented_batch tensors:
        # peak_mz/peak_intensity/context_mask/peak_valid_mask: [B, N]
        # target_masks: [B, K, N], precursor_mz: [B] when present.
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
