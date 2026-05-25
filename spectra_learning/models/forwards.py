from __future__ import annotations

from typing import Any, Literal, overload

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from spectra_learning.models.objectives import CovariancePooler


class ForwardMixin:
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
        self: Any,
        augmented_batch: dict[str, Tensor],
        return_collapse_data: Literal[False] = False,
        covariance_pooler: CovariancePooler | None = None,
    ) -> dict[str, Tensor]: ...

    @overload
    def forward_augmented(
        self: Any,
        augmented_batch: dict[str, Tensor],
        return_collapse_data: Literal[True],
        covariance_pooler: CovariancePooler | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]: ...

    def forward_augmented(
        self: Any,
        augmented_batch: dict[str, Tensor],
        return_collapse_data: bool = False,
        covariance_pooler: CovariancePooler | None = None,
    ) -> dict[str, Tensor] | tuple[dict[str, Tensor], dict[str, Tensor]]:
        # augmented_batch tensors:
        # peak_mz/peak_intensity/context_mask/peak_valid_mask: [B, N]
        # target_masks: [B, K, N], precursor_mz: [B] when present.
        if self.training_mode == "mae":
            return self.forward_mae(
                augmented_batch,
                return_collapse_data=return_collapse_data,
                covariance_pooler=covariance_pooler,
            )

        peak_mz = augmented_batch["peak_mz"]
        peak_intensity = augmented_batch["peak_intensity"]
        peak_valid_mask = augmented_batch["peak_valid_mask"]
        precursor_mz = augmented_batch.get("precursor_mz", None)
        context_mask = augmented_batch["context_mask"] & peak_valid_mask
        target_masks = augmented_batch["target_masks"] & peak_valid_mask.unsqueeze(1)
        (
            teacher_target_features,
            teacher_peak_emb,
            teacher_cls_emb,
            context_emb,
            context_cls_emb,
        ) = self._encode_augmented_teacher_and_context(
            peak_mz,
            peak_intensity,
            peak_valid_mask,
            context_mask,
            target_masks,
            precursor_mz=precursor_mz,
        )
        predictor_output_features, predictor_output = self._predict_augmented_targets(
            context_emb,
            context_mask,
            target_masks,
            context_cls_emb=context_cls_emb,
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
        masked_prediction_term = self.masked_token_loss_weight * masked_prediction_loss
        jepa_mae_term, jepa_mae_metrics = self._jepa_mae_metrics(
            predictor_output,
            peak_mz,
            peak_intensity,
            target_masks,
            context_emb,
        )
        covariance_term, covariance_metrics = self._covariance_pooling_metrics(
            context_emb,
            context_mask,
            covariance_pooler,
        )
        loss = (
            masked_prediction_term
            + jepa_mae_term
            + covariance_term
        )
        valid_peak_count = peak_valid_mask.float().sum().clamp_min(1.0)
        collapse_data: dict[str, Tensor] = {}
        if return_collapse_data:
            pooled_mean = self.pool(teacher_peak_emb, peak_valid_mask)
            collapse_data = {
                "teacher_peak_emb": teacher_peak_emb.detach(),
                "teacher_cls_emb": teacher_cls_emb.detach(),
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
        metrics.update(covariance_metrics)
        if return_collapse_data:
            return metrics, collapse_data
        return metrics

    @overload
    def forward_mae(
        self: Any,
        augmented_batch: dict[str, Tensor],
        return_collapse_data: Literal[False] = False,
        covariance_pooler: CovariancePooler | None = None,
    ) -> dict[str, Tensor]: ...

    @overload
    def forward_mae(
        self: Any,
        augmented_batch: dict[str, Tensor],
        return_collapse_data: Literal[True],
        covariance_pooler: CovariancePooler | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]: ...

    def forward_mae(
        self: Any,
        augmented_batch: dict[str, Tensor],
        return_collapse_data: bool = False,
        covariance_pooler: CovariancePooler | None = None,
    ) -> dict[str, Tensor] | tuple[dict[str, Tensor], dict[str, Tensor]]:
        # augmented_batch tensors:
        # peak_mz/peak_intensity/context_mask/peak_valid_mask: [B, N]
        # target_masks: [B, K, N], precursor_mz: [B] when present.
        peak_mz = augmented_batch["peak_mz"]
        peak_intensity = augmented_batch["peak_intensity"]
        peak_valid_mask = augmented_batch["peak_valid_mask"]
        precursor_mz = augmented_batch.get("precursor_mz", None)
        context_mask = augmented_batch["context_mask"] & peak_valid_mask
        target_masks = augmented_batch["target_masks"] & peak_valid_mask.unsqueeze(1)

        context_mz, context_intensity, context_visible_mask = self._context_encoder_inputs(
            peak_mz,
            peak_intensity,
            context_mask,
            target_masks,
        )

        context_encoded = self.encoder(
            context_mz,
            context_intensity,
            valid_mask=peak_valid_mask,
            visible_mask=context_visible_mask,
            precursor_mz=precursor_mz,
        )
        context_emb, context_cls_emb = self._split_encoder_output(
            self.encoder,
            context_encoded,
            peak_valid_mask,
        )
        predictor_output_features, predictor_output = self._predict_augmented_targets(
            context_emb,
            context_mask,
            target_masks,
            context_cls_emb=(
                context_cls_emb if self.encoder.use_cls_token else None
            ),
        )
        mae_term, mae_metrics = self._mae_metrics(
            predictor_output,
            peak_mz,
            peak_intensity,
            target_masks,
            context_emb,
        )
        covariance_term, covariance_metrics = self._covariance_pooling_metrics(
            context_emb,
            context_visible_mask,
            covariance_pooler,
        )
        loss = mae_term + covariance_term
        valid_peak_count = peak_valid_mask.float().sum().clamp_min(1.0)
        metrics = {
            "loss": loss,
            "context_fraction": context_mask.float().sum() / valid_peak_count,
        }
        metrics.update(self._target_mask_metrics(target_masks, valid_peak_count))
        metrics.update(mae_metrics)
        metrics.update(covariance_metrics)
        if return_collapse_data:
            return metrics, {}
        return metrics

    def encode(
        self: Any,
        batch: dict[str, Tensor],
    ) -> Float[Tensor, "batch dim"] | Float[Tensor, "batch cls_tokens dim"]:
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
        )
        peak_x, cls_x = self._split_encoder_output(self.encoder, encoded, valid)
        return cls_x
