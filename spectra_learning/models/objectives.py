from __future__ import annotations

from typing import Any, Protocol, cast

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn


class CovariancePooler(Protocol):
    def reconstruction_loss(
        self,
        peak_embeddings: Float[Tensor, "batch peaks dim"],
        valid_mask: Bool[Tensor, "batch peaks"],
    ) -> Float[Tensor, ""]: ...


class ObjectiveMixin:
    def _embedding_loss(
        self: Any,
        prediction: Float[Tensor, "*batch dim"],
        target: Float[Tensor, "*batch dim"],
    ) -> Float[Tensor, "*batch"]:
        prediction = prediction.float()
        target = target.float()
        return (prediction - target).square().mean(dim=-1)

    def _jepa_mae_targets(
        self: Any,
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
        self: Any,
        logits: Float[Tensor, "... classes"],
        targets: Int[Tensor, "..."],
        valid_mask: Bool[Tensor, "..."],
    ) -> Float[Tensor, ""]:
        per_token = F.cross_entropy(
            logits.flatten(0, -2).float(),
            targets.reshape(-1),
            reduction="none",
        ).reshape_as(valid_mask)
        weights = valid_mask.float()
        return (per_token * weights).sum() / weights.sum().clamp_min(1.0)

    def _jepa_mae_value_prediction_loss(
        self: Any,
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
        if self.masked_token_input_mode == "mz_sentinel":
            zero = mz_loss.new_zeros(())
            return mz_loss, mz_loss, zero, mz_accuracy, zero

        intensity_logits = cast(nn.Linear, self.jepa_mae_intensity_head)(
            predicted_latents
        )
        intensity_loss = self._masked_ce_loss(
            intensity_logits,
            intensity_target,
            target_masks,
        )
        value_loss = mz_loss + intensity_loss
        intensity_accuracy = (
            (intensity_logits.argmax(dim=-1) == intensity_target).float()
            * target_weights.float()
        ).sum() / target_weights.float().sum().clamp_min(1.0)
        return value_loss, mz_loss, intensity_loss, mz_accuracy, intensity_accuracy

    def _predict_augmented_targets(
        self: Any,
        context_emb: Float[Tensor, "batch peaks dim"],
        context_mask: Bool[Tensor, "batch peaks"],
        target_masks: Bool[Tensor, "batch views peaks"],
    ) -> tuple[
        Float[Tensor, "batch views peaks target_dim"],
        Float[Tensor, "batch views peaks target_dim"],
    ]:
        batch_size, num_target_blocks, num_peaks = target_masks.shape
        context_mask_by_view = context_mask.unsqueeze(1)
        # predictor_input: [B, K, N, D]
        predictor_input = (
            context_emb.unsqueeze(1).expand(-1, num_target_blocks, -1, -1)
            * context_mask_by_view.unsqueeze(-1)
        )
        if self.masked_token_input_mode == "mz_sentinel":
            predictor_input = torch.where(
                target_masks.unsqueeze(-1),
                context_emb.unsqueeze(1).expand(-1, num_target_blocks, -1, -1),
                predictor_input,
            )
        else:
            predictor_input = torch.where(
                target_masks.unsqueeze(-1),
                self.latent_mask_token.view(1, 1, 1, -1).to(context_emb),
                predictor_input,
            )
        predictor_visible_mask = (context_mask_by_view | target_masks).reshape(
            batch_size,
            num_target_blocks,
            num_peaks,
        )
        predictor_visible_mask = predictor_visible_mask.reshape(
            batch_size * num_target_blocks,
            predictor_visible_mask.shape[2],
        )
        # Flatten target views into the batch: [B, K, T, D] -> [B*K, T, D].
        flat_predictor_input = predictor_input.reshape(
            batch_size * num_target_blocks,
            predictor_input.shape[2],
            -1,
        )
        predictor_features = self.predict_masked_target_features(
            flat_predictor_input,
            predictor_visible_mask,
        )
        predictor_features = predictor_features.reshape(
            batch_size,
            num_target_blocks,
            predictor_input.shape[2],
            -1,
        )
        predictor_features = predictor_features[:, :, :num_peaks]
        predictor_output = self.project_targets(predictor_features)
        return predictor_features, predictor_output

    def _masked_prediction_loss(
        self: Any,
        predictor_output: Float[Tensor, "batch views peaks target_dim"],
        teacher_targets: Float[Tensor, "batch peaks target_dim"],
        target_masks: Bool[Tensor, "batch views peaks"],
    ) -> Float[Tensor, ""]:
        per_token = self._embedding_loss(predictor_output, teacher_targets.unsqueeze(1))
        target_weights = target_masks.float()
        return (per_token * target_weights).sum() / target_weights.sum().clamp_min(1.0)

    def _jepa_mae_metrics(
        self: Any,
        predictor_output: Float[Tensor, "batch views peaks target_dim"],
        peak_mz: Float[Tensor, "batch peaks"],
        peak_intensity: Float[Tensor, "batch peaks"],
        target_masks: Bool[Tensor, "batch views peaks"],
        reference: Float[Tensor, "*batch dim"],
    ) -> tuple[Float[Tensor, ""], dict[str, Tensor]]:
        if self.jepa_mae_loss_weight <= 0:
            return reference.new_tensor(0.0), {}
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
        loss_weight = reference.new_tensor(self.jepa_mae_loss_weight)
        term = loss_weight * value_loss.to(dtype=reference.dtype)
        return term, {
            "jepa_mae_loss": value_loss.to(dtype=reference.dtype),
            "jepa_mae_term": term,
        }

    def _mae_metrics(
        self: Any,
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
        loss_weight = reference.new_tensor(self.mae_loss_weight)
        term = loss_weight * value_loss.to(dtype=reference.dtype)
        return term, {
            "mae_loss": value_loss.to(dtype=reference.dtype),
            "mae_term": term,
            "mae_mz_loss": mz_loss.to(dtype=reference.dtype),
            "mae_intensity_loss": intensity_loss.to(dtype=reference.dtype),
            "mae_mz_accuracy": mz_accuracy.to(dtype=reference.dtype),
            "mae_intensity_accuracy": intensity_accuracy.to(dtype=reference.dtype),
        }

    def _covariance_pooling_metrics(
        self: Any,
        embeddings: Float[Tensor, "batch peaks dim"],
        valid_mask: Bool[Tensor, "batch peaks"],
        covariance_pooler: CovariancePooler | None = None,
    ) -> tuple[Float[Tensor, ""], dict[str, Tensor]]:
        if (
            not self.train_covariance_pooling
            or self.covariance_pooling_loss_weight <= 0
            or covariance_pooler is None
        ):
            return embeddings.new_tensor(0.0), {}

        covariance_loss = covariance_pooler.reconstruction_loss(embeddings, valid_mask)

        term = embeddings.new_tensor(self.covariance_pooling_loss_weight) * (
            covariance_loss.to(dtype=embeddings.dtype)
        )
        return term, {
            "covariance_pooling_loss": covariance_loss.to(dtype=embeddings.dtype),
            "covariance_pooling_term": term,
        }

    def pool(
        self: Any,
        embeddings: Float[Tensor, "batch tokens dim"],
        valid_mask: Bool[Tensor, "batch peaks"],
    ) -> Float[Tensor, "batch dim"]:
        num_extra_tokens = embeddings.shape[1] - valid_mask.shape[1]
        if num_extra_tokens > 0:
            embeddings = embeddings[:, : valid_mask.shape[1]]
        mask = valid_mask.unsqueeze(-1).float()
        return (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
