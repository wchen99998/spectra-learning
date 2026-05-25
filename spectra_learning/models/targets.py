from __future__ import annotations

from typing import Any

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from spectra_learning.models.common import _active_autocast_context
from spectra_learning.models.transformer import create_visible_attention_mask


class TargetProjectionMixin:
    def _apply_group_target_normalization(
        self: Any,
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
        self: Any,
        x: Float[Tensor, "batch peaks dim"],
    ) -> Float[Tensor, "batch peaks dim"]:
        return self._apply_group_target_normalization(x, self.jepa_target_group_dim)

    def _add_predictor_positions(
        self: Any,
        x: Float[Tensor, "batch tokens dim"],
    ) -> Float[Tensor, "batch tokens dim"]:
        positions = torch.arange(x.shape[1], device=x.device)
        return x + self.predictor_position_embedding(positions).to(dtype=x.dtype)

    def predict_masked_latents(
        self: Any,
        x: Float[Tensor, "batch tokens dim"],
        visible_mask: Bool[Tensor, "batch tokens"],
    ) -> Float[Tensor, "batch tokens dim"]:
        x = self._add_predictor_positions(x)
        x = self.encoder_to_predictor_proj(x)
        if len(self.masked_latent_predictor) > 0:
            predictor_attn_mask = create_visible_attention_mask(visible_mask)
            for block in self.masked_latent_predictor:
                x = block(
                    x,
                    attn_mask=predictor_attn_mask,
                )
        x = self.predictor_final_norm(x)
        return x

    def project_targets(
        self: Any,
        x: Float[Tensor, "*batch dim"],
    ) -> Float[Tensor, "*batch target_dim"]:
        return self.target_projector(x)

    def project_teacher_targets(
        self: Any,
        x: Float[Tensor, "*batch dim"],
    ) -> Float[Tensor, "*batch target_dim"]:
        projector = (
            self.teacher_target_projector
            if self.teacher_target_projector is not None
            else self.target_projector
        )
        return projector(x)

    def predict_masked_target_features(
        self: Any,
        x: Float[Tensor, "batch tokens dim"],
        visible_mask: Bool[Tensor, "batch tokens"],
    ) -> Float[Tensor, "batch tokens target_dim"]:
        return self.masked_latent_readout(
            self.predict_masked_latents(
                x,
                visible_mask,
            )
        )

    def predict_masked_targets(
        self: Any,
        x: Float[Tensor, "batch tokens dim"],
        visible_mask: Bool[Tensor, "batch tokens"],
    ) -> Float[Tensor, "batch tokens target_dim"]:
        return self.project_targets(
            self.predict_masked_target_features(
                x,
                visible_mask,
            )
        )

    def _compute_jepa_teacher_target_features(
        self: Any,
        peak_mz: Float[Tensor, "batch peaks"],
        peak_intensity: Float[Tensor, "batch peaks"],
        peak_valid_mask: Bool[Tensor, "batch peaks"],
        precursor_mz: Float[Tensor, "batch"] | None = None,
    ) -> Float[Tensor, "batch peaks target_dim"]:
        with _active_autocast_context(peak_mz.device.type):
            teacher_encoder = (
                self.teacher_encoder
                if self.teacher_encoder is not None
                else self.encoder
            )
            teacher_peak_outputs = teacher_encoder.forward_peak_block_outputs(
                peak_mz,
                peak_intensity,
                valid_mask=peak_valid_mask,
                visible_mask=peak_valid_mask,
                block_indices=self.jepa_target_layers,
                precursor_mz=precursor_mz,
            )
            # teacher_peak_outputs: target_layers * [B, N, D] -> [B, N, L*D]
            return torch.cat(teacher_peak_outputs, dim=-1)

    def _compute_jepa_teacher_targets(
        self: Any,
        peak_mz: Float[Tensor, "batch peaks"],
        peak_intensity: Float[Tensor, "batch peaks"],
        peak_valid_mask: Bool[Tensor, "batch peaks"],
        precursor_mz: Float[Tensor, "batch"] | None = None,
    ) -> Float[Tensor, "batch peaks target_dim"]:
        with torch.no_grad():
            teacher_target_features = self._compute_jepa_teacher_target_features(
                peak_mz,
                peak_intensity,
                peak_valid_mask,
                precursor_mz=precursor_mz,
            )
            return self.project_teacher_targets(
                self._apply_jepa_target_normalization(teacher_target_features)
            )

    def _context_encoder_inputs(
        self: Any,
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
        self: Any,
        augmented_batch: dict[str, Tensor],
    ) -> Float[Tensor, "batch peaks target_dim"]:
        return self._compute_jepa_teacher_targets(
            augmented_batch["peak_mz"],
            augmented_batch["peak_intensity"],
            augmented_batch["peak_valid_mask"],
            precursor_mz=augmented_batch.get("precursor_mz", None),
        )

    def _encode_augmented_teacher_and_context(
        self: Any,
        peak_mz: Float[Tensor, "batch peaks"],
        peak_intensity: Float[Tensor, "batch peaks"],
        peak_valid_mask: Bool[Tensor, "batch peaks"],
        context_mask: Bool[Tensor, "batch peaks"],
        target_masks: Bool[Tensor, "batch views peaks"],
        precursor_mz: Float[Tensor, "batch"] | None = None,
    ) -> tuple[
        Float[Tensor, "batch peaks target_dim"],
        Float[Tensor, "batch peaks dim"],
        Float[Tensor, "batch peaks dim"],
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
                teacher_encoded, teacher_peak_outputs = (
                    self.teacher_encoder.forward_with_block_outputs(
                        peak_mz,
                        peak_intensity,
                        valid_mask=peak_valid_mask,
                        visible_mask=peak_valid_mask,
                        block_indices=self.jepa_target_layers,
                        precursor_mz=precursor_mz,
                    )
                )
            context_encoded = self.encoder(
                context_mz,
                context_intensity,
                valid_mask=peak_valid_mask,
                visible_mask=context_visible_mask,
                precursor_mz=precursor_mz,
            )
            teacher_target_features = torch.cat(teacher_peak_outputs, dim=-1)
            return (
                teacher_target_features,
                teacher_encoded,
                context_encoded,
            )
        encoded, teacher_peak_outputs = self.encoder.forward_with_block_outputs(
            torch.cat([peak_mz, context_mz], dim=0),
            torch.cat([peak_intensity, context_intensity], dim=0),
            valid_mask=torch.cat([peak_valid_mask, peak_valid_mask], dim=0),
            visible_mask=torch.cat([peak_valid_mask, context_visible_mask], dim=0),
            block_indices=self.jepa_target_layers,
            precursor_mz=(
                None
                if precursor_mz is None
                else torch.cat([precursor_mz, precursor_mz], dim=0)
            ),
            )
        teacher_target_features = torch.cat(
            [peak_output[:batch_size] for peak_output in teacher_peak_outputs],
            dim=-1,
        )
        return (
            teacher_target_features,
            encoded[:batch_size],
            encoded[batch_size:],
        )

    def _compute_pooled_teacher_peak_targets(
        self: Any,
        peak_mz: Float[Tensor, "batch peaks"],
        peak_intensity: Float[Tensor, "batch peaks"],
        peak_valid_mask: Bool[Tensor, "batch peaks"],
        visible_mask: Bool[Tensor, "batch peaks"] | None = None,
        precursor_mz: Float[Tensor, "batch"] | None = None,
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
            )
        return self.pool(teacher_encoded, visible_mask)
