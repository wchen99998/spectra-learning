from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
import torch
from flax import nnx

from spectra_learning.models.common_jax import (
    Array,
    Identity,
    Linear,
    RMSNorm,
    activation_checkpoint_policy,
    assign_param,
    gelu,
    resolve_jax_compute_dtype,
    should_activation_checkpoint,
)
from spectra_learning.models.encoder_jax import PeakSetEncoder
from spectra_learning.models.pairmixer_jax import (
    SUPPORTED_PAIRMIXER_TRANSITION_TYPES,
    _active_indices,
    _gather_single,
)
from spectra_learning.models.peak_features_jax import PeakFeatureEmbedder
from spectra_learning.models.settings import (
    PeakSetJEPASettings,
    ema_teacher_momentum_at as resolve_ema_teacher_momentum,
    load_frozen_teacher_settings,
)
from spectra_learning.models.spectrum_metadata import jax_spectrum_metadata_from_batch
from spectra_learning.models.transformer_jax import CrossAttentionBlock


class TargetProjector(nnx.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        compute_dtype: object = jnp.float32,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        rngs = nnx.Rngs(0) if rngs is None else rngs
        self.linear0 = Linear(in_dim, in_dim, compute_dtype=compute_dtype, rngs=rngs)
        self.linear2 = Linear(in_dim, out_dim, compute_dtype=compute_dtype, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        return self.linear2(gelu(self.linear0(x)))

    def load_torch_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
    ) -> None:
        self.linear0.load_torch_state_dict(state_dict, f"{prefix}.0")
        self.linear2.load_torch_state_dict(state_dict, f"{prefix}.2")


class PeakSetJEPAJax(nnx.Module):
    def __init__(
        self,
        settings: PeakSetJEPASettings | None = None,
        *,
        rngs: nnx.Rngs | None = None,
        **overrides: Any,
    ) -> None:
        rngs = nnx.Rngs(0) if rngs is None else rngs
        cfg = PeakSetJEPASettings.create(settings, **overrides)
        frozen_teacher_cfg = load_frozen_teacher_settings(cfg)
        self.settings = cfg
        self.training_mode = cfg.training_mode.lower()
        self.model_dim = cfg.model_dim
        self.encoder_use_cls_token = cfg.encoder_use_cls_token
        self.predictor_dim = cfg.predictor_dim if cfg.predictor_dim is not None else cfg.model_dim
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
        self.use_fastmixer = self.pairmixer_block_type in {
            "fastmixer",
            "fastmixer-dense",
        }
        self.pairmixer_fast_max_visible_tokens = cfg.pairmixer_fast_max_visible_tokens
        self.pairmixer_fast_target_max_visible_tokens = (
            cfg.predictor_target_max_tokens
            if cfg.predictor_target_max_tokens is not None
            else cfg.num_peaks
        )
        self.pairmixer_fast_encoder_max_visible_tokens = (
            cfg.pairmixer_fast_encoder_max_visible_tokens
            if cfg.pairmixer_fast_encoder_max_visible_tokens is not None
            else cfg.pairmixer_fast_max_visible_tokens
        )
        self.pairmixer_transition_type = cfg.pairmixer_transition_type.lower()
        if self.pairmixer_transition_type not in SUPPORTED_PAIRMIXER_TRANSITION_TYPES:
            raise ValueError(
                "pairmixer_transition_type must be one of ('swiglu', 'feedforward')"
            )
        self.encoder_num_layers = cfg.encoder_num_layers
        self.norm_eps = cfg.norm_eps
        self.jepa_num_target_blocks = cfg.jepa_num_target_blocks
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
        self.jepa_target_dim = self.teacher_model_dim
        raw_target_projector_dim = (
            self.teacher_model_dim if cfg.target_projector_dim is None else cfg.target_projector_dim
        )
        self.use_target_projector = raw_target_projector_dim >= 0
        self.target_projector_dim = (
            raw_target_projector_dim if self.use_target_projector else self.jepa_target_dim
        )
        self.jepa_target_group_dim = self.teacher_model_dim
        self.jepa_target_normalization = cfg.jepa_target_normalization.lower()
        self.masked_token_input_mode = cfg.masked_token_input_mode.lower()
        self.masked_mz_sentinel = cfg.masked_mz_sentinel
        self.mae_loss_weight = cfg.mae_loss_weight
        self.masked_token_loss_weight = (
            0.0 if self.training_mode == "mae" else cfg.masked_token_loss_weight
        )
        self.jepa_mae_loss_weight = (
            0.0 if self.training_mode == "mae" else cfg.jepa_mae_loss_weight
        )
        if cfg.distogram_loss_weight > 0 or cfg.latent_pair_loss_weight > 0:
            raise ValueError(
                "pair prediction losses are unavailable with the cross-attention predictor"
            )
        self.jepa_mae_mz_bin_size = cfg.jepa_mae_mz_bin_size
        self.jepa_mae_intensity_bin_size = cfg.jepa_mae_intensity_bin_size
        self.mae_intensity_loss_weight = cfg.mae_intensity_loss_weight
        self.jepa_mae_mz_max = cfg.jepa_mae_mz_max
        self.jepa_mae_intensity_max = cfg.jepa_mae_intensity_max
        self.jepa_mae_num_mz_bins = math.ceil(
            self.jepa_mae_mz_max / self.jepa_mae_mz_bin_size
        )
        self.jepa_mae_num_intensity_bins = math.ceil(
            self.jepa_mae_intensity_max / self.jepa_mae_intensity_bin_size
        )
        self.num_peak_tokens = cfg.num_peaks
        self.predictor_target_max_tokens = (
            cfg.predictor_target_max_tokens
            if cfg.predictor_target_max_tokens is not None
            else self.num_peak_tokens
        )
        self.activation_checkpoint_mode = cfg.activation_checkpoint_mode.lower()
        self.activation_checkpoint_every_n_layers = cfg.activation_checkpoint_every_n_layers
        self.activation_checkpoint_modules = cfg.activation_checkpoint_modules
        self.compute_dtype = resolve_jax_compute_dtype(cfg.autocast_dtype)

        self.encoder = self._build_encoder(cfg, rngs)
        self.teacher_encoder = (
            self._build_encoder(frozen_teacher_cfg or cfg, rngs)
            if self.use_ema_teacher or self.use_frozen_teacher
            else None
        )
        if self.teacher_encoder is not None:
            teacher_cfg = frozen_teacher_cfg or cfg
            teacher_full_tokens = teacher_cfg.num_peaks + int(
                teacher_cfg.encoder_use_cls_token
            )
            self.teacher_encoder.pairmixer_fast_max_visible_tokens = (
                teacher_full_tokens
            )
            for block in self.teacher_encoder.blocks:
                block.fastmixer_max_visible_tokens = teacher_full_tokens
        self.latent_mask_token = nnx.Param(
            rngs.params.normal((self.predictor_dim,), dtype=jnp.float32) * 0.02
        )
        self.encoder_to_predictor_proj = (
            Linear(
                self.model_dim,
                self.predictor_dim,
                bias=False,
                compute_dtype=self.compute_dtype,
                rngs=rngs,
            )
            if self.predictor_dim != self.model_dim
            else Identity()
        )
        predictor_blocks = []
        for _ in range(cfg.masked_latent_predictor_num_layers):
            block = CrossAttentionBlock(
                dim=self.predictor_dim,
                n_heads=cfg.masked_latent_predictor_num_heads,
                norm_eps=self.norm_eps,
                hidden_dim=math.ceil(
                    self.predictor_dim * cfg.attention_mlp_multiple
                ),
                max_sequence_length=(
                    self.num_peak_tokens + int(self.encoder_use_cls_token)
                ),
                compute_dtype=self.compute_dtype,
                rngs=rngs,
            )
            predictor_blocks.append(block)
        self.masked_latent_predictor = nnx.List(predictor_blocks)
        self.predictor_final_norm = (
            RMSNorm(self.predictor_dim, eps=self.norm_eps, affine=False)
            if cfg.predictor_apply_final_norm
            else None
        )
        self.masked_latent_readout = Linear(
            self.predictor_dim,
            self.jepa_target_dim,
            compute_dtype=self.compute_dtype,
            rngs=rngs,
        )
        self.target_projector = (
            TargetProjector(
                self.jepa_target_dim,
                self.target_projector_dim,
                compute_dtype=self.compute_dtype,
                rngs=rngs,
            )
            if self.use_target_projector
            else Identity()
        )
        self.teacher_target_projector = (
            TargetProjector(
                self.jepa_target_dim,
                self.target_projector_dim,
                compute_dtype=self.compute_dtype,
                rngs=rngs,
            )
            if (self.use_ema_teacher or self.use_frozen_teacher)
            and self.use_target_projector
            else None
        )
        self.jepa_mae_mz_head = (
            Linear(
                self.target_projector_dim,
                self.jepa_mae_num_mz_bins,
                compute_dtype=self.compute_dtype,
                rngs=rngs,
            )
            if self.jepa_mae_loss_weight > 0 or self.training_mode == "mae"
            else None
        )
        self.jepa_mae_intensity_head = (
            Linear(
                self.target_projector_dim,
                self.jepa_mae_num_intensity_bins,
                compute_dtype=self.compute_dtype,
                rngs=rngs,
            )
            if (
                (self.jepa_mae_loss_weight > 0 or self.training_mode == "mae")
                and self.mae_intensity_loss_weight > 0.0
            )
            else None
        )

    def _build_encoder(self, cfg: PeakSetJEPASettings, rngs: nnx.Rngs) -> PeakSetEncoder:
        return PeakSetEncoder(
            model_dim=cfg.model_dim,
            embedder=PeakFeatureEmbedder(
                model_dim=cfg.model_dim,
                hidden_dim=cfg.feature_mlp_hidden_dim,
                fourier_mlp_hidden_dim=cfg.encoder_fourier_mlp_hidden_dim,
                fourier_mlp_num_layers=cfg.encoder_fourier_mlp_num_layers,
                fourier_x_min=cfg.encoder_fourier_x_min,
                fourier_x_max=cfg.encoder_fourier_x_max,
                fourier_num_freqs=cfg.encoder_fourier_num_freqs,
                mz_scale=cfg.encoder_mz_scale,
                mz_embedding=cfg.encoder_mz_embedding,
                token_bin_size=cfg.encoder_mz_token_bin_size,
                token_embedding_dim=cfg.encoder_mz_token_embedding_dim,
                compute_dtype=self.compute_dtype,
                rngs=rngs,
            ),
            num_layers=cfg.encoder_num_layers,
            num_heads=cfg.encoder_num_heads,
            attention_mlp_multiple=cfg.attention_mlp_multiple,
            norm_eps=cfg.norm_eps,
            use_position_embedding=cfg.encoder_use_position_embedding,
            apply_final_norm=cfg.encoder_apply_final_norm,
            apply_final_pair_norm=cfg.encoder_apply_final_pair_norm,
            num_peaks=cfg.num_peaks,
            use_cls_token=cfg.encoder_use_cls_token,
            pairmixer_block_type=self.pairmixer_block_type,
            pairmixer_transition_type=cfg.pairmixer_transition_type.lower(),
            pair_dim=cfg.pairmixer_pair_dim,
            pair_feature_hidden_dim=cfg.pairmixer_pair_feature_hidden_dim,
            pairmixer_dropout=cfg.pairmixer_dropout,
            pairmixer_use_pair_bias=cfg.pairmixer_use_pair_bias,
            pairmixer_mz_scale=cfg.pairmixer_mz_scale,
            pairmixer_precursor_mz_scale=cfg.pairmixer_precursor_mz_scale,
            pairmixer_mz_embedding=cfg.pairmixer_mz_embedding,
            pairmixer_mz_token_bin_size=cfg.pairmixer_mz_token_bin_size,
            pairmixer_mz_token_embedding_dim=cfg.pairmixer_mz_token_embedding_dim,
            pairmixer_fourier_num_freqs=cfg.pairmixer_fourier_num_freqs,
            pairmixer_fourier_x_min=cfg.pairmixer_fourier_x_min,
            pairmixer_fourier_x_max=cfg.pairmixer_fourier_x_max,
            pairmixer_relative_fourier_x_min=cfg.pairmixer_relative_fourier_x_min,
            pairmixer_relative_fourier_x_max=cfg.pairmixer_relative_fourier_x_max,
            pairmixer_fast_max_visible_tokens=(
                self.pairmixer_fast_encoder_max_visible_tokens
            ),
            activation_checkpoint_mode=cfg.activation_checkpoint_mode,
            activation_checkpoint_every_n_layers=(
                cfg.activation_checkpoint_every_n_layers
            ),
            activation_checkpoint_modules=cfg.activation_checkpoint_modules,
            compute_dtype=self.compute_dtype,
            rngs=rngs,
        )

    def set_fastmixer_capacities(
        self,
        encoder_tokens: int,
        predictor_tokens: int,
        target_tokens: int,
    ) -> None:
        self.pairmixer_fast_encoder_max_visible_tokens = encoder_tokens
        self.pairmixer_fast_max_visible_tokens = predictor_tokens
        self.pairmixer_fast_target_max_visible_tokens = target_tokens
        self.encoder.pairmixer_fast_max_visible_tokens = encoder_tokens
        for block in self.encoder.blocks:
            block.fastmixer_max_visible_tokens = encoder_tokens

    def __call__(
        self,
        augmented_batch: dict[str, Array],
        *,
        return_collapse_data: bool = False,
        loss_only: bool = False,
    ) -> dict[str, Array] | tuple[dict[str, Array], dict[str, Array]]:
        if self.training_mode == "mae":
            return self.forward_mae(
                augmented_batch,
                return_collapse_data=return_collapse_data,
                loss_only=loss_only,
            )
        return self.forward_augmented(
            augmented_batch,
            return_collapse_data=return_collapse_data,
            loss_only=loss_only,
        )

    @staticmethod
    def _target_mask_metrics(target_masks: Array, valid_peak_count: Array) -> dict[str, Array]:
        target_entries = target_masks.astype(jnp.float32).sum()
        target_union = jnp.any(target_masks, axis=1).astype(jnp.float32).sum()
        target_overlap_entries = target_entries - target_union
        per_view_denominator = valid_peak_count * max(target_masks.shape[1], 1)
        return {
            "target_fraction": target_entries / per_view_denominator,
            "target_fraction_per_view": target_entries / per_view_denominator,
            "target_union_fraction": target_union / valid_peak_count,
            "target_entry_fraction": target_entries / valid_peak_count,
            "target_overlap_entries": target_overlap_entries,
        }

    def forward_augmented(
        self,
        augmented_batch: dict[str, Array],
        *,
        return_collapse_data: bool = False,
        loss_only: bool = False,
    ) -> dict[str, Array] | tuple[dict[str, Array], dict[str, Array]]:
        if (
            self.use_fastmixer
            and self.teacher_encoder is not None
            and self.masked_token_loss_weight > 0.0
            and self.jepa_mae_loss_weight <= 0.0
            and not return_collapse_data
        ):
            return self._forward_augmented_fastmixer_target_only(
                augmented_batch,
                loss_only=loss_only,
            )
        peak_mz = augmented_batch["peak_mz"]
        peak_intensity = augmented_batch["peak_intensity"]
        peak_valid_mask = augmented_batch["peak_valid_mask"]
        precursor_mz = augmented_batch.get("precursor_mz", None)
        spectrum_metadata = jax_spectrum_metadata_from_batch(augmented_batch)
        context_mask = augmented_batch["context_mask"] & peak_valid_mask
        target_masks = augmented_batch["target_masks"] & peak_valid_mask[:, None, :]
        (
            teacher_target_features,
            teacher_peak_emb,
            context_emb,
        ) = self._encode_augmented_teacher_and_context(
            peak_mz,
            peak_intensity,
            peak_valid_mask,
            context_mask,
            target_masks,
            precursor_mz=precursor_mz,
            spectrum_metadata=spectrum_metadata,
        )
        predictor_output_features, predictor_output = (
            self._predict_augmented_target_outputs(
                context_emb,
                context_mask,
                target_masks,
            )
        )
        teacher_target_features_normalized = self._apply_jepa_target_normalization(
            jax.lax.stop_gradient(teacher_target_features)
        )
        teacher_targets = jax.lax.stop_gradient(
            self.project_teacher_targets(teacher_target_features_normalized)
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
        loss = masked_prediction_term + jepa_mae_term
        if loss_only:
            return {"loss": loss}
        valid_peak_count = jnp.maximum(peak_valid_mask.astype(jnp.float32).sum(), 1.0)
        collapse_data: dict[str, Array] = {}
        if return_collapse_data:
            pooled_mean = self.pool(teacher_peak_emb, peak_valid_mask)
            collapse_data = {
                "teacher_peak_emb": teacher_peak_emb,
                "context_emb": context_emb,
                "context_mask": context_mask,
                "peak_valid_mask": peak_valid_mask,
                "target_masks": target_masks,
                "teacher_target_features": teacher_target_features,
                "teacher_target_features_normalized": (
                    teacher_target_features_normalized
                ),
                "teacher_targets": teacher_targets,
                "predictor_output_features": predictor_output_features,
                "predictor_output": predictor_output,
                "pooled_mean": pooled_mean,
            }
        metrics = {
            "loss": loss,
            "masked_prediction_loss": masked_prediction_loss,
            "masked_prediction_term": masked_prediction_term,
            "context_fraction": context_mask.astype(jnp.float32).sum()
            / valid_peak_count,
        }
        metrics.update(self._target_mask_metrics(target_masks, valid_peak_count))
        metrics.update(jepa_mae_metrics)
        if return_collapse_data:
            return metrics, collapse_data
        return metrics

    def _forward_augmented_fastmixer_target_only(
        self,
        augmented_batch: dict[str, Array],
        *,
        loss_only: bool,
    ) -> dict[str, Array]:
        peak_mz = augmented_batch["peak_mz"]
        peak_intensity = augmented_batch["peak_intensity"]
        peak_valid_mask = augmented_batch["peak_valid_mask"]
        precursor_mz = augmented_batch.get("precursor_mz", None)
        spectrum_metadata = jax_spectrum_metadata_from_batch(augmented_batch)
        context_mask = augmented_batch["context_mask"] & peak_valid_mask
        target_masks = augmented_batch["target_masks"] & peak_valid_mask[:, None, :]

        teacher_target_features = jax.lax.stop_gradient(
            self._compute_jepa_teacher_target_features(
                peak_mz,
                peak_intensity,
                peak_valid_mask,
                precursor_mz=precursor_mz,
                spectrum_metadata=spectrum_metadata,
            )
        )
        context_mz, context_intensity, context_visible_mask = (
            self._context_encoder_inputs(
                peak_mz,
                peak_intensity,
                context_mask,
                target_masks,
            )
        )
        (
            context_encoded_compact,
            _,
            enc_idx,
            enc_compact_mask,
        ) = self.encoder.forward_with_pair_compact(
            context_mz,
            context_intensity,
            valid_mask=peak_valid_mask,
            visible_mask=context_visible_mask,
            max_visible_tokens=self.pairmixer_fast_encoder_max_visible_tokens,
            precursor_mz=precursor_mz,
            spectrum_metadata=spectrum_metadata,
        )
        predictor_targets, target_positions, target_mask = (
            self._predict_augmented_target_outputs_fastmixer_compact(
                context_encoded_compact,
                enc_idx,
                enc_compact_mask,
                target_masks,
            )
        )

        batch_size, num_target_blocks, num_peaks = target_masks.shape
        teacher_features_by_view = jnp.broadcast_to(
            teacher_target_features[:, None],
            (
                batch_size,
                num_target_blocks,
                num_peaks,
                teacher_target_features.shape[-1],
            ),
        ).reshape(
            batch_size * num_target_blocks,
            num_peaks,
            teacher_target_features.shape[-1],
        )
        teacher_target_features_compact = _gather_single(
            teacher_features_by_view,
            target_positions,
        )
        teacher_targets = jax.lax.stop_gradient(
            self.project_teacher_targets(
                self._apply_jepa_target_normalization(
                    teacher_target_features_compact
                )
            )
        )
        target_weights = target_mask.astype(jnp.float32)
        per_target = self._embedding_loss(predictor_targets, teacher_targets)
        masked_prediction_loss = (per_target * target_weights).sum() / jnp.maximum(
            target_weights.sum(),
            1.0,
        )
        loss = masked_prediction_loss * self.masked_token_loss_weight
        if loss_only:
            return {"loss": loss}

        valid_peak_count = jnp.maximum(
            peak_valid_mask.astype(jnp.float32).sum(),
            1.0,
        )
        metrics = {
            "loss": loss,
            "masked_prediction_loss": masked_prediction_loss,
            "masked_prediction_term": loss,
            "context_fraction": context_mask.astype(jnp.float32).sum()
            / valid_peak_count,
        }
        metrics.update(self._target_mask_metrics(target_masks, valid_peak_count))
        return metrics

    def forward_mae(
        self,
        augmented_batch: dict[str, Array],
        *,
        return_collapse_data: bool = False,
        loss_only: bool = False,
    ) -> dict[str, Array] | tuple[dict[str, Array], dict[str, Array]]:
        if self.use_fastmixer:
            return self._forward_mae_fastmixer_compact(
                augmented_batch,
                return_collapse_data=return_collapse_data,
                loss_only=loss_only,
            )
        peak_mz = augmented_batch["peak_mz"]
        peak_intensity = augmented_batch["peak_intensity"]
        peak_valid_mask = augmented_batch["peak_valid_mask"]
        precursor_mz = augmented_batch.get("precursor_mz", None)
        spectrum_metadata = jax_spectrum_metadata_from_batch(augmented_batch)
        context_mask = augmented_batch["context_mask"] & peak_valid_mask
        target_masks = augmented_batch["target_masks"] & peak_valid_mask[:, None, :]
        context_mz, context_intensity, context_visible_mask = self._context_encoder_inputs(
            peak_mz,
            peak_intensity,
            context_mask,
            target_masks,
        )
        context_encoded = self._encode_mae_context(
            context_mz,
            context_intensity,
            peak_valid_mask,
            context_visible_mask,
            precursor_mz,
            spectrum_metadata,
        )
        _, predictor_output = (
            self._predict_augmented_target_outputs(
                context_encoded,
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
            compute_accuracy=not loss_only,
        )
        loss = mae_term
        if loss_only:
            return {"loss": loss}
        valid_peak_count = jnp.maximum(peak_valid_mask.astype(jnp.float32).sum(), 1.0)
        metrics = {
            "loss": loss,
            "context_fraction": context_mask.astype(jnp.float32).sum() / valid_peak_count,
        }
        metrics.update(self._target_mask_metrics(target_masks, valid_peak_count))
        metrics.update(mae_metrics)
        if return_collapse_data:
            return metrics, {}
        return metrics

    def _forward_mae_fastmixer_compact(
        self,
        augmented_batch: dict[str, Array],
        *,
        return_collapse_data: bool = False,
        loss_only: bool = False,
    ) -> dict[str, Array] | tuple[dict[str, Array], dict[str, Array]]:
        peak_mz = augmented_batch["peak_mz"]
        peak_intensity = augmented_batch["peak_intensity"]
        peak_valid_mask = augmented_batch["peak_valid_mask"]
        precursor_mz = augmented_batch.get("precursor_mz", None)
        spectrum_metadata = jax_spectrum_metadata_from_batch(augmented_batch)
        context_mask = augmented_batch["context_mask"] & peak_valid_mask
        target_masks = augmented_batch["target_masks"] & peak_valid_mask[:, None, :]
        context_mz, context_intensity, context_visible_mask = self._context_encoder_inputs(
            peak_mz,
            peak_intensity,
            context_mask,
            target_masks,
        )
        (
            context_encoded_compact,
            _,
            enc_idx,
            enc_compact_mask,
        ) = self.encoder.forward_with_pair_compact(
            context_mz,
            context_intensity,
            valid_mask=peak_valid_mask,
            visible_mask=context_visible_mask,
            max_visible_tokens=self.pairmixer_fast_encoder_max_visible_tokens,
            precursor_mz=precursor_mz,
            spectrum_metadata=spectrum_metadata,
        )
        (
            predictor_output_compact,
            target_positions,
            target_mask,
        ) = self._predict_augmented_target_outputs_fastmixer_compact(
            context_encoded_compact,
            enc_idx,
            enc_compact_mask,
            target_masks,
        )
        flat_peak_mz, flat_peak_intensity = self._flatten_peak_values_for_target_views(
            peak_mz,
            peak_intensity,
            target_masks.shape[1],
        )
        peak_idx = jnp.minimum(target_positions, peak_mz.shape[1] - 1)
        peak_mz_compact = jnp.take_along_axis(flat_peak_mz, peak_idx, axis=1)
        peak_intensity_compact = jnp.take_along_axis(
            flat_peak_intensity,
            peak_idx,
            axis=1,
        )
        mae_term, mae_metrics = self._mae_metrics_compact(
            predictor_output_compact,
            peak_mz_compact,
            peak_intensity_compact,
            target_mask,
            reference=context_encoded_compact,
            compute_accuracy=not loss_only,
        )
        loss = mae_term
        if loss_only:
            return {"loss": loss}
        valid_peak_count = jnp.maximum(peak_valid_mask.astype(jnp.float32).sum(), 1.0)
        metrics = {
            "loss": loss,
            "context_fraction": context_mask.astype(jnp.float32).sum() / valid_peak_count,
        }
        metrics.update(self._target_mask_metrics(target_masks, valid_peak_count))
        metrics.update(mae_metrics)
        if return_collapse_data:
            return metrics, {}
        return metrics

    def _context_encoder_inputs(
        self,
        peak_mz: Array,
        peak_intensity: Array,
        context_mask: Array,
        target_masks: Array,
    ) -> tuple[Array, Array, Array]:
        if self.masked_token_input_mode != "mz_sentinel":
            return peak_mz, peak_intensity, context_mask
        target_union = jnp.any(target_masks, axis=1)
        masked_mz = jnp.where(target_union, self.masked_mz_sentinel, peak_mz)
        return masked_mz, peak_intensity, context_mask | target_union

    def _encode_mae_context(
        self,
        context_mz: Array,
        context_intensity: Array,
        peak_valid_mask: Array,
        context_visible_mask: Array,
        precursor_mz: Array | None,
        spectrum_metadata: Array | None,
    ) -> Array:
        return self.encoder(
            context_mz,
            context_intensity,
            valid_mask=peak_valid_mask,
            visible_mask=context_visible_mask,
            precursor_mz=precursor_mz,
            spectrum_metadata=spectrum_metadata,
        )

    def _predict_masked_latents(
        self,
        query: Array,
        memory: Array,
        query_positions: Array,
        memory_positions: Array,
        query_mask: Array,
        memory_mask: Array,
    ) -> Array:
        memory = self.encoder_to_predictor_proj(memory)
        for block_idx, block in enumerate(self.masked_latent_predictor, start=1):
            if should_activation_checkpoint(
                mode=self.activation_checkpoint_mode,
                modules=self.activation_checkpoint_modules,
                module="predictor",
                block_idx=block_idx,
                every_n=self.activation_checkpoint_every_n_layers,
            ):
                query = nnx.remat(
                    _call_cross_attention_block,
                    policy=activation_checkpoint_policy(
                        self.activation_checkpoint_mode
                    ),
                )(
                    block,
                    query,
                    memory,
                    query_positions,
                    memory_positions,
                    query_mask,
                    memory_mask,
                )
            else:
                query = block(
                    query,
                    memory,
                    query_positions,
                    memory_positions,
                    query_mask,
                    memory_mask,
                )
        if self.predictor_final_norm is not None:
            query = self.predictor_final_norm(query)
        return query * query_mask[..., None].astype(query.dtype)

    def project_targets(self, x: Array) -> Array:
        return self.target_projector(x)

    def project_teacher_targets(self, x: Array) -> Array:
        projector = (
            self.teacher_target_projector
            if self.teacher_target_projector is not None
            else self.target_projector
        )
        return projector(x)

    def _apply_group_target_normalization(self, x: Array, group_dim: int) -> Array:
        if self.jepa_target_normalization == "none":
            return x
        orig_dtype = x.dtype
        x_float = x.astype(jnp.float32).reshape(*x.shape[:-1], -1, group_dim)
        mean = jnp.mean(x_float, axis=-1, keepdims=True)
        std = jnp.maximum(jnp.std(x_float, axis=-1, keepdims=True), 1e-6)
        return ((x_float - mean) / std).reshape(x.shape).astype(orig_dtype)

    def _apply_jepa_target_normalization(self, x: Array) -> Array:
        return self._apply_group_target_normalization(x, self.jepa_target_group_dim)

    def _compute_jepa_teacher_target_features(
        self,
        peak_mz: Array,
        peak_intensity: Array,
        peak_valid_mask: Array,
        *,
        precursor_mz: Array | None = None,
        spectrum_metadata: Array | None = None,
    ) -> Array:
        teacher_encoder = self.teacher_encoder if self.teacher_encoder is not None else self.encoder
        teacher_encoded, _ = teacher_encoder.forward_with_pair(
            peak_mz,
            peak_intensity,
            valid_mask=peak_valid_mask,
            visible_mask=peak_valid_mask,
            precursor_mz=precursor_mz,
            spectrum_metadata=spectrum_metadata,
        )
        return teacher_encoded[:, : peak_mz.shape[1]]

    def compute_teacher_targets(self, augmented_batch: dict[str, Array]) -> Array:
        teacher_target_features = self._compute_jepa_teacher_target_features(
            augmented_batch["peak_mz"],
            augmented_batch["peak_intensity"],
            augmented_batch["peak_valid_mask"],
            precursor_mz=augmented_batch.get("precursor_mz", None),
            spectrum_metadata=jax_spectrum_metadata_from_batch(augmented_batch),
        )
        return self.project_teacher_targets(
            self._apply_jepa_target_normalization(teacher_target_features)
        )

    def _encode_augmented_teacher_and_context(
        self,
        peak_mz: Array,
        peak_intensity: Array,
        peak_valid_mask: Array,
        context_mask: Array,
        target_masks: Array,
        *,
        precursor_mz: Array | None = None,
        spectrum_metadata: Array | None = None,
    ) -> tuple[Array, Array, Array]:
        batch_size = peak_mz.shape[0]
        context_mz, context_intensity, context_visible_mask = self._context_encoder_inputs(
            peak_mz,
            peak_intensity,
            context_mask,
            target_masks,
        )
        if self.teacher_encoder is not None:
            teacher_encoded = self.teacher_encoder(
                peak_mz,
                peak_intensity,
                valid_mask=peak_valid_mask,
                visible_mask=peak_valid_mask,
                precursor_mz=precursor_mz,
                spectrum_metadata=spectrum_metadata,
            )
            context_encoded = self.encoder(
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
                context_encoded,
            )
        encoded = self.encoder(
            jnp.concatenate([peak_mz, context_mz], axis=0),
            jnp.concatenate([peak_intensity, context_intensity], axis=0),
            valid_mask=jnp.concatenate([peak_valid_mask, peak_valid_mask], axis=0),
            visible_mask=jnp.concatenate([peak_valid_mask, context_visible_mask], axis=0),
            precursor_mz=(
                None
                if precursor_mz is None
                else jnp.concatenate([precursor_mz, precursor_mz], axis=0)
            ),
            spectrum_metadata=(
                None
                if spectrum_metadata is None
                else jnp.concatenate([spectrum_metadata, spectrum_metadata], axis=0)
            ),
        )
        return (
            encoded[:batch_size, : peak_mz.shape[1]],
            encoded[:batch_size],
            encoded[batch_size:],
        )

    def _predict_augmented_target_outputs(
        self,
        context_emb: Array,
        context_mask: Array,
        target_masks: Array,
    ) -> tuple[Array, Array]:
        batch_size, num_target_blocks, num_peaks = target_masks.shape
        memory_peak_mask = context_mask
        if self.masked_token_input_mode == "mz_sentinel":
            memory_peak_mask = memory_peak_mask | jnp.any(target_masks, axis=1)
        memory_mask = memory_peak_mask
        if self.encoder_use_cls_token:
            memory_mask = jnp.concatenate(
                [
                    memory_peak_mask,
                    jnp.ones((batch_size, 1), dtype=jnp.bool_),
                ],
                axis=1,
            )
        memory_positions, compact_memory_mask = _active_indices(
            memory_mask,
            memory_mask.shape[1],
        )
        memory = _gather_single(context_emb, memory_positions)
        compact_memory_tokens = memory.shape[1]
        flat_memory = jnp.broadcast_to(
            memory[:, None],
            (
                batch_size,
                num_target_blocks,
                compact_memory_tokens,
                memory.shape[-1],
            ),
        ).reshape(
            batch_size * num_target_blocks,
            compact_memory_tokens,
            memory.shape[-1],
        )
        flat_memory_positions = jnp.broadcast_to(
            memory_positions[:, None],
            (batch_size, num_target_blocks, compact_memory_tokens),
        ).reshape(
            batch_size * num_target_blocks,
            compact_memory_tokens,
        )
        flat_memory_mask = jnp.broadcast_to(
            compact_memory_mask[:, None],
            (batch_size, num_target_blocks, compact_memory_tokens),
        ).reshape(
            batch_size * num_target_blocks,
            compact_memory_tokens,
        )
        (
            compact_features,
            compact_output,
            target_positions,
            target_mask,
        ) = self._predict_compact_target_outputs(
            flat_memory,
            flat_memory_positions,
            flat_memory_mask,
            target_masks.reshape(batch_size * num_target_blocks, num_peaks),
        )
        flat_batch_indices = jnp.broadcast_to(
            jnp.arange(batch_size * num_target_blocks)[:, None],
            target_positions.shape,
        )
        flat_features = jnp.zeros(
            (
                batch_size * num_target_blocks,
                num_peaks,
                compact_features.shape[-1],
            ),
            dtype=compact_features.dtype,
        ).at[flat_batch_indices, target_positions].set(compact_features)
        flat_output = jnp.zeros(
            (
                batch_size * num_target_blocks,
                num_peaks,
                compact_output.shape[-1],
            ),
            dtype=compact_output.dtype,
        ).at[flat_batch_indices, target_positions].set(compact_output)
        predictor_features = flat_features.reshape(
            batch_size,
            num_target_blocks,
            num_peaks,
            -1,
        )
        predictor_output = flat_output.reshape(
            batch_size,
            num_target_blocks,
            num_peaks,
            -1,
        )
        return predictor_features, predictor_output

    @staticmethod
    def _flatten_peak_values_for_target_views(
        peak_mz: Array,
        peak_intensity: Array,
        num_target_blocks: int,
    ) -> tuple[Array, Array]:
        batch_size, num_peaks = peak_mz.shape
        view_shape = (batch_size, num_target_blocks, num_peaks)
        return (
            jnp.broadcast_to(peak_mz[:, None], view_shape).reshape(
                batch_size * num_target_blocks,
                num_peaks,
            ),
            jnp.broadcast_to(peak_intensity[:, None], view_shape).reshape(
                batch_size * num_target_blocks,
                num_peaks,
            ),
        )

    def _predict_compact_target_outputs(
        self,
        memory: Array,
        memory_positions: Array,
        memory_mask: Array,
        target_masks: Array,
    ) -> tuple[Array, Array, Array, Array]:
        target_positions, target_mask = _active_indices(
            target_masks,
            self.pairmixer_fast_target_max_visible_tokens,
        )
        query = jnp.broadcast_to(
            self.latent_mask_token[None, None, :],
            (
                target_masks.shape[0],
                target_positions.shape[1],
                self.predictor_dim,
            ),
        )
        query = query * target_mask[..., None].astype(query.dtype)
        latents = self._predict_masked_latents(
            query,
            memory,
            target_positions,
            memory_positions,
            target_mask,
            memory_mask,
        )
        features = self.masked_latent_readout(latents)
        output = self.project_targets(features)
        return features, output, target_positions, target_mask

    def _predict_augmented_target_outputs_fastmixer_compact(
        self,
        context_emb_compact: Array,
        enc_idx: Array,
        enc_compact_mask: Array,
        target_masks: Array,
    ) -> tuple[Array, Array, Array]:
        batch_size, num_target_blocks, num_peaks = target_masks.shape
        flat_batch_size = batch_size * num_target_blocks
        compact_encoder_tokens = context_emb_compact.shape[1]
        flat_memory = jnp.broadcast_to(
            context_emb_compact[:, None],
            (
                batch_size,
                num_target_blocks,
                compact_encoder_tokens,
                context_emb_compact.shape[-1],
            ),
        ).reshape(
            flat_batch_size,
            compact_encoder_tokens,
            context_emb_compact.shape[-1],
        )
        flat_memory_positions = jnp.broadcast_to(
            enc_idx[:, None],
            (batch_size, num_target_blocks, compact_encoder_tokens),
        ).reshape(
            flat_batch_size,
            compact_encoder_tokens,
        )
        flat_memory_mask = jnp.broadcast_to(
            enc_compact_mask[:, None],
            (batch_size, num_target_blocks, compact_encoder_tokens),
        ).reshape(
            flat_batch_size,
            compact_encoder_tokens,
        )
        _, output, target_positions, target_mask = (
            self._predict_compact_target_outputs(
                flat_memory,
                flat_memory_positions,
                flat_memory_mask,
                target_masks.reshape(flat_batch_size, num_peaks),
            )
        )
        return output, target_positions, target_mask

    def _embedding_loss(self, prediction: Array, target: Array) -> Array:
        return jnp.square(prediction.astype(jnp.float32) - target.astype(jnp.float32)).mean(
            axis=-1
        )

    def _masked_prediction_loss(
        self,
        predictor_output: Array,
        teacher_targets: Array,
        target_masks: Array,
    ) -> Array:
        per_token = self._embedding_loss(predictor_output, teacher_targets[:, None])
        target_weights = target_masks.astype(jnp.float32)
        return (per_token * target_weights).sum() / jnp.maximum(
            target_weights.sum(),
            1.0,
        )

    def _jepa_mae_targets(self, peak_mz: Array, peak_intensity: Array) -> tuple[Array, Array]:
        mz_target = jnp.floor(
            peak_mz.astype(jnp.float32) * self.jepa_mae_mz_max / self.jepa_mae_mz_bin_size
        ).astype(jnp.int32)
        intensity_target = jnp.floor(
            peak_intensity.astype(jnp.float32) / self.jepa_mae_intensity_bin_size
        ).astype(jnp.int32)
        return (
            jnp.clip(mz_target, 0, self.jepa_mae_num_mz_bins - 1),
            jnp.clip(intensity_target, 0, self.jepa_mae_num_intensity_bins - 1),
        )

    def _jepa_mae_value_prediction_loss(
        self,
        predicted_latents: Array,
        peak_mz: Array,
        peak_intensity: Array,
        target_masks: Array,
        *,
        compute_accuracy: bool = True,
    ) -> tuple[Array, Array, Array, Array, Array]:
        assert self.jepa_mae_mz_head is not None
        mz_logits = self.jepa_mae_mz_head(predicted_latents)
        mz_target, intensity_target = self._jepa_mae_targets(peak_mz, peak_intensity)
        view_shape = mz_logits.shape[:3]
        mz_target = jnp.broadcast_to(mz_target[:, None, :], view_shape)
        intensity_target = jnp.broadcast_to(intensity_target[:, None, :], view_shape)
        mz_loss = _masked_ce_loss_from_logits(mz_logits, mz_target, target_masks)
        target_weights = target_masks.astype(jnp.float32)
        zero = mz_loss * 0.0
        mz_accuracy = zero
        if compute_accuracy:
            mz_accuracy = (
                (
                    (jnp.argmax(mz_logits, axis=-1) == mz_target).astype(jnp.float32)
                    * target_weights
                ).sum()
                / jnp.maximum(target_weights.sum(), 1.0)
            )
        if (
            self.masked_token_input_mode == "mz_sentinel"
            or self.mae_intensity_loss_weight <= 0.0
        ):
            return mz_loss, mz_loss, zero, mz_accuracy, zero
        assert self.jepa_mae_intensity_head is not None
        intensity_logits = self.jepa_mae_intensity_head(predicted_latents)
        intensity_loss = _masked_ce_loss_from_logits(
            intensity_logits,
            intensity_target,
            target_masks,
        )
        value_loss = mz_loss + intensity_loss * self.mae_intensity_loss_weight
        intensity_accuracy = zero
        if compute_accuracy:
            intensity_accuracy = (
                (
                    (jnp.argmax(intensity_logits, axis=-1) == intensity_target).astype(jnp.float32)
                    * target_weights
                ).sum()
                / jnp.maximum(target_weights.sum(), 1.0)
            )
        return value_loss, mz_loss, intensity_loss, mz_accuracy, intensity_accuracy

    def _mae_metrics(
        self,
        predictor_output: Array,
        peak_mz: Array,
        peak_intensity: Array,
        target_masks: Array,
        reference: Array,
        *,
        compute_accuracy: bool = True,
    ) -> tuple[Array, dict[str, Array]]:
        del reference
        value_loss, mz_loss, intensity_loss, mz_accuracy, intensity_accuracy = (
            self._jepa_mae_value_prediction_loss(
                predictor_output,
                peak_mz,
                peak_intensity,
                target_masks,
                compute_accuracy=compute_accuracy,
            )
        )
        term = value_loss * self.mae_loss_weight
        return term, {
            "mae_loss": value_loss,
            "mae_term": term,
            "mae_mz_loss": mz_loss,
            "mae_intensity_loss": intensity_loss,
            "mae_mz_accuracy": mz_accuracy,
            "mae_intensity_accuracy": intensity_accuracy,
        }

    def _mae_metrics_compact(
        self,
        predictor_output_compact: Array,
        peak_mz_compact: Array,
        peak_intensity_compact: Array,
        target_slot: Array,
        reference: Array,
        *,
        compute_accuracy: bool = True,
    ) -> tuple[Array, dict[str, Array]]:
        del reference
        assert self.jepa_mae_mz_head is not None
        mz_target, intensity_target = self._jepa_mae_targets(
            peak_mz_compact,
            peak_intensity_compact,
        )
        if compute_accuracy:
            mz_logits = self.jepa_mae_mz_head(predictor_output_compact)
            mz_loss = _masked_ce_loss_from_logits(mz_logits, mz_target, target_slot)
            target_weights = target_slot.astype(jnp.float32)
            mz_accuracy = (
                (
                    (jnp.argmax(mz_logits, axis=-1) == mz_target).astype(jnp.float32)
                    * target_weights
                ).sum()
                / jnp.maximum(target_weights.sum(), 1.0)
            )
        else:
            mz_loss = _linear_head_masked_ce_loss(
                self.jepa_mae_mz_head,
                predictor_output_compact,
                mz_target,
                target_slot,
            )
            mz_accuracy = mz_loss * 0.0

        zero = mz_loss * 0.0
        if (
            self.masked_token_input_mode == "mz_sentinel"
            or self.mae_intensity_loss_weight <= 0.0
        ):
            value_loss = mz_loss
            intensity_loss = zero
            intensity_accuracy = zero
        else:
            assert self.jepa_mae_intensity_head is not None
            intensity_logits = self.jepa_mae_intensity_head(predictor_output_compact)
            intensity_loss = _masked_ce_loss_from_logits(
                intensity_logits,
                intensity_target,
                target_slot,
            )
            value_loss = mz_loss + intensity_loss * self.mae_intensity_loss_weight
            if compute_accuracy:
                target_weights = target_slot.astype(jnp.float32)
                intensity_accuracy = (
                    (
                        (
                            jnp.argmax(intensity_logits, axis=-1)
                            == intensity_target
                        ).astype(jnp.float32)
                        * target_weights
                    ).sum()
                    / jnp.maximum(target_weights.sum(), 1.0)
                )
            else:
                intensity_accuracy = zero

        term = value_loss * self.mae_loss_weight
        return term, {
            "mae_loss": value_loss,
            "mae_term": term,
            "mae_mz_loss": mz_loss,
            "mae_intensity_loss": intensity_loss,
            "mae_mz_accuracy": mz_accuracy,
            "mae_intensity_accuracy": intensity_accuracy,
        }

    def _jepa_mae_metrics(
        self,
        predictor_output: Array,
        peak_mz: Array,
        peak_intensity: Array,
        target_masks: Array,
        reference: Array,
    ) -> tuple[Array, dict[str, Array]]:
        if self.jepa_mae_loss_weight <= 0:
            return reference.reshape(-1)[0] * 0.0, {}
        value_loss, _mz_loss, _intensity_loss, _mz_accuracy, _intensity_accuracy = (
            self._jepa_mae_value_prediction_loss(
                predictor_output,
                peak_mz,
                peak_intensity,
                target_masks,
                compute_accuracy=False,
            )
        )
        term = value_loss * self.jepa_mae_loss_weight
        return term, {"jepa_mae_loss": value_loss, "jepa_mae_term": term}

    def pool(self, embeddings: Array, valid_mask: Array) -> Array:
        if embeddings.shape[1] > valid_mask.shape[1]:
            embeddings = embeddings[:, : valid_mask.shape[1]]
        mask = valid_mask[..., None].astype(embeddings.dtype)
        return (embeddings * mask).sum(axis=1) / jnp.maximum(mask.sum(axis=1), 1.0)

    def encode(self, batch: dict[str, Array]) -> Array:
        encoded = self.encoder(
            batch["peak_mz"],
            batch["peak_intensity"],
            valid_mask=batch["peak_valid_mask"],
            visible_mask=batch["peak_valid_mask"],
            precursor_mz=batch.get("precursor_mz", None),
            spectrum_metadata=jax_spectrum_metadata_from_batch(batch),
        )
        return self.pool(encoded, batch["peak_valid_mask"])

    def ema_teacher_momentum_at(self, step: int, total_steps: int) -> float:
        return resolve_ema_teacher_momentum(
            schedule=self.ema_teacher_schedule,
            start=self.ema_teacher_momentum_start,
            mid=self.ema_teacher_momentum_mid,
            final=self.ema_teacher_momentum_final,
            peak_fraction=self.ema_teacher_schedule_peak_fraction,
            step=step,
            total_steps=total_steps,
        )

    def sync_ema_teacher(self) -> None:
        if self.teacher_encoder is not None:
            _copy_param_state(self.teacher_encoder, self.encoder)
        if self.teacher_target_projector is not None:
            _copy_param_state(self.teacher_target_projector, self.target_projector)

    def update_ema_teacher(self, step: int, total_steps: int) -> float | None:
        if not self.use_ema_teacher or self.teacher_encoder is None:
            return None
        momentum = self.ema_teacher_momentum_at(step, total_steps)
        _ema_update_module(self.teacher_encoder, self.encoder, momentum)
        if self.teacher_target_projector is not None:
            _ema_update_module(
                self.teacher_target_projector,
                self.target_projector,
                momentum,
            )
        return momentum

    def load_torch_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        assign_param(self.latent_mask_token, state_dict["latent_mask_token"])
        self.encoder.load_torch_state_dict(state_dict, "encoder")
        if self.teacher_encoder is not None:
            self.teacher_encoder.load_torch_state_dict(state_dict, "teacher_encoder")
        if isinstance(self.encoder_to_predictor_proj, Linear):
            self.encoder_to_predictor_proj.load_torch_state_dict(
                state_dict,
                "encoder_to_predictor_proj",
            )
        for idx, block in enumerate(self.masked_latent_predictor):
            block.load_torch_state_dict(state_dict, f"masked_latent_predictor.{idx}")
        if self.predictor_final_norm is not None:
            self.predictor_final_norm.load_torch_state_dict(
                state_dict,
                "predictor_final_norm",
            )
        self.masked_latent_readout.load_torch_state_dict(
            state_dict,
            "masked_latent_readout",
        )
        if isinstance(self.target_projector, TargetProjector):
            self.target_projector.load_torch_state_dict(state_dict, "target_projector")
        if self.teacher_target_projector is not None:
            self.teacher_target_projector.load_torch_state_dict(
                state_dict,
                "teacher_target_projector",
            )
        if self.jepa_mae_mz_head is not None:
            self.jepa_mae_mz_head.load_torch_state_dict(state_dict, "jepa_mae_mz_head")
        if self.jepa_mae_intensity_head is not None:
            self.jepa_mae_intensity_head.load_torch_state_dict(
                state_dict,
                "jepa_mae_intensity_head",
            )
def _cross_entropy_from_logits(logits: Array, targets: Array) -> Array:
    logits = logits.astype(jnp.float32)
    classes = jnp.arange(logits.shape[-1])
    target_one_hot = (classes == targets[..., None]).astype(logits.dtype)
    return -(jax.nn.log_softmax(logits, axis=-1) * target_one_hot).sum(axis=-1)


def _masked_ce_loss_from_logits(
    logits: Array,
    targets: Array,
    valid_mask: Array,
) -> Array:
    per_token = _cross_entropy_from_logits(logits, targets)
    weights = valid_mask.astype(jnp.float32)
    return (per_token * weights).sum() / jnp.maximum(weights.sum(), 1.0)


def _linear_head_masked_ce_loss(
    linear: Linear,
    x: Array,
    targets: Array,
    valid_mask: Array,
) -> Array:
    loss_rows, weight_rows = _linear_head_masked_ce_rows(
        linear,
        x,
        targets,
        valid_mask,
    )
    return loss_rows.sum() / jnp.maximum(weight_rows.sum(), 1.0)


def _linear_head_masked_ce_rows(
    linear: Linear,
    x: Array,
    targets: Array,
    valid_mask: Array,
) -> tuple[Array, Array]:
    logits = linear(x)
    weights = valid_mask.astype(jnp.float32)
    return _cross_entropy_from_logits(logits, targets) * weights, weights


def _call_cross_attention_block(
    block: CrossAttentionBlock,
    query: Array,
    memory: Array,
    query_positions: Array,
    memory_positions: Array,
    query_mask: Array,
    memory_mask: Array,
) -> Array:
    return block(
        query,
        memory,
        query_positions,
        memory_positions,
        query_mask,
        memory_mask,
    )


def _copy_param_state(target: nnx.Module, source: nnx.Module) -> None:
    nnx.update(target, nnx.as_pure(nnx.state(source, nnx.Param)))


def _ema_update_module(
    teacher: nnx.Module,
    student: nnx.Module,
    momentum: float,
) -> None:
    teacher_state = nnx.as_pure(nnx.state(teacher, nnx.Param))
    student_state = nnx.as_pure(nnx.state(student, nnx.Param))
    updated = jax.tree.map(
        lambda teacher_value, student_value: (
            teacher_value * momentum + student_value * (1.0 - momentum)
        ),
        teacher_state,
        student_state,
    )
    nnx.update(teacher, updated)
