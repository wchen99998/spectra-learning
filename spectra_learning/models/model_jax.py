from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import torch
from flax import nnx

from spectra_learning.config import load_config
from spectra_learning.models.common_jax import (
    Array,
    Identity,
    LayerNorm,
    Linear,
    activation_checkpoint_policy,
    assign_param,
    batch_to_jax,
    build_frozen_2d_position_embedding,
    build_frozen_position_embedding,
    gelu,
    resolve_jax_compute_dtype,
    should_activation_checkpoint,
)
from spectra_learning.models.encoder_jax import PeakSetEncoder
from spectra_learning.models.pairmixer_jax import PairMixerBlock
from spectra_learning.models.peak_features_jax import PeakFeatureEmbedder
from spectra_learning.models.settings import PeakSetJEPASettings
from spectra_learning.training.checkpointing import load_torch_checkpoint


class TargetProjector(nnx.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        compute_dtype: object = jnp.float32,
    ) -> None:
        self.linear0 = Linear(in_dim, in_dim, compute_dtype=compute_dtype)
        self.linear2 = Linear(in_dim, out_dim, compute_dtype=compute_dtype)

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
        **overrides: Any,
    ) -> None:
        cfg = PeakSetJEPASettings.create(settings, **overrides)
        frozen_teacher_cfg = _load_frozen_teacher_settings(cfg)
        self.settings = cfg
        self.training_mode = cfg.training_mode.lower()
        self.model_dim = cfg.model_dim
        self.predictor_dim = cfg.predictor_dim if cfg.predictor_dim is not None else cfg.model_dim
        self.predictor_pair_dim = (
            cfg.pairmixer_pair_dim if cfg.pairmixer_pair_dim is not None else cfg.model_dim
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
        self.teacher_pair_dim = (
            _pair_dim(frozen_teacher_cfg)
            if frozen_teacher_cfg is not None
            else self.predictor_pair_dim
        )
        self.jepa_target_layers = (
            [self.teacher_encoder_num_layers]
            if self.training_mode == "mae_teacher_jepa" or cfg.jepa_target_layers is None
            else [int(layer_idx) for layer_idx in cfg.jepa_target_layers]
        )
        self.num_jepa_target_layers = len(self.jepa_target_layers)
        self.jepa_target_dim = self.num_jepa_target_layers * self.teacher_model_dim
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
        self.distogram_loss_weight = cfg.distogram_loss_weight
        self.latent_pair_loss_weight = (
            0.0 if self.training_mode == "mae" else cfg.latent_pair_loss_weight
        )
        self.latent_pair_target_normalization = cfg.latent_pair_target_normalization.lower()
        self.distogram_mz_max = cfg.distogram_mz_max
        self.distogram_loss_chunk_size = cfg.distogram_loss_chunk_size
        self.jepa_mae_mz_bin_size = cfg.jepa_mae_mz_bin_size
        self.jepa_mae_intensity_bin_size = cfg.jepa_mae_intensity_bin_size
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
        self.num_peak_tokens = cfg.num_peaks
        self.num_predictor_input_tokens = self.num_peak_tokens + 1
        self.activation_checkpoint_mode = cfg.activation_checkpoint_mode.lower()
        self.activation_checkpoint_every_n_layers = cfg.activation_checkpoint_every_n_layers
        self.activation_checkpoint_modules = cfg.activation_checkpoint_modules
        self.mae_context_encoder_pack_tokens = cfg.mae_context_encoder_pack_tokens
        self.compute_dtype = resolve_jax_compute_dtype(cfg.autocast_dtype)

        self.encoder = self._build_encoder(cfg)
        self.teacher_encoder = (
            self._build_encoder(frozen_teacher_cfg or cfg)
            if self.use_ema_teacher or self.use_frozen_teacher
            else None
        )
        self.latent_mask_token = nnx.Param(jnp.zeros((self.model_dim,), dtype=jnp.float32))
        self.pair_mask_token = nnx.Param(
            jnp.zeros((self.predictor_pair_dim,), dtype=jnp.float32)
        )
        self.encoder_to_predictor_proj = (
            Linear(
                self.model_dim,
                self.predictor_dim,
                bias=False,
                compute_dtype=self.compute_dtype,
            )
            if self.predictor_dim != self.model_dim
            else Identity()
        )
        self.predictor_position_embedding = build_frozen_position_embedding(
            self.num_predictor_input_tokens,
            self.model_dim,
        )
        self.predictor_pair_position_embedding = build_frozen_2d_position_embedding(
            self.num_predictor_input_tokens,
            self.predictor_pair_dim,
        )
        self.masked_latent_predictor = nnx.List(
            [
                PairMixerBlock(
                    single_dim=self.predictor_dim,
                    pair_dim=self.predictor_pair_dim,
                    num_heads=cfg.masked_latent_predictor_num_heads,
                    attention_mlp_multiple=cfg.attention_mlp_multiple,
                    norm_eps=self.norm_eps,
                    dropout=cfg.predictor_dropout,
                    use_pair_bias_attention=cfg.pairmixer_use_pair_bias_attention,
                    compute_dtype=self.compute_dtype,
                )
                for _ in range(cfg.masked_latent_predictor_num_layers)
            ]
        )
        self.predictor_final_norm = (
            LayerNorm(self.predictor_dim, eps=self.norm_eps, affine=False)
            if cfg.predictor_apply_final_norm
            else None
        )
        self.masked_latent_readout = Linear(
            self.predictor_dim,
            self.jepa_target_dim,
            compute_dtype=self.compute_dtype,
        )
        self.masked_pair_readout = (
            Linear(
                self.predictor_pair_dim,
                self.teacher_pair_dim,
                compute_dtype=self.compute_dtype,
            )
            if self.latent_pair_loss_weight > 0
            and self.predictor_pair_dim != self.teacher_pair_dim
            else Identity()
        )
        self.target_projector = (
            TargetProjector(
                self.jepa_target_dim,
                self.target_projector_dim,
                compute_dtype=self.compute_dtype,
            )
            if self.use_target_projector
            else Identity()
        )
        self.teacher_target_projector = (
            TargetProjector(
                self.jepa_target_dim,
                self.target_projector_dim,
                compute_dtype=self.compute_dtype,
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
            )
            if self.jepa_mae_loss_weight > 0 or self.training_mode == "mae"
            else None
        )
        self.jepa_mae_intensity_head = (
            Linear(
                self.target_projector_dim,
                self.jepa_mae_num_intensity_bins,
                compute_dtype=self.compute_dtype,
            )
            if self.jepa_mae_loss_weight > 0 or self.training_mode == "mae"
            else None
        )
        self.distogram_head = (
            Linear(
                self.predictor_pair_dim,
                self.distogram_num_bins,
                compute_dtype=self.compute_dtype,
            )
            if self.distogram_loss_weight > 0
            else None
        )

    def _build_encoder(self, cfg: PeakSetJEPASettings) -> PeakSetEncoder:
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
                fourier_input_scale=cfg.encoder_fourier_input_scale,
                use_fourier_features=cfg.encoder_use_fourier_features,
                compute_dtype=self.compute_dtype,
            ),
            num_layers=cfg.encoder_num_layers,
            num_heads=cfg.encoder_num_heads,
            attention_mlp_multiple=cfg.attention_mlp_multiple,
            norm_eps=cfg.norm_eps,
            use_position_embedding=cfg.encoder_use_position_embedding,
            apply_final_norm=cfg.encoder_apply_final_norm,
            apply_final_pair_norm=cfg.encoder_apply_final_pair_norm,
            num_peaks=cfg.num_peaks,
            pair_dim=cfg.pairmixer_pair_dim,
            pair_feature_hidden_dim=cfg.pairmixer_pair_feature_hidden_dim,
            pairmixer_dropout=cfg.pairmixer_dropout,
            pairmixer_use_pair_bias_attention=cfg.pairmixer_use_pair_bias_attention,
            pairmixer_mz_scale=cfg.pairmixer_mz_scale,
            pairmixer_precursor_mz_scale=cfg.pairmixer_precursor_mz_scale,
            pairmixer_use_fourier_features=cfg.pairmixer_use_fourier_features,
            pairmixer_fourier_num_freqs=cfg.pairmixer_fourier_num_freqs,
            pairmixer_fourier_x_min=cfg.pairmixer_fourier_x_min,
            pairmixer_fourier_x_max=cfg.pairmixer_fourier_x_max,
            pairmixer_relative_fourier_x_min=cfg.pairmixer_relative_fourier_x_min,
            pairmixer_relative_fourier_x_max=cfg.pairmixer_relative_fourier_x_max,
            activation_checkpoint_mode=cfg.activation_checkpoint_mode,
            activation_checkpoint_every_n_layers=(
                cfg.activation_checkpoint_every_n_layers
            ),
            activation_checkpoint_modules=cfg.activation_checkpoint_modules,
            compute_dtype=self.compute_dtype,
        )

    def __call__(
        self,
        augmented_batch: dict[str, Array | torch.Tensor],
        *,
        return_collapse_data: bool = False,
        loss_only: bool = False,
    ) -> dict[str, Array] | tuple[dict[str, Array], dict[str, Array]]:
        batch = (
            batch_to_jax(augmented_batch)
            if any(isinstance(value, torch.Tensor) for value in augmented_batch.values())
            else augmented_batch
        )
        if self.training_mode == "mae":
            return self.forward_mae(
                batch,
                return_collapse_data=return_collapse_data,
                loss_only=loss_only,
            )
        return self.forward_augmented(
            batch,
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
        peak_mz = augmented_batch["peak_mz"]
        peak_intensity = augmented_batch["peak_intensity"]
        peak_valid_mask = augmented_batch["peak_valid_mask"]
        precursor_mz = augmented_batch.get("precursor_mz", None)
        context_mask = augmented_batch["context_mask"] & peak_valid_mask
        target_masks = augmented_batch["target_masks"] & peak_valid_mask[:, None, :]
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
        predictor_visible_masks = context_mask[:, None, :] | target_masks
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
        metrics.update(distogram_metrics)
        metrics.update(latent_pair_metrics)
        if return_collapse_data:
            return metrics, collapse_data
        return metrics

    def forward_mae(
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
        context_mask = augmented_batch["context_mask"] & peak_valid_mask
        target_masks = augmented_batch["target_masks"] & peak_valid_mask[:, None, :]
        context_mz, context_intensity, context_visible_mask = self._context_encoder_inputs(
            peak_mz,
            peak_intensity,
            context_mask,
            target_masks,
        )
        context_encoded, context_pair = self._encode_mae_context(
            context_mz,
            context_intensity,
            peak_valid_mask,
            context_visible_mask,
            precursor_mz,
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
            compute_accuracy=not loss_only,
        )
        predictor_visible_masks = context_mask[:, None, :] | target_masks
        distogram_term, distogram_metrics = self._distogram_metrics(
            predictor_pair,
            peak_mz,
            target_masks,
            predictor_visible_masks,
            predictor_output,
        )
        loss = mae_term + distogram_term
        if loss_only:
            return {"loss": loss}
        valid_peak_count = jnp.maximum(peak_valid_mask.astype(jnp.float32).sum(), 1.0)
        metrics = {
            "loss": loss,
            "context_fraction": context_mask.astype(jnp.float32).sum() / valid_peak_count,
        }
        metrics.update(self._target_mask_metrics(target_masks, valid_peak_count))
        metrics.update(mae_metrics)
        metrics.update(distogram_metrics)
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
    ) -> tuple[Array, Array]:
        def full_encoder(_):
            context_encoded, _, context_pair = self.encoder.forward_with_block_outputs(
                context_mz,
                context_intensity,
                valid_mask=peak_valid_mask,
                visible_mask=context_visible_mask,
                precursor_mz=precursor_mz,
            )
            return context_encoded, context_pair

        pack_tokens = self.mae_context_encoder_pack_tokens
        if pack_tokens <= 0 or self.encoder.use_position_embedding:
            return full_encoder(None)
        return self._encode_packed_mae_context(
            context_mz,
            context_intensity,
            context_visible_mask,
            precursor_mz,
            pack_tokens,
        )

    def _encode_packed_mae_context(
        self,
        context_mz: Array,
        context_intensity: Array,
        context_visible_mask: Array,
        precursor_mz: Array | None,
        pack_tokens: int,
    ) -> tuple[Array, Array]:
        num_peaks = context_mz.shape[1]
        positions = jnp.arange(num_peaks, dtype=jnp.int32)
        sort_key = jnp.where(
            context_visible_mask,
            positions[None, :],
            positions[None, :] + num_peaks,
        )
        packed_indices = jnp.argsort(sort_key, axis=1)[:, :pack_tokens]
        packed_mask = jnp.take_along_axis(context_visible_mask, packed_indices, axis=1)
        packed_mz = jnp.take_along_axis(context_mz, packed_indices, axis=1)
        packed_intensity = jnp.take_along_axis(
            context_intensity,
            packed_indices,
            axis=1,
        )
        packed_encoded, _, packed_pair = self.encoder.forward_with_block_outputs(
            packed_mz,
            packed_intensity,
            valid_mask=packed_mask,
            visible_mask=packed_mask,
            precursor_mz=precursor_mz,
        )
        return self._scatter_packed_mae_context(
            packed_encoded,
            packed_pair,
            packed_indices,
            packed_mask,
            num_peaks,
        )

    def _scatter_packed_mae_context(
        self,
        packed_encoded: Array,
        packed_pair: Array,
        packed_indices: Array,
        packed_mask: Array,
        num_peaks: int,
    ) -> tuple[Array, Array]:
        batch_size, pack_tokens = packed_indices.shape
        batch_indices = jnp.arange(batch_size)
        peak_values = packed_encoded[:, :pack_tokens] * packed_mask[..., None].astype(
            packed_encoded.dtype
        )
        full_peak = jnp.zeros(
            (batch_size, num_peaks, packed_encoded.shape[-1]),
            dtype=packed_encoded.dtype,
        )
        full_peak = full_peak.at[batch_indices[:, None], packed_indices].add(
            peak_values
        )
        context_encoded = jnp.concatenate(
            [full_peak, packed_encoded[:, pack_tokens : pack_tokens + 1]],
            axis=1,
        )

        num_tokens = num_peaks + 1
        pair_dim = packed_pair.shape[-1]
        full_pair = jnp.zeros(
            (batch_size, num_tokens, num_tokens, pair_dim),
            dtype=packed_pair.dtype,
        )
        pair_mask = (
            packed_mask[:, :, None] & packed_mask[:, None, :]
        )[..., None].astype(packed_pair.dtype)
        peak_pair = packed_pair[:, :pack_tokens, :pack_tokens] * pair_mask
        full_pair = full_pair.at[
            batch_indices[:, None, None],
            packed_indices[:, :, None],
            packed_indices[:, None, :],
        ].add(peak_pair)
        mask_f = packed_mask[..., None].astype(packed_pair.dtype)
        full_pair = full_pair.at[
            batch_indices[:, None],
            packed_indices,
            num_peaks,
        ].add(packed_pair[:, :pack_tokens, pack_tokens] * mask_f)
        full_pair = full_pair.at[
            batch_indices[:, None],
            num_peaks,
            packed_indices,
        ].add(packed_pair[:, pack_tokens, :pack_tokens] * mask_f)
        full_pair = full_pair.at[batch_indices, num_peaks, num_peaks].set(
            packed_pair[:, pack_tokens, pack_tokens]
        )
        return context_encoded, full_pair

    def _add_predictor_positions(self, x: Array) -> Array:
        positions = jnp.arange(x.shape[1])
        return x + self.predictor_position_embedding(positions).astype(x.dtype)

    def _add_predictor_pair_positions(self, pair: Array) -> Array:
        num_tokens = pair.shape[1]
        positions = jnp.arange(num_tokens * num_tokens)
        encoding = self.predictor_pair_position_embedding(positions)
        encoding = encoding.reshape(num_tokens, num_tokens, -1)
        return pair + encoding.astype(pair.dtype)

    def _predict_masked_latents_and_pair(
        self,
        x: Array,
        pair: Array,
        visible_mask: Array,
    ) -> tuple[Array, Array]:
        x = self._add_predictor_positions(x)
        x = self.encoder_to_predictor_proj(x)
        pair = self._add_predictor_pair_positions(pair)
        for block_idx, block in enumerate(self.masked_latent_predictor, start=1):
            if should_activation_checkpoint(
                mode=self.activation_checkpoint_mode,
                modules=self.activation_checkpoint_modules,
                module="predictor",
                block_idx=block_idx,
                every_n=self.activation_checkpoint_every_n_layers,
            ):
                x, pair = nnx.remat(
                    _call_pair_mixer_block,
                    policy=activation_checkpoint_policy(
                        self.activation_checkpoint_mode
                    ),
                )(block, x, pair, visible_mask, visible_mask)
            else:
                x, pair = block(x, pair, visible_mask, visible_mask)
        if self.predictor_final_norm is not None:
            x = self.predictor_final_norm(x)
        pair_visible_mask = visible_mask[:, :, None] & visible_mask[:, None, :]
        pair = pair * pair_visible_mask[..., None].astype(pair.dtype)
        return x, pair

    def predict_masked_target_features_with_pair(
        self,
        x: Array,
        pair: Array,
        visible_mask: Array,
    ) -> tuple[Array, Array]:
        x, pair = self._predict_masked_latents_and_pair(x, pair, visible_mask)
        return self.masked_latent_readout(x), pair

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
    ) -> Array:
        teacher_encoder = self.teacher_encoder if self.teacher_encoder is not None else self.encoder
        teacher_peak_outputs = teacher_encoder.forward_peak_block_outputs(
            peak_mz,
            peak_intensity,
            valid_mask=peak_valid_mask,
            visible_mask=peak_valid_mask,
            block_indices=self.jepa_target_layers,
            precursor_mz=precursor_mz,
        )
        return jnp.concatenate(
            [output[:, : peak_mz.shape[1]] for output in teacher_peak_outputs],
            axis=-1,
        )

    def compute_teacher_targets(self, augmented_batch: dict[str, Array | torch.Tensor]) -> Array:
        batch = (
            batch_to_jax(augmented_batch)
            if any(isinstance(value, torch.Tensor) for value in augmented_batch.values())
            else augmented_batch
        )
        teacher_target_features = self._compute_jepa_teacher_target_features(
            batch["peak_mz"],
            batch["peak_intensity"],
            batch["peak_valid_mask"],
            precursor_mz=batch.get("precursor_mz", None),
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
    ) -> tuple[Array, Array, Array, Array, Array]:
        batch_size = peak_mz.shape[0]
        context_mz, context_intensity, context_visible_mask = self._context_encoder_inputs(
            peak_mz,
            peak_intensity,
            context_mask,
            target_masks,
        )
        if self.teacher_encoder is not None:
            teacher_encoded, teacher_peak_outputs, teacher_pair = (
                self.teacher_encoder.forward_with_block_outputs(
                    peak_mz,
                    peak_intensity,
                    valid_mask=peak_valid_mask,
                    visible_mask=peak_valid_mask,
                    block_indices=self.jepa_target_layers,
                    precursor_mz=precursor_mz,
                )
            )
            context_encoded, _, context_pair = self.encoder.forward_with_block_outputs(
                context_mz,
                context_intensity,
                valid_mask=peak_valid_mask,
                visible_mask=context_visible_mask,
                precursor_mz=precursor_mz,
            )
            teacher_target_features = jnp.concatenate(teacher_peak_outputs, axis=-1)
            return (
                teacher_target_features[:, : peak_mz.shape[1]],
                teacher_encoded,
                teacher_pair[:, : peak_mz.shape[1], : peak_mz.shape[1]],
                context_encoded,
                context_pair,
            )
        encoded, teacher_peak_outputs, pair = self.encoder.forward_with_block_outputs(
            jnp.concatenate([peak_mz, context_mz], axis=0),
            jnp.concatenate([peak_intensity, context_intensity], axis=0),
            valid_mask=jnp.concatenate([peak_valid_mask, peak_valid_mask], axis=0),
            visible_mask=jnp.concatenate([peak_valid_mask, context_visible_mask], axis=0),
            block_indices=self.jepa_target_layers,
            precursor_mz=(
                None
                if precursor_mz is None
                else jnp.concatenate([precursor_mz, precursor_mz], axis=0)
            ),
        )
        teacher_target_features = jnp.concatenate(
            [
                peak_output[:batch_size, : peak_mz.shape[1]]
                for peak_output in teacher_peak_outputs
            ],
            axis=-1,
        )
        return (
            teacher_target_features,
            encoded[:batch_size],
            pair[:batch_size, : peak_mz.shape[1], : peak_mz.shape[1]],
            encoded[batch_size:],
            pair[batch_size:],
        )

    def _predict_augmented_target_outputs(
        self,
        context_emb: Array,
        context_pair: Array,
        context_mask: Array,
        target_masks: Array,
    ) -> tuple[Array, Array, Array]:
        batch_size, num_target_blocks, num_peaks = target_masks.shape
        context_mask_by_view = context_mask[:, None, :]
        context_peak_emb = context_emb[:, :num_peaks]
        context_cls_emb = context_emb[:, num_peaks : num_peaks + 1]
        predictor_input = (
            jnp.broadcast_to(
                context_peak_emb[:, None],
                (batch_size, num_target_blocks, num_peaks, context_peak_emb.shape[-1]),
            )
            * context_mask_by_view[..., None]
        )
        latent_mask_token = jnp.broadcast_to(
            self.latent_mask_token[...],
            (batch_size, num_target_blocks, num_peaks, self.latent_mask_token[...].shape[0]),
        )
        predictor_input = jnp.where(
            target_masks[..., None],
            latent_mask_token.astype(predictor_input.dtype),
            predictor_input,
        )
        predictor_input = jnp.concatenate(
            [
                predictor_input,
                jnp.broadcast_to(
                    context_cls_emb[:, None],
                    (batch_size, num_target_blocks, 1, context_cls_emb.shape[-1]),
                ),
            ],
            axis=2,
        )
        predictor_visible_mask = context_mask_by_view | target_masks
        predictor_visible_mask = jnp.concatenate(
            [
                predictor_visible_mask,
                jnp.ones_like(predictor_visible_mask[:, :, :1]),
            ],
            axis=2,
        )
        predictor_pair = jnp.broadcast_to(
            context_pair[:, None],
            (
                batch_size,
                num_target_blocks,
                context_pair.shape[1],
                context_pair.shape[2],
                context_pair.shape[3],
            ),
        )
        context_token_mask = jnp.concatenate(
            [
                jnp.broadcast_to(
                    context_mask_by_view,
                    (batch_size, num_target_blocks, context_mask.shape[-1]),
                ),
                jnp.ones_like(context_mask_by_view[:, :, :1]).repeat(
                    num_target_blocks,
                    axis=1,
                ),
            ],
            axis=2,
        )
        context_pair_mask = context_token_mask[:, :, :, None] & context_token_mask[:, :, None, :]
        predictor_pair = predictor_pair * context_pair_mask[..., None].astype(predictor_pair.dtype)
        target_token_mask = jnp.concatenate(
            [target_masks, jnp.zeros_like(target_masks[:, :, :1])],
            axis=2,
        )
        target_pair_mask = target_token_mask[:, :, :, None] | target_token_mask[:, :, None, :]
        pair_mask_token = jnp.broadcast_to(
            self.pair_mask_token[...],
            predictor_pair.shape,
        )
        predictor_pair = jnp.where(
            target_pair_mask[..., None],
            pair_mask_token.astype(predictor_pair.dtype),
            predictor_pair,
        )
        predictor_pair_mask = predictor_visible_mask[:, :, :, None] & predictor_visible_mask[:, :, None, :]
        predictor_pair = predictor_pair * predictor_pair_mask[..., None].astype(predictor_pair.dtype)
        flat_visible_mask = predictor_visible_mask.reshape(
            batch_size * num_target_blocks,
            num_peaks + 1,
        )
        flat_predictor_input = predictor_input.reshape(
            batch_size * num_target_blocks,
            predictor_input.shape[2],
            predictor_input.shape[-1],
        )
        flat_predictor_pair = predictor_pair.reshape(
            batch_size * num_target_blocks,
            predictor_pair.shape[2],
            predictor_pair.shape[3],
            predictor_pair.shape[-1],
        )
        predictor_features, predictor_pair = self.predict_masked_target_features_with_pair(
            flat_predictor_input,
            flat_predictor_pair,
            flat_visible_mask,
        )
        predictor_features = predictor_features.reshape(
            batch_size,
            num_target_blocks,
            predictor_input.shape[2],
            -1,
        )[:, :, :num_peaks]
        predictor_pair = predictor_pair.reshape(
            batch_size,
            num_target_blocks,
            flat_predictor_pair.shape[1],
            flat_predictor_pair.shape[2],
            -1,
        )[:, :, :num_peaks, :num_peaks]
        predictor_output = self.project_targets(predictor_features)
        return predictor_features, predictor_output, predictor_pair

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

    def _masked_ce_loss(self, logits: Array, targets: Array, valid_mask: Array) -> Array:
        per_token = _cross_entropy_from_logits(logits, targets)
        weights = valid_mask.astype(jnp.float32)
        return (per_token * weights).sum() / jnp.maximum(weights.sum(), 1.0)

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
        mz_loss = self._masked_ce_loss(mz_logits, mz_target, target_masks)
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
        if self.masked_token_input_mode == "mz_sentinel":
            return mz_loss, mz_loss, zero, mz_accuracy, zero
        assert self.jepa_mae_intensity_head is not None
        intensity_logits = self.jepa_mae_intensity_head(predicted_latents)
        intensity_loss = self._masked_ce_loss(
            intensity_logits,
            intensity_target,
            target_masks,
        )
        value_loss = mz_loss + intensity_loss
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

    def _distogram_targets(self, peak_mz: Array) -> Array:
        mz_da = peak_mz.astype(jnp.float32) * self.distogram_mz_max
        pair_distance = jnp.abs(mz_da[:, :, None] - mz_da[:, None, :])
        return jnp.clip(
            jnp.floor(pair_distance / self.jepa_mae_mz_bin_size).astype(jnp.int32),
            0,
            self.distogram_num_bins - 1,
        )

    def _target_pair_mask(self, target_masks: Array, predictor_visible_masks: Array) -> Array:
        target_pair_mask = target_masks[:, :, :, None] | target_masks[:, :, None, :]
        visible_pair_mask = predictor_visible_masks[:, :, :, None] & predictor_visible_masks[:, :, None, :]
        diagonal = jnp.eye(target_masks.shape[-1], dtype=jnp.bool_)[None, None]
        return target_pair_mask & visible_pair_mask & ~diagonal

    def _distogram_metrics(
        self,
        predictor_pair: Array,
        peak_mz: Array,
        target_masks: Array,
        predictor_visible_masks: Array,
        reference: Array,
    ) -> tuple[Array, dict[str, Array]]:
        if self.distogram_loss_weight <= 0:
            return reference.reshape(-1)[0] * 0.0, {}
        assert self.distogram_head is not None
        pair_mask = self._target_pair_mask(target_masks, predictor_visible_masks)
        sym_pair = predictor_pair + jnp.swapaxes(predictor_pair, 2, 3)
        targets = jnp.broadcast_to(
            self._distogram_targets(peak_mz)[:, None],
            sym_pair.shape[:-1],
        )
        logits = self.distogram_head(sym_pair)
        per_pair = _cross_entropy_from_logits(logits, targets)
        weights = pair_mask.astype(jnp.float32)
        distogram_loss = (per_pair * weights).sum() / jnp.maximum(weights.sum(), 1.0)
        term = distogram_loss * self.distogram_loss_weight
        return term, {"distogram_loss": distogram_loss, "distogram_term": term}

    def _latent_pair_metrics(
        self,
        predictor_pair: Array,
        teacher_pair: Array,
        target_masks: Array,
        predictor_visible_masks: Array,
        reference: Array,
    ) -> tuple[Array, dict[str, Array]]:
        if self.latent_pair_loss_weight <= 0:
            return reference.reshape(-1)[0] * 0.0, {}
        pair_mask = self._target_pair_mask(target_masks, predictor_visible_masks)
        predicted_pair = self.masked_pair_readout(predictor_pair)
        teacher_pair_targets = jax.lax.stop_gradient(teacher_pair)
        if self.latent_pair_target_normalization == "layernorm":
            pair_float = teacher_pair_targets.astype(jnp.float32)
            pair_mean = jnp.mean(pair_float, axis=-1, keepdims=True)
            pair_var = jnp.mean(jnp.square(pair_float - pair_mean), axis=-1, keepdims=True)
            teacher_pair_targets = (pair_float - pair_mean) * jax.lax.rsqrt(
                pair_var + 1e-5
            )
        teacher_pair_targets = teacher_pair_targets.astype(predicted_pair.dtype)
        teacher_pair_targets = jnp.broadcast_to(
            teacher_pair_targets[:, None],
            predicted_pair.shape,
        )
        per_pair = self._embedding_loss(predicted_pair, teacher_pair_targets)
        pair_weights = pair_mask.astype(jnp.float32)
        latent_pair_loss = (per_pair * pair_weights).sum() / jnp.maximum(
            pair_weights.sum(),
            1.0,
        )
        term = latent_pair_loss * self.latent_pair_loss_weight
        return term, {"latent_pair_loss": latent_pair_loss, "latent_pair_term": term}

    def pool(self, embeddings: Array, valid_mask: Array) -> Array:
        if embeddings.shape[1] > valid_mask.shape[1]:
            embeddings = embeddings[:, : valid_mask.shape[1]]
        mask = valid_mask[..., None].astype(embeddings.dtype)
        return (embeddings * mask).sum(axis=1) / jnp.maximum(mask.sum(axis=1), 1.0)

    def encode(self, batch: dict[str, Array | torch.Tensor]) -> Array:
        batch = (
            batch_to_jax(batch)
            if any(isinstance(value, torch.Tensor) for value in batch.values())
            else batch
        )
        encoded = self.encoder(
            batch["peak_mz"],
            batch["peak_intensity"],
            valid_mask=batch["peak_valid_mask"],
            visible_mask=batch["peak_valid_mask"],
            precursor_mz=batch.get("precursor_mz", None),
        )
        return self.pool(encoded, batch["peak_valid_mask"])

    def ema_teacher_momentum_at(self, step: int, total_steps: int) -> float:
        if self.ema_teacher_schedule == "constant":
            return self.ema_teacher_momentum_start
        progress = min(1.0, max(0.0, float(step) / float(max(1, total_steps))))
        if self.ema_teacher_schedule == "slow-fast-slow":
            peak = min(1.0, max(1e-6, self.ema_teacher_schedule_peak_fraction))
            if progress <= peak:
                phase = progress / peak
                eased = 0.5 - 0.5 * math.cos(math.pi * phase)
                return self.ema_teacher_momentum_start + eased * (
                    self.ema_teacher_momentum_mid - self.ema_teacher_momentum_start
                )
            phase = (progress - peak) / max(1e-6, 1.0 - peak)
            eased = 0.5 - 0.5 * math.cos(math.pi * phase)
            return self.ema_teacher_momentum_mid + eased * (
                self.ema_teacher_momentum_final - self.ema_teacher_momentum_mid
            )
        if self.ema_teacher_schedule == "cosine":
            progress = 0.5 - 0.5 * math.cos(math.pi * progress)
        return self.ema_teacher_momentum_start + progress * (
            self.ema_teacher_momentum_final - self.ema_teacher_momentum_start
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
        assign_param(self.pair_mask_token, state_dict["pair_mask_token"])
        self.encoder.load_torch_state_dict(state_dict, "encoder")
        if self.teacher_encoder is not None:
            teacher_prefix = (
                "teacher_encoder"
                if any(key.startswith("teacher_encoder.") for key in state_dict)
                else "encoder"
            )
            self.teacher_encoder.load_torch_state_dict(state_dict, teacher_prefix)
        if isinstance(self.encoder_to_predictor_proj, Linear):
            self.encoder_to_predictor_proj.load_torch_state_dict(
                state_dict,
                "encoder_to_predictor_proj",
            )
        self.predictor_position_embedding.load_torch_state_dict(
            state_dict,
            "predictor_position_embedding",
        )
        self.predictor_pair_position_embedding.load_torch_state_dict(
            state_dict,
            "predictor_pair_position_embedding",
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
        if isinstance(self.masked_pair_readout, Linear):
            self.masked_pair_readout.load_torch_state_dict(
                state_dict,
                "masked_pair_readout",
            )
        if isinstance(self.target_projector, TargetProjector):
            self.target_projector.load_torch_state_dict(state_dict, "target_projector")
        if self.teacher_target_projector is not None:
            teacher_projector_prefix = (
                "teacher_target_projector"
                if any(key.startswith("teacher_target_projector.") for key in state_dict)
                else "target_projector"
            )
            self.teacher_target_projector.load_torch_state_dict(
                state_dict,
                teacher_projector_prefix,
            )
        if self.jepa_mae_mz_head is not None:
            self.jepa_mae_mz_head.load_torch_state_dict(state_dict, "jepa_mae_mz_head")
        if self.jepa_mae_intensity_head is not None:
            self.jepa_mae_intensity_head.load_torch_state_dict(
                state_dict,
                "jepa_mae_intensity_head",
            )
        if self.distogram_head is not None:
            self.distogram_head.load_torch_state_dict(state_dict, "distogram_head")

    def load_torch_checkpoint(self, checkpoint_path: str | Path) -> None:
        ckpt = load_torch_checkpoint(checkpoint_path, map_location="cpu", weights_only=True)
        self.load_torch_state_dict(ckpt["model"])


def _cross_entropy_from_logits(logits: Array, targets: Array) -> Array:
    logits = logits.astype(jnp.float32)
    classes = jnp.arange(logits.shape[-1])
    target_one_hot = (classes == targets[..., None]).astype(logits.dtype)
    return -(jax.nn.log_softmax(logits, axis=-1) * target_one_hot).sum(axis=-1)


def _call_pair_mixer_block(
    block: PairMixerBlock,
    single: Array,
    pair: Array,
    peak_mask: Array,
    token_mask: Array,
) -> tuple[Array, Array]:
    return block(single, pair, peak_mask, token_mask, deterministic=True)


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


def _load_frozen_teacher_settings(
    cfg: PeakSetJEPASettings,
) -> PeakSetJEPASettings | None:
    if cfg.training_mode.lower() != "mae_teacher_jepa":
        return None
    if cfg.frozen_teacher_config_path is None:
        return None
    return PeakSetJEPASettings.from_config(load_config(cfg.frozen_teacher_config_path))


def _pair_dim(cfg: PeakSetJEPASettings) -> int:
    return cfg.model_dim if cfg.pairmixer_pair_dim is None else cfg.pairmixer_pair_dim
