from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
import torch
from jax.scipy.special import logsumexp
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
from spectra_learning.models.encoder_jax import PairOnlyEncoder, PeakSetEncoder
from spectra_learning.models.pairmixer_jax import (
    PairOnlyBlock,
    run_pair_only_blocks,
    SUPPORTED_PAIRMIXER_TRANSITION_TYPES,
    _active_indices,
    _apply_pair_rope,
    _gather_pair,
    _gather_single,
)
from spectra_learning.models.peak_features_jax import PeakFeatureEmbedder
from spectra_learning.models.settings import (
    PAIR_DISTOGRAM_TRAINING_MODE,
    PeakSetJEPASettings,
    encoder_uses_pair_path,
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


def build_peak_set_encoder_jax(
    cfg: PeakSetJEPASettings,
    *,
    compute_dtype: object,
    pairmixer_block_type: str,
    pairmixer_fast_encoder_max_visible_tokens: int | None,
    rngs: nnx.Rngs,
) -> PeakSetEncoder | PairOnlyEncoder:
    if cfg.training_mode.lower() == PAIR_DISTOGRAM_TRAINING_MODE:
        return PairOnlyEncoder(cfg, compute_dtype=compute_dtype, rngs=rngs)
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
            compute_dtype=compute_dtype,
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
        use_pair_path=encoder_uses_pair_path(cfg),
        pairmixer_block_type=pairmixer_block_type,
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
        metadata_schema=cfg.encoder_metadata_schema,
        metadata_conditioning=cfg.encoder_metadata_conditioning,
        metadata_condition_dim=cfg.encoder_metadata_condition_dim,
        mz_quantization_bin_size=(
            cfg.jepa_mae_mz_bin_size
            if cfg.mae_objective == "absorbing_diffusion"
            else None
        ),
        pairmixer_fast_max_visible_tokens=(
            pairmixer_fast_encoder_max_visible_tokens
        ),
        activation_checkpoint_mode=cfg.activation_checkpoint_mode,
        activation_checkpoint_every_n_layers=(
            cfg.activation_checkpoint_every_n_layers
        ),
        activation_checkpoint_modules=cfg.activation_checkpoint_modules,
        compute_dtype=compute_dtype,
        rngs=rngs,
    )


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
        self.mae_objective = cfg.mae_objective
        if self.mae_objective not in {"masked_reconstruction", "absorbing_diffusion"}:
            raise ValueError(f"Unknown mae_objective: {self.mae_objective}")
        if self.mae_objective == "absorbing_diffusion":
            required = {
                "training_mode": "mae",
                "masked_token_input_mode": "latent_token",
                "jepa_num_target_blocks": 1,
                "encoder_use_cls_token": True,
                "mae_mz_objective": "exact_ce",
                "mae_mz_multiscale_weight": 0.0,
                "mae_intensity_loss_weight": 0.0,
                "mae_loss_weight": 1.0,
                "distogram_loss_weight": 0.0,
                "use_ema_teacher": False,
            }
            for name, expected in required.items():
                if getattr(cfg, name) != expected:
                    raise ValueError(f"absorbing_diffusion requires {name}={expected!r}")
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
        if self.pairmixer_fast_max_visible_tokens is None and (
            self.use_fastmixer or self.training_mode == PAIR_DISTOGRAM_TRAINING_MODE
        ):
            self.pairmixer_fast_max_visible_tokens = cfg.num_peaks + int(
                cfg.encoder_use_cls_token
            )
        self.pairmixer_fast_target_max_visible_tokens = (
            cfg.predictor_target_max_tokens
            if cfg.predictor_target_max_tokens is not None
            else cfg.num_peaks
        )
        self.pairmixer_fast_encoder_max_visible_tokens = (
            cfg.pairmixer_fast_encoder_max_visible_tokens
            if cfg.pairmixer_fast_encoder_max_visible_tokens is not None
            else self.pairmixer_fast_max_visible_tokens
        )
        self.pairmixer_transition_type = cfg.pairmixer_transition_type.lower()
        if self.pairmixer_transition_type not in SUPPORTED_PAIRMIXER_TRANSITION_TYPES:
            raise ValueError(
                "pairmixer_transition_type must be one of ('swiglu', 'feedforward')"
            )
        self.encoder_num_layers = cfg.encoder_num_layers
        self.norm_eps = cfg.norm_eps
        self.encoder_metadata_schema = cfg.encoder_metadata_schema
        self.jepa_num_target_blocks = cfg.jepa_num_target_blocks
        self.is_pair_distogram = self.training_mode == PAIR_DISTOGRAM_TRAINING_MODE
        if self.is_pair_distogram:
            if cfg.masked_token_input_mode != "latent_token":
                raise ValueError("pair_distogram requires masked_token_input_mode='latent_token'")
            if cfg.distogram_loss_weight <= 0:
                raise ValueError("pair_distogram requires distogram_loss_weight > 0")
            if cfg.mae_loss_weight or cfg.jepa_mae_loss_weight or cfg.masked_token_loss_weight:
                raise ValueError("pair_distogram requires other loss weights to be zero")
            if cfg.encoder_metadata_schema and cfg.encoder_metadata_conditioning != "adaln_zero":
                raise ValueError("pair_distogram metadata requires adaln_zero conditioning")
        self.encoder_pair_dim = (
            cfg.model_dim
            if cfg.pairmixer_pair_dim is None
            else cfg.pairmixer_pair_dim
        )
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
        if cfg.latent_pair_loss_weight > 0:
            raise ValueError(
                "latent pair prediction is unavailable with the cross-attention predictor"
            )
        self.distogram_loss_weight = cfg.distogram_loss_weight
        self.distogram_pair_dim = cfg.distogram_pair_dim
        self.distogram_multiscale_widths = tuple(
            int(width) for width in cfg.distogram_multiscale_widths
        )
        self.distogram_multiscale_weight = float(cfg.distogram_multiscale_weight)
        self.distogram_mz_max = cfg.distogram_mz_max
        self.jepa_mae_mz_bin_size = cfg.jepa_mae_mz_bin_size
        self.jepa_mae_intensity_bin_size = cfg.jepa_mae_intensity_bin_size
        self.mae_intensity_loss_weight = cfg.mae_intensity_loss_weight
        self.mae_mz_objective = cfg.mae_mz_objective.lower()
        if self.mae_mz_objective not in {
            "exact_ce",
            "multiscale_ce",
            "sparse_ordinal_ce",
        }:
            raise ValueError(
                "mae_mz_objective must be one of "
                "('exact_ce', 'multiscale_ce', 'sparse_ordinal_ce')"
            )
        self.mae_mz_ordinal_radius_bins = int(cfg.mae_mz_ordinal_radius_bins)
        self.mae_mz_ordinal_epsilon = float(cfg.mae_mz_ordinal_epsilon)
        if self.mae_mz_ordinal_radius_bins < 0:
            raise ValueError("mae_mz_ordinal_radius_bins must be >= 0")
        if not 0.0 <= self.mae_mz_ordinal_epsilon <= 1.0:
            raise ValueError("mae_mz_ordinal_epsilon must be between 0 and 1")
        self.jepa_mae_mz_max = cfg.jepa_mae_mz_max
        self.jepa_mae_intensity_max = cfg.jepa_mae_intensity_max
        self.mae_mz_multiscale_widths = tuple(
            int(width) for width in cfg.mae_mz_multiscale_widths
        )
        self.mae_mz_multiscale_weight = float(cfg.mae_mz_multiscale_weight)
        self.mae_gap_bucket_edges_da = tuple(
            float(edge) for edge in cfg.mae_gap_bucket_edges_da
        )
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
            if self.teacher_encoder.use_pair_path:
                for block in self.teacher_encoder.blocks:
                    block.fastmixer_max_visible_tokens = teacher_full_tokens
        self.latent_mask_token = (
            None
            if self.is_pair_distogram
            else nnx.Param(
                rngs.params.normal((self.predictor_dim,), dtype=jnp.float32) * 0.02
            )
        )
        self.encoder_to_predictor_proj = (
            Linear(
                self.model_dim,
                self.predictor_dim,
                bias=False,
                compute_dtype=self.compute_dtype,
                rngs=rngs,
            )
            if not self.is_pair_distogram and self.predictor_dim != self.model_dim
            else Identity()
        )
        predictor_blocks: list[Any] = []
        for index in range(cfg.masked_latent_predictor_num_layers):
            block = (
                PairOnlyBlock(
                    pair_dim=self.encoder_pair_dim,
                    attention_mlp_multiple=cfg.predictor_mlp_multiple,
                    norm_eps=self.norm_eps,
                    transition_type=self.pairmixer_transition_type,
                    triangle_attention_num_heads=(
                        cfg.pairmixer_triangle_attention_num_heads
                        if cfg.pairmixer_triangle_attention_every_n_layers
                        and (index + 1) % cfg.pairmixer_triangle_attention_every_n_layers == 0
                        else None
                    ),
                    compute_dtype=self.compute_dtype,
                    rngs=rngs,
                )
                if self.is_pair_distogram
                else CrossAttentionBlock(
                    dim=self.predictor_dim,
                    n_heads=cfg.masked_latent_predictor_num_heads,
                    norm_eps=self.norm_eps,
                    hidden_dim=math.ceil(
                        self.predictor_dim * cfg.predictor_mlp_multiple
                    ),
                    max_sequence_length=(
                        self.num_peak_tokens + int(self.encoder_use_cls_token)
                    ),
                    compute_dtype=self.compute_dtype,
                    rngs=rngs,
                )
            )
            predictor_blocks.append(block)
        self.masked_latent_predictor = nnx.List(predictor_blocks)
        # Marks pair entries whose m/z was hidden from the encoder.
        self.pair_mask_token = (
            nnx.Param(
                rngs.params.normal((self.encoder_pair_dim,), dtype=jnp.float32)
                * 0.02
            )
            if self.is_pair_distogram
            else None
        )
        self.predictor_pair_norm = (
            RMSNorm(self.encoder_pair_dim, eps=self.norm_eps, affine=True)
            if self.is_pair_distogram and cfg.predictor_apply_final_norm
            else None
        )
        self.predictor_final_norm = (
            RMSNorm(self.predictor_dim, eps=self.norm_eps, affine=False)
            if not self.is_pair_distogram and cfg.predictor_apply_final_norm
            else None
        )
        self.masked_latent_readout = (
            None
            if self.is_pair_distogram
            else Linear(
                self.predictor_dim,
                self.jepa_target_dim,
                compute_dtype=self.compute_dtype,
                rngs=rngs,
            )
        )
        self.target_projector = (
            TargetProjector(
                self.jepa_target_dim,
                self.target_projector_dim,
                compute_dtype=self.compute_dtype,
                rngs=rngs,
            )
            if not self.is_pair_distogram and self.use_target_projector
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
        # A distogram-only run sets mae_loss_weight to 0; without this the m/z
        # and intensity heads are still built and still run every step.
        self.uses_mae_value_heads = self.jepa_mae_loss_weight > 0 or (
            self.training_mode == "mae" and self.mae_loss_weight > 0
        )
        if cfg.mae_mz_head not in {"linear", "readout"}:
            raise ValueError(f"Unknown mae_mz_head: {cfg.mae_mz_head}")
        if cfg.mae_mz_head == "readout":
            if self.jepa_mae_num_mz_bins != self.target_projector_dim:
                raise ValueError("mae_mz_head='readout' requires one m/z bin per readout dimension")
        self.jepa_mae_mz_head = (
            (
                Identity()
                if cfg.mae_mz_head == "readout"
                else Linear(
                    self.target_projector_dim,
                    self.jepa_mae_num_mz_bins,
                    compute_dtype=self.compute_dtype,
                    rngs=rngs,
                )
            )
            if self.uses_mae_value_heads
            else None
        )
        self.jepa_mae_intensity_head = (
            Linear(
                self.target_projector_dim,
                self.jepa_mae_num_intensity_bins,
                compute_dtype=self.compute_dtype,
                rngs=rngs,
            )
            if self.uses_mae_value_heads and self.mae_intensity_loss_weight > 0.0
            else None
        )
        # The distogram scores every (target, partner) pair, so the tensor the
        # head reads is [B, targets, partners, dim]. Projecting the predictor
        # latents down first is what keeps that affordable: at predictor_dim it
        # is the largest activation in the step.
        distogram_input_dim = (
            self.encoder_pair_dim if self.is_pair_distogram else self.predictor_dim
        )
        self.distogram_pair_proj = (
            Linear(
                distogram_input_dim,
                self.distogram_pair_dim,
                bias=False,
                compute_dtype=self.compute_dtype,
                rngs=rngs,
            )
            if self.distogram_loss_weight > 0 and self.distogram_pair_dim is not None
            else None
        )
        self.distogram_head = (
            Linear(
                distogram_input_dim
                if self.distogram_pair_dim is None
                else self.distogram_pair_dim,
                self.distogram_num_bins,
                compute_dtype=self.compute_dtype,
                rngs=rngs,
            )
            if self.distogram_loss_weight > 0
            else None
        )

    def _build_encoder(
        self, cfg: PeakSetJEPASettings, rngs: nnx.Rngs,
    ) -> PeakSetEncoder | PairOnlyEncoder:
        return build_peak_set_encoder_jax(
            cfg,
            compute_dtype=self.compute_dtype,
            pairmixer_block_type=self.pairmixer_block_type,
            pairmixer_fast_encoder_max_visible_tokens=(
                self.pairmixer_fast_encoder_max_visible_tokens
            ),
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
        if self.encoder.use_pair_path and not self.is_pair_distogram:
            for block in self.encoder.blocks:
                block.fastmixer_max_visible_tokens = encoder_tokens

    def __call__(
        self,
        augmented_batch: dict[str, Array],
        *,
        return_collapse_data: bool = False,
        loss_only: bool = False,
    ) -> dict[str, Array] | tuple[dict[str, Array], dict[str, Array]]:
        if self.is_pair_distogram:
            return self.forward_pair_distogram(
                augmented_batch,
                return_collapse_data=return_collapse_data,
                loss_only=loss_only,
            )
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
        spectrum_metadata = jax_spectrum_metadata_from_batch(
            augmented_batch, self.encoder_metadata_schema
        )
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
        predictor_output_features, predictor_output, distogram_loss = (
            self._predict_augmented_target_outputs(
                context_emb,
                context_mask,
                target_masks,
                peak_mz,
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
        distogram_term, distogram_metrics = self._distogram_metrics(
            distogram_loss,
            predictor_output,
        )
        loss = masked_prediction_term + jepa_mae_term + distogram_term
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
        spectrum_metadata = jax_spectrum_metadata_from_batch(
            augmented_batch, self.encoder_metadata_schema
        )
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
            enc_idx,
            enc_compact_mask,
        ) = self.encoder.forward_compact(
            context_mz,
            context_intensity,
            valid_mask=peak_valid_mask,
            visible_mask=context_visible_mask,
            max_visible_tokens=self.pairmixer_fast_encoder_max_visible_tokens,
            precursor_mz=precursor_mz,
            spectrum_metadata=spectrum_metadata,
        )
        predictor_targets, target_positions, target_mask, distogram_loss = (
            self._predict_augmented_target_outputs_fastmixer_compact(
                context_encoded_compact,
                enc_idx,
                enc_compact_mask,
                context_mask,
                target_masks,
                peak_mz,
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
        masked_prediction_term = (
            masked_prediction_loss * self.masked_token_loss_weight
        )
        distogram_term, distogram_metrics = self._distogram_metrics(
            distogram_loss,
            predictor_targets,
        )
        loss = masked_prediction_term + distogram_term
        if loss_only:
            return {"loss": loss}

        valid_peak_count = jnp.maximum(
            peak_valid_mask.astype(jnp.float32).sum(),
            1.0,
        )
        metrics = {
            "loss": loss,
            "masked_prediction_loss": masked_prediction_loss,
            "masked_prediction_term": masked_prediction_term,
            "context_fraction": context_mask.astype(jnp.float32).sum()
            / valid_peak_count,
        }
        metrics.update(self._target_mask_metrics(target_masks, valid_peak_count))
        metrics.update(distogram_metrics)
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
        spectrum_metadata = jax_spectrum_metadata_from_batch(
            augmented_batch, self.encoder_metadata_schema
        )
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
        _, predictor_output, distogram_loss = (
            self._predict_augmented_target_outputs(
                context_encoded,
                context_mask,
                target_masks,
                peak_mz,
            )
        )
        target_gap_da = None
        if not loss_only and self.mae_gap_bucket_edges_da and self.uses_mae_value_heads:
            gap = _nearest_context_gap_da(
                peak_mz,
                peak_mz,
                context_mask,
                self.jepa_mae_mz_max,
            )
            target_gap_da = jnp.broadcast_to(gap[:, None, :], target_masks.shape)
        if self.uses_mae_value_heads:
            mae_term, mae_metrics = self._mae_metrics(
                predictor_output,
                peak_mz,
                peak_intensity,
                target_masks,
                context_encoded,
                compute_accuracy=not loss_only,
                target_gap_da=target_gap_da,
            )
        else:
            mae_term, mae_metrics = context_encoded.reshape(-1)[0] * 0.0, {}
        distogram_term, distogram_metrics = self._distogram_metrics(
            distogram_loss,
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
        spectrum_metadata = jax_spectrum_metadata_from_batch(
            augmented_batch, self.encoder_metadata_schema
        )
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
            enc_idx,
            enc_compact_mask,
        ) = self.encoder.forward_compact(
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
            distogram_loss,
        ) = self._predict_augmented_target_outputs_fastmixer_compact(
            context_encoded_compact,
            enc_idx,
            enc_compact_mask,
            context_mask,
            target_masks,
            peak_mz,
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
        target_gap_da = None
        if not loss_only and self.mae_gap_bucket_edges_da and self.uses_mae_value_heads:
            num_views = target_masks.shape[1]
            flat_context_mask = jnp.broadcast_to(
                context_mask[:, None, :],
                (context_mask.shape[0], num_views, context_mask.shape[1]),
            ).reshape(-1, context_mask.shape[1])
            target_gap_da = _nearest_context_gap_da(
                peak_mz_compact,
                flat_peak_mz,
                flat_context_mask,
                self.jepa_mae_mz_max,
            )
        if self.uses_mae_value_heads:
            mae_term, mae_metrics = self._mae_metrics_compact(
                predictor_output_compact,
                peak_mz_compact,
                peak_intensity_compact,
                target_mask,
                reference=context_encoded_compact,
                compute_accuracy=not loss_only,
                target_gap_da=target_gap_da,
            )
        else:
            mae_term, mae_metrics = context_encoded_compact.reshape(-1)[0] * 0.0, {}
        distogram_term, distogram_metrics = self._distogram_metrics(
            distogram_loss,
            predictor_output_compact,
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

    def forward_pair_distogram(
        self,
        augmented_batch: dict[str, Array],
        *,
        return_collapse_data: bool = False,
        loss_only: bool = False,
    ) -> dict[str, Array] | tuple[dict[str, Array], dict[str, Array]]:
        """Encode context pairs and query missing distances in the predictor."""
        peak_mz = augmented_batch["peak_mz"]
        peak_intensity = augmented_batch["peak_intensity"]
        peak_valid_mask = augmented_batch["peak_valid_mask"]
        precursor_mz = augmented_batch.get("precursor_mz", None)
        spectrum_metadata = jax_spectrum_metadata_from_batch(
            augmented_batch, self.encoder_metadata_schema
        )
        context_mask = augmented_batch["context_mask"] & peak_valid_mask
        target_masks = augmented_batch["target_masks"] & peak_valid_mask[:, None, :]
        target_union = jnp.any(target_masks, axis=1)
        context_mask = context_mask & ~target_union
        pair, idx, compact_token_mask = self.encoder(
            peak_mz,
            peak_intensity,
            valid_mask=peak_valid_mask,
            visible_mask=context_mask,
            max_visible_tokens=self.pairmixer_fast_encoder_max_visible_tokens,
            precursor_mz=precursor_mz,
            spectrum_metadata=spectrum_metadata,
        )
        distogram_losses = self._distogram_loss_pair_branch(
            pair,
            idx,
            compact_token_mask,
            target_union,
            peak_mz,
        )
        loss, distogram_metrics = self._distogram_metrics(distogram_losses, pair)
        if loss_only:
            return {"loss": loss}
        valid_peak_count = jnp.maximum(peak_valid_mask.astype(jnp.float32).sum(), 1.0)
        metrics = {
            "loss": loss,
            "context_fraction": context_mask.astype(jnp.float32).sum()
            / valid_peak_count,
        }
        metrics.update(self._target_mask_metrics(target_masks, valid_peak_count))
        metrics.update(distogram_metrics)
        if return_collapse_data:
            return metrics, {}
        return metrics

    def _pair_predictor_inputs(
        self,
        pair: Array,
        idx: Array,
        compact_token_mask: Array,
        target_union: Array,
    ) -> tuple[Array, Array, Array]:
        """Keep encoded context pairs and seed new pairs with masked queries."""
        num_cls = int(self.encoder_use_cls_token)
        batch_idx = jnp.arange(pair.shape[0])[:, None]
        context = jnp.zeros(
            (pair.shape[0], self.num_peak_tokens + num_cls), dtype=jnp.bool_,
        ).at[batch_idx, idx].set(compact_token_mask)
        predictor_idx, predictor_mask = _active_indices(
            context[:, :self.num_peak_tokens] | target_union,
            self.pairmixer_fast_max_visible_tokens - num_cls,
        )
        if num_cls:
            predictor_idx = jnp.concatenate([
                predictor_idx,
                jnp.full((pair.shape[0], 1), self.num_peak_tokens, dtype=idx.dtype),
            ], axis=1)
            predictor_mask = jnp.concatenate([
                predictor_mask, jnp.ones((pair.shape[0], 1), dtype=jnp.bool_),
            ], axis=1)

        # Remap the compact context square directly into predictor order.
        matches = (
            (predictor_idx[:, :, None] == idx[:, None, :])
            & compact_token_mask[:, None, :]
        )
        encoder_slots = jnp.argmax(matches, axis=-1)
        encoded = _gather_pair(pair, encoder_slots)
        is_context = jnp.any(matches, axis=-1)
        context_pair = is_context[:, :, None] & is_context[:, None, :]
        # Use the encoder's 2D input rotary convention for masked queries:
        # rotate each channel half by its endpoint's original position, with
        # no target m/z or intensity features.
        num_tokens = predictor_idx.shape[1]
        query = jnp.broadcast_to(
            self.pair_mask_token[...].astype(pair.dtype),
            (pair.shape[0], num_tokens, num_tokens, self.encoder_pair_dim),
        )
        masked_pair = _apply_pair_rope(query, predictor_idx)
        predicted = jnp.where(context_pair[..., None], encoded, masked_pair)
        valid_pair = predictor_mask[:, :, None] & predictor_mask[:, None, :]
        predicted = predicted * valid_pair[..., None].astype(pair.dtype)
        return predicted, predictor_idx, predictor_mask

    def _distogram_loss_pair_branch(
        self,
        pair: Array,
        idx: Array,
        compact_token_mask: Array,
        target_union: Array,
        peak_mz: Array,
    ) -> dict[str, Array]:
        assert self.distogram_head is not None
        predicted, idx, compact_token_mask = self._pair_predictor_inputs(
            pair, idx, compact_token_mask, target_union,
        )
        num_cls = int(self.encoder_use_cls_token)
        slot_is_peak = compact_token_mask & (idx < peak_mz.shape[1])
        slot_is_target = jnp.take_along_axis(
            jnp.pad(target_union, ((0, 0), (0, num_cls))), idx, axis=1,
        ) & slot_is_peak
        slot_mz = jnp.take_along_axis(
            jnp.pad(peak_mz, ((0, 0), (0, num_cls))), idx, axis=1,
        )
        predicted = run_pair_only_blocks(
            self.masked_latent_predictor, predicted, compact_token_mask, None,
            checkpoint_mode=(
                self.activation_checkpoint_mode
                if "predictor" in self.activation_checkpoint_modules else "none"
            ),
            checkpoint_every_n=self.activation_checkpoint_every_n_layers,
        )
        if self.predictor_pair_norm is not None:
            predicted = self.predictor_pair_norm(predicted)

        # Score the masked rows only. Every unknown pair touches a masked peak,
        # so those rows already cover the whole supervised set, and the 2,000-way
        # head is the most expensive thing in the step: at 32 masked peaks in
        # 110 slots this is 3,520 pairs rather than 12,100.
        target_slots, target_slot_mask = _active_indices(
            slot_is_target,
            min(
                predicted.shape[1],
                self.jepa_num_target_blocks * self.pairmixer_fast_target_max_visible_tokens,
            ),
        )
        batch_size, num_tokens, _, pair_dim = predicted.shape
        rows_pair = _gather_single(
            predicted.reshape(batch_size, num_tokens, num_tokens * pair_dim),
            target_slots,
        ).reshape(batch_size, target_slots.shape[1], num_tokens, pair_dim)
        row_mz = jnp.take_along_axis(slot_mz, target_slots, axis=1)
        row_positions = jnp.take_along_axis(idx, target_slots, axis=1)
        column_positions = idx
        pair_mask = (
            target_slot_mask[:, :, None]
            & slot_is_peak[:, None, :]
            & (row_positions[:, :, None] != column_positions[:, None, :])
        )
        distance = (
            jnp.abs(row_mz[:, :, None] - slot_mz[:, None, :]) * self.distogram_mz_max
        )
        targets = jnp.clip(
            jnp.floor(distance / self.jepa_mae_mz_bin_size).astype(jnp.int32),
            0,
            self.distogram_num_bins - 1,
        )
        features = self._distogram_pair_latents(rows_pair)
        rows = self._distogram_score(features, targets)
        weights = pair_mask.astype(jnp.float32)
        denominator = jnp.maximum(weights.sum(), 1.0)
        return {
            (
                "distogram_loss" if width == 1 else f"distogram_ce_{width}bins"
            ): (row * weights).sum()
            / denominator
            for width, row in rows.items()
        }

    def _distogram_score(self, features: Array, targets: Array) -> dict[int, Array]:
        widths = (
            (1, *self.distogram_multiscale_widths)
            if self._use_distogram_multiscale
            else (1,)
        )
        # These logits are the largest tensor in the step - 2,000 classes on
        # every pair - so recomputing them in the backward is usually a better
        # trade than holding them, unlike a per-token head where it is not.
        score = (
            nnx.remat(
                _distogram_score_rows,
                policy=activation_checkpoint_policy(
                    self.activation_checkpoint_mode
                ),
                static_argnums=(3,),
            )
            if should_activation_checkpoint(
                mode=self.activation_checkpoint_mode,
                modules=self.activation_checkpoint_modules,
                module="distogram",
                block_idx=1,
                every_n=1,
            )
            else _distogram_score_rows
        )
        return score(self.distogram_head, features, targets, widths)

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
        teacher_encoded = teacher_encoder(
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
            spectrum_metadata=jax_spectrum_metadata_from_batch(
                augmented_batch, self.encoder_metadata_schema
            ),
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
        peak_mz: Array,
    ) -> tuple[Array, Array, dict[str, Array] | None]:
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
            compact_latents,
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
        distogram_loss = None
        if self.distogram_loss_weight > 0:
            flat_context_mask = jnp.broadcast_to(
                context_mask[:, None],
                (batch_size, num_target_blocks, num_peaks),
            ).reshape(batch_size * num_target_blocks, num_peaks)
            flat_peak_mz = jnp.broadcast_to(
                peak_mz[:, None],
                (batch_size, num_target_blocks, num_peaks),
            ).reshape(batch_size * num_target_blocks, num_peaks)
            distogram_loss = self._distogram_loss_compact(
                compact_latents,
                target_positions,
                target_mask,
                flat_memory,
                flat_memory_positions,
                flat_memory_mask,
                flat_context_mask,
                flat_peak_mz,
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
        return predictor_features, predictor_output, distogram_loss

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
    ) -> tuple[Array, Array, Array, Array, Array]:
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
        return latents, features, output, target_positions, target_mask

    def _predict_augmented_target_outputs_fastmixer_compact(
        self,
        context_emb_compact: Array,
        enc_idx: Array,
        enc_compact_mask: Array,
        context_mask: Array,
        target_masks: Array,
        peak_mz: Array,
    ) -> tuple[Array, Array, Array, dict[str, Array] | None]:
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
        latents, _, output, target_positions, target_mask = (
            self._predict_compact_target_outputs(
                flat_memory,
                flat_memory_positions,
                flat_memory_mask,
                target_masks.reshape(flat_batch_size, num_peaks),
            )
        )
        distogram_loss = None
        if self.distogram_loss_weight > 0:
            flat_context_mask = jnp.broadcast_to(
                context_mask[:, None],
                (batch_size, num_target_blocks, num_peaks),
            ).reshape(flat_batch_size, num_peaks)
            flat_peak_mz = jnp.broadcast_to(
                peak_mz[:, None],
                (batch_size, num_target_blocks, num_peaks),
            ).reshape(flat_batch_size, num_peaks)
            distogram_loss = self._distogram_loss_compact(
                latents,
                target_positions,
                target_mask,
                flat_memory,
                flat_memory_positions,
                flat_memory_mask,
                flat_context_mask,
                flat_peak_mz,
            )
        return output, target_positions, target_mask, distogram_loss

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

    def _distogram_targets(self, peak_mz: Array) -> Array:
        mz_da = peak_mz.astype(jnp.float32) * self.distogram_mz_max
        pair_distance = jnp.abs(mz_da[:, :, None] - mz_da[:, None, :])
        return jnp.clip(
            jnp.floor(pair_distance / self.jepa_mae_mz_bin_size).astype(jnp.int32),
            0,
            self.distogram_num_bins - 1,
        )

    @property
    def _use_distogram_multiscale(self) -> bool:
        return (
            bool(self.distogram_multiscale_widths)
            and self.distogram_multiscale_weight > 0.0
        )

    def _distogram_pair_latents(self, latents: Array) -> Array:
        if self.distogram_pair_proj is None:
            return latents
        return self.distogram_pair_proj(latents)

    def _distogram_loss_compact(
        self,
        target_latents: Array,
        target_positions: Array,
        target_mask: Array,
        memory: Array,
        memory_positions: Array,
        memory_mask: Array,
        context_mask: Array,
        peak_mz: Array,
    ) -> dict[str, Array]:
        assert self.distogram_head is not None
        context_latents = self.encoder_to_predictor_proj(memory)
        if self.predictor_final_norm is not None:
            context_latents = self.predictor_final_norm(context_latents)
        padded_context_mask = jnp.pad(
            context_mask,
            ((0, 0), (0, int(self.encoder_use_cls_token))),
        )
        context_slot = (
            jnp.take_along_axis(
                padded_context_mask,
                memory_positions,
                axis=1,
            )
            & memory_mask
        )

        # Project before the outer difference, not after: the projection is
        # linear in the token count and the difference is quadratic in it.
        target_pair_latents = self._distogram_pair_latents(target_latents)
        partner_pair_latents = jnp.concatenate(
            [
                self._distogram_pair_latents(context_latents),
                target_pair_latents,
            ],
            axis=1,
        )
        partner_positions = jnp.concatenate(
            [memory_positions, target_positions],
            axis=1,
        )
        partner_mask = jnp.concatenate([context_slot, target_mask], axis=1)
        pair_mask = (
            target_mask[:, :, None]
            & partner_mask[:, None, :]
            & (target_positions[:, :, None] != partner_positions[:, None, :])
        )
        pair_features = jnp.abs(
            target_pair_latents[:, :, None, :] - partner_pair_latents[:, None, :, :]
        )

        padded_peak_mz = jnp.pad(
            peak_mz,
            ((0, 0), (0, int(self.encoder_use_cls_token))),
        )
        target_mz = jnp.take_along_axis(
            padded_peak_mz,
            target_positions,
            axis=1,
        )
        partner_mz = jnp.take_along_axis(
            padded_peak_mz,
            partner_positions,
            axis=1,
        )
        pair_distance = (
            jnp.abs(target_mz[:, :, None] - partner_mz[:, None, :])
            * self.distogram_mz_max
        )
        targets = jnp.clip(
            jnp.floor(pair_distance / self.jepa_mae_mz_bin_size).astype(jnp.int32),
            0,
            self.distogram_num_bins - 1,
        )

        # A compact target-context entry represents both directions in the
        # original symmetric full pair mask. Target-target directions are
        # already both present in the compact target rows.
        context_weights = jnp.full(context_slot.shape, 2.0, dtype=jnp.float32)
        partner_weights = jnp.concatenate(
            [context_weights, target_mask.astype(jnp.float32)],
            axis=1,
        )
        weights = partner_weights[:, None, :] * pair_mask.astype(jnp.float32)
        denominator = jnp.maximum(weights.sum(), 1.0)
        rows = self._distogram_score(pair_features, targets)
        losses = {
            (
                "distogram_loss"
                if width == 1
                else f"distogram_ce_{width}bins"
            ): (row * weights).sum() / denominator
            for width, row in rows.items()
        }
        return losses

    def _distogram_metrics(
        self,
        distogram_losses: dict[str, Array] | None,
        reference: Array,
    ) -> tuple[Array, dict[str, Array]]:
        if self.distogram_loss_weight <= 0:
            return reference.reshape(-1)[0] * 0.0, {}
        assert distogram_losses is not None
        metrics = dict(distogram_losses)
        objective = metrics["distogram_loss"]
        coarse = [
            value for name, value in metrics.items() if name != "distogram_loss"
        ]
        if coarse:
            multiscale = sum(coarse) / float(len(coarse))
            metrics["distogram_multiscale_loss"] = multiscale
            objective = objective + self.distogram_multiscale_weight * multiscale
        term = objective * self.distogram_loss_weight
        metrics["distogram_objective"] = objective
        metrics["distogram_term"] = term
        return term, metrics

    @property
    def _use_mz_multiscale(self) -> bool:
        return (
            self.mae_mz_objective == "multiscale_ce"
            and bool(self.mae_mz_multiscale_widths)
            and self.mae_mz_multiscale_weight > 0.0
        )

    def _mz_cross_entropy_rows(self, mz_logits: Array, mz_target: Array) -> Array:
        if self.mae_mz_objective == "sparse_ordinal_ce":
            return _sparse_ordinal_cross_entropy_from_logits(
                mz_logits,
                mz_target,
                self.mae_mz_ordinal_radius_bins,
                self.mae_mz_ordinal_epsilon,
            )
        return _cross_entropy_from_logits(mz_logits, mz_target)

    def _mz_multiscale_loss(
        self,
        mz_logits: Array,
        mz_target: Array,
        target_mask: Array,
    ) -> tuple[Array, dict[str, Array]]:
        if not self._use_mz_multiscale:
            zero = mz_logits.reshape(-1)[0] * 0.0
            return zero, {}
        rows = _coarse_marginal_ce_rows(
            mz_logits,
            mz_target,
            self.mae_mz_multiscale_widths,
        )
        metrics = {
            f"mae_mz_ce_{width}bins": _masked_mean(value, target_mask)
            for width, value in rows.items()
        }
        mean_loss = sum(metrics.values()) / float(len(metrics))
        metrics["mae_mz_multiscale_loss"] = mean_loss
        return mean_loss, metrics

    def _gap_bucket_metrics(
        self,
        per_token_ce: Array,
        target_mask: Array,
        gap_da: Array | None,
    ) -> dict[str, Array]:
        if gap_da is None or not self.mae_gap_bucket_edges_da:
            return {}
        edges = self.mae_gap_bucket_edges_da
        labels = _gap_bucket_labels(edges)
        bounds = (0.0, *edges, float("inf"))
        total = jnp.maximum(target_mask.astype(jnp.float32).sum(), 1.0)
        metrics: dict[str, Array] = {}
        for index, label in enumerate(labels):
            lower, upper = bounds[index], bounds[index + 1]
            in_bucket = target_mask & (gap_da >= lower)
            if upper != float("inf"):
                in_bucket = in_bucket & (gap_da < upper)
            weights = in_bucket.astype(jnp.float32)
            metrics[f"mae_mz_loss_gap_{label}da"] = (
                per_token_ce * weights
            ).sum() / jnp.maximum(weights.sum(), 1.0)
            metrics[f"mae_gap_fraction_{label}da"] = weights.sum() / total
        return metrics

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
    ) -> tuple[Array, Array, Array, Array, Array, Array, dict[str, Array]]:
        assert self.jepa_mae_mz_head is not None
        mz_logits = self.jepa_mae_mz_head(predicted_latents)
        mz_target, intensity_target = self._jepa_mae_targets(peak_mz, peak_intensity)
        view_shape = mz_logits.shape[:3]
        mz_target = jnp.broadcast_to(mz_target[:, None, :], view_shape)
        intensity_target = jnp.broadcast_to(intensity_target[:, None, :], view_shape)
        per_token_ce = self._mz_cross_entropy_rows(mz_logits, mz_target)
        mz_loss = self._mae_mean(per_token_ce, target_masks)
        multiscale_term, multiscale_metrics = self._mz_multiscale_loss(
            mz_logits,
            mz_target,
            target_masks,
        )
        mz_objective = mz_loss
        if multiscale_metrics:
            mz_objective = mz_loss + self.mae_mz_multiscale_weight * multiscale_term
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
            return (
                mz_objective,
                mz_loss,
                zero,
                mz_accuracy,
                zero,
                per_token_ce,
                multiscale_metrics,
            )
        assert self.jepa_mae_intensity_head is not None
        intensity_logits = self.jepa_mae_intensity_head(predicted_latents)
        intensity_loss = _masked_ce_loss_from_logits(
            intensity_logits,
            intensity_target,
            target_masks,
        )
        value_loss = mz_objective + intensity_loss * self.mae_intensity_loss_weight
        intensity_accuracy = zero
        if compute_accuracy:
            intensity_accuracy = (
                (
                    (jnp.argmax(intensity_logits, axis=-1) == intensity_target).astype(jnp.float32)
                    * target_weights
                ).sum()
                / jnp.maximum(target_weights.sum(), 1.0)
            )
        return (
            value_loss,
            mz_loss,
            intensity_loss,
            mz_accuracy,
            intensity_accuracy,
            per_token_ce,
            multiscale_metrics,
        )

    def _mae_metrics(
        self,
        predictor_output: Array,
        peak_mz: Array,
        peak_intensity: Array,
        target_masks: Array,
        reference: Array,
        *,
        compute_accuracy: bool = True,
        target_gap_da: Array | None = None,
    ) -> tuple[Array, dict[str, Array]]:
        del reference
        (
            value_loss,
            mz_loss,
            intensity_loss,
            mz_accuracy,
            intensity_accuracy,
            per_token_ce,
            multiscale_metrics,
        ) = self._jepa_mae_value_prediction_loss(
            predictor_output,
            peak_mz,
            peak_intensity,
            target_masks,
            compute_accuracy=compute_accuracy,
        )
        extra_metrics = dict(multiscale_metrics)
        if compute_accuracy:
            extra_metrics.update(
                self._gap_bucket_metrics(per_token_ce, target_masks, target_gap_da)
            )
        term = value_loss * self.mae_loss_weight
        return term, {
            "mae_loss": value_loss,
            "mae_term": term,
            "mae_mz_loss": mz_loss,
            "mae_intensity_loss": intensity_loss,
            "mae_mz_accuracy": mz_accuracy,
            "mae_intensity_accuracy": intensity_accuracy,
            **extra_metrics,
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
        target_gap_da: Array | None = None,
    ) -> tuple[Array, dict[str, Array]]:
        del reference
        assert self.jepa_mae_mz_head is not None
        mz_target, intensity_target = self._jepa_mae_targets(
            peak_mz_compact,
            peak_intensity_compact,
        )
        extra_metrics: dict[str, Array] = {}
        multiscale_term = None
        if (
            compute_accuracy
            or self._use_mz_multiscale
            or self.mae_mz_objective == "sparse_ordinal_ce"
        ):
            mz_logits = self.jepa_mae_mz_head(predictor_output_compact)
            per_token_ce = self._mz_cross_entropy_rows(mz_logits, mz_target)
            mz_loss = self._mae_mean(per_token_ce, target_slot)
            target_weights = target_slot.astype(jnp.float32)
            mz_accuracy = (
                (
                    (jnp.argmax(mz_logits, axis=-1) == mz_target).astype(jnp.float32)
                    * target_weights
                ).sum()
                / jnp.maximum(target_weights.sum(), 1.0)
                if compute_accuracy
                else mz_loss * 0.0
            )
            if self._use_mz_multiscale:
                multiscale_term, multiscale_metrics = self._mz_multiscale_loss(
                    mz_logits,
                    mz_target,
                    target_slot,
                )
                extra_metrics.update(multiscale_metrics)
            if compute_accuracy:
                extra_metrics.update(
                    self._gap_bucket_metrics(
                        per_token_ce,
                        target_slot,
                        target_gap_da,
                    )
                )
        else:
            mz_loss = _linear_head_masked_ce_loss(
                self.jepa_mae_mz_head,
                predictor_output_compact,
                mz_target,
                target_slot,
                per_example=self.mae_objective == "absorbing_diffusion",
            )
            mz_accuracy = mz_loss * 0.0

        zero = mz_loss * 0.0
        mz_objective = mz_loss
        if multiscale_term is not None:
            mz_objective = mz_loss + self.mae_mz_multiscale_weight * multiscale_term
        if (
            self.masked_token_input_mode == "mz_sentinel"
            or self.mae_intensity_loss_weight <= 0.0
        ):
            value_loss = mz_objective
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
            value_loss = mz_objective + intensity_loss * self.mae_intensity_loss_weight
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
            **extra_metrics,
        }

    def _mae_mean(self, values: Array, mask: Array) -> Array:
        if self.mae_objective == "absorbing_diffusion":
            return _masked_per_example_mean(values, mask)
        return _masked_mean(values, mask)

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
        value_loss = self._jepa_mae_value_prediction_loss(
            predictor_output,
            peak_mz,
            peak_intensity,
            target_masks,
            compute_accuracy=False,
        )[0]
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
            spectrum_metadata=jax_spectrum_metadata_from_batch(
                batch, self.encoder_metadata_schema
            ),
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
        if self.distogram_pair_proj is not None:
            self.distogram_pair_proj.load_torch_state_dict(
                state_dict,
                "distogram_pair_proj",
            )
        if self.distogram_head is not None:
            self.distogram_head.load_torch_state_dict(
                state_dict,
                "distogram_head",
            )


_GAP_SENTINEL_DA = 1.0e6


def _distogram_score_rows(
    head: Linear,
    pair_features: Array,
    targets: Array,
    widths: tuple[int, ...],
) -> dict[int, Array]:
    return _grouped_marginal_ce_rows(head(pair_features), targets, widths)


def _grouped_marginal_ce_rows(
    logits: Array,
    targets: Array,
    widths: tuple[int, ...],
) -> dict[int, Array]:
    """`_coarse_marginal_ce_rows` without materialising every coarse bin.

    The coarse loss only ever reads the group that holds the target, so the
    other ``num_bins / width - 1`` groups are computed and thrown away:

        -log p_coarse[g(y)] = logsumexp(all logits)
                              - logsumexp(logits in group g(y))

    Gathering the ``width`` members of that one group costs ``sum(widths)``
    values per row against ``len(widths) * num_bins``. On the distogram, whose
    rows are [batch, targets, partners] rather than [batch, targets], that is
    126 values instead of 12,000 and is the difference between fitting on a
    v6e chip and not.

    Indices past the last bin are the padding the coarse form adds when
    ``num_bins`` is not a multiple of ``width``; masking them to -inf leaves
    the group sum unchanged, so this matches `_coarse_marginal_ce_rows`.
    """
    logits = logits.astype(jnp.float32)
    num_bins = logits.shape[-1]
    log_partition = logsumexp(logits, axis=-1)
    rows: dict[int, Array] = {}
    for width in widths:
        if width == 1:
            group = jnp.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
        else:
            index = (
                jnp.floor_divide(targets, width) * width
            )[..., None] + jnp.arange(width)
            members = jnp.take_along_axis(
                logits,
                jnp.minimum(index, num_bins - 1),
                axis=-1,
            )
            group = logsumexp(
                jnp.where(index < num_bins, members, -jnp.inf),
                axis=-1,
            )
        rows[width] = log_partition - group
    return rows


def _coarse_marginal_ce_rows(
    logits: Array,
    targets: Array,
    widths: tuple[int, ...],
) -> dict[int, Array]:
    """Per-token cross entropy of the softmax marginalised onto coarser bins.

    A flat categorical loss over 0.5 Da bins scores a 3 Da miss and a 300 Da miss
    identically. Summing the same softmax over groups of ``width`` adjacent bins
    yields a proper scoring rule at that coarser resolution, so the model is also
    rewarded for landing in the right neighbourhood.
    """
    log_probabilities = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    num_bins = log_probabilities.shape[-1]
    rows: dict[int, Array] = {}
    for width in widths:
        padding = (-num_bins) % width
        padded = (
            log_probabilities
            if padding == 0
            else jnp.pad(
                log_probabilities,
                [(0, 0)] * (log_probabilities.ndim - 1) + [(0, padding)],
                constant_values=-jnp.inf,
            )
        )
        grouped = padded.reshape(*padded.shape[:-1], -1, width)
        coarse = logsumexp(grouped, axis=-1)
        index = jnp.floor_divide(targets, width)[..., None]
        rows[width] = -jnp.take_along_axis(coarse, index, axis=-1)[..., 0]
    return rows


def _masked_mean(values: Array, valid_mask: Array) -> Array:
    weights = valid_mask.astype(jnp.float32)
    return (values * weights).sum() / jnp.maximum(weights.sum(), 1.0)


def _masked_per_example_mean(values: Array, valid_mask: Array) -> Array:
    """Uniform K sampling needs CE/K per spectrum, with no extra 1/r factor."""
    weights = valid_mask.astype(jnp.float32)
    axes = tuple(range(1, values.ndim))
    return jnp.mean(
        (values * weights).sum(axis=axes)
        / jnp.maximum(weights.sum(axis=axes), 1.0)
    )


def _nearest_context_gap_da(
    target_mz: Array,
    peak_mz: Array,
    context_mask: Array,
    mz_max: float,
) -> Array:
    """Distance in Da from each target peak to the nearest visible context peak."""
    distance = jnp.abs(target_mz[..., None] - peak_mz[..., None, :])
    distance = jnp.where(
        context_mask[..., None, :],
        distance,
        jnp.asarray(_GAP_SENTINEL_DA / max(mz_max, 1.0), dtype=distance.dtype),
    )
    return distance.min(axis=-1) * mz_max


def _gap_bucket_labels(edges: tuple[float, ...]) -> tuple[str, ...]:
    labels = []
    lower = 0.0
    for edge in edges:
        labels.append(f"{lower:g}_{edge:g}")
        lower = edge
    labels.append(f"{lower:g}_inf")
    return tuple(labels)


def _cross_entropy_from_logits(logits: Array, targets: Array) -> Array:
    logits = logits.astype(jnp.float32)
    target_logits = jnp.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
    return logsumexp(logits, axis=-1) - target_logits


def _sparse_ordinal_cross_entropy_from_logits(
    logits: Array,
    targets: Array,
    radius: int,
    epsilon: float,
) -> Array:
    logits = logits.astype(jnp.float32)
    offsets = jnp.arange(-radius, radius + 1)
    indices = targets[..., None] + offsets
    valid = (indices >= 0) & (indices < logits.shape[-1])
    indices = jnp.clip(indices, 0, logits.shape[-1] - 1)

    triangular = (radius + 1 - jnp.abs(offsets)).astype(logits.dtype)
    local_weights = triangular * valid.astype(logits.dtype)
    local_weights = local_weights / local_weights.sum(axis=-1, keepdims=True)
    center = (offsets == 0).astype(logits.dtype)
    target_weights = epsilon * local_weights + (1.0 - epsilon) * center

    indexed_logits = jnp.take_along_axis(logits, indices, axis=-1)
    expected_target_logit = (target_weights * indexed_logits).sum(axis=-1)
    return logsumexp(logits, axis=-1) - expected_target_logit


def _masked_ce_loss_from_logits(
    logits: Array,
    targets: Array,
    valid_mask: Array,
) -> Array:
    per_token = _cross_entropy_from_logits(logits, targets)
    weights = valid_mask.astype(jnp.float32)
    return (per_token * weights).sum() / jnp.maximum(weights.sum(), 1.0)


def _linear_head_masked_ce_loss(
    linear: Linear | Identity,
    x: Array,
    targets: Array,
    valid_mask: Array,
    *,
    per_example: bool = False,
) -> Array:
    loss_rows, weight_rows = _linear_head_masked_ce_rows(
        linear,
        x,
        targets,
        valid_mask,
    )
    if per_example:
        return _masked_per_example_mean(loss_rows, valid_mask)
    return loss_rows.sum() / jnp.maximum(weight_rows.sum(), 1.0)


def _linear_head_masked_ce_rows(
    linear: Linear | Identity,
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
