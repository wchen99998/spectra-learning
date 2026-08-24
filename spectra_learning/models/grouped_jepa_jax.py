from __future__ import annotations

import copy
import math
from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from flax import nnx

from spectra_learning.models.common_jax import Array, Identity, Linear, RMSNorm
from spectra_learning.models.encoder_jax import PeakSetEncoder
from spectra_learning.models.model_jax import PeakSetJEPAJax
from spectra_learning.models.settings import PeakSetJEPASettings
from spectra_learning.models.spectrum_metadata import jax_spectrum_metadata_from_batch
from spectra_learning.models.transformer_jax import CrossAttentionBlock


GROUP_JEPA_TEACHER_TARGET_KEY = "group_jepa_teacher_target"
GROUP_JEPA_TEACHER_TARGET_AGE_KEY = "group_jepa_teacher_target_age"


class GroupedJEPACrossAttentionPredictor(nnx.Module):
    def __init__(
        self,
        settings: PeakSetJEPASettings,
        *,
        compute_dtype: object,
        rngs: nnx.Rngs,
    ) -> None:
        model_dim = settings.model_dim
        predictor_dim = settings.predictor_dim or model_dim
        self.query_token = nnx.Param(
            rngs.params.normal((predictor_dim,), dtype=jnp.float32) * 0.02
        )
        self.memory_projection = (
            Linear(
                model_dim,
                predictor_dim,
                bias=False,
                compute_dtype=compute_dtype,
                rngs=rngs,
            )
            if predictor_dim != model_dim
            else Identity()
        )
        self.blocks = nnx.List(
            [
                CrossAttentionBlock(
                    dim=predictor_dim,
                    n_heads=settings.masked_latent_predictor_num_heads,
                    norm_eps=settings.norm_eps,
                    hidden_dim=math.ceil(
                        predictor_dim * settings.predictor_mlp_multiple
                    ),
                    max_sequence_length=(
                        settings.num_peaks + int(settings.encoder_use_cls_token)
                    ),
                    compute_dtype=compute_dtype,
                    rngs=rngs,
                )
                for _ in range(settings.masked_latent_predictor_num_layers)
            ]
        )
        self.final_norm = (
            RMSNorm(predictor_dim, eps=settings.norm_eps, affine=False)
            if settings.predictor_apply_final_norm
            else Identity()
        )
        self.readout = Linear(
            predictor_dim,
            model_dim,
            compute_dtype=compute_dtype,
            rngs=rngs,
        )

    def __call__(self, memory: Array, peak_valid_mask: Array) -> Array:
        batch_size, num_tokens, _model_dim = memory.shape
        memory = self.memory_projection(memory)
        memory_mask = jnp.concatenate(
            [
                peak_valid_mask,
                jnp.ones((batch_size, 1), dtype=jnp.bool_),
            ],
            axis=1,
        )
        memory_positions = jnp.broadcast_to(
            jnp.arange(num_tokens, dtype=jnp.int32)[None],
            (batch_size, num_tokens),
        )
        query = jnp.broadcast_to(
            self.query_token[None, None],
            (batch_size, 1, self.query_token.shape[0]),
        )
        query_positions = jnp.full(
            (batch_size, 1),
            num_tokens - 1,
            dtype=jnp.int32,
        )
        query_mask = jnp.ones((batch_size, 1), dtype=jnp.bool_)
        for block in self.blocks:
            query = block(
                query,
                memory,
                query_positions,
                memory_positions,
                query_mask,
                memory_mask,
            )
        return self.readout(self.final_norm(query[:, 0]))


class GroupedSpectrumJEPAJax(nnx.Module):
    def __init__(
        self,
        settings: PeakSetJEPASettings,
        *,
        teacher_spectra_per_group: int,
        ema_momentum: float | None,
        invariance_loss_weight: float = 1.0,
        visreg_loss_weight: float = 0.0,
        visreg_num_projections: int = 0,
        visreg_center_weight: float = 1.0,
        visreg_scale_weight: float = 1.0,
        visreg_shape_weight: float = 1.0,
        visreg_data_axis_name: str | None = None,
        rngs: nnx.Rngs,
    ) -> None:
        student = PeakSetJEPAJax(settings, rngs=rngs)
        self.encoder: PeakSetEncoder = student.encoder
        self.teacher_encoder: PeakSetEncoder | None = (
            copy.deepcopy(self.encoder) if ema_momentum is not None else None
        )
        self.predictor = GroupedJEPACrossAttentionPredictor(
            settings,
            compute_dtype=student.compute_dtype,
            rngs=rngs,
        )
        self.teacher_spectra_per_group = teacher_spectra_per_group
        self.ema_momentum = ema_momentum
        self.invariance_loss_weight = invariance_loss_weight
        self.visreg_loss_weight = visreg_loss_weight
        self.visreg_center_weight = visreg_center_weight
        self.visreg_scale_weight = visreg_scale_weight
        self.visreg_shape_weight = visreg_shape_weight
        self.visreg_data_axis_name = visreg_data_axis_name
        self.visreg_directions = (
            _normalized_random_directions(
                settings.model_dim,
                visreg_num_projections,
                rngs.params(),
            )
            if visreg_loss_weight > 0.0
            else None
        )

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        teacher_target = batch.get(GROUP_JEPA_TEACHER_TARGET_KEY)
        target_age_steps = batch.get(
            GROUP_JEPA_TEACHER_TARGET_AGE_KEY,
            0,
        )
        spectra_batch = {
            key: value
            for key, value in batch.items()
            if key
            not in {
                GROUP_JEPA_TEACHER_TARGET_KEY,
                GROUP_JEPA_TEACHER_TARGET_AGE_KEY,
            }
        }
        if teacher_target is None:
            if self.teacher_encoder is None:
                teacher_cls = self._teacher_cls(self.encoder, spectra_batch)
                teacher_target = jax.lax.stop_gradient(
                    teacher_cls.astype(jnp.float32).mean(axis=1)
                )
            else:
                teacher_cls = None
                teacher_target = self.teacher_target(spectra_batch)
            target_age_steps = 0
        else:
            teacher_cls = None
        return self.student_metrics(
            spectra_batch,
            jax.lax.stop_gradient(teacher_target),
            target_age_steps=target_age_steps,
            teacher_cls=teacher_cls,
        )

    def teacher_target(self, batch: dict[str, Any]) -> Any:
        teacher_count = self.teacher_spectra_per_group
        teacher_batch = {
            key: value[:, :teacher_count]
            for key, value in batch.items()
        }
        teacher_encoder = (
            self.teacher_encoder
            if self.teacher_encoder is not None
            else self.encoder
        )
        teacher_cls = self._encode_cls(teacher_encoder, teacher_batch)
        return jax.lax.stop_gradient(
            teacher_cls.astype(jnp.float32).mean(axis=1)
        )

    def student_metrics(
        self,
        batch: dict[str, Any],
        teacher_target: Any,
        *,
        target_age_steps: Any,
        teacher_cls: Any | None = None,
    ) -> dict[str, Any]:
        teacher_count = self.teacher_spectra_per_group
        student_batch = {
            key: value[:, teacher_count:]
            for key, value in batch.items()
        }
        student_tokens = self._encode_tokens(self.encoder, student_batch)
        student_cls = student_tokens[..., -1, :].astype(jnp.float32)
        groups, spectra, tokens, model_dim = student_tokens.shape
        student_prediction = self.predictor(
            student_tokens.reshape(groups * spectra, tokens, model_dim),
            student_batch["peak_valid_mask"].reshape(
                groups * spectra,
                student_batch["peak_valid_mask"].shape[-1],
            ),
        ).reshape(groups, spectra, model_dim).astype(jnp.float32)
        squared_error = jnp.square(
            student_prediction - teacher_target[:, None, :]
        )
        per_student_loss = squared_error.mean(axis=-1)
        invariance_loss = per_student_loss.mean()
        visreg_metrics = self._visreg_metrics(teacher_cls, student_cls)
        invariance_term = invariance_loss * self.invariance_loss_weight
        visreg_term = visreg_metrics["visreg_loss"] * self.visreg_loss_weight
        loss = invariance_term + visreg_term
        teacher_rms = jnp.sqrt(jnp.mean(jnp.square(teacher_target)))
        student_rms = jnp.sqrt(jnp.mean(jnp.square(student_cls)))
        prediction_rms = jnp.sqrt(jnp.mean(jnp.square(student_prediction)))
        return {
            "loss": loss,
            "group_jepa_loss": invariance_loss,
            "group_jepa_invariance_term": invariance_term,
            "group_jepa_visreg_term": visreg_term,
            "teacher_cls_rms": teacher_rms,
            "student_cls_rms": student_rms,
            "teacher_cls_variance": jnp.var(teacher_target, axis=0).mean(),
            "student_cls_variance": jnp.var(
                student_cls.reshape(-1, student_cls.shape[-1]),
                axis=0,
            ).mean(),
            "student_prediction_rms": prediction_rms,
            "student_prediction_variance": jnp.var(
                student_prediction.reshape(-1, student_prediction.shape[-1]),
                axis=0,
            ).mean(),
            "ema_teacher_momentum": jnp.asarray(
                self.ema_momentum or 0.0,
                dtype=jnp.float32,
            ),
            "teacher_target_age_steps": jnp.asarray(
                target_age_steps,
                dtype=jnp.float32,
            ).mean(),
            **visreg_metrics,
        }

    def _visreg_metrics(
        self,
        teacher_cls: Any | None,
        student_cls: Any,
    ) -> dict[str, Any]:
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        if self.visreg_directions is None:
            return {
                "visreg_loss": zero,
                "visreg_center_loss": zero,
                "visreg_scale_loss": zero,
                "visreg_shape_loss": zero,
            }
        embeddings = student_cls
        if teacher_cls is not None:
            embeddings = jnp.concatenate(
                [teacher_cls.astype(jnp.float32), embeddings],
                axis=1,
            )
        embeddings = jnp.swapaxes(embeddings, 0, 1)
        if self.visreg_data_axis_name is not None:
            embeddings = jax.lax.all_gather(
                embeddings,
                self.visreg_data_axis_name,
                axis=1,
                tiled=True,
            )
        return visreg_metrics(
            embeddings,
            self.visreg_directions,
            center_weight=self.visreg_center_weight,
            scale_weight=self.visreg_scale_weight,
            shape_weight=self.visreg_shape_weight,
        )

    def _teacher_cls(
        self,
        encoder: PeakSetEncoder,
        batch: dict[str, Any],
    ) -> Any:
        teacher_count = self.teacher_spectra_per_group
        teacher_batch = {
            key: value[:, :teacher_count]
            for key, value in batch.items()
        }
        return self._encode_cls(encoder, teacher_batch)

    @staticmethod
    def _encode_cls(
        encoder: PeakSetEncoder,
        batch: dict[str, Any],
    ) -> Any:
        return GroupedSpectrumJEPAJax._encode_tokens(encoder, batch)[..., -1, :]

    @staticmethod
    def _encode_tokens(
        encoder: PeakSetEncoder,
        batch: dict[str, Any],
    ) -> Any:
        groups, spectra = batch["peak_mz"].shape[:2]
        flat_batch = {
            key: value.reshape(groups * spectra, *value.shape[2:])
            for key, value in batch.items()
        }
        encoded = encoder(
            flat_batch["peak_mz"],
            flat_batch["peak_intensity"],
            valid_mask=flat_batch["peak_valid_mask"],
            visible_mask=flat_batch["peak_valid_mask"],
            precursor_mz=flat_batch.get("precursor_mz", None),
            spectrum_metadata=jax_spectrum_metadata_from_batch(flat_batch),
        )
        return encoded.reshape(
            groups,
            spectra,
            encoded.shape[-2],
            encoded.shape[-1],
        )


def _normalized_random_directions(
    embedding_dim: int,
    num_projections: int,
    key: Array,
) -> Array:
    directions = jax.random.normal(
        key,
        (embedding_dim, num_projections),
        dtype=jnp.float32,
    )
    return directions / jnp.linalg.norm(directions, axis=0, keepdims=True)


def visreg_metrics(
    embeddings: Array,
    directions: Array,
    *,
    center_weight: float,
    scale_weight: float,
    shape_weight: float,
) -> dict[str, Array]:
    batch_size = embeddings.shape[1]
    mean = embeddings.mean(axis=1, keepdims=True)
    center_loss = jnp.square(mean).mean()

    centered = embeddings - mean
    std = jnp.sqrt(
        jnp.maximum(jnp.sum(jnp.square(centered), axis=1), 1e-12)
        / batch_size
    )
    scale_loss = jnp.square(std - 1.0).mean()

    normalized = centered / jax.lax.stop_gradient(std[:, None])
    projected = jnp.einsum("vbd,dk->vbk", normalized, directions)
    projected = jnp.sort(projected, axis=1)
    quantiles = (
        jnp.arange(1, batch_size + 1, dtype=jnp.float32)
        / (batch_size + 1)
    )
    normal_quantiles = (
        math.sqrt(2.0) * jsp.special.erfinv(2.0 * quantiles - 1.0)
    )
    shape_loss = jnp.square(
        projected - normal_quantiles[None, :, None]
    ).mean()
    loss = (
        center_weight * center_loss
        + scale_weight * scale_loss
        + shape_weight * shape_loss
    )
    return {
        "visreg_loss": loss,
        "visreg_center_loss": center_loss,
        "visreg_scale_loss": scale_loss,
        "visreg_shape_loss": shape_loss,
    }
