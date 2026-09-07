from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from spectra_learning.models.common_jax import Array, resolve_jax_compute_dtype
from spectra_learning.models.encoder_jax import PeakSetEncoder
from spectra_learning.models.model_jax import build_peak_set_encoder_jax
from spectra_learning.models.settings import PeakSetJEPASettings
from spectra_learning.models.spectrum_metadata import jax_spectrum_metadata_from_batch


def sigreg_loss(
    embeddings: Array,
    directions: Array,
    *,
    knots: int = 17,
) -> Array:
    embeddings = embeddings.astype(jnp.float32)
    t = jnp.linspace(0.0, 3.0, knots, dtype=jnp.float32)
    dt = 3.0 / (knots - 1)
    trapezoid_weights = jnp.full((knots,), 2.0 * dt, dtype=jnp.float32)
    trapezoid_weights = trapezoid_weights.at[0].set(dt)
    trapezoid_weights = trapezoid_weights.at[-1].set(dt)
    normal_characteristic_function = jnp.exp(-jnp.square(t) / 2.0)
    integration_weights = trapezoid_weights * normal_characteristic_function

    projected = embeddings @ directions
    characteristic_arguments = projected[..., None] * t
    cosine_mean = jnp.cos(characteristic_arguments).mean(axis=-3)
    sine_mean = jnp.sin(characteristic_arguments).mean(axis=-3)
    error = (
        jnp.square(cosine_mean - normal_characteristic_function)
        + jnp.square(sine_mean)
    )
    statistic = error @ integration_weights
    return statistic.mean() * embeddings.shape[-2]


def sample_sigreg_directions(
    model_dim: int,
    num_slices: int,
    *,
    seed: int,
    step: Array,
) -> Array:
    key = jax.random.fold_in(jax.random.key(seed), step)
    directions = jax.random.normal(
        key,
        (model_dim, num_slices),
        dtype=jnp.float32,
    )
    return directions / jnp.linalg.norm(directions, axis=0, keepdims=True)


def sjepa_metrics(
    global_cls: Array,
    local_cls: Array,
    directions: Array,
    *,
    invariance_loss_weight: float,
    sigreg_loss_weight: float,
    teacher_stop_gradient: bool = True,
    sigreg_global_cls: Array | None = None,
    sigreg_local_cls: Array | None = None,
) -> dict[str, Array]:
    teacher_cls = global_cls.mean(axis=1)
    if teacher_stop_gradient:
        teacher_cls = jax.lax.stop_gradient(teacher_cls)
    squared_error = jnp.square(teacher_cls[:, None, :] - local_cls)
    invariance_loss = squared_error.mean()
    embeddings = jnp.concatenate([teacher_cls[:, None, :], local_cls], axis=1)
    sigreg_teacher_cls = (
        global_cls if sigreg_global_cls is None else sigreg_global_cls
    ).mean(axis=1)
    if teacher_stop_gradient:
        sigreg_teacher_cls = jax.lax.stop_gradient(sigreg_teacher_cls)
    sigreg_embeddings = jnp.concatenate(
        [
            sigreg_teacher_cls[:, None, :],
            local_cls if sigreg_local_cls is None else sigreg_local_cls,
        ],
        axis=1,
    ).swapaxes(0, 1)
    regularizer_loss = sigreg_loss(sigreg_embeddings, directions)
    invariance_term = invariance_loss_weight * invariance_loss
    regularizer_term = sigreg_loss_weight * regularizer_loss
    return {
        "loss": invariance_term + regularizer_term,
        "sjepa_loss": invariance_loss,
        "sjepa_invariance_term": invariance_term,
        "sigreg_loss": regularizer_loss,
        "sjepa_sigreg_term": regularizer_term,
        "global_cls_rms": jnp.sqrt(jnp.mean(jnp.square(teacher_cls))),
        "local_cls_rms": jnp.sqrt(jnp.mean(jnp.square(local_cls))),
        "cls_variance": jnp.var(
            embeddings.reshape(-1, embeddings.shape[-1]),
            axis=0,
        ).mean(),
    }


class SJEPASpectrumJax(nnx.Module):
    def __init__(
        self,
        settings: PeakSetJEPASettings,
        *,
        num_global_views: int,
        num_local_views: int,
        global_view_fraction: float,
        local_view_fraction: float,
        invariance_loss_weight: float,
        sigreg_loss_weight: float,
        teacher_stop_gradient: bool = True,
        sigreg_num_slices: int,
        sigreg_data_axis_name: str | None,
        sigreg_seed: int,
        rngs: nnx.Rngs,
    ) -> None:
        self.encoder: PeakSetEncoder = build_peak_set_encoder_jax(
            settings,
            compute_dtype=resolve_jax_compute_dtype(settings.autocast_dtype),
            pairmixer_block_type=settings.pairmixer_block_type.lower(),
            pairmixer_fast_encoder_max_visible_tokens=(
                settings.pairmixer_fast_encoder_max_visible_tokens
            ),
            rngs=rngs,
        )
        self.encoder_metadata_schema = settings.encoder_metadata_schema
        self.num_global_views = num_global_views
        self.num_local_views = num_local_views
        self.global_view_max_tokens = (
            max(round(settings.num_peaks * global_view_fraction), 1) + 1
        )
        self.local_view_max_tokens = (
            max(round(settings.num_peaks * local_view_fraction), 1) + 1
        )
        self.invariance_loss_weight = invariance_loss_weight
        self.sigreg_loss_weight = sigreg_loss_weight
        self.teacher_stop_gradient = teacher_stop_gradient
        self.sigreg_num_slices = sigreg_num_slices
        self.sigreg_data_axis_name = sigreg_data_axis_name
        self.sigreg_seed = sigreg_seed
        self.model_dim = settings.model_dim

    def __call__(self, batch: dict[str, Any]) -> dict[str, Array]:
        global_masks = (
            batch["sjepa_global_view_masks"]
            & batch["peak_valid_mask"][:, None, :]
        )
        local_masks = (
            batch["sjepa_local_view_masks"]
            & batch["peak_valid_mask"][:, None, :]
        )
        global_cls = self._encode_views(
            batch,
            global_masks,
            max_visible_tokens=self.global_view_max_tokens,
        )
        local_cls = self._encode_views(
            batch,
            local_masks,
            max_visible_tokens=self.local_view_max_tokens,
        )
        sigreg_global_cls = global_cls
        sigreg_local_cls = local_cls
        if self.sigreg_data_axis_name is not None:
            sigreg_global_cls = jax.lax.all_gather(
                sigreg_global_cls,
                self.sigreg_data_axis_name,
                axis=0,
                tiled=True,
            )
            sigreg_local_cls = jax.lax.all_gather(
                sigreg_local_cls,
                self.sigreg_data_axis_name,
                axis=0,
                tiled=True,
            )
        sigreg_step = batch.get(
            "sjepa_sigreg_step",
            jnp.zeros((), dtype=jnp.uint32),
        ).reshape(-1)[0]
        sigreg_directions = sample_sigreg_directions(
            self.model_dim,
            self.sigreg_num_slices,
            seed=self.sigreg_seed,
            step=sigreg_step,
        )
        metrics = sjepa_metrics(
            global_cls,
            local_cls,
            sigreg_directions,
            invariance_loss_weight=self.invariance_loss_weight,
            sigreg_loss_weight=self.sigreg_loss_weight,
            teacher_stop_gradient=self.teacher_stop_gradient,
            sigreg_global_cls=sigreg_global_cls,
            sigreg_local_cls=sigreg_local_cls,
        )
        valid_peak_count = batch["peak_valid_mask"].astype(jnp.float32).sum()
        return {
            **metrics,
            "global_view_fraction": (
                global_masks.astype(jnp.float32).sum()
                / (valid_peak_count * self.num_global_views)
            ),
            "local_view_fraction": (
                local_masks.astype(jnp.float32).sum()
                / (valid_peak_count * self.num_local_views)
            ),
        }

    def _encode_views(
        self,
        batch: dict[str, Any],
        view_masks: Array,
        *,
        max_visible_tokens: int,
    ) -> Array:
        batch_size, num_views, num_peaks = view_masks.shape
        peak_mz = _repeat_views(batch["peak_mz"], num_views)
        peak_intensity = _repeat_views(batch["peak_intensity"], num_views)
        valid_mask = _repeat_views(batch["peak_valid_mask"], num_views)
        precursor_mz = batch.get("precursor_mz", None)
        if precursor_mz is not None:
            precursor_mz = _repeat_views(precursor_mz, num_views)
        spectrum_metadata = jax_spectrum_metadata_from_batch(
            batch,
            self.encoder_metadata_schema,
        )
        if spectrum_metadata is not None:
            spectrum_metadata = _repeat_views(spectrum_metadata, num_views)
        flat_view_masks = view_masks.reshape(batch_size * num_views, num_peaks)
        encoded, _positions, _compact_mask = self.encoder.forward_compact(
            peak_mz,
            peak_intensity,
            valid_mask=valid_mask,
            visible_mask=flat_view_masks,
            max_visible_tokens=max_visible_tokens,
            precursor_mz=precursor_mz,
            spectrum_metadata=spectrum_metadata,
        )
        cls_positions = flat_view_masks.sum(axis=-1).astype(jnp.int32)
        cls = jnp.take_along_axis(
            encoded,
            cls_positions[:, None, None],
            axis=1,
        )[:, 0]
        return cls.astype(jnp.float32).reshape(
            batch_size,
            num_views,
            encoded.shape[-1],
        )


def _repeat_views(value: Array, num_views: int) -> Array:
    repeated = jnp.broadcast_to(
        value[:, None],
        (value.shape[0], num_views, *value.shape[1:]),
    )
    return repeated.reshape(value.shape[0] * num_views, *value.shape[1:])
