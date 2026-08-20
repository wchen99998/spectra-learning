from __future__ import annotations

import math
from typing import cast

import jax
import jax.numpy as jnp
import torch
from flax import nnx

from spectra_learning.data.spectra import PEAK_MZ_MAX
from spectra_learning.models.common_jax import (
    Array,
    Linear,
    RMSNorm,
    pair_mask,
    scaled_dot_product_attention,
    silu,
)
from spectra_learning.models.peak_features_jax import FourierFeatures, MzTokenEmbedding
from spectra_learning.models.transformer_jax import FeedForward, SwiGLUFeedForward


SUPPORTED_PAIRMIXER_TRANSITION_TYPES = {"swiglu", "feedforward"}


def _apply_rope(tensor: Array, positions: Array) -> Array:
    head_dim = tensor.shape[-1]
    inv_freq = 1.0 / (
        10_000.0
        ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim)
    )
    angles = positions[..., None] * inv_freq
    cos = jnp.cos(angles)[:, None].astype(tensor.dtype)
    sin = jnp.sin(angles)[:, None].astype(tensor.dtype)
    even = tensor[..., 0::2]
    odd = tensor[..., 1::2]
    return jnp.stack(
        (even * cos - odd * sin, even * sin + odd * cos),
        axis=-1,
    ).reshape(tensor.shape)


def _preferred_acc_dtype(dtype: object) -> object | None:
    dtype = jnp.dtype(dtype)
    if dtype == jnp.bfloat16:
        return jnp.float32
    return None


def _dot_precision(dtype: object) -> object | None:
    dtype = jnp.dtype(dtype)
    if dtype == jnp.bfloat16:
        return jax.lax.Precision.DEFAULT
    return jax.lax.Precision.HIGHEST


def _linear_with_preferred_acc(linear: Linear, x: Array) -> Array:
    compute_dtype = linear.compute_dtype
    x = x.astype(compute_dtype)
    weight = linear.weight[...].astype(compute_dtype)
    acc_dtype = _preferred_acc_dtype(compute_dtype)
    y = jax.lax.dot_general(
        x,
        weight,
        dimension_numbers=(
            ((x.ndim - 1,), (weight.ndim - 1,)),
            ((), ()),
        ),
        precision=_dot_precision(compute_dtype),
        preferred_element_type=acc_dtype,
    )
    if linear.bias is not None:
        y = y + linear.bias[...].astype(y.dtype)
    if acc_dtype is not None:
        y = y.astype(compute_dtype)
    return y


def _swiglu_with_preferred_acc(feed_forward: SwiGLUFeedForward, x: Array) -> Array:
    return _linear_with_preferred_acc(
        feed_forward.fc3,
        silu(_linear_with_preferred_acc(feed_forward.fc1, x))
        * _linear_with_preferred_acc(feed_forward.fc2, x),
    )


def _feedforward_with_preferred_acc(feed_forward: FeedForward, x: Array) -> Array:
    return _linear_with_preferred_acc(
        feed_forward.w2,
        silu(_linear_with_preferred_acc(feed_forward.w1, x)),
    )


def _transition_with_preferred_acc(
    feed_forward: FeedForward | SwiGLUFeedForward,
    x: Array,
) -> Array:
    if isinstance(feed_forward, SwiGLUFeedForward):
        return _swiglu_with_preferred_acc(feed_forward, x)
    return _feedforward_with_preferred_acc(feed_forward, x)


def _build_pairmixer_transition(
    dim: int,
    *,
    hidden_dim: int,
    transition_type: str,
    compute_dtype: object,
    rngs: nnx.Rngs,
) -> FeedForward | SwiGLUFeedForward:
    if transition_type == "swiglu":
        return SwiGLUFeedForward(
            dim,
            hidden_dim=hidden_dim,
            compute_dtype=compute_dtype,
            rngs=rngs,
        )
    if transition_type == "feedforward":
        return FeedForward(
            dim,
            hidden_dim=hidden_dim,
            compute_dtype=compute_dtype,
            rngs=rngs,
        )
    raise ValueError(
        "pairmixer_transition_type must be one of ('swiglu', 'feedforward')"
    )


def _active_indices(token_mask: Array, max_active_tokens: int) -> tuple[Array, Array]:
    batch_size, num_tokens = token_mask.shape
    compact_len = min(max_active_tokens, num_tokens)
    position = jnp.arange(num_tokens, dtype=jnp.int32)
    mask_f = token_mask.astype(jnp.float32)
    inv_mask_f = (~token_mask).astype(jnp.float32)
    active_count = jnp.sum(mask_f, axis=-1).astype(jnp.int32)
    active_rank = (jnp.cumsum(mask_f, axis=-1) - 1.0).astype(jnp.int32)
    inactive_rank = (jnp.cumsum(inv_mask_f, axis=-1) - 1.0).astype(jnp.int32)
    slot = jnp.where(token_mask, active_rank, active_count[:, None] + inactive_rank)
    batch_idx = jnp.broadcast_to(
        jnp.arange(batch_size, dtype=jnp.int32)[:, None],
        (batch_size, num_tokens),
    )
    position_idx = jnp.broadcast_to(position[None, :], (batch_size, num_tokens))
    idx_full = jnp.zeros((batch_size, num_tokens), dtype=jnp.int32)
    idx_full = idx_full.at[batch_idx, slot].set(position_idx)
    idx = idx_full[:, :compact_len]
    compact_position = jnp.arange(compact_len, dtype=jnp.int32)
    compact_mask = compact_position[None, :] < active_count[:, None]
    return idx, compact_mask


def _selector_from_idx(idx: Array, num_tokens: int, dtype: object) -> Array:
    return jax.nn.one_hot(idx, num_tokens, dtype=dtype)


def _flat_indexed_gather_rows(
    data_2d: Array,
    flat_idx: Array,
    out_shape: tuple[int, ...],
) -> Array:
    return data_2d[flat_idx.reshape(-1), :].reshape(out_shape)


def _gather_single(single: Array, idx: Array) -> Array:
    batch_size, num_tokens, dim = single.shape
    compact_len = idx.shape[1]
    batch_offsets = (
        jnp.arange(batch_size, dtype=jnp.int32)[:, None]
        * jnp.asarray(num_tokens, dtype=jnp.int32)
    )
    flat_idx = batch_offsets + idx.astype(jnp.int32)
    return _flat_indexed_gather_rows(
        single.reshape(batch_size * num_tokens, dim),
        flat_idx,
        (batch_size, compact_len, dim),
    )


def _gather_pair(pair: Array, idx: Array) -> Array:
    batch_size, num_tokens, _, pair_dim = pair.shape
    compact_len = idx.shape[1]
    batch_offsets = (
        jnp.arange(batch_size, dtype=jnp.int32)[:, None, None]
        * jnp.asarray(num_tokens * num_tokens, dtype=jnp.int32)
    )
    flat_idx = (
        batch_offsets
        + idx[:, :, None].astype(jnp.int32)
        * jnp.asarray(num_tokens, dtype=jnp.int32)
        + idx[:, None, :].astype(jnp.int32)
    )
    return _flat_indexed_gather_rows(
        pair.reshape(batch_size * num_tokens * num_tokens, pair_dim),
        flat_idx,
        (batch_size, compact_len, compact_len, pair_dim),
    )


def _scatter_pair(pair_compact: Array, idx: Array, out_shape: tuple[int, ...]) -> Array:
    num_tokens = out_shape[1]
    selector = _selector_from_idx(idx, num_tokens, pair_compact.dtype)
    precision = _dot_precision(pair_compact.dtype)
    acc_dtype = _preferred_acc_dtype(pair_compact.dtype)
    tmp = jnp.einsum(
        "bki,bklc->bilc",
        selector,
        pair_compact,
        precision=precision,
        preferred_element_type=acc_dtype,
    ).astype(pair_compact.dtype)
    out = jnp.einsum(
        "bilc,blj->bijc",
        tmp,
        selector,
        precision=precision,
        preferred_element_type=acc_dtype,
    )
    return out.astype(pair_compact.dtype).reshape(out_shape)


def _scatter_compact_pair_bias(delta: Array, idx: Array, num_tokens: int) -> Array:
    selector = _selector_from_idx(idx, num_tokens, delta.dtype)
    precision = _dot_precision(delta.dtype)
    acc_dtype = _preferred_acc_dtype(delta.dtype)
    tmp = jnp.einsum(
        "bki,bklh->bilh",
        selector,
        delta,
        precision=precision,
        preferred_element_type=acc_dtype,
    ).astype(delta.dtype)
    return jnp.einsum(
        "bilh,blj->bijh",
        tmp,
        selector,
        precision=precision,
        preferred_element_type=acc_dtype,
    )


class PairFeatureEmbedder(nnx.Module):
    def __init__(
        self,
        *,
        single_dim: int,
        pair_dim: int,
        hidden_dim: int,
        mz_scale: float = PEAK_MZ_MAX,
        precursor_mz_scale: float = PEAK_MZ_MAX,
        mz_embedding: str = "fourier",
        token_bin_size: float = 0.1,
        token_embedding_dim: int = 128,
        fourier_num_freqs: int = 16,
        fourier_x_min: float = 1e-2,
        fourier_x_max: float = PEAK_MZ_MAX,
        relative_fourier_x_min: float = 1e-3,
        relative_fourier_x_max: float = 1.0,
        compute_dtype: object = jnp.float32,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        rngs = nnx.Rngs(0) if rngs is None else rngs
        self.mz_scale = mz_scale
        self.precursor_mz_scale = precursor_mz_scale
        self.mz_embedding = mz_embedding.lower()
        if self.mz_embedding not in {"fourier", "token"}:
            raise ValueError("mz_embedding must be one of ('fourier', 'token')")
        raw_dim = 14
        if self.mz_embedding == "fourier":
            self.pair_fourier = FourierFeatures(
                x_min=fourier_x_min,
                x_max=fourier_x_max,
                num_freqs=fourier_num_freqs,
            )
            self.relative_pair_fourier = FourierFeatures(
                x_min=relative_fourier_x_min,
                x_max=relative_fourier_x_max,
                num_freqs=fourier_num_freqs,
            )
            raw_dim += 3 * self.pair_fourier.num_features()
            raw_dim += self.relative_pair_fourier.num_features()
        else:
            self.mz_features = MzTokenEmbedding(
                mz_scale=mz_scale,
                bin_size=token_bin_size,
                embedding_dim=token_embedding_dim,
            )
            raw_dim += self.mz_features.num_features()
        self.raw_proj = nnx.List(
            [
                Linear(raw_dim, hidden_dim, compute_dtype=compute_dtype, rngs=rngs),
                Linear(hidden_dim, pair_dim, compute_dtype=compute_dtype, rngs=rngs),
            ]
        )
        self.single_pair_proj = nnx.List(
            [
                Linear(
                    4 * single_dim,
                    hidden_dim,
                    compute_dtype=compute_dtype,
                    rngs=rngs,
                ),
                Linear(hidden_dim, pair_dim, compute_dtype=compute_dtype, rngs=rngs),
            ]
        )

    def _reference_mass_da(
        self,
        peak_mz: Array,
        valid_mask: Array,
        precursor_mz: Array | None,
    ) -> Array:
        if precursor_mz is not None:
            return jnp.maximum(precursor_mz.astype(jnp.float32) * self.precursor_mz_scale, 1.0)
        mz_da = peak_mz.astype(jnp.float32) * self.mz_scale
        return jnp.maximum(jnp.max(mz_da * valid_mask.astype(jnp.float32), axis=1), 1.0)

    def _fourier_values(self, fourier: FourierFeatures, values: Array) -> Array:
        encoded = fourier(values[..., None])
        return encoded.reshape(*values.shape[:-1], values.shape[-1] * fourier.num_features())

    def _raw_proj_call(self, raw: Array) -> Array:
        return self.raw_proj[1](silu(self.raw_proj[0](raw)))

    def _single_pair_proj_call(self, single_pair: Array) -> Array:
        return self.single_pair_proj[1](silu(self.single_pair_proj[0](single_pair)))

    def __call__(
        self,
        peak_mz: Array,
        peak_intensity: Array,
        single: Array,
        valid_mask: Array,
        *,
        precursor_mz: Array | None = None,
    ) -> Array:
        peak_mz = peak_mz.astype(jnp.float32)
        intensity = peak_intensity.astype(jnp.float32)
        intensity_i = intensity[:, :, None]
        intensity_j = intensity[:, None, :]
        diag = jnp.eye(peak_mz.shape[1], dtype=peak_mz.dtype)[None, :, :]
        if self.mz_embedding == "fourier":
            mz_da = peak_mz * self.mz_scale
            reference_mass = self._reference_mass_da(
                peak_mz,
                valid_mask,
                precursor_mz,
            )[:, None, None]
            mz_i = mz_da[:, :, None]
            mz_j = mz_da[:, None, :]
            d = mz_j - mz_i
            abs_d = jnp.abs(d)
            relative_d = d / reference_mass
            complement = mz_i + mz_j - reference_mass
            raw_parts = [
                d[..., None] / self.mz_scale,
                abs_d[..., None] / self.mz_scale,
                relative_d[..., None],
                complement[..., None] / self.mz_scale,
                jnp.broadcast_to(mz_i, d.shape)[..., None] / reference_mass[..., None],
                jnp.broadcast_to(mz_j, d.shape)[..., None] / reference_mass[..., None],
                jnp.broadcast_to(intensity_i, d.shape)[..., None],
                jnp.broadcast_to(intensity_j, d.shape)[..., None],
                jnp.broadcast_to(intensity_i * intensity_j, d.shape)[..., None],
                jnp.broadcast_to(jnp.log1p(intensity_i), d.shape)[..., None],
                jnp.broadcast_to(jnp.log1p(intensity_j), d.shape)[..., None],
                jnp.sign(d)[..., None],
                jnp.broadcast_to(diag, d.shape)[..., None],
                (d > 0).astype(peak_mz.dtype)[..., None],
            ]
            raw_parts.extend(
                [
                    self._fourier_values(
                        self.pair_fourier,
                        jnp.stack([d, abs_d, complement], axis=-1),
                    ),
                    self._fourier_values(
                        self.relative_pair_fourier,
                        relative_d[..., None],
                    ),
                ]
            )
        else:
            mz_token = self.mz_features.token_ids(peak_mz)
            token_delta = mz_token[:, None, :] - mz_token[:, :, None]
            raw_parts = [
                jnp.zeros((*token_delta.shape, 6), dtype=peak_mz.dtype),
                jnp.broadcast_to(intensity_i, token_delta.shape)[..., None],
                jnp.broadcast_to(intensity_j, token_delta.shape)[..., None],
                jnp.broadcast_to(
                    intensity_i * intensity_j,
                    token_delta.shape,
                )[..., None],
                jnp.broadcast_to(jnp.log1p(intensity_i), token_delta.shape)[
                    ..., None
                ],
                jnp.broadcast_to(jnp.log1p(intensity_j), token_delta.shape)[
                    ..., None
                ],
                jnp.sign(token_delta).astype(peak_mz.dtype)[..., None],
                jnp.broadcast_to(diag, token_delta.shape)[..., None],
                (token_delta > 0).astype(peak_mz.dtype)[..., None],
                self.mz_features.embedding(jnp.abs(token_delta)),
            ]
        raw = jnp.concatenate(raw_parts, axis=-1)
        single_i = single[:, :, None, :]
        single_j = single[:, None, :, :]
        num_peaks = single.shape[1]
        single_pair = jnp.concatenate(
            [
                jnp.broadcast_to(single_i, (single.shape[0], num_peaks, num_peaks, single.shape[-1])),
                jnp.broadcast_to(single_j, (single.shape[0], num_peaks, num_peaks, single.shape[-1])),
                single_i * single_j,
                single_j - single_i,
            ],
            axis=-1,
        )
        z = self._raw_proj_call(raw.astype(single.dtype)) + self._single_pair_proj_call(
            single_pair
        )
        return z * pair_mask(valid_mask)[..., None].astype(z.dtype)

    def load_torch_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
    ) -> None:
        if self.mz_embedding == "fourier":
            self.pair_fourier.load_torch_state_dict(state_dict, f"{prefix}.pair_fourier")
            self.relative_pair_fourier.load_torch_state_dict(
                state_dict,
                f"{prefix}.relative_pair_fourier",
            )
        else:
            self.mz_features.load_torch_state_dict(state_dict, f"{prefix}.mz_features")
        self.raw_proj[0].load_torch_state_dict(state_dict, f"{prefix}.raw_proj.0")
        self.raw_proj[1].load_torch_state_dict(state_dict, f"{prefix}.raw_proj.2")
        self.single_pair_proj[0].load_torch_state_dict(
            state_dict,
            f"{prefix}.single_pair_proj.0",
        )
        self.single_pair_proj[1].load_torch_state_dict(
            state_dict,
            f"{prefix}.single_pair_proj.2",
        )


class TriangleMultiplicativeUpdate(nnx.Module):
    def __init__(
        self,
        pair_dim: int,
        *,
        direction: str,
        norm_eps: float,
        compute_dtype: object = jnp.float32,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        rngs = nnx.Rngs(0) if rngs is None else rngs
        self.direction = direction
        self.compute_dtype = compute_dtype
        self.norm_in = RMSNorm(pair_dim, eps=norm_eps)
        self.p_in = Linear(pair_dim, 2 * pair_dim, compute_dtype=compute_dtype, rngs=rngs)
        self.g_in = Linear(
            pair_dim,
            2 * pair_dim,
            compute_dtype=compute_dtype,
            init="gate",
            rngs=rngs,
        )
        self.norm_out = RMSNorm(pair_dim, eps=norm_eps)
        self.p_out = Linear(
            pair_dim,
            pair_dim,
            compute_dtype=compute_dtype,
            init="zeros",
            rngs=rngs,
        )
        self.g_out = Linear(
            pair_dim,
            pair_dim,
            compute_dtype=compute_dtype,
            init="gate",
            rngs=rngs,
        )

    def __call__(self, x: Array, token_mask: Array, mask: Array) -> Array:
        pair_mask_f = mask[..., None].astype(x.dtype)
        x_norm = self.norm_in(x)
        projected = self.p_in(x_norm) * jax.nn.sigmoid(self.g_in(x_norm))
        a, b = jnp.split(projected, 2, axis=-1)
        if self.direction == "outgoing":
            a = a * token_mask[:, None, :, None].astype(a.dtype)
            update = jnp.sum(
                a[:, :, None, :, :] * b[:, None, :, :, :],
                axis=3,
            )
        else:
            a = a * token_mask[:, :, None, None].astype(a.dtype)
            update = jnp.sum(
                a[:, :, :, None, :] * b[:, :, None, :, :],
                axis=1,
            )
        update = self.p_out(self.norm_out(update))
        update = update * jax.nn.sigmoid(self.g_out(x_norm))
        return update * pair_mask_f

    def load_torch_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
    ) -> None:
        self.norm_in.load_torch_state_dict(state_dict, f"{prefix}.norm_in")
        self.p_in.load_torch_state_dict(state_dict, f"{prefix}.p_in")
        self.g_in.load_torch_state_dict(state_dict, f"{prefix}.g_in")
        self.norm_out.load_torch_state_dict(state_dict, f"{prefix}.norm_out")
        self.p_out.load_torch_state_dict(state_dict, f"{prefix}.p_out")
        self.g_out.load_torch_state_dict(state_dict, f"{prefix}.g_out")


class AttentionPairBias(nnx.Module):
    def __init__(
        self,
        *,
        single_dim: int,
        pair_dim: int,
        num_heads: int,
        norm_eps: float,
        use_pair_bias: bool = True,
        use_rope: bool = True,
        input_norm_affine: bool = True,
        zero_output: bool = True,
        compute_dtype: object = jnp.float32,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        rngs = nnx.Rngs(0) if rngs is None else rngs
        self.num_heads = num_heads
        self.head_dim = single_dim // num_heads
        if use_rope:
            assert self.head_dim % 2 == 0
        self.use_pair_bias = use_pair_bias
        self.use_rope = use_rope
        self.single_norm = RMSNorm(
            single_dim,
            eps=norm_eps,
            affine=input_norm_affine,
        )
        self.pair_norm = (
            RMSNorm(pair_dim, eps=norm_eps) if use_pair_bias else None
        )
        self.q_norm = RMSNorm(self.head_dim, eps=norm_eps, affine=False)
        self.k_norm = RMSNorm(self.head_dim, eps=norm_eps, affine=False)
        self.qkv = Linear(
            single_dim,
            3 * single_dim,
            bias=False,
            compute_dtype=compute_dtype,
            rngs=rngs,
        )
        self.pair_bias = (
            Linear(
                pair_dim,
                num_heads,
                bias=False,
                compute_dtype=compute_dtype,
                rngs=rngs,
            )
            if use_pair_bias
            else None
        )
        self.g = Linear(
            single_dim,
            single_dim,
            compute_dtype=compute_dtype,
            init="gate",
            rngs=rngs,
        )
        self.o = Linear(
            single_dim,
            single_dim,
            compute_dtype=compute_dtype,
            init="zeros" if zero_output else "xavier_normal",
            rngs=rngs,
        )

    def __call__(
        self,
        single: Array,
        pair: Array | None,
        token_mask: Array,
        num_peak_tokens: int,
        token_positions: Array | None = None,
        normalized_single: Array | None = None,
    ) -> Array:
        batch_size, num_tokens, single_dim = single.shape
        single_norm = (
            self.single_norm(single)
            if normalized_single is None
            else normalized_single
        )
        qkv = self.qkv(single_norm).reshape(
            batch_size,
            num_tokens,
            3,
            self.num_heads,
            self.head_dim,
        )
        q, k, v = jnp.moveaxis(qkv, 2, 0)
        q = jnp.swapaxes(q, 1, 2)
        k = jnp.swapaxes(k, 1, 2)
        v = jnp.swapaxes(v, 1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.use_rope:
            if token_positions is None:
                token_positions = jnp.broadcast_to(
                    jnp.arange(num_tokens),
                    (batch_size, num_tokens),
                )
            q = _apply_rope(q, token_positions)
            k = _apply_rope(k, token_positions)
        if self.use_pair_bias:
            pair = cast(Array, pair)
            peak_bias = jnp.transpose(
                self.pair_bias(self.pair_norm(pair)),
                (0, 3, 1, 2),
            )
            extra_tokens = num_tokens - num_peak_tokens
            attn_bias = jnp.pad(
                peak_bias,
                ((0, 0), (0, 0), (0, extra_tokens), (0, extra_tokens)),
            ).astype(jnp.float32)
        else:
            attn_bias = jnp.zeros(
                (batch_size, 1, 1, num_tokens),
                dtype=jnp.float32,
            )
        attn_bias = jnp.where(
            token_mask[:, None, None, :],
            attn_bias,
            jnp.asarray(-jnp.inf, dtype=attn_bias.dtype),
        )
        out = scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        out = jnp.swapaxes(out, 1, 2).reshape(batch_size, num_tokens, single_dim)
        out = out * jax.nn.sigmoid(self.g(single_norm))
        return self.o(out)

    def load_torch_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
    ) -> None:
        self.single_norm.load_torch_state_dict(state_dict, f"{prefix}.single_norm")
        self.qkv.load_torch_state_dict(state_dict, f"{prefix}.qkv")
        if self.use_pair_bias:
            self.pair_norm.load_torch_state_dict(state_dict, f"{prefix}.pair_norm")
            self.pair_bias.load_torch_state_dict(state_dict, f"{prefix}.pair_bias")
        self.g.load_torch_state_dict(state_dict, f"{prefix}.g")
        self.o.load_torch_state_dict(state_dict, f"{prefix}.o")


class SingleMixerBlock(nnx.Module):
    def __init__(
        self,
        *,
        single_dim: int,
        num_heads: int,
        attention_mlp_multiple: float,
        norm_eps: float,
        use_rope: bool = True,
        transition_type: str = "swiglu",
        condition_dim: int | None = None,
        compute_dtype: object = jnp.float32,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        rngs = nnx.Rngs(0) if rngs is None else rngs
        self.transition_type = transition_type.lower()
        self.condition_dim = condition_dim
        if self.transition_type not in SUPPORTED_PAIRMIXER_TRANSITION_TYPES:
            raise ValueError(
                "pairmixer_transition_type must be one of ('swiglu', 'feedforward')"
            )
        self.single_attention = AttentionPairBias(
            single_dim=single_dim,
            pair_dim=single_dim,
            num_heads=num_heads,
            norm_eps=norm_eps,
            use_pair_bias=False,
            use_rope=use_rope,
            input_norm_affine=condition_dim is None,
            zero_output=condition_dim is None,
            compute_dtype=compute_dtype,
            rngs=rngs,
        )
        self.single_attention_post_norm = RMSNorm(single_dim, eps=norm_eps)
        self.single_transition_norm = RMSNorm(
            single_dim,
            eps=norm_eps,
            affine=condition_dim is None,
        )
        self.single_transition = _build_pairmixer_transition(
            single_dim,
            hidden_dim=math.ceil(single_dim * attention_mlp_multiple),
            transition_type=self.transition_type,
            compute_dtype=compute_dtype,
            rngs=rngs,
        )
        self.single_transition_post_norm = RMSNorm(single_dim, eps=norm_eps)
        self.adaLN_modulation = (
            Linear(
                condition_dim,
                6 * single_dim,
                compute_dtype=compute_dtype,
                init="zeros",
                rngs=rngs,
            )
            if condition_dim is not None
            else None
        )
        if condition_dim is not None and self.transition_type == "swiglu":
            transition = cast(SwiGLUFeedForward, self.single_transition)
            transition.fc3.weight[...] = rngs.params.normal(
                transition.fc3.weight.shape,
                dtype=jnp.float32,
            ) / math.sqrt(transition.fc3.weight.shape[1])

    def __call__(
        self,
        single: Array,
        token_mask: Array,
        token_positions: Array | None = None,
        condition: Array | None = None,
    ) -> Array:
        if self.adaLN_modulation is not None:
            (
                attn_shift,
                attn_scale,
                attn_gate,
                mlp_shift,
                mlp_scale,
                mlp_gate,
            ) = jnp.split(self.adaLN_modulation(condition), 6, axis=-1)
            attention_input = self.single_attention.single_norm(single)
            attention_input = (
                attention_input * (1 + attn_scale[:, None])
                + attn_shift[:, None]
            )
            attention_update = self.single_attention(
                single,
                None,
                token_mask,
                token_mask.shape[1],
                token_positions,
                attention_input,
            )
            single = single + attn_gate[:, None] * self.single_attention_post_norm(
                attention_update
            )
            transition_input = self.single_transition_norm(single)
            transition_input = (
                transition_input * (1 + mlp_scale[:, None]) + mlp_shift[:, None]
            )
            transition_update = _transition_with_preferred_acc(
                self.single_transition,
                transition_input,
            )
            return single + mlp_gate[:, None] * self.single_transition_post_norm(
                transition_update
            )
        attention_update = self.single_attention(
            single,
            None,
            token_mask,
            token_mask.shape[1],
            token_positions,
        )
        single = single + self.single_attention_post_norm(attention_update)
        return single + self.single_transition_post_norm(
            _transition_with_preferred_acc(
                self.single_transition,
                self.single_transition_norm(single),
            )
        )

    def load_torch_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
    ) -> None:
        self.single_attention.load_torch_state_dict(
            state_dict,
            f"{prefix}.single_attention",
        )
        self.single_attention_post_norm.load_torch_state_dict(
            state_dict,
            f"{prefix}.single_attention_post_norm",
        )
        self.single_transition_norm.load_torch_state_dict(
            state_dict,
            f"{prefix}.single_transition_norm",
        )
        self.single_transition.load_torch_state_dict(
            state_dict,
            f"{prefix}.single_transition",
        )
        self.single_transition_post_norm.load_torch_state_dict(
            state_dict,
            f"{prefix}.single_transition_post_norm",
        )
        if self.adaLN_modulation is not None:
            self.adaLN_modulation.load_torch_state_dict(
                state_dict,
                f"{prefix}.adaLN_modulation",
            )


class GatedSingleToPairUpdate(nnx.Module):
    def __init__(
        self,
        *,
        single_dim: int,
        pair_dim: int,
        norm_eps: float,
        compute_dtype: object = jnp.float32,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        rngs = nnx.Rngs(0) if rngs is None else rngs
        self.single_norm = RMSNorm(single_dim, eps=norm_eps)
        self.pair_norm = RMSNorm(pair_dim, eps=norm_eps)
        self.left = Linear(single_dim, pair_dim, compute_dtype=compute_dtype, rngs=rngs)
        self.right = Linear(single_dim, pair_dim, compute_dtype=compute_dtype, rngs=rngs)
        self.out = Linear(pair_dim, pair_dim, compute_dtype=compute_dtype, rngs=rngs)
        self.gate = Linear(
            pair_dim,
            pair_dim,
            compute_dtype=compute_dtype,
            init="gate",
            rngs=rngs,
        )

    def __call__(self, single: Array, pair: Array, pair_mask_value: Array) -> Array:
        single_norm = self.single_norm(single)
        left = self.left(single_norm)[:, :, None, :]
        right = self.right(single_norm)[:, None, :, :]
        update = self.out(left * right)
        gate = jax.nn.sigmoid(self.gate(self.pair_norm(pair)))
        update = gate * update
        return update * pair_mask_value[..., None].astype(update.dtype)

    def load_torch_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
    ) -> None:
        self.single_norm.load_torch_state_dict(state_dict, f"{prefix}.single_norm")
        self.pair_norm.load_torch_state_dict(state_dict, f"{prefix}.pair_norm")
        self.left.load_torch_state_dict(state_dict, f"{prefix}.left")
        self.right.load_torch_state_dict(state_dict, f"{prefix}.right")
        self.out.load_torch_state_dict(state_dict, f"{prefix}.out")
        self.gate.load_torch_state_dict(state_dict, f"{prefix}.gate")


class PairMixerBlock(nnx.Module):
    def __init__(
        self,
        *,
        single_dim: int,
        pair_dim: int,
        num_heads: int,
        attention_mlp_multiple: float,
        norm_eps: float,
        dropout: float,
        use_single_to_pair_update: bool = False,
        use_pair_bias: bool = True,
        use_rope: bool = True,
        use_fastmixer: bool = False,
        fastmixer_max_visible_tokens: int | None = None,
        transition_type: str = "swiglu",
        compute_dtype: object = jnp.float32,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        rngs = nnx.Rngs(0) if rngs is None else rngs
        self.dropout = dropout
        self.use_single_to_pair_update = use_single_to_pair_update
        self.use_fastmixer = use_fastmixer
        self.transition_type = transition_type.lower()
        if self.transition_type not in SUPPORTED_PAIRMIXER_TRANSITION_TYPES:
            raise ValueError(
                "pairmixer_transition_type must be one of ('swiglu', 'feedforward')"
            )
        self.fastmixer_max_visible_tokens = (
            0
            if fastmixer_max_visible_tokens is None
            else int(fastmixer_max_visible_tokens)
        )
        if self.use_fastmixer:
            assert self.fastmixer_max_visible_tokens > 0
        self.tri_mul_out = TriangleMultiplicativeUpdate(
            pair_dim,
            direction="outgoing",
            norm_eps=norm_eps,
            compute_dtype=compute_dtype,
            rngs=rngs,
        )
        self.tri_mul_out_post_norm = RMSNorm(pair_dim, eps=norm_eps)
        self.tri_mul_in = TriangleMultiplicativeUpdate(
            pair_dim,
            direction="incoming",
            norm_eps=norm_eps,
            compute_dtype=compute_dtype,
            rngs=rngs,
        )
        self.tri_mul_in_post_norm = RMSNorm(pair_dim, eps=norm_eps)
        self.pair_transition_norm = RMSNorm(pair_dim, eps=norm_eps)
        self.pair_transition = _build_pairmixer_transition(
            pair_dim,
            hidden_dim=math.ceil(pair_dim * attention_mlp_multiple),
            transition_type=self.transition_type,
            compute_dtype=compute_dtype,
            rngs=rngs,
        )
        self.pair_transition_post_norm = RMSNorm(pair_dim, eps=norm_eps)
        if self.use_single_to_pair_update:
            self.single_to_pair_update = GatedSingleToPairUpdate(
                single_dim=single_dim,
                pair_dim=pair_dim,
                norm_eps=norm_eps,
                compute_dtype=compute_dtype,
                rngs=rngs,
            )
            self.single_to_pair_post_norm = RMSNorm(pair_dim, eps=norm_eps)
        self.single_attention = AttentionPairBias(
            single_dim=single_dim,
            pair_dim=pair_dim,
            num_heads=num_heads,
            norm_eps=norm_eps,
            use_pair_bias=use_pair_bias,
            use_rope=use_rope,
            compute_dtype=compute_dtype,
            rngs=rngs,
        )
        self.single_attention_post_norm = RMSNorm(single_dim, eps=norm_eps)
        self.single_transition_norm = RMSNorm(single_dim, eps=norm_eps)
        self.single_transition = _build_pairmixer_transition(
            single_dim,
            hidden_dim=math.ceil(single_dim * attention_mlp_multiple),
            transition_type=self.transition_type,
            compute_dtype=compute_dtype,
            rngs=rngs,
        )
        self.single_transition_post_norm = RMSNorm(single_dim, eps=norm_eps)

    def __call__(
        self,
        single: Array,
        pair: Array,
        peak_mask: Array,
        token_mask: Array,
        *,
        deterministic: bool = True,
        rng: Array | None = None,
    ) -> tuple[Array, Array]:
        del rng
        pair_mask_value = pair_mask(peak_mask)
        if self.use_fastmixer:
            return self._fastmixer_call(
                single,
                pair,
                peak_mask,
                token_mask,
                pair_mask_value,
            )
        pair = pair + self.tri_mul_out_post_norm(
            self.tri_mul_out(pair, peak_mask, pair_mask_value)
        )
        pair = pair + self.tri_mul_in_post_norm(
            self.tri_mul_in(pair, peak_mask, pair_mask_value)
        )
        pair = pair + self.pair_transition_post_norm(
            self.pair_transition(self.pair_transition_norm(pair))
        )
        pair = pair * pair_mask_value[..., None].astype(pair.dtype)
        if self.use_single_to_pair_update:
            pair = pair + self.single_to_pair_post_norm(
                self.single_to_pair_update(
                    single,
                    pair,
                    pair_mask_value,
                )
            )
            pair = pair * pair_mask_value[..., None].astype(pair.dtype)
        attention_update = self.single_attention(
            single,
            pair,
            token_mask,
            peak_mask.shape[1],
        )
        single = single + self.single_attention_post_norm(attention_update)
        single = single + self.single_transition_post_norm(
            self.single_transition(self.single_transition_norm(single))
        )
        return single, pair

    def _fastmixer_call(
        self,
        single: Array,
        pair: Array,
        peak_mask: Array,
        token_mask: Array,
        pair_mask_value: Array,
    ) -> tuple[Array, Array]:
        del pair_mask_value
        idx, compact_token_mask = _active_indices(
            peak_mask,
            self.fastmixer_max_visible_tokens,
        )
        pair_compact = _gather_pair(pair, idx)
        single, pair_compact = self.fastmixer_compact_call(
            single,
            pair_compact,
            idx,
            compact_token_mask,
            token_mask,
        )
        pair = _scatter_pair(pair_compact, idx, pair.shape)
        return single, pair

    def fastmixer_compact_call(
        self,
        single: Array,
        pair_compact: Array,
        idx: Array,
        compact_token_mask: Array,
        token_mask: Array,
    ) -> tuple[Array, Array]:
        single_compact = _gather_single(single, idx)
        pair_mask_compact = compact_token_mask[:, :, None] & compact_token_mask[:, None, :]

        pair_compact = pair_compact + self.tri_mul_out_post_norm(
            self._fast_triangle_update(
                self.tri_mul_out,
                pair_compact,
                compact_token_mask,
                pair_mask_compact,
            )
        )
        pair_compact = pair_compact + self.tri_mul_in_post_norm(
            self._fast_triangle_update(
                self.tri_mul_in,
                pair_compact,
                compact_token_mask,
                pair_mask_compact,
            )
        )
        pair_compact = pair_compact + self.pair_transition_post_norm(
            _transition_with_preferred_acc(
                self.pair_transition,
                self.pair_transition_norm(pair_compact),
            )
        )
        pair_compact = pair_compact * pair_mask_compact[..., None].astype(
            pair_compact.dtype
        )
        if self.use_single_to_pair_update:
            pair_compact = pair_compact + self.single_to_pair_post_norm(
                self._fast_single_to_pair_update(
                    single_compact,
                    pair_compact,
                    pair_mask_compact,
                )
            )
            pair_compact = pair_compact * pair_mask_compact[..., None].astype(
                pair_compact.dtype
            )

        single = single + self.single_attention_post_norm(
            self._fast_attention_pair_bias(
                single,
                pair_compact,
                idx,
                token_mask,
            )
        )
        single = single + self.single_transition_post_norm(
            _transition_with_preferred_acc(
                self.single_transition,
                self.single_transition_norm(single),
            )
        )
        return single, pair_compact

    def fastmixer_compact_only_call(
        self,
        single_compact: Array,
        pair_compact: Array,
        compact_token_mask: Array,
        token_positions: Array | None = None,
    ) -> tuple[Array, Array]:
        pair_mask_compact = compact_token_mask[:, :, None] & compact_token_mask[:, None, :]

        pair_compact = pair_compact + self.tri_mul_out_post_norm(
            self._fast_triangle_update(
                self.tri_mul_out,
                pair_compact,
                compact_token_mask,
                pair_mask_compact,
            )
        )
        pair_compact = pair_compact + self.tri_mul_in_post_norm(
            self._fast_triangle_update(
                self.tri_mul_in,
                pair_compact,
                compact_token_mask,
                pair_mask_compact,
            )
        )
        pair_compact = pair_compact + self.pair_transition_post_norm(
            _transition_with_preferred_acc(
                self.pair_transition,
                self.pair_transition_norm(pair_compact),
            )
        )
        pair_compact = pair_compact * pair_mask_compact[..., None].astype(
            pair_compact.dtype
        )
        if self.use_single_to_pair_update:
            pair_compact = pair_compact + self.single_to_pair_post_norm(
                self._fast_single_to_pair_update(
                    single_compact,
                    pair_compact,
                    pair_mask_compact,
                )
            )
            pair_compact = pair_compact * pair_mask_compact[..., None].astype(
                pair_compact.dtype
            )

        single_compact = single_compact + self.single_attention_post_norm(
            self._fast_attention_pair_bias_compact(
                single_compact,
                pair_compact,
                compact_token_mask,
                token_positions,
            )
        )
        single_compact = single_compact + self.single_transition_post_norm(
            _transition_with_preferred_acc(
                self.single_transition,
                self.single_transition_norm(single_compact),
            )
        )
        single_compact = single_compact * compact_token_mask[..., None].astype(
            single_compact.dtype
        )
        return single_compact, pair_compact

    def _fast_triangle_update(
        self,
        module: TriangleMultiplicativeUpdate,
        pair: Array,
        token_mask: Array,
        mask: Array,
    ) -> Array:
        pair_mask_f = mask[..., None].astype(pair.dtype)
        x_norm = module.norm_in(pair)
        projected = _linear_with_preferred_acc(
            module.p_in,
            x_norm,
        ) * jax.nn.sigmoid(_linear_with_preferred_acc(module.g_in, x_norm))
        output_gate = jax.nn.sigmoid(
            _linear_with_preferred_acc(module.g_out, x_norm)
        )
        a, b = jnp.split(projected, 2, axis=-1)
        if module.direction == "outgoing":
            a = a * token_mask[:, None, :, None].astype(a.dtype)
            update = jnp.sum(
                a[:, :, None, :, :] * b[:, None, :, :, :],
                axis=3,
            )
        else:
            a = a * token_mask[:, :, None, None].astype(a.dtype)
            update = jnp.sum(
                a[:, :, :, None, :] * b[:, :, None, :, :],
                axis=1,
            )
        update = _linear_with_preferred_acc(module.p_out, module.norm_out(update))
        update = update * output_gate
        return update * pair_mask_f

    def _fast_single_to_pair_update(
        self,
        single: Array,
        pair: Array,
        pair_mask_value: Array,
    ) -> Array:
        module = self.single_to_pair_update
        single_norm = module.single_norm(single)
        left = _linear_with_preferred_acc(module.left, single_norm)[:, :, None, :]
        right = _linear_with_preferred_acc(module.right, single_norm)[:, None, :, :]
        update = _linear_with_preferred_acc(module.out, left * right)
        gate = jax.nn.sigmoid(
            _linear_with_preferred_acc(module.gate, module.pair_norm(pair))
        )
        update = gate * update
        return update * pair_mask_value[..., None].astype(update.dtype)

    def _fast_attention_pair_bias(
        self,
        single: Array,
        pair_compact: Array,
        idx: Array,
        token_mask: Array,
    ) -> Array:
        module = self.single_attention
        batch_size, num_tokens, single_dim = single.shape
        single_norm = module.single_norm(single)
        qkv = _linear_with_preferred_acc(module.qkv, single_norm).reshape(
            batch_size,
            num_tokens,
            3,
            module.num_heads,
            module.head_dim,
        )
        q, k, v = jnp.moveaxis(qkv, 2, 0)
        q = jnp.swapaxes(q, 1, 2)
        k = jnp.swapaxes(k, 1, 2)
        v = jnp.swapaxes(v, 1, 2)
        q = module.q_norm(q)
        k = module.k_norm(k)
        if module.use_rope:
            token_positions = jnp.broadcast_to(
                jnp.arange(num_tokens),
                (batch_size, num_tokens),
            )
            q = _apply_rope(q, token_positions)
            k = _apply_rope(k, token_positions)

        if module.use_pair_bias:
            pair_bias_compact = _linear_with_preferred_acc(
                module.pair_bias,
                module.pair_norm(pair_compact),
            )
            pair_bias = _scatter_compact_pair_bias(
                pair_bias_compact,
                idx,
                num_tokens,
            ).astype(jnp.float32)
            attn_bias = jnp.transpose(pair_bias, (0, 3, 1, 2))
        else:
            attn_bias = jnp.zeros(
                (batch_size, 1, 1, num_tokens),
                dtype=jnp.float32,
            )
        attn_bias = jnp.where(
            token_mask[:, None, None, :],
            attn_bias,
            jnp.asarray(-jnp.inf, dtype=attn_bias.dtype),
        )

        scores = jnp.einsum(
            "...qd,...kd->...qk",
            q,
            k,
            precision=_dot_precision(q.dtype),
            preferred_element_type=_preferred_acc_dtype(q.dtype),
        ) / math.sqrt(module.head_dim)
        scores = scores + attn_bias
        attn = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(v.dtype)
        out = jnp.einsum(
            "...qk,...kd->...qd",
            attn,
            v,
            precision=_dot_precision(v.dtype),
            preferred_element_type=_preferred_acc_dtype(v.dtype),
        ).astype(v.dtype)
        out = jnp.swapaxes(out, 1, 2).reshape(batch_size, num_tokens, single_dim)
        out = out * jax.nn.sigmoid(_linear_with_preferred_acc(module.g, single_norm))
        return _linear_with_preferred_acc(module.o, out)

    def _fast_attention_pair_bias_compact(
        self,
        single_compact: Array,
        pair_compact: Array,
        compact_token_mask: Array,
        token_positions: Array | None = None,
    ) -> Array:
        module = self.single_attention
        batch_size, num_tokens, single_dim = single_compact.shape
        single_norm = module.single_norm(single_compact)
        qkv = _linear_with_preferred_acc(module.qkv, single_norm).reshape(
            batch_size,
            num_tokens,
            3,
            module.num_heads,
            module.head_dim,
        )
        q, k, v = jnp.moveaxis(qkv, 2, 0)
        q = jnp.swapaxes(q, 1, 2)
        k = jnp.swapaxes(k, 1, 2)
        v = jnp.swapaxes(v, 1, 2)
        q = module.q_norm(q)
        k = module.k_norm(k)
        if module.use_rope:
            if token_positions is None:
                token_positions = jnp.broadcast_to(
                    jnp.arange(num_tokens),
                    (batch_size, num_tokens),
                )
            q = _apply_rope(q, token_positions)
            k = _apply_rope(k, token_positions)

        if module.use_pair_bias:
            pair_bias = _linear_with_preferred_acc(
                module.pair_bias,
                module.pair_norm(pair_compact),
            )
            attn_bias = jnp.transpose(pair_bias, (0, 3, 1, 2))
        else:
            attn_bias = jnp.zeros(
                (batch_size, 1, 1, num_tokens),
                dtype=jnp.float32,
            )
        attn_bias = jnp.where(
            compact_token_mask[:, None, None, :],
            attn_bias.astype(jnp.float32),
            jnp.asarray(-jnp.inf, dtype=jnp.float32),
        )

        scores = (
            jnp.einsum(
                "...qd,...kd->...qk",
                q,
                k,
                precision=_dot_precision(q.dtype),
                preferred_element_type=_preferred_acc_dtype(q.dtype),
            ).astype(jnp.float32)
            / math.sqrt(module.head_dim)
        )
        scores = scores + attn_bias
        attn = jax.nn.softmax(scores, axis=-1).astype(v.dtype)
        out = jnp.einsum(
            "...qk,...kd->...qd",
            attn,
            v,
            precision=_dot_precision(v.dtype),
            preferred_element_type=_preferred_acc_dtype(v.dtype),
        ).astype(v.dtype)
        out = jnp.swapaxes(out, 1, 2).reshape(batch_size, num_tokens, single_dim)
        out = out * jax.nn.sigmoid(_linear_with_preferred_acc(module.g, single_norm))
        out = _linear_with_preferred_acc(module.o, out)
        return out * compact_token_mask[..., None].astype(out.dtype)

    def load_torch_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
    ) -> None:
        self.tri_mul_out.load_torch_state_dict(state_dict, f"{prefix}.tri_mul_out")
        self.tri_mul_out_post_norm.load_torch_state_dict(
            state_dict,
            f"{prefix}.tri_mul_out_post_norm",
        )
        self.tri_mul_in.load_torch_state_dict(state_dict, f"{prefix}.tri_mul_in")
        self.tri_mul_in_post_norm.load_torch_state_dict(
            state_dict,
            f"{prefix}.tri_mul_in_post_norm",
        )
        self.pair_transition_norm.load_torch_state_dict(
            state_dict,
            f"{prefix}.pair_transition_norm",
        )
        self.pair_transition.load_torch_state_dict(
            state_dict,
            f"{prefix}.pair_transition",
        )
        self.pair_transition_post_norm.load_torch_state_dict(
            state_dict,
            f"{prefix}.pair_transition_post_norm",
        )
        if self.use_single_to_pair_update:
            self.single_to_pair_update.load_torch_state_dict(
                state_dict,
                f"{prefix}.single_to_pair_update",
            )
            self.single_to_pair_post_norm.load_torch_state_dict(
                state_dict,
                f"{prefix}.single_to_pair_post_norm",
            )
        self.single_attention.load_torch_state_dict(
            state_dict,
            f"{prefix}.single_attention",
        )
        self.single_attention_post_norm.load_torch_state_dict(
            state_dict,
            f"{prefix}.single_attention_post_norm",
        )
        self.single_transition_norm.load_torch_state_dict(
            state_dict,
            f"{prefix}.single_transition_norm",
        )
        self.single_transition.load_torch_state_dict(
            state_dict,
            f"{prefix}.single_transition",
        )
        self.single_transition_post_norm.load_torch_state_dict(
            state_dict,
            f"{prefix}.single_transition_post_norm",
        )
