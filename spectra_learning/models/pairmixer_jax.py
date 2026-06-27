from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import torch
from flax import nnx

from spectra_learning.data.spectra import PEAK_MZ_MAX
from spectra_learning.models.common_jax import (
    Array,
    LayerNorm,
    Linear,
    pair_mask,
    scaled_dot_product_attention,
    silu,
)
from spectra_learning.models.peak_features_jax import FourierFeatures
from spectra_learning.models.transformer_jax import SwiGLUFeedForward


COMMON_MASS_DIFFERENCES_DA = (
    1.003355,
    17.026549,
    18.010565,
    28.031300,
    44.026215,
    57.021464,
    71.037114,
    97.052764,
    99.068414,
    113.084064,
    129.042593,
    147.068414,
)


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


def _active_indices(token_mask: Array, max_active_tokens: int) -> tuple[Array, Array]:
    batch_size, num_tokens = token_mask.shape
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
    idx = idx_full[:, :max_active_tokens]
    compact_position = jnp.arange(max_active_tokens, dtype=jnp.int32)
    compact_mask = compact_position[None, :] < active_count[:, None]
    return idx, compact_mask


def _selector_from_idx(idx: Array, num_tokens: int, dtype: object) -> Array:
    return jax.nn.one_hot(idx, num_tokens, dtype=dtype)


def _gather_single(single: Array, idx: Array) -> Array:
    return jnp.take_along_axis(single, idx[..., None], axis=1)


def _gather_pair(pair: Array, idx: Array) -> Array:
    num_tokens = pair.shape[1]
    selector = _selector_from_idx(idx, num_tokens, pair.dtype)
    precision = _dot_precision(pair.dtype)
    acc_dtype = _preferred_acc_dtype(pair.dtype)
    tmp = jnp.einsum(
        "bki,bijc->bkjc",
        selector,
        pair,
        precision=precision,
        preferred_element_type=acc_dtype,
    ).astype(pair.dtype)
    out = jnp.einsum(
        "bkjc,blj->bklc",
        tmp,
        selector,
        precision=precision,
        preferred_element_type=acc_dtype,
    )
    return out.astype(pair.dtype)


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
        sigma_ppm: float = 20.0,
        use_fourier_features: bool = True,
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
        self.sigma_ppm = sigma_ppm
        self.use_fourier_features = use_fourier_features
        self.mass_differences = jnp.asarray(COMMON_MASS_DIFFERENCES_DA, dtype=jnp.float32)
        raw_dim = 14 + len(COMMON_MASS_DIFFERENCES_DA)
        if self.use_fourier_features:
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
        mz_da = peak_mz.astype(jnp.float32) * self.mz_scale
        intensity = peak_intensity.astype(jnp.float32)
        reference_mass = self._reference_mass_da(peak_mz, valid_mask, precursor_mz)[
            :, None, None
        ]
        mz_i = mz_da[:, :, None]
        mz_j = mz_da[:, None, :]
        d = mz_j - mz_i
        abs_d = jnp.abs(d)
        relative_d = d / reference_mass
        complement = mz_i + mz_j - reference_mass
        ppm = 1e6 * (abs_d[..., None] - self.mass_differences) / self.mass_differences
        radial = jnp.exp(-0.5 * jnp.square(ppm / self.sigma_ppm))

        intensity_i = intensity[:, :, None]
        intensity_j = intensity[:, None, :]
        diag = jnp.eye(peak_mz.shape[1], dtype=peak_mz.dtype)[None, :, :]
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
            radial,
        ]
        if self.use_fourier_features:
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
        if self.use_fourier_features:
            self.pair_fourier.load_torch_state_dict(state_dict, f"{prefix}.pair_fourier")
            self.relative_pair_fourier.load_torch_state_dict(
                state_dict,
                f"{prefix}.relative_pair_fourier",
            )
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
        self.norm_in = LayerNorm(pair_dim, eps=norm_eps)
        self.p_in = Linear(pair_dim, 2 * pair_dim, compute_dtype=compute_dtype, rngs=rngs)
        self.g_in = Linear(
            pair_dim,
            2 * pair_dim,
            compute_dtype=compute_dtype,
            init="gate",
            rngs=rngs,
        )
        self.norm_out = LayerNorm(pair_dim, eps=norm_eps)
        self.p_out = Linear(pair_dim, pair_dim, compute_dtype=compute_dtype, rngs=rngs)
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
        compute_dtype: object = jnp.float32,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        rngs = nnx.Rngs(0) if rngs is None else rngs
        self.num_heads = num_heads
        self.head_dim = single_dim // num_heads
        self.single_norm = LayerNorm(single_dim, eps=norm_eps)
        self.pair_norm = LayerNorm(pair_dim, eps=norm_eps)
        self.qkv = Linear(
            single_dim,
            3 * single_dim,
            bias=False,
            compute_dtype=compute_dtype,
            rngs=rngs,
        )
        self.pair_bias = Linear(
            pair_dim,
            num_heads,
            bias=False,
            compute_dtype=compute_dtype,
            rngs=rngs,
        )
        self.g = Linear(
            single_dim,
            single_dim,
            compute_dtype=compute_dtype,
            init="gate",
            rngs=rngs,
        )
        self.o = Linear(single_dim, single_dim, compute_dtype=compute_dtype, rngs=rngs)

    def __call__(
        self,
        single: Array,
        pair: Array,
        token_mask: Array,
        num_peak_tokens: int,
    ) -> Array:
        batch_size, num_tokens, single_dim = single.shape
        single_norm = self.single_norm(single)
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
        peak_bias = jnp.transpose(self.pair_bias(self.pair_norm(pair)), (0, 3, 1, 2))
        extra_tokens = num_tokens - num_peak_tokens
        attn_bias = jnp.pad(
            peak_bias,
            ((0, 0), (0, 0), (0, extra_tokens), (0, extra_tokens)),
        ).astype(jnp.float32)
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
        self.pair_norm.load_torch_state_dict(state_dict, f"{prefix}.pair_norm")
        self.qkv.load_torch_state_dict(state_dict, f"{prefix}.qkv")
        self.pair_bias.load_torch_state_dict(state_dict, f"{prefix}.pair_bias")
        self.g.load_torch_state_dict(state_dict, f"{prefix}.g")
        self.o.load_torch_state_dict(state_dict, f"{prefix}.o")


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
        self.single_norm = LayerNorm(single_dim, eps=norm_eps)
        self.pair_norm = LayerNorm(pair_dim, eps=norm_eps)
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
        use_fastmixer: bool = False,
        fastmixer_max_visible_tokens: int | None = None,
        compute_dtype: object = jnp.float32,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        rngs = nnx.Rngs(0) if rngs is None else rngs
        self.dropout = dropout
        self.use_single_to_pair_update = use_single_to_pair_update
        self.use_fastmixer = use_fastmixer
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
        self.tri_mul_in = TriangleMultiplicativeUpdate(
            pair_dim,
            direction="incoming",
            norm_eps=norm_eps,
            compute_dtype=compute_dtype,
            rngs=rngs,
        )
        self.pair_transition_norm = LayerNorm(pair_dim, eps=norm_eps)
        self.pair_transition = SwiGLUFeedForward(
            pair_dim,
            hidden_dim=math.ceil(pair_dim * attention_mlp_multiple),
            compute_dtype=compute_dtype,
            rngs=rngs,
        )
        if self.use_single_to_pair_update:
            self.single_to_pair_update = GatedSingleToPairUpdate(
                single_dim=single_dim,
                pair_dim=pair_dim,
                norm_eps=norm_eps,
                compute_dtype=compute_dtype,
                rngs=rngs,
            )
        self.single_attention = AttentionPairBias(
            single_dim=single_dim,
            pair_dim=pair_dim,
            num_heads=num_heads,
            norm_eps=norm_eps,
            compute_dtype=compute_dtype,
            rngs=rngs,
        )
        self.single_transition_norm = LayerNorm(single_dim, eps=norm_eps)
        self.single_transition = SwiGLUFeedForward(
            single_dim,
            hidden_dim=math.ceil(single_dim * attention_mlp_multiple),
            compute_dtype=compute_dtype,
            rngs=rngs,
        )

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
        pair = pair + self.tri_mul_out(pair, peak_mask, pair_mask_value)
        pair = pair + self.tri_mul_in(pair, peak_mask, pair_mask_value)
        pair = pair + self.pair_transition(self.pair_transition_norm(pair))
        pair = pair * pair_mask_value[..., None].astype(pair.dtype)
        if self.use_single_to_pair_update:
            pair = pair + self.single_to_pair_update(
                single,
                pair,
                pair_mask_value,
            )
            pair = pair * pair_mask_value[..., None].astype(pair.dtype)
        attention_update = self.single_attention(
            single,
            pair,
            token_mask,
            peak_mask.shape[1],
        )
        single = single + attention_update
        single = single + self.single_transition(self.single_transition_norm(single))
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
        single_compact = _gather_single(single, idx)
        pair_compact = _gather_pair(pair, idx)
        pair_mask_compact = compact_token_mask[:, :, None] & compact_token_mask[:, None, :]

        pair_compact = pair_compact + self._fast_triangle_update(
            self.tri_mul_out,
            pair_compact,
            compact_token_mask,
            pair_mask_compact,
        )
        pair_compact = pair_compact + self._fast_triangle_update(
            self.tri_mul_in,
            pair_compact,
            compact_token_mask,
            pair_mask_compact,
        )
        pair_compact = pair_compact + _swiglu_with_preferred_acc(
            self.pair_transition,
            self.pair_transition_norm(pair_compact),
        )
        pair_compact = pair_compact * pair_mask_compact[..., None].astype(
            pair_compact.dtype
        )
        if self.use_single_to_pair_update:
            pair_compact = pair_compact + self._fast_single_to_pair_update(
                single_compact,
                pair_compact,
                pair_mask_compact,
            )
            pair_compact = pair_compact * pair_mask_compact[..., None].astype(
                pair_compact.dtype
            )

        single = single + self._fast_attention_pair_bias(
            single,
            pair_compact,
            idx,
            token_mask,
        )
        single = single + _swiglu_with_preferred_acc(
            self.single_transition,
            self.single_transition_norm(single),
        )
        pair = _scatter_pair(pair_compact, idx, pair.shape)
        return single, pair

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
        update = update * jax.nn.sigmoid(_linear_with_preferred_acc(module.g_out, x_norm))
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

        pair_norm_compact = module.pair_norm(pair_compact)
        pair_bias_compact = _linear_with_preferred_acc(
            module.pair_bias,
            pair_norm_compact,
        )
        zero_norm = module.pair_norm.bias[...].astype(pair_compact.dtype)
        zero_bias = _linear_with_preferred_acc(
            module.pair_bias,
            zero_norm[None, :],
        )[0]
        delta_compact = (
            pair_bias_compact.astype(jnp.float32)
            - zero_bias.astype(jnp.float32)[None, None, None, :]
        ).astype(pair_bias_compact.dtype)
        delta_dense = _scatter_compact_pair_bias(delta_compact, idx, num_tokens)
        pair_bias = (
            delta_dense.astype(jnp.float32)
            + zero_bias.astype(jnp.float32)[None, None, None, :]
        )
        attn_bias = jnp.transpose(pair_bias, (0, 3, 1, 2))
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

    def load_torch_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
    ) -> None:
        self.tri_mul_out.load_torch_state_dict(state_dict, f"{prefix}.tri_mul_out")
        self.tri_mul_in.load_torch_state_dict(state_dict, f"{prefix}.tri_mul_in")
        self.pair_transition_norm.load_torch_state_dict(
            state_dict,
            f"{prefix}.pair_transition_norm",
        )
        self.pair_transition.load_torch_state_dict(
            state_dict,
            f"{prefix}.pair_transition",
        )
        if self.use_single_to_pair_update:
            self.single_to_pair_update.load_torch_state_dict(
                state_dict,
                f"{prefix}.single_to_pair_update",
            )
        self.single_attention.load_torch_state_dict(
            state_dict,
            f"{prefix}.single_attention",
        )
        self.single_transition_norm.load_torch_state_dict(
            state_dict,
            f"{prefix}.single_transition_norm",
        )
        self.single_transition.load_torch_state_dict(
            state_dict,
            f"{prefix}.single_transition",
        )
