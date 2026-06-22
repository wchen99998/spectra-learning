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
    assign_param,
    pair_mask,
    scaled_dot_product_attention,
    silu,
)
from spectra_learning.models.peak_features_jax import FourierFeatures
from spectra_learning.models.transformer_jax import FeedForward


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
    ) -> None:
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
                Linear(raw_dim, hidden_dim, compute_dtype=compute_dtype),
                Linear(hidden_dim, pair_dim, compute_dtype=compute_dtype),
            ]
        )
        self.single_pair_proj = nnx.List(
            [
                Linear(4 * single_dim, hidden_dim, compute_dtype=compute_dtype),
                Linear(hidden_dim, pair_dim, compute_dtype=compute_dtype),
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
    ) -> None:
        self.direction = direction
        self.compute_dtype = compute_dtype
        self.norm_in = LayerNorm(pair_dim, eps=norm_eps)
        self.p_in = Linear(pair_dim, 2 * pair_dim, compute_dtype=compute_dtype)
        self.g_in = Linear(pair_dim, 2 * pair_dim, compute_dtype=compute_dtype)
        self.norm_out = LayerNorm(pair_dim, eps=norm_eps)
        self.p_out = Linear(pair_dim, pair_dim, compute_dtype=compute_dtype)
        self.g_out = Linear(pair_dim, pair_dim, compute_dtype=compute_dtype)

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


class LearnedTriangleMediatorAssignment(nnx.Module):
    def __init__(
        self,
        single_dim: int,
        num_mediators: int,
        *,
        norm_eps: float,
        compute_dtype: object = jnp.float32,
    ) -> None:
        self.single_dim = single_dim
        self.num_mediators = num_mediators
        self.compute_dtype = compute_dtype
        self.matmul_precision = (
            jax.lax.Precision.DEFAULT if compute_dtype == jnp.bfloat16 else None
        )
        self.mediator_token = nnx.Param(
            jnp.zeros((num_mediators, single_dim), dtype=jnp.float32)
        )
        self.norm = LayerNorm(single_dim, eps=norm_eps)
        self.wq = Linear(single_dim, single_dim, bias=False, compute_dtype=compute_dtype)
        self.wk = Linear(single_dim, single_dim, bias=False, compute_dtype=compute_dtype)

    def __call__(self, single: Array) -> Array:
        batch_size = single.shape[0]
        q = self.wq(self.norm(single))
        mediator = jnp.broadcast_to(
            self.mediator_token[...].astype(single.dtype),
            (batch_size, self.num_mediators, self.single_dim),
        )
        k = self.wk(mediator)
        scores = jnp.einsum(
            "bid,bmd->bim",
            q,
            k,
            precision=self.matmul_precision,
        ) / math.sqrt(self.single_dim)
        return jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(q.dtype)

    def load_torch_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
    ) -> None:
        assign_param(self.mediator_token, state_dict[f"{prefix}.mediator_token"])
        self.norm.load_torch_state_dict(state_dict, f"{prefix}.norm")
        self.wq.load_torch_state_dict(state_dict, f"{prefix}.wq")
        self.wk.load_torch_state_dict(state_dict, f"{prefix}.wk")


class MediatedTriangleMultiplicativeUpdate(nnx.Module):
    def __init__(
        self,
        pair_dim: int,
        *,
        direction: str,
        norm_eps: float,
        mediator_eps: float,
        compute_dtype: object = jnp.float32,
    ) -> None:
        self.direction = direction
        self.mediator_eps = mediator_eps
        self.compute_dtype = compute_dtype
        self.matmul_precision = (
            jax.lax.Precision.DEFAULT if compute_dtype == jnp.bfloat16 else None
        )
        self.norm_in = LayerNorm(pair_dim, eps=norm_eps)
        self.p_in = Linear(pair_dim, 2 * pair_dim, compute_dtype=compute_dtype)
        self.g_in = Linear(pair_dim, 2 * pair_dim, compute_dtype=compute_dtype)
        self.norm_out = LayerNorm(pair_dim, eps=norm_eps)
        self.p_out = Linear(pair_dim, pair_dim, compute_dtype=compute_dtype)
        self.g_out = Linear(pair_dim, pair_dim, compute_dtype=compute_dtype)

    def _metric(self, mediator_assignment: Array, token_mask: Array) -> tuple[Array, Array]:
        mediator = mediator_assignment * token_mask[..., None].astype(
            mediator_assignment.dtype
        )
        mediator_float = mediator.astype(jnp.float32)
        gram = jnp.einsum(
            "bkm,bkn->bmn",
            mediator_float,
            mediator_float,
            precision=self.matmul_precision,
        )
        eye = jnp.eye(gram.shape[-1], dtype=jnp.float32)[None, :, :]
        metric = jnp.linalg.inv(gram + self.mediator_eps * eye)
        return mediator, metric.astype(mediator.dtype)

    def __call__(
        self,
        x: Array,
        token_mask: Array,
        mask: Array,
        mediator_assignment: Array,
    ) -> Array:
        pair_mask_f = mask[..., None].astype(x.dtype)
        mediator, metric = self._metric(mediator_assignment, token_mask)
        x_norm = self.norm_in(x)
        projected = self.p_in(x_norm) * jax.nn.sigmoid(self.g_in(x_norm))
        a, b = jnp.split(projected, 2, axis=-1)
        if self.direction == "outgoing":
            a_landmark = jnp.einsum(
                "bikc,bkm->bimc",
                a,
                mediator,
                precision=self.matmul_precision,
            )
            b_landmark = jnp.einsum(
                "bjkc,bkm->bjmc",
                b,
                mediator,
                precision=self.matmul_precision,
            )
        else:
            a_landmark = jnp.einsum(
                "bkic,bkm->bimc",
                a,
                mediator,
                precision=self.matmul_precision,
            )
            b_landmark = jnp.einsum(
                "bkjc,bkm->bjmc",
                b,
                mediator,
                precision=self.matmul_precision,
            )
        a_landmark = jnp.einsum(
            "bimc,bmn->binc",
            a_landmark,
            metric,
            precision=self.matmul_precision,
        )
        update = jnp.einsum(
            "binc,bjnc->bijc",
            a_landmark,
            b_landmark,
            precision=self.matmul_precision,
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
    ) -> None:
        self.num_heads = num_heads
        self.head_dim = single_dim // num_heads
        self.single_norm = LayerNorm(single_dim, eps=norm_eps)
        self.pair_norm = LayerNorm(pair_dim, eps=norm_eps)
        self.qkv = Linear(
            single_dim,
            3 * single_dim,
            bias=False,
            compute_dtype=compute_dtype,
        )
        self.pair_bias = Linear(
            pair_dim,
            num_heads,
            bias=False,
            compute_dtype=compute_dtype,
        )
        self.g = Linear(single_dim, single_dim, compute_dtype=compute_dtype)
        self.o = Linear(single_dim, single_dim, compute_dtype=compute_dtype)

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
    ) -> None:
        self.single_norm = LayerNorm(single_dim, eps=norm_eps)
        self.pair_norm = LayerNorm(pair_dim, eps=norm_eps)
        self.left = Linear(single_dim, pair_dim, compute_dtype=compute_dtype)
        self.right = Linear(single_dim, pair_dim, compute_dtype=compute_dtype)
        self.out = Linear(pair_dim, pair_dim, compute_dtype=compute_dtype)
        self.gate = Linear(pair_dim, pair_dim, compute_dtype=compute_dtype)

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
        triangle_mediator_num_mediators: int | None = None,
        triangle_mediator_eps: float = 1e-4,
        use_single_to_pair_update: bool = False,
        compute_dtype: object = jnp.float32,
    ) -> None:
        self.dropout = dropout
        self.use_triangle_mediator = triangle_mediator_num_mediators is not None
        self.use_single_to_pair_update = use_single_to_pair_update
        if self.use_triangle_mediator:
            self.triangle_mediator_assignment = LearnedTriangleMediatorAssignment(
                single_dim,
                triangle_mediator_num_mediators,
                norm_eps=norm_eps,
                compute_dtype=compute_dtype,
            )
            self.tri_mul_out = MediatedTriangleMultiplicativeUpdate(
                pair_dim,
                direction="outgoing",
                norm_eps=norm_eps,
                mediator_eps=triangle_mediator_eps,
                compute_dtype=compute_dtype,
            )
            self.tri_mul_in = MediatedTriangleMultiplicativeUpdate(
                pair_dim,
                direction="incoming",
                norm_eps=norm_eps,
                mediator_eps=triangle_mediator_eps,
                compute_dtype=compute_dtype,
            )
        else:
            self.tri_mul_out = TriangleMultiplicativeUpdate(
                pair_dim,
                direction="outgoing",
                norm_eps=norm_eps,
                compute_dtype=compute_dtype,
            )
            self.tri_mul_in = TriangleMultiplicativeUpdate(
                pair_dim,
                direction="incoming",
                norm_eps=norm_eps,
                compute_dtype=compute_dtype,
            )
        self.pair_transition_norm = LayerNorm(pair_dim, eps=norm_eps)
        self.pair_transition = FeedForward(
            pair_dim,
            hidden_dim=math.ceil(pair_dim * attention_mlp_multiple),
            compute_dtype=compute_dtype,
        )
        if self.use_single_to_pair_update:
            self.single_to_pair_update = GatedSingleToPairUpdate(
                single_dim=single_dim,
                pair_dim=pair_dim,
                norm_eps=norm_eps,
                compute_dtype=compute_dtype,
            )
        self.single_attention = AttentionPairBias(
            single_dim=single_dim,
            pair_dim=pair_dim,
            num_heads=num_heads,
            norm_eps=norm_eps,
            compute_dtype=compute_dtype,
        )
        self.single_transition_norm = LayerNorm(single_dim, eps=norm_eps)
        self.single_transition = FeedForward(
            single_dim,
            hidden_dim=math.ceil(single_dim * attention_mlp_multiple),
            compute_dtype=compute_dtype,
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
        if self.use_triangle_mediator:
            mediator_assignment = self.triangle_mediator_assignment(single)
            pair = pair + self.tri_mul_out(
                pair,
                peak_mask,
                pair_mask_value,
                mediator_assignment,
            )
            pair = pair + self.tri_mul_in(
                pair,
                peak_mask,
                pair_mask_value,
                mediator_assignment,
            )
        else:
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

    def load_torch_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
    ) -> None:
        if self.use_triangle_mediator:
            self.triangle_mediator_assignment.load_torch_state_dict(
                state_dict,
                f"{prefix}.triangle_mediator_assignment",
            )
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
