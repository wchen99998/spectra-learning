from __future__ import annotations

from math import log10

import jax.numpy as jnp
import torch
from flax import nnx

from spectra_learning.data.spectra import PEAK_MZ_MAX
from spectra_learning.models.common_jax import Array, Linear, MLP, load_param, silu


class FourierFeatures(nnx.Module):
    def __init__(
        self,
        x_min: float = 3e-3,
        x_max: float = 1000.0,
        *,
        num_freqs: int = 256,
    ) -> None:
        assert x_min > 0.0
        assert x_max > x_min
        self.num_freqs = num_freqs
        wavelengths = jnp.logspace(
            log10(x_min),
            log10(x_max),
            num_freqs,
            dtype=jnp.float32,
        )
        self.b = nnx.Param((1.0 / wavelengths)[None, :])

    def __call__(self, x: Array) -> Array:
        angles = 2.0 * jnp.pi * x @ self.b[...]
        return jnp.concatenate([jnp.cos(angles), jnp.sin(angles)], axis=-1)

    def num_features(self) -> int:
        return 2 * self.b[...].shape[1]

    def load_torch_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
    ) -> None:
        load_param(self.b, state_dict, f"{prefix}.b")


class PeakFeatureEmbedder(nnx.Module):
    def __init__(
        self,
        *,
        model_dim: int,
        hidden_dim: int,
        fourier_mlp_hidden_dim: int | None = None,
        fourier_mlp_num_layers: int = 2,
        fourier_x_min: float = 3e-3,
        fourier_x_max: float = 1000.0,
        fourier_num_freqs: int = 256,
        fourier_input_scale: float = PEAK_MZ_MAX,
        use_fourier_features: bool = True,
        compute_dtype: object = jnp.float32,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        rngs = nnx.Rngs(0) if rngs is None else rngs
        self.use_fourier_features = use_fourier_features
        self.fourier_input_scale = fourier_input_scale
        fourier_hidden_dim = (
            hidden_dim if fourier_mlp_hidden_dim is None else fourier_mlp_hidden_dim
        )
        if self.use_fourier_features:
            fourier_dim = model_dim // 2
            raw_dim = model_dim - fourier_dim
            self.mz_fourier = FourierFeatures(
                x_min=fourier_x_min,
                x_max=fourier_x_max,
                num_freqs=fourier_num_freqs,
            )
            self.fourier_ffn = MLP(
                self.mz_fourier.num_features(),
                fourier_hidden_dim,
                fourier_dim,
                fourier_mlp_num_layers,
                compute_dtype=compute_dtype,
                rngs=rngs,
            )
            self.raw_ffn = nnx.List(
                [
                    Linear(3, hidden_dim, compute_dtype=compute_dtype, rngs=rngs),
                    Linear(hidden_dim, raw_dim, compute_dtype=compute_dtype, rngs=rngs),
                ]
            )
        else:
            self.raw_ffn = MLP(
                3,
                fourier_hidden_dim,
                model_dim,
                fourier_mlp_num_layers,
                compute_dtype=compute_dtype,
                rngs=rngs,
            )
        self.output_proj = Linear(
            model_dim,
            model_dim,
            compute_dtype=compute_dtype,
            rngs=rngs,
        )

    def _prepare_fourier_mz(self, peak_mz: Array) -> Array:
        return peak_mz[..., None] * self.fourier_input_scale

    def _raw_ffn_call(self, raw: Array) -> Array:
        if isinstance(self.raw_ffn, MLP):
            return self.raw_ffn(raw)
        return self.raw_ffn[1](silu(self.raw_ffn[0](raw)))

    def __call__(self, peak_mz: Array, peak_intensity: Array) -> Array:
        peak_mz = peak_mz.astype(jnp.float32)
        peak_intensity = peak_intensity.astype(jnp.float32)
        mz = peak_mz[..., None]
        intensity = peak_intensity[..., None]
        log_intensity = jnp.log1p(peak_intensity)[..., None]
        raw = self._raw_ffn_call(jnp.concatenate([mz, intensity, log_intensity], axis=-1))
        if not self.use_fourier_features:
            return self.output_proj(raw)
        fourier = self.fourier_ffn(self.mz_fourier(self._prepare_fourier_mz(peak_mz)))
        return self.output_proj(jnp.concatenate([fourier, raw], axis=-1))

    def load_torch_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
    ) -> None:
        if self.use_fourier_features:
            self.mz_fourier.load_torch_state_dict(state_dict, f"{prefix}.mz_fourier")
            self.fourier_ffn.load_torch_state_dict(state_dict, f"{prefix}.fourier_ffn")
            self.raw_ffn[0].load_torch_state_dict(state_dict, f"{prefix}.raw_ffn.0")
            self.raw_ffn[1].load_torch_state_dict(state_dict, f"{prefix}.raw_ffn.2")
        else:
            self.raw_ffn.load_torch_state_dict(state_dict, f"{prefix}.raw_ffn")
        self.output_proj.load_torch_state_dict(state_dict, f"{prefix}.output_proj")
