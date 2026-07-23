from __future__ import annotations

from math import ceil, isclose, log10

import torch
from jaxtyping import Float
from torch import Tensor, nn

from spectra_learning.data.spectra import PEAK_MZ_MAX


MZ_EMBEDDING_MODES = {"fourier", "discrete"}


class FourierFeatures(nn.Module):
    def __init__(
        self,
        x_min: float = 3e-3,
        x_max: float = 1000.0,
        *,
        num_freqs: int = 256,
    ) -> None:
        super().__init__()
        assert x_min > 0.0
        assert x_max > x_min

        self.num_freqs = num_freqs

        wavelengths = torch.logspace(
            start=log10(x_min),
            end=log10(x_max),
            steps=num_freqs,
            dtype=torch.float32,
        )
        self.register_buffer("b", (1.0 / wavelengths).unsqueeze(0))

    def forward(self, x: Float[Tensor, "*batch 1"]) -> Float[Tensor, "*batch fourier"]:
        # x: [..., 1] -> [..., 2 * num_freqs]
        angles = 2 * torch.pi * x @ self.b
        return torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)

    def num_features(self) -> int:
        return 2 * self.b.shape[1]


class DiscretizedMzFeatures(nn.Module):
    def __init__(
        self,
        *,
        mz_scale: float,
        bin_size: float,
        coarse_bin_size: float,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        assert mz_scale > 0.0
        assert bin_size > 0.0
        assert coarse_bin_size > bin_size
        assert isclose(
            coarse_bin_size / bin_size,
            round(coarse_bin_size / bin_size),
        )

        self.mz_scale = mz_scale
        self.bin_size = bin_size
        self.coarse_bin_size = coarse_bin_size
        self.num_fine_bins = round(coarse_bin_size / bin_size)
        self.coarse_embedding = nn.Embedding(
            ceil(mz_scale / coarse_bin_size) + 1,
            embedding_dim,
        )
        self.fine_embedding = nn.Embedding(self.num_fine_bins, embedding_dim)
        nn.init.normal_(self.coarse_embedding.weight, std=0.5)
        nn.init.normal_(self.fine_embedding.weight, std=0.5)

    def _indices(
        self,
        peak_mz: Float[Tensor, "*batch"],
    ) -> tuple[Tensor, Tensor]:
        mz_da = (peak_mz * self.mz_scale).clamp(0.0, self.mz_scale)
        coarse = torch.floor(mz_da / self.coarse_bin_size).long()
        residual = mz_da - coarse * self.coarse_bin_size
        fine = torch.floor(residual / self.bin_size + 0.5).long()
        coarse = coarse + torch.div(
            fine,
            self.num_fine_bins,
            rounding_mode="floor",
        )
        fine = fine.remainder(self.num_fine_bins)
        return coarse, fine

    def quantized_mz(
        self,
        peak_mz: Float[Tensor, "*batch"],
    ) -> Float[Tensor, "*batch"]:
        coarse, fine = self._indices(peak_mz)
        mz_da = coarse * self.coarse_bin_size + fine * self.bin_size
        return mz_da.to(dtype=peak_mz.dtype) / self.mz_scale

    def forward(
        self,
        peak_mz: Float[Tensor, "*batch"],
    ) -> Float[Tensor, "*batch features"]:
        coarse, fine = self._indices(peak_mz)
        return self.coarse_embedding(coarse) + self.fine_embedding(fine)

    def num_features(self) -> int:
        return self.coarse_embedding.embedding_dim


def _build_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int,
) -> nn.Sequential:
    assert num_layers >= 2
    layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
    for _ in range(num_layers - 2):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


def _init_mlp(module: nn.Sequential) -> None:
    for layer in module:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)


class PeakFeatureEmbedder(nn.Module):
    mz_features: FourierFeatures | DiscretizedMzFeatures
    mz_ffn: nn.Sequential
    raw_ffn: nn.Sequential
    output_proj: nn.Linear

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
        mz_scale: float = PEAK_MZ_MAX,
        mz_embedding: str = "fourier",
        discrete_bin_size: float = 0.02,
        discrete_coarse_bin_size: float = 1.0,
        discrete_embedding_dim: int = 70,
    ) -> None:
        super().__init__()
        self.mz_embedding = mz_embedding.lower()
        if self.mz_embedding not in MZ_EMBEDDING_MODES:
            raise ValueError(
                "mz_embedding must be one of ('fourier', 'discrete')"
            )
        fourier_hidden_dim = (
            hidden_dim if fourier_mlp_hidden_dim is None else fourier_mlp_hidden_dim
        )
        self.mz_scale = mz_scale
        mz_dim = model_dim // 2
        raw_dim = model_dim - mz_dim

        # Keep the random stream for all shared model parameters identical across
        # embedding ablations.
        with torch.random.fork_rng(devices=[]):
            if self.mz_embedding == "fourier":
                self.mz_features = FourierFeatures(
                    x_min=fourier_x_min,
                    x_max=fourier_x_max,
                    num_freqs=fourier_num_freqs,
                )
            else:
                self.mz_features = DiscretizedMzFeatures(
                    mz_scale=mz_scale,
                    bin_size=discrete_bin_size,
                    coarse_bin_size=discrete_coarse_bin_size,
                    embedding_dim=discrete_embedding_dim,
                )
            self.mz_ffn = _build_mlp(
                self.mz_features.num_features(),
                fourier_hidden_dim,
                mz_dim,
                fourier_mlp_num_layers,
            )
            _init_mlp(self.mz_ffn)
        self.raw_ffn = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, raw_dim),
        )
        self.output_proj = nn.Linear(model_dim, model_dim)

        _init_mlp(self.raw_ffn)
        nn.init.xavier_normal_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def _prepare_fourier_mz(
        self,
        peak_mz: Float[Tensor, "batch peaks"],
    ) -> Float[Tensor, "batch peaks 1"]:
        # peak_mz: [B, N] -> [B, N, 1], in Da-scale units for Fourier features.
        return peak_mz.unsqueeze(-1) * self.mz_scale

    def forward(
        self,
        peak_mz: Float[Tensor, "batch peaks"],
        peak_intensity: Float[Tensor, "batch peaks"],
    ) -> Float[Tensor, "batch peaks dim"]:
        with torch.autocast(device_type=peak_mz.device.type, enabled=False):
            peak_mz = peak_mz.float()
            peak_intensity = peak_intensity.float()
            embedded_mz = peak_mz
            if self.mz_embedding == "discrete":
                embedded_mz = self.mz_features.quantized_mz(peak_mz)
            # mz/intensity/log_intensity: [B, N, 1]; raw: [B, N, D_raw]
            mz = embedded_mz.unsqueeze(-1)
            intensity = peak_intensity.unsqueeze(-1)
            log_intensity = torch.log1p(peak_intensity).unsqueeze(-1)
            raw = self.raw_ffn(torch.cat([mz, intensity, log_intensity], dim=-1))
            mz_features = (
                self.mz_features(self._prepare_fourier_mz(peak_mz))
                if self.mz_embedding == "fourier"
                else self.mz_features(peak_mz)
            )
            mz_embedding = self.mz_ffn(mz_features)
            return self.output_proj(torch.cat([mz_embedding, raw], dim=-1))
