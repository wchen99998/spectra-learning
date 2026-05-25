from __future__ import annotations

from math import log10

import torch
from jaxtyping import Float
from torch import Tensor, nn

from spectra_learning.data.spectra import PEAK_MZ_MAX


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


class PeakFeatureEmbedder(nn.Module):
    mz_fourier: FourierFeatures
    fourier_ffn: nn.Sequential
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
        fourier_input_scale: float = PEAK_MZ_MAX,
        use_fourier_features: bool = True,
    ) -> None:
        super().__init__()
        self.use_fourier_features = use_fourier_features
        fourier_hidden_dim = (
            hidden_dim if fourier_mlp_hidden_dim is None else fourier_mlp_hidden_dim
        )

        self.fourier_input_scale = fourier_input_scale
        if self.use_fourier_features:
            fourier_dim = model_dim // 2
            raw_dim = model_dim - fourier_dim
            self.mz_fourier = FourierFeatures(
                x_min=fourier_x_min,
                x_max=fourier_x_max,
                num_freqs=fourier_num_freqs,
            )
            self.fourier_ffn = _build_mlp(
                self.mz_fourier.num_features(),
                fourier_hidden_dim,
                fourier_dim,
                fourier_mlp_num_layers,
            )
            self.raw_ffn = nn.Sequential(
                nn.Linear(3, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, raw_dim),
            )
        else:
            self.raw_ffn = _build_mlp(
                3,
                fourier_hidden_dim,
                model_dim,
                fourier_mlp_num_layers,
            )
        self.output_proj = nn.Linear(model_dim, model_dim)

        modules = [self.raw_ffn]
        if self.use_fourier_features:
            modules.append(self.fourier_ffn)
        for module in modules:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    nn.init.zeros_(layer.bias)
        nn.init.xavier_normal_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def _prepare_fourier_mz(
        self,
        peak_mz: Float[Tensor, "batch peaks"],
    ) -> Float[Tensor, "batch peaks 1"]:
        # peak_mz: [B, N] -> [B, N, 1], in Da-scale units for Fourier features.
        return peak_mz.unsqueeze(-1) * self.fourier_input_scale

    def forward(
        self,
        peak_mz: Float[Tensor, "batch peaks"],
        peak_intensity: Float[Tensor, "batch peaks"],
    ) -> Float[Tensor, "batch peaks dim"]:
        with torch.autocast(device_type=peak_mz.device.type, enabled=False):
            peak_mz = peak_mz.float()
            peak_intensity = peak_intensity.float()
            # mz/intensity/log_intensity: [B, N, 1]; raw: [B, N, D_raw]
            mz = peak_mz.unsqueeze(-1)
            intensity = peak_intensity.unsqueeze(-1)
            log_intensity = torch.log1p(peak_intensity).unsqueeze(-1)
            raw = self.raw_ffn(torch.cat([mz, intensity, log_intensity], dim=-1))
            if not self.use_fourier_features:
                return self.output_proj(raw)
            # fourier: [B, N, D_fourier]; cat([fourier, raw]): [B, N, D]
            fourier = self.fourier_ffn(
                self.mz_fourier(self._prepare_fourier_mz(peak_mz))
            )
            return self.output_proj(torch.cat([fourier, raw], dim=-1))
