from __future__ import annotations

from math import ceil, log10

import torch
from torch import nn

from utils.spectra_preprocessing import PEAK_MZ_MAX


class FourierFeatures(nn.Module):
    def __init__(
        self,
        strategy: str = "log_spaced",
        x_min: float = 3e-3,
        x_max: float = 1000.0,
        *,
        trainable: bool = False,
        funcs: str = "both",
        sigma: float = 10.0,
        num_freqs: int = 256,
    ) -> None:
        super().__init__()
        assert strategy in {"random", "voronov_et_al", "lin_float_int", "log_spaced"}
        assert funcs in {"both", "sin", "cos"}
        assert x_min > 0.0
        assert x_max > x_min

        self.funcs = funcs
        self.strategy = strategy
        self.trainable = trainable
        self.num_freqs = num_freqs

        if strategy == "random":
            b = torch.randn(num_freqs, dtype=torch.float32) * sigma
        elif strategy in {"log_spaced", "voronov_et_al"}:
            wavelengths = torch.logspace(
                start=log10(x_min),
                end=log10(x_max),
                steps=num_freqs,
                dtype=torch.float32,
            )
            b = 1.0 / wavelengths
        else:
            periods = torch.tensor(
                [x_min * i for i in range(2, ceil(1.0 / x_min), 2)]
                + [float(i) for i in range(2, ceil(x_max), 1)],
                dtype=torch.float32,
            )
            if num_freqs < periods.numel():
                idx = torch.linspace(0, periods.numel() - 1, steps=num_freqs)
                periods = periods[idx.round().to(torch.long)]
            b = 1.0 / periods
        self.b = nn.Parameter(b.unsqueeze(0), requires_grad=trainable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = 2 * torch.pi * x @ self.b
        if self.funcs == "both":
            return torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)
        if self.funcs == "cos":
            return torch.cos(angles)
        return torch.sin(angles)

    def num_features(self) -> int:
        return self.b.shape[1] if self.funcs != "both" else 2 * self.b.shape[1]


class PeakFeatureEmbedder(nn.Module):
    def __init__(
        self,
        *,
        model_dim: int,
        hidden_dim: int,
        fourier_strategy: str = "log_spaced",
        fourier_x_min: float = 3e-3,
        fourier_x_max: float = 1000.0,
        fourier_funcs: str = "both",
        fourier_num_freqs: int = 256,
        fourier_sigma: float = 10.0,
        fourier_trainable: bool = False,
        fourier_input_scale: float = PEAK_MZ_MAX,
    ) -> None:
        super().__init__()
        fourier_dim = model_dim // 2
        raw_dim = model_dim - fourier_dim

        self.fourier_input_scale = float(fourier_input_scale)
        self.mz_fourier = FourierFeatures(
            strategy=fourier_strategy,
            x_min=fourier_x_min,
            x_max=fourier_x_max,
            trainable=fourier_trainable,
            funcs=fourier_funcs,
            sigma=fourier_sigma,
            num_freqs=fourier_num_freqs,
        )
        self.fourier_ffn = nn.Sequential(
            nn.Linear(self.mz_fourier.num_features(), hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, fourier_dim),
        )
        self.raw_ffn = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, raw_dim),
        )
        self.output_proj = nn.Linear(model_dim, model_dim)

        for module in (self.fourier_ffn, self.raw_ffn):
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    nn.init.zeros_(layer.bias)
        nn.init.xavier_normal_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def _prepare_fourier_mz(self, peak_mz: torch.Tensor) -> torch.Tensor:
        return peak_mz.unsqueeze(-1) * self.fourier_input_scale

    def forward(self, peak_mz: torch.Tensor, peak_intensity: torch.Tensor) -> torch.Tensor:
        mz = peak_mz.unsqueeze(-1)
        intensity = peak_intensity.unsqueeze(-1)
        log_intensity = torch.log1p(peak_intensity.clamp(min=0.0)).unsqueeze(-1)
        fourier = self.fourier_ffn(self.mz_fourier(self._prepare_fourier_mz(peak_mz)))
        raw = self.raw_ffn(torch.cat([mz, intensity, log_intensity], dim=-1))
        return self.output_proj(torch.cat([fourier, raw], dim=-1))
