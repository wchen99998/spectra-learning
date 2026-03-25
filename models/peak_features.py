from __future__ import annotations

from math import ceil

import torch
from torch import nn


class FourierFeatures(nn.Module):
    def __init__(
        self,
        strategy: str = "lin_float_int",
        x_min: float = 1e-4,
        x_max: float = 1000.0,
        *,
        trainable: bool = True,
        funcs: str = "sin",
        sigma: float = 10.0,
        num_freqs: int = 512,
    ) -> None:
        super().__init__()
        assert strategy in {"random", "voronov_et_al", "lin_float_int"}
        assert funcs in {"both", "sin", "cos"}
        assert x_min < 1.0

        self.funcs = funcs
        self.strategy = strategy
        self.trainable = trainable
        self.num_freqs = num_freqs

        if strategy == "random":
            b = torch.randn(num_freqs, dtype=torch.float32) * sigma
        elif strategy == "voronov_et_al":
            b = torch.tensor(
                [
                    1.0 / (x_min * (x_max / x_min) ** (2 * i / (num_freqs - 2)))
                    for i in range(1, num_freqs)
                ],
                dtype=torch.float32,
            )
        else:
            b = torch.tensor(
                [1.0 / (x_min * i) for i in range(2, ceil(1.0 / x_min), 2)]
                + [1.0 / i for i in range(2, ceil(x_max), 1)],
                dtype=torch.float32,
            )
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
        fourier_strategy: str = "lin_float_int",
        fourier_x_min: float = 1e-4,
        fourier_x_max: float = 1000.0,
        fourier_funcs: str = "sin",
        fourier_num_freqs: int = 512,
        fourier_sigma: float = 10.0,
        fourier_trainable: bool = True,
    ) -> None:
        super().__init__()
        fourier_dim = model_dim // 2
        raw_dim = model_dim - fourier_dim

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
            nn.Linear(2, hidden_dim),
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

    def forward(self, peak_mz: torch.Tensor, peak_intensity: torch.Tensor) -> torch.Tensor:
        mz = peak_mz.unsqueeze(-1)
        intensity = peak_intensity.unsqueeze(-1)
        fourier = self.fourier_ffn(self.mz_fourier(mz))
        raw = self.raw_ffn(torch.cat([mz, intensity], dim=-1))
        return self.output_proj(torch.cat([fourier, raw], dim=-1))
