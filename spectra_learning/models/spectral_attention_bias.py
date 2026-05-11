from __future__ import annotations

import math
from math import ceil, log10

import torch
import torch.nn.functional as F
from torch import nn


def _init_weight(param: nn.Parameter, std: float) -> None:
    if float(std) == 0.0:
        nn.init.zeros_(param)
    else:
        nn.init.normal_(param, mean=0.0, std=float(std))


def _make_frequency_b(
    *,
    strategy: str,
    num_freqs: int,
    x_min: float,
    x_max: float,
    sigma: float,
) -> torch.Tensor:
    strategy = str(strategy).lower()
    num_freqs = int(num_freqs)
    if num_freqs <= 0:
        raise ValueError("num_freqs must be positive")
    if x_min <= 0.0:
        raise ValueError("x_min must be positive")
    if x_max <= x_min:
        raise ValueError("x_max must be larger than x_min")

    if strategy == "random":
        return torch.randn(num_freqs, dtype=torch.float32) * float(sigma)

    if strategy in {"log_spaced", "voronov_et_al"}:
        wavelengths = torch.logspace(
            start=log10(float(x_min)),
            end=log10(float(x_max)),
            steps=num_freqs,
            dtype=torch.float32,
        )
        return 1.0 / wavelengths

    if strategy == "lin_float_int":
        periods = torch.tensor(
            [float(x_min) * i for i in range(2, ceil(1.0 / float(x_min)), 2)]
            + [float(i) for i in range(2, ceil(float(x_max)), 1)],
            dtype=torch.float32,
        )
        if periods.numel() == 0:
            periods = torch.linspace(float(x_min), float(x_max), steps=num_freqs)
        if num_freqs < periods.numel():
            idx = torch.linspace(0, periods.numel() - 1, steps=num_freqs)
            periods = periods[idx.round().to(torch.long)]
        elif num_freqs > periods.numel():
            extra = torch.logspace(
                log10(float(x_min)),
                log10(float(x_max)),
                steps=num_freqs - periods.numel(),
            )
            periods = torch.cat([periods, extra], dim=0)
        return 1.0 / periods

    raise ValueError(f"Unsupported Fourier strategy: {strategy!r}")


class FourierFrequencyBank(nn.Module):
    """Shared Fourier frequency bank using angles 2 pi b x."""

    def __init__(
        self,
        *,
        num_freqs: int,
        strategy: str = "log_spaced",
        x_min: float = 3e-3,
        x_max: float = 1000.0,
        sigma: float = 10.0,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        b = _make_frequency_b(
            strategy=strategy,
            num_freqs=num_freqs,
            x_min=x_min,
            x_max=x_max,
            sigma=sigma,
        )
        self.b = nn.Parameter(b, requires_grad=bool(trainable))

    @property
    def num_freqs(self) -> int:
        return int(self.b.numel())

    def angles(self, x_da: torch.Tensor) -> torch.Tensor:
        return (2.0 * math.pi) * x_da.float().unsqueeze(-1) * self.b.float()

    def sin_cos(self, x_da: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        angles = self.angles(x_da)
        return torch.sin(angles), torch.cos(angles)


class HarmonicRelativeLossBias(nn.Module):
    """Exact factorized Fourier bias for psi_h(m_i - m_j)."""

    def __init__(
        self,
        *,
        num_heads: int,
        num_freqs: int,
        strategy: str = "log_spaced",
        x_min: float = 3e-3,
        x_max: float = 1000.0,
        sigma: float = 10.0,
        trainable_freqs: bool = False,
        directional: bool = True,
        init_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.freqs = FourierFrequencyBank(
            num_freqs=num_freqs,
            strategy=strategy,
            x_min=x_min,
            x_max=x_max,
            sigma=sigma,
            trainable=trainable_freqs,
        )
        self.num_heads = int(num_heads)
        self.directional = bool(directional)
        self.cos_weight = nn.Parameter(torch.empty(self.num_heads, num_freqs))
        self.sin_weight = (
            nn.Parameter(torch.empty(self.num_heads, num_freqs))
            if self.directional
            else None
        )
        _init_weight(self.cos_weight, init_std)
        if self.sin_weight is not None:
            _init_weight(self.sin_weight, init_std)

        self.register_buffer(
            "output_scale",
            torch.tensor(1.0 / math.sqrt(max(1, num_freqs)), dtype=torch.float32),
            persistent=False,
        )

    def forward(self, mass_da: torch.Tensor) -> torch.Tensor:
        s, c = self.freqs.sin_cos(mass_da)
        cw = self.cos_weight.float() * self.output_scale

        bias = torch.einsum("bir,hr,bjr->bhij", c, cw, c)
        bias = bias + torch.einsum("bir,hr,bjr->bhij", s, cw, s)

        if self.sin_weight is not None:
            sw = self.sin_weight.float() * self.output_scale
            bias = bias + torch.einsum("bir,hr,bjr->bhij", s, sw, c)
            bias = bias - torch.einsum("bir,hr,bjr->bhij", c, sw, s)

        return bias


class PrecursorNeutralLossBias(nn.Module):
    """Key-only precursor neutral-loss bias eta_h(p - m_j)."""

    def __init__(
        self,
        *,
        num_heads: int,
        num_freqs: int,
        strategy: str = "log_spaced",
        x_min: float = 3e-3,
        x_max: float = 1000.0,
        sigma: float = 10.0,
        trainable_freqs: bool = False,
        use_cos: bool = True,
        use_sin: bool = True,
        init_std: float = 0.0,
    ) -> None:
        super().__init__()
        if not use_cos and not use_sin:
            raise ValueError("At least one of use_cos/use_sin must be True")
        self.freqs = FourierFrequencyBank(
            num_freqs=num_freqs,
            strategy=strategy,
            x_min=x_min,
            x_max=x_max,
            sigma=sigma,
            trainable=trainable_freqs,
        )
        self.num_heads = int(num_heads)
        self.use_cos = bool(use_cos)
        self.use_sin = bool(use_sin)

        if self.use_cos:
            self.cos_weight = nn.Parameter(torch.empty(self.num_heads, num_freqs))
            _init_weight(self.cos_weight, init_std)
        else:
            self.register_parameter("cos_weight", None)

        if self.use_sin:
            self.sin_weight = nn.Parameter(torch.empty(self.num_heads, num_freqs))
            _init_weight(self.sin_weight, init_std)
        else:
            self.register_parameter("sin_weight", None)

        self.register_buffer(
            "output_scale",
            torch.tensor(1.0 / math.sqrt(max(1, num_freqs)), dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        mass_da: torch.Tensor,
        precursor_da: torch.Tensor,
    ) -> torch.Tensor:
        neutral_loss = precursor_da.float().unsqueeze(1) - mass_da.float()
        s, c = self.freqs.sin_cos(neutral_loss)

        bias = None
        if self.cos_weight is not None:
            cw = self.cos_weight.float() * self.output_scale
            bias = torch.einsum("bnr,hr->bhn", c, cw)
        if self.sin_weight is not None:
            sw = self.sin_weight.float() * self.output_scale
            sin_bias = torch.einsum("bnr,hr->bhn", s, sw)
            bias = sin_bias if bias is None else bias + sin_bias

        assert bias is not None
        return bias.unsqueeze(2)


class RBFRelativeLossBias(nn.Module):
    """Mass-tolerance-aware RBF bias for psi_h(m_i - m_j)."""

    def __init__(
        self,
        *,
        num_heads: int,
        num_basis: int = 64,
        delta_min: float = -1000.0,
        delta_max: float = 1000.0,
        init_sigma: float | None = None,
        use_absolute_delta: bool = False,
        learnable_centers: bool = True,
        learnable_sigma: bool = True,
        init_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.num_basis = int(num_basis)
        self.use_absolute_delta = bool(use_absolute_delta)

        lo = 0.0 if self.use_absolute_delta else float(delta_min)
        hi = float(delta_max)
        centers = torch.linspace(lo, hi, steps=self.num_basis, dtype=torch.float32)
        self.centers = nn.Parameter(centers, requires_grad=bool(learnable_centers))

        if init_sigma is None:
            init_sigma = max((hi - lo) / max(1, self.num_basis - 1), 1e-3)
        self.log_sigma = nn.Parameter(
            torch.full(
                (self.num_basis,),
                math.log(float(init_sigma)),
                dtype=torch.float32,
            ),
            requires_grad=bool(learnable_sigma),
        )
        self.weight = nn.Parameter(torch.empty(self.num_heads, self.num_basis))
        _init_weight(self.weight, init_std)

        self.register_buffer(
            "output_scale",
            torch.tensor(1.0 / math.sqrt(max(1, self.num_basis)), dtype=torch.float32),
            persistent=False,
        )

    def forward(self, mass_da: torch.Tensor) -> torch.Tensor:
        delta = mass_da.float().unsqueeze(2) - mass_da.float().unsqueeze(1)
        if self.use_absolute_delta:
            delta = delta.abs()

        sigma = self.log_sigma.float().exp().clamp_min(1e-4)
        z = (delta.unsqueeze(-1) - self.centers.float()) / sigma
        rbf = torch.exp(-0.5 * z.square())

        w = self.weight.float() * self.output_scale
        return torch.einsum("bijr,hr->bhij", rbf, w)


class LegacyDreamsDifferenceBias(nn.Module):
    """Ablation module for beta_ijh = w_h^T Phi(m_i) - w_h^T Phi(m_j)."""

    def __init__(
        self,
        *,
        num_heads: int,
        num_freqs: int,
        strategy: str = "log_spaced",
        x_min: float = 3e-3,
        x_max: float = 1000.0,
        sigma: float = 10.0,
        trainable_freqs: bool = False,
        init_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.freqs = FourierFrequencyBank(
            num_freqs=num_freqs,
            strategy=strategy,
            x_min=x_min,
            x_max=x_max,
            sigma=sigma,
            trainable=trainable_freqs,
        )
        self.num_heads = int(num_heads)
        self.weight = nn.Parameter(torch.empty(self.num_heads, 2 * num_freqs))
        _init_weight(self.weight, init_std)
        self.register_buffer(
            "output_scale",
            torch.tensor(1.0 / math.sqrt(max(1, 2 * num_freqs)), dtype=torch.float32),
            persistent=False,
        )

    def forward(self, mass_da: torch.Tensor) -> torch.Tensor:
        s, c = self.freqs.sin_cos(mass_da)
        features = torch.cat([c, s], dim=-1)
        w = self.weight.float() * self.output_scale
        token_score = torch.einsum("bnf,hf->bhn", features, w)
        return token_score.unsqueeze(-1) - token_score.unsqueeze(-2)


class IntensityPairBias(nn.Module):
    """Optional learned rho_h(I_i, I_j)."""

    def __init__(
        self,
        *,
        num_heads: int,
        hidden_dim: int = 16,
        init_last_zero: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(6, int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), self.num_heads),
        )
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)
        if init_last_zero:
            last = self.mlp[-1]
            assert isinstance(last, nn.Linear)
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, intensity: torch.Tensor) -> torch.Tensor:
        intensity = intensity.float()
        bsz, n = intensity.shape

        iq = intensity.unsqueeze(2).expand(bsz, n, n)
        ik = intensity.unsqueeze(1).expand(bsz, n, n)
        log_iq = torch.log1p(iq)
        log_ik = torch.log1p(ik)

        features = torch.stack(
            [
                iq,
                ik,
                log_iq,
                log_ik,
                (iq - ik).abs(),
                iq * ik,
            ],
            dim=-1,
        )

        out = self.mlp(features)
        return out.permute(0, 3, 1, 2).contiguous()


class SpectralGraphormerBias(nn.Module):
    """Combines spectral Graphormer-style additive attention biases."""

    def __init__(
        self,
        *,
        num_heads: int,
        mass_scale: float = 1000.0,
        precursor_scale: float = 1000.0,
        first_token_is_precursor: bool = False,
        relative_kind: str = "none",
        num_freqs: int = 128,
        fourier_strategy: str = "log_spaced",
        fourier_x_min: float = 3e-3,
        fourier_x_max: float = 1000.0,
        fourier_sigma: float = 10.0,
        fourier_trainable: bool = False,
        use_precursor_bias: bool = False,
        precursor_num_freqs: int | None = None,
        use_intensity_bias: bool = False,
        intensity_hidden_dim: int = 16,
        rbf_num_basis: int = 64,
        rbf_delta_min: float = -1000.0,
        rbf_delta_max: float = 1000.0,
        rbf_init_sigma: float | None = None,
        rbf_use_absolute_delta: bool = False,
        init_std: float = 0.0,
        bias_clip: float | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.mass_scale = float(mass_scale)
        self.precursor_scale = float(precursor_scale)
        self.first_token_is_precursor = bool(first_token_is_precursor)
        self.bias_clip = None if bias_clip is None else float(bias_clip)

        kind = str(relative_kind).lower()
        if kind in {"", "none", "false", "off"}:
            self.relative_bias = None
        elif kind in {"harmonic", "fourier", "relative_fourier"}:
            self.relative_bias = HarmonicRelativeLossBias(
                num_heads=self.num_heads,
                num_freqs=int(num_freqs),
                strategy=fourier_strategy,
                x_min=fourier_x_min,
                x_max=fourier_x_max,
                sigma=fourier_sigma,
                trainable_freqs=fourier_trainable,
                directional=True,
                init_std=init_std,
            )
        elif kind in {"harmonic_even", "cos", "cosine", "magnitude"}:
            self.relative_bias = HarmonicRelativeLossBias(
                num_heads=self.num_heads,
                num_freqs=int(num_freqs),
                strategy=fourier_strategy,
                x_min=fourier_x_min,
                x_max=fourier_x_max,
                sigma=fourier_sigma,
                trainable_freqs=fourier_trainable,
                directional=False,
                init_std=init_std,
            )
        elif kind in {"rbf", "radial_basis"}:
            self.relative_bias = RBFRelativeLossBias(
                num_heads=self.num_heads,
                num_basis=int(rbf_num_basis),
                delta_min=float(rbf_delta_min),
                delta_max=float(rbf_delta_max),
                init_sigma=rbf_init_sigma,
                use_absolute_delta=bool(rbf_use_absolute_delta),
                init_std=init_std,
            )
        elif kind in {"legacy_dreams", "dreams_difference", "dreams"}:
            self.relative_bias = LegacyDreamsDifferenceBias(
                num_heads=self.num_heads,
                num_freqs=int(num_freqs),
                strategy=fourier_strategy,
                x_min=fourier_x_min,
                x_max=fourier_x_max,
                sigma=fourier_sigma,
                trainable_freqs=fourier_trainable,
                init_std=init_std,
            )
        else:
            raise ValueError(f"Unsupported spectral relative_kind: {relative_kind!r}")

        if use_precursor_bias:
            self.precursor_bias = PrecursorNeutralLossBias(
                num_heads=self.num_heads,
                num_freqs=int(precursor_num_freqs or num_freqs),
                strategy=fourier_strategy,
                x_min=fourier_x_min,
                x_max=fourier_x_max,
                sigma=fourier_sigma,
                trainable_freqs=fourier_trainable,
                use_cos=True,
                use_sin=True,
                init_std=init_std,
            )
        else:
            self.precursor_bias = None

        self.intensity_bias = (
            IntensityPairBias(
                num_heads=self.num_heads,
                hidden_dim=intensity_hidden_dim,
                init_last_zero=(float(init_std) == 0.0),
            )
            if use_intensity_bias
            else None
        )

    def _resolve_precursor_da(
        self,
        mass_da: torch.Tensor,
        precursor_mz: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if precursor_mz is not None:
            return precursor_mz.float() * self.precursor_scale
        if self.first_token_is_precursor:
            return mass_da[:, 0]
        return None

    def forward(
        self,
        peak_mz: torch.Tensor,
        *,
        peak_intensity: torch.Tensor | None = None,
        precursor_mz: torch.Tensor | None = None,
        num_special_tokens: int = 0,
    ) -> torch.Tensor | None:
        with torch.autocast(device_type=peak_mz.device.type, enabled=False):
            return self._forward_fp32(
                peak_mz.float(),
                peak_intensity=(
                    None if peak_intensity is None else peak_intensity.float()
                ),
                precursor_mz=None if precursor_mz is None else precursor_mz.float(),
                num_special_tokens=num_special_tokens,
            )

    def _forward_fp32(
        self,
        peak_mz: torch.Tensor,
        *,
        peak_intensity: torch.Tensor | None,
        precursor_mz: torch.Tensor | None,
        num_special_tokens: int,
    ) -> torch.Tensor | None:
        mass_da = peak_mz.float() * self.mass_scale

        pair_pieces: list[torch.Tensor] = []
        key_pieces: list[torch.Tensor] = []

        if self.relative_bias is not None:
            pair_pieces.append(self.relative_bias(mass_da))

        if self.precursor_bias is not None:
            precursor_da = self._resolve_precursor_da(mass_da, precursor_mz)
            if precursor_da is not None:
                key_pieces.append(self.precursor_bias(mass_da, precursor_da))

        if self.intensity_bias is not None and peak_intensity is not None:
            pair_pieces.append(self.intensity_bias(peak_intensity))

        if not pair_pieces and not key_pieces:
            return None

        pair_bias = None
        for piece in pair_pieces:
            pair_bias = piece if pair_bias is None else pair_bias + piece

        key_bias = None
        for piece in key_pieces:
            key_bias = piece if key_bias is None else key_bias + piece

        s = int(num_special_tokens)
        if s > 0:
            if pair_bias is not None:
                pair_bias = F.pad(pair_bias, (0, s, 0, s), value=0.0)
            if key_bias is not None:
                peak_count = peak_mz.shape[1]
                key_bias = key_bias.expand(-1, -1, peak_count, -1)
                key_bias = F.pad(key_bias, (0, s, 0, s), value=0.0)

        bias = pair_bias if pair_bias is not None else key_bias
        if pair_bias is not None and key_bias is not None:
            bias = pair_bias + key_bias

        if self.bias_clip is not None:
            bias = bias.clamp(min=-self.bias_clip, max=self.bias_clip)

        return bias
