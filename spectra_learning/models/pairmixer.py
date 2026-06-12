from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float
from torch import Tensor, nn

from spectra_learning.data.spectra import PEAK_MZ_MAX
from spectra_learning.models.peak_features import FourierFeatures
from spectra_learning.models.transformer import Attention, FeedForward, _build_norm


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


def _init_linear(linear: nn.Linear, *, gate: bool = False) -> None:
    if gate:
        nn.init.zeros_(linear.weight)
        if linear.bias is not None:
            nn.init.ones_(linear.bias)
        return
    nn.init.xavier_normal_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


def _pair_mask(
    peak_mask: Bool[Tensor, "batch peaks"],
) -> Bool[Tensor, "batch peaks peaks"]:
    return peak_mask.unsqueeze(2) & peak_mask.unsqueeze(1)


class PairFeatureEmbedder(nn.Module):
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
    ) -> None:
        super().__init__()
        self.mz_scale = mz_scale
        self.precursor_mz_scale = precursor_mz_scale
        self.sigma_ppm = sigma_ppm
        self.use_fourier_features = use_fourier_features
        self.register_buffer(
            "mass_differences",
            torch.tensor(COMMON_MASS_DIFFERENCES_DA, dtype=torch.float32),
            persistent=False,
        )
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
        self.raw_proj = nn.Sequential(
            nn.Linear(raw_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, pair_dim),
        )
        self.single_pair_proj = nn.Sequential(
            nn.Linear(4 * single_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, pair_dim),
        )
        for module in (self.raw_proj, self.single_pair_proj):
            for layer in module:
                if isinstance(layer, nn.Linear):
                    _init_linear(layer)

    def _reference_mass_da(
        self,
        peak_mz: Float[Tensor, "batch peaks"],
        valid_mask: Bool[Tensor, "batch peaks"],
        precursor_mz: Float[Tensor, "batch"] | None,
    ) -> Float[Tensor, "batch"]:
        # peak_mz: [B, N], precursor_mz: [B] -> reference mass: [B]
        if precursor_mz is not None:
            return (precursor_mz.float() * self.precursor_mz_scale).clamp_min(1.0)
        mz_da = peak_mz.float() * self.mz_scale
        return (mz_da * valid_mask.float()).amax(dim=1).clamp_min(1.0)

    def _fourier_values(
        self,
        fourier: FourierFeatures,
        values: Float[Tensor, "*batch features"],
    ) -> Float[Tensor, "*batch fourier"]:
        # values: [..., F] -> [..., F * 2 * num_freqs]
        return fourier(values.unsqueeze(-1)).flatten(start_dim=-2)

    def forward(
        self,
        peak_mz: Float[Tensor, "batch peaks"],
        peak_intensity: Float[Tensor, "batch peaks"],
        single: Float[Tensor, "batch peaks dim"],
        valid_mask: Bool[Tensor, "batch peaks"],
        precursor_mz: Float[Tensor, "batch"] | None = None,
    ) -> Float[Tensor, "batch peaks peaks pair"]:
        with torch.autocast(device_type=peak_mz.device.type, enabled=False):
            mz_da = peak_mz.float() * self.mz_scale
            intensity = peak_intensity.float()
            reference_mass = self._reference_mass_da(
                peak_mz,
                valid_mask,
                precursor_mz,
            ).view(-1, 1, 1)
            # mz_i: [B, N, 1], mz_j: [B, 1, N], d/abs_d/relative_d: [B, N, N]
            mz_i = mz_da.unsqueeze(2)
            mz_j = mz_da.unsqueeze(1)
            d = mz_j - mz_i
            abs_d = d.abs()
            relative_d = d / reference_mass
            complement = mz_i + mz_j - reference_mass
            mass_diffs = self.mass_differences.to(device=peak_mz.device)
            # radial: [B, N, N, num_common_mass_differences]
            ppm = 1e6 * (abs_d.unsqueeze(-1) - mass_diffs) / mass_diffs
            radial = torch.exp(-0.5 * (ppm / self.sigma_ppm).square())

            intensity_i = intensity.unsqueeze(2)
            intensity_j = intensity.unsqueeze(1)
            diag = torch.eye(
                peak_mz.shape[1],
                device=peak_mz.device,
                dtype=peak_mz.dtype,
            ).view(1, peak_mz.shape[1], peak_mz.shape[1])
            raw_parts = [
                d.unsqueeze(-1) / self.mz_scale,
                abs_d.unsqueeze(-1) / self.mz_scale,
                relative_d.unsqueeze(-1),
                complement.unsqueeze(-1) / self.mz_scale,
                mz_i.expand_as(d).unsqueeze(-1) / reference_mass.unsqueeze(-1),
                mz_j.expand_as(d).unsqueeze(-1) / reference_mass.unsqueeze(-1),
                intensity_i.expand_as(d).unsqueeze(-1),
                intensity_j.expand_as(d).unsqueeze(-1),
                (intensity_i * intensity_j).expand_as(d).unsqueeze(-1),
                torch.log1p(intensity_i).expand_as(d).unsqueeze(-1),
                torch.log1p(intensity_j).expand_as(d).unsqueeze(-1),
                torch.sign(d).unsqueeze(-1),
                diag.expand_as(d).unsqueeze(-1),
                (d > 0).to(dtype=peak_mz.dtype).unsqueeze(-1),
                radial,
            ]
            if self.use_fourier_features:
                raw_parts.extend(
                    [
                        self._fourier_values(
                            self.pair_fourier,
                            torch.stack([d, abs_d, complement], dim=-1),
                        ),
                        self._fourier_values(
                            self.relative_pair_fourier,
                            relative_d.unsqueeze(-1),
                        ),
                    ]
                )
            raw = torch.cat(raw_parts, dim=-1)
            # raw: [B, N, N, pair_raw_features]
        single_i = single.unsqueeze(2)
        single_j = single.unsqueeze(1)
        single_pair = torch.cat(
            [
                single_i.expand(-1, -1, single.shape[1], -1),
                single_j.expand(-1, single.shape[1], -1, -1),
                single_i * single_j,
                single_j - single_i,
            ],
            dim=-1,
        )
        # single_pair: [B, N, N, 4 * D]; z: [B, N, N, P]
        z = self.raw_proj(raw.to(dtype=single.dtype)) + self.single_pair_proj(single_pair)
        return z * _pair_mask(valid_mask).unsqueeze(-1).to(dtype=z.dtype)


class TriangleMultiplicativeUpdate(nn.Module):
    def __init__(
        self,
        pair_dim: int,
        *,
        direction: str,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.direction = direction
        self.norm_eps = norm_eps
        self.norm_in = _build_norm(pair_dim, eps=norm_eps)
        self.p_in = nn.Linear(pair_dim, 2 * pair_dim)
        self.g_in = nn.Linear(pair_dim, 2 * pair_dim)
        self.norm_out = _build_norm(pair_dim, eps=norm_eps)
        self.p_out = nn.Linear(pair_dim, pair_dim)
        self.g_out = nn.Linear(pair_dim, pair_dim)
        _init_linear(self.p_in)
        _init_linear(self.g_in, gate=True)
        _init_linear(self.p_out)
        _init_linear(self.g_out, gate=True)

    def forward(
        self,
        x: Float[Tensor, "batch peaks peaks pair"],
        mask: Bool[Tensor, "batch peaks peaks"],
    ) -> Float[Tensor, "batch peaks peaks pair"]:
        pair_mask = mask.unsqueeze(-1).to(dtype=x.dtype)
        x_norm = self.norm_in(x)
        a, b = (self.p_in(x_norm) * torch.sigmoid(self.g_in(x_norm))).chunk(2, dim=-1)
        a = a * pair_mask
        b = b * pair_mask
        # a/b: [B, N, N, P]; update contracts the middle triangle node k.
        if self.direction == "outgoing":
            update = torch.einsum("bikc,bjkc->bijc", a, b)
        else:
            update = torch.einsum("bkic,bkjc->bijc", a, b)
        update = self.p_out(self.norm_out(update))
        update = update * torch.sigmoid(self.g_out(x_norm))
        return update * pair_mask


class CommutedLowRankTriangle(nn.Module):
    def __init__(
        self,
        pair_dim: int,
        *,
        hidden_dim: int,
        num_mediators: int,
        rank: int,
        direction: str,
        norm_eps: float,
        keep_out_gate: bool = True,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.direction = direction
        self.keep_out_gate = keep_out_gate
        self.norm_in = _build_norm(pair_dim, eps=norm_eps)
        self.q = nn.Parameter(
            torch.randn(num_mediators, rank) / math.sqrt(num_mediators)
        )
        self.left_proj = nn.Linear(pair_dim, hidden_dim)
        self.right_proj = nn.Linear(pair_dim, hidden_dim)
        self.left_gate = nn.Linear(pair_dim, hidden_dim)
        self.right_gate = nn.Linear(pair_dim, hidden_dim)
        self.out_gate = nn.Linear(pair_dim, hidden_dim) if keep_out_gate else None
        self.norm_out = _build_norm(hidden_dim, eps=norm_eps)
        self.p_out = nn.Linear(hidden_dim, pair_dim)
        _init_linear(self.left_proj)
        _init_linear(self.right_proj)
        _init_linear(self.left_gate, gate=True)
        _init_linear(self.right_gate, gate=True)
        if self.out_gate is not None:
            _init_linear(self.out_gate, gate=True)
        _init_linear(self.p_out)

    def forward(
        self,
        x: Float[Tensor, "batch peaks peaks pair"],
        mask: Bool[Tensor, "batch peaks peaks"],
    ) -> Float[Tensor, "batch peaks peaks pair"]:
        pair_mask = mask.unsqueeze(-1).to(dtype=x.dtype)
        x_norm = self.norm_in(x)
        x_masked = x_norm * pair_mask
        q = self.q.softmax(dim=0).to(dtype=x.dtype)
        if self.direction == "outgoing":
            x_comp = torch.einsum("bikc,km->bimc", x_masked, q)
        else:
            x_comp = torch.einsum("bkic,km->bimc", x_masked, q)
        left = self.left_proj(x_comp)
        right = self.right_proj(x_comp)
        left = left * torch.sigmoid(self.left_gate(x_comp))
        right = right * torch.sigmoid(self.right_gate(x_comp))
        update = torch.einsum("bimh,bjmh->bijh", left, right)
        update = self.norm_out(update)
        if self.out_gate is not None:
            update = update * torch.sigmoid(self.out_gate(x_norm))
        return self.p_out(update) * pair_mask


class TriangleAttention(nn.Module):
    def __init__(
        self,
        pair_dim: int,
        *,
        num_heads: int,
        ending: bool,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = pair_dim // num_heads
        self.ending = ending
        self.norm = _build_norm(pair_dim, eps=norm_eps)
        self.qkv = nn.Linear(pair_dim, 3 * pair_dim, bias=False)
        self.bias = nn.Linear(pair_dim, num_heads, bias=False)
        self.g = nn.Linear(pair_dim, pair_dim)
        self.o = nn.Linear(pair_dim, pair_dim)
        _init_linear(self.qkv)
        _init_linear(self.bias)
        _init_linear(self.g, gate=True)
        _init_linear(self.o)

    def _start_attention(
        self,
        x: Float[Tensor, "batch peaks peaks pair"],
        peak_mask: Bool[Tensor, "batch peaks"],
        pair_mask: Bool[Tensor, "batch peaks peaks"],
    ) -> Float[Tensor, "batch peaks peaks pair"]:
        batch_size, num_peaks, _, pair_dim = x.shape
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm).view(
            batch_size,
            num_peaks,
            num_peaks,
            3,
            self.num_heads,
            self.head_dim,
        )
        q, k, v = qkv.unbind(dim=3)
        q = q.permute(0, 1, 3, 2, 4)
        k = k.permute(0, 1, 3, 2, 4)
        v = v.permute(0, 1, 3, 2, 4)
        bias = self.bias(x_norm).permute(0, 3, 1, 2).unsqueeze(1)
        key_mask = peak_mask[:, None, None, None, :].expand(
            batch_size,
            num_peaks,
            1,
            1,
            num_peaks,
        )
        scores = (
            torch.einsum("bnhqd,bnhkd->bnhqk", q, k)
            * (1.0 / math.sqrt(self.head_dim))
            + bias
        )
        scores = scores.masked_fill(~key_mask, float("-inf"))
        attn = torch.softmax(scores.float(), dim=-1).to(dtype=v.dtype)
        out = torch.einsum("bnhqk,bnhkd->bnhqd", attn, v)
        out = out.permute(0, 1, 3, 2, 4).reshape(
            batch_size,
            num_peaks,
            num_peaks,
            pair_dim,
        )
        out = out * torch.sigmoid(self.g(x_norm))
        out = self.o(out)
        return out * pair_mask.unsqueeze(-1).to(dtype=out.dtype)

    def forward(
        self,
        x: Float[Tensor, "batch peaks peaks pair"],
        peak_mask: Bool[Tensor, "batch peaks"],
        pair_mask: Bool[Tensor, "batch peaks peaks"],
    ) -> Float[Tensor, "batch peaks peaks pair"]:
        if self.ending:
            return self._start_attention(
                x.transpose(1, 2),
                peak_mask,
                pair_mask.transpose(1, 2),
            ).transpose(1, 2)
        return self._start_attention(x, peak_mask, pair_mask)


class AttentionPairBias(nn.Module):
    def __init__(
        self,
        *,
        single_dim: int,
        pair_dim: int,
        num_heads: int,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = single_dim // num_heads
        self.single_norm = _build_norm(single_dim, eps=norm_eps)
        self.pair_norm = _build_norm(pair_dim, eps=norm_eps)
        self.qkv = nn.Linear(single_dim, 3 * single_dim, bias=False)
        self.pair_bias = nn.Linear(pair_dim, num_heads, bias=False)
        self.g = nn.Linear(single_dim, single_dim)
        self.o = nn.Linear(single_dim, single_dim)
        _init_linear(self.qkv)
        _init_linear(self.pair_bias)
        _init_linear(self.g, gate=True)
        _init_linear(self.o)

    def forward(
        self,
        single: Float[Tensor, "batch tokens dim"],
        pair: Float[Tensor, "batch peaks peaks pair"],
        token_mask: Bool[Tensor, "batch tokens"],
        num_peak_tokens: int,
    ) -> Float[Tensor, "batch tokens dim"]:
        batch_size, num_tokens, single_dim = single.shape
        single_norm = self.single_norm(single)
        qkv = self.qkv(single_norm).view(
            batch_size,
            num_tokens,
            3,
            self.num_heads,
            self.head_dim,
        )
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        peak_bias = self.pair_bias(self.pair_norm(pair)).permute(0, 3, 1, 2)
        extra_tokens = num_tokens - num_peak_tokens
        attn_bias = F.pad(peak_bias, (0, extra_tokens, 0, extra_tokens)).float()
        attn_bias = attn_bias.masked_fill(
            ~token_mask[:, None, None, :],
            float("-inf"),
        ).contiguous()
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        out = out.transpose(1, 2).contiguous().view(batch_size, num_tokens, single_dim)
        out = out * torch.sigmoid(self.g(single_norm))
        return self.o(out)


class PairMixerBlock(nn.Module):
    def __init__(
        self,
        *,
        single_dim: int,
        pair_dim: int,
        num_heads: int,
        attention_mlp_multiple: float,
        norm_eps: float,
        dropout: float,
        triangle_mediator_rank: int = 0,
        use_commuted_low_rank_triangle: bool = False,
        max_mediator_tokens: int = 0,
        use_pair_bias_attention: bool = False,
    ) -> None:
        super().__init__()
        self.use_pair_bias_attention = use_pair_bias_attention
        if use_commuted_low_rank_triangle and triangle_mediator_rank > 0:
            self.tri_mul_out = CommutedLowRankTriangle(
                pair_dim,
                hidden_dim=pair_dim,
                num_mediators=max_mediator_tokens,
                rank=triangle_mediator_rank,
                direction="outgoing",
                norm_eps=norm_eps,
            )
            self.tri_mul_in = CommutedLowRankTriangle(
                pair_dim,
                hidden_dim=pair_dim,
                num_mediators=max_mediator_tokens,
                rank=triangle_mediator_rank,
                direction="incoming",
                norm_eps=norm_eps,
            )
        else:
            self.tri_mul_out = TriangleMultiplicativeUpdate(
                pair_dim,
                direction="outgoing",
                norm_eps=norm_eps,
            )
            self.tri_mul_in = TriangleMultiplicativeUpdate(
                pair_dim,
                direction="incoming",
                norm_eps=norm_eps,
            )
        self.pair_transition_norm = _build_norm(
            pair_dim,
            eps=norm_eps,
        )
        self.pair_transition = FeedForward(
            pair_dim,
            hidden_dim=math.ceil(pair_dim * attention_mlp_multiple),
        )
        if self.use_pair_bias_attention:
            self.single_attention = AttentionPairBias(
                single_dim=single_dim,
                pair_dim=pair_dim,
                num_heads=num_heads,
                norm_eps=norm_eps,
            )
        else:
            self.single_attention_norm = _build_norm(
                single_dim,
                eps=norm_eps,
            )
            self.single_attention = Attention(
                single_dim,
                num_heads,
            )
        self.single_transition_norm = _build_norm(
            single_dim,
            eps=norm_eps,
        )
        self.single_transition = FeedForward(
            single_dim,
            hidden_dim=math.ceil(single_dim * attention_mlp_multiple),
        )
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        single: Float[Tensor, "batch tokens dim"],
        pair: Float[Tensor, "batch peaks peaks pair"],
        peak_mask: Bool[Tensor, "batch peaks"],
        token_mask: Bool[Tensor, "batch tokens"],
    ) -> tuple[
        Float[Tensor, "batch tokens dim"],
        Float[Tensor, "batch peaks peaks pair"],
    ]:
        pair_mask = _pair_mask(peak_mask)
        pair = pair + self.drop(self.tri_mul_out(pair, pair_mask))
        pair = pair + self.drop(self.tri_mul_in(pair, pair_mask))
        pair = pair + self.drop(self.pair_transition(self.pair_transition_norm(pair)))
        pair = pair * pair_mask.unsqueeze(-1).to(dtype=pair.dtype)
        if self.use_pair_bias_attention:
            single = single + self.drop(
                self.single_attention(
                    single,
                    pair,
                    token_mask,
                    peak_mask.shape[1],
                )
            )
        else:
            single = single + self.drop(
                self.single_attention(
                    self.single_attention_norm(single),
                    attn_mask=token_mask[:, None, None, :],
                )
            )
        single = single + self.drop(
            self.single_transition(self.single_transition_norm(single))
        )
        return single, pair
