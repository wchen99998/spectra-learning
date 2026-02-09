"""Peak-set SIGReg model for mass spectrometry pretraining.

Architecture:
- PeakSetEncoder: MLP embedder -> non-causal Transformer -> peak embeddings
- Two-view spectrum augmentation (drop + jitter)
- Optional projector on pooled embeddings
- SIGReg objective: MSE(z1, z2) + lambda * BCS(z1, z2)
"""

from __future__ import annotations

import math

import torch
from torch import nn

from models.losses import BCSLoss
from networks import transformer_torch


class FourierFeatures(nn.Module):
    """Log-spaced sinusoidal features for scalar inputs (NeRF-style)."""

    def __init__(
        self,
        num_frequencies: int = 32,
        min_freq: float = 1.0,
        max_freq: float = 100.0,
        learnable: bool = False,
    ):
        super().__init__()
        freqs = torch.logspace(
            math.log10(min_freq),
            math.log10(max_freq),
            num_frequencies,
        )
        if learnable:
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer("freqs", freqs)
        self.output_dim = 2 * num_frequencies

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N] -> [B, N, 2*num_frequencies]
        projected = x.unsqueeze(-1) * self.freqs * (2.0 * math.pi)
        return torch.cat([projected.sin(), projected.cos()], dim=-1)


class PeakFeatureEmbedder(nn.Module):
    """Embeds raw peak features (mz, intensity, precursor) into model dim."""

    def __init__(
        self,
        model_dim: int,
        hidden_dim: int,
        mz_fourier_num_frequencies: int = 32,
        mz_fourier_min_freq: float = 1.0,
        mz_fourier_max_freq: float = 100.0,
        mz_fourier_learnable: bool = False,
    ):
        super().__init__()
        self.mz_fourier = FourierFeatures(
            num_frequencies=mz_fourier_num_frequencies,
            min_freq=mz_fourier_min_freq,
            max_freq=mz_fourier_max_freq,
            learnable=mz_fourier_learnable,
        )
        # input: fourier(mz) + raw_mz + intensity + precursor
        input_dim = self.mz_fourier.output_dim + 3
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, model_dim),
        )
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        peak_mz: torch.Tensor,
        peak_intensity: torch.Tensor,
        precursor_mz: torch.Tensor,
    ) -> torch.Tensor:
        precursor = precursor_mz.unsqueeze(1).expand_as(peak_mz)
        mz_fourier = self.mz_fourier(peak_mz)
        scalars = torch.stack([peak_mz, peak_intensity, precursor], dim=-1)
        features = torch.cat([mz_fourier, scalars], dim=-1)
        return self.mlp(features)


def _build_non_causal_blocks(
    *,
    dim: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int | None,
    attention_mlp_multiple: float,
    norm_eps: float = 1e-5,
) -> nn.ModuleList:
    heads = int(num_heads)
    kv_heads = heads if num_kv_heads is None else int(num_kv_heads)
    hidden_dim = int(math.ceil(dim * attention_mlp_multiple))
    blocks: list[transformer_torch.TransformerBlock] = []
    for _ in range(num_layers):
        blocks.append(
            transformer_torch.TransformerBlock(
                dim=dim,
                n_heads=heads,
                n_kv_heads=kv_heads,
                causal=False,
                norm_eps=norm_eps,
                mlp_type="swish",
                multiple_of=4,
                hidden_dim=hidden_dim,
                w_init_scale=1.0,
                use_rotary_embeddings=False,
            )
        )
    return nn.ModuleList(blocks)


class PeakSetEncoder(nn.Module):
    """Transformer encoder for peak sets (non-causal, no positional encoding)."""

    def __init__(
        self,
        *,
        model_dim: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        attention_mlp_multiple: float = 4.0,
        feature_mlp_hidden_dim: int = 128,
        mz_fourier_num_frequencies: int = 32,
        mz_fourier_min_freq: float = 1.0,
        mz_fourier_max_freq: float = 100.0,
        mz_fourier_learnable: bool = False,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.embedder = PeakFeatureEmbedder(
            model_dim,
            feature_mlp_hidden_dim,
            mz_fourier_num_frequencies=mz_fourier_num_frequencies,
            mz_fourier_min_freq=mz_fourier_min_freq,
            mz_fourier_max_freq=mz_fourier_max_freq,
            mz_fourier_learnable=mz_fourier_learnable,
        )
        self.blocks = _build_non_causal_blocks(
            dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            attention_mlp_multiple=attention_mlp_multiple,
        )
        self.norm = nn.RMSNorm(model_dim, eps=1e-5)

    def forward(
        self,
        peak_mz: torch.Tensor,
        peak_intensity: torch.Tensor,
        precursor_mz: torch.Tensor,
    ) -> torch.Tensor:
        x = self.embedder(peak_mz, peak_intensity, precursor_mz)
        for block in self.blocks:
            x = block(x, freqs_cos=None, freqs_sin=None)
        return self.norm(x)


class PeakSetSIGReg(nn.Module):
    """Peak-set strict SIGReg model with two-view augmentation."""

    def __init__(
        self,
        *,
        num_peaks: int = 60,
        model_dim: int = 768,
        encoder_num_layers: int = 20,
        encoder_num_heads: int = 12,
        encoder_num_kv_heads: int | None = None,
        attention_mlp_multiple: float = 4.0,
        feature_mlp_hidden_dim: int = 128,
        mz_fourier_num_frequencies: int = 32,
        mz_fourier_min_freq: float = 1.0,
        mz_fourier_max_freq: float = 100.0,
        mz_fourier_learnable: bool = False,
        sigreg_use_projector: bool = True,
        sigreg_proj_hidden_dim: int = 2048,
        sigreg_proj_output_dim: int = 128,
        bcs_num_slices: int = 256,
        sigreg_lambda: float = 10.0,
        sigreg_drop_prob: float = 0.20,
        sigreg_mz_jitter_std: float = 0.005,
        sigreg_intensity_jitter_std: float = 0.05,
    ):
        super().__init__()
        self.num_peaks = num_peaks
        self.model_dim = model_dim
        self.sigreg_dim = sigreg_proj_output_dim if sigreg_use_projector else model_dim

        self.sigreg_drop_prob = sigreg_drop_prob
        self.sigreg_mz_jitter_std = sigreg_mz_jitter_std
        self.sigreg_intensity_jitter_std = sigreg_intensity_jitter_std

        self.encoder = PeakSetEncoder(
            model_dim=model_dim,
            num_layers=encoder_num_layers,
            num_heads=encoder_num_heads,
            num_kv_heads=encoder_num_kv_heads,
            attention_mlp_multiple=attention_mlp_multiple,
            feature_mlp_hidden_dim=feature_mlp_hidden_dim,
            mz_fourier_num_frequencies=mz_fourier_num_frequencies,
            mz_fourier_min_freq=mz_fourier_min_freq,
            mz_fourier_max_freq=mz_fourier_max_freq,
            mz_fourier_learnable=mz_fourier_learnable,
        )

        if sigreg_use_projector:
            self.projector = nn.Sequential(
                nn.Linear(model_dim, sigreg_proj_hidden_dim),
                nn.RMSNorm(sigreg_proj_hidden_dim),
                nn.SiLU(),
                nn.Linear(sigreg_proj_hidden_dim, sigreg_proj_hidden_dim),
                nn.RMSNorm(sigreg_proj_hidden_dim),
                nn.SiLU(),
                nn.Linear(sigreg_proj_hidden_dim, sigreg_proj_output_dim),
            )
        else:
            self.projector = nn.Identity()

        self.sigreg_loss = BCSLoss(num_slices=bcs_num_slices, lmbd=sigreg_lambda)

    def _masked_mean_pool(
        self,
        embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = valid_mask.unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (embeddings * mask).sum(dim=1) / denom

    def _augment_view(
        self,
        peak_mz: torch.Tensor,
        peak_intensity: torch.Tensor,
        peak_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        drop = (torch.rand_like(peak_mz) < self.sigreg_drop_prob) & peak_valid_mask
        view_valid = peak_valid_mask & (~drop)

        mz = peak_mz + torch.randn_like(peak_mz) * self.sigreg_mz_jitter_std
        mz = torch.clamp(mz, min=0.0, max=1.0)

        intensity = peak_intensity + torch.randn_like(peak_intensity) * self.sigreg_intensity_jitter_std
        intensity = torch.clamp(intensity, min=0.0, max=1.0)

        mz = torch.where(view_valid, mz, torch.zeros_like(mz))
        intensity = torch.where(view_valid, intensity, torch.zeros_like(intensity))

        max_intensity = intensity.max(dim=1, keepdim=True).values.clamp(min=1e-6)
        intensity = intensity / max_intensity
        intensity = torch.where(view_valid, intensity, torch.zeros_like(intensity))

        return mz, intensity, view_valid

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        train: bool = True,
        bcs_projection: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del train
        peak_mz = batch["peak_mz"]
        peak_intensity = batch["peak_intensity"]
        peak_valid_mask = batch["peak_valid_mask"]
        precursor_mz = batch["precursor_mz"]

        view1_mz, view1_int, view1_valid = self._augment_view(
            peak_mz,
            peak_intensity,
            peak_valid_mask,
        )
        view2_mz, view2_int, view2_valid = self._augment_view(
            peak_mz,
            peak_intensity,
            peak_valid_mask,
        )

        # Fuse both views into a single encoder + projector pass [2B, N, ...]
        fused_mz = torch.cat([view1_mz, view2_mz], dim=0)
        fused_int = torch.cat([view1_int, view2_int], dim=0)
        fused_precursor = torch.cat([precursor_mz, precursor_mz], dim=0)
        fused_valid = torch.cat([view1_valid, view2_valid], dim=0)

        fused_emb = self.encoder(fused_mz, fused_int, fused_precursor)
        fused_pooled = self._masked_mean_pool(fused_emb, fused_valid)
        fused_z = self.projector(fused_pooled)
        z1, z2 = fused_z.chunk(2, dim=0)

        loss_dict = self.sigreg_loss(z1, z2, proj=bcs_projection)
        valid_fraction = fused_valid.float().mean()

        return {
            "loss": loss_dict["loss"],
            "bcs_loss": loss_dict["bcs_loss"],
            "invariance_loss": loss_dict["invariance_loss"],
            "valid_fraction": valid_fraction,
        }

    def encode(
        self,
        batch: dict[str, torch.Tensor],
        *,
        train: bool = False,
    ) -> torch.Tensor:
        del train
        peak_mz = batch["peak_mz"]
        peak_intensity = batch["peak_intensity"]
        peak_valid_mask = batch["peak_valid_mask"]
        precursor_mz = batch["precursor_mz"]

        embeddings = self.encoder(peak_mz, peak_intensity, precursor_mz)
        pooled = self._masked_mean_pool(embeddings, peak_valid_mask)
        return self.projector(pooled)

    def compute_loss(
        self,
        batch: dict[str, torch.Tensor],
        *,
        train: bool = False,
    ):
        metrics = self(batch, train=train)
        return metrics["loss"], metrics

    def sample_bcs_projection(
        self,
        *,
        device: torch.device,
        seed: int | None = None,
    ) -> torch.Tensor:
        return self.sigreg_loss.sample_projection(
            self.sigreg_dim,
            device=device,
            seed=seed,
        )


PeakSetJEPA = PeakSetSIGReg
