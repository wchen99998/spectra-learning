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


class PeakFeatureEmbedder(nn.Module):
    """Embeds raw peak features (mz, intensity, precursor) into model dim."""

    def __init__(self, model_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
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
        features = torch.stack([peak_mz, peak_intensity, precursor], dim=-1)
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
    ):
        super().__init__()
        self.model_dim = model_dim
        self.embedder = PeakFeatureEmbedder(model_dim, feature_mlp_hidden_dim)
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
        )

        if sigreg_use_projector:
            self.projector = nn.Sequential(
                nn.Linear(model_dim, sigreg_proj_hidden_dim),
                nn.BatchNorm1d(sigreg_proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(sigreg_proj_hidden_dim, sigreg_proj_hidden_dim),
                nn.BatchNorm1d(sigreg_proj_hidden_dim),
                nn.ReLU(),
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

        emb1 = self.encoder(view1_mz, view1_int, precursor_mz)
        emb2 = self.encoder(view2_mz, view2_int, precursor_mz)

        pooled1 = self._masked_mean_pool(emb1, view1_valid)
        pooled2 = self._masked_mean_pool(emb2, view2_valid)

        z1 = self.projector(pooled1)
        z2 = self.projector(pooled2)

        loss_dict = self.sigreg_loss(z1, z2, proj=bcs_projection)
        valid_fraction = (view1_valid.float().mean() + view2_valid.float().mean()) / 2.0

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
        return self._masked_mean_pool(embeddings, peak_valid_mask)

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
