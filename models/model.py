"""Peak-set SIGReg model for mass spectrometry pretraining.

Architecture:
- PeakSetEncoder: MLP embedder -> non-causal Transformer -> peak embeddings
- Two-view spectrum augmentation (one masked view + one unmasked jittered view)
- Optional projector on pooled embeddings
- SIGReg objective: MSE(z1, z2) + lambda * BCS(z1, z2)
"""

from __future__ import annotations

import math

import torch
from torch import nn

from models.augmentation import (
    augment_masked_view,
    augment_sigreg_batch,
    augment_unmasked_view,
)
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
        valid_mask: torch.Tensor | None = None,
        masked_positions: torch.Tensor | None = None,
        mask_token: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embedder(peak_mz, peak_intensity, precursor_mz)
        if masked_positions is not None:
            token = mask_token.view(1, 1, -1).to(dtype=x.dtype, device=x.device)
            x = torch.where(masked_positions.unsqueeze(-1), token, x)
        for block in self.blocks:
            x = block(x, freqs_cos=None, freqs_sin=None, attention_mask=valid_mask)
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
        sigreg_contiguous_mask_fraction: float = 0.25,
        sigreg_contiguous_mask_min_len: int = 1,
        sigreg_mz_jitter_std: float = 0.005,
        sigreg_intensity_jitter_std: float = 0.05,
        pooling_type: str = "pma",
        pma_num_heads: int | None = None,
        pma_num_seeds: int = 1,
    ):
        super().__init__()
        self.num_peaks = num_peaks
        self.model_dim = model_dim
        self.sigreg_dim = sigreg_proj_output_dim if sigreg_use_projector else model_dim

        self.sigreg_contiguous_mask_fraction = sigreg_contiguous_mask_fraction
        self.sigreg_contiguous_mask_min_len = int(sigreg_contiguous_mask_min_len)
        self.sigreg_mz_jitter_std = sigreg_mz_jitter_std
        self.sigreg_intensity_jitter_std = sigreg_intensity_jitter_std
        self.pooling_type = pooling_type
        self.pma_num_seeds = int(pma_num_seeds)

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

        pma_heads = int(encoder_num_heads) if pma_num_heads is None else int(pma_num_heads)
        self.pool_query = nn.Parameter(torch.empty(self.pma_num_seeds, model_dim))
        nn.init.xavier_normal_(self.pool_query)
        self.pool_mha = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=pma_heads,
            batch_first=True,
        )
        self.mask_token = nn.Parameter(torch.empty(model_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.sigreg_loss = BCSLoss(num_slices=bcs_num_slices, lmbd=sigreg_lambda)

    def _mean_pool(
        self,
        embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = valid_mask.unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (embeddings * mask).sum(dim=1) / denom

    def pool(
        self,
        embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.pooling_type == "mean":
            return self._mean_pool(embeddings, valid_mask)
        if self.pooling_type == "pma":
            query = self.pool_query.unsqueeze(0).expand(embeddings.shape[0], -1, -1)
            pooled, _ = self.pool_mha(
                query=query,
                key=embeddings,
                value=embeddings,
                key_padding_mask=~valid_mask,
                need_weights=False,
            )
            return pooled.mean(dim=1)
        raise NotImplementedError(f"Unknown pooling type: {self.pooling_type}")

    def _augment_view(
        self,
        peak_mz: torch.Tensor,
        peak_intensity: torch.Tensor,
        peak_valid_mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        return augment_masked_view(
            peak_mz,
            peak_intensity,
            peak_valid_mask,
            contiguous_mask_fraction=self.sigreg_contiguous_mask_fraction,
            contiguous_mask_min_len=self.sigreg_contiguous_mask_min_len,
            mz_jitter_std=self.sigreg_mz_jitter_std,
            intensity_jitter_std=self.sigreg_intensity_jitter_std,
        )

    def _augment_unmasked_view(
        self,
        peak_mz: torch.Tensor,
        peak_intensity: torch.Tensor,
        peak_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return augment_unmasked_view(
            peak_mz,
            peak_intensity,
            peak_valid_mask,
            mz_jitter_std=self.sigreg_mz_jitter_std,
            intensity_jitter_std=self.sigreg_intensity_jitter_std,
        )

    def augment_batch(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return augment_sigreg_batch(
            batch,
            contiguous_mask_fraction=self.sigreg_contiguous_mask_fraction,
            contiguous_mask_min_len=self.sigreg_contiguous_mask_min_len,
            mz_jitter_std=self.sigreg_mz_jitter_std,
            intensity_jitter_std=self.sigreg_intensity_jitter_std,
        )

    def forward_augmented(
        self,
        augmented_batch: dict[str, torch.Tensor],
        *,
        bcs_projection: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        fused_mz = augmented_batch["fused_mz"]
        fused_intensity = augmented_batch["fused_intensity"]
        fused_precursor_mz = augmented_batch["fused_precursor_mz"]
        fused_valid_mask = augmented_batch["fused_valid_mask"]
        fused_masked_positions = augmented_batch["fused_masked_positions"]
        masked_fraction = augmented_batch["view1_masked_fraction"]
        density_interval_fraction = augmented_batch["view1_density_interval_fraction"]

        fused_emb = self.encoder(
            fused_mz,
            fused_intensity,
            fused_precursor_mz,
            valid_mask=fused_valid_mask,
            masked_positions=fused_masked_positions,
            mask_token=self.mask_token,
        )
        fused_pooled = self.pool(fused_emb, fused_valid_mask)
        fused_z = self.projector(fused_pooled)
        z1, z2 = fused_z.chunk(2, dim=0)
        representation_variance = fused_z.var(dim=0, unbiased=False).mean()

        loss_dict = self.sigreg_loss(z1, z2, proj=bcs_projection)
        valid_fraction = fused_valid_mask.float().mean()

        return {
            "loss": loss_dict["loss"],
            "bcs_loss": loss_dict["bcs_loss"],
            "invariance_loss": loss_dict["invariance_loss"],
            "valid_fraction": valid_fraction,
            "masked_fraction": masked_fraction,
            "density_interval_fraction": density_interval_fraction,
            "representation_variance": representation_variance,
        }

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        train: bool = True,
        bcs_projection: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del train
        augmented_batch = self.augment_batch(batch)
        return self.forward_augmented(augmented_batch, bcs_projection=bcs_projection)

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

        embeddings = self.encoder(peak_mz, peak_intensity, precursor_mz, valid_mask=peak_valid_mask)
        pooled = self.pool(embeddings, peak_valid_mask)
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
