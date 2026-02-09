"""Peak-set JEPA model for mass spectrometry pretraining.

Architecture:
- PeakSetEncoder: MLP embedder -> non-causal Transformer -> peak embeddings
- PeakMaskSampler: samples fixed-size context/target index sets
- PeakSetPredictor: predicts target embeddings from context + target queries
- PeakSetJEPA: full JEPA container with prediction + BCS regularization losses
"""

from __future__ import annotations

import math

import torch
from torch import nn

from models.losses import BCSLoss, squared_prediction_loss
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
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

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
        """Returns peak embeddings [B, N, D]."""
        x = self.embedder(peak_mz, peak_intensity, precursor_mz)
        for block in self.blocks:
            x = block(x, freqs_cos=None, freqs_sin=None)
        return self.norm(x)


class PeakMaskSampler(nn.Module):
    """Samples fixed-size context and target index sets for JEPA masking.

    Valid peaks are prioritised; invalid (padded) peaks are pushed to the end
    of the random permutation so they are only selected when there are fewer
    valid peaks than num_context + num_target.
    """

    def __init__(self, num_peaks: int, num_context: int, num_target: int):
        super().__init__()
        self.num_peaks = num_peaks
        self.num_context = num_context
        self.num_target = num_target

    def forward(
        self,
        peak_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (ctx_indices, tgt_indices, ctx_valid, tgt_valid).

        All outputs have static shapes suitable for torch.compile.
        """
        B, N = peak_valid_mask.shape
        device = peak_valid_mask.device

        rand = torch.rand(B, N, device=device)
        rand = rand + (~peak_valid_mask).float() * 2.0
        perm = rand.argsort(dim=1)

        ctx_indices = perm[:, : self.num_context]
        tgt_indices = perm[:, self.num_context : self.num_context + self.num_target]

        ctx_valid = torch.gather(peak_valid_mask, 1, ctx_indices)
        tgt_valid = torch.gather(peak_valid_mask, 1, tgt_indices)

        return ctx_indices, tgt_indices, ctx_valid, tgt_valid


class PeakSetPredictor(nn.Module):
    """Predicts target peak embeddings from context embeddings and target queries.

    Architecture: embed target peak features as query tokens, concatenate with
    context embeddings, apply non-causal self-attention, extract target outputs.
    """

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
        self.query_embedder = PeakFeatureEmbedder(model_dim, feature_mlp_hidden_dim)
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
        context_embeddings: torch.Tensor,
        target_mz: torch.Tensor,
        target_intensity: torch.Tensor,
        precursor_mz: torch.Tensor,
    ) -> torch.Tensor:
        """Returns predicted target embeddings [B, N_tgt, D]."""
        target_queries = self.query_embedder(
            target_mz, target_intensity, precursor_mz
        )
        N_ctx = context_embeddings.shape[1]
        x = torch.cat([context_embeddings, target_queries], dim=1)

        for block in self.blocks:
            x = block(x, freqs_cos=None, freqs_sin=None)
        x = self.norm(x)

        return x[:, N_ctx:, :]


class PeakSetJEPA(nn.Module):
    """Peak-set JEPA model for mass spectrometry pretraining.

    Components:
    1. PeakSetEncoder: encodes all peaks into latent embeddings
    2. PeakMaskSampler: splits peaks into context and target sets
    3. PeakSetPredictor: predicts target embeddings from context + target queries
    4. Losses: MSE prediction loss + BCS (Batched Characteristic Slicing) regularizer
    """

    def __init__(
        self,
        *,
        num_peaks: int = 60,
        model_dim: int = 768,
        encoder_num_layers: int = 20,
        encoder_num_heads: int = 12,
        encoder_num_kv_heads: int | None = None,
        predictor_num_layers: int = 4,
        predictor_num_heads: int = 12,
        predictor_num_kv_heads: int | None = None,
        attention_mlp_multiple: float = 4.0,
        feature_mlp_hidden_dim: int = 128,
        target_ratio: float = 0.4,
        pred_weight: float = 1.0,
        bcs_num_slices: int = 256,
        bcs_lambda: float = 10.0,
    ):
        super().__init__()
        self.num_peaks = num_peaks
        self.model_dim = model_dim
        self.pred_weight = pred_weight

        num_target = int(num_peaks * target_ratio)
        num_context = num_peaks - num_target

        self.encoder = PeakSetEncoder(
            model_dim=model_dim,
            num_layers=encoder_num_layers,
            num_heads=encoder_num_heads,
            num_kv_heads=encoder_num_kv_heads,
            attention_mlp_multiple=attention_mlp_multiple,
            feature_mlp_hidden_dim=feature_mlp_hidden_dim,
        )

        self.mask_sampler = PeakMaskSampler(num_peaks, num_context, num_target)

        self.predictor = PeakSetPredictor(
            model_dim=model_dim,
            num_layers=predictor_num_layers,
            num_heads=predictor_num_heads,
            num_kv_heads=predictor_num_kv_heads,
            attention_mlp_multiple=attention_mlp_multiple,
            feature_mlp_hidden_dim=feature_mlp_hidden_dim,
        )

        self.bcs_loss = BCSLoss(num_slices=bcs_num_slices, lmbd=bcs_lambda)

    def _masked_mean_pool(
        self,
        embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Masked mean pooling over peaks. Returns [B, D]."""
        mask = valid_mask.unsqueeze(-1).float()
        return (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        train: bool = True,
    ) -> dict[str, torch.Tensor]:
        del train
        peak_mz = batch["peak_mz"]
        peak_intensity = batch["peak_intensity"]
        peak_valid_mask = batch["peak_valid_mask"]
        precursor_mz = batch["precursor_mz"]

        # Full encoder pass
        all_embeddings = self.encoder(
            peak_mz, peak_intensity, precursor_mz
        )

        # BCS regularizer on pooled embeddings (static shape [B, D])
        pooled = self._masked_mean_pool(all_embeddings, peak_valid_mask)
        bcs_loss = self.bcs_loss(pooled)

        # Sample context / target masks
        ctx_idx, tgt_idx, ctx_valid, tgt_valid = self.mask_sampler(peak_valid_mask)

        # Gather context and target embeddings
        expand_d = ctx_idx.unsqueeze(-1).expand(-1, -1, self.model_dim)
        ctx_embeddings = torch.gather(all_embeddings, 1, expand_d)

        expand_t = tgt_idx.unsqueeze(-1).expand(-1, -1, self.model_dim)
        tgt_embeddings = torch.gather(all_embeddings, 1, expand_t).detach()

        # Gather target peak features for predictor queries
        tgt_mz = torch.gather(peak_mz, 1, tgt_idx)
        tgt_intensity = torch.gather(peak_intensity, 1, tgt_idx)

        # Predict target embeddings
        pred_tgt = self.predictor(
            ctx_embeddings, tgt_mz, tgt_intensity, precursor_mz
        )

        # Prediction loss
        pred_loss = squared_prediction_loss(pred_tgt, tgt_embeddings, tgt_valid)

        loss = self.pred_weight * pred_loss + bcs_loss

        return {
            "loss": loss,
            "pred_loss": pred_loss,
            "bcs_loss": bcs_loss,
            "target_valid_fraction": peak_valid_mask.float().mean(),
        }

    def encode(
        self,
        batch: dict[str, torch.Tensor],
        *,
        train: bool = False,
    ) -> torch.Tensor:
        """Returns pooled spectrum embedding [B, D] for downstream tasks."""
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
