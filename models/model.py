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
from networks import set_transformer_torch, transformer_torch
from networks.transformer_torch import create_padding_block_mask


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
        # Compute in fp32: high frequencies (up to 50k) produce products ~314k
        # which overflow fp16 (max 65504) and lose all precision in bf16
        # (7-bit mantissa → quantization step ~2048 at that magnitude).
        projected = x.float().unsqueeze(-1) * self.freqs.float() * (2.0 * math.pi)
        return torch.cat([projected.sin(), projected.cos()], dim=-1).to(x.dtype)


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
        # input: fourier(mz) + raw_mz + intensity
        input_dim = self.mz_fourier.output_dim + 1 + 1
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
    ) -> torch.Tensor:
        mz_fourier = self.mz_fourier(peak_mz)
        features = torch.cat([mz_fourier, peak_mz.unsqueeze(-1), peak_intensity.unsqueeze(-1)], dim=-1)
        return self.mlp(features)


def _build_non_causal_blocks(
    *,
    dim: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int | None,
    attention_mlp_multiple: float,
    use_rope: bool = False,
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
                use_rotary_embeddings=use_rope,
            )
        )
    return nn.ModuleList(blocks)


def _build_isab_blocks(
    *,
    dim: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int | None,
    attention_mlp_multiple: float,
    num_inducing_points: int = 32,
    norm_eps: float = 1e-5,
) -> nn.ModuleList:
    heads = int(num_heads)
    kv_heads = heads if num_kv_heads is None else int(num_kv_heads)
    blocks: list[set_transformer_torch.ISAB] = []
    for _ in range(num_layers):
        blocks.append(
            set_transformer_torch.ISAB(
                dim=dim,
                num_inducing_points=num_inducing_points,
                n_heads=heads,
                n_kv_heads=kv_heads,
                attention_mlp_multiple=attention_mlp_multiple,
                norm_eps=norm_eps,
            )
        )
    return nn.ModuleList(blocks)


def _build_rope_frequencies(
    *,
    sequence_length: int,
    inv_freq: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(sequence_length, device=inv_freq.device, dtype=inv_freq.dtype)
    angles = torch.outer(positions, inv_freq)
    angles = torch.repeat_interleave(angles, repeats=2, dim=-1)
    freqs_cos = angles.cos().to(dtype=dtype).unsqueeze(0).unsqueeze(2)
    freqs_sin = angles.sin().to(dtype=dtype).unsqueeze(0).unsqueeze(2)
    return freqs_cos, freqs_sin


class PeakSetEncoder(nn.Module):
    """Transformer encoder for peak sets (non-causal, optional RoPE)."""

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
        use_rope: bool = False,
        fp16_high_precision_stem: bool = False,
        encoder_block_type: str = "transformer",
        isab_num_inducing_points: int = 32,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.use_rope = bool(use_rope)
        self.fp16_high_precision_stem = bool(fp16_high_precision_stem)
        self.encoder_block_type = encoder_block_type
        self.embedder = PeakFeatureEmbedder(
            model_dim,
            feature_mlp_hidden_dim,
            mz_fourier_num_frequencies=mz_fourier_num_frequencies,
            mz_fourier_min_freq=mz_fourier_min_freq,
            mz_fourier_max_freq=mz_fourier_max_freq,
            mz_fourier_learnable=mz_fourier_learnable,
        )
        head_dim = model_dim // int(num_heads)
        inv_freq = 1.0 / (
            10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / float(head_dim))
        )
        self.rope_inv_freq: torch.Tensor
        self.register_buffer("rope_inv_freq", inv_freq, persistent=False)
        if self.encoder_block_type == "isab":
            self.blocks = _build_isab_blocks(
                dim=model_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                attention_mlp_multiple=attention_mlp_multiple,
                num_inducing_points=isab_num_inducing_points,
            )
        else:
            self.blocks = _build_non_causal_blocks(
                dim=model_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                attention_mlp_multiple=attention_mlp_multiple,
                use_rope=self.use_rope,
            )
        self.norm = nn.RMSNorm(model_dim, eps=1e-5)

    def forward(
        self,
        peak_mz: torch.Tensor,
        peak_intensity: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device_type = peak_mz.device.type

        if self.encoder_block_type == "isab":
            x = self.embedder(peak_mz, peak_intensity)
            kv_block_mask = None
            q_block_mask = None
            if valid_mask is not None:
                m = self.blocks[0].inducing_points.shape[0]
                kv_block_mask = set_transformer_torch.create_kv_padding_block_mask(valid_mask, q_len=m)
                q_block_mask = set_transformer_torch.create_q_padding_block_mask(valid_mask, kv_len=m)
            for block in self.blocks:
                x = block(x, kv_block_mask=kv_block_mask, q_block_mask=q_block_mask)
            return self.norm(x)

        autocast_dtype = None
        run_high_precision_stem = False
        if self.fp16_high_precision_stem and self.embedder.mlp[0].weight.dtype == torch.float32:
            if torch.is_autocast_enabled(device_type):
                autocast_dtype = torch.get_autocast_dtype(device_type)
            run_high_precision_stem = (
                peak_mz.dtype == torch.float16
                or peak_intensity.dtype == torch.float16
                or autocast_dtype == torch.float16
            )

        block_mask = None
        start_block = 0
        if run_high_precision_stem:
            with torch.autocast(device_type=device_type, enabled=False):
                x = self.embedder(peak_mz.float(), peak_intensity.float())
                stem_freqs_cos = None
                stem_freqs_sin = None
                if self.use_rope:
                    stem_freqs_cos, stem_freqs_sin = _build_rope_frequencies(
                        sequence_length=x.shape[1],
                        inv_freq=self.rope_inv_freq,
                        dtype=torch.float32,
                    )
                if valid_mask is not None:
                    block_mask = create_padding_block_mask(valid_mask)
                if len(self.blocks) > 0:
                    x = self.blocks[0](
                        x,
                        freqs_cos=stem_freqs_cos,
                        freqs_sin=stem_freqs_sin,
                        block_mask=block_mask,
                    )
                    start_block = 1
            target_dtype = autocast_dtype if autocast_dtype is not None else peak_mz.dtype
            x = x.to(dtype=target_dtype)
        else:
            x = self.embedder(peak_mz, peak_intensity)
            if valid_mask is not None:
                block_mask = create_padding_block_mask(valid_mask)

        freqs_cos = None
        freqs_sin = None
        if self.use_rope:
            freqs_cos, freqs_sin = _build_rope_frequencies(
                sequence_length=x.shape[1],
                inv_freq=self.rope_inv_freq,
                dtype=x.dtype,
            )

        for block in self.blocks[start_block:]:
            x = block(x, freqs_cos=freqs_cos, freqs_sin=freqs_sin, block_mask=block_mask)
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
        encoder_use_rope: bool = False,
        encoder_fp16_high_precision_stem: bool = False,
        sigreg_use_projector: bool = True,
        sigreg_proj_hidden_dim: int = 2048,
        sigreg_proj_output_dim: int = 128,
        bcs_num_slices: int = 256,
        sigreg_lambda: float = 10.0,
        sigreg_contiguous_mask_fraction: float = 0.25,
        sigreg_contiguous_mask_min_len: int = 1,
        sigreg_random_mask_prob: float = 0.0,
        sigreg_mz_jitter_std: float = 0.005,
        sigreg_intensity_jitter_std: float = 0.05,
        pooling_type: str = "pma",
        pma_fp16_high_precision: bool = False,
        pma_num_heads: int | None = None,
        pma_num_seeds: int = 1,
        encoder_block_type: str = "transformer",
        isab_num_inducing_points: int = 32,
    ):
        super().__init__()
        self.num_peaks = num_peaks
        self.model_dim = model_dim
        self.sigreg_dim = sigreg_proj_output_dim if sigreg_use_projector else model_dim

        self.sigreg_contiguous_mask_fraction = sigreg_contiguous_mask_fraction
        self.sigreg_contiguous_mask_min_len = int(sigreg_contiguous_mask_min_len)
        self.sigreg_random_mask_prob = sigreg_random_mask_prob
        self.sigreg_mz_jitter_std = sigreg_mz_jitter_std
        self.sigreg_intensity_jitter_std = sigreg_intensity_jitter_std
        self.pooling_type = pooling_type
        self.pma_fp16_high_precision = bool(pma_fp16_high_precision)
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
            use_rope=encoder_use_rope,
            fp16_high_precision_stem=encoder_fp16_high_precision_stem,
            encoder_block_type=encoder_block_type,
            isab_num_inducing_points=isab_num_inducing_points,
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
        self.pool_norm = nn.RMSNorm(model_dim, eps=1e-5)

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
            device_type = embeddings.device.type
            autocast_dtype = None
            run_high_precision_pma = False
            if self.pma_fp16_high_precision and self.pool_mha.in_proj_weight.dtype == torch.float32:
                if torch.is_autocast_enabled(device_type):
                    autocast_dtype = torch.get_autocast_dtype(device_type)
                run_high_precision_pma = embeddings.dtype == torch.float16 or autocast_dtype == torch.float16

            if run_high_precision_pma:
                with torch.autocast(device_type=device_type, enabled=False):
                    query = self.pool_query.float().unsqueeze(0).expand(embeddings.shape[0], -1, -1)
                    pooled, _ = self.pool_mha(
                        query=query,
                        key=embeddings.float(),
                        value=embeddings.float(),
                        key_padding_mask=~valid_mask,
                        need_weights=False,
                    )
                    pooled = self.pool_norm(pooled.mean(dim=1))
                target_dtype = autocast_dtype if autocast_dtype is not None else embeddings.dtype
                return pooled.to(dtype=target_dtype)

            query = self.pool_query.unsqueeze(0).expand(embeddings.shape[0], -1, -1)
            pooled, _ = self.pool_mha(
                query=query,
                key=embeddings,
                value=embeddings,
                key_padding_mask=~valid_mask,
                need_weights=False,
            )
            return self.pool_norm(pooled.mean(dim=1))
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
    ]:
        return augment_masked_view(
            peak_mz,
            peak_intensity,
            peak_valid_mask,
            contiguous_mask_fraction=self.sigreg_contiguous_mask_fraction,
            contiguous_mask_min_len=self.sigreg_contiguous_mask_min_len,
            random_mask_prob=self.sigreg_random_mask_prob,
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
            random_mask_prob=self.sigreg_random_mask_prob,
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
        fused_valid_mask = augmented_batch["fused_valid_mask"]
        masked_fraction = augmented_batch["view1_masked_fraction"]

        fused_emb = self.encoder(
            fused_mz,
            fused_intensity,
            valid_mask=fused_valid_mask,
        )
        fused_pooled = self.pool(fused_emb, fused_valid_mask)
        fused_z = self.projector(fused_pooled)
        z1, z2 = fused_z.chunk(2, dim=0)
        representation_variance = fused_z.var(dim=0, unbiased=False).mean()
        encoder_variance = fused_pooled.var(dim=0, unbiased=False).mean()

        # Alignment: mean cosine similarity between paired views (higher = better)
        z1_norm = nn.functional.normalize(z1, dim=-1)
        z2_norm = nn.functional.normalize(z2, dim=-1)
        alignment = (z1_norm * z2_norm).sum(dim=-1).mean()
        # Uniformity: mean cosine similarity between unpaired samples (lower = better)
        uniformity = (z1_norm @ z2_norm.T).fill_diagonal_(0).sum() / (z1.shape[0] * (z1.shape[0] - 1))

        loss_dict = self.sigreg_loss(z1, z2, proj=bcs_projection)
        valid_fraction = fused_valid_mask.float().mean()

        return {
            "loss": loss_dict["loss"],
            "bcs_loss": loss_dict["bcs_loss"],
            "invariance_loss": loss_dict["invariance_loss"],
            "valid_fraction": valid_fraction,
            "masked_fraction": masked_fraction,
            "representation_variance": representation_variance,
            "encoder_variance": encoder_variance,
            "alignment": alignment,
            "uniformity": uniformity,
        }

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        train: bool = True,
        bcs_projection: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        augmented_batch = self.augment_batch(batch)
        return self.forward_augmented(augmented_batch, bcs_projection=bcs_projection)

    def encode(
        self,
        batch: dict[str, torch.Tensor],
        *,
        train: bool = False,
    ) -> torch.Tensor:
        peak_mz = batch["peak_mz"]
        peak_intensity = batch["peak_intensity"]
        peak_valid_mask = batch["peak_valid_mask"]

        embeddings = self.encoder(peak_mz, peak_intensity, valid_mask=peak_valid_mask)
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
