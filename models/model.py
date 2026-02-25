"""Peak-set SIGReg model for mass spectrometry pretraining.

Architecture:
- PeakSetEncoder: MLP embedder -> non-causal Transformer -> peak embeddings
- Multi-crop augmentation (2 global + 6 local views with random peak retention)
- Optional projector on pooled embeddings
- Objective: SIGReg Gaussianity regularizer + optional masked latent prediction
"""

from __future__ import annotations

import math

import torch
from torch import nn

from models.losses import SIGReg
from networks import set_transformer_torch, transformer_torch
from networks.transformer_torch import (
    create_masked_context_block_mask,
    create_padding_block_mask,
)


class PeakFeatureEmbedder(nn.Module):
    """Embeds raw peak features into model dim."""

    def __init__(
        self,
        model_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        input_dim = 3
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
        prec = precursor_mz.reshape(-1, 1)
        # neutral_loss = torch.clamp(prec - peak_mz, min=0.0)
        log_intensity = torch.log1p(peak_intensity)
        features = torch.cat(
            [
                peak_mz.unsqueeze(-1),
                # neutral_loss.unsqueeze(-1),
                peak_intensity.unsqueeze(-1),
                log_intensity.unsqueeze(-1),
            ],
            dim=-1,
        )
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
    qk_norm: bool = False,
    post_norm: bool = False,
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
                qk_norm=qk_norm,
                post_norm=post_norm,
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


def _build_mass_rope_omega(
    *,
    head_dim: int,
    mz_max: float,
    mz_precision: float,
    device: torch.device,
) -> torch.Tensor:
    half_dim = head_dim // 2
    lambda_min = 2.0 * float(mz_precision)
    lambda_max = float(mz_max)
    lambdas = torch.logspace(
        math.log10(lambda_min),
        math.log10(lambda_max),
        steps=half_dim,
        device=device,
        dtype=torch.float32,
    )
    return (2.0 * math.pi) / lambdas


def _build_rope_freqs_from_positions(
    *,
    positions_da: torch.Tensor,
    omega: torch.Tensor,
    out_dtype: torch.dtype,
    modulo_2pi: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    angles = positions_da.float().unsqueeze(-1) * omega.view(1, 1, -1)
    if modulo_2pi:
        angles = torch.remainder(angles, 2.0 * math.pi)
    angles = torch.repeat_interleave(angles, repeats=2, dim=-1)
    freqs_cos = angles.cos().to(dtype=out_dtype).unsqueeze(2)
    freqs_sin = angles.sin().to(dtype=out_dtype).unsqueeze(2)
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
        use_rope: bool = False,
        rope_mz_max: float = 1000.0,
        rope_mz_precision: float = 0.1,
        rope_complement_heads: int | None = None,
        rope_modulo_2pi: bool = True,
        num_peaks: int = 60,
        masked_token_position_mode: str = "index",
        masked_token_attention_mode: str = "bidirectional",
        fp16_high_precision_stem: bool = False,
        encoder_block_type: str = "transformer",
        isab_num_inducing_points: int = 32,
        qk_norm: bool = False,
        post_norm: bool = False,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.use_rope = bool(use_rope)
        self.fp16_high_precision_stem = bool(fp16_high_precision_stem)
        self.encoder_block_type = encoder_block_type
        self.embedder = PeakFeatureEmbedder(model_dim, feature_mlp_hidden_dim)
        self.rope_mz_max = float(rope_mz_max)
        self.rope_mz_precision = float(rope_mz_precision)
        self.rope_modulo_2pi = bool(rope_modulo_2pi)
        self.num_peaks = int(num_peaks)
        self.masked_token_position_mode = str(masked_token_position_mode).lower()
        if self.masked_token_position_mode not in {"mz", "index"}:
            raise ValueError(
                f"Unknown masked_token_position_mode: {self.masked_token_position_mode!r}"
            )
        self.masked_token_attention_mode = str(masked_token_attention_mode).lower()
        if self.masked_token_attention_mode not in {"bidirectional", "masked_query_to_unmasked_kv"}:
            raise ValueError(
                f"Unknown masked_token_attention_mode: {self.masked_token_attention_mode!r}"
            )
        self.masked_index_embedding = None
        if self.masked_token_position_mode == "index":
            self.masked_index_embedding = nn.Embedding(self.num_peaks, model_dim)
            nn.init.normal_(self.masked_index_embedding.weight, std=0.02)
        heads = int(num_heads)
        complement_heads = heads // 2 if rope_complement_heads is None else int(rope_complement_heads)
        complement_heads = max(0, min(complement_heads, heads))
        self.rope_mass_heads = heads - complement_heads
        head_dim = model_dim // heads
        omega = _build_mass_rope_omega(
            head_dim=head_dim,
            mz_max=self.rope_mz_max,
            mz_precision=self.rope_mz_precision,
            device=torch.device("cpu"),
        )
        self.register_buffer("rope_omega", omega, persistent=False)
        if self.encoder_block_type == "isab":
            if self.use_rope:
                raise ValueError("mass-aware RoPE is not implemented for encoder_block_type='isab'")
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
                qk_norm=qk_norm,
                post_norm=post_norm,
            )
        self.final_norm = nn.RMSNorm(model_dim, eps=1e-5)

    def forward(
        self,
        peak_mz: torch.Tensor,
        peak_intensity: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        precursor_mz: torch.Tensor | None = None,
        masked_positions: torch.Tensor | None = None,
        mask_token: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device_type = peak_mz.device.type
        index_positions_da = None
        if masked_positions is not None and self.masked_token_position_mode == "index":
            base_positions = torch.linspace(
                0.0,
                self.rope_mz_max,
                steps=peak_mz.shape[1],
                device=peak_mz.device,
                dtype=torch.float32,
            )
            index_positions_da = base_positions.unsqueeze(0).expand(peak_mz.shape[0], -1)

        def _apply_mask_token(x_in: torch.Tensor) -> torch.Tensor:
            if masked_positions is None or mask_token is None:
                return x_in
            token = mask_token.view(1, 1, -1).to(dtype=x_in.dtype, device=x_in.device)
            x_out = torch.where(masked_positions.unsqueeze(-1), token, x_in)
            if self.masked_index_embedding is not None:
                index_ids = torch.arange(
                    x_in.shape[1], device=x_in.device, dtype=torch.long
                ).unsqueeze(0).expand(x_in.shape[0], -1)
                index_embed = self.masked_index_embedding(index_ids).to(dtype=x_in.dtype)
                x_out = x_out + torch.where(
                    masked_positions.unsqueeze(-1),
                    index_embed,
                    torch.zeros_like(index_embed),
                )
            return x_out

        if self.encoder_block_type == "isab":
            x = self.embedder(peak_mz, peak_intensity, precursor_mz)
            x = _apply_mask_token(x)
            kv_block_mask = None
            q_block_mask = None
            if valid_mask is not None:
                m = self.blocks[0].inducing_points.shape[0]
                kv_block_mask = set_transformer_torch.create_kv_padding_block_mask(valid_mask, q_len=m)
                q_block_mask = set_transformer_torch.create_q_padding_block_mask(valid_mask, kv_len=m)
            for block in self.blocks:
                x = block(x, kv_block_mask=kv_block_mask, q_block_mask=q_block_mask)
            return self.final_norm(x)

        block_mask = None
        if valid_mask is not None:
            if (
                masked_positions is not None
                and self.masked_token_attention_mode == "masked_query_to_unmasked_kv"
            ):
                block_mask = create_masked_context_block_mask(valid_mask, masked_positions)
            else:
                block_mask = create_padding_block_mask(valid_mask)

        def _compute_rope(dtype: torch.dtype) -> tuple[
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
        ]:
            if not self.use_rope:
                return None, None, None, None

            mz_da = peak_mz.float() * self.rope_mz_max
            prec_da = precursor_mz.float().reshape(-1) * self.rope_mz_max
            neutral_loss_da = torch.clamp(prec_da.unsqueeze(1) - mz_da, min=0.0)
            if masked_positions is not None and index_positions_da is not None:
                mz_da = torch.where(masked_positions, index_positions_da, mz_da)
                neutral_loss_da = torch.where(
                    masked_positions,
                    index_positions_da,
                    neutral_loss_da,
                )
            omega = self.rope_omega
            if omega.device != peak_mz.device:
                omega = omega.to(device=peak_mz.device)

            freqs_cos, freqs_sin = _build_rope_freqs_from_positions(
                positions_da=mz_da,
                omega=omega,
                out_dtype=dtype,
                modulo_2pi=self.rope_modulo_2pi,
            )
            freqs_cos_q, freqs_sin_q = _build_rope_freqs_from_positions(
                positions_da=neutral_loss_da,
                omega=omega,
                out_dtype=dtype,
                modulo_2pi=self.rope_modulo_2pi,
            )
            return freqs_cos, freqs_sin, freqs_cos_q, freqs_sin_q

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

        start_block = 0
        if run_high_precision_stem:
            with torch.autocast(device_type=device_type, enabled=False):
                x = self.embedder(peak_mz.float(), peak_intensity.float(), precursor_mz.float())
                x = _apply_mask_token(x)
                stem_freqs_cos, stem_freqs_sin, stem_freqs_cos_q, stem_freqs_sin_q = _compute_rope(torch.float32)
                if len(self.blocks) > 0:
                    x = self.blocks[0](
                        x,
                        freqs_cos=stem_freqs_cos,
                        freqs_sin=stem_freqs_sin,
                        freqs_cos_q=stem_freqs_cos_q,
                        freqs_sin_q=stem_freqs_sin_q,
                        q_rope_head_split=self.rope_mass_heads,
                        block_mask=block_mask,
                    )
                    start_block = 1
            target_dtype = autocast_dtype if autocast_dtype is not None else peak_mz.dtype
            x = x.to(dtype=target_dtype)
        else:
            x = self.embedder(peak_mz, peak_intensity, precursor_mz)
            x = _apply_mask_token(x)
        freqs_cos, freqs_sin, freqs_cos_q, freqs_sin_q = _compute_rope(x.dtype)

        for block in self.blocks[start_block:]:
            x = block(
                x,
                freqs_cos=freqs_cos,
                freqs_sin=freqs_sin,
                freqs_cos_q=freqs_cos_q,
                freqs_sin_q=freqs_sin_q,
                q_rope_head_split=self.rope_mass_heads,
                block_mask=block_mask,
            )
        return self.final_norm(x)


class PeakSetSIGReg(nn.Module):
    """Peak-set SIGReg model with multi-crop augmentation."""

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
        encoder_use_rope: bool = False,
        rope_mz_max: float = 1000.0,
        rope_mz_precision: float = 0.1,
        rope_complement_heads: int | None = None,
        rope_modulo_2pi: bool = True,
        use_masked_token_input: bool = False,
        masked_token_position_mode: str = "index",
        masked_token_attention_mode: str = "bidirectional",
        masked_token_loss_weight: float = 0.0,
        masked_latent_predictor_hidden_dim: int = 0,
        encoder_fp16_high_precision_stem: bool = False,
        sigreg_use_projector: bool = True,
        sigreg_proj_hidden_dim: int = 2048,
        sigreg_proj_output_dim: int = 128,
        sigreg_proj_norm: str = "rmsnorm",
        sigreg_num_slices: int = 256,
        sigreg_lambda: float = 0.1,
        multicrop_num_global_views: int = 2,
        multicrop_num_local_views: int = 6,
        multicrop_global_keep_fraction: float = 0.80,
        multicrop_local_keep_fraction: float = 0.25,
        sigreg_mz_jitter_std: float = 0.005,
        sigreg_intensity_jitter_std: float = 0.05,
        pooling_type: str = "pma",
        pma_fp16_high_precision: bool = False,
        pma_num_heads: int | None = None,
        pma_num_seeds: int = 1,
        encoder_block_type: str = "transformer",
        isab_num_inducing_points: int = 32,
        encoder_qk_norm: bool = False,
        encoder_post_norm: bool = False,
    ):
        super().__init__()
        self.num_peaks = num_peaks
        self.model_dim = model_dim
        self.sigreg_dim = sigreg_proj_output_dim if sigreg_use_projector else model_dim

        self.multicrop_num_global_views = int(multicrop_num_global_views)
        self.multicrop_num_local_views = int(multicrop_num_local_views)
        self.multicrop_global_keep_fraction = float(multicrop_global_keep_fraction)
        self.multicrop_local_keep_fraction = float(multicrop_local_keep_fraction)
        self.num_views = self.multicrop_num_global_views + self.multicrop_num_local_views
        self.sigreg_mz_jitter_std = sigreg_mz_jitter_std
        self.sigreg_intensity_jitter_std = sigreg_intensity_jitter_std
        self.pooling_type = pooling_type
        self.pma_fp16_high_precision = bool(pma_fp16_high_precision)
        self.pma_num_seeds = int(pma_num_seeds)
        self.use_masked_token_input = bool(use_masked_token_input)
        self.masked_token_loss_weight = float(masked_token_loss_weight)
        self.masked_latent_predictor_hidden_dim = int(masked_latent_predictor_hidden_dim)

        self.encoder = PeakSetEncoder(
            model_dim=model_dim,
            num_layers=encoder_num_layers,
            num_heads=encoder_num_heads,
            num_kv_heads=encoder_num_kv_heads,
            attention_mlp_multiple=attention_mlp_multiple,
            feature_mlp_hidden_dim=feature_mlp_hidden_dim,
            use_rope=encoder_use_rope,
            rope_mz_max=rope_mz_max,
            rope_mz_precision=rope_mz_precision,
            rope_complement_heads=rope_complement_heads,
            rope_modulo_2pi=rope_modulo_2pi,
            num_peaks=num_peaks,
            masked_token_position_mode=masked_token_position_mode,
            masked_token_attention_mode=masked_token_attention_mode,
            fp16_high_precision_stem=encoder_fp16_high_precision_stem,
            encoder_block_type=encoder_block_type,
            isab_num_inducing_points=isab_num_inducing_points,
            qk_norm=encoder_qk_norm,
            post_norm=encoder_post_norm,
        )

        if sigreg_use_projector:
            proj_norm = str(sigreg_proj_norm).lower()

            def _make_norm(dim: int) -> nn.Module:
                if proj_norm == "rmsnorm":
                    return nn.RMSNorm(dim, eps=1e-5)
                if proj_norm == "batchnorm":
                    return nn.BatchNorm1d(dim, eps=1e-5)
                if proj_norm == "layernorm":
                    return nn.LayerNorm(dim, eps=1e-5)
                if proj_norm in {"none", "identity"}:
                    return nn.Identity()
                raise ValueError(f"Unknown sigreg_proj_norm: {sigreg_proj_norm}")

            self.projector = nn.Sequential(
                nn.Linear(model_dim, sigreg_proj_hidden_dim),
                _make_norm(sigreg_proj_hidden_dim),
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
        self.mask_token = nn.Parameter(torch.empty(model_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        predictor_hidden_dim = (
            model_dim
            if self.masked_latent_predictor_hidden_dim <= 0
            else self.masked_latent_predictor_hidden_dim
        )
        self.masked_latent_predictor = nn.Sequential(
            nn.Linear(model_dim, predictor_hidden_dim),
            nn.SiLU(),
            nn.Linear(predictor_hidden_dim, model_dim),
        )

        self.sigreg = SIGReg(num_slices=int(sigreg_num_slices))

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
        pooled, _ = self.pool_with_raw(embeddings, valid_mask)
        return pooled

    def pool_with_raw(
        self,
        embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pooling_type == "mean":
            pooled = self._mean_pool(embeddings, valid_mask)
            return pooled, pooled
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
                    pooled_raw = pooled.mean(dim=1)
                    pooled = self.pool_norm(pooled_raw)
                target_dtype = autocast_dtype if autocast_dtype is not None else embeddings.dtype
                return pooled.to(dtype=target_dtype), pooled_raw.to(dtype=target_dtype)

            query = self.pool_query.unsqueeze(0).expand(embeddings.shape[0], -1, -1)
            pooled, _ = self.pool_mha(
                query=query,
                key=embeddings,
                value=embeddings,
                key_padding_mask=~valid_mask,
                need_weights=False,
            )
            pooled_raw = pooled.mean(dim=1)
            pooled = self.pool_norm(pooled_raw)
            return pooled, pooled_raw
        raise NotImplementedError(f"Unknown pooling type: {self.pooling_type}")

    def forward_augmented(
        self,
        augmented_batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        fused_mz = augmented_batch["fused_mz"]
        fused_intensity = augmented_batch["fused_intensity"]
        fused_valid_mask = augmented_batch["fused_valid_mask"]
        fused_precursor_mz = augmented_batch["fused_precursor_mz"]
        fused_masked_positions = augmented_batch.get("fused_masked_positions")
        if fused_masked_positions is None:
            fused_masked_positions = torch.zeros_like(fused_valid_mask)
        else:
            fused_masked_positions = fused_masked_positions & fused_valid_mask
        encoder_masked_positions = None
        encoder_mask_token = None
        if self.use_masked_token_input:
            encoder_masked_positions = fused_masked_positions
            encoder_mask_token = self.mask_token
        student_intensity = fused_intensity
        if self.use_masked_token_input:
            student_intensity = fused_intensity.masked_fill(encoder_masked_positions, 0.0)

        fused_emb = self.encoder(
            fused_mz,
            student_intensity,
            valid_mask=fused_valid_mask,
            precursor_mz=fused_precursor_mz,
            masked_positions=encoder_masked_positions,
            mask_token=encoder_mask_token,
        )
        fused_pooled, fused_pooled_raw = self.pool_with_raw(fused_emb, fused_valid_mask)
        fused_z = self.projector(fused_pooled)

        V = self.num_views
        proj = fused_z.reshape(V, -1, fused_z.size(-1))  # [V, B, D]

        # SIGReg Gaussianity (samples random A internally).
        sigreg_loss = self.sigreg(proj)
        loss = sigreg_loss
        masked_latent_loss = torch.zeros((), dtype=loss.dtype, device=loss.device)
        masked_fraction = fused_masked_positions.float().mean()
        if self.masked_token_loss_weight > 0.0:
            with torch.no_grad():
                target_emb = self.encoder(
                    fused_mz,
                    fused_intensity,
                    valid_mask=fused_valid_mask,
                    precursor_mz=fused_precursor_mz,
                    masked_positions=None,
                    mask_token=None,
                )
            predicted_emb = self.masked_latent_predictor(fused_emb)
            per_token = (predicted_emb - target_emb).square().mean(dim=-1)
            mask = fused_masked_positions.float()
            masked_latent_loss = (per_token * mask).sum() / mask.sum().clamp_min(1.0)
            loss = loss + self.masked_token_loss_weight * masked_latent_loss

        representation_variance = fused_z.float().var(dim=0, unbiased=False).mean()
        encoder_variance = fused_pooled.float().var(dim=0, unbiased=False).mean()
        encoder_variance_raw = fused_pooled_raw.float().var(dim=0, unbiased=False).mean()
        encoder_pooled_rms = fused_pooled.float().pow(2).mean(dim=-1).sqrt().mean()
        encoder_pooled_raw_rms = fused_pooled_raw.float().pow(2).mean(dim=-1).sqrt().mean()
        pool_norm_weight_abs_mean = self.pool_norm.weight.abs().mean()

        # Alignment/uniformity on first two views (globals)
        z1, z2 = proj[0], proj[1]
        z1_norm = nn.functional.normalize(z1, dim=-1)
        z2_norm = nn.functional.normalize(z2, dim=-1)
        alignment = (z1_norm * z2_norm).sum(dim=-1).mean()
        uniformity = (z1_norm @ z2_norm.T).fill_diagonal_(0).sum() / (z1.shape[0] * (z1.shape[0] - 1))

        valid_fraction = fused_valid_mask.float().mean()

        return {
            "loss": loss,
            "sigreg_loss": sigreg_loss,
            "masked_latent_loss": masked_latent_loss,
            "valid_fraction": valid_fraction,
            "masked_fraction": masked_fraction,
            "representation_variance": representation_variance,
            "encoder_variance": encoder_variance,
            "encoder_variance_raw": encoder_variance_raw,
            "encoder_pooled_rms": encoder_pooled_rms,
            "encoder_pooled_raw_rms": encoder_pooled_raw_rms,
            "pool_norm_weight_abs_mean": pool_norm_weight_abs_mean,
            "alignment": alignment,
            "uniformity": uniformity,
        }

    def encode(
        self,
        batch: dict[str, torch.Tensor],
        *,
        train: bool = False,
    ) -> torch.Tensor:
        peak_mz = batch["peak_mz"]
        peak_intensity = batch["peak_intensity"]
        peak_valid_mask = batch["peak_valid_mask"]
        precursor_mz = batch["precursor_mz"]

        embeddings = self.encoder(peak_mz, peak_intensity, valid_mask=peak_valid_mask, precursor_mz=precursor_mz)
        pooled = self.pool(embeddings, peak_valid_mask)
        return self.projector(pooled)
