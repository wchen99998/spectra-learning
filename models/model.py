"""Peak-set representation-regularized model for mass spectrometry pretraining.

Architecture:
- PeakSetEncoder: MLP embedder -> non-causal Transformer -> peak embeddings
- Multi-crop augmentation (1 full-spectrum global + K local masked views)
- Objective: selectable regularizer (SIGReg, VICReg, GECO-weighted SIGReg, or GECO-weighted VICReg) + optional masked latent prediction
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from models.losses import SIGReg, VICRegLoss
from networks import transformer_torch
from networks.transformer_torch import (
    create_padding_block_mask,
)


def _normalize_keep_fraction_range(
    keep_fraction: float | tuple[float, float] | list[float],
) -> tuple[float, float]:
    if isinstance(keep_fraction, (tuple, list)):
        return float(keep_fraction[0]), float(keep_fraction[1])
    value = float(keep_fraction)
    return value, value


def _build_norm(dim: int, eps: float, norm_type: str) -> nn.Module:
    kind = str(norm_type).lower()
    if kind == "rmsnorm":
        return nn.RMSNorm(dim, eps=eps)
    if kind == "layernorm":
        return nn.LayerNorm(dim, eps=eps)
    raise ValueError(f"Unsupported norm_type: {norm_type}")


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
    ) -> torch.Tensor:
        log_intensity = torch.log1p(peak_intensity)
        features = torch.cat(
            [
                peak_mz.unsqueeze(-1),
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
    norm_type: str = "rmsnorm",
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
                norm_type=norm_type,
            )
        )
    return nn.ModuleList(blocks)


def _build_standard_rope_inv_freq(
    *,
    head_dim: int,
    base: float,
    device: torch.device,
) -> torch.Tensor:
    half_dim = head_dim // 2
    freq_idx = torch.arange(half_dim, device=device, dtype=torch.float32)
    return 1.0 / (float(base) ** (freq_idx / half_dim))


def _build_rope_freqs_from_positions(
    *,
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    angles = positions.float().unsqueeze(-1) * inv_freq.view(1, 1, -1)
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
        qk_norm: bool = False,
        norm_type: str = "rmsnorm",
    ):
        super().__init__()
        self.model_dim = model_dim
        self.use_rope = bool(use_rope)
        self.norm_type = str(norm_type).lower()
        self.embedder = PeakFeatureEmbedder(model_dim, feature_mlp_hidden_dim)
        heads = int(num_heads)
        head_dim = model_dim // heads
        inv_freq = _build_standard_rope_inv_freq(
            head_dim=head_dim,
            base=10000.0,
            device=torch.device("cpu"),
        )
        self.register_buffer("rope_inv_freq", inv_freq, persistent=False)
        self.blocks = _build_non_causal_blocks(
            dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            attention_mlp_multiple=attention_mlp_multiple,
            use_rope=self.use_rope,
            qk_norm=qk_norm,
            post_norm=False,
            norm_type=self.norm_type,
        )

    def forward(
        self,
        peak_mz: torch.Tensor,
        peak_intensity: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        block_mask = None
        if valid_mask is not None:
            block_mask = create_padding_block_mask(valid_mask)

        def _compute_rope(dtype: torch.dtype) -> tuple[
            torch.Tensor | None,
            torch.Tensor | None,
        ]:
            if not self.use_rope:
                return None, None

            positions = torch.arange(
                peak_mz.shape[1],
                device=peak_mz.device,
                dtype=torch.float32,
            ).unsqueeze(0)
            inv_freq = self.rope_inv_freq
            if inv_freq.device != peak_mz.device:
                inv_freq = inv_freq.to(device=peak_mz.device)

            freqs_cos, freqs_sin = _build_rope_freqs_from_positions(
                positions=positions,
                inv_freq=inv_freq,
                out_dtype=dtype,
            )
            return freqs_cos, freqs_sin

        x = self.embedder(peak_mz, peak_intensity)
        freqs_cos, freqs_sin = _compute_rope(x.dtype)

        for block in self.blocks:
            x = block(
                x,
                freqs_cos=freqs_cos,
                freqs_sin=freqs_sin,
                block_mask=block_mask,
            )
        return x


class PeakSetSIGReg(nn.Module):
    """Peak-set model with multi-crop augmentation and selectable regularizer."""

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
        masked_token_loss_weight: float = 0.0,
        masked_token_loss_type: str = "l1",
        representation_regularizer: str = "sigreg",
        masked_latent_predictor_num_layers: int = 2,
        sigreg_num_slices: int = 256,
        sigreg_lambda: float = 0.1,
        sigreg_lambda_warmup_steps: int = 0,
        gco_std_target: float = 0.8,
        gco_alpha: float = 0.99,
        gco_eta: float = 1e-3,
        gco_log_lambda_init: float = -8.0,
        gco_log_lambda_min: float = -12.0,
        gco_log_lambda_max: float = 2.0,
        vicreg_beta: float = 1e-3,
        vicreg_sim_coeff: float = 0.0,
        vicreg_std_coeff: float = 25.0,
        vicreg_cov_coeff: float = 1.0,
        multicrop_num_local_views: int = 6,
        multicrop_local_keep_fraction: float | tuple[float, float] = 0.25,
        use_ema_teacher_target: bool = False,
        teacher_ema_decay: float = 0.996,
        teacher_ema_decay_start: float = 0.0,
        teacher_ema_decay_warmup_steps: int = 0,
        sigreg_mz_jitter_std: float = 0.005,
        sigreg_intensity_jitter_std: float = 0.05,
        pooling_type: str = "pma",
        pma_fp16_high_precision: bool = False,
        pma_num_heads: int | None = None,
        pma_num_seeds: int = 1,
        encoder_qk_norm: bool = False,
        normalize_jepa_targets: bool = False,
        norm_type: str = "rmsnorm",
    ):
        super().__init__()
        self.num_peaks = num_peaks
        self.model_dim = model_dim

        self.multicrop_num_local_views = int(multicrop_num_local_views)
        self.multicrop_local_keep_fraction = _normalize_keep_fraction_range(
            multicrop_local_keep_fraction
        )
        # One global full-spectrum view + K local masked views.
        self.num_views = 1 + self.multicrop_num_local_views
        self.use_ema_teacher_target = bool(use_ema_teacher_target)
        self.teacher_ema_decay = float(teacher_ema_decay)
        self.teacher_ema_decay_start = float(teacher_ema_decay_start)
        self.teacher_ema_decay_warmup_steps = int(teacher_ema_decay_warmup_steps)
        self.sigreg_mz_jitter_std = sigreg_mz_jitter_std
        self.sigreg_intensity_jitter_std = sigreg_intensity_jitter_std
        self.sigreg_lambda = float(sigreg_lambda)
        self.sigreg_lambda_warmup_steps = int(sigreg_lambda_warmup_steps)
        self.std_target = float(gco_std_target)
        self.gco_alpha = float(gco_alpha)
        self.gco_eta = float(gco_eta)
        self.gco_log_lambda_min = float(gco_log_lambda_min)
        self.gco_log_lambda_max = float(gco_log_lambda_max)
        self.register_buffer(
            "sigreg_lambda_target",
            torch.tensor(self.sigreg_lambda, dtype=torch.float32),
        )
        self.register_buffer(
            "sigreg_lambda_current",
            torch.tensor(
                self.sigreg_lambda if self.sigreg_lambda_warmup_steps <= 0 else 0.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "sigreg_lambda_step",
            torch.zeros((), dtype=torch.int64),
        )
        self.register_buffer(
            "sigreg_lambda_warmup_steps_tensor",
            torch.tensor(
                max(self.sigreg_lambda_warmup_steps, 1),
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "gco_log_lambda",
            torch.tensor(float(gco_log_lambda_init), dtype=torch.float32),
        )
        self.register_buffer(
            "gco_c_ema",
            torch.tensor(0.0, dtype=torch.float32),
        )
        self.register_buffer(
            "teacher_ema_decay_start_tensor",
            torch.tensor(self.teacher_ema_decay_start, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "teacher_ema_decay_target",
            torch.tensor(self.teacher_ema_decay, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "teacher_ema_decay_current",
            torch.tensor(
                self.teacher_ema_decay
                if self.teacher_ema_decay_warmup_steps <= 0
                else self.teacher_ema_decay_start,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "teacher_ema_decay_step",
            torch.zeros((), dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "teacher_ema_decay_warmup_steps_tensor",
            torch.tensor(
                max(self.teacher_ema_decay_warmup_steps, 1),
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.pooling_type = pooling_type
        self.pma_fp16_high_precision = bool(pma_fp16_high_precision)
        self.pma_num_seeds = int(pma_num_seeds)
        self.masked_token_loss_weight = float(masked_token_loss_weight)
        self.masked_token_loss_type = str(masked_token_loss_type).lower()
        self.normalize_jepa_targets = bool(normalize_jepa_targets)
        self.representation_regularizer = str(representation_regularizer).lower()
        if self.representation_regularizer == "gco":
            self.representation_regularizer = "gco-sigreg"
        self.norm_type = str(norm_type).lower()
        self.masked_latent_predictor_num_layers = int(masked_latent_predictor_num_layers)
        self.vicreg_beta = float(vicreg_beta)
        if self.representation_regularizer in ("vicreg", "gco-vicreg"):
            assert self.multicrop_num_local_views >= 2, (
                "representation_regularizer in {'vicreg', 'gco-vicreg'} requires at least two local views "
                "(multicrop_num_local_views >= 2)."
            )
        if self.representation_regularizer in ("gco-sigreg", "gco-vicreg"):
            assert self.multicrop_num_local_views >= 1, (
                "representation_regularizer in {'gco-sigreg', 'gco-vicreg'} requires at least one local view "
                "(multicrop_num_local_views >= 1)."
            )

        self.encoder = PeakSetEncoder(
            model_dim=model_dim,
            num_layers=encoder_num_layers,
            num_heads=encoder_num_heads,
            num_kv_heads=encoder_num_kv_heads,
            attention_mlp_multiple=attention_mlp_multiple,
            feature_mlp_hidden_dim=feature_mlp_hidden_dim,
            use_rope=encoder_use_rope,
            qk_norm=encoder_qk_norm,
            norm_type=self.norm_type,
        )
        if self.use_ema_teacher_target:
            self.teacher_encoder: AveragedModel | None = AveragedModel(
                self.encoder,
                multi_avg_fn=get_ema_multi_avg_fn(self.teacher_ema_decay),
                use_buffers=True,
            )
            self.teacher_encoder.requires_grad_(False)
            self.teacher_encoder.eval()
        else:
            self.teacher_encoder = None

        pma_heads = int(encoder_num_heads) if pma_num_heads is None else int(pma_num_heads)
        self.pool_query = nn.Parameter(torch.empty(self.pma_num_seeds, model_dim))
        nn.init.xavier_normal_(self.pool_query)
        self.pool_mha = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=pma_heads,
            batch_first=True,
        )
        self.pool_norm = _build_norm(model_dim, eps=1e-5, norm_type=self.norm_type)
        self.latent_mask_token = nn.Parameter(torch.empty(self.model_dim))
        nn.init.normal_(self.latent_mask_token, std=0.02)
        predictor_max_heads = min(int(encoder_num_heads), self.model_dim // 16)
        self.predictor_num_heads = max(1, predictor_max_heads)
        while self.model_dim % self.predictor_num_heads != 0:
            self.predictor_num_heads -= 1
        predictor_head_dim = self.model_dim // self.predictor_num_heads
        predictor_inv_freq = _build_standard_rope_inv_freq(
            head_dim=predictor_head_dim,
            base=10000.0,
            device=torch.device("cpu"),
        )
        self.register_buffer("predictor_rope_inv_freq", predictor_inv_freq, persistent=False)
        self.masked_latent_predictor = _build_non_causal_blocks(
            dim=self.model_dim,
            num_layers=self.masked_latent_predictor_num_layers,
            num_heads=self.predictor_num_heads,
            num_kv_heads=None,
            attention_mlp_multiple=attention_mlp_multiple,
            use_rope=encoder_use_rope,
            qk_norm=encoder_qk_norm,
            post_norm=False,
            norm_type=self.norm_type,
        )

        self.sigreg = SIGReg(num_slices=int(sigreg_num_slices))
        self.vicreg = VICRegLoss(
            sim_coeff=float(vicreg_sim_coeff),
            std_coeff=float(vicreg_std_coeff),
            cov_coeff=float(vicreg_cov_coeff),
        )

    @torch.no_grad()
    def advance_sigreg_lambda_schedule(self) -> None:
        if self.representation_regularizer != "sigreg":
            return
        if self.sigreg_lambda_warmup_steps <= 0:
            self.sigreg_lambda_current.copy_(self.sigreg_lambda_target)
            return
        step = self.sigreg_lambda_step.to(dtype=self.sigreg_lambda_current.dtype)
        ratio = torch.clamp(step / self.sigreg_lambda_warmup_steps_tensor, min=0.0, max=1.0)
        self.sigreg_lambda_current.copy_(self.sigreg_lambda_target * ratio)
        self.sigreg_lambda_step.add_(1)

    @torch.no_grad()
    def advance_teacher_ema_decay_schedule(self) -> None:
        if self.teacher_ema_decay_warmup_steps <= 0:
            self.teacher_ema_decay_current.copy_(self.teacher_ema_decay_target)
            return
        step = self.teacher_ema_decay_step.to(dtype=self.teacher_ema_decay_current.dtype)
        ratio = torch.clamp(step / self.teacher_ema_decay_warmup_steps_tensor, min=0.0, max=1.0)
        delta = self.teacher_ema_decay_target - self.teacher_ema_decay_start_tensor
        self.teacher_ema_decay_current.copy_(self.teacher_ema_decay_start_tensor + delta * ratio)
        self.teacher_ema_decay_step.add_(1)

    def predict_masked_latents(
        self,
        local_token_emb_remasked: torch.Tensor,
        local_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Predictor is a lightweight non-causal transformer stack over token
        # latents, using the local valid-mask to block out padding tokens.
        predictor_block_mask = create_padding_block_mask(local_valid_mask)
        freqs_cos = None
        freqs_sin = None
        if self.encoder.use_rope:
            positions = torch.arange(
                local_token_emb_remasked.shape[1],
                device=local_token_emb_remasked.device,
                dtype=torch.float32,
            ).unsqueeze(0)
            inv_freq = self.predictor_rope_inv_freq
            if inv_freq.device != local_token_emb_remasked.device:
                inv_freq = inv_freq.to(device=local_token_emb_remasked.device)
            freqs_cos, freqs_sin = _build_rope_freqs_from_positions(
                positions=positions,
                inv_freq=inv_freq,
                out_dtype=local_token_emb_remasked.dtype,
            )
        x = local_token_emb_remasked
        for block in self.masked_latent_predictor:
            x = block(
                x,
                freqs_cos=freqs_cos,
                freqs_sin=freqs_sin,
                block_mask=predictor_block_mask,
            )
        return x

    def train(self, mode: bool = True) -> "PeakSetSIGReg":
        super().train(mode)
        if self.teacher_encoder is not None:
            # Teacher is an EMA target network and should stay in eval mode.
            self.teacher_encoder.eval()
        return self

    @torch.no_grad()
    def update_teacher(self) -> None:
        if self.teacher_encoder is None:
            return
        self.advance_teacher_ema_decay_schedule()
        self.teacher_encoder.multi_avg_fn = get_ema_multi_avg_fn(float(self.teacher_ema_decay_current))
        self.teacher_encoder.update_parameters(self.encoder)

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
        # Data pipeline contract:
        # - View 0 is the global full-spectrum view.
        # - Views 1..V-1 are local masked views.
        # - fused_masked_positions is always present.
        fused_masked_positions = augmented_batch["fused_masked_positions"] & fused_valid_mask

        V = self.num_views
        target_global_view_idx = 0

        # Global target view is always full-spectrum context. Only non-global views
        # use masked positions for encoder masking and latent remasking.
        effective_masked_positions = fused_masked_positions.reshape(V, -1, fused_masked_positions.shape[1]).clone()
        effective_masked_positions[target_global_view_idx] = False
        effective_masked_positions = effective_masked_positions.reshape_as(fused_masked_positions)
        encoder_masked_positions = effective_masked_positions
        encoder_visible_mask = fused_valid_mask & (~encoder_masked_positions)

        fused_emb = self.encoder(
            fused_mz,
            fused_intensity,
            valid_mask=encoder_visible_mask,
        )

        B = fused_emb.shape[0] // V
        N = fused_emb.shape[1]
        token_emb = fused_emb.reshape(V, B, N, fused_emb.shape[2])
        token_valid = fused_valid_mask.reshape(V, B, N)
        token_masked = effective_masked_positions.reshape(V, B, N)

        # Local predictors regress to a fixed global latent target.
        # When EMA teacher is enabled, the target comes from teacher(global view).
        if self.teacher_encoder is not None:
            global_slice = slice(0, B)
            with torch.no_grad():
                global_token_target = self.teacher_encoder(
                    fused_mz[global_slice],
                    fused_intensity[global_slice],
                    valid_mask=fused_valid_mask[global_slice],
                ).detach()
        else:
            global_token_target = token_emb[target_global_view_idx].detach()
        latent_mask_token = self.latent_mask_token.view(1, 1, -1).to(
            dtype=fused_emb.dtype,
            device=fused_emb.device,
        )
        L = V - 1  # number of local views
        # Batch all local views for a single predictor pass instead of
        # iterating per-view, doubling the effective batch size for GEMMs
        # and halving the number of kernel launches.
        all_local_emb = token_emb[target_global_view_idx + 1:]    # [L, B, N, D]
        all_local_valid = token_valid[target_global_view_idx + 1:]  # [L, B, N]
        all_local_mask = token_masked[target_global_view_idx + 1:]  # [L, B, N]
        all_local_emb_flat = all_local_emb.reshape(L * B, N, -1)
        all_local_valid_flat = all_local_valid.reshape(L * B, N)
        all_local_mask_flat = all_local_mask.reshape(L * B, N)

        # Re-mask local latents at masked slots before prediction so the
        # predictor must infer masked content from surrounding context.
        all_local_remasked = torch.where(
            all_local_mask_flat.unsqueeze(-1),
            latent_mask_token,
            all_local_emb_flat,
        )
        all_local_pred = self.predict_masked_latents(
            all_local_remasked,
            all_local_valid_flat,
        )
        # Compute per-token regression loss across all local views at once.
        # Reshape to [L, B, N, D_loss] for broadcasting against [B, N, D_loss].
        all_local_pred_views = all_local_pred.reshape(L, B, N, -1)
        loss_pred = all_local_pred_views
        loss_target = global_token_target.unsqueeze(0)
        if self.normalize_jepa_targets:
            loss_pred = torch.nn.functional.normalize(loss_pred, dim=-1)
            loss_target = torch.nn.functional.normalize(loss_target, dim=-1)

        if self.masked_token_loss_type == "cosine":
            per_token_reg = 2.0 - 2.0 * (loss_pred * loss_target).sum(dim=-1)
        elif self.masked_token_loss_type == "l2":
            per_token_reg = (loss_pred - loss_target).square().mean(dim=-1)
        elif self.masked_token_loss_type == "l2_sum":
            per_token_reg = (loss_pred - loss_target).square().sum(dim=-1)  # == cosine distance for unit vectors
        elif self.masked_token_loss_type == "l1":
            per_token_reg = (loss_pred - loss_target).abs().mean(dim=-1)
        else:
            raise ValueError(...)
        all_local_mask_float = all_local_mask.float()  # [L, B, N]
        reg_num = (per_token_reg * all_local_mask_float).sum()
        reg_den = all_local_mask_float.sum()
        local_global_loss = reg_num / reg_den.clamp_min(1.0)

        if self.sigreg_lambda_warmup_steps > 0:
            sigreg_lambda_current = self.sigreg_lambda_current.to(dtype=fused_emb.dtype)
        else:
            sigreg_lambda_current = fused_emb.new_tensor(self.sigreg_lambda)

        jepa_term = self.masked_token_loss_weight * local_global_loss
        # Apply representation regularizers on encoder token embeddings
        # from local views only, masking only true padding tokens.
        local_emb = token_emb[1:]                          # [L, B, N, D]
        local_valid = token_valid[1:]                       # [L, B, N]
        regularizer_emb_flat = local_emb.reshape(L * B, N, -1)
        regularizer_valid_flat = local_valid.reshape(L * B, N)
        regularizer_emb_by_view = local_emb
        regularizer_valid_by_view = local_valid

        # --- Collapse monitoring (detached, no grad) ---
        with torch.no_grad():
            collapse_metrics = self._collapse_metrics(
                fused_emb, encoder_visible_mask, B, V, target_global_view_idx,
            )
            local_to_global_emb_std_ratio = (
                collapse_metrics["local_emb_std"] / collapse_metrics["global_emb_std"]
            )
            regularizer_flat = regularizer_emb_flat.float().reshape(-1, regularizer_emb_flat.shape[-1])
            regularizer_weights = regularizer_valid_flat.reshape(-1).float()
            regularizer_count = regularizer_weights.sum().clamp_min(1.0)
            regularizer_weights_col = regularizer_weights.unsqueeze(-1)
            regularizer_mean = (regularizer_flat * regularizer_weights_col).sum(0) / regularizer_count
            regularizer_var = (
                (regularizer_flat - regularizer_mean).square() * regularizer_weights_col
            ).sum(0) / regularizer_count
            encoder_emb_std = torch.sqrt(regularizer_var + 1e-6).mean()
            gco_constraint = self.std_target - encoder_emb_std.float()

        gco_std_penalty = torch.relu(gco_constraint).to(dtype=fused_emb.dtype)
        gco_lambda = self.gco_log_lambda.exp().to(dtype=fused_emb.dtype)

        if self.representation_regularizer in ("gco-sigreg", "gco-vicreg"):
            with torch.no_grad():
                if self.training:
                    self.gco_c_ema.mul_(self.gco_alpha).add_((1.0 - self.gco_alpha) * gco_constraint)
                    self.gco_log_lambda.add_(self.gco_eta * self.gco_c_ema)
                    self.gco_log_lambda.clamp_(self.gco_log_lambda_min, self.gco_log_lambda_max)
                gco_lambda = self.gco_log_lambda.exp().to(dtype=fused_emb.dtype)

        # Regularizer path (SIGReg, VICReg, GECO-weighted SIGReg, or GECO-weighted VICReg).
        if self.representation_regularizer == "sigreg" and self.sigreg_lambda > 0:
            token_sigreg_loss = self.sigreg(
                regularizer_emb_flat,
                valid_mask=regularizer_valid_flat,
            )
            sigreg_term = sigreg_lambda_current * token_sigreg_loss
            vicreg_loss = fused_emb.new_tensor(0.0)
            vicreg_term = fused_emb.new_tensor(0.0)
        elif self.representation_regularizer == "vicreg":
            pair_valid_mask = regularizer_valid_by_view[0] & regularizer_valid_by_view[1]
            vicreg_loss = self.vicreg(
                regularizer_emb_by_view[0],
                regularizer_emb_by_view[1],
                valid_mask=pair_valid_mask,
            )
            vicreg_term = fused_emb.new_tensor(self.vicreg_beta) * vicreg_loss
            token_sigreg_loss = fused_emb.new_tensor(0.0)
            sigreg_term = fused_emb.new_tensor(0.0)
        elif self.representation_regularizer == "gco-sigreg":
            token_sigreg_loss = self.sigreg(
                regularizer_emb_flat,
                valid_mask=regularizer_valid_flat,
            )
            sigreg_lambda_current = gco_lambda
            sigreg_term = gco_lambda * token_sigreg_loss
            vicreg_loss = fused_emb.new_tensor(0.0)
            vicreg_term = fused_emb.new_tensor(0.0)
        elif self.representation_regularizer == "gco-vicreg":
            pair_valid_mask = regularizer_valid_by_view[0] & regularizer_valid_by_view[1]
            vicreg_loss = self.vicreg(
                regularizer_emb_by_view[0],
                regularizer_emb_by_view[1],
                valid_mask=pair_valid_mask,
            )
            sigreg_lambda_current = gco_lambda
            vicreg_term = gco_lambda * vicreg_loss
            token_sigreg_loss = fused_emb.new_tensor(0.0)
            sigreg_term = fused_emb.new_tensor(0.0)
        else:
            # "none" or sigreg with lambda=0: skip regularization.
            token_sigreg_loss = fused_emb.new_tensor(0.0)
            sigreg_term = fused_emb.new_tensor(0.0)
            vicreg_loss = fused_emb.new_tensor(0.0)
            vicreg_term = fused_emb.new_tensor(0.0)

        regularizer_term = sigreg_term + vicreg_term
        loss = jepa_term + regularizer_term
        safe_jepa = jepa_term.clamp_min(1e-8)
        target_regularizer_term_over_jepa_term = regularizer_term / safe_jepa
        target_sigreg_term_over_jepa_term = sigreg_term / safe_jepa
        target_vicreg_term_over_jepa_term = vicreg_term / safe_jepa
        # Report masking rate on valid local peaks only; global target view is
        # full-spectrum and excluded from this metric.
        local_valid = token_valid[target_global_view_idx + 1:]
        local_masked = token_masked[target_global_view_idx + 1:]
        local_valid_count = local_valid.float().sum().clamp_min(1.0)
        masked_fraction = local_masked.float().sum() / local_valid_count
        valid_fraction = (local_valid & (~local_masked)).float().sum() / local_valid_count

        pool_norm_weight_abs_mean = self.pool_norm.weight.abs().mean()

        return {
            "loss": loss,
            "sigreg_loss": token_sigreg_loss,
            "token_sigreg_loss": token_sigreg_loss,
            "local_global_loss": local_global_loss,
            "local_global_l1_loss": local_global_loss,
            "regularizer_loss": token_sigreg_loss + vicreg_loss,
            "regularizer_term": regularizer_term,
            "sigreg_term": sigreg_term,
            "vicreg_loss": vicreg_loss,
            "vicreg_term": vicreg_term,
            "jepa_term": jepa_term,
            "target_regularizer_term_over_jepa_term": target_regularizer_term_over_jepa_term,
            "target_sigreg_term_over_jepa_term": target_sigreg_term_over_jepa_term,
            "target_vicreg_term_over_jepa_term": target_vicreg_term_over_jepa_term,
            "valid_fraction": valid_fraction,
            "masked_fraction": masked_fraction,
            "sigreg_lambda_current": sigreg_lambda_current,
            "gco_lambda": gco_lambda,
            "gco_log_lambda": self.gco_log_lambda.to(dtype=fused_emb.dtype),
            "gco_c_ema": self.gco_c_ema.to(dtype=fused_emb.dtype),
            "gco_constraint": gco_constraint.to(dtype=fused_emb.dtype),
            "gco_std_penalty": gco_std_penalty,
            "encoder_emb_std": encoder_emb_std.to(dtype=fused_emb.dtype),
            "pool_norm_weight_abs_mean": pool_norm_weight_abs_mean,
            "local_to_global_emb_std_ratio": local_to_global_emb_std_ratio,
            **collapse_metrics,
        }

    @torch.no_grad()
    def _collapse_metrics(
        self,
        fused_emb: torch.Tensor,
        visible_mask: torch.Tensor,
        B: int,
        V: int,
        target_global_view_idx: int,
    ) -> dict[str, torch.Tensor]:
        """Compute per-view collapse indicators (detached, no grad).

        Returns metrics for global and local views:
        - ``*_emb_std``: mean per-dimension std over valid tokens (low = dimensional collapse)
        - ``*_emb_norm``: mean L2 norm of valid token embeddings
        """
        emb = fused_emb.float()  # [V*B, N, D]
        mask = visible_mask  # [V*B, N]

        def _stats(x: torch.Tensor, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            flat = x.reshape(-1, x.shape[-1])  # [S*N, D]
            w = m.reshape(-1).float()  # [S*N]
            n = w.sum().clamp_min(1.0)
            w_col = w.unsqueeze(-1)  # [S*N, 1]
            mean = (flat * w_col).sum(0) / n  # [D]
            var = ((flat - mean).square() * w_col).sum(0) / n  # [D]
            per_dim_std = var.sqrt().mean()
            mean_norm = (flat.norm(dim=-1) * w).sum() / n
            return per_dim_std, mean_norm

        global_std, global_norm = _stats(emb[:B], mask[:B])
        local_std, local_norm = _stats(emb[B:], mask[B:])

        return {
            "global_emb_std": global_std,
            "local_emb_std": local_std,
            "global_emb_norm": global_norm,
            "local_emb_norm": local_norm,
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

        embeddings = self.encoder(peak_mz, peak_intensity, valid_mask=peak_valid_mask)
        pooled = self.pool(embeddings, peak_valid_mask)
        return pooled
