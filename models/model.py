from __future__ import annotations

import math

import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from models.losses import SIGReg
from networks import transformer_torch
from networks.transformer_torch import (
    _build_norm,
    create_visible_block_mask,
)


class PeakFeatureEmbedder(nn.Module):
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
        log_intensity = torch.log1p(peak_intensity.clamp(min=0.0))
        return self.mlp(torch.stack([peak_mz, peak_intensity, log_intensity], dim=-1))


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
    block_kwargs = dict(
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
    return nn.ModuleList(
        [transformer_torch.TransformerBlock(**block_kwargs) for _ in range(num_layers)]
    )


def _build_standard_rope_inv_freq(
    *,
    head_dim: int,
    base: float,
    device: torch.device,
) -> torch.Tensor:
    half_dim = head_dim // 2
    freq_idx = torch.arange(half_dim, device=device, dtype=torch.float32)
    return 1.0 / (float(base) ** (freq_idx / half_dim))


def _compute_rope_freqs(
    use_rope: bool,
    seq_len: int,
    inv_freq: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not use_rope:
        return None, None
    positions = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(0)
    if inv_freq.device != device:
        inv_freq = inv_freq.to(device=device)
    angles = positions.unsqueeze(-1) * inv_freq.view(1, 1, -1)
    angles = torch.repeat_interleave(angles, repeats=2, dim=-1)
    return angles.cos().to(dtype=dtype).unsqueeze(2), angles.sin().to(
        dtype=dtype
    ).unsqueeze(2)


def _masked_embedding_stats(
    emb: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    flat = emb.float().reshape(-1, emb.shape[-1])
    weights = valid_mask.reshape(-1).float()
    count = weights.sum().clamp_min(1.0)
    weights_col = weights.unsqueeze(-1)
    mean = (flat * weights_col).sum(0) / count
    centered = flat - mean
    weighted_centered = centered * weights_col
    cov = centered.transpose(0, 1) @ weighted_centered / count
    var = cov.diagonal()
    var_scale = var.clamp_min(1e-12)
    corr = cov / torch.sqrt(var_scale.unsqueeze(0) * var_scale.unsqueeze(1))
    offdiag_den = cov.shape[0] * (cov.shape[0] - 1)
    return {
        "emb_std": torch.sqrt(var + 1e-6).mean(),
        "emb_norm": (flat.norm(dim=-1) * weights).sum() / count,
        "emb_var_mean": var.mean(),
        "emb_var_floor": var.amin(),
        "emb_cov_offdiag_abs_mean": (cov.abs().sum() - cov.diagonal().abs().sum())
        / offdiag_den,
        "emb_corr_offdiag_abs_mean": (corr.abs().sum() - corr.diagonal().abs().sum())
        / offdiag_den,
    }


class PeakSetEncoder(nn.Module):
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
        self.use_rope = bool(use_rope)
        norm_type = str(norm_type).lower()
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
            norm_type=norm_type,
        )

    def forward(
        self,
        peak_mz: torch.Tensor,
        peak_intensity: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        visible_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if visible_mask is not None and valid_mask is not None:
            attn_mask = visible_mask & valid_mask
        else:
            attn_mask = visible_mask if visible_mask is not None else valid_mask
        block_mask = (
            create_visible_block_mask(attn_mask) if attn_mask is not None else None
        )

        x = self.embedder(peak_mz, peak_intensity)
        freqs_cos, freqs_sin = _compute_rope_freqs(
            self.use_rope, peak_mz.shape[1], self.rope_inv_freq, peak_mz.device, x.dtype
        )

        for block in self.blocks:
            x = block(
                x,
                freqs_cos=freqs_cos,
                freqs_sin=freqs_sin,
                block_mask=block_mask,
            )
        return x


class PeakSetSIGReg(nn.Module):
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
        gco_var_floor_target: float = 1e-3,
        gco_corr_target: float = 0.6,
        gco_alpha: float = 0.99,
        gco_eta: float = 1e-3,
        gco_log_lambda_init: float = -8.0,
        gco_log_lambda_min: float = -12.0,
        gco_log_lambda_max: float = 2.0,
        jepa_num_target_blocks: int = 2,
        jepa_context_fraction: float = 0.5,
        jepa_target_fraction: float = 0.25,
        jepa_block_min_len: int = 1,
        use_ema_teacher_target: bool = False,
        teacher_ema_decay: float = 0.996,
        teacher_ema_decay_start: float = 0.0,
        teacher_ema_decay_warmup_steps: int = 0,
        sigreg_mz_jitter_std: float = 0.005,
        sigreg_intensity_jitter_std: float = 0.05,
        pooling_type: str = "pma",
        pma_num_heads: int | None = None,
        pma_num_seeds: int = 1,
        encoder_qk_norm: bool = False,
        normalize_jepa_targets: bool = False,
        norm_type: str = "rmsnorm",
        use_precursor_token: bool = False,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.use_precursor_token = bool(use_precursor_token)

        self.jepa_num_target_blocks = int(jepa_num_target_blocks)
        _teacher_ema_decay = float(teacher_ema_decay)
        _teacher_ema_decay_start = float(teacher_ema_decay_start)
        self.teacher_ema_decay_warmup_steps = int(teacher_ema_decay_warmup_steps)
        self.sigreg_lambda = float(sigreg_lambda)
        self.sigreg_lambda_warmup_steps = int(sigreg_lambda_warmup_steps)
        self.gco_var_floor_target = float(gco_var_floor_target)
        self.gco_corr_target = float(gco_corr_target)
        self.gco_alpha = float(gco_alpha)
        self.gco_eta = float(gco_eta)
        self.gco_log_lambda_min = float(gco_log_lambda_min)
        self.gco_log_lambda_max = float(gco_log_lambda_max)

        def _f32(v: float) -> torch.Tensor:
            return torch.tensor(v, dtype=torch.float32)

        _reg = self.register_buffer
        _sr_init = self.sigreg_lambda if self.sigreg_lambda_warmup_steps <= 0 else 0.0
        _reg("sigreg_lambda_target", _f32(self.sigreg_lambda))
        _reg("sigreg_lambda_current", _f32(_sr_init))
        _reg("sigreg_lambda_step", torch.zeros((), dtype=torch.int64))
        _reg(
            "sigreg_lambda_warmup_steps_tensor",
            _f32(max(self.sigreg_lambda_warmup_steps, 1)),
            persistent=False,
        )
        _reg("gco_log_lambda", _f32(float(gco_log_lambda_init)))
        _reg("gco_c_ema", _f32(0.0))
        _reg(
            "teacher_ema_decay_start_tensor",
            _f32(_teacher_ema_decay_start),
            persistent=False,
        )
        _reg("teacher_ema_decay_target", _f32(_teacher_ema_decay), persistent=False)
        _ema_init = (
            _teacher_ema_decay
            if self.teacher_ema_decay_warmup_steps <= 0
            else _teacher_ema_decay_start
        )
        _reg("teacher_ema_decay_current", _f32(_ema_init), persistent=False)
        _reg(
            "teacher_ema_decay_step",
            torch.zeros((), dtype=torch.int64),
            persistent=False,
        )
        _reg(
            "teacher_ema_decay_warmup_steps_tensor",
            _f32(max(self.teacher_ema_decay_warmup_steps, 1)),
            persistent=False,
        )
        self.pooling_type = pooling_type
        self.pma_num_seeds = int(pma_num_seeds)
        self.masked_token_loss_weight = float(masked_token_loss_weight)
        self.masked_token_loss_type = str(masked_token_loss_type).lower()
        self.normalize_jepa_targets = bool(normalize_jepa_targets)
        self.representation_regularizer = str(representation_regularizer).lower()
        if self.representation_regularizer == "gco":
            self.representation_regularizer = "gco-sigreg"
        self.norm_type = str(norm_type).lower()
        _predictor_num_layers = int(masked_latent_predictor_num_layers)
        if self.jepa_num_target_blocks < 1:
            raise ValueError("jepa_num_target_blocks must be >= 1")
        if self.representation_regularizer not in ("sigreg", "gco-sigreg", "none", ""):
            raise ValueError(
                f"Unsupported regularizer: {self.representation_regularizer!r}"
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
        if use_ema_teacher_target:
            self.teacher_encoder: AveragedModel | None = AveragedModel(
                self.encoder,
                multi_avg_fn=get_ema_multi_avg_fn(_teacher_ema_decay),
                use_buffers=True,
            )
            self.teacher_encoder.requires_grad_(False)
            self.teacher_encoder.eval()
        else:
            self.teacher_encoder = None

        pma_heads = (
            int(encoder_num_heads) if pma_num_heads is None else int(pma_num_heads)
        )
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
        pred_heads = max(1, min(int(encoder_num_heads), self.model_dim // 16))
        while self.model_dim % pred_heads != 0:
            pred_heads -= 1
        predictor_head_dim = self.model_dim // pred_heads
        predictor_inv_freq = _build_standard_rope_inv_freq(
            head_dim=predictor_head_dim,
            base=10000.0,
            device=torch.device("cpu"),
        )
        self.register_buffer(
            "predictor_rope_inv_freq", predictor_inv_freq, persistent=False
        )
        self.masked_latent_predictor = _build_non_causal_blocks(
            dim=self.model_dim,
            num_layers=_predictor_num_layers,
            num_heads=pred_heads,
            num_kv_heads=None,
            attention_mlp_multiple=attention_mlp_multiple,
            use_rope=encoder_use_rope,
            qk_norm=encoder_qk_norm,
            post_norm=False,
            norm_type=self.norm_type,
        )

        self.sigreg = SIGReg(num_slices=int(sigreg_num_slices))

    @torch.no_grad()
    def advance_sigreg_lambda_schedule(self) -> None:
        if self.representation_regularizer != "sigreg":
            return
        if self.sigreg_lambda_warmup_steps <= 0:
            self.sigreg_lambda_current.copy_(self.sigreg_lambda_target)
            return
        step = self.sigreg_lambda_step.to(dtype=self.sigreg_lambda_current.dtype)
        ratio = torch.clamp(
            step / self.sigreg_lambda_warmup_steps_tensor, min=0.0, max=1.0
        )
        self.sigreg_lambda_current.copy_(self.sigreg_lambda_target * ratio)
        self.sigreg_lambda_step.add_(1)

    @torch.no_grad()
    def advance_teacher_ema_decay_schedule(self) -> None:
        if self.teacher_ema_decay_warmup_steps <= 0:
            self.teacher_ema_decay_current.copy_(self.teacher_ema_decay_target)
            return
        step = self.teacher_ema_decay_step.to(
            dtype=self.teacher_ema_decay_current.dtype
        )
        ratio = torch.clamp(
            step / self.teacher_ema_decay_warmup_steps_tensor, min=0.0, max=1.0
        )
        delta = self.teacher_ema_decay_target - self.teacher_ema_decay_start_tensor
        self.teacher_ema_decay_current.copy_(
            self.teacher_ema_decay_start_tensor + delta * ratio
        )
        self.teacher_ema_decay_step.add_(1)

    def predict_masked_latents(
        self,
        x: torch.Tensor,
        visible_mask: torch.Tensor,
    ) -> torch.Tensor:
        predictor_block_mask = create_visible_block_mask(visible_mask)
        freqs_cos, freqs_sin = _compute_rope_freqs(
            self.encoder.use_rope,
            x.shape[1],
            self.predictor_rope_inv_freq,
            x.device,
            x.dtype,
        )
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
            self.teacher_encoder.eval()
        return self

    @torch.no_grad()
    def update_teacher(self) -> None:
        if self.teacher_encoder is None:
            return
        self.advance_teacher_ema_decay_schedule()
        self.teacher_encoder.multi_avg_fn = get_ema_multi_avg_fn(
            float(self.teacher_ema_decay_current)
        )
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

    @staticmethod
    def prepend_precursor_token(
        peak_mz: torch.Tensor,
        peak_intensity: torch.Tensor,
        peak_valid_mask: torch.Tensor,
        precursor_mz: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        target_masks: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        B, N = peak_mz.shape
        device = peak_mz.device
        dtype = peak_mz.dtype

        pre_mz = precursor_mz.unsqueeze(1)
        pre_int = torch.full((B, 1), -1.0, device=device, dtype=dtype)
        pre_valid = torch.ones(B, 1, device=device, dtype=torch.bool)

        result: dict[str, torch.Tensor] = {
            "peak_mz": torch.cat([pre_mz, peak_mz], dim=1),
            "peak_intensity": torch.cat([pre_int, peak_intensity], dim=1),
            "peak_valid_mask": torch.cat([pre_valid, peak_valid_mask], dim=1),
        }

        if context_mask is not None:
            pre_ctx = torch.ones(B, 1, device=device, dtype=torch.bool)
            result["context_mask"] = torch.cat([pre_ctx, context_mask], dim=1)

        if target_masks is not None:
            K = target_masks.shape[1]
            pre_tgt = torch.zeros(B, K, 1, device=device, dtype=torch.bool)
            result["target_masks"] = torch.cat([pre_tgt, target_masks], dim=2)

        return result

    def forward_augmented(
        self,
        augmented_batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        peak_mz = augmented_batch["peak_mz"]
        peak_intensity = augmented_batch["peak_intensity"]
        peak_valid_mask = augmented_batch["peak_valid_mask"]
        context_mask = augmented_batch["context_mask"] & peak_valid_mask
        target_masks = augmented_batch["target_masks"] & peak_valid_mask.unsqueeze(1)

        B, N = peak_mz.shape
        K = self.jepa_num_target_blocks

        context_emb = self.encoder(
            peak_mz,
            peak_intensity,
            valid_mask=peak_valid_mask,
            visible_mask=context_mask,
        )

        def _expand_for_targets(t: torch.Tensor) -> torch.Tensor:
            return t.unsqueeze(1).expand(-1, K, -1).reshape(B * K, N)

        peak_mz_targets = _expand_for_targets(peak_mz)
        peak_intensity_targets = _expand_for_targets(peak_intensity)
        peak_valid_targets = _expand_for_targets(peak_valid_mask)
        target_masks_flat = target_masks.reshape(B * K, N)
        target_emb_flat = self.encoder(
            peak_mz_targets,
            peak_intensity_targets,
            valid_mask=peak_valid_targets,
            visible_mask=target_masks_flat,
        )
        target_emb = target_emb_flat.reshape(B, K, N, -1).permute(1, 0, 2, 3)
        target_masks_by_view = target_masks.permute(1, 0, 2)

        if self.teacher_encoder is not None:
            with torch.no_grad():
                target_token_target_flat = self.teacher_encoder(
                    peak_mz_targets,
                    peak_intensity_targets,
                    valid_mask=peak_valid_targets,
                    visible_mask=target_masks_flat,
                ).detach()
            target_token_target = target_token_target_flat.reshape(B, K, N, -1).permute(
                1, 0, 2, 3
            )
        else:
            target_token_target = target_emb.detach()

        predictor_context_mask = context_mask.unsqueeze(0)
        predictor_union_mask = predictor_context_mask | target_masks_by_view
        context_emb_by_view = context_emb.unsqueeze(0).expand(K, -1, -1, -1)
        predictor_input = context_emb_by_view * predictor_context_mask.unsqueeze(-1)
        latent_mask_token = self.latent_mask_token.view(1, 1, 1, -1).to(
            dtype=context_emb.dtype,
            device=context_emb.device,
        )
        predictor_input = torch.where(
            target_masks_by_view.unsqueeze(-1),
            latent_mask_token,
            predictor_input,
        )
        predictor_output_flat = self.predict_masked_latents(
            predictor_input.reshape(B * K, N, -1),
            predictor_union_mask.reshape(B * K, N),
        )
        predictor_output = predictor_output_flat.reshape(K, B, N, -1)

        loss_pred = predictor_output
        loss_target = target_token_target
        if self.normalize_jepa_targets:
            loss_pred = torch.nn.functional.normalize(loss_pred, dim=-1)
            loss_target = torch.nn.functional.normalize(loss_target, dim=-1)

        if self.masked_token_loss_type == "cosine":
            per_token_reg = 2.0 - 2.0 * (loss_pred * loss_target).sum(dim=-1)
        elif self.masked_token_loss_type == "l2":
            per_token_reg = (loss_pred - loss_target).square().mean(dim=-1)
        elif self.masked_token_loss_type == "l2_sum":
            per_token_reg = (loss_pred - loss_target).square().sum(dim=-1)
        elif self.masked_token_loss_type == "l1":
            per_token_reg = (loss_pred - loss_target).abs().mean(dim=-1)
        else:
            raise ValueError(
                f"Unsupported masked_token_loss_type: {self.masked_token_loss_type}"
            )
        target_mask_float = target_masks_by_view.float()
        reg_num = (per_token_reg * target_mask_float).sum()
        reg_den = target_mask_float.sum().clamp_min(1.0)
        local_global_loss = reg_num / reg_den

        if self.sigreg_lambda_warmup_steps > 0:
            sigreg_lambda_current = self.sigreg_lambda_current.to(
                dtype=context_emb.dtype
            )
        else:
            sigreg_lambda_current = context_emb.new_tensor(self.sigreg_lambda)

        jepa_term = self.masked_token_loss_weight * local_global_loss

        branch_emb = torch.cat([context_emb.unsqueeze(0), target_emb], dim=0)
        branch_visible = torch.cat(
            [context_mask.unsqueeze(0), target_masks_by_view], dim=0
        )
        V = branch_emb.shape[0]
        fused_emb = branch_emb.reshape(V * B, N, -1)
        fused_visible = branch_visible.reshape(V * B, N)

        with torch.no_grad():
            collapse_metrics = self._collapse_metrics(fused_emb, fused_visible, B, V)
            local_to_global_emb_std_ratio = (
                collapse_metrics["local_emb_std"] / collapse_metrics["global_emb_std"]
            )
            reg_stats = _masked_embedding_stats(fused_emb, fused_visible)
            gco_var_floor_constraint = (
                self.gco_var_floor_target - reg_stats["emb_var_floor"].float()
            )
            gco_corr_constraint = (
                reg_stats["emb_corr_offdiag_abs_mean"].float() - self.gco_corr_target
            )
            gco_constraint = torch.maximum(
                gco_var_floor_constraint,
                gco_corr_constraint,
            )

        gco_constraint_penalty = torch.relu(gco_constraint).to(dtype=context_emb.dtype)
        gco_lambda = self.gco_log_lambda.exp().to(dtype=context_emb.dtype)

        if self.representation_regularizer == "gco-sigreg":
            with torch.no_grad():
                if self.training:
                    self.gco_c_ema.mul_(self.gco_alpha).add_(
                        (1.0 - self.gco_alpha) * gco_constraint
                    )
                    self.gco_log_lambda.add_(self.gco_eta * self.gco_c_ema)
                    self.gco_log_lambda.clamp_(
                        self.gco_log_lambda_min, self.gco_log_lambda_max
                    )
                gco_lambda = self.gco_log_lambda.exp().to(dtype=context_emb.dtype)

        use_sigreg = (
            self.representation_regularizer == "sigreg" and self.sigreg_lambda > 0
        )
        use_gco = self.representation_regularizer == "gco-sigreg"
        if use_sigreg or use_gco:
            token_sigreg_loss = self.sigreg(fused_emb, valid_mask=fused_visible)
            if use_gco:
                sigreg_lambda_current = gco_lambda
            sigreg_term = sigreg_lambda_current * token_sigreg_loss
        else:
            token_sigreg_loss = context_emb.new_tensor(0.0)
            sigreg_term = context_emb.new_tensor(0.0)

        loss = jepa_term + sigreg_term
        valid_peak_count = peak_valid_mask.float().sum().clamp_min(1.0)

        _dt = context_emb.dtype
        return {
            "loss": loss,
            "token_sigreg_loss": token_sigreg_loss,
            "local_global_loss": local_global_loss,
            "sigreg_term": sigreg_term,
            "jepa_term": jepa_term,
            "target_sigreg_term_over_jepa_term": sigreg_term
            / jepa_term.clamp_min(1e-8),
            "context_fraction": context_mask.float().sum() / valid_peak_count,
            "masked_fraction": target_masks.float().sum() / valid_peak_count,
            "sigreg_lambda_current": sigreg_lambda_current,
            "gco_lambda": gco_lambda,
            "gco_log_lambda": self.gco_log_lambda.to(dtype=_dt),
            "gco_c_ema": self.gco_c_ema.to(dtype=_dt),
            "gco_constraint": gco_constraint.to(dtype=_dt),
            "gco_var_floor_constraint": gco_var_floor_constraint.to(dtype=_dt),
            "gco_corr_constraint": gco_corr_constraint.to(dtype=_dt),
            "gco_constraint_penalty": gco_constraint_penalty,
            **{f"encoder_{k}": v.to(dtype=_dt) for k, v in reg_stats.items()},
            "pool_norm_weight_abs_mean": self.pool_norm.weight.abs().mean(),
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
    ) -> dict[str, torch.Tensor]:
        emb = fused_emb.float().reshape(V, B, fused_emb.shape[1], fused_emb.shape[2])
        mask = visible_mask.reshape(V, B, visible_mask.shape[1])
        result: dict[str, torch.Tensor] = {}
        for prefix, e, m in [("global", emb[0], mask[0]), ("local", emb[1:], mask[1:])]:
            for k, v in _masked_embedding_stats(e, m).items():
                result[f"{prefix}_{k}"] = v
        return result

    def encode(
        self,
        batch: dict[str, torch.Tensor],
        *,
        train: bool = False,
    ) -> torch.Tensor:
        mz, intensity, valid = (
            batch["peak_mz"],
            batch["peak_intensity"],
            batch["peak_valid_mask"],
        )
        return self.pool(self.encoder(mz, intensity, valid_mask=valid), valid)
