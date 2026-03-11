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


def _build_non_causal_blocks(
    *,
    dim: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int | None,
    attention_mlp_multiple: float,
    norm_eps: float = 1e-5,
    qk_norm: bool = False,
    norm_type: str = "rmsnorm",
) -> nn.ModuleList:
    block_kwargs = dict(
        dim=dim,
        n_heads=int(num_heads),
        n_kv_heads=int(num_heads) if num_kv_heads is None else int(num_kv_heads),
        norm_eps=norm_eps,
        hidden_dim=int(math.ceil(dim * attention_mlp_multiple)),
        qk_norm=qk_norm,
        norm_type=norm_type,
    )
    return nn.ModuleList(
        [transformer_torch.TransformerBlock(**block_kwargs) for _ in range(num_layers)]
    )


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
    angles = positions.unsqueeze(-1) * inv_freq.to(device=device).view(1, 1, -1)
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
    centered = flat - (flat * weights_col).sum(0) / count
    cov = centered.transpose(0, 1) @ (centered * weights_col) / count
    var = cov.diagonal()
    vs = var.clamp_min(1e-12)
    corr = cov / torch.sqrt(vs.unsqueeze(0) * vs.unsqueeze(1))
    d = cov.shape[0] * (cov.shape[0] - 1)
    return {
        "emb_std": torch.sqrt(var + 1e-6).mean(),
        "emb_norm": (flat.norm(dim=-1) * weights).sum() / count,
        "emb_var_mean": var.mean(),
        "emb_var_floor": var.amin(),
        "emb_cov_offdiag_abs_mean": (cov.abs().sum() - cov.diagonal().abs().sum()) / d,
        "emb_corr_offdiag_abs_mean": (corr.abs().sum() - corr.diagonal().abs().sum())
        / d,
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
        self.embedder = nn.Sequential(
            nn.Linear(3, feature_mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(feature_mlp_hidden_dim, model_dim),
        )
        for layer in self.embedder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        _h = model_dim // int(num_heads) // 2
        self.register_buffer(
            "rope_inv_freq",
            1.0 / (10000.0 ** (torch.arange(_h, dtype=torch.float32) / _h)),
            persistent=False,
        )
        self.blocks = _build_non_causal_blocks(
            dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            attention_mlp_multiple=attention_mlp_multiple,
            qk_norm=qk_norm,
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
        log_intensity = torch.log1p(peak_intensity.clamp(min=0.0))
        x = self.embedder(torch.stack([peak_mz, peak_intensity, log_intensity], dim=-1))
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
        gco_constraints: list[dict] = (),
        gco_alpha: float = 0.99,
        gco_eta: float = 1e-3,
        gco_log_lambda_init: float = -8.0,
        gco_log_lambda_min: float = -12.0,
        gco_log_lambda_max: float = 2.0,
        jepa_num_target_blocks: int = 2,
        use_ema_teacher_target: bool = False,
        teacher_ema_decay: float = 0.996,
        teacher_ema_decay_start: float = 0.0,
        teacher_ema_decay_warmup_steps: int = 0,
        pooling_type: str = "pma",
        pma_num_heads: int | None = None,
        pma_num_seeds: int = 1,
        encoder_qk_norm: bool = False,
        norm_type: str = "rmsnorm",
        use_precursor_token: bool = False,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.use_precursor_token = bool(use_precursor_token)
        self.jepa_num_target_blocks = int(jepa_num_target_blocks)
        self.sigreg_lambda = float(sigreg_lambda)
        self.sigreg_lambda_warmup_steps = int(sigreg_lambda_warmup_steps)
        self.representation_regularizer = str(representation_regularizer).lower()
        if self.representation_regularizer == "gco":
            self.representation_regularizer = "gco-sigreg"
        if self.representation_regularizer not in ("sigreg", "gco-sigreg", "none", ""):
            raise ValueError(
                f"Unsupported regularizer: {self.representation_regularizer!r}"
            )
        self.gco_alpha = float(gco_alpha)
        self.gco_eta = float(gco_eta)
        self.gco_log_lambda_min = float(gco_log_lambda_min)
        self.gco_log_lambda_max = float(gco_log_lambda_max)
        # Configurable GCO constraints
        self.gco_constraint_keys: list[str] = []
        self.gco_constraint_signs: list[float] = []
        gco_targets: list[float] = []
        for c in gco_constraints:
            self.gco_constraint_keys.append(c["metric"])
            self.gco_constraint_signs.append(
                -1.0 if c["bound"] == "lower" else 1.0
            )
            gco_targets.append(float(c["target"]))
        _f = torch.float32
        _reg = self.register_buffer
        _reg(
            "gco_constraint_targets",
            torch.tensor(gco_targets, dtype=_f) if gco_targets else torch.empty(0, dtype=_f),
        )
        _reg("gco_log_lambda", torch.tensor(float(gco_log_lambda_init), dtype=_f))
        _reg("gco_c_ema", torch.tensor(0.0, dtype=_f))
        _sr_init = self.sigreg_lambda if self.sigreg_lambda_warmup_steps <= 0 else 0.0
        _reg("sigreg_lambda_target", torch.tensor(self.sigreg_lambda, dtype=_f))
        _reg("sigreg_lambda_current", torch.tensor(_sr_init, dtype=_f))
        _reg("sigreg_lambda_step", torch.zeros((), dtype=torch.int64))
        _reg(
            "sigreg_lambda_warmup_steps_tensor",
            torch.tensor(max(self.sigreg_lambda_warmup_steps, 1), dtype=_f),
            persistent=False,
        )
        # EMA teacher buffers
        teacher_ema_decay_start = float(teacher_ema_decay_start)
        teacher_ema_decay = float(teacher_ema_decay)
        teacher_ema_decay_warmup_steps = int(teacher_ema_decay_warmup_steps)
        _reg(
            "teacher_ema_decay_start_tensor",
            torch.tensor(teacher_ema_decay_start, dtype=_f),
            persistent=False,
        )
        _reg(
            "teacher_ema_decay_target",
            torch.tensor(teacher_ema_decay, dtype=_f),
            persistent=False,
        )
        _reg(
            "teacher_ema_decay_current",
            torch.tensor(
                teacher_ema_decay
                if teacher_ema_decay_warmup_steps <= 0
                else teacher_ema_decay_start,
                dtype=_f,
            ),
            persistent=False,
        )
        _reg(
            "teacher_ema_decay_step",
            torch.zeros((), dtype=torch.int64),
            persistent=False,
        )
        _reg(
            "teacher_ema_decay_warmup_steps_tensor",
            torch.tensor(max(teacher_ema_decay_warmup_steps, 1), dtype=_f),
            persistent=False,
        )
        self.pooling_type = pooling_type
        self.pma_num_seeds = int(pma_num_seeds)
        self.masked_token_loss_weight = float(masked_token_loss_weight)
        self.masked_token_loss_type = str(masked_token_loss_type).lower()
        self.norm_type = str(norm_type).lower()
        if self.jepa_num_target_blocks < 1:
            raise ValueError("jepa_num_target_blocks must be >= 1")
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
        # EMA teacher encoder
        if bool(use_ema_teacher_target):
            self.teacher_encoder: AveragedModel | None = AveragedModel(
                self.encoder,
                multi_avg_fn=get_ema_multi_avg_fn(teacher_ema_decay),
                use_buffers=True,
            )
            self.teacher_encoder.requires_grad_(False)
            self.teacher_encoder.eval()
        else:
            self.teacher_encoder = None
        self.pool_query = nn.Parameter(torch.empty(self.pma_num_seeds, model_dim))
        nn.init.xavier_normal_(self.pool_query)
        self.pool_mha = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=int(encoder_num_heads)
            if pma_num_heads is None
            else int(pma_num_heads),
            batch_first=True,
        )
        self.pool_norm = _build_norm(model_dim, eps=1e-5, norm_type=self.norm_type)
        self.latent_mask_token = nn.Parameter(torch.empty(self.model_dim))
        nn.init.normal_(self.latent_mask_token, std=0.02)
        pred_heads = max(1, min(int(encoder_num_heads), self.model_dim // 16))
        while self.model_dim % pred_heads != 0:
            pred_heads -= 1
        _ph = self.model_dim // pred_heads // 2
        self.register_buffer(
            "predictor_rope_inv_freq",
            1.0 / (10000.0 ** (torch.arange(_ph, dtype=torch.float32) / _ph)),
            persistent=False,
        )
        self.masked_latent_predictor = _build_non_causal_blocks(
            dim=self.model_dim,
            num_layers=int(masked_latent_predictor_num_layers),
            num_heads=pred_heads,
            num_kv_heads=None,
            attention_mlp_multiple=attention_mlp_multiple,
            qk_norm=encoder_qk_norm,
            norm_type=self.norm_type,
        )
        self.sigreg = SIGReg(num_slices=int(sigreg_num_slices))

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

    @torch.no_grad()
    def advance_teacher_ema_decay_schedule(self) -> None:
        step = self.teacher_ema_decay_step.to(
            dtype=self.teacher_ema_decay_current.dtype
        )
        ratio = torch.clamp(
            step / self.teacher_ema_decay_warmup_steps_tensor, max=1.0
        )
        delta = self.teacher_ema_decay_target - self.teacher_ema_decay_start_tensor
        self.teacher_ema_decay_current.copy_(
            self.teacher_ema_decay_start_tensor + delta * ratio
        )
        self.teacher_ema_decay_step.add_(1)

    @torch.no_grad()
    def advance_sigreg_lambda_schedule(self) -> None:
        if self.representation_regularizer != "sigreg":
            return
        if self.sigreg_lambda_warmup_steps <= 0:
            return
        step = self.sigreg_lambda_step.to(dtype=self.sigreg_lambda_current.dtype)
        ratio = torch.clamp(step / self.sigreg_lambda_warmup_steps_tensor, max=1.0)
        self.sigreg_lambda_current.copy_(self.sigreg_lambda_target * ratio)
        self.sigreg_lambda_step.add_(1)

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

    def pool(
        self,
        embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.pool_with_raw(embeddings, valid_mask)[0]

    def pool_with_raw(
        self,
        embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pooling_type == "mean":
            mask = valid_mask.unsqueeze(-1).float()
            pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            return pooled, pooled
        if self.pooling_type == "pma":
            pooled, _ = self.pool_mha(
                query=self.pool_query.unsqueeze(0).expand(embeddings.shape[0], -1, -1),
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
        pre_int = torch.full((B, 1), -1.0, device=device, dtype=peak_mz.dtype)
        pre_valid = torch.ones(B, 1, device=device, dtype=torch.bool)
        result: dict[str, torch.Tensor] = {
            "peak_mz": torch.cat([precursor_mz.unsqueeze(1), peak_mz], dim=1),
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
        target_emb = (
            self.encoder(
                peak_mz.repeat_interleave(K, dim=0),
                peak_intensity.repeat_interleave(K, dim=0),
                valid_mask=peak_valid_mask.repeat_interleave(K, dim=0),
                visible_mask=target_masks.reshape(B * K, N),
            )
            .reshape(B, K, N, -1)
            .permute(1, 0, 2, 3)
        )
        target_masks_by_view = target_masks.permute(1, 0, 2)
        if self.teacher_encoder is not None:
            with torch.no_grad():
                target_token_target = (
                    self.teacher_encoder(
                        peak_mz.repeat_interleave(K, dim=0),
                        peak_intensity.repeat_interleave(K, dim=0),
                        valid_mask=peak_valid_mask.repeat_interleave(K, dim=0),
                        visible_mask=target_masks.reshape(B * K, N),
                    )
                    .reshape(B, K, N, -1)
                    .permute(1, 0, 2, 3)
                    .detach()
                )
        else:
            target_token_target = target_emb.detach()

        ctx_mask_v = context_mask.unsqueeze(0)
        context_emb_by_view = context_emb.unsqueeze(0).expand(K, -1, -1, -1)
        predictor_input = context_emb_by_view * ctx_mask_v.unsqueeze(-1)
        predictor_input = torch.where(
            target_masks_by_view.unsqueeze(-1),
            self.latent_mask_token.view(1, 1, 1, -1).to(context_emb),
            predictor_input,
        )
        predictor_output = self.predict_masked_latents(
            predictor_input.reshape(B * K, N, -1),
            (ctx_mask_v | target_masks_by_view).reshape(B * K, N),
        ).reshape(K, B, N, -1)
        if self.masked_token_loss_type == "l2":
            per_token_reg = (
                (predictor_output - target_token_target).square().mean(dim=-1)
            )
        elif self.masked_token_loss_type == "l1":
            per_token_reg = (predictor_output - target_token_target).abs().mean(dim=-1)
        else:
            raise ValueError(
                f"Unsupported masked_token_loss_type: {self.masked_token_loss_type}"
            )
        target_mask_float = target_masks_by_view.float()
        reg_num = (per_token_reg * target_mask_float).sum()
        reg_den = target_mask_float.sum().clamp_min(1.0)
        local_global_loss = reg_num / reg_den
        sigreg_lambda_current = (
            self.sigreg_lambda_current.to(dtype=context_emb.dtype)
            if self.sigreg_lambda_warmup_steps > 0
            else context_emb.new_tensor(self.sigreg_lambda)
        )
        jepa_term = self.masked_token_loss_weight * local_global_loss
        branch_emb = torch.cat([context_emb.unsqueeze(0), target_emb], dim=0)
        branch_visible = torch.cat(
            [context_mask.unsqueeze(0), target_masks_by_view], dim=0
        )
        V = branch_emb.shape[0]
        fused_emb = branch_emb.reshape(V * B, N, -1)
        fused_visible = branch_visible.reshape(V * B, N)
        with torch.no_grad():
            emb_f = fused_emb.float().reshape(V, B, N, -1)
            mask_f = fused_visible.reshape(V, B, N)
            collapse_metrics: dict[str, torch.Tensor] = {}
            for prefix, e, m in [
                ("global", emb_f[0], mask_f[0]),
                ("local", emb_f[1:], mask_f[1:]),
            ]:
                for k, v in _masked_embedding_stats(e, m).items():
                    collapse_metrics[f"{prefix}_{k}"] = v
            reg_stats = _masked_embedding_stats(fused_emb, fused_visible)
            # Configurable GCO constraints — look up from both reg_stats and collapse_metrics
            all_stats = {**reg_stats, **collapse_metrics}
            if self.gco_constraint_keys:
                constraint_vals = torch.stack([
                    sign * (all_stats[key].float() - self.gco_constraint_targets[i])
                    for i, (key, sign) in enumerate(
                        zip(self.gco_constraint_keys, self.gco_constraint_signs)
                    )
                ])
                gco_constraint = constraint_vals.amax(dim=0)
            else:
                gco_constraint = torch.tensor(0.0, device=fused_emb.device)
                constraint_vals = None
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
        valid_peak_count = peak_valid_mask.float().sum().clamp_min(1.0)
        metrics = {
            "loss": jepa_term + sigreg_term,
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
            "gco_log_lambda": self.gco_log_lambda.to(dtype=context_emb.dtype),
            "gco_c_ema": self.gco_c_ema.to(dtype=context_emb.dtype),
            "gco_constraint": gco_constraint.to(dtype=context_emb.dtype),
            **{f"encoder_{k}": v.to(context_emb.dtype) for k, v in reg_stats.items()},
            "pool_norm_weight_abs_mean": self.pool_norm.weight.abs().mean(),
            "local_to_global_emb_std_ratio": collapse_metrics["local_emb_std"]
            / collapse_metrics["global_emb_std"],
            **collapse_metrics,
        }
        if constraint_vals is not None:
            for i, key in enumerate(self.gco_constraint_keys):
                metrics[f"gco_constraint_{key}"] = constraint_vals[i].to(
                    dtype=context_emb.dtype
                )
        return metrics

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        mz, intensity, valid = (
            batch["peak_mz"],
            batch["peak_intensity"],
            batch["peak_valid_mask"],
        )
        return self.pool(self.encoder(mz, intensity, valid_mask=valid), valid)
