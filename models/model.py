import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from models.losses import SIGReg
from networks import transformer_torch
from networks.transformer_torch import _build_norm, _rotate_half, create_visible_block_mask


def _apply_depth_scaled_init(blocks: nn.ModuleList, num_layers: int) -> None:
    """Scale residual output projections by 1/sqrt(2*num_layers) (GPT-2 style).

    In pre-norm transformers each residual addition contributes ~unit variance,
    so after 2*L sub-layers the activation norm grows by sqrt(2*L).  Scaling
    the output projections (wo in attention, w2 in FFN) keeps the total
    variance growth O(1) regardless of depth.
    """
    if num_layers <= 0:
        return
    scale = 1.0 / math.sqrt(2.0 * num_layers)
    for block in blocks:
        if hasattr(block, "attention"):
            block.attention.wo.weight.data.mul_(scale)
        if hasattr(block, "feed_forward"):
            block.feed_forward.w2.weight.data.mul_(scale)


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
    blocks = nn.ModuleList(
        [transformer_torch.TransformerBlock(**block_kwargs) for _ in range(num_layers)]
    )
    _apply_depth_scaled_init(blocks, num_layers)
    return blocks


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


class CrossAttention(nn.Module):
    """Cross-attention: Q from prediction queries, KV from source embeddings."""

    def __init__(self, dim: int, n_heads: int, *, n_kv_heads: int | None = None,
                 qk_norm: bool = False, norm_type: str = "rmsnorm"):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.head_dim = self.dim // self.n_heads
        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(self.dim, 2 * self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.dim, self.dim, bias=False)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = _build_norm(self.head_dim, eps=None, norm_type=norm_type)
            self.k_norm = _build_norm(self.head_dim, eps=None, norm_type=norm_type)
        nn.init.xavier_normal_(self.wq.weight)
        nn.init.xavier_normal_(self.wkv.weight)
        nn.init.xavier_normal_(self.wo.weight)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, *,
                freqs_cos: torch.Tensor | None = None,
                freqs_sin: torch.Tensor | None = None,
                memory_mask: torch.Tensor | None = None) -> torch.Tensor:
        bsz, tgt_len, _ = x.shape
        mem_len = memory.shape[1]
        xq = self.wq(x).view(bsz, tgt_len, self.n_heads, self.head_dim)
        kv = self.wkv(memory)
        xk, xv = kv.split(
            [self.n_kv_heads * self.head_dim, self.n_kv_heads * self.head_dim], dim=-1)
        xk = xk.view(bsz, mem_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, mem_len, self.n_kv_heads, self.head_dim)
        # RoPE on Q only (prediction positions); no RoPE on K
        if freqs_cos is not None and freqs_sin is not None:
            q_rot = _rotate_half(xq)
            xq = (xq * freqs_cos) + (q_rot * freqs_sin)
        if self.qk_norm:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)
        q = xq.transpose(1, 2)  # [B, H, T, D]
        k = xk.transpose(1, 2)  # [B, H, S, D]
        v = xv.transpose(1, 2)
        attn_mask = None
        if memory_mask is not None:
            # memory_mask: [B, S] -> [B, 1, 1, S]
            attn_mask = memory_mask[:, None, None, :].to(dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(attn_mask == 0, float("-inf")).masked_fill(attn_mask == 1, 0.0)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn = attn.transpose(1, 2).contiguous().view(bsz, tgt_len, self.dim)
        return self.wo(attn)


class TemporalDecoderBlock(nn.Module):
    """Decoder block: self-attention + cross-attention + FFN."""

    def __init__(self, *, dim: int, n_heads: int, n_kv_heads: int | None,
                 norm_eps: float, hidden_dim: int | None,
                 qk_norm: bool = False, norm_type: str = "rmsnorm"):
        super().__init__()
        self.attention = transformer_torch.Attention(
            dim, n_heads, n_kv_heads=n_kv_heads,
            qk_norm=qk_norm, norm_type=norm_type,
        )
        self.cross_attn = CrossAttention(dim, n_heads, n_kv_heads=n_kv_heads,
                                          qk_norm=qk_norm, norm_type=norm_type)
        self.feed_forward = transformer_torch.FeedForward(dim, hidden_dim=hidden_dim)
        self.attention_norm = _build_norm(dim, eps=None, norm_type=norm_type)
        self.cross_attn_norm = _build_norm(dim, eps=None, norm_type=norm_type)
        self.ffn_norm = _build_norm(dim, eps=None, norm_type=norm_type)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, *,
                freqs_cos: torch.Tensor | None,
                freqs_sin: torch.Tensor | None,
                memory_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cos=freqs_cos,
                               freqs_sin=freqs_sin)
        h = h + self.cross_attn(self.cross_attn_norm(h), memory,
                                freqs_cos=freqs_cos, freqs_sin=freqs_sin,
                                memory_mask=memory_mask)
        return h + self.feed_forward(self.ffn_norm(h))


def _apply_temporal_depth_scaled_init(blocks: nn.ModuleList, num_layers: int) -> None:
    """Depth-scaled init for temporal decoder blocks.

    Each block has 3 residual sub-layers (self-attn, cross-attn, FFN),
    so scale by 1/sqrt(3*num_layers).
    """
    if num_layers <= 0:
        return
    scale = 1.0 / math.sqrt(3.0 * num_layers)
    for block in blocks:
        block.attention.wo.weight.data.mul_(scale)
        block.cross_attn.wo.weight.data.mul_(scale)
        block.feed_forward.w2.weight.data.mul_(scale)


def _build_temporal_decoder_blocks(*, dim: int, num_layers: int, num_heads: int,
                                    num_kv_heads: int | None, attention_mlp_multiple: float,
                                    norm_eps: float = 1e-5, qk_norm: bool = False,
                                    norm_type: str = "rmsnorm") -> nn.ModuleList:
    block_kwargs = dict(
        dim=dim, n_heads=int(num_heads),
        n_kv_heads=int(num_heads) if num_kv_heads is None else int(num_kv_heads),
        norm_eps=norm_eps, hidden_dim=int(math.ceil(dim * attention_mlp_multiple)),
        qk_norm=qk_norm, norm_type=norm_type,
    )
    blocks = nn.ModuleList([TemporalDecoderBlock(**block_kwargs) for _ in range(num_layers)])
    _apply_temporal_depth_scaled_init(blocks, num_layers)
    return blocks


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
        self.final_norm = _build_norm(model_dim, eps=None, norm_type=norm_type)

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
        return self.final_norm(x)


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
        normalize_jepa_targets: bool = False,
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
        teacher_ema_update_every: int = 1,
        encoder_qk_norm: bool = False,
        norm_type: str = "rmsnorm",
        use_precursor_token: bool = False,
        num_peaks: int = 64,
        temporal_predictor_num_layers: int = 0,
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
        self.teacher_ema_update_every = int(teacher_ema_update_every)
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
        _reg("teacher_ema_update_step", torch.zeros((), dtype=torch.int64))
        self.masked_token_loss_weight = float(masked_token_loss_weight)
        self.masked_token_loss_type = str(masked_token_loss_type).lower()
        self.normalize_jepa_targets = bool(normalize_jepa_targets)
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
        self.predictor_final_norm = _build_norm(self.model_dim, eps=None, norm_type=self.norm_type)
        self.sigreg = SIGReg(num_slices=int(sigreg_num_slices))

        # Temporal predictor for frame -> next-frame prediction.
        N = int(num_peaks)
        self.temporal_predictor_num_layers = int(temporal_predictor_num_layers)
        if self.temporal_predictor_num_layers > 0:
            self.temporal_predictor = _build_temporal_decoder_blocks(
                dim=model_dim, num_layers=self.temporal_predictor_num_layers,
                num_heads=pred_heads, num_kv_heads=None,
                attention_mlp_multiple=attention_mlp_multiple,
                qk_norm=encoder_qk_norm, norm_type=self.norm_type,
            )
            self.temporal_rt_proj = nn.Sequential(
                nn.Linear(1, model_dim), nn.SiLU(), nn.Linear(model_dim, model_dim),
            )
            for layer in self.temporal_rt_proj:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    nn.init.zeros_(layer.bias)
            self.temporal_query_tokens = nn.Parameter(torch.empty(N, model_dim))
            nn.init.trunc_normal_(self.temporal_query_tokens, std=0.02)

    def train(self, mode: bool = True) -> "PeakSetSIGReg":
        super().train(mode)
        if self.teacher_encoder is not None:
            self.teacher_encoder.eval()
        return self

    @torch.no_grad()
    def update_teacher(self) -> None:
        if self.teacher_encoder is None:
            return
        step = int(self.teacher_ema_update_step.item())
        self.teacher_ema_update_step.add_(1)
        if step % self.teacher_ema_update_every != 0:
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
        return self.predictor_final_norm(x)

    def pool(
        self,
        embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = valid_mask.unsqueeze(-1).float()
        return (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

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
        )
        # Teacher sees full valid spectrum; loss mask selects target positions per block
        teacher = self.teacher_encoder if self.teacher_encoder is not None else self.encoder
        with torch.no_grad():
            teacher_full = teacher(
                peak_mz,
                peak_intensity,
                valid_mask=peak_valid_mask,
                visible_mask=peak_valid_mask,
            ).detach()
        target_token_target = teacher_full.unsqueeze(1).expand(-1, K, -1, -1)

        ctx_mask_v = context_mask.unsqueeze(1)
        context_emb_by_view = context_emb.unsqueeze(1).expand(-1, K, -1, -1)
        predictor_input = context_emb_by_view * ctx_mask_v.unsqueeze(-1)
        predictor_input = torch.where(
            target_masks.unsqueeze(-1),
            self.latent_mask_token.view(1, 1, 1, -1).to(context_emb),
            predictor_input,
        )
        predictor_output = self.predict_masked_latents(
            predictor_input.reshape(B * K, N, -1),
            (ctx_mask_v | target_masks).reshape(B * K, N),
        ).reshape(B, K, N, -1)
        loss_pred = predictor_output
        loss_target = target_token_target
        if self.normalize_jepa_targets:
            loss_pred = F.normalize(loss_pred, dim=-1)
            loss_target = F.normalize(loss_target, dim=-1)
        if self.masked_token_loss_type == "l2":
            per_token_reg = (
                (loss_pred - loss_target).square().mean(dim=-1)
            )
        elif self.masked_token_loss_type == "l2_sum":
            per_token_reg = (
                (loss_pred - loss_target).square().sum(dim=-1)
            )
        elif self.masked_token_loss_type == "l1":
            per_token_reg = (loss_pred - loss_target).abs().mean(dim=-1)
        else:
            raise ValueError(
                f"Unsupported masked_token_loss_type: {self.masked_token_loss_type}"
            )
        target_mask_float = target_masks.float()
        reg_num = (per_token_reg * target_mask_float).sum()
        reg_den = target_mask_float.sum().clamp_min(1.0)
        local_global_loss = reg_num / reg_den
        sigreg_lambda_current = (
            self.sigreg_lambda_current.to(dtype=context_emb.dtype)
            if self.sigreg_lambda_warmup_steps > 0
            else context_emb.new_tensor(self.sigreg_lambda)
        )
        jepa_term = self.masked_token_loss_weight * local_global_loss
        branch_emb = torch.cat([context_emb.unsqueeze(1), target_emb], dim=1)
        branch_visible = torch.cat(
            [context_mask.unsqueeze(1), target_masks], dim=1
        )
        V = branch_emb.shape[1]
        fused_emb = branch_emb.reshape(V * B, N, -1)
        fused_visible = branch_visible.reshape(V * B, N)
        with torch.no_grad():
            emb_f = fused_emb.float().reshape(B, V, N, -1)
            mask_f = fused_visible.reshape(B, V, N)
            collapse_metrics: dict[str, torch.Tensor] = {}
            for prefix, e, m in [
                ("global", emb_f[:, 0], mask_f[:, 0]),
                ("local", emb_f[:, 1:], mask_f[:, 1:]),
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

    @torch.no_grad()
    def compute_next_frame_teacher_embeddings(
        self, batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute teacher embeddings for the next frame."""
        teacher = self.teacher_encoder if self.teacher_encoder is not None else self.encoder
        return teacher(
            batch["next_frame_peak_mz"],
            batch["next_frame_peak_intensity"],
            valid_mask=batch["next_frame_peak_valid_mask"],
            visible_mask=batch["next_frame_peak_valid_mask"],
        )

    def forward_temporal(
        self,
        batch: dict[str, torch.Tensor],
        teacher_embeddings: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict next-frame token embeddings from the full current frame."""
        frame_mz = batch["frame_peak_mz"]
        frame_int = batch["frame_peak_intensity"]
        frame_valid = batch["frame_peak_valid_mask"]
        next_frame_mz = batch["next_frame_peak_mz"]
        next_frame_int = batch["next_frame_peak_intensity"]
        next_frame_valid = batch["next_frame_peak_valid_mask"]
        frame_rt = batch["frame_rt"]
        next_frame_rt = batch["next_frame_rt"]
        B = frame_mz.shape[0]

        frame_emb = self.encoder(
            frame_mz,
            frame_int,
            valid_mask=frame_valid,
            visible_mask=frame_valid,
        )  # [B, N, D]

        delta_rt = (next_frame_rt - frame_rt).unsqueeze(-1)  # [B, 1] in minutes
        rt_emb = self.temporal_rt_proj(delta_rt)  # [B, D]

        queries = self.temporal_query_tokens.unsqueeze(0).expand(B, -1, -1)
        queries = queries + rt_emb.unsqueeze(1)

        N_tgt = queries.shape[1]
        freqs_cos, freqs_sin = _compute_rope_freqs(
            True, N_tgt, self.predictor_rope_inv_freq, queries.device, queries.dtype
        )
        for block in self.temporal_predictor:
            queries = block(queries, frame_emb,
                            freqs_cos=freqs_cos, freqs_sin=freqs_sin,
                            memory_mask=frame_valid)
        predicted_next_frame = queries  # [B, N, D]

        if teacher_embeddings is not None:
            next_frame_emb = teacher_embeddings
        else:
            teacher = self.teacher_encoder if self.teacher_encoder is not None else self.encoder
            with torch.no_grad():
                next_frame_emb = teacher(
                    next_frame_mz,
                    next_frame_int,
                    valid_mask=next_frame_valid,
                    visible_mask=next_frame_valid,
                ).detach()

        if self.masked_token_loss_type == "l2":
            per_token = (predicted_next_frame - next_frame_emb).square().mean(dim=-1)
        elif self.masked_token_loss_type == "l1":
            per_token = (predicted_next_frame - next_frame_emb).abs().mean(dim=-1)
        else:
            per_token = (predicted_next_frame - next_frame_emb).square().mean(dim=-1)

        next_frame_mask = next_frame_valid.float()
        loss = (per_token * next_frame_mask).sum() / next_frame_mask.sum().clamp_min(1.0)

        return {
            "loss": loss,
            "next_frame_pred_loss": loss.detach(),
        }

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        mz, intensity, valid = (
            batch["peak_mz"],
            batch["peak_intensity"],
            batch["peak_valid_mask"],
        )
        return self.pool(self.encoder(mz, intensity, valid_mask=valid), valid)
