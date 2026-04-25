import math
import torch
import torch.nn.functional as F
from torch import nn

from models.losses import SIGReg, SlotwiseSIGReg
from models.peak_features import PeakFeatureEmbedder
from networks import transformer_torch
from networks.transformer_torch import _build_norm, create_visible_attention_mask
from utils.spectra_preprocessing import PRECURSOR_TOKEN_INTENSITY


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
    dropout: float = 0.0,
) -> nn.ModuleList:
    block_kwargs = dict(
        dim=dim,
        n_heads=int(num_heads),
        n_kv_heads=int(num_heads) if num_kv_heads is None else int(num_kv_heads),
        norm_eps=norm_eps,
        hidden_dim=int(math.ceil(dim * attention_mlp_multiple)),
        qk_norm=qk_norm,
        norm_type=norm_type,
        dropout=dropout,
    )
    blocks = nn.ModuleList(
        [transformer_torch.TransformerBlock(**block_kwargs) for _ in range(num_layers)]
    )
    _apply_depth_scaled_init(blocks, num_layers)
    return blocks


def _build_sincos_position_table(num_positions: int, dim: int) -> torch.Tensor:
    half_dim = dim // 2
    positions = torch.arange(num_positions, dtype=torch.float32).unsqueeze(1)
    if half_dim == 0:
        return torch.zeros(num_positions, dim, dtype=torch.float32)
    scales = torch.exp(
        -math.log(10000.0) * torch.arange(half_dim, dtype=torch.float32) / half_dim
    )
    angles = positions * scales.unsqueeze(0)
    table = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if dim % 2 == 1:
        table = F.pad(table, (0, 1))
    return table


def _build_frozen_position_embedding(num_positions: int, dim: int) -> nn.Embedding:
    embedding = nn.Embedding(num_positions, dim)
    with torch.no_grad():
        embedding.weight.copy_(_build_sincos_position_table(num_positions, dim))
    embedding.weight.requires_grad_(False)
    return embedding


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


def _merge_visible_mask(
    valid_mask: torch.Tensor | None,
    visible_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    if visible_mask is not None and valid_mask is not None:
        return visible_mask & valid_mask
    return visible_mask if visible_mask is not None else valid_mask


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
                memory_mask: torch.Tensor | None = None) -> torch.Tensor:
        bsz, tgt_len, _ = x.shape
        mem_len = memory.shape[1]
        xq = self.wq(x).view(bsz, tgt_len, self.n_heads, self.head_dim)
        kv = self.wkv(memory)
        xk, xv = kv.split(
            [self.n_kv_heads * self.head_dim, self.n_kv_heads * self.head_dim], dim=-1)
        xk = xk.view(bsz, mem_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, mem_len, self.n_kv_heads, self.head_dim)
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
                memory_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x))
        h = h + self.cross_attn(self.cross_attn_norm(h), memory,
                                memory_mask=memory_mask)
        return h + self.feed_forward(self.ffn_norm(h))


class CovariancePool(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        compressed_dim: int,
    ) -> None:
        super().__init__()
        self.left_proj = nn.Linear(input_dim, compressed_dim, bias=False)
        self.right_proj = nn.Linear(input_dim, compressed_dim, bias=False)
        nn.init.xavier_normal_(self.left_proj.weight)
        nn.init.xavier_normal_(self.right_proj.weight)

    def forward(
        self,
        peak_embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = valid_mask.unsqueeze(-1).to(dtype=peak_embeddings.dtype)
        left = self.left_proj(peak_embeddings) * mask
        right = self.right_proj(peak_embeddings) * mask
        denom = mask.sum(dim=1).clamp(min=1.0)
        covariance = left.transpose(1, 2) @ right
        covariance = covariance / denom.unsqueeze(-1)
        return covariance.flatten(start_dim=1)


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
        fourier_strategy: str = "log_spaced",
        fourier_x_min: float = 3e-3,
        fourier_x_max: float = 1000.0,
        fourier_funcs: str = "both",
        fourier_num_freqs: int = 256,
        fourier_sigma: float = 10.0,
        fourier_trainable: bool = False,
        fourier_input_scale: float = 1000.0,
        qk_norm: bool = False,
        norm_type: str = "rmsnorm",
        apply_final_norm: bool = True,
        num_peaks: int = 64,
        use_position_embedding: bool = True,
        num_register_tokens: int = 0,
    ):
        super().__init__()
        self.num_layers = int(num_layers)
        norm_type = str(norm_type).lower()
        self.num_register_tokens = int(num_register_tokens)
        self.use_position_embedding = bool(use_position_embedding)
        self.embedder = PeakFeatureEmbedder(
            model_dim=model_dim,
            hidden_dim=feature_mlp_hidden_dim,
            fourier_strategy=fourier_strategy,
            fourier_x_min=fourier_x_min,
            fourier_x_max=fourier_x_max,
            fourier_funcs=fourier_funcs,
            fourier_num_freqs=fourier_num_freqs,
            fourier_sigma=fourier_sigma,
            fourier_trainable=fourier_trainable,
            fourier_input_scale=fourier_input_scale,
        )
        self.position_embedding = _build_frozen_position_embedding(
            int(num_peaks),
            model_dim,
        )
        self.cls_token = nn.Parameter(torch.empty(model_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.num_register_tokens > 0:
            self.register_tokens = nn.Parameter(
                torch.empty(self.num_register_tokens, model_dim)
            )
            nn.init.trunc_normal_(self.register_tokens, std=0.02)
        else:
            self.register_tokens = None
        self.blocks = _build_non_causal_blocks(
            dim=model_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            attention_mlp_multiple=attention_mlp_multiple,
            qk_norm=qk_norm,
            norm_type=norm_type,
        )
        self.final_norm = (
            _build_norm(model_dim, eps=None, norm_type=norm_type)
            if apply_final_norm
            else nn.Identity()
        )

    def _add_positions(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_position_embedding:
            return x
        return x + self.position_embedding(
            torch.arange(x.shape[1], device=x.device)
        ).unsqueeze(0).to(dtype=x.dtype)

    def _append_special_tokens(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        cls = self.cls_token.view(1, 1, -1).expand(x.shape[0], -1, -1).to(dtype=x.dtype)
        if self.register_tokens is None:
            special = cls
        else:
            registers = self.register_tokens.unsqueeze(0).expand(x.shape[0], -1, -1)
            special = torch.cat([cls, registers.to(dtype=x.dtype)], dim=1)
        x = torch.cat([x, special], dim=1)
        if attn_mask is None:
            return x, None
        special_mask = torch.ones(
            x.shape[0],
            special.shape[1],
            device=x.device,
            dtype=torch.bool,
        )
        return x, torch.cat([attn_mask, special_mask], dim=1)

    @staticmethod
    def split_peak_and_cls(
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return x[:, :-1], x[:, -1]

    def forward_with_block_outputs(
        self,
        peak_mz: torch.Tensor,
        peak_intensity: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        visible_mask: torch.Tensor | None = None,
        block_indices: list[int] | tuple[int, ...] = (),
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        block_indices = tuple(int(idx) for idx in block_indices)
        attn_mask = _merge_visible_mask(valid_mask, visible_mask)
        x = self._add_positions(self.embedder(peak_mz, peak_intensity))
        seq_len = peak_mz.shape[1]
        selected = set(block_indices)
        selected_peak_outputs: dict[int, torch.Tensor] = {}
        x, attn_mask = self._append_special_tokens(x, attn_mask)
        attn_mask = (
            create_visible_attention_mask(attn_mask) if attn_mask is not None else None
        )
        for block_idx, block in enumerate(self.blocks, start=1):
            x = block(
                x,
                attn_mask=attn_mask,
            )
            if block_idx in selected and block_idx != self.num_layers:
                selected_peak_outputs[block_idx] = x[:, :seq_len]
        x = self.final_norm(x)
        if self.num_layers in selected:
            selected_peak_outputs[self.num_layers] = x[:, :seq_len]
        peak_x = x[:, :seq_len]
        cls_x = x[:, seq_len]
        output = torch.cat([peak_x, cls_x.unsqueeze(1)], dim=1)
        return output, [selected_peak_outputs[idx] for idx in block_indices]

    def forward_peak_block_outputs(
        self,
        peak_mz: torch.Tensor,
        peak_intensity: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        visible_mask: torch.Tensor | None = None,
        block_indices: list[int] | tuple[int, ...] = (),
    ) -> list[torch.Tensor]:
        _, peak_block_outputs = self.forward_with_block_outputs(
            peak_mz,
            peak_intensity,
            valid_mask=valid_mask,
            visible_mask=visible_mask,
            block_indices=block_indices,
        )
        return peak_block_outputs

    def forward(
        self,
        peak_mz: torch.Tensor,
        peak_intensity: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        visible_mask: torch.Tensor | None = None,
        return_cls_token: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        output, _ = self.forward_with_block_outputs(
            peak_mz,
            peak_intensity,
            valid_mask=valid_mask,
            visible_mask=visible_mask,
        )
        if return_cls_token:
            return output, output[:, -1]
        return output


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
        encoder_fourier_strategy: str = "log_spaced",
        encoder_fourier_x_min: float = 3e-3,
        encoder_fourier_x_max: float = 1000.0,
        encoder_fourier_funcs: str = "both",
        encoder_fourier_num_freqs: int = 256,
        encoder_fourier_sigma: float = 10.0,
        encoder_fourier_trainable: bool = False,
        encoder_fourier_input_scale: float = 1000.0,
        masked_token_loss_weight: float = 0.0,
        masked_token_loss_type: str = "l1",
        jepa_target_normalization: str = "none",
        jepa_target_layers: list[int] | tuple[int, ...] | None = None,
        representation_regularizer: str = "none",
        masked_latent_predictor_num_layers: int = 2,
        masked_latent_predictor_num_heads: int = 8,
        sigreg_num_slices: int = 256,
        sigreg_lambda: float = 0.02,
        sigreg_precursor_scale: float = 1.0,
        jepa_num_target_blocks: int = 2,
        jepa_context_fraction: float = 0.5,
        jepa_target_fraction: float = 0.25,
        encoder_qk_norm: bool = False,
        norm_type: str = "rmsnorm",
        encoder_use_position_embedding: bool = True,
        encoder_apply_final_norm: bool = True,
        predictor_apply_final_norm: bool = True,
        use_precursor_token: bool = False,
        num_peaks: int = 64,
        temporal_predictor_num_layers: int = 0,
        encoder_num_register_tokens: int = 0,
        predictor_num_register_tokens: int = 0,
        predictor_dim: int | None = None,
        target_projector_dim: int | None = None,
        predictor_dropout: float = 0.0,
        train_covariance_pooling: bool = False,
        covariance_pooling_dim: int = 32,
        covariance_pooling_sigreg_lambda: float | None = None,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.predictor_dim = predictor_dim if predictor_dim is not None else model_dim
        self.target_projector_dim = (
            int(target_projector_dim)
            if target_projector_dim is not None
            else self.model_dim
        )
        self.encoder_num_layers = int(encoder_num_layers)
        self.use_precursor_token = bool(use_precursor_token)
        self.jepa_num_target_blocks = int(jepa_num_target_blocks)
        self.jepa_target_layers = (
            [self.encoder_num_layers]
            if jepa_target_layers is None
            else [int(layer_idx) for layer_idx in jepa_target_layers]
        )
        if not self.jepa_target_layers:
            raise ValueError("jepa_target_layers must not be empty")
        if min(self.jepa_target_layers) < 1 or max(self.jepa_target_layers) > self.encoder_num_layers:
            raise ValueError("jepa_target_layers must be within encoder depth")
        self.num_jepa_target_layers = len(self.jepa_target_layers)
        self.jepa_target_dim = self.num_jepa_target_layers * self.model_dim
        self.representation_regularizer = str(representation_regularizer).lower()
        if self.representation_regularizer == "sigreg":
            self.representation_regularizer = "sigreg-enc"
        if self.representation_regularizer == "slog-sigreg-pred":
            self.representation_regularizer = "slot-sigreg-pred"
        if self.representation_regularizer == "slog-sigreg-proj":
            self.representation_regularizer = "slot-sigreg-proj"
        if self.representation_regularizer not in (
            "none",
            "",
            "sigreg-enc",
            "sigreg-pred",
            "sigreg-proj",
            "slot-sigreg-enc",
            "slot-sigreg-pred",
            "slot-sigreg-proj",
        ):
            raise ValueError(
                f"Unsupported regularizer: {self.representation_regularizer!r}"
            )
        self.sigreg_lambda = float(sigreg_lambda)
        self.sigreg_precursor_scale = float(sigreg_precursor_scale)
        self.train_covariance_pooling = bool(train_covariance_pooling)
        self.covariance_pooling_sigreg_lambda = (
            float(covariance_pooling_sigreg_lambda)
            if covariance_pooling_sigreg_lambda is not None
            else float(sigreg_lambda)
        )
        self.masked_token_loss_weight = float(masked_token_loss_weight)
        self.masked_token_loss_type = str(masked_token_loss_type).lower()
        self.jepa_target_normalization = str(jepa_target_normalization).lower()
        if self.jepa_target_normalization not in ("none", "zscore"):
            raise ValueError(
                "jepa_target_normalization must be one of ('none', 'zscore')"
            )
        self.norm_type = str(norm_type).lower()
        self.temporal_predictor_num_layers = int(temporal_predictor_num_layers)
        self.predictor_num_register_tokens = int(predictor_num_register_tokens)
        if self.jepa_num_target_blocks < 1:
            raise ValueError("jepa_num_target_blocks must be >= 1")
        N = int(num_peaks) + int(self.use_precursor_token)
        self.encoder = PeakSetEncoder(
            model_dim=model_dim,
            num_layers=self.encoder_num_layers,
            num_heads=encoder_num_heads,
            num_kv_heads=encoder_num_kv_heads,
            attention_mlp_multiple=attention_mlp_multiple,
            feature_mlp_hidden_dim=feature_mlp_hidden_dim,
            fourier_strategy=encoder_fourier_strategy,
            fourier_x_min=encoder_fourier_x_min,
            fourier_x_max=encoder_fourier_x_max,
            fourier_funcs=encoder_fourier_funcs,
            fourier_num_freqs=encoder_fourier_num_freqs,
            fourier_sigma=encoder_fourier_sigma,
            fourier_trainable=encoder_fourier_trainable,
            fourier_input_scale=encoder_fourier_input_scale,
            qk_norm=encoder_qk_norm,
            norm_type=self.norm_type,
            use_position_embedding=encoder_use_position_embedding,
            apply_final_norm=encoder_apply_final_norm,
            num_peaks=N,
            num_register_tokens=encoder_num_register_tokens,
        )
        self.latent_mask_token = nn.Parameter(torch.empty(self.model_dim))
        nn.init.normal_(self.latent_mask_token, std=0.02)

        if self.predictor_dim != self.model_dim:
            self.encoder_to_predictor_proj = nn.Linear(
                self.model_dim, self.predictor_dim, bias=False,
            )
            nn.init.xavier_normal_(self.encoder_to_predictor_proj.weight)
        else:
            self.encoder_to_predictor_proj = nn.Identity()

        self.predictor_position_embedding = _build_frozen_position_embedding(
            N,
            self.model_dim,
        )
        if self.predictor_num_register_tokens > 0:
            self.predictor_register_tokens = nn.Parameter(
                torch.empty(self.predictor_num_register_tokens, self.model_dim)
            )
            nn.init.trunc_normal_(self.predictor_register_tokens, std=0.02)
        else:
            self.predictor_register_tokens = None
        self.masked_latent_predictor = _build_non_causal_blocks(
            dim=self.predictor_dim,
            num_layers=int(masked_latent_predictor_num_layers),
            num_heads=int(masked_latent_predictor_num_heads),
            num_kv_heads=None,
            attention_mlp_multiple=attention_mlp_multiple,
            qk_norm=encoder_qk_norm,
            norm_type=self.norm_type,
            dropout=predictor_dropout,
        )
        self.predictor_final_norm = (
            _build_norm(self.predictor_dim, eps=None, norm_type=self.norm_type)
            if predictor_apply_final_norm
            else nn.Identity()
        )
        self.masked_latent_readout = nn.Linear(self.predictor_dim, self.jepa_target_dim)
        nn.init.xavier_normal_(self.masked_latent_readout.weight)
        nn.init.zeros_(self.masked_latent_readout.bias)
        self.target_projector = nn.Sequential(
            nn.Linear(self.jepa_target_dim, self.jepa_target_dim),
            nn.GELU(),
            nn.Linear(self.jepa_target_dim, self.target_projector_dim),
        )
        for layer in self.target_projector:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        sigreg_cls = (
            SlotwiseSIGReg
            if self.representation_regularizer
            in ("slot-sigreg-enc", "slot-sigreg-pred", "slot-sigreg-proj")
            else SIGReg
        )
        self.sigreg = sigreg_cls(num_slices=int(sigreg_num_slices))
        if self.train_covariance_pooling:
            self.covariance_pooler = CovariancePool(
                input_dim=self.model_dim,
                compressed_dim=int(covariance_pooling_dim),
            )
            self.covariance_sigreg = SIGReg(num_slices=int(sigreg_num_slices))
        # Temporal predictor for frame -> next-frame prediction.
        if self.temporal_predictor_num_layers > 0:
            self.temporal_predictor = _build_temporal_decoder_blocks(
                dim=model_dim, num_layers=self.temporal_predictor_num_layers,
                num_heads=int(masked_latent_predictor_num_heads), num_kv_heads=None,
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
            self.temporal_query_token = nn.Parameter(torch.empty(model_dim))
            nn.init.trunc_normal_(self.temporal_query_token, std=0.02)

    def _apply_group_target_normalization(
        self,
        x: torch.Tensor,
        group_dim: int,
    ) -> torch.Tensor:
        if self.jepa_target_normalization == "none":
            return x
        x = x.reshape(*x.shape[:-1], -1, group_dim)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        return ((x - mean) / std).reshape(*x.shape[:-2], -1)

    def _apply_jepa_target_normalization(self, x: torch.Tensor) -> torch.Tensor:
        return self._apply_group_target_normalization(x, self.model_dim)

    def _append_predictor_register_tokens(
        self,
        x: torch.Tensor,
        visible_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.predictor_register_tokens is None:
            return x, visible_mask
        registers = self.predictor_register_tokens.unsqueeze(0).expand(x.shape[0], -1, -1)
        x = torch.cat([x, registers.to(dtype=x.dtype)], dim=1)
        if visible_mask is None:
            return x, None
        register_mask = torch.ones(
            x.shape[0],
            self.predictor_num_register_tokens,
            device=x.device,
            dtype=torch.bool,
        )
        return x, torch.cat([visible_mask, register_mask], dim=1)

    def _add_predictor_positions(self, x: torch.Tensor) -> torch.Tensor:
        # Real predictor/query slots get absolute positions; register tokens are
        # appended later and stay unpositioned.
        return x + self.predictor_position_embedding(
            torch.arange(x.shape[1], device=x.device)
        ).unsqueeze(0).to(dtype=x.dtype)

    def predict_masked_latents(
        self,
        x: torch.Tensor,
        visible_mask: torch.Tensor,
    ) -> torch.Tensor:
        if len(self.masked_latent_predictor) == 0:
            return x
        x = self._add_predictor_positions(x)
        x, visible_mask = self._append_predictor_register_tokens(x, visible_mask)
        x = self.encoder_to_predictor_proj(x)
        predictor_attn_mask = create_visible_attention_mask(visible_mask)
        for block in self.masked_latent_predictor:
            x = block(
                x,
                attn_mask=predictor_attn_mask,
            )
        x = self.predictor_final_norm(x)
        if self.predictor_num_register_tokens > 0:
            x = x[:, :-self.predictor_num_register_tokens]
        return x

    def project_targets(self, x: torch.Tensor) -> torch.Tensor:
        return self.target_projector(x)

    def predict_masked_target_features(
        self,
        x: torch.Tensor,
        visible_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.masked_latent_readout(
            self.predict_masked_latents(
                x,
                visible_mask,
            )
        )

    def predict_masked_targets(
        self,
        x: torch.Tensor,
        visible_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.project_targets(
            self.predict_masked_target_features(
                x,
                visible_mask,
            )
        )

    def _compute_jepa_teacher_target_features(
        self,
        peak_mz: torch.Tensor,
        peak_intensity: torch.Tensor,
        peak_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        amp_dtype = (
            torch.get_autocast_dtype("cuda")
            if torch.is_autocast_enabled("cuda")
            else torch.bfloat16
        )
        with torch.autocast("cuda", dtype=amp_dtype):
            teacher_peak_outputs = self.encoder.forward_peak_block_outputs(
                peak_mz,
                peak_intensity,
                valid_mask=peak_valid_mask,
                visible_mask=peak_valid_mask,
                block_indices=self.jepa_target_layers,
            )
            return torch.cat(teacher_peak_outputs, dim=-1)

    def _compute_jepa_teacher_targets(
        self,
        peak_mz: torch.Tensor,
        peak_intensity: torch.Tensor,
        peak_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            teacher_target_features = self._compute_jepa_teacher_target_features(
                peak_mz,
                peak_intensity,
                peak_valid_mask,
            )
            return self.project_targets(
                self._apply_jepa_target_normalization(teacher_target_features)
            )

    def compute_teacher_targets(
        self,
        augmented_batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self._compute_jepa_teacher_targets(
            augmented_batch["peak_mz"],
            augmented_batch["peak_intensity"],
            augmented_batch["peak_valid_mask"],
        )

    def _encode_augmented_teacher_and_context(
        self,
        peak_mz: torch.Tensor,
        peak_intensity: torch.Tensor,
        peak_valid_mask: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = peak_mz.shape[0]
        encoded, teacher_peak_outputs = self.encoder.forward_with_block_outputs(
            torch.cat([peak_mz, peak_mz], dim=0),
            torch.cat([peak_intensity, peak_intensity], dim=0),
            valid_mask=torch.cat([peak_valid_mask, peak_valid_mask], dim=0),
            visible_mask=torch.cat([peak_valid_mask, context_mask], dim=0),
            block_indices=self.jepa_target_layers,
        )
        teacher_target_features = torch.cat(
            [peak_output[:batch_size] for peak_output in teacher_peak_outputs],
            dim=-1,
        )
        teacher_peak_emb, _ = self.encoder.split_peak_and_cls(encoded[:batch_size])
        context_emb, _ = self.encoder.split_peak_and_cls(encoded[batch_size:])
        return teacher_target_features, teacher_peak_emb, context_emb

    def _compute_pooled_teacher_peak_targets(
        self,
        peak_mz: torch.Tensor,
        peak_intensity: torch.Tensor,
        peak_valid_mask: torch.Tensor,
        visible_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if visible_mask is None:
            visible_mask = peak_valid_mask
        amp_dtype = (
            torch.get_autocast_dtype("cuda")
            if torch.is_autocast_enabled("cuda")
            else torch.bfloat16
        )
        with torch.autocast("cuda", dtype=amp_dtype):
            teacher_encoded = self.encoder(
                peak_mz,
                peak_intensity,
                valid_mask=peak_valid_mask,
                visible_mask=visible_mask,
            )
            teacher_peak_emb, _ = self.encoder.split_peak_and_cls(teacher_encoded)
        return self.pool(teacher_peak_emb, visible_mask)

    def _embedding_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if self.masked_token_loss_type == "l2":
            return (prediction - target).square().mean(dim=-1)
        if self.masked_token_loss_type == "l2_sum":
            return (prediction - target).square().sum(dim=-1)
        if self.masked_token_loss_type == "l1":
            return (prediction - target).abs().mean(dim=-1)
        raise ValueError(
            f"Unsupported masked_token_loss_type: {self.masked_token_loss_type}"
        )

    def pool(
        self,
        embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if embeddings.shape[1] == valid_mask.shape[1] + 1:
            embeddings = embeddings[:, :-1]
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
        B = peak_mz.shape[0]
        device = peak_mz.device
        pre_int = torch.full(
            (B, 1),
            PRECURSOR_TOKEN_INTENSITY,
            device=device,
            dtype=peak_mz.dtype,
        )
        pre_valid = torch.ones(B, 1, device=device, dtype=torch.bool)
        result: dict[str, torch.Tensor] = {
            "peak_mz": torch.cat([precursor_mz.unsqueeze(1), peak_mz], dim=1),
            "peak_intensity": torch.cat([pre_int, peak_intensity], dim=1),
            "peak_valid_mask": torch.cat([pre_valid, peak_valid_mask], dim=1),
        }
        if context_mask is not None:
            pre_ctx = torch.zeros(B, 1, device=device, dtype=torch.bool)
            result["context_mask"] = torch.cat([pre_ctx, context_mask], dim=1)
        if target_masks is not None:
            K = target_masks.shape[1]
            pre_tgt = torch.zeros(B, K, 1, device=device, dtype=torch.bool)
            result["target_masks"] = torch.cat([pre_tgt, target_masks], dim=2)
        return result

    def _get_temporal_frame_inputs(
        self,
        batch: dict[str, torch.Tensor],
        prefix: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        peak_mz = batch[f"{prefix}_peak_mz"]
        peak_intensity = batch[f"{prefix}_peak_intensity"]
        peak_valid_mask = batch[f"{prefix}_peak_valid_mask"]
        if not self.use_precursor_token:
            return peak_mz, peak_intensity, peak_valid_mask
        with_precursor = self.prepend_precursor_token(
            peak_mz,
            peak_intensity,
            peak_valid_mask,
            batch[f"{prefix}_precursor_mz"],
        )
        return (
            with_precursor["peak_mz"],
            with_precursor["peak_intensity"],
            with_precursor["peak_valid_mask"],
        )

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
        (
            teacher_target_features,
            teacher_peak_emb,
            context_emb,
        ) = self._encode_augmented_teacher_and_context(
            peak_mz,
            peak_intensity,
            peak_valid_mask,
            context_mask,
        )
        ctx_mask_v = context_mask.unsqueeze(1)
        context_emb_by_view = context_emb.unsqueeze(1).expand(-1, K, -1, -1)
        predictor_input = context_emb_by_view * ctx_mask_v.unsqueeze(-1)
        predictor_input = torch.where(
            target_masks.unsqueeze(-1),
            self.latent_mask_token.view(1, 1, 1, -1).to(context_emb),
            predictor_input,
        )
        predictor_output_features = self.predict_masked_target_features(
            predictor_input.reshape(B * K, N, -1),
            (ctx_mask_v | target_masks).reshape(B * K, N),
        ).reshape(B, K, N, -1)
        predictor_output = self.project_targets(predictor_output_features)
        with torch.no_grad():
            teacher_targets = self.project_targets(
                self._apply_jepa_target_normalization(teacher_target_features.detach())
            )
        loss_target = teacher_targets.unsqueeze(1)
        per_token_reg = self._embedding_loss(predictor_output, loss_target)

        cls_loss_weight = context_emb.new_tensor(0.0)
        cls_embedding_loss = context_emb.new_tensor(0.0)
        cls_embedding_term = context_emb.new_tensor(0.0)
        cls_visible_mask = torch.zeros_like(context_mask)
        target_mask_float = target_masks.float()
        reg_num = (per_token_reg * target_mask_float).sum()
        reg_den = target_mask_float.sum().clamp_min(1.0)
        local_global_loss = reg_num / reg_den
        jepa_term = self.masked_token_loss_weight * local_global_loss
        use_sigreg_enc = (
            self.representation_regularizer in ("sigreg-enc", "slot-sigreg-enc")
            and self.sigreg_lambda > 0
        )
        use_sigreg_pred = (
            self.representation_regularizer in ("sigreg-pred", "slot-sigreg-pred")
            and self.sigreg_lambda > 0
        )
        use_sigreg_proj = (
            self.representation_regularizer in ("sigreg-proj", "slot-sigreg-proj")
            and self.sigreg_lambda > 0
        )
        regularizer_lambda_current = context_emb.new_tensor(0.0)
        regularizer_loss = context_emb.new_tensor(0.0)
        regularizer_term = context_emb.new_tensor(0.0)
        sigreg_lambda_current = context_emb.new_tensor(0.0)
        token_sigreg_loss = context_emb.new_tensor(0.0)
        context_token_sigreg_loss = context_emb.new_tensor(0.0)
        teacher_token_sigreg_loss = context_emb.new_tensor(0.0)
        pred_token_sigreg_loss = context_emb.new_tensor(0.0)
        projected_student_token_sigreg_loss = context_emb.new_tensor(0.0)
        projected_teacher_token_sigreg_loss = context_emb.new_tensor(0.0)
        sigreg_term = context_emb.new_tensor(0.0)
        context_sigreg_term = context_emb.new_tensor(0.0)
        teacher_sigreg_term = context_emb.new_tensor(0.0)
        pred_sigreg_term = context_emb.new_tensor(0.0)
        projected_student_sigreg_term = context_emb.new_tensor(0.0)
        projected_teacher_sigreg_term = context_emb.new_tensor(0.0)
        covariance_pooling_sigreg_loss = context_emb.new_tensor(0.0)
        covariance_pooling_sigreg_term = context_emb.new_tensor(0.0)
        covariance_pooling_sigreg_lambda_current = context_emb.new_tensor(0.0)
        if use_sigreg_enc:
            sigreg_lambda_current = context_emb.new_tensor(self.sigreg_lambda)
            context_sigreg_weights = context_mask.float()
            if self.use_precursor_token:
                context_sigreg_weights = context_sigreg_weights.clone()
                context_sigreg_weights[..., 0] *= self.sigreg_precursor_scale
            context_token_sigreg_loss = self.sigreg(
                context_emb.float(),
                valid_mask=context_sigreg_weights,
            )
            token_sigreg_loss = context_token_sigreg_loss
            context_sigreg_term = sigreg_lambda_current * context_token_sigreg_loss.to(
                dtype=context_emb.dtype
            )
            sigreg_term = context_sigreg_term
            regularizer_lambda_current = sigreg_lambda_current
            regularizer_loss = token_sigreg_loss.to(dtype=context_emb.dtype)
            regularizer_term = sigreg_term
        elif use_sigreg_pred:
            sigreg_lambda_current = context_emb.new_tensor(self.sigreg_lambda)
            sigreg_weights = target_masks.float()
            if self.use_precursor_token:
                sigreg_weights = sigreg_weights.clone()
                sigreg_weights[..., 0] *= self.sigreg_precursor_scale
            pred_token_sigreg_loss = self.sigreg(
                predictor_output_features.float(),
                valid_mask=sigreg_weights,
            )
            token_sigreg_loss = pred_token_sigreg_loss
            pred_sigreg_term = sigreg_lambda_current * pred_token_sigreg_loss.to(
                dtype=context_emb.dtype
            )
            sigreg_term = pred_sigreg_term
            regularizer_lambda_current = sigreg_lambda_current
            regularizer_loss = token_sigreg_loss.to(dtype=context_emb.dtype)
            regularizer_term = sigreg_term
        elif use_sigreg_proj:
            sigreg_lambda_current = context_emb.new_tensor(self.sigreg_lambda)
            student_sigreg_weights = target_masks.float()
            if self.use_precursor_token:
                student_sigreg_weights = student_sigreg_weights.clone()
                student_sigreg_weights[..., 0] *= self.sigreg_precursor_scale
            projected_student_token_sigreg_loss = self.sigreg(
                predictor_output.float(),
                valid_mask=student_sigreg_weights,
            )
            token_sigreg_loss = projected_student_token_sigreg_loss
            projected_student_sigreg_term = sigreg_lambda_current * (
                projected_student_token_sigreg_loss.to(dtype=context_emb.dtype)
            )
            sigreg_term = projected_student_sigreg_term
            regularizer_lambda_current = sigreg_lambda_current
            regularizer_loss = token_sigreg_loss.to(dtype=context_emb.dtype)
            regularizer_term = sigreg_term
        if self.train_covariance_pooling and self.covariance_pooling_sigreg_lambda > 0:
            covariance_pooling_sigreg_lambda_current = context_emb.new_tensor(
                self.covariance_pooling_sigreg_lambda
            )
            covariance_embedding = self.covariance_pooler(
                teacher_peak_emb.float(),
                peak_valid_mask,
            )
            covariance_pooling_sigreg_loss = self.covariance_sigreg(
                covariance_embedding,
            )
            covariance_pooling_sigreg_term = (
                covariance_pooling_sigreg_lambda_current
                * covariance_pooling_sigreg_loss.to(dtype=context_emb.dtype)
            )
        loss = (
            jepa_term
            + cls_embedding_term
            + regularizer_term
            + covariance_pooling_sigreg_term
        )
        with torch.no_grad():
            collapse_metrics: dict[str, torch.Tensor] = {}
            for k, v in _masked_embedding_stats(context_emb, context_mask).items():
                collapse_metrics[f"global_{k}"] = v
            reg_stats = _masked_embedding_stats(context_emb, context_mask)
        valid_peak_count = peak_valid_mask.float().sum().clamp_min(1.0)
        metrics = {
            "loss": loss,
            "local_global_loss": local_global_loss,
            "jepa_term": jepa_term,
            "cls_embedding_loss": cls_embedding_loss,
            "cls_embedding_term": cls_embedding_term,
            "cls_embedding_loss_weight": cls_loss_weight,
            "regularizer_loss": regularizer_loss,
            "regularizer_term": regularizer_term,
            "regularizer_lambda_current": regularizer_lambda_current,
            "sigreg_loss": token_sigreg_loss.to(dtype=context_emb.dtype),
            "token_sigreg_loss": token_sigreg_loss.to(dtype=context_emb.dtype),
            "context_token_sigreg_loss": context_token_sigreg_loss.to(
                dtype=context_emb.dtype
            ),
            "teacher_token_sigreg_loss": teacher_token_sigreg_loss.to(
                dtype=context_emb.dtype
            ),
            "pred_token_sigreg_loss": pred_token_sigreg_loss.to(dtype=context_emb.dtype),
            "projected_student_token_sigreg_loss": projected_student_token_sigreg_loss.to(
                dtype=context_emb.dtype
            ),
            "projected_teacher_token_sigreg_loss": projected_teacher_token_sigreg_loss.to(
                dtype=context_emb.dtype
            ),
            "sigreg_term": sigreg_term,
            "context_sigreg_term": context_sigreg_term,
            "teacher_sigreg_term": teacher_sigreg_term,
            "pred_sigreg_term": pred_sigreg_term,
            "projected_student_sigreg_term": projected_student_sigreg_term,
            "projected_teacher_sigreg_term": projected_teacher_sigreg_term,
            "sigreg_lambda_current": sigreg_lambda_current,
            "covariance_pooling_sigreg_loss": covariance_pooling_sigreg_loss.to(
                dtype=context_emb.dtype
            ),
            "covariance_pooling_sigreg_term": covariance_pooling_sigreg_term,
            "covariance_pooling_sigreg_lambda_current": (
                covariance_pooling_sigreg_lambda_current
            ),
            "target_regularizer_term_over_jepa_term": regularizer_term
            / jepa_term.clamp_min(1e-8),
            "target_sigreg_term_over_jepa_term": sigreg_term
            / jepa_term.clamp_min(1e-8),
            "covariance_pooling_sigreg_term_over_jepa_term": (
                covariance_pooling_sigreg_term / jepa_term.clamp_min(1e-8)
            ),
            "context_fraction": context_mask.float().sum() / valid_peak_count,
            "masked_fraction": target_masks.float().sum() / valid_peak_count,
            "cls_visible_fraction": cls_visible_mask.float().sum() / valid_peak_count,
            **{f"encoder_{k}": v.to(context_emb.dtype) for k, v in reg_stats.items()},
            **collapse_metrics,
        }
        return metrics

    def compute_next_frame_teacher_embeddings(
        self, batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute teacher embeddings for the next frame."""
        next_frame_mz, next_frame_int, next_frame_valid = self._get_temporal_frame_inputs(
            batch,
            "next_frame",
        )
        teacher_embeddings = self.encoder(
            next_frame_mz,
            next_frame_int,
            valid_mask=next_frame_valid,
            visible_mask=next_frame_valid,
        )
        teacher_embeddings, _ = self.encoder.split_peak_and_cls(
            teacher_embeddings
        )
        return teacher_embeddings

    def forward_temporal(
        self,
        batch: dict[str, torch.Tensor],
        teacher_embeddings: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict next-frame token embeddings from the full current frame."""
        if self.temporal_predictor_num_layers <= 0:
            raise ValueError(
                "forward_temporal requires temporal_predictor_num_layers > 0"
            )
        frame_mz, frame_int, frame_valid = self._get_temporal_frame_inputs(
            batch,
            "frame",
        )
        next_frame_mz, next_frame_int, next_frame_valid = self._get_temporal_frame_inputs(
            batch,
            "next_frame",
        )
        frame_rt = batch["frame_rt"]
        next_frame_rt = batch["next_frame_rt"]
        B = frame_mz.shape[0]

        frame_encoded = self.encoder(
            frame_mz,
            frame_int,
            valid_mask=frame_valid,
            visible_mask=frame_valid,
        )  # [B, N, D]
        frame_emb, _ = self.encoder.split_peak_and_cls(frame_encoded)

        delta_rt = (next_frame_rt - frame_rt).unsqueeze(-1)  # [B, 1] in minutes
        rt_emb = self.temporal_rt_proj(delta_rt)  # [B, D]

        queries = self.temporal_query_token.view(1, 1, -1).expand(
            B, frame_emb.shape[1], -1
        )
        queries = queries + rt_emb.unsqueeze(1)
        queries = self._add_predictor_positions(queries)
        queries, _ = self._append_predictor_register_tokens(queries, None)

        for block in self.temporal_predictor:
            queries = block(queries, frame_emb, memory_mask=frame_valid)
        if self.predictor_num_register_tokens > 0:
            queries = queries[:, :-self.predictor_num_register_tokens]
        predicted_next_frame = queries  # [B, N, D]

        if teacher_embeddings is not None:
            next_frame_emb = teacher_embeddings
        else:
            next_frame_emb = self.compute_next_frame_teacher_embeddings(batch)

        per_token = self._embedding_loss(predicted_next_frame, next_frame_emb)

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
        encoded = self.encoder(
            mz,
            intensity,
            valid_mask=valid,
            visible_mask=valid,
        )
        _, cls_x = self.encoder.split_peak_and_cls(encoded)
        return cls_x
