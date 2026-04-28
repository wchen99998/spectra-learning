import copy
import math
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import nn

from models.losses import SIGReg, SlotwiseSIGReg
from models.peak_features import PeakFeatureEmbedder
from models.spectral_attention_bias import SpectralGraphormerBias
from networks import transformer_torch
from networks.transformer_torch import _build_norm, create_visible_attention_mask
from utils.spectra_preprocessing import PEAK_MZ_MAX, PRECURSOR_TOKEN_INTENSITY


def _active_autocast_context(device_type: str):
    if torch.is_autocast_enabled(device_type):
        return torch.autocast(
            device_type=device_type,
            dtype=torch.get_autocast_dtype(device_type),
        )
    return nullcontext()


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


def _masked_flatten(
    emb: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = emb.float().reshape(-1, emb.shape[-1])
    weights = valid_mask.reshape(-1).float()
    count = weights.sum().clamp_min(1.0)
    weights_col = weights.unsqueeze(-1)
    mean = (flat * weights_col).sum(0) / count
    return flat, weights, mean


def _weighted_covariance(
    emb: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    flat, weights, mean = _masked_flatten(emb, valid_mask)
    centered = flat - mean
    with torch.autocast(device_type=emb.device.type, enabled=False):
        cov = (
            centered.transpose(0, 1)
            @ (centered * weights.unsqueeze(-1))
            / weights.sum().clamp_min(1.0)
        )
    return cov, flat, weights, mean


def _sample_for_pairwise_cosine(
    flat: torch.Tensor,
    weights: torch.Tensor,
    max_samples: int = 1024,
) -> torch.Tensor:
    valid = flat[weights > 0]
    if valid.shape[0] <= max_samples:
        return valid
    stride = math.ceil(valid.shape[0] / max_samples)
    return valid[::stride][:max_samples]


def _pairwise_cosine_stats(
    flat: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    sample = _sample_for_pairwise_cosine(flat, weights)
    normed = F.normalize(sample, dim=-1)
    cos = normed @ normed.transpose(0, 1)
    n = sample.shape[0]
    offdiag = cos[~torch.eye(n, device=cos.device, dtype=torch.bool)]
    return offdiag.mean(), offdiag.std(unbiased=False)


def _embedding_geometry_metrics(
    emb: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    cov, flat, weights, mean = _weighted_covariance(emb, valid_mask)
    var = cov.diagonal()
    trace = var.sum()
    eigvals = torch.linalg.eigvalsh(cov.float()).clamp_min(0.0)
    eig_mass = eigvals / trace.clamp_min(1e-12)
    effective_rank = torch.exp(
        -(eig_mass * eig_mass.clamp_min(1e-12).log()).sum()
    )
    participation_ratio = trace.square() / eigvals.square().sum().clamp_min(1e-12)
    norm = flat.norm(dim=-1)
    weighted_norm = norm[weights > 0]
    pairwise_mean, pairwise_std = _pairwise_cosine_stats(flat, weights)
    return {
        "trace_cov": trace,
        "effective_rank": effective_rank,
        "effective_rank_frac": effective_rank / cov.shape[0],
        "participation_ratio": participation_ratio,
        "top_eigen_mass": eig_mass[-1],
        "mean_norm": mean.norm(),
        "embedding_norm_p50": torch.quantile(weighted_norm, 0.50),
        "embedding_norm_p95": torch.quantile(weighted_norm, 0.95),
        "per_dim_std_min": torch.sqrt(var.clamp_min(0.0)).amin(),
        "per_dim_std_p05": torch.quantile(torch.sqrt(var.clamp_min(0.0)), 0.05),
        "pairwise_cosine_mean": pairwise_mean,
        "pairwise_cosine_std": pairwise_std,
    }


def _prefix_metrics(
    prefix: str,
    metrics: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {f"{prefix}/{key}": value for key, value in metrics.items()}


def _token_prediction_r2(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    weights = valid_mask.unsqueeze(-1).float()
    count = weights.sum().clamp_min(1.0)
    mean = (target.float() * weights).sum(dim=(0, 1, 2)) / count
    residual = (prediction.float() - target.float()).square() * weights
    centered = (target.float() - mean).square() * weights
    return 1.0 - residual.sum() / centered.sum().clamp_min(1e-12)


def _pooled_prediction_r2(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    weights = valid_mask.unsqueeze(-1).float()
    denom = weights.sum(dim=2).clamp_min(1.0)
    pooled_prediction = (prediction.float() * weights).sum(dim=2) / denom
    pooled_target = (target.float() * weights).sum(dim=2) / denom
    mean = pooled_target.mean(dim=(0, 1))
    return 1.0 - (pooled_prediction - pooled_target).square().sum() / (
        pooled_target - mean
    ).square().sum().clamp_min(1e-12)


def _within_spectrum_pairwise_cosine(
    emb: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    normed = F.normalize(emb.float(), dim=-1)
    cos = normed @ normed.transpose(1, 2)
    pair_mask = valid_mask.unsqueeze(1) & valid_mask.unsqueeze(2)
    pair_mask = pair_mask & ~torch.eye(
        emb.shape[1], device=emb.device, dtype=torch.bool
    ).unsqueeze(0)
    return cos[pair_mask].mean()


def _collapse_diagnostics(
    *,
    teacher_peak_emb: torch.Tensor,
    teacher_cls_emb: torch.Tensor,
    context_emb: torch.Tensor,
    context_mask: torch.Tensor,
    peak_valid_mask: torch.Tensor,
    target_masks: torch.Tensor,
    teacher_target_features: torch.Tensor,
    teacher_target_features_normalized: torch.Tensor,
    teacher_targets: torch.Tensor,
    predictor_output_features: torch.Tensor,
    predictor_output: torch.Tensor,
    pooled_mean: torch.Tensor,
) -> dict[str, torch.Tensor]:
    metrics: dict[str, torch.Tensor] = {}
    metrics.update(
        _prefix_metrics(
            "repr/token",
            _embedding_geometry_metrics(teacher_peak_emb, peak_valid_mask),
        )
    )
    metrics.update(
        _prefix_metrics(
            "repr/student_context",
            _embedding_geometry_metrics(context_emb, context_mask),
        )
    )
    spectrum_mask = torch.ones(
        pooled_mean.shape[0],
        device=pooled_mean.device,
        dtype=torch.bool,
    )
    metrics.update(
        _prefix_metrics(
            "repr/spec_mean",
            _embedding_geometry_metrics(
                pooled_mean.unsqueeze(1),
                spectrum_mask.unsqueeze(1),
            ),
        )
    )
    metrics.update(
        _prefix_metrics(
            "repr/cls",
            _embedding_geometry_metrics(
                teacher_cls_emb.unsqueeze(1),
                spectrum_mask.unsqueeze(1),
            ),
        )
    )
    probe_feature = torch.cat([teacher_cls_emb.float(), pooled_mean.float()], dim=-1)
    metrics.update(
        _prefix_metrics(
            "repr/spec_probe_feature",
            _embedding_geometry_metrics(
                probe_feature.unsqueeze(1),
                spectrum_mask.unsqueeze(1),
            ),
        )
    )

    residual = teacher_peak_emb - pooled_mean.unsqueeze(1)
    within = _embedding_geometry_metrics(residual, peak_valid_mask)
    metrics.update(_prefix_metrics("repr/within", within))
    between = metrics["repr/spec_mean/trace_cov"]
    metrics["repr/between/trace_cov"] = between
    metrics["repr/within_to_between_trace_ratio"] = (
        within["trace_cov"] / between.clamp_min(1e-12)
    )
    metrics["repr/mean_pairwise_token_cosine_within_spectrum"] = (
        _within_spectrum_pairwise_cosine(teacher_peak_emb, peak_valid_mask)
    )
    metrics["repr/token_to_spectrum_norm_ratio"] = (
        metrics["repr/token/embedding_norm_p50"]
        / metrics["repr/spec_mean/embedding_norm_p50"].clamp_min(1e-12)
    )
    metrics["repr/cls_to_mean_peak_cosine"] = F.cosine_similarity(
        teacher_cls_emb.float(),
        pooled_mean.float(),
        dim=-1,
    ).mean()
    metrics["repr/cls_norm_over_mean_peak_norm"] = (
        teacher_cls_emb.float().norm(dim=-1).mean()
        / pooled_mean.float().norm(dim=-1).mean().clamp_min(1e-12)
    )

    expanded_teacher_features = teacher_target_features.unsqueeze(1).expand_as(
        predictor_output_features
    )
    expanded_teacher_features_normalized = (
        teacher_target_features_normalized.unsqueeze(1).expand_as(
            predictor_output_features
        )
    )
    expanded_teacher_targets = teacher_targets.unsqueeze(1).expand_as(predictor_output)
    metrics.update(
        _prefix_metrics(
            "repr/jepa_target_features_raw",
            _embedding_geometry_metrics(expanded_teacher_features, target_masks),
        )
    )
    metrics.update(
        _prefix_metrics(
            "repr/jepa_target_features_normalized",
            _embedding_geometry_metrics(
                expanded_teacher_features_normalized,
                target_masks,
            ),
        )
    )
    metrics.update(
        _prefix_metrics(
            "repr/jepa_target_projected",
            _embedding_geometry_metrics(expanded_teacher_targets, target_masks),
        )
    )
    metrics.update(
        _prefix_metrics(
            "repr/jepa_prediction_features",
            _embedding_geometry_metrics(predictor_output_features, target_masks),
        )
    )
    metrics.update(
        _prefix_metrics(
            "repr/jepa_prediction_projected",
            _embedding_geometry_metrics(predictor_output, target_masks),
        )
    )
    target_trace = metrics["repr/jepa_target_projected/trace_cov"]
    pred_trace = metrics["repr/jepa_prediction_projected/trace_cov"]
    metrics["jepa/target_var"] = target_trace
    metrics["jepa/pred_var"] = pred_trace
    metrics["jepa/pred_target_var_ratio"] = pred_trace / target_trace.clamp_min(
        1e-12
    )
    metrics["jepa/token_prediction_R2"] = _token_prediction_r2(
        predictor_output,
        expanded_teacher_targets,
        target_masks,
    )
    metrics["jepa/spectrum_pooled_prediction_R2"] = _pooled_prediction_r2(
        predictor_output,
        expanded_teacher_targets,
        target_masks,
    )
    metrics["jepa/teacher_target_effective_rank"] = metrics[
        "repr/jepa_target_projected/effective_rank"
    ]
    metrics["jepa/predictor_output_effective_rank"] = metrics[
        "repr/jepa_prediction_projected/effective_rank"
    ]
    return metrics


def _merge_visible_mask(
    valid_mask: torch.Tensor | None,
    visible_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    if visible_mask is not None and valid_mask is not None:
        return visible_mask & valid_mask
    return visible_mask if visible_mask is not None else valid_mask


def _masked_mean_pool(
    embeddings: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    mask = valid_mask.unsqueeze(-1).to(dtype=embeddings.dtype)
    return (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


class CrossAttention(nn.Module):
    """Cross-attention: Q from prediction queries, KV from source embeddings."""

    def __init__(self, dim: int, n_heads: int, *, n_kv_heads: int | None = None,
                 qk_norm: bool = False, norm_type: str = "rmsnorm",
                 norm_eps: float = 1e-5):
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
            self.q_norm = _build_norm(self.head_dim, eps=norm_eps, norm_type=norm_type)
            self.k_norm = _build_norm(self.head_dim, eps=norm_eps, norm_type=norm_type)
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
            qk_norm=qk_norm, norm_type=norm_type, norm_eps=norm_eps,
        )
        self.cross_attn = CrossAttention(dim, n_heads, n_kv_heads=n_kv_heads,
                                          qk_norm=qk_norm, norm_type=norm_type,
                                          norm_eps=norm_eps)
        self.feed_forward = transformer_torch.FeedForward(dim, hidden_dim=hidden_dim)
        self.attention_norm = _build_norm(dim, eps=norm_eps, norm_type=norm_type)
        self.cross_attn_norm = _build_norm(dim, eps=norm_eps, norm_type=norm_type)
        self.ffn_norm = _build_norm(dim, eps=norm_eps, norm_type=norm_type)

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
        with torch.autocast(device_type=peak_embeddings.device.type, enabled=False):
            peak_embeddings = peak_embeddings.float()
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
        fourier_mlp_hidden_dim: int | None = None,
        fourier_mlp_num_layers: int = 2,
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
        norm_eps: float = 1e-5,
        apply_final_norm: bool = True,
        num_peaks: int = 64,
        use_position_embedding: bool = True,
        use_cls_token: bool = True,
        num_register_tokens: int = 0,
        use_precursor_token: bool = False,
        spectral_bias_relative_kind: str = "none",
        spectral_bias_use_precursor: bool = False,
        spectral_bias_use_intensity: bool = False,
        spectral_bias_num_freqs: int = 128,
        spectral_bias_fourier_strategy: str = "log_spaced",
        spectral_bias_fourier_x_min: float = 3e-3,
        spectral_bias_fourier_x_max: float = 1000.0,
        spectral_bias_fourier_sigma: float = 10.0,
        spectral_bias_fourier_trainable: bool = False,
        spectral_bias_mass_scale: float = 1000.0,
        spectral_bias_precursor_scale: float = 1000.0,
        spectral_bias_rbf_num_basis: int = 64,
        spectral_bias_rbf_delta_min: float = -1000.0,
        spectral_bias_rbf_delta_max: float = 1000.0,
        spectral_bias_rbf_use_absolute_delta: bool = False,
        spectral_bias_intensity_hidden_dim: int = 16,
        spectral_bias_init_std: float = 0.0,
        spectral_bias_clip: float | None = None,
    ):
        super().__init__()
        self.num_layers = int(num_layers)
        norm_type = str(norm_type).lower()
        self.use_cls_token = bool(use_cls_token)
        self.num_register_tokens = int(num_register_tokens)
        self.use_precursor_token = bool(use_precursor_token)
        self.use_position_embedding = bool(use_position_embedding)
        relative_kind = str(spectral_bias_relative_kind).lower()
        spectral_bias_enabled = (
            relative_kind not in {"", "none", "false", "off"}
            or bool(spectral_bias_use_precursor)
            or bool(spectral_bias_use_intensity)
        )
        self.embedder = PeakFeatureEmbedder(
            model_dim=model_dim,
            hidden_dim=feature_mlp_hidden_dim,
            fourier_mlp_hidden_dim=fourier_mlp_hidden_dim,
            fourier_mlp_num_layers=fourier_mlp_num_layers,
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
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.empty(model_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None
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
            norm_eps=norm_eps,
            qk_norm=qk_norm,
            norm_type=norm_type,
        )
        self.final_norm = (
            _build_norm(model_dim, eps=norm_eps, norm_type=norm_type)
            if apply_final_norm
            else nn.Identity()
        )
        if spectral_bias_enabled:
            self.spectral_attn_biases = nn.ModuleList(
                [
                    SpectralGraphormerBias(
                        num_heads=int(num_heads),
                        mass_scale=float(spectral_bias_mass_scale),
                        precursor_scale=float(spectral_bias_precursor_scale),
                        first_token_is_precursor=self.use_precursor_token,
                        relative_kind=spectral_bias_relative_kind,
                        num_freqs=int(spectral_bias_num_freqs),
                        fourier_strategy=str(spectral_bias_fourier_strategy),
                        fourier_x_min=float(spectral_bias_fourier_x_min),
                        fourier_x_max=float(spectral_bias_fourier_x_max),
                        fourier_sigma=float(spectral_bias_fourier_sigma),
                        fourier_trainable=bool(spectral_bias_fourier_trainable),
                        use_precursor_bias=bool(spectral_bias_use_precursor),
                        use_intensity_bias=bool(spectral_bias_use_intensity),
                        intensity_hidden_dim=int(spectral_bias_intensity_hidden_dim),
                        rbf_num_basis=int(spectral_bias_rbf_num_basis),
                        rbf_delta_min=float(spectral_bias_rbf_delta_min),
                        rbf_delta_max=float(spectral_bias_rbf_delta_max),
                        rbf_use_absolute_delta=bool(
                            spectral_bias_rbf_use_absolute_delta
                        ),
                        init_std=float(spectral_bias_init_std),
                        bias_clip=spectral_bias_clip,
                    )
                    for _ in range(self.num_layers)
                ]
            )
        else:
            self.spectral_attn_biases = None

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
        special_tokens = []
        if self.cls_token is not None:
            cls = self.cls_token.view(1, 1, -1).expand(x.shape[0], -1, -1)
            special_tokens.append(cls.to(dtype=x.dtype))
        if self.register_tokens is not None:
            registers = self.register_tokens.unsqueeze(0).expand(x.shape[0], -1, -1)
            special_tokens.append(registers.to(dtype=x.dtype))
        if not special_tokens:
            return x, attn_mask
        special = torch.cat(special_tokens, dim=1)
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

    def _spectral_attn_bias(
        self,
        block_idx: int,
        peak_mz: torch.Tensor,
        peak_intensity: torch.Tensor,
        precursor_mz: torch.Tensor | None,
        *,
        num_special_tokens: int,
    ) -> torch.Tensor | None:
        if self.spectral_attn_biases is None:
            return None
        return self.spectral_attn_biases[int(block_idx)](
            peak_mz,
            peak_intensity=peak_intensity,
            precursor_mz=precursor_mz,
            num_special_tokens=int(num_special_tokens),
        )

    def split_peak_and_cls(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_cls_token:
            return x[:, :-1], x[:, -1]
        return x, x.mean(dim=1)

    def forward_with_block_outputs(
        self,
        peak_mz: torch.Tensor,
        peak_intensity: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        visible_mask: torch.Tensor | None = None,
        block_indices: list[int] | tuple[int, ...] = (),
        precursor_mz: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        block_indices = tuple(int(idx) for idx in block_indices)
        attn_mask = _merge_visible_mask(valid_mask, visible_mask)
        x = self._add_positions(self.embedder(peak_mz, peak_intensity))
        seq_len = peak_mz.shape[1]
        selected = set(block_indices)
        selected_peak_outputs: dict[int, torch.Tensor] = {}
        special_len = int(self.use_cls_token) + self.num_register_tokens
        x, attn_mask = self._append_special_tokens(x, attn_mask)
        attn_mask = (
            create_visible_attention_mask(attn_mask) if attn_mask is not None else None
        )
        for block_idx, block in enumerate(self.blocks, start=1):
            attn_bias = self._spectral_attn_bias(
                block_idx - 1,
                peak_mz,
                peak_intensity,
                precursor_mz,
                num_special_tokens=special_len,
            )
            x = block(
                x,
                attn_mask=attn_mask,
                attn_bias=attn_bias,
            )
            if block_idx in selected and block_idx != self.num_layers:
                selected_peak_outputs[block_idx] = x[:, :seq_len]
        x = self.final_norm(x)
        if self.num_layers in selected:
            selected_peak_outputs[self.num_layers] = x[:, :seq_len]
        peak_x = x[:, :seq_len]
        if self.use_cls_token:
            cls_x = x[:, seq_len]
            output = torch.cat([peak_x, cls_x.unsqueeze(1)], dim=1)
        else:
            output = peak_x
        return output, [selected_peak_outputs[idx] for idx in block_indices]

    def forward_peak_block_outputs(
        self,
        peak_mz: torch.Tensor,
        peak_intensity: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        visible_mask: torch.Tensor | None = None,
        block_indices: list[int] | tuple[int, ...] = (),
        precursor_mz: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        _, peak_block_outputs = self.forward_with_block_outputs(
            peak_mz,
            peak_intensity,
            valid_mask=valid_mask,
            visible_mask=visible_mask,
            block_indices=block_indices,
            precursor_mz=precursor_mz,
        )
        return peak_block_outputs

    def forward(
        self,
        peak_mz: torch.Tensor,
        peak_intensity: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        visible_mask: torch.Tensor | None = None,
        return_cls_token: bool = False,
        precursor_mz: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        output, _ = self.forward_with_block_outputs(
            peak_mz,
            peak_intensity,
            valid_mask=valid_mask,
            visible_mask=visible_mask,
            precursor_mz=precursor_mz,
        )
        if return_cls_token:
            peak_x, cls_x = self.split_peak_and_cls(output)
            if not self.use_cls_token and valid_mask is not None:
                cls_x = _masked_mean_pool(peak_x, valid_mask)
            return output, cls_x
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
        encoder_fourier_mlp_hidden_dim: int | None = None,
        encoder_fourier_mlp_num_layers: int = 2,
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
        jepa_mae_loss_weight: float = 0.0,
        jepa_mae_mz_bin_size: float = 2.5,
        jepa_mae_intensity_bin_size: float = 0.1,
        jepa_mae_mz_max: float = PEAK_MZ_MAX,
        jepa_mae_intensity_max: float = 1.0,
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
        norm_eps: float = 1e-5,
        encoder_use_position_embedding: bool = True,
        encoder_apply_final_norm: bool = True,
        predictor_apply_final_norm: bool = True,
        encoder_use_cls_token: bool = True,
        use_precursor_token: bool = False,
        spectral_bias_relative_kind: str = "none",
        spectral_bias_use_precursor: bool = False,
        spectral_bias_use_intensity: bool = False,
        spectral_bias_num_freqs: int = 128,
        spectral_bias_fourier_strategy: str = "log_spaced",
        spectral_bias_fourier_x_min: float = 3e-3,
        spectral_bias_fourier_x_max: float = 1000.0,
        spectral_bias_fourier_sigma: float = 10.0,
        spectral_bias_fourier_trainable: bool = False,
        spectral_bias_mass_scale: float = 1000.0,
        spectral_bias_precursor_scale: float = 1000.0,
        spectral_bias_rbf_num_basis: int = 64,
        spectral_bias_rbf_delta_min: float = -1000.0,
        spectral_bias_rbf_delta_max: float = 1000.0,
        spectral_bias_rbf_use_absolute_delta: bool = False,
        spectral_bias_intensity_hidden_dim: int = 16,
        spectral_bias_init_std: float = 0.0,
        spectral_bias_clip: float | None = None,
        num_peaks: int = 64,
        temporal_predictor_num_layers: int = 0,
        encoder_num_register_tokens: int = 0,
        predictor_num_register_tokens: int = 0,
        predictor_dim: int | None = None,
        target_projector_dim: int | None = None,
        use_target_projector: bool = True,
        predictor_dropout: float = 0.0,
        train_covariance_pooling: bool = False,
        covariance_pooling_dim: int = 32,
        use_ema_teacher: bool = False,
        ema_teacher_momentum: float = 0.996,
        ema_teacher_momentum_mid: float | None = None,
        ema_teacher_momentum_final: float | None = None,
        ema_teacher_schedule_peak_fraction: float = 0.35,
        ema_teacher_schedule: str = "constant",
    ):
        super().__init__()
        self.model_dim = model_dim
        self.predictor_dim = predictor_dim if predictor_dim is not None else model_dim
        self.use_target_projector = bool(use_target_projector)
        self.encoder_num_layers = int(encoder_num_layers)
        self.encoder_use_cls_token = bool(encoder_use_cls_token)
        self.use_precursor_token = bool(use_precursor_token)
        self.jepa_num_target_blocks = int(jepa_num_target_blocks)
        self.jepa_target_layers = (
            [self.encoder_num_layers]
            if jepa_target_layers is None
            else [int(layer_idx) for layer_idx in jepa_target_layers]
        )
        if not self.jepa_target_layers:
            raise ValueError("jepa_target_layers must not be empty")
        if (
            min(self.jepa_target_layers) < 1
            or max(self.jepa_target_layers) > self.encoder_num_layers
        ):
            raise ValueError("jepa_target_layers must be within encoder depth")
        self.num_jepa_target_layers = len(self.jepa_target_layers)
        self.jepa_target_dim = self.num_jepa_target_layers * self.model_dim
        requested_target_projector_dim = (
            int(target_projector_dim)
            if target_projector_dim is not None
            else self.model_dim
        )
        self.target_projector_dim = (
            requested_target_projector_dim
            if self.use_target_projector
            else self.jepa_target_dim
        )
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
        self.masked_token_loss_weight = float(masked_token_loss_weight)
        self.masked_token_loss_type = str(masked_token_loss_type).lower()
        self.jepa_mae_loss_weight = float(jepa_mae_loss_weight)
        self.jepa_mae_mz_bin_size = float(jepa_mae_mz_bin_size)
        self.jepa_mae_intensity_bin_size = float(jepa_mae_intensity_bin_size)
        self.jepa_mae_mz_max = float(jepa_mae_mz_max)
        self.jepa_mae_intensity_max = float(jepa_mae_intensity_max)
        self.jepa_mae_num_mz_bins = int(
            math.ceil(self.jepa_mae_mz_max / self.jepa_mae_mz_bin_size)
        )
        self.jepa_mae_num_intensity_bins = int(
            math.ceil(self.jepa_mae_intensity_max / self.jepa_mae_intensity_bin_size)
        )
        self.jepa_target_normalization = str(jepa_target_normalization).lower()
        if self.jepa_target_normalization not in ("none", "zscore"):
            raise ValueError(
                "jepa_target_normalization must be one of ('none', 'zscore')"
            )
        self.norm_type = str(norm_type).lower()
        self.norm_eps = float(norm_eps)
        self.temporal_predictor_num_layers = int(temporal_predictor_num_layers)
        self.predictor_num_register_tokens = int(predictor_num_register_tokens)
        if self.jepa_num_target_blocks < 1:
            raise ValueError("jepa_num_target_blocks must be >= 1")
        num_peak_tokens = int(num_peaks) + int(self.use_precursor_token)
        self.encoder = PeakSetEncoder(
            model_dim=model_dim,
            num_layers=self.encoder_num_layers,
            num_heads=encoder_num_heads,
            num_kv_heads=encoder_num_kv_heads,
            attention_mlp_multiple=attention_mlp_multiple,
            feature_mlp_hidden_dim=feature_mlp_hidden_dim,
            fourier_mlp_hidden_dim=encoder_fourier_mlp_hidden_dim,
            fourier_mlp_num_layers=encoder_fourier_mlp_num_layers,
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
            norm_eps=self.norm_eps,
            use_position_embedding=encoder_use_position_embedding,
            apply_final_norm=encoder_apply_final_norm,
            num_peaks=num_peak_tokens,
            use_cls_token=self.encoder_use_cls_token,
            num_register_tokens=encoder_num_register_tokens,
            use_precursor_token=self.use_precursor_token,
            spectral_bias_relative_kind=spectral_bias_relative_kind,
            spectral_bias_use_precursor=spectral_bias_use_precursor,
            spectral_bias_use_intensity=spectral_bias_use_intensity,
            spectral_bias_num_freqs=spectral_bias_num_freqs,
            spectral_bias_fourier_strategy=spectral_bias_fourier_strategy,
            spectral_bias_fourier_x_min=spectral_bias_fourier_x_min,
            spectral_bias_fourier_x_max=spectral_bias_fourier_x_max,
            spectral_bias_fourier_sigma=spectral_bias_fourier_sigma,
            spectral_bias_fourier_trainable=spectral_bias_fourier_trainable,
            spectral_bias_mass_scale=spectral_bias_mass_scale,
            spectral_bias_precursor_scale=spectral_bias_precursor_scale,
            spectral_bias_rbf_num_basis=spectral_bias_rbf_num_basis,
            spectral_bias_rbf_delta_min=spectral_bias_rbf_delta_min,
            spectral_bias_rbf_delta_max=spectral_bias_rbf_delta_max,
            spectral_bias_rbf_use_absolute_delta=spectral_bias_rbf_use_absolute_delta,
            spectral_bias_intensity_hidden_dim=spectral_bias_intensity_hidden_dim,
            spectral_bias_init_std=spectral_bias_init_std,
            spectral_bias_clip=spectral_bias_clip,
        )
        self.use_ema_teacher = bool(use_ema_teacher)
        self.ema_teacher_momentum = float(ema_teacher_momentum)
        self.ema_teacher_momentum_mid = (
            float(ema_teacher_momentum_mid)
            if ema_teacher_momentum_mid is not None
            else self.ema_teacher_momentum
        )
        self.ema_teacher_momentum_final = (
            float(ema_teacher_momentum_final)
            if ema_teacher_momentum_final is not None
            else self.ema_teacher_momentum
        )
        self.ema_teacher_schedule_peak_fraction = float(
            ema_teacher_schedule_peak_fraction
        )
        self.ema_teacher_schedule = str(ema_teacher_schedule).lower()
        if self.ema_teacher_schedule not in (
            "constant",
            "linear",
            "cosine",
            "slow-fast-slow",
        ):
            raise ValueError(
                "ema_teacher_schedule must be one of "
                "('constant', 'linear', 'cosine', 'slow-fast-slow')"
            )
        if self.use_ema_teacher:
            self.teacher_encoder = copy.deepcopy(self.encoder)
            self.teacher_encoder.requires_grad_(False)
        else:
            self.teacher_encoder = None
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
            num_peak_tokens,
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
            norm_eps=self.norm_eps,
            qk_norm=encoder_qk_norm,
            norm_type=self.norm_type,
            dropout=predictor_dropout,
        )
        self.predictor_final_norm = (
            _build_norm(self.predictor_dim, eps=self.norm_eps, norm_type=self.norm_type)
            if predictor_apply_final_norm
            else nn.Identity()
        )
        self.masked_latent_readout = nn.Linear(self.predictor_dim, self.jepa_target_dim)
        nn.init.xavier_normal_(self.masked_latent_readout.weight)
        nn.init.zeros_(self.masked_latent_readout.bias)
        if self.use_target_projector:
            self.target_projector = nn.Sequential(
                nn.Linear(self.jepa_target_dim, self.jepa_target_dim),
                nn.GELU(),
                nn.Linear(self.jepa_target_dim, self.target_projector_dim),
            )
            for layer in self.target_projector:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    nn.init.zeros_(layer.bias)
        else:
            self.target_projector = nn.Identity()
        if self.use_ema_teacher:
            self.teacher_target_projector = copy.deepcopy(self.target_projector)
            self.teacher_target_projector.requires_grad_(False)
        else:
            self.teacher_target_projector = None
        if self.jepa_mae_loss_weight > 0:
            self.jepa_mae_mz_head = nn.Linear(
                self.target_projector_dim,
                self.jepa_mae_num_mz_bins,
            )
            self.jepa_mae_intensity_head = nn.Linear(
                self.target_projector_dim,
                self.jepa_mae_num_intensity_bins,
            )
            nn.init.xavier_normal_(self.jepa_mae_mz_head.weight)
            nn.init.zeros_(self.jepa_mae_mz_head.bias)
            nn.init.xavier_normal_(self.jepa_mae_intensity_head.weight)
            nn.init.zeros_(self.jepa_mae_intensity_head.bias)
        else:
            self.jepa_mae_mz_head = None
            self.jepa_mae_intensity_head = None
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
                norm_eps=self.norm_eps, qk_norm=encoder_qk_norm,
                norm_type=self.norm_type,
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

    def ema_teacher_momentum_at(
        self,
        step: int,
        total_steps: int,
    ) -> float:
        if self.ema_teacher_schedule == "constant":
            return self.ema_teacher_momentum
        progress = min(1.0, max(0.0, float(step) / float(max(1, total_steps))))
        if self.ema_teacher_schedule == "slow-fast-slow":
            peak = min(1.0, max(1e-6, self.ema_teacher_schedule_peak_fraction))
            if progress <= peak:
                phase = progress / peak
                eased = 0.5 - 0.5 * math.cos(math.pi * phase)
                return self.ema_teacher_momentum + eased * (
                    self.ema_teacher_momentum_mid - self.ema_teacher_momentum
                )
            phase = (progress - peak) / max(1e-6, 1.0 - peak)
            eased = 0.5 - 0.5 * math.cos(math.pi * phase)
            return self.ema_teacher_momentum_mid + eased * (
                self.ema_teacher_momentum_final - self.ema_teacher_momentum_mid
            )
        if self.ema_teacher_schedule == "cosine":
            progress = 0.5 - 0.5 * math.cos(math.pi * progress)
        return self.ema_teacher_momentum + progress * (
            self.ema_teacher_momentum_final - self.ema_teacher_momentum
        )

    @torch.no_grad()
    def sync_ema_teacher(self) -> None:
        if self.teacher_encoder is not None:
            self.teacher_encoder.load_state_dict(self.encoder.state_dict())
        if self.teacher_target_projector is not None:
            self.teacher_target_projector.load_state_dict(
                self.target_projector.state_dict()
            )

    @staticmethod
    @torch.no_grad()
    def _update_ema_module(
        teacher: nn.Module,
        student: nn.Module,
        momentum: float,
    ) -> None:
        for teacher_param, student_param in zip(
            teacher.parameters(),
            student.parameters(),
        ):
            teacher_param.lerp_(student_param, 1.0 - momentum)
        for teacher_buffer, student_buffer in zip(
            teacher.buffers(),
            student.buffers(),
        ):
            if torch.is_floating_point(teacher_buffer):
                teacher_buffer.lerp_(student_buffer, 1.0 - momentum)
            else:
                teacher_buffer.copy_(student_buffer)

    @torch.no_grad()
    def update_ema_teacher(
        self,
        step: int,
        total_steps: int,
    ) -> float | None:
        if self.teacher_encoder is None:
            return None
        momentum = self.ema_teacher_momentum_at(step, total_steps)
        self._update_ema_module(self.teacher_encoder, self.encoder, momentum)
        if self.teacher_target_projector is not None:
            self._update_ema_module(
                self.teacher_target_projector,
                self.target_projector,
                momentum,
            )
        return momentum

    def _apply_group_target_normalization(
        self,
        x: torch.Tensor,
        group_dim: int,
    ) -> torch.Tensor:
        if self.jepa_target_normalization == "none":
            return x
        orig_dtype = x.dtype
        x = x.float().reshape(*x.shape[:-1], -1, group_dim)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        normalized = ((x - mean) / std).reshape(*x.shape[:-2], -1)
        return normalized.to(dtype=orig_dtype)

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

    def project_teacher_targets(self, x: torch.Tensor) -> torch.Tensor:
        projector = (
            self.teacher_target_projector
            if self.teacher_target_projector is not None
            else self.target_projector
        )
        return projector(x)

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

    def _split_encoder_output(
        self,
        encoder: PeakSetEncoder,
        embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        peak_embeddings, cls_embedding = encoder.split_peak_and_cls(embeddings)
        if not encoder.use_cls_token:
            cls_embedding = self.pool(peak_embeddings, valid_mask)
        return peak_embeddings, cls_embedding

    def _compute_jepa_teacher_target_features(
        self,
        peak_mz: torch.Tensor,
        peak_intensity: torch.Tensor,
        peak_valid_mask: torch.Tensor,
        precursor_mz: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with _active_autocast_context(peak_mz.device.type):
            teacher_encoder = (
                self.teacher_encoder
                if self.teacher_encoder is not None
                else self.encoder
            )
            teacher_peak_outputs = teacher_encoder.forward_peak_block_outputs(
                peak_mz,
                peak_intensity,
                valid_mask=peak_valid_mask,
                visible_mask=peak_valid_mask,
                block_indices=self.jepa_target_layers,
                precursor_mz=precursor_mz,
            )
            return torch.cat(teacher_peak_outputs, dim=-1)

    def _compute_jepa_teacher_targets(
        self,
        peak_mz: torch.Tensor,
        peak_intensity: torch.Tensor,
        peak_valid_mask: torch.Tensor,
        precursor_mz: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            teacher_target_features = self._compute_jepa_teacher_target_features(
                peak_mz,
                peak_intensity,
                peak_valid_mask,
                precursor_mz=precursor_mz,
            )
            return self.project_teacher_targets(
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
            precursor_mz=augmented_batch.get("precursor_mz", None),
        )

    def _encode_augmented_teacher_and_context(
        self,
        peak_mz: torch.Tensor,
        peak_intensity: torch.Tensor,
        peak_valid_mask: torch.Tensor,
        context_mask: torch.Tensor,
        precursor_mz: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = peak_mz.shape[0]
        if self.teacher_encoder is not None:
            with torch.no_grad(), _active_autocast_context(peak_mz.device.type):
                teacher_encoded, teacher_peak_outputs = (
                    self.teacher_encoder.forward_with_block_outputs(
                        peak_mz,
                        peak_intensity,
                        valid_mask=peak_valid_mask,
                        visible_mask=peak_valid_mask,
                        block_indices=self.jepa_target_layers,
                        precursor_mz=precursor_mz,
                    )
                )
            context_encoded = self.encoder(
                peak_mz,
                peak_intensity,
                valid_mask=peak_valid_mask,
                visible_mask=context_mask,
                precursor_mz=precursor_mz,
            )
            teacher_target_features = torch.cat(teacher_peak_outputs, dim=-1)
            teacher_peak_emb, teacher_cls_emb = self._split_encoder_output(
                self.teacher_encoder,
                teacher_encoded,
                peak_valid_mask,
            )
            context_emb, _ = self._split_encoder_output(
                self.encoder,
                context_encoded,
                peak_valid_mask,
            )
            return (
                teacher_target_features,
                teacher_peak_emb,
                teacher_cls_emb,
                context_emb,
            )
        encoded, teacher_peak_outputs = self.encoder.forward_with_block_outputs(
            torch.cat([peak_mz, peak_mz], dim=0),
            torch.cat([peak_intensity, peak_intensity], dim=0),
            valid_mask=torch.cat([peak_valid_mask, peak_valid_mask], dim=0),
            visible_mask=torch.cat([peak_valid_mask, context_mask], dim=0),
            block_indices=self.jepa_target_layers,
            precursor_mz=(
                None
                if precursor_mz is None
                else torch.cat([precursor_mz, precursor_mz], dim=0)
            ),
        )
        teacher_target_features = torch.cat(
            [peak_output[:batch_size] for peak_output in teacher_peak_outputs],
            dim=-1,
        )
        teacher_peak_emb, teacher_cls_emb = self._split_encoder_output(
            self.encoder,
            encoded[:batch_size],
            peak_valid_mask,
        )
        context_emb, _ = self._split_encoder_output(
            self.encoder,
            encoded[batch_size:],
            peak_valid_mask,
        )
        return teacher_target_features, teacher_peak_emb, teacher_cls_emb, context_emb

    def _compute_pooled_teacher_peak_targets(
        self,
        peak_mz: torch.Tensor,
        peak_intensity: torch.Tensor,
        peak_valid_mask: torch.Tensor,
        visible_mask: torch.Tensor | None = None,
        precursor_mz: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if visible_mask is None:
            visible_mask = peak_valid_mask
        with _active_autocast_context(peak_mz.device.type):
            teacher_encoder = (
                self.teacher_encoder
                if self.teacher_encoder is not None
                else self.encoder
            )
            teacher_encoded = teacher_encoder(
                peak_mz,
                peak_intensity,
                valid_mask=peak_valid_mask,
                visible_mask=visible_mask,
                precursor_mz=precursor_mz,
            )
            teacher_peak_emb, _ = self._split_encoder_output(
                teacher_encoder,
                teacher_encoded,
                peak_valid_mask,
            )
        return self.pool(teacher_peak_emb, visible_mask)

    def _embedding_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        prediction = prediction.float()
        target = target.float()
        if self.masked_token_loss_type == "l2":
            return (prediction - target).square().mean(dim=-1)
        if self.masked_token_loss_type == "l2_sum":
            return (prediction - target).square().sum(dim=-1)
        if self.masked_token_loss_type == "l1":
            return (prediction - target).abs().mean(dim=-1)
        raise ValueError(
            f"Unsupported masked_token_loss_type: {self.masked_token_loss_type}"
        )

    def _jepa_mae_targets(
        self,
        peak_mz: torch.Tensor,
        peak_intensity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mz_target = torch.floor(
            peak_mz.float() * self.jepa_mae_mz_max / self.jepa_mae_mz_bin_size
        ).long()
        intensity_target = torch.floor(
            peak_intensity.float() / self.jepa_mae_intensity_bin_size
        ).long()
        return (
            mz_target.clamp(0, self.jepa_mae_num_mz_bins - 1),
            intensity_target.clamp(0, self.jepa_mae_num_intensity_bins - 1),
        )

    def _masked_ce_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        per_token = F.cross_entropy(
            logits.flatten(0, -2).float(),
            targets.reshape(-1),
            reduction="none",
        ).reshape_as(valid_mask)
        weights = valid_mask.float()
        return (per_token * weights).sum() / weights.sum().clamp_min(1.0)

    def _jepa_mae_value_prediction_loss(
        self,
        predicted_latents: torch.Tensor,
        peak_mz: torch.Tensor,
        peak_intensity: torch.Tensor,
        target_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mz_logits = self.jepa_mae_mz_head(predicted_latents)
        intensity_logits = self.jepa_mae_intensity_head(predicted_latents)
        mz_target, intensity_target = self._jepa_mae_targets(peak_mz, peak_intensity)
        view_shape = (mz_logits.shape[0], mz_logits.shape[1], mz_logits.shape[2])
        mz_target = mz_target.unsqueeze(1).expand(view_shape)
        intensity_target = intensity_target.unsqueeze(1).expand(view_shape)
        mz_loss = self._masked_ce_loss(mz_logits, mz_target, target_masks)
        intensity_loss = self._masked_ce_loss(
            intensity_logits,
            intensity_target,
            target_masks,
        )
        value_loss = mz_loss + intensity_loss
        target_weights = target_masks
        mz_accuracy = (
            (mz_logits.argmax(dim=-1) == mz_target).float() * target_weights.float()
        ).sum() / target_weights.float().sum().clamp_min(1.0)
        intensity_accuracy = (
            (intensity_logits.argmax(dim=-1) == intensity_target).float()
            * target_weights.float()
        ).sum() / target_weights.float().sum().clamp_min(1.0)
        return value_loss, mz_loss, intensity_loss, mz_accuracy, intensity_accuracy

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
        return_collapse_data: bool = False,
    ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        peak_mz = augmented_batch["peak_mz"]
        peak_intensity = augmented_batch["peak_intensity"]
        peak_valid_mask = augmented_batch["peak_valid_mask"]
        precursor_mz = augmented_batch.get("precursor_mz", None)
        context_mask = augmented_batch["context_mask"] & peak_valid_mask
        target_masks = augmented_batch["target_masks"] & peak_valid_mask.unsqueeze(1)
        B, N = peak_mz.shape
        K = self.jepa_num_target_blocks
        (
            teacher_target_features,
            teacher_peak_emb,
            teacher_cls_emb,
            context_emb,
        ) = self._encode_augmented_teacher_and_context(
            peak_mz,
            peak_intensity,
            peak_valid_mask,
            context_mask,
            precursor_mz=precursor_mz,
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
        teacher_target_features_normalized = self._apply_jepa_target_normalization(
            teacher_target_features.detach()
        )
        with torch.no_grad():
            teacher_targets = self.project_teacher_targets(
                teacher_target_features_normalized
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
        jepa_mae_loss = context_emb.new_tensor(0.0)
        jepa_mae_mz_loss = context_emb.new_tensor(0.0)
        jepa_mae_intensity_loss = context_emb.new_tensor(0.0)
        jepa_mae_term = context_emb.new_tensor(0.0)
        jepa_mae_mz_accuracy = context_emb.new_tensor(0.0)
        jepa_mae_intensity_accuracy = context_emb.new_tensor(0.0)
        jepa_mae_loss_weight = context_emb.new_tensor(self.jepa_mae_loss_weight)
        if self.jepa_mae_loss_weight > 0:
            (
                jepa_mae_loss,
                jepa_mae_mz_loss,
                jepa_mae_intensity_loss,
                jepa_mae_mz_accuracy,
                jepa_mae_intensity_accuracy,
            ) = self._jepa_mae_value_prediction_loss(
                predictor_output,
                peak_mz,
                peak_intensity,
                target_masks,
            )
            jepa_mae_term = jepa_mae_loss_weight * jepa_mae_loss.to(
                dtype=context_emb.dtype
            )
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
        if self.train_covariance_pooling and self.sigreg_lambda > 0:
            covariance_embedding = self.covariance_pooler(
                teacher_peak_emb.float(),
                peak_valid_mask,
            )
            covariance_pooling_sigreg_loss = self.covariance_sigreg(
                covariance_embedding.float(),
            )
            covariance_pooling_sigreg_term = (
                context_emb.new_tensor(self.sigreg_lambda)
                * covariance_pooling_sigreg_loss.to(dtype=context_emb.dtype)
            )
        loss = (
            jepa_term
            + jepa_mae_term
            + cls_embedding_term
            + regularizer_term
            + covariance_pooling_sigreg_term
        )
        valid_peak_count = peak_valid_mask.float().sum().clamp_min(1.0)
        context_weights = context_mask.float()
        teacher_student_context_output_norm = (
            (teacher_peak_emb.float() - context_emb.float()).norm(dim=-1)
            * context_weights
        ).sum() / context_weights.sum().clamp_min(1.0)
        collapse_data: dict[str, torch.Tensor] = {}
        if return_collapse_data:
            pooled_mean = self.pool(teacher_peak_emb, peak_valid_mask)
            collapse_data = {
                "teacher_peak_emb": teacher_peak_emb.detach(),
                "teacher_cls_emb": teacher_cls_emb.detach(),
                "context_emb": context_emb.detach(),
                "context_mask": context_mask.detach(),
                "peak_valid_mask": peak_valid_mask.detach(),
                "target_masks": target_masks.detach(),
                "teacher_target_features": teacher_target_features.detach(),
                "teacher_target_features_normalized": (
                    teacher_target_features_normalized.detach()
                ),
                "teacher_targets": teacher_targets.detach(),
                "predictor_output_features": predictor_output_features.detach(),
                "predictor_output": predictor_output.detach(),
                "pooled_mean": pooled_mean.detach(),
            }
        metrics = {
            "loss": loss,
            "local_global_loss": local_global_loss,
            "jepa_term": jepa_term,
            "jepa_mae_loss": jepa_mae_loss.to(dtype=context_emb.dtype),
            "jepa_mae_mz_loss": jepa_mae_mz_loss.to(dtype=context_emb.dtype),
            "jepa_mae_intensity_loss": jepa_mae_intensity_loss.to(
                dtype=context_emb.dtype
            ),
            "jepa_mae_term": jepa_mae_term,
            "jepa_mae_loss_weight": jepa_mae_loss_weight,
            "jepa_mae_mz_accuracy": jepa_mae_mz_accuracy.to(dtype=context_emb.dtype),
            "jepa_mae_intensity_accuracy": jepa_mae_intensity_accuracy.to(
                dtype=context_emb.dtype
            ),
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
            "target_regularizer_term_over_jepa_term": regularizer_term
            / jepa_term.clamp_min(1e-8),
            "target_sigreg_term_over_jepa_term": sigreg_term
            / jepa_term.clamp_min(1e-8),
            "jepa_mae_term_over_jepa_term": jepa_mae_term
            / jepa_term.clamp_min(1e-8),
            "covariance_pooling_sigreg_term_over_jepa_term": (
                covariance_pooling_sigreg_term / jepa_term.clamp_min(1e-8)
            ),
            "teacher_student_context_output_norm": (
                teacher_student_context_output_norm.to(dtype=context_emb.dtype)
            ),
            "context_fraction": context_mask.float().sum() / valid_peak_count,
            "masked_fraction": target_masks.float().sum() / valid_peak_count,
            "cls_visible_fraction": cls_visible_mask.float().sum() / valid_peak_count,
        }
        if return_collapse_data:
            return metrics, collapse_data
        return metrics

    def compute_next_frame_teacher_embeddings(
        self, batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute teacher embeddings for the next frame."""
        next_frame_mz, next_frame_int, next_frame_valid = self._get_temporal_frame_inputs(
            batch,
            "next_frame",
        )
        teacher_encoder = (
            self.teacher_encoder
            if self.teacher_encoder is not None
            else self.encoder
        )
        if self.teacher_encoder is None:
            teacher_embeddings = teacher_encoder(
                next_frame_mz,
                next_frame_int,
                valid_mask=next_frame_valid,
                visible_mask=next_frame_valid,
                precursor_mz=batch.get("next_frame_precursor_mz", None),
            )
        else:
            with torch.no_grad():
                teacher_embeddings = teacher_encoder(
                    next_frame_mz,
                    next_frame_int,
                    valid_mask=next_frame_valid,
                    visible_mask=next_frame_valid,
                    precursor_mz=batch.get("next_frame_precursor_mz", None),
                )
        teacher_embeddings, _ = self._split_encoder_output(
            teacher_encoder,
            teacher_embeddings,
            next_frame_valid,
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
            precursor_mz=batch.get("frame_precursor_mz", None),
        )  # [B, N, D]
        frame_emb, _ = self._split_encoder_output(
            self.encoder,
            frame_encoded,
            frame_valid,
        )

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
            precursor_mz=batch.get("precursor_mz", None),
        )
        peak_x, cls_x = self._split_encoder_output(self.encoder, encoded, valid)
        return cls_x
