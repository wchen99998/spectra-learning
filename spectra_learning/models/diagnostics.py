import math

import torch
import torch.nn.functional as F


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
