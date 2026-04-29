from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import input_pipeline
from utils.intensity_aware_masking import (
    AWARE_MIXED_MASK_CONFIG,
    INTENSITY_AWARE_MASK_STRATEGY,
    sample_intensity_aware_masks_torch,
)
from utils.spectra_preprocessing import (
    DEFAULT_MIN_PEAK_INTENSITY,
    DEFAULT_PRECURSOR_PEAK_EXCLUSION_WINDOW_DA,
    PEAK_MZ_MAX,
    preprocess_peak_batch_torch,
)


def _load_config(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("masking_eval_config", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.get_config()


def _quantiles(x: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "p05": float(np.quantile(x, 0.05)),
        "p25": float(np.quantile(x, 0.25)),
        "p50": float(np.quantile(x, 0.50)),
        "p75": float(np.quantile(x, 0.75)),
        "p95": float(np.quantile(x, 0.95)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def _binary_rate_ci(k: float, n: float) -> dict[str, float]:
    p = float(k / n)
    se = float(np.sqrt(p * (1.0 - p) / n))
    return {"rate": p, "se": se, "ci95_low": p - 1.96 * se, "ci95_high": p + 1.96 * se}


def _safe_entropy_terms(p: np.ndarray) -> np.ndarray:
    out = np.zeros_like(p)
    positive = p > 0.0
    out[positive] = -p[positive] * np.log(p[positive])
    return out


def _safe_set_entropy(p: np.ndarray, mask: np.ndarray) -> np.ndarray:
    terms = _safe_entropy_terms(p)
    return (terms * mask).sum(axis=1)


def _renormalized_entropy(p: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mass = (p * mask).sum(axis=1)
    set_p = np.divide(
        p * mask,
        mass[:, None],
        out=np.zeros_like(p),
        where=mass[:, None] > 0.0,
    )
    return _safe_entropy_terms(set_p).sum(axis=1)


def _mask_probability_mass(p: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return (p * mask).sum(axis=1)


def _entropy_split_stats(
    *,
    p: np.ndarray,
    valid: np.ndarray,
    context: np.ndarray,
    targets: np.ndarray,
    target_union: np.ndarray,
    ignored: np.ndarray,
) -> dict[str, Any]:
    total_entropy = _safe_entropy_terms(p).sum(axis=1)
    valid_count = valid.sum(axis=1).astype(np.float64)
    norm_entropy = np.divide(
        total_entropy,
        np.log(np.maximum(valid_count, 2.0)),
        out=np.zeros_like(total_entropy),
        where=valid_count > 1,
    )
    effective_peaks = np.exp(total_entropy)

    masses = {
        "context": _mask_probability_mass(p, context),
        "target_union": _mask_probability_mass(p, target_union),
        "ignored": _mask_probability_mass(p, ignored),
    }
    entropy_values = {
        "context": _safe_set_entropy(p, context),
        "target_union": _safe_set_entropy(p, target_union),
        "ignored": _safe_set_entropy(p, ignored),
    }
    entropy_fractions = {
        name: np.divide(
            values,
            total_entropy,
            out=np.zeros_like(values),
            where=total_entropy > 0.0,
        )
        for name, values in entropy_values.items()
    }
    surprise_per_mass = {
        name: np.divide(
            entropy_values[name],
            masses[name],
            out=np.zeros_like(masses[name]),
            where=masses[name] > 0.0,
        )
        for name in masses
    }
    renorm_entropy = {
        name: _renormalized_entropy(p, mask)
        for name, mask in (
            ("context", context),
            ("target_union", target_union),
            ("ignored", ignored),
        )
    }
    renorm_effective_peaks = {
        name: np.exp(values)
        for name, values in renorm_entropy.items()
    }

    target_view_mass = np.stack(
        [_mask_probability_mass(p, targets[:, idx]) for idx in range(targets.shape[1])],
        axis=1,
    )
    target_view_entropy = np.stack(
        [_safe_set_entropy(p, targets[:, idx]) for idx in range(targets.shape[1])],
        axis=1,
    )
    target_view_entropy_fraction = np.divide(
        target_view_entropy,
        total_entropy[:, None],
        out=np.zeros_like(target_view_entropy),
        where=total_entropy[:, None] > 0.0,
    )
    overlap_mask = targets[:, 0] & targets[:, 1] if targets.shape[1] >= 2 else np.zeros_like(context)
    overlap_mass = _mask_probability_mass(p, overlap_mask)
    overlap_entropy = _safe_set_entropy(p, overlap_mask)

    top_idx = np.argmax(np.where(valid, p, -1.0), axis=1)
    rows = np.arange(p.shape[0])
    top_mass = p[rows, top_idx]
    top_context = context[rows, top_idx]
    top_target = target_union[rows, top_idx]
    top_ignored = ignored[rows, top_idx]

    q = np.stack([masses["context"], masses["target_union"], masses["ignored"]], axis=1)
    group_entropy = _safe_entropy_terms(q).sum(axis=1)
    conditional_entropy = (
        q[:, 0] * renorm_entropy["context"]
        + q[:, 1] * renorm_entropy["target_union"]
        + q[:, 2] * renorm_entropy["ignored"]
    )
    chain_error = np.abs(total_entropy - group_entropy - conditional_entropy)

    spearman_target = stats.spearmanr(norm_entropy, masses["target_union"])
    spearman_context = stats.spearmanr(norm_entropy, masses["context"])
    spearman_ignored = stats.spearmanr(norm_entropy, masses["ignored"])

    return {
        "spectrum_entropy_nats": _quantiles(total_entropy),
        "spectrum_entropy_bits": _quantiles(total_entropy / np.log(2.0)),
        "normalized_entropy": _quantiles(norm_entropy),
        "effective_peak_count": _quantiles(effective_peaks),
        "probability_mass": {
            name: _quantiles(values)
            for name, values in masses.items()
        } | {
            "target_per_view": _quantiles(target_view_mass.reshape(-1)),
            "target_view_overlap": _quantiles(overlap_mass),
        },
        "entropy_contribution_nats": {
            name: _quantiles(values)
            for name, values in entropy_values.items()
        } | {
            "target_per_view": _quantiles(target_view_entropy.reshape(-1)),
            "target_view_overlap": _quantiles(overlap_entropy),
        },
        "entropy_fraction": {
            name: _quantiles(values)
            for name, values in entropy_fractions.items()
        } | {
            "target_per_view": _quantiles(target_view_entropy_fraction.reshape(-1)),
        },
        "entropy_fraction_minus_probability_mass": {
            name: _quantiles(entropy_fractions[name] - masses[name])
            for name in masses
        },
        "surprise_nats_per_probability_mass": {
            name: _quantiles(values)
            for name, values in surprise_per_mass.items()
        },
        "renormalized_set_entropy_nats": {
            name: _quantiles(values)
            for name, values in renorm_entropy.items()
        },
        "renormalized_set_effective_peak_count": {
            name: _quantiles(values)
            for name, values in renorm_effective_peaks.items()
        },
        "partition_chain_rule": {
            "group_entropy_nats": _quantiles(group_entropy),
            "conditional_entropy_nats": _quantiles(conditional_entropy),
            "group_entropy_fraction": _quantiles(
                np.divide(
                    group_entropy,
                    total_entropy,
                    out=np.zeros_like(group_entropy),
                    where=total_entropy > 0.0,
                )
            ),
            "conditional_entropy_fraction": _quantiles(
                np.divide(
                    conditional_entropy,
                    total_entropy,
                    out=np.zeros_like(conditional_entropy),
                    where=total_entropy > 0.0,
                )
            ),
            "max_chain_rule_abs_error": float(chain_error.max()),
        },
        "base_peak": {
            "probability_mass": _quantiles(top_mass),
            "context_rate": float(top_context.mean()),
            "target_union_rate": float(top_target.mean()),
            "ignored_rate": float(top_ignored.mean()),
        },
        "entropy_mass_correlations": {
            "normalized_entropy_vs_context_mass_spearman_r": float(spearman_context.statistic),
            "normalized_entropy_vs_context_mass_spearman_p": float(spearman_context.pvalue),
            "normalized_entropy_vs_target_union_mass_spearman_r": float(spearman_target.statistic),
            "normalized_entropy_vs_target_union_mass_spearman_p": float(spearman_target.pvalue),
            "normalized_entropy_vs_ignored_mass_spearman_r": float(spearman_ignored.statistic),
            "normalized_entropy_vs_ignored_mass_spearman_p": float(spearman_ignored.pvalue),
        },
    }


def _rank_exposure_stats(selection: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    observed = selection.sum(axis=0).astype(np.float64)
    exposure = valid.sum(axis=0).astype(np.float64)
    total_observed = observed.sum()
    expected = exposure * (total_observed / exposure.sum())
    chi = stats.chisquare(observed, f_exp=expected)
    rates = observed / np.maximum(exposure, 1.0)
    slot = np.arange(len(rates), dtype=np.float64)
    exposed = exposure > 0
    spearman = stats.spearmanr(slot[exposed], rates[exposed])
    return {
        "mean_rate": float(total_observed / exposure.sum()),
        "min_rate": float(rates[exposed].min()),
        "max_rate": float(rates[exposed].max()),
        "first_slot_rate": float(rates[0]),
        "last_exposed_slot_rate": float(rates[np.where(exposed)[0][-1]]),
        "chi_square": float(chi.statistic),
        "chi_square_p": float(chi.pvalue),
        "spearman_r": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "rates_by_slot": [float(v) for v in rates],
    }


def _by_valid_count_bin(
    valid_count: np.ndarray,
    metrics: dict[str, np.ndarray],
) -> dict[str, dict[str, dict[str, float]]]:
    bins = (
        ("01_08", 1, 8),
        ("09_16", 9, 16),
        ("17_32", 17, 32),
        ("33_48", 33, 48),
        ("49_64", 49, 64),
    )
    out: dict[str, dict[str, dict[str, float]]] = {}
    for label, lo, hi in bins:
        keep = (valid_count >= lo) & (valid_count <= hi)
        if not keep.any():
            continue
        out[label] = {
            name: _quantiles(values[keep])
            for name, values in metrics.items()
        }
        out[label]["num_spectra"] = {"mean": float(keep.sum())}
    return out


def _mannwhitney_auc(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    max_n = 200_000
    if len(a) > max_n:
        a = a[rng.choice(len(a), size=max_n, replace=False)]
    if len(b) > max_n:
        b = b[rng.choice(len(b), size=max_n, replace=False)]
    result = stats.mannwhitneyu(a, b, alternative="two-sided")
    auc = float(result.statistic / (len(a) * len(b)))
    return {"auc": auc, "p": float(result.pvalue)}


def _run_lengths_and_widths(mask: np.ndarray, mz_da: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lengths: list[int] = []
    widths: list[float] = []
    for row_mask, row_mz, row_valid in zip(mask, mz_da, valid, strict=True):
        active = np.flatnonzero(row_valid)
        compressed = row_mask[active]
        positions = np.flatnonzero(compressed)
        if len(positions) == 0:
            continue
        starts = [int(positions[0])]
        ends: list[int] = []
        for prev, current in zip(positions[:-1], positions[1:], strict=False):
            if int(current) != int(prev) + 1:
                ends.append(int(prev))
                starts.append(int(current))
        ends.append(int(positions[-1]))
        active_mz = row_mz[active]
        for start, end in zip(starts, ends, strict=True):
            lengths.append(end - start + 1)
            widths.append(float(active_mz[end] - active_mz[start]))
    return np.asarray(lengths, dtype=np.float64), np.asarray(widths, dtype=np.float64)


def _nearest_context_da(
    target_masks: np.ndarray,
    context: np.ndarray,
    mz_da: np.ndarray,
) -> np.ndarray:
    distances: list[float] = []
    for row_targets, row_context, row_mz in zip(target_masks, context, mz_da, strict=True):
        context_mz = row_mz[row_context]
        if len(context_mz) == 0:
            continue
        for row_target in row_targets:
            target_mz = row_mz[row_target]
            if len(target_mz) == 0:
                continue
            diff = np.abs(target_mz[:, None] - context_mz[None, :])
            distances.extend(np.min(diff, axis=1).tolist())
    return np.asarray(distances, dtype=np.float64)


def _collect_values(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return values[mask]


def _evaluate_strategy(
    *,
    name: str,
    peak_valid_mask: torch.Tensor,
    mz_da: np.ndarray,
    intensity: np.ndarray,
    precursor_da: np.ndarray,
    cfg: Any,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    strategy = input_pipeline._normalize_mask_strategy_name(name)
    if strategy == INTENSITY_AWARE_MASK_STRATEGY:
        context, targets = sample_intensity_aware_masks_torch(
            peak_valid_mask,
            torch.as_tensor(
                intensity,
                dtype=torch.float32,
                device=peak_valid_mask.device,
            ),
            torch.as_tensor(
                mz_da,
                dtype=torch.float32,
                device=peak_valid_mask.device,
            ),
            num_target_blocks=int(cfg.get("jepa_num_target_blocks", 2)),
            **{
                key: float(cfg.get(f"jepa_intensity_aware_{key}", value))
                for key, value in AWARE_MIXED_MASK_CONFIG.items()
            },
        )
    else:
        context, targets = input_pipeline._sample_block_masks_torch(
            peak_valid_mask,
            num_target_blocks=int(cfg.get("jepa_num_target_blocks", 2)),
            context_fraction=float(cfg.get("jepa_context_fraction", 0.5)),
            target_fraction=float(cfg.get("jepa_target_fraction", 0.25)),
            block_min_len=int(cfg.get("jepa_block_min_len", 1)),
            mask_strategy=name,
            mask_lengths=tuple(int(v) for v in cfg.get("jepa_mask_lengths", (1, 2, 4, 8, 16))),
            mask_round_from=int(cfg.get("jepa_mask_round_from", len(cfg.get("jepa_mask_lengths", (1, 2, 4, 8, 16))))),
        )
    valid = peak_valid_mask.cpu().numpy().astype(bool)
    context_np = context.cpu().numpy().astype(bool)
    targets_np = targets.cpu().numpy().astype(bool)
    target_union = targets_np.any(axis=1)
    total_union = context_np | target_union
    ignored = valid & ~total_union
    p = intensity * valid
    p = p / p.sum(axis=1, keepdims=True)
    valid_count = valid.sum(axis=1).astype(np.float64)
    context_count = context_np.sum(axis=1).astype(np.float64)
    target_counts = targets_np.sum(axis=2).astype(np.float64)
    target_union_count = target_union.sum(axis=1).astype(np.float64)
    total_union_count = total_union.sum(axis=1).astype(np.float64)
    ignored_count = ignored.sum(axis=1).astype(np.float64)
    pair_overlap = (targets_np[:, 0] & targets_np[:, 1]).sum(axis=1).astype(np.float64)
    pair_union = (targets_np[:, 0] | targets_np[:, 1]).sum(axis=1).astype(np.float64)
    pair_jaccard = pair_overlap / np.maximum(pair_union, 1.0)

    context_lengths, context_widths = _run_lengths_and_widths(context_np, mz_da, valid)
    target_lengths, target_widths = _run_lengths_and_widths(target_union, mz_da, valid)
    nearest_context = _nearest_context_da(targets_np, context_np, mz_da)

    rng = np.random.default_rng(seed)
    valid_mz = _collect_values(mz_da, valid)
    context_mz = _collect_values(mz_da, context_np)
    target_mz = _collect_values(mz_da, target_union)
    valid_intensity = _collect_values(intensity, valid)
    context_intensity = _collect_values(intensity, context_np)
    target_intensity = _collect_values(intensity, target_union)
    neutral_loss = precursor_da[:, None] - mz_da
    valid_nl = _collect_values(neutral_loss, valid)
    target_nl = _collect_values(neutral_loss, target_union)

    return {
        "counts": {
            "valid": _quantiles(valid_count),
            "context": _quantiles(context_count),
            "target_per_view": _quantiles(target_counts.reshape(-1)),
            "target_union": _quantiles(target_union_count),
            "context_or_target_union": _quantiles(total_union_count),
            "ignored_valid": _quantiles(ignored_count),
            "target_view_overlap": _quantiles(pair_overlap),
            "target_view_jaccard": _quantiles(pair_jaccard),
        },
        "fractions": {
            "context_over_valid": _quantiles(context_count / valid_count),
            "target_per_view_over_valid": _quantiles(target_counts.reshape(-1) / np.repeat(valid_count, targets_np.shape[1])),
            "target_union_over_valid": _quantiles(target_union_count / valid_count),
            "context_or_target_union_over_valid": _quantiles(total_union_count / valid_count),
            "ignored_over_valid": _quantiles(ignored_count / valid_count),
        },
        "fractions_by_valid_peak_count": _by_valid_count_bin(
            valid_count,
            {
                "context_over_valid": context_count / valid_count,
                "target_union_over_valid": target_union_count / valid_count,
                "context_or_target_union_over_valid": total_union_count / valid_count,
                "ignored_over_valid": ignored_count / valid_count,
            },
        ),
        "coverage_ci": {
            "context": _binary_rate_ci(float(context_np.sum()), float(valid.sum())),
            "target_union": _binary_rate_ci(float(target_union.sum()), float(valid.sum())),
            "context_or_target_union": _binary_rate_ci(float(total_union.sum()), float(valid.sum())),
            "ignored": _binary_rate_ci(float(ignored.sum()), float(valid.sum())),
        },
        "rank_exposure": {
            "context": _rank_exposure_stats(context_np, valid),
            "target_union": _rank_exposure_stats(target_union, valid),
            "ignored": _rank_exposure_stats(ignored, valid),
        },
        "block_geometry": {
            "context_run_length": _quantiles(context_lengths),
            "context_run_width_da": _quantiles(context_widths),
            "target_union_run_length": _quantiles(target_lengths),
            "target_union_run_width_da": _quantiles(target_widths),
            "target_to_nearest_context_da": _quantiles(nearest_context),
        },
        "mass_intensity_bias": {
            "valid_mz_da": _quantiles(valid_mz),
            "context_mz_da": _quantiles(context_mz),
            "target_union_mz_da": _quantiles(target_mz),
            "target_vs_valid_mz_mannwhitney_auc": _mannwhitney_auc(target_mz, valid_mz, rng),
            "valid_intensity": _quantiles(valid_intensity),
            "context_intensity": _quantiles(context_intensity),
            "target_union_intensity": _quantiles(target_intensity),
            "target_vs_valid_intensity_mannwhitney_auc": _mannwhitney_auc(target_intensity, valid_intensity, rng),
            "valid_neutral_loss_da": _quantiles(valid_nl),
            "target_union_neutral_loss_da": _quantiles(target_nl),
            "target_vs_valid_neutral_loss_mannwhitney_auc": _mannwhitney_auc(target_nl, valid_nl, rng),
        },
        "entropy": _entropy_split_stats(
            p=p,
            valid=valid,
            context=context_np,
            targets=targets_np,
            target_union=target_union,
            ignored=ignored,
        ),
    }


def _sample_raw_rows(
    *,
    gems_dir: Path,
    train_shards: list[str],
    train_lengths: list[int],
    num_samples: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    starts = np.zeros(len(train_lengths) + 1, dtype=np.int64)
    np.cumsum(np.asarray(train_lengths, dtype=np.int64), out=starts[1:])
    global_indices = rng.integers(0, int(starts[-1]), size=int(num_samples), endpoint=False)
    spectra_parts: list[np.ndarray] = []
    precursor_parts: list[np.ndarray] = []
    for shard_idx, shard_name in enumerate(train_shards):
        lo, hi = int(starts[shard_idx]), int(starts[shard_idx + 1])
        chosen = global_indices[(global_indices >= lo) & (global_indices < hi)] - lo
        if len(chosen) == 0:
            continue
        order = np.argsort(chosen)
        chosen_sorted = chosen[order]
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        shard_dir = gems_dir / "train" / shard_name
        spectra = np.load(shard_dir / "spectra.npy", mmap_mode="r")
        precursor = np.load(shard_dir / "precursor_mz_raw.npy", mmap_mode="r")
        spectra_parts.append(np.asarray(spectra[chosen_sorted], dtype=np.float32)[inverse])
        precursor_parts.append(np.asarray(precursor[chosen_sorted], dtype=np.float32)[inverse])
    spectra_np = np.concatenate(spectra_parts, axis=0)
    precursor_np = np.concatenate(precursor_parts, axis=0)
    return torch.from_numpy(spectra_np), torch.from_numpy(precursor_np)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/gems_small.py"))
    parser.add_argument("--num-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        help="Mask strategies to evaluate. Defaults to config strategy plus contiguous/ragged/all.",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/masking_statistics.json"))
    args = parser.parse_args()

    cfg = _load_config(args.config)
    artifact_dir = Path(str(cfg.get("artifact_dir", "data/gems_artifacts"))).expanduser()
    gems_dir = artifact_dir / "gems"
    metadata = json.loads((gems_dir / "metadata.json").read_text())
    spectra, precursor_raw = _sample_raw_rows(
        gems_dir=gems_dir,
        train_shards=list(metadata["train_shards"]),
        train_lengths=[int(v) for v in metadata["train_lengths"]],
        num_samples=args.num_samples,
        seed=args.seed,
    )
    pre = preprocess_peak_batch_torch(
        spectra[:, 0, :],
        spectra[:, 1, :],
        precursor_raw,
        num_peaks=int(cfg.get("num_peaks", 64)),
        peak_drop_min_intensity=float(cfg.get("peak_drop_min_intensity", cfg.get("min_peak_intensity", DEFAULT_MIN_PEAK_INTENSITY))),
        peak_ordering=str(cfg.get("peak_ordering", "mz")),
        max_precursor_mz=float(cfg.get("max_precursor_mz", 1000.0)),
        precursor_peak_exclusion_window_da=float(cfg.get("precursor_peak_exclusion_window_da", DEFAULT_PRECURSOR_PEAK_EXCLUSION_WINDOW_DA)),
        min_peak_intensity=float(cfg.get("min_peak_intensity", DEFAULT_MIN_PEAK_INTENSITY)),
    )
    valid = pre["peak_valid_mask"]
    nonempty = valid.any(dim=1)
    for key in ("peak_mz", "peak_intensity", "peak_valid_mask", "precursor_mz"):
        pre[key] = pre[key][nonempty]
    precursor_raw = precursor_raw[nonempty]

    configured = str(cfg.get("jepa_mask_strategy", "contiguous")).lower()
    if args.strategies is None:
        strategies = tuple(dict.fromkeys([configured, "contiguous", "ragged", "all"]))
    else:
        strategies = tuple(args.strategies)

    jepa_policy: dict[str, Any] = {
        "num_target_blocks": int(cfg.get("jepa_num_target_blocks", 2)),
        "mask_strategy": configured,
    }
    if input_pipeline._normalize_mask_strategy_name(configured) == INTENSITY_AWARE_MASK_STRATEGY:
        jepa_policy["intensity_aware"] = {
            key: float(cfg.get(f"jepa_intensity_aware_{key}", value))
            for key, value in AWARE_MIXED_MASK_CONFIG.items()
        }
    else:
        jepa_policy |= {
            "context_fraction": float(cfg.get("jepa_context_fraction", 0.5)),
            "target_fraction": float(cfg.get("jepa_target_fraction", 0.25)),
            "block_min_len": int(cfg.get("jepa_block_min_len", 1)),
            "mask_lengths": list(cfg.get("jepa_mask_lengths", (1, 2, 4, 8, 16))),
            "mask_round_from": int(cfg.get("jepa_mask_round_from", len(cfg.get("jepa_mask_lengths", (1, 2, 4, 8, 16))))),
        }

    mz_da = pre["peak_mz"].cpu().numpy().astype(np.float64) * PEAK_MZ_MAX
    intensity = pre["peak_intensity"].cpu().numpy().astype(np.float64)
    precursor_da = precursor_raw.cpu().numpy().astype(np.float64)
    result = {
        "config": str(args.config),
        "artifact": str(gems_dir),
        "sampled_spectra": int(args.num_samples),
        "nonempty_spectra": int(pre["peak_valid_mask"].shape[0]),
        "num_peaks": int(cfg.get("num_peaks", 64)),
        "peak_ordering": str(cfg.get("peak_ordering", "mz")),
        "use_precursor_token": bool(cfg.get("use_precursor_token", False)),
        "jepa": jepa_policy,
        "preprocessed_valid_peak_count": _quantiles(pre["peak_valid_mask"].sum(dim=1).cpu().numpy().astype(np.float64)),
        "strategies": {},
    }
    for offset, strategy in enumerate(strategies):
        result["strategies"][strategy] = _evaluate_strategy(
            name=strategy,
            peak_valid_mask=pre["peak_valid_mask"],
            mz_da=mz_da,
            intensity=intensity,
            precursor_da=precursor_da,
            cfg=cfg,
            seed=args.seed + 1009 * offset,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))

    primary = result["strategies"][configured]
    print(json.dumps({
        "output": str(args.output),
        "config": result["config"],
        "sampled_spectra": result["sampled_spectra"],
        "nonempty_spectra": result["nonempty_spectra"],
        "configured_strategy": configured,
        "valid_peak_count": result["preprocessed_valid_peak_count"],
        "configured_counts": primary["counts"],
        "configured_fractions": primary["fractions"],
        "configured_block_geometry": primary["block_geometry"],
        "configured_rank_exposure": {
            key: {
                metric: value
                for metric, value in stats_dict.items()
                if metric != "rates_by_slot"
            }
            for key, stats_dict in primary["rank_exposure"].items()
        },
        "configured_mass_intensity_bias": {
            "target_vs_valid_mz_auc": primary["mass_intensity_bias"]["target_vs_valid_mz_mannwhitney_auc"],
            "target_vs_valid_intensity_auc": primary["mass_intensity_bias"]["target_vs_valid_intensity_mannwhitney_auc"],
            "target_vs_valid_neutral_loss_auc": primary["mass_intensity_bias"]["target_vs_valid_neutral_loss_mannwhitney_auc"],
        },
        "configured_entropy": {
            "spectrum_entropy_bits": primary["entropy"]["spectrum_entropy_bits"],
            "normalized_entropy": primary["entropy"]["normalized_entropy"],
            "effective_peak_count": primary["entropy"]["effective_peak_count"],
            "probability_mass": primary["entropy"]["probability_mass"],
            "entropy_fraction": primary["entropy"]["entropy_fraction"],
            "base_peak": primary["entropy"]["base_peak"],
            "entropy_mass_correlations": primary["entropy"]["entropy_mass_correlations"],
        },
    }, indent=2))


if __name__ == "__main__":
    main()
