"""Small, CPU-testable statistics for the periodic-teacher investigation."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def macro_auc(prediction: np.ndarray, target: np.ndarray) -> float:
    """Macro AUROC with average ranks for ties and undefined bits excluded."""
    positive = target.sum(0)
    negative = len(target) - positive
    valid = (positive > 0) & (negative > 0)
    ranks = rankdata(prediction[:, valid], axis=0)
    numerator = (ranks * target[:, valid]).sum(0) - positive[valid] * (positive[valid] + 1) / 2
    return float(np.mean(numerator / (positive[valid] * negative[valid])))


def rank_correlation(a: np.ndarray, b: np.ndarray, controls: np.ndarray | None = None) -> float:
    """Spearman, or partial Spearman with tied ranks and an intercept."""
    x, y = rankdata(a), rankdata(b)
    if controls is not None:
        design = np.column_stack([np.ones(len(x)), rankdata(controls, axis=0)])
        x = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
        y = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    return float(np.corrcoef(x, y)[0, 1])


def nearest_mass(train_mass: np.ndarray, query_mass: np.ndarray) -> np.ndarray:
    """Actual closest mass, considering both sides of searchsorted."""
    order = np.argsort(train_mass, kind="stable")
    values = train_mass[order]
    right = np.searchsorted(values, query_mass).clip(0, len(values) - 1)
    left = (right - 1).clip(0)
    use_left = abs(values[left] - query_mass) <= abs(values[right] - query_mass)
    return order[np.where(use_left, left, right)]


def paired_mean_interval(a: np.ndarray, b: np.ndarray, seed: int = 20260929, repeats: int = 2000) -> dict:
    delta = np.asarray(b) - np.asarray(a)
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(repeats)])
    return {"delta": float(delta.mean()), "ci95": np.quantile(means, [.025, .975]).tolist(), "repeats": repeats}
