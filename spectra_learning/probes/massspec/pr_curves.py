from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PrecisionRecallCurve:
    label: str
    targets: np.ndarray
    probabilities: np.ndarray
    title: str

    @property
    def class_labels(self) -> list[str]:
        return [f"not {self.label}", self.label]

    @property
    def two_class_probabilities(self) -> np.ndarray:
        return np.stack([1.0 - self.probabilities, self.probabilities], axis=1)


def build_precision_recall_curve(
    *,
    prefix: str,
    name: str,
    pred: np.ndarray,
    target: np.ndarray,
) -> PrecisionRecallCurve:
    return PrecisionRecallCurve(
        label=name,
        targets=(target.reshape(-1) > 0).astype(np.int64),
        probabilities=pred.reshape(-1).astype(np.float64),
        title=f"{prefix} {name} precision-recall",
    )


def precision_recall_points(curve: PrecisionRecallCurve) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-curve.probabilities)
    target = curve.targets[order] > 0
    true_positives = np.cumsum(target, dtype=np.float64)
    false_positives = np.cumsum(~target, dtype=np.float64)
    positives = float(np.count_nonzero(target))
    precision = true_positives / np.maximum(true_positives + false_positives, 1.0)
    recall = true_positives / positives if positives > 0 else np.zeros_like(precision)
    return (
        np.concatenate([np.asarray([0.0], dtype=np.float64), recall]),
        np.concatenate([np.asarray([1.0], dtype=np.float64), precision]),
    )
