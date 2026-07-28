from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.metrics import r2_score

from spectra_learning.config.msg_probe import validate_msg_probe_config
from spectra_learning.probes.massspec.msg_settings import MsgProbeTaskSpec
from spectra_learning.probes.massspec.pr_curves import build_precision_recall_curve


EpochState = dict[str, Any]


def probe_task_names(task_spec: MsgProbeTaskSpec) -> tuple[str, ...]:
    task_names = task_spec.regression_tasks + task_spec.binary_tasks
    if task_spec.maccs_bits > 0:
        task_names += (task_spec.fingerprint_task,)
    return task_names


def probe_prediction_names(task_spec: MsgProbeTaskSpec) -> tuple[str, ...]:
    return probe_task_names(task_spec)


def probe_task_output_dims(task_spec: MsgProbeTaskSpec) -> dict[str, int]:
    if task_spec.maccs_bits <= 0:
        return {}
    return {
        task_spec.fingerprint_task: (
            len(task_spec.regression_tasks) + task_spec.maccs_bits
        )
    }


def new_epoch_state(task_spec: MsgProbeTaskSpec) -> EpochState:
    task_names = probe_prediction_names(task_spec)
    return {
        "count": 0,
        "predictions": {name: [] for name in task_names},
        "targets": {name: [] for name in task_names},
    }


def merge_epoch_states(
    states: list[EpochState],
    task_spec: MsgProbeTaskSpec,
) -> EpochState:
    merged = new_epoch_state(task_spec)
    merged["count"] = sum(int(state["count"]) for state in states)
    merged_predictions = merged["predictions"]
    merged_targets = merged["targets"]
    for state in states:
        predictions = state["predictions"]
        targets = state["targets"]
        for name in probe_prediction_names(task_spec):
            merged_predictions[name].extend(predictions[name])
            merged_targets[name].extend(targets[name])
    return merged


def score_epoch_state(
    *,
    prefix: str,
    epoch_state: EpochState,
    task_spec: MsgProbeTaskSpec,
    include_pr_curves: bool = False,
) -> dict[str, Any]:
    count = int(epoch_state["count"])
    metrics: dict[str, Any] = {f"{prefix}/samples": float(count)}
    regression_r2_values = []
    regression_mae_values = []
    predictions = epoch_state["predictions"]
    targets = epoch_state["targets"]
    for name in task_spec.regression_tasks:
        pred = np.concatenate(predictions[name], axis=0)
        target = np.concatenate(targets[name], axis=0)
        metrics[f"{prefix}/r2_{name}"] = float(r2_score(target, pred))
        metrics[f"{prefix}/mae_{name}"] = float(np.mean(np.abs(target - pred)))
        regression_r2_values.append(metrics[f"{prefix}/r2_{name}"])
        regression_mae_values.append(metrics[f"{prefix}/mae_{name}"])
    for name in task_spec.binary_tasks:
        pred = np.concatenate(predictions[name], axis=0).astype(np.float64)
        target = np.concatenate(targets[name], axis=0).astype(np.float64)
        metrics.update(_binary_metrics(prefix, name, pred, target))
        if include_pr_curves and name in ("fluorine", "sulfur"):
            metrics[f"{prefix}/pr_curve_{name}"] = build_precision_recall_curve(
                prefix=prefix,
                name=name,
                pred=pred,
                target=target,
            )
    if task_spec.maccs_bits > 0:
        fingerprint_task = task_spec.fingerprint_task
        pred = np.concatenate(predictions[fingerprint_task], axis=0)
        target = np.concatenate(targets[fingerprint_task], axis=0)
        metrics.update(fingerprint_metrics(prefix, fingerprint_task, pred, target))
    if regression_r2_values:
        metrics[f"{prefix}/r2_mean"] = float(np.mean(regression_r2_values))
        metrics[f"{prefix}/mae_mean"] = float(np.mean(regression_mae_values))
    return metrics


def _binary_metrics(
    prefix: str,
    name: str,
    pred: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    positives = float(target.sum())
    negatives = float(target.shape[0]) - positives
    if positives > 0 and negatives > 0:
        order = np.argsort(pred)
        target_ordered = target[order]
        negatives_before = np.cumsum(1.0 - target_ordered)
        auc = float(
            (target_ordered * negatives_before).sum() / (positives * negatives)
        )
        target_descending = target_ordered[::-1]
        true_positives_at_rank = np.cumsum(target_descending)
        ranks = np.arange(1, target_descending.shape[0] + 1, dtype=np.float64)
        average_precision = float(
            (target_descending * true_positives_at_rank / ranks).sum() / positives
        )
    else:
        auc = float("nan")
        average_precision = float("nan")
    predicted = pred >= 0.5
    target_bits = target > 0
    true_positives = float(np.count_nonzero(predicted & target_bits))
    predicted_positives = float(np.count_nonzero(predicted))
    return {
        f"{prefix}/positive_{name}": positives,
        f"{prefix}/auc_{name}": auc,
        f"{prefix}/average_precision_{name}": average_precision,
        f"{prefix}/recall_{name}": (
            true_positives / positives if positives > 0 else float("nan")
        ),
        f"{prefix}/precision_{name}": (
            true_positives / predicted_positives
            if predicted_positives > 0
            else float("nan")
        ),
    }


def fingerprint_metrics(
    prefix: str,
    fingerprint_task: str,
    pred: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    positives = target.sum(axis=0)
    valid_metric_mask = (positives > 0) & (positives < target.shape[0])
    if np.count_nonzero(valid_metric_mask) > 0:
        valid_target = target[:, valid_metric_mask].astype(np.float64)
        valid_pred = pred[:, valid_metric_mask].astype(np.float64)
        valid_positives = positives[valid_metric_mask].astype(np.float64)
        valid_negatives = float(target.shape[0]) - valid_positives
        ascending = np.argsort(valid_pred, axis=0)
        target_ascending = np.take_along_axis(valid_target, ascending, axis=0)
        negatives_before = np.cumsum(1.0 - target_ascending, axis=0)
        auc_values = (
            (target_ascending * negatives_before).sum(axis=0)
            / (valid_positives * valid_negatives)
        )
        target_descending = target_ascending[::-1]
        true_positives_at_rank = np.cumsum(target_descending, axis=0)
        ranks = np.arange(
            1,
            target_descending.shape[0] + 1,
            dtype=np.float64,
        )[:, None]
        average_precision_values = (
            (target_descending * true_positives_at_rank / ranks).sum(axis=0)
            / valid_positives
        )
    else:
        auc_values = np.asarray([], dtype=np.float64)
        average_precision_values = np.asarray([], dtype=np.float64)
    positive_mask = positives > 0
    bit_pred = pred >= 0.5
    target_bits = target > 0
    true_positives = (bit_pred & target_bits).sum(axis=0)
    predicted_positives = bit_pred.sum(axis=0)
    recall_values = true_positives[positive_mask] / positives[positive_mask]
    precision_values = np.divide(
        true_positives[positive_mask],
        predicted_positives[positive_mask],
        out=np.zeros_like(true_positives[positive_mask], dtype=np.float64),
        where=predicted_positives[positive_mask] > 0,
    )
    intersection = np.count_nonzero(bit_pred & target_bits, axis=1)
    union = np.count_nonzero(bit_pred | target_bits, axis=1)
    tanimoto_values = intersection / np.maximum(union, 1)
    dot = np.sum(pred * target, axis=1)
    cosine_values = dot / np.maximum(
        np.linalg.norm(pred, axis=1) * np.linalg.norm(target, axis=1),
        1e-12,
    )
    return {
        f"{prefix}/num_{fingerprint_task}_auc_bits": float(len(auc_values)),
        f"{prefix}/num_{fingerprint_task}_average_precision_bits": float(
            len(average_precision_values)
        ),
        f"{prefix}/num_{fingerprint_task}_recall_bits": float(len(recall_values)),
        f"{prefix}/num_{fingerprint_task}_precision_bits": float(
            len(precision_values)
        ),
        f"{prefix}/auc_{fingerprint_task}_mean": (
            float(np.mean(auc_values)) if len(auc_values) else float("nan")
        ),
        f"{prefix}/average_precision_{fingerprint_task}_mean": (
            float(np.mean(average_precision_values))
            if len(average_precision_values)
            else float("nan")
        ),
        f"{prefix}/recall_{fingerprint_task}_mean": (
            float(np.mean(recall_values)) if len(recall_values) else float("nan")
        ),
        f"{prefix}/precision_{fingerprint_task}_mean": (
            float(np.mean(precision_values)) if len(precision_values) else float("nan")
        ),
        f"{prefix}/tanimoto_{fingerprint_task}_mean": float(
            np.mean(tanimoto_values)
        ),
        f"{prefix}/cosine_{fingerprint_task}_mean": float(np.mean(cosine_values)),
    }


def resolve_probe_warmup_steps(config: Any, steps_per_epoch: int) -> int:
    warmup_epochs = config.get("msg_probe_warmup_epochs", None)
    if warmup_epochs is not None:
        return int(round(float(warmup_epochs) * steps_per_epoch))
    return int(config.get("msg_probe_warmup_steps", 100))


def resolve_msg_probe_select_metric(config: Any) -> str:
    validate_msg_probe_config(config)
    return str(
        config.get(
            "msg_probe_select_metric",
            "msg_probe/val/auc_fluorine",
        )
    )


def msg_probe_metric_higher_is_better(metric_key: str) -> bool:
    return "/mae_" not in metric_key


def msg_probe_variant_metric_key(variant: str, metric_key: str) -> str:
    variant_prefix = f"msg_probe/{variant}/"
    if metric_key.startswith(variant_prefix):
        return metric_key
    if metric_key.startswith("msg_probe/"):
        return variant_prefix + metric_key[len("msg_probe/") :]
    return metric_key


def average_metric_dicts(metric_dicts: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    artifacts: dict[str, Any] = {}
    for metrics in metric_dicts:
        for key, value in metrics.items():
            if not isinstance(value, (int, float, np.number)):
                artifacts.setdefault(key, value)
                continue
            totals[key] = totals.get(key, 0.0) + value
            counts[key] = counts.get(key, 0) + 1
    return {
        **{key: totals[key] / counts[key] for key in totals},
        **artifacts,
    }


def run_repeated_probe(
    *,
    repeat_count: int,
    metric_prefix: str,
    run_once: Callable[
        [int, Callable[[dict[str, float]], None] | None],
        dict[str, Any],
    ],
    on_epoch_end: Callable[[dict[str, float]], None] | None = None,
    on_repeat_start: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    if repeat_count == 1:
        metrics = dict(run_once(0, on_epoch_end))
        if metrics:
            metrics[f"{metric_prefix}/repeats"] = 1.0
        return metrics

    repeat_metrics: list[dict[str, Any]] = []
    repeat_curves: list[list[dict[str, float]]] = []
    for repeat_idx in range(repeat_count):
        if on_repeat_start is not None:
            on_repeat_start(repeat_idx, repeat_count)
        repeat_curve: list[dict[str, float]] = []
        metrics = run_once(repeat_idx, repeat_curve.append)
        if metrics:
            repeat_metrics.append(metrics)
        if repeat_curve:
            repeat_curves.append(repeat_curve)

    averaged_metrics = average_metric_dicts(repeat_metrics)
    if averaged_metrics:
        averaged_metrics[f"{metric_prefix}/repeats"] = float(repeat_count)
    if on_epoch_end is not None and repeat_curves:
        num_epochs = max(len(curve) for curve in repeat_curves)
        for epoch_idx in range(num_epochs):
            epoch_metrics = average_metric_dicts(
                [curve[epoch_idx] for curve in repeat_curves if epoch_idx < len(curve)]
            )
            if epoch_metrics:
                on_epoch_end(epoch_metrics)
    return averaged_metrics
