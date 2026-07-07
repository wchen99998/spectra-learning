import csv
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ml_collections import config_dict

from spectra_learning.probes.massspec.pr_curves import (
    PrecisionRecallCurve,
    precision_recall_points,
)
from spectra_learning.training.naming import auto_run_name


def _build_wandb_init_kwargs(config: Any | None) -> dict[str, Any]:
    if config is None:
        return {}
    wandb_kwargs = dict(config.get("wandb_kwargs", {}) or {})
    resume_id = str(config.get("wandb_resume_id", "") or "")
    if not resume_id and bool(config.get("wandb_resume_from_env", True)):
        resume_id = os.environ.get("WANDB_RESUME_ID", "")
    if resume_id:
        wandb_kwargs.setdefault("id", resume_id)
        wandb_kwargs.setdefault("resume", "must")
        wandb_kwargs.pop("name", None)
        return wandb_kwargs
    if "name" not in wandb_kwargs:
        wandb_kwargs["name"] = auto_run_name(config)
    return wandb_kwargs


def _use_wandb_shared_mode(config: Any | None) -> bool:
    if config is None:
        return False
    return bool(config.get("wandb_shared_mode", False))


def _to_serialisable_config(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, Mapping):
        return {str(k): _to_serialisable_config(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_serialisable_config(v) for v in value]
    return str(value)


def _config_to_wandb_dict(config: Any | None) -> dict[str, Any]:
    if config is None:
        return {}
    if callable(getattr(config, "to_dict", None)):
        return dict(_to_serialisable_config(config.to_dict()))
    if isinstance(config, Mapping):
        return dict(_to_serialisable_config(config))
    return dict(_to_serialisable_config(vars(config)))


class MetricLogger:
    def log_hyperparams(self, params: dict[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        pass

    @property
    def experiment(self) -> Any:
        return None


class WandbMetricLogger(MetricLogger):
    def __init__(self, config: config_dict.ConfigDict, workdir: Path) -> None:
        import wandb

        wandb_kwargs = _build_wandb_init_kwargs(config)
        if _use_wandb_shared_mode(config):
            primary = bool(config.get("wandb_shared_primary", True))
            settings_kwargs: dict[str, Any] = {
                "mode": "shared",
                "x_primary": primary,
                "x_label": str(
                    config.get(
                        "wandb_shared_label",
                        "train" if primary else "worker",
                    )
                ),
            }
            if not primary:
                settings_kwargs["x_update_finish_state"] = bool(
                    config.get("wandb_shared_update_finish_state", False)
                )
            wandb_kwargs["settings"] = wandb.Settings(**settings_kwargs)
        self._run = wandb.init(
            project=config.get("wandb_project", "md4"),
            dir=str(workdir),
            config=_config_to_wandb_dict(config),
            **wandb_kwargs,
        )
        self._run.define_metric("global_step")
        self._run.define_metric("train/*", step_metric="global_step")
        self._run.define_metric("val/*", step_metric="global_step")
        self._run.define_metric("msg_probe/*", step_metric="global_step")
        self._run.define_metric("run/*", step_metric="global_step")
        self._run.define_metric("model/*", step_metric="global_step")

    @property
    def experiment(self) -> Any:
        return self._run

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        self._run.config.update(params, allow_val_change=True)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self._run.log(
            _serialise_metrics(metrics, enable_wandb_artifacts=True),
            step=step,
        )


class CSVMetricLogger(MetricLogger):
    def __init__(self, workdir: Path) -> None:
        self.path = workdir / "csv_logs" / "metrics.csv"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames: list[str] = ["step"]
        self._rows: list[dict[str, Any]] = []

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        row = {"step": step, **_serialise_metrics(metrics)}
        for key in row:
            if key not in self._fieldnames:
                self._fieldnames.append(key)
        self._rows.append(row)
        self._write_rows()

    def _write_rows(self) -> None:
        with self.path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writeheader()
            writer.writerows(self._rows)


def _wandb_precision_recall_image(curve: PrecisionRecallCurve) -> Any:
    import matplotlib.pyplot as plt
    import wandb

    recall, precision = precision_recall_points(curve)
    fig, ax = plt.subplots(figsize=(5.0, 4.0), dpi=140)
    ax.plot(recall, precision, linewidth=2.0)
    ax.set_title(curve.title)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    image = wandb.Image(fig)
    plt.close(fig)
    return image


def _wandb_precision_recall_native(curve: PrecisionRecallCurve) -> Any:
    import wandb

    return wandb.plot.pr_curve(
        y_true=curve.targets.tolist(),
        y_probas=curve.two_class_probabilities.tolist(),
        labels=curve.class_labels,
        classes_to_plot=[1],
        title=curve.title,
    )


def _serialise_metrics(
    metrics: dict[str, Any],
    *,
    enable_wandb_artifacts: bool = False,
) -> dict[str, Any]:
    serialised = {}
    for key, value in metrics.items():
        if isinstance(value, PrecisionRecallCurve):
            if enable_wandb_artifacts:
                serialised[f"{key}/image"] = _wandb_precision_recall_image(value)
                serialised[f"{key}/native"] = _wandb_precision_recall_native(value)
            continue
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            serialised[key] = value.item() if value.ndim == 0 else value.tolist()
        elif isinstance(value, np.generic):
            serialised[key] = value.item()
        elif isinstance(value, np.ndarray):
            serialised[key] = value.item() if value.ndim == 0 else value.tolist()
        else:
            serialised[key] = value
    return serialised


def build_logger(config: config_dict.ConfigDict, workdir: Path) -> MetricLogger:
    if config.get("enable_wandb", False):
        logger = WandbMetricLogger(config, workdir)
        logger.log_hyperparams(_config_to_wandb_dict(config))
        return logger
    return CSVMetricLogger(workdir)


def log_msg_probe_metrics(
    logger: Any,
    metrics: dict[str, Any],
    global_step: int,
    *,
    enable_wandb: bool,
) -> None:
    if enable_wandb:
        logger.log_metrics({"global_step": float(global_step), **metrics})
    else:
        logger.log_metrics(metrics, step=global_step)
