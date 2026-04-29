import csv
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ml_collections import config_dict

from spectra_learning.training.naming import auto_run_name


def _build_wandb_init_kwargs(config: Any | None) -> dict[str, Any]:
    if config is None:
        return {}
    wandb_kwargs = dict(config.get("wandb_kwargs", {}) or {})
    resume_id = str(config.get("wandb_resume_id", "") or "")
    if not resume_id:
        resume_id = os.environ.get("WANDB_RESUME_ID", "")
    if resume_id:
        wandb_kwargs.setdefault("id", resume_id)
        wandb_kwargs.setdefault("resume", "must")
        wandb_kwargs.pop("name", None)
        return wandb_kwargs
    if "name" not in wandb_kwargs:
        wandb_kwargs["name"] = auto_run_name(config)
    return wandb_kwargs


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
        self._run = wandb.init(
            project=config.get("wandb_project", "md4"),
            dir=str(workdir),
            config=_config_to_wandb_dict(config),
            **wandb_kwargs,
        )

    @property
    def experiment(self) -> Any:
        return self._run

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        self._run.config.update(params, allow_val_change=True)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self._run.log(_serialise_metrics(metrics), step=step)


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


def _serialise_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    serialised = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
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
