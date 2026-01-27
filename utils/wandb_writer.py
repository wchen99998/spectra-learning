"""Minimal WandB helpers used by the Lightning training entrypoint."""

from __future__ import annotations

import datetime
import os
from typing import Any, Mapping

import numpy as np


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def build_wandb_init_kwargs(config: Any | None) -> dict[str, Any]:
    if config is None:
        return {}

    wandb_kwargs = dict(_config_get(config, "wandb_kwargs", {}) or {})

    resume_id = os.environ.get("WANDB_RESUME_ID")
    if resume_id:
        wandb_kwargs.setdefault("id", resume_id)
        wandb_kwargs.setdefault("resume", "must")
        wandb_kwargs.pop("name", None)
        return wandb_kwargs

    prefix = _config_get(config, "wandb_run_name_prefix")
    if prefix and "name" not in wandb_kwargs:
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M")
        wandb_kwargs["name"] = f"{str(prefix).strip()}-{timestamp}"

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


def config_to_wandb_dict(config: Any | None) -> dict[str, Any]:
    if config is None:
        return {}
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        return dict(_to_serialisable_config(to_dict()))
    if isinstance(config, Mapping):
        return dict(_to_serialisable_config(config))
    return dict(_to_serialisable_config(vars(config)))
