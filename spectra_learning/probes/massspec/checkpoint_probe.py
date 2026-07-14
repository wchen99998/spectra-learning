import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ml_collections import config_dict

from spectra_learning.models.factory import build_model_from_config
from spectra_learning.models.pooling import SinglePairCovariancePool
from spectra_learning.probes.massspec.msg_probe import run_msg_probe
from spectra_learning.probes.massspec.pr_curves import PrecisionRecallCurve
from spectra_learning.training.checkpointing import (
    load_resume_covariance_pooler_state,
    load_resume_model_state,
    load_torch_checkpoint,
)
from spectra_learning.training.logging import build_logger, log_msg_probe_metrics
from spectra_learning.training.storage import (
    local_scratch_dir,
    normalize_storage_path,
    storage_join,
    storage_mkdir,
    write_text,
)


DEFAULT_STANDALONE_WANDB_PROJECT = "msg-probe-evaluations"


def run_checkpoint_msg_probe(
    *,
    config_json: str,
    checkpoint_path: str | Path,
    workdir: str | Path,
    global_step: int,
    wandb_project: str | None = DEFAULT_STANDALONE_WANDB_PROJECT,
) -> dict[str, Any]:
    config = config_dict.ConfigDict(json.loads(config_json))
    checkpoint_path = normalize_storage_path(checkpoint_path)
    checkpoint = load_torch_checkpoint(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if checkpoint.get("training_mode", None):
        config.training_mode = checkpoint["training_mode"]
    config.source_wandb_run_id = str(checkpoint["wandb_run_id"] or "")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(config)
    load_resume_model_state(model, checkpoint["model"])
    covariance_pooler = _checkpoint_covariance_pooler(
        config,
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        device=device,
    )
    return run_msg_probe_evaluation(
        config=config,
        model=model,
        workdir=workdir,
        global_step=global_step,
        checkpoint_path=checkpoint_path,
        covariance_pooler=covariance_pooler,
        device=device,
        wandb_project=wandb_project,
    )


def run_msg_probe_evaluation(
    *,
    config: config_dict.ConfigDict,
    model: torch.nn.Module,
    workdir: str | Path,
    global_step: int,
    checkpoint_path: str | Path | None = None,
    covariance_pooler: torch.nn.Module | None = None,
    device: torch.device | None = None,
    wandb_project: str | None = DEFAULT_STANDALONE_WANDB_PROJECT,
) -> dict[str, Any]:
    workdir = normalize_storage_path(workdir)
    local_workdir = local_scratch_dir(workdir)
    local_workdir.mkdir(parents=True, exist_ok=True)
    storage_mkdir(workdir)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _configure_standalone_wandb(
        config,
        checkpoint_path=checkpoint_path,
        global_step=global_step,
        wandb_project=wandb_project,
    )
    model.to(device).eval()
    if covariance_pooler is not None:
        covariance_pooler.to(device).eval()
    logger = build_logger(config, local_workdir)
    metrics = run_msg_probe(
        config=config,
        model=model,
        device=device,
        covariance_pooler=covariance_pooler,
    )
    log_msg_probe_metrics(
        logger,
        metrics,
        global_step,
        enable_wandb=bool(config.get("enable_wandb", False)),
    )
    metrics_path = storage_join(workdir, f"msg_probe_step-{global_step:08d}.json")
    write_text(
        metrics_path,
        json.dumps(_json_metrics(metrics), indent=2, sort_keys=True),
    )
    return metrics


def _configure_standalone_wandb(
    config: config_dict.ConfigDict,
    *,
    checkpoint_path: str | Path | None,
    global_step: int,
    wandb_project: str | None,
) -> None:
    config.msg_probe_checkpoint_path = (
        "" if checkpoint_path is None else str(checkpoint_path)
    )
    config.msg_probe_global_step = int(global_step)
    config.wandb_resume_id = ""
    config.wandb_resume_from_env = False
    config.wandb_shared_mode = False
    if wandb_project is None:
        config.enable_wandb = False
        return
    config.enable_wandb = True
    config.wandb_project = wandb_project
    wandb_kwargs = dict(config.get("wandb_kwargs", {}) or {})
    wandb_kwargs.setdefault(
        "name",
        _standalone_wandb_name(
            checkpoint_path=checkpoint_path,
            global_step=global_step,
        ),
    )
    config.wandb_kwargs = wandb_kwargs


def _standalone_wandb_name(
    *,
    checkpoint_path: str | Path | None,
    global_step: int,
) -> str:
    if checkpoint_path is None:
        checkpoint_name = "supplied-model"
    else:
        checkpoint_name = Path(str(checkpoint_path).rstrip("/")).name.removesuffix(".pt")
    return f"msg_probe_step-{global_step:08d}_{checkpoint_name}"


def _json_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, PrecisionRecallCurve):
            continue
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            payload[key] = value.item() if value.ndim == 0 else value.tolist()
        elif isinstance(value, np.generic):
            payload[key] = value.item()
        elif isinstance(value, np.ndarray):
            payload[key] = value.item() if value.ndim == 0 else value.tolist()
        else:
            payload[key] = value
    return payload


def _checkpoint_covariance_pooler(
    config: config_dict.ConfigDict,
    *,
    checkpoint_path: str | Path,
    checkpoint: dict,
    device: torch.device,
) -> torch.nn.Module | None:
    pooler_name = checkpoint.get("covariance_pooler_checkpoint", None)
    if not pooler_name:
        return None
    compressed_dim = int(
        config.get("contrastive_covariance_dim", config.get("covariance_pooling_dim", 32))
    )
    pooler = SinglePairCovariancePool(
        single_dim=int(config.model_dim),
        pair_dim=int(config.get("pairmixer_pair_dim", config.model_dim)),
        compressed_dim=compressed_dim,
        include_diagonal=bool(
            config.get("contrastive_single_pair_include_diagonal", False)
        ),
    )
    load_resume_covariance_pooler_state(pooler, checkpoint_path, checkpoint)
    return pooler.to(device)
