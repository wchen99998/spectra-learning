import json
from pathlib import Path

import torch
from ml_collections import config_dict

from spectra_learning.models.pooling import build_covariance_pooler_from_config
from spectra_learning.probes.massspec.msg_probe import run_msg_probe
from spectra_learning.training.api import build_logger, build_model_from_config
from spectra_learning.training.checkpointing import (
    load_resume_covariance_pooler_state,
    load_resume_model_state,
)
from spectra_learning.training.logging import log_msg_probe_metrics


def run_checkpoint_msg_probe(
    *,
    config_json: str,
    checkpoint_path: str | Path,
    workdir: str | Path,
    global_step: int,
) -> dict[str, float]:
    config = config_dict.ConfigDict(json.loads(config_json))
    checkpoint_path = Path(checkpoint_path)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    wandb_run_id = str(checkpoint.get("wandb_run_id", "") or "")
    if wandb_run_id:
        config.wandb_resume_id = wandb_run_id
        config.wandb_shared_mode = True
        config.wandb_shared_primary = False
        config.wandb_shared_label = f"probe_step_{global_step}"
        config.wandb_shared_update_finish_state = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(config)
    load_resume_model_state(model, checkpoint["model"])
    model.to(device).eval()

    covariance_pooler = build_covariance_pooler_from_config(config)
    load_resume_covariance_pooler_state(
        covariance_pooler,
        checkpoint_path,
        checkpoint,
    )
    if covariance_pooler is not None:
        covariance_pooler.to(device).eval()

    logger = build_logger(config, workdir)
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
    metrics_path = workdir / f"msg_probe_step-{global_step:08d}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics
