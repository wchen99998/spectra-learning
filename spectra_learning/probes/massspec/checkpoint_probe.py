import json
from pathlib import Path

import torch
from ml_collections import config_dict

from spectra_learning.models.pooling import SinglePairCovariancePool
from spectra_learning.probes.massspec.msg_probe import run_msg_probe
from spectra_learning.training.api import build_logger, build_model_from_config
from spectra_learning.training.checkpointing import (
    load_resume_covariance_pooler_state,
    load_torch_checkpoint,
    load_resume_model_state,
)
from spectra_learning.training.logging import log_msg_probe_metrics
from spectra_learning.training.storage import (
    local_scratch_dir,
    normalize_storage_path,
    storage_join,
    storage_mkdir,
    write_text,
)


def run_checkpoint_msg_probe(
    *,
    config_json: str,
    checkpoint_path: str | Path,
    workdir: str | Path,
    global_step: int,
) -> dict[str, float]:
    config = config_dict.ConfigDict(json.loads(config_json))
    checkpoint_path = normalize_storage_path(checkpoint_path)
    workdir = normalize_storage_path(workdir)
    local_workdir = local_scratch_dir(workdir)
    local_workdir.mkdir(parents=True, exist_ok=True)
    storage_mkdir(workdir)

    checkpoint = load_torch_checkpoint(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if checkpoint.get("training_mode", None):
        config.training_mode = checkpoint["training_mode"]
    wandb_run_id = str(checkpoint["wandb_run_id"] or "")
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
    covariance_pooler = _checkpoint_covariance_pooler(
        config,
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        device=device,
    )

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
    write_text(metrics_path, json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


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
