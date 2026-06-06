import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from ml_collections import config_dict

from spectra_learning.training.schedules import LRSchedulerLike, make_cosine_schedule
from spectra_learning.training.storage import (
    StoragePath,
    read_text,
    storage_exists,
    storage_join,
)


def deepspeed_enabled(config: config_dict.ConfigDict) -> bool:
    return bool(
        _config_get(config, "use_deepspeed", _config_get(config, "deepspeed", False))
    )


def gradient_accumulation_steps(config: config_dict.ConfigDict) -> int:
    return int(_config_get(config, "gradient_accumulation_steps", 1))


def initialize_deepspeed(
    *,
    config: config_dict.ConfigDict,
    model: torch.nn.Module,
    total_steps: int,
    train_micro_batch_size_per_gpu: int,
    autocast_dtype: torch.dtype | None,
    grad_clip_norm: float | None,
) -> tuple[torch.nn.Module, list[torch.optim.Optimizer], list[LRSchedulerLike]]:
    import deepspeed

    ds_config = build_deepspeed_config(
        config,
        train_micro_batch_size_per_gpu=train_micro_batch_size_per_gpu,
        autocast_dtype=autocast_dtype,
        grad_clip_norm=grad_clip_norm,
    )
    scheduler_factory = _scheduler_factory(config, total_steps)
    engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
        lr_scheduler=scheduler_factory,
    )
    return engine, [optimizer], [scheduler]


def build_deepspeed_config(
    config: config_dict.ConfigDict,
    *,
    train_micro_batch_size_per_gpu: int,
    autocast_dtype: torch.dtype | None,
    grad_clip_norm: float | None,
) -> dict[str, Any]:
    ds_config: dict[str, Any] = {
        "train_batch_size": int(config.batch_size),
        "train_micro_batch_size_per_gpu": int(train_micro_batch_size_per_gpu),
        "gradient_accumulation_steps": gradient_accumulation_steps(config),
        "zero_optimization": {
            "stage": int(_config_get(config, "deepspeed_zero_stage", 2)),
        },
        "optimizer": _deepspeed_optimizer_config(config),
    }
    if grad_clip_norm is not None and grad_clip_norm > 0:
        ds_config["gradient_clipping"] = float(grad_clip_norm)
    if autocast_dtype == torch.bfloat16:
        ds_config["bf16"] = {"enabled": True}
    elif autocast_dtype == torch.float16:
        ds_config["fp16"] = {"enabled": True, "loss_scale": 0}
    return _deep_update(ds_config, _deepspeed_config_overrides(config))


def _deepspeed_optimizer_config(config: config_dict.ConfigDict) -> dict[str, Any]:
    optimizer_type = str(_config_get(config, "optimizer", "adamw")).lower()
    if optimizer_type == "muon":
        weight_decay = float(
            _config_get(config, "muon_weight_decay", None)
            or _config_get(config, "weight_decay", 0.0)
        )
        return {
            "type": "Muon",
            "params": {
                "lr": float(config.learning_rate),
                "momentum": float(_config_get(config, "muon_momentum", 0.95)),
                "weight_decay": weight_decay,
                "muon_lr": float(_config_get(config, "muon_lr", config.learning_rate)),
                "adam_lr": float(_config_get(config, "adamw_lr", config.learning_rate)),
                "ns_method": str(_config_get(config, "deepspeed_muon_ns_method", "gram")),
            },
        }
    return {
        "type": "Adam",
        "params": {
            "lr": float(config.learning_rate),
            "betas": [0.9, float(_config_get(config, "b2", 0.999))],
            "eps": float(_config_get(config, "adam_eps", 1e-8)),
            "weight_decay": float(_config_get(config, "weight_decay", 0.0)),
            "adam_w_mode": True,
            "torch_adam": bool(_config_get(config, "deepspeed_torch_adam", False)),
        },
    }


def _scheduler_factory(
    config: config_dict.ConfigDict,
    total_steps: int,
) -> Callable[[torch.optim.Optimizer], LRSchedulerLike]:
    def build(optimizer: torch.optim.Optimizer) -> LRSchedulerLike:
        return make_cosine_schedule(
            optimizer,
            total_steps,
            int(_config_get(config, "warmup_steps", 0)),
            _config_get(config, "min_learning_rate", None),
        )

    return build


def is_deepspeed_engine(model: torch.nn.Module) -> bool:
    return (
        model.__class__.__name__ == "DeepSpeedEngine"
        or (
            hasattr(model, "is_gradient_accumulation_boundary")
            and hasattr(model, "backward")
            and hasattr(model, "step")
            and hasattr(model, "module")
        )
    )


def deepspeed_checkpoint_tag(global_step: int, name: str | None = None) -> str:
    return name or f"step-{global_step:08d}"


def save_deepspeed_checkpoint(
    *,
    checkpoint_dir: StoragePath,
    engine: torch.nn.Module,
    global_step: int,
    epoch: int,
    loss: float,
    wandb_run_id: str | None,
    name: str | None = None,
) -> None:
    tag = deepspeed_checkpoint_tag(global_step, name)
    engine.save_checkpoint(
        str(checkpoint_dir),
        tag=tag,
        client_state={
            "global_step": global_step,
            "epoch": epoch,
            "loss": loss,
            "wandb_run_id": wandb_run_id,
            "training_mode": "pretrain",
        },
        save_latest=True,
    )


def restore_deepspeed_training_state(
    *,
    config: config_dict.ConfigDict,
    checkpoint_dir: StoragePath,
    engine: torch.nn.Module,
    steps_per_epoch: int,
) -> tuple[int, int, int]:
    tag = latest_deepspeed_checkpoint_tag(checkpoint_dir)
    if tag is None:
        return 0, 0, 0
    logging.info("Resuming DeepSpeed training from checkpoint tag: %s", tag)
    _, client_state = engine.load_checkpoint(str(checkpoint_dir), tag=tag)
    resume_wandb_id = client_state.get("wandb_run_id")
    if resume_wandb_id:
        config.wandb_resume_id = resume_wandb_id
    global_step = int(client_state["global_step"])
    start_epoch = int(client_state["epoch"])
    resume_offset = global_step - start_epoch * steps_per_epoch
    start_epoch += resume_offset // steps_per_epoch
    resume_offset %= steps_per_epoch
    return start_epoch, global_step, resume_offset


def latest_deepspeed_checkpoint_tag(checkpoint_dir: StoragePath) -> str | None:
    latest_path = storage_join(checkpoint_dir, "latest")
    if not storage_exists(latest_path):
        return None
    return read_text(latest_path).strip()


def _deepspeed_config_overrides(config: config_dict.ConfigDict) -> dict[str, Any]:
    path = str(_config_get(config, "deepspeed_config_path", "")).strip()
    overrides = json.loads(Path(path).read_text()) if path else {}
    inline = _config_get(config, "deepspeed_config", None)
    if inline is not None:
        overrides = _deep_update(overrides, _plain_dict(inline))
    return overrides


def _plain_dict(value: Any) -> Any:
    if isinstance(value, config_dict.ConfigDict):
        return {key: _plain_dict(item) for key, item in value.items()}
    if isinstance(value, dict):
        return {key: _plain_dict(item) for key, item in value.items()}
    return value


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _config_get(config: config_dict.ConfigDict, key: str, default: Any) -> Any:
    return config.get(key, default)
