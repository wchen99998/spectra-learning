from typing import Any

import torch
from ml_collections import config_dict

from spectra_learning.training.schedules import (
    LRSchedulerLike,
    make_cosine_schedule,
)


def is_weight_decay_target(name: str, param: Any) -> bool:
    return param.ndim >= 2 and name.endswith("weight")


def build_adamw_param_groups(
    decay_params: list[torch.nn.Parameter],
    no_decay_params: list[torch.nn.Parameter],
    weight_decay: float,
) -> list[dict]:
    param_groups = []
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})
    if decay_params:
        param_groups.append({"params": decay_params, "weight_decay": weight_decay})
    return param_groups


def build_optimizers(
    config: config_dict.ConfigDict,
    model: torch.nn.Module,
    total_steps: int,
    device: torch.device,
) -> tuple[list[torch.optim.Optimizer], list[LRSchedulerLike]]:
    settings = _optimizer_settings(config, device)
    return _build_single_adamw_optimizer(config, model, total_steps, settings)


def _optimizer_settings(
    config: config_dict.ConfigDict,
    device: torch.device,
) -> dict:
    is_cuda = device.type == "cuda"
    fused_cfg = _config_get(config, "optimizer_fused", None)
    return {
        "base_lr": float(config.learning_rate),
        "warmup_steps": int(_config_get(config, "warmup_steps", 0)),
        "min_learning_rate": _config_get(config, "min_learning_rate", None),
        "b2": float(_config_get(config, "b2", 0.999)),
        "weight_decay": float(config.weight_decay),
        "is_cuda": is_cuda,
        "fused": is_cuda if fused_cfg is None else bool(fused_cfg) and is_cuda,
    }


def _adamw(
    param_groups: list[dict],
    *,
    lr: float,
    b2: float,
    fused: bool,
) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        param_groups,
        lr=lr,
        betas=(0.9, b2),
        fused=fused,
    )


def _build_single_adamw_optimizer(
    config: config_dict.ConfigDict,
    model: torch.nn.Module,
    total_steps: int,
    settings: dict,
) -> tuple[list[torch.optim.Optimizer], list[LRSchedulerLike]]:
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and is_weight_decay_target(name, param):
            decay_params.append(param)
        elif param.requires_grad:
            no_decay_params.append(param)
    param_groups = build_adamw_param_groups(
        decay_params,
        no_decay_params,
        settings["weight_decay"],
    )
    optimizer = _adamw(
        param_groups,
        lr=settings["base_lr"],
        b2=settings["b2"],
        fused=settings["fused"],
    )
    scheduler = make_cosine_schedule(
        optimizer,
        total_steps,
        settings["warmup_steps"],
        settings["min_learning_rate"],
    )
    return [optimizer], [scheduler]


def _config_get(config: config_dict.ConfigDict, key: str, default: Any) -> Any:
    return config.get(key, default)
