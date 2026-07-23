import torch
from ml_collections import config_dict

from spectra_learning.training.schedules import (
    LRSchedulerLike,
    make_cosine_schedule,
)


def build_optimizers(
    config: config_dict.ConfigDict,
    model: torch.nn.Module,
    total_steps: int,
    device: torch.device,
) -> tuple[list[torch.optim.Optimizer], list[LRSchedulerLike]]:
    optimizer_name = str(config.get("optimizer", "adam")).lower()
    if optimizer_name != "adam":
        raise ValueError("PyTorch training requires optimizer='adam'")
    if float(config.get("weight_decay", 0.0)) != 0.0:
        raise ValueError("Plain Adam requires weight_decay=0")
    settings = _optimizer_settings(config, device)
    return _build_single_adam_optimizer(model, total_steps, settings)


def _optimizer_settings(
    config: config_dict.ConfigDict,
    device: torch.device,
) -> dict:
    is_cuda = device.type == "cuda"
    fused_cfg = config.get("optimizer_fused", None)
    return {
        "base_lr": float(config.learning_rate),
        "warmup_steps": int(config.get("warmup_steps", 0)),
        "min_learning_rate": config.get("min_learning_rate", None),
        "b2": float(config.get("b2", 0.999)),
        "fused": is_cuda if fused_cfg is None else bool(fused_cfg) and is_cuda,
    }


def _adam(
    parameters: list[torch.nn.Parameter],
    *,
    lr: float,
    b2: float,
    fused: bool,
) -> torch.optim.Adam:
    return torch.optim.Adam(
        parameters,
        lr=lr,
        betas=(0.9, b2),
        fused=fused,
    )


def _build_single_adam_optimizer(
    model: torch.nn.Module,
    total_steps: int,
    settings: dict,
) -> tuple[list[torch.optim.Optimizer], list[LRSchedulerLike]]:
    parameters = [param for param in model.parameters() if param.requires_grad]
    optimizer = _adam(
        parameters,
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
