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
    is_cuda = device.type == "cuda"
    fused_cfg = config.get("optimizer_fused", None)
    parameters = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.Adam(
        parameters,
        lr=float(config.learning_rate),
        betas=(0.9, float(config.get("b2", 0.999))),
        fused=is_cuda if fused_cfg is None else bool(fused_cfg) and is_cuda,
    )
    scheduler = make_cosine_schedule(
        optimizer,
        total_steps,
        int(config.get("warmup_steps", 0)),
        config.get("min_learning_rate", None),
    )
    return [optimizer], [scheduler]
