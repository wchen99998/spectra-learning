import logging

import torch
from ml_collections import config_dict


def parse_autocast_dtype(value: object) -> torch.dtype | None:
    name = str(value).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp32", "float32", "none"}:
        return None
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    raise ValueError(f"Unsupported autocast_dtype: {name}")


def build_grad_scaler(
    autocast_dtype: torch.dtype | None,
    device: torch.device,
) -> torch.amp.GradScaler:
    return torch.amp.GradScaler(
        device=device.type,
        enabled=device.type == "cuda" and autocast_dtype == torch.float16,
    )


def collect_and_log_param_metrics(model: torch.nn.Module) -> dict[str, float]:
    by_module: dict[str, list[int]] = {}
    total = trainable = 0
    for name, param in model.named_parameters():
        numel = param.numel()
        module_name = name.split(".", 1)[0]
        counts = by_module.setdefault(module_name, [0, 0])
        total += numel
        counts[0] += numel
        if param.requires_grad:
            trainable += numel
            counts[1] += numel
    logging.info(
        "Model parameters: total=%s trainable=%s non_trainable=%s",
        f"{total:,}",
        f"{trainable:,}",
        f"{total - trainable:,}",
    )
    metrics: dict[str, float] = {
        "model/params_total": float(total),
        "model/params_trainable": float(trainable),
        "model/params_non_trainable": float(total - trainable),
    }
    for module_name in sorted(by_module):
        mod_total, mod_train = by_module[module_name]
        logging.info(
            "  [%s] total=%s trainable=%s",
            module_name,
            f"{mod_total:,}",
            f"{mod_train:,}",
        )
        metrics[f"model/params_total/{module_name}"] = float(mod_total)
        metrics[f"model/params_trainable/{module_name}"] = float(mod_train)
    return metrics


def trainable_parameter_count(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def estimate_training_flops_per_sample(
    config: config_dict.ConfigDict,
    model: torch.nn.Module,
) -> float:
    configured = config.get("training_flops_per_sample", None)
    if configured is not None:
        return float(configured)
    multiplier = float(config.get("training_flops_per_parameter", 6.0))
    return multiplier * float(trainable_parameter_count(model))


def estimate_training_flops_per_optimizer_step(
    config: config_dict.ConfigDict,
    model: torch.nn.Module,
    global_batch_size: int,
) -> float:
    configured = config.get("training_flops_per_optimizer_step", None)
    if configured is not None:
        return float(configured)
    return estimate_training_flops_per_sample(config, model) * float(global_batch_size)


def cumulative_training_flops(global_step: int, flops_per_optimizer_step: float) -> float:
    return float(global_step) * float(flops_per_optimizer_step)
