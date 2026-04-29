from __future__ import annotations

import math

import torch


def learning_rate_at_step(
    step: int,
    *,
    base_lr: float,
    total_steps: int,
    warmup_steps: int,
    min_learning_rate: float | None = None,
) -> float:
    warmup = min(1.0, step / warmup_steps) if warmup_steps > 0 else 1.0
    ratio = max(0.0, max(0, step - warmup_steps) / max(1, total_steps - warmup_steps))
    mult = 0.5 * (1.0 + math.cos(math.pi * ratio))
    min_lr = min_learning_rate if min_learning_rate is not None else 0.1 * base_lr
    return max(min_lr, mult * base_lr) * warmup


def make_cosine_schedule(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr: float | None,
) -> torch.optim.lr_scheduler.LRScheduler:
    base_lr = float(optimizer.param_groups[0]["lr"])
    eta_min = float(min_lr) if min_lr is not None else 0.1 * base_lr
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps - warmup_steps),
        eta_min=eta_min,
    )
    if warmup_steps > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )
    return cosine


def scaled_min_lr(min_lr: float | None, ratio: float) -> float | None:
    return None if min_lr is None else float(min_lr) * ratio
