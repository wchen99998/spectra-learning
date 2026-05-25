from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

import torch

_WARMUP_START_FACTOR = 1e-8


class LRSchedulerLike(Protocol):
    def step(self, epoch: int | None = None) -> None: ...

    def get_last_lr(self) -> Sequence[float | torch.Tensor]: ...

    def state_dict(self) -> dict: ...

    def load_state_dict(self, state_dict: dict) -> None: ...


def learning_rate_at_step(
    step: int,
    *,
    base_lr: float,
    total_steps: int,
    warmup_steps: int,
    min_learning_rate: float | None = None,
) -> float:
    min_lr = min_learning_rate if min_learning_rate is not None else 0.1 * base_lr
    if warmup_steps > 0 and step < warmup_steps:
        warmup = _WARMUP_START_FACTOR + (
            1.0 - _WARMUP_START_FACTOR
        ) * step / warmup_steps
        return base_lr * warmup
    ratio = max(
        0.0,
        min(1.0, max(0, step - warmup_steps) / max(1, total_steps - warmup_steps)),
    )
    mult = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + (base_lr - min_lr) * mult


def _lr_to_float(lr) -> float:
    return float(lr.detach().cpu()) if torch.is_tensor(lr) else float(lr)


class WarmupCosineSchedule:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_steps: int,
        min_lr: float | None,
    ) -> None:
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.base_lrs = [_lr_to_float(group["lr"]) for group in optimizer.param_groups]
        base_lr = self.base_lrs[0]
        self.eta_min = min_lr if min_lr is not None else 0.1 * base_lr
        self.last_epoch = 0
        self._last_lr: list[float] = []
        self._set_lrs(self._compute_lrs(self.last_epoch))

    def _compute_lrs(self, step: int) -> list[float]:
        return [
            learning_rate_at_step(
                step,
                base_lr=base_lr,
                total_steps=self.total_steps,
                warmup_steps=self.warmup_steps,
                min_learning_rate=self.eta_min,
            )
            for base_lr in self.base_lrs
        ]

    def _set_lrs(self, lrs: list[float]) -> None:
        self._last_lr = lrs
        for group, lr in zip(self.optimizer.param_groups, lrs, strict=True):
            current_lr = group["lr"]
            if torch.is_tensor(current_lr):
                current_lr.fill_(lr)
            else:
                group["lr"] = lr

    def step(self, epoch: int | None = None) -> None:
        self.last_epoch = self.last_epoch + 1 if epoch is None else epoch
        self._set_lrs(self._compute_lrs(self.last_epoch))

    def get_last_lr(self) -> list[float]:
        return list(self._last_lr)

    def state_dict(self) -> dict:
        return {
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "base_lrs": self.base_lrs,
            "eta_min": self.eta_min,
            "last_epoch": self.last_epoch,
            "_last_lr": self._last_lr,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.total_steps = int(state_dict["total_steps"])
        self.warmup_steps = int(state_dict["warmup_steps"])
        self.base_lrs = [float(lr) for lr in state_dict["base_lrs"]]
        self.eta_min = float(state_dict["eta_min"])
        self.last_epoch = int(state_dict["last_epoch"])
        self._set_lrs(self._compute_lrs(self.last_epoch))


def make_cosine_schedule(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr: float | None,
) -> WarmupCosineSchedule:
    return WarmupCosineSchedule(optimizer, total_steps, warmup_steps, min_lr)
