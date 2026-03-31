from __future__ import annotations

import math


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


class CapturableCosineSchedule:
    def __init__(
        self,
        optimizer: "torch.optim.Optimizer",
        *,
        base_lr: float,
        total_steps: int,
        warmup_steps: int,
        min_lr: float | None = None,
        device: str | "torch.device" = "cuda",
    ) -> None:
        import torch

        self.optimizer = optimizer
        self._step = torch.zeros((), dtype=torch.int64, device=device)
        self._base_lr = torch.tensor(base_lr, dtype=torch.float64, device=device)
        self._warmup_steps = torch.tensor(
            max(warmup_steps, 1),
            dtype=torch.float64,
            device=device,
        )
        self._has_warmup = warmup_steps > 0
        self._effective_total = torch.tensor(
            max(1, total_steps - warmup_steps),
            dtype=torch.float64,
            device=device,
        )
        self._min_lr = torch.tensor(
            min_lr if min_lr is not None else 0.1 * base_lr,
            dtype=torch.float64,
            device=device,
        )
        self._pi = torch.tensor(math.pi, dtype=torch.float64, device=device)

    def step(self) -> None:
        import torch

        self._step.add_(1)
        step_f = self._step.to(torch.float64)
        if self._has_warmup:
            warmup = torch.clamp(step_f / self._warmup_steps, max=1.0)
            effective_step = torch.clamp(
                step_f - self._warmup_steps,
                min=0.0,
            )
        else:
            warmup = torch.ones((), dtype=torch.float64, device=self._step.device)
            effective_step = step_f
        ratio = torch.clamp(effective_step / self._effective_total, min=0.0)
        mult = 0.5 * (1.0 + torch.cos(self._pi * ratio))
        lr = torch.max(self._min_lr, mult * self._base_lr) * warmup
        lr_val = lr.item()
        for group in self.optimizer.param_groups:
            if hasattr(group["lr"], "copy_"):
                group["lr"].copy_(lr)
            else:
                group["lr"] = lr_val

    def state_dict(self) -> dict[str, object]:
        return {"_step": self._step.item()}

    def load_state_dict(self, state: dict[str, object]) -> None:
        import torch

        self._step.copy_(
            torch.tensor(state["_step"], dtype=torch.int64),
        )
