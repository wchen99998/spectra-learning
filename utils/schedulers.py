from __future__ import annotations

import math


def learning_rate_at_step(
    step: int,
    *,
    base_lr: float,
    total_steps: int,
    warmup_steps: int,
    schedule_type: str = "cosine",
    min_learning_rate: float | None = None,
) -> float:
    if warmup_steps > 0:
        warmup = min(1.0, step / warmup_steps)
        effective_step = max(0, step - warmup_steps)
        effective_total = max(1, total_steps - warmup_steps)
    else:
        warmup = 1.0
        effective_step = step
        effective_total = max(1, total_steps)

    # Parse schedule string (e.g. "cosine" or "cyclic_cosine;cycle_length=500")
    parts = schedule_type.split(";")
    schedule_base = parts[0]
    parsed: dict[str, str] = {}
    for kv in parts[1:]:
        if "=" in kv:
            k, v = kv.split("=", 1)
            parsed[k.strip()] = v.strip()

    if schedule_base == "cosine":
        ratio = max(0.0, effective_step / max(1, effective_total))
        mult = 0.5 * (1.0 + math.cos(math.pi * ratio))
        min_lr_val = (
            min_learning_rate if min_learning_rate is not None else 0.1 * base_lr
        )
        lr = max(min_lr_val, mult * base_lr)
    elif schedule_base == "constant":
        lr = base_lr
    elif schedule_base == "cyclic_cosine":
        cycle_length = int(parsed.get("cycle_length", max(1, total_steps // 10)))
        min_lr = float(parsed.get("min_lr", 0.0))
        decay_factor = float(parsed.get("decay_factor", 1.0))

        cycle_index = effective_step // cycle_length
        pos_in_cycle = effective_step % cycle_length
        peak_lr = base_lr * (decay_factor**cycle_index)
        cosine_ratio = pos_in_cycle / cycle_length
        lr = min_lr + 0.5 * (peak_lr - min_lr) * (
            1.0 + math.cos(math.pi * cosine_ratio)
        )
    else:
        raise NotImplementedError(f"Unknown schedule type: {schedule_base}")

    return lr * warmup


class CapturableCosineSchedule:
    """Warmup + cosine-decay LR schedule using only CUDA tensor ops.

    Unlike ``LambdaLR``, every operation in ``step()`` is a pure tensor op,
    so the entire optimizer-step + schedule-step can live inside a single
    ``torch.compile`` CUDA-graph region with no graph breaks.

    Supports ``state_dict()`` / ``load_state_dict()`` for checkpoint resume.
    """

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
        # Mutable step counter — lives on *device* so .add_ is a GPU op.
        self._step = torch.zeros((), dtype=torch.int64, device=device)
        # Pre-computed constants (all on device).
        self._base_lr = torch.tensor(base_lr, dtype=torch.float64, device=device)
        self._warmup_steps = torch.tensor(
            max(warmup_steps, 1),
            dtype=torch.float64,
            device=device,
        )
        self._has_warmup = warmup_steps > 0
        eff_total = (
            max(1, total_steps - warmup_steps)
            if warmup_steps > 0
            else max(1, total_steps)
        )
        self._effective_total = torch.tensor(
            eff_total,
            dtype=torch.float64,
            device=device,
        )
        self._min_lr = torch.tensor(
            min_lr if min_lr is not None else 0.1 * base_lr,
            dtype=torch.float64,
            device=device,
        )
        self._pi = torch.tensor(math.pi, dtype=torch.float64, device=device)
        self._warmup_steps_int = torch.tensor(
            warmup_steps,
            dtype=torch.int64,
            device=device,
        )

    # -- compilable step --------------------------------------------------

    def step(self) -> None:
        import torch

        self._step.add_(1)
        step_f = self._step.to(torch.float64)

        # Warmup multiplier: clamp(step / warmup_steps, max=1)
        if self._has_warmup:
            warmup = torch.clamp(step_f / self._warmup_steps, max=1.0)
            effective_step = torch.clamp(
                step_f - self._warmup_steps,
                min=0.0,
            )
        else:
            warmup = torch.ones((), dtype=torch.float64, device=self._step.device)
            effective_step = step_f

        # Cosine decay
        ratio = torch.clamp(effective_step / self._effective_total, min=0.0)
        mult = 0.5 * (1.0 + torch.cos(self._pi * ratio))
        lr = torch.max(self._min_lr, mult * self._base_lr) * warmup

        # Write to every param group (in-place, no CPU round-trip).
        for group in self.optimizer.param_groups:
            group["lr"].copy_(lr)

    # -- checkpoint support -----------------------------------------------

    def state_dict(self) -> dict[str, object]:
        return {"_step": self._step.item()}

    def load_state_dict(self, state: dict[str, object]) -> None:
        import torch

        self._step.copy_(
            torch.tensor(state["_step"], dtype=torch.int64),
        )
