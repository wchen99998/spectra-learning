from __future__ import annotations

import math


def parse_schedule(schedule_type: str) -> tuple[str, dict[str, str]]:
    parts = schedule_type.split(";")
    base = parts[0]
    parsed: dict[str, str] = {}
    for kv in parts[1:]:
        if "=" in kv:
            k, v = kv.split("=", 1)
            parsed[k.strip()] = v.strip()
    return base, parsed


def cosine_decay(
    base_lr: float,
    step: int,
    total_steps: int,
    *,
    min_lr: float | None,
) -> float:
    total_steps = max(1, total_steps)
    ratio = max(0.0, step / total_steps)
    mult = 0.5 * (1.0 + math.cos(math.pi * ratio))
    decayed = mult * base_lr
    min_lr_value = min_lr if min_lr is not None else 0.1 * base_lr
    return max(min_lr_value, decayed)


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

    schedule_base, parsed = parse_schedule(schedule_type)

    if schedule_base == "cosine":
        lr = cosine_decay(
            base_lr,
            effective_step,
            effective_total,
            min_lr=min_learning_rate,
        )
    elif schedule_base == "constant":
        lr = base_lr
    elif schedule_base == "cyclic_cosine":
        cycle_length = int(parsed.get("cycle_length", max(1, total_steps // 10)))
        min_lr = float(parsed.get("min_lr", 0.0))
        decay_factor = float(parsed.get("decay_factor", 1.0))

        cycle_index = effective_step // cycle_length
        pos_in_cycle = effective_step % cycle_length
        peak_lr = base_lr * (decay_factor ** cycle_index)
        cosine_ratio = pos_in_cycle / cycle_length
        lr = min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * cosine_ratio))
    else:
        raise NotImplementedError(f"Unknown schedule type: {schedule_base}")

    return lr * warmup
