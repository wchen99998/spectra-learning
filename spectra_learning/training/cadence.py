from __future__ import annotations

from typing import Any


def total_training_steps(config: Any, datamodule: Any) -> int:
    total_steps = max(1, int(float(config.num_epochs) * datamodule.train_steps))
    training_max_steps = config.get("training_max_steps", None)
    if training_max_steps is None:
        return total_steps
    return min(total_steps, max(1, int(training_max_steps)))


def resolve_step_interval(
    config: Any,
    datamodule: Any,
    total_steps: int,
    key: str,
    *,
    default: float,
) -> int:
    raw = float(config.get(key, default))
    if raw < 0:
        return int(raw)
    if raw == 0:
        return total_steps
    if 0 < raw <= 1:
        reference_steps = (
            total_steps
            if float(config.get("num_epochs", 1)) < 1
            else int(datamodule.train_steps)
        )
        return max(1, int(raw * reference_steps))
    return int(raw)


def should_run_at_step(interval: int, global_step: int) -> bool:
    return interval > 0 and global_step % interval == 0


def should_run_at_step_or_final(
    interval: int,
    global_step: int,
    *,
    total_steps: int,
    run_at_final_step: bool,
) -> bool:
    if should_run_at_step(interval, global_step):
        return True
    return bool(run_at_final_step and interval > 0 and global_step == total_steps)


def msg_probe_interval(config: Any, datamodule: Any, total_steps: int) -> int:
    return resolve_step_interval(
        config,
        datamodule,
        total_steps,
        "msg_probe_every_n_steps",
        default=0,
    )


def validation_interval(config: Any, datamodule: Any, total_steps: int) -> int:
    return resolve_step_interval(
        config,
        datamodule,
        total_steps,
        "val_every_n_steps",
        default=-1,
    )


def validation_steps(config: Any) -> int:
    return int(config.get("val_num_steps", 64))
