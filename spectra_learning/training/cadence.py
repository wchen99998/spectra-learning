from __future__ import annotations

from typing import Any


def resolve_step_interval(
    config: Any,
    datamodule: Any,
    total_steps: int,
    key: str,
    *,
    default: float,
) -> int:
    raw = float(_config_get(config, key, default))
    if raw < 0:
        return int(raw)
    if raw == 0:
        return total_steps
    if 0 < raw <= 1:
        reference_steps = (
            total_steps
            if float(_config_get(config, "num_epochs", 1)) < 1
            else int(datamodule.train_steps)
        )
        return max(1, int(raw * reference_steps))
    return int(raw)


def should_run_at_step(interval: int, global_step: int) -> bool:
    return interval > 0 and global_step % interval == 0


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
    return int(_config_get(config, "val_num_steps", 64))


def _config_get(config: Any, key: str, default: Any) -> Any:
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)
