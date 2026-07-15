from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class JepaMaskStage:
    index: int
    context_fraction: float
    target_fraction: float
    gradient_accumulation_steps: int


def jepa_mask_stages(config: Any) -> tuple[JepaMaskStage, ...]:
    context_fractions = tuple(
        float(value)
        for value in config.get(
            "jepa_context_fraction_schedule",
            (config.get("jepa_context_fraction", 0.5),),
        )
    )
    target_fractions = tuple(
        float(value)
        for value in config.get(
            "jepa_target_fraction_schedule",
            (config.get("jepa_target_fraction", 0.25),),
        )
    )
    gradient_accumulation_steps = tuple(
        int(value)
        for value in config.get(
            "gradient_accumulation_steps_schedule",
            (int(config.get("gradient_accumulation_steps", 1)),)
            * len(context_fractions),
        )
    )
    if not (
        len(context_fractions)
        == len(target_fractions)
        == len(gradient_accumulation_steps)
    ):
        raise ValueError(
            "mask and gradient accumulation schedules must have equal lengths"
        )
    return tuple(
        JepaMaskStage(index, context_fraction, target_fraction, accumulation_steps)
        for index, (context_fraction, target_fraction, accumulation_steps) in enumerate(
            zip(
                context_fractions,
                target_fractions,
                gradient_accumulation_steps,
                strict=True,
            )
        )
    )


def jepa_mask_stage_boundaries(config: Any) -> tuple[float, ...]:
    stages = jepa_mask_stages(config)
    return tuple(
        float(value)
        for value in config.get(
            "jepa_mask_schedule_step_fractions",
            tuple(index / len(stages) for index in range(1, len(stages))),
        )
    )


def jepa_mask_stage_index(config: Any, global_step: int, total_steps: int) -> int:
    boundaries = tuple(
        int(step)
        for step in config.get(
            "jepa_mask_schedule_steps",
            tuple(
                math.ceil(total_steps * fraction)
                for fraction in jepa_mask_stage_boundaries(config)
            ),
        )
    )
    return bisect_right(boundaries, global_step)


def jepa_mask_stage(config: Any, global_step: int, total_steps: int) -> JepaMaskStage:
    stages = jepa_mask_stages(config)
    return stages[jepa_mask_stage_index(config, global_step, total_steps)]
