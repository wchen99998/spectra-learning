from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from typing import Any

from spectra_learning.data.spectra import DEFAULT_NUM_PEAKS


@dataclass(frozen=True)
class JepaMaskStage:
    index: int
    context_fraction: float
    target_fraction: float
    gradient_accumulation_steps: int
    num_peaks: int


SCHEDULE_KEYS = (
    "jepa_context_fraction_schedule",
    "jepa_target_fraction_schedule",
    "gradient_accumulation_steps_schedule",
    # Peak-count stages (e.g. 31 peaks, then 63). The model is built for
    # config.num_peaks, which must be the largest stage.
    "num_peaks_schedule",
)


def jepa_mask_stages(config: Any) -> tuple[JepaMaskStage, ...]:
    num_stages = max(len(config.get(key, ())) for key in SCHEDULE_KEYS) or 1
    max_num_peaks = int(config.get("num_peaks", DEFAULT_NUM_PEAKS))
    defaults = (
        config.get("jepa_context_fraction", 0.5),
        config.get("jepa_target_fraction", 0.25),
        config.get("gradient_accumulation_steps", 1),
        max_num_peaks,
    )
    schedules = [
        tuple(config.get(key, (default,) * num_stages))
        for key, default in zip(SCHEDULE_KEYS, defaults, strict=True)
    ]
    if any(len(schedule) != num_stages for schedule in schedules):
        raise ValueError(f"{', '.join(SCHEDULE_KEYS)} must have equal lengths")
    if max(schedules[3]) != max_num_peaks:
        raise ValueError("num_peaks must equal the largest num_peaks_schedule value")
    return tuple(
        JepaMaskStage(index, float(context), float(target), int(accumulation), int(peaks))
        for index, (context, target, accumulation, peaks) in enumerate(
            zip(*schedules, strict=True)
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
