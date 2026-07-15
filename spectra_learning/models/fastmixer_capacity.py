from __future__ import annotations

from typing import Any

from spectra_learning.data.gems.mask_schedule import jepa_mask_stages
from spectra_learning.data.gems.masking import jepa_mask_lengths_for_valid_count


def pairmixer_fast_full_visible_tokens(config: Any) -> int:
    return int(config.get("num_peaks", 64)) + 1


def _mask_strategy_names(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = value.split(",")
    else:
        values = value
    return tuple(str(item).strip().lower() for item in values if str(item).strip())


def _mae_encoder_predictor_visible_tokens(
    config: Any,
    *,
    context_fraction: float | None = None,
    target_fraction: float | None = None,
) -> tuple[int, int]:
    num_peaks = int(config.get("num_peaks", 64))
    strategies = _mask_strategy_names(
        config.get("jepa_mask_strategy", "contiguous")
    )
    if "all" in strategies or "intensity_aware" in strategies:
        full_visible = pairmixer_fast_full_visible_tokens(config)
        return full_visible, full_visible
    num_target_blocks = int(config.get("jepa_num_target_blocks", 2))
    context_fraction = (
        float(config.get("jepa_context_fraction", 0.5))
        if context_fraction is None
        else context_fraction
    )
    target_fraction = (
        float(config.get("jepa_target_fraction", 0.25))
        if target_fraction is None
        else target_fraction
    )
    block_min_len = int(config.get("jepa_block_min_len", 1))
    allow_target_overlap = bool(config.get("jepa_allow_target_overlap", False))
    masked_token_input_mode = str(
        config.get("masked_token_input_mode", "latent_token")
    ).lower()

    encoder_max_visible = 1
    predictor_max_visible = 1
    for valid_count in range(1, num_peaks + 1):
        context_len, target_len = jepa_mask_lengths_for_valid_count(
            valid_count,
            num_target_blocks=num_target_blocks,
            context_fraction=context_fraction,
            target_fraction=target_fraction,
            block_min_len=block_min_len,
            allow_target_overlap=allow_target_overlap,
        )
        encoder_visible = context_len + 1
        if masked_token_input_mode == "mz_sentinel":
            target_union_len = min(
                max(valid_count - context_len, 0),
                num_target_blocks * target_len,
            )
            encoder_visible = context_len + target_union_len + 1
        predictor_visible = context_len + target_len + 1
        encoder_max_visible = max(encoder_max_visible, encoder_visible)
        predictor_max_visible = max(predictor_max_visible, predictor_visible)
    return encoder_max_visible, predictor_max_visible


def pairmixer_fast_mae_stage_visible_tokens(
    config: Any,
) -> tuple[tuple[int, int], ...]:
    return tuple(
        _mae_encoder_predictor_visible_tokens(
            config,
            context_fraction=stage.context_fraction,
            target_fraction=stage.target_fraction,
        )
        for stage in jepa_mask_stages(config)
    )


def pairmixer_fast_stage_capacities(
    config: Any,
) -> tuple[tuple[int, int], ...]:
    return pairmixer_fast_mae_stage_visible_tokens(config)


def pairmixer_stage_projection_kernels(
    config: Any,
) -> tuple[tuple[str, str], ...]:
    stage_count = len(jepa_mask_stages(config))
    encoder_kernel = str(
        config.get("pairmixer_encoder_projection_kernel", "xla")
    )
    predictor_kernel = str(
        config.get("pairmixer_predictor_projection_kernel", "xla")
    )
    encoder_kernels = config.get(
        "pairmixer_encoder_projection_kernel_schedule",
        (encoder_kernel,) * stage_count,
    )
    predictor_kernels = config.get(
        "pairmixer_predictor_projection_kernel_schedule",
        (predictor_kernel,) * stage_count,
    )
    return tuple(
        (str(encoder), str(predictor))
        for encoder, predictor in zip(
            encoder_kernels,
            predictor_kernels,
            strict=True,
        )
    )


def pairmixer_fast_mae_encoder_visible_tokens(config: Any) -> int:
    encoder_visible, _ = _mae_encoder_predictor_visible_tokens(config)
    return encoder_visible


def pairmixer_fast_mae_visible_tokens(config: Any) -> int:
    encoder_visible, predictor_visible = _mae_encoder_predictor_visible_tokens(config)
    return max(encoder_visible, predictor_visible)


def pairmixer_fast_required_visible_tokens(
    config: Any,
    *,
    mode: str = "train",
) -> int:
    if mode == "full_visible":
        return pairmixer_fast_full_visible_tokens(config)
    training_mode = str(config.get("training_mode", "jepa")).lower()
    if training_mode == "mae":
        return pairmixer_fast_mae_visible_tokens(config)
    return pairmixer_fast_full_visible_tokens(config)


def resolve_pairmixer_fast_max_visible_tokens(
    config: Any,
    *,
    mode: str = "train",
) -> int | None:
    block_type = str(config.get("pairmixer_block_type", "dense")).lower()
    if "pairmixer_fast_max_visible_tokens" in config:
        raise ValueError(
            "pairmixer_fast_max_visible_tokens is derived from the data config; "
            "remove it from experiment configs"
        )
    if block_type not in {"fastmixer", "fastmixer-dense"}:
        return None
    return pairmixer_fast_required_visible_tokens(config, mode=mode)


def resolve_pairmixer_fast_encoder_max_visible_tokens(
    config: Any,
    *,
    mode: str = "train",
) -> int | None:
    block_type = str(config.get("pairmixer_block_type", "dense")).lower()
    if block_type not in {"fastmixer", "fastmixer-dense"}:
        return None
    if mode != "train":
        return pairmixer_fast_required_visible_tokens(config, mode=mode)
    training_mode = str(config.get("training_mode", "jepa")).lower()
    if training_mode == "mae":
        return pairmixer_fast_mae_encoder_visible_tokens(config)
    return pairmixer_fast_full_visible_tokens(config)
