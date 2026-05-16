import math

import torch

DEFAULT_JEPA_MASK_STRATEGY = "contiguous"
DEFAULT_JEPA_MASK_LENGTHS = (1, 2, 4, 8, 16)
JEPA_MASK_STRATEGIES = ("contiguous", "ragged", "random")

_DEFAULT_JEPA_MASK_STRATEGY = DEFAULT_JEPA_MASK_STRATEGY
_DEFAULT_JEPA_MASK_LENGTHS = DEFAULT_JEPA_MASK_LENGTHS
_JEPA_MASK_STRATEGIES = JEPA_MASK_STRATEGIES


def _normalize_mask_strategy_name(mask_strategy: str) -> str:
    strategy = mask_strategy.lower()
    if strategy == "ragged_blocks":
        strategy = "ragged"
    return strategy


def _sample_ragged_block_mask_1d_torch(
    active_positions: torch.Tensor,
    *,
    masked_fraction: float,
    lengths: tuple[int, ...],
    round_from: int,
) -> torch.Tensor:
    active_count = int(active_positions.sum().item())
    if active_count == 0:
        return torch.zeros_like(active_positions)
    compressed_positions = torch.cumsum(active_positions.to(torch.int64), dim=0)
    compressed_positions = compressed_positions - active_positions.to(torch.int64)
    bs = torch.rand(len(lengths), device=active_positions.device)
    bs = bs / bs.sum()
    masks_by_length: list[torch.Tensor] = []
    for length_idx, length in enumerate(lengths):
        block_len = length
        max_elem = math.ceil(masked_fraction * active_count / block_len)
        coeff_float = float(bs[length_idx].item()) * max_elem
        coeff = math.ceil(coeff_float) if length_idx < round_from else round(coeff_float)
        if coeff == 0:
            masks_by_length.append(torch.zeros_like(active_positions))
            continue
        effective_len = min(block_len, active_count)
        starts = torch.randint(
            0,
            active_count - effective_len + 1,
            (coeff,),
            device=active_positions.device,
        )
        block_mask = (compressed_positions.unsqueeze(0) >= starts.unsqueeze(1)) & (
            compressed_positions.unsqueeze(0) < (starts + effective_len).unsqueeze(1)
        )
        masks_by_length.append((block_mask & active_positions.unsqueeze(0)).any(dim=0))
    return torch.stack(masks_by_length, dim=0).any(dim=0)


def _sample_contiguous_mask_1d_torch(
    active_positions: torch.Tensor,
    *,
    mask_count: int,
) -> torch.Tensor:
    active_count = int(active_positions.sum().item())
    if active_count == 0:
        return torch.zeros_like(active_positions)
    count = min(mask_count, active_count)
    start = int(torch.randint(active_count - count + 1, (), device=active_positions.device).item())
    compressed_positions = torch.cumsum(active_positions.to(torch.int64), dim=0)
    compressed_positions = compressed_positions - active_positions.to(torch.int64)
    mask = (compressed_positions >= start) & (compressed_positions < start + count)
    return mask & active_positions


def _sample_random_mask_1d_torch(
    active_positions: torch.Tensor,
    *,
    mask_count: int,
) -> torch.Tensor:
    active_indices = torch.nonzero(active_positions, as_tuple=False).squeeze(-1)
    count = min(mask_count, active_indices.numel())
    selected = active_indices[
        torch.randperm(active_indices.numel(), device=active_positions.device)[
            :count
        ]
    ]
    mask = torch.zeros_like(active_positions)
    mask[selected] = True
    return mask


def _fit_mask_to_count(
    mask: torch.Tensor,
    active_positions: torch.Tensor,
    *,
    mask_count: int,
) -> torch.Tensor:
    count = min(mask_count, int(active_positions.sum().item()))
    if count == 0:
        return torch.zeros_like(active_positions)

    out = mask & active_positions
    current = int(out.sum().item())
    if current > count:
        selected = torch.nonzero(out, as_tuple=False).squeeze(-1)
        keep = selected[
            torch.randperm(selected.numel(), device=out.device)[:count]
        ]
        out = torch.zeros_like(active_positions)
        out[keep] = True
        return out

    if current < count:
        candidates = torch.nonzero(active_positions & ~out, as_tuple=False).squeeze(-1)
        add = candidates[
            torch.randperm(candidates.numel(), device=out.device)[
                : count - current
            ]
        ]
        out = out.clone()
        out[add] = True

    return out


def _sample_mask_strategy_torch(
    mask_strategy: str,
    *,
    device: torch.device,
) -> str:
    strategy = _normalize_mask_strategy_name(mask_strategy)
    if strategy == "all":
        return JEPA_MASK_STRATEGIES[
            int(torch.randint(len(JEPA_MASK_STRATEGIES), (), device=device).item())
        ]
    return strategy


def _sample_all_mask_strategies_torch(
    batch_size: int,
    *,
    device: torch.device,
) -> list[str]:
    strategies: list[str] = []
    while len(strategies) < batch_size:
        order = torch.randperm(len(JEPA_MASK_STRATEGIES), device=device)
        strategies.extend(JEPA_MASK_STRATEGIES[int(idx.item())] for idx in order)
    return strategies[: batch_size]


def _target_lengths(
    valid_count: int,
    *,
    num_target_blocks: int,
    context_fraction: float,
    target_fraction: float,
    block_min_len: int,
    allow_target_overlap: bool,
) -> tuple[int, int]:
    desired_context = max(
        round(valid_count * context_fraction),
        block_min_len,
    )
    if num_target_blocks == 0:
        return min(desired_context, valid_count), 0
    reserve_for_targets = min(valid_count, num_target_blocks * block_min_len)
    context_len = min(desired_context, max(valid_count - reserve_for_targets, 0))
    desired_target = max(
        round(valid_count * target_fraction),
        block_min_len,
    )
    available_for_targets = max(valid_count - context_len, 0)
    if allow_target_overlap:
        target_len = min(desired_target, available_for_targets)
    else:
        target_len = min(
            desired_target,
            math.ceil(float(available_for_targets) / float(num_target_blocks)),
        )
    return context_len, target_len


def _sample_row_mask(
    row_valid: torch.Tensor,
    strategy: str,
    *,
    mask_count: int,
    mask_lengths: tuple[int, ...],
    mask_round_from: int,
) -> torch.Tensor:
    active_count = int(row_valid.sum().item())
    count = min(mask_count, active_count)
    if count == 0:
        return torch.zeros_like(row_valid)
    if strategy == "contiguous":
        mask = _sample_contiguous_mask_1d_torch(row_valid, mask_count=count)
    elif strategy == "random":
        mask = _sample_random_mask_1d_torch(row_valid, mask_count=count)
    else:
        mask = _sample_ragged_block_mask_1d_torch(
            row_valid,
            masked_fraction=float(count) / float(active_count),
            lengths=mask_lengths,
            round_from=mask_round_from,
        )
    return _fit_mask_to_count(
        mask,
        row_valid,
        mask_count=count,
    )


def _reserve_target_capacity(
    row_context: torch.Tensor,
    row_valid: torch.Tensor,
    *,
    reserve_count: int,
) -> torch.Tensor:
    available = int((row_valid & ~row_context).sum().item())
    needed = min(reserve_count, int(row_valid.sum().item())) - available
    if needed <= 0:
        return row_context
    context_indices = torch.nonzero(row_context, as_tuple=False).squeeze(-1)
    drop = context_indices[
        torch.randperm(context_indices.numel(), device=row_context.device)[:needed]
    ]
    out = row_context.clone()
    out[drop] = False
    return out


def _sample_target_rows(
    target_masks: torch.Tensor,
    row_idx: int,
    valid_target_positions: torch.Tensor,
    strategy: str,
    *,
    target_len: int,
    mask_lengths: tuple[int, ...],
    mask_round_from: int,
    allow_target_overlap: bool,
) -> None:
    available = valid_target_positions.clone()
    if not bool(available.any()):
        return
    for block_idx in range(target_masks.shape[1]):
        block_count = min(target_len, int(available.sum().item()))
        if block_count == 0:
            return
        target_masks[row_idx, block_idx] = _sample_row_mask(
            available,
            strategy,
            mask_count=block_count,
            mask_lengths=mask_lengths,
            mask_round_from=mask_round_from,
        )
        if not allow_target_overlap:
            available = available & ~target_masks[row_idx, block_idx]
        else:
            available = valid_target_positions


def _sample_block_masks_torch(
    peak_valid_mask: torch.Tensor,
    *,
    num_target_blocks: int,
    context_fraction: float,
    target_fraction: float,
    block_min_len: int,
    mask_strategy: str = DEFAULT_JEPA_MASK_STRATEGY,
    mask_lengths: tuple[int, ...] = DEFAULT_JEPA_MASK_LENGTHS,
    mask_round_from: int = len(DEFAULT_JEPA_MASK_LENGTHS),
    allow_target_overlap: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    strategy = _normalize_mask_strategy_name(mask_strategy)
    if strategy not in {*JEPA_MASK_STRATEGIES, "all"}:
        raise ValueError(f"Unsupported JEPA mask strategy: {mask_strategy!r}")
    lengths = tuple(length for length in mask_lengths)
    round_from = mask_round_from
    batch_size, num_peaks = peak_valid_mask.shape
    context_mask = torch.zeros_like(peak_valid_mask)
    target_masks = torch.zeros(
        batch_size,
        num_target_blocks,
        num_peaks,
        dtype=torch.bool,
        device=peak_valid_mask.device,
    )
    all_row_strategies = (
        _sample_all_mask_strategies_torch(batch_size, device=peak_valid_mask.device)
        if strategy == "all"
        else None
    )
    for row_idx in range(batch_size):
        row_valid = peak_valid_mask[row_idx]
        valid_count = int(row_valid.sum().item())
        if valid_count == 0:
            continue
        row_strategy = (
            all_row_strategies[row_idx]
            if all_row_strategies is not None
            else _sample_mask_strategy_torch(strategy, device=peak_valid_mask.device)
        )
        context_len, target_len = _target_lengths(
            valid_count,
            num_target_blocks=num_target_blocks,
            context_fraction=context_fraction,
            target_fraction=target_fraction,
            block_min_len=block_min_len,
            allow_target_overlap=allow_target_overlap,
        )
        row_context = _sample_row_mask(
            row_valid,
            row_strategy,
            mask_count=context_len,
            mask_lengths=lengths,
            mask_round_from=round_from,
        )
        row_context = _reserve_target_capacity(
            row_context,
            row_valid,
            reserve_count=num_target_blocks * block_min_len,
        )
        context_mask[row_idx] = row_context
        if num_target_blocks > 0 and target_len > 0:
            _sample_target_rows(
                target_masks,
                row_idx,
                row_valid & ~row_context,
                row_strategy,
                target_len=target_len,
                mask_lengths=lengths,
                mask_round_from=round_from,
                allow_target_overlap=allow_target_overlap,
            )
    return context_mask, target_masks
