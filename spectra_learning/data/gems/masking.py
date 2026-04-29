import math

import torch

DEFAULT_JEPA_MASK_STRATEGY = "contiguous"
DEFAULT_JEPA_MASK_LENGTHS = (1, 2, 4, 8, 16)
JEPA_MASK_STRATEGIES = ("contiguous", "ragged")

_DEFAULT_JEPA_MASK_STRATEGY = DEFAULT_JEPA_MASK_STRATEGY
_DEFAULT_JEPA_MASK_LENGTHS = DEFAULT_JEPA_MASK_LENGTHS
_JEPA_MASK_STRATEGIES = JEPA_MASK_STRATEGIES


def _normalize_mask_strategy_name(mask_strategy: str) -> str:
    strategy = str(mask_strategy).lower()
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
        block_len = int(length)
        max_elem = int(
            math.ceil(float(masked_fraction) * float(active_count) / float(block_len))
        )
        coeff_float = float(bs[length_idx].item()) * float(max_elem)
        coeff = int(math.ceil(coeff_float)) if length_idx < round_from else int(round(coeff_float))
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
    count = min(int(mask_count), active_count)
    start = int(torch.randint(active_count - count + 1, (), device=active_positions.device).item())
    compressed_positions = torch.cumsum(active_positions.to(torch.int64), dim=0)
    compressed_positions = compressed_positions - active_positions.to(torch.int64)
    mask = (compressed_positions >= start) & (compressed_positions < start + count)
    return mask & active_positions


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


def _target_lengths(
    valid_count: int,
    *,
    num_target_blocks: int,
    context_fraction: float,
    target_fraction: float,
    block_min_len: int,
) -> tuple[int, int]:
    desired_context = max(int(round(valid_count * float(context_fraction))), int(block_min_len))
    if num_target_blocks == 0:
        return min(desired_context, valid_count), 0
    reserve_for_targets = min(valid_count, int(num_target_blocks) * int(block_min_len))
    context_len = min(desired_context, max(valid_count - reserve_for_targets, 1))
    desired_target = max(int(round(valid_count * float(target_fraction))), int(block_min_len))
    target_len = min(desired_target, max(valid_count - context_len, 0))
    return context_len, target_len


def _sample_row_mask(
    row_valid: torch.Tensor,
    strategy: str,
    *,
    mask_count: int,
    mask_lengths: tuple[int, ...],
    mask_round_from: int,
) -> torch.Tensor:
    if strategy == "contiguous":
        return _sample_contiguous_mask_1d_torch(row_valid, mask_count=mask_count)
    return _sample_ragged_block_mask_1d_torch(
        row_valid,
        masked_fraction=float(mask_count) / float(row_valid.sum().item()),
        lengths=mask_lengths,
        round_from=mask_round_from,
    )


def _sample_target_rows(
    target_masks: torch.Tensor,
    row_idx: int,
    valid_target_positions: torch.Tensor,
    strategy: str,
    *,
    target_len: int,
    mask_lengths: tuple[int, ...],
    mask_round_from: int,
) -> None:
    if strategy == "contiguous":
        for block_idx in range(int(target_masks.shape[1])):
            target_masks[row_idx, block_idx] = _sample_contiguous_mask_1d_torch(
                valid_target_positions,
                mask_count=target_len,
            )
        return
    available = int(valid_target_positions.sum().item())
    if available == 0:
        return
    for block_idx in range(int(target_masks.shape[1])):
        target_masks[row_idx, block_idx] = _sample_ragged_block_mask_1d_torch(
            valid_target_positions,
            masked_fraction=float(target_len) / float(available),
            lengths=mask_lengths,
            round_from=mask_round_from,
        )


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
) -> tuple[torch.Tensor, torch.Tensor]:
    strategy = _normalize_mask_strategy_name(mask_strategy)
    if strategy not in {*JEPA_MASK_STRATEGIES, "all"}:
        raise ValueError(f"Unsupported JEPA mask strategy: {mask_strategy!r}")
    lengths = tuple(int(length) for length in mask_lengths)
    round_from = int(mask_round_from)
    batch_size, num_peaks = peak_valid_mask.shape
    context_mask = torch.zeros_like(peak_valid_mask)
    target_masks = torch.zeros(
        batch_size,
        int(num_target_blocks),
        num_peaks,
        dtype=torch.bool,
        device=peak_valid_mask.device,
    )
    for row_idx in range(batch_size):
        row_valid = peak_valid_mask[row_idx]
        valid_count = int(row_valid.sum().item())
        if valid_count == 0:
            continue
        row_strategy = _sample_mask_strategy_torch(strategy, device=peak_valid_mask.device)
        context_len, target_len = _target_lengths(
            valid_count,
            num_target_blocks=int(num_target_blocks),
            context_fraction=context_fraction,
            target_fraction=target_fraction,
            block_min_len=block_min_len,
        )
        row_context = _sample_row_mask(
            row_valid,
            row_strategy,
            mask_count=context_len,
            mask_lengths=lengths,
            mask_round_from=round_from,
        )
        context_mask[row_idx] = row_context
        if int(num_target_blocks) > 0 and target_len > 0:
            _sample_target_rows(
                target_masks,
                row_idx,
                row_valid & ~row_context,
                row_strategy,
                target_len=target_len,
                mask_lengths=lengths,
                mask_round_from=round_from,
            )
    return context_mask, target_masks
