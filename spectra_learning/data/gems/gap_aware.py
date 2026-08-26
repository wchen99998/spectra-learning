"""Gap-aware target masking.

Uniform random masking splits the target budget between two populations that want
different things from the model. Measured on real batches at 1B scale: masked peaks
within 2 Da of a visible peak carry half the tokens but only a quarter of the loss,
and their gradient sits at cosine −0.775 to the gradient from isolated peaks — at a
young checkpoint as well as a converged one. The combined gradient is five to seven
times smaller than either part.

This strategy samples context exactly as ``random`` does, then draws targets with a
configurable weight per distance-to-nearest-context-peak bucket, so the mix is a
knob rather than an accident of the peak distribution.

The weights are sampling priorities, not target proportions: targets are drawn without
replacement, and at the production fractions (context 0.60, target 0.25) they already
consume most of the peaks left over from context selection. A zero-weight bucket is a
hard exclusion. If fewer positive-weight peaks are available than requested, the mask
is deliberately shorter instead of being backfilled from an excluded bucket.
"""

from __future__ import annotations

import torch

from spectra_learning.data.gems.masking import (
    _sample_random_mask_1d_torch,
    _reserve_target_capacity,
    jepa_mask_lengths_for_valid_count,
)

GAP_AWARE_MASK_STRATEGY = "gap_aware"

DEFAULT_GAP_AWARE_BUCKET_EDGES_DA = (2.0, 5.0)
DEFAULT_GAP_AWARE_BUCKET_WEIGHTS = (0.2, 0.4, 0.4)


def gap_to_nearest(
    peak_mz_da: torch.Tensor,
    context_mask: torch.Tensor,
) -> torch.Tensor:
    """Distance in Da from every peak to the nearest peak selected as context."""
    if not bool(context_mask.any()):
        return torch.full_like(peak_mz_da, float("inf"))
    context_mz = peak_mz_da[context_mask]
    return (peak_mz_da[:, None] - context_mz[None, :]).abs().min(dim=1).values


def bucket_sampling_weights(
    gap_da: torch.Tensor,
    available: torch.Tensor,
    *,
    bucket_edges_da: tuple[float, ...],
    bucket_weights: tuple[float, ...],
) -> torch.Tensor:
    """Per-peak sampling priorities with equal total weight per non-empty bucket."""
    edges = torch.tensor(bucket_edges_da, dtype=gap_da.dtype, device=gap_da.device)
    bucket = torch.bucketize(gap_da, edges, right=True)
    weights = torch.zeros_like(gap_da)
    for index, share in enumerate(bucket_weights):
        in_bucket = available & (bucket == index)
        count = int(in_bucket.sum().item())
        if count == 0 or share <= 0.0:
            continue
        weights[in_bucket] = share / float(count)
    return weights


def _sample_weighted(weights: torch.Tensor, count: int) -> torch.Tensor:
    positive = int((weights > 0).sum().item())
    take = min(count, positive)
    mask = torch.zeros_like(weights, dtype=torch.bool)
    if take <= 0:
        return mask
    chosen = torch.multinomial(weights, take, replacement=False)
    mask[chosen] = True
    return mask


def sample_gap_aware_masks_torch(
    peak_valid_mask: torch.Tensor,
    peak_mz_da: torch.Tensor,
    *,
    num_target_blocks: int,
    context_fraction: float,
    target_fraction: float,
    block_min_len: int,
    allow_target_overlap: bool,
    bucket_edges_da: tuple[float, ...] = DEFAULT_GAP_AWARE_BUCKET_EDGES_DA,
    bucket_weights: tuple[float, ...] = DEFAULT_GAP_AWARE_BUCKET_WEIGHTS,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(bucket_weights) != len(bucket_edges_da) + 1:
        raise ValueError(
            "gap-aware masking needs one more bucket weight than bucket edge; got "
            f"{len(bucket_weights)} weights for {len(bucket_edges_da)} edges"
        )
    batch_size, num_peaks = peak_valid_mask.shape
    context_mask = torch.zeros_like(peak_valid_mask)
    target_masks = torch.zeros(
        batch_size,
        num_target_blocks,
        num_peaks,
        dtype=torch.bool,
        device=peak_valid_mask.device,
    )
    for row_idx in range(batch_size):
        row_valid = peak_valid_mask[row_idx]
        valid_count = int(row_valid.sum().item())
        if valid_count == 0:
            continue
        context_len, target_len = jepa_mask_lengths_for_valid_count(
            valid_count,
            num_target_blocks=num_target_blocks,
            context_fraction=context_fraction,
            target_fraction=target_fraction,
            block_min_len=block_min_len,
            allow_target_overlap=allow_target_overlap,
        )
        row_context = _sample_random_mask_1d_torch(
            row_valid,
            mask_count=min(context_len, valid_count),
        )
        row_context = _reserve_target_capacity(
            row_context,
            row_valid,
            reserve_count=num_target_blocks * block_min_len,
        )
        context_mask[row_idx] = row_context
        if num_target_blocks <= 0 or target_len <= 0:
            continue
        gap_da = gap_to_nearest(peak_mz_da[row_idx], row_context)
        available = row_valid & ~row_context
        for block_idx in range(num_target_blocks):
            if not bool(available.any()):
                break
            weights = bucket_sampling_weights(
                gap_da,
                available,
                bucket_edges_da=bucket_edges_da,
                bucket_weights=bucket_weights,
            )
            block = _sample_weighted(weights, min(target_len, int(available.sum())))
            target_masks[row_idx, block_idx] = block
            if not allow_target_overlap:
                available = available & ~block
    return context_mask, target_masks
