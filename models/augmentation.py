from __future__ import annotations

import torch


def _random_keep_view(
    peak_mz: torch.Tensor,
    peak_intensity: torch.Tensor,
    peak_valid_mask: torch.Tensor,
    *,
    keep_fraction: float,
    mz_jitter_std: float,
    intensity_jitter_std: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a single multicrop view by randomly keeping *keep_fraction* of valid peaks.

    Returns (view_mz, view_intensity, view_valid_mask) with dropped peaks
    compacted to the front.
    """
    batch_size, num_peaks = peak_mz.shape

    valid_counts = peak_valid_mask.sum(dim=1)  # [B]
    keep_counts = (valid_counts.float() * keep_fraction).round().clamp(min=1).long()

    # Random scores; -inf on invalid positions so they sort last
    scores = torch.rand(batch_size, num_peaks, device=peak_mz.device)
    scores = torch.where(peak_valid_mask, scores, torch.full_like(scores, float("-inf")))
    order = scores.argsort(dim=1, descending=True)  # highest scores first

    # Build keep mask in the *sorted* space, then scatter back
    positions = torch.arange(num_peaks, device=peak_mz.device).unsqueeze(0)
    keep_sorted = positions < keep_counts.unsqueeze(1)
    # Map back to original positions
    inverse_order = order.argsort(dim=1)
    keep_mask = keep_sorted.gather(1, inverse_order) & peak_valid_mask

    # Compact kept peaks to the front
    pack_keys = torch.where(keep_mask, positions.expand_as(peak_mz), positions.expand_as(peak_mz) + num_peaks)
    compact_order = pack_keys.argsort(dim=1, stable=True)

    compact_mz = peak_mz.gather(1, compact_order)
    compact_int = peak_intensity.gather(1, compact_order)
    compact_valid = keep_mask.gather(1, compact_order)

    # Zero out invalid positions
    compact_mz = torch.where(compact_valid, compact_mz, torch.zeros_like(compact_mz))
    compact_int = torch.where(compact_valid, compact_int, torch.zeros_like(compact_int))

    # Jitter
    mz_noise = torch.randn_like(compact_mz) * mz_jitter_std
    view_mz = torch.where(compact_valid, (compact_mz + mz_noise).clamp(0.0, 1.0), torch.zeros_like(compact_mz))

    int_noise = torch.randn_like(compact_int) * intensity_jitter_std
    view_int = torch.where(compact_valid, (compact_int + int_noise).clamp(0.0, 1.0), torch.zeros_like(compact_int))

    # Re-normalize intensity
    max_int = view_int.max(dim=1, keepdim=True).values.clamp(min=1e-6)
    view_int = view_int / max_int
    view_int = torch.where(compact_valid, view_int, torch.zeros_like(view_int))

    return view_mz, view_int, compact_valid


def augment_multicrop_batch(
    batch: dict[str, torch.Tensor],
    *,
    num_global_views: int = 2,
    num_local_views: int = 6,
    global_keep_fraction: float = 0.80,
    local_keep_fraction: float = 0.25,
    mz_jitter_std: float,
    intensity_jitter_std: float,
) -> dict[str, torch.Tensor]:
    """Multi-crop augmentation: *num_global_views* globals + *num_local_views* locals.

    Returns fused tensors ``[V*B, N]`` where ``V = num_global_views + num_local_views``.
    """
    peak_mz = batch["peak_mz"]
    peak_intensity = batch["peak_intensity"]
    peak_valid_mask = batch["peak_valid_mask"]
    precursor_mz = batch["precursor_mz"]

    all_mz: list[torch.Tensor] = []
    all_int: list[torch.Tensor] = []
    all_valid: list[torch.Tensor] = []

    for _ in range(num_global_views):
        mz, intensity, valid = _random_keep_view(
            peak_mz, peak_intensity, peak_valid_mask,
            keep_fraction=global_keep_fraction,
            mz_jitter_std=mz_jitter_std,
            intensity_jitter_std=intensity_jitter_std,
        )
        all_mz.append(mz)
        all_int.append(intensity)
        all_valid.append(valid)

    for _ in range(num_local_views):
        mz, intensity, valid = _random_keep_view(
            peak_mz, peak_intensity, peak_valid_mask,
            keep_fraction=local_keep_fraction,
            mz_jitter_std=mz_jitter_std,
            intensity_jitter_std=intensity_jitter_std,
        )
        all_mz.append(mz)
        all_int.append(intensity)
        all_valid.append(valid)

    num_views = num_global_views + num_local_views
    return {
        "fused_mz": torch.cat(all_mz, dim=0),
        "fused_intensity": torch.cat(all_int, dim=0),
        "fused_precursor_mz": precursor_mz.repeat(num_views),
        "fused_valid_mask": torch.cat(all_valid, dim=0),
    }
