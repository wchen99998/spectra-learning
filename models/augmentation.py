from __future__ import annotations

import torch


def augment_masked_view(
    peak_mz: torch.Tensor,
    peak_intensity: torch.Tensor,
    peak_valid_mask: torch.Tensor,
    *,
    contiguous_mask_fraction: float,
    contiguous_mask_min_len: int,
    mz_jitter_std: float,
    intensity_jitter_std: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    batch_size, num_peaks = peak_mz.shape
    has_valid = peak_valid_mask.any(dim=1)

    sort_keys = torch.where(
        peak_valid_mask,
        peak_mz,
        torch.full_like(peak_mz, float("inf")),
    )
    sorted_order = torch.argsort(sort_keys, dim=1)
    sorted_valid = torch.gather(peak_valid_mask, dim=1, index=sorted_order)
    valid_counts = sorted_valid.sum(dim=1)
    interval_start = torch.zeros(batch_size, dtype=torch.long, device=peak_mz.device)
    interval_end = torch.clamp(valid_counts - 1, min=0)
    interval_lengths = valid_counts
    density_interval_lengths = valid_counts
    pos = torch.arange(num_peaks, device=peak_mz.device)

    raw_mask_len = torch.floor(
        interval_lengths.float() * contiguous_mask_fraction
    ).to(torch.long)
    mask_len = torch.maximum(
        raw_mask_len,
        torch.full_like(raw_mask_len, contiguous_mask_min_len),
    )
    mask_len = torch.minimum(mask_len, interval_lengths)
    mask_len = torch.where(has_valid, mask_len, torch.zeros_like(mask_len))

    max_start_offset = interval_lengths - mask_len + 1
    sampled_offset = torch.floor(
        torch.rand(batch_size, device=peak_mz.device) * max_start_offset.float(),
    ).to(torch.long)
    sampled_offset = torch.where(has_valid, sampled_offset, torch.zeros_like(sampled_offset))
    mask_start = interval_start + sampled_offset
    mask_end = mask_start + mask_len

    sorted_positions = pos.view(1, num_peaks)
    masked_sorted = (
        has_valid.view(batch_size, 1)
        & (sorted_positions >= mask_start.view(batch_size, 1))
        & (sorted_positions < mask_end.view(batch_size, 1))
        & (sorted_positions < valid_counts.view(batch_size, 1))
    )
    masked = torch.zeros_like(peak_valid_mask)
    masked.scatter_(1, sorted_order, masked_sorted)
    masked = masked & peak_valid_mask

    view_valid = peak_valid_mask
    jitterable = peak_valid_mask & (~masked)

    mz = torch.zeros_like(peak_mz)
    mz[jitterable] = peak_mz[jitterable] + (
        torch.randn_like(peak_mz[jitterable]) * mz_jitter_std
    )
    mz = torch.clamp(mz, min=0.0, max=1.0)

    intensity = torch.zeros_like(peak_intensity)
    intensity[jitterable] = peak_intensity[jitterable] + (
        torch.randn_like(peak_intensity[jitterable]) * intensity_jitter_std
    )
    intensity = torch.clamp(intensity, min=0.0, max=1.0)

    max_intensity = intensity.max(dim=1, keepdim=True).values.clamp(min=1e-6)
    intensity = intensity / max_intensity
    intensity = torch.where(jitterable, intensity, 0.0)

    valid_counts = peak_valid_mask.sum(dim=1).clamp(min=1)
    masked_counts = masked.sum(dim=1)
    masked_fraction = (masked_counts.float() / valid_counts.float()).mean()
    density_interval_fraction = (
        density_interval_lengths.float() / valid_counts.float()
    ).mean()

    return (
        mz,
        intensity,
        view_valid,
        masked,
        masked_fraction,
        density_interval_fraction,
    )


def augment_unmasked_view(
    peak_mz: torch.Tensor,
    peak_intensity: torch.Tensor,
    peak_valid_mask: torch.Tensor,
    *,
    mz_jitter_std: float,
    intensity_jitter_std: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    view_valid = peak_valid_mask
    masked = torch.zeros_like(peak_valid_mask)

    mz = torch.zeros_like(peak_mz)
    mz[view_valid] = peak_mz[view_valid] + (
        torch.randn_like(peak_mz[view_valid]) * mz_jitter_std
    )
    mz = torch.clamp(mz, min=0.0, max=1.0)

    intensity = torch.zeros_like(peak_intensity)
    intensity[view_valid] = peak_intensity[view_valid] + (
        torch.randn_like(peak_intensity[view_valid]) * intensity_jitter_std
    )
    intensity = torch.clamp(intensity, min=0.0, max=1.0)

    max_intensity = intensity.max(dim=1, keepdim=True).values.clamp(min=1e-6)
    intensity = intensity / max_intensity
    intensity = torch.where(view_valid, intensity, 0.0)

    return mz, intensity, view_valid, masked


def augment_sigreg_batch(
    batch: dict[str, torch.Tensor],
    *,
    contiguous_mask_fraction: float,
    contiguous_mask_min_len: int,
    mz_jitter_std: float,
    intensity_jitter_std: float,
) -> dict[str, torch.Tensor]:
    peak_mz = batch["peak_mz"]
    peak_intensity = batch["peak_intensity"]
    peak_valid_mask = batch["peak_valid_mask"]
    precursor_mz = batch["precursor_mz"]

    view1_mz, view1_int, view1_valid, view1_masked, view1_masked_fraction, view1_density_interval_fraction = augment_masked_view(
        peak_mz,
        peak_intensity,
        peak_valid_mask,
        contiguous_mask_fraction=contiguous_mask_fraction,
        contiguous_mask_min_len=contiguous_mask_min_len,
        mz_jitter_std=mz_jitter_std,
        intensity_jitter_std=intensity_jitter_std,
    )
    view2_mz, view2_int, view2_valid, view2_masked = augment_unmasked_view(
        peak_mz,
        peak_intensity,
        peak_valid_mask,
        mz_jitter_std=mz_jitter_std,
        intensity_jitter_std=intensity_jitter_std,
    )

    return {
        "fused_mz": torch.cat([view1_mz, view2_mz], dim=0),
        "fused_intensity": torch.cat([view1_int, view2_int], dim=0),
        "fused_precursor_mz": torch.cat([precursor_mz, precursor_mz], dim=0),
        "fused_valid_mask": torch.cat([view1_valid, view2_valid], dim=0),
        "fused_masked_positions": torch.cat([view1_masked, view2_masked], dim=0),
        "view1_masked_fraction": view1_masked_fraction,
        "view1_density_interval_fraction": view1_density_interval_fraction,
    }
