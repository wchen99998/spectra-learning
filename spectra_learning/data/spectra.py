from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

NUM_PEAKS_INPUT = 128
PEAK_MZ_MIN = 20.0
PEAK_MZ_MAX = 1000.0
DEFAULT_MIN_PEAK_INTENSITY = 1e-4
DEFAULT_MAX_PRECURSOR_MZ = 1000.0
DEFAULT_PRECURSOR_PEAK_EXCLUSION_WINDOW_DA = 0.0


def preprocess_peak_batch_numpy(
    spectra: np.ndarray,
    precursor_mz: np.ndarray,
    *,
    num_peaks: int,
    peak_drop_min_intensity: float,
    peak_ordering: str,
    max_precursor_mz: float,
    precursor_peak_exclusion_window_da: float = (
        DEFAULT_PRECURSOR_PEAK_EXCLUSION_WINDOW_DA
    ),
    min_peak_intensity: float = DEFAULT_MIN_PEAK_INTENSITY,
) -> dict[str, np.ndarray]:
    mz = spectra[:, 0, :].astype(np.float32, copy=False)
    intensity = spectra[:, 1, :].astype(np.float32, copy=False)
    intensity_threshold = max(min_peak_intensity, peak_drop_min_intensity)
    window = precursor_peak_exclusion_window_da
    precursor_upper = precursor_mz[:, None] - window
    keep = (
        (mz >= PEAK_MZ_MIN)
        & (mz <= PEAK_MZ_MAX)
        & (intensity >= intensity_threshold)
        & ((window <= 0.0) | (mz <= precursor_upper))
    )
    mz = np.where(keep, mz, 0.0)
    intensity = np.where(keep, intensity, 0.0)

    if mz.shape[1] > num_peaks:
        topk_idx = np.argpartition(-intensity, kth=num_peaks - 1, axis=1)[:, :num_peaks]
        rows = np.arange(mz.shape[0])[:, None]
        mz = mz[rows, topk_idx]
        intensity = intensity[rows, topk_idx]
        order = np.argsort(-intensity, axis=1, kind="stable")
        mz = np.take_along_axis(mz, order, axis=1)
        intensity = np.take_along_axis(intensity, order, axis=1)
    elif mz.shape[1] < num_peaks:
        pad = num_peaks - mz.shape[1]
        mz = np.pad(mz, ((0, 0), (0, pad)))
        intensity = np.pad(intensity, ((0, 0), (0, pad)))

    max_intensity = np.maximum(intensity.max(axis=1, keepdims=True), 1e-8)
    intensity = intensity / max_intensity
    valid = intensity > 0
    if peak_ordering == "mz":
        sort_key = np.where(valid, mz, np.inf)
        order = np.argsort(sort_key, axis=1, kind="stable")
    else:
        sort_key = np.where(valid, intensity, -np.inf)
        order = np.argsort(-sort_key, axis=1, kind="stable")
    mz = np.take_along_axis(mz, order, axis=1)
    intensity = np.take_along_axis(intensity, order, axis=1)
    valid = np.take_along_axis(valid, order, axis=1)
    mz = np.where(valid, mz, 0.0)
    intensity = np.where(valid, intensity, 0.0)
    precursor = (
        np.clip(precursor_mz, 0.0, max_precursor_mz).astype(np.float32)
        / max_precursor_mz
    )
    return {
        "peak_mz": (mz / PEAK_MZ_MAX).astype(np.float32),
        "peak_intensity": intensity.astype(np.float32),
        "peak_valid_mask": valid.astype(bool, copy=False),
        "precursor_mz": precursor,
    }


def preprocess_peak_batch_torch(
    mz: torch.Tensor,
    intensity: torch.Tensor,
    precursor_mz: torch.Tensor,
    *,
    num_peaks: int,
    peak_drop_min_intensity: float,
    peak_ordering: str,
    max_precursor_mz: float,
    precursor_peak_exclusion_window_da: float = (
        DEFAULT_PRECURSOR_PEAK_EXCLUSION_WINDOW_DA
    ),
    min_peak_intensity: float = DEFAULT_MIN_PEAK_INTENSITY,
) -> dict[str, torch.Tensor]:
    intensity_threshold = max(min_peak_intensity, peak_drop_min_intensity)
    window = precursor_peak_exclusion_window_da
    precursor_upper = precursor_mz[:, None] - window
    keep = (
        (mz >= PEAK_MZ_MIN)
        & (mz <= PEAK_MZ_MAX)
        & (intensity >= intensity_threshold)
        & ((window <= 0.0) | (mz <= precursor_upper))
    )
    mz = torch.where(keep, mz, torch.zeros_like(mz))
    intensity = torch.where(keep, intensity, torch.zeros_like(intensity))

    if mz.shape[1] > num_peaks:
        intensity, topk_idx = torch.topk(intensity, k=num_peaks, dim=1, sorted=True)
        mz = torch.gather(mz, 1, topk_idx)
    elif mz.shape[1] < num_peaks:
        pad = num_peaks - mz.shape[1]
        mz = F.pad(mz, (0, pad))
        intensity = F.pad(intensity, (0, pad))

    max_intensity = torch.clamp(intensity.amax(dim=1, keepdim=True), min=1e-8)
    intensity = intensity / max_intensity
    valid = intensity > 0
    if peak_ordering == "mz":
        sort_key = torch.where(valid, mz, torch.full_like(mz, float("inf")))
        order = torch.argsort(sort_key, dim=1, stable=True)
    else:
        sort_key = torch.where(
            valid, intensity, torch.full_like(intensity, float("-inf"))
        )
        order = torch.argsort(sort_key, dim=1, descending=True, stable=True)
    mz = torch.gather(mz, 1, order)
    intensity = torch.gather(intensity, 1, order)
    valid = torch.gather(valid, 1, order)
    mz = torch.where(valid, mz, torch.zeros_like(mz))
    intensity = torch.where(valid, intensity, torch.zeros_like(intensity))
    precursor = torch.clamp(precursor_mz, 0.0, max_precursor_mz) / max_precursor_mz
    return {
        "peak_mz": mz / PEAK_MZ_MAX,
        "peak_intensity": intensity,
        "peak_valid_mask": valid,
        "precursor_mz": precursor,
    }
