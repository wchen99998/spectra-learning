from __future__ import annotations

import re

import numpy as np
import torch
import torch.nn.functional as F

NUM_PEAKS_INPUT = 128
DEFAULT_NUM_PEAKS = 60
PEAK_MZ_MIN = 20.0
PEAK_MZ_MAX = 1000.0
DEFAULT_MIN_PEAK_INTENSITY = 1e-4
PEAK_INTENSITY_NORMALIZATION_FLOOR = 1e-8
DEFAULT_MIN_PRECURSOR_MZ = 1.0
DEFAULT_MAX_PRECURSOR_MZ = 1000.0
DEFAULT_PRECURSOR_PEAK_EXCLUSION_WINDOW_DA = 0.0
COLLISION_ENERGY_MAX = 100.0
PRECURSOR_CHARGE_MAX = 21.0
ASSUMED_PRECURSOR_CHARGE = 1.0
SPECTRUM_METADATA_KEYS = (
    "collision_energy",
    "charge",
)
_PRECURSOR_CHARGE_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")


def canonicalize_precursor_charge_numpy(charge: np.ndarray) -> np.ndarray:
    charge = np.abs(charge.astype(np.float32, copy=False))
    return np.where(
        np.isfinite(charge) & (charge > 0.0),
        charge,
        np.float32(ASSUMED_PRECURSOR_CHARGE),
    )


def canonicalize_precursor_charge_torch(charge: torch.Tensor) -> torch.Tensor:
    charge = charge.to(dtype=torch.float32).abs()
    return torch.where(
        torch.isfinite(charge) & (charge > 0.0),
        charge,
        torch.full_like(charge, ASSUMED_PRECURSOR_CHARGE),
    )


def parse_precursor_charge(value: object) -> float:
    match = _PRECURSOR_CHARGE_PATTERN.search(str(value))
    charge = abs(float(match.group())) if match is not None else ASSUMED_PRECURSOR_CHARGE
    return (
        charge
        if np.isfinite(charge) and charge > 0.0
        else ASSUMED_PRECURSOR_CHARGE
    )


def spectra_from_peak_lists(
    mz_lists: list[list[float]],
    intensity_lists: list[list[float]],
) -> np.ndarray:
    spectra = np.zeros((len(mz_lists), 2, NUM_PEAKS_INPUT), dtype=np.float32)
    for row, (mz, intensity) in enumerate(
        zip(mz_lists, intensity_lists, strict=True)
    ):
        num_peaks = min(len(mz), NUM_PEAKS_INPUT)
        spectra[row, 0, :num_peaks] = np.asarray(mz[:num_peaks], dtype=np.float32)
        spectra[row, 1, :num_peaks] = np.asarray(
            intensity[:num_peaks],
            dtype=np.float32,
        )
    return spectra


def _select_top_intensity_numpy(
    mz: np.ndarray,
    intensity: np.ndarray,
    *,
    num_peaks: int,
) -> tuple[np.ndarray, np.ndarray]:
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
    return mz, intensity


def _select_top_intensity_torch(
    mz: torch.Tensor,
    intensity: torch.Tensor,
    *,
    num_peaks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mz.shape[1] > num_peaks:
        intensity, topk_idx = torch.topk(intensity, k=num_peaks, dim=1, sorted=True)
        mz = torch.gather(mz, 1, topk_idx)
    elif mz.shape[1] < num_peaks:
        pad = num_peaks - mz.shape[1]
        mz = F.pad(mz, (0, pad))
        intensity = F.pad(intensity, (0, pad))
    return mz, intensity


def _normalize_peak_intensity_numpy(intensity: np.ndarray) -> np.ndarray:
    max_intensity = np.maximum(
        intensity.max(axis=1, keepdims=True),
        PEAK_INTENSITY_NORMALIZATION_FLOOR,
    )
    return intensity / max_intensity


def _usable_peak_mask_numpy(
    mz: np.ndarray,
    normalized_intensity: np.ndarray,
    precursor_mz: np.ndarray,
    *,
    min_peak_intensity: float,
    peak_drop_min_intensity: float,
    precursor_peak_exclusion_window_da: float,
) -> np.ndarray:
    intensity_threshold = max(min_peak_intensity, peak_drop_min_intensity)
    window = precursor_peak_exclusion_window_da
    precursor_upper = precursor_mz[:, None] - window
    return (
        (mz >= PEAK_MZ_MIN)
        & (mz <= PEAK_MZ_MAX)
        & (normalized_intensity >= intensity_threshold)
        & ((window <= 0.0) | (mz <= precursor_upper))
    )


def usable_peak_counts_numpy(
    spectra: np.ndarray,
    precursor_mz: np.ndarray,
    *,
    min_peak_intensity: float = DEFAULT_MIN_PEAK_INTENSITY,
    peak_drop_min_intensity: float = DEFAULT_MIN_PEAK_INTENSITY,
    precursor_peak_exclusion_window_da: float = (
        DEFAULT_PRECURSOR_PEAK_EXCLUSION_WINDOW_DA
    ),
) -> np.ndarray:
    mz = spectra[:, 0, :].astype(np.float32, copy=False)
    intensity = spectra[:, 1, :].astype(np.float32, copy=False)
    normalized_intensity = _normalize_peak_intensity_numpy(intensity)
    usable = _usable_peak_mask_numpy(
        mz,
        normalized_intensity,
        precursor_mz,
        min_peak_intensity=min_peak_intensity,
        peak_drop_min_intensity=peak_drop_min_intensity,
        precursor_peak_exclusion_window_da=(
            precursor_peak_exclusion_window_da
        ),
    )
    return usable.sum(axis=1)


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
    if peak_ordering not in {"mz", "intensity"}:
        raise ValueError(f"Unknown peak_ordering: {peak_ordering}")
    mz = spectra[:, 0, :].astype(np.float32, copy=False)
    intensity = spectra[:, 1, :].astype(np.float32, copy=False)
    relative_intensity = _normalize_peak_intensity_numpy(intensity)
    keep = _usable_peak_mask_numpy(
        mz,
        relative_intensity,
        precursor_mz,
        min_peak_intensity=min_peak_intensity,
        peak_drop_min_intensity=peak_drop_min_intensity,
        precursor_peak_exclusion_window_da=(
            precursor_peak_exclusion_window_da
        ),
    )
    mz = np.where(keep, mz, 0.0)
    intensity = np.where(keep, intensity, 0.0)

    mz, intensity = _select_top_intensity_numpy(
        mz,
        intensity,
        num_peaks=num_peaks,
    )

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
    valid[:, 0] |= ~valid.any(axis=1)
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
    if peak_ordering not in {"mz", "intensity"}:
        raise ValueError(f"Unknown peak_ordering: {peak_ordering}")
    input_max_intensity = torch.clamp(
        intensity.amax(dim=1, keepdim=True),
        min=PEAK_INTENSITY_NORMALIZATION_FLOOR,
    )
    relative_intensity = intensity / input_max_intensity
    intensity_threshold = max(min_peak_intensity, peak_drop_min_intensity)
    window = precursor_peak_exclusion_window_da
    precursor_upper = precursor_mz[:, None] - window
    keep = (
        (mz >= PEAK_MZ_MIN)
        & (mz <= PEAK_MZ_MAX)
        & (relative_intensity >= intensity_threshold)
        & ((window <= 0.0) | (mz <= precursor_upper))
    )
    mz = torch.where(keep, mz, torch.zeros_like(mz))
    intensity = torch.where(keep, intensity, torch.zeros_like(intensity))

    mz, intensity = _select_top_intensity_torch(
        mz,
        intensity,
        num_peaks=num_peaks,
    )

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
    valid[:, 0] |= ~valid.any(dim=1)
    precursor = torch.clamp(precursor_mz, 0.0, max_precursor_mz) / max_precursor_mz
    return {
        "peak_mz": mz / PEAK_MZ_MAX,
        "peak_intensity": intensity,
        "peak_valid_mask": valid,
        "precursor_mz": precursor,
    }
