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
PEAK_FILTERING_TOP_INTENSITY = "top_intensity"
PEAK_FILTERING_GROUPED = "grouped"
DEFAULT_PEAK_FILTERING = PEAK_FILTERING_TOP_INTENSITY
DEFAULT_GROUPED_PEAK_SHOULDER_DA = 0.05
DEFAULT_GROUPED_PEAK_ISOTOPE_CHARGES = (1, 2, 3)
COLLISION_ENERGY_MAX = 100.0
PRECURSOR_CHARGE_MAX = 21.0
ASSUMED_PRECURSOR_CHARGE = 1.0
SPECTRUM_METADATA_KEYS = (
    "collision_energy",
    "charge",
)

_SIRIUS_MZ_ISOTOPE_ERROR_DA = 0.002
_SIRIUS_ISOTOPE_RANGES_DA = (
    (
        0.99664664 - _SIRIUS_MZ_ISOTOPE_ERROR_DA,
        1.00342764 + _SIRIUS_MZ_ISOTOPE_ERROR_DA,
    ),
    (
        1.99653883209004 - _SIRIUS_MZ_ISOTOPE_ERROR_DA,
        2.0067426280592295 + _SIRIUS_MZ_ISOTOPE_ERROR_DA,
    ),
    (
        2.9950584 - _SIRIUS_MZ_ISOTOPE_ERROR_DA,
        3.00995027 + _SIRIUS_MZ_ISOTOPE_ERROR_DA,
    ),
    (
        3.99359037 - _SIRIUS_MZ_ISOTOPE_ERROR_DA,
        4.01300058 + _SIRIUS_MZ_ISOTOPE_ERROR_DA,
    ),
    (
        4.9937908 - _SIRIUS_MZ_ISOTOPE_ERROR_DA,
        5.01572941 + _SIRIUS_MZ_ISOTOPE_ERROR_DA,
    ),
)


def _find_group_parent(parent: list[int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union_group_parent(parent: list[int], a: int, b: int) -> None:
    root_a = _find_group_parent(parent, a)
    root_b = _find_group_parent(parent, b)
    if root_a != root_b:
        parent[root_b] = root_a


def _group_peak_indices_numpy(
    mz: np.ndarray,
    *,
    shoulder_da: float,
    isotope_charges: tuple[int, ...],
) -> list[np.ndarray]:
    parent = list(range(len(mz)))
    search = mz.searchsorted
    for i, value in enumerate(mz):
        shoulder_end = int(search(value + shoulder_da, side="right"))
        for j in range(i + 1, shoulder_end):
            _union_group_parent(parent, i, j)
        for charge in isotope_charges:
            if charge > 1 and value / charge < 100.0:
                continue
            inv_charge = 1.0 / charge
            for lo_delta, hi_delta in _SIRIUS_ISOTOPE_RANGES_DA:
                lo = int(search(value + lo_delta * inv_charge, side="left"))
                hi = int(search(value + hi_delta * inv_charge, side="right"))
                start = max(i + 1, lo)
                if hi <= start:
                    break
                for j in range(start, hi):
                    _union_group_parent(parent, i, j)

    groups_by_root: dict[int, list[int]] = {}
    for i in range(len(mz)):
        root = _find_group_parent(parent, i)
        if root not in groups_by_root:
            groups_by_root[root] = []
        groups_by_root[root].append(i)
    return [np.asarray(indices, dtype=np.int64) for indices in groups_by_root.values()]


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


def _select_grouped_peaks_numpy(
    mz: np.ndarray,
    intensity: np.ndarray,
    *,
    num_peaks: int,
    shoulder_da: float,
    isotope_charges: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    selected_mz = np.zeros((mz.shape[0], num_peaks), dtype=mz.dtype)
    selected_intensity = np.zeros((intensity.shape[0], num_peaks), dtype=intensity.dtype)
    for row_idx in range(mz.shape[0]):
        valid = intensity[row_idx] > 0
        row_mz = mz[row_idx, valid]
        row_intensity = intensity[row_idx, valid]
        if row_mz.size == 0:
            continue
        mz_order = np.argsort(row_mz, kind="stable")
        row_mz = row_mz[mz_order]
        row_intensity = row_intensity[mz_order]
        groups = _group_peak_indices_numpy(
            row_mz,
            shoulder_da=shoulder_da,
            isotope_charges=isotope_charges,
        )
        representatives = np.asarray(
            [group[row_intensity[group].argmax()] for group in groups],
            dtype=np.int64,
        )
        order = np.argsort(-row_intensity[representatives], kind="stable")
        representatives = representatives[order[:num_peaks]]
        selected_mz[row_idx, : representatives.size] = row_mz[representatives]
        selected_intensity[row_idx, : representatives.size] = row_intensity[
            representatives
        ]
    return selected_mz, selected_intensity


def _group_peak_indices_torch(
    mz: torch.Tensor,
    *,
    shoulder_da: float,
    isotope_charges: tuple[int, ...],
) -> list[list[int]]:
    parent = list(range(mz.numel()))
    for i in range(mz.numel()):
        value = float(mz[i])
        shoulder_end = int(
            torch.searchsorted(mz, value + shoulder_da, right=True).item()
        )
        for j in range(i + 1, shoulder_end):
            _union_group_parent(parent, i, j)
        for charge in isotope_charges:
            if charge > 1 and value / charge < 100.0:
                continue
            pattern_edges: list[int] = []
            for lo_delta, hi_delta in _SIRIUS_ISOTOPE_RANGES_DA:
                lo = int(
                    torch.searchsorted(
                        mz,
                        value + lo_delta / charge,
                        right=False,
                    ).item()
                )
                hi = int(
                    torch.searchsorted(
                        mz,
                        value + hi_delta / charge,
                        right=True,
                    ).item()
                )
                if hi <= max(i + 1, lo):
                    break
                pattern_edges.extend(range(max(i + 1, lo), hi))
            for j in pattern_edges:
                _union_group_parent(parent, i, j)

    groups_by_root: dict[int, list[int]] = {}
    for i in range(mz.numel()):
        root = _find_group_parent(parent, i)
        if root not in groups_by_root:
            groups_by_root[root] = []
        groups_by_root[root].append(i)
    return list(groups_by_root.values())


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


def _select_grouped_peaks_torch(
    mz: torch.Tensor,
    intensity: torch.Tensor,
    *,
    num_peaks: int,
    shoulder_da: float,
    isotope_charges: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    selected_mz = torch.zeros(
        mz.shape[0],
        num_peaks,
        dtype=mz.dtype,
        device=mz.device,
    )
    selected_intensity = torch.zeros(
        intensity.shape[0],
        num_peaks,
        dtype=intensity.dtype,
        device=intensity.device,
    )
    for row_idx in range(mz.shape[0]):
        valid = intensity[row_idx] > 0
        row_mz = mz[row_idx, valid]
        row_intensity = intensity[row_idx, valid]
        if row_mz.numel() == 0:
            continue
        mz_order = torch.argsort(row_mz, stable=True)
        row_mz = row_mz[mz_order]
        row_intensity = row_intensity[mz_order]
        groups = _group_peak_indices_torch(
            row_mz,
            shoulder_da=shoulder_da,
            isotope_charges=isotope_charges,
        )
        representatives = []
        for group in groups:
            group_idx = torch.tensor(group, dtype=torch.long, device=mz.device)
            representative = group_idx[torch.argmax(row_intensity[group_idx])]
            representatives.append(representative)
        representative_idx = torch.stack(representatives)
        group_scores = row_intensity[representative_idx]
        order = torch.argsort(group_scores, descending=True, stable=True)[:num_peaks]
        representative_idx = representative_idx[order]
        selected_mz[row_idx, : representative_idx.numel()] = row_mz[representative_idx]
        selected_intensity[row_idx, : representative_idx.numel()] = row_intensity[
            representative_idx
        ]
    return selected_mz, selected_intensity


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
    peak_filtering: str = DEFAULT_PEAK_FILTERING,
    grouped_peak_shoulder_da: float = DEFAULT_GROUPED_PEAK_SHOULDER_DA,
    grouped_peak_isotope_charges: tuple[int, ...] = (
        DEFAULT_GROUPED_PEAK_ISOTOPE_CHARGES
    ),
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

    if peak_filtering == PEAK_FILTERING_GROUPED:
        mz, intensity = _select_grouped_peaks_numpy(
            mz,
            intensity,
            num_peaks=num_peaks,
            shoulder_da=grouped_peak_shoulder_da,
            isotope_charges=grouped_peak_isotope_charges,
        )
    else:
        mz, intensity = _select_top_intensity_numpy(
            mz,
            intensity,
            num_peaks=num_peaks,
        )

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
    peak_filtering: str = DEFAULT_PEAK_FILTERING,
    grouped_peak_shoulder_da: float = DEFAULT_GROUPED_PEAK_SHOULDER_DA,
    grouped_peak_isotope_charges: tuple[int, ...] = (
        DEFAULT_GROUPED_PEAK_ISOTOPE_CHARGES
    ),
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

    if peak_filtering == PEAK_FILTERING_GROUPED:
        mz, intensity = _select_grouped_peaks_torch(
            mz,
            intensity,
            num_peaks=num_peaks,
            shoulder_da=grouped_peak_shoulder_da,
            isotope_charges=grouped_peak_isotope_charges,
        )
    else:
        mz, intensity = _select_top_intensity_torch(
            mz,
            intensity,
            num_peaks=num_peaks,
        )

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
