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
PEAK_GROUP_PADDING_ID = -1
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
    max_intensity = spectra[:, 1].max(axis=1, keepdims=True)
    np.divide(
        spectra[:, 1],
        np.maximum(max_intensity, 1e-8),
        out=spectra[:, 1],
    )
    return spectra


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


def _groups_from_parent_numpy(parent: list[int]) -> list[np.ndarray]:
    groups_by_root: dict[int, list[int]] = {}
    for i in range(len(parent)):
        root = _find_group_parent(parent, i)
        if root not in groups_by_root:
            groups_by_root[root] = []
        groups_by_root[root].append(i)
    return [np.asarray(indices, dtype=np.int64) for indices in groups_by_root.values()]


def _group_shoulder_peak_indices_numpy(
    mz: np.ndarray,
    *,
    shoulder_da: float,
) -> list[np.ndarray]:
    parent = list(range(len(mz)))
    search = mz.searchsorted
    for i, value in enumerate(mz):
        shoulder_end = int(search(value + shoulder_da, side="right"))
        for j in range(i + 1, shoulder_end):
            _union_group_parent(parent, i, j)
    return _groups_from_parent_numpy(parent)


def _group_isotope_peak_indices_numpy(
    mz: np.ndarray,
    *,
    isotope_charges: tuple[int, ...],
) -> list[np.ndarray]:
    parent = list(range(len(mz)))
    search = mz.searchsorted
    for i, value in enumerate(mz):
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
    return _groups_from_parent_numpy(parent)


def _collapse_shoulder_peaks_numpy(
    mz: np.ndarray,
    intensity: np.ndarray,
    *,
    shoulder_da: float,
) -> tuple[np.ndarray, np.ndarray]:
    groups = _group_shoulder_peak_indices_numpy(mz, shoulder_da=shoulder_da)
    representatives = np.asarray(
        [group[intensity[group].argmax()] for group in groups],
        dtype=np.int64,
    )
    order = np.argsort(mz[representatives], kind="stable")
    representatives = representatives[order]
    return mz[representatives], intensity[representatives]


def _select_top_intensity_numpy(
    mz: np.ndarray,
    intensity: np.ndarray,
    *,
    num_peaks: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    group_id = np.broadcast_to(
        np.arange(mz.shape[1], dtype=np.int32),
        mz.shape,
    ).copy()
    group_id = np.where(intensity > 0, group_id, PEAK_GROUP_PADDING_ID)
    return mz, intensity, group_id


def _select_grouped_peaks_numpy(
    mz: np.ndarray,
    intensity: np.ndarray,
    *,
    num_peaks: int,
    shoulder_da: float,
    isotope_charges: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected_mz = np.zeros((mz.shape[0], num_peaks), dtype=mz.dtype)
    selected_intensity = np.zeros((intensity.shape[0], num_peaks), dtype=intensity.dtype)
    selected_group_id = np.full(
        (mz.shape[0], num_peaks),
        PEAK_GROUP_PADDING_ID,
        dtype=np.int32,
    )
    for row_idx in range(mz.shape[0]):
        valid = intensity[row_idx] > 0
        row_mz = mz[row_idx, valid]
        row_intensity = intensity[row_idx, valid]
        if row_mz.size == 0:
            continue
        mz_order = np.argsort(row_mz, kind="stable")
        row_mz = row_mz[mz_order]
        row_intensity = row_intensity[mz_order]
        row_mz, row_intensity = _collapse_shoulder_peaks_numpy(
            row_mz,
            row_intensity,
            shoulder_da=shoulder_da,
        )
        groups = _group_isotope_peak_indices_numpy(
            row_mz,
            isotope_charges=isotope_charges,
        )
        group_scores = np.asarray(
            [row_intensity[group].max() for group in groups],
            dtype=row_intensity.dtype,
        )
        group_order = np.argsort(-group_scores, kind="stable")
        offset = 0
        output_group_id = 0
        for group_idx in group_order:
            group = groups[int(group_idx)]
            if offset + group.size > num_peaks:
                continue
            order = np.argsort(row_mz[group], kind="stable")
            peak_indices = group[order]
            end = offset + peak_indices.size
            selected_mz[row_idx, offset:end] = row_mz[peak_indices]
            selected_intensity[row_idx, offset:end] = row_intensity[peak_indices]
            selected_group_id[row_idx, offset:end] = output_group_id
            offset = end
            output_group_id += 1
    return selected_mz, selected_intensity, selected_group_id


def _select_top_intensity_torch(
    mz: torch.Tensor,
    intensity: torch.Tensor,
    *,
    num_peaks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mz.shape[1] > num_peaks:
        intensity, topk_idx = torch.topk(intensity, k=num_peaks, dim=1, sorted=True)
        mz = torch.gather(mz, 1, topk_idx)
    elif mz.shape[1] < num_peaks:
        pad = num_peaks - mz.shape[1]
        mz = F.pad(mz, (0, pad))
        intensity = F.pad(intensity, (0, pad))
    group_id = torch.arange(num_peaks, dtype=torch.int32, device=mz.device).expand(
        mz.shape[0],
        num_peaks,
    )
    group_id = torch.where(
        intensity > 0,
        group_id,
        torch.full_like(group_id, PEAK_GROUP_PADDING_ID),
    )
    return mz, intensity, group_id


def _select_grouped_peaks_torch(
    mz: torch.Tensor,
    intensity: torch.Tensor,
    *,
    num_peaks: int,
    shoulder_da: float,
    isotope_charges: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert mz.device.type == "cpu" and intensity.device.type == "cpu", (
        "grouped peak filtering runs on CPU before device transfer"
    )
    assert mz.dtype == torch.float32 and intensity.dtype == torch.float32, (
        "grouped peak filtering expects float32 spectra"
    )
    selected_mz, selected_intensity, selected_group_id = _select_grouped_peaks_numpy(
        mz.numpy(),
        intensity.numpy(),
        num_peaks=num_peaks,
        shoulder_da=shoulder_da,
        isotope_charges=isotope_charges,
    )
    return (
        torch.from_numpy(selected_mz),
        torch.from_numpy(selected_intensity),
        torch.from_numpy(selected_group_id),
    )


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
        mz, intensity, group_id = _select_grouped_peaks_numpy(
            mz,
            intensity,
            num_peaks=num_peaks,
            shoulder_da=grouped_peak_shoulder_da,
            isotope_charges=grouped_peak_isotope_charges,
        )
    else:
        mz, intensity, group_id = _select_top_intensity_numpy(
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
    group_id = np.take_along_axis(group_id, order, axis=1)
    mz = np.where(valid, mz, 0.0)
    intensity = np.where(valid, intensity, 0.0)
    group_id = np.where(valid, group_id, PEAK_GROUP_PADDING_ID)
    precursor = (
        np.clip(precursor_mz, 0.0, max_precursor_mz).astype(np.float32)
        / max_precursor_mz
    )
    return {
        "peak_mz": (mz / PEAK_MZ_MAX).astype(np.float32),
        "peak_intensity": intensity.astype(np.float32),
        "peak_valid_mask": valid.astype(bool, copy=False),
        "peak_group_id": group_id.astype(np.int32, copy=False),
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
        mz, intensity, group_id = _select_grouped_peaks_torch(
            mz,
            intensity,
            num_peaks=num_peaks,
            shoulder_da=grouped_peak_shoulder_da,
            isotope_charges=grouped_peak_isotope_charges,
        )
    else:
        mz, intensity, group_id = _select_top_intensity_torch(
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
    group_id = torch.gather(group_id, 1, order)
    mz = torch.where(valid, mz, torch.zeros_like(mz))
    intensity = torch.where(valid, intensity, torch.zeros_like(intensity))
    group_id = torch.where(
        valid,
        group_id,
        torch.full_like(group_id, PEAK_GROUP_PADDING_ID),
    )
    precursor = torch.clamp(precursor_mz, 0.0, max_precursor_mz) / max_precursor_mz
    return {
        "peak_mz": mz / PEAK_MZ_MAX,
        "peak_intensity": intensity,
        "peak_valid_mask": valid,
        "peak_group_id": group_id,
        "precursor_mz": precursor,
    }
