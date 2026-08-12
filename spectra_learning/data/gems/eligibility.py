from __future__ import annotations

import numpy as np

from spectra_learning.data.spectra import (
    DEFAULT_MIN_PEAK_INTENSITY,
    DEFAULT_PRECURSOR_PEAK_EXCLUSION_WINDOW_DA,
    PEAK_INTENSITY_NORMALIZATION_FLOOR,
    PEAK_MZ_MAX,
    PEAK_MZ_MIN,
    usable_peak_counts_numpy,
)

MASSIVE_V2_ELIGIBILITY_VERSION = "bounded_precursor_rt_usable_peaks_ms2_v5"
MASSIVE_V2_REQUIRED_MS_LEVEL = 2
MASSIVE_V2_MIN_PRECURSOR_MZ = 1.0
MASSIVE_V2_MAX_PRECURSOR_MZ = 1000.0
MASSIVE_V2_MIN_RETENTION_TIME_EXCLUSIVE = 0.0
MASSIVE_V2_MIN_RELATIVE_PEAK_INTENSITY = DEFAULT_MIN_PEAK_INTENSITY
MASSIVE_V2_PRECURSOR_PEAK_EXCLUSION_WINDOW_DA = (
    DEFAULT_PRECURSOR_PEAK_EXCLUSION_WINDOW_DA
)
MASSIVE_V2_MIN_USABLE_PEAKS = 2


def massive_v2_eligibility_contract() -> dict[str, object]:
    return {
        "version": MASSIVE_V2_ELIGIBILITY_VERSION,
        "ms_level": MASSIVE_V2_REQUIRED_MS_LEVEL,
        "min_precursor_mz": MASSIVE_V2_MIN_PRECURSOR_MZ,
        "max_precursor_mz": MASSIVE_V2_MAX_PRECURSOR_MZ,
        "min_retention_time_exclusive": (
            MASSIVE_V2_MIN_RETENTION_TIME_EXCLUSIVE
        ),
        "requires_finite_precursor_mz": True,
        "requires_finite_retention_time": True,
        "min_peak_mz": PEAK_MZ_MIN,
        "max_peak_mz": PEAK_MZ_MAX,
        "min_relative_peak_intensity": (
            MASSIVE_V2_MIN_RELATIVE_PEAK_INTENSITY
        ),
        "peak_intensity_normalization_max_floor": (
            PEAK_INTENSITY_NORMALIZATION_FLOOR
        ),
        "precursor_peak_exclusion_window_da": (
            MASSIVE_V2_PRECURSOR_PEAK_EXCLUSION_WINDOW_DA
        ),
        "min_usable_peaks": MASSIVE_V2_MIN_USABLE_PEAKS,
    }


def massive_v2_training_eligibility_numpy(
    spectra: np.ndarray,
    precursor_mz: np.ndarray,
    retention_time: np.ndarray,
) -> np.ndarray:
    usable_peak_counts = usable_peak_counts_numpy(
        spectra,
        precursor_mz,
        min_peak_intensity=MASSIVE_V2_MIN_RELATIVE_PEAK_INTENSITY,
        peak_drop_min_intensity=MASSIVE_V2_MIN_RELATIVE_PEAK_INTENSITY,
        precursor_peak_exclusion_window_da=(
            MASSIVE_V2_PRECURSOR_PEAK_EXCLUSION_WINDOW_DA
        ),
    )
    return (
        np.isfinite(precursor_mz)
        & (precursor_mz >= MASSIVE_V2_MIN_PRECURSOR_MZ)
        & (precursor_mz <= MASSIVE_V2_MAX_PRECURSOR_MZ)
        & np.isfinite(retention_time)
        & (retention_time > MASSIVE_V2_MIN_RETENTION_TIME_EXCLUSIVE)
        & (usable_peak_counts >= MASSIVE_V2_MIN_USABLE_PEAKS)
    )
