from __future__ import annotations

from typing import Any

from spectra_learning.data.spectra import (
    COLLISION_ENERGY_MAX,
    DEFAULT_GROUPED_PEAK_ISOTOPE_CHARGES,
    DEFAULT_GROUPED_PEAK_SHOULDER_DA,
    DEFAULT_MAX_PRECURSOR_MZ,
    DEFAULT_MIN_PRECURSOR_MZ,
    DEFAULT_MIN_PEAK_INTENSITY,
    DEFAULT_NUM_PEAKS,
    DEFAULT_PEAK_FILTERING,
    DEFAULT_PRECURSOR_PEAK_EXCLUSION_WINDOW_DA,
    NUM_PEAKS_INPUT,
    PEAK_FILTERING_GROUPED,
    PEAK_FILTERING_TOP_INTENSITY,
    PEAK_MZ_MAX,
    PEAK_MZ_MIN,
)

PEAK_PREPROCESSING_CONTRACT_VERSION = 1
_LOCAL_PROVENANCE_KEYS = {
    "artifact_dir",
    "gems_dir",
    "gems_manifest",
}


def peak_preprocessing_contract(config: Any) -> dict[str, Any]:
    min_peak_intensity = float(
        config.get("min_peak_intensity", DEFAULT_MIN_PEAK_INTENSITY)
    )
    peak_filtering = str(config.get("peak_filtering", DEFAULT_PEAK_FILTERING))
    if peak_filtering not in {
        PEAK_FILTERING_TOP_INTENSITY,
        PEAK_FILTERING_GROUPED,
    }:
        raise ValueError(f"Unknown peak_filtering: {peak_filtering}")
    peak_ordering = str(config.get("peak_ordering", "mz"))
    if peak_ordering not in {"mz", "intensity"}:
        raise ValueError(f"Unknown peak_ordering: {peak_ordering}")
    return {
        "version": PEAK_PREPROCESSING_CONTRACT_VERSION,
        "num_peaks_input": NUM_PEAKS_INPUT,
        "num_peaks": int(config.get("num_peaks", DEFAULT_NUM_PEAKS)),
        "peak_mz_min": PEAK_MZ_MIN,
        "peak_mz_max": PEAK_MZ_MAX,
        "min_precursor_mz": float(
            config.get("min_precursor_mz", DEFAULT_MIN_PRECURSOR_MZ)
        ),
        "max_precursor_mz": float(
            config.get("max_precursor_mz", DEFAULT_MAX_PRECURSOR_MZ)
        ),
        "min_peak_intensity": min_peak_intensity,
        "peak_drop_min_intensity": float(
            config.get("peak_drop_min_intensity", min_peak_intensity)
        ),
        "peak_filtering": peak_filtering,
        "grouped_peak_shoulder_da": float(
            config.get(
                "grouped_peak_shoulder_da",
                DEFAULT_GROUPED_PEAK_SHOULDER_DA,
            )
        ),
        "grouped_peak_isotope_charges": [
            int(charge)
            for charge in config.get(
                "grouped_peak_isotope_charges",
                DEFAULT_GROUPED_PEAK_ISOTOPE_CHARGES,
            )
        ],
        "peak_ordering": peak_ordering,
        "precursor_peak_exclusion_window_da": float(
            config.get(
                "precursor_peak_exclusion_window_da",
                DEFAULT_PRECURSOR_PEAK_EXCLUSION_WINDOW_DA,
            )
        ),
        "intensity_normalization": (
            "base_peak_before_threshold_and_after_peak_selection"
        ),
        "empty_spectrum": "zero_placeholder_with_first_slot_valid",
        "collision_energy_max": COLLISION_ENERGY_MAX,
        "charge_normalization": (
            "absolute; missing_nonfinite_or_zero_to_1; "
            "model_clips_0_21_then_divides_by_21"
        ),
    }


def validate_peak_preprocessing_contract(
    checkpoint: dict[str, Any],
    config: Any,
) -> None:
    expected = peak_preprocessing_contract(config)
    actual = checkpoint["peak_preprocessing"]
    if actual != expected:
        raise ValueError(
            "Checkpoint peak preprocessing does not match the requested config: "
            f"checkpoint={actual}, requested={expected}"
        )


def data_provenance_contract(info: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in info.items()
        if key not in _LOCAL_PROVENANCE_KEYS
        and not key.endswith(("_cache_dir", "_source_dir"))
    }


def validate_data_provenance_contract(
    checkpoint: dict[str, Any],
    info: dict[str, Any],
) -> None:
    expected = data_provenance_contract(info)
    actual = checkpoint["data_provenance"]
    if actual != expected:
        raise ValueError(
            "Checkpoint data provenance does not match the requested data: "
            f"checkpoint={actual}, requested={expected}"
        )
