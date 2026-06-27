from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from ml_collections import config_dict

from spectra_learning.data.gems.intensity_aware import AWARE_MIXED_MASK_CONFIG
from spectra_learning.data.spectra import (
    DEFAULT_MAX_PRECURSOR_MZ,
    DEFAULT_GROUPED_PEAK_ISOTOPE_CHARGES,
    DEFAULT_GROUPED_PEAK_SHOULDER_DA,
    DEFAULT_MIN_PEAK_INTENSITY,
    DEFAULT_PEAK_FILTERING,
    DEFAULT_PRECURSOR_PEAK_EXCLUSION_WINDOW_DA,
)

DEFAULT_BATCH_SIZE = 512
DEFAULT_ARTIFACT_DIR = Path("data/gems_artifacts")
DEFAULT_GEMS_HDF5_REPO_ID = "novogaia/massive-v1-ms2-100m-stratified-x16"
DEFAULT_GEMS_HDF5_MANIFEST = "fdataloader_shards.json"
NUM_PEAKS_OUTPUT = 60
GEMS_METADATA_FILENAME = "metadata.json"


def _config_get(config: config_dict.ConfigDict, key: str, default: Any) -> Any:
    return config.get(key, default)


def _config_mask_strategy(value: Any) -> str | tuple[str, ...]:
    if isinstance(value, str):
        return value
    return tuple(str(strategy) for strategy in value)


@dataclass(frozen=True)
class GemsDataConfig:
    artifact_dir: Path
    gems_hdf5_repo_id: str
    gems_hdf5_revision: str
    gems_hdf5_manifest: str
    gems_hdf5_spectrum_dataset: str
    gems_hdf5_precursor_dataset: str
    gems_hdf5_rows_per_block: int
    batch_size: int
    gradient_accumulation_steps: int
    drop_remainder: bool
    max_precursor_mz: float
    min_peak_intensity: float
    peak_drop_min_intensity: float
    peak_filtering: str
    grouped_peak_shoulder_da: float
    grouped_peak_isotope_charges: tuple[int, ...]
    peak_ordering: str
    precursor_peak_exclusion_window_da: float
    jepa_num_target_blocks: int
    jepa_context_fraction: float
    jepa_target_fraction: float
    jepa_block_min_len: int
    jepa_mask_strategy: str | tuple[str, ...]
    jepa_mask_lengths: tuple[int, ...]
    jepa_mask_round_from: int
    jepa_intensity_aware_mask_config: dict[str, float]
    jepa_allow_target_overlap: bool
    num_peaks: int
    dataloader_pin_memory: bool
    dataloader_num_workers: int
    dataloader_prefetch_factor: int
    dataloader_persistent_workers: bool
    dataloader_multiprocessing_context: str
    dataloader_output_format: str

    @classmethod
    def from_config(cls, config: config_dict.ConfigDict) -> "GemsDataConfig":
        artifact_dir = (
            Path(_config_get(config, "artifact_dir", str(DEFAULT_ARTIFACT_DIR)))
            .expanduser()
            .resolve()
        )
        min_peak_intensity = float(
            _config_get(config, "min_peak_intensity", DEFAULT_MIN_PEAK_INTENSITY)
        )
        jepa_mask_lengths = tuple(
            int(length)
            for length in _config_get(config, "jepa_mask_lengths", (1, 2, 4, 8, 16))
        )
        dataloader_num_workers = int(_config_get(config, "dataloader_num_workers", 1))
        return cls(
            artifact_dir=artifact_dir,
            gems_hdf5_repo_id=str(
                _config_get(
                    config,
                    "gems_hdf5_repo_id",
                    DEFAULT_GEMS_HDF5_REPO_ID,
                )
            ).strip(),
            gems_hdf5_revision=str(
                _config_get(config, "gems_hdf5_revision", "main")
            ),
            gems_hdf5_manifest=str(
                _config_get(
                    config,
                    "gems_hdf5_manifest",
                    DEFAULT_GEMS_HDF5_MANIFEST,
                )
            ),
            gems_hdf5_spectrum_dataset=str(
                _config_get(config, "gems_hdf5_spectrum_dataset", "spectrum")
            ),
            gems_hdf5_precursor_dataset=str(
                _config_get(config, "gems_hdf5_precursor_dataset", "precursor_mz")
            ),
            gems_hdf5_rows_per_block=int(
                _config_get(config, "gems_hdf5_rows_per_block", 0)
            ),
            batch_size=int(_config_get(config, "batch_size", DEFAULT_BATCH_SIZE)),
            gradient_accumulation_steps=int(
                _config_get(config, "gradient_accumulation_steps", 1)
            ),
            drop_remainder=bool(_config_get(config, "drop_remainder", True)),
            max_precursor_mz=float(
                _config_get(config, "max_precursor_mz", DEFAULT_MAX_PRECURSOR_MZ)
            ),
            min_peak_intensity=min_peak_intensity,
            peak_drop_min_intensity=float(
                _config_get(config, "peak_drop_min_intensity", min_peak_intensity)
            ),
            peak_filtering=str(
                _config_get(config, "peak_filtering", DEFAULT_PEAK_FILTERING)
            ),
            grouped_peak_shoulder_da=float(
                _config_get(
                    config,
                    "grouped_peak_shoulder_da",
                    DEFAULT_GROUPED_PEAK_SHOULDER_DA,
                )
            ),
            grouped_peak_isotope_charges=tuple(
                int(charge)
                for charge in _config_get(
                    config,
                    "grouped_peak_isotope_charges",
                    DEFAULT_GROUPED_PEAK_ISOTOPE_CHARGES,
                )
            ),
            peak_ordering=str(_config_get(config, "peak_ordering", "mz")),
            precursor_peak_exclusion_window_da=float(
                _config_get(
                    config,
                    "precursor_peak_exclusion_window_da",
                    DEFAULT_PRECURSOR_PEAK_EXCLUSION_WINDOW_DA,
                )
            ),
            jepa_num_target_blocks=int(_config_get(config, "jepa_num_target_blocks", 2)),
            jepa_context_fraction=float(
                _config_get(config, "jepa_context_fraction", 0.5)
            ),
            jepa_target_fraction=float(
                _config_get(config, "jepa_target_fraction", 0.25)
            ),
            jepa_block_min_len=int(_config_get(config, "jepa_block_min_len", 1)),
            jepa_mask_strategy=_config_mask_strategy(
                _config_get(config, "jepa_mask_strategy", "contiguous")
            ),
            jepa_mask_lengths=jepa_mask_lengths,
            jepa_mask_round_from=int(
                _config_get(config, "jepa_mask_round_from", len(jepa_mask_lengths))
            ),
            jepa_intensity_aware_mask_config={
                key: float(_config_get(config, f"jepa_intensity_aware_{key}", value))
                for key, value in AWARE_MIXED_MASK_CONFIG.items()
            },
            jepa_allow_target_overlap=bool(
                _config_get(config, "jepa_allow_target_overlap", False)
            ),
            num_peaks=int(_config_get(config, "num_peaks", NUM_PEAKS_OUTPUT)),
            dataloader_pin_memory=bool(
                _config_get(config, "dataloader_pin_memory", torch.cuda.is_available())
            ),
            dataloader_num_workers=dataloader_num_workers,
            dataloader_prefetch_factor=int(
                _config_get(config, "dataloader_prefetch_factor", 2)
            ),
            dataloader_persistent_workers=bool(
                _config_get(
                    config,
                    "dataloader_persistent_workers",
                    dataloader_num_workers > 0,
                )
            ),
            dataloader_multiprocessing_context=str(
                _config_get(config, "dataloader_multiprocessing_context", "")
            ),
            dataloader_output_format=str(
                _config_get(config, "dataloader_output_format", "torch")
            ),
        )
