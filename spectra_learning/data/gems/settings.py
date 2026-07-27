from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from ml_collections import config_dict

from spectra_learning.data.contracts import peak_preprocessing_contract
from spectra_learning.data.gems.intensity_aware import AWARE_MIXED_MASK_CONFIG
from spectra_learning.data.spectra import DEFAULT_NUM_PEAKS

DEFAULT_BATCH_SIZE = 512
DEFAULT_ARTIFACT_DIR = Path("data/gems_artifacts")
DEFAULT_GEMS_HDF5_REPO_ID = "novogaia/massive-v1-ms2-100m-stratified-x16"
DEFAULT_GEMS_HDF5_REVISION = "7ff47061cbde23e4cdd113378dcfb489e86b32c4"
DEFAULT_GEMS_HDF5_MANIFEST = "fdataloader_shards.json"
NUM_PEAKS_OUTPUT = DEFAULT_NUM_PEAKS
GEMS_METADATA_FILENAME = "metadata.json"


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
    gems_hdf5_retention_time_dataset: str
    gems_hdf5_ms_level_dataset: str
    gems_hdf5_rows_per_block: int
    batch_size: int
    gradient_accumulation_steps: int
    drop_remainder: bool
    min_precursor_mz: float
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
            Path(config.get("artifact_dir", str(DEFAULT_ARTIFACT_DIR)))
            .expanduser()
            .resolve()
        )
        preprocessing = peak_preprocessing_contract(config)
        jepa_mask_lengths = tuple(
            int(length)
            for length in config.get("jepa_mask_lengths", (1, 2, 4, 8, 16))
        )
        dataloader_num_workers = int(config.get("dataloader_num_workers", 1))
        return cls(
            artifact_dir=artifact_dir,
            gems_hdf5_repo_id=str(
                config.get(
                    "gems_hdf5_repo_id",
                    DEFAULT_GEMS_HDF5_REPO_ID,
                )
            ).strip(),
            gems_hdf5_revision=str(
                config.get(
                    "gems_hdf5_revision",
                    DEFAULT_GEMS_HDF5_REVISION,
                )
            ),
            gems_hdf5_manifest=str(
                config.get(
                    "gems_hdf5_manifest",
                    DEFAULT_GEMS_HDF5_MANIFEST,
                )
            ),
            gems_hdf5_spectrum_dataset=str(
                config.get("gems_hdf5_spectrum_dataset", "spectrum")
            ),
            gems_hdf5_precursor_dataset=str(
                config.get("gems_hdf5_precursor_dataset", "precursor_mz")
            ),
            gems_hdf5_retention_time_dataset=str(
                config.get("gems_hdf5_retention_time_dataset", "RT")
            ),
            gems_hdf5_ms_level_dataset=str(
                config.get("gems_hdf5_ms_level_dataset", "MS level")
            ),
            gems_hdf5_rows_per_block=int(
                config.get("gems_hdf5_rows_per_block", 0)
            ),
            batch_size=int(config.get("batch_size", DEFAULT_BATCH_SIZE)),
            gradient_accumulation_steps=int(
                config.get("gradient_accumulation_steps", 1)
            ),
            drop_remainder=bool(config.get("drop_remainder", True)),
            min_precursor_mz=preprocessing["min_precursor_mz"],
            max_precursor_mz=preprocessing["max_precursor_mz"],
            min_peak_intensity=preprocessing["min_peak_intensity"],
            peak_drop_min_intensity=preprocessing["peak_drop_min_intensity"],
            peak_filtering=preprocessing["peak_filtering"],
            grouped_peak_shoulder_da=preprocessing["grouped_peak_shoulder_da"],
            grouped_peak_isotope_charges=tuple(
                preprocessing["grouped_peak_isotope_charges"]
            ),
            peak_ordering=preprocessing["peak_ordering"],
            precursor_peak_exclusion_window_da=preprocessing[
                "precursor_peak_exclusion_window_da"
            ],
            jepa_num_target_blocks=int(config.get("jepa_num_target_blocks", 2)),
            jepa_context_fraction=float(
                config.get("jepa_context_fraction", 0.5)
            ),
            jepa_target_fraction=float(
                config.get("jepa_target_fraction", 0.25)
            ),
            jepa_block_min_len=int(config.get("jepa_block_min_len", 1)),
            jepa_mask_strategy=_config_mask_strategy(
                config.get("jepa_mask_strategy", "contiguous")
            ),
            jepa_mask_lengths=jepa_mask_lengths,
            jepa_mask_round_from=int(
                config.get("jepa_mask_round_from", len(jepa_mask_lengths))
            ),
            jepa_intensity_aware_mask_config={
                key: float(config.get(f"jepa_intensity_aware_{key}", value))
                for key, value in AWARE_MIXED_MASK_CONFIG.items()
            },
            jepa_allow_target_overlap=bool(
                config.get("jepa_allow_target_overlap", False)
            ),
            num_peaks=preprocessing["num_peaks"],
            dataloader_pin_memory=bool(
                config.get("dataloader_pin_memory", torch.cuda.is_available())
            ),
            dataloader_num_workers=dataloader_num_workers,
            dataloader_prefetch_factor=int(
                config.get("dataloader_prefetch_factor", 2)
            ),
            dataloader_persistent_workers=bool(
                config.get(
                    "dataloader_persistent_workers",
                    dataloader_num_workers > 0,
                )
            ),
            dataloader_multiprocessing_context=str(
                config.get("dataloader_multiprocessing_context", "")
            ),
            dataloader_output_format=str(
                config.get("dataloader_output_format", "torch")
            ),
        )
