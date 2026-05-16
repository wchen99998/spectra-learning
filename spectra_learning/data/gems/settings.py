from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from ml_collections import config_dict

from spectra_learning.data.gems.intensity_aware import AWARE_MIXED_MASK_CONFIG
from spectra_learning.data.spectra import (
    DEFAULT_MAX_PRECURSOR_MZ,
    DEFAULT_MIN_PEAK_INTENSITY,
    DEFAULT_PRECURSOR_PEAK_EXCLUSION_WINDOW_DA,
)

DEFAULT_BATCH_SIZE = 512
DEFAULT_ARTIFACT_DIR = Path("data/gems_artifacts")
NUM_PEAKS_OUTPUT = 60
GEMS_METADATA_FILENAME = "metadata.json"


def _config_get(config: config_dict.ConfigDict, key: str, default: Any) -> Any:
    return config.get(key, default)


@dataclass(frozen=True)
class GemsDataConfig:
    artifact_dir: Path
    gems_native_repo_id: str
    gems_native_revision: str
    gems_native_source_hdf5_path: str
    gems_native_source_url: str
    batch_size: int
    drop_remainder: bool
    max_precursor_mz: float
    min_peak_intensity: float
    peak_drop_min_intensity: float
    peak_ordering: str
    precursor_peak_exclusion_window_da: float
    jepa_num_target_blocks: int
    jepa_context_fraction: float
    jepa_target_fraction: float
    jepa_block_min_len: int
    jepa_mask_strategy: str
    jepa_mask_lengths: tuple[int, ...]
    jepa_mask_round_from: int
    jepa_intensity_aware_mask_config: dict[str, float]
    jepa_allow_target_overlap: bool
    use_precursor_token: bool
    num_peaks: int
    dataloader_pin_memory: bool
    dataloader_num_workers: int
    dataloader_prefetch_factor: int
    dataloader_persistent_workers: bool

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
            gems_native_repo_id=str(
                _config_get(config, "gems_native_repo_id", "")
            ).strip(),
            gems_native_revision=str(_config_get(config, "gems_native_revision", "main")),
            gems_native_source_hdf5_path=str(
                _config_get(config, "gems_native_source_hdf5_path", "")
            ).strip(),
            gems_native_source_url=str(
                _config_get(config, "gems_native_source_url", "")
            ).strip(),
            batch_size=int(_config_get(config, "batch_size", DEFAULT_BATCH_SIZE)),
            drop_remainder=bool(_config_get(config, "drop_remainder", True)),
            max_precursor_mz=float(
                _config_get(config, "max_precursor_mz", DEFAULT_MAX_PRECURSOR_MZ)
            ),
            min_peak_intensity=min_peak_intensity,
            peak_drop_min_intensity=float(
                _config_get(config, "peak_drop_min_intensity", min_peak_intensity)
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
            jepa_mask_strategy=str(
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
            use_precursor_token=bool(_config_get(config, "use_precursor_token", False)),
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
        )
