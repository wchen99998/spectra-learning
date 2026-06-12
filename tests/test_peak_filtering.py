import numpy as np
import torch
from ml_collections import config_dict

from spectra_learning.data.gems.settings import GemsDataConfig
from spectra_learning.data.spectra import (
    DEFAULT_GROUPED_PEAK_SHOULDER_DA,
    PEAK_FILTERING_GROUPED,
    PEAK_FILTERING_TOP_INTENSITY,
    PEAK_MZ_MAX,
    preprocess_peak_batch_numpy,
    preprocess_peak_batch_torch,
)


def test_grouped_peak_filtering_keeps_group_representatives_torch() -> None:
    mz = torch.tensor([[100.0, 100.03, 101.002, 150.0, 151.0, 300.0, 400.0]])
    intensity = torch.tensor([[1.0, 0.90, 0.95, 0.85, 0.80, 0.70, 0.60]])
    precursor_mz = torch.tensor([500.0])

    top_intensity = preprocess_peak_batch_torch(
        mz,
        intensity,
        precursor_mz,
        num_peaks=4,
        peak_drop_min_intensity=0.0,
        peak_ordering="mz",
        max_precursor_mz=1000.0,
        peak_filtering=PEAK_FILTERING_TOP_INTENSITY,
    )
    grouped = preprocess_peak_batch_torch(
        mz,
        intensity,
        precursor_mz,
        num_peaks=4,
        peak_drop_min_intensity=0.0,
        peak_ordering="mz",
        max_precursor_mz=1000.0,
        peak_filtering=PEAK_FILTERING_GROUPED,
        grouped_peak_shoulder_da=0.05,
        grouped_peak_isotope_charges=(1,),
    )

    top_mz = top_intensity["peak_mz"][0, top_intensity["peak_valid_mask"][0]]
    grouped_mz = grouped["peak_mz"][0, grouped["peak_valid_mask"][0]]
    assert torch.allclose(
        top_mz * PEAK_MZ_MAX,
        torch.tensor([100.0, 100.03, 101.002, 150.0]),
        atol=1e-4,
    )
    assert torch.allclose(
        grouped_mz * PEAK_MZ_MAX,
        torch.tensor([100.0, 150.0, 300.0, 400.0]),
        atol=1e-4,
    )


def test_grouped_peak_filtering_keeps_group_representatives_numpy() -> None:
    spectra = np.zeros((1, 2, 7), dtype=np.float32)
    spectra[0, 0] = [100.0, 100.03, 101.002, 150.0, 151.0, 300.0, 400.0]
    spectra[0, 1] = [1.0, 0.90, 0.95, 0.85, 0.80, 0.70, 0.60]

    grouped = preprocess_peak_batch_numpy(
        spectra,
        np.asarray([500.0], dtype=np.float32),
        num_peaks=4,
        peak_drop_min_intensity=0.0,
        peak_ordering="mz",
        max_precursor_mz=1000.0,
        peak_filtering=PEAK_FILTERING_GROUPED,
        grouped_peak_shoulder_da=0.05,
        grouped_peak_isotope_charges=(1,),
    )

    grouped_mz = grouped["peak_mz"][0, grouped["peak_valid_mask"][0]]
    assert np.allclose(
        grouped_mz * PEAK_MZ_MAX,
        np.asarray([100.0, 150.0, 300.0, 400.0], dtype=np.float32),
        atol=1e-4,
    )


def test_gems_data_config_reads_grouped_peak_filtering_fields() -> None:
    cfg = config_dict.ConfigDict()
    cfg.peak_filtering = PEAK_FILTERING_GROUPED
    cfg.grouped_peak_shoulder_da = 0.05
    cfg.grouped_peak_isotope_charges = (1, 2, 3)

    data_config = GemsDataConfig.from_config(cfg)

    assert DEFAULT_GROUPED_PEAK_SHOULDER_DA == 0.05
    assert data_config.peak_filtering == PEAK_FILTERING_GROUPED
    assert data_config.grouped_peak_shoulder_da == 0.05
    assert data_config.grouped_peak_isotope_charges == (1, 2, 3)
