import numpy as np
import torch
from ml_collections import config_dict

from spectra_learning.data.gems.collate import GemsBatchCollator
from spectra_learning.data.gems.settings import GemsDataConfig
from spectra_learning.data.spectra import (
    ASSUMED_PRECURSOR_CHARGE,
    COLLISION_ENERGY_MAX,
    DEFAULT_GROUPED_PEAK_SHOULDER_DA,
    PEAK_GROUP_PADDING_ID,
    PEAK_FILTERING_GROUPED,
    PEAK_FILTERING_TOP_INTENSITY,
    PEAK_MZ_MAX,
    PRECURSOR_CHARGE_MAX,
    preprocess_peak_batch_numpy,
    preprocess_peak_batch_torch,
)
from spectra_learning.models.spectrum_metadata import torch_spectrum_metadata_from_batch


def _numpy_spectrum_metadata(
    collision_energy: float,
    charge: float,
) -> dict[str, np.float32]:
    return {
        "collision_energy": np.float32(collision_energy),
        "charge": np.float32(charge),
    }


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
        torch.tensor([100.0, 101.002, 150.0, 151.0]),
        atol=1e-4,
    )
    grouped_group_id = grouped["peak_group_id"][0, grouped["peak_valid_mask"][0]]
    assert torch.equal(grouped_group_id, torch.tensor([0, 0, 1, 1], dtype=torch.int32))
    padding_group_id = grouped["peak_group_id"][0, ~grouped["peak_valid_mask"][0]]
    assert (padding_group_id == PEAK_GROUP_PADDING_ID).all()


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
        np.asarray([100.0, 101.002, 150.0, 151.0], dtype=np.float32),
        atol=1e-4,
    )
    grouped_group_id = grouped["peak_group_id"][0, grouped["peak_valid_mask"][0]]
    assert np.array_equal(grouped_group_id, np.asarray([0, 0, 1, 1], dtype=np.int32))


def test_gems_data_config_reads_grouped_peak_filtering_fields() -> None:
    cfg = config_dict.ConfigDict()
    cfg.peak_filtering = PEAK_FILTERING_GROUPED
    cfg.grouped_peak_shoulder_da = 0.05
    cfg.grouped_peak_isotope_charges = (1, 2, 3)
    cfg.dataloader_output_format = "numpy"

    data_config = GemsDataConfig.from_config(cfg)

    assert DEFAULT_GROUPED_PEAK_SHOULDER_DA == 0.05
    assert data_config.peak_filtering == PEAK_FILTERING_GROUPED
    assert data_config.grouped_peak_shoulder_da == 0.05
    assert data_config.grouped_peak_isotope_charges == (1, 2, 3)
    assert data_config.dataloader_output_format == "numpy"


def test_gems_collator_can_return_numpy_batch() -> None:
    collator = GemsBatchCollator(
        augment=True,
        num_target_blocks=1,
        context_fraction=0.5,
        target_fraction=0.25,
        block_min_len=1,
        num_peaks=4,
        max_precursor_mz=1000.0,
        min_peak_intensity=1e-4,
        peak_drop_min_intensity=1e-4,
        peak_ordering="mz",
        precursor_peak_exclusion_window_da=0.0,
        peak_filtering=PEAK_FILTERING_GROUPED,
        grouped_peak_shoulder_da=0.05,
        grouped_peak_isotope_charges=(1,),
        mask_strategy="contiguous",
        output_format="numpy",
    )
    samples = [
        {
            "spectra": np.asarray(
                [
                    [100.0, 100.03, 101.002, 150.0, 300.0],
                    [1.0, 0.9, 0.8, 0.7, 0.6],
                ],
                dtype=np.float32,
            ),
            "precursor_mz_raw": np.float32(500.0),
            **_numpy_spectrum_metadata(20.0, 2.0),
        }
    ]

    batch = collator(samples)

    assert isinstance(batch["peak_mz"], np.ndarray)
    assert isinstance(batch["peak_group_id"], np.ndarray)
    assert isinstance(batch["context_mask"], np.ndarray)
    assert isinstance(batch["target_masks"], np.ndarray)
    assert batch["peak_mz"].shape == (1, 4)
    assert batch["context_mask"].dtype == bool


def test_gems_collator_normalizes_spectrum_metadata() -> None:
    collator = GemsBatchCollator(
        augment=False,
        num_target_blocks=1,
        context_fraction=0.5,
        target_fraction=0.5,
        block_min_len=1,
        num_peaks=4,
        max_precursor_mz=1000.0,
        min_peak_intensity=0.0,
        peak_drop_min_intensity=0.0,
        peak_ordering="mz",
        precursor_peak_exclusion_window_da=0.0,
    )
    spectra = torch.zeros(2, 128, dtype=torch.float32)
    spectra[0, :2] = torch.tensor([100.0, 120.0])
    spectra[1, :2] = torch.tensor([1.0, 0.5])

    batch = collator(
        [
            {
                "spectra": spectra,
                "precursor_mz_raw": torch.tensor(500.0, dtype=torch.float32),
                "collision_energy": torch.tensor(150.0, dtype=torch.float32),
                "charge": torch.tensor(21.0, dtype=torch.float32),
            }
        ]
    )

    torch.testing.assert_close(
        batch["collision_energy"],
        torch.tensor([1.0], dtype=torch.float32),
    )
    torch.testing.assert_close(
        batch["charge"],
        torch.tensor([21.0], dtype=torch.float32),
    )
    torch.testing.assert_close(
        torch_spectrum_metadata_from_batch(batch),
        torch.tensor([[1.0, 1.0]], dtype=torch.float32),
    )
    assert COLLISION_ENERGY_MAX == 100.0
    assert PRECURSOR_CHARGE_MAX == 21.0
    assert ASSUMED_PRECURSOR_CHARGE == 1.0
