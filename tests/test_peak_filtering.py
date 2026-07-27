import numpy as np
import pytest
import torch
from ml_collections import config_dict

from spectra_learning.data.gems.collate import GemsBatchCollator
from spectra_learning.data.gems.settings import GemsDataConfig
from spectra_learning.data.contracts import peak_preprocessing_contract
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
    spectra_from_peak_lists,
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


@pytest.mark.parametrize(
    "config",
    (
        {"peak_filtering": "typo"},
        {"peak_ordering": "typo"},
    ),
)
def test_peak_preprocessing_contract_rejects_unknown_modes(config) -> None:
    with pytest.raises(ValueError, match="Unknown peak_"):
        peak_preprocessing_contract(config)


@pytest.mark.parametrize(
    "overrides",
    (
        {"peak_filtering": "typo"},
        {"peak_ordering": "typo"},
    ),
)
def test_peak_preprocessor_rejects_unknown_modes(overrides) -> None:
    kwargs = {
        "num_peaks": 1,
        "peak_drop_min_intensity": 0.0,
        "peak_ordering": "mz",
        "max_precursor_mz": 1000.0,
        "peak_filtering": PEAK_FILTERING_TOP_INTENSITY,
        **overrides,
    }
    with pytest.raises(ValueError, match="Unknown peak_"):
        preprocess_peak_batch_torch(
            torch.tensor([[100.0]]),
            torch.tensor([[1.0]]),
            torch.tensor([200.0]),
            **kwargs,
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


def test_numpy_and_torch_peak_preprocessing_randomized_parity() -> None:
    rng = np.random.default_rng(3491)
    for peak_filtering in (
        PEAK_FILTERING_TOP_INTENSITY,
        PEAK_FILTERING_GROUPED,
    ):
        for peak_ordering in ("mz", "intensity"):
            for input_peaks in (9, 41):
                mz = rng.uniform(-20.0, 1100.0, (7, input_peaks)).astype(np.float32)
                intensity = rng.uniform(-0.1, 1.5, (7, input_peaks)).astype(
                    np.float32
                )
                precursor_mz = rng.uniform(50.0, 1050.0, 7).astype(np.float32)
                bases = rng.uniform(120.0, 700.0, 7).astype(np.float32)
                mz[:, :4] = np.stack(
                    [bases, bases + 0.03, bases + 1.002, bases + 2.004],
                    axis=1,
                )
                intensity[:, :4] = rng.uniform(0.1, 1.5, (7, 4))
                spectra = np.stack([mz, intensity], axis=1)
                kwargs = {
                    "num_peaks": 16,
                    "peak_drop_min_intensity": 0.08,
                    "peak_ordering": peak_ordering,
                    "max_precursor_mz": 1000.0,
                    "precursor_peak_exclusion_window_da": 0.1,
                    "min_peak_intensity": 0.05,
                    "peak_filtering": peak_filtering,
                    "grouped_peak_shoulder_da": 0.05,
                    "grouped_peak_isotope_charges": (1, 2, 3),
                }

                numpy_batch = preprocess_peak_batch_numpy(
                    spectra,
                    precursor_mz,
                    **kwargs,
                )
                torch_batch = preprocess_peak_batch_torch(
                    torch.from_numpy(mz),
                    torch.from_numpy(intensity),
                    torch.from_numpy(precursor_mz),
                    **kwargs,
                )

                for key, expected in numpy_batch.items():
                    actual = torch_batch[key].numpy()
                    if np.issubdtype(expected.dtype, np.floating):
                        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
                    else:
                        np.testing.assert_array_equal(actual, expected)


def test_peak_preprocessing_uses_relative_intensity_thresholds() -> None:
    raw = np.zeros((1, 2, 128), dtype=np.float32)
    raw[0, 0, :3] = [100.0, 200.0, 300.0]
    raw[0, 1, :3] = [1_000_000.0, 200.0, 10.0]
    normalized = spectra_from_peak_lists(
        [[100.0, 200.0, 300.0]],
        [[1_000_000.0, 200.0, 10.0]],
    )
    precursor_mz = np.asarray([500.0], dtype=np.float32)
    kwargs = {
        "num_peaks": 3,
        "peak_drop_min_intensity": 1e-4,
        "peak_ordering": "mz",
        "max_precursor_mz": 1000.0,
        "min_peak_intensity": 1e-4,
    }

    raw_numpy = preprocess_peak_batch_numpy(raw, precursor_mz, **kwargs)
    normalized_numpy = preprocess_peak_batch_numpy(normalized, precursor_mz, **kwargs)
    raw_torch = preprocess_peak_batch_torch(
        torch.from_numpy(raw[:, 0]),
        torch.from_numpy(raw[:, 1]),
        torch.from_numpy(precursor_mz),
        **kwargs,
    )
    normalized_torch = preprocess_peak_batch_torch(
        torch.from_numpy(normalized[:, 0]),
        torch.from_numpy(normalized[:, 1]),
        torch.from_numpy(precursor_mz),
        **kwargs,
    )

    for key, expected in raw_numpy.items():
        for actual in (
            normalized_numpy[key],
            raw_torch[key].numpy(),
            normalized_torch[key].numpy(),
        ):
            if np.issubdtype(expected.dtype, np.floating):
                np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
            else:
                np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        raw_numpy["peak_valid_mask"],
        [[True, True, False]],
    )


def test_peak_preprocessing_marks_zero_placeholder_valid() -> None:
    spectra = np.zeros((1, 2, 3), dtype=np.float32)
    spectra[0, 0] = [100.0, 200.0, 300.0]
    precursor_mz = np.asarray([500.0], dtype=np.float32)
    kwargs = {
        "num_peaks": 3,
        "peak_drop_min_intensity": 1e-4,
        "peak_ordering": "mz",
        "max_precursor_mz": 1000.0,
    }

    numpy_batch = preprocess_peak_batch_numpy(spectra, precursor_mz, **kwargs)
    torch_batch = preprocess_peak_batch_torch(
        torch.from_numpy(spectra[:, 0]),
        torch.from_numpy(spectra[:, 1]),
        torch.from_numpy(precursor_mz),
        **kwargs,
    )

    for key, expected in numpy_batch.items():
        np.testing.assert_array_equal(torch_batch[key].numpy(), expected)
    np.testing.assert_array_equal(
        numpy_batch["peak_valid_mask"],
        [[True, False, False]],
    )
    np.testing.assert_array_equal(numpy_batch["peak_mz"], np.zeros((1, 3)))
    np.testing.assert_array_equal(numpy_batch["peak_intensity"], np.zeros((1, 3)))
    np.testing.assert_array_equal(
        numpy_batch["peak_group_id"],
        np.full((1, 3), PEAK_GROUP_PADDING_ID),
    )


def test_spectra_from_peak_lists_pads_truncates_and_normalizes() -> None:
    spectra = spectra_from_peak_lists(
        [[100.0, 200.0], list(np.arange(200, dtype=np.float32))],
        [[2.0, 1.0], list(np.arange(200, dtype=np.float32))],
    )

    assert spectra.shape == (2, 2, 128)
    assert spectra.dtype == np.float32
    np.testing.assert_array_equal(spectra[0, 0, :3], [100.0, 200.0, 0.0])
    np.testing.assert_allclose(spectra[0, 1, :3], [1.0, 0.5, 0.0])
    assert spectra[1, 0, -1] == 127.0
    assert spectra[1, 1, -1] == 1.0


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


def test_gems_collator_canonicalizes_precursor_charge() -> None:
    collator = GemsBatchCollator(
        augment=False,
        num_target_blocks=1,
        context_fraction=0.5,
        target_fraction=0.5,
        block_min_len=1,
        num_peaks=2,
        max_precursor_mz=1000.0,
        min_peak_intensity=0.0,
        peak_drop_min_intensity=0.0,
        peak_ordering="mz",
        precursor_peak_exclusion_window_da=0.0,
    )
    spectra = torch.zeros(2, 128, dtype=torch.float32)
    spectra[0, 0] = 100.0
    spectra[1, 0] = 1.0
    samples = [
        {
            "spectra": spectra.clone(),
            "precursor_mz_raw": torch.tensor(500.0),
            "collision_energy": torch.tensor(35.0),
            "charge": torch.tensor(charge),
        }
        for charge in (0.0, -2.0, float("nan"))
    ]

    batch = collator(samples)

    torch.testing.assert_close(
        batch["charge"],
        torch.tensor([1.0, 2.0, 1.0]),
    )
