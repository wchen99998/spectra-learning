import numpy as np
import torch

from spectra_learning.models.model import PeakSetJEPA
from spectra_learning.models.settings import PeakSetJEPASettings
from spectra_learning.models.spectrum_metadata import (
    MASSIVE_V2_ACQUISITION_SCHEMA,
    MASSIVE_V2_CONDITION_DIM,
    drop_massive_v2_metadata_torch,
    instrument_family_id,
    polarity_id,
    torch_spectrum_metadata_from_batch,
)


def _metadata_batch() -> dict[str, torch.Tensor]:
    return {
        "precursor_mz": torch.tensor([0.5, 0.7]),
        "precursor_mz_present": torch.ones(2),
        "collision_energy": torch.tensor([0.25, 0.4]),
        "collision_energy_present": torch.ones(2),
        "charge": torch.tensor([2.0, 1.0]),
        "charge_present": torch.ones(2),
        "mass_accuracy": torch.tensor([0.6, 0.4]),
        "mass_accuracy_present": torch.ones(2),
        "retention_time_fraction": torch.tensor([0.2, 0.8]),
        "retention_time_present": torch.ones(2),
        "precursor_intensity_zscore": torch.tensor([-0.2, 0.4]),
        "precursor_intensity_present": torch.ones(2),
        "polarity_id": torch.tensor([1, 2]),
        "acquisition_type_id": torch.tensor([1, 2]),
        "isolation_window_lower_offset": torch.tensor([0.1, 0.2]),
        "isolation_window_upper_offset": torch.tensor([0.2, 0.3]),
        "isolation_window_present": torch.ones(2),
        "instrument_family_id": torch.tensor([1, 2]),
    }


def test_metadata_taxonomy_prefers_specific_instrument_families() -> None:
    assert instrument_family_id("LTQ Orbitrap XL") == 1
    assert instrument_family_id("Q-TOF") == 2
    assert instrument_family_id("5800 TOF/TOF") == 3
    assert instrument_family_id("LTQ ion trap") == 4
    assert polarity_id("[M+H]+") == 1
    assert polarity_id("[M-H]-") == 2


def test_massive_v2_condition_vector_and_grouped_dropout() -> None:
    metadata = torch_spectrum_metadata_from_batch(
        _metadata_batch(),
        MASSIVE_V2_ACQUISITION_SCHEMA,
    )
    assert metadata.shape == (2, MASSIVE_V2_CONDITION_DIM)
    torch.testing.assert_close(metadata[:, 4], torch.tensor([2 / 21, 1 / 21]))
    dropped = drop_massive_v2_metadata_torch(metadata, 1.0)
    torch.testing.assert_close(dropped[:, :12], torch.zeros(2, 12))
    torch.testing.assert_close(dropped[:, 12], torch.ones(2))
    torch.testing.assert_close(dropped[:, 15], torch.ones(2))
    torch.testing.assert_close(dropped[:, 18:21], torch.zeros(2, 3))
    torch.testing.assert_close(dropped[:, 21], torch.ones(2))


def _settings() -> PeakSetJEPASettings:
    return PeakSetJEPASettings(
        training_mode="mae",
        model_dim=32,
        encoder_num_layers=1,
        encoder_num_heads=4,
        feature_mlp_hidden_dim=32,
        encoder_fourier_mlp_hidden_dim=32,
        encoder_fourier_num_freqs=4,
        num_peaks=6,
        predictor_dim=32,
        masked_latent_predictor_num_layers=1,
        masked_latent_predictor_num_heads=4,
        pairmixer_transition_type="feedforward",
        encoder_metadata_schema=MASSIVE_V2_ACQUISITION_SCHEMA,
        encoder_metadata_conditioning="adaln_zero",
        encoder_metadata_condition_dim=16,
    )


def test_adaln_zero_initialization_and_first_step_gate_gradient() -> None:
    model = PeakSetJEPA(_settings())
    block = model.encoder.blocks[0]
    assert torch.count_nonzero(block.adaLN_modulation.weight) == 0
    assert torch.count_nonzero(model.encoder.final_adaLN_modulation.weight) == 0
    mz = torch.rand(2, 6)
    intensity = torch.rand(2, 6)
    valid = torch.ones(2, 6, dtype=torch.bool)
    metadata = torch_spectrum_metadata_from_batch(
        _metadata_batch(),
        MASSIVE_V2_ACQUISITION_SCHEMA,
    )
    output = model.encoder(
        mz,
        intensity,
        valid_mask=valid,
        spectrum_metadata=metadata,
    )
    output.square().mean().backward()
    assert block.adaLN_modulation.weight.grad.abs().sum() > 0


def test_metadata_changes_output_after_modulation_is_enabled() -> None:
    model = PeakSetJEPA(_settings()).eval()
    block = model.encoder.blocks[0]
    with torch.no_grad():
        block.adaLN_modulation.weight[2 * model.model_dim : 3 * model.model_dim].fill_(
            0.01
        )
    mz = torch.rand(2, 6)
    intensity = torch.rand(2, 6)
    valid = torch.ones(2, 6, dtype=torch.bool)
    first = torch_spectrum_metadata_from_batch(
        _metadata_batch(),
        MASSIVE_V2_ACQUISITION_SCHEMA,
    )
    second = first.clone()
    second[:, 0] += 0.5
    with torch.no_grad():
        first_output = model.encoder(
            mz, intensity, valid_mask=valid, spectrum_metadata=first
        )
        second_output = model.encoder(
            mz, intensity, valid_mask=valid, spectrum_metadata=second
        )
    assert not np.allclose(first_output.numpy(), second_output.numpy())
