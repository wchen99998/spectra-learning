import numpy as np

from spectra_learning.config import load_config
from scripts.backfill_legacy_jax_checkpoint_reference import (
    LEGACY_METADATA_DIM,
    legacy_reference_inputs,
)


def test_legacy_reference_inputs_reproduce_launch_pipeline() -> None:
    config = load_config(
        "configs/1b_singlemixer_dense_muon_metadata_adaln.py"
    )

    raw_input, encoder_input = legacy_reference_inputs(config)

    assert encoder_input["spectrum_metadata"].shape == (2, LEGACY_METADATA_DIM)
    np.testing.assert_allclose(
        np.asarray(encoder_input["peak_intensity"]).max(axis=1),
        np.ones(2),
    )
    np.testing.assert_array_equal(
        np.asarray(encoder_input["spectrum_metadata"][:, 10:12]),
        np.stack(
            [
                np.asarray(raw_input["precursor_intensity_zscore"]),
                np.asarray(raw_input["precursor_intensity_present"]),
            ],
            axis=-1,
        ),
    )
