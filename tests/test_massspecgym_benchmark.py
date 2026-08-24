import numpy as np

from scripts.modal_massspecgym_maccs_probe import _massspecgym_acquisition_metadata


def test_massspecgym_acquisition_metadata_preserves_presence_and_taxonomy() -> None:
    metadata = _massspecgym_acquisition_metadata(
        {
            "collision_energy": [None, 0.0, 25.0, 150.0],
            "adduct": ["[M+H]+", "[M+Na]+", "[M+H]+", "[M+Na]+"],
            "instrument_type": ["Orbitrap", "QTOF", "", "Orbitrap"],
        }
    )

    np.testing.assert_array_equal(
        metadata["collision_energy"], [0.0, 0.0, 0.25, 1.0]
    )
    np.testing.assert_array_equal(
        metadata["collision_energy_present"], [0.0, 0.0, 1.0, 1.0]
    )
    np.testing.assert_array_equal(metadata["charge"], np.ones(4))
    np.testing.assert_array_equal(metadata["charge_present"], np.ones(4))
    np.testing.assert_array_equal(metadata["polarity_id"], np.ones(4))
    np.testing.assert_array_equal(metadata["instrument_family_id"], [1, 2, 0, 1])
