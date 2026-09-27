"""Check physical ion units and alignment without loading spectrum embeddings."""

import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ion_logmixture_data import load_ion_training_inputs
from ion_logmixture_data import PROTON_MASS


@pytest.fixture
def inputs(tmp_path):
    root, data = tmp_path / "input", tmp_path / "prepared"
    (root / "data").mkdir(parents=True)
    (root / "msg15").mkdir()
    data.mkdir()
    tsv = root / "msg15/MassSpecGym1.5.tsv"
    tsv.write_text("precursor_mz\tadduct\tsmiles\n200\t[M+H]+\tunused\n600\t[M+Na]+\tunused\n"
                   "300\t[M+H]1+\tunused\n350\t[M+2H]2+\tunused\n400\t\tunused\n")
    digest = hashlib.sha256(tsv.read_bytes()).hexdigest()
    config = tmp_path / "encoder_metadata.json"
    config.write_text(json.dumps({"task_contract": {"config": {"max_precursor_mz": 500.0}}}))
    embedding = {"samples": 5, "embedding_shape": [5, 3, 8], "data_root": str(data),
                 "config_metadata_path": str(config), "tsv_sha256": digest}
    (root / "embedding_metadata.json").write_text(json.dumps(embedding))
    (data / "preparation_metadata.json").write_text(json.dumps({"tsv_sha256": digest}))
    mask = np.array([[True, True, False]] * 5)
    normalized_mz = np.array([[.05, .1, 0], [.08, .16, 0], [.12, .2, 0], [.07, .11, 0], [.03, .09, 0]], np.float32)
    normalized_precursor = np.array([.4, 1, .6, .7, .8], np.float32)
    np.savez(data / "prepared_massspecgym.npz", peak_mz=normalized_mz, peak_valid_mask=mask,
             precursor_mz=normalized_precursor)
    np.save(root / "data/token_valid_mask.npy", np.column_stack([mask, np.ones(5, bool)]))
    # Preserve the supplied legacy coordinate even when it differs from Da ratios.
    fractions = normalized_mz / normalized_precursor[:, None]
    return root, {"kind": "raw", "embedding_metadata": embedding}, mask, fractions


def test_physical_units_exact_adducts_and_legacy_fraction_preservation(inputs):
    root, peak_input, mask, fractions = inputs
    query, provenance = load_ion_training_inputs(*inputs)
    assert set(query) == {"fraction", "mz", "eligible", "adduct_mass"}
    np.testing.assert_array_equal(query["fraction"], fractions)
    np.testing.assert_allclose(query["mz"], [[50, 100, 0], [80, 160, 0], [120, 200, 0], [70, 110, 0], [30, 90, 0]], atol=1e-5)
    np.testing.assert_array_equal(query["eligible"], [True, True, False, False, False])
    np.testing.assert_array_equal(query["adduct_mass"][[0, 2, 3, 4]], np.full(4, PROTON_MASS, np.float32))
    assert query["adduct_mass"][1] == np.float32(22.989769282 - .000548579909065)
    assert provenance["units"]["peak_mz_scale_da"] == 1000
    assert provenance["units"]["precursor_mz_scale_da"] == 500
    assert provenance["units"]["clipped_precursor_rows"] == 1
    assert provenance["eligible_rows"] == 2
    assert provenance["tokens_read"] is False
    assert not (root / "covariance_embeddings.float16.npy").exists()


def test_wrong_tsv_provenance_is_rejected(inputs):
    root, _, _, _ = inputs
    with (root / "msg15/MassSpecGym1.5.tsv").open("a") as stream:
        stream.write("500\t[M+H]+\tunused\n")
    with pytest.raises(ValueError, match="Raw ion TSV differs"):
        load_ion_training_inputs(*inputs)


@pytest.mark.parametrize("array", ["peak_valid_mask", "precursor_mz"])
def test_misaligned_prepared_physical_inputs_are_rejected(inputs, array):
    _, peak_input, _, _ = inputs
    path = Path(peak_input["embedding_metadata"]["data_root"]) / "prepared_massspecgym.npz"
    with np.load(path) as saved:
        prepared = {name: saved[name] for name in saved.files}
    if array == "peak_valid_mask":
        prepared[array][0] = [True, False, True]
    else:
        prepared[array][0] = .2
    np.savez(path, **prepared)
    with pytest.raises(AssertionError):
        load_ion_training_inputs(*inputs)


def test_bad_token_slot_mapping_is_rejected(inputs):
    root, _, _, _ = inputs
    path = root / "data/token_valid_mask.npy"
    mask = np.load(path)
    mask[1] = [True, False, True, True]
    np.save(path, mask)
    with pytest.raises(AssertionError, match="Frozen physical peak slots"):
        load_ion_training_inputs(*inputs)


def test_unsupported_peak_input_kind_is_rejected(inputs):
    inputs[1]["kind"] = "pca"
    with pytest.raises(ValueError, match="same raw-token embedding metadata"):
        load_ion_training_inputs(*inputs)


@pytest.fixture
def trainable_inputs(inputs):
    from trainable_jepa_peak import load_spectra

    root, raw_input, mask, fractions = inputs
    embedding = raw_input["embedding_metadata"]
    embedding["metadata_input_dim"] = 28
    (root / "embedding_metadata.json").write_text(json.dumps(embedding))
    metadata_path = Path(embedding["config_metadata_path"])
    metadata_path.write_text(json.dumps({"task_contract": {"config": {
        "max_precursor_mz": 500.0, "encoder_metadata_schema": "massive_v2_acquisition_v1",
    }}}))
    data_root = Path(embedding["data_root"])
    prepared_path = data_root / "prepared_massspecgym.npz"
    with np.load(prepared_path) as data:
        prepared = {name: data[name] for name in data.files}
    prepared.update(
        peak_intensity=np.arange(15, dtype=np.float32).reshape(5, 3) / 15,
        split=np.array([0, 0, 1, 2, 2], np.int32),
        collision_energy=np.arange(5, dtype=np.float32) / 10,
        collision_energy_present=np.ones(5, bool), charge=np.ones(5, np.float32),
        charge_present=np.ones(5, bool), polarity_id=np.ones(5, np.int32),
        instrument_family_id=np.arange(5, dtype=np.int32),
    )
    np.savez(prepared_path, **prepared)
    np.save(root / "data/fold.npy", np.array(["train", "train", "val", "test", "test"]))
    packed, actual_mask, peak_input = load_spectra(root, data_root, metadata_path)
    np.testing.assert_array_equal(actual_mask, mask)
    return root, peak_input, mask, fractions, packed, prepared


def test_trainable_input_binding_preserves_frozen_ion_arrays(trainable_inputs):
    root, peak_input, mask, fractions, packed, prepared = trainable_inputs
    trainable, provenance = load_ion_training_inputs(root, peak_input, mask, fractions)
    raw, frozen_provenance = load_ion_training_inputs(
        root, {"kind": "raw", "embedding_metadata": peak_input["embedding_metadata"]}, mask, fractions)
    for name in raw:
        assert trainable[name].dtype == raw[name].dtype
        assert trainable[name].tobytes() == raw[name].tobytes()
    assert provenance["trainable_jepa_input_verified"] is True
    assert "trainable_jepa_input_verified" not in frozen_provenance
    assert provenance["query_array_sha256"] == frozen_provenance["query_array_sha256"]
    for name in frozen_provenance["source_sha256"]:
        assert provenance["source_sha256"][name] == frozen_provenance["source_sha256"][name]
    assert provenance["source_sha256"]["encoder_metadata"] == peak_input["metadata_sha256"]
    for name in ("prepared_spectra", "preparation_metadata"):
        assert provenance["source_sha256"][name] == peak_input[f"{name}_sha256"]
    assert peak_input["shape"] == [5, 3, 32]
    assert packed.dtype == np.float32
    np.testing.assert_array_equal(packed[..., 0], prepared["peak_mz"])
    np.testing.assert_array_equal(packed[..., 1], prepared["peak_intensity"])
    np.testing.assert_array_equal(packed[..., 2], mask)
    np.testing.assert_array_equal(packed[:, 0, 3], prepared["precursor_mz"])
    assert not packed[:, 1:, 3:].any()


@pytest.mark.parametrize("name", ["peak_mz", "peak_intensity", "collision_energy", "instrument_family_id", "split"])
def test_changed_prepared_arrays_with_same_tsv_and_masks_are_rejected(trainable_inputs, name):
    root, peak_input, mask, fractions, _, prepared = trainable_inputs
    prepared[name].flat[0] += 1
    path = Path(peak_input["data_root"]) / "prepared_massspecgym.npz"
    np.savez(path, **prepared)
    with pytest.raises(ValueError, match="prepared_spectra changed"):
        load_ion_training_inputs(root, peak_input, mask, fractions)


@pytest.mark.parametrize("name", ["data_root", "metadata_path"])
def test_other_source_path_is_rejected_even_with_identical_bytes(trainable_inputs, tmp_path, name):
    root, peak_input, mask, fractions, _, _ = trainable_inputs
    original = Path(peak_input[name])
    other = tmp_path / f"other_{name}"
    if original.is_dir():
        shutil.copytree(original, other)
    else:
        shutil.copyfile(original, other)
    peak_input[name] = str(other)
    with pytest.raises(ValueError, match="must use the same"):
        load_ion_training_inputs(root, peak_input, mask, fractions)


@pytest.mark.parametrize("name", ["metadata_path", "preparation_metadata"])
def test_changed_metadata_with_unchanged_tsv_or_mass_scale_is_rejected(trainable_inputs, name):
    root, peak_input, mask, fractions, _, _ = trainable_inputs
    path = (Path(peak_input["metadata_path"]) if name == "metadata_path"
            else Path(peak_input["data_root"]) / "preparation_metadata.json")
    content = json.loads(path.read_text())
    content["changed_after_spectrum_loading"] = True
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="changed before loading ion inputs"):
        load_ion_training_inputs(root, peak_input, mask, fractions)


@pytest.mark.parametrize("name", ["metadata_path", "metadata_sha256", "prepared_spectra_sha256",
                                  "preparation_metadata_sha256"])
def test_missing_trainable_input_identity_is_rejected(trainable_inputs, name):
    root, peak_input, mask, fractions, _, _ = trainable_inputs
    del peak_input[name]
    with pytest.raises(ValueError, match="provenance is incomplete"):
        load_ion_training_inputs(root, peak_input, mask, fractions)
