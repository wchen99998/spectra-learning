import json

import numpy as np
from rdkit import Chem

from scripts.prepare_encoder_chemistry_panel import (
    chemical_labels, matched_controls, prepare, repeated_rows,
    select_primary_rows, verify_identities, weak_loss_flags,
)


def key(smiles):
    return Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))


def test_identity_conflicts_exclude_whole_groups_and_crossfold_leaks():
    smiles = np.array(["CCO", "CCO", "CCN", "CCN", "CCC", "CCC", "CO"])
    source = np.array([key(s)[:14] for s in smiles])
    source[1] = key("CCN")[:14]
    fold = np.array(["train", "train", "train", "train", "train", "test", "val"])
    identity, audit = verify_identities(smiles, source, fold)
    # Both groups implicated by a wrong source label are excluded, as is the
    # computed connectivity shared across official folds.
    np.testing.assert_array_equal(identity["identity_eligible"], [False] * 6 + [True])
    assert audit["source_connectivity_mismatch_rows"] == 1
    assert audit["crossfold_computed_connectivities"]["train/test"] == [key("CCC")[:14]]


def test_primary_selection_is_grouped_and_repeats_require_exact_graph():
    identity = {
        "molecule": np.array(["a", "a", "a", "b", "b", "c", "c"]),
        "computed_inchikey": np.array(["a-full", "a-full", "a-full", "b-full", "b-full", "c-full", "c-other"]),
        "canonical_smiles": np.array(["a", "a", "tautomer", "b", "b", "c", "stereoisomer"]),
    }
    fold = np.array(["train"] * 3 + ["val"] * 2 + ["test"] * 2)
    eligible = np.ones(7, bool)
    rows = select_primary_rows(identity, fold, eligible, {"train": 1, "val": 1, "test": 1}, 17)
    assert len(set(identity["molecule"][rows])) == 3
    np.testing.assert_array_equal(rows, select_primary_rows(identity, fold, eligible,
                                                         {"train": 1, "val": 1, "test": 1}, 17))
    repeats, primary_index = repeated_rows(np.array([0, 3, 5]), identity, eligible)
    np.testing.assert_array_equal(repeats, [1, 4])
    np.testing.assert_array_equal(primary_index, [0, 1])


def test_controls_require_all_calipers_and_never_cross_fold_or_identity():
    n = 9
    data = {"fold": np.array(["test"] * 8 + ["train"]),
            "molecule": np.array(["a", "b", "a", "c", "d", "e", "f", "g", "h"]),
            "instrument_type": np.array(["Orbitrap"] * n),
            "adduct": np.array(["[M+H]+"] * 6 + ["[M+Na]+"] + ["[M+H]+"] * 2),
            "precursor_mz_raw": np.array([100., 102., 100., 106., 100., 100., 100., 100., 100.]),
            "collision_energy_raw": np.array([20., 25., 20., 20., 40., 20., 20., np.nan, 20.]),
            "collision_energy_present": np.array([1.] * 7 + [0., 1.]),
            "peak_valid_mask": np.arange(20)[None, :] < np.array([5, 8, 5, 5, 5, 15, 5, 5, 5])[:, None],
            "rows": np.arange(n)}
    binary = np.array([[0, 1]] + [[1, 1]] * (n - 1), np.uint8)
    controls, audit = matched_controls(data, binary)
    assert controls["donor"][0] == 1
    np.testing.assert_array_equal(controls["opposite_concept_donor"][0], [1, -1])
    assert controls["eligible_donor_count"][0] == 1
    assert controls["donor"][-1] == -1
    assert audit["audits"]["nuisance"]["mass_gap_da"]["max"] <= 5
    # The closest molecule cannot be used if the label is unchanged.
    binary[1, 0] = 0
    changed, _ = matched_controls(data, binary)
    assert changed["opposite_concept_donor"][0, 0] == -1


def test_weak_loss_matches_respect_adduct_and_peak_mask():
    data = {"precursor_mz_raw": np.array([200., 200., 200.]),
            "peak_mz": np.array([[.181989435, .180]] * 3),
            "peak_valid_mask": np.array([[True, True], [True, True], [False, True]]),
            "adduct": np.array(["[M+H]+", "[M+Na]+", "[M+H]+"])}
    result = weak_loss_flags(data)
    assert result["peak_matches"][0, 0, 0]
    assert not result["peak_matches"][1:].any()
    assert not result["peak_matches"][0, 1].any()


def test_saved_panel_keeps_rows_labels_and_native_inputs_aligned(tmp_path):
    smiles = ["CCCO", "CCCO", "CCCN", "CCCN", "CCCC", "CCCC"]
    folds = ["train", "train", "val", "val", "test", "test"]
    tsv = tmp_path / "spectra.tsv"
    tsv.write_text("identifier\tmzs\tintensities\tsmiles\tinchikey\tformula\tprecursor_mz\tadduct\tinstrument_type\tcollision_energy\tfold\n" + "".join(
        f"r{i}\t25,30,35,40,45,50\t1,.5,.4,.3,.2,.1\t{s}\t{key(s)[:14]}\tX\t100\t[M+H]+\tOrbitrap\t{20+i}\t{f}\n"
        for i, (s, f) in enumerate(zip(smiles, folds, strict=True))))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"num_peaks": 127, "peak_drop_min_intensity": .01,
        "peak_ordering": "mz", "max_precursor_mz": 1000,
        "precursor_peak_exclusion_window_da": 1., "min_peak_intensity": .0001}))
    out = tmp_path / "panel"
    manifest = prepare(tsv, config, out, train_size=1, val_size=1, test_size=1)
    with np.load(out / "inputs.npz") as data, np.load(out / "labels.npz") as labels, np.load(out / "repeat_inputs.npz") as repeat:
        assert data["peak_mz"].shape == (3, 127)
        np.testing.assert_allclose(data["peak_mz"][:, :6], [[.025, .030, .035, .040, .045, .050]] * 3)
        np.testing.assert_array_equal(data["rows"], labels["rows"])
        np.testing.assert_array_equal(repeat["primary_source_row"], data["rows"][repeat["primary_index"]])
        np.testing.assert_array_equal(labels["binary"], chemical_labels(data["smiles"])["binary"])
        assert labels["binary"].shape == (3, 22)
        assert labels["maccs"].shape == (3, 166)
        assert labels["morgan"].shape == (3, 1024)
        assert not data["unseen_nonempty_train_scaffold"].any()
    assert manifest["peak_fragment_ground_truth_available"] is False
    assert manifest["repeated_acquisitions"]["test"]["alternate_spectra"] == 1
