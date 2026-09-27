"""Complete molecular support and fixed selection for raw-token retrieval."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_atom_flare_jax as preparation
from atom_flare_data import GraphStore


@pytest.fixture
def input_root(tmp_path):
    root = tmp_path / "input"
    (root / "data").mkdir(parents=True)
    (root / "msg15").mkdir()
    folds = np.array(["train", "val", "train", "val", "test", "val"])
    smiles = ["C", "N", "CC", "O", "F", "Cl"]
    keys = [f"MOLECULE{i:06d}-STEREO" for i in range(len(smiles))]
    np.save(root / "data/fold.npy", folds)
    text = ["smiles\tinchikey\tfold"] + [
        f"{smile}\t{key}\t{fold}" for smile, key, fold in zip(smiles, keys, folds, strict=True)]
    (root / "msg15/MassSpecGym1.5.tsv").write_text("\n".join(text) + "\n")
    # Exercise support beyond the old first-16 bank and the selected atom cap.
    pools = {smile: [smile] for smile in smiles}
    pools["C"] = ["C" * n for n in range(1, 22)] + ["C" * 129]
    pools["F"] = ["F", "Br"]
    (root / "msg15/MassSpecGym1.5_retrieval_candidates_formula.json").write_text(json.dumps(pools))
    return root


def test_complete_pool_atom_capacity_and_selected_rows(input_root, tmp_path):
    out = tmp_path / "prepared"
    preparation.prepare_graphs(input_root, out, workers=1, val_n=2)
    store = GraphStore(out / "graphs")
    assert {"C" * 21, "C" * 129, "Br"}.issubset(store.smiles)
    assert store.metadata["max_nodes"] == 129
    rows = preparation.read_rows(input_root)
    expected_val = np.sort(np.random.default_rng(0).choice([1, 3, 5], 2, replace=False))
    with np.load(out / "rows.npz") as saved:
        np.testing.assert_array_equal(saved["train"], [0, 2])
        np.testing.assert_array_equal(saved["val"], expected_val)
        np.testing.assert_array_equal(saved["test"], [4])
        np.testing.assert_array_equal(saved["smiles"], rows["smiles"])
        np.testing.assert_array_equal(saved["inchikey14"], [s[:14] for s in rows["inchikey"]])
    selection = json.loads((out / "selection.json").read_text())
    assert selection["seed"] == selection["negative_bank_size"] == 0
    assert selection["validation_sha256"] == hashlib.sha256(expected_val.tobytes()).hexdigest()
    assert set(p.name for p in out.iterdir()) == {"graphs", "rows.npz", "selection.json"}


def test_fold_misalignment_rejected_before_building_graphs(input_root, tmp_path):
    np.save(input_root / "data/fold.npy", np.array(["train"] * 6))
    with pytest.raises(ValueError, match="fold alignment"):
        preparation.prepare_graphs(input_root, tmp_path / "prepared", workers=1, val_n=2)
    assert not (tmp_path / "prepared").exists()


def test_over_capacity_molecule_is_rejected_not_truncated(input_root, tmp_path):
    path = input_root / "msg15/MassSpecGym1.5_retrieval_candidates_formula.json"
    pool = json.loads(path.read_text())
    pool["C"].append("C" * 161)
    path.write_text(json.dumps(pool))
    with pytest.raises(ValueError, match="161 atoms.*max_atoms=160"):
        preparation.prepare_graphs(input_root, tmp_path / "prepared", workers=1, val_n=2)


def test_cli_uses_selected_defaults(input_root, tmp_path, monkeypatch):
    recorded = []
    monkeypatch.setattr(preparation, "prepare_graphs", lambda *a, **k: recorded.append((a, k)))
    out = tmp_path / "prepared"
    monkeypatch.setattr(sys, "argv", ["prepare", "--input-root", str(input_root), "--out", str(out)])
    preparation.main()
    assert recorded == [((input_root, out), {"workers": 16})]


@pytest.mark.parametrize("obsolete", [["--phase", "pca"], ["--dim", "2"], ["--seed", "1"],
                                      ["--val-n", "1"], ["--negative-bank-size", "16"],
                                      ["--max-atoms", "96"]])
def test_cli_rejects_removed_recipe_options(tmp_path, monkeypatch, obsolete):
    monkeypatch.setattr(sys, "argv", ["prepare", "--input-root", str(tmp_path),
                                      "--out", str(tmp_path / "prepared"), *obsolete])
    with pytest.raises(SystemExit) as exc:
        preparation.main()
    assert exc.value.code == 2


def test_json_helper_replaces_file_without_leaving_partial(tmp_path):
    path = tmp_path / "provenance.json"
    preparation.write_json(path, {"before": 1})
    preparation.write_json(path, {"after": [2, 3]})
    assert json.loads(path.read_text()) == {"after": [2, 3]}
    assert not path.with_suffix(".json.partial").exists()
