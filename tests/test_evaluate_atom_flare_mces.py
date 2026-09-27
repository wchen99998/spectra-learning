"""Selected-only MCES reports and explicitly requested historical comparisons."""

import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


@pytest.mark.parametrize(("with_baseline", "with_seed"), [(False, True), (True, True), (False, False)])
def test_selected_report(tmp_path, monkeypatch, with_baseline, with_seed):
    path = Path(__file__).resolve().parents[1] / "scripts/evaluate_atom_flare_mces.py"
    spec = importlib.util.spec_from_file_location("evaluate_mces_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    seed = tmp_path / "results/local_moljepa_atom_latest/mces_seed.json"
    seed.parent.mkdir(parents=True)
    seed.write_text(json.dumps({"CC||CCC": 1.0}))
    seed.with_name("mces_seed_provenance.json").write_text(json.dumps({
        "threshold": 15, "sha256": hashlib.sha256(seed.read_bytes()).hexdigest(),
        "compatibility": {"myopic_mces": importlib.metadata.version("myopic-mces"),
                          "rdkit": importlib.metadata.version("rdkit")},
    }))
    selected, baseline = tmp_path / "selected", tmp_path / "baseline"
    common = {"rows": np.array([7, 11]), "truth_smiles": np.array(["CC", "CC"]),
              "inchikey14": np.array(["compound_a", "compound_a"]),
              "candidate_counts": np.array([2, 2])}
    selected_top = ["CC", "CCC"] if with_seed else ["CC", "CC"]
    for root, top in ((selected, selected_top), (baseline, ["CC", "CC"])):
        (root / "run").mkdir(parents=True)
        np.savez(root / "run/test_ranks.npz", **common, top_smiles=np.array(top))
    argv = [str(path), "--root", str(selected), "--workers", "1"]
    if with_baseline:
        argv += ["--baseline", str(baseline)]
    if with_seed:
        argv += ["--seed-cache", str(seed)]
    else:
        seed.unlink()
        seed.with_name("mces_seed_provenance.json").unlink()
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
    expected = {"global_glacier" if with_baseline else "selected":
                {"mces_at_1": .5 if with_seed else 0., "n": 2, "n_failed": 0}}
    if with_baseline:
        expected["baseline"] = {"mces_at_1": 0., "n": 2, "n_failed": 0}
    report = selected / "full_results"
    assert json.loads((report / "mces_summary.json").read_text()) == expected
    protocol = json.loads((report / "mces_protocol.json").read_text())
    assert set(protocol["rank_files_sha256"]) == set(expected)
    assert protocol["threshold"] == 15 and protocol["time_limit"] is None
    assert protocol["seed_cache"] == (str(seed) if with_seed else None)
    with np.load(report / "mces_per_query.npz") as values:
        np.testing.assert_array_equal(values["rows"], common["rows"])
        np.testing.assert_array_equal(values["global_glacier" if with_baseline else "selected"], [0., 1. if with_seed else 0.])
