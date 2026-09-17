import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture
def wrapper(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("frozen_flare_wrapper_test", scripts / "run_frozen_flare.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cache_path(tmp_path):
    (tmp_path / "smiles.json").write_text(json.dumps(["CC"]))
    (tmp_path / "metadata.json").write_text(json.dumps({"checkpoint_sha256": "a" * 64}))
    np.save(tmp_path / "offsets.npy", np.array([0, 2], dtype=np.int64))
    np.save(tmp_path / "atom_embeddings.float16.npy", np.ones((2, 4), dtype=np.float16))
    return tmp_path


@pytest.mark.parametrize("filename", ["09_jepa_flare.py", "10_candidate_flare.py"])
def test_wrapper_preserves_imported_spectrum_encoder_and_replaces_only_molecule_path(
    wrapper, cache_path, monkeypatch, filename,
):
    class OriginalSpectrum(torch.nn.Module):
        pass

    molecular_module = SimpleNamespace()
    hybrid = SimpleNamespace(HybridSpectrumEncoder=OriginalSpectrum, fm=molecular_module)
    stage = hybrid if filename == "09_jepa_flare.py" else SimpleNamespace(hybrid=hybrid)
    called = []

    def run_stage():
        assert hybrid.HybridSpectrumEncoder is OriginalSpectrum
        graphs = molecular_module.build_graph_cache(["CC"], "unused.pt", 1)
        atom_map = molecular_module.MolGNN(512, 8, 3, .8, feature_set="basic")
        assert isinstance(atom_map, wrapper.AtomCompatibility)
        assert set(atom_map.state_dict()) == {"out.weight"}
        assert atom_map.out.weight.shape == (8, 4)
        atoms = graphs["CC"][0]
        atom_map.train()
        torch.testing.assert_close(atom_map(atoms), atom_map(atoms), atol=0, rtol=0)
        assert sys.argv == [str(cache_path / filename), "--mode", "evidence"]
        called.append(True)

    stage.main = run_stage
    spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module: None))
    monkeypatch.setattr(wrapper.importlib.util, "spec_from_file_location", lambda *args: spec)
    monkeypatch.setattr(wrapper.importlib.util, "module_from_spec", lambda spec: stage)
    monkeypatch.setattr(sys, "argv", [
        "run_frozen_flare.py", "--source", str(cache_path / filename),
        "--atom-cache", str(cache_path), "--expected-checkpoint-sha256", "a" * 64,
        "--", "--mode", "evidence",
    ])
    wrapper.main()
    assert called == [True]


def test_wrapper_requires_the_launchers_expected_encoder_checkpoint(wrapper, cache_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "run_frozen_flare.py", "--source", "10_candidate_flare.py",
        "--atom-cache", str(cache_path), "--expected-checkpoint-sha256", "b" * 64,
    ])
    with pytest.raises(ValueError, match="expected pretrained checkpoint SHA-256"):
        wrapper.main()
