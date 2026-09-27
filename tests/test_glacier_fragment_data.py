"""Original-order chemistry, complete cache coverage and JAX fragment packing."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from atom_flare_data import build_graph_store
from glacier_fragment_data import (
    FragmentStore, batch_fragments, build_fragment_record, build_fragment_store,
)


def _record(smiles, atomsets, intensities, **kwargs):
    return build_fragment_record(
        smiles, [np.asarray(atoms, np.int16) for atoms in atomsets], intensities, **kwargs,
    )


def _proposals(directory, graphs, shards, unsupported=()):
    directory.mkdir()
    names = []
    for index, rows in enumerate(shards):
        name = f"part-{index:04d}.npz"
        ids, fragments, members, atoms, intensity = [], [0], [0], [], []
        for graph_id, sets, values in rows:
            ids.append(graph_id)
            fragments.append(fragments[-1] + len(sets))
            for atomset, value in zip(sets, values, strict=True):
                atoms.extend(atomset)
                members.append(len(atoms))
                intensity.append(value)
        np.savez(directory / name, graph_ids=np.asarray(ids, np.int32),
                 frag_off=np.asarray(fragments, np.int64), mem_off=np.asarray(members, np.int64),
                 atoms=np.asarray(atoms, np.int16), inten=np.asarray(intensity, np.float32))
        names.append(name)
    metadata = dict(smiles_sha256=graphs.metadata["smiles_sha256"], completed_shards=names,
                    unsupported_graph_ids=list(unsupported))
    (directory / "metadata.json").write_text(json.dumps(metadata))
    return directory


def test_converter_has_no_torch_runtime_dependency():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import glacier_fragment_data; assert 'torch' not in sys.modules"],
        env={**os.environ, "PYTHONPATH": str(scripts)}, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_filter_order_deduplication_boundary_and_intact_anchor():
    result = _record("CCCC(C)O", [[5, 4, 3], [3, 4, 5], [0], [2, 1, 0], list(range(6))],
                     [0.9, 0.8, 0.7, 0.6, 0.5])
    assert result["F"] == 3
    np.testing.assert_array_equal(result["atoms"], [0, 1, 2, 3, 4, 5, 3, 4, 5, 0, 1, 2])
    np.testing.assert_array_equal(result["atom_len"], [6, 3, 3])
    np.testing.assert_array_equal(result["bnd_atoms"], [3, 2])
    np.testing.assert_array_equal(result["bnd_len"], [0, 1, 1])
    np.testing.assert_array_equal(result["n_cuts"], [0, 1, 1])
    np.testing.assert_array_equal(result["kind"], [0, 1, 1])
    np.testing.assert_array_equal(result["inten"], np.asarray([1, 0.9, 0.6], np.float32))
    np.testing.assert_allclose(result["mass_frac"][1:].sum(), 1)
    capped = _record("CCCC(C)O", [[3, 4, 5], [0, 1, 2]], [0.9, 0.6], f_max=2)
    np.testing.assert_array_equal(capped["atoms"], result["atoms"][:9])
    filtered = _record("CCCC(C)O", [[0, 1, 2]], [0.9], min_frac=0.9)
    assert filtered["F"] == 1
    intact_only = _record("CC", [[0]], [0.9], min_atoms=1, f_max=1)
    assert intact_only["F"] == 1


def test_cut_cap_and_disconnected_most_common_isotope_mass():
    capped = _record("C" * 20, [list(range(0, 20, 2))], [0.8])
    assert capped["n_cuts"][1] == 7
    h = 1.00782503207
    plain = _record("CCCC.[Na+]", [[0, 1, 2]], [0.5])
    labelled = _record("[13CH3]CCC.[Na+]", [[0, 1, 2]], [0.5])
    np.testing.assert_array_equal(plain["mass_frac"], labelled["mass_frac"])
    np.testing.assert_array_equal(plain["total"], labelled["total"])
    assert plain["total"] > 4 * 12 + 10 * h  # Disconnected sodium remains in denominator.


@pytest.mark.parametrize("atoms,intensity,error", [
    ([[0, 1, 20]], [0.2], "outside"),
    ([[0, -1, 2]], [0.2], "outside"),
    ([[0, 1, 2]], [float("nan")], "intensity"),
    ([[0, 1, 2]], [1.1], "intensity"),
])
def test_corrupt_predictions_raise(atoms, intensity, error):
    with pytest.raises(ValueError, match=error):
        _record("CCCC", atoms, intensity)


def test_cache_reorders_shards_and_batches_without_partial_fragments(tmp_path):
    graphs = build_graph_store(["CCCC(C)O", "CCO", "c1ccccc1"], tmp_path / "graphs")
    branch, small, ring = graphs.ids(["CCCC(C)O", "CCO", "c1ccccc1"])
    proposals = _proposals(tmp_path / "proposals", graphs, [
        [(ring, [[0, 1, 2]], [0.4]), (small, [], [])],
        [(branch, [[3, 4, 5], [0, 1, 2]], [0.9, 0.6])],
    ])
    store = build_fragment_store(graphs, proposals, tmp_path / "fragments", f_max=4)
    assert isinstance(store.atoms, np.memmap)
    assert len(store) == 3
    assert store.metadata["intact_only_molecules"] == 1
    np.testing.assert_array_equal(store.counts[[branch, small, ring]], [3, 1, 2])
    assert store.mem_off[-1] == len(store.atoms)
    assert store.bnd_off[-1] == len(store.bnd_atoms)
    reopened = FragmentStore(tmp_path / "fragments")
    batch = batch_fragments(reopened, [ring, branch, ring], graph_capacity=4, atom_cap=7)
    np.testing.assert_array_equal(batch["fragment_mask"],
                                  [[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]])
    np.testing.assert_array_equal(batch["member_mean"][0], batch["member_mean"][2])
    np.testing.assert_allclose(batch["member_mean"].sum(-1), batch["fragment_mask"])
    np.testing.assert_array_equal(batch["boundary_mean"][:, 0], 0)
    np.testing.assert_allclose(batch["boundary_mean"][0, 1, [0, 2]], 0.5)
    np.testing.assert_allclose(batch["fragment_log_n_atoms"][0, :2], np.log([6, 3]))
    np.testing.assert_array_equal(batch["fragment_intensity"][1, :3], np.asarray([1, 0.9, 0.6], np.float32))
    np.testing.assert_array_equal(batch["member_mean"][:, :, 6], 0)
    # Pooling uses original molecule atom slots and produces the exact atom means.
    hidden = np.arange(4 * 7 * 2).reshape(4, 7, 2)
    pooled = np.einsum("gfa,gad->gfd", batch["member_mean"], hidden)
    np.testing.assert_allclose(pooled[1, 1], hidden[1, [3, 4, 5]].mean(0))
    assert len(build_fragment_store(graphs, proposals, tmp_path / "fragments", f_max=4)) == 3
    with pytest.raises(ValueError, match="differs"):
        build_fragment_store(graphs, proposals, tmp_path / "fragments", f_max=8)
    with pytest.raises(ValueError, match="atom_cap"):
        batch_fragments(store, [ring], graph_capacity=1, atom_cap=5)
    with pytest.raises(ValueError, match="capacity"):
        batch_fragments(store, [ring, small], graph_capacity=1, atom_cap=6)
    with pytest.raises(ValueError, match="outside"):
        batch_fragments(store, [-1], graph_capacity=1, atom_cap=6)


@pytest.mark.parametrize("failure", ["missing", "duplicate", "fingerprint", "offsets"])
def test_cache_rejects_incomplete_or_misaligned_proposals(tmp_path, failure):
    graphs = build_graph_store(["CCC", "CCCC"], tmp_path / "graphs")
    rows = [(0, [], []), (1, [], [])]
    if failure == "missing":
        rows = rows[:1]
    elif failure == "duplicate":
        rows[1] = rows[0]
    proposals = _proposals(tmp_path / "proposals", graphs, [rows])
    if failure == "fingerprint":
        metadata = json.loads((proposals / "metadata.json").read_text())
        metadata["smiles_sha256"] = "different"
        (proposals / "metadata.json").write_text(json.dumps(metadata))
    elif failure == "offsets":
        with np.load(proposals / "part-0000.npz") as source:
            shard = {key: source[key] for key in source.files}
        shard["mem_off"] = np.asarray([1], np.int64)
        np.savez(proposals / "part-0000.npz", **shard)
    with pytest.raises(ValueError, match={"missing": "Missing", "duplicate": "Duplicate",
                                        "fingerprint": "fingerprint", "offsets": "offsets"}[failure]):
        build_fragment_store(graphs, proposals, tmp_path / "out")
    assert not (tmp_path / "out" / "metadata.json").exists()


def test_explicit_graphormer_unsupported_fallback_is_recorded(tmp_path):
    graphs = build_graph_store(["C" * 101], tmp_path / "graphs")
    proposals = _proposals(tmp_path / "proposals", graphs, [[(0, [], [])]], unsupported=[0])
    store = build_fragment_store(graphs, proposals, tmp_path / "fragments")
    assert store.unsupported[0]
    assert store.metadata["unsupported_molecules"] == 1
    assert store.counts[0] == 1
    batch = batch_fragments(store, [0], graph_capacity=1, atom_cap=128)
    assert batch["fragment_mask"].sum() == 1
    np.testing.assert_allclose(batch["member_mean"][0, 0, :101], 1 / 101)
    assert not batch["member_mean"][0, 0, 101:].any()


def test_parallel_shards_match_serial_array_bytes_and_metadata(tmp_path):
    graphs = build_graph_store(
        ["CCCC(C)O", "CCO", "c1ccccc1", "CCCC", "[13CH3]CCC.[Na+]", "C" * 101],
        tmp_path / "graphs",
    )
    branch, small, ring, chain, labelled, unsupported = graphs.ids(
        ["CCCC(C)O", "CCO", "c1ccccc1", "CCCC", "[13CH3]CCC.[Na+]", "C" * 101])
    proposals = _proposals(tmp_path / "proposals", graphs, [
        [(ring, [[0, 1, 2], [3, 4, 5]], [0.8, 0.6]), (small, [], [])],
        [(unsupported, [], [])],
        [(branch, [[3, 4, 5], [5, 4, 3], [0], [0, 1, 2]], [0.9, 0.8, 0.7, 0.6])],
        [(chain, [[0, 1, 2]], [0.4]), (labelled, [[0, 1, 2]], [0.3])],
    ], unsupported=[int(unsupported)])
    serial = build_fragment_store(graphs, proposals, tmp_path / "serial", f_max=4)
    parallel = build_fragment_store(graphs, proposals, tmp_path / "parallel", f_max=4, workers=2)
    assert parallel.metadata == serial.metadata
    assert parallel.metadata["unsupported_molecules"] == 1
    np.testing.assert_array_equal(parallel.counts[[branch, small, ring]], [3, 1, 3])
    filenames = {path.name for path in serial.directory.iterdir()}
    assert filenames == {path.name for path in parallel.directory.iterdir()}
    for name in filenames:
        assert (parallel.directory / name).read_bytes() == (serial.directory / name).read_bytes(), name
    assert not list(serial.directory.glob("processed-*"))
    assert not list(parallel.directory.glob("processed-*"))


@pytest.mark.parametrize("failure", ["duplicate", "missing", "chemistry"])
def test_parallel_shard_failure_leaves_no_complete_artifact_or_temporary_files(tmp_path, failure):
    graphs = build_graph_store(["CCC", "CCCC", "CCCCC", "CCCCCC", "CCN", "CCO"], tmp_path / "graphs")
    shards = [[(i, [[0, 1, 2]], [0.5])] for i in range(len(graphs))]
    if failure == "duplicate":
        # More than the four buffered tasks lets earlier shards finish before
        # the parent detects a duplicate graph ID in a later source shard.
        shards.append([(0, [], [])])
    elif failure == "missing":
        shards.pop()
    else:
        shards[-1] = [(len(graphs) - 1, [[0, 1, 99]], [0.5])]
    proposals = _proposals(tmp_path / "proposals", graphs, shards)
    original = {path.name: path.read_bytes() for path in proposals.iterdir()}
    out = tmp_path / "fragments"
    with pytest.raises(ValueError, match={"duplicate": "Duplicate", "missing": "Missing",
                                         "chemistry": "outside original SMILES"}[failure]):
        build_fragment_store(graphs, proposals, out, workers=2)
    assert out.is_dir() and list(out.iterdir()) == []
    assert {path.name: path.read_bytes() for path in proposals.iterdir()} == original


def test_numerical_parity_with_upstream_glacier_constructor():
    source = Path(os.environ.get(
        "JEPA_FPS_GLACIER_SOURCE", "/tmp/jepa-fps-glacier-audit-20260918/scripts/flare_frag.py",
    ))
    if not source.is_file():
        pytest.skip("Set JEPA_FPS_GLACIER_SOURCE to upstream scripts/flare_frag.py for parity")
    pytest.importorskip("torch_geometric")
    spec = importlib.util.spec_from_file_location("upstream_glacier_fragments", source)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    for smiles, sets, values in [
        ("CCCC(C)O", [[3, 4, 5], [5, 4, 3], [0, 1, 2], [0]], [0.9, 0.8, 0.5, 0.1]),
        ("c1ccccc1", [[0, 1, 2], [3, 4, 5]], [0.8, 0.6]),
        ("[13CH3]CCC.[Na+]", [[0, 1, 2]], [0.2]),
        ("C" * 20, [list(range(0, 20, 2))], [0.7]),
        ("CCO", [], []),
    ]:
        for cap in [2, 128]:
            actual = _record(smiles, sets, values, f_max=cap)
            expected = upstream._build_one_glacier(smiles, sets, cap, 3, 0.05, values)
            assert actual.keys() == expected.keys()
            for key in actual:
                np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
