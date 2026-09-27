"""Mapped graph reuse must establish tensor identity, not only graph counts."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.atom_flare_data import build_graph_store
from scripts import validate_fullpool_graph_reuse as audit


def _sha(path):
    return sha256(path.read_bytes()).hexdigest()


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def _case(tmp_path):
    old = build_graph_store(["CCO", "O"], tmp_path / "old-prepared/graphs")
    full = build_graph_store(["CC", "CCO", "CO", "O", "S"], tmp_path / "full-prepared/graphs")
    fragments = []
    for label, graph in (("old", old), ("full", full)):
        path = tmp_path / f"{label}-fragments"
        metadata = {"schema": "glacier_fragments_original_order_v1", "molecules": len(graph),
                    "smiles_sha256": graph.metadata["smiles_sha256"], "fragments": len(graph),
                    "memberships": graph.metadata["atoms"], "boundary_memberships": 0,
                    "f_max": 128, "min_atoms": 3, "min_frac": 0.05}
        _write(path / "metadata.json", metadata)
        fragments.append((path, metadata))
    (old_fragments, old_fragment_meta), (full_fragments, full_fragment_meta) = fragments
    index = tmp_path / "index"
    index.mkdir()
    np.save(index / "old_to_full.npy", np.asarray([1, 3], np.int32))
    metadata = {"schema": "glacier_fullpool_index_v1", "complete": True,
                "old_graphs": str(old.directory), "full_graphs": str(full.directory),
                "existing_fragments": str(old_fragments), "reused_molecules": len(old), "full_molecules": len(full),
                "old_smiles_sha256": old.metadata["smiles_sha256"], "full_smiles_sha256": full.metadata["smiles_sha256"],
                "input_metadata_sha256": {str(path / "metadata.json"): _sha(path / "metadata.json")
                                          for path in (old.directory, full.directory, old_fragments)},
                "mapping_sha256": {"old_to_full.npy": _sha(index / "old_to_full.npy")}}
    _write(index / "metadata.json", metadata)
    source = tmp_path / "original-head/manifest.json"
    _write(source, {"graphs": old.metadata, "fragments": old_fragment_meta,
                    "args": {"prepared": str(old.directory.parent), "fragments": str(old_fragments)}})
    fragment_report = tmp_path / "fragment_reuse.json"
    _write(fragment_report, {
        "schema": "glacier_fullpool_fragment_reuse_validation_v1", "complete": True, "status": "passed",
        "fields": sorted(audit.FRAGMENT_FIELDS),
        "index": {"path": str(index), "metadata_sha256": _sha(index / "metadata.json")},
        "mapping": {"path": str(index / "old_to_full.npy"), "sha256": _sha(index / "old_to_full.npy")},
        "old_fragments": {"path": str(old_fragments), "metadata_sha256": _sha(old_fragments / "metadata.json"),
                          "molecules": len(old), "smiles_sha256": old.metadata["smiles_sha256"]},
        "full_fragments": {"path": str(full_fragments), "metadata_sha256": _sha(full_fragments / "metadata.json"),
                           "molecules": len(full), "smiles_sha256": full.metadata["smiles_sha256"]},
        "compared": {"molecules": len(old), **{k: old_fragment_meta[k] for k in
                     ("fragments", "memberships", "boundary_memberships")}},
        "additional_molecules": len(full) - len(old),
        "settings": {k: old_fragment_meta[k] for k in ("f_max", "min_atoms", "min_frac")},
    })
    return index, source, fragment_report, tmp_path / "report.json", old, full


def _mutate(path, position, value):
    array = np.load(path)
    array[position] = value
    np.save(path, array)


def _repin_mapping(index, proof):
    metadata = json.loads((index / "metadata.json").read_text())
    metadata["mapping_sha256"]["old_to_full.npy"] = _sha(index / "old_to_full.npy")
    _write(index / "metadata.json", metadata)
    report = json.loads(proof.read_text())
    report["mapping"]["sha256"] = _sha(index / "old_to_full.npy")
    report["index"]["metadata_sha256"] = _sha(index / "metadata.json")
    _write(proof, report)


def test_every_shared_tensor_matches_with_shifted_offsets_edgeless_graph_and_bounded_loads(tmp_path, monkeypatch):
    index, source, proof, out, old, full = _case(tmp_path)
    originals = {p: _sha(p) for p in tmp_path.rglob("*") if p.is_file()}
    loads = []
    original_load = np.load

    def checked_load(path, *args, **kwargs):
        loads.append((Path(path).name, kwargs))
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(audit.np, "load", checked_load)
    result = audit.validate(index, source, proof, out, chunk_molecules=1)
    assert result == json.loads(out.read_text())
    assert result["complete"] and result["status"] == "passed"
    assert result["compared"] == {"molecules": 2, "nodes": 4, "directed_edges": 4}
    assert result["fields"] == list(audit.FIELDS)
    assert result["additional_molecules"] == 3
    assert len(loads) == 15
    assert all(kwargs == {"mmap_mode": "r", "allow_pickle": False} for _, kwargs in loads)
    for name, graph in (("old_graphs", old), ("full_graphs", full)):
        assert result[name]["metadata"] == graph.metadata
        assert result[name]["metadata_sha256"] == _sha(graph.directory / "metadata.json")
    assert result["source_manifest"]["sha256"] == _sha(source)
    assert result["fragment_reuse"]["sha256"] == _sha(proof)
    for field in audit.FIELDS:
        assert result["shared_payload_sha256"][field] == sha256(getattr(old, field).tobytes()).hexdigest()
    assert all(_sha(path) == digest for path, digest in originals.items())


def test_chunk_size_does_not_change_proof_and_new_only_payloads_are_outside_scope(tmp_path):
    index, source, proof, out, old, full = _case(tmp_path)
    first = audit.validate(index, source, proof, out, chunk_molecules=1)
    _mutate(full.directory / "nodes.npy", (0, 0), np.float32(123))
    second = audit.validate(index, source, proof, tmp_path / "second.json", chunk_molecules=3)
    assert first["shared_payload_sha256"] == second["shared_payload_sha256"]
    assert first["compared"] == second["compared"]


@pytest.mark.parametrize("field", audit.FIELDS)
def test_same_counts_but_any_changed_shared_payload_fails(tmp_path, field):
    index, source, proof, out, old, full = _case(tmp_path)
    offset = int(full.node_offsets[1] if field in ("nodes", "afrac") else full.edge_offsets[1])
    position = (offset, 0) if field in ("nodes", "edges", "afrac") else offset
    value = getattr(full, field)[position]
    replacement = np.nextafter(value, np.float32(np.inf)) if value.dtype.kind == "f" else (int(value) + 1) % 3
    _mutate(full.directory / f"{field}.npy", position, replacement)
    with pytest.raises(ValueError, match=f"Shared graph {field} differs"):
        audit.validate(index, source, proof, out, chunk_molecules=1)
    assert not out.exists() and not out.with_name(out.name + ".tmp").exists()


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_identical_nonfinite_payloads_cannot_pass(tmp_path, value):
    index, source, proof, out, old, full = _case(tmp_path)
    for graph, position in ((old, 0), (full, int(full.node_offsets[1]))):
        _mutate(graph.directory / "nodes.npy", (position, 0), value)
    with pytest.raises(ValueError, match="nonfinite"):
        audit.validate(index, source, proof, out)
    assert not out.exists()


def test_signed_zero_difference_is_detected_by_exact_byte_comparison(tmp_path):
    index, source, proof, out, old, full = _case(tmp_path)
    _mutate(old.directory / "nodes.npy", (0, 0), np.float32(0.0))
    _mutate(full.directory / "nodes.npy", (int(full.node_offsets[1]), 0), np.float32(-0.0))
    with pytest.raises(ValueError, match="Shared graph nodes differs"):
        audit.validate(index, source, proof, out)


@pytest.mark.parametrize("field", ["senders", "receivers"])
def test_identical_endpoints_outside_own_molecule_fail_even_if_inside_chunk(tmp_path, field):
    index, source, proof, out, old, full = _case(tmp_path)
    _mutate(old.directory / f"{field}.npy", 0, 3)
    _mutate(full.directory / f"{field}.npy", int(full.edge_offsets[1]), 3)
    with pytest.raises(ValueError, match="Invalid molecule-local"):
        audit.validate(index, source, proof, out, chunk_molecules=2)


@pytest.mark.parametrize("corruption", ["dtype", "shape", "offset_endpoint", "offset_count"])
def test_bad_array_layout_or_offsets_fail(tmp_path, corruption):
    index, source, proof, out, old, full = _case(tmp_path)
    if corruption == "dtype":
        np.save(full.directory / "nodes.npy", np.asarray(full.nodes, np.float64))
    elif corruption == "shape":
        np.save(full.directory / "nodes.npy", np.asarray(full.nodes[:, :-1]))
    elif corruption == "offset_endpoint":
        _mutate(full.directory / "node_offsets.npy", -1, 100)
    else:
        _mutate(full.directory / "node_offsets.npy", 2, int(full.node_offsets[2]) + 1)
    with pytest.raises(ValueError, match="Invalid|counts differ"):
        audit.validate(index, source, proof, out)
    assert not out.exists()


@pytest.mark.parametrize("ids,repin", [([2, 3], False), ([1, 1], True), ([1, 5], True), ([3, 1], True)])
def test_mapping_hash_order_uniqueness_and_bounds_are_enforced(tmp_path, ids, repin):
    index, source, proof, out, old, full = _case(tmp_path)
    np.save(index / "old_to_full.npy", np.asarray(ids, np.int32))
    if repin:
        _repin_mapping(index, proof)
    with pytest.raises(ValueError, match="mapping"):
        audit.validate(index, source, proof, out, chunk_molecules=1)
    assert not out.exists()


@pytest.mark.parametrize("target,key", [("source", "graphs"), ("source", "fragments"),
                                        ("proof", "index"), ("proof", "mapping"),
                                        ("proof", "old_fragments"), ("proof", "full_fragments")])
def test_source_head_and_linked_proof_identities_must_match(tmp_path, target, key):
    index, source, proof, out, old, full = _case(tmp_path)
    path = source if target == "source" else proof
    value = json.loads(path.read_text())
    if target == "source":
        value[key]["molecules"] += 1
    elif key == "mapping":
        value[key]["sha256"] = "wrong"
    else:
        value[key]["metadata_sha256"] = "wrong"
    _write(path, value)
    with pytest.raises(ValueError, match="manifest|identity"):
        audit.validate(index, source, proof, out)
    assert not out.exists()


def test_late_comparison_failure_never_publishes_partial_success(tmp_path):
    index, source, proof, out, old, full = _case(tmp_path)
    _mutate(full.directory / "afrac.npy", (int(full.node_offsets[3]), 0), np.float32(.125))
    with pytest.raises(ValueError, match=r"old IDs \[1, 2\)"):
        audit.validate(index, source, proof, out, chunk_molecules=1)
    assert not out.exists() and not out.with_name(out.name + ".tmp").exists()


def test_existing_report_is_preserved_and_failed_publish_cleans_owned_temporary(tmp_path, monkeypatch):
    index, source, proof, out, old, full = _case(tmp_path)
    out.write_text("already audited")
    with pytest.raises(FileExistsError):
        audit.validate(index, source, proof, out)
    assert out.read_text() == "already audited"
    out.unlink()

    def fail_replace(self, target):
        raise OSError("synthetic publish failure")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="publish failure"):
        audit.validate(index, source, proof, out)
    assert not out.exists() and not out.with_name(out.name + ".tmp").exists()
