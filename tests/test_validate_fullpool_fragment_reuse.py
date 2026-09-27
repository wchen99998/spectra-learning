"""Every reused fragment must match after graph IDs and CSR offsets move."""

from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from atom_flare_data import build_graph_store
from glacier_fragment_data import build_fragment_store
from prepare_glacier_fullpool_index import prepare
import validate_fullpool_fragment_reuse as reuse


def _sha(path):
    return sha256(path.read_bytes()).hexdigest()


def _write_json(path, value):
    path.write_text(json.dumps(value) + "\n")


def _update(path, **values):
    metadata = json.loads(path.read_text())
    metadata.update(values)
    _write_json(path, metadata)


def _proposals(path, graphs, *, reverse=False):
    path.mkdir()
    definitions = {
        "CCCC(C)O": ([[5, 4, 3], [2, 1, 0]], [0.9, 0.6]),
        "c1ccccc1": ([[0, 1, 2], [3, 4, 5]], [0.8, 0.4]),
    }
    ids, fragment_offsets, member_offsets, members, intensities = [], [0], [0], [], []
    for graph_id in reversed(range(len(graphs))):
        sets, values = definitions.get(graphs.smiles[graph_id], ([], []))
        if reverse and graphs.smiles[graph_id] == "CCCC(C)O":
            sets, values = list(reversed(sets)), list(reversed(values))
        ids.append(graph_id)
        for atoms, intensity in zip(sets, values, strict=True):
            members.extend(atoms)
            member_offsets.append(len(members))
            intensities.append(intensity)
        fragment_offsets.append(len(intensities))
    np.savez(path / "shard.npz", graph_ids=np.array(ids, np.int32),
             frag_off=np.array(fragment_offsets, np.int64), mem_off=np.array(member_offsets, np.int64),
             atoms=np.array(members, np.int16), inten=np.array(intensities, np.float32))
    _write_json(path / "metadata.json", {
        "smiles_sha256": graphs.metadata["smiles_sha256"], "molecules": len(graphs),
        "selected_molecules": len(graphs), "complete": True,
        "completed_shards": ["shard.npz"], "unsupported_graph_ids": [],
    })


def _case(tmp_path, *, reverse=False):
    old = build_graph_store(["CCCC(C)O", "CCO", "c1ccccc1"], tmp_path / "old-graphs")
    full = build_graph_store([*old.smiles, "CCN", "NCC", "OCC"], tmp_path / "full-graphs")
    old_proposals, full_proposals = tmp_path / "old-proposals", tmp_path / "full-proposals"
    _proposals(old_proposals, old)
    _proposals(full_proposals, full, reverse=reverse)
    old_fragments = build_fragment_store(old, old_proposals, tmp_path / "old-fragments", f_max=4)
    full_fragments = build_fragment_store(full, full_proposals, tmp_path / "full-fragments", f_max=4)
    index = tmp_path / "index"
    prepare(old.directory, full.directory, old_fragments.directory, old_proposals, index, parts=2)
    return index, old_fragments, full_fragments, tmp_path / "report.json"


def _mutate(path, position, value):
    array = np.load(path)
    array[position] = value
    np.save(path, array)


def test_every_shared_molecule_matches_across_noncontiguous_ids_and_shifted_offsets(tmp_path, monkeypatch):
    index, old, full, out = _case(tmp_path)
    np.testing.assert_array_equal(np.load(index / "old_to_full.npy"), [0, 2, 5])
    assert old.frag_off[-2] != full.frag_off[-2]
    originals = {p: _sha(p) for root in (old.directory, full.directory, index)
                 for p in root.rglob("*") if p.is_file()}
    monkeypatch.setattr(reuse, "CHUNK_MOLECULES", 1)  # Includes zero boundary-member chunks.
    result = reuse.validate(index, full.directory, out)
    assert result == json.loads(out.read_text())
    assert result["status"] == "passed" and result["complete"]
    assert result["compared"] == {
        "molecules": 3, "fragments": 7, "memberships": 27, "boundary_memberships": 6,
    }
    assert result["additional_molecules"] == 3
    assert result["settings"] == {"f_max": 4, "min_atoms": 3, "min_frac": 0.05}
    assert result["mapping"]["sha256"] == _sha(index / "old_to_full.npy")
    assert result["old_fragments"]["metadata_sha256"] == _sha(old.directory / "metadata.json")
    assert result["full_fragments"]["metadata_sha256"] == _sha(full.directory / "metadata.json")
    assert result["index"]["metadata_sha256"] == _sha(index / "metadata.json")
    assert all(_sha(path) == digest for path, digest in originals.items())


@pytest.mark.parametrize("field,position,value", [
    ("inten", 1, np.nextafter(np.float32(0.9), np.float32(1.0))),
    ("mass_frac", 1, np.float32(0.333)), ("total_mass", 0, np.float32(100)),
    ("n_atoms", 1, 4), ("n_cuts", 1, 2), ("kind", 1, 0),
    ("atoms", 6, 2), ("bnd_atoms", 0, 2), ("unsupported", 2, True),
    ("mem_off", 2, 8), ("bnd_off", 2, 0), ("frag_off", 1, 2),
])
def test_exact_field_or_count_changes_fail_without_report(tmp_path, field, position, value):
    index, old, full, out = _case(tmp_path)
    _mutate(full.directory / f"{field}.npy", position, value)
    with pytest.raises(ValueError, match="differs|Invalid"):
        reuse.validate(index, full.directory, out)
    assert not out.exists() and not out.with_name(out.name + ".tmp").exists()


def test_fragment_order_is_part_of_identity(tmp_path):
    index, old, full, out = _case(tmp_path, reverse=True)
    with pytest.raises(ValueError, match="Shared fragment .* differs"):
        reuse.validate(index, full.directory, out)
    assert not out.exists()


def test_failure_after_successful_chunks_never_publishes_partial_pass(tmp_path, monkeypatch, capsys):
    index, old, full, out = _case(tmp_path)
    monkeypatch.setattr(reuse, "CHUNK_MOLECULES", 1)
    last_fragment = int(full.frag_off[-1] - 1)
    _mutate(full.directory / "inten.npy", last_fragment, np.float32(0.125))
    with pytest.raises(ValueError, match=r"IDs \[2, 3\)"):
        reuse.validate(index, full.directory, out)
    assert not out.exists() and not out.with_name(out.name + ".tmp").exists()


@pytest.mark.parametrize("field,value", [
    ("smiles_sha256", "wrong"), ("molecules", 7), ("schema", "wrong"),
    ("f_max", 8), ("min_atoms", 2), ("min_frac", 0.1),
])
def test_metadata_identity_or_filtering_mismatch_is_rejected(tmp_path, field, value):
    index, old, full, out = _case(tmp_path)
    _update(full.directory / "metadata.json", **{field: value})
    with pytest.raises(ValueError, match="identity|settings"):
        reuse.validate(index, full.directory, out)
    assert not out.exists()


def test_old_metadata_is_pinned_to_index_preparation(tmp_path):
    index, old, full, out = _case(tmp_path)
    _update(old.directory / "metadata.json", extra="changed")
    with pytest.raises(ValueError, match="changed since index preparation"):
        reuse.validate(index, full.directory, out)
    assert not out.exists()


@pytest.mark.parametrize("update_hash", [False, True])
def test_mapping_tampering_cannot_repeat_an_old_destination(tmp_path, update_hash):
    index, old, full, out = _case(tmp_path)
    mapping_path = index / "old_to_full.npy"
    np.save(mapping_path, np.array([0, 0, 5], np.int32))
    if update_hash:
        manifest = json.loads((index / "metadata.json").read_text())
        manifest["mapping_sha256"]["old_to_full.npy"] = _sha(mapping_path)
        _write_json(index / "metadata.json", manifest)
    with pytest.raises(ValueError, match="mapping"):
        reuse.validate(index, full.directory, out)
    assert not out.exists()


def test_dtype_changes_are_rejected_even_with_equal_values(tmp_path):
    index, old, full, out = _case(tmp_path)
    np.save(full.directory / "inten.npy", np.asarray(full.inten, np.float64))
    with pytest.raises(ValueError, match="dtype"):
        reuse.validate(index, full.directory, out)
    assert not out.exists()


def test_existing_report_is_refused_and_preserved(tmp_path):
    index, old, full, out = _case(tmp_path)
    out.write_text("existing report")
    with pytest.raises(FileExistsError):
        reuse.validate(index, full.directory, out)
    assert out.read_text() == "existing report"
