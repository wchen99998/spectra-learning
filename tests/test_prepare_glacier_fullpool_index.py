"""Exact graph-ID remapping and provenance checks for missing-fragment indexing."""

from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_glacier_fullpool_index import graph_counts, prepare, subset_mapping


def _digest(values):
    return sha256(b"".join(value.encode() + b"\0" for value in values)).hexdigest()


def _json(path, value):
    path.write_text(json.dumps(value) + "\n")


def _index(path, smiles, counts):
    path.mkdir()
    metadata = {
        "schema": "atom_flare_55dfbdf_v1", "molecules": len(smiles),
        "smiles_sha256": _digest(smiles),
    }
    for kind, column, total in [("node", 0, "atoms"), ("edge", 1, "directed_edges")]:
        offsets = np.concatenate(([0], np.cumsum([counts[s][column] for s in smiles])))
        np.save(path / f"{kind}_offsets.npy", offsets)
        metadata[total] = int(offsets[-1])
    _json(path / "smiles.json", smiles)
    _json(path / "metadata.json", metadata)


def _case(tmp_path, *, no_missing=False):
    # Preserve distinct original spellings, including stereochemistry; no
    # canonicalization may collapse CCO and OCC or change their atom ordering.
    counts = {"C": (1, 0), "CC": (2, 2), "CCO": (3, 4), "CO": (2, 2),
              "N[C@@H](C)C(=O)O": (6, 10), "O": (1, 0), "OCC": (3, 4)}
    full = sorted(counts)
    old = full if no_missing else ["CC", "N[C@@H](C)C(=O)O", "OCC"]
    paths = {name: tmp_path / name for name in ("old", "full", "fragments", "proposals", "out")}
    _index(paths["old"], old, counts)
    _index(paths["full"], full, counts)
    for label in ("fragments", "proposals"):
        paths[label].mkdir()
        metadata = {"smiles_sha256": _digest(old), "molecules": len(old)}
        if label == "proposals":
            metadata.update(complete=True, selected_molecules=len(old))
        _json(paths[label] / "metadata.json", metadata)
    fragment_path = paths["fragments"] / "metadata.json"
    fragment_metadata = json.loads(fragment_path.read_text())
    proposal_metadata = json.loads((paths["proposals"] / "metadata.json").read_text())
    fragment_metadata["proposal_metadata_sha256"] = sha256(
        json.dumps(proposal_metadata, sort_keys=True, separators=(",", ":")).encode(),
    ).hexdigest()
    _json(fragment_path, fragment_metadata)
    return paths, old, full


def _prepare(paths, *, parts=3):
    return prepare(paths["old"], paths["full"], paths["fragments"],
                   paths["proposals"], paths["out"], parts=parts)


def test_sorted_subset_mapping_preserves_exact_ids():
    old_to_full, full_to_old = subset_mapping(["CC", "CO", "O"], ["C", "CC", "CCO", "CO", "N", "O", "OCC"])
    np.testing.assert_array_equal(old_to_full, [1, 3, 5])
    np.testing.assert_array_equal(full_to_old, [-1, 0, -1, 1, -1, 2, -1])
    assert old_to_full.dtype == full_to_old.dtype == np.int32


@pytest.mark.parametrize("old", [["B", "C"], ["C", "CC"], ["C", "P"]])
def test_missing_old_molecule_is_rejected_at_any_position(old):
    with pytest.raises(ValueError, match="Old molecule is absent"):
        subset_mapping(old, ["C", "CN", "O"])


def test_prepare_partitions_cover_only_missing_ids_and_preserve_original_smiles(tmp_path):
    paths, old, full = _case(tmp_path)
    result = _prepare(paths)
    expected_shared = np.array([full.index(s) for s in old], np.int32)
    expected_missing = np.array([i for i, s in enumerate(full) if s not in old], np.int32)
    old_to_full = np.load(paths["out"] / "old_to_full.npy")
    full_to_old = np.load(paths["out"] / "full_to_old.npy")
    missing = np.load(paths["out"] / "missing_full_ids.npy")
    np.testing.assert_array_equal(old_to_full, expected_shared)
    np.testing.assert_array_equal(full_to_old[expected_shared], np.arange(len(old)))
    np.testing.assert_array_equal(full_to_old[expected_missing], -1)
    np.testing.assert_array_equal(missing, expected_missing)
    assert result["reused_molecules"] == len(old)
    assert result["missing_molecules"] == len(full) - len(old)
    assert result["full_molecules"] == len(full)
    assert result["shared_node_and_edge_counts_equal"]
    assert not result["graph_feature_arrays_compared"]
    assert not result["new_proposals_generated"] and not result["training_performed"]
    assert len(result["partitions"]) == 3
    partition_ids = []
    for partition in result["partitions"]:
        directory = Path(partition["directory"])
        ids = np.load(directory / "local_to_full.npy")
        values = json.loads((directory / "smiles.json").read_text())
        metadata = json.loads((directory / "metadata.json").read_text())
        assert len(ids) > 0 and ids.dtype == np.int32
        assert values == [full[int(i)] for i in ids]
        assert metadata["molecules"] == len(ids)
        assert metadata["smiles_sha256"] == _digest(values)
        assert metadata["full_graph_smiles_sha256"] == _digest(full)
        assert metadata["local_to_full_sha256"] == sha256((directory / "local_to_full.npy").read_bytes()).hexdigest()
        assert metadata["atom_order"] == "Chem.MolFromSmiles(original_graph_store_smiles)"
        assert not metadata["contains_graph_feature_arrays"]
        assert {p.name for p in directory.iterdir()} == {"local_to_full.npy", "smiles.json", "metadata.json"}
        partition_ids.extend(ids.tolist())
    assert partition_ids == expected_missing.tolist()
    assert sorted(partition_ids + old_to_full.tolist()) == list(range(len(full)))
    assert result == json.loads((paths["out"] / "metadata.json").read_text())
    for name, digest in result["mapping_sha256"].items():
        assert digest == sha256((paths["out"] / name).read_bytes()).hexdigest()
    for filename, digest in result["input_metadata_sha256"].items():
        assert digest == sha256(Path(filename).read_bytes()).hexdigest()


@pytest.mark.parametrize("label,field,value,message", [
    ("fragments", "smiles_sha256", "wrong", "fragments"),
    ("proposals", "smiles_sha256", "wrong", "proposals"),
    ("fragments", "molecules", 999, "fragments"),
    ("fragments", "proposal_metadata_sha256", "wrong", "proposal"),
    ("proposals", "molecules", 999, "proposals"),
    ("proposals", "complete", False, "completed proposals"),
    ("proposals", "selected_molecules", 1, "completed proposals"),
    ("full", "smiles_sha256", "wrong", "fingerprint"),
    ("full", "schema", "wrong", "schema"),
    ("full", "molecules", 999, "metadata"),
])
def test_prepare_rejects_provenance_or_index_metadata_mismatch(tmp_path, label, field, value, message):
    paths, _, _ = _case(tmp_path)
    metadata_path = paths[label] / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata[field] = value
    _json(metadata_path, metadata)
    with pytest.raises(ValueError, match=message):
        _prepare(paths)
    assert not paths["out"].exists()


@pytest.mark.parametrize("mutation,message", [("unsorted", "strictly sorted"), ("duplicate", "unique"), ("empty", "metadata")])
def test_prepare_rejects_invalid_smiles_index_even_with_matching_digest(tmp_path, mutation, message):
    paths, _, full = _case(tmp_path)
    if mutation == "unsorted":
        full[0], full[1] = full[1], full[0]
    elif mutation == "duplicate":
        full[1] = full[0]
    else:
        full[0] = ""
    _json(paths["full"] / "smiles.json", full)
    metadata = json.loads((paths["full"] / "metadata.json").read_text())
    metadata["smiles_sha256"] = _digest(full)
    _json(paths["full"] / "metadata.json", metadata)
    with pytest.raises(ValueError, match=message):
        _prepare(paths)
    assert not paths["out"].exists()


@pytest.mark.parametrize("kind,total,increment", [("node", "atoms", 1), ("edge", "directed_edges", 2)])
def test_shared_molecule_count_mismatch_is_rejected(tmp_path, kind, total, increment):
    paths, old, full = _case(tmp_path)
    path = paths["full"] / f"{kind}_offsets.npy"
    offsets = np.load(path)
    offsets[full.index(old[0]) + 1:] += increment
    np.save(path, offsets)
    metadata_path = paths["full"] / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata[total] = int(offsets[-1])
    _json(metadata_path, metadata)
    with pytest.raises(ValueError, match=f"Shared molecules have different {kind} counts"):
        _prepare(paths)
    assert not paths["out"].exists()


def test_existing_output_is_refused_without_overwriting(tmp_path):
    paths, _, _ = _case(tmp_path)
    paths["out"].mkdir()
    marker = paths["out"] / "existing.txt"
    marker.write_text("preserve me")
    with pytest.raises(FileExistsError):
        _prepare(paths)
    assert marker.read_text() == "preserve me"
    assert list(paths["out"].iterdir()) == [marker]


def test_no_missing_molecules_produces_identity_maps_without_partitions(tmp_path):
    paths, old, _ = _case(tmp_path, no_missing=True)
    result = _prepare(paths, parts=8)
    for name in ("old_to_full", "full_to_old"):
        np.testing.assert_array_equal(np.load(paths["out"] / f"{name}.npy"), np.arange(len(old)))
    missing = np.load(paths["out"] / "missing_full_ids.npy")
    assert missing.shape == (0,) and missing.dtype == np.int32
    assert result["missing_molecules"] == 0 and result["partitions"] == []
    assert not list(paths["out"].glob("part-*"))


@pytest.mark.parametrize("kind,offsets,total", [("node", [0, 3, 2, 5], "atoms"), ("edge", [0, 4, 2, 6], "directed_edges")])
def test_unsigned_descending_offsets_are_rejected(tmp_path, kind, offsets, total):
    np.save(tmp_path / f"{kind}_offsets.npy", np.asarray(offsets, np.uint64))
    metadata = {"molecules": 3, total: offsets[-1]}
    with pytest.raises(ValueError, match=f"Invalid {kind} (offsets|counts)"):
        graph_counts(tmp_path, metadata, kind)
