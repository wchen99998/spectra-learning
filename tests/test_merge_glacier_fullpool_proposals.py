"""Raw proposal merge preserves atom order and requires exact complete coverage."""

from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from merge_glacier_fullpool_proposals import merge


IDENTITY = {
    "schema": "glacier_learned_proposals_original_order_v1",
    "source_commit": "pinned-upstream-commit", "generator_sha256": "pinned-generator",
    "checkpoint_sha256": "pinned-weights", "precision": "float32",
    "collision_energies": [20.0, 30.0, 40.0], "adduct": "[M+H]+", "instrument": "Orbitrap",
    "atom_order": "Chem.MolFromSmiles(original_graph_store_smiles)",
    "aggregation": "union atom sets; maximum predicted intensity over shifts and energies",
}


def _sha(path):
    return sha256(path.read_bytes()).hexdigest()


def _write_json(path, value):
    path.write_text(json.dumps(value) + "\n")


def _update(path, **values):
    metadata = json.loads(path.read_text())
    metadata.update(values)
    _write_json(path, metadata)


def _write_shard(path, ids, empty=()):
    fragments, members, atoms, intensities = [0], [0], [], []
    for graph_id in ids:
        if graph_id not in empty:
            # Deliberately non-sorted local atom membership must stay untouched.
            atoms.extend([2, 0])
            members.append(len(atoms))
            intensities.append(0.125 * (int(graph_id) + 1))
        fragments.append(len(intensities))
    np.savez(path, graph_ids=np.asarray(ids, np.int32),
             frag_off=np.asarray(fragments, np.int64), mem_off=np.asarray(members, np.int64),
             atoms=np.asarray(atoms, np.int16), inten=np.asarray(intensities, np.float32),
             extra_payload=np.array([[5, 9]], np.uint8))


def _source(path, fingerprint, count, chunks, *, empty=(), unsupported=()):
    path.mkdir(parents=True)
    names = []
    for number, ids in enumerate(chunks):
        name = f"shard-{number:06d}.npz"
        _write_shard(path / name, ids, empty)
        names.append(name)
    _write_json(path / "metadata.json", {
        **IDENTITY, "molecules": count, "selected_molecules": count, "complete": True,
        "smiles_sha256": fingerprint, "completed_shards": names,
        "unsupported_graph_ids": list(unsupported), "selection_seed": 0, "selection_limit": 0,
        "benchmark": {"molecules_per_second": 99},
    })


def _case(tmp_path, *, no_missing=False):
    index, new, old, out = [tmp_path / name for name in ("index", "new", "old", "out")]
    index.mkdir()
    # Independent expected identity: reused full IDs 1,4,6; new full IDs 0,2,3,5.
    maps = {
        "old_to_full.npy": [0, 1, 2] if no_missing else [1, 4, 6],
        "full_to_old.npy": [0, 1, 2] if no_missing else [-1, 0, -1, -1, 1, -1, 2],
        "missing_full_ids.npy": [] if no_missing else [0, 2, 3, 5],
    }
    for name, ids in maps.items():
        np.save(index / name, np.asarray(ids, np.int32))
    _source(old, "old-smiles", 3, [[2, 0], [1]], empty=[1], unsupported=[1])
    partitions = []
    for number, ids in enumerate([] if no_missing else [[0, 2], [3, 5]]):
        name = f"part-{number:02d}"
        directory = index / name
        directory.mkdir()
        np.save(directory / "local_to_full.npy", np.asarray(ids, np.int32))
        metadata = {
            "schema": "glacier_proposal_smiles_index_v1", "molecules": 2,
            "smiles_sha256": f"part-{number}-smiles", "full_graph_smiles_sha256": "full-smiles",
            "local_to_full_sha256": _sha(directory / "local_to_full.npy"),
            "atom_order": IDENTITY["atom_order"], "contains_graph_feature_arrays": False,
        }
        _write_json(directory / "metadata.json", metadata)
        partitions.append({"directory": str(directory), **metadata})
        # Empty supported predictions are valid too, independent of unsupported IDs.
        _source(new / name, metadata["smiles_sha256"], 2, [[1, 0]], empty=[0],
                unsupported=[0] if number == 1 else [])
    _write_json(index / "metadata.json", {
        "schema": "glacier_fullpool_index_v1", "complete": True,
        "full_molecules": 3 if no_missing else 7, "reused_molecules": 3,
        "missing_molecules": 0 if no_missing else 4,
        "existing_proposals": str(old), "old_smiles_sha256": "old-smiles",
        "full_smiles_sha256": "full-smiles", "partitions": partitions,
        "mapping_sha256": {name: _sha(index / name) for name in maps},
        "input_metadata_sha256": {str(old / "metadata.json"): _sha(old / "metadata.json"),
                                  "original-fragments/metadata.json": "bound-fragment-metadata"},
    })
    return index, new, old, out


def _change_shard(path, **values):
    with np.load(path) as shard:
        arrays = {name: shard[name] for name in shard.files}
    arrays.update(values)
    np.savez(path, **arrays)


def test_merge_preserves_every_payload_and_original_and_records_exact_provenance(tmp_path, capsys):
    index, new, old, out = _case(tmp_path)
    originals = {p: _sha(p) for root in (old, new, index) for p in root.rglob("*") if p.is_file()}
    result = merge(index, new, out)
    assert result == json.loads((out / "metadata.json").read_text())
    assert result["complete"] and result["molecules"] == result["selected_molecules"] == 7
    assert result["smiles_sha256"] == "full-smiles"
    assert result["unsupported_graph_ids"] == [3, 4]
    assert result["selection_limit"] == 0
    assert "benchmark" not in result and "selection_seed" not in result
    assert result["operation"] == "merge_existing_proposals"
    provenance = result["merge_provenance"]
    assert provenance["global_coverage_verified"] and not provenance["new_predictions_generated"]
    assert provenance["changed_arrays"] == ["graph_ids"]
    assert provenance["index_metadata_sha256"] == _sha(index / "metadata.json")
    assert provenance["index_input_metadata_sha256"]["original-fragments/metadata.json"] == "bound-fragment-metadata"
    full_ids = []
    expected = [[6, 1], [4], [2, 0], [5, 3]]
    shard_number = 0
    for source in provenance["sources"]:
        source_path = Path(source["directory"])
        assert source["metadata_sha256"] == _sha(source_path / "metadata.json")
        assert source["mapping_sha256"] == _sha(Path(source["mapping"]))
        for source_name, output_name in zip(source["input_shards"], source["output_shards"], strict=True):
            with np.load(source_path / source_name) as before, np.load(out / output_name) as after:
                assert before.files == after.files
                np.testing.assert_array_equal(after["graph_ids"], expected[shard_number])
                full_ids.extend(after["graph_ids"].tolist())
                for field in before.files:
                    if field != "graph_ids":
                        assert before[field].dtype == after[field].dtype
                        assert before[field].shape == after[field].shape
                        assert before[field].tobytes() == after[field].tobytes()
            shard_number += 1
    assert sorted(full_ids) == list(range(7))
    assert all(_sha(path) == digest for path, digest in originals.items())
    assert "Completed source part-01: 2 molecules, 1 shards" in capsys.readouterr().out


def test_no_missing_graphs_needs_only_original_proposals(tmp_path):
    index, new, old, out = _case(tmp_path, no_missing=True)
    result = merge(index, new, out)
    assert result["molecules"] == 3 and result["unsupported_graph_ids"] == [1]
    assert len(result["merge_provenance"]["sources"]) == 1
    with np.load(out / result["completed_shards"][0]) as shard:
        np.testing.assert_array_equal(shard["graph_ids"], [2, 0])


@pytest.mark.parametrize("field,value", [
    ("source_commit", "other"), ("generator_sha256", "other"), ("checkpoint_sha256", "other"),
    ("precision", "float16"), ("collision_energies", [20, 30]), ("adduct", "[M+Na]+"),
    ("instrument", "other"), ("atom_order", "canonical"), ("aggregation", "mean"),
    ("schema", "other"), ("complete", False), ("molecules", 999), ("selected_molecules", 1),
    ("smiles_sha256", "other"),
])
def test_source_provenance_or_completion_mismatch_is_rejected_before_copy(tmp_path, field, value):
    index, new, old, out = _case(tmp_path)
    _update(new / "part-00" / "metadata.json", **{field: value})
    with pytest.raises(ValueError):
        merge(index, new, out)
    assert not out.exists()


def test_old_proposal_metadata_is_bound_to_prepared_index(tmp_path):
    index, new, old, out = _case(tmp_path)
    _update(old / "metadata.json", benchmark={"molecules_per_second": 100})
    with pytest.raises(ValueError, match="changed since index preparation"):
        merge(index, new, out)
    assert not out.exists()


@pytest.mark.parametrize("names", [["../escape.npz"], ["/tmp/escape.npz"], ["a\\b.npz"],
                                   ["shard-000000.npz", "shard-000000.npz"], ["not-npz"]])
def test_unsafe_or_duplicate_shard_names_are_rejected(tmp_path, names):
    index, new, old, out = _case(tmp_path)
    _update(new / "part-00" / "metadata.json", completed_shards=names)
    with pytest.raises(ValueError, match="shard"):
        merge(index, new, out)
    assert not out.exists()


@pytest.mark.parametrize("ids,message", [
    ([1, 1], "Duplicate"), ([0], "Missing"), ([], "nonempty"),
    ([-1, 0], "source-local"), ([2, 0], "source-local"),
])
def test_bad_shard_coverage_leaves_partial_output_without_completion(tmp_path, ids, message):
    index, new, old, out = _case(tmp_path)
    _write_shard(new / "part-00" / "shard-000000.npz", ids, empty=[0])
    with pytest.raises(ValueError, match=message):
        merge(index, new, out)
    assert list(out.glob("*.npz"))  # Old source was copied before this failure.
    assert not (out / "metadata.json").exists()
    assert not (out / "metadata.json.tmp").exists()


def test_graph_duplicates_across_shards_are_rejected(tmp_path):
    index, new, old, out = _case(tmp_path)
    directory = new / "part-00"
    _write_shard(directory / "extra.npz", [0], empty=[0])
    _update(directory / "metadata.json", completed_shards=["shard-000000.npz", "extra.npz"])
    with pytest.raises(ValueError, match="Duplicate proposal graph IDs"):
        merge(index, new, out)
    assert not (out / "metadata.json").exists()


def test_unsigned_oversize_ids_do_not_wrap_during_remap(tmp_path):
    index, new, old, out = _case(tmp_path)
    _change_shard(new / "part-00" / "shard-000000.npz", graph_ids=np.array([2**64 - 1, 0], np.uint64))
    with pytest.raises(ValueError, match="source-local"):
        merge(index, new, out)
    assert not (out / "metadata.json").exists()


@pytest.mark.parametrize("name", ["old_to_full.npy", "full_to_old.npy", "missing_full_ids.npy", "part-00/local_to_full.npy"])
def test_tampered_mapping_hash_is_rejected(tmp_path, name):
    index, new, old, out = _case(tmp_path)
    path = index / name
    values = np.load(path)
    values[0] += 1
    np.save(path, values)
    with pytest.raises(ValueError, match="Mapping SHA mismatch"):
        merge(index, new, out)
    assert not out.exists()


def test_hash_matching_but_inconsistent_inverse_map_is_rejected(tmp_path):
    index, new, old, out = _case(tmp_path)
    name = "full_to_old.npy"
    np.save(index / name, np.array([-1, 2, -1, -1, 1, -1, 0], np.int32))
    manifest = json.loads((index / "metadata.json").read_text())
    manifest["mapping_sha256"][name] = _sha(index / name)
    _write_json(index / "metadata.json", manifest)
    with pytest.raises(ValueError, match="bijections"):
        merge(index, new, out)
    assert not out.exists()


def test_hash_matching_partition_overlap_is_rejected(tmp_path):
    index, new, old, out = _case(tmp_path)
    path = index / "part-01" / "local_to_full.npy"
    np.save(path, np.array([2, 5], np.int32))  # Repeats part-00 full ID 2; loses full ID 3.
    _update(path.parent / "metadata.json", local_to_full_sha256=_sha(path))
    manifest = json.loads((index / "metadata.json").read_text())
    manifest["partitions"][1]["local_to_full_sha256"] = _sha(path)
    _write_json(index / "metadata.json", manifest)
    with pytest.raises(ValueError, match="cover the missing IDs exactly once"):
        merge(index, new, out)
    assert not out.exists()


@pytest.mark.parametrize("unsupported", [[2], [-1], [0, 0], [1]])
def test_invalid_unsupported_ids_or_predicted_unsupported_fragments_are_rejected(tmp_path, unsupported):
    index, new, old, out = _case(tmp_path)
    _update(new / "part-00" / "metadata.json", unsupported_graph_ids=unsupported)
    with pytest.raises(ValueError, match="[Uu]nsupported"):
        merge(index, new, out)
    assert not (out / "metadata.json").exists()


def test_unsigned_descending_offsets_are_rejected(tmp_path):
    index, new, old, out = _case(tmp_path)
    _change_shard(new / "part-00" / "shard-000000.npz", frag_off=np.array([0, 2, 1], np.uint64))
    with pytest.raises(ValueError, match="Invalid proposal frag_off"):
        merge(index, new, out)
    assert not (out / "metadata.json").exists()


def test_existing_output_is_refused_without_overwriting(tmp_path):
    index, new, old, out = _case(tmp_path)
    out.mkdir()
    marker = out / "existing.txt"
    marker.write_text("preserve me")
    with pytest.raises(FileExistsError):
        merge(index, new, out)
    assert marker.read_text() == "preserve me"
    assert list(out.iterdir()) == [marker]


def test_real_graph_index_merge_and_fragment_assembly_preserve_atom_membership(tmp_path):
    from atom_flare_data import build_graph_store
    from glacier_fragment_data import build_fragment_store
    from prepare_glacier_fullpool_index import prepare

    # CCO and OCC deliberately retain different original atom orderings.
    old = build_graph_store(["NCC", "CCO"], tmp_path / "old-graphs")
    full = build_graph_store(["OCC", "CCO", "CCC", "NCC", "CCN"], tmp_path / "full-graphs")
    old_proposals = tmp_path / "old-proposals"
    _source(old_proposals, old.metadata["smiles_sha256"], 2, [[1, 0]])
    # Each prediction is a connected two-atom fragment, listed in reverse order.
    _change_shard(old_proposals / "shard-000000.npz", atoms=np.array([2, 1, 2, 1], np.int16))
    old_fragments = build_fragment_store(old, old_proposals, tmp_path / "old-fragments", min_atoms=2)
    index, new, merged = tmp_path / "index", tmp_path / "new-proposals", tmp_path / "merged"
    manifest = prepare(old.directory, full.directory, old_fragments.directory, old_proposals, index, parts=2)
    for partition in manifest["partitions"]:
        name = Path(partition["directory"]).name
        count = partition["molecules"]
        _source(new / name, partition["smiles_sha256"], count, [list(reversed(range(count)))])
        _change_shard(new / name / "shard-000000.npz", atoms=np.tile(np.array([2, 1], np.int16), count))
    merge(index, new, merged)
    assembled = build_fragment_store(full, merged, tmp_path / "full-fragments", min_atoms=2)

    assert full.smiles == ["CCC", "CCN", "CCO", "NCC", "OCC"]
    np.testing.assert_array_equal(np.load(index / "old_to_full.npy"), [2, 3])
    np.testing.assert_array_equal(assembled.counts, [2, 2, 2, 2, 2])
    np.testing.assert_array_equal(assembled.atom_counts, [3, 3, 3, 3, 3])
    assert assembled.metadata["proposals"]["operation"] == "merge_existing_proposals"
    assert assembled.metadata["smiles_sha256"] == full.metadata["smiles_sha256"]
    predicted_fraction = {}
    for graph_id, expected_intensity in enumerate([0.125, 0.25, 0.125, 0.25, 0.125]):
        first, last = assembled.frag_off[graph_id:graph_id + 2]
        np.testing.assert_array_equal(assembled.kind[first:last], [0, 1])
        np.testing.assert_array_equal(assembled.n_cuts[first:last], [0, 1])
        np.testing.assert_array_equal(assembled.inten[first:last], [1, expected_intensity])
        anchor_lo, predicted_lo, predicted_hi = assembled.mem_off[first:last + 1]
        np.testing.assert_array_equal(assembled.atoms[anchor_lo:predicted_lo], [0, 1, 2])
        np.testing.assert_array_equal(assembled.atoms[predicted_lo:predicted_hi], [1, 2])
        predicted_fraction[full.smiles[graph_id]] = assembled.mass_frac[first + 1]
    # Same spelling-dependent atom IDs select C-O in CCO and C-C in OCC.
    assert predicted_fraction["CCO"] != predicted_fraction["OCC"]
    for old_id, full_id in enumerate([2, 3]):
        old_lo, old_hi = old_fragments.frag_off[old_id:old_id + 2]
        full_lo, full_hi = assembled.frag_off[full_id:full_id + 2]
        for field in ("mass_frac", "n_atoms", "n_cuts", "kind", "inten"):
            np.testing.assert_array_equal(getattr(old_fragments, field)[old_lo:old_hi],
                                          getattr(assembled, field)[full_lo:full_hi])
        np.testing.assert_array_equal(old_fragments.total_mass[old_id], assembled.total_mass[full_id])
