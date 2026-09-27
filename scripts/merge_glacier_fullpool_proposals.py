"""Merge pinned GLACIER proposal shards by exact, verified graph-ID mappings.

Only graph_ids change; molecular atom indices and all other NPZ arrays are
preserved. A failed merge leaves an incomplete output directory without final
metadata. Existing outputs are refused, including incomplete previous attempts.
"""

import argparse
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np


GENERATION_FIELDS = (
    "schema", "source_commit", "generator_sha256", "checkpoint_sha256",
    "precision", "collision_energies", "adduct", "instrument", "atom_order", "aggregation",
)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def _mapping(path: Path, expected_sha: str, length: int) -> np.ndarray:
    if _sha256(path) != expected_sha:
        raise ValueError(f"Mapping SHA mismatch: {path}")
    result = np.load(path, mmap_mode="r", allow_pickle=False)
    if result.dtype != np.int32 or result.shape != (length,):
        raise ValueError(f"Mapping must be an int32 vector of length {length}: {path}")
    return result


def _source(path: Path, fingerprint: str, count: int) -> tuple[dict, str]:
    metadata_path = path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if (metadata.get("complete") is not True or metadata.get("molecules") != count
            or metadata.get("selected_molecules") != count):
        raise ValueError(f"Incomplete source proposal coverage: {path}")
    if metadata.get("smiles_sha256") != fingerprint:
        raise ValueError(f"Source proposal fingerprint mismatch: {path}")
    if (metadata.get("schema") != "glacier_learned_proposals_original_order_v1"
            or any(field not in metadata for field in GENERATION_FIELDS)):
        raise ValueError(f"Missing or incompatible generation provenance: {path}")
    names = metadata["completed_shards"]
    if (not isinstance(names, list) or any(
            not isinstance(name, str) or Path(name).name != name
            or "\\" in name or not name.endswith(".npz") for name in names)):
        raise ValueError(f"Source shard names must be safe NPZ basenames: {path}")
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate source shard basenames: {path}")
    return metadata, _sha256(metadata_path)


def merge(index: Path, new_proposals: Path, out: Path) -> dict:
    if out.exists():
        raise FileExistsError(f"Merge output already exists: {out}")
    index_path = index / "metadata.json"
    manifest = json.loads(index_path.read_text())
    if manifest.get("schema") != "glacier_fullpool_index_v1" or manifest.get("complete") is not True:
        raise ValueError("Expected a completed full-pool index")
    full_n, old_n, missing_n = (manifest[key] for key in
                                ("full_molecules", "reused_molecules", "missing_molecules"))
    if not 0 <= old_n <= full_n <= np.iinfo(np.int32).max or missing_n != full_n - old_n:
        raise ValueError("Invalid full-pool index counts")
    hashes = manifest["mapping_sha256"]
    old_map = _mapping(index / "old_to_full.npy", hashes["old_to_full.npy"], old_n)
    inverse = _mapping(index / "full_to_old.npy", hashes["full_to_old.npy"], full_n)
    missing = _mapping(index / "missing_full_ids.npy", hashes["missing_full_ids.npy"], missing_n)
    if (np.any(old_map < 0) or np.any(old_map >= full_n)
            or np.any(old_map[1:] <= old_map[:-1])
            or np.any(inverse < -1) or np.any(inverse >= old_n)
            or not np.array_equal(inverse[old_map], np.arange(old_n))
            or not np.array_equal(missing, np.flatnonzero(inverse == -1))):
        raise ValueError("Index mappings are not consistent bijections and complements")

    old_path = Path(manifest["existing_proposals"])
    old_meta, old_sha = _source(old_path, manifest["old_smiles_sha256"], old_n)
    if old_sha != manifest["input_metadata_sha256"].get(str(old_path / "metadata.json")):
        raise ValueError("Existing proposal metadata changed since index preparation")
    identity = {field: old_meta[field] for field in GENERATION_FIELDS}
    sources = [("old", old_path, old_map, index / "old_to_full.npy", old_meta, old_sha)]
    consumed = 0
    for number, partition in enumerate(manifest["partitions"]):
        name = f"part-{number:02d}"
        if Path(partition["directory"]).name != name:
            raise ValueError("Index partition names must be consecutive part-NN directories")
        directory = index / name
        expected = {key: value for key, value in partition.items() if key != "directory"}
        if json.loads((directory / "metadata.json").read_text()) != expected:
            raise ValueError(f"Partition metadata differs from the prepared index: {name}")
        if (partition["schema"] != "glacier_proposal_smiles_index_v1"
                or partition["full_graph_smiles_sha256"] != manifest["full_smiles_sha256"]
                or partition["atom_order"] != identity["atom_order"]):
            raise ValueError(f"Partition provenance differs from the full index: {name}")
        count = partition["molecules"]
        local_map = _mapping(directory / "local_to_full.npy", partition["local_to_full_sha256"], count)
        if count < 1 or not np.array_equal(local_map, missing[consumed:consumed + count]):
            raise ValueError(f"Partition maps must cover the missing IDs exactly once in order: {name}")
        consumed += count
        path = new_proposals / name
        metadata, digest = _source(path, partition["smiles_sha256"], count)
        if any(metadata[field] != identity[field] for field in GENERATION_FIELDS):
            raise ValueError(f"Generation provenance mismatch: {name}")
        sources.append((name, path, local_map, directory / "local_to_full.npy", metadata, digest))
    if consumed != missing_n:
        raise ValueError("Partition maps do not cover all missing IDs")

    out.mkdir(parents=True, exist_ok=False)
    global_seen = np.zeros(full_n, bool)
    completed, source_records, unsupported_full = [], [], []
    for label, source, mapping, mapping_path, metadata, digest in sources:
        local_seen = np.zeros(len(mapping), bool)
        raw_unsupported = metadata.get("unsupported_graph_ids")
        if not isinstance(raw_unsupported, list) or any(type(value) is not int for value in raw_unsupported):
            raise ValueError(f"Unsupported IDs must be an explicit integer list: {source}")
        unsupported = np.asarray(raw_unsupported, np.int64)
        if (np.any(unsupported < 0) or np.any(unsupported >= len(mapping))
                or len(np.unique(unsupported)) != len(unsupported)):
            raise ValueError(f"Invalid or duplicate unsupported graph IDs: {source}")
        unsupported_mask = np.zeros(len(mapping), bool)
        unsupported_mask[unsupported] = True
        unsupported_full.extend(mapping[unsupported].tolist())
        first_output = len(completed)
        for filename in metadata["completed_shards"]:
            with np.load(source / filename, allow_pickle=False) as shard:
                if len(set(shard.files)) != len(shard.files):
                    raise ValueError(f"Duplicate NPZ array names: {source / filename}")
                if not {"graph_ids", "frag_off", "mem_off", "atoms", "inten"} <= set(shard.files):
                    raise ValueError(f"Missing proposal payload arrays: {source / filename}")
                arrays = {name: shard[name] for name in shard.files}
            ids = arrays["graph_ids"]
            if (ids.ndim != 1 or ids.dtype.kind not in "iu" or len(ids) == 0
                    or np.any(ids < 0) or np.any(ids >= len(mapping))):
                raise ValueError(f"Proposal IDs must be nonempty source-local integer IDs: {source / filename}")
            full_ids = mapping[ids]
            if (len(np.unique(ids)) != len(ids) or local_seen[ids].any()
                    or global_seen[full_ids].any()):
                raise ValueError(f"Duplicate proposal graph IDs: {source / filename}")
            for field, rows, size in (("frag_off", len(ids), len(arrays["inten"])),
                                      ("mem_off", len(arrays["inten"]), len(arrays["atoms"]))):
                offsets = arrays[field]
                if (offsets.dtype.kind not in "iu" or offsets.shape != (rows + 1,)
                        or offsets[0] != 0 or offsets[-1] != size
                        or np.any(offsets[1:] < offsets[:-1])):
                    raise ValueError(f"Invalid proposal {field}: {source / filename}")
            if np.any(np.diff(arrays["frag_off"])[unsupported_mask[ids]] != 0):
                raise ValueError("Unsupported graph IDs cannot have predicted fragments")
            arrays["graph_ids"] = np.asarray(full_ids, np.int32)
            output_name = f"shard-{len(completed):06d}.npz"
            np.savez(out / output_name, **arrays)
            completed.append(output_name)
            local_seen[ids] = True
            global_seen[full_ids] = True
            if len(completed) % 100 == 0:
                print(f"Merged {len(completed)} shards; current source {label}", flush=True)
        if not local_seen.all():
            raise ValueError(f"Missing proposals for {int((~local_seen).sum())} source graph IDs: {source}")
        source_records.append({
            "label": label, "directory": str(source), "metadata_sha256": digest,
            "smiles_sha256": metadata["smiles_sha256"], "molecules": len(mapping),
            "mapping": str(mapping_path), "mapping_sha256": _sha256(mapping_path),
            "input_shards": metadata["completed_shards"],
            "output_shards": completed[first_output:],
            "unsupported_molecules": len(unsupported),
        })
        print(f"Completed source {label}: {len(mapping)} molecules, "
              f"{len(completed) - first_output} shards", flush=True)
    if not global_seen.all():
        raise ValueError("Merged proposals do not cover every full-pool graph")
    result = {
        **identity, "smiles_sha256": manifest["full_smiles_sha256"],
        "molecules": full_n, "selected_molecules": full_n, "selection_limit": 0, "complete": True,
        "completed_shards": completed, "unsupported_graph_ids": sorted(unsupported_full),
        "operation": "merge_existing_proposals",
        "merge_provenance": {
            "schema": "glacier_fullpool_proposal_merge_v1", "index": str(index),
            "index_metadata_sha256": _sha256(index_path),
            "index_input_metadata_sha256": manifest["input_metadata_sha256"],
            "index_mapping_sha256": hashes, "sources": source_records,
            "script_sha256": _sha256(Path(__file__)), "changed_arrays": ["graph_ids"],
            "new_predictions_generated": False, "global_coverage_verified": True,
            "selection_semantics": "All full-pool molecules; no subsampling during merge",
        },
    }
    temporary = out / "metadata.json.tmp"
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(out / "metadata.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--new-proposals", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = merge(args.index, args.new_proposals, args.out)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
