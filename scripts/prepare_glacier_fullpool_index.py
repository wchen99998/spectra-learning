"""Map existing fragments to full-pool graph IDs and index missing molecules.

The unchanged GLACIER generator only reads smiles.json and its fingerprint.
Each partition is a molecule-list input, not a complete training GraphStore;
local_to_full.npy maps its future proposal IDs to the full graph store.
No fragments, embeddings, or retrieval scores are generated here.
"""

import argparse
from hashlib import sha256
from itertools import pairwise
import json
from pathlib import Path

import numpy as np


def smiles_digest(smiles: list[str]) -> str:
    digest = sha256()
    for value in smiles:
        digest.update(value.encode() + b"\0")
    return digest.hexdigest()


def read_index(path: Path) -> tuple[list[str], dict]:
    metadata = json.loads((path / "metadata.json").read_text())
    smiles = json.loads((path / "smiles.json").read_text())
    if metadata["schema"] != "atom_flare_55dfbdf_v1":
        raise ValueError("Expected the existing chem49/bond8 graph schema")
    if len(smiles) != metadata["molecules"] or not all(isinstance(s, str) and s for s in smiles):
        raise ValueError("Molecule list does not match graph metadata")
    if any(a >= b for a, b in pairwise(smiles)):
        raise ValueError("Graph SMILES must be unique and strictly sorted")
    if smiles_digest(smiles) != metadata["smiles_sha256"]:
        raise ValueError("Molecule fingerprint differs from graph metadata")
    return smiles, metadata


def subset_mapping(old: list[str], full: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Map two validated sorted indexes, rejecting any missing old molecule."""
    if len(full) > np.iinfo(np.int32).max:
        raise ValueError("Graph IDs exceed the int32 training representation")
    old_to_full = np.empty(len(old), np.int32)
    full_to_old = np.full(len(full), -1, np.int32)
    old_id = 0
    for full_id, value in enumerate(full):
        if old_id == len(old):
            break
        if value > old[old_id]:
            raise ValueError(f"Old molecule is absent from full pool: {old[old_id]}")
        if value == old[old_id]:
            old_to_full[old_id] = full_id
            full_to_old[full_id] = old_id
            old_id += 1
    if old_id != len(old):
        raise ValueError(f"Old molecule is absent from full pool: {old[old_id]}")
    return old_to_full, full_to_old


def graph_counts(path: Path, metadata: dict, kind: str) -> np.ndarray:
    offsets = np.load(path / f"{kind}_offsets.npy", mmap_mode="r", allow_pickle=False)
    total = metadata["atoms" if kind == "node" else "directed_edges"]
    if (offsets.dtype.kind not in "iu" or offsets.shape != (metadata["molecules"] + 1,)
            or offsets[0] != 0 or offsets[-1] != total):
        raise ValueError(f"Invalid {kind} offsets")
    if (offsets[1:] < offsets[:-1]).any():
        raise ValueError(f"Invalid {kind} offsets: counts must be nonnegative")
    counts = np.diff(offsets)
    if (counts < (1 if kind == "node" else 0)).any():
        raise ValueError(f"Invalid {kind} counts")
    return counts


def prepare(old_graphs: Path, full_graphs: Path, fragments: Path,
            proposals: Path, out: Path, parts: int = 4) -> dict:
    if parts < 1:
        raise ValueError("Partition count must be positive")
    old, old_meta = read_index(old_graphs)
    full, full_meta = read_index(full_graphs)
    fragment_meta = json.loads((fragments / "metadata.json").read_text())
    proposal_meta = json.loads((proposals / "metadata.json").read_text())
    for label, metadata in (("fragments", fragment_meta), ("proposals", proposal_meta)):
        if (metadata["smiles_sha256"] != old_meta["smiles_sha256"]
                or metadata["molecules"] != len(old)):
            raise ValueError(f"Old {label} do not match the old graph index")
    if not proposal_meta["complete"] or proposal_meta["selected_molecules"] != len(old):
        raise ValueError("Reuse requires completed proposals covering the entire old index")
    proposal_digest = sha256(
        json.dumps(proposal_meta, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if fragment_meta["proposal_metadata_sha256"] != proposal_digest:
        raise ValueError("Fragment provenance differs from the existing proposals")
    old_to_full, full_to_old = subset_mapping(old, full)
    for kind in ("node", "edge"):
        before = graph_counts(old_graphs, old_meta, kind)
        after = graph_counts(full_graphs, full_meta, kind)
        if not np.array_equal(before, after[old_to_full]):
            raise ValueError(f"Shared molecules have different {kind} counts")
    missing = np.flatnonzero(full_to_old < 0).astype(np.int32)
    out.mkdir(parents=True, exist_ok=False)
    for name, array in (("old_to_full", old_to_full), ("full_to_old", full_to_old),
                        ("missing_full_ids", missing)):
        np.save(out / f"{name}.npy", array)
    partitions = []
    if len(missing):
        for number, ids in enumerate(np.array_split(missing, min(parts, len(missing)))):
            directory = out / f"part-{number:02d}"
            directory.mkdir()
            values = [full[int(i)] for i in ids]
            (directory / "smiles.json").write_text(json.dumps(values) + "\n")
            np.save(directory / "local_to_full.npy", ids)
            metadata = {
                "schema": "glacier_proposal_smiles_index_v1", "molecules": len(values),
                "smiles_sha256": smiles_digest(values),
                "full_graph_smiles_sha256": full_meta["smiles_sha256"],
                "local_to_full_sha256": sha256((directory / "local_to_full.npy").read_bytes()).hexdigest(),
                "atom_order": "Chem.MolFromSmiles(original_graph_store_smiles)",
                "contains_graph_feature_arrays": False,
            }
            (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            partitions.append({"directory": str(directory), **metadata})
    result = {
        "schema": "glacier_fullpool_index_v1", "complete": True,
        "old_graphs": str(old_graphs), "full_graphs": str(full_graphs),
        "existing_fragments": str(fragments), "existing_proposals": str(proposals),
        "old_smiles_sha256": old_meta["smiles_sha256"],
        "full_smiles_sha256": full_meta["smiles_sha256"],
        "reused_molecules": len(old), "full_molecules": len(full),
        "missing_molecules": len(missing), "partitions": partitions,
        "shared_node_and_edge_counts_equal": True,
        "graph_feature_arrays_compared": False,
        "new_proposals_generated": False, "training_performed": False,
        "input_metadata_sha256": {str(path / "metadata.json"):
            sha256((path / "metadata.json").read_bytes()).hexdigest()
            for path in (old_graphs, full_graphs, fragments, proposals)},
        "mapping_sha256": {name: sha256((out / name).read_bytes()).hexdigest() for name in
                           ("old_to_full.npy", "full_to_old.npy", "missing_full_ids.npy")},
        "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out / "metadata.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("old-graphs", "full-graphs", "fragments", "proposals", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--parts", type=int, default=4, help="Maximum number of nonempty partitions")
    args = parser.parse_args()
    result = prepare(args.old_graphs, args.full_graphs, args.fragments, args.proposals, args.out, args.parts)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
