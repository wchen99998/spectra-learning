"""Compare every original graph with its mapped expanded-store graph.

Uses the existing exact-SMILES index and fragment-reuse proof. Graph payloads
are memory mapped and compared in bounded molecule chunks. No model or training
code is loaded. A pass report is published atomically only after all comparisons.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import file_digest, sha256
import json
from pathlib import Path
import time

import numpy as np


FIELDS = ("nodes", "edges", "senders", "receivers", "afrac")
FRAGMENT_FIELDS = {"n_cuts", "kind", "mass_frac", "n_atoms", "inten", "atoms",
                   "bnd_atoms", "total_mass", "unsupported"}


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _arrays(directory: Path, metadata: dict, chunk_molecules: int) -> dict[str, np.ndarray]:
    if (metadata["schema"] != "atom_flare_55dfbdf_v1"
            or (metadata["atom_dim"], metadata["bond_dim"], metadata["fraction_slots"]) != (49, 8, 6)):
        raise ValueError("Expected the existing chem49/bond8/fraction6 graph schema")
    molecules, nodes, edges = (metadata[key] for key in ("molecules", "atoms", "directed_edges"))
    layout = {"nodes": ((nodes, 49), np.float32), "edges": ((edges, 8), np.float32),
              "senders": ((edges,), np.int16), "receivers": ((edges,), np.int16),
              "afrac": ((nodes, 6), np.float32),
              "node_offsets": ((molecules + 1,), np.int64), "edge_offsets": ((molecules + 1,), np.int64)}
    arrays = {}
    for name, (shape, dtype) in layout.items():
        value = np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        if value.shape != shape or value.dtype != dtype or not value.flags.c_contiguous:
            raise ValueError(f"Invalid graph array shape, dtype or layout: {directory / name}")
        arrays[name] = value
    for kind, total, minimum in (("node", nodes, 1), ("edge", edges, 0)):
        offsets = arrays[f"{kind}_offsets"]
        if offsets[0] != 0 or offsets[-1] != total:
            raise ValueError(f"Invalid {kind} offset endpoints: {directory}")
        for start in range(0, molecules, chunk_molecules):
            lengths = np.diff(offsets[start:min(start + chunk_molecules, molecules) + 1])
            if np.any(lengths < minimum):
                raise ValueError(f"Invalid {kind} offsets: {directory}")
    return arrays


def _ranges(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    offsets = np.cumsum(lengths, dtype=np.int64) - lengths
    return np.arange(int(lengths.sum()), dtype=np.int64) + np.repeat(starts - offsets, lengths)


def validate(index: Path, source_manifest: Path, fragment_reuse: Path, out: Path,
             *, chunk_molecules: int = 256) -> dict:
    if out.exists():
        raise FileExistsError(f"Validation report already exists: {out}")
    if chunk_molecules < 1:
        raise ValueError("chunk_molecules must be positive")
    started = time.monotonic()
    index_path = index / "metadata.json"
    index_meta, source, reuse = _read(index_path), _read(source_manifest), _read(fragment_reuse)
    if index_meta.get("schema") != "glacier_fullpool_index_v1" or index_meta.get("complete") is not True:
        raise ValueError("Expected a completed full-pool index")
    if (reuse.get("schema") != "glacier_fullpool_fragment_reuse_validation_v1"
            or reuse.get("complete") is not True or reuse.get("status") != "passed"
            or set(reuse.get("fields", [])) != FRAGMENT_FIELDS):
        raise ValueError("Expected a completed exact fragment-reuse report")
    old_path, full_path = Path(index_meta["old_graphs"]), Path(index_meta["full_graphs"])
    mapping_path = index / "old_to_full.npy"
    old_fragment_path = Path(index_meta["existing_fragments"])
    full_fragment_path = Path(reuse["full_fragments"]["path"])
    metadata_paths = [path / "metadata.json" for path in
                      (old_path, full_path, old_fragment_path, full_fragment_path)]
    old_meta, full_meta, old_fragments, full_fragments = map(_read, metadata_paths)
    input_paths = [index_path, source_manifest, fragment_reuse, mapping_path, *metadata_paths]
    input_hashes = {str(path): _sha256(path) for path in input_paths}
    for path in metadata_paths[:3]:
        if input_hashes[str(path)] != index_meta["input_metadata_sha256"].get(str(path)):
            raise ValueError(f"Metadata changed since index preparation: {path}")
    index_hash, mapping_hash = input_hashes[str(index_path)], input_hashes[str(mapping_path)]
    if (Path(reuse["index"]["path"]) != index or reuse["index"]["metadata_sha256"] != index_hash
            or Path(reuse["mapping"]["path"]) != mapping_path
            or reuse["mapping"]["sha256"] != mapping_hash
            or index_meta["mapping_sha256"]["old_to_full.npy"] != mapping_hash):
        raise ValueError("Fragment proof index or mapping identity differs")
    old_n, full_n = index_meta["reused_molecules"], index_meta["full_molecules"]
    if not 0 < old_n <= full_n:
        raise ValueError("Invalid original/full molecule counts")
    for label, graph, fragment, fragment_path, count, fingerprint in (
        ("old_fragments", old_meta, old_fragments, old_fragment_path, old_n, index_meta["old_smiles_sha256"]),
        ("full_fragments", full_meta, full_fragments, full_fragment_path, full_n, index_meta["full_smiles_sha256"]),
    ):
        identity = reuse[label]
        if (graph["molecules"] != count or fragment["molecules"] != count
                or graph["smiles_sha256"] != fingerprint or fragment["smiles_sha256"] != fingerprint
                or fragment["schema"] != "glacier_fragments_original_order_v1"
                or identity["molecules"] != count or identity["smiles_sha256"] != fingerprint
                or Path(identity["path"]) != fragment_path
                or identity["metadata_sha256"] != input_hashes[str(fragment_path / "metadata.json")]):
            raise ValueError(f"Graph or fragment identity differs: {label}")
    if (source["graphs"] != old_meta or source["fragments"] != old_fragments
            or Path(source["args"]["prepared"]) / "graphs" != old_path
            or Path(source["args"]["fragments"]) != old_fragment_path):
        raise ValueError("Original head manifest does not describe the certified old inputs")
    expected_compared = {"molecules": old_n, **{name: old_fragments[name] for name in
                         ("fragments", "memberships", "boundary_memberships")}}
    settings = {name: old_fragments[name] for name in ("f_max", "min_atoms", "min_frac")}
    if (reuse["compared"] != expected_compared or reuse["additional_molecules"] != full_n - old_n
            or reuse["settings"] != settings or any(full_fragments[k] != v for k, v in settings.items())):
        raise ValueError("Fragment proof does not cover every original molecule with matching settings")

    mapping = np.load(mapping_path, mmap_mode="r", allow_pickle=False)
    if mapping.dtype != np.int32 or mapping.shape != (old_n,):
        raise ValueError("Invalid old-to-full mapping shape or dtype")
    previous = -1
    for start in range(0, old_n, chunk_molecules):
        ids = mapping[start:min(start + chunk_molecules, old_n)]
        if (ids[0] <= previous or np.any(ids < 0) or np.any(ids >= full_n)
                or np.any(ids[1:] <= ids[:-1])):
            raise ValueError("Old-to-full mapping must contain unique increasing valid IDs")
        previous = int(ids[-1])
    old, full = (_arrays(path, meta, chunk_molecules) for path, meta in
                 ((old_path, old_meta), (full_path, full_meta)))
    payload_hashes = {field: sha256() for field in FIELDS}
    compared = {"molecules": 0, "nodes": 0, "directed_edges": 0}
    for start in range(0, old_n, chunk_molecules):
        stop = min(start + chunk_molecules, old_n)
        ids = mapping[start:stop]
        selected, bounds, counts = {}, {}, {}
        for kind in ("node", "edge"):
            before = old[f"{kind}_offsets"][start:stop + 1]
            lengths = np.diff(before)
            after = full[f"{kind}_offsets"]
            if not np.array_equal(lengths, after[ids + 1] - after[ids]):
                raise ValueError(f"Shared graph {kind} counts differ for old IDs [{start}, {stop})")
            selected[kind] = _ranges(after[ids], lengths)
            bounds[kind] = int(before[0]), int(before[-1])
            counts[kind] = lengths
        edge_node_counts = np.repeat(counts["node"], counts["edge"])
        for field in FIELDS:
            kind = "node" if field in ("nodes", "afrac") else "edge"
            first, last = bounds[kind]
            before, after = old[field][first:last], full[field][selected[kind]]
            if (before.dtype.kind == "f" and not np.isfinite(before).all()) or not np.array_equal(
                    before.view(np.uint8), after.view(np.uint8)):
                raise ValueError(f"Shared graph {field} differs or is nonfinite for old IDs [{start}, {stop})")
            if field in ("senders", "receivers") and (np.any(before < 0) or np.any(before >= edge_node_counts)):
                raise ValueError(f"Invalid molecule-local {field} for old IDs [{start}, {stop})")
            payload_hashes[field].update(before.tobytes(order="C"))
        compared["molecules"] += stop - start
        compared["nodes"] += int(counts["node"].sum())
        compared["directed_edges"] += int(counts["edge"].sum())
        if (start // chunk_molecules + 1) % 100 == 0 or stop == old_n:
            print(f"Verified reused graphs: {stop:,}/{old_n:,} molecules", flush=True)

    if input_hashes != {str(path): _sha256(path) for path in input_paths}:
        raise ValueError("Input metadata, mapping, or proof changed during comparison")
    result = {
        "schema": "glacier_fullpool_graph_reuse_validation_v1", "status": "passed", "complete": True,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "comparison": "Exact dtype and C-order payload bytes; no numerical tolerance; shared floats finite",
        "fields": list(FIELDS), "compared": compared, "additional_molecules": full_n - old_n,
        "chunk_molecules": chunk_molecules,
        "old_graphs": {"path": str(old_path), "metadata_sha256": input_hashes[str(metadata_paths[0])], "metadata": old_meta},
        "full_graphs": {"path": str(full_path), "metadata_sha256": input_hashes[str(metadata_paths[1])], "metadata": full_meta},
        "source_manifest": {"path": str(source_manifest), "sha256": input_hashes[str(source_manifest)]},
        "fragment_reuse": {"path": str(fragment_reuse), "sha256": input_hashes[str(fragment_reuse)]},
        "index": {"path": str(index), "metadata_sha256": index_hash},
        "mapping": {"path": str(mapping_path), "sha256": mapping_hash},
        "input_metadata_sha256": {str(path): input_hashes[str(path)] for path in metadata_paths},
        "shared_payload_sha256": {name: digest.hexdigest() for name, digest in payload_hashes.items()},
        "payload_hash_semantics": "C-order data bytes in original molecule/atom/edge order, excluding NPY headers; identical for old and mapped full payloads",
        "molecule_identity_basis": "Exact-SMILES subset mapping from the pinned completed index; SMILES JSON files are not reread here",
        "limitations": ["Only shared graph payloads were compared; additional molecules' payloads were not audited.",
                        "No whole expanded-store payload hashes or fragment payload rechecks; stores must remain immutable during and after certification.",
                        "No checkpoint inference or validation-score parity was performed."],
        "source_sha256": {Path(__file__).name: _sha256(Path(__file__))},
        "numpy_version": np.__version__, "seconds": time.monotonic() - started,
        "training_performed": False,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + ".tmp")
    stream = temporary.open("x")
    try:
        with stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
        temporary.replace(out)
    finally:
        temporary.unlink(missing_ok=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("index", "source-manifest", "fragment-reuse", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--chunk-molecules", type=int, default=256)
    args = parser.parse_args()
    report = validate(args.index, args.source_manifest, args.fragment_reuse, args.out,
                      chunk_molecules=args.chunk_molecules)
    print(json.dumps({name: report[name] for name in ("status", "compared", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
