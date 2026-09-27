"""Prove exact reuse of every original molecule in an expanded fragment store.

Only shared molecules are compared; proposal merging and assembly establish
coverage of the additional molecules. No pass report is written on failure.
"""

import argparse
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np

from glacier_fragment_data import FRAGMENT_FIELDS, SCHEMA


CHUNK_MOLECULES = 256


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def _arrays(directory: Path, metadata: dict) -> dict[str, np.ndarray]:
    n, f = metadata["molecules"], metadata["fragments"]
    layout = {
        "frag_off": (n + 1, np.int64), "mem_off": (f + 1, np.int64),
        "bnd_off": (f + 1, np.int64), "atoms": (metadata["memberships"], np.int16),
        "bnd_atoms": (metadata["boundary_memberships"], np.int16),
        "total_mass": (n, np.float32), "unsupported": (n, np.bool_),
        **{name: (f, dtype) for name, dtype in FRAGMENT_FIELDS.items()},
    }
    arrays = {}
    for name, (length, dtype) in layout.items():
        array = np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        if array.shape != (length,) or array.dtype != dtype:
            raise ValueError(f"Invalid fragment array shape or dtype: {directory / name}")
        arrays[name] = array
    for offset, payload in (("frag_off", "inten"), ("mem_off", "atoms"), ("bnd_off", "bnd_atoms")):
        if arrays[offset][0] != 0 or arrays[offset][-1] != len(arrays[payload]):
            raise ValueError(f"Invalid fragment offsets: {directory / offset}")
    return arrays


def _ranges(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Concatenate integer ranges without a Python loop over fragments."""
    output_starts = np.cumsum(lengths, dtype=np.int64) - lengths
    return np.arange(int(lengths.sum()), dtype=np.int64) + np.repeat(starts - output_starts, lengths)


def _equal(left: np.ndarray, right: np.ndarray, field: str, start: int, stop: int) -> None:
    if not np.array_equal(left, right):
        raise ValueError(f"Shared fragment {field} differs for old molecule IDs [{start}, {stop})")


def validate(index: Path, fragments: Path, out: Path) -> dict:
    if out.exists():
        raise FileExistsError(f"Validation report already exists: {out}")
    index_path = index / "metadata.json"
    manifest = json.loads(index_path.read_text())
    if manifest.get("schema") != "glacier_fullpool_index_v1" or manifest.get("complete") is not True:
        raise ValueError("Expected a completed full-pool index")
    old_path = Path(manifest["existing_fragments"])
    metadata_paths = [old_path / "metadata.json", fragments / "metadata.json"]
    old_meta, full_meta = [json.loads(path.read_text()) for path in metadata_paths]
    old_n, full_n = manifest["reused_molecules"], manifest["full_molecules"]
    if not 0 < old_n <= full_n:
        raise ValueError("Invalid shared/full molecule counts")
    for metadata, fingerprint, count in (
        (old_meta, manifest["old_smiles_sha256"], old_n),
        (full_meta, manifest["full_smiles_sha256"], full_n),
    ):
        if (metadata.get("schema") != SCHEMA or metadata.get("smiles_sha256") != fingerprint
                or metadata.get("molecules") != count):
            raise ValueError("Fragment store identity differs from the prepared index")
    if any(old_meta[name] != full_meta[name] for name in ("f_max", "min_atoms", "min_frac")):
        raise ValueError("Fragment filtering settings differ")
    old_sha = _sha256(metadata_paths[0])
    if old_sha != manifest["input_metadata_sha256"].get(str(metadata_paths[0])):
        raise ValueError("Original fragment metadata changed since index preparation")
    mapping_path = index / "old_to_full.npy"
    mapping_sha = _sha256(mapping_path)
    if mapping_sha != manifest["mapping_sha256"]["old_to_full.npy"]:
        raise ValueError("Old-to-full mapping SHA mismatch")
    mapping = np.load(mapping_path, mmap_mode="r", allow_pickle=False)
    if (mapping.dtype != np.int32 or mapping.shape != (old_n,)
            or np.any(mapping < 0) or np.any(mapping >= full_n)
            or np.any(mapping[1:] <= mapping[:-1])):
        raise ValueError("Old-to-full mapping must contain unique increasing valid IDs")
    old, full = _arrays(old_path, old_meta), _arrays(fragments, full_meta)
    counts = dict(molecules=0, fragments=0, memberships=0, boundary_memberships=0)
    for start in range(0, old_n, CHUNK_MOLECULES):
        stop = min(start + CHUNK_MOLECULES, old_n)
        ids = mapping[start:stop]
        old_offsets = old["frag_off"][start:stop + 1]
        lengths = np.diff(old_offsets)
        full_starts, full_stops = full["frag_off"][ids], full["frag_off"][ids + 1]
        if (np.any(lengths < 1) or np.any(lengths > old_meta["f_max"])
                or old_offsets[0] < 0 or old_offsets[-1] > old_meta["fragments"]
                or np.any(full_starts < 0) or np.any(full_stops > full_meta["fragments"])):
            raise ValueError(f"Invalid fragment count/offset at old molecule IDs [{start}, {stop})")
        _equal(lengths, full_stops - full_starts, "count", start, stop)
        first, last = old_offsets[0], old_offsets[-1]
        full_fragments = _ranges(full_starts, lengths)
        for field in FRAGMENT_FIELDS:
            _equal(old[field][first:last], full[field][full_fragments], field, start, stop)
        for offset, members, label in (("mem_off", "atoms", "memberships"),
                                        ("bnd_off", "bnd_atoms", "boundary_memberships")):
            old_member_offsets = old[offset][first:last + 1]
            old_lengths = np.diff(old_member_offsets)
            full_member_starts = full[offset][full_fragments]
            full_member_stops = full[offset][full_fragments + 1]
            if (np.any(old_lengths < 0) or old_member_offsets[0] < 0
                    or old_member_offsets[-1] > len(old[members])
                    or np.any(full_member_starts < 0) or np.any(full_member_stops > len(full[members]))):
                raise ValueError(f"Invalid shared {offset} at old molecule IDs [{start}, {stop})")
            # Equal per-fragment lengths prove equality of every per-molecule
            # offset vector after subtracting its first (global) offset.
            _equal(old_lengths, full_member_stops - full_member_starts, offset, start, stop)
            member_starts = full[offset][full_starts]
            member_lengths = full[offset][full_stops] - member_starts
            full_members = _ranges(member_starts, member_lengths)
            _equal(old[members][old_member_offsets[0]:old_member_offsets[-1]],
                   full[members][full_members], members, start, stop)
            counts[label] += int(old_lengths.sum())
        for field in ("total_mass", "unsupported"):
            _equal(old[field][start:stop], full[field][ids], field, start, stop)
        counts["molecules"] += stop - start
        counts["fragments"] += int(lengths.sum())
        if (start // CHUNK_MOLECULES + 1) % 100 == 0 or stop == old_n:
            print(f"Verified reused fragments: {stop:,}/{old_n:,} molecules", flush=True)
    result = {
        "schema": "glacier_fullpool_fragment_reuse_validation_v1", "status": "passed", "complete": True,
        "comparison": "Exact dtype and array values; no numerical tolerance",
        "compared": counts, "additional_molecules": full_n - old_n,
        "additional_molecule_coverage": "Validated upstream by proposal merge and fragment assembly",
        "fields": [*FRAGMENT_FIELDS, "atoms", "bnd_atoms", "total_mass", "unsupported"],
        "offsets": "Exact fragment counts and per-molecule normalized mem_off/bnd_off",
        "chunk_molecules": CHUNK_MOLECULES,
        "settings": {name: old_meta[name] for name in ("f_max", "min_atoms", "min_frac")},
        "index": {"path": str(index), "metadata_sha256": _sha256(index_path)},
        "mapping": {"path": str(mapping_path), "sha256": mapping_sha},
        "old_fragments": {"path": str(old_path), "metadata_sha256": old_sha,
                          "smiles_sha256": old_meta["smiles_sha256"], "molecules": old_n},
        "full_fragments": {"path": str(fragments), "metadata_sha256": _sha256(metadata_paths[1]),
                           "smiles_sha256": full_meta["smiles_sha256"], "molecules": full_n},
        "source_sha256": {Path(__file__).name: _sha256(Path(__file__)),
                          "glacier_fragment_data.py": _sha256(Path(__file__).with_name("glacier_fragment_data.py"))},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(out)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--fragments", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.index, args.fragments, args.out), indent=2), flush=True)


if __name__ == "__main__":
    main()
