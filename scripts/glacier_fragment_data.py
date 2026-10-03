"""GLACIER predicted atom sets -> NumPy fragment features in original SMILES order.

The proposal generator must remap GLACIER's canonical atom indices before writing
shards. This module never enumerates substitute fragments and never drops missing
molecules. The intact anchor is always present, matching upstream flare_frag.py.
No PyTorch dependency is needed for preparing or consuming this cache.
"""

import argparse
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from functools import partial
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from rdkit import Chem

from atom_flare_data import GraphStore, H_MONO, concatenated_ranges


SCHEMA = "glacier_fragments_original_order_v1"
FRAGMENT_FIELDS = {
    "n_cuts": np.int8, "kind": np.int8, "mass_frac": np.float32,
    "n_atoms": np.int16, "inten": np.float32,
}


def build_fragment_record(
    smiles: str, atomsets: Iterable[np.ndarray], intensities: Iterable[float], *,
    f_max: int = 128, min_atoms: int = 3, min_frac: float = 0.05,
) -> dict[str, np.ndarray | int | np.float32]:
    """Upstream _build_one_glacier semantics, rejecting corrupt atom indices.

    Predicted sets arrive in decreasing intensity order. Filtering preserves that
    order; duplicate memberships keep the first prediction. Masses intentionally
    use the most common isotope plus implicit H, as in upstream FLARE.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"Invalid or empty fragment molecule: {smiles}")
    n = mol.GetNumAtoms()
    table = Chem.GetPeriodicTable()
    mass = np.asarray([
        table.GetMostCommonIsotopeMass(a.GetAtomicNum()) + a.GetTotalNumHs() * H_MONO
        for a in mol.GetAtoms()
    ], np.float64)
    total = float(mass.sum())
    adjacency: list[list[int]] = [[] for _ in range(n)]
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adjacency[u].append(v)
        adjacency[v].append(u)
    members = [list(range(n))]
    boundaries: list[list[int]] = [[]]
    cuts, kinds, fractions, intensity = [0], [0], [1.0], [1.0]
    seen = {frozenset(range(n))}
    for raw_atoms, value in zip(atomsets, intensities, strict=True):
        raw_atoms = np.asarray(raw_atoms)
        if raw_atoms.ndim != 1 or raw_atoms.dtype.kind not in "iu":
            raise ValueError("Fragment memberships must be one-dimensional integer arrays")
        if np.any(raw_atoms < 0) or np.any(raw_atoms >= n):
            raise ValueError("Fragment membership is outside original SMILES atom order")
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Fragment intensity must be finite and in [0, 1]")
        atoms = sorted(set(raw_atoms.tolist()))
        key = frozenset(atoms)
        if len(atoms) < min_atoms or key in seen or len(members) >= f_max:
            continue
        fraction = float(mass[atoms].sum()) / total if total > 0 else 0.0
        if fraction < min_frac:
            continue
        boundary, broken = [], 0
        for atom in atoms:
            leaving = sum(neighbor not in key for neighbor in adjacency[atom])
            if leaving:
                boundary.append(atom)
                broken += leaving
        seen.add(key)
        members.append(atoms)
        boundaries.append(boundary)
        cuts.append(min(broken, 7))
        kinds.append(1)
        fractions.append(fraction)
        intensity.append(float(value))
    lengths = np.asarray([len(a) for a in members], np.int32)
    return {
        "atoms": np.asarray([a for fragment in members for a in fragment], np.int16),
        "atom_len": lengths,
        "bnd_atoms": np.asarray([a for boundary in boundaries for a in boundary], np.int16),
        "bnd_len": np.asarray([len(a) for a in boundaries], np.int32),
        "n_cuts": np.asarray(cuts, np.int8), "kind": np.asarray(kinds, np.int8),
        "mass_frac": np.asarray(fractions, np.float32),
        "n_atoms": lengths.astype(np.int16), "inten": np.asarray(intensity, np.float32),
        "total": np.float32(total), "F": len(members),
    }


def _offsets(lengths: np.ndarray) -> np.ndarray:
    return np.concatenate((np.zeros(1, np.int64), np.cumsum(lengths, dtype=np.int64)))


def _check_offsets(values: np.ndarray, rows: int, total: int, name: str) -> None:
    if (values.dtype.kind not in "iu" or values.shape != (rows + 1,)
            or values[0] != 0 or values[-1] != total or np.any(values[1:] < values[:-1])):
        raise ValueError(f"Invalid proposal {name} offsets")


class FragmentStore:
    """Memory-mapped fragment CSR arrays indexed by the unchanged GraphStore IDs."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.metadata = json.loads((self.directory / "metadata.json").read_text())
        if self.metadata["schema"] != SCHEMA:
            raise ValueError("Fragment cache does not contain original-order GLACIER fragments")
        for name in (
            "frag_off", "mem_off", "atoms", "bnd_off", "bnd_atoms", "total_mass",
            "unsupported", *FRAGMENT_FIELDS,
        ):
            setattr(self, name, np.load(self.directory / f"{name}.npy", mmap_mode="r"))
        self.counts = np.diff(self.frag_off).astype(np.int32)
        self.atom_counts = np.asarray(self.n_atoms[self.frag_off[:-1]], np.int32)

    def __len__(self) -> int:
        return self.metadata["molecules"]


def _process_proposal_shard(
    task: tuple[Path, Path, np.ndarray, list[str], np.ndarray, np.ndarray], *,
    f_max: int, min_atoms: int, min_frac: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    proposal_path, processed_path, ids, smiles, atom_counts, unsupported = task
    with np.load(proposal_path, allow_pickle=False) as shard:
        if not np.array_equal(shard["graph_ids"], ids):
            raise ValueError("Proposal graph IDs changed after coverage validation")
        frag_off, mem_off = shard["frag_off"], shard["mem_off"]
        atoms, inten = shard["atoms"], shard["inten"]
        if atoms.ndim != 1 or atoms.dtype.kind not in "iu" or inten.ndim != 1:
            raise ValueError("Invalid proposal atom or intensity array")
        _check_offsets(frag_off, len(ids), len(inten), "frag_off")
        _check_offsets(mem_off, len(inten), len(atoms), "mem_off")
        records = []
        for index, molecule in enumerate(smiles):
            lo, hi = frag_off[index:index + 2]
            if unsupported[index] and hi != lo:
                raise ValueError("Unsupported GLACIER graph has predicted fragments")
            sets = (atoms[mem_off[j]:mem_off[j + 1]] for j in range(lo, hi))
            record = build_fragment_record(
                molecule, sets, inten[lo:hi], f_max=f_max,
                min_atoms=min_atoms, min_frac=min_frac,
            )
            if record["n_atoms"][0] != atom_counts[index]:
                raise ValueError("Fragment molecule atom count differs from graph store")
            records.append(record)
        fragment_counts = np.asarray([record["F"] for record in records], np.int32)
        member_counts = np.asarray([len(record["atoms"]) for record in records], np.int64)
        boundary_counts = np.asarray([len(record["bnd_atoms"]) for record in records], np.int64)
        packed = {
            field: np.concatenate([record[field] for record in records])
            for field in ("atoms", "atom_len", "bnd_atoms", "bnd_len", *FRAGMENT_FIELDS)
        }
        np.savez(
            processed_path, graph_ids=ids, frag_off=_offsets(fragment_counts),
            mol_mem_off=_offsets(member_counts), mol_bnd_off=_offsets(boundary_counts),
            total_mass=np.asarray([record["total"] for record in records], np.float32),
            **packed,
        )
    return ids, fragment_counts, member_counts, boundary_counts


def build_fragment_store(
    graphs: GraphStore, proposals: str | Path, directory: str | Path, *,
    f_max: int = 128, min_atoms: int = 3, min_frac: float = 0.05, workers: int = 1,
) -> FragmentStore:
    """Transform bounded shards once, then assemble arrays in GraphStore ID order.

    Proposal metadata must list completed_shards, smiles_sha256 and explicit
    unsupported_graph_ids (>100 atoms). Every graph ID must occur exactly once.
    Empty predictions are valid; absent rows are an error. Temporary processed
    shards bound RAM use and avoid doing RDKit chemistry twice. Optional workers
    process independent shards; global coverage and final ordered assembly stay
    in the parent. Each task carries only its shard's molecule strings.
    """
    if f_max < 1 or min_atoms < 1 or not 0 <= min_frac <= 1:
        raise ValueError("Invalid fragment cap, minimum atom count, or mass fraction")
    if workers < 1:
        raise ValueError("Fragment processing workers must be positive")
    proposals, directory = Path(proposals), Path(directory)
    source = json.loads((proposals / "metadata.json").read_text())
    if source["smiles_sha256"] != graphs.metadata["smiles_sha256"]:
        raise ValueError("Proposal molecule fingerprint differs from the graph store")
    shard_names = source["completed_shards"]
    if len(set(shard_names)) != len(shard_names):
        raise ValueError("Duplicate completed proposal shards")
    if any(Path(name).name != name or not name.endswith(".npz") for name in shard_names):
        raise ValueError("Completed shard names must be .npz basenames")
    identity = {
        "schema": SCHEMA, "smiles_sha256": source["smiles_sha256"],
        "proposal_metadata_sha256": sha256(
            json.dumps(source, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "f_max": f_max, "min_atoms": min_atoms, "min_frac": min_frac,
    }
    manifest = directory / "metadata.json"
    if manifest.exists():
        store = FragmentStore(directory)
        if any(store.metadata[key] != value for key, value in identity.items()):
            raise ValueError("Existing fragment cache differs from requested proposals or settings")
        return store
    unsupported_ids = np.asarray(source.get("unsupported_graph_ids", []), np.int64)
    if (unsupported_ids.ndim != 1 or np.any(unsupported_ids < 0)
            or np.any(unsupported_ids >= len(graphs))):
        raise ValueError("Unsupported graph IDs are outside the graph store")
    if len(unsupported_ids) and np.any(graphs.counts[unsupported_ids] <= 100):
        raise ValueError("Unsupported GLACIER graphs must exceed 100 atoms")
    unsupported = np.zeros(len(graphs), bool)
    unsupported[unsupported_ids] = True
    seen = np.zeros(len(graphs), bool)
    fragment_counts = np.zeros(len(graphs), np.int32)
    member_counts = np.zeros(len(graphs), np.int64)
    boundary_counts = np.zeros(len(graphs), np.int64)
    directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="processed-", dir=directory) as temporary:
        temporary = Path(temporary)
        def tasks():
            for shard_index, name in enumerate(shard_names):
                with np.load(proposals / name, allow_pickle=False) as shard:
                    ids = shard["graph_ids"]
                if (ids.ndim != 1 or ids.dtype.kind not in "iu" or len(ids) == 0
                        or np.any(ids < 0) or np.any(ids >= len(graphs))):
                    raise ValueError("Proposal graph IDs must be nonempty valid integer IDs")
                if len(np.unique(ids)) != len(ids) or np.any(seen[ids]):
                    raise ValueError("Duplicate proposal graph IDs")
                seen[ids] = True
                yield (proposals / name, temporary / f"{shard_index}.npz", ids,
                       [graphs.smiles[graph_id] for graph_id in ids], graphs.counts[ids], unsupported[ids])

        process = partial(_process_proposal_shard, f_max=f_max, min_atoms=min_atoms, min_frac=min_frac)
        # Exit the executor before deleting the temporary directory, including
        # on worker/coverage errors, so no worker can recreate a cleaned file.
        with (ProcessPoolExecutor(max_workers=workers) if workers > 1 else nullcontext()) as executor:
            results = map(process, tasks()) if executor is None else executor.map(
                process, tasks(), buffersize=2 * workers,
            )
            for shard_index, (ids, fragments, members, boundaries) in enumerate(results):
                fragment_counts[ids], member_counts[ids], boundary_counts[ids] = fragments, members, boundaries
                print(
                    f"GLACIER fragment shards: {shard_index + 1}/{len(shard_names)}", flush=True,
                )
        if not seen.all():
            raise ValueError(f"Missing GLACIER proposals for {int((~seen).sum())} graph IDs")
        frag_off = _offsets(fragment_counts)
        mol_mem_off, mol_bnd_off = _offsets(member_counts), _offsets(boundary_counts)
        n_fragments = int(frag_off[-1])
        arrays = {}
        shapes = {
            "atoms": (int(mol_mem_off[-1]),), "bnd_atoms": (int(mol_bnd_off[-1]),),
            "mem_off": (n_fragments + 1,), "bnd_off": (n_fragments + 1,),
            "total_mass": (len(graphs),), **{field: (n_fragments,) for field in FRAGMENT_FIELDS},
        }
        types = {"atoms": np.int16, "bnd_atoms": np.int16, "mem_off": np.int64,
                 "bnd_off": np.int64, "total_mass": np.float32, **FRAGMENT_FIELDS}
        for field, shape in shapes.items():
            arrays[field] = np.lib.format.open_memmap(
                directory / f"{field}.npy", mode="w+", shape=shape, dtype=types[field],
            )
        for shard_index in range(len(shard_names)):
            with np.load(temporary / f"{shard_index}.npz", allow_pickle=False) as shard:
                # NPZ members are materialized once per shard, never once per molecule.
                data = {field: shard[field] for field in shard.files}
                for index, graph_id in enumerate(data["graph_ids"]):
                    flo, fhi = frag_off[graph_id:graph_id + 2]
                    sfl, sfh = data["frag_off"][index:index + 2]
                    mlo, mhi = mol_mem_off[graph_id:graph_id + 2]
                    sml, smh = data["mol_mem_off"][index:index + 2]
                    blo, bhi = mol_bnd_off[graph_id:graph_id + 2]
                    sbl, sbh = data["mol_bnd_off"][index:index + 2]
                    arrays["atoms"][mlo:mhi] = data["atoms"][sml:smh]
                    arrays["bnd_atoms"][blo:bhi] = data["bnd_atoms"][sbl:sbh]
                    arrays["mem_off"][flo:fhi + 1] = mlo + _offsets(data["atom_len"][sfl:sfh])
                    arrays["bnd_off"][flo:fhi + 1] = blo + _offsets(data["bnd_len"][sfl:sfh])
                    arrays["total_mass"][graph_id] = data["total_mass"][index]
                    for field in FRAGMENT_FIELDS:
                        arrays[field][flo:fhi] = data[field][sfl:sfh]
        for array in arrays.values():
            array.flush()
    np.save(directory / "frag_off.npy", frag_off)
    np.save(directory / "unsupported.npy", unsupported)
    metadata = {
        **identity, "molecules": len(graphs), "fragments": n_fragments,
        "memberships": int(mol_mem_off[-1]), "boundary_memberships": int(mol_bnd_off[-1]),
        "max_fragments": int(fragment_counts.max()),
        "intact_only_molecules": int((fragment_counts == 1).sum()),
        "unsupported_molecules": int(unsupported.sum()), "proposals": source,
    }
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    return FragmentStore(directory)


def batch_fragments(
    store: FragmentStore, graph_ids: Iterable[int], *, graph_capacity: int, atom_cap: int,
) -> dict[str, np.ndarray]:
    """Pack fragment mean-pooling matrices without truncating any molecule."""
    ids = np.asarray(list(graph_ids), np.int32)
    if (ids < 0).any() or (ids >= len(store)).any():
        raise ValueError("Graph ID is outside the fragment store")
    if len(ids) > graph_capacity:
        raise ValueError("Fragment batch exceeds graph capacity")
    if len(ids) and store.atom_counts[ids].max() > atom_cap:
        raise ValueError("Fragment batch exceeds atom_cap; truncating atoms is not supported")
    shape = (graph_capacity, store.metadata["f_max"])
    result = {
        "member_mean": np.zeros((*shape, atom_cap), np.float32),
        "boundary_mean": np.zeros((*shape, atom_cap), np.float32),
        "fragment_mask": np.zeros(shape, bool),
        "fragment_n_cuts": np.zeros(shape, np.int32),
        "fragment_kind": np.zeros(shape, np.int32),
        "fragment_mass_frac": np.zeros(shape, np.float32),
        "fragment_log_n_atoms": np.zeros(shape, np.float32),
        "fragment_intensity": np.zeros(shape, np.float32),
    }
    lo = store.frag_off[ids].astype(np.int64)
    counts = (store.frag_off[ids + 1] - store.frag_off[ids]).astype(np.int64)
    fragments = concatenated_ranges(lo, counts)
    slot = np.repeat(np.arange(len(ids)), counts)
    position = np.arange(len(fragments)) - np.repeat(np.cumsum(counts) - counts, counts)
    result["fragment_mask"][slot, position] = True
    result["fragment_n_cuts"][slot, position] = store.n_cuts[fragments]
    result["fragment_kind"][slot, position] = store.kind[fragments]
    result["fragment_mass_frac"][slot, position] = store.mass_frac[fragments]
    result["fragment_log_n_atoms"][slot, position] = np.log(store.n_atoms[fragments].astype(np.float32))
    result["fragment_intensity"][slot, position] = store.inten[fragments]
    for name, offsets, atoms in (("member_mean", store.mem_off, store.atoms),
                                 ("boundary_mean", store.bnd_off, store.bnd_atoms)):
        start = offsets[fragments].astype(np.int64)
        lengths = offsets[fragments + 1].astype(np.int64) - start
        members = atoms[concatenated_ranges(start, lengths)]
        weight = (1.0 / np.maximum(lengths, 1)).astype(np.float32)
        result[name][np.repeat(slot, lengths), np.repeat(position, lengths), members] = np.repeat(weight, lengths)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--f-max", type=int, default=128)
    parser.add_argument("--min-atoms", type=int, default=3)
    parser.add_argument("--min-frac", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel proposal-shard chemistry workers; final assembly remains serial")
    args = parser.parse_args()
    store = build_fragment_store(
        GraphStore(args.graphs), args.proposals, args.out,
        f_max=args.f_max, min_atoms=args.min_atoms, min_frac=args.min_frac, workers=args.workers,
    )
    print(json.dumps(store.metadata, indent=2))


if __name__ == "__main__":
    main()
