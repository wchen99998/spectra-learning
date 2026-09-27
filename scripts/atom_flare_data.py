"""NumPy/RDKit atom graphs matching jepa-fps 55dfbdf (chem49, bond8).

The on-disk arrays are memory mapped; even the full formula candidate universe
does not require keeping its molecular features in RAM. Training consumes only
ordinary NumPy arrays, with no PyTorch or PyG dependency.
"""

from collections import deque
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from itertools import islice
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.*")
ATOM_FDIM = 49
BOND_FDIM = 8
FRACTION_SLOTS = 6
H_MONO = 1.00782503207
ELEMENTS = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Si", "Se", "As", "H"]
_DEGREE = list(range(7))
_CHARGE = [-2, -1, 0, 1, 2]
_NUMH = list(range(5))
_HYB = [
    Chem.rdchem.HybridizationType.S, Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2,
]
_CHI = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED, Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
]
_BONDT = [
    Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC,
]


def _onehot(value, vocabulary: list) -> list[float]:
    result = [0.0] * (len(vocabulary) + 1)
    result[vocabulary.index(value) if value in vocabulary else -1] = 1.0
    return result


def _atom_features(atom: Chem.Atom) -> list[float]:
    return (
        _onehot(atom.GetSymbol(), ELEMENTS)
        + _onehot(atom.GetTotalDegree(), _DEGREE)
        + _onehot(atom.GetFormalCharge(), _CHARGE)
        + _onehot(atom.GetTotalNumHs(), _NUMH)
        + _onehot(atom.GetHybridization(), _HYB)
        + _onehot(atom.GetChiralTag(), _CHI)
        + [float(atom.GetIsAromatic()), float(atom.IsInRing()), atom.GetMass() / 100.0]
    )


def _bond_features(bond: Chem.Bond) -> list[float]:
    return _onehot(bond.GetBondType(), _BONDT) + [
        float(bond.GetIsConjugated()), float(bond.IsInRing()),
        float(bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE),
    ]


def atom_retained_fractions(mol: Chem.Mol) -> np.ndarray:
    """Incident bridge near-side masses / whole-molecule mass, then intact 1.

    Exactly as upstream: masses include implicit hydrogens and use the most
    common isotope, including for explicitly isotope-labelled atoms; fractions
    are deduplicated, sorted, capped at six, and padded with -1. Disconnected
    components contribute to the denominator but not a cut's reachable mass.
    """
    table = Chem.GetPeriodicTable()
    masses = [table.GetMostCommonIsotopeMass(a.GetAtomicNum())
              + a.GetTotalNumHs() * H_MONO for a in mol.GetAtoms()]
    total = float(sum(masses))
    adjacency: list[list[int]] = [[] for _ in masses]
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adjacency[u].append(v)
        adjacency[v].append(u)

    def reachable(start: int, u: int, v: int) -> tuple[float, set[int]]:
        seen = {start}
        pending = deque([start])
        mass = masses[start]
        while pending:
            node = pending.popleft()
            for neighbor in adjacency[node]:
                if (node == u and neighbor == v) or (node == v and neighbor == u):
                    continue
                if neighbor not in seen:
                    seen.add(neighbor)
                    mass += masses[neighbor]
                    pending.append(neighbor)
        return mass, seen

    fractions = [{1.0} for _ in masses]
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        mass_u, seen_u = reachable(u, u, v)
        if v in seen_u:
            continue
        mass_v, _ = reachable(v, u, v)
        if total > 0:
            fractions[u].add(mass_u / total)
            fractions[v].add(mass_v / total)
    result = np.full((len(masses), FRACTION_SLOTS), -1, np.float32)
    for i, values in enumerate(fractions):
        ordered = sorted(values)[:FRACTION_SLOTS]
        result[i, :len(ordered)] = ordered
    return result


def featurize(smiles: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Return nodes, directed edge_index[2,E], edge features, retained fractions.

    Invalid and empty molecules return None, as upstream. Atom order and the
    alternating forward/reverse bond order are RDKit's original input order.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    nodes = np.asarray([_atom_features(a) for a in mol.GetAtoms()], np.float32)
    edge_index = np.empty((2, mol.GetNumBonds() * 2), np.int32)
    edges = np.empty((mol.GetNumBonds() * 2, BOND_FDIM), np.float32)
    for i, bond in enumerate(mol.GetBonds()):
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index[:, 2 * i:2 * i + 2] = ((u, v), (v, u))
        edges[2 * i:2 * i + 2] = _bond_features(bond)
    return nodes, edge_index, edges, atom_retained_fractions(mol)


def _counts(smiles: str) -> tuple[int, int]:
    mol = Chem.MolFromSmiles(smiles)
    return (0, 0) if mol is None else (mol.GetNumAtoms(), mol.GetNumBonds() * 2)


def _map_chunk(function, values: list[str]) -> list:
    return [function(value) for value in values]


def _mapped(function, values: list[str], workers: int, chunk_size: int) -> Iterator:
    if workers == 1:
        yield from map(function, values)
    else:
        # Limit outstanding chunks: fast workers must not queue the full graph
        # universe in RAM while the writer flushes a slower disk.
        source = iter(values)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending = deque()
            for _ in range(workers * 2):
                chunk = list(islice(source, chunk_size))
                if chunk:
                    pending.append(pool.submit(_map_chunk, function, chunk))
            while pending:
                yield from pending.popleft().result()
                chunk = list(islice(source, chunk_size))
                if chunk:
                    pending.append(pool.submit(_map_chunk, function, chunk))


class GraphStore:
    """Read-only flat molecular arrays, indexed by sorted, exact input SMILES."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.metadata = json.loads((self.directory / "metadata.json").read_text())
        if self.metadata["schema"] != "atom_flare_55dfbdf_v1":
            raise ValueError("Graph cache does not contain the upstream chem49/bond8 schema")
        self.smiles: list[str] = json.loads((self.directory / "smiles.json").read_text())
        self.row_of = {smiles: i for i, smiles in enumerate(self.smiles)}
        for name in ("nodes", "edges", "senders", "receivers", "afrac", "node_offsets", "edge_offsets"):
            setattr(self, name, np.load(self.directory / f"{name}.npy", mmap_mode="r"))
        self.counts = np.diff(self.node_offsets).astype(np.int32)
        self.edge_counts = np.diff(self.edge_offsets).astype(np.int32)

    def __len__(self) -> int:
        return len(self.smiles)

    def ids(self, smiles: Iterable[str]) -> np.ndarray:
        """Map valid strings to IDs; a missing string raises KeyError."""
        return np.fromiter((self.row_of[s] for s in smiles), dtype=np.int32)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        lo, hi = self.node_offsets[index:index + 2]
        elo, ehi = self.edge_offsets[index:index + 2]
        return (self.nodes[lo:hi], np.stack((self.senders[elo:ehi], self.receivers[elo:ehi])),
                self.edges[elo:ehi], self.afrac[lo:hi])


def build_graph_store(
    smiles: Iterable[str], directory: str | Path, *, max_atoms: int = 128,
    workers: int = 1, chunk_size: int = 128,
) -> GraphStore:
    """Build a streaming, memory-mapped cache; omit invalid molecules only.

    A light first pass determines offsets; the second pass writes one graph at
    a time. Cache identity includes every requested SMILES, even invalid ones.
    Existing completed stores must match exactly, preventing stale subset reuse.
    """
    if not 0 < max_atoms <= np.iinfo(np.int16).max:
        raise ValueError("max_atoms must fit positive int16 molecule-local indices")
    if workers < 1 or chunk_size < 1:
        raise ValueError("workers and chunk_size must be positive")
    unique = sorted(set(smiles))
    digest = sha256()
    for value in unique:
        digest.update(value.encode())
        digest.update(b"\0")
    fingerprint = digest.hexdigest()
    directory = Path(directory)
    manifest = directory / "metadata.json"
    if manifest.exists():
        store = GraphStore(directory)
        if store.metadata["smiles_sha256"] != fingerprint:
            raise ValueError(f"Requested molecule set differs from graph cache {directory}")
        if store.metadata["max_nodes"] > max_atoms:
            raise ValueError("Graph cache contains molecules exceeding requested max_atoms")
        return store
    sizes = np.empty((len(unique), 2), np.int32)
    for i, counts in enumerate(_mapped(_counts, unique, workers, chunk_size)):
        sizes[i] = counts
    if len(sizes) == 0 or not np.any(sizes[:, 0] > 0):
        raise ValueError("No valid, nonempty molecules to cache")
    too_large = np.flatnonzero(sizes[:, 0] > max_atoms)
    if len(too_large):
        index = int(too_large[0])
        raise ValueError(f"Molecule has {sizes[index, 0]} atoms, exceeding max_atoms={max_atoms}: {unique[index]}")
    valid = sizes[:, 0] > 0
    invalid = [s for s, keep in zip(unique, valid) if not keep]
    unique = [s for s, keep in zip(unique, valid) if keep]
    sizes = sizes[valid]
    node_offsets = np.concatenate((np.zeros(1, np.int64), sizes[:, 0].cumsum(dtype=np.int64)))
    edge_offsets = np.concatenate((np.zeros(1, np.int64), sizes[:, 1].cumsum(dtype=np.int64)))
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "node_offsets.npy", node_offsets)
    np.save(directory / "edge_offsets.npy", edge_offsets)
    shapes = {
        "nodes": (int(node_offsets[-1]), ATOM_FDIM),
        "edges": (int(edge_offsets[-1]), BOND_FDIM),
        "senders": (int(edge_offsets[-1]),), "receivers": (int(edge_offsets[-1]),),
        "afrac": (int(node_offsets[-1]), FRACTION_SLOTS),
    }
    arrays = {name: np.lib.format.open_memmap(
        directory / f"{name}.npy", mode="w+", shape=shape,
        dtype=np.int16 if name in ("senders", "receivers") else np.float32,
    ) for name, shape in shapes.items()}
    for i, graph in enumerate(_mapped(featurize, unique, workers, chunk_size)):
        nodes, endpoints, edges, fractions = graph
        lo, hi = node_offsets[i:i + 2]
        elo, ehi = edge_offsets[i:i + 2]
        arrays["nodes"][lo:hi] = nodes
        arrays["edges"][elo:ehi] = edges
        arrays["senders"][elo:ehi] = endpoints[0]
        arrays["receivers"][elo:ehi] = endpoints[1]
        arrays["afrac"][lo:hi] = fractions
        if (i + 1) % 10000 == 0:
            print(f"Atom FLARE graphs: {i + 1:,}/{len(unique):,}", flush=True)
    for array in arrays.values():
        array.flush()
    (directory / "smiles.json").write_text(json.dumps(unique))
    (directory / "invalid_smiles.json").write_text(json.dumps(invalid))
    metadata = {
        "schema": "atom_flare_55dfbdf_v1", "smiles_sha256": fingerprint,
        "molecules": len(unique), "invalid_molecules": len(invalid),
        "atoms": int(node_offsets[-1]), "directed_edges": int(edge_offsets[-1]),
        "max_nodes": int(sizes[:, 0].max()), "max_edges": int(sizes[:, 1].max()),
        "molecules_over_upstream_atom_cap96": int((sizes[:, 0] > 96).sum()),
        "atom_dim": ATOM_FDIM, "bond_dim": BOND_FDIM, "fraction_slots": FRACTION_SLOTS,
    }
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    return GraphStore(directory)


def batch_graphs(
    store: GraphStore, graph_ids: Iterable[int], *, graph_capacity: int,
    node_capacity: int, edge_capacity: int, atom_cap: int,
) -> dict[str, np.ndarray]:
    """Pack a disjoint graph union with explicit static capacities for JAX.

    Repeated IDs remain distinct graph instances, matching upstream true
    molecules in a batch. The caller deduplicates only its sampled decoys.
    No molecule is truncated; capacity overflow raises before packing.
    """
    ids = np.asarray(list(graph_ids), np.int32)
    if (ids < 0).any() or (ids >= len(store)).any():
        raise ValueError("Graph ID is outside the graph store")
    counts, edge_counts = store.counts[ids], store.edge_counts[ids]
    if len(ids) > graph_capacity or counts.sum() > node_capacity or edge_counts.sum() > edge_capacity:
        raise ValueError("Graph batch exceeds graph, node, or edge capacity")
    if len(ids) and counts.max() > atom_cap:
        raise ValueError("Graph batch exceeds atom_cap; truncating atoms is not supported")
    result = {
        "nodes": np.zeros((node_capacity, ATOM_FDIM), np.float32),
        "edges": np.zeros((edge_capacity, BOND_FDIM), np.float32),
        "senders": np.zeros(edge_capacity, np.int32),
        "receivers": np.zeros(edge_capacity, np.int32),
        "node_mask": np.zeros(node_capacity, bool),
        "edge_mask": np.zeros(edge_capacity, bool),
        "graph_idx": np.zeros(node_capacity, np.int32),
        "node_slots": np.zeros(node_capacity, np.int32),
        "atom_mask": np.zeros((graph_capacity, atom_cap), bool),
        "afrac": np.full((graph_capacity, atom_cap, FRACTION_SLOTS), -1, np.float32),
        "n_node": np.zeros(graph_capacity, np.int32),
    }
    npos, epos = 0, 0
    for slot, index in enumerate(ids):
        lo, hi = store.node_offsets[index:index + 2]
        elo, ehi = store.edge_offsets[index:index + 2]
        n, e = int(hi - lo), int(ehi - elo)
        result["nodes"][npos:npos + n] = store.nodes[lo:hi]
        result["edges"][epos:epos + e] = store.edges[elo:ehi]
        result["senders"][epos:epos + e] = store.senders[elo:ehi].astype(np.int32) + npos
        result["receivers"][epos:epos + e] = store.receivers[elo:ehi].astype(np.int32) + npos
        result["node_mask"][npos:npos + n] = True
        result["edge_mask"][epos:epos + e] = True
        result["graph_idx"][npos:npos + n] = slot
        result["node_slots"][npos:npos + n] = np.arange(n)
        result["atom_mask"][slot, :n] = True
        result["afrac"][slot, :n] = store.afrac[lo:hi]
        result["n_node"][slot] = n
        npos += n
        epos += e
    return result
