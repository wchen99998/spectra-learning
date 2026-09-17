"""Original FLARE chemistry, combined without changing RDKit atom order."""
from collections import deque
import numpy as np
from rdkit import Chem, RDLogger
from flare_features import chemistry_features, fragment_features, morgan_atom_ids
RDLogger.DisableLog("rdApp.*")

ELEMENTS = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Si", "Se", "As", "H"]
_ELEM_IX = {e: i for i, e in enumerate(ELEMENTS)}
_DEGREE = list(range(7))          # 0..6+
_CHARGE = [-2, -1, 0, 1, 2]
_NUMH = list(range(5))            # 0..4+
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


def _onehot(v, vocab):
    x = [0.0] * (len(vocab) + 1)
    x[vocab.index(v) if v in vocab else -1] = 1.0
    return x


def _atom_feats(a):
    return (
        _onehot(a.GetSymbol(), ELEMENTS)
        + _onehot(a.GetTotalDegree(), _DEGREE)
        + _onehot(a.GetFormalCharge(), _CHARGE)
        + _onehot(a.GetTotalNumHs(), _NUMH)
        + _onehot(a.GetHybridization(), _HYB)
        + _onehot(a.GetChiralTag(), _CHI)
        + [float(a.GetIsAromatic()), float(a.IsInRing()), a.GetMass() / 100.0]
    )


def _bond_feats(b):
    return (
        _onehot(b.GetBondType(), _BONDT)
        + [float(b.GetIsConjugated()), float(b.IsInRing()), float(b.GetStereo() != Chem.rdchem.BondStereo.STEREONONE)]
    )


ATOM_FDIM = len(_atom_feats(Chem.MolFromSmiles("CC").GetAtomWithIdx(0)))
BOND_FDIM = len(_bond_feats(Chem.MolFromSmiles("CC").GetBondWithIdx(0)))


H_MONO = 1.00782503207  # monoisotopic mass of 1H


def _node_masses(mol, monoisotopic=True):
    """Per-atom mass INCLUDING that atom's implicit hydrogens (H folds into its heavy
    atom, matching the implicit-H graph in mol_to_graph). Monoisotopic by default."""
    pt = Chem.GetPeriodicTable()
    hmass = H_MONO if monoisotopic else pt.GetAtomicWeight(1)
    out = []
    for a in mol.GetAtoms():
        z = a.GetAtomicNum()
        base = pt.GetMostCommonIsotopeMass(z) if monoisotopic else a.GetMass()
        out.append(base + a.GetTotalNumHs() * hmass)
    return out


def _adjacency(mol):
    adj = [[] for _ in range(mol.GetNumAtoms())]
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        adj[i].append(j); adj[j].append(i)
    return adj


def _reach_mass(adj, node_mass, start, blocked):
    """BFS mass + visited set from `start`, refusing to traverse the single bond
    `blocked` (an unordered (u, v) pair). If the other endpoint is NOT reached, the
    bond is a bridge and the visited set is exactly one fragment."""
    bu, bv = blocked
    seen = {start}
    dq = deque([start])
    total = node_mass[start]
    while dq:
        x = dq.popleft()
        for y in adj[x]:
            if (x == bu and y == bv) or (x == bv and y == bu):
                continue
            if y not in seen:
                seen.add(y); total += node_mass[y]; dq.append(y)
    return total, seen


def fragment_masses(mol, monoisotopic=True):
    """Per-atom incident-bridge fragment masses.

    Returns (per_atom, total_mass):
      per_atom[i] = list of (far, near) monoisotopic-mass tuples, one per bridge bond
                    incident to atom i. `far` = mass of the fragment on the NEIGHBOUR's
                    side (what splits off if that bond breaks); `near` = mass of the
                    fragment on atom i's own side. Ring atoms with only ring bonds get
                    an empty list. Either side can be the observed ion (charge retention
                    is not graph-decidable), so both are returned.
      total_mass  = monoisotopic mass of the whole input (all fragments if disconnected).

    O(E*(V+E)); trivial for molecule-sized graphs. Accepts a SMILES or an RDKit Mol;
    returns None if unparseable/empty."""
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    node_mass = _node_masses(mol, monoisotopic)
    adj = _adjacency(mol)
    per_atom = [[] for _ in range(mol.GetNumAtoms())]
    for b in mol.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        mass_u, seen_u = _reach_mass(adj, node_mass, u, (u, v))
        if v in seen_u:
            continue  # non-bridge (ring bond): a single cut doesn't fragment
        mass_v, _ = _reach_mass(adj, node_mass, v, (u, v))
        per_atom[u].append((mass_v, mass_u))  # v's side splits off from u
        per_atom[v].append((mass_u, mass_v))  # u's side splits off from v
    return per_atom, float(sum(node_mass))


# We cache only the RAW mass scalars per atom (below) and Fourier-EXPAND them on the
# GPU at batch time (see MolGNN.forward / expand_mass). Baking the full Fourier block
# into the cache blew it up to ~22 GB for the 595k candidates and OOM'd a 30 GB box;
# the raw form is ~4 extra float32 columns. This mirrors the peak side, which also
# Fourier-expands m/z at batch time.
RAW_MASS_DIM = 6           # [min_far, max_far, mean_far, total_mass, n_bridges/4, has_bridge]
_MASS_COLS = 4             # the first 4 raw columns are masses (Da) to Fourier-expand


def mass_expanded_dim(periods):
    """Atom-feature width AFTER Fourier expansion: chem + 4*2P mass + 2 flag scalars."""
    return ATOM_FDIM + _MASS_COLS * (2 * len(periods)) + 2


def raw_atom_mass(mol):
    """Per-atom raw fragment-mass columns (n_atoms, 6), monoisotopic Da:
      [min_far, max_far, mean_far, total_mass, n_bridges/4 (clipped), has_bridge].
    `min/max/mean far` summarize the masses that split off across the atom's incident
    bridges (0 for ring-only atoms, flagged by has_bridge=0); `total_mass` is the
    precursor (same for every atom) so the model can relate a `far` mass to M-far.
    Kept RAW so the fine mass-defect survives until the (float64) Fourier at batch time.
    Returns None if unparseable."""
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    res = fragment_masses(mol)
    if res is None:
        return None
    per_atom, total = res
    out = np.zeros((mol.GetNumAtoms(), RAW_MASS_DIM), dtype=np.float32)
    out[:, 3] = total
    for i, cuts in enumerate(per_atom):
        if not cuts:
            continue
        fars = np.array([f for f, _ in cuts], dtype=np.float64)
        out[i, 0] = fars.min(); out[i, 1] = fars.max(); out[i, 2] = fars.mean()
        out[i, 4] = min(len(cuts), 4) / 4.0
        out[i, 5] = 1.0
    return out


def atom_retained_fractions(mol, F=6):
    """Per-atom padded (n_atoms, F) float32 of RETAINED fractions: the fraction of the
    molecule's monoisotopic mass in the fragment that CONTAINS this atom when one of its
    incident bridges cleaves (near-side / total), PLUS 1.0 (the intact precursor contains
    every atom). Pad with -1. These are the fragment fractions a peak (mz/precursor) is
    matched against in the peak-side agreement term -- a peak retaining fraction fp should
    align to atoms that CAN sit in a fragment of fraction fp. Ring-only atoms get just
    [1.0]. Sorted ascending, capped at F (keeps the smallest; 1.0 always present if it
    fits). Returns None if unparseable."""
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    res = fragment_masses(mol)
    if res is None:
        return None
    per_atom, total = res
    out = -np.ones((mol.GetNumAtoms(), F), dtype=np.float32)
    for i, cuts in enumerate(per_atom):
        fracs = {1.0}
        if total > 0:
            for far, near in cuts:
                fracs.add(near / total)               # fraction of the a-side fragment
        fs = sorted(fracs)[:F]
        out[i, :len(fs)] = fs
    return out


def features(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not 0 < mol.GetNumAtoms() <= 128:
        raise ValueError(f'Invalid or oversized candidate: {smiles}')
    nodes = np.asarray([_atom_feats(a) for a in mol.GetAtoms()], dtype=np.float32)
    endpoints = np.asarray([(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()], dtype=np.int16).reshape(-1, 2)
    edges = np.asarray([_bond_feats(b) for b in mol.GetBonds()], dtype=np.float32).reshape(-1, BOND_FDIM)
    chemical_nodes, chemical_edges = chemistry_features(mol)
    fragment_nodes, fragment_edges = fragment_features(mol)
    nodes = np.concatenate((nodes, chemical_nodes, fragment_nodes, morgan_atom_ids(mol)), axis=1)
    # Preserve both orientations of cut-side descriptors; the model uses them in
    # directed messages before forming orientation-invariant bond states.
    edges = np.concatenate((np.repeat(edges, 2, axis=0), chemical_edges, fragment_edges), axis=1)
    fractions = atom_retained_fractions(mol)
    if not all(np.isfinite(a).all() for a in (nodes, edges, fractions)):
        raise ValueError(f'Nonfinite candidate features: {smiles}')
    return nodes, endpoints, edges, fractions

