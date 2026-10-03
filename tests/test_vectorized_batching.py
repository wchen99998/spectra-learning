"""Vectorized graph/fragment packing must equal the original per-graph loops bit for bit."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from atom_flare_data import ATOM_FDIM, BOND_FDIM, FRACTION_SLOTS, batch_graphs, build_graph_store
from glacier_fragment_data import batch_fragments, build_fragment_store
from test_glacier_fragment_data import _proposals

SMILES = ["CCCC(C)O", "CCO", "c1ccccc1", "[Na+]", "CC(=O)Nc1ccc(O)cc1", "OC(=O)CCc1ccccc1",
          "CCN(CC)CC", "C1CCOC1", "CC(C)Cc1ccc(C(C)C(=O)O)cc1", "O=C(O)c1ccccc1O"] * 3


def reference_graphs(store, ids, *, graph_capacity, node_capacity, edge_capacity, atom_cap):
    result = {
        "nodes": np.zeros((node_capacity, ATOM_FDIM), np.float32),
        "edges": np.zeros((edge_capacity, BOND_FDIM), np.float32),
        "senders": np.zeros(edge_capacity, np.int32), "receivers": np.zeros(edge_capacity, np.int32),
        "node_mask": np.zeros(node_capacity, bool), "edge_mask": np.zeros(edge_capacity, bool),
        "graph_idx": np.zeros(node_capacity, np.int32), "node_slots": np.zeros(node_capacity, np.int32),
        "atom_mask": np.zeros((graph_capacity, atom_cap), bool),
        "afrac": np.full((graph_capacity, atom_cap, FRACTION_SLOTS), -1, np.float32),
        "n_node": np.zeros(graph_capacity, np.int32),
    }
    npos = epos = 0
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


def reference_fragments(store, ids, *, graph_capacity, atom_cap):
    shape = (graph_capacity, store.metadata["f_max"])
    result = {
        "member_mean": np.zeros((*shape, atom_cap), np.float32),
        "boundary_mean": np.zeros((*shape, atom_cap), np.float32),
        "fragment_mask": np.zeros(shape, bool), "fragment_n_cuts": np.zeros(shape, np.int32),
        "fragment_kind": np.zeros(shape, np.int32), "fragment_mass_frac": np.zeros(shape, np.float32),
        "fragment_log_n_atoms": np.zeros(shape, np.float32), "fragment_intensity": np.zeros(shape, np.float32),
    }
    for slot, graph_id in enumerate(ids):
        lo, hi = store.frag_off[graph_id:graph_id + 2]
        count = int(hi - lo)
        result["fragment_mask"][slot, :count] = True
        result["fragment_n_cuts"][slot, :count] = store.n_cuts[lo:hi]
        result["fragment_kind"][slot, :count] = store.kind[lo:hi]
        result["fragment_mass_frac"][slot, :count] = store.mass_frac[lo:hi]
        result["fragment_log_n_atoms"][slot, :count] = np.log(store.n_atoms[lo:hi].astype(np.float32))
        result["fragment_intensity"][slot, :count] = store.inten[lo:hi]
        for name, offs, atoms, floor in (("member_mean", store.mem_off, store.atoms, 0),
                                         ("boundary_mean", store.bnd_off, store.bnd_atoms, 1)):
            offsets = offs[lo:hi + 1]
            lengths = np.diff(offsets)
            rows = np.repeat(np.arange(count), lengths)
            members = atoms[offsets[0]:offsets[-1]]
            result[name][slot, rows, members] = np.repeat(1.0 / np.maximum(lengths, floor or 1e-300), lengths)
    return result


def test_vectorized_packing_matches_reference_loops(tmp_path):
    graphs = build_graph_store(SMILES, tmp_path / "graphs")
    ids_all = graphs.ids(sorted(set(SMILES)))
    rng = np.random.default_rng(0)
    rows = []
    for graph_id in ids_all:
        n = int(graphs.counts[graph_id])
        sets = [sorted(rng.choice(n, rng.integers(1, n + 1), replace=False).tolist())
                for _ in range(int(rng.integers(0, 6)))] if n >= 3 else []
        rows.append((graph_id, sets, rng.uniform(0.05, 1, len(sets)).tolist()))
    proposals = _proposals(tmp_path / "proposals", graphs, [rows[:4], rows[4:]])
    store = build_fragment_store(graphs, proposals, tmp_path / "fragments", f_max=6, min_atoms=1, min_frac=0.0)
    for trial in range(20):
        ids = rng.choice(ids_all, int(rng.integers(1, 12)), replace=True).tolist()
        kw = dict(graph_capacity=12, node_capacity=400, edge_capacity=800, atom_cap=16)
        new, old = batch_graphs(graphs, ids, **kw), reference_graphs(graphs, ids, **kw)
        assert new.keys() == old.keys()
        for key in old:
            np.testing.assert_array_equal(new[key], old[key], err_msg=key)
            assert new[key].dtype == old[key].dtype, key
        new = batch_fragments(store, ids, graph_capacity=12, atom_cap=16)
        old = reference_fragments(store, ids, graph_capacity=12, atom_cap=16)
        assert new.keys() == old.keys()
        for key in old:
            np.testing.assert_array_equal(new[key], old[key], err_msg=key)
            assert new[key].dtype == old[key].dtype, key
