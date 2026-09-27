"""Chemistry and graph-layout checks for the atom FLARE JAX input path."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from atom_flare_data import GraphStore, atom_retained_fractions, batch_graphs, build_graph_store, featurize


def test_chem49_bond8_and_original_rdkit_order():
    nodes, endpoints, edges, fractions = featurize("CCO")
    assert nodes.shape == (3, 49)
    assert edges.shape == (4, 8)
    np.testing.assert_array_equal(endpoints, [[0, 1, 1, 2], [1, 0, 2, 1]])
    np.testing.assert_array_equal(edges, np.tile([1, 0, 0, 0, 0, 0, 0, 0], (4, 1)))
    # C methyl: element C, total degree 4, neutral, 3H, SP3, unspecified chirality.
    expected = np.zeros(49, np.float32)
    expected[[0, 19, 25, 32, 38, 42]] = 1
    expected[48] = 0.12011
    np.testing.assert_array_equal(nodes[0], expected)
    assert nodes[2, 2] == 1  # Input atom order preserves terminal oxygen.
    assert fractions.shape == (3, 6)


def test_incident_bridge_fractions_do_not_propagate_to_other_atoms():
    mol = Chem.MolFromSmiles("CCO")
    actual = atom_retained_fractions(mol)
    h = 1.00782503207
    methyl, methylene, hydroxyl = 12 + 3 * h, 12 + 2 * h, 15.99491462 + h
    total = methyl + methylene + hydroxyl
    expected = np.full((3, 6), -1, np.float32)
    expected[0, :2] = [methyl / total, 1]
    expected[1, :3] = sorted([(methylene + hydroxyl) / total, (methyl + methylene) / total, 1])
    expected[2, :2] = [hydroxyl / total, 1]
    np.testing.assert_allclose(actual, expected, rtol=1e-7)
    ring = atom_retained_fractions(Chem.MolFromSmiles("c1ccccc1"))
    np.testing.assert_array_equal(ring[:, 0], 1)
    np.testing.assert_array_equal(ring[:, 1:], -1)


def test_disconnected_mass_denominator_and_symmetric_deduplication():
    table = Chem.GetPeriodicTable()
    fractions = atom_retained_fractions(Chem.MolFromSmiles("CC.[Na+]"))
    carbon_h3 = 12 + 3 * 1.00782503207
    expected = carbon_h3 / (2 * carbon_h3 + table.GetMostCommonIsotopeMass(11))
    np.testing.assert_allclose(fractions[:2, 0], expected, rtol=1e-7)
    np.testing.assert_array_equal(fractions[2], [1, -1, -1, -1, -1, -1])
    symmetric = atom_retained_fractions(Chem.MolFromSmiles("CCC"))
    assert (symmetric[1] >= 0).sum() == 2


def test_single_atom_stereo_and_unknown_element():
    atom, index, edges, fractions = featurize("[Na+]")
    assert index.shape == (2, 0)
    assert edges.shape == (0, 8)
    assert atom[0, 14] == 1  # Out-of-vocabulary element channel.
    assert atom[0, 26] == 1  # +1 formal charge.
    np.testing.assert_array_equal(fractions, [[1, -1, -1, -1, -1, -1]])
    _, _, stereo_edges, _ = featurize("F/C=C/F")
    np.testing.assert_array_equal(stereo_edges[2:4, 7], 1)
    assert featurize("not_a_molecule") is None
    assert featurize("") is None


def test_graph_store_roundtrip_duplicates_padding_and_disjoint_edges(tmp_path):
    directory = tmp_path / "graphs"
    store = build_graph_store(["CCO", "[Na+]", "CCO", "not_a_molecule"], directory)
    assert len(store) == 2
    assert isinstance(store.nodes, np.memmap)
    assert store.metadata["invalid_molecules"] == 1
    assert store.metadata["max_nodes"] == 3
    assert store.metadata["max_edges"] == 4
    assert json.loads((directory / "invalid_smiles.json").read_text()) == ["not_a_molecule"]
    for smiles in store.smiles:
        for saved, expected in zip(store[store.row_of[smiles]], featurize(smiles)):
            np.testing.assert_array_equal(saved, expected)
    reopened = GraphStore(directory)
    ids = reopened.ids(["CCO", "[Na+]", "CCO"])
    packed = batch_graphs(reopened, ids, graph_capacity=4, node_capacity=9, edge_capacity=10, atom_cap=4)
    np.testing.assert_array_equal(packed["n_node"], [3, 1, 3, 0])
    np.testing.assert_array_equal(packed["graph_idx"][:7], [0, 0, 0, 1, 2, 2, 2])
    np.testing.assert_array_equal(packed["node_slots"][:7], [0, 1, 2, 0, 0, 1, 2])
    np.testing.assert_array_equal(packed["senders"][:8], [0, 1, 1, 2, 4, 5, 5, 6])
    np.testing.assert_array_equal(packed["receivers"][:8], [1, 0, 2, 1, 5, 4, 6, 5])
    np.testing.assert_array_equal(packed["nodes"][:3], packed["nodes"][4:7])
    assert not packed["node_mask"][7:].any()
    assert not packed["edge_mask"][8:].any()
    assert not packed["atom_mask"][3].any()
    np.testing.assert_array_equal(packed["afrac"][3], -1)
    assert len(build_graph_store(["not_a_molecule", "CCO", "[Na+]"], directory)) == 2
    with pytest.raises(ValueError, match="molecule set differs"):
        build_graph_store(["CCO"], directory)


def test_capacities_raise_instead_of_silently_truncating(tmp_path):
    with pytest.raises(ValueError, match="exceeding max_atoms"):
        build_graph_store(["CCCC"], tmp_path / "too_small", max_atoms=3)
    store = build_graph_store(["CCO"], tmp_path / "graphs")
    capacities = dict(graph_capacity=1, node_capacity=3, edge_capacity=4, atom_cap=3)
    for key in capacities:
        smaller = capacities | {key: capacities[key] - 1}
        with pytest.raises(ValueError, match="exceeds"):
            batch_graphs(store, [0], **smaller)
    with pytest.raises(ValueError, match="outside"):
        batch_graphs(store, [-1], **capacities)
    with pytest.raises(ValueError, match="No valid"):
        build_graph_store([""], tmp_path / "empty")


def test_parallel_preparation_and_zero_edge_store(tmp_path):
    store = build_graph_store(["[Na+]", "[Cl-]"], tmp_path / "ions", workers=2, chunk_size=1)
    assert len(store) == 2
    assert store.edges.shape == (0, 8)
    assert store.senders.shape == (0,)
    batch = batch_graphs(store, [0, 1], graph_capacity=2, node_capacity=2, edge_capacity=1, atom_cap=1)
    assert batch["node_mask"].all()
    assert not batch["edge_mask"].any()
