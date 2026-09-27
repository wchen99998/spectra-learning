"""Molecular context reaches distant atoms without mixing disjoint graphs."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from spectra_learning.probes.massspec.atom_flare_jax import MolGNN, normalize
from spectra_learning.probes.massspec.atom_global_context_jax import MoleculeGlobalContext


def test_context_transmits_a_perturbation_beyond_three_gine_hops():
    # An eight-edge chain leaves atom 8 outside atom 0's three-hop GINE field.
    nodes = jax.random.normal(jax.random.key(1), (9, 49))
    senders = jnp.concatenate((jnp.arange(8), jnp.arange(1, 9)))
    receivers = jnp.concatenate((jnp.arange(1, 9), jnp.arange(8)))
    edges = jax.random.normal(jax.random.key(2), (16, 8))
    node_mask = jnp.ones(9, dtype=bool)
    edge_mask = jnp.ones(16, dtype=bool)
    gine = MolGNN(hidden=12, out_dim=8, dropout=0)
    args = (edges, senders, receivers, node_mask, edge_mask)
    gine_vars = gine.init(jax.random.key(3), nodes, *args)
    original = gine.apply(gine_vars, nodes, *args)
    perturbed = gine.apply(gine_vars, nodes.at[8].add(5.0), *args)
    np.testing.assert_array_equal(original[0], perturbed[0])
    assert np.linalg.norm(np.asarray(original[8] - perturbed[8])) > 1e-3

    model = MoleculeGlobalContext(dim=8)
    graph_idx = jnp.zeros(9, dtype=jnp.int32)
    variables = model.init(jax.random.key(4), original, graph_idx, node_mask, num_graphs=1)
    before = model.apply(variables, original, graph_idx, node_mask, num_graphs=1)
    after = model.apply(variables, perturbed, graph_idx, node_mask, num_graphs=1)
    assert np.linalg.norm(np.asarray(before[0] - after[0])) > 1e-4
    np.testing.assert_allclose(jnp.linalg.norm(before, axis=-1), 1.0, atol=1e-6)


def test_padding_permutation_empty_slots_and_graph_isolation():
    atoms = normalize(jax.random.normal(jax.random.key(5), (7, 8)))
    graph_idx = jnp.array([0, 0, 0, 1, 1, 1, 1])
    mask = jnp.ones(7, dtype=bool)
    model = MoleculeGlobalContext(dim=8)
    variables = model.init(jax.random.key(6), atoms, graph_idx, mask, num_graphs=3)
    apply = jax.jit(lambda x, idx, valid: model.apply(
        variables, x, idx, valid, num_graphs=3,
    ))
    original = apply(atoms, graph_idx, mask)

    # The third graph is empty; padded IDs may be invalid and values non-finite.
    padded = jnp.pad(atoms, ((0, 3), (0, 0)), constant_values=jnp.nan)
    padded_idx = jnp.concatenate((graph_idx, jnp.array([-1, 300, 2])))
    padded_mask = jnp.arange(10) < 7
    with_padding = apply(padded, padded_idx, padded_mask)
    np.testing.assert_allclose(with_padding[:7], original, atol=1e-6)
    np.testing.assert_array_equal(with_padding[7:], 0.0)

    order = jnp.array([5, 2, 6, 0, 3, 1, 4])
    permuted = apply(atoms[order], graph_idx[order], mask[order])
    np.testing.assert_allclose(permuted, original[order], atol=1e-6)

    changed = apply(atoms.at[3:].set(-atoms[3:]), graph_idx, mask)
    np.testing.assert_array_equal(changed[:3], original[:3])
    assert np.linalg.norm(np.asarray(changed[3:] - original[3:])) > 1e-3


def test_gradients_reach_remote_atoms_and_parameters_but_not_other_graphs():
    atoms = normalize(jax.random.normal(jax.random.key(7), (9, 8)))
    graph_idx = jnp.array([0, 0, 0, 0, 0, 1, 1, 1, 500])
    mask = jnp.arange(9) < 8
    model = MoleculeGlobalContext(dim=8)
    variables = model.init(jax.random.key(8), atoms, graph_idx, mask, num_graphs=3)
    weights = jax.random.normal(jax.random.key(9), (8,))

    def loss(params, inputs):
        output = model.apply({"params": params}, inputs, graph_idx, mask, num_graphs=3)
        return output[0] @ weights

    parameter_grad, atom_grad = jax.grad(loss, argnums=(0, 1))(variables["params"], atoms)
    for gradient in jax.tree.leaves((parameter_grad, atom_grad)):
        assert np.isfinite(np.asarray(gradient)).all()
    for gradient in jax.tree.leaves(parameter_grad):
        assert np.linalg.norm(np.asarray(gradient)) > 1e-6
    assert np.linalg.norm(np.asarray(atom_grad[4])) > 1e-6
    np.testing.assert_array_equal(atom_grad[5:], 0.0)


def test_production_width_adds_131584_trainable_parameters():
    model = MoleculeGlobalContext()
    variables = model.init(
        jax.random.key(10), jnp.zeros((2, 256)), jnp.zeros(2, dtype=jnp.int32),
        jnp.ones(2, dtype=bool), num_graphs=1,
    )
    assert sum(leaf.size for leaf in jax.tree.leaves(variables["params"])) == 131_584


def test_hidden_state_residual_can_preserve_magnitude_without_normalization():
    atoms = jax.random.normal(jax.random.key(11), (5, 12)) * 3.0
    graph_idx = jnp.array([0, 0, 0, 1, 1])
    mask = jnp.array([True, True, True, True, False])
    raw = MoleculeGlobalContext(dim=12, normalize_output=False)
    unit = MoleculeGlobalContext(dim=12)
    variables = raw.init(jax.random.key(12), atoms, graph_idx, mask, num_graphs=2)
    contextual = raw.apply(variables, atoms, graph_idx, mask, num_graphs=2)
    normalized = unit.apply(variables, atoms, graph_idx, mask, num_graphs=2)
    np.testing.assert_allclose(normalize(contextual), normalized, atol=1e-6)
    np.testing.assert_allclose(contextual[0] - atoms[0], contextual[2] - atoms[2], atol=1e-6)
    assert np.all(np.asarray(jnp.linalg.norm(contextual[:4], axis=-1)) > 2.0)
    np.testing.assert_array_equal(contextual[4], 0.0)
