"""Broadcast a learned whole-molecule summary into local GINE atom tokens.

Pooling covers every valid atom in each disjoint graph before any token cap is
applied. The summary extends the information available to each atom beyond its
local message-passing neighborhood. Mean pooling still cannot distinguish all
graphs that share the same multiset of local atom representations.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn

from spectra_learning.probes.massspec.atom_flare_jax import Linear, normalize


class MoleculeGlobalContext(nn.Module):
    """Mean pool, transform, broadcast-add, and optionally normalize atoms."""

    dim: int = 256
    normalize_output: bool = True

    @nn.compact
    def __call__(
        self,
        atoms: jax.Array,
        graph_idx: jax.Array,
        node_mask: jax.Array,
        *,
        num_graphs: int,
    ) -> jax.Array:
        """Contextualize flat atoms; ``num_graphs`` must be static under JIT.

        Valid nodes have graph IDs in ``[0, num_graphs)``. Padded graph IDs and
        atom values are ignored, including when they are outside these ranges.
        No BatchNorm or operation across molecules is introduced here.
        """
        atoms = jnp.where(node_mask[:, None], atoms.astype(jnp.float32), 0.0)
        safe_idx = jnp.where(node_mask, graph_idx, 0)
        sums = jnp.zeros((num_graphs, self.dim), dtype=jnp.float32).at[safe_idx].add(atoms)
        counts = jnp.zeros((num_graphs,), dtype=jnp.float32).at[safe_idx].add(node_mask)
        means = sums / jnp.maximum(counts[:, None], 1.0)
        context = Linear(self.dim, name="linear1")(means)
        context = jax.nn.relu(context)
        context = Linear(self.dim, name="linear2")(context)
        result = atoms + context[safe_idx]
        if self.normalize_output:
            result = normalize(result)
        return jnp.where(node_mask[:, None], result, 0.0)
