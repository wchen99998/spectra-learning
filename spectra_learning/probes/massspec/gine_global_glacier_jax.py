"""Whole-molecule GINE context followed by learned GLACIER fragment pooling."""

import jax.numpy as jnp
from flax import linen as nn

from spectra_learning.probes.massspec.atom_flare_jax import MolGNN
from spectra_learning.probes.massspec.atom_global_context_jax import MoleculeGlobalContext
from spectra_learning.probes.massspec.glacier_fragments_jax import FragmentPooler


class GlobalGlacierGNN(nn.Module):
    hidden: int = 496
    out_dim: int = 256
    dropout: float = 0.1

    @nn.compact
    def __call__(self, graph, *, train: bool = False):
        h = MolGNN(self.hidden, self.out_dim, 3, self.dropout, name="gine")(
            *(graph[name] for name in ("nodes", "edges", "senders", "receivers", "node_mask", "edge_mask")),
            train=train, return_hidden=True,
        )
        h = MoleculeGlobalContext(self.hidden, normalize_output=False, name="global_context")(
            h, graph["graph_idx"], graph["node_mask"], num_graphs=graph["atom_mask"].shape[0],
        )
        # The molecular mean includes all real atoms, before fragment pooling.
        dense = jnp.zeros((*graph["atom_mask"].shape, self.hidden), jnp.float32)
        dense = dense.at[graph["graph_idx"], graph["node_slots"]].add(h)
        tokens, gate = FragmentPooler(self.out_dim, self.hidden, dropout=self.dropout,
                              name="fragments")(
            dense, graph["member_mean"], graph["boundary_mean"], graph["fragment_mask"],
            graph["fragment_n_cuts"], graph["fragment_kind"], graph["fragment_mass_frac"],
            graph["fragment_log_n_atoms"], graph["fragment_intensity"], train=train,
        )
        return tokens, gate
