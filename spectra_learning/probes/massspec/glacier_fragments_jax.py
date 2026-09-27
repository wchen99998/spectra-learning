"""JAX dynamic fragment embeddings and intensity-aware gates from GLACIER.

Fragment membership is fixed preprocessing; pooled hidden vectors remain part of
the differentiable GINE computation. Predicted intensity enters
only the fragment gate, never the fragment embedding. Q/K projections are applied
externally after this module's normalized embeddings.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn

from spectra_learning.probes.massspec.atom_flare_jax import Linear, normalize


class FragmentPooler(nn.Module):
    out_dim: int = 256
    hidden: int = 256
    emb: int = 8
    dropout: float = 0.1
    n_cuts_vocab: int = 8
    kind_vocab: int = 4

    @nn.compact
    def __call__(
        self, H, member_mean, boundary_mean, fragment_mask, n_cuts, kind,
        mass_frac, log_n_atoms, intensity, *, train: bool = False,
    ):
        """Return unit fragment vectors and masked gate logits.

        H is (graphs, atoms, hidden), with padded atoms zeroed. Membership and
        boundary mean matrices are (graphs, fragments, atoms), with 1/count for
        selected atoms and zero elsewhere. Other arrays are (graphs, fragments).
        An empty boundary has an all-zero row. Fully empty graph slots are valid.
        """
        H = H.astype(jnp.float32)
        member_mean = jnp.where(fragment_mask[..., None], member_mean, 0.0)
        boundary_mean = jnp.where(fragment_mask[..., None], boundary_mean, 0.0)
        h_mean = jnp.einsum("gfa,gah->gfh", member_mean, H)
        h_boundary = jnp.einsum("gfa,gah->gfh", boundary_mean, H)
        cuts = nn.Embed(
            self.n_cuts_vocab, self.emb,
            embedding_init=nn.initializers.normal(stddev=1.0), name="cuts_emb",
        )(jnp.where(fragment_mask, n_cuts, 0))
        kinds = nn.Embed(
            self.kind_vocab, self.emb,
            embedding_init=nn.initializers.normal(stddev=1.0), name="kind_emb",
        )(jnp.where(fragment_mask, kind, 0))
        mass_frac = jnp.where(fragment_mask, mass_frac, 0.0)
        log_n_atoms = jnp.where(fragment_mask, log_n_atoms, 0.0)
        features = jnp.concatenate(
            [h_mean, h_boundary, cuts, kinds, mass_frac[..., None], log_n_atoms[..., None]],
            axis=-1,
        )
        z = nn.LayerNorm(epsilon=1e-5, use_fast_variance=False, name="norm")(features)
        z = Linear(self.hidden, name="linear1")(z)
        z = jax.nn.gelu(z, approximate=False)
        z = nn.Dropout(self.dropout)(z, deterministic=not train)
        z = normalize(Linear(self.out_dim, name="linear2")(z))
        intensity = jnp.maximum(jnp.where(fragment_mask, intensity, 0.0), 0.0)
        gate_features = jnp.concatenate(
            [features, intensity[..., None], jnp.log1p(intensity)[..., None]], axis=-1,
        )
        gate = Linear(1, name="gate")(gate_features)[..., 0]
        return jnp.where(fragment_mask[..., None], z, 0.0), jnp.where(fragment_mask, gate, -jnp.inf)
