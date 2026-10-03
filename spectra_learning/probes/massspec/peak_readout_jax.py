"""Contextual readout of frozen per-peak JEPA tokens for retrieval.

The baseline PeakEncoder maps every peak token independently. This readout lets peaks
attend to each other (pre-LN transformer, key-padding mask) before the same unit-norm
projection, so each peak query is formed in the context of the whole spectrum.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn

from spectra_learning.probes.massspec.atom_flare_jax import Linear, normalize


class AttentionPeakReadout(nn.Module):
    out_dim: int = 256
    layers: int = 2
    heads: int = 4
    dropout: float = 0.1

    @nn.compact
    def __call__(self, tokens: jax.Array, mask: jax.Array | None = None, *, train: bool = False) -> jax.Array:
        if mask is None:
            mask = jnp.ones(tokens.shape[:2], bool)
        x = nn.LayerNorm(epsilon=1e-5, use_fast_variance=False, name="norm")(tokens.astype(jnp.float32))
        x = Linear(self.out_dim, name="linear1")(x)
        attention_mask = nn.make_attention_mask(mask, mask)
        for layer in range(self.layers):
            h = nn.LayerNorm(epsilon=1e-5, name=f"attn_norm_{layer}")(x)
            h = nn.MultiHeadDotProductAttention(num_heads=self.heads, dropout_rate=self.dropout,
                                                deterministic=not train, name=f"attn_{layer}")(h, h, mask=attention_mask)
            x = x + h
            h = nn.LayerNorm(epsilon=1e-5, name=f"mlp_norm_{layer}")(x)
            h = nn.Dense(4 * self.out_dim, name=f"mlp_{layer}_0")(h)
            h = nn.Dropout(self.dropout)(jax.nn.gelu(h, approximate=False), deterministic=not train)
            x = x + nn.Dense(self.out_dim, name=f"mlp_{layer}_1")(h)
        x = jax.nn.gelu(x, approximate=False)
        x = nn.Dropout(self.dropout)(x, deterministic=not train)
        return normalize(Linear(self.out_dim, name="linear2")(x))
