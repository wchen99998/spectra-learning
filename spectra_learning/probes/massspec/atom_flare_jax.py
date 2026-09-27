"""Shared JAX peak projection, residual GINE and symmetric retrieval loss.

Graph masks exclude capacity padding from message passing and BatchNorm.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import linen as nn

NEG = -1e4


def normalize(x: jax.Array) -> jax.Array:
    """PyTorch F.normalize's epsilon, including finite gradients at zero."""
    return x / jnp.sqrt(jnp.maximum(jnp.sum(x * x, axis=-1, keepdims=True), 1e-24))


class Linear(nn.Module):
    """PyTorch/PyG default: uniform weights and biases within ±1/sqrt(fan_in).

    GINEConv's extra PyG Linear also defaults to kaiming_uniform(a=sqrt(5)),
    which has these same bounds; it does not use Glorot initialization.
    """

    features: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        bound = 1.0 / math.sqrt(x.shape[-1])

        def init(key, shape, dtype=jnp.float32):
            return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)

        kernel = self.param("kernel", init, (x.shape[-1], self.features))
        bias = self.param("bias", init, (self.features,))
        return x.astype(jnp.float32) @ kernel + bias


class PeakEncoder(nn.Module):
    out_dim: int = 256
    dropout: float = 0.1

    @nn.compact
    def __call__(self, tokens: jax.Array, *, train: bool = False) -> jax.Array:
        x = nn.LayerNorm(epsilon=1e-5, use_fast_variance=False, name="norm")(
            tokens.astype(jnp.float32)
        )
        x = Linear(self.out_dim, name="linear1")(x)
        x = jax.nn.gelu(x, approximate=False)
        x = nn.Dropout(self.dropout)(x, deterministic=not train)
        return normalize(Linear(self.out_dim, name="linear2")(x))


class QueryKeyProjection(nn.Module):
    """Asymmetric bilinear compatibility, identity-initialized without normalization."""

    dim: int = 256

    def setup(self):
        self.query = self.param("query", lambda key: jnp.eye(self.dim, dtype=jnp.float32))
        self.key = self.param("key", lambda key: jnp.eye(self.dim, dtype=jnp.float32))

    def project_queries(self, peaks):
        queries = peaks.astype(jnp.float32) @ self.query
        return queries

    def project_keys(self, atoms):
        keys = atoms.astype(jnp.float32) @ self.key
        return keys

    def __call__(self, peaks, atoms):
        return self.project_queries(peaks), self.project_keys(atoms)


class MaskedBatchNorm(nn.Module):
    """BatchNorm1d with PyTorch's variance and momentum conventions.

    Forward normalization uses population variance, while the running variance
    receives its unbiased estimate. Padding must not count toward either. As in
    the upstream training batches, training requires at least two real nodes.
    """

    momentum: float = 0.1
    epsilon: float = 1e-5

    @nn.compact
    def __call__(self, x: jax.Array, mask: jax.Array, *, train: bool) -> jax.Array:
        width = x.shape[-1]
        scale = self.param("scale", nn.initializers.ones, (width,))
        bias = self.param("bias", nn.initializers.zeros, (width,))
        running_mean = self.variable("batch_stats", "mean", jnp.zeros, (width,))
        running_var = self.variable("batch_stats", "var", jnp.ones, (width,))
        if train:
            count = mask.sum().astype(jnp.float32)
            mean = jnp.where(mask[:, None], x, 0.0).sum(axis=0) / count
            var = jnp.where(mask[:, None], (x - mean) ** 2, 0.0).sum(axis=0) / count
            if not self.is_initializing():
                running_mean.value = jax.lax.stop_gradient(
                    (1 - self.momentum) * running_mean.value + self.momentum * mean
                )
                running_var.value = jax.lax.stop_gradient(
                    (1 - self.momentum) * running_var.value
                    + self.momentum * var * count / (count - 1)
                )
        else:
            mean, var = running_mean.value, running_var.value
        result = (x - mean) * jax.lax.rsqrt(var + self.epsilon) * scale + bias
        return jnp.where(mask[:, None], result, 0.0)


class MolGNN(nn.Module):
    hidden: int = 256
    out_dim: int = 256
    n_layers: int = 3
    dropout: float = 0.1

    @nn.compact
    def __call__(
        self,
        nodes: jax.Array,
        edges: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
        node_mask: jax.Array,
        edge_mask: jax.Array,
        *,
        train: bool = False,
        return_hidden: bool = False,
    ) -> jax.Array:
        h = Linear(self.hidden, name="node_enc")(nodes)
        e = Linear(self.hidden, name="edge_enc")(edges)
        for layer in range(self.n_layers):
            # GINEConv(edge_dim=hidden) projects the already encoded edges again.
            edge_h = Linear(self.hidden, name=f"edge_{layer}")(e)
            messages = jax.nn.relu(h[senders] + edge_h)
            messages = jnp.where(edge_mask[:, None], messages, 0.0)
            aggregated = jnp.zeros_like(h).at[receivers].add(messages)
            # Upstream GINE epsilon is fixed at zero, not a trainable parameter.
            update = Linear(self.hidden, name=f"mlp_{layer}_0")(h + aggregated)
            update = jax.nn.relu(update)
            update = Linear(self.hidden, name=f"mlp_{layer}_2")(update)
            update = MaskedBatchNorm(name=f"norm_{layer}")(update, node_mask, train=train)
            update = jax.nn.relu(update)
            update = nn.Dropout(self.dropout, name=f"dropout_{layer}")(
                update, deterministic=not train
            )
            h = h + update
        if return_hidden:
            return jnp.where(node_mask[:, None], h, 0.0)
        result = normalize(Linear(self.out_dim, name="out")(h))
        return jnp.where(node_mask[:, None], result, 0.0)










def info_nce(score, ik_codes, tau=0.05, hn=None, hn_valid=None):
    """Symmetric InfoNCE; formula negatives extend the forward denominator."""
    diagonal = jnp.arange(score.shape[0])
    same = ik_codes[:, None] == ik_codes[None, :]
    off_diagonal = same & ~jnp.eye(score.shape[0], dtype=jnp.bool_)
    logits = jnp.where(off_diagonal, -jnp.inf, score / tau)
    row_logits = logits
    if hn is not None:
        row_logits = jnp.concatenate([logits, jnp.where(hn_valid, hn / tau, -jnp.inf)], axis=1)
    positive = logits[diagonal, diagonal]
    s2m = jax.scipy.special.logsumexp(row_logits, axis=1) - positive
    m2s = jax.scipy.special.logsumexp(logits, axis=0) - positive
    return 0.5 * (s2m.mean() + m2s.mean())
