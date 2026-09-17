"""Permutation-equivariant edge-to-node statistics for frozen pair FLARE.

Keep diagonal states and the two directed off-diagonal means separate. For
any linear channel projection W, mean_j(Z_ij) W = mean_j(Z_ij W), so these
caches exactly support learned linear incoming/outgoing message aggregation.
They do not preserve arbitrary nonlinear functions of individual edges.
"""

import jax
import jax.numpy as jnp


def pair_peak_statistics(
    pair: jax.Array, token_mask: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return [diagonal, outgoing, incoming] per peak; repeat CLS to that width.

    The last position is CLS. Padding, CLS edges, and self edges are excluded
    from both means. A peak with no physical neighbor gets zero messages.
    Reductions use float32 even when the frozen encoder computes in bfloat16.
    """
    pair = pair.astype(jnp.float32)
    valid = token_mask[:, :-1]
    count = valid.shape[1]
    physical = pair[:, :-1, :-1]
    edge_mask = (
        valid[:, :, None] & valid[:, None, :]
        & ~jnp.eye(count, dtype=jnp.bool_)[None]
    )
    edges = jnp.where(edge_mask[..., None], physical, 0.0)
    denominator = jnp.maximum(valid.sum(-1) - 1, 1)[:, None, None]
    outgoing = edges.sum(axis=2) / denominator
    incoming = edges.sum(axis=1) / denominator
    diagonal = physical[:, jnp.arange(count), jnp.arange(count)]
    diagonal = jnp.where(valid[..., None], diagonal, 0.0)
    peaks = jnp.concatenate([diagonal, outgoing, incoming], axis=-1)
    cls = jnp.tile(pair[:, -1, -1], (1, 3))
    return peaks, cls


def restore_peak_order(
    compact: jax.Array, indices: jax.Array, valid: jax.Array, peak_slots: int,
) -> jax.Array:
    """Scatter compact physical tokens into their canonical m/z-ordered slots.

    Padded compact indices can repeat valid indices; scatter-add with zeroed
    padding prevents a padded entry from overwriting a real peak.
    """
    output = jnp.zeros((compact.shape[0], peak_slots, compact.shape[-1]), compact.dtype)
    values = jnp.where(valid[..., None], compact, 0.0)
    return output.at[jnp.arange(len(compact))[:, None], indices].add(values)
