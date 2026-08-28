"""ZeRO-1 optimizer-state sharding for the Adam update.

Adam is elementwise: element ``i`` of the update depends only on element ``i``
of the gradient, of the two moments and of the parameter. So unlike Muon's
Newton-Schulz, which needs a whole matrix at once, the flattened parameter
vector can simply be cut into one slice per data-parallel rank. Each rank then
holds ``1/N`` of the moments and does ``1/N`` of the update arithmetic, and the
result is bit-comparable to the replicated update.

The gradient arrives by ``psum_scatter`` rather than ``psum``. A ring
all-reduce is internally a reduce-scatter followed by an all-gather, each phase
moving ``B(P-1)/P`` bytes per rank, so the wire traffic is unchanged; what
changes is that the second phase now carries the *update*, after the moments
have been applied, instead of the raw gradient.

Slicing is per leaf rather than over one concatenated buffer. Concatenating
would materialise a second full-size copy of the gradient tree - 4 GB at 1B
parameters in fp32 - which is the opposite of what sharding the state is for.
Each leaf is instead zero-padded up to a multiple of the shard count, which
costs at most ``N-1`` elements per leaf.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.sharding import PartitionSpec as P


class Zero1AdamState(NamedTuple):
    """Adam state where ``mu``/``nu`` hold only this rank's slice of each leaf."""

    count: Array
    mu: Any
    nu: Any


def zero1_state_specs(shard_axis: str) -> Zero1AdamState:
    """`shard_map` in/out specs for the state: count replicated, moments split."""
    return Zero1AdamState(count=P(), mu=P(shard_axis), nu=P(shard_axis))


ZERO1_ALIGNMENT = 4096


def padded_size(size: int, shards: int, *, alignment: int = ZERO1_ALIGNMENT) -> int:
    """``size`` rounded up to a whole number of shards.

    Padding to ``lcm(shards, alignment)`` rather than to ``shards`` keeps the
    stored shape identical for every power-of-two shard count up to the
    alignment, so a checkpoint written on 16 chips still restores on 64. The
    waste is bounded by one alignment block per leaf - about 13 MB across a
    1B-parameter model, against the 8 GB of moments being sharded.
    """
    block = math.lcm(shards, alignment)
    return -(-size // block) * block


def local_slice_size(size: int, shards: int) -> int:
    return padded_size(size, shards) // shards


def zero1_adam_init(
    params: Any,
    *,
    shards: int,
    moment_dtype: Any = jnp.float32,
) -> Zero1AdamState:
    """The *global* moment arrays, one flat padded vector per leaf.

    Passed into `shard_map` under `zero1_state_specs`, each rank receives
    only its own contiguous slice, which is the whole point: the moments
    cost ``1/shards`` of HBM per chip instead of a full replica.
    """

    def local_zeros(leaf: Array) -> Array:
        return jnp.zeros((padded_size(int(leaf.size), shards),), moment_dtype)

    return Zero1AdamState(
        count=jnp.zeros([], jnp.int32),
        mu=jax.tree.map(local_zeros, params),
        nu=jax.tree.map(local_zeros, params),
    )


def _flatten_padded(leaf: Array, shards: int) -> Array:
    flat = leaf.reshape(-1)
    pad = padded_size(int(leaf.size), shards) - int(leaf.size)
    if pad:
        flat = jnp.pad(flat, (0, pad))
    return flat


def _local_of_replicated(leaf: Array, *, shards: int, rank: Array) -> Array:
    """This rank's slice of a leaf that every rank already holds in full."""
    flat = _flatten_padded(leaf, shards)
    return jax.lax.dynamic_index_in_dim(
        flat.reshape(shards, -1), rank, axis=0, keepdims=False
    )


def zero1_adam_update(
    grads: Any,
    state: Zero1AdamState,
    params: Any,
    *,
    shard_axis: str,
    shards: int,
    learning_rate: Any,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    weight_decay: float = 0.0,
    decay_mask: Any = None,
    clip_norm: float = 0.0,
    scatter_dtype: Any = jnp.float32,
    gather_dtype: Any = jnp.float32,
) -> tuple[Any, Zero1AdamState]:
    """The replicated ``chain(clip_by_global_norm, adamw)`` update, sharded.

    ``grads`` are the per-rank partial gradients, *not* yet summed across the
    data axis - this replaces the ``psum`` the caller would otherwise do.
    Returns updates in the same tree structure as ``params``, already summed
    and gathered, so the caller can apply them to the replicated parameters.
    """
    rank = jax.lax.axis_index(shard_axis)

    def scatter(leaf: Array) -> Array:
        flat = _flatten_padded(leaf, shards)
        summed = jax.lax.psum_scatter(
            flat.astype(scatter_dtype), shard_axis, tiled=True
        )
        return summed.astype(jnp.float32)

    local_grads = jax.tree.map(scatter, grads)

    if clip_norm > 0.0:
        # Every element lives on exactly one rank and the padding is zero, so
        # summing the local sums of squares gives the true global norm.
        local_sq = sum(
            jnp.sum(jnp.square(leaf)) for leaf in jax.tree.leaves(local_grads)
        )
        global_norm = jnp.sqrt(jax.lax.psum(local_sq, shard_axis))
        scale = clip_norm / jnp.maximum(clip_norm, global_norm)
        local_grads = jax.tree.map(lambda leaf: leaf * scale, local_grads)

    count = state.count + 1
    mu = jax.tree.map(
        lambda m, g: ((1 - b1) * g + b1 * m.astype(jnp.float32)),
        state.mu,
        local_grads,
    )
    nu = jax.tree.map(
        lambda v, g: ((1 - b2) * jnp.square(g) + b2 * v.astype(jnp.float32)),
        state.nu,
        local_grads,
    )
    mu_hat = jax.tree.map(lambda m: m / (1 - b1**count), mu)
    nu_hat = jax.tree.map(lambda v: v / (1 - b2**count), nu)
    local_updates = jax.tree.map(
        lambda m, v: m / (jnp.sqrt(v + eps_root) + eps),
        mu_hat,
        nu_hat,
    )

    if weight_decay != 0.0:
        mask = (
            decay_mask
            if decay_mask is not None
            else jax.tree.map(lambda _leaf: True, params)
        )

        def decay(update: Array, param: Array, decayed: bool) -> Array:
            if not decayed:
                return update
            return update + weight_decay * _local_of_replicated(
                param, shards=shards, rank=rank
            )

        local_updates = jax.tree.map(decay, local_updates, params, mask)

    step_size = -(
        learning_rate(state.count) if callable(learning_rate) else learning_rate
    )
    local_updates = jax.tree.map(lambda u: step_size * u, local_updates)

    def gather(local_update: Array, param: Array) -> Array:
        full = jax.lax.all_gather(
            local_update.astype(gather_dtype), shard_axis, tiled=True
        )
        return full[: param.size].astype(param.dtype).reshape(param.shape)

    updates = jax.tree.map(gather, local_updates, params)
    moment_dtype = jax.tree.leaves(state.mu)[0].dtype
    return updates, Zero1AdamState(
        count=count,
        mu=jax.tree.map(lambda m: m.astype(moment_dtype), mu),
        nu=jax.tree.map(lambda v: v.astype(moment_dtype), nu),
    )


def zero1_adam_config(config: Any) -> dict[str, Any]:
    """The knobs `zero1_adam_update` needs, read off a training config."""
    return {
        "b1": float(config.get("b1", 0.9)),
        "b2": float(config.get("b2", 0.999)),
        "weight_decay": float(config.get("weight_decay", 0.0)),
        "clip_norm": float(config.get("grad_clip_norm", 0.0)),
        "scatter_dtype": jnp.dtype(
            config.get("jax_grad_all_reduce_dtype", "float32")
        ),
        "gather_dtype": jnp.dtype(
            config.get("jax_zero1_gather_dtype", "float32")
        ),
    }


def zero1_moment_dtype(config: Any) -> Any:
    state_dtype = str(config.get("optimizer_state_dtype", "fp32")).lower()
    if state_dtype in {"bf16", "bfloat16"}:
        return jnp.bfloat16
    return jnp.float32


__all__ = [
    "Zero1AdamState",
    "zero1_adam_config",
    "zero1_adam_init",
    "zero1_adam_init_sharded",
    "zero1_flat_init",
    "zero1_flat_update",
    "zero1_adam_update",
    "zero1_moment_dtype",
    "zero1_state_specs",
    "local_slice_size",
    "padded_size",
]


def zero1_adam_init_sharded(
    params: Any,
    *,
    shards: int,
    mesh: Any,
    shard_axis: str,
    moment_dtype: Any = jnp.float32,
) -> Zero1AdamState:
    """`zero1_adam_init` as a globally sharded array, without a host round-trip.

    The moments are zeros, so they are built directly on device under the
    target sharding rather than created on the host and transferred. That keeps
    a 1B-parameter state off the host entirely and works unchanged across
    processes, where each host only ever owns its own slice.
    """
    treedef = jax.tree.structure(params)
    sizes = [int(leaf.size) for leaf in jax.tree.leaves(params)]

    def build() -> Zero1AdamState:
        moments = [
            jnp.zeros((padded_size(size, shards),), moment_dtype) for size in sizes
        ]
        return Zero1AdamState(
            count=jnp.zeros([], jnp.int32),
            mu=jax.tree.unflatten(treedef, moments),
            nu=jax.tree.unflatten(treedef, list(moments)),
        )

    replicated = jax.sharding.NamedSharding(mesh, P())
    split = jax.sharding.NamedSharding(mesh, P(shard_axis))
    out_shardings = Zero1AdamState(
        count=replicated,
        mu=jax.tree.unflatten(treedef, [split] * len(sizes)),
        nu=jax.tree.unflatten(treedef, [split] * len(sizes)),
    )
    return jax.jit(build, out_shardings=out_shardings)()


# ---------------------------------------------------------------------------
# Flat variant: one concatenated buffer instead of one collective per leaf.
#
# The per-leaf form issues two collectives for each of the ~400 parameter
# tensors. A replicated `psum` over the same tree is batched by XLA into a
# handful of large transfers, so per-leaf sharding trades 32 ms of update
# arithmetic for far more than that in collective latency. Concatenating first
# costs one full-size temporary but reduces the traffic to exactly two
# collectives.
# ---------------------------------------------------------------------------


def _flat_layout(params: Any, shards: int) -> tuple[list, Any, int, int]:
    leaves, treedef = jax.tree.flatten(params)
    total = sum(int(leaf.size) for leaf in leaves)
    return leaves, treedef, total, padded_size(total, shards)


def _concatenate(leaves: list, total: int, padded: int) -> Array:
    flat = jnp.concatenate([leaf.reshape(-1) for leaf in leaves])
    if padded > total:
        flat = jnp.pad(flat, (0, padded - total))
    return flat


def zero1_flat_init(
    params: Any,
    *,
    shards: int,
    mesh: Any,
    shard_axis: str,
    moment_dtype: Any = jnp.float32,
) -> Zero1AdamState:
    _leaves, _treedef, _total, padded = _flat_layout(params, shards)
    replicated = jax.sharding.NamedSharding(mesh, P())
    split = jax.sharding.NamedSharding(mesh, P(shard_axis))

    def build() -> Zero1AdamState:
        return Zero1AdamState(
            count=jnp.zeros([], jnp.int32),
            mu=jnp.zeros((padded,), moment_dtype),
            nu=jnp.zeros((padded,), moment_dtype),
        )

    return jax.jit(
        build,
        out_shardings=Zero1AdamState(
            count=replicated, mu=split, nu=split
        ),
    )()


def zero1_flat_update(
    grads: Any,
    state: Zero1AdamState,
    params: Any,
    *,
    shard_axis: str,
    shards: int,
    learning_rate: Any,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    weight_decay: float = 0.0,
    decay_mask: Any = None,
    clip_norm: float = 0.0,
    scatter_dtype: Any = jnp.float32,
    gather_dtype: Any = jnp.float32,
) -> tuple[Any, Zero1AdamState]:
    leaves, treedef, total, padded = _flat_layout(params, shards)
    grad_leaves = jax.tree.leaves(grads)

    local = jax.lax.psum_scatter(
        _concatenate(grad_leaves, total, padded).astype(scatter_dtype),
        shard_axis,
        tiled=True,
    ).astype(jnp.float32)

    if clip_norm > 0.0:
        global_sq = jax.lax.psum(jnp.sum(jnp.square(local)), shard_axis)
        local = local * (
            clip_norm / jnp.maximum(clip_norm, jnp.sqrt(global_sq))
        )

    count = state.count + 1
    mu = (1 - b1) * local + b1 * state.mu.astype(jnp.float32)
    nu = (1 - b2) * jnp.square(local) + b2 * state.nu.astype(jnp.float32)
    update = (mu / (1 - b1**count)) / (
        jnp.sqrt(nu / (1 - b2**count) + eps_root) + eps
    )

    if weight_decay != 0.0:
        mask = (
            jax.tree.leaves(decay_mask)
            if decay_mask is not None
            else [True] * len(leaves)
        )
        decayed = _concatenate(
            [
                leaf if keep else jnp.zeros_like(leaf)
                for leaf, keep in zip(leaves, mask)
            ],
            total,
            padded,
        )
        slice_size = padded // shards
        update = update + weight_decay * jax.lax.dynamic_slice(
            decayed, (jax.lax.axis_index(shard_axis) * slice_size,), (slice_size,)
        )

    step_size = -(
        learning_rate(state.count) if callable(learning_rate) else learning_rate
    )
    gathered = jax.lax.all_gather(
        (step_size * update).astype(gather_dtype), shard_axis, tiled=True
    ).astype(jnp.float32)

    updates = []
    offset = 0
    for leaf in leaves:
        size = int(leaf.size)
        updates.append(
            jax.lax.slice(gathered, (offset,), (offset + size,))
            .astype(leaf.dtype)
            .reshape(leaf.shape)
        )
        offset += size
    moment_dtype = state.mu.dtype
    return jax.tree.unflatten(treedef, updates), Zero1AdamState(
        count=count,
        mu=mu.astype(moment_dtype),
        nu=nu.astype(moment_dtype),
    )
