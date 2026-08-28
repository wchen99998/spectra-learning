"""ZeRO-1 Adam must reproduce the replicated `chain(clip, adamw)` update.

Adam is elementwise, so cutting the flattened parameter vector into one slice
per data-parallel rank changes only where the arithmetic happens, not what it
computes. That has to hold across several steps, because the bias correction
and the learning-rate schedule both depend on a step counter that the sharded
path maintains separately.

The check runs in a subprocess because faking four CPU devices requires
XLA_FLAGS to be set before JAX initialises its backend, which an in-process
test cannot guarantee once another test has already imported JAX.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

from spectra_learning.training.optimizer_sharding import (
    local_slice_size,
    padded_size,
)

_EQUIVALENCE_SCRIPT = """
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import PartitionSpec as P

from spectra_learning.training.optimizer_sharding import (
    zero1_adam_init,
    zero1_adam_update,
    zero1_state_specs,
)

AXIS = "data"
STEPS = 3
B1, B2, WD, CLIP = 0.9, 0.95, 0.1, 1.0
schedule = lambda count: 0.01 / (1.0 + 0.05 * count)
# Sizes that are deliberately not multiples of the shard count, so the padding
# path is exercised: 6 and 30 and 130 all leave a remainder at 4 shards.
shapes = {
    "enc": {"weight": (13, 10), "bias": (6,)},
    "mlp": {"weight": (30, 7), "bias": (30,)},
    "head": {"weight": (5, 26), "scale": (130,)},
}
mask_fn = lambda tree: jax.tree.map(lambda v: v.ndim >= 2, tree)

key = jax.random.key(0)
params = {}
for block, entries in shapes.items():
    params[block] = {}
    for name, shape in entries.items():
        key, sub = jax.random.split(key)
        params[block][name] = jax.random.normal(sub, shape)


def per_rank_grads(shards):
    grads = []
    k = jax.random.key(7)
    for _ in range(STEPS):
        k, sub = jax.random.split(k)
        keys = jax.random.split(sub, shards)
        grads.append(
            jax.tree.map(
                lambda v: jnp.stack(
                    [jax.random.normal(keys[r], v.shape) for r in range(shards)]
                ),
                params,
            )
        )
    return grads


def reference(step_grads):
    tx = optax.chain(
        optax.clip_by_global_norm(CLIP),
        optax.adamw(schedule, b1=B1, b2=B2, weight_decay=WD, mask=mask_fn),
    )
    state = tx.init(params)
    tree = params
    for stacked in step_grads:
        summed = jax.tree.map(lambda g: jnp.sum(g, axis=0), stacked)
        updates, state = tx.update(summed, state, tree)
        tree = optax.apply_updates(tree, updates)
    return tree


def sharded(step_grads, shards, scatter_dtype, gather_dtype):
    mesh = jax.make_mesh((shards,), (AXIS,), devices=jax.devices()[:shards])
    specs = zero1_state_specs(AXIS)

    @jax.jit
    @jax.shard_map(
        mesh=mesh,
        in_specs=(P(), P(AXIS), specs),
        out_specs=(P(), specs),
        axis_names={AXIS},
        check_vma=False,
    )
    def run(tree, stacked, state):
        grad = jax.tree.map(lambda g: g[0], stacked)
        updates, state = zero1_adam_update(
            grad,
            state,
            tree,
            shard_axis=AXIS,
            shards=shards,
            learning_rate=schedule,
            b1=B1,
            b2=B2,
            weight_decay=WD,
            decay_mask=mask_fn(tree),
            clip_norm=CLIP,
            scatter_dtype=scatter_dtype,
            gather_dtype=gather_dtype,
        )
        return optax.apply_updates(tree, updates), state

    def place(tree_, spec):
        return jax.tree.map(
            lambda v: jax.device_put(
                v, jax.sharding.NamedSharding(mesh, spec)
            ),
            tree_,
        )

    state = zero1_adam_init(params, shards=shards)
    state = state._replace(
        count=place(state.count, P()),
        mu=place(state.mu, P(AXIS)),
        nu=place(state.nu, P(AXIS)),
    )
    tree = place(params, P())
    for stacked in step_grads:
        tree, state = run(tree, place(stacked, P(AXIS)), state)
    return tree, state


for shards in (2, 4):
    step_grads = per_rank_grads(shards)
    want = reference(step_grads)
    got, state = sharded(step_grads, shards, jnp.float32, jnp.float32)
    jax.tree.map(
        lambda a, b: np.testing.assert_allclose(
            np.asarray(a), np.asarray(b), rtol=2e-5, atol=2e-6
        ),
        want,
        got,
    )
    # The moments must actually be sharded, not quietly replicated.
    for leaf, param in zip(jax.tree.leaves(state.mu), jax.tree.leaves(params)):
        assert leaf.size >= param.size, (leaf.size, param.size)
        assert leaf.sharding.num_devices == shards, leaf.sharding

# bf16 on the wire is a lossy but bounded approximation of the same update.
step_grads = per_rank_grads(4)
want = reference(step_grads)
got, _ = sharded(step_grads, 4, jnp.bfloat16, jnp.bfloat16)
jax.tree.map(
    lambda a, b: np.testing.assert_allclose(
        np.asarray(a), np.asarray(b), rtol=5e-2, atol=5e-3
    ),
    want,
    got,
)
print("EQUIVALENT")
"""


def test_zero1_update_matches_replicated_adamw() -> None:
    environment = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
    }
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_EQUIVALENCE_SCRIPT)],
        capture_output=True,
        text=True,
        env=environment,
        timeout=900,
    )
    assert "EQUIVALENT" in result.stdout, result.stdout + result.stderr


def test_padding_divides_evenly_and_ignores_shard_count() -> None:
    """The stored moment shape must not depend on how many chips are training.

    Otherwise a checkpoint written on 16 chips could not be resumed on 64, which
    is exactly the migration this optimizer path exists to enable.
    """
    powers_of_two = (1, 2, 4, 8, 16, 32, 64, 128)
    for size in (1, 6, 130, 1536, 4096, 4097, 1_000_000):
        stored = {padded_size(size, shards) for shards in powers_of_two}
        assert len(stored) == 1, (size, stored)
        for shards in powers_of_two:
            assert padded_size(size, shards) >= size
            assert padded_size(size, shards) % shards == 0
            assert (
                local_slice_size(size, shards) * shards
                == padded_size(size, shards)
            )


def test_padding_still_divides_for_non_power_of_two_shards() -> None:
    for shards in (3, 6, 12):
        for size in (1, 130, 1536):
            assert padded_size(size, shards) % shards == 0
