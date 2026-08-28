"""Sharded Newton-Schulz must produce the same update as the replicated one.

Muon's orthogonalisation is identical work on every data-parallel rank and does
not shrink as chips are added, so it is sharded by assigning whole tensors to
ranks. Each rank returns zeros for the tensors it does not own and a sum over
the data axis rebuilds the full update, which has to be exactly what the
unsharded transform produces.

The equivalence check runs in a subprocess because faking four CPU devices
requires XLA_FLAGS to be set before JAX initialises its backend, which an
in-process test cannot guarantee once another test has already imported JAX.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import jax
import optax

from spectra_learning.training.muon import (
    _jax_muon_weight_dimension_numbers,
    _jax_orthogonalization_owners,
)

_EQUIVALENCE_SCRIPT = """
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P

from spectra_learning.training.muon import build_muon_transform

AXIS = "data"
key = jax.random.key(0)
shapes = {
    "b0": {"qkv": (192, 64), "out": (64, 64), "fc12": (256, 64)},
    "b1": {"qkv": (192, 64), "fc3": (64, 128), "wide": (48, 96)},
    "b2": {"fc12": (256, 64), "out": (64, 64), "tall": (128, 32)},
}
params = {}
for block, entries in shapes.items():
    params[block] = {}
    for name, shape in entries.items():
        key, sub = jax.random.split(key)
        params[block][name] = {"weight": jax.random.normal(sub, shape)}
grads = jax.tree.map(lambda v: jax.random.normal(key, v.shape) * 0.01, params)
mask = lambda tree: jax.tree.map(lambda v: v.ndim >= 2, tree)


def config(reduce_dtype):
    return {
        "weight_decay": 0.1,
        "muon_ns_steps": 5,
        "muon_shard_reduce_dtype": reduce_dtype,
        # XLA's CPU backend has no bf16 x bf16 -> f32 dot.
        "muon_ns_dtype": "float32",
    }


def transform(cfg, shard_axis, shard_count):
    return build_muon_transform(
        cfg,
        learning_rate=0.01,
        adam_learning_rate=0.001,
        weight_decay_mask=mask,
        shard_axis=shard_axis,
        shard_count=shard_count,
    )


reference_tx = transform(config("float32"), None, 1)
reference, _ = reference_tx.update(grads, reference_tx.init(params), params)


def sharded(reduce_dtype, shard_count):
    tx = transform(config(reduce_dtype), AXIS, shard_count)
    state = tx.init(params)
    mesh = jax.make_mesh((shard_count,), (AXIS,), devices=jax.devices()[:shard_count])

    @jax.jit
    @jax.shard_map(
        mesh=mesh,
        in_specs=(P(), P(), P()),
        out_specs=P(),
        axis_names={AXIS},
        check_vma=False,
    )
    def run(tree, grad, opt_state):
        updates, _ = tx.update(grad, opt_state, tree)
        return updates

    return run(params, grads, state)


for count in (2, 4):
    jax.tree.map(
        lambda a, b: np.testing.assert_allclose(
            np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-6
        ),
        reference,
        sharded("float32", count),
    )
jax.tree.map(
    lambda a, b: np.testing.assert_allclose(
        np.asarray(a), np.asarray(b), rtol=2e-2, atol=2e-3
    ),
    reference,
    sharded("bfloat16", 4),
)
print("EQUIVALENT")
"""


def test_sharded_update_matches_replicated_update() -> None:
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
        timeout=600,
    )
    assert "EQUIVALENT" in result.stdout, result.stdout + result.stderr


def test_owners_cover_every_weight_and_balance() -> None:
    weights = {
        f"w{index}": {"weight": jax.numpy.zeros(shape)}
        for index, shape in enumerate(
            [(192, 64), (64, 64), (256, 64), (64, 128), (48, 96), (128, 32)]
        )
    }
    dimension_numbers = _jax_muon_weight_dimension_numbers(weights)
    for shards in (1, 2, 4, 6):
        owners = _jax_orthogonalization_owners(
            weights, dimension_numbers, ns_steps=5, shards=shards
        )
        assigned = [entry["weight"] for entry in owners.values()]
        assert len(assigned) == len(weights)
        assert all(0 <= owner < shards for owner in assigned)
        # Every rank must get work, or the split leaves chips idle.
        assert len(set(assigned)) == shards


def test_dimension_numbers_split_fused_projections() -> None:
    tree = {"layer": {"qkv": {"weight": jax.numpy.zeros((3, 64, 64))}}}
    dimension_numbers = _jax_muon_weight_dimension_numbers(tree)
    fused = dimension_numbers["layer"]["qkv"]["weight"]
    assert fused == optax.contrib.MuonDimensionNumbers(
        reduction_axis=2, output_axis=1
    )
