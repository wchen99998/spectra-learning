"""Replicating the train state must not leave a second copy pinned on one device.

`nnx.split` returns states that share device buffers with the model, so the
parameters built during construction stay resident on whichever single device
made them until the model is rebound. At 1B parameters that is 4 GB on one chip
per host, and since every device runs the same program, the busiest device sets
the micro-batch ceiling for all of them.

Runs in a subprocess so four CPU devices can be faked before JAX initialises.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

_SCRIPT = """
import jax
from flax import nnx

from spectra_learning.config import load_config
from spectra_learning.models.factory_jax import build_model_from_config
from spectra_learning.training.pretrain_jax import (
    _jax_data_mesh_for_device_count,
    init_pure_optax_train_state,
    prepare_jax_training_config,
    replicate_trainable_state_on_mesh,
    trainable_param_filter,
)

DEVICES = 4
config = load_config(
    "configs/1b_singlemixer_dense_adamw_metadata_adaln_multiscale_n128_b2048_500k_hd256.py",
    {
        "model_dim": 128,
        "predictor_dim": 128,
        "encoder_num_heads": 2,
        "masked_latent_predictor_num_heads": 2,
        "encoder_num_layers": 2,
        "num_peaks": 16,
        "jax_mesh_devices": str(DEVICES),
        "jax_distributed_initialize": False,
        "jax_enable_compilation_cache": False,
        "enable_wandb": False,
    },
)
prepare_jax_training_config(config)
mesh = _jax_data_mesh_for_device_count(DEVICES)

model = build_model_from_config(config)
graphdef, params, static_state, opt_state, optimizer = init_pure_optax_train_state(
    config, model, total_steps=4
)

before = jax.tree.leaves(nnx.state(model, nnx.Param))
assert before, "model exposed no parameters"
# Establish there is something to fix: construction leaves every parameter on a
# single device, and the split aliases those same buffers.
assert all(leaf.sharding.num_devices == 1 for leaf in before), (
    "parameters were already distributed; this test would prove nothing"
)
assert any(
    leaf is split for leaf in before for split in jax.tree.leaves(params)
), "nnx.split no longer aliases the model"

params, static_state = replicate_trainable_state_on_mesh(
    model, params, static_state, mesh
)

after = jax.tree.leaves(nnx.state(model, nnx.Param))
# `trainable_param_filter` routes the frozen teacher parameters into
# static_state, so the model's parameters span both replicated trees.
live = jax.tree.leaves(params) + jax.tree.leaves(static_state)
live_ids = {id(leaf) for leaf in live}
for model_leaf in after:
    # Identity, not equality: a copy would still pin a second buffer.
    assert id(model_leaf) in live_ids, "model holds its own copy of a parameter"
    assert model_leaf.sharding.num_devices == DEVICES, model_leaf.sharding
print("RELEASED")
"""


def test_replication_releases_the_unreplicated_parameters() -> None:
    environment = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
    }
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_SCRIPT)],
        capture_output=True,
        text=True,
        env=environment,
        timeout=900,
    )
    assert "RELEASED" in result.stdout, (
        result.stdout[-4000:] + result.stderr[-4000:]
    )
