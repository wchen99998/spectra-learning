"""The ZeRO-1 training step must move the parameters exactly like the replicated one.

`tests/test_zero1_adam.py` checks the update rule in isolation. This checks the
wiring around it: that gradients are divided by the global denominator before
the scatter rather than after, that the shard_map specs carry a sharded
optimizer state through unchanged, and that the parameters coming out are the
ones the replicated path would have produced.

Runs in a subprocess so four CPU devices can be faked before JAX initialises.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

_SCRIPT = """
import jax
import sys
import jax.numpy as jnp
import numpy as np
from flax import nnx

from spectra_learning.config import load_config
from spectra_learning.models.factory_jax import build_model_from_config
from spectra_learning.models.spectrum_metadata import INSTRUMENT_FAMILIES
from spectra_learning.training.optimizer_sharding import (
    zero1_adam_init_sharded,
    zero1_flat_init,
)
from spectra_learning.training.pretrain_jax import (
    JAX_DATA_AXIS,
    _jax_data_mesh_for_device_count,
    _replicate_tree_on_data_mesh,
    init_pure_optax_train_state,
    jax_zero1_settings,
    make_pure_accumulated_train_step,
    numpy_batch_to_jax,
    prepare_jax_training_config,
)

DEVICES = 4
ACCUM = 2
MICRO = 2
PEAKS = 16

overrides = {
    "batch_size": MICRO * DEVICES * ACCUM,
    "gradient_accumulation_steps": ACCUM,
    "jax_mesh_devices": str(DEVICES),
    "jax_distributed_initialize": False,
    "jax_enable_compilation_cache": False,
    "enable_wandb": False,
    "training_max_steps": 4,
    "model_dim": 128,
    "predictor_dim": 128,
    "encoder_num_heads": 2,
    "masked_latent_predictor_num_heads": 2,
    "encoder_num_layers": 2,
    "num_peaks": PEAKS,
    # Exercise decoupled decay and clipping, both of which the sharded path
    # has to reproduce from partial gradients.
    "weight_decay": 0.1,
    "grad_clip_norm": 1.0,
}
frozen_teacher = sys.argv[1] == 'frozen_bf16'
if frozen_teacher:
    overrides.update(
        training_mode='mae_teacher_jepa', optimizer='adam',
        optimizer_state_dtype='bf16', weight_decay=0.0,
        masked_token_loss_weight=1.0, mae_loss_weight=0.0,
        contrastive_loss_weight=0.0, online_probe_loss_weight=0.0,
    )
config = load_config(
    "configs/1b_singlemixer_dense_adamw_metadata_adaln_multiscale_n128_b2048_500k_hd256.py",
    overrides,
)
prepare_jax_training_config(config)

mesh = _jax_data_mesh_for_device_count(DEVICES)
metadata_dim = 21 + len(INSTRUMENT_FAMILIES)
rng = np.random.default_rng(0)
shape = (ACCUM, MICRO * DEVICES, PEAKS)
context = PEAKS // 2
masks = np.zeros(shape, dtype=np.bool_)
masks[..., :context] = True
targets = np.zeros((*shape[:2], 1, PEAKS), dtype=np.bool_)
targets[..., context : context + 4] = True
host_batch = {
    "peak_mz": rng.random(shape, dtype=np.float32),
    "peak_intensity": rng.random(shape, dtype=np.float32),
    "peak_valid_mask": np.ones(shape, dtype=np.bool_),
    "context_mask": masks,
    "target_masks": targets,
    "precursor_mz": np.full(shape[:2], 0.8, dtype=np.float32),
    "spectrum_metadata": np.zeros((*shape[:2], metadata_dim), dtype=np.float32),
}
batch = numpy_batch_to_jax(host_batch, data_mesh=mesh, batch_axis=1)


def run(zero1_shards, mode='flat'):
    model = build_model_from_config(config)
    graphdef, params, static_state, opt_state, optimizer = (
        init_pure_optax_train_state(
            config, model, total_steps=4, zero1_shards=zero1_shards
        )
    )
    params = _replicate_tree_on_data_mesh(params, mesh)
    static_state = _replicate_tree_on_data_mesh(static_state, mesh)
    if zero1_shards:
        init = zero1_flat_init if mode == 'flat' else zero1_adam_init_sharded
        opt_state = init(
            params, shards=zero1_shards, mesh=mesh, shard_axis=JAX_DATA_AXIS,
            moment_dtype=jnp.bfloat16 if frozen_teacher else jnp.float32,
        )
        assert all(
            x.dtype == (jnp.bfloat16 if frozen_teacher else jnp.float32)
            for x in jax.tree.leaves(opt_state.mu)
        )
    else:
        opt_state = _replicate_tree_on_data_mesh(opt_state, mesh)
    step = make_pure_accumulated_train_step(
        graphdef,
        optimizer,
        sharded=True,
        data_mesh=mesh,
        grad_all_reduce_dtype=jnp.float32,
        zero1_shards=zero1_shards,
        zero1_mode=mode,
        zero1_settings=(
            jax_zero1_settings(config, total_steps=4) if zero1_shards else None
        ),
    )
    for _ in range(3):
        params, opt_state, _metrics = step(params, static_state, opt_state, batch)
    return params


replicated = run(0)
for mode in ('per_leaf', 'flat'):
    sharded = run(DEVICES, mode)
    jax.tree.map(
        lambda a, b: np.testing.assert_allclose(
            np.asarray(a), np.asarray(b), rtol=2e-4, atol=2e-6
        ),
        replicated,
        sharded,
    )
    print('mode', mode, 'matches')
print("EQUIVALENT")
"""


@pytest.mark.parametrize('recipe', ('mae_fp32', 'frozen_bf16'))
def test_zero1_train_step_matches_replicated_train_step(recipe: str) -> None:
    environment = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
    }
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_SCRIPT), recipe],
        capture_output=True,
        text=True,
        env=environment,
        timeout=1800,
    )
    assert "EQUIVALENT" in result.stdout, result.stdout[-4000:] + result.stderr[-4000:]
