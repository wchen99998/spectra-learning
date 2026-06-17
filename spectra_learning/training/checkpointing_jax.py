from __future__ import annotations

from typing import Any

import jax
import orbax.checkpoint as ocp

from spectra_learning.training.storage import StoragePath, storage_join


def jax_checkpoint_dir(checkpoint_dir: StoragePath) -> str:
    return str(storage_join(checkpoint_dir, "orbax"))


def build_jax_checkpoint_manager(
    checkpoint_dir: StoragePath,
    *,
    max_to_keep: int = 5,
    enable_async_checkpointing: bool = True,
) -> ocp.CheckpointManager:
    options = ocp.CheckpointManagerOptions(
        max_to_keep=max_to_keep,
        enable_async_checkpointing=enable_async_checkpointing,
    )
    return ocp.CheckpointManager(jax_checkpoint_dir(checkpoint_dir), options=options)


def save_jax_training_state(
    manager: ocp.CheckpointManager,
    step: int,
    state: Any,
) -> None:
    manager.save(step, args=ocp.args.StandardSave(state))


def restore_jax_training_state(
    manager: ocp.CheckpointManager,
    step: int,
    state: Any,
) -> Any:
    # Restore against abstract targets that carry the live shardings so every
    # array lands back on its original mesh layout on every host.
    target = jax.tree.map(_abstract_value, state)
    return manager.restore(step, args=ocp.args.StandardRestore(target))


def _abstract_value(value: Any) -> Any:
    if isinstance(value, jax.Array):
        return jax.ShapeDtypeStruct(value.shape, value.dtype, sharding=value.sharding)
    return value
