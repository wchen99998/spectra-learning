from __future__ import annotations

import json
from typing import Any

import jax
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx
from flax.traverse_util import flatten_dict, unflatten_dict

from spectra_learning.training.storage import StoragePath, storage_join


def jax_checkpoint_dir(checkpoint_dir: StoragePath) -> str:
    return str(storage_join(checkpoint_dir, "orbax"))


def build_jax_checkpoint_manager(
    checkpoint_dir: StoragePath,
    *,
    max_to_keep: int | None = 5,
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
    *,
    metadata: dict[str, Any],
) -> None:
    manager.save(
        step,
        args=ocp.args.Composite(
            state=ocp.args.StandardSave(state),
            metadata=ocp.args.JsonSave(_canonical_metadata(metadata)),
        ),
    )


def restore_jax_training_state(
    manager: ocp.CheckpointManager,
    step: int,
    state: Any,
    *,
    expected_metadata: dict[str, Any],
) -> Any:
    # Capture shapes and shardings before releasing the initialized state.
    # Otherwise Orbax materializes a second full state on the accelerator and
    # the resume-only memory peak can exceed TPU HBM.
    target = jax.tree.map(_abstract_value, state)
    jax.block_until_ready(state)
    for value in jax.tree.leaves(state):
        if isinstance(value, jax.Array):
            value.delete()
    restored = manager.restore(
        step,
        args=ocp.args.Composite(
            state=ocp.args.StandardRestore(target),
            metadata=ocp.args.JsonRestore(),
        ),
    )
    actual_metadata = _canonical_metadata(restored.metadata)
    expected_metadata = _canonical_metadata(expected_metadata)
    if actual_metadata != expected_metadata:
        raise ValueError(
            "JAX checkpoint training contract mismatch: "
            f"expected={json.dumps(expected_metadata, sort_keys=True)} "
            f"actual={json.dumps(actual_metadata, sort_keys=True)}. "
            "Start a new workdir for the current training contract."
        )
    return restored.state


def restore_frozen_teacher_encoder(
    checkpoint_path: StoragePath,
    teacher_state: nnx.State,
) -> nnx.State:
    state_path = storage_join(checkpoint_path, "state")
    with ocp.StandardCheckpointer() as checkpointer:
        metadata = checkpointer.metadata(state_path).item_metadata

    teacher_values = {
        tuple(str(part) for part in path): (path, value)
        for path, value in nnx.to_flat_state(teacher_state)
    }
    source_metadata = {
        root: flatten_dict(metadata[root]["encoder"])
        for root in ("trainable_params", "static_state")
    }
    source_paths = {
        tuple(str(part) for part in path)
        for root_metadata in source_metadata.values()
        for path in root_metadata
    }
    if source_paths != teacher_values.keys():
        missing = sorted(teacher_values.keys() - source_paths)
        unexpected = sorted(source_paths - teacher_values.keys())
        raise ValueError(
            "Frozen teacher encoder checkpoint mismatch: "
            f"missing={missing} unexpected={unexpected}"
        )

    target: dict[str, dict[str, Any]] = {}
    restore_args: dict[str, dict[str, Any]] = {}
    for root, root_metadata in source_metadata.items():
        target_leaves = {}
        restore_arg_leaves = {}
        for source_path, leaf_metadata in root_metadata.items():
            serialized_path = tuple(str(part) for part in source_path)
            _, value = teacher_values[serialized_path]
            if (
                tuple(value.shape) != tuple(leaf_metadata.shape)
                or np.dtype(value.dtype) != np.dtype(leaf_metadata.dtype)
            ):
                raise ValueError(
                    "Frozen teacher encoder leaf mismatch at "
                    f"{'.'.join(serialized_path)}: "
                    f"checkpoint=({tuple(leaf_metadata.shape)}, {leaf_metadata.dtype}) "
                    f"model=({tuple(value.shape)}, {value.dtype})"
                )
            target_leaves[source_path] = jax.ShapeDtypeStruct(
                value.shape,
                value.dtype,
                sharding=value.sharding,
            )
            restore_arg_leaves[source_path] = ocp.ArrayRestoreArgs(
                sharding=value.sharding,
                global_shape=value.shape,
                dtype=value.dtype,
            )
        target[root] = {"encoder": unflatten_dict(target_leaves)}
        restore_args[root] = {"encoder": unflatten_dict(restore_arg_leaves)}

    with ocp.PyTreeCheckpointer() as checkpointer:
        restored = checkpointer.restore(
            state_path,
            args=ocp.args.PyTreeRestore(
                item=target,
                restore_args=restore_args,
                partial_restore=True,
            ),
        )

    restored_values = {}
    for root in ("trainable_params", "static_state"):
        for source_path, value in flatten_dict(restored[root]["encoder"]).items():
            restored_values[tuple(str(part) for part in source_path)] = value
    return nnx.from_flat_state(
        (native_path, restored_values[serialized_path])
        for serialized_path, (native_path, _value) in teacher_values.items()
    )


def jax_training_checkpoint_metadata(
    training_task: str,
    task_contract: dict[str, Any],
) -> dict[str, Any]:
    return _canonical_metadata(
        {
            "format_version": 1,
            "training_task": training_task,
            "task_contract": task_contract,
        }
    )


def _abstract_value(value: Any) -> Any:
    if isinstance(value, jax.Array):
        return jax.ShapeDtypeStruct(value.shape, value.dtype, sharding=value.sharding)
    return value


def _canonical_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(metadata, sort_keys=True))
