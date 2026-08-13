from __future__ import annotations

import json
import logging
import os
import signal
import threading
from typing import Any
from urllib import error, parse, request

import jax
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx
from flax.traverse_util import flatten_dict, unflatten_dict

from spectra_learning.training.storage import StoragePath, storage_join


_GCE_METADATA_ROOT = "http://metadata.google.internal/computeMetadata/v1/instance"
_GCE_METADATA_HEADERS = {"Metadata-Flavor": "Google"}


class EmergencyCheckpointMonitor:
    """Turn VM termination notices into a durable checkpoint request."""

    def __init__(self, *, watch_gce_metadata: bool) -> None:
        self._watch_gce_metadata = watch_gce_metadata
        self._requested = threading.Event()
        self._stopped = threading.Event()
        self._reason: str | None = None
        self._previous_sigterm_handler: Any = None

    @classmethod
    def for_current_environment(cls) -> EmergencyCheckpointMonitor:
        cluster_info = os.environ.get("SKYPILOT_CLUSTER_INFO")
        watch_gce_metadata = (
            cluster_info is not None
            and json.loads(cluster_info).get("cloud", "").lower() == "gcp"
        )
        return cls(watch_gce_metadata=watch_gce_metadata)

    def __enter__(self) -> EmergencyCheckpointMonitor:
        self._previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, self._handle_sigterm)
        if self._watch_gce_metadata:
            for metadata_key in ("preempted", "maintenance-event"):
                threading.Thread(
                    target=self._watch_metadata_key,
                    args=(metadata_key,),
                    name=f"spectra-{metadata_key}-watcher",
                    daemon=True,
                ).start()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        del exc_type, exc_value, traceback
        self._stopped.set()
        signal.signal(signal.SIGTERM, self._previous_sigterm_handler)

    @property
    def requested(self) -> bool:
        return self._requested.is_set()

    @property
    def reason(self) -> str | None:
        return self._reason

    def request(self, reason: str) -> None:
        if not self._requested.is_set():
            self._reason = reason
            self._requested.set()

    def wait_for_forced_termination(self) -> None:
        logging.warning(
            "Emergency checkpoint is durable; waiting for the VM to terminate."
        )
        while True:
            self._stopped.wait(3600)

    def _handle_sigterm(self, signum: int, frame: Any) -> None:
        del signum, frame
        self.request("SIGTERM")

    def _watch_metadata_key(self, metadata_key: str) -> None:
        etag = ""
        while not self._stopped.is_set() and not self._requested.is_set():
            query = {"wait_for_change": "true", "timeout_sec": "60"}
            if etag:
                query["last_etag"] = etag
            url = f"{_GCE_METADATA_ROOT}/{metadata_key}?{parse.urlencode(query)}"
            metadata_request = request.Request(url, headers=_GCE_METADATA_HEADERS)
            try:
                with request.urlopen(metadata_request, timeout=65) as response:
                    value = response.read().decode().strip()
                    etag = response.headers.get("ETag", "")
            except (error.URLError, TimeoutError):
                continue
            if metadata_key == "preempted" and value == "TRUE":
                self.request("GCE preemption notice")
            if metadata_key == "maintenance-event" and value.startswith("TERMINATE"):
                self.request("GCE host-maintenance notice")


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
    allowed_config_keys: tuple[str, ...] = (),
    allowed_dataset_keys: tuple[str, ...] = (),
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
    for key in allowed_config_keys:
        actual_metadata["task_contract"]["config"].pop(key, None)
        expected_metadata["task_contract"]["config"].pop(key, None)
    for key in allowed_dataset_keys:
        actual_metadata["task_contract"]["dataset"].pop(key, None)
        expected_metadata["task_contract"]["dataset"].pop(key, None)
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
    *,
    path_renames: dict[str, str] | None = None,
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

    def target_path(path: tuple[Any, ...]) -> tuple[str, ...]:
        renames = {} if path_renames is None else path_renames
        return tuple(renames.get(str(part), str(part)) for part in path)

    source_paths = {
        target_path(path)
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
            serialized_path = target_path(source_path)
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
            restored_values[target_path(source_path)] = value
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
