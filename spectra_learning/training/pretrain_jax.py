from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from flax import nnx
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from ml_collections import config_dict
from tqdm import tqdm

from spectra_learning.data.gems.datamodule import GemsNativeDataModule
from spectra_learning.models.common_jax import Array
from spectra_learning.models.factory_jax import build_model_from_config
from spectra_learning.models.model_jax import PeakSetJEPAJax
from spectra_learning.probes.massspec.msg_probe_jax import run_msg_probe_jax
from spectra_learning.probes.massspec.msg_settings import (
    msg_probe_variants_from_config,
)
from spectra_learning.training.cadence import (
    msg_probe_interval,
    should_run_at_step,
    should_run_at_step_or_final,
    validation_interval,
    validation_steps,
)
from spectra_learning.training.checkpointing_jax import (
    build_jax_checkpoint_manager,
    restore_jax_training_state,
    save_jax_training_state,
)
from spectra_learning.training.jax_runtime_flags import configure_jax_tpu_xla_flags
from spectra_learning.training.logging import (
    MetricLogger,
    build_logger,
    log_msg_probe_metrics,
)
from spectra_learning.training.schedules import learning_rate_at_step
from spectra_learning.training.storage import (
    local_scratch_dir,
    normalize_storage_path,
    storage_join,
    storage_mkdir,
)


JAX_DATA_AXIS = "data"


@dataclass(frozen=True)
class _StagedJaxTrainMetrics:
    metrics: dict[str, Array]
    epoch: int
    global_step: int
    total_steps: int


def trainable_param_filter(path: tuple[object, ...], value: object) -> bool:
    if not isinstance(value, nnx.Param):
        return False
    frozen_modules = {
        "teacher_encoder",
        "teacher_target_projector",
        "position_embedding",
        "predictor_position_embedding",
        "predictor_pair_position_embedding",
    }
    if any(part in frozen_modules for part in path):
        return False
    return path[-1] != "b"


def configure_jax_runtime(config: Any) -> None:
    configure_jax_tpu_xla_flags()
    if bool(_config_get(config, "jax_log_compiles", False)) or _env_enabled(
        "JAX_LOG_COMPILES"
    ):
        jax.config.update("jax_log_compiles", True)
    if bool(_config_get(config, "jax_explain_cache_misses", False)) or _env_enabled(
        "JAX_EXPLAIN_CACHE_MISSES"
    ):
        jax.config.update("jax_explain_cache_misses", True)
    compilation_cache_dir = str(_config_get(config, "jax_compilation_cache_dir", ""))
    if compilation_cache_dir:
        jax.config.update("jax_compilation_cache_dir", compilation_cache_dir)
        jax.config.update(
            "jax_enable_compilation_cache",
            bool(_config_get(config, "jax_enable_compilation_cache", True)),
        )
    min_compile_time = _config_get(
        config, "jax_persistent_cache_min_compile_time_secs", None
    )
    if min_compile_time is not None:
        jax.config.update(
            "jax_persistent_cache_min_compile_time_secs",
            float(min_compile_time),
        )
    min_entry_size = _config_get(
        config, "jax_persistent_cache_min_entry_size_bytes", None
    )
    if min_entry_size is not None:
        jax.config.update(
            "jax_persistent_cache_min_entry_size_bytes",
            int(min_entry_size),
        )


def initialize_jax_distributed(config: Any) -> None:
    if jax.distributed.is_initialized():
        return
    enabled = bool(_config_get(config, "jax_distributed_initialize", False)) or any(
        os.environ.get(key)
        for key in (
            "JAX_DISTRIBUTED_INITIALIZE",
            "JAX_COORDINATOR_ADDRESS",
            "JAX_COORDINATOR_ADDR",
            "JAX_NUM_PROCESSES",
            "JAX_PROCESS_COUNT",
        )
    )
    if not enabled:
        return
    kwargs = {
        key: value
        for key, value in {
            "coordinator_address": _config_or_env(
                config,
                "jax_coordinator_address",
                ("JAX_COORDINATOR_ADDRESS", "JAX_COORDINATOR_ADDR"),
            )
            or None,
            "num_processes": _optional_int(
                _config_or_env(
                    config,
                    "jax_num_processes",
                    ("JAX_NUM_PROCESSES", "JAX_PROCESS_COUNT", "WORLD_SIZE"),
                )
            ),
            "process_id": _optional_int(
                _config_or_env(
                    config,
                    "jax_process_id",
                    ("JAX_PROCESS_ID", "JAX_PROCESS_INDEX", "RANK"),
                )
            ),
            "local_device_ids": _local_device_ids(
                _config_or_env(
                    config,
                    "jax_local_device_ids",
                    ("JAX_LOCAL_DEVICE_IDS", "LOCAL_DEVICE_IDS"),
                )
            ),
            "cluster_detection_method": _config_or_env(
                config,
                "jax_cluster_detection_method",
                ("JAX_CLUSTER_DETECTION_METHOD",),
            )
            or None,
            "initialization_timeout": int(
                _config_or_env(
                    config,
                    "jax_initialization_timeout",
                    ("JAX_INITIALIZATION_TIMEOUT",),
                    300,
                )
            ),
        }.items()
        if value is not None
    }
    jax.distributed.initialize(**kwargs)


def build_jax_optimizer(
    config: Any,
    model: PeakSetJEPAJax,
    *,
    total_steps: int | None = None,
) -> nnx.Optimizer:
    optimizer = build_jax_optax_transform(config, total_steps=total_steps)
    return nnx.Optimizer(model, optimizer, wrt=trainable_param_filter)


def build_jax_optax_transform(
    config: Any,
    *,
    total_steps: int | None = None,
) -> optax.GradientTransformation:
    optimizer = str(_config_get(config, "optimizer", "adamw")).lower()
    if optimizer == "muon":
        return _jax_muon_transform(config, total_steps=total_steps)
    return _jax_adamw_transform(config, total_steps=total_steps)


def _jax_adamw_transform(
    config: Any,
    *,
    total_steps: int | None = None,
) -> optax.GradientTransformation:
    return optax.adamw(
        learning_rate=_jax_learning_rate_schedule(config, total_steps=total_steps),
        b2=float(_config_get(config, "b2", 0.999)),
        weight_decay=float(_config_get(config, "weight_decay", 0.0)),
        mask=_jax_weight_decay_mask,
    )


def _jax_muon_transform(
    config: Any,
    *,
    total_steps: int | None = None,
) -> optax.GradientTransformation:
    return _jax_split_qkv_transform(
        optax.contrib.muon(
            learning_rate=_jax_learning_rate_schedule(config, total_steps=total_steps),
            ns_coeffs=_config_get(config, "muon_ns_coeffs", (3.4445, -4.7750, 2.0315)),
            ns_steps=int(_config_get(config, "muon_ns_steps", 5)),
            beta=float(_config_get(config, "muon_beta", 0.95)),
            eps=float(_config_get(config, "muon_eps", 1e-8)),
            weight_decay=float(_config_get(config, "weight_decay", 0.0)),
            weight_decay_mask=_jax_weight_decay_mask,
            mu_dtype=_config_get(config, "muon_mu_dtype", None),
            nesterov=bool(_config_get(config, "muon_nesterov", True)),
            adaptive=bool(_config_get(config, "muon_adaptive", False)),
            preconditioning=str(_config_get(config, "muon_preconditioning", "frobenius")),
            adam_b1=float(_config_get(config, "muon_adam_b1", 0.9)),
            adam_b2=float(
                _config_get(config, "muon_adam_b2", _config_get(config, "b2", 0.999))
            ),
            adam_eps_root=float(_config_get(config, "muon_adam_eps_root", 0.0)),
            adam_weight_decay=float(_config_get(config, "muon_adam_weight_decay", 0.0)),
            adam_learning_rate=_jax_muon_adam_learning_rate_schedule(
                config,
                total_steps=total_steps,
            ),
            muon_weight_dimension_numbers=_jax_muon_weight_dimension_numbers,
            consistent_rms=_jax_muon_consistent_rms(config),
        )
    )


def _jax_muon_adam_learning_rate_schedule(
    config: Any,
    *,
    total_steps: int | None = None,
):
    adam_learning_rate = _config_get(config, "muon_adam_learning_rate", None)
    if adam_learning_rate is None:
        return None
    return _jax_learning_rate_schedule(
        config,
        total_steps=total_steps,
        base_lr=float(adam_learning_rate),
        min_lr=_config_get(config, "muon_adam_min_learning_rate", None),
    )


def _jax_muon_consistent_rms(config: Any) -> float | None:
    adjust_lr_fn = str(_config_get(config, "muon_adjust_lr_fn", "") or "").lower()
    if adjust_lr_fn == "match_rms_adamw":
        return 0.2
    return _config_get(config, "muon_consistent_rms", None)


def _jax_learning_rate_schedule(
    config: Any,
    *,
    total_steps: int | None = None,
    base_lr: float | None = None,
    min_lr: Any = None,
):
    if base_lr is None:
        base_lr = float(_config_get(config, "learning_rate", 1e-3))
        if min_lr is None:
            min_lr = _config_get(config, "min_learning_rate", None)
    else:
        base_lr = float(base_lr)
    warmup_steps = int(_config_get(config, "warmup_steps", 0))
    return _jax_cosine_learning_rate_schedule(
        config,
        total_steps=total_steps,
        base_lr=base_lr,
        warmup_steps=warmup_steps,
        min_lr=min_lr,
    )


def _jax_cosine_learning_rate_schedule(
    config: Any,
    *,
    total_steps: int | None,
    base_lr: float,
    warmup_steps: int,
    min_lr: Any,
):
    min_lr = float(min_lr) if min_lr is not None else 0.1 * base_lr
    raw_total_steps = total_steps or _config_get(config, "training_max_steps", None)
    if raw_total_steps is None:
        return base_lr
    total_steps = int(raw_total_steps)

    def schedule(step):
        step = jnp.asarray(step, dtype=jnp.float32)
        warmup = base_lr * (1e-8 + (1.0 - 1e-8) * step / max(1, warmup_steps))
        ratio = jnp.clip(
            (step - float(warmup_steps)) / float(max(1, total_steps - warmup_steps)),
            0.0,
            1.0,
        )
        decay = min_lr + (base_lr - min_lr) * 0.5 * (1.0 + jnp.cos(jnp.pi * ratio))
        if warmup_steps <= 0:
            return decay
        return jnp.where(step < warmup_steps, warmup, decay)

    return schedule


def _scheduled_jax_learning_rate(
    config: Any,
    *,
    global_step: int,
    total_steps: int,
) -> float:
    return learning_rate_at_step(
        global_step,
        base_lr=float(_config_get(config, "learning_rate", 1e-3)),
        total_steps=total_steps,
        warmup_steps=int(_config_get(config, "warmup_steps", 0)),
        min_learning_rate=_config_get(config, "min_learning_rate", None),
    )


def _jax_weight_decay_mask(params: Any) -> Any:
    return jax.tree.map_with_path(
        lambda path, value: _tree_path_key(path[-1]) == "weight" and value.ndim >= 2,
        params,
    )


def _jax_muon_weight_dimension_numbers(params: Any) -> Any:
    return jax.tree.map_with_path(
        lambda path, value: (
            _jax_muon_weight_dimension_number(path, value)
            if _tree_path_key(path[-1]) == "weight" and value.ndim >= 2
            else None
        ),
        params,
    )


def _jax_muon_weight_dimension_number(
    path: tuple[Any, ...],
    value: Any,
) -> optax.contrib.MuonDimensionNumbers:
    if _jax_qkv_weight_path(path) and value.ndim == 3:
        return optax.contrib.MuonDimensionNumbers(reduction_axis=2, output_axis=1)
    return optax.contrib.MuonDimensionNumbers(reduction_axis=1, output_axis=0)


def _jax_split_qkv_transform(
    transform: optax.GradientTransformation,
) -> optax.GradientTransformation:
    def init_fn(params):
        return transform.init(_jax_split_qkv_tree(params))

    def update_fn(updates, state, params=None):
        split_updates = _jax_split_qkv_tree(updates)
        split_params = None if params is None else _jax_split_qkv_tree(params)
        split_updates, state = transform.update(split_updates, state, split_params)
        return _jax_unsplit_qkv_tree(split_updates, updates), state

    return optax.GradientTransformation(init_fn, update_fn)


def _jax_split_qkv_tree(tree: Any) -> Any:
    return jax.tree.map_with_path(_jax_split_qkv_leaf, tree)


def _jax_unsplit_qkv_tree(tree: Any, reference: Any) -> Any:
    return jax.tree.map_with_path(
        lambda path, value: _jax_unsplit_qkv_leaf(path, value, reference),
        tree,
    )


def _jax_split_qkv_leaf(path: tuple[Any, ...], value: Any) -> Any:
    if _jax_split_qkv_leaf_path(path, value):
        return value.reshape(3, value.shape[0] // 3, value.shape[1])
    return value


def _jax_unsplit_qkv_leaf(path: tuple[Any, ...], value: Any, reference: Any) -> Any:
    if _jax_split_qkv_leaf_path(path, _jax_tree_get_path(reference, path)):
        return value.reshape(value.shape[0] * value.shape[1], value.shape[2])
    return value


def _jax_split_qkv_leaf_path(path: tuple[Any, ...], value: Any) -> bool:
    return (
        _jax_qkv_weight_path(path)
        and value.ndim == 2
        and value.shape[0] % 3 == 0
    )


def _jax_qkv_weight_path(path: tuple[Any, ...]) -> bool:
    parts = _tree_path_parts(path)
    return len(parts) >= 2 and parts[-1] == "weight" and parts[-2] in {"qkv", "wqkv"}


def _jax_tree_get_path(tree: Any, path: tuple[Any, ...]) -> Any:
    value = tree
    for path_entry in path:
        value = value[_tree_path_key(path_entry)]
    return value


def _tree_path_parts(path: tuple[Any, ...]) -> tuple[str, ...]:
    return tuple(str(_tree_path_key(path_entry)) for path_entry in path)


def _tree_path_key(path_entry: Any) -> Any:
    return getattr(path_entry, "key", path_entry)


def numpy_batch_to_jax(
    batch: dict[str, Any],
    *,
    data_mesh: Mesh | None = None,
    batch_axis: int = 0,
) -> dict[str, Array]:
    if data_mesh is None:
        return {key: _numpy_array_to_jax(value) for key, value in batch.items()}
    return _put_batch_on_data_mesh(batch, data_mesh, batch_axis=batch_axis)


def _put_batch_on_data_mesh(
    batch: dict[str, Any],
    data_mesh: Mesh,
    *,
    batch_axis: int,
) -> dict[str, Array]:
    return {
        key: _put_batch_array_on_data_mesh(value, data_mesh, batch_axis=batch_axis)
        for key, value in batch.items()
    }


def _put_batch_array_on_data_mesh(
    value: Any,
    data_mesh: Mesh,
    *,
    batch_axis: int,
) -> Array:
    spec = P(
        *([None] * batch_axis),
        JAX_DATA_AXIS,
        *([None] * (value.ndim - batch_axis - 1)),
    )
    sharding = NamedSharding(data_mesh, spec)
    if jax.process_count() > 1:
        return jax.make_array_from_process_local_data(sharding, value)
    return jax.device_put(_numpy_array_to_jax(value), sharding)


def _numpy_array_to_jax(value: Any) -> Array:
    if isinstance(value, jax.Array):
        return value
    assert isinstance(value, np.ndarray | np.generic)
    return jnp.asarray(value)


def _replicate_tree_on_data_mesh(tree: Any, data_mesh: Mesh) -> Any:
    sharding = NamedSharding(data_mesh, P())

    def replicate(value: Any) -> Any:
        if not isinstance(value, jax.Array):
            return value
        if jax.process_count() > 1:
            return multihost_utils.host_local_array_to_global_array(
                np.asarray(value),
                data_mesh,
                P(),
            )
        return jax.device_put(value, sharding)

    return jax.tree.map(replicate, tree)


def _jax_data_mesh(config: Any | None = None) -> Mesh:
    device_count = (
        jax.device_count() if config is None else _jax_data_parallel_devices(config)
    )
    return _jax_data_mesh_for_device_count(device_count)


@cache
def _jax_data_mesh_for_device_count(device_count: int) -> Mesh:
    devices = (
        jax.devices()
        if device_count == jax.device_count()
        else jax.devices()[:device_count]
    )
    if jax.process_count() == 1:
        return jax.make_mesh((device_count,), (JAX_DATA_AXIS,), devices=devices)
    mesh_devices = _host_contiguous_mesh_devices(devices)
    return Mesh(mesh_devices.reshape((device_count,)), (JAX_DATA_AXIS,))


def _host_contiguous_mesh_devices(devices: list[Any]) -> np.ndarray:
    indexed_devices = enumerate(devices)
    return np.asarray(
        [
            device
            for _index, device in sorted(
                indexed_devices,
                key=lambda item: (item[1].process_index, item[0]),
            )
        ]
    )


@nnx.jit
def jax_grad_step(
    model: PeakSetJEPAJax,
    batch: dict[str, Array],
) -> tuple[tuple[Array, dict[str, Array]], nnx.State]:
    def loss_fn(model: PeakSetJEPAJax):
        metrics = model(batch)
        return metrics["loss"], metrics

    return nnx.value_and_grad(
        loss_fn,
        has_aux=True,
        argnums=nnx.DiffState(0, trainable_param_filter),
    )(model)


@nnx.jit
def jax_apply_grads(
    model: PeakSetJEPAJax,
    optimizer: nnx.Optimizer,
    grads: nnx.State,
) -> Array:
    optimizer.update(model, grads)
    return optimizer.step[...]


@nnx.jit
def jax_train_step(
    model: PeakSetJEPAJax,
    optimizer: nnx.Optimizer,
    batch: dict[str, Array],
) -> dict[str, Array]:
    def loss_fn(model: PeakSetJEPAJax):
        metrics = model(batch)
        return metrics["loss"], metrics

    (_loss, metrics), grads = nnx.value_and_grad(
        loss_fn,
        has_aux=True,
        argnums=nnx.DiffState(0, trainable_param_filter),
    )(model)
    optimizer.update(model, grads)
    return metrics


def jax_sharded_grad_step(
    model: PeakSetJEPAJax,
    batch: dict[str, Array],
) -> tuple[tuple[Array, dict[str, Array]], nnx.State]:
    return _jax_sharded_grad_step_fn(jax.device_count())(model, batch)


def jax_sharded_apply_grads(
    model: PeakSetJEPAJax,
    optimizer: nnx.Optimizer,
    grads: nnx.State,
) -> Array:
    return _jax_sharded_apply_grads_fn(jax.device_count())(model, optimizer, grads)


@cache
def _jax_sharded_grad_step_fn(device_count: int):
    data_mesh = _jax_data_mesh_for_device_count(device_count)

    @nnx.jit
    @nnx.shard_map(
        mesh=data_mesh,
        in_specs=(P(), P(JAX_DATA_AXIS)),
        out_specs=(P(), P()),
        axis_names={JAX_DATA_AXIS},
        check_vma=False,
    )
    def grad_step(
        model: PeakSetJEPAJax,
        batch: dict[str, Array],
    ) -> tuple[tuple[Array, dict[str, Array]], nnx.State]:
        def loss_fn(model: PeakSetJEPAJax):
            metrics = model(batch)
            return metrics["loss"], metrics

        (loss, metrics), grads = nnx.value_and_grad(
            loss_fn,
            has_aux=True,
            argnums=nnx.DiffState(0, trainable_param_filter),
        )(model)
        loss = jax.lax.pmean(loss, JAX_DATA_AXIS)
        metrics = jax.tree.map(
            lambda value: jax.lax.pmean(value, JAX_DATA_AXIS),
            metrics,
        )
        grads = jax.tree.map(lambda value: jax.lax.pmean(value, JAX_DATA_AXIS), grads)
        return (loss, metrics), grads

    return grad_step


@cache
def _jax_sharded_apply_grads_fn(device_count: int):
    data_mesh = _jax_data_mesh_for_device_count(device_count)

    @nnx.jit
    @nnx.shard_map(
        mesh=data_mesh,
        in_specs=(P(), P(), P()),
        out_specs=P(),
        axis_names={JAX_DATA_AXIS},
    )
    def apply_grads(
        model: PeakSetJEPAJax,
        optimizer: nnx.Optimizer,
        grads: nnx.State,
    ) -> Array:
        optimizer.update(model, grads)
        return optimizer.step[...]

    return apply_grads


def init_pure_optax_train_state(
    config: Any,
    model: PeakSetJEPAJax,
    *,
    total_steps: int | None = None,
) -> tuple[Any, nnx.State, nnx.State, Any, optax.GradientTransformation]:
    graphdef, trainable_params, static_state = nnx.split(
        model,
        trainable_param_filter,
        ...,
    )
    trainable_params = nnx.as_pure(trainable_params)
    static_state = nnx.as_pure(static_state)
    optimizer = build_jax_optax_transform(config, total_steps=total_steps)
    opt_state = optimizer.init(trainable_params)
    return graphdef, trainable_params, static_state, opt_state, optimizer


def make_pure_accumulated_train_step(
    graphdef: Any,
    optimizer: optax.GradientTransformation,
    *,
    sharded: bool,
    data_mesh: Mesh | None = None,
):
    def accumulated_metrics_and_grads(
        trainable_params: nnx.State,
        static_state: nnx.State,
        batch: dict[str, Array],
    ) -> tuple[dict[str, Array], nnx.State]:
        def loss_fn(params: nnx.State, micro_batch: dict[str, Array]):
            functional_model = nnx.merge(graphdef, params, static_state)
            metrics = functional_model(micro_batch)
            return metrics["loss"], metrics

        def micro_batch_grad(micro_batch: dict[str, Array]):
            return jax.value_and_grad(loss_fn, has_aux=True)(
                trainable_params,
                micro_batch,
            )

        def scan_body(carry: tuple[nnx.State, dict[str, Array]], micro_batch):
            grad_accumulator, metric_accumulator = carry
            (_micro_loss, micro_metrics), micro_grads = micro_batch_grad(micro_batch)
            grad_accumulator = jax.tree.map(
                lambda lhs, rhs: lhs + rhs,
                grad_accumulator,
                micro_grads,
            )
            metric_accumulator = jax.tree.map(
                lambda lhs, rhs: lhs + rhs,
                metric_accumulator,
                micro_metrics,
            )
            return (grad_accumulator, metric_accumulator), None

        first_batch = jax.tree.map(lambda value: value[0], batch)
        remaining_batches = jax.tree.map(lambda value: value[1:], batch)
        (_loss, metrics), grads = micro_batch_grad(first_batch)

        (grads, metrics), _ = jax.lax.scan(
            scan_body,
            (grads, metrics),
            remaining_batches,
        )
        num_micro_batches = float(jax.tree.leaves(batch)[0].shape[0])
        grads = jax.tree.map(lambda value: value / num_micro_batches, grads)
        metrics = jax.tree.map(lambda value: value / num_micro_batches, metrics)
        return metrics, grads

    if sharded:
        data_mesh = _jax_data_mesh() if data_mesh is None else data_mesh

        @jax.jit(donate_argnums=(0, 2))
        @jax.shard_map(
            mesh=data_mesh,
            in_specs=(P(), P(), P(), P(None, JAX_DATA_AXIS)),
            out_specs=(P(), P(), P()),
            axis_names={JAX_DATA_AXIS},
            check_vma=False,
        )
        def pure_sharded_accumulated_train_step(
            trainable_params: nnx.State,
            static_state: nnx.State,
            opt_state: Any,
            batch: dict[str, Array],
        ) -> tuple[nnx.State, Any, dict[str, Array]]:
            metrics, grads = accumulated_metrics_and_grads(
                trainable_params,
                static_state,
                batch,
            )
            metrics = jax.tree.map(
                lambda value: jax.lax.pmean(value, JAX_DATA_AXIS),
                metrics,
            )
            grads = jax.tree.map(
                lambda value: jax.lax.pmean(value, JAX_DATA_AXIS),
                grads,
            )
            updates, opt_state = optimizer.update(
                grads,
                opt_state,
                trainable_params,
            )
            trainable_params = optax.apply_updates(trainable_params, updates)
            return trainable_params, opt_state, metrics

        return pure_sharded_accumulated_train_step

    @jax.jit(donate_argnums=(0, 2))
    def pure_accumulated_train_step(
        trainable_params: nnx.State,
        static_state: nnx.State,
        opt_state: Any,
        batch: dict[str, Array],
    ) -> tuple[nnx.State, Any, dict[str, Array]]:
        metrics, grads = accumulated_metrics_and_grads(
            trainable_params,
            static_state,
            batch,
        )
        updates, opt_state = optimizer.update(
            grads,
            opt_state,
            trainable_params,
        )
        trainable_params = optax.apply_updates(trainable_params, updates)
        return trainable_params, opt_state, metrics

    return pure_accumulated_train_step


def make_pure_eval_step(
    graphdef: Any,
    *,
    sharded: bool,
    data_mesh: Mesh | None = None,
):
    if sharded:
        data_mesh = _jax_data_mesh() if data_mesh is None else data_mesh

        @jax.jit
        @jax.shard_map(
            mesh=data_mesh,
            in_specs=(P(), P(), P(JAX_DATA_AXIS)),
            out_specs=P(),
            axis_names={JAX_DATA_AXIS},
            check_vma=False,
        )
        def pure_sharded_eval_step(
            trainable_params: nnx.State,
            static_state: nnx.State,
            batch: dict[str, Array],
        ) -> dict[str, Array]:
            functional_model = nnx.merge(graphdef, trainable_params, static_state)
            metrics = functional_model(batch)
            return jax.tree.map(
                lambda value: jax.lax.pmean(value, JAX_DATA_AXIS),
                metrics,
            )

        return pure_sharded_eval_step

    @jax.jit
    def pure_eval_step(
        trainable_params: nnx.State,
        static_state: nnx.State,
        batch: dict[str, Array],
    ) -> dict[str, Array]:
        functional_model = nnx.merge(graphdef, trainable_params, static_state)
        return functional_model(batch)

    return pure_eval_step


def train_and_evaluate_jax(
    config: config_dict.ConfigDict,
    workdir: str | Path,
) -> dict[str, object]:
    configure_jax_runtime(config)
    initialize_jax_distributed(config)
    workdir = normalize_storage_path(workdir)
    local_workdir = local_scratch_dir(workdir)
    local_workdir.mkdir(parents=True, exist_ok=True)
    is_main_process = jax.process_index() == 0
    if is_main_process:
        storage_mkdir(workdir)
    multihost_utils.sync_global_devices("spectra_learning_jax_workdir_ready")
    torch.manual_seed(int(config.seed))
    config.dataloader_pin_memory = False
    config.dataloader_persistent_workers = False
    config.dataloader_output_format = "numpy"
    if int(_config_get(config, "dataloader_num_workers", 0)) > 0:
        config.dataloader_multiprocessing_context = str(
            _config_get(config, "dataloader_multiprocessing_context", "forkserver")
            or "forkserver"
        )
    datamodule = GemsNativeDataModule(
        config,
        seed=int(config.seed),
        distributed_world_size=jax.process_count(),
        distributed_rank=jax.process_index(),
        distributed_local_rank=0,
    )
    total_steps = _total_training_steps(config, datamodule)
    model = build_model_from_config(config)
    checkpoint_dir = storage_join(workdir, "checkpoints")
    if is_main_process:
        storage_mkdir(checkpoint_dir)
    multihost_utils.sync_global_devices("spectra_learning_jax_checkpoint_dir_ready")
    jax_checkpoint_max_to_keep = _config_get(config, "jax_checkpoint_max_to_keep", 5)
    if jax_checkpoint_max_to_keep is not None:
        jax_checkpoint_max_to_keep = int(jax_checkpoint_max_to_keep)
    checkpoint_manager = build_jax_checkpoint_manager(
        checkpoint_dir,
        max_to_keep=jax_checkpoint_max_to_keep,
        enable_async_checkpointing=bool(
            _config_get(config, "jax_enable_async_checkpointing", True)
        ),
    )
    resume_step = checkpoint_manager.latest_step()
    if resume_step is None:
        initialize_jax_model_from_torch_seed(config, model)
    logger = build_logger(config, local_workdir) if is_main_process else MetricLogger()
    metrics = _run_jax_training_loop(
        config=config,
        datamodule=datamodule,
        model=model,
        logger=logger,
        total_steps=total_steps,
        checkpoint_manager=checkpoint_manager,
        resume_step=resume_step,
    )
    checkpoint_manager.close()
    data_parallel_devices = _jax_data_parallel_devices(config)
    run_metrics = {
        "run/world_size": float(jax.process_count()),
        "run/jax_device_count": float(jax.device_count()),
        "run/jax_process_index": float(jax.process_index()),
        "run/jax_process_count": float(jax.process_count()),
        "run/jax_data_parallel_devices": float(data_parallel_devices),
        "run/global_batch_size": float(datamodule.global_batch_size),
        "run/local_batch_size": float(datamodule.batch_size),
        "run/device_microbatch_size": float(
            datamodule.batch_size // data_parallel_devices
            if jax.process_count() == 1
            else datamodule.batch_size // jax.local_device_count()
        ),
        "run/gradient_accumulation_steps": float(
            int(_config_get(config, "gradient_accumulation_steps", 1))
        ),
        "run/device_backend": "jax",
    }
    results = {**metrics, **run_metrics}
    if is_main_process:
        final_global_step = int(metrics["run/final_global_step"])
        logger.log_metrics(
            {"global_step": float(final_global_step), **results},
            step=final_global_step,
        )
    return results


def initialize_jax_model_from_torch_seed(
    config: config_dict.ConfigDict,
    model: PeakSetJEPAJax,
) -> None:
    from spectra_learning.models.factory import (
        build_model_from_config as build_torch_model_from_config,
    )

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(config.seed))
        torch_model = build_torch_model_from_config(config)
    model.load_torch_state_dict(torch_model.state_dict())


def _run_jax_training_loop(
    *,
    config: config_dict.ConfigDict,
    datamodule: GemsNativeDataModule,
    model: PeakSetJEPAJax,
    logger: MetricLogger,
    total_steps: int,
    checkpoint_manager: Any,
    resume_step: int | None,
) -> dict[str, object]:
    log_every_n_steps = int(_config_get(config, "log_every_n_steps", 50))
    warmup_steps = int(_config_get(config, "throughput_warmup_steps", 0))
    grad_accum_steps = int(_config_get(config, "gradient_accumulation_steps", 1))
    if model.use_ema_teacher:
        raise ValueError("JAX training uses pure Optax and does not support EMA teachers.")
    data_parallel_devices = _jax_data_parallel_devices(config)
    data_mesh = _jax_data_mesh_for_device_count(data_parallel_devices)
    use_sharded_step = data_parallel_devices > 1
    pure_graphdef = None
    pure_full_graphdef = None
    pure_trainable_params = None
    pure_static_state = None
    pure_opt_state = None
    pure_train_step = None
    pure_full_static_state = None
    pure_full_train_step = None
    pure_pack_train_steps = []
    context_encoder_pack_tokens = int(
        getattr(model, "mae_context_encoder_pack_tokens", 0)
    )
    context_encoder_pack_choices = _context_encoder_pack_choices(
        config,
        context_encoder_pack_tokens,
    )
    if context_encoder_pack_choices:
        model.mae_context_encoder_pack_tokens = 0
        (
            pure_full_graphdef,
            _full_trainable_params,
            pure_full_static_state,
            _full_opt_state,
            _full_optimizer,
        ) = init_pure_optax_train_state(
            config,
            model,
            total_steps=total_steps,
        )
        model.mae_context_encoder_pack_tokens = context_encoder_pack_tokens
        pure_optimizer = None
        pack_train_states = []
        for pack_tokens in context_encoder_pack_choices:
            model.mae_context_encoder_pack_tokens = pack_tokens
            (
                pack_graphdef,
                pack_trainable_params,
                pack_static_state,
                pack_opt_state,
                pack_optimizer,
            ) = init_pure_optax_train_state(
                config,
                model,
                total_steps=total_steps,
            )
            if pure_trainable_params is None:
                pure_trainable_params = pack_trainable_params
                pure_static_state = pack_static_state
                pure_opt_state = pack_opt_state
                pure_optimizer = pack_optimizer
            pack_train_states.append(
                (pack_tokens, pack_static_state, pack_graphdef)
            )
        model.mae_context_encoder_pack_tokens = context_encoder_pack_tokens
        pure_pack_train_steps = [
            (
                pack_tokens,
                pack_static_state,
                make_pure_accumulated_train_step(
                    pack_graphdef,
                    pure_optimizer,
                    sharded=use_sharded_step,
                    data_mesh=data_mesh,
                ),
            )
            for pack_tokens, pack_static_state, pack_graphdef in pack_train_states
        ]
    else:
        (
            pure_graphdef,
            pure_trainable_params,
            pure_static_state,
            pure_opt_state,
            pure_optimizer,
        ) = init_pure_optax_train_state(
            config,
            model,
            total_steps=total_steps,
        )
    if not pure_pack_train_steps:
        pure_train_step = make_pure_accumulated_train_step(
            pure_graphdef,
            pure_optimizer,
            sharded=use_sharded_step,
            data_mesh=data_mesh,
        )
    if context_encoder_pack_choices:
        pure_full_train_step = make_pure_accumulated_train_step(
            pure_full_graphdef,
            pure_optimizer,
            sharded=use_sharded_step,
            data_mesh=data_mesh,
        )
    eval_graphdef = (
        pure_full_graphdef if pure_full_graphdef is not None else pure_graphdef
    )
    pure_eval_step = make_pure_eval_step(
        eval_graphdef,
        sharded=use_sharded_step,
        data_mesh=data_mesh,
    )
    if use_sharded_step:
        # Commit the training state to the data mesh once so every training
        # step uses the same input-sharding signature.
        pure_trainable_params = _replicate_tree_on_data_mesh(
            pure_trainable_params,
            data_mesh,
        )
        pure_opt_state = _replicate_tree_on_data_mesh(pure_opt_state, data_mesh)
        pure_static_state = _replicate_tree_on_data_mesh(
            pure_static_state,
            data_mesh,
        )
        if pure_full_static_state is not None:
            pure_full_static_state = _replicate_tree_on_data_mesh(
                pure_full_static_state,
                data_mesh,
            )
        pure_pack_train_steps = [
            (
                pack_tokens,
                _replicate_tree_on_data_mesh(pack_static_state, data_mesh),
                pack_train_step,
            )
            for pack_tokens, pack_static_state, pack_train_step in (
                pure_pack_train_steps
            )
        ]
    checkpoint_every_steps = int(_config_get(config, "checkpoint_every_steps", 0))
    val_every_n_steps = validation_interval(config, datamodule, total_steps)
    val_num_steps = validation_steps(config)
    msg_probe_every_n_steps = msg_probe_interval(config, datamodule, total_steps)
    msg_probe_variants = msg_probe_variants_from_config(config)

    def jax_checkpoint_state() -> dict[str, Any]:
        return {
            "trainable_params": pure_trainable_params,
            "static_state": pure_static_state,
            "opt_state": pure_opt_state,
        }

    def save_checkpoint(step: int) -> None:
        save_jax_training_state(checkpoint_manager, step, jax_checkpoint_state())

    def maybe_save_checkpoint(step: int) -> None:
        if checkpoint_every_steps <= 0 or step % checkpoint_every_steps != 0:
            return
        save_checkpoint(step)

    start_step = 0
    if resume_step is not None:
        restored = restore_jax_training_state(
            checkpoint_manager,
            int(resume_step),
            jax_checkpoint_state(),
        )
        pure_trainable_params = restored["trainable_params"]
        pure_static_state = restored["static_state"]
        pure_opt_state = restored["opt_state"]
        # Pack variants only differ in graphdef; their static states are
        # numerically identical, so they all share the restored tree.
        if pure_full_static_state is not None:
            pure_full_static_state = pure_static_state
        pure_pack_train_steps = [
            (pack_tokens, pure_static_state, pack_train_step)
            for pack_tokens, _pack_static_state, pack_train_step in (
                pure_pack_train_steps
            )
        ]
        start_step = int(resume_step)
    timing_barriers = bool(_config_get(config, "jax_timing_barriers", False))
    compile_stall_threshold_seconds = float(
        _config_get(config, "jax_compile_stall_threshold_seconds", 0.0)
    )
    profile_dir = str(_config_get(config, "jax_profile_dir", ""))
    profile_start_step = int(_config_get(config, "jax_profile_start_step", warmup_steps))
    profile_steps = int(_config_get(config, "jax_profile_steps", 0))
    profile_end_step = (
        profile_start_step + profile_steps if profile_steps > 0 else total_steps
    )
    profile_started = False
    profile_active = False
    timing = {
        "dataloader_seconds": 0.0,
        "transfer_seconds": 0.0,
        "grad_seconds": 0.0,
        "accumulate_seconds": 0.0,
        "apply_seconds": 0.0,
        "compiled_step_seconds": 0.0,
        "measured_microbatches": 0.0,
    }
    non_train_timing = {
        "checkpoint_seconds": 0.0,
        "validation_seconds": 0.0,
        "msg_probe_seconds": 0.0,
        "model_update_seconds": 0.0,
        "profile_seconds": 0.0,
    }
    measured_non_train_timing = {name: 0.0 for name in non_train_timing}

    def add_non_train_timing(
        name: str,
        elapsed: float,
        *,
        include_measured: bool = True,
    ) -> None:
        non_train_timing[name] += elapsed
        if include_measured and measured_start is not None:
            measured_non_train_timing[name] += elapsed

    loop_epochs = max(1, math.ceil(float(config.num_epochs)))
    global_step = start_step
    start_epoch = min(start_step // datamodule.train_steps, loop_epochs - 1)
    last_metrics: dict[str, Array] = {}
    pending_train_metrics: _StagedJaxTrainMetrics | None = None
    last_validation_metrics: dict[str, float] = {}
    last_msg_probe_metrics: dict[str, float] = {}
    train_start = time.perf_counter()
    measured_start: float | None = None
    measured_steps = 0
    context_encoder_packed_steps = 0
    context_encoder_full_fallback_steps = 0
    measured_context_encoder_packed_steps = 0
    measured_context_encoder_full_fallback_steps = 0
    context_encoder_pack_steps_by_size = {
        pack: 0 for pack in context_encoder_pack_choices
    }
    measured_context_encoder_pack_steps_by_size = {
        pack: 0 for pack in context_encoder_pack_choices
    }
    for epoch in range(start_epoch, loop_epochs):
        epoch_start_batch = (
            global_step - epoch * datamodule.train_steps if epoch == start_epoch else 0
        )
        loader = datamodule.train_loader_for_epoch(epoch, start_batch=epoch_start_batch)
        loader_iter = iter(loader)
        pbar = tqdm(
            total=min(
                datamodule.train_steps - epoch_start_batch,
                total_steps - global_step,
            ),
            desc=f"Epoch {epoch}",
            unit="step",
            disable=jax.process_index() != 0,
        )
        while global_step < total_steps:
            dataloader_elapsed = 0.0
            transfer_elapsed = 0.0
            max_context_count = 0
            micro_batches = []
            for _ in range(grad_accum_steps):
                dataloader_start = time.perf_counter()
                try:
                    torch_batch = next(loader_iter)
                except StopIteration:
                    break
                dataloader_elapsed += time.perf_counter() - dataloader_start
                if pure_full_train_step is not None:
                    active_context = (
                        torch_batch["context_mask"] & torch_batch["peak_valid_mask"]
                    )
                    if isinstance(active_context, np.ndarray):
                        context_count = int(active_context.sum(axis=1).max())
                    else:
                        context_count = int(active_context.sum(dim=1).max().item())
                    max_context_count = max(max_context_count, int(context_count))
                transfer_start = time.perf_counter()
                micro_batches.append(torch_batch)
                transfer_elapsed += time.perf_counter() - transfer_start
            if len(micro_batches) < grad_accum_steps:
                break
            batch = numpy_batch_to_jax(
                _stack_micro_batches(micro_batches),
                data_mesh=data_mesh if use_sharded_step else None,
                batch_axis=1,
            )
            if timing_barriers:
                jax.block_until_ready(batch)
            timing_enabled = measured_start is not None
            if timing_enabled:
                timing["dataloader_seconds"] += dataloader_elapsed
                timing["transfer_seconds"] += transfer_elapsed
                timing["measured_microbatches"] += float(grad_accum_steps)
            if measured_start is None and global_step >= warmup_steps:
                jax.effects_barrier()
                measured_start = time.perf_counter()
                timing_enabled = True
            if (
                profile_dir
                and not profile_started
                and global_step >= profile_start_step
            ):
                phase_start = time.perf_counter()
                jax.effects_barrier()
                jax.profiler.start_trace(profile_dir)
                add_non_train_timing(
                    "profile_seconds",
                    time.perf_counter() - phase_start,
                )
                profile_started = True
                profile_active = True

            if pure_pack_train_steps:
                max_context_count = _process_global_max_int(max_context_count)
            if pure_pack_train_steps:
                selected_train_step = pure_full_train_step
                selected_static_state = pure_full_static_state
                used_context_full_fallback = True
                selected_pack_tokens = 0
                for (
                    pack_tokens,
                    pack_static_state,
                    pack_train_step,
                ) in pure_pack_train_steps:
                    if max_context_count <= pack_tokens:
                        selected_train_step = pack_train_step
                        selected_static_state = pack_static_state
                        selected_pack_tokens = pack_tokens
                        used_context_full_fallback = False
                        break
            else:
                selected_train_step = pure_train_step
                selected_static_state = pure_static_state
                used_context_full_fallback = False
                selected_pack_tokens = 0
            if pure_full_train_step is not None:
                if used_context_full_fallback:
                    context_encoder_full_fallback_steps += 1
                    if timing_enabled:
                        measured_context_encoder_full_fallback_steps += 1
                else:
                    context_encoder_packed_steps += 1
                    if selected_pack_tokens:
                        context_encoder_pack_steps_by_size[selected_pack_tokens] += 1
                    if timing_enabled:
                        measured_context_encoder_packed_steps += 1
                        if selected_pack_tokens:
                            measured_context_encoder_pack_steps_by_size[
                                selected_pack_tokens
                            ] += 1

            step_start = time.perf_counter()
            pure_trainable_params, pure_opt_state, metrics = selected_train_step(
                pure_trainable_params,
                selected_static_state,
                pure_opt_state,
                batch,
            )
            if timing_barriers:
                jax.block_until_ready((pure_trainable_params, pure_opt_state, metrics))
            step_elapsed = time.perf_counter() - step_start
            _raise_on_jax_compile_stall(
                step_elapsed,
                threshold_seconds=compile_stall_threshold_seconds,
                global_step=global_step,
                branch="pure_optax_scan",
            )
            if timing_enabled:
                timing["compiled_step_seconds"] += step_elapsed
            last_metrics = metrics
            global_step += 1
            if measured_start is not None:
                measured_steps += 1
            pbar.update(1)
            _log_jax_train_metrics(config, logger, pbar, pending_train_metrics)
            pending_train_metrics = None
            if _should_log_jax_train_metrics(global_step, log_every_n_steps):
                pending_train_metrics = _stage_jax_train_metrics(
                    metrics,
                    epoch=epoch,
                    global_step=global_step,
                    total_steps=total_steps,
                )
            phase_start = time.perf_counter()
            maybe_save_checkpoint(global_step)
            add_non_train_timing(
                "checkpoint_seconds",
                time.perf_counter() - phase_start,
            )
            if should_run_at_step(val_every_n_steps, global_step):
                phase_start = time.perf_counter()
                eval_static_state = (
                    pure_full_static_state
                    if pure_full_static_state is not None
                    else pure_static_state
                )
                last_validation_metrics = _evaluate_jax_validation_loss(
                    datamodule=datamodule,
                    trainable_params=pure_trainable_params,
                    static_state=eval_static_state,
                    eval_step=pure_eval_step,
                    max_steps=val_num_steps,
                    use_sharded_step=use_sharded_step,
                    data_mesh=data_mesh,
                )
                _log_jax_validation_metrics(
                    logger,
                    pbar,
                    last_validation_metrics,
                    global_step=global_step,
                )
                add_non_train_timing(
                    "validation_seconds",
                    time.perf_counter() - phase_start,
                )
            if should_run_at_step_or_final(
                msg_probe_every_n_steps,
                global_step,
                total_steps=total_steps,
                run_at_final_step=bool(
                    _config_get(config, "msg_probe_at_final_step", False)
                ),
            ):
                phase_start = time.perf_counter()
                last_msg_probe_metrics = _run_distributed_msg_probe_jax(
                    config=config,
                    model=model,
                    logger=logger,
                    variants=msg_probe_variants,
                    global_step=global_step,
                    trainable_params=pure_trainable_params,
                    data_mesh=data_mesh,
                )
                add_non_train_timing(
                    "msg_probe_seconds",
                    time.perf_counter() - phase_start,
                )
            if profile_active and global_step >= profile_end_step:
                phase_start = time.perf_counter()
                jax.effects_barrier()
                jax.profiler.stop_trace()
                add_non_train_timing(
                    "profile_seconds",
                    time.perf_counter() - phase_start,
                )
                profile_active = False
        if pending_train_metrics is not None and (
            global_step >= total_steps or epoch == loop_epochs - 1
        ):
            _log_jax_train_metrics(config, logger, pbar, pending_train_metrics)
            pending_train_metrics = None
        pbar.close()
        _shutdown_torch_loader_iterator(loader_iter)
        del loader_iter, loader
        if global_step >= total_steps:
            break
    jax.block_until_ready(pure_trainable_params)
    phase_start = time.perf_counter()
    nnx.update(model, pure_trainable_params)
    jax.effects_barrier()
    add_non_train_timing(
        "model_update_seconds",
        time.perf_counter() - phase_start,
    )
    post_model_update_time = time.perf_counter()
    if profile_active:
        phase_start = time.perf_counter()
        jax.profiler.stop_trace()
        add_non_train_timing(
            "profile_seconds",
            time.perf_counter() - phase_start,
        )
    measured_wall_elapsed = (
        post_model_update_time - measured_start if measured_start is not None else 0.0
    )
    measured_non_train_elapsed = sum(measured_non_train_timing.values())
    measured_train_elapsed = max(measured_wall_elapsed - measured_non_train_elapsed, 0.0)
    if global_step > start_step and checkpoint_manager.latest_step() != global_step:
        phase_start = time.perf_counter()
        save_checkpoint(global_step)
        add_non_train_timing(
            "checkpoint_seconds",
            time.perf_counter() - phase_start,
            include_measured=False,
        )
    checkpoint_manager.wait_until_finished()
    wall_elapsed = time.perf_counter() - train_start
    train_elapsed = max(
        wall_elapsed
        - sum(non_train_timing.values()),
        0.0,
    )
    global_batch_size = int(datamodule.global_batch_size)
    train_metrics: dict[str, float] = {"train/loss": float("nan")}
    if last_metrics:
        host_train_metrics = _jax_metrics_to_host(last_metrics, prefix="train/")
        if jax.process_index() == 0:
            train_metrics = host_train_metrics
    result = {
        "run/final_global_step": float(global_step),
        "run/wall_elapsed_seconds": wall_elapsed,
        "run/train_elapsed_seconds": train_elapsed,
        "run/non_train_elapsed_seconds": sum(non_train_timing.values()),
        "run/steps_per_second": (
            float(global_step) / train_elapsed if train_elapsed > 0 else 0.0
        ),
        "run/samples_per_second": (
            float(global_step) * global_batch_size / train_elapsed
            if train_elapsed > 0
            else 0.0
        ),
        "run/wall_steps_per_second": (
            float(global_step) / wall_elapsed if wall_elapsed > 0 else 0.0
        ),
        "run/wall_samples_per_second": (
            float(global_step) * global_batch_size / wall_elapsed
            if wall_elapsed > 0
            else 0.0
        ),
        "run/measured_steps": float(measured_steps),
        "run/measured_wall_elapsed_seconds": measured_wall_elapsed,
        "run/measured_non_train_elapsed_seconds": measured_non_train_elapsed,
        "run/measured_elapsed_seconds": measured_train_elapsed,
        "run/measured_steps_per_second": (
            float(measured_steps) / measured_train_elapsed
            if measured_train_elapsed > 0
            else 0.0
        ),
        "run/measured_samples_per_second": (
            float(measured_steps) * global_batch_size / measured_train_elapsed
            if measured_train_elapsed > 0
            else 0.0
        ),
        "run/measured_wall_steps_per_second": (
            float(measured_steps) / measured_wall_elapsed
            if measured_wall_elapsed > 0
            else 0.0
        ),
        "run/measured_wall_samples_per_second": (
            float(measured_steps) * global_batch_size / measured_wall_elapsed
            if measured_wall_elapsed > 0
            else 0.0
        ),
        **train_metrics,
    }
    for name, value in non_train_timing.items():
        result[f"run/{name}"] = value
    result.update(last_validation_metrics)
    result.update(last_msg_probe_metrics)
    if context_encoder_pack_choices:
        result.update(
            {
                "run/context_encoder_pack_tokens": float(context_encoder_pack_tokens),
                "run/context_encoder_packed_steps": float(context_encoder_packed_steps),
                "run/context_encoder_full_fallback_steps": float(
                    context_encoder_full_fallback_steps
                ),
                "run/measured_context_encoder_packed_steps": float(
                    measured_context_encoder_packed_steps
                ),
                "run/measured_context_encoder_full_fallback_steps": float(
                    measured_context_encoder_full_fallback_steps
                ),
            }
        )
        for pack_tokens, count in context_encoder_pack_steps_by_size.items():
            result[f"run/context_encoder_pack_{pack_tokens}_steps"] = float(count)
            result[f"run/measured_context_encoder_pack_{pack_tokens}_steps"] = float(
                measured_context_encoder_pack_steps_by_size[pack_tokens]
            )
    for name, value in timing.items():
        result[f"run/profile_{name}"] = value
    if timing["measured_microbatches"] > 0:
        for name in (
            "dataloader_seconds",
            "transfer_seconds",
            "grad_seconds",
            "accumulate_seconds",
            "apply_seconds",
            "compiled_step_seconds",
        ):
            result[f"run/profile_{name}_per_microbatch"] = (
                timing[name] / timing["measured_microbatches"]
            )
    if measured_steps > 0:
        for name in (
            "dataloader_seconds",
            "transfer_seconds",
            "grad_seconds",
            "accumulate_seconds",
            "apply_seconds",
            "compiled_step_seconds",
        ):
            result[f"run/profile_{name}_per_step"] = timing[name] / measured_steps
    return result


def _log_jax_train_metrics(
    config: config_dict.ConfigDict,
    logger: MetricLogger,
    pbar: tqdm,
    staged: _StagedJaxTrainMetrics | None,
) -> None:
    if staged is None:
        return
    host_metrics = _jax_metrics_to_host(staged.metrics, prefix="train/")
    if jax.process_index() != 0:
        return
    pbar.set_postfix(
        loss=f"{host_metrics['train/loss']:.4f}",
        step=staged.global_step,
    )
    host_metrics["train/learning_rate"] = _scheduled_jax_learning_rate(
        config,
        global_step=staged.global_step,
        total_steps=staged.total_steps,
    )
    host_metrics["epoch"] = float(staged.epoch)
    host_metrics["global_step"] = float(staged.global_step)
    logger.log_metrics(host_metrics, step=staged.global_step)


def _should_log_jax_train_metrics(global_step: int, every_n_steps: int) -> bool:
    return every_n_steps > 0 and global_step % every_n_steps == 0


def _stage_jax_train_metrics(
    metrics: dict[str, Array],
    *,
    epoch: int,
    global_step: int,
    total_steps: int,
) -> _StagedJaxTrainMetrics:
    return _StagedJaxTrainMetrics(
        metrics=jax.copy_to_host_async(metrics),
        epoch=epoch,
        global_step=global_step,
        total_steps=total_steps,
    )


def _jax_metrics_to_host(
    metrics: dict[str, Array],
    *,
    prefix: str = "",
) -> dict[str, float]:
    return {
        f"{prefix}{key}": float(np.asarray(value))
        for key, value in jax.device_get(metrics).items()
    }


def _evaluate_jax_validation_loss(
    *,
    datamodule: GemsNativeDataModule,
    trainable_params: nnx.State,
    static_state: nnx.State,
    eval_step: Any,
    max_steps: int,
    use_sharded_step: bool,
    data_mesh: Mesh,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    steps = 0
    for torch_batch in datamodule.val_loader_for_eval(augment=True):
        if steps >= max_steps:
            break
        batch = numpy_batch_to_jax(
            torch_batch,
            data_mesh=data_mesh if use_sharded_step else None,
            batch_axis=0,
        )
        metrics = jax.device_get(eval_step(trainable_params, static_state, batch))
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(np.asarray(value))
        steps += 1
    return {f"val/{key}": value / float(steps) for key, value in totals.items()}


def _log_jax_validation_metrics(
    logger: MetricLogger,
    pbar: tqdm,
    metrics: dict[str, float],
    *,
    global_step: int,
) -> None:
    if jax.process_index() != 0:
        return
    pbar.set_postfix(val_loss=f"{metrics['val/loss']:.4f}", step=global_step)
    logger.log_metrics(
        {
            "global_step": float(global_step),
            **metrics,
        },
        step=global_step,
    )


def run_and_log_msg_probe_jax(
    *,
    config: config_dict.ConfigDict,
    model: PeakSetJEPAJax,
    logger: MetricLogger,
    variants: tuple[str, ...],
    global_step: int,
    data_mesh: Mesh | None = None,
) -> dict[str, float]:
    probe_data_mesh = (
        data_mesh
        if bool(_config_get(config, "jax_msg_probe_shard_batches", False))
        else None
    )
    probe_metrics = run_msg_probe_jax(
        config=config,
        model=model,
        data_mesh=probe_data_mesh,
        online_maccs_only=True,
    )
    if jax.process_index() != 0:
        return probe_metrics
    log_msg_probe_metrics(
        logger,
        probe_metrics,
        global_step,
        enable_wandb=bool(_config_get(config, "enable_wandb", False)),
    )
    fingerprint_task = "maccs"
    for variant in variants:
        prefix = f"msg_probe/{variant}"
        epoch_key = f"{prefix}/epoch"
        if epoch_key in probe_metrics:
            logging.info(
                "step=%d msg_probe[%s] best_epoch=%.2f test_auc_%s_mean=%.4f",
                global_step,
                variant,
                probe_metrics[epoch_key],
                fingerprint_task,
                probe_metrics[f"{prefix}/test/auc_{fingerprint_task}_mean"],
            )
    return probe_metrics


def _run_distributed_msg_probe_jax(
    *,
    config: config_dict.ConfigDict,
    model: PeakSetJEPAJax,
    logger: MetricLogger,
    variants: tuple[str, ...],
    global_step: int,
    trainable_params: Any,
    data_mesh: Mesh,
) -> dict[str, float]:
    nnx.update(model, trainable_params)
    with jax.set_mesh(data_mesh):
        probe_metrics = run_and_log_msg_probe_jax(
            config=config,
            model=model,
            logger=logger,
            variants=variants,
            global_step=global_step,
            data_mesh=data_mesh,
        )
    multihost_utils.sync_global_devices(
        f"spectra_learning_jax_msg_probe_{global_step}"
    )
    return probe_metrics if jax.process_index() == 0 else {}


def _raise_on_jax_compile_stall(
    elapsed_seconds: float,
    *,
    threshold_seconds: float,
    global_step: int,
    branch: str,
) -> None:
    if threshold_seconds <= 0.0 or elapsed_seconds <= threshold_seconds:
        return
    raise RuntimeError(
        "JAX step exceeded compile stall threshold: "
        f"step={global_step} branch={branch} elapsed={elapsed_seconds:.2f}s "
        f"threshold={threshold_seconds:.2f}s. Enable jax_log_compiles and "
        "jax_explain_cache_misses to inspect the recompilation key."
    )


def _limit_context_count(
    batch: dict[str, Array],
    max_context_count: int,
) -> dict[str, Array]:
    peak_valid_mask = batch["peak_valid_mask"]
    valid_rank = jnp.cumsum(peak_valid_mask.astype(jnp.int32), axis=-1)
    return {
        **batch,
        "context_mask": peak_valid_mask & (valid_rank <= max_context_count),
    }


def _total_training_steps(
    config: config_dict.ConfigDict,
    datamodule: GemsNativeDataModule,
) -> int:
    total_steps = max(1, int(float(config.num_epochs) * datamodule.train_steps))
    training_max_steps = _config_get(config, "training_max_steps", None)
    if training_max_steps is None:
        return total_steps
    return min(total_steps, max(1, int(training_max_steps)))


def _jax_data_parallel_devices(config: Any) -> int:
    requested = _config_get(config, "jax_mesh_devices", None)
    if requested is None:
        return jax.device_count()
    if isinstance(requested, str):
        return jax.device_count() if requested.lower() == "all" else int(requested)
    return int(requested)


def _stack_micro_batches(batches: list[dict[str, Any]]) -> dict[str, Any]:
    return jax.tree.map(_stack_micro_batch_values, *batches)


def _stack_micro_batch_values(*values: Any) -> Any:
    first = values[0]
    if isinstance(first, np.ndarray):
        return np.stack(values)
    return jnp.stack(values)


def _process_global_max_int(value: int) -> int:
    if jax.process_count() <= 1:
        return int(value)
    gathered = multihost_utils.process_allgather(np.asarray(value, dtype=np.int32))
    return int(np.asarray(gathered).max())


def _shutdown_torch_loader_iterator(loader_iter: Any) -> None:
    shutdown_workers = getattr(loader_iter, "_shutdown_workers", None)
    if callable(shutdown_workers):
        shutdown_workers()


def _context_encoder_pack_choices(config: Any, default_pack_tokens: int) -> tuple[int, ...]:
    raw_choices = _config_get(config, "mae_context_encoder_pack_token_choices", ())
    if isinstance(raw_choices, str):
        choices = [int(value) for value in raw_choices.split(",") if value]
    else:
        choices = [int(value) for value in raw_choices]
    if default_pack_tokens > 0:
        choices.append(default_pack_tokens)
    return tuple(sorted({choice for choice in choices if choice > 0}))


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def _config_get(config: Any, key: str, default: Any) -> Any:
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _config_or_env(
    config: Any,
    key: str,
    env_keys: tuple[str, ...],
    default: Any = "",
) -> Any:
    value = _config_get(config, key, None)
    if value not in (None, ""):
        return value
    for env_key in env_keys:
        value = os.environ.get(env_key)
        if value not in (None, ""):
            return value
    return default


def _optional_int(value: Any) -> int | None:
    if value in (None, "", "none", "None"):
        return None
    return int(value)


def _local_device_ids(value: Any) -> tuple[int, ...] | int | None:
    if value in (None, "", "none", "None"):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return tuple(int(part) for part in value.split(",") if part)
    return tuple(int(part) for part in value)
