from __future__ import annotations

import gc
import logging
import math
import os
import time
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Callable, Literal

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

from spectra_learning.config import config_to_dict
from spectra_learning.data.gems.artifacts import MASSIVE_V2_HDF5_FORMAT
from spectra_learning.data.gems.datamodule import GemsDataModule
from spectra_learning.data.gems.mask_schedule import (
    jepa_mask_stage_index,
    jepa_mask_stages,
)
from spectra_learning.models.common_jax import Array
from spectra_learning.models.factory_jax import build_model_from_config
from spectra_learning.models.fastmixer_capacity import pairmixer_fast_stage_capacities
from spectra_learning.models.model_jax import PeakSetJEPAJax
from spectra_learning.probes.massspec.msg_probe_jax import run_msg_probe_jax
from spectra_learning.probes.massspec.msg_settings import (
    msg_probe_variants_from_config,
)
from spectra_learning.training.cadence import (
    msg_probe_interval,
    should_run_at_step,
    should_run_at_step_or_final,
    total_training_steps,
    validation_interval,
    validation_steps,
)
from spectra_learning.training.checkpointing_jax import (
    EmergencyCheckpointMonitor,
    build_jax_checkpoint_manager,
    jax_training_checkpoint_metadata,
    restore_frozen_teacher_encoder,
    restore_jax_training_state,
    save_jax_training_state,
)
from spectra_learning.training.configuration import finalize_config, save_config
from spectra_learning.training.logging import (
    MetricLogger,
    build_logger,
    log_msg_probe_metrics,
)
from spectra_learning.training import muon
from spectra_learning.training.schedules import learning_rate_at_step
from spectra_learning.training.storage import (
    local_scratch_dir,
    normalize_storage_path,
    storage_join,
    storage_mkdir,
)


JAX_DATA_AXIS = "data"
JaxMetricReduction = Literal["mean", "token_weighted"]


@dataclass(frozen=True)
class JaxTrainingTask:
    name: str
    build_datamodule: Callable[[config_dict.ConfigDict, int, int], Any]
    build_model: Callable[[config_dict.ConfigDict, Any], Any]
    checkpoint_contract: Callable[
        [config_dict.ConfigDict, Any, int],
        dict[str, Any],
    ]
    metric_reduction: JaxMetricReduction = "mean"
    enable_msg_probe: bool = False
    initialize_model: Callable[[config_dict.ConfigDict, Any], None] | None = None
    validate_model: Callable[[Any], None] | None = None
    run_metadata: Callable[[Any], dict[str, object]] | None = None
    log_start: Callable[[Any, int], None] | None = None


@dataclass(frozen=True)
class _StagedJaxTrainMetrics:
    metrics: dict[str, Array]
    epoch: int
    global_step: int
    total_steps: int


@dataclass
class _JaxTrainState:
    trainable_params: Any
    static_state: Any
    opt_state: Any

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "trainable_params": self.trainable_params,
            "static_state": self.static_state,
            "opt_state": self.opt_state,
        }


def trainable_param_filter(path: tuple[object, ...], value: object) -> bool:
    if not isinstance(value, nnx.Param):
        return False
    frozen_modules = {
        "teacher_encoder",
        "teacher_target_projector",
        "position_embedding",
    }
    if any(part in frozen_modules for part in path):
        return False
    return path[-1] != "b"


def collect_jax_param_metrics(model: Any) -> dict[str, float]:
    params = nnx.state(model, nnx.Param)
    trainable_params = nnx.state(model, trainable_param_filter)
    total_by_module = _jax_param_counts_by_module(params)
    trainable_by_module = _jax_param_counts_by_module(trainable_params)
    total = sum(total_by_module.values())
    trainable = sum(trainable_by_module.values())
    logging.info(
        "Model parameters: total=%s trainable=%s non_trainable=%s",
        f"{total:,}",
        f"{trainable:,}",
        f"{total - trainable:,}",
    )
    metrics: dict[str, float] = {
        "model/params_total": float(total),
        "model/params_trainable": float(trainable),
        "model/params_non_trainable": float(total - trainable),
    }
    for module_name in sorted(total_by_module):
        module_total = total_by_module[module_name]
        module_trainable = trainable_by_module.get(module_name, 0)
        logging.info(
            "  [%s] total=%s trainable=%s",
            module_name,
            f"{module_total:,}",
            f"{module_trainable:,}",
        )
        metrics[f"model/params_total/{module_name}"] = float(module_total)
        metrics[f"model/params_trainable/{module_name}"] = float(module_trainable)
    return metrics


def _jax_param_counts_by_module(state: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path, value in jax.tree_util.tree_flatten_with_path(state)[0]:
        module_name = str(_tree_path_key(path[0])) if path else "<root>"
        counts[module_name] = counts.get(module_name, 0) + int(value.size)
    return counts


def _jax_tree_numel(tree: Any) -> int:
    return sum(int(value.size) for value in jax.tree.leaves(tree))


def _jax_tree_l2(tree: Any) -> Array:
    total = sum(
        (
            jnp.sum(jnp.square(value.astype(jnp.float32)))
            for value in jax.tree.leaves(tree)
        ),
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    return jnp.sqrt(total)


def _jax_update_scale_metrics(
    params: Any,
    grads: Any,
    updates: Any,
) -> dict[str, Array]:
    num_params = jnp.asarray(float(_jax_tree_numel(params)), dtype=jnp.float32)
    param_l2 = _jax_tree_l2(params)
    grad_l2 = _jax_tree_l2(grads)
    update_l2 = _jax_tree_l2(updates)
    denom = jnp.sqrt(num_params)
    return {
        "param_l2": param_l2,
        "param_rms": param_l2 / denom,
        "grad_l2": grad_l2,
        "grad_rms": grad_l2 / denom,
        "update_l2": update_l2,
        "update_rms": update_l2 / denom,
        "update_to_param_l2": update_l2 / param_l2,
    }


def configure_jax_runtime(config: Any) -> None:
    if bool(config.get("jax_log_compiles", False)) or _env_enabled(
        "JAX_LOG_COMPILES"
    ):
        jax.config.update("jax_log_compiles", True)
    if bool(config.get("jax_explain_cache_misses", False)) or _env_enabled(
        "JAX_EXPLAIN_CACHE_MISSES"
    ):
        jax.config.update("jax_explain_cache_misses", True)
    compilation_cache_dir = str(config.get("jax_compilation_cache_dir", ""))
    if compilation_cache_dir:
        jax.config.update("jax_compilation_cache_dir", compilation_cache_dir)
        jax.config.update(
            "jax_enable_compilation_cache",
            bool(config.get("jax_enable_compilation_cache", True)),
        )
    min_compile_time = config.get("jax_persistent_cache_min_compile_time_secs", None)
    if min_compile_time is not None:
        jax.config.update(
            "jax_persistent_cache_min_compile_time_secs",
            float(min_compile_time),
        )
    min_entry_size = config.get("jax_persistent_cache_min_entry_size_bytes", None)
    if min_entry_size is not None:
        jax.config.update(
            "jax_persistent_cache_min_entry_size_bytes",
            int(min_entry_size),
        )


def initialize_jax_distributed(config: Any) -> None:
    if jax.distributed.is_initialized():
        return
    enabled = bool(config.get("jax_distributed_initialize", False)) or any(
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


def build_jax_optax_transform(
    config: Any,
    *,
    total_steps: int | None = None,
) -> optax.GradientTransformation:
    optimizer = str(config.get("optimizer", "adamw")).lower()
    if optimizer == "muon":
        transform = _jax_muon_transform(config, total_steps=total_steps)
    else:
        transform = _jax_adamw_transform(config, total_steps=total_steps)
    grad_clip_norm = float(config.get("grad_clip_norm", 0.0))
    if grad_clip_norm > 0.0:
        return optax.chain(optax.clip_by_global_norm(grad_clip_norm), transform)
    return transform


def _jax_adamw_transform(
    config: Any,
    *,
    total_steps: int | None = None,
) -> optax.GradientTransformation:
    learning_rate = _jax_learning_rate_schedule(config, total_steps=total_steps)
    b1 = float(config.get("b1", 0.9))
    b2 = float(config.get("b2", 0.999))
    weight_decay = float(config.get("weight_decay", 0.0))
    optimizer_state_dtype = str(
        config.get("optimizer_state_dtype", "fp32")
    ).lower()
    if optimizer_state_dtype in {"bf16", "bfloat16"}:
        return optax.chain(
            _jax_scale_by_adam_bf16_states(b1=b1, b2=b2),
            optax.add_decayed_weights(weight_decay, mask=_jax_weight_decay_mask),
            optax.scale_by_learning_rate(learning_rate),
        )
    return optax.adamw(
        learning_rate=learning_rate,
        b1=b1,
        b2=b2,
        weight_decay=weight_decay,
        mask=_jax_weight_decay_mask,
    )


def _jax_scale_by_adam_bf16_states(
    *,
    b1: float,
    b2: float,
) -> optax.GradientTransformation:
    adam = optax.scale_by_adam(b1=b1, b2=b2, mu_dtype=jnp.bfloat16)

    def init_fn(params):
        state = adam.init(params)
        # Optax casts each updated nu back to the dtype of state.nu.
        nu = jax.tree.map(lambda value: value.astype(jnp.bfloat16), state.nu)
        return state._replace(nu=nu)

    return optax.GradientTransformation(init_fn, adam.update)


def _jax_muon_transform(
    config: Any,
    *,
    total_steps: int | None = None,
) -> optax.GradientTransformation:
    learning_rate = _jax_learning_rate_schedule(config, total_steps=total_steps)
    adam_learning_rate = _jax_muon_adam_learning_rate_schedule(
        config,
        total_steps=total_steps,
    )
    return muon.build_muon_transform(
        config,
        learning_rate=learning_rate,
        adam_learning_rate=adam_learning_rate,
        weight_decay_mask=_jax_weight_decay_mask,
    )


def _jax_muon_adam_learning_rate_schedule(
    config: Any,
    *,
    total_steps: int | None = None,
):
    adam_learning_rate = config.get("muon_adam_learning_rate", None)
    if adam_learning_rate is None:
        return None
    return _jax_learning_rate_schedule(
        config,
        total_steps=total_steps,
        base_lr=float(adam_learning_rate),
        min_lr=config.get("muon_adam_min_learning_rate", None),
    )


def _jax_learning_rate_schedule(
    config: Any,
    *,
    total_steps: int | None = None,
    base_lr: float | None = None,
    min_lr: Any = None,
):
    if base_lr is None:
        base_lr = float(config.get("learning_rate", 1e-3))
        if min_lr is None:
            min_lr = config.get("min_learning_rate", None)
    else:
        base_lr = float(base_lr)
    warmup_steps = int(config.get("warmup_steps", 0))
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
    raw_total_steps = total_steps or config.get("training_max_steps", None)
    if raw_total_steps is None:
        return base_lr
    total_steps = int(raw_total_steps)
    schedule_start_step = int(
        config.get("learning_rate_schedule_start_step", 0)
    )
    schedule_steps = max(1, total_steps - schedule_start_step)
    warmup_start_lr = float(
        config.get("warmup_start_learning_rate", base_lr * 1e-8)
    )

    def schedule(step):
        step = jnp.asarray(step, dtype=jnp.float32)
        local_step = jnp.maximum(0.0, step - float(schedule_start_step))
        warmup = warmup_start_lr + (
            base_lr - warmup_start_lr
        ) * local_step / max(1, warmup_steps)
        ratio = jnp.clip(
            (local_step - float(warmup_steps))
            / float(max(1, schedule_steps - warmup_steps)),
            0.0,
            1.0,
        )
        decay = min_lr + (base_lr - min_lr) * 0.5 * (1.0 + jnp.cos(jnp.pi * ratio))
        if warmup_steps <= 0:
            return decay
        return jnp.where(local_step < warmup_steps, warmup, decay)

    return schedule


def _scheduled_jax_learning_rate(
    config: Any,
    *,
    global_step: int,
    total_steps: int,
) -> float:
    return learning_rate_at_step(
        global_step,
        base_lr=float(config.get("learning_rate", 1e-3)),
        total_steps=total_steps,
        warmup_steps=int(config.get("warmup_steps", 0)),
        min_learning_rate=config.get("min_learning_rate", None),
        schedule_start_step=int(
            config.get("learning_rate_schedule_start_step", 0)
        ),
        warmup_start_learning_rate=config.get(
            "warmup_start_learning_rate", None
        ),
    )


def _jax_weight_decay_mask(params: Any) -> Any:
    return jax.tree.map_with_path(
        lambda path, value: _tree_path_key(path[-1]) == "weight" and value.ndim >= 2,
        params,
    )


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
    value = _host_array_for_jax_process_local_data(value)
    spec = P(
        *([None] * batch_axis),
        JAX_DATA_AXIS,
        *([None] * (value.ndim - batch_axis - 1)),
    )
    sharding = NamedSharding(data_mesh, spec)
    if jax.process_count() > 1:
        return jax.make_array_from_process_local_data(sharding, value)
    return jax.device_put(_numpy_array_to_jax(value), sharding)


def _host_array_for_jax_process_local_data(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, jax.Array):
        return np.asarray(value)
    return value


def _numpy_array_to_jax(value: Any) -> Array:
    if isinstance(value, jax.Array):
        return value
    if isinstance(value, torch.Tensor):
        return jnp.asarray(value.detach().cpu().numpy())
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


def init_pure_optax_train_state(
    config: Any,
    model: Any,
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
    log_update_stats: bool = False,
    metric_reduction: JaxMetricReduction = "mean",
):
    def accumulated_metrics_and_grads(
        trainable_params: nnx.State,
        static_state: nnx.State,
        batch: dict[str, Array],
    ) -> tuple[dict[str, Array], nnx.State, Array]:
        def loss_fn(params: nnx.State, micro_batch: dict[str, Array]):
            functional_model = nnx.merge(graphdef, params, static_state)
            metrics = functional_model(micro_batch)
            loss = metrics["loss"]
            if metric_reduction == "token_weighted":
                loss = loss * metrics["target_tokens"]
            return loss, metrics

        def micro_batch_grad(micro_batch: dict[str, Array]):
            return jax.value_and_grad(loss_fn, has_aux=True)(
                trainable_params,
                micro_batch,
            )

        def scan_body(
            carry: tuple[nnx.State, dict[str, Array], Array],
            micro_batch: dict[str, Array],
        ):
            grad_accumulator, metric_accumulator, grad_denominator = carry
            (_micro_loss, micro_metrics), micro_grads = micro_batch_grad(micro_batch)
            grad_accumulator = jax.tree.map(
                lambda lhs, rhs: lhs + rhs,
                grad_accumulator,
                micro_grads,
            )
            micro_metric_totals = _jax_metric_totals(
                micro_metrics,
                metric_reduction=metric_reduction,
            )
            metric_accumulator = jax.tree.map(
                lambda lhs, rhs: lhs + rhs,
                metric_accumulator,
                micro_metric_totals,
            )
            grad_denominator = grad_denominator + _jax_gradient_denominator(
                micro_metrics,
                metric_reduction=metric_reduction,
            )
            return (grad_accumulator, metric_accumulator, grad_denominator), None

        first_batch = jax.tree.map(lambda value: value[0], batch)
        remaining_batches = jax.tree.map(lambda value: value[1:], batch)
        (_loss, metrics), grads = micro_batch_grad(first_batch)
        metric_totals = _jax_metric_totals(
            metrics,
            metric_reduction=metric_reduction,
        )
        grad_denominator = _jax_gradient_denominator(
            metrics,
            metric_reduction=metric_reduction,
        )

        (grads, metric_totals, grad_denominator), _ = jax.lax.scan(
            scan_body,
            (grads, metric_totals, grad_denominator),
            remaining_batches,
        )
        return metric_totals, grads, grad_denominator

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
            metric_totals, grads, grad_denominator = accumulated_metrics_and_grads(
                trainable_params,
                static_state,
                batch,
            )
            metric_totals = jax.tree.map(
                lambda value: jax.lax.psum(value, JAX_DATA_AXIS),
                metric_totals,
            )
            grads = jax.tree.map(
                lambda value: jax.lax.psum(value, JAX_DATA_AXIS),
                grads,
            )
            grad_denominator = jax.lax.psum(grad_denominator, JAX_DATA_AXIS)
            grads = jax.tree.map(
                lambda value: value / jnp.maximum(grad_denominator, 1.0),
                grads,
            )
            metrics = _finalize_jax_metric_totals(
                metric_totals,
                metric_reduction=metric_reduction,
                mean_denominator=grad_denominator,
            )
            updates, opt_state = optimizer.update(
                grads,
                opt_state,
                trainable_params,
            )
            if log_update_stats:
                metrics = {
                    **metrics,
                    **_jax_update_scale_metrics(trainable_params, grads, updates),
                }
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
        metric_totals, grads, grad_denominator = accumulated_metrics_and_grads(
            trainable_params,
            static_state,
            batch,
        )
        grads = jax.tree.map(
            lambda value: value / jnp.maximum(grad_denominator, 1.0),
            grads,
        )
        metrics = _finalize_jax_metric_totals(
            metric_totals,
            metric_reduction=metric_reduction,
            mean_denominator=grad_denominator,
        )
        updates, opt_state = optimizer.update(
            grads,
            opt_state,
            trainable_params,
        )
        if log_update_stats:
            metrics = {
                **metrics,
                **_jax_update_scale_metrics(trainable_params, grads, updates),
            }
        trainable_params = optax.apply_updates(trainable_params, updates)
        return trainable_params, opt_state, metrics

    return pure_accumulated_train_step


def make_pure_eval_step(
    graphdef: Any,
    *,
    sharded: bool,
    data_mesh: Mesh | None = None,
    metric_reduction: JaxMetricReduction = "mean",
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
            if metric_reduction == "token_weighted":
                metric_totals = _jax_metric_totals(
                    metrics,
                    metric_reduction=metric_reduction,
                )
                return jax.tree.map(
                    lambda value: jax.lax.psum(value, JAX_DATA_AXIS),
                    metric_totals,
                )
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
        metrics = functional_model(batch)
        return _jax_metric_totals(metrics, metric_reduction=metric_reduction)

    return pure_eval_step


def _jax_metric_totals(
    metrics: dict[str, Array],
    *,
    metric_reduction: JaxMetricReduction,
) -> dict[str, Array]:
    if metric_reduction == "mean":
        return metrics
    totals = {}
    for key, value in metrics.items():
        if key == "target_tokens" or key.startswith("target_tokens/"):
            totals[key] = value
            continue
        weight_key = _jax_token_metric_weight_key(key)
        totals[key] = value * metrics[weight_key]
    return totals


def _finalize_jax_metric_totals(
    totals: dict[str, Array],
    *,
    metric_reduction: JaxMetricReduction,
    mean_denominator: Array | float,
) -> dict[str, Array]:
    if metric_reduction == "mean":
        denominator = jnp.maximum(jnp.asarray(mean_denominator), 1.0)
        return jax.tree.map(lambda value: value / denominator, totals)
    metrics = {}
    for key, value in totals.items():
        if key == "target_tokens" or key.startswith("target_tokens/"):
            metrics[key] = value
            continue
        weight_key = _jax_token_metric_weight_key(key)
        metrics[key] = value / jnp.maximum(totals[weight_key], 1.0)
    return metrics


def _jax_gradient_denominator(
    metrics: dict[str, Array],
    *,
    metric_reduction: JaxMetricReduction,
) -> Array:
    if metric_reduction == "token_weighted":
        return metrics["target_tokens"].astype(jnp.float32)
    return jnp.ones((), dtype=jnp.float32)


def _jax_token_metric_weight_key(key: str) -> str:
    if key in {"loss", "token_accuracy"}:
        return "target_tokens"
    metric_name, separator, suffix = key.partition("/")
    if separator and metric_name in {"loss", "token_accuracy"}:
        return f"target_tokens/{suffix}"
    raise ValueError(f"Token-weighted JAX metric has no target-token count: {key}")


def train_and_evaluate_jax(
    config: config_dict.ConfigDict,
    workdir: str | Path,
) -> dict[str, object]:
    task = JaxTrainingTask(
        name="pretrain",
        build_datamodule=_build_pretrain_jax_datamodule,
        build_model=lambda task_config, _datamodule: build_model_from_config(task_config),
        checkpoint_contract=_pretrain_jax_checkpoint_contract,
        enable_msg_probe=True,
        initialize_model=initialize_jax_pretrain_model,
        validate_model=_validate_pretrain_jax_model,
    )
    return train_and_evaluate_jax_task(config, workdir, task=task)


def train_and_evaluate_jax_task(
    config: config_dict.ConfigDict,
    workdir: str | Path,
    *,
    task: JaxTrainingTask,
) -> dict[str, object]:
    with EmergencyCheckpointMonitor.for_current_environment() as emergency_checkpoint:
        return _train_and_evaluate_jax_task(
            config,
            workdir,
            task=task,
            emergency_checkpoint=emergency_checkpoint,
        )


def _train_and_evaluate_jax_task(
    config: config_dict.ConfigDict,
    workdir: str | Path,
    *,
    task: JaxTrainingTask,
    emergency_checkpoint: EmergencyCheckpointMonitor,
) -> dict[str, object]:
    configure_jax_runtime(config)
    initialize_jax_distributed(config)
    workdir = normalize_storage_path(workdir)
    local_workdir = local_scratch_dir(workdir)
    local_workdir.mkdir(parents=True, exist_ok=True)
    is_main_process = jax.process_index() == 0
    if is_main_process:
        storage_mkdir(workdir)
    multihost_utils.sync_global_devices(
        f"spectra_learning_jax_{task.name}_workdir_ready"
    )
    torch.manual_seed(int(config.seed))
    prepare_jax_training_config(config)
    _validate_jax_task_probe_config(config, task)
    if is_main_process:
        save_config(config, workdir)
    datamodule = task.build_datamodule(
        config,
        jax.process_count(),
        jax.process_index(),
    )
    total_steps = total_training_steps(config, datamodule)
    model = task.build_model(config, datamodule)
    if task.validate_model is not None:
        task.validate_model(model)
    task_contract = task.checkpoint_contract(config, datamodule, total_steps)
    checkpoint_metadata = jax_training_checkpoint_metadata(task.name, task_contract)
    checkpoint_dir = storage_join(workdir, "checkpoints")
    if is_main_process:
        storage_mkdir(checkpoint_dir)
    multihost_utils.sync_global_devices(
        f"spectra_learning_jax_{task.name}_checkpoint_dir_ready"
    )
    jax_checkpoint_max_to_keep = config.get("jax_checkpoint_max_to_keep", 5)
    if jax_checkpoint_max_to_keep is not None:
        jax_checkpoint_max_to_keep = int(jax_checkpoint_max_to_keep)
    checkpoint_manager = build_jax_checkpoint_manager(
        checkpoint_dir,
        max_to_keep=jax_checkpoint_max_to_keep,
        enable_async_checkpointing=bool(
            config.get("jax_enable_async_checkpointing", True)
        ),
    )
    resume_step = checkpoint_manager.latest_step()
    if resume_step is None and task.initialize_model is not None:
        task.initialize_model(config, model)
    logger = build_logger(config, local_workdir) if is_main_process else MetricLogger()
    param_metrics = collect_jax_param_metrics(model)
    if is_main_process:
        if task.log_start is not None:
            task.log_start(datamodule, total_steps)
        logging.info(
            "Training JAX task %s for %d optimizer steps on %d process(es).",
            task.name,
            total_steps,
            jax.process_count(),
        )
        param_metrics_step = int(resume_step or 0)
        logger.log_metrics(
            {"global_step": float(param_metrics_step), **param_metrics},
            step=param_metrics_step,
        )
    metrics = _run_jax_training_loop(
        config=config,
        datamodule=datamodule,
        model=model,
        logger=logger,
        total_steps=total_steps,
        checkpoint_manager=checkpoint_manager,
        resume_step=resume_step,
        checkpoint_metadata=checkpoint_metadata,
        metric_reduction=task.metric_reduction,
        enable_msg_probe=task.enable_msg_probe,
        emergency_checkpoint=emergency_checkpoint,
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
            datamodule.gradient_accumulation_steps
        ),
        "run/device_backend": "jax",
        "run/training_task": task.name,
    }
    if task.run_metadata is not None:
        run_metrics.update(task.run_metadata(datamodule))
    results = {**metrics, **run_metrics, **param_metrics}
    if is_main_process:
        final_global_step = int(metrics["run/final_global_step"])
        logger.log_metrics(
            {"global_step": float(final_global_step), **results},
            step=final_global_step,
        )
        logger.finish()
    return results


def prepare_jax_training_config(config: config_dict.ConfigDict) -> None:
    if "jax_msg_probe_shard_batches" in config:
        raise ValueError(
            "jax_msg_probe_shard_batches has been removed; MSG probe batches are unsharded"
        )
    config.dataloader_pin_memory = False
    config.dataloader_persistent_workers = False
    config.dataloader_output_format = "numpy"
    if int(config.get("dataloader_num_workers", 0)) > 0:
        config.dataloader_multiprocessing_context = str(
            config.get("dataloader_multiprocessing_context", "forkserver")
            or "forkserver"
        )
    finalize_config(config)


def jax_config_checkpoint_contract(
    config: config_dict.ConfigDict,
) -> dict[str, Any]:
    effective_config = config_to_dict(config)
    effective_config.pop("config_path", None)
    effective_config.pop("jax_resume_allowed_config_keys", None)
    return {"config": effective_config}


def _build_pretrain_jax_datamodule(
    config: config_dict.ConfigDict,
    process_count: int,
    process_index: int,
) -> GemsDataModule:
    return GemsDataModule(
        config,
        seed=int(config.seed),
        distributed_world_size=process_count,
        distributed_rank=process_index,
        distributed_local_rank=0,
    )


def _pretrain_jax_checkpoint_contract(
    config: config_dict.ConfigDict,
    datamodule: GemsDataModule,
    _total_steps: int,
) -> dict[str, Any]:
    contract = jax_config_checkpoint_contract(config)
    if datamodule.artifact.format == MASSIVE_V2_HDF5_FORMAT:
        contract["dataset"] = {
            "format": datamodule.info["gems_hdf5_format"],
            "repo_id": datamodule.info["gems_hdf5_repo_id"],
            "revision": datamodule.info["gems_hdf5_revision"],
            "manifest_sha256": datamodule.info["gems_manifest_sha256"],
            "shard_plan_sha256": datamodule.info[
                "gems_shard_plan_sha256"
            ],
        }
    return contract


def _validate_pretrain_jax_model(model: Any) -> None:
    if model.use_ema_teacher:
        raise ValueError("JAX training uses pure Optax and does not support EMA teachers.")


def _validate_jax_task_probe_config(
    config: config_dict.ConfigDict,
    task: JaxTrainingTask,
) -> None:
    if task.enable_msg_probe:
        return
    probe_interval = float(config.get("msg_probe_every_n_steps", -1.0))
    probe_at_final = bool(config.get("msg_probe_at_final_step", False))
    if probe_interval >= 0.0 or probe_at_final:
        raise ValueError(
            f"JAX task {task.name!r} does not support the MSG probe; "
            "set msg_probe_every_n_steps=-1 and msg_probe_at_final_step=False."
        )


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


def initialize_jax_pretrain_model(
    config: config_dict.ConfigDict,
    model: PeakSetJEPAJax,
) -> None:
    if model.use_frozen_teacher:
        return
    initialize_jax_model_from_torch_seed(config, model)


_JAX_PROFILE_TIMING_NAMES = (
    "dataloader_seconds",
    "transfer_seconds",
    "grad_seconds",
    "accumulate_seconds",
    "apply_seconds",
    "compiled_step_seconds",
)
_JAX_TIME_LIMIT_CHECK_EVERY_STEPS = 100


def _jax_wall_time() -> float:
    return time.time()


def _jax_training_deadline(config: config_dict.ConfigDict) -> float | None:
    max_duration_hours = config.get("max_duration_hours", None)
    if max_duration_hours is None:
        return None
    if jax.process_index() == 0:
        logging.info(
            "Training wall-clock budget: %.2f hours",
            float(max_duration_hours),
        )
    return _jax_wall_time() + float(max_duration_hours) * 3600.0


def _jax_process_bool_broadcast(value: bool) -> bool:
    if jax.process_count() == 1:
        return value
    broadcast = multihost_utils.broadcast_one_to_all(
        np.asarray(value, dtype=np.int32),
        is_source=jax.process_index() == 0,
    )
    return bool(np.asarray(broadcast).item())


class _JaxTrainingLoop:
    def __init__(
        self,
        *,
        config: config_dict.ConfigDict,
        datamodule: Any,
        model: Any,
        logger: MetricLogger,
        total_steps: int,
        checkpoint_manager: Any,
        resume_step: int | None,
        checkpoint_metadata: dict[str, Any],
        metric_reduction: JaxMetricReduction,
        enable_msg_probe: bool,
        emergency_checkpoint: EmergencyCheckpointMonitor | None,
    ) -> None:
        self.config = config
        self.datamodule = datamodule
        self.model = model
        self.logger = logger
        self.total_steps = total_steps
        self.checkpoint_manager = checkpoint_manager
        self.checkpoint_metadata = checkpoint_metadata
        self.metric_reduction = metric_reduction
        self.resume_step = resume_step
        self.emergency_checkpoint = emergency_checkpoint

        self.log_every_n_steps = int(config.get("log_every_n_steps", 50))
        self.warmup_steps = int(config.get("throughput_warmup_steps", 0))
        self.grad_accum_steps = int(config.get("gradient_accumulation_steps", 1))
        self.deadline = _jax_training_deadline(config)
        self.time_limit_check_every_steps = int(
            config.get(
                "jax_time_limit_check_every_steps",
                _JAX_TIME_LIMIT_CHECK_EVERY_STEPS,
            )
        )
        self.stopped_for_time_limit = False
        self.stopped_for_termination = False
        self.use_mask_schedule = (
            isinstance(model, PeakSetJEPAJax)
            and "jepa_context_fraction_schedule" in config
        )
        self.mask_stages = jepa_mask_stages(config) if self.use_mask_schedule else ()
        self.mask_capacities = (
            pairmixer_fast_stage_capacities(config)
            if self.use_mask_schedule
            else ()
        )
        self.mask_stage_index = -1 if self.use_mask_schedule else 0
        data_parallel_devices = _jax_data_parallel_devices(config)
        self.data_mesh = _jax_data_mesh_for_device_count(data_parallel_devices)
        self.use_sharded_step = data_parallel_devices > 1
        self.state, self.train_steps, self.eval_steps = self._initialize_train_state()
        self.train_step = self.train_steps[0]
        self.eval_step = self.eval_steps[0]

        self.checkpoint_every_steps = int(config.get("checkpoint_every_steps", 0))
        self.val_every_n_steps = validation_interval(config, datamodule, total_steps)
        self.val_num_steps = validation_steps(config)
        self.msg_probe_every_n_steps = (
            msg_probe_interval(config, datamodule, total_steps)
            if enable_msg_probe
            else -1
        )
        self.msg_probe_variants = (
            msg_probe_variants_from_config(config) if enable_msg_probe else ()
        )
        self.start_step = self._restore_checkpoint(resume_step)
        nnx.update(
            self.model,
            self.state.trainable_params,
            self.state.static_state,
        )
        self.global_step = self.start_step
        self._activate_mask_stage(self.global_step)

        self.timing_barriers = bool(config.get("jax_timing_barriers", False))
        self.compile_stall_threshold_seconds = float(
            config.get("jax_compile_stall_threshold_seconds", 0.0)
        )
        self.profile_dir = str(config.get("jax_profile_dir", ""))
        self.profile_start_step = int(
            config.get("jax_profile_start_step", self.warmup_steps)
        )
        profile_steps = int(config.get("jax_profile_steps", 0))
        self.profile_end_step = (
            self.profile_start_step + profile_steps
            if profile_steps > 0
            else total_steps
        )
        self.profile_started = False
        self.profile_active = False

        self.timing = {name: 0.0 for name in _JAX_PROFILE_TIMING_NAMES}
        self.timing["measured_microbatches"] = 0.0
        self.non_train_timing = {
            "checkpoint_seconds": 0.0,
            "validation_seconds": 0.0,
            "msg_probe_seconds": 0.0,
            "model_update_seconds": 0.0,
            "profile_seconds": 0.0,
        }
        self.measured_non_train_timing = {
            name: 0.0 for name in self.non_train_timing
        }
        self.train_start = 0.0
        self.measured_start: float | None = None
        self.measured_steps = 0
        self.last_metrics: dict[str, Array] = {}
        self.pending_train_metrics: _StagedJaxTrainMetrics | None = None
        self.last_validation_metrics: dict[str, float] = {}
        self.last_msg_probe_metrics: dict[str, float] = {}

    def _initialize_train_state(
        self,
    ) -> tuple[
        _JaxTrainState,
        tuple[Callable[..., Any], ...],
        tuple[Callable[..., Any], ...],
    ]:
        graphdef, params, static_state, opt_state, optimizer = (
            init_pure_optax_train_state(
                self.config,
                self.model,
                total_steps=self.total_steps,
            )
        )

        def make_steps(
            stage_graphdef: Any,
        ) -> tuple[Callable[..., Any], Callable[..., Any]]:
            return (
                make_pure_accumulated_train_step(
                    stage_graphdef,
                    optimizer,
                    sharded=self.use_sharded_step,
                    data_mesh=self.data_mesh,
                    log_update_stats=bool(
                        self.config.get("jax_log_update_stats", False)
                    ),
                    metric_reduction=self.metric_reduction,
                ),
                make_pure_eval_step(
                    stage_graphdef,
                    sharded=self.use_sharded_step,
                    data_mesh=self.data_mesh,
                    metric_reduction=self.metric_reduction,
                ),
            )

        if not self.use_mask_schedule:
            train_step, eval_step = make_steps(graphdef)
            train_steps = [train_step]
            eval_steps = [eval_step]
        else:
            train_steps = []
            eval_steps = []
        for capacity in self.mask_capacities:
            self.model.set_fastmixer_capacities(*capacity)
            stage_graphdef, _, _ = nnx.split(
                self.model,
                trainable_param_filter,
                ...,
            )
            train_step, eval_step = make_steps(stage_graphdef)
            train_steps.append(train_step)
            eval_steps.append(eval_step)
        if self.use_mask_schedule:
            self.model.set_fastmixer_capacities(*self.mask_capacities[0])
        state = _JaxTrainState(params, static_state, opt_state)
        if self.use_sharded_step:
            params = _replicate_tree_on_data_mesh(
                state.trainable_params,
                self.data_mesh,
            )
            opt_state = _replicate_tree_on_data_mesh(
                state.opt_state,
                self.data_mesh,
            )
            static_state = _replicate_tree_on_data_mesh(
                state.static_state,
                self.data_mesh,
            )
            state = _JaxTrainState(
                params,
                static_state,
                opt_state,
            )
        if self.resume_step is None and isinstance(self.model, PeakSetJEPAJax):
            if self.model.use_frozen_teacher:
                teacher_checkpoint = self.config.frozen_teacher_checkpoint_path
                state.static_state["teacher_encoder"] = (
                    restore_frozen_teacher_encoder(
                        teacher_checkpoint,
                        state.static_state["teacher_encoder"],
                    )
                )
                if jax.process_index() == 0:
                    logging.info(
                        "Restored frozen JAX teacher encoder from %s",
                        teacher_checkpoint,
                    )
        return state, tuple(train_steps), tuple(eval_steps)

    def _activate_mask_stage(self, global_step: int) -> bool:
        if not self.use_mask_schedule:
            return False
        stage_index = jepa_mask_stage_index(
            self.config,
            global_step,
            self.total_steps,
        )
        if stage_index == self.mask_stage_index:
            return False
        if self.mask_stage_index >= 0:
            jax.effects_barrier()
            self.train_steps[self.mask_stage_index].clear_cache()
            self.eval_steps[self.mask_stage_index].clear_cache()
            gc.collect()
            logging.info(
                "Released JAX executable cache for MAE mask stage %d",
                self.mask_stage_index + 1,
            )
        stage = self.mask_stages[stage_index]
        encoder_tokens, predictor_tokens, target_tokens = self.mask_capacities[
            stage_index
        ]
        self.datamodule.set_mask_fractions(
            stage.context_fraction,
            stage.target_fraction,
        )
        self.datamodule.set_gradient_accumulation_steps(
            stage.gradient_accumulation_steps
        )
        self.grad_accum_steps = stage.gradient_accumulation_steps
        self.model.set_fastmixer_capacities(
            encoder_tokens,
            predictor_tokens,
            target_tokens,
        )
        self.train_step = self.train_steps[stage_index]
        self.eval_step = self.eval_steps[stage_index]
        self.mask_stage_index = stage_index
        logging.info(
            "MAE mask stage %d: context_fraction=%.2f target_fraction=%.2f "
            "encoder_tokens=%d predictor_tokens=%d target_tokens=%d "
            "accumulation_steps=%d",
            stage_index + 1,
            stage.context_fraction,
            stage.target_fraction,
            encoder_tokens,
            predictor_tokens,
            target_tokens,
            stage.gradient_accumulation_steps,
        )
        return True

    def _restore_checkpoint(self, resume_step: int | None) -> int:
        if resume_step is None:
            return 0
        restored = restore_jax_training_state(
            self.checkpoint_manager,
            int(resume_step),
            self.state.checkpoint_state(),
            expected_metadata=self.checkpoint_metadata,
            allowed_config_keys=tuple(
                self.config.get("jax_resume_allowed_config_keys", ())
            ),
        )
        self.state = _JaxTrainState(
            restored["trainable_params"],
            restored["static_state"],
            restored["opt_state"],
        )
        return int(resume_step)

    def run(self) -> dict[str, object]:
        if self._emergency_checkpoint_requested():
            self._save_emergency_checkpoint_and_wait()
            return self._finish()
        loop_epochs = max(1, math.ceil(float(self.config.num_epochs)))
        start_epoch = min(
            self.start_step // self.datamodule.train_steps,
            loop_epochs - 1,
        )
        self.train_start = time.perf_counter()
        for epoch in range(start_epoch, loop_epochs):
            self._run_epoch(epoch, start_epoch=start_epoch, loop_epochs=loop_epochs)
            if (
                self.stopped_for_time_limit
                or self.stopped_for_termination
                or self.global_step >= self.total_steps
            ):
                break
        return self._finish()

    def _run_epoch(self, epoch: int, *, start_epoch: int, loop_epochs: int) -> None:
        epoch_start_batch = (
            self.global_step - epoch * self.datamodule.train_steps
            if epoch == start_epoch
            else 0
        )
        loader = self.datamodule.train_loader_for_epoch(
            epoch,
            start_batch=epoch_start_batch,
        )
        loader_iter = iter(loader)
        loader_stage_index = self.mask_stage_index
        pbar = tqdm(
            total=min(
                self.datamodule.train_steps - epoch_start_batch,
                self.total_steps - self.global_step,
            ),
            desc=f"Epoch {epoch}",
            unit="step",
            disable=jax.process_index() != 0,
        )
        while self.global_step < self.total_steps:
            if self._time_limit_reached():
                break
            if loader_stage_index != self.mask_stage_index:
                _shutdown_torch_loader_iterator(loader_iter)
                del loader_iter, loader
                current_epoch_batch = (
                    self.global_step - epoch * self.datamodule.train_steps
                )
                loader = self.datamodule.train_loader_for_epoch(
                    epoch,
                    start_batch=current_epoch_batch,
                )
                loader_iter = iter(loader)
                loader_stage_index = self.mask_stage_index
            batch = self._next_accumulated_batch(loader_iter)
            if batch is None:
                break
            self._start_measurement_if_ready()
            self._start_profile_if_ready()
            metrics = self._train_batch(batch)
            pbar.update(1)
            if self._emergency_checkpoint_requested():
                self._save_emergency_checkpoint_and_wait()
                break
            self._log_or_stage_train_metrics(metrics, epoch=epoch, pbar=pbar)
            self._activate_mask_stage(self.global_step)
            self._run_scheduled_work(pbar)
        if self.pending_train_metrics is not None and (
            self.stopped_for_time_limit
            or self.stopped_for_termination
            or self.global_step >= self.total_steps
            or epoch == loop_epochs - 1
        ):
            _log_jax_train_metrics(
                self.config,
                self.logger,
                pbar,
                self.pending_train_metrics,
            )
            self.pending_train_metrics = None
        pbar.close()
        _shutdown_torch_loader_iterator(loader_iter)
        del loader_iter, loader

    def _time_limit_reached(self) -> bool:
        if self.deadline is None:
            return False
        should_check = (
            self.global_step == self.start_step
            or self.global_step % self.time_limit_check_every_steps == 0
        )
        if not should_check:
            return False
        source_reached = (
            jax.process_index() == 0 and _jax_wall_time() >= self.deadline
        )
        if not _jax_process_bool_broadcast(source_reached):
            return False
        if jax.process_index() == 0:
            logging.info(
                "Reached max_duration_hours at global_step=%d.",
                self.global_step,
            )
        self.stopped_for_time_limit = True
        return True

    def _emergency_checkpoint_requested(self) -> bool:
        if self.emergency_checkpoint is None:
            return False
        requested = np.asarray(
            self.emergency_checkpoint.requested,
            dtype=np.int32,
        )
        if jax.process_count() > 1:
            requested = multihost_utils.process_allgather(requested)
        return bool(np.asarray(requested).any())

    def _save_emergency_checkpoint_and_wait(self) -> None:
        assert self.emergency_checkpoint is not None
        self.stopped_for_termination = True
        reason = self.emergency_checkpoint.reason or "another JAX process terminated"
        if jax.process_index() == 0:
            logging.warning(
                "Termination requested at global_step=%d (%s); writing emergency "
                "checkpoint.",
                self.global_step,
                reason,
            )
        phase_start = time.perf_counter()
        if self.checkpoint_manager.latest_step() != self.global_step:
            self._save_checkpoint(self.global_step)
        self.checkpoint_manager.wait_until_finished()
        self._add_non_train_timing(
            "checkpoint_seconds",
            time.perf_counter() - phase_start,
        )
        if jax.process_index() == 0:
            logging.warning(
                "Emergency checkpoint at global_step=%d is durable.",
                self.global_step,
            )
        self.emergency_checkpoint.wait_for_forced_termination()

    def _next_accumulated_batch(self, loader_iter: Any) -> dict[str, Any] | None:
        dataloader_elapsed = 0.0
        transfer_elapsed = 0.0
        micro_batches = []
        for _ in range(self.grad_accum_steps):
            dataloader_start = time.perf_counter()
            try:
                torch_batch = next(loader_iter)
            except StopIteration:
                break
            dataloader_elapsed += time.perf_counter() - dataloader_start
            transfer_start = time.perf_counter()
            micro_batches.append(torch_batch)
            transfer_elapsed += time.perf_counter() - transfer_start
        if len(micro_batches) < self.grad_accum_steps:
            return None
        batch = numpy_batch_to_jax(
            _stack_micro_batches(micro_batches),
            data_mesh=self.data_mesh if self.use_sharded_step else None,
            batch_axis=1,
        )
        if self.timing_barriers:
            jax.block_until_ready(batch)
        if self.measured_start is not None:
            self.timing["dataloader_seconds"] += dataloader_elapsed
            self.timing["transfer_seconds"] += transfer_elapsed
            self.timing["measured_microbatches"] += float(self.grad_accum_steps)
        return batch

    def _start_measurement_if_ready(self) -> None:
        if self.measured_start is not None or self.global_step < self.warmup_steps:
            return
        jax.effects_barrier()
        self.measured_start = time.perf_counter()

    def _start_profile_if_ready(self) -> None:
        if (
            not self.profile_dir
            or self.profile_started
            or self.global_step < self.profile_start_step
        ):
            return
        phase_start = time.perf_counter()
        jax.effects_barrier()
        jax.profiler.start_trace(self.profile_dir)
        self._add_non_train_timing(
            "profile_seconds",
            time.perf_counter() - phase_start,
        )
        self.profile_started = True
        self.profile_active = True

    def _train_batch(self, batch: dict[str, Any]) -> dict[str, Array]:
        step_start = time.perf_counter()
        params, opt_state, metrics = self.train_step(
            self.state.trainable_params,
            self.state.static_state,
            self.state.opt_state,
            batch,
        )
        self.state.trainable_params = params
        self.state.opt_state = opt_state
        if self.timing_barriers:
            jax.block_until_ready((params, opt_state, metrics))
        step_elapsed = time.perf_counter() - step_start
        _raise_on_jax_compile_stall(
            step_elapsed,
            threshold_seconds=self.compile_stall_threshold_seconds,
            global_step=self.global_step,
            branch="pure_optax_scan",
        )
        if self.measured_start is not None:
            self.timing["compiled_step_seconds"] += step_elapsed
            self.measured_steps += 1
        self.last_metrics = metrics
        self.global_step += 1
        return metrics

    def _log_or_stage_train_metrics(
        self,
        metrics: dict[str, Array],
        *,
        epoch: int,
        pbar: tqdm,
    ) -> None:
        _log_jax_train_metrics(
            self.config,
            self.logger,
            pbar,
            self.pending_train_metrics,
        )
        self.pending_train_metrics = None
        if _should_log_jax_train_metrics(
            self.global_step,
            self.log_every_n_steps,
        ):
            self.pending_train_metrics = _stage_jax_train_metrics(
                metrics,
                epoch=epoch,
                global_step=self.global_step,
                total_steps=self.total_steps,
            )

    def _run_scheduled_work(self, pbar: tqdm) -> None:
        phase_start = time.perf_counter()
        self._save_periodic_checkpoint()
        self._add_non_train_timing(
            "checkpoint_seconds",
            time.perf_counter() - phase_start,
        )
        self._run_validation_if_due(pbar)
        self._run_msg_probe_if_due()
        if self.profile_active and self.global_step >= self.profile_end_step:
            self._stop_profile()

    def _save_checkpoint(self, step: int) -> None:
        save_jax_training_state(
            self.checkpoint_manager,
            step,
            self.state.checkpoint_state(),
            metadata=self.checkpoint_metadata,
        )

    def _save_periodic_checkpoint(self) -> None:
        if (
            self.checkpoint_every_steps > 0
            and self.global_step % self.checkpoint_every_steps == 0
        ):
            self._save_checkpoint(self.global_step)

    def _run_validation_if_due(self, pbar: tqdm) -> None:
        if not should_run_at_step(self.val_every_n_steps, self.global_step):
            return
        phase_start = time.perf_counter()
        self.last_validation_metrics = _evaluate_jax_validation_loss(
            datamodule=self.datamodule,
            trainable_params=self.state.trainable_params,
            static_state=self.state.static_state,
            eval_step=self.eval_step,
            max_steps=self.val_num_steps,
            use_sharded_step=self.use_sharded_step,
            data_mesh=self.data_mesh,
            metric_reduction=self.metric_reduction,
        )
        _log_jax_validation_metrics(
            self.logger,
            pbar,
            self.last_validation_metrics,
            global_step=self.global_step,
        )
        self._add_non_train_timing(
            "validation_seconds",
            time.perf_counter() - phase_start,
        )

    def _run_msg_probe_if_due(self) -> None:
        if not should_run_at_step_or_final(
            self.msg_probe_every_n_steps,
            self.global_step,
            total_steps=self.total_steps,
            run_at_final_step=bool(self.config.get("msg_probe_at_final_step", False)),
        ):
            return
        phase_start = time.perf_counter()
        self.last_msg_probe_metrics = _run_distributed_msg_probe_jax(
            config=self.config,
            model=self.model,
            logger=self.logger,
            variants=self.msg_probe_variants,
            global_step=self.global_step,
            trainable_params=self.state.trainable_params,
            data_mesh=self.data_mesh,
        )
        self._add_non_train_timing(
            "msg_probe_seconds",
            time.perf_counter() - phase_start,
        )

    def _stop_profile(self, *, synchronize: bool = True) -> None:
        phase_start = time.perf_counter()
        if synchronize:
            jax.effects_barrier()
        jax.profiler.stop_trace()
        self._add_non_train_timing(
            "profile_seconds",
            time.perf_counter() - phase_start,
        )
        self.profile_active = False

    def _add_non_train_timing(
        self,
        name: str,
        elapsed: float,
        *,
        include_measured: bool = True,
    ) -> None:
        self.non_train_timing[name] += elapsed
        if include_measured and self.measured_start is not None:
            self.measured_non_train_timing[name] += elapsed

    def _finish(self) -> dict[str, object]:
        jax.block_until_ready(self.state.trainable_params)
        phase_start = time.perf_counter()
        nnx.update(self.model, self.state.trainable_params)
        jax.effects_barrier()
        self._add_non_train_timing(
            "model_update_seconds",
            time.perf_counter() - phase_start,
        )
        post_model_update_time = time.perf_counter()
        if self.profile_active:
            self._stop_profile(synchronize=False)
        measured_wall_elapsed = (
            post_model_update_time - self.measured_start
            if self.measured_start is not None
            else 0.0
        )
        measured_non_train_elapsed = sum(self.measured_non_train_timing.values())
        measured_train_elapsed = max(
            measured_wall_elapsed - measured_non_train_elapsed,
            0.0,
        )
        if (
            self.global_step > self.start_step
            and self.checkpoint_manager.latest_step() != self.global_step
        ):
            phase_start = time.perf_counter()
            self._save_checkpoint(self.global_step)
            self._add_non_train_timing(
                "checkpoint_seconds",
                time.perf_counter() - phase_start,
                include_measured=False,
            )
        self.checkpoint_manager.wait_until_finished()
        wall_elapsed = time.perf_counter() - self.train_start
        train_elapsed = max(wall_elapsed - sum(self.non_train_timing.values()), 0.0)
        return self._build_result(
            wall_elapsed=wall_elapsed,
            train_elapsed=train_elapsed,
            measured_wall_elapsed=measured_wall_elapsed,
            measured_non_train_elapsed=measured_non_train_elapsed,
            measured_train_elapsed=measured_train_elapsed,
        )

    def _build_result(
        self,
        *,
        wall_elapsed: float,
        train_elapsed: float,
        measured_wall_elapsed: float,
        measured_non_train_elapsed: float,
        measured_train_elapsed: float,
    ) -> dict[str, object]:
        global_batch_size = int(self.datamodule.global_batch_size)
        train_metrics: dict[str, float] = {"train/loss": float("nan")}
        if self.last_metrics:
            host_metrics = _jax_metrics_to_host(self.last_metrics, prefix="train/")
            if jax.process_index() == 0:
                train_metrics = host_metrics
        result: dict[str, object] = {
            "run/final_global_step": float(self.global_step),
            "run/stopped_for_time_limit": float(self.stopped_for_time_limit),
            "run/stopped_for_termination": float(self.stopped_for_termination),
            "run/wall_elapsed_seconds": wall_elapsed,
            "run/train_elapsed_seconds": train_elapsed,
            "run/non_train_elapsed_seconds": sum(self.non_train_timing.values()),
            "run/steps_per_second": _rate(self.global_step, train_elapsed),
            "run/samples_per_second": _rate(
                self.global_step * global_batch_size,
                train_elapsed,
            ),
            "run/wall_steps_per_second": _rate(self.global_step, wall_elapsed),
            "run/wall_samples_per_second": _rate(
                self.global_step * global_batch_size,
                wall_elapsed,
            ),
            "run/measured_steps": float(self.measured_steps),
            "run/measured_wall_elapsed_seconds": measured_wall_elapsed,
            "run/measured_non_train_elapsed_seconds": measured_non_train_elapsed,
            "run/measured_elapsed_seconds": measured_train_elapsed,
            "run/measured_steps_per_second": _rate(
                self.measured_steps,
                measured_train_elapsed,
            ),
            "run/measured_samples_per_second": _rate(
                self.measured_steps * global_batch_size,
                measured_train_elapsed,
            ),
            "run/measured_wall_steps_per_second": _rate(
                self.measured_steps,
                measured_wall_elapsed,
            ),
            "run/measured_wall_samples_per_second": _rate(
                self.measured_steps * global_batch_size,
                measured_wall_elapsed,
            ),
            **train_metrics,
        }
        result.update(
            {f"run/{name}": value for name, value in self.non_train_timing.items()}
        )
        result.update(self.last_validation_metrics)
        result.update(self.last_msg_probe_metrics)
        result.update(
            {f"run/profile_{name}": value for name, value in self.timing.items()}
        )
        measured_microbatches = self.timing["measured_microbatches"]
        if measured_microbatches > 0:
            result.update(
                {
                    f"run/profile_{name}_per_microbatch": (
                        self.timing[name] / measured_microbatches
                    )
                    for name in _JAX_PROFILE_TIMING_NAMES
                }
            )
        if self.measured_steps > 0:
            result.update(
                {
                    f"run/profile_{name}_per_step": (
                        self.timing[name] / self.measured_steps
                    )
                    for name in _JAX_PROFILE_TIMING_NAMES
                }
            )
        return result


def _rate(count: int, elapsed: float) -> float:
    return float(count) / elapsed if elapsed > 0 else 0.0


def _run_jax_training_loop(
    *,
    config: config_dict.ConfigDict,
    datamodule: Any,
    model: Any,
    logger: MetricLogger,
    total_steps: int,
    checkpoint_manager: Any,
    resume_step: int | None,
    checkpoint_metadata: dict[str, Any],
    metric_reduction: JaxMetricReduction,
    enable_msg_probe: bool,
    emergency_checkpoint: EmergencyCheckpointMonitor | None = None,
) -> dict[str, object]:
    return _JaxTrainingLoop(
        config=config,
        datamodule=datamodule,
        model=model,
        logger=logger,
        total_steps=total_steps,
        checkpoint_manager=checkpoint_manager,
        resume_step=resume_step,
        checkpoint_metadata=checkpoint_metadata,
        metric_reduction=metric_reduction,
        enable_msg_probe=enable_msg_probe,
        emergency_checkpoint=emergency_checkpoint,
    ).run()


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
    datamodule: Any,
    trainable_params: nnx.State,
    static_state: nnx.State,
    eval_step: Any,
    max_steps: int,
    use_sharded_step: bool,
    data_mesh: Mesh,
    metric_reduction: JaxMetricReduction,
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
    if metric_reduction == "mean":
        metrics = {key: value / float(steps) for key, value in totals.items()}
    else:
        metrics = _finalize_token_metric_totals_host(totals)
    return {f"val/{key}": value for key, value in metrics.items()}


def _finalize_token_metric_totals_host(
    totals: dict[str, float],
) -> dict[str, float]:
    metrics = {}
    for key, value in totals.items():
        if key == "target_tokens" or key.startswith("target_tokens/"):
            metrics[key] = value
            continue
        weight_key = _jax_token_metric_weight_key(key)
        metrics[key] = value / max(totals[weight_key], 1.0)
    return metrics


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
) -> dict[str, float]:
    probe_metrics = run_msg_probe_jax(
        config=config,
        model=model,
        data_mesh=None,
        online_maccs_only=False,
    )
    if jax.process_index() != 0:
        return probe_metrics
    log_msg_probe_metrics(
        logger,
        probe_metrics,
        global_step,
        enable_wandb=bool(config.get("enable_wandb", False)),
    )
    for variant in variants:
        prefix = f"msg_probe/{variant}"
        epoch_key = f"{prefix}/epoch"
        if epoch_key in probe_metrics:
            logging.info(
                "step=%d msg_probe[%s] best_epoch=%.2f test_auc_fluorine=%.4f",
                global_step,
                variant,
                probe_metrics[epoch_key],
                probe_metrics[f"{prefix}/test/auc_fluorine"],
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


def _jax_data_parallel_devices(config: Any) -> int:
    requested = config.get("jax_mesh_devices", None)
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
    if isinstance(first, torch.Tensor):
        return np.stack([value.detach().cpu().numpy() for value in values])
    return jnp.stack(values)


def _shutdown_torch_loader_iterator(loader_iter: Any) -> None:
    shutdown_workers = getattr(loader_iter, "_shutdown_workers", None)
    if callable(shutdown_workers):
        shutdown_workers()


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def _config_or_env(
    config: Any,
    key: str,
    env_keys: tuple[str, ...],
    default: Any = "",
) -> Any:
    value = config.get(key, None)
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
