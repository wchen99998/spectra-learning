from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import optax
import torch
from flax import nnx
from jax.sharding import Mesh, PartitionSpec as P
from ml_collections import config_dict
from tqdm import tqdm

from spectra_learning.data.gems.datamodule import GemsNativeDataModule
from spectra_learning.models.common_jax import Array, batch_to_jax
from spectra_learning.models.factory_jax import build_model_from_config
from spectra_learning.models.model_jax import PeakSetJEPAJax
from spectra_learning.training.checkpointing import training_checkpoint_paths
from spectra_learning.training.logging import MetricLogger
from spectra_learning.training.storage import (
    local_scratch_dir,
    normalize_storage_path,
    storage_join,
    storage_mkdir,
)


JAX_NNX_DEVICE_BACKENDS = {"jax_nnx", "nnx", "flax", "flax_nnx", "jax_native"}
JAX_DATA_AXIS = "data"
JAX_DATA_MESH = Mesh(jax.devices(), (JAX_DATA_AXIS,))


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


def use_jax_nnx_backend(config: Any) -> bool:
    backend = str(_config_get(config, "device_backend", "auto")).lower()
    return backend in JAX_NNX_DEVICE_BACKENDS or bool(
        _config_get(config, "use_jax_nnx", False)
    )


def build_jax_optimizer(config: Any, model: PeakSetJEPAJax) -> nnx.Optimizer:
    grad_accum_steps = int(_config_get(config, "gradient_accumulation_steps", 1))
    optimizer = build_jax_optax_transform(config)
    use_multistep = (
        bool(_config_get(config, "jax_optax_multistep_accumulation", False))
        and _jax_data_parallel_devices(config) == 1
    )
    if use_multistep:
        optimizer = optax.MultiSteps(
            optimizer,
            grad_accum_steps,
            use_grad_mean=True,
        ).gradient_transformation()
    return nnx.Optimizer(model, optimizer, wrt=trainable_param_filter)


def build_jax_optax_transform(config: Any) -> optax.GradientTransformation:
    return optax.adamw(
        learning_rate=float(_config_get(config, "learning_rate", 1e-3)),
        b2=float(_config_get(config, "b2", 0.999)),
        weight_decay=float(_config_get(config, "weight_decay", 0.0)),
    )


def torch_batch_to_jax(batch: dict[str, torch.Tensor]) -> dict[str, Array]:
    return batch_to_jax(batch)


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


@nnx.jit
def jax_multistep_train_step(
    model: PeakSetJEPAJax,
    optimizer: nnx.Optimizer,
    batch: dict[str, Array],
) -> tuple[dict[str, Array], Array]:
    def loss_fn(model: PeakSetJEPAJax):
        metrics = model(batch)
        return metrics["loss"], metrics

    (_loss, metrics), grads = nnx.value_and_grad(
        loss_fn,
        has_aux=True,
        argnums=nnx.DiffState(0, trainable_param_filter),
    )(model)
    optimizer.update(model, grads)
    return metrics, optimizer.step[...]


@nnx.jit
def jax_local_grad_step(
    model: PeakSetJEPAJax,
    batch: dict[str, Array],
) -> nnx.State:
    def loss_fn(model: PeakSetJEPAJax):
        return model(batch, loss_only=True)["loss"]

    return nnx.grad(
        loss_fn,
        argnums=nnx.DiffState(0, trainable_param_filter),
    )(model)


@nnx.jit
def jax_accumulate_local_grads(
    model: PeakSetJEPAJax,
    batch: dict[str, Array],
    accumulated_grads: nnx.State,
) -> nnx.State:
    grads = jax_local_grad_step(model, batch)
    return jax.tree.map(lambda lhs, rhs: lhs + rhs, accumulated_grads, grads)


@nnx.jit
def jax_apply_accumulated_train_step(
    model: PeakSetJEPAJax,
    optimizer: nnx.Optimizer,
    batch: dict[str, Array],
    accumulated_grads: nnx.State,
    accumulation_scale: Array,
) -> tuple[dict[str, Array], Array]:
    def loss_fn(model: PeakSetJEPAJax):
        return model(batch, loss_only=True)["loss"]

    loss, grads = nnx.value_and_grad(
        loss_fn,
        argnums=nnx.DiffState(0, trainable_param_filter),
    )(model)
    grads = jax.tree.map(
        lambda lhs, rhs: (lhs + rhs) * accumulation_scale,
        accumulated_grads,
        grads,
    )
    optimizer.update(model, grads)
    return {"loss": loss}, optimizer.step[...]


@nnx.jit(donate_argnums=(0, 1))
def jax_accumulated_train_step(
    model: PeakSetJEPAJax,
    optimizer: nnx.Optimizer,
    batch: dict[str, Array],
) -> tuple[dict[str, Array], Array]:
    metrics, grads = _accumulated_metrics_and_grads(model, batch)
    optimizer.update(model, grads)
    return metrics, optimizer.step[...]


@nnx.jit
@nnx.shard_map(
    mesh=JAX_DATA_MESH,
    in_specs=(P(), P(JAX_DATA_AXIS)),
    out_specs=(P(), P()),
    axis_names={JAX_DATA_AXIS},
    check_vma=False,
)
def jax_sharded_grad_step(
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
    metrics = jax.tree.map(lambda value: jax.lax.pmean(value, JAX_DATA_AXIS), metrics)
    grads = jax.tree.map(lambda value: jax.lax.pmean(value, JAX_DATA_AXIS), grads)
    return (loss, metrics), grads


@nnx.jit
@nnx.shard_map(
    mesh=JAX_DATA_MESH,
    in_specs=(P(), P(JAX_DATA_AXIS)),
    out_specs=P(),
    axis_names={JAX_DATA_AXIS},
    check_vma=False,
)
def jax_sharded_grad_step_grads_only(
    model: PeakSetJEPAJax,
    batch: dict[str, Array],
) -> nnx.State:
    def loss_fn(model: PeakSetJEPAJax):
        return model(batch, loss_only=True)["loss"]

    grads = nnx.grad(
        loss_fn,
        argnums=nnx.DiffState(0, trainable_param_filter),
    )(model)
    return jax.tree.map(lambda value: jax.lax.pmean(value, JAX_DATA_AXIS), grads)


@nnx.jit
@nnx.shard_map(
    mesh=JAX_DATA_MESH,
    in_specs=(P(), P(JAX_DATA_AXIS)),
    out_specs=P(),
    axis_names={JAX_DATA_AXIS},
    check_vma=False,
)
def jax_sharded_local_grad_step(
    model: PeakSetJEPAJax,
    batch: dict[str, Array],
) -> nnx.State:
    def loss_fn(model: PeakSetJEPAJax):
        return model(batch, loss_only=True)["loss"]

    return nnx.grad(
        loss_fn,
        argnums=nnx.DiffState(0, trainable_param_filter),
    )(model)


@nnx.jit
@nnx.shard_map(
    mesh=JAX_DATA_MESH,
    in_specs=(P(), P(JAX_DATA_AXIS), P()),
    out_specs=P(),
    axis_names={JAX_DATA_AXIS},
    check_vma=False,
)
def jax_sharded_accumulate_local_grads(
    model: PeakSetJEPAJax,
    batch: dict[str, Array],
    accumulated_grads: nnx.State,
) -> nnx.State:
    def loss_fn(model: PeakSetJEPAJax):
        return model(batch, loss_only=True)["loss"]

    grads = nnx.grad(
        loss_fn,
        argnums=nnx.DiffState(0, trainable_param_filter),
    )(model)
    return jax.tree.map(lambda lhs, rhs: lhs + rhs, accumulated_grads, grads)


@nnx.jit
@nnx.shard_map(
    mesh=JAX_DATA_MESH,
    in_specs=(P(), P(), P()),
    out_specs=P(),
    axis_names={JAX_DATA_AXIS},
)
def jax_sharded_apply_grads(
    model: PeakSetJEPAJax,
    optimizer: nnx.Optimizer,
    grads: nnx.State,
) -> Array:
    optimizer.update(model, grads)
    return optimizer.step[...]


@nnx.jit
@nnx.shard_map(
    mesh=JAX_DATA_MESH,
    in_specs=(P(), P(), P(JAX_DATA_AXIS), P(), P()),
    out_specs=(P(), P()),
    axis_names={JAX_DATA_AXIS},
    check_vma=False,
)
def jax_sharded_apply_accumulated_train_step(
    model: PeakSetJEPAJax,
    optimizer: nnx.Optimizer,
    batch: dict[str, Array],
    accumulated_grads: nnx.State,
    accumulation_scale: Array,
) -> tuple[dict[str, Array], Array]:
    def loss_fn(model: PeakSetJEPAJax):
        return model(batch, loss_only=True)["loss"]

    loss, grads = nnx.value_and_grad(
        loss_fn,
        argnums=nnx.DiffState(0, trainable_param_filter),
    )(model)
    grads = jax.tree.map(
        lambda lhs, rhs: (lhs + rhs) * accumulation_scale,
        accumulated_grads,
        grads,
    )
    grads = jax.tree.map(lambda value: jax.lax.pmean(value, JAX_DATA_AXIS), grads)
    metrics = {"loss": jax.lax.pmean(loss, JAX_DATA_AXIS)}
    optimizer.update(model, grads)
    return metrics, optimizer.step[...]


@nnx.jit(donate_argnums=(0, 1))
@nnx.shard_map(
    mesh=JAX_DATA_MESH,
    in_specs=(P(), P(), P(None, JAX_DATA_AXIS)),
    out_specs=(P(), P()),
    axis_names={JAX_DATA_AXIS},
    check_vma=False,
)
def jax_sharded_accumulated_train_step(
    model: PeakSetJEPAJax,
    optimizer: nnx.Optimizer,
    batch: dict[str, Array],
) -> tuple[dict[str, Array], Array]:
    metrics, grads = _accumulated_metrics_and_grads(model, batch)
    metrics = jax.tree.map(lambda value: jax.lax.pmean(value, JAX_DATA_AXIS), metrics)
    grads = jax.tree.map(lambda value: jax.lax.pmean(value, JAX_DATA_AXIS), grads)
    optimizer.update(model, grads)
    return metrics, optimizer.step[...]


def _accumulated_metrics_and_grads(
    model: PeakSetJEPAJax,
    batch: dict[str, Array],
) -> tuple[dict[str, Array], nnx.State]:
    graphdef, trainable_params, static_state = nnx.split(
        model,
        trainable_param_filter,
        ...,
    )

    def loss_fn(params: nnx.State, micro_batch: dict[str, Array]):
        functional_model = nnx.merge(graphdef, params, static_state)
        return functional_model(micro_batch, loss_only=True)["loss"]

    def micro_batch_grad(micro_batch: dict[str, Array]):
        return jax.value_and_grad(loss_fn)(
            trainable_params,
            micro_batch,
        )

    first_batch = jax.tree.map(lambda value: value[0], batch)
    remaining_batches = jax.tree.map(lambda value: value[1:], batch)
    loss, grads = micro_batch_grad(first_batch)

    def scan_body(carry: tuple[nnx.State, Array], micro_batch):
        grad_accumulator, loss_accumulator = carry
        micro_loss, micro_grads = micro_batch_grad(micro_batch)
        grad_accumulator = jax.tree.map(
            lambda lhs, rhs: lhs + rhs,
            grad_accumulator,
            micro_grads,
        )
        return (grad_accumulator, loss_accumulator + micro_loss), None

    (grads, loss), _ = jax.lax.scan(
        scan_body,
        (grads, loss),
        remaining_batches,
    )
    num_micro_batches = float(jax.tree.leaves(batch)[0].shape[0])
    grads = jax.tree.map(lambda value: value / num_micro_batches, grads)
    loss = loss / num_micro_batches
    return {"loss": loss}, grads


def init_pure_optax_train_state(
    config: Any,
    model: PeakSetJEPAJax,
) -> tuple[Any, nnx.State, nnx.State, Any, optax.GradientTransformation]:
    graphdef, trainable_params, static_state = nnx.split(
        model,
        trainable_param_filter,
        ...,
    )
    trainable_params = nnx.as_pure(trainable_params)
    static_state = nnx.as_pure(static_state)
    optimizer = build_jax_optax_transform(config)
    opt_state = optimizer.init(trainable_params)
    return graphdef, trainable_params, static_state, opt_state, optimizer


def make_pure_accumulated_train_step(
    graphdef: Any,
    optimizer: optax.GradientTransformation,
    *,
    sharded: bool,
    scan_zero_init: bool = False,
):
    def accumulated_metrics_and_grads(
        trainable_params: nnx.State,
        static_state: nnx.State,
        batch: dict[str, Array],
    ) -> tuple[dict[str, Array], nnx.State]:
        def loss_fn(params: nnx.State, micro_batch: dict[str, Array]):
            functional_model = nnx.merge(graphdef, params, static_state)
            return functional_model(micro_batch, loss_only=True)["loss"]

        def micro_batch_grad(micro_batch: dict[str, Array]):
            return jax.value_and_grad(loss_fn)(
                trainable_params,
                micro_batch,
            )

        def scan_body(carry: tuple[nnx.State, Array], micro_batch):
            grad_accumulator, loss_accumulator = carry
            micro_loss, micro_grads = micro_batch_grad(micro_batch)
            grad_accumulator = jax.tree.map(
                lambda lhs, rhs: lhs + rhs,
                grad_accumulator,
                micro_grads,
            )
            return (grad_accumulator, loss_accumulator + micro_loss), None

        if scan_zero_init:
            grads = jax.tree.map(jnp.zeros_like, trainable_params)
            loss = jnp.asarray(0.0, dtype=jnp.float32)
            (grads, loss), _ = jax.lax.scan(
                scan_body,
                (grads, loss),
                batch,
            )
            num_micro_batches = float(jax.tree.leaves(batch)[0].shape[0])
            grads = jax.tree.map(lambda value: value / num_micro_batches, grads)
            loss = loss / num_micro_batches
            return {"loss": loss}, grads

        first_batch = jax.tree.map(lambda value: value[0], batch)
        remaining_batches = jax.tree.map(lambda value: value[1:], batch)
        loss, grads = micro_batch_grad(first_batch)

        (grads, loss), _ = jax.lax.scan(
            scan_body,
            (grads, loss),
            remaining_batches,
        )
        num_micro_batches = float(jax.tree.leaves(batch)[0].shape[0])
        grads = jax.tree.map(lambda value: value / num_micro_batches, grads)
        loss = loss / num_micro_batches
        return {"loss": loss}, grads

    if sharded:

        @jax.jit(donate_argnums=(0, 2))
        @jax.shard_map(
            mesh=JAX_DATA_MESH,
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


def train_and_evaluate_jax(
    config: config_dict.ConfigDict,
    workdir: str | Path,
) -> dict[str, object]:
    workdir = normalize_storage_path(workdir)
    local_workdir = local_scratch_dir(workdir)
    local_workdir.mkdir(parents=True, exist_ok=True)
    storage_mkdir(workdir)
    torch.manual_seed(int(config.seed))
    datamodule = GemsNativeDataModule(config, seed=int(config.seed))
    total_steps = _total_training_steps(config, datamodule)
    model = build_model_from_config(config)
    checkpoint_dir = storage_join(workdir, "checkpoints")
    storage_mkdir(checkpoint_dir)
    checkpoints = training_checkpoint_paths(checkpoint_dir)
    if checkpoints:
        model.load_torch_checkpoint(checkpoints[-1])
    optimizer = build_jax_optimizer(config, model)
    logger = MetricLogger()
    metrics = _run_jax_training_loop(
        config=config,
        datamodule=datamodule,
        model=model,
        optimizer=optimizer,
        logger=logger,
        total_steps=total_steps,
    )
    data_parallel_devices = _jax_data_parallel_devices(config)
    return {
        **metrics,
        "run/world_size": float(data_parallel_devices),
        "run/jax_device_count": float(jax.device_count()),
        "run/global_batch_size": float(datamodule.global_batch_size),
        "run/local_batch_size": float(datamodule.batch_size),
        "run/device_microbatch_size": float(
            datamodule.batch_size // data_parallel_devices
        ),
        "run/gradient_accumulation_steps": float(
            int(_config_get(config, "gradient_accumulation_steps", 1))
        ),
        "run/device_backend": "jax_nnx",
    }


def _run_jax_training_loop(
    *,
    config: config_dict.ConfigDict,
    datamodule: GemsNativeDataModule,
    model: PeakSetJEPAJax,
    optimizer: nnx.Optimizer,
    logger: MetricLogger,
    total_steps: int,
) -> dict[str, object]:
    del logger
    log_every_n_steps = int(_config_get(config, "log_every_n_steps", 50))
    warmup_steps = int(_config_get(config, "throughput_warmup_steps", 0))
    grad_accum_steps = int(_config_get(config, "gradient_accumulation_steps", 1))
    use_sharded_step = _jax_data_parallel_devices(config) > 1
    use_optax_multistep = (
        bool(
            _config_get(
                config,
                "jax_optax_multistep_accumulation",
                False,
            )
        )
        and grad_accum_steps > 1
        and not use_sharded_step
    )
    use_compiled_accumulation = grad_accum_steps > 1 and not use_optax_multistep
    use_scan_accumulation = (
        bool(_config_get(config, "jax_scan_accumulation", False))
        and use_compiled_accumulation
    )
    use_device_accumulation = (
        bool(_config_get(config, "jax_device_accumulation", True))
        and use_compiled_accumulation
        and not use_scan_accumulation
    )
    use_pure_optax_step = (
        bool(_config_get(config, "jax_pure_optax_step", False))
        and use_scan_accumulation
        and not model.use_ema_teacher
    )
    pure_trainable_params = None
    pure_static_state = None
    pure_opt_state = None
    pure_train_step = None
    pure_full_static_state = None
    pure_full_train_step = None
    context_encoder_pack_tokens = int(
        getattr(model, "mae_context_encoder_pack_tokens", 0)
    )
    if use_pure_optax_step:
        if context_encoder_pack_tokens > 0:
            model.mae_context_encoder_pack_tokens = 0
            (
                pure_full_graphdef,
                _full_trainable_params,
                pure_full_static_state,
                _full_opt_state,
                _full_optimizer,
            ) = init_pure_optax_train_state(config, model)
            model.mae_context_encoder_pack_tokens = context_encoder_pack_tokens
        (
            pure_graphdef,
            pure_trainable_params,
            pure_static_state,
            pure_opt_state,
            pure_optimizer,
        ) = init_pure_optax_train_state(config, model)
        scan_zero_init = bool(_config_get(config, "jax_scan_zero_init", False))
        pure_train_step = make_pure_accumulated_train_step(
            pure_graphdef,
            pure_optimizer,
            sharded=use_sharded_step,
            scan_zero_init=scan_zero_init,
        )
        if context_encoder_pack_tokens > 0:
            pure_full_train_step = make_pure_accumulated_train_step(
                pure_full_graphdef,
                pure_optimizer,
                sharded=use_sharded_step,
                scan_zero_init=scan_zero_init,
            )
    timing_barriers = bool(_config_get(config, "jax_timing_barriers", False))
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
    loop_epochs = max(1, math.ceil(float(config.num_epochs)))
    global_step = 0
    accumulation_step = 0
    accumulated_grads = None
    last_metrics: dict[str, Array] = {}
    train_start = time.perf_counter()
    measured_start: float | None = None
    measured_steps = 0
    context_encoder_packed_steps = 0
    context_encoder_full_fallback_steps = 0
    measured_context_encoder_packed_steps = 0
    measured_context_encoder_full_fallback_steps = 0
    for epoch in range(loop_epochs):
        loader = datamodule.train_loader_for_epoch(epoch)
        loader_iter = iter(loader)
        pbar = tqdm(
            total=min(datamodule.train_steps, total_steps - global_step),
            desc=f"Epoch {epoch}",
            unit="step",
        )
        while global_step < total_steps:
            if use_scan_accumulation:
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
                        context_count = (
                            (
                                torch_batch["context_mask"]
                                & torch_batch["peak_valid_mask"]
                            )
                            .sum(dim=1)
                            .max()
                            .item()
                        )
                        max_context_count = max(max_context_count, int(context_count))
                    transfer_start = time.perf_counter()
                    micro_batches.append(torch_batch_to_jax(torch_batch))
                    if timing_barriers:
                        jax.block_until_ready(micro_batches[-1])
                    transfer_elapsed += time.perf_counter() - transfer_start
                if len(micro_batches) < grad_accum_steps:
                    break
                batch = _stack_micro_batches(micro_batches)
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
                    jax.effects_barrier()
                    jax.profiler.start_trace(profile_dir)
                    profile_started = True
                    profile_active = True
                step_start = time.perf_counter()
                if use_pure_optax_step:
                    selected_train_step = pure_train_step
                    selected_static_state = pure_static_state
                    used_context_full_fallback = False
                    if (
                        pure_full_train_step is not None
                        and max_context_count > context_encoder_pack_tokens
                    ):
                        selected_train_step = pure_full_train_step
                        selected_static_state = pure_full_static_state
                        used_context_full_fallback = True
                    if pure_full_train_step is not None:
                        if used_context_full_fallback:
                            context_encoder_full_fallback_steps += 1
                            if timing_enabled:
                                measured_context_encoder_full_fallback_steps += 1
                        else:
                            context_encoder_packed_steps += 1
                            if timing_enabled:
                                measured_context_encoder_packed_steps += 1
                    pure_trainable_params, pure_opt_state, metrics = selected_train_step(
                        pure_trainable_params,
                        selected_static_state,
                        pure_opt_state,
                        batch,
                    )
                    if timing_barriers:
                        jax.block_until_ready(
                            (pure_trainable_params, pure_opt_state, metrics)
                        )
                else:
                    metrics, apply_token = (
                        jax_sharded_accumulated_train_step(model, optimizer, batch)
                        if use_sharded_step
                        else jax_accumulated_train_step(model, optimizer, batch)
                    )
                    if timing_barriers:
                        jax.block_until_ready((metrics, apply_token))
                if timing_enabled:
                    timing["compiled_step_seconds"] += (
                        time.perf_counter() - step_start
                    )
                if not use_pure_optax_step:
                    ema_momentum = model.update_ema_teacher(global_step + 1, total_steps)
                    if ema_momentum is not None:
                        metrics["ema_teacher_momentum"] = jnp.asarray(ema_momentum)
                last_metrics = metrics
                global_step += 1
                if measured_start is not None:
                    measured_steps += 1
                pbar.update(1)
                if log_every_n_steps > 0 and global_step % log_every_n_steps == 0:
                    pbar.set_postfix(
                        loss=f"{float(jax.device_get(metrics['loss'])):.4f}",
                        step=global_step,
                    )
                if profile_active and global_step >= profile_end_step:
                    jax.effects_barrier()
                    jax.profiler.stop_trace()
                    profile_active = False
                continue
            dataloader_start = time.perf_counter()
            try:
                torch_batch = next(loader_iter)
            except StopIteration:
                break
            dataloader_elapsed = time.perf_counter() - dataloader_start
            if global_step >= total_steps:
                break
            timing_enabled = measured_start is not None
            if timing_enabled:
                timing["dataloader_seconds"] += dataloader_elapsed
                timing["measured_microbatches"] += 1.0
            transfer_start = time.perf_counter()
            batch = torch_batch_to_jax(torch_batch)
            if timing_barriers:
                jax.block_until_ready(batch)
            transfer_elapsed = time.perf_counter() - transfer_start
            if timing_enabled:
                timing["transfer_seconds"] += transfer_elapsed
            if measured_start is None and global_step >= warmup_steps:
                jax.effects_barrier()
                measured_start = time.perf_counter()
                timing_enabled = True
            if (
                profile_dir
                and not profile_started
                and global_step >= profile_start_step
            ):
                jax.effects_barrier()
                jax.profiler.start_trace(profile_dir)
                profile_started = True
                profile_active = True
            if use_optax_multistep:
                step_start = time.perf_counter()
                metrics, apply_token = jax_multistep_train_step(
                    model,
                    optimizer,
                    batch,
                )
                if timing_barriers:
                    jax.block_until_ready((metrics, apply_token))
                if timing_enabled:
                    timing["grad_seconds"] += time.perf_counter() - step_start
                accumulation_step += 1
                if accumulation_step % grad_accum_steps != 0:
                    continue
                accumulation_step = 0
                ema_momentum = model.update_ema_teacher(global_step + 1, total_steps)
                if ema_momentum is not None:
                    metrics["ema_teacher_momentum"] = jnp.asarray(ema_momentum)
                last_metrics = metrics
                global_step += 1
                if measured_start is not None:
                    measured_steps += 1
                pbar.update(1)
                if log_every_n_steps > 0 and global_step % log_every_n_steps == 0:
                    pbar.set_postfix(
                        loss=f"{float(jax.device_get(metrics['loss'])):.4f}",
                        step=global_step,
                    )
                if profile_active and global_step >= profile_end_step:
                    jax.effects_barrier()
                    jax.profiler.stop_trace()
                    profile_active = False
                continue
            next_micro_step_is_boundary = (
                (accumulation_step + 1) % grad_accum_steps == 0
            )
            if use_device_accumulation:
                step_start = time.perf_counter()
                if next_micro_step_is_boundary:
                    scale = jnp.asarray(1.0 / float(grad_accum_steps), dtype=jnp.float32)
                    metrics, apply_token = (
                        jax_sharded_apply_accumulated_train_step(
                            model,
                            optimizer,
                            batch,
                            accumulated_grads,
                            scale,
                        )
                        if use_sharded_step
                        else jax_apply_accumulated_train_step(
                            model,
                            optimizer,
                            batch,
                            accumulated_grads,
                            scale,
                        )
                    )
                    if timing_barriers:
                        jax.block_until_ready((metrics, apply_token))
                    if timing_enabled:
                        timing["compiled_step_seconds"] += (
                            time.perf_counter() - step_start
                        )
                    ema_momentum = model.update_ema_teacher(
                        global_step + 1,
                        total_steps,
                    )
                    if ema_momentum is not None:
                        metrics["ema_teacher_momentum"] = jnp.asarray(ema_momentum)
                    accumulated_grads = None
                    accumulation_step = 0
                    last_metrics = metrics
                    global_step += 1
                    if measured_start is not None:
                        measured_steps += 1
                    pbar.update(1)
                    if log_every_n_steps > 0 and global_step % log_every_n_steps == 0:
                        pbar.set_postfix(
                            loss=f"{float(jax.device_get(metrics['loss'])):.4f}",
                            step=global_step,
                        )
                    if profile_active and global_step >= profile_end_step:
                        jax.effects_barrier()
                        jax.profiler.stop_trace()
                        profile_active = False
                    continue
                accumulated_grads = (
                    (
                        jax_sharded_local_grad_step(model, batch)
                        if use_sharded_step
                        else jax_local_grad_step(model, batch)
                    )
                    if accumulated_grads is None
                    else (
                        jax_sharded_accumulate_local_grads(
                            model,
                            batch,
                            accumulated_grads,
                        )
                        if use_sharded_step
                        else jax_accumulate_local_grads(
                            model,
                            batch,
                            accumulated_grads,
                        )
                    )
                )
                if timing_barriers:
                    jax.block_until_ready(accumulated_grads)
                if timing_enabled:
                    timing["grad_seconds"] += time.perf_counter() - step_start
                accumulation_step += 1
                continue
            grad_start = time.perf_counter()
            metrics = None
            if use_sharded_step and not next_micro_step_is_boundary:
                grads = jax_sharded_grad_step_grads_only(model, batch)
                if timing_barriers:
                    jax.block_until_ready(grads)
            else:
                (_loss, metrics), grads = (
                    jax_sharded_grad_step(model, batch)
                    if use_sharded_step
                    else jax_grad_step(model, batch)
                )
                if timing_barriers:
                    jax.block_until_ready((metrics, grads))
            if timing_barriers:
                jax.block_until_ready(grads)
            grad_elapsed = time.perf_counter() - grad_start
            if timing_enabled:
                timing["grad_seconds"] += grad_elapsed
            accumulated_grads = (
                grads
                if accumulated_grads is None
                else _timed_tree_map(
                    lambda lhs, rhs: lhs + rhs,
                    accumulated_grads,
                    grads,
                )
            )
            if timing_barriers:
                jax.block_until_ready(accumulated_grads)
            accumulate_elapsed = time.perf_counter() - grad_start - grad_elapsed
            if timing_enabled:
                timing["accumulate_seconds"] += accumulate_elapsed
            accumulation_step += 1
            if accumulation_step % grad_accum_steps != 0:
                continue
            if grad_accum_steps > 1:
                accumulate_start = time.perf_counter()
                accumulated_grads = jax.tree.map(
                    lambda grad: grad / float(grad_accum_steps),
                    accumulated_grads,
                )
                if timing_barriers:
                    jax.block_until_ready(accumulated_grads)
                if timing_enabled:
                    timing["accumulate_seconds"] += (
                        time.perf_counter() - accumulate_start
                    )
            apply_start = time.perf_counter()
            if use_sharded_step:
                apply_token = jax_sharded_apply_grads(model, optimizer, accumulated_grads)
            else:
                apply_token = jax_apply_grads(model, optimizer, accumulated_grads)
            if timing_barriers:
                jax.block_until_ready(apply_token)
            if timing_enabled:
                timing["apply_seconds"] += time.perf_counter() - apply_start
            ema_momentum = model.update_ema_teacher(global_step + 1, total_steps)
            if ema_momentum is not None:
                metrics["ema_teacher_momentum"] = jnp.asarray(ema_momentum)
            accumulated_grads = None
            accumulation_step = 0
            last_metrics = metrics
            global_step += 1
            if measured_start is not None:
                measured_steps += 1
            pbar.update(1)
            if log_every_n_steps > 0 and global_step % log_every_n_steps == 0:
                pbar.set_postfix(
                    loss=f"{float(jax.device_get(metrics['loss'])):.4f}",
                    step=global_step,
                )
            if profile_active and global_step >= profile_end_step:
                jax.effects_barrier()
                jax.profiler.stop_trace()
                profile_active = False
        pbar.close()
        if global_step >= total_steps:
            break
    if use_pure_optax_step:
        jax.block_until_ready(pure_trainable_params)
        nnx.update(model, pure_trainable_params)
    jax.effects_barrier()
    if profile_active:
        jax.profiler.stop_trace()
    elapsed = time.perf_counter() - train_start
    measured_elapsed = (
        time.perf_counter() - measured_start if measured_start is not None else 0.0
    )
    global_batch_size = int(datamodule.global_batch_size)
    loss = float(jax.device_get(last_metrics["loss"])) if last_metrics else float("nan")
    result = {
        "run/final_global_step": float(global_step),
        "run/train_elapsed_seconds": elapsed,
        "run/steps_per_second": float(global_step) / elapsed if elapsed > 0 else 0.0,
        "run/samples_per_second": (
            float(global_step) * global_batch_size / elapsed if elapsed > 0 else 0.0
        ),
        "run/measured_steps": float(measured_steps),
        "run/measured_elapsed_seconds": measured_elapsed,
        "run/measured_steps_per_second": (
            float(measured_steps) / measured_elapsed if measured_elapsed > 0 else 0.0
        ),
        "run/measured_samples_per_second": (
            float(measured_steps) * global_batch_size / measured_elapsed
            if measured_elapsed > 0
            else 0.0
        ),
        "train/loss": loss,
    }
    if context_encoder_pack_tokens > 0:
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
    requested = _config_get(config, "jax_mesh_devices", jax.device_count())
    if isinstance(requested, str):
        return jax.device_count() if requested.lower() == "all" else int(requested)
    return int(requested)


def _timed_tree_map(fn: Any, *trees: Any) -> Any:
    return jax.tree.map(fn, *trees)


def _stack_micro_batches(batches: list[dict[str, Array]]) -> dict[str, Array]:
    return jax.tree.map(lambda *values: jnp.stack(values), *batches)


def _config_get(config: Any, key: str, default: Any) -> Any:
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)
