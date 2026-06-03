from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import optax
import torch
import torchax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from ml_collections import config_dict
from torchax import interop
from tqdm import tqdm

from spectra_learning.data.gems.datamodule import GemsNativeDataModule
from spectra_learning.training.batch import TRAIN_BATCH_KEYS
from spectra_learning.training.checkpointing import save_torch_checkpoint
from spectra_learning.training.distributed import DistributedContext
from spectra_learning.training.logging import MetricLogger, build_logger
from spectra_learning.training.optimization import is_weight_decay_target
from spectra_learning.training.pretrain import (
    _config_get,
    initialize_frozen_teacher,
    install_stop_signal_handlers,
    seed_all,
    total_training_steps,
)
from spectra_learning.training.storage import (
    StoragePath,
    local_scratch_dir,
    normalize_storage_path,
    storage_join,
    storage_mkdir,
)

_WARMUP_START_FACTOR = 1e-8


def train_and_evaluate_torchax(
    config: config_dict.ConfigDict,
    workdir: str | Path,
) -> dict[str, object]:
    install_stop_signal_handlers()
    torchax.enable_globally()
    torchax.enable_performance_mode()
    workdir = normalize_storage_path(workdir)
    local_workdir = local_scratch_dir(workdir)
    local_workdir.mkdir(parents=True, exist_ok=True)
    storage_mkdir(workdir)
    seed_all(int(config.seed))
    datamodule = GemsNativeDataModule(config, seed=int(config.seed))
    total_steps = total_training_steps(config, datamodule)
    loop_epochs = max(1, math.ceil(float(config.num_epochs)))
    logging.info("TorchAX devices: %s", jax.devices())
    logging.info("Training for %s epochs (%d steps).", config.num_epochs, total_steps)
    logging.info("Steps per epoch: %d", datamodule.train_steps)
    mesh = jax.make_mesh((jax.device_count(),), ("data",))
    model = _build_torchax_model(config)
    weights, static_state = _split_trainable_and_static_state(model)
    with jax.set_mesh(mesh):
        weights = _replicate_tree(weights, mesh)
        static_state = _replicate_tree(static_state, mesh)
        optimizer = _build_optax_optimizer(config, weights, total_steps)
        opt_state = interop.call_jax(optimizer.init, weights)
        train_step = _make_train_step(model, optimizer)
        checkpoint_dir = storage_join(workdir, "checkpoints")
        storage_mkdir(checkpoint_dir)
        logger = build_logger(config, local_workdir) if bool(
            _config_get(config, "enable_wandb", False)
        ) else MetricLogger()
        metrics, weights = _run_torchax_loop(
            config=config,
            datamodule=datamodule,
            weights=weights,
            static_state=static_state,
            opt_state=opt_state,
            train_step=train_step,
            logger=logger,
            checkpoint_dir=checkpoint_dir,
            loop_epochs=loop_epochs,
            total_steps=total_steps,
            mesh=mesh,
        )
    return metrics


def _build_torchax_model(config: config_dict.ConfigDict) -> torch.nn.Module:
    from spectra_learning.training.api import build_model_from_config

    model = build_model_from_config(config)
    initialize_frozen_teacher(
        config,
        model,
        distributed=DistributedContext(
            rank=0,
            local_rank=0,
            world_size=1,
            device=torch.device("cpu"),
        ),
    )
    return model.to("jax").train()


def _split_trainable_and_static_state(
    model: torch.nn.Module,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    state_dict_keys = set(model.state_dict())
    weights = {
        name: param
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    static_state = {
        name: param
        for name, param in model.named_parameters()
        if not param.requires_grad
    }
    static_state.update(
        {
            name: buffer
            for name, buffer in model.named_buffers()
            if name in state_dict_keys
        }
    )
    return weights, static_state


def _make_train_step(
    model: torch.nn.Module,
    optimizer: optax.GradientTransformation,
):
    def loss_and_metrics(
        weights: dict[str, torch.Tensor],
        static_state: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        metrics = torch.func.functional_call(model, (weights, static_state), (batch,))
        return metrics["loss"], metrics

    grad_fn = interop.jax_value_and_grad(
        loss_and_metrics,
        kwargs_for_value_and_grad={"has_aux": True},
    )

    def step(weights, static_state, opt_state, batch):
        (loss, metrics), grads = grad_fn(weights, static_state, batch)
        updates, opt_state = interop.call_jax(
            optimizer.update,
            grads,
            opt_state,
            weights,
        )
        weights = interop.call_jax(optax.apply_updates, weights, updates)
        metrics["loss"] = loss
        return metrics, weights, opt_state

    return interop.jax_jit(step, kwargs_for_jax_jit={"donate_argnums": (0, 2)})


def _build_optax_optimizer(
    config: config_dict.ConfigDict,
    weights: dict[str, torch.Tensor],
    total_steps: int,
) -> optax.GradientTransformation:
    optimizer_type = str(_config_get(config, "optimizer", "adamw")).lower()
    transforms = []
    grad_clip_norm = _config_get(config, "grad_clip_norm", None)
    if grad_clip_norm is not None and float(grad_clip_norm) > 0:
        transforms.append(optax.clip_by_global_norm(float(grad_clip_norm)))
    if optimizer_type == "muon":
        transforms.append(_build_muon_optax_optimizer(config, weights, total_steps))
    elif optimizer_type == "adamw":
        transforms.append(
            optax.adamw(
                learning_rate=_optax_learning_rate_schedule(config, total_steps),
                b1=0.9,
                b2=float(_config_get(config, "b2", 0.999)),
                weight_decay=float(config.weight_decay),
                mask={
                    name: is_weight_decay_target(name, param)
                    for name, param in weights.items()
                },
            )
        )
    else:
        raise ValueError("optimizer must be one of ('adamw', 'muon')")
    return optax.chain(*transforms)


def _optax_learning_rate_schedule(
    config: config_dict.ConfigDict,
    total_steps: int,
    *,
    learning_rate: float | None = None,
):
    base_lr = float(config.learning_rate if learning_rate is None else learning_rate)
    raw_min_lr = _config_get(config, "min_learning_rate", None)
    min_lr = 0.1 * base_lr if raw_min_lr is None else float(raw_min_lr)
    warmup_steps = int(_config_get(config, "warmup_steps", 0))
    cosine_steps = max(1, total_steps - warmup_steps)

    def schedule(step):
        step = jnp.asarray(step, dtype=jnp.float32)
        progress = jnp.clip((step - warmup_steps) / cosine_steps, 0.0, 1.0)
        cosine = min_lr + 0.5 * (base_lr - min_lr) * (
            1.0 + jnp.cos(jnp.pi * progress)
        )
        warmup = base_lr * (
            _WARMUP_START_FACTOR
            + (1.0 - _WARMUP_START_FACTOR) * step / max(1, warmup_steps)
        )
        return jnp.where(step < warmup_steps, warmup, cosine)

    return schedule


def _build_muon_optax_optimizer(
    config: config_dict.ConfigDict,
    weights: dict[str, torch.Tensor],
    total_steps: int,
) -> optax.GradientTransformation:
    muon_lr = float(_config_get(config, "muon_lr", None) or config.learning_rate)
    adamw_lr = float(_config_get(config, "adamw_lr", None) or config.learning_rate)
    muon_weight_decay = float(
        _config_get(config, "muon_weight_decay", None) or config.weight_decay
    )
    return optax.contrib.muon(
        learning_rate=_optax_learning_rate_schedule(
            config,
            total_steps,
            learning_rate=muon_lr,
        ),
        weight_decay=muon_weight_decay,
        muon_weight_dimension_numbers={
            name: (
                optax.contrib.MuonDimensionNumbers()
                if is_weight_decay_target(name, param)
                else None
            )
            for name, param in weights.items()
        },
        adam_learning_rate=_optax_learning_rate_schedule(
            config,
            total_steps,
            learning_rate=adamw_lr,
        ),
        adam_b1=0.9,
        adam_b2=float(_config_get(config, "b2", 0.999)),
        adam_weight_decay=0.0,
        beta=float(_config_get(config, "muon_momentum", 0.95)),
        nesterov=bool(_config_get(config, "muon_nesterov", True)),
        ns_steps=int(_config_get(config, "muon_ns_steps", 5)),
        consistent_rms=_muon_consistent_rms(config),
    )


def _muon_consistent_rms(config: config_dict.ConfigDict) -> float | None:
    adjust_lr_fn = _config_get(config, "muon_adjust_lr_fn", "match_rms_adamw")
    if adjust_lr_fn is None or str(adjust_lr_fn) == "original":
        return None
    if str(adjust_lr_fn) == "match_rms_adamw":
        return 0.2
    raise ValueError("muon_adjust_lr_fn must be one of ('original', 'match_rms_adamw')")


def _run_torchax_loop(
    *,
    config: config_dict.ConfigDict,
    datamodule: GemsNativeDataModule,
    weights: dict[str, torch.Tensor],
    static_state: dict[str, torch.Tensor],
    opt_state: Any,
    train_step,
    logger,
    checkpoint_dir: StoragePath,
    loop_epochs: int,
    total_steps: int,
    mesh: Mesh,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    log_every_n_steps = int(_config_get(config, "log_every_n_steps", 50))
    checkpoint_every_steps = int(config.checkpoint_every_steps)
    global_step = 0
    last_metrics: dict[str, torch.Tensor] = {}
    training_start_time = time.perf_counter()
    for epoch in range(loop_epochs):
        train_loader = datamodule.train_loader_for_epoch(epoch)
        epoch_steps = min(datamodule.train_steps, total_steps - global_step)
        torchax.disable_globally()
        train_iter = iter(train_loader)
        torchax.enable_globally()
        pbar = tqdm(
            total=epoch_steps,
            desc=f"Epoch {epoch}",
            unit="step",
            disable=bool(_config_get(config, "disable_tqdm", False)),
        )
        for _ in range(epoch_steps):
            if global_step >= total_steps:
                break
            torchax.disable_globally()
            batch = next(train_iter)
            torchax.enable_globally()
            batch = _place_batch(batch, mesh)
            last_metrics, weights, opt_state = train_step(
                weights,
                static_state,
                opt_state,
                batch,
            )
            _block_until_ready(last_metrics["loss"])
            global_step += 1
            pbar.update(1)
            if log_every_n_steps > 0 and global_step % log_every_n_steps == 0:
                _log_torchax_train_metrics(
                    logger,
                    pbar,
                    last_metrics,
                    config=config,
                    total_steps=total_steps,
                    epoch=epoch,
                    global_step=global_step,
                )
            if checkpoint_every_steps > 0 and global_step % checkpoint_every_steps == 0:
                _save_torchax_checkpoint(
                    storage_join(checkpoint_dir, f"step-{global_step:08d}.pt"),
                    weights,
                    static_state,
                    global_step,
                    epoch,
                    _metric_float(last_metrics["loss"]),
                )
        pbar.close()
        if global_step >= total_steps:
            break
    _save_torchax_checkpoint(
        storage_join(checkpoint_dir, "last.pt"),
        weights,
        static_state,
        global_step,
        max(0, global_step // datamodule.train_steps),
        _metric_float(last_metrics["loss"]),
    )
    elapsed = time.perf_counter() - training_start_time
    results = {
        f"train/{key}": _metric_float(value)
        for key, value in last_metrics.items()
    }
    results.update(
        {
            "run/final_global_step": float(global_step),
            "run/world_size": float(jax.device_count()),
            "run/global_batch_size": float(datamodule.global_batch_size),
            "run/local_batch_size": float(datamodule.batch_size),
            "run/train_elapsed_seconds": elapsed,
            "run/steps_per_second": float(global_step) / elapsed if elapsed > 0 else 0.0,
            "run/samples_per_second": (
                float(global_step) * datamodule.global_batch_size / elapsed
                if elapsed > 0
                else 0.0
            ),
        }
    )
    return results, weights


def _place_batch(
    batch: dict[str, torch.Tensor],
    mesh: Mesh,
) -> dict[str, torch.Tensor]:
    env = torchax.default_env()
    axis_size = jax.device_count()
    return {
        key: _place_tensor(value, env, mesh, axis_size)
        for key, value in batch.items()
        if key in TRAIN_BATCH_KEYS and isinstance(value, torch.Tensor)
    }


def _place_tensor(
    tensor: torch.Tensor,
    env,
    mesh: Mesh,
    axis_size: int,
) -> torch.Tensor:
    array = env.t2j_copy(tensor)
    spec = (
        PartitionSpec("data")
        if axis_size > 1 and tensor.ndim > 0 and tensor.shape[0] % axis_size == 0
        else PartitionSpec()
    )
    return interop.torch_view_elem(jax.device_put(array, NamedSharding(mesh, spec)))


def _replicate_tree(
    tree,
    mesh: Mesh,
):
    sharding = NamedSharding(mesh, PartitionSpec())
    return interop.torch_view(
        jax.tree_util.tree_map(
            lambda value: jax.device_put(value, sharding),
            interop.jax_view(tree),
        )
    )


def _save_torchax_checkpoint(
    path: StoragePath,
    weights: dict[str, torch.Tensor],
    static_state: dict[str, torch.Tensor],
    global_step: int,
    epoch: int,
    loss: float,
) -> None:
    state = {}
    state.update(static_state)
    state.update(weights)
    save_torch_checkpoint(
        {
            "model": _cpu_state_dict(state),
            "global_step": global_step,
            "epoch": epoch,
            "loss": loss,
            "wandb_run_id": None,
        },
        path,
    )


def _cpu_state_dict(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        name: _torchax_tensor_to_cpu(value)
        for name, value in state.items()
    }


def _torchax_tensor_to_cpu(value: torch.Tensor) -> torch.Tensor:
    if hasattr(value, "torch"):
        return value.torch().detach().cpu()
    return value.detach().cpu()


def _metric_float(value) -> float:
    if hasattr(value, "torch") or hasattr(value, "detach"):
        return float(_torchax_tensor_to_cpu(value))
    return float(jax.device_get(value))


def _log_torchax_train_metrics(
    logger,
    pbar: tqdm,
    metrics: dict[str, torch.Tensor],
    *,
    config: config_dict.ConfigDict,
    total_steps: int,
    epoch: int,
    global_step: int,
) -> None:
    loss_val = _metric_float(metrics["loss"])
    pbar.set_postfix(loss=f"{loss_val:.4f}", step=global_step)
    logging.info("TorchAX train step=%d epoch=%d loss=%.4f", global_step, epoch, loss_val)
    log_metrics = {
        f"train/{key}": _metric_float(value)
        for key, value in metrics.items()
    }
    log_metrics.update(_learning_rate_metrics(config, total_steps, global_step))
    log_metrics["epoch"] = epoch
    log_metrics["global_step"] = global_step
    logger.log_metrics(log_metrics, step=global_step)


def _learning_rate_metrics(
    config: config_dict.ConfigDict,
    total_steps: int,
    global_step: int,
) -> dict[str, float]:
    optimizer_type = str(_config_get(config, "optimizer", "adamw")).lower()
    if optimizer_type == "muon":
        muon_lr = float(_config_get(config, "muon_lr", None) or config.learning_rate)
        adamw_lr = float(_config_get(config, "adamw_lr", None) or config.learning_rate)
        return {
            "train/lr_muon": _metric_float(
                _optax_learning_rate_schedule(
                    config,
                    total_steps,
                    learning_rate=muon_lr,
                )(global_step)
            ),
            "train/lr_adamw": _metric_float(
                _optax_learning_rate_schedule(
                    config,
                    total_steps,
                    learning_rate=adamw_lr,
                )(global_step)
            ),
        }
    return {
        "train/learning_rate": _metric_float(
            _optax_learning_rate_schedule(config, total_steps)(global_step)
        )
    }


def _block_until_ready(value: torch.Tensor) -> None:
    jax.block_until_ready(interop.jax_view_elem(value))
