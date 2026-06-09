from __future__ import annotations

import logging
import math
import time
import warnings
from pathlib import Path
from typing import Any

import torch
from ml_collections import config_dict
from tqdm import tqdm

from spectra_learning.data.gems.datamodule import GemsNativeDataModule
from spectra_learning.training.activation_checkpointing import apply_activation_checkpointing
from spectra_learning.training.api import (
    build_logger,
    build_model_from_config,
    collect_and_log_param_metrics,
    estimate_training_flops_per_optimizer_step,
)
from spectra_learning.training.batch import move_batch_to_device
from spectra_learning.training.checkpointing import (
    load_torch_checkpoint,
    save_torch_checkpoint,
    training_checkpoint_paths,
)
from spectra_learning.training.distributed import barrier, init_distributed_from_env
from spectra_learning.training.logging import MetricLogger
from spectra_learning.training.storage import (
    local_scratch_dir,
    normalize_storage_path,
    storage_join,
    storage_mkdir,
)
from spectra_learning.training.torchax_runtime import (
    build_torchax_mesh,
    enable_torchax,
    initialize_torchax_distributed,
    place_torchax_tensor_like,
    replicate_torchax_tree,
    shard_torchax_batch,
    synchronize_torchax,
    tensor_to_portable_cpu,
)


def train_and_evaluate_torchax(
    config: config_dict.ConfigDict,
    workdir: str | Path,
) -> dict[str, object]:
    import optax
    import torchax.interop

    warnings.filterwarnings("ignore", message="Some donated buffers were not usable")
    initialize_torchax_distributed(config)
    enable_torchax()
    distributed = init_distributed_from_env("jax")
    workdir = normalize_storage_path(workdir)
    local_workdir = local_scratch_dir(workdir)
    if distributed.is_main:
        local_workdir.mkdir(parents=True, exist_ok=True)
        storage_mkdir(workdir)
    config.dataloader_pin_memory = False
    config.dataloader_num_workers = 0
    seed = int(config.seed)
    torch.manual_seed(seed)
    datamodule = GemsNativeDataModule(
        config,
        seed=seed,
        distributed_world_size=distributed.world_size,
        distributed_rank=distributed.rank,
    )
    total_steps = _total_training_steps(config, datamodule)
    device = distributed.device
    model = build_model_from_config(config)
    model.to(device).train()
    apply_activation_checkpointing(model, config)
    model_param_metrics = (
        collect_and_log_param_metrics(model) if distributed.is_main else {}
    )
    flops_per_optimizer_step = estimate_training_flops_per_optimizer_step(
        config,
        model,
        datamodule.global_batch_size,
    )
    if distributed.is_main:
        model_param_metrics["model/flops_per_optimizer_step_estimate"] = (
            flops_per_optimizer_step
        )
        model_param_metrics["model/flops_per_sample_estimate"] = (
            flops_per_optimizer_step / float(datamodule.global_batch_size)
        )
    checkpoint_dir = storage_join(workdir, "checkpoints")
    if distributed.is_main:
        storage_mkdir(checkpoint_dir)
    logger = build_logger(config, local_workdir) if distributed.is_main else MetricLogger()
    params = {
        name: param
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    buffers = dict(model.named_buffers())
    mesh = build_torchax_mesh(config)
    params = replicate_torchax_tree(params, mesh)
    buffers = replicate_torchax_tree(buffers, mesh)
    optimizer = optax.adamw(
        learning_rate=float(config.learning_rate),
        b2=float(_config_get(config, "b2", 0.999)),
        weight_decay=float(config.weight_decay),
    )
    lazy_opt_state = bool(_config_get(config, "torchax_lazy_opt_state", True))
    has_checkpoints = bool(training_checkpoint_paths(checkpoint_dir))
    opt_state = (
        None
        if lazy_opt_state and not has_checkpoints
        else torchax.interop.call_jax(optimizer.init, params)
    )
    global_step, params, opt_state = _restore_torchax_state(
        checkpoint_dir,
        params,
        opt_state,
    )
    grad_step = _make_torchax_grad_step(model, mesh=mesh, params=params)
    apply_grads = _make_torchax_apply_grads(optimizer)
    accumulate_grads = _make_torchax_accumulate_grads()
    scale_grads = _make_torchax_scale_grads()
    if bool(_config_get(config, "torchax_jit", True)):
        grad_step = torchax.interop.jax_jit(grad_step)
        apply_grads = torchax.interop.jax_jit(
            apply_grads,
            kwargs_for_jax_jit={"donate_argnums": (0, 1, 2)},
        )
        accumulate_grads = torchax.interop.jax_jit(
            accumulate_grads,
            kwargs_for_jax_jit={"donate_argnums": (0, 1)},
        )
        scale_grads = torchax.interop.jax_jit(
            scale_grads,
            kwargs_for_jax_jit={"donate_argnums": (0,)},
        )
    if distributed.is_main:
        logger.log_metrics(model_param_metrics, step=global_step)
    metrics = _run_torchax_loop(
        config=config,
        datamodule=datamodule,
        model=model,
        params=params,
        buffers=buffers,
        opt_state=opt_state,
        grad_step=grad_step,
        apply_grads=apply_grads,
        accumulate_grads=accumulate_grads,
        scale_grads=scale_grads,
        optimizer=optimizer,
        logger=logger,
        checkpoint_dir=checkpoint_dir,
        global_step=global_step,
        total_steps=total_steps,
        flops_per_optimizer_step=flops_per_optimizer_step,
        device=device,
        mesh=mesh,
        distributed_is_main=distributed.is_main,
    )
    barrier(distributed)
    return {
        **metrics,
        **model_param_metrics,
        "run/world_size": float(distributed.world_size),
        "run/global_batch_size": float(datamodule.global_batch_size),
        "run/local_batch_size": float(datamodule.batch_size),
        "run/device_backend": "jax",
        "run/torchax_mesh_devices": float(mesh.device_count),
    }


def _make_torchax_grad_step(
    model: torch.nn.Module,
    *,
    mesh: Any | None = None,
    params: dict[str, torch.Tensor] | None = None,
):
    import jax
    from jax.sharding import PartitionSpec as P
    import torchax.interop

    def loss_fn(params, buffers, batch):
        metrics = torch.func.functional_call(model, (params, buffers), (batch,))
        return metrics["loss"]

    grad_fn = torchax.interop.jax_value_and_grad(loss_fn)

    if mesh is not None and mesh.enabled:
        param_specs = jax.tree.map(lambda _: P(), params)

        def pmean_tree(tree):
            return jax.tree.map(
                lambda value: jax.lax.pmean(value, mesh.axis_name),
                tree,
            )

        def grad_step(params, buffers, batch):
            loss, grads = grad_fn(params, buffers, batch)
            loss = torchax.interop.call_jax(
                lambda value: jax.lax.pmean(value, mesh.axis_name),
                loss,
            )
            grads = torchax.interop.call_jax(pmean_tree, grads)
            return loss, grads

        return torchax.interop.jax_shard_map(
            grad_step,
            {
                "mesh": mesh.mesh,
                "out_specs": (P(), param_specs),
                "check_vma": False,
            },
        )

    def grad_step(params, buffers, batch):
        return grad_fn(params, buffers, batch)

    return grad_step


def _make_torchax_apply_grads(optimizer):
    import optax
    import torchax.interop

    def apply_grads(params, opt_state, grads):
        updates, opt_state = torchax.interop.call_jax(
            optimizer.update,
            grads,
            opt_state,
            params,
        )
        params = torchax.interop.call_jax(optax.apply_updates, params, updates)
        return params, opt_state

    return apply_grads


def _make_torchax_accumulate_grads():
    import jax
    import torchax.interop

    def accumulate_grads(accumulated_grads, grads):
        return torchax.interop.call_jax(
            lambda lhs, rhs: jax.tree.map(lambda acc, grad: acc + grad, lhs, rhs),
            accumulated_grads,
            grads,
        )

    return accumulate_grads


def _make_torchax_scale_grads():
    import jax
    import torchax.interop

    def scale_grads(grads, scale):
        return torchax.interop.call_jax(
            lambda tree, factor: jax.tree.map(lambda grad: grad * factor, tree),
            grads,
            scale,
        )

    return scale_grads


def _make_torchax_train_step(model: torch.nn.Module, optimizer):
    grad_step = _make_torchax_grad_step(model)
    apply_grads = _make_torchax_apply_grads(optimizer)

    def train_step(params, buffers, opt_state, batch):
        loss, grads = grad_step(params, buffers, batch)
        params, opt_state = apply_grads(params, opt_state, grads)
        return loss, params, opt_state

    return train_step


def _run_torchax_loop(
    *,
    config: config_dict.ConfigDict,
    datamodule: GemsNativeDataModule,
    model: torch.nn.Module,
    params: dict[str, torch.Tensor],
    buffers: dict[str, torch.Tensor],
    opt_state: Any | None,
    grad_step,
    apply_grads,
    accumulate_grads,
    scale_grads,
    optimizer,
    logger,
    checkpoint_dir,
    global_step: int,
    total_steps: int,
    flops_per_optimizer_step: float,
    device: torch.device,
    mesh,
    distributed_is_main: bool,
) -> dict[str, object]:
    import torchax

    log_every_n_steps = int(_config_get(config, "log_every_n_steps", 50))
    checkpoint_every_steps = int(_config_get(config, "checkpoint_every_steps", 0))
    warmup_steps = int(_config_get(config, "throughput_warmup_steps", 0))
    grad_accum_steps = int(_config_get(config, "gradient_accumulation_steps", 1))
    offload_opt_state = bool(
        _config_get(config, "torchax_offload_opt_state_during_accumulation", True)
    )
    start_global_step = global_step
    train_start = time.perf_counter()
    measured_start: float | None = None
    measured_steps = 0
    last_loss = torch.tensor(float("nan"))
    accumulated_grads = None
    accumulation_step = 0
    loop_epochs = max(1, math.ceil(float(config.num_epochs)))
    for epoch in range(loop_epochs):
        loader = datamodule.train_loader_for_epoch(epoch)
        pbar = tqdm(
            total=min(datamodule.train_steps, total_steps - global_step),
            desc=f"Epoch {epoch}",
            unit="step",
            disable=not distributed_is_main,
        )
        torchax.disable_globally()
        loader_iter = iter(loader)
        while global_step < total_steps:
            torchax.disable_globally()
            batch = next(loader_iter, None)
            if batch is None:
                break
            if global_step >= total_steps:
                break
            torchax.enable_globally()
            moved = move_batch_to_device(batch, device)
            moved = shard_torchax_batch(moved, mesh)
            if (
                measured_start is None
                and (global_step - start_global_step) >= warmup_steps
            ):
                synchronize_torchax()
                measured_start = time.perf_counter()
            last_loss, grads = grad_step(params, buffers, moved)
            accumulated_grads = (
                grads
                if accumulated_grads is None
                else accumulate_grads(accumulated_grads, grads)
            )
            accumulation_step += 1
            if accumulation_step % grad_accum_steps != 0:
                continue
            if grad_accum_steps > 1:
                accumulated_grads = scale_grads(
                    accumulated_grads,
                    1.0 / float(grad_accum_steps),
                )
            if opt_state is None:
                import torchax.interop

                opt_state = torchax.interop.call_jax(optimizer.init, params)
            else:
                opt_state = _opt_state_to_device(opt_state, mesh)
            params, opt_state = apply_grads(params, opt_state, accumulated_grads)
            if offload_opt_state and grad_accum_steps > 1:
                opt_state = _opt_state_to_cpu(opt_state)
            accumulated_grads = None
            accumulation_step = 0
            global_step += 1
            if measured_start is not None:
                measured_steps += 1
            pbar.update(1)
            if (
                distributed_is_main
                and log_every_n_steps > 0
                and global_step % log_every_n_steps == 0
            ):
                _log_torchax_metrics(
                    config,
                    logger,
                    pbar,
                    last_loss,
                    global_step=global_step,
                    flops_per_optimizer_step=flops_per_optimizer_step,
                )
            if (
                distributed_is_main
                and checkpoint_every_steps > 0
                and global_step % checkpoint_every_steps == 0
            ):
                _save_torchax_training_checkpoint(
                    checkpoint_dir,
                    model,
                    params,
                    buffers,
                    opt_state,
                    global_step=global_step,
                    epoch=epoch,
                    loss=float(last_loss.cpu()),
                )
        pbar.close()
        if global_step >= total_steps:
            break
    synchronize_torchax()
    elapsed = time.perf_counter() - train_start
    measured_elapsed = (
        time.perf_counter() - measured_start if measured_start is not None else 0.0
    )
    if distributed_is_main:
        _save_torchax_training_checkpoint(
            checkpoint_dir,
            model,
            params,
            buffers,
            opt_state,
            global_step=global_step,
            epoch=max(0, global_step // datamodule.train_steps),
            loss=float(last_loss.cpu()),
            name="last.pt",
        )
    return {
        "run/final_global_step": float(global_step),
        "run/train_elapsed_seconds": elapsed,
        "run/steps_per_second": (
            float(global_step - start_global_step) / elapsed if elapsed > 0 else 0.0
        ),
        "run/samples_per_second": (
            float(global_step - start_global_step)
            * int(datamodule.global_batch_size)
            / elapsed
            if elapsed > 0
            else 0.0
        ),
        "run/measured_steps": float(measured_steps),
        "run/measured_elapsed_seconds": measured_elapsed,
        "run/measured_steps_per_second": (
            float(measured_steps) / measured_elapsed if measured_elapsed > 0 else 0.0
        ),
        "run/measured_samples_per_second": (
            float(measured_steps) * int(datamodule.global_batch_size) / measured_elapsed
            if measured_elapsed > 0
            else 0.0
        ),
        "train/loss": float(last_loss.cpu()),
    }


def _save_torchax_training_checkpoint(
    checkpoint_dir,
    model: torch.nn.Module,
    params: dict[str, torch.Tensor],
    buffers: dict[str, torch.Tensor],
    opt_state: Any,
    *,
    global_step: int,
    epoch: int,
    loss: float,
    name: str | None = None,
) -> None:
    assert opt_state is not None
    checkpoint_name = name or f"step-{global_step:08d}.pt"
    state_keys = set(model.state_dict().keys())
    state_dict = {
        state_key: tensor
        for name, tensor in [*params.items(), *buffers.items()]
        if (state_key := _state_dict_key(name)) in state_keys
    }
    save_torch_checkpoint(
        {
            "model": state_dict,
            "optimizers": [],
            "optax_leaves": _opt_state_checkpoint_leaves(opt_state),
            "schedulers": [],
            "grad_scaler": None,
            "global_step": global_step,
            "epoch": epoch,
            "loss": loss,
            "wandb_run_id": None,
            "checkpoint_backend": "torchax",
        },
        storage_join(checkpoint_dir, checkpoint_name),
    )


def _restore_torchax_state(
    checkpoint_dir,
    params: dict[str, torch.Tensor],
    opt_state: Any | None,
) -> tuple[int, dict[str, torch.Tensor], Any | None]:
    checkpoints = training_checkpoint_paths(checkpoint_dir)
    if not checkpoints:
        return 0, params, opt_state
    ckpt_path = checkpoints[-1]
    logging.info("Resuming TorchAX training from checkpoint: %s", ckpt_path)
    ckpt = load_torch_checkpoint(ckpt_path, map_location="cpu", weights_only=True)
    state_key_to_param_key = {_state_dict_key(key): key for key in params}
    loaded_params = {
        state_key_to_param_key[key]: place_torchax_tensor_like(
            value,
            params[state_key_to_param_key[key]],
        )
        for key, value in ckpt["model"].items()
        if key in state_key_to_param_key
    }
    params.update(loaded_params)
    assert opt_state is not None
    opt_state = _restore_opt_state(opt_state, ckpt.get("optax_leaves", []))
    return int(ckpt["global_step"]), params, opt_state


def _opt_state_checkpoint_leaves(opt_state: Any) -> list[Any]:
    import jax

    return list(jax.tree.leaves(opt_state))


def _opt_state_to_cpu(opt_state: Any) -> Any:
    import jax

    return jax.tree.map(
        lambda leaf: tensor_to_portable_cpu(leaf)
        if isinstance(leaf, torch.Tensor)
        else leaf,
        opt_state,
    )


def _opt_state_to_device(opt_state: Any, mesh: Any) -> Any:
    import jax

    opt_state = jax.tree.map(
        lambda leaf: leaf.to("jax") if isinstance(leaf, torch.Tensor) else leaf,
        opt_state,
    )
    return replicate_torchax_tree(opt_state, mesh)


def _restore_opt_state(opt_state: Any, leaves: list[torch.Tensor]) -> Any:
    import jax

    if not leaves:
        return opt_state
    jax_leaves = [
        place_torchax_tensor_like(leaf, reference)
        for leaf, reference in zip(leaves, jax.tree.leaves(opt_state), strict=True)
    ]
    return jax.tree.unflatten(jax.tree.structure(opt_state), jax_leaves)


def _state_dict_key(name: str) -> str:
    return name.replace("._checkpoint_wrapped_module", "")


def _log_torchax_metrics(
    config: config_dict.ConfigDict,
    logger,
    pbar: tqdm,
    loss: torch.Tensor,
    *,
    global_step: int,
    flops_per_optimizer_step: float,
) -> None:
    loss_value = float(loss.cpu())
    pbar.set_postfix(loss=f"{loss_value:.4f}", step=global_step)
    cumulative_flops = float(global_step) * float(flops_per_optimizer_step)
    logger.log_metrics(
        {
            "train/loss": loss_value,
            "train/learning_rate": float(config.learning_rate),
            "train/cumulative_flops": cumulative_flops,
            "train/cumulative_peta_flops": cumulative_flops / 1e15,
            "train/flops_per_optimizer_step": float(flops_per_optimizer_step),
            "global_step": global_step,
        },
        step=global_step,
    )


def _total_training_steps(
    config: config_dict.ConfigDict,
    datamodule: GemsNativeDataModule,
) -> int:
    total_steps = max(1, int(float(config.num_epochs) * datamodule.train_steps))
    training_max_steps = _config_get(config, "training_max_steps", None)
    if training_max_steps is None:
        return total_steps
    return min(total_steps, max(1, int(training_max_steps)))


def _config_get(config: Any, key: str, default: Any) -> Any:
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)
