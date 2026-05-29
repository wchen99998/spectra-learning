import gc
import logging
import math
import random
import signal
import time
import warnings
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch._dynamo
import torch._inductor.config as inductor_config
from ml_collections import config_dict
from tqdm import tqdm

from spectra_learning.data.gems.datamodule import GemsNativeDataModule
from spectra_learning.training.batch import BatchPrefetcher
from spectra_learning.training.checkpointing import (
    AsyncCheckpointWriter,
    load_frozen_teacher_weights,
    load_grad_scaler_state,
    load_optimizer_state,
    load_resume_model_state,
    load_torch_checkpoint,
    training_checkpoint_paths,
)
from spectra_learning.training.distributed import (
    DistributedContext,
    any_rank,
    barrier,
    cleanup_distributed,
    init_distributed_from_env,
    reduce_metric_tensors,
    unwrap_model,
    wrap_distributed_model,
)
from spectra_learning.training.logging import MetricLogger, log_msg_probe_metrics
from spectra_learning.training.modal_probe import (
    save_and_submit_modal_msg_probe,
    should_run_msg_probe_on_modal,
)
from spectra_learning.training.modules import PretrainModule, split_pretrain_module
from spectra_learning.training.optimization import build_optimizers
from spectra_learning.training.schedules import LRSchedulerLike
from spectra_learning.training.storage import (
    StoragePath,
    local_scratch_dir,
    normalize_storage_path,
    storage_join,
    storage_mkdir,
    storage_parent,
)
from spectra_learning.training.steps import train_step_impl
from spectra_learning.probes.massspec.msg_probe import (
    msg_probe_variants_from_config,
    resolve_msg_probe_fingerprint,
    run_msg_probe,
)
from spectra_learning.models.model import PeakSetJEPA
from spectra_learning.training.api import (
    build_grad_scaler,
    build_logger,
    build_model_from_config,
    collect_and_log_param_metrics,
    parse_autocast_dtype,
)

warnings.filterwarnings("ignore", message="Profiler function.*will be ignored")
torch.set_float32_matmul_precision("high")
inductor_config.triton.unique_kernel_names = True
inductor_config.fx_graph_cache = True
inductor_config.epilogue_fusion = True
inductor_config.shape_padding = True

_STOP_REQUESTED = False


def _config_get(config: config_dict.ConfigDict, key: str, default: Any) -> Any:
    return config.get(key, default)


def _handle_stop_signal(signum: int, frame: object) -> None:
    del frame
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    logging.warning("Received signal %d; stopping after the current step.", signum)


def install_stop_signal_handlers() -> None:
    signal.signal(signal.SIGTERM, _handle_stop_signal)
    signal.signal(signal.SIGINT, _handle_stop_signal)


def stop_requested() -> bool:
    return _STOP_REQUESTED


def stop_requested_on_any_rank(distributed: DistributedContext) -> bool:
    if not distributed.is_distributed or not torch.distributed.is_initialized():
        return stop_requested()
    return any_rank(stop_requested(), distributed)


def train_and_evaluate(
    config: config_dict.ConfigDict,
    workdir: str | Path,
) -> dict[str, object]:
    install_stop_signal_handlers()
    distributed = init_distributed_from_env()
    configure_torch_runtime(config)
    workdir = normalize_storage_path(workdir)
    local_workdir = local_scratch_dir(workdir)
    if distributed.is_main:
        local_workdir.mkdir(parents=True, exist_ok=True)
        storage_mkdir(workdir)
    barrier(distributed)
    seed_all(int(config.seed))
    datamodule = GemsNativeDataModule(
        config,
        seed=int(config.seed),
        distributed_world_size=distributed.world_size,
        distributed_rank=distributed.rank,
    )
    total_steps = total_training_steps(config, datamodule)
    loop_epochs = max(1, math.ceil(float(config.num_epochs)))
    if distributed.is_main:
        logging.info("Training for %s epochs (%d steps).", config.num_epochs, total_steps)
        logging.info("Steps per epoch: %d", datamodule.train_steps)
        logging.info(
            "Distributed: world_size=%d global_batch_size=%d local_batch_size=%d",
            distributed.world_size,
            datamodule.global_batch_size,
            datamodule.batch_size,
        )
    device = distributed.device
    clear_cuda_cache(device)
    model = build_model_from_config(config)
    initialize_frozen_teacher(config, model, distributed)
    train_module = PretrainModule(model)
    model_param_metrics = (
        collect_and_log_param_metrics(train_module) if distributed.is_main else {}
    )
    train_module.to(device).train()
    autocast_dtype = parse_autocast_dtype(_config_get(config, "autocast_dtype", "bf16"))
    grad_scaler = build_grad_scaler(autocast_dtype, device)
    optimizers, schedulers = build_optimizers(config, train_module, total_steps, device)
    checkpoint_dir = storage_join(workdir, "checkpoints")
    if distributed.is_main:
        storage_mkdir(checkpoint_dir)
    logger = build_logger(config, local_workdir) if distributed.is_main else MetricLogger()
    start_epoch, global_step, resume_offset = restore_training_state(
        config=config,
        checkpoint_dir=checkpoint_dir,
        model=model,
        optimizers=optimizers,
        schedulers=schedulers,
        grad_scaler=grad_scaler,
        steps_per_epoch=datamodule.train_steps,
        device=device,
    )
    if distributed.is_main:
        logger.log_metrics(model_param_metrics, step=global_step)
    compile_forward(train_module, config)
    train_model = wrap_distributed_model(
        train_module,
        distributed,
        static_graph=bool(_config_get(config, "ddp_static_graph", True)),
        find_unused_parameters=bool(
            _config_get(config, "ddp_find_unused_parameters", False)
        ),
    )
    checkpoint_writer = AsyncCheckpointWriter()
    last_msg_probe_metrics = run_training_loop(
        config=config,
        datamodule=datamodule,
        model=train_model,
        optimizers=optimizers,
        schedulers=schedulers,
        logger=logger,
        checkpoint_dir=checkpoint_dir,
        start_epoch=start_epoch,
        loop_epochs=loop_epochs,
        resume_offset=resume_offset,
        global_step=global_step,
        total_steps=total_steps,
        device=device,
        autocast_dtype=autocast_dtype,
        grad_scaler=grad_scaler,
        distributed=distributed,
        checkpoint_writer=checkpoint_writer,
    )
    final_global_step = int(cast(float, last_msg_probe_metrics["run/final_global_step"]))
    if distributed.is_main:
        base_model, _ = split_pretrain_module(unwrap_model(train_model))
        checkpoint_writer.save_checkpoint(
            storage_join(checkpoint_dir, "last.pt"),
            base_model,
            optimizers,
            schedulers,
            final_global_step,
            final_global_step // datamodule.train_steps,
            float("nan"),
            getattr(logger.experiment, "id", None),
            grad_scaler=grad_scaler,
        )
    checkpoint_writer.close()
    barrier(distributed)
    results = {
        **last_msg_probe_metrics,
        **model_param_metrics,
        "run/final_global_step": float(final_global_step),
        "run/world_size": float(distributed.world_size),
        "run/global_batch_size": float(datamodule.global_batch_size),
        "run/local_batch_size": float(datamodule.batch_size),
    }
    cleanup_distributed(distributed)
    return results


def initialize_frozen_teacher(
    config: config_dict.ConfigDict,
    model: PeakSetJEPA,
    distributed: DistributedContext,
) -> None:
    if str(_config_get(config, "training_mode", "jepa")).lower() != "mae_teacher_jepa":
        return
    checkpoint_path = str(config.frozen_teacher_checkpoint_path)
    if distributed.is_main:
        logging.info("Loading frozen MAE teacher from %s.", checkpoint_path)
    load_frozen_teacher_weights(model, checkpoint_path)


def run_training_loop(
    *,
    config: config_dict.ConfigDict,
    datamodule: Any,
    model: torch.nn.Module,
    optimizers: list[torch.optim.Optimizer],
    schedulers: list[LRSchedulerLike],
    logger,
    checkpoint_dir: StoragePath,
    start_epoch: int,
    loop_epochs: int,
    resume_offset: int,
    global_step: int,
    total_steps: int,
    device: torch.device,
    autocast_dtype: torch.dtype | None = None,
    grad_scaler: torch.amp.GradScaler | None = None,
    distributed: DistributedContext | None = None,
    checkpoint_writer: AsyncCheckpointWriter | None = None,
) -> dict[str, object]:
    if distributed is None:
        distributed = DistributedContext(
            rank=0,
            local_rank=0,
            world_size=1,
            device=device,
        )
    if autocast_dtype is None:
        autocast_dtype = parse_autocast_dtype(
            _config_get(config, "autocast_dtype", "bf16")
        )
    if grad_scaler is None:
        grad_scaler = build_grad_scaler(autocast_dtype, device)
    owns_checkpoint_writer = checkpoint_writer is None
    if checkpoint_writer is None:
        checkpoint_writer = AsyncCheckpointWriter()
    log_every_n_steps = int(_config_get(config, "log_every_n_steps", 50))
    collapse_every_n_steps = int(
        _config_get(config, "collapse_metrics_every_n_steps", log_every_n_steps)
    )
    checkpoint_every_steps = int(config.checkpoint_every_steps)
    grad_clip_norm = optional_float(_config_get(config, "grad_clip_norm", None))
    msg_probe_every_n_steps = msg_probe_interval(config, datamodule, total_steps)
    msg_probe_variants = msg_probe_variants_from_config(config)
    device_prefetch_size = int(_config_get(config, "device_prefetch_size", 1))
    deadline = training_deadline(config)
    wandb_run = getattr(logger, "experiment", None)
    last_msg_probe_metrics: dict[str, object] = {}
    stopped_for_time_limit = False
    stopped_for_signal = False
    initial_global_step = global_step
    modal_probe_call_ids: list[str] = []
    training_start_time = time.perf_counter()
    throughput_warmup_steps = int(_config_get(config, "throughput_warmup_steps", 0))
    measured_start_time: float | None = None
    measured_steps = 0
    for epoch in range(start_epoch, loop_epochs):
        if distributed.is_main:
            logging.info("Starting epoch %d at global_step=%d", epoch, global_step)
        epoch_resume_offset = resume_offset if epoch == start_epoch else 0
        if distributed.is_main and epoch_resume_offset:
            logging.info(
                "Resuming epoch %d from batch offset %d.",
                epoch,
                epoch_resume_offset,
            )
        train_loader = datamodule.train_loader_for_epoch(
            epoch,
            start_batch=epoch_resume_offset,
        )
        prefetcher = BatchPrefetcher(
            iter(train_loader),
            device,
            prefetch_size=device_prefetch_size,
        )
        epoch_steps = min(
            datamodule.train_steps - epoch_resume_offset,
            total_steps - global_step,
        )
        pbar = tqdm(
            total=epoch_steps,
            desc=f"Epoch {epoch}",
            unit="step",
            disable=not distributed.is_main,
        )
        while global_step < total_steps and (batch := prefetcher.next()) is not None:
            if stop_requested_on_any_rank(distributed):
                if distributed.is_main:
                    logging.info("Received stop request at global_step=%d.", global_step)
                stopped_for_signal = True
                break
            if deadline is not None and any_rank(
                time.perf_counter() >= deadline,
                distributed,
            ):
                if distributed.is_main:
                    logging.info("Reached max_duration_hours at global_step=%d.", global_step)
                stopped_for_time_limit = True
                break
            if measured_start_time is None and (
                global_step - initial_global_step
            ) >= throughput_warmup_steps:
                synchronize_device(device)
                barrier(distributed)
                measured_start_time = time.perf_counter()
            metrics = train_step_impl(
                model,
                batch,
                optimizers,
                schedulers,
                autocast_dtype,
                grad_clip_norm,
                grad_scaler=grad_scaler,
                compute_collapse_metrics=(
                    collapse_every_n_steps > 0
                    and (global_step + 1) % collapse_every_n_steps == 0
                ),
                global_step=global_step,
                total_steps=total_steps,
            )
            global_step += 1
            if measured_start_time is not None:
                measured_steps += 1
            pbar.update(1)
            should_log = log_every_n_steps > 0 and global_step % log_every_n_steps == 0
            log_metrics = (
                reduce_metric_tensors(metrics, distributed) if should_log else metrics
            )
            if distributed.is_main:
                log_train_metrics(
                    config,
                    logger,
                    pbar,
                    log_metrics,
                    optimizers,
                    epoch=epoch,
                    global_step=global_step,
                    every_n_steps=log_every_n_steps,
                )
            if global_step % checkpoint_every_steps == 0:
                if distributed.is_main:
                    base_model, _ = split_pretrain_module(unwrap_model(model))
                    checkpoint_writer.save_checkpoint(
                        storage_join(checkpoint_dir, f"step-{global_step:08d}.pt"),
                        base_model,
                        optimizers,
                        schedulers,
                        global_step,
                        global_step // datamodule.train_steps,
                        float(metrics["loss"]),
                        getattr(wandb_run, "id", None),
                        grad_scaler=grad_scaler,
                        prune_checkpoint_dir=checkpoint_dir,
                        keep_top_k=15,
                    )
                barrier(distributed)
            if msg_probe_every_n_steps > 0 and global_step % msg_probe_every_n_steps == 0:
                base_model, _ = split_pretrain_module(unwrap_model(model))
                if should_run_msg_probe_on_modal(config):
                    last_msg_probe_metrics = submit_and_log_modal_msg_probe(
                        config=config,
                        model=base_model,
                        logger=logger,
                        checkpoint_dir=checkpoint_dir,
                        global_step=global_step,
                        epoch=epoch,
                        loss=float(metrics["loss"].detach()),
                        distributed=distributed,
                    )
                    call_id = last_msg_probe_metrics.get("msg_probe/modal/call_id")
                    if isinstance(call_id, str):
                        modal_probe_call_ids.append(call_id)
                else:
                    last_msg_probe_metrics = dict(
                        run_and_log_msg_probe(
                            config,
                            base_model,
                            device,
                            logger,
                            msg_probe_variants,
                            global_step,
                            distributed,
                        )
                    )
                barrier(distributed)
        pbar.close()
        if distributed.is_main:
            logging.info("Finished epoch %d at global_step=%d", epoch, global_step)
        if stopped_for_time_limit or stopped_for_signal or global_step >= total_steps:
            break
    synchronize_device(device)
    barrier(distributed)
    training_elapsed = time.perf_counter() - training_start_time
    measured_elapsed = (
        time.perf_counter() - measured_start_time
        if measured_start_time is not None
        else 0.0
    )
    global_batch_size = int(datamodule.global_batch_size)
    last_msg_probe_metrics["run/stopped_for_time_limit"] = float(stopped_for_time_limit)
    last_msg_probe_metrics["run/stopped_for_signal"] = float(stopped_for_signal)
    last_msg_probe_metrics["run/final_global_step"] = float(global_step)
    last_msg_probe_metrics["run/train_elapsed_seconds"] = training_elapsed
    last_msg_probe_metrics["run/steps_per_second"] = (
        float(global_step - initial_global_step) / training_elapsed
        if training_elapsed > 0
        else 0.0
    )
    last_msg_probe_metrics["run/samples_per_second"] = (
        float(global_step - initial_global_step) * global_batch_size / training_elapsed
        if training_elapsed > 0
        else 0.0
    )
    last_msg_probe_metrics["run/measured_steps"] = float(measured_steps)
    last_msg_probe_metrics["run/measured_elapsed_seconds"] = measured_elapsed
    last_msg_probe_metrics["run/measured_steps_per_second"] = (
        float(measured_steps) / measured_elapsed
        if measured_elapsed > 0
        else 0.0
    )
    last_msg_probe_metrics["run/measured_samples_per_second"] = (
        float(measured_steps) * global_batch_size / measured_elapsed
        if measured_elapsed > 0
        else 0.0
    )
    if modal_probe_call_ids:
        last_msg_probe_metrics["run/modal_probe_call_ids"] = modal_probe_call_ids
        last_msg_probe_metrics["run/modal_probe_calls"] = float(len(modal_probe_call_ids))
    if owns_checkpoint_writer:
        checkpoint_writer.close()
    return last_msg_probe_metrics


def configure_torch_runtime(config: config_dict.ConfigDict) -> None:
    if str(_config_get(config, "optimizer", "adamw")).lower() == "muon":
        limit = int(_config_get(config, "dynamo_recompile_limit", 64))
        torch._dynamo.config.recompile_limit = limit
        torch._dynamo.config.cache_size_limit = limit
        logging.info("TorchDynamo cache limits set to %d for Muon.", limit)


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def clear_cuda_cache(device: torch.device) -> None:
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        logging.info(
            "CUDA memory before model init: free=%.2f GiB total=%.2f GiB",
            free_bytes / 1024**3,
            total_bytes / 1024**3,
        )


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def total_training_steps(
    config: config_dict.ConfigDict,
    datamodule: GemsNativeDataModule,
) -> int:
    total_steps = max(1, int(float(config.num_epochs) * datamodule.train_steps))
    training_max_steps = _config_get(config, "training_max_steps", None)
    if training_max_steps is None:
        return total_steps
    return min(total_steps, max(1, int(training_max_steps)))


def restore_training_state(
    *,
    config: config_dict.ConfigDict,
    checkpoint_dir: StoragePath,
    model: PeakSetJEPA,
    optimizers: list[torch.optim.Optimizer],
    schedulers: list[LRSchedulerLike],
    steps_per_epoch: int,
    device: torch.device,
    grad_scaler: torch.amp.GradScaler | None = None,
) -> tuple[int, int, int]:
    checkpoints = training_checkpoint_paths(checkpoint_dir)
    if not checkpoints:
        return 0, 0, 0
    ckpt_path = checkpoints[-1]
    logging.info("Resuming from checkpoint: %s", ckpt_path)
    ckpt = load_torch_checkpoint(ckpt_path, map_location=device, weights_only=True)
    resume_wandb_id = ckpt.get("wandb_run_id")
    if resume_wandb_id:
        config.wandb_resume_id = resume_wandb_id
    load_resume_model_state(model, ckpt["model"])
    for optimizer, state in zip(optimizers, ckpt["optimizers"], strict=True):
        load_optimizer_state(optimizer, state)
    for scheduler, state in zip(schedulers, ckpt["schedulers"], strict=True):
        scheduler.load_state_dict(state)
    load_grad_scaler_state(grad_scaler, ckpt.get("grad_scaler"))
    global_step = int(ckpt["global_step"])
    start_epoch = int(ckpt["epoch"])
    resume_offset = global_step - start_epoch * steps_per_epoch
    start_epoch += resume_offset // steps_per_epoch
    resume_offset %= steps_per_epoch
    return start_epoch, global_step, resume_offset


def compile_forward(model: torch.nn.Module, config: config_dict.ConfigDict) -> None:
    compile_mode = str(_config_get(config, "compile_mode", "max-autotune"))
    if compile_mode.lower() == "none":
        return
    inductor_config.shape_padding = not compile_mode.startswith("max-autotune")
    model.compile(
        mode=compile_mode,
        fullgraph=False,
    )


def log_train_metrics(
    config: config_dict.ConfigDict,
    logger,
    pbar: tqdm,
    metrics: dict[str, torch.Tensor],
    optimizers: list[torch.optim.Optimizer],
    *,
    epoch: int,
    global_step: int,
    every_n_steps: int,
) -> None:
    if every_n_steps <= 0 or global_step % every_n_steps != 0:
        return
    loss_val = float(metrics["loss"].detach())
    pbar.set_postfix(loss=f"{loss_val:.4f}", step=global_step)
    log_metrics = {f"train/{key}": float(value.detach()) for key, value in metrics.items()}
    log_metrics.update(learning_rate_metrics(config, optimizers))
    log_metrics["epoch"] = epoch
    log_metrics["global_step"] = global_step
    logger.log_metrics(log_metrics, step=global_step)


def learning_rate_metrics(
    config: config_dict.ConfigDict,
    optimizers: list[torch.optim.Optimizer],
) -> dict[str, float]:
    optimizer_type = str(_config_get(config, "optimizer", "adamw")).lower()
    if optimizer_type == "muon":
        metrics = {}
        for idx, optimizer in enumerate(optimizers):
            label = getattr(optimizer, "_spectra_lr_label", idx)
            metrics[f"train/lr_{label}"] = float(optimizer.param_groups[0]["lr"])
        return metrics
    return {"train/learning_rate": float(optimizers[0].param_groups[0]["lr"])}


def msg_probe_interval(
    config: config_dict.ConfigDict,
    datamodule: GemsNativeDataModule,
    total_steps: int,
) -> int:
    raw = float(_config_get(config, "msg_probe_every_n_steps", 0))
    if 0 < raw <= 1:
        reference_steps = total_steps if float(config.num_epochs) < 1 else datamodule.train_steps
        return max(1, int(raw * reference_steps))
    return int(raw)


def run_and_log_msg_probe(
    config: config_dict.ConfigDict,
    model: PeakSetJEPA,
    device: torch.device,
    logger,
    variants: tuple[str, ...],
    global_step: int,
    distributed: DistributedContext | None = None,
) -> dict[str, float]:
    if distributed is None:
        distributed = DistributedContext(
            rank=0,
            local_rank=0,
            world_size=1,
            device=device,
        )
    probe_metrics = run_msg_probe(
        config=config,
        model=model,
        device=device,
        distributed=distributed,
    )
    if distributed.is_main:
        log_msg_probe_metrics(
            logger,
            probe_metrics,
            global_step,
            enable_wandb=bool(_config_get(config, "enable_wandb", False)),
        )
        fingerprint_task = resolve_msg_probe_fingerprint(config)
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


def submit_and_log_modal_msg_probe(
    *,
    config: config_dict.ConfigDict,
    model: PeakSetJEPA,
    logger,
    checkpoint_dir: StoragePath,
    global_step: int,
    epoch: int,
    loss: float,
    distributed: DistributedContext,
) -> dict[str, object]:
    if not distributed.is_main:
        return {}
    metrics = save_and_submit_modal_msg_probe(
        config=config,
        model=model,
        checkpoint_dir=checkpoint_dir,
        workdir=storage_parent(checkpoint_dir),
        global_step=global_step,
        epoch=epoch,
        loss=loss,
        wandb_run_id=getattr(logger.experiment, "id", None),
    )
    log_msg_probe_metrics(
        logger,
        {key: value for key, value in metrics.items() if isinstance(value, float)},
        global_step,
        enable_wandb=bool(_config_get(config, "enable_wandb", False)),
    )
    return metrics


def training_deadline(config: config_dict.ConfigDict) -> float | None:
    max_duration_hours = _config_get(config, "max_duration_hours", None)
    if max_duration_hours is None:
        return None
    logging.info("Training wall-clock budget: %.2f hours", float(max_duration_hours))
    return time.perf_counter() + float(max_duration_hours) * 3600.0


def optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


_configure_dynamo_for_optimizer = configure_torch_runtime
