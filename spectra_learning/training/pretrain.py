import gc
import logging
import math
import random
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch._dynamo
import torch._inductor.config as inductor_config
from ml_collections import config_dict
from tqdm import tqdm

from spectra_learning.data.gems.datamodule import GemsNativeDataModule
from spectra_learning.training.batch import BatchPrefetcher
from spectra_learning.training.checkpointing import (
    load_optimizer_state,
    load_resume_model_state,
    prune_checkpoints,
    save_checkpoint,
)
from spectra_learning.training.optimization import build_optimizers
from spectra_learning.training.steps import train_step_impl
from spectra_learning.probes.massspec.msg_probe import msg_probe_variants_from_config, run_msg_probe
from spectra_learning.training.api import (
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


def train_and_evaluate(
    config: config_dict.ConfigDict,
    workdir: str | Path,
) -> dict[str, float]:
    configure_torch_runtime(config)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    seed_all(int(config.seed))
    datamodule = GemsNativeDataModule(config, seed=int(config.seed))
    total_steps = total_training_steps(config, datamodule)
    loop_epochs = max(1, math.ceil(float(config.num_epochs)))
    logging.info("Training for %s epochs (%d steps).", config.num_epochs, total_steps)
    logging.info("Steps per epoch: %d", datamodule.train_steps)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clear_cuda_cache(device)
    model = build_model_from_config(config)
    model_param_metrics = collect_and_log_param_metrics(model)
    model.to(device).train()
    optimizers, schedulers = build_optimizers(config, model, total_steps, device)
    checkpoint_dir = workdir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger(config, workdir)
    start_epoch, global_step, resume_offset = restore_training_state(
        config=config,
        checkpoint_dir=checkpoint_dir,
        model=model,
        optimizers=optimizers,
        schedulers=schedulers,
        steps_per_epoch=datamodule.train_steps,
        device=device,
    )
    logger.log_metrics(model_param_metrics, step=global_step)
    compile_forward(model, config)
    last_msg_probe_metrics = run_training_loop(
        config=config,
        datamodule=datamodule,
        model=model,
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
    )
    final_global_step = int(last_msg_probe_metrics["run/final_global_step"])
    save_checkpoint(
        checkpoint_dir / "last.pt",
        model,
        optimizers,
        schedulers,
        final_global_step,
        final_global_step // datamodule.train_steps,
        float("nan"),
        getattr(logger.experiment, "id", None),
    )
    return {
        **last_msg_probe_metrics,
        **model_param_metrics,
        "run/final_global_step": float(final_global_step),
    }


def run_training_loop(
    *,
    config: config_dict.ConfigDict,
    datamodule: GemsNativeDataModule,
    model: torch.nn.Module,
    optimizers: list[torch.optim.Optimizer],
    schedulers: list[torch.optim.lr_scheduler.LRScheduler],
    logger,
    checkpoint_dir: Path,
    start_epoch: int,
    loop_epochs: int,
    resume_offset: int,
    global_step: int,
    total_steps: int,
    device: torch.device,
) -> dict[str, float]:
    autocast_dtype = parse_autocast_dtype(config.get("autocast_dtype", "bf16"))
    log_every_n_steps = int(config.get("log_every_n_steps", 50))
    collapse_every_n_steps = int(
        config.get("collapse_metrics_every_n_steps", log_every_n_steps)
    )
    checkpoint_every_steps = int(config.checkpoint_every_steps)
    grad_clip_norm = optional_float(config.get("grad_clip_norm", None))
    msg_probe_every_n_steps = msg_probe_interval(config, datamodule, total_steps)
    msg_probe_variants = msg_probe_variants_from_config(config)
    device_prefetch_size = int(config.get("device_prefetch_size", 1))
    deadline = training_deadline(config)
    wandb_run = getattr(logger, "experiment", None)
    last_msg_probe_metrics: dict[str, float] = {}
    stopped_for_time_limit = False
    for epoch in range(start_epoch, loop_epochs):
        logging.info("Starting epoch %d at global_step=%d", epoch, global_step)
        train_loader = datamodule.train_loader_for_epoch(epoch)
        prefetcher = BatchPrefetcher(
            iter(train_loader),
            device,
            prefetch_size=device_prefetch_size,
        )
        epoch_resume_offset = resume_offset if epoch == start_epoch else 0
        for _ in range(epoch_resume_offset):
            if prefetcher.next() is None:
                break
        epoch_steps = min(
            datamodule.train_steps - epoch_resume_offset,
            total_steps - global_step,
        )
        pbar = tqdm(total=epoch_steps, desc=f"Epoch {epoch}", unit="step")
        while global_step < total_steps and (batch := prefetcher.next()) is not None:
            if deadline is not None and time.perf_counter() >= deadline:
                logging.info("Reached max_duration_hours at global_step=%d.", global_step)
                stopped_for_time_limit = True
                break
            metrics = train_step_impl(
                model,
                batch,
                optimizers,
                schedulers,
                autocast_dtype,
                grad_clip_norm,
                compute_collapse_metrics=(
                    collapse_every_n_steps > 0
                    and (global_step + 1) % collapse_every_n_steps == 0
                ),
                global_step=global_step,
                total_steps=total_steps,
            )
            global_step += 1
            pbar.update(1)
            log_train_metrics(
                config,
                logger,
                pbar,
                metrics,
                optimizers,
                epoch=epoch,
                global_step=global_step,
                every_n_steps=log_every_n_steps,
            )
            if global_step % checkpoint_every_steps == 0:
                save_checkpoint(
                    checkpoint_dir / f"step-{global_step:08d}.pt",
                    model,
                    optimizers,
                    schedulers,
                    global_step,
                    global_step // datamodule.train_steps,
                    float(metrics["loss"]),
                    getattr(wandb_run, "id", None),
                )
                prune_checkpoints(checkpoint_dir, keep_top_k=15)
            if msg_probe_every_n_steps > 0 and global_step % msg_probe_every_n_steps == 0:
                last_msg_probe_metrics = run_and_log_msg_probe(
                    config,
                    model,
                    device,
                    logger,
                    msg_probe_variants,
                    global_step,
                )
        pbar.close()
        logging.info("Finished epoch %d at global_step=%d", epoch, global_step)
        if stopped_for_time_limit or global_step >= total_steps:
            break
    last_msg_probe_metrics["run/stopped_for_time_limit"] = float(stopped_for_time_limit)
    last_msg_probe_metrics["run/final_global_step"] = float(global_step)
    return last_msg_probe_metrics


def configure_torch_runtime(config: config_dict.ConfigDict) -> None:
    if str(config.get("optimizer", "adamw")).lower() == "muon":
        limit = int(config.get("dynamo_recompile_limit", 64))
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


def total_training_steps(
    config: config_dict.ConfigDict,
    datamodule: GemsNativeDataModule,
) -> int:
    total_steps = max(1, int(float(config.num_epochs) * datamodule.train_steps))
    training_max_steps = config.get("training_max_steps", None)
    if training_max_steps is None:
        return total_steps
    return min(total_steps, max(1, int(training_max_steps)))


def restore_training_state(
    *,
    config: config_dict.ConfigDict,
    checkpoint_dir: Path,
    model: torch.nn.Module,
    optimizers: list[torch.optim.Optimizer],
    schedulers: list[torch.optim.lr_scheduler.LRScheduler],
    steps_per_epoch: int,
    device: torch.device,
) -> tuple[int, int, int]:
    checkpoints = sorted(checkpoint_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    if not checkpoints:
        return 0, 0, 0
    ckpt_path = checkpoints[-1]
    logging.info("Resuming from checkpoint: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    resume_wandb_id = ckpt.get("wandb_run_id")
    if resume_wandb_id:
        config.wandb_resume_id = resume_wandb_id
    load_resume_model_state(model, ckpt["model"])
    for optimizer, state in zip(optimizers, ckpt["optimizers"], strict=True):
        load_optimizer_state(optimizer, state)
    for scheduler, state in zip(schedulers, ckpt["schedulers"], strict=True):
        scheduler.load_state_dict(state)
    global_step = int(ckpt["global_step"])
    start_epoch = int(ckpt["epoch"])
    resume_offset = global_step - start_epoch * steps_per_epoch
    start_epoch += resume_offset // steps_per_epoch
    resume_offset %= steps_per_epoch
    return start_epoch, global_step, resume_offset


def compile_forward(model: torch.nn.Module, config: config_dict.ConfigDict) -> None:
    compile_mode = str(config.get("compile_mode", "max-autotune"))
    model.forward_augmented = torch.compile(
        model.forward_augmented,
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
    optimizer_type = str(config.get("optimizer", "adamw")).lower()
    has_predictor_lr = float(config.get("predictor_learning_rate_ratio", 1.0)) != 1.0
    if optimizer_type == "muon":
        metrics = {"train/lr_muon": float(optimizers[0].param_groups[0]["lr"])}
        if has_predictor_lr:
            metrics["train/lr_predictor_muon"] = float(optimizers[1].param_groups[0]["lr"])
        return metrics
    metrics = {"train/learning_rate": float(optimizers[0].param_groups[0]["lr"])}
    if has_predictor_lr:
        metrics["train/predictor_learning_rate"] = float(optimizers[1].param_groups[0]["lr"])
    return metrics


def msg_probe_interval(
    config: config_dict.ConfigDict,
    datamodule: GemsNativeDataModule,
    total_steps: int,
) -> int:
    raw = float(config.get("msg_probe_every_n_steps", 0))
    if 0 < raw <= 1:
        reference_steps = total_steps if float(config.num_epochs) < 1 else datamodule.train_steps
        return max(1, int(raw * reference_steps))
    return int(raw)


def run_and_log_msg_probe(
    config: config_dict.ConfigDict,
    model: torch.nn.Module,
    device: torch.device,
    logger,
    variants: tuple[str, ...],
    global_step: int,
) -> dict[str, float]:
    probe_metrics = run_msg_probe(config=config, model=model, device=device)
    logger.log_metrics(probe_metrics, step=global_step)
    for variant in variants:
        prefix = f"msg_probe/{variant}"
        epoch_key = f"{prefix}/epoch"
        if epoch_key in probe_metrics:
            logging.info(
                "step=%d msg_probe[%s] best_epoch=%.2f test_auc_maccs_mean=%.4f",
                global_step,
                variant,
                probe_metrics[epoch_key],
                probe_metrics[f"{prefix}/test/auc_maccs_mean"],
            )
    return probe_metrics


def training_deadline(config: config_dict.ConfigDict) -> float | None:
    max_duration_hours = config.get("max_duration_hours", None)
    if max_duration_hours is None:
        return None
    logging.info("Training wall-clock budget: %.2f hours", float(max_duration_hours))
    return time.perf_counter() + float(max_duration_hours) * 3600.0


def optional_float(value: object) -> float | None:
    return None if value is None else float(value)


_configure_dynamo_for_optimizer = configure_torch_runtime
