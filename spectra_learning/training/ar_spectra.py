from __future__ import annotations

import logging
import math
import os
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ContextManager

import numpy as np
import torch
import torch.distributed as dist
from ml_collections import config_dict
from torch.nn.parallel import DistributedDataParallel

from spectra_learning.data.ar_spectra import SpectraARGemsDataModule
from spectra_learning.models.ar_spectra import build_spectra_ar_model_from_config
from spectra_learning.training.configuration import save_config
from spectra_learning.training.logging import MetricLogger, build_logger


def _config_get(config: config_dict.ConfigDict, key: str, default: Any) -> Any:
    return config.get(key, default)


@dataclass(frozen=True)
class ARDistributedContext:
    world_size: int
    rank: int
    local_rank: int
    is_distributed: bool
    is_main: bool
    device: torch.device
    device_id: int | None


def init_ar_distributed(config: config_dict.ConfigDict) -> ARDistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_name = str(_config_get(config, "device", "auto")).lower()
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    use_cuda = device_name.startswith("cuda") and torch.cuda.is_available()
    is_distributed = world_size > 1
    device_id = local_rank if use_cuda else None
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if is_distributed:
        if use_cuda:
            dist.init_process_group(backend="nccl", device_id=device)
        else:
            dist.init_process_group(backend="gloo")
    return ARDistributedContext(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        is_distributed=is_distributed,
        is_main=rank == 0,
        device=device,
        device_id=device_id,
    )


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def autocast_context(
    device: torch.device,
    dtype_name: str,
) -> ContextManager:
    if device.type != "cuda" or dtype_name == "fp32":
        return nullcontext()
    if dtype_name == "bf16":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    if dtype_name == "fp16":
        return torch.amp.autocast("cuda", dtype=torch.float16)
    raise ValueError(f"Unknown autocast dtype: {dtype_name}")


def train_and_evaluate_ar_spectra(
    config: config_dict.ConfigDict,
    workdir: str | Path,
) -> dict[str, float]:
    torch.set_float32_matmul_precision("high")
    workdir = Path(workdir)
    distributed = init_ar_distributed(config)
    if distributed.is_main:
        workdir.mkdir(parents=True, exist_ok=True)
        save_config(config, workdir)
    seed = int(_config_get(config, "seed", 0))
    seed_all(seed + distributed.rank)

    device = distributed.device
    datamodule = SpectraARGemsDataModule(
        config,
        seed=seed,
        distributed_world_size=distributed.world_size,
        distributed_rank=distributed.rank,
        distributed_local_rank=distributed.local_rank,
    )
    base_model = build_spectra_ar_model_from_config(
        config,
        datamodule.ar_tokenizer,
    ).to(device)
    model: torch.nn.Module = base_model
    if distributed.is_distributed:
        ddp_kwargs: dict[str, Any] = {}
        if distributed.device_id is not None:
            ddp_kwargs["device_ids"] = [distributed.device_id]
            ddp_kwargs["output_device"] = distributed.device_id
        model = DistributedDataParallel(base_model, **ddp_kwargs)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(_config_get(config, "learning_rate", 3e-4)),
        betas=(
            float(_config_get(config, "b1", 0.9)),
            float(_config_get(config, "b2", 0.95)),
        ),
        weight_decay=float(_config_get(config, "weight_decay", 0.01)),
    )
    gradient_accumulation_steps = int(
        _config_get(config, "gradient_accumulation_steps", 1)
    )
    total_steps = min(
        int(_config_get(config, "training_max_steps", datamodule.train_steps)),
        max(1, math.ceil(float(_config_get(config, "num_epochs", 1.0))))
        * datamodule.train_steps,
    )
    log_every_n_steps = int(_config_get(config, "log_every_n_steps", 20))
    val_every_n_steps = int(_config_get(config, "val_every_n_steps", 0))
    val_num_steps = int(_config_get(config, "val_num_steps", 20))
    checkpoint_every_n_steps = int(_config_get(config, "checkpoint_every_n_steps", 0))
    grad_clip_norm = float(_config_get(config, "grad_clip_norm", 1.0))
    autocast_dtype = str(_config_get(config, "autocast_dtype", "bf16")).lower()
    logger = build_logger(config, workdir) if distributed.is_main else MetricLogger()

    if distributed.is_main:
        logging.info("AR tokenizer: %s", asdict(datamodule.ar_tokenizer_config))
        logging.info("AR vocab size: %d", datamodule.ar_tokenizer.vocab_size)
        logging.info("AR sequence length: %d", datamodule.ar_tokenizer.sequence_length)
        logging.info(
            "Training AR spectra model for %d optimizer steps on %d rank(s).",
            total_steps,
            distributed.world_size,
        )

    global_step = 0
    last_metrics: dict[str, float] = {}
    optimizer.zero_grad(set_to_none=True)
    loop_epochs = max(1, math.ceil(float(_config_get(config, "num_epochs", 1.0))))
    for epoch in range(loop_epochs):
        for micro_step, batch in enumerate(datamodule.train_loader_for_epoch(epoch)):
            model.train()
            batch = move_batch_to_device(batch, device)
            with autocast_context(device, autocast_dtype):
                output = model(batch)
                loss = output["loss"] / gradient_accumulation_steps
            loss.backward()
            if (micro_step + 1) % gradient_accumulation_steps != 0:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            last_metrics = _scalar_metrics(output)

            if distributed.is_main and (
                global_step % log_every_n_steps == 0 or global_step == 1
            ):
                logger.log_metrics(
                    {
                        "global_step": float(global_step),
                        **_prefix_metrics(last_metrics, "train"),
                    },
                    step=global_step,
                )
                logging.info(
                    "step=%d train_loss=%.4f train_token_accuracy=%.4f",
                    global_step,
                    last_metrics["loss"],
                    last_metrics["token_accuracy"],
                )
            if val_every_n_steps > 0 and global_step % val_every_n_steps == 0:
                if distributed.is_distributed:
                    dist.barrier()
                if distributed.is_main:
                    validation_metrics = evaluate_ar_spectra(
                        base_model,
                        datamodule.val_loader,
                        device,
                        max_steps=val_num_steps,
                        autocast_dtype=autocast_dtype,
                    )
                    last_metrics.update(validation_metrics)
                    logger.log_metrics(
                        {"global_step": float(global_step), **validation_metrics},
                        step=global_step,
                    )
                    logging.info(
                        "step=%d val_loss=%.4f val_token_accuracy=%.4f %s",
                        global_step,
                        last_metrics["val/loss"],
                        last_metrics["val/token_accuracy"],
                        _validation_accuracy_summary(last_metrics),
                    )
                if distributed.is_distributed:
                    dist.barrier()
            if (
                checkpoint_every_n_steps > 0
                and global_step % checkpoint_every_n_steps == 0
            ):
                if distributed.is_main:
                    save_checkpoint(
                        workdir / f"step_{global_step}.pt",
                        base_model,
                        optimizer,
                        datamodule,
                        global_step,
                        config,
                    )
                if distributed.is_distributed:
                    dist.barrier()
            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            break

    if distributed.is_main:
        validation_metrics = evaluate_ar_spectra(
            base_model,
            datamodule.val_loader,
            device,
            max_steps=val_num_steps,
            autocast_dtype=autocast_dtype,
        )
        last_metrics.update(validation_metrics)
        save_checkpoint(
            workdir / "last.pt",
            base_model,
            optimizer,
            datamodule,
            global_step,
            config,
        )
        last_metrics["run/final_global_step"] = float(global_step)
        last_metrics["run/ar_vocab_size"] = float(datamodule.ar_tokenizer.vocab_size)
        last_metrics["run/ar_sequence_length"] = float(
            datamodule.ar_tokenizer.sequence_length
        )
        last_metrics["run/world_size"] = float(distributed.world_size)
        final_metrics = {
            key: value
            for key, value in last_metrics.items()
            if key.startswith("val/") or key.startswith("run/")
        }
        logger.log_metrics(
            {"global_step": float(global_step), **final_metrics},
            step=global_step,
        )
        wandb_run = getattr(logger, "experiment", None)
        if wandb_run is not None:
            wandb_run.finish()
    if distributed.is_distributed:
        dist.barrier()
        dist.destroy_process_group()
    return last_metrics


@torch.no_grad()
def evaluate_ar_spectra(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    max_steps: int,
    autocast_dtype: str = "bf16",
) -> dict[str, float]:
    model.eval()
    weighted_metrics: dict[str, float] = {}
    metric_weights: dict[str, float] = {}
    token_totals: dict[str, float] = {}
    for step, batch in enumerate(loader):
        if step >= max_steps:
            break
        batch = move_batch_to_device(batch, device)
        with autocast_context(device, autocast_dtype):
            output = model(batch)
        tokens_by_suffix = _target_token_counts(output)
        for suffix, tokens in tokens_by_suffix.items():
            token_totals[f"val/target_tokens{suffix}"] = (
                token_totals.get(f"val/target_tokens{suffix}", 0.0) + tokens
            )
        for key, value in output.items():
            if key == "logits" or key.startswith("target_tokens"):
                continue
            weight = _metric_weight(key, tokens_by_suffix)
            weighted_metrics[key] = (
                weighted_metrics.get(key, 0.0) + float(value.detach().cpu()) * weight
            )
            metric_weights[key] = metric_weights.get(key, 0.0) + weight
    metrics = {
        f"val/{key}": weighted_value / max(metric_weights[key], 1.0)
        for key, weighted_value in weighted_metrics.items()
    }
    metrics.update(token_totals)
    return metrics


def _target_token_counts(output: dict[str, torch.Tensor]) -> dict[str, float]:
    counts = {"": float(output["target_tokens"].detach().cpu())}
    for key, value in output.items():
        if key.startswith("target_tokens/"):
            counts[f"/{key.removeprefix('target_tokens/')}"] = float(
                value.detach().cpu()
            )
    return counts


def _metric_weight(key: str, tokens_by_suffix: dict[str, float]) -> float:
    if "/" not in key:
        return tokens_by_suffix[""]
    suffix = "/" + key.split("/", 1)[1]
    return tokens_by_suffix[suffix]


def _validation_accuracy_summary(metrics: dict[str, float]) -> str:
    items = []
    for key in sorted(metrics):
        if key.startswith("val/token_accuracy/"):
            name = key.removeprefix("val/token_accuracy/")
            items.append(f"val_acc/{name}={metrics[key]:.4f}")
    return " ".join(items)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    datamodule: SpectraARGemsDataModule,
    global_step: int,
    config: config_dict.ConfigDict,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": global_step,
            "config": config.to_dict(),
            "tokenizer_config": asdict(datamodule.ar_tokenizer_config),
            "data_info": datamodule.info,
        },
        path,
    )


def _scalar_metrics(output: dict[str, torch.Tensor]) -> dict[str, float]:
    return {
        key: float(value.detach().cpu())
        for key, value in output.items()
        if key != "logits"
    }


def _prefix_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}/{key}": value for key, value in metrics.items()}
