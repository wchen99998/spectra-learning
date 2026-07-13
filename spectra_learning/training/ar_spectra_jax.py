from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import torch
from jax.experimental import multihost_utils
from ml_collections import config_dict

from spectra_learning.data.ar_spectra import SpectraARGemsDataModule
from spectra_learning.models.ar_spectra_jax import build_spectra_ar_model_jax_from_config
from spectra_learning.training.checkpointing_jax import build_jax_checkpoint_manager
from spectra_learning.training.configuration import save_config
from spectra_learning.training.logging import MetricLogger, build_logger
from spectra_learning.training.pretrain_jax import (
    _config_get,
    _jax_data_parallel_devices,
    _run_jax_training_loop,
    collect_jax_param_metrics,
    configure_jax_runtime,
    initialize_jax_distributed,
)
from spectra_learning.training.storage import (
    local_scratch_dir,
    normalize_storage_path,
    storage_join,
    storage_mkdir,
)


def train_and_evaluate_ar_spectra_jax(
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
    multihost_utils.sync_global_devices("spectra_ar_jax_workdir_ready")

    torch.manual_seed(int(config.seed))
    config.dataloader_pin_memory = False
    config.dataloader_persistent_workers = False
    if int(_config_get(config, "dataloader_num_workers", 0)) > 0:
        config.dataloader_multiprocessing_context = str(
            _config_get(config, "dataloader_multiprocessing_context", "forkserver")
            or "forkserver"
        )
    config.msg_probe_every_n_steps = float(
        _config_get(config, "msg_probe_every_n_steps", -1.0)
    )
    config.msg_probe_at_final_step = bool(
        _config_get(config, "msg_probe_at_final_step", False)
    )
    if is_main_process:
        save_config(config, workdir)

    datamodule = SpectraARGemsDataModule(
        config,
        seed=int(config.seed),
        distributed_world_size=jax.process_count(),
        distributed_rank=jax.process_index(),
        distributed_local_rank=0,
    )
    total_steps = _total_training_steps(config, datamodule)
    model = build_spectra_ar_model_jax_from_config(config, datamodule.ar_tokenizer)
    checkpoint_dir = storage_join(workdir, "checkpoints")
    if is_main_process:
        storage_mkdir(checkpoint_dir)
    multihost_utils.sync_global_devices("spectra_ar_jax_checkpoint_dir_ready")
    checkpoint_manager = build_jax_checkpoint_manager(
        checkpoint_dir,
        max_to_keep=_jax_checkpoint_max_to_keep(config),
        enable_async_checkpointing=bool(
            _config_get(config, "jax_enable_async_checkpointing", True)
        ),
    )
    logger = build_logger(config, local_workdir) if is_main_process else MetricLogger()
    param_metrics = collect_jax_param_metrics(model)
    if is_main_process:
        logging.info("AR tokenizer: %s", asdict(datamodule.ar_tokenizer_config))
        logging.info("AR vocab size: %d", datamodule.ar_tokenizer.vocab_size)
        logging.info("AR sequence length: %d", datamodule.ar_tokenizer.sequence_length)
        logging.info(
            "Training JAX AR spectra model for %d optimizer steps on %d process(es).",
            total_steps,
            jax.process_count(),
        )
        logger.log_metrics(param_metrics, step=int(checkpoint_manager.latest_step() or 0))
    metrics = _run_jax_training_loop(
        config=config,
        datamodule=datamodule,
        model=model,
        logger=logger,
        total_steps=total_steps,
        checkpoint_manager=checkpoint_manager,
        resume_step=checkpoint_manager.latest_step(),
    )
    checkpoint_manager.close()
    data_parallel_devices = _jax_data_parallel_devices(config)
    run_metrics: dict[str, object] = {
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
        "run/training_task": "ar_spectra",
        "run/ar_vocab_size": float(datamodule.ar_tokenizer.vocab_size),
        "run/ar_sequence_length": float(datamodule.ar_tokenizer.sequence_length),
    }
    results = {**metrics, **run_metrics, **param_metrics}
    if is_main_process:
        final_global_step = int(metrics["run/final_global_step"])
        logger.log_metrics(
            {"global_step": float(final_global_step), **results},
            step=final_global_step,
        )
        wandb_run = getattr(logger, "experiment", None)
        if wandb_run is not None:
            wandb_run.finish()
    return results


def _total_training_steps(
    config: config_dict.ConfigDict,
    datamodule: SpectraARGemsDataModule,
) -> int:
    total_steps = max(1, int(float(config.num_epochs) * datamodule.train_steps))
    training_max_steps = _config_get(config, "training_max_steps", None)
    if training_max_steps is None:
        return total_steps
    return min(total_steps, max(1, int(training_max_steps)))


def _jax_checkpoint_max_to_keep(config: config_dict.ConfigDict) -> int | None:
    value: Any = _config_get(config, "jax_checkpoint_max_to_keep", 5)
    if value is None:
        return None
    return int(value)
