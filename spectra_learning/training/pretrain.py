import gc
import logging
import math
import random
import signal
import time
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch._dynamo
import torch._inductor.config as inductor_config
from ml_collections import config_dict
from tqdm import tqdm

from spectra_learning.data.contracts import (
    data_provenance_contract,
    peak_preprocessing_contract,
    validate_data_provenance_contract,
    validate_peak_preprocessing_contract,
)
from spectra_learning.data.gems.datamodule import GemsDataModule
from spectra_learning.models.factory import build_model_from_config
from spectra_learning.models.model import PeakSetJEPA
from spectra_learning.probes.massspec.msg_probe import (
    msg_probe_variants_from_config,
    run_msg_probe,
)
from spectra_learning.training.activation_checkpointing import apply_activation_checkpointing
from spectra_learning.training.batch import BatchPrefetcher
from spectra_learning.training.cadence import (
    msg_probe_interval as resolve_msg_probe_interval,
    should_run_at_step,
    should_run_at_step_or_final,
    total_training_steps,
    validation_interval,
    validation_steps,
)
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
    max_across_ranks,
    reduce_metric_tensors,
    unwrap_model,
    wrap_distributed_model,
)
from spectra_learning.training.configuration import save_config
from spectra_learning.training.logging import (
    MetricLogger,
    build_logger,
    log_msg_probe_metrics,
)
from spectra_learning.training.optimization import build_optimizers
from spectra_learning.training.performance import (
    compile_forward as compile_training_forward,
    register_bf16_adam_state_hooks,
)
from spectra_learning.training.runtime import (
    build_grad_scaler,
    collect_and_log_param_metrics,
    cumulative_training_flops,
    estimate_training_flops_per_optimizer_step,
    parse_autocast_dtype,
)
from spectra_learning.training.schedules import LRSchedulerLike
from spectra_learning.training.storage import (
    StoragePath,
    local_scratch_dir,
    normalize_storage_path,
    storage_join,
    storage_mkdir,
)
from spectra_learning.training.steps import train_step_impl

warnings.filterwarnings("ignore", message="Profiler function.*will be ignored")
torch.set_float32_matmul_precision("high")
inductor_config.triton.unique_kernel_names = True
inductor_config.fx_graph_cache = True
inductor_config.epilogue_fusion = True
inductor_config.shape_padding = True

_STOP_REQUESTED = False


def gradient_accumulation_steps(config: config_dict.ConfigDict) -> int:
    return int(config.get("gradient_accumulation_steps", 1))


def effective_compile_mode(config: config_dict.ConfigDict) -> str:
    compile_mode = str(config.get("compile_mode", "max-autotune"))
    if (
        compile_mode.lower() == "max-autotune"
        and gradient_accumulation_steps(config) > 1
    ):
        return "max-autotune-no-cudagraphs"
    return compile_mode


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
    workdir = normalize_storage_path(workdir)
    local_workdir = local_scratch_dir(workdir)
    if distributed.is_main:
        local_workdir.mkdir(parents=True, exist_ok=True)
        storage_mkdir(workdir)
    barrier(distributed)
    seed_all(int(config.seed))
    datamodule = GemsDataModule(
        config,
        seed=int(config.seed),
        distributed_world_size=distributed.world_size,
        distributed_rank=distributed.rank,
        distributed_local_rank=distributed.local_rank,
    )
    total_steps = total_training_steps(config, datamodule)
    loop_epochs = max(1, math.ceil(float(config.num_epochs)))
    if distributed.is_main:
        logging.info("Training for %s epochs (%d steps).", config.num_epochs, total_steps)
        logging.info("Steps per epoch: %d", datamodule.train_steps)
        logging.info(
            "Distributed: world_size=%d global_batch_size=%d "
            "local_micro_batch_size=%d grad_accum_steps=%d",
            distributed.world_size,
            datamodule.global_batch_size,
            datamodule.batch_size,
            gradient_accumulation_steps(config),
        )
    device = distributed.device
    clear_cuda_cache(device)
    model = build_model_from_config(config)
    initialize_frozen_teacher(config, model, distributed)
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
    model.to(device).train()
    apply_activation_checkpointing(model, config)
    autocast_dtype = parse_autocast_dtype(config.get("autocast_dtype", "bf16"))
    grad_clip_norm = optional_float(config.get("grad_clip_norm", None))
    grad_scaler = build_grad_scaler(autocast_dtype, device)
    checkpoint_dir = storage_join(workdir, "checkpoints")
    if distributed.is_main:
        storage_mkdir(checkpoint_dir)
    optimizers, schedulers = build_optimizers(
        config,
        model,
        total_steps,
        device,
    )
    register_bf16_adam_state_hooks(optimizers, config)
    start_epoch, global_step, resume_offset = restore_training_state(
        config=config,
        checkpoint_dir=checkpoint_dir,
        model=model,
        optimizers=optimizers,
        schedulers=schedulers,
        grad_scaler=grad_scaler,
        steps_per_epoch=datamodule.train_steps,
        device=device,
        data_provenance=dict(datamodule.info),
    )
    if distributed.is_main:
        save_config(config, workdir)
        logger = build_logger(config, local_workdir)
    else:
        logger = MetricLogger()
    compile_forward(model, config)
    train_model = wrap_distributed_model(
        model,
        distributed,
        static_graph=bool(
            config.get(
                "ddp_static_graph",
                gradient_accumulation_steps(config) == 1,
            )
        ),
        find_unused_parameters=bool(
            config.get("ddp_find_unused_parameters", False)
        ),
    )
    if distributed.is_main:
        logger.log_metrics(model_param_metrics, step=global_step)
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
        flops_per_optimizer_step=flops_per_optimizer_step,
    )
    final_global_step = int(cast(float, last_msg_probe_metrics["run/final_global_step"]))
    if distributed.is_main:
        base_model = cast(PeakSetJEPA, unwrap_model(train_model))
        checkpoint_writer.save_checkpoint(
            storage_join(checkpoint_dir, "last.pt"),
            base_model,
            optimizers,
            schedulers,
            final_global_step,
            final_global_step // datamodule.train_steps,
            float("nan"),
            getattr(logger.experiment, "id", None),
            peak_preprocessing=peak_preprocessing_contract(config),
            data_provenance=data_provenance_contract(datamodule.info),
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
        "run/gradient_accumulation_steps": float(gradient_accumulation_steps(config)),
    }
    cleanup_distributed(distributed)
    return results


def initialize_frozen_teacher(
    config: config_dict.ConfigDict,
    model: PeakSetJEPA,
    distributed: DistributedContext,
) -> None:
    if str(config.get("training_mode", "jepa")).lower() != "mae_teacher_jepa":
        return
    checkpoint_path = str(config.frozen_teacher_checkpoint_path)
    if distributed.is_main:
        logging.info("Loading frozen MAE teacher from %s.", checkpoint_path)
    load_frozen_teacher_weights(model, checkpoint_path, config=config)


@dataclass(frozen=True)
class _TorchTrainingLoopInputs:
    config: config_dict.ConfigDict
    datamodule: Any
    model: torch.nn.Module
    optimizers: list[torch.optim.Optimizer]
    schedulers: list[LRSchedulerLike]
    logger: Any
    checkpoint_dir: StoragePath
    start_epoch: int
    loop_epochs: int
    resume_offset: int
    global_step: int
    total_steps: int
    device: torch.device
    autocast_dtype: torch.dtype | None
    grad_scaler: torch.amp.GradScaler | None
    distributed: DistributedContext | None
    checkpoint_writer: AsyncCheckpointWriter | None
    flops_per_optimizer_step: float | None


class _TorchTrainingLoop:
    def __init__(self, inputs: _TorchTrainingLoopInputs) -> None:
        self.config = inputs.config
        self.datamodule = inputs.datamodule
        self.model = inputs.model
        self.optimizers = inputs.optimizers
        self.schedulers = inputs.schedulers
        self.logger = inputs.logger
        self.checkpoint_dir = inputs.checkpoint_dir
        self.start_epoch = inputs.start_epoch
        self.loop_epochs = inputs.loop_epochs
        self.resume_offset = inputs.resume_offset
        self.global_step = inputs.global_step
        self.total_steps = inputs.total_steps
        self.device = inputs.device

        self.distributed = inputs.distributed
        if self.distributed is None:
            self.distributed = DistributedContext(
                rank=0,
                local_rank=0,
                world_size=1,
                device=self.device,
            )
        self.autocast_dtype = inputs.autocast_dtype
        if self.autocast_dtype is None:
            self.autocast_dtype = parse_autocast_dtype(
                self.config.get("autocast_dtype", "bf16")
            )
        self.grad_scaler = inputs.grad_scaler
        if self.grad_scaler is None:
            self.grad_scaler = build_grad_scaler(
                self.autocast_dtype,
                self.device,
            )
        self.owns_checkpoint_writer = inputs.checkpoint_writer is None
        self.checkpoint_writer = inputs.checkpoint_writer
        if self.checkpoint_writer is None:
            self.checkpoint_writer = AsyncCheckpointWriter()
        self.flops_per_optimizer_step = inputs.flops_per_optimizer_step
        if self.flops_per_optimizer_step is None:
            self.flops_per_optimizer_step = (
                estimate_training_flops_per_optimizer_step(
                    self.config,
                    unwrap_model(self.model),
                    int(self.datamodule.global_batch_size),
                )
            )

        self.log_every_n_steps = int(self.config.get("log_every_n_steps", 50))
        self.collapse_every_n_steps = int(
            self.config.get(
                "collapse_metrics_every_n_steps",
                self.log_every_n_steps,
            )
        )
        self.checkpoint_every_steps = int(self.config.checkpoint_every_steps)
        self.grad_clip_norm = optional_float(
            self.config.get("grad_clip_norm", None)
        )
        self.grad_accum_steps = gradient_accumulation_steps(self.config)
        self.msg_probe_every_n_steps = msg_probe_interval(
            self.config,
            self.datamodule,
            self.total_steps,
        )
        self.val_every_n_steps = validation_interval(
            self.config,
            self.datamodule,
            self.total_steps,
        )
        self.val_num_steps = validation_steps(self.config)
        self.msg_probe_variants = msg_probe_variants_from_config(self.config)
        self.device_prefetch_size = int(
            self.config.get("device_prefetch_size", 1)
        )
        self.deadline = training_deadline(self.config)
        self.wandb_run = getattr(self.logger, "experiment", None)
        self.last_msg_probe_metrics: dict[str, object] = {}
        self.last_validation_metrics: dict[str, object] = {}
        self.stopped_for_time_limit = False
        self.stopped_for_signal = False
        self.initial_global_step = self.global_step
        self.training_start_time = 0.0
        self.throughput_warmup_steps = 0
        self.measured_start_time: float | None = None
        self.measured_steps = 0
        self.validation_seconds = 0.0
        self.msg_probe_seconds = 0.0
        self.profiler: Any = None

    def run(self) -> dict[str, object]:
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.training_start_time = time.perf_counter()
        self.throughput_warmup_steps = int(
            self.config.get("throughput_warmup_steps", 0)
        )
        self.profiler = make_torch_profiler(
            self.config,
            self.distributed,
            self.device,
        )
        if self.profiler is not None:
            self.profiler.start()
        for epoch in range(self.start_epoch, self.loop_epochs):
            self._run_epoch(epoch)
            if (
                self.stopped_for_time_limit
                or self.stopped_for_signal
                or self.global_step >= self.total_steps
            ):
                break
        return self._finish()

    def _run_epoch(self, epoch: int) -> None:
        if self.distributed.is_main:
            logging.info(
                "Starting epoch %d at global_step=%d",
                epoch,
                self.global_step,
            )
        epoch_resume_offset = (
            self.resume_offset if epoch == self.start_epoch else 0
        )
        if self.distributed.is_main and epoch_resume_offset:
            logging.info(
                "Resuming epoch %d from batch offset %d.",
                epoch,
                epoch_resume_offset,
            )
        train_loader = self.datamodule.train_loader_for_epoch(
            epoch,
            start_batch=epoch_resume_offset,
        )
        prefetcher = BatchPrefetcher(
            iter(train_loader),
            self.device,
            prefetch_size=self.device_prefetch_size,
        )
        pbar = tqdm(
            total=min(
                self.datamodule.train_steps - epoch_resume_offset,
                self.total_steps - self.global_step,
            ),
            desc=f"Epoch {epoch}",
            unit="step",
            disable=(
                not self.distributed.is_main
                or bool(self.config.get("disable_progress_bar", False))
            ),
        )
        accumulation_step = 0
        while (
            self.global_step < self.total_steps
            and (batch := prefetcher.next()) is not None
        ):
            if self._stop_requested():
                break
            self._start_measurement_if_ready()
            metrics = self._train_microbatch(batch, accumulation_step)
            accumulation_step += 1
            optimizer_step = bool(
                float(
                    metrics.get(
                        "optimizer_step",
                        metrics["loss"].new_tensor(1.0),
                    )
                )
            )
            if optimizer_step:
                self._finish_optimizer_step(metrics, epoch, pbar)
        pbar.close()
        if self.distributed.is_main:
            logging.info(
                "Finished epoch %d at global_step=%d",
                epoch,
                self.global_step,
            )

    def _stop_requested(self) -> bool:
        if stop_requested_on_any_rank(self.distributed):
            if self.distributed.is_main:
                logging.info(
                    "Received stop request at global_step=%d.",
                    self.global_step,
                )
            self.stopped_for_signal = True
            return True
        if self.deadline is not None and any_rank(
            time.perf_counter() >= self.deadline,
            self.distributed,
        ):
            if self.distributed.is_main:
                logging.info(
                    "Reached max_duration_hours at global_step=%d.",
                    self.global_step,
                )
            self.stopped_for_time_limit = True
            return True
        return False

    def _start_measurement_if_ready(self) -> None:
        if self.measured_start_time is not None or (
            self.global_step - self.initial_global_step
        ) < self.throughput_warmup_steps:
            return
        synchronize_device(self.device)
        barrier(self.distributed)
        self.measured_start_time = time.perf_counter()

    def _train_microbatch(
        self,
        batch: Any,
        accumulation_step: int,
    ) -> dict[str, torch.Tensor]:
        next_micro_step_is_boundary = (
            (accumulation_step + 1) % self.grad_accum_steps == 0
        )
        return train_step_impl(
            self.model,
            batch,
            self.optimizers,
            self.schedulers,
            self.autocast_dtype,
            self.grad_clip_norm,
            grad_scaler=self.grad_scaler,
            compute_collapse_metrics=(
                next_micro_step_is_boundary
                and self.collapse_every_n_steps > 0
                and (self.global_step + 1) % self.collapse_every_n_steps == 0
            ),
            global_step=self.global_step,
            total_steps=self.total_steps,
            gradient_accumulation_steps=self.grad_accum_steps,
            accumulation_step=accumulation_step,
        )

    def _finish_optimizer_step(
        self,
        metrics: dict[str, torch.Tensor],
        epoch: int,
        pbar: Any,
    ) -> None:
        self.global_step += 1
        if self.profiler is not None:
            self.profiler.step()
        if self.measured_start_time is not None:
            self.measured_steps += 1
        pbar.update(1)
        self._log_train_step(metrics, epoch, pbar)
        self._save_periodic_checkpoint(metrics)
        self._run_validation_if_due(pbar)
        self._run_msg_probe_if_due()
        if self.distributed.is_main:
            self.checkpoint_writer.log_completed_failures()

    def _log_train_step(
        self,
        metrics: dict[str, torch.Tensor],
        epoch: int,
        pbar: Any,
    ) -> None:
        should_log = (
            self.log_every_n_steps > 0
            and self.global_step % self.log_every_n_steps == 0
        )
        log_metrics = (
            reduce_metric_tensors(metrics, self.distributed)
            if should_log
            else metrics
        )
        if self.distributed.is_main:
            log_train_metrics(
                self.config,
                self.logger,
                pbar,
                log_metrics,
                self.optimizers,
                epoch=epoch,
                global_step=self.global_step,
                every_n_steps=self.log_every_n_steps,
                flops_per_optimizer_step=self.flops_per_optimizer_step,
            )

    def _save_periodic_checkpoint(
        self,
        metrics: dict[str, torch.Tensor],
    ) -> None:
        if self.global_step % self.checkpoint_every_steps != 0:
            return
        if self.distributed.is_main:
            base_model = cast(PeakSetJEPA, unwrap_model(self.model))
            self.checkpoint_writer.save_checkpoint(
                storage_join(
                    self.checkpoint_dir,
                    f"step-{self.global_step:08d}.pt",
                ),
                base_model,
                self.optimizers,
                self.schedulers,
                self.global_step,
                self.global_step // self.datamodule.train_steps,
                float(metrics["loss"]),
                getattr(self.wandb_run, "id", None),
                peak_preprocessing=peak_preprocessing_contract(self.config),
                data_provenance=data_provenance_contract(self.datamodule.info),
                grad_scaler=self.grad_scaler,
                prune_checkpoint_dir=self.checkpoint_dir,
                keep_top_k=15,
            )
        barrier(self.distributed)

    def _run_validation_if_due(self, pbar: Any) -> None:
        if not should_run_at_step(
            self.val_every_n_steps,
            self.global_step,
        ):
            return
        if self.device.type == "cuda":
            synchronize_device(self.device)
        started = time.perf_counter()
        val_metrics = evaluate_validation_loss(
            datamodule=self.datamodule,
            model=self.model,
            device=self.device,
            autocast_dtype=self.autocast_dtype,
            distributed=self.distributed,
            max_steps=self.val_num_steps,
            prefetch_size=self.device_prefetch_size,
        )
        if self.device.type == "cuda":
            synchronize_device(self.device)
        self.validation_seconds += time.perf_counter() - started
        self.last_validation_metrics = {
            f"val/{key}": float(value.detach())
            for key, value in val_metrics.items()
        }
        if self.distributed.is_main:
            log_validation_metrics(
                self.logger,
                pbar,
                self.last_validation_metrics,
                global_step=self.global_step,
            )

    def _run_msg_probe_if_due(self) -> None:
        if not should_run_at_step_or_final(
            self.msg_probe_every_n_steps,
            self.global_step,
            total_steps=self.total_steps,
            run_at_final_step=bool(
                self.config.get("msg_probe_at_final_step", False)
            ),
        ):
            return
        if self.device.type == "cuda":
            synchronize_device(self.device)
        started = time.perf_counter()
        base_model = cast(PeakSetJEPA, unwrap_model(self.model))
        self.last_msg_probe_metrics = dict(
            run_and_log_msg_probe(
                self.config,
                base_model,
                self.device,
                self.logger,
                self.msg_probe_variants,
                self.global_step,
                self.distributed,
            )
        )
        if self.device.type == "cuda":
            synchronize_device(self.device)
        barrier(self.distributed)
        self.msg_probe_seconds += time.perf_counter() - started

    def _finish(self) -> dict[str, object]:
        synchronize_device(self.device)
        barrier(self.distributed)
        if self.profiler is not None:
            self.profiler.stop()
        training_elapsed = time.perf_counter() - self.training_start_time
        measured_elapsed = (
            time.perf_counter() - self.measured_start_time
            if self.measured_start_time is not None
            else 0.0
        )
        result = self._build_result(training_elapsed, measured_elapsed)
        if self.owns_checkpoint_writer:
            self.checkpoint_writer.close()
        return result

    def _build_result(
        self,
        training_elapsed: float,
        measured_elapsed: float,
    ) -> dict[str, object]:
        global_batch_size = int(self.datamodule.global_batch_size)
        result = self.last_msg_probe_metrics
        result["run/stopped_for_time_limit"] = float(
            self.stopped_for_time_limit
        )
        result["run/stopped_for_signal"] = float(self.stopped_for_signal)
        result.update(self.last_validation_metrics)
        result["run/final_global_step"] = float(self.global_step)
        result["run/train_elapsed_seconds"] = training_elapsed
        completed_steps = float(self.global_step - self.initial_global_step)
        result["run/steps_per_second"] = (
            completed_steps / training_elapsed if training_elapsed > 0 else 0.0
        )
        result["run/samples_per_second"] = (
            completed_steps * global_batch_size / training_elapsed
            if training_elapsed > 0
            else 0.0
        )
        result["run/measured_steps"] = float(self.measured_steps)
        result["run/measured_elapsed_seconds"] = measured_elapsed
        measured_training_seconds = max(
            0.0,
            measured_elapsed - self.validation_seconds - self.msg_probe_seconds,
        )
        result["run/validation_seconds"] = self.validation_seconds
        result["run/msg_probe_seconds"] = self.msg_probe_seconds
        result["run/measured_training_seconds"] = measured_training_seconds
        result["run/measured_steps_per_second"] = (
            float(self.measured_steps) / measured_elapsed
            if measured_elapsed > 0
            else 0.0
        )
        result["run/measured_samples_per_second"] = (
            float(self.measured_steps) * global_batch_size / measured_elapsed
            if measured_elapsed > 0
            else 0.0
        )
        result["run/measured_training_steps_per_second"] = (
            float(self.measured_steps) / measured_training_seconds
            if measured_training_seconds > 0
            else 0.0
        )
        result["run/measured_training_samples_per_second"] = (
            float(self.measured_steps) * global_batch_size / measured_training_seconds
            if measured_training_seconds > 0
            else 0.0
        )
        if self.device.type == "cuda":
            peak_allocated = max_across_ranks(
                float(torch.cuda.max_memory_allocated(self.device)),
                self.distributed,
            )
            peak_reserved = max_across_ranks(
                float(torch.cuda.max_memory_reserved(self.device)),
                self.distributed,
            )
            result["run/peak_cuda_memory_allocated_bytes"] = peak_allocated
            result["run/peak_cuda_memory_reserved_bytes"] = peak_reserved
        return result


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
    flops_per_optimizer_step: float | None = None,
) -> dict[str, object]:
    return _TorchTrainingLoop(
        _TorchTrainingLoopInputs(
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
            autocast_dtype=autocast_dtype,
            grad_scaler=grad_scaler,
            distributed=distributed,
            checkpoint_writer=checkpoint_writer,
            flops_per_optimizer_step=flops_per_optimizer_step,
        )
    ).run()


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


def make_torch_profiler(
    config: config_dict.ConfigDict,
    distributed: DistributedContext,
    device: torch.device,
) -> torch.profiler.profile | None:
    profile_dir = str(config.get("torch_profile_dir", "") or "")
    if not profile_dir:
        return None
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=int(config.get("torch_profile_wait_steps", 1)),
            warmup=int(config.get("torch_profile_warmup_steps", 1)),
            active=int(config.get("torch_profile_active_steps", 3)),
            repeat=int(config.get("torch_profile_repeat", 1)),
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            profile_dir,
            worker_name=f"rank{distributed.rank}",
        ),
        record_shapes=bool(config.get("torch_profile_record_shapes", False)),
        profile_memory=bool(config.get("torch_profile_memory", False)),
        with_stack=bool(config.get("torch_profile_with_stack", False)),
        with_flops=bool(config.get("torch_profile_with_flops", False)),
    )


def restore_training_state(
    *,
    config: config_dict.ConfigDict,
    checkpoint_dir: StoragePath,
    model: PeakSetJEPA,
    optimizers: list[torch.optim.Optimizer],
    schedulers: list[LRSchedulerLike],
    steps_per_epoch: int,
    device: torch.device,
    data_provenance: dict[str, Any],
    grad_scaler: torch.amp.GradScaler | None = None,
) -> tuple[int, int, int]:
    checkpoints = training_checkpoint_paths(checkpoint_dir)
    if not checkpoints:
        return 0, 0, 0
    ckpt_path = checkpoints[-1]
    logging.info("Resuming from checkpoint: %s", ckpt_path)
    ckpt = load_torch_checkpoint(ckpt_path, map_location=device, weights_only=True)
    validate_peak_preprocessing_contract(ckpt, config)
    validate_data_provenance_contract(ckpt, data_provenance)
    _ = ckpt["loss"]
    resume_wandb_id = ckpt["wandb_run_id"]
    if resume_wandb_id:
        config.wandb_resume_id = resume_wandb_id
    load_resume_model_state(model, ckpt["model"])
    for optimizer, state in zip(optimizers, ckpt["optimizers"], strict=True):
        load_optimizer_state(optimizer, state)
    for scheduler, state in zip(schedulers, ckpt["schedulers"], strict=True):
        scheduler.load_state_dict(state)
    load_grad_scaler_state(grad_scaler, ckpt["grad_scaler"])
    global_step = int(ckpt["global_step"])
    start_epoch = int(ckpt["epoch"])
    resume_offset = global_step - start_epoch * steps_per_epoch
    start_epoch += resume_offset // steps_per_epoch
    resume_offset %= steps_per_epoch
    return start_epoch, global_step, resume_offset


def compile_forward(model: torch.nn.Module, config: config_dict.ConfigDict) -> None:
    requested_compile_mode = str(config.get("compile_mode", "max-autotune"))
    compile_mode = effective_compile_mode(config)
    if compile_mode.lower() == "none":
        return
    inductor_config.shape_padding = not compile_mode.startswith("max-autotune")
    inductor_config.triton.cudagraph_skip_dynamic_graphs = bool(
        config.get("cudagraph_skip_dynamic_graphs", False)
    )
    cudagraph_trees = config.get("cudagraph_trees", None)
    if cudagraph_trees is not None:
        inductor_config.triton.cudagraph_trees = bool(cudagraph_trees)
    if inductor_config.triton.cudagraph_skip_dynamic_graphs:
        logging.info("Skipping CUDA Graph capture for dynamic-shape Inductor graphs.")
    if compile_mode != requested_compile_mode:
        logging.info(
            "Using torch.compile mode %s for requested mode %s with "
            "gradient_accumulation_steps=%d.",
            compile_mode,
            requested_compile_mode,
            gradient_accumulation_steps(config),
        )
    compile_training_forward(model, config, compile_mode=compile_mode)


def evaluate_validation_loss(
    *,
    datamodule: Any,
    model: torch.nn.Module,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
    distributed: DistributedContext,
    max_steps: int,
    prefetch_size: int,
) -> dict[str, torch.Tensor]:
    was_training = model.training
    model.eval()
    totals: dict[str, torch.Tensor] = {}
    steps = 0
    prefetcher = BatchPrefetcher(
        iter(datamodule.val_loader),
        device,
        prefetch_size=min(prefetch_size, max_steps),
    )

    def autocast_context():
        if autocast_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=device.type, dtype=autocast_dtype)

    with torch.no_grad():
        while steps < max_steps and (batch := prefetcher.next()) is not None:
            with autocast_context():
                metrics = cast(dict[str, torch.Tensor], model(batch))
            for key, value in metrics.items():
                totals[key] = totals.get(key, torch.zeros_like(value)) + value.detach()
            steps += 1
    if was_training:
        model.train()
    averaged = {key: value / float(steps) for key, value in totals.items()}
    return reduce_metric_tensors(averaged, distributed)


def log_validation_metrics(
    logger,
    pbar: tqdm,
    metrics: dict[str, object],
    *,
    global_step: int,
) -> None:
    pbar.set_postfix(val_loss=f"{float(metrics['val/loss']):.4f}", step=global_step)
    logger.log_metrics(
        {
            **metrics,
            "global_step": global_step,
        },
        step=global_step,
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
    flops_per_optimizer_step: float,
) -> None:
    if every_n_steps <= 0 or global_step % every_n_steps != 0:
        return
    loss_val = float(metrics["loss"].detach())
    pbar.set_postfix(loss=f"{loss_val:.4f}", step=global_step)
    log_metrics = {
        f"train/{key}": float(value.detach())
        for key, value in metrics.items()
    }
    log_metrics.update(learning_rate_metrics(config, optimizers))
    cumulative_flops = cumulative_training_flops(global_step, flops_per_optimizer_step)
    log_metrics["train/cumulative_flops"] = cumulative_flops
    log_metrics["train/cumulative_peta_flops"] = cumulative_flops / 1e15
    log_metrics["train/flops_per_optimizer_step"] = float(flops_per_optimizer_step)
    log_metrics["epoch"] = epoch
    log_metrics["global_step"] = global_step
    logger.log_metrics(log_metrics, step=global_step)


def learning_rate_metrics(
    config: config_dict.ConfigDict,
    optimizers: list[torch.optim.Optimizer],
) -> dict[str, float]:
    return {"train/learning_rate": float(optimizers[0].param_groups[0]["lr"])}


def msg_probe_interval(
    config: config_dict.ConfigDict,
    datamodule: GemsDataModule,
    total_steps: int,
) -> int:
    return resolve_msg_probe_interval(config, datamodule, total_steps)


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
        online_maccs_only=True,
    )
    if distributed.is_main:
        log_msg_probe_metrics(
            logger,
            probe_metrics,
            global_step,
            enable_wandb=bool(config.get("enable_wandb", False)),
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


def training_deadline(config: config_dict.ConfigDict) -> float | None:
    max_duration_hours = config.get("max_duration_hours", None)
    if max_duration_hours is None:
        return None
    logging.info("Training wall-clock budget: %.2f hours", float(max_duration_hours))
    return time.perf_counter() + float(max_duration_hours) * 3600.0


def optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
