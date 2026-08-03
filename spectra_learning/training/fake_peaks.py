from __future__ import annotations

import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from ml_collections import config_dict
from torch import Tensor, nn
from tqdm import tqdm

from spectra_learning.data.contracts import (
    data_provenance_contract,
    peak_preprocessing_contract,
    validate_data_provenance_contract,
    validate_peak_preprocessing_contract,
)
from spectra_learning.data.gems.datamodule import GemsDataModule
from spectra_learning.data.spectra import PEAK_MZ_MAX
from spectra_learning.models.fake_peaks import FakePeakDiscriminator
from spectra_learning.models.model import PeakSetJEPA
from spectra_learning.models.settings import PeakSetJEPASettings
from spectra_learning.models.spectrum_metadata import torch_spectrum_metadata_from_batch
from spectra_learning.training.cadence import (
    should_run_at_step,
    total_training_steps,
    validation_interval,
    validation_steps,
)
from spectra_learning.training.checkpointing import (
    load_torch_checkpoint,
    save_torch_checkpoint,
)
from spectra_learning.training.configuration import save_config
from spectra_learning.training.logging import build_logger
from spectra_learning.training.optimization import build_optimizers
from spectra_learning.training.performance import compile_forward
from spectra_learning.training.storage import (
    StoragePath,
    local_scratch_dir,
    normalize_storage_path,
    storage_exists,
    storage_join,
    storage_mkdir,
)


def sample_fake_mz(
    logits: Tensor,
    true_mz: Tensor,
    target_mask: Tensor,
    *,
    bin_size: float,
    top_k: int,
    temperature: float,
) -> tuple[Tensor, Tensor]:
    true_scaled = true_mz.float() * PEAK_MZ_MAX / bin_size
    true_bins = torch.floor(true_scaled).long().clamp(0, logits.shape[-1] - 1)
    if top_k == 1:
        sampled_bins = logits.argmax(dim=-1)
    else:
        top_logits, top_bins = logits.topk(min(top_k, logits.shape[-1]), dim=-1)
        sampled = torch.multinomial(
            F.softmax(top_logits.float() / temperature, dim=-1).flatten(0, -2),
            1,
        ).reshape(true_mz.shape)
        sampled_bins = torch.gather(top_bins, -1, sampled.unsqueeze(-1)).squeeze(-1)
    exact_bin_mask = target_mask & sampled_bins.eq(true_bins)
    sub_bin_residual = torch.rand_like(true_mz)
    sampled_mz = (sampled_bins.float() + sub_bin_residual) * bin_size / PEAK_MZ_MAX
    return torch.where(target_mask, sampled_mz, true_mz), exact_bin_mask


def build_fake_peak_batch(
    batch: dict[str, Tensor],
    generated_mz: Tensor,
    generator_exact_bin_mask: Tensor,
) -> dict[str, Tensor]:
    target_mask = batch["target_masks"][:, 0] & batch["peak_valid_mask"]
    donor_dimension = 0 if batch["peak_intensity"].shape[0] > 1 else 1
    # ponytail: this generator has no intensity head; replace this donor when it does.
    generated_intensity = torch.where(
        target_mask,
        torch.roll(batch["peak_intensity"], shifts=1, dims=donor_dimension),
        batch["peak_intensity"],
    )
    order = generated_intensity.argsort(dim=1, descending=True)
    generated = dict(batch)
    generated.update(
        {
            "peak_mz": generated_mz,
            "peak_intensity": generated_intensity,
            "true_peak_mz": batch["peak_mz"],
            "true_peak_intensity": batch["peak_intensity"],
            "fake_peak_mask": target_mask,
            "generator_exact_bin_mask": generator_exact_bin_mask,
            "student_visible_mask": batch["context_mask"] | target_mask,
        }
    )
    per_peak_keys = (
        "peak_mz",
        "peak_intensity",
        "peak_valid_mask",
        "context_mask",
        "true_peak_mz",
        "true_peak_intensity",
        "fake_peak_mask",
        "generator_exact_bin_mask",
        "student_visible_mask",
    )
    for key in per_peak_keys:
        generated[key] = torch.gather(generated[key], 1, order)
    generated["target_masks"] = torch.gather(
        generated["target_masks"],
        2,
        order.unsqueeze(1).expand_as(generated["target_masks"]),
    )
    return generated


class FrozenPeakGenerator:
    def __init__(self, config: config_dict.ConfigDict) -> None:
        self.device = torch.device(str(config.generator_device))
        checkpoint = load_torch_checkpoint(
            str(config.generator_checkpoint_path),
            map_location="cpu",
            weights_only=True,
        )
        if checkpoint["format_version"] != 1:
            raise ValueError("Unsupported frozen generator format.")
        if checkpoint["source_checkpoint"] != str(config.generator_source_checkpoint):
            raise ValueError("Frozen generator source checkpoint mismatch.")
        settings = PeakSetJEPASettings(**checkpoint["settings"])
        self.model = PeakSetJEPA(settings)
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(self.device).eval().requires_grad_(False)
        compile_forward(
            self.model,
            config,
            compile_mode=str(config.generator_compile_mode),
        )
        self.bin_size = float(settings.jepa_mae_mz_bin_size)
        self.top_k = int(config.generator_top_k)
        self.temperature = float(config.generator_temperature)

    @torch.inference_mode()
    def __call__(self, cpu_batch: dict[str, Tensor]) -> tuple[dict[str, Tensor], float]:
        started = time.perf_counter()
        torch.cuda.set_device(self.device)
        batch = {
            key: value.to(self.device, non_blocking=True)
            for key, value in cpu_batch.items()
        }
        peak_valid_mask = batch["peak_valid_mask"]
        context_mask = batch["context_mask"] & peak_valid_mask
        target_masks = batch["target_masks"] & peak_valid_mask.unsqueeze(1)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            context_emb = self.model.encoder(
                batch["peak_mz"],
                batch["peak_intensity"],
                valid_mask=peak_valid_mask,
                visible_mask=context_mask,
                precursor_mz=batch.get("precursor_mz"),
                spectrum_metadata=torch_spectrum_metadata_from_batch(batch),
            )
            _, predictor_output = self.model._predict_augmented_target_outputs(
                context_emb,
                context_mask,
                target_masks,
            )
            mz_head = self.model.jepa_mae_mz_head
            assert mz_head is not None
            logits = mz_head(predictor_output)[:, 0]
        generated_mz, exact_bin_mask = sample_fake_mz(
            logits,
            batch["peak_mz"],
            target_masks[:, 0],
            bin_size=self.bin_size,
            top_k=self.top_k,
            temperature=self.temperature,
        )
        generated = build_fake_peak_batch(
            batch,
            generated_mz,
            exact_bin_mask,
        )
        generated = {key: value.cpu() for key, value in generated.items()}
        return generated, time.perf_counter() - started


class GeneratedBatchPrefetcher:
    def __init__(
        self,
        batches: Iterable[dict[str, Tensor]],
        generator: FrozenPeakGenerator,
    ) -> None:
        self._batches = iter(batches)
        self._generator = generator
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="frozen-peak-generator",
        )
        self._future = self._submit_next()

    def _submit_next(self) -> Future[tuple[dict[str, Tensor], float]] | None:
        batch = next(self._batches, None)
        return None if batch is None else self._executor.submit(self._generator, batch)

    def next(self) -> tuple[dict[str, Tensor], float, float, float] | None:
        if self._future is None:
            return None
        started = time.perf_counter()
        batch, generation_seconds = self._future.result()
        generator_wait_seconds = time.perf_counter() - started
        started = time.perf_counter()
        self._future = self._submit_next()
        data_wait_seconds = time.perf_counter() - started
        return (
            batch,
            generation_seconds,
            generator_wait_seconds,
            data_wait_seconds,
        )

    def wait(self) -> None:
        if self._future is not None:
            self._future.result()

    def close(self) -> None:
        if self._future is not None:
            self._future.result()
        self._executor.shutdown(wait=True)


def _to_device(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def _evaluate(
    model: FakePeakDiscriminator,
    generator: FrozenPeakGenerator,
    batches: Iterable[dict[str, Tensor]],
    device: torch.device,
    max_steps: int,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    steps = 0
    prefetcher = GeneratedBatchPrefetcher(islice(batches, max_steps), generator)
    try:
        while steps < max_steps and (item := prefetcher.next()) is not None:
            batch, _, _, _ = item
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                metrics = model(_to_device(batch, device))
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            steps += 1
    finally:
        prefetcher.close()
    model.train()
    return {key: value / steps for key, value in totals.items()}


def _save_checkpoint(
    path: StoragePath,
    model: FakePeakDiscriminator,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    global_step: int,
    loss: float,
    total_steps: int,
    steps_per_epoch: int,
    data_info: dict[str, Any],
    config: config_dict.ConfigDict,
    wandb_run_id: str | None,
) -> None:
    save_torch_checkpoint(
        {
            "format_version": 2,
            "model": model.state_dict(),
            "optimizers": [optimizer.state_dict()],
            "schedulers": [scheduler.state_dict()],
            "global_step": global_step,
            "epoch": global_step // steps_per_epoch,
            "loss": loss,
            "total_steps": total_steps,
            "steps_per_epoch": steps_per_epoch,
            "wandb_run_id": wandb_run_id,
            "generator_source_checkpoint": str(config.generator_source_checkpoint),
            "peak_preprocessing": peak_preprocessing_contract(config),
            "data_provenance": data_provenance_contract(data_info),
        },
        path,
    )


def train_fake_peak_discriminator(
    config: config_dict.ConfigDict,
    workdir: str | Path,
) -> dict[str, object]:
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("fake_peak training uses one process controlling both GPUs.")
    workdir = normalize_storage_path(workdir)
    local_workdir = local_scratch_dir(workdir)
    local_workdir.mkdir(parents=True, exist_ok=True)
    storage_mkdir(workdir)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(
        local_workdir / "generator_torchinductor"
    )
    checkpoint_path = storage_join(workdir, "checkpoints", "last.pt")

    torch.manual_seed(int(config.seed))
    train_device = torch.device(str(config.discriminator_device))
    generator = FrozenPeakGenerator(config)
    model = FakePeakDiscriminator(config).to(train_device).train()
    datamodule = GemsDataModule(config, seed=int(config.seed))
    total_steps = total_training_steps(config, datamodule)
    accumulation_steps = int(config.gradient_accumulation_steps)
    optimizers, schedulers = build_optimizers(
        config,
        model,
        total_steps,
        train_device,
    )
    optimizer, scheduler = optimizers[0], schedulers[0]
    global_step = 0
    if storage_exists(checkpoint_path):
        checkpoint = load_torch_checkpoint(checkpoint_path, map_location="cpu")
        if checkpoint["format_version"] != 2:
            raise ValueError("Unsupported fake-peak checkpoint format; use a new workdir.")
        if checkpoint["generator_source_checkpoint"] != str(
            config.generator_source_checkpoint
        ):
            raise ValueError("Checkpoint generator source mismatch.")
        if checkpoint["total_steps"] != total_steps:
            raise ValueError("Checkpoint training length does not match the config.")
        if checkpoint["steps_per_epoch"] != datamodule.train_steps:
            raise ValueError("Checkpoint epoch length does not match the dataset.")
        validate_peak_preprocessing_contract(checkpoint, config)
        validate_data_provenance_contract(checkpoint, datamodule.info)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizers"][0])
        scheduler.load_state_dict(checkpoint["schedulers"][0])
        global_step = int(checkpoint["global_step"])
        if checkpoint["wandb_run_id"]:
            config.wandb_resume_id = checkpoint["wandb_run_id"]
    save_config(config, workdir)
    logger = build_logger(config, local_workdir)
    wandb_run_id = getattr(logger.experiment, "id", None)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    logger.log_metrics(
        {
            "global_step": global_step,
            "model/params_total": parameter_count,
            "model/generator_params_frozen": sum(
                parameter.numel() for parameter in generator.model.parameters()
            ),
        },
        step=global_step,
    )

    val_metrics = _evaluate(
        model,
        generator,
        datamodule.val_loader,
        train_device,
        validation_steps(config),
    )
    logger.log_metrics(
        {
            "global_step": global_step,
            **{f"val/{key}": value for key, value in val_metrics.items()},
        },
        step=global_step,
    )

    checkpoint_every = int(config.checkpoint_every_steps)
    log_every = int(config.log_every_n_steps)
    val_every = validation_interval(config, datamodule, total_steps)
    val_num_steps = validation_steps(config)
    progress = tqdm(
        total=total_steps,
        initial=global_step,
        disable=bool(config.get("disable_progress_bar", False)),
    )
    started = time.perf_counter()
    epoch = global_step // datamodule.train_steps
    resume_offset = global_step % datamodule.train_steps
    while global_step < total_steps:
        prefetcher = GeneratedBatchPrefetcher(
            datamodule.train_loader_for_epoch(epoch, start_batch=resume_offset),
            generator,
        )
        try:
            for _ in range(resume_offset, datamodule.train_steps):
                if global_step >= total_steps:
                    break
                optimizer.zero_grad(set_to_none=True)
                metric_totals: dict[str, Tensor] = {}
                generation_seconds = 0.0
                generator_wait_seconds = 0.0
                data_wait_seconds = 0.0
                step_started = time.perf_counter()
                for _ in range(accumulation_steps):
                    item = prefetcher.next()
                    assert item is not None
                    cpu_batch, generated_for, waited_for, waited_for_data = item
                    generation_seconds += generated_for
                    generator_wait_seconds += waited_for
                    data_wait_seconds += waited_for_data
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        metrics = model(_to_device(cpu_batch, train_device))
                        (metrics["loss"] / accumulation_steps).backward()
                    for key, value in metrics.items():
                        metric_totals[key] = metric_totals.get(
                            key,
                            torch.zeros_like(value),
                        ) + value.detach()
                grad_clip_norm = float(config.get("grad_clip_norm", 0.0))
                if grad_clip_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
                scheduler.step()
                torch.cuda.synchronize(train_device)
                global_step += 1
                progress.update(1)
                step_seconds = time.perf_counter() - step_started
                averaged = {
                    key: value / accumulation_steps
                    for key, value in metric_totals.items()
                }
                if should_run_at_step(log_every, global_step):
                    logger.log_metrics(
                        {
                            "global_step": global_step,
                            **{
                                f"train/{key}": float(value)
                                for key, value in averaged.items()
                            },
                            "train/learning_rate": optimizer.param_groups[0]["lr"],
                            "run/step_seconds": step_seconds,
                            "run/generator_seconds": generation_seconds,
                            "run/generator_wait_seconds": generator_wait_seconds,
                            "run/data_wait_seconds": data_wait_seconds,
                            "run/generator_wait_fraction": (
                                generator_wait_seconds / step_seconds
                            ),
                            "run/input_wait_fraction": (
                                (generator_wait_seconds + data_wait_seconds)
                                / step_seconds
                            ),
                        },
                        step=global_step,
                    )
                if should_run_at_step(checkpoint_every, global_step):
                    storage_mkdir(storage_join(workdir, "checkpoints"))
                    _save_checkpoint(
                        checkpoint_path,
                        model,
                        optimizer,
                        scheduler,
                        global_step,
                        float(averaged["loss"]),
                        total_steps,
                        datamodule.train_steps,
                        datamodule.info,
                        config,
                        wandb_run_id,
                    )
                if should_run_at_step(val_every, global_step):
                    prefetcher.wait()
                    val_metrics = _evaluate(
                        model,
                        generator,
                        datamodule.val_loader,
                        train_device,
                        val_num_steps,
                    )
                    logger.log_metrics(
                        {
                            "global_step": global_step,
                            **{
                                f"val/{key}": value
                                for key, value in val_metrics.items()
                            },
                        },
                        step=global_step,
                    )
        finally:
            prefetcher.close()
        epoch += 1
        resume_offset = 0

    progress.close()
    if not should_run_at_step(val_every, global_step):
        val_metrics = _evaluate(
            model,
            generator,
            datamodule.val_loader,
            train_device,
            val_num_steps,
        )
    results: dict[str, object] = {
        "run/final_global_step": float(global_step),
        "run/train_elapsed_seconds": time.perf_counter() - started,
        "run/generator_prefetch_depth": 1.0,
        "run/discriminator_device": str(train_device),
        "run/generator_device": str(generator.device),
        **{f"val/{key}": value for key, value in val_metrics.items()},
    }
    logger.log_metrics({"global_step": global_step, **results}, step=global_step)
    storage_mkdir(storage_join(workdir, "checkpoints"))
    _save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        scheduler,
        global_step,
        float(val_metrics["loss"]),
        total_steps,
        datamodule.train_steps,
        datamodule.info,
        config,
        wandb_run_id,
    )
    logger.finish()
    return results
