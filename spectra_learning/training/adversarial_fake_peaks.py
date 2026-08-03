from __future__ import annotations

import os
import time
from dataclasses import asdict
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
from spectra_learning.data.gems.settings import GemsDataConfig
from spectra_learning.data.spectra import PEAK_MZ_MAX
from spectra_learning.models.fake_peaks import (
    DynamicPeakGenerator,
    FakePeakDiscriminator,
    logit_uniform_nll,
)
from spectra_learning.models.settings import PeakSetJEPASettings
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
from spectra_learning.training.configuration import finalize_config, save_config
from spectra_learning.training.logging import build_logger
from spectra_learning.training.optimization import build_optimizers
from spectra_learning.training.storage import (
    StoragePath,
    local_scratch_dir,
    normalize_storage_path,
    storage_exists,
    storage_join,
    storage_mkdir,
)

ADVERSARIAL_FAKE_PEAK_CHECKPOINT_FORMAT_VERSION = 3


def generator_config(
    config: config_dict.ConfigDict,
) -> config_dict.ConfigDict:
    result = config.copy_and_resolve_references()
    result.update(config.generator_model)
    result.learning_rate = config.generator_learning_rate
    result.min_learning_rate = config.generator_min_learning_rate
    result.warmup_steps = config.generator_warmup_steps
    return result


def adversarial_weight_at_step(
    config: config_dict.ConfigDict,
    global_step: int,
) -> float:
    warmup_steps = int(config.generator_adversarial_warmup_steps)
    progress = 1.0 if warmup_steps == 0 else min(1.0, global_step / warmup_steps)
    return float(config.generator_adversarial_loss_weight) * progress


def _clamped_class_targets(
    values: Tensor,
    *,
    value_scale: float,
    bin_size: float,
    num_classes: int,
) -> tuple[Tensor, Tensor]:
    scaled = values.float() * value_scale / bin_size
    classes = torch.floor(scaled).long().clamp(0, num_classes - 1)
    residuals = (scaled - classes.float()).clamp(0.0, 1.0)
    return classes, residuals


def _slice_batch(
    batch: dict[str, Tensor],
    start: int,
    end: int,
) -> dict[str, Tensor]:
    return {key: value[start:end] for key, value in batch.items()}


def _to_device(
    batch: dict[str, Tensor],
    device: torch.device,
) -> dict[str, Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }


def _generator_metrics(
    generated: dict[str, Tensor],
    source: dict[str, Tensor],
    config: config_dict.ConfigDict,
) -> tuple[Tensor, dict[str, Tensor], Tensor]:
    target = generated["target_mask"]
    weights = target.float()
    count = weights.sum()
    predicted_mz = generated["predicted_mz"].float()
    predicted_intensity = generated["predicted_intensity"].float()
    true_mz = source["peak_mz"].float()
    true_intensity = source["peak_intensity"].float()

    mz_bin_size = float(config.jepa_mae_mz_bin_size)
    intensity_bin_size = float(config.jepa_mae_intensity_bin_size)
    true_mz_bins, true_mz_residual = _clamped_class_targets(
        true_mz,
        value_scale=PEAK_MZ_MAX,
        bin_size=mz_bin_size,
        num_classes=generated["mz_logits"].shape[-1],
    )
    true_intensity_bins, true_intensity_residual = _clamped_class_targets(
        true_intensity,
        value_scale=1.0,
        bin_size=intensity_bin_size,
        num_classes=generated["intensity_logits"].shape[-1],
    )
    mz_bin_per_peak = F.cross_entropy(
        generated["mz_logits"].transpose(1, 2),
        true_mz_bins,
        reduction="none",
    )
    intensity_bin_per_peak = F.cross_entropy(
        generated["intensity_logits"].transpose(1, 2),
        true_intensity_bins,
        reduction="none",
    )
    mz_residual_per_peak = logit_uniform_nll(
        generated["mz_residual_shift"],
        true_mz_residual,
    )
    intensity_residual_per_peak = logit_uniform_nll(
        generated["intensity_residual_shift"],
        true_intensity_residual,
    )
    mz_bin_loss = (mz_bin_per_peak * weights).sum() / count
    intensity_bin_loss = (intensity_bin_per_peak * weights).sum() / count
    mz_residual_loss = (mz_residual_per_peak * weights).sum() / count
    intensity_residual_loss = (
        intensity_residual_per_peak * weights
    ).sum() / count
    mz_loss = mz_bin_loss + mz_residual_loss
    intensity_loss = intensity_bin_loss + intensity_residual_loss
    reconstruction_loss = (
        float(config.generator_mz_loss_weight) * mz_loss
        + float(config.generator_intensity_loss_weight) * intensity_loss
    )

    predicted_mz_bins = torch.floor(
        predicted_mz * PEAK_MZ_MAX / mz_bin_size
    ).long().clamp(0, generated["mz_logits"].shape[-1] - 1)
    exact_mz = target & predicted_mz_bins.eq(true_mz_bins)
    predicted_intensity_bins = torch.floor(
        predicted_intensity / intensity_bin_size
    ).long().clamp(0, generated["intensity_logits"].shape[-1] - 1)

    def moments(value: Tensor) -> tuple[Tensor, Tensor]:
        mean = (value * weights).sum() / count
        variance = ((value - mean).square() * weights).sum() / count
        return mean, variance.sqrt()

    generated_mz_mean, generated_mz_std = moments(predicted_mz)
    target_mz_mean, target_mz_std = moments(true_mz)
    generated_intensity_mean, generated_intensity_std = moments(
        predicted_intensity
    )
    target_intensity_mean, target_intensity_std = moments(true_intensity)
    generated_mz_residual_mean, generated_mz_residual_std = moments(
        generated["mz_residual"]
    )
    target_mz_residual_mean, target_mz_residual_std = moments(
        true_mz_residual
    )
    (
        generated_intensity_residual_mean,
        generated_intensity_residual_std,
    ) = moments(generated["intensity_residual"])
    (
        target_intensity_residual_mean,
        target_intensity_residual_std,
    ) = moments(true_intensity_residual)
    metrics = {
        "reconstruction_loss": reconstruction_loss,
        "mz_loss": mz_loss,
        "intensity_loss": intensity_loss,
        "mz_bin_loss": mz_bin_loss,
        "intensity_bin_loss": intensity_bin_loss,
        "mz_residual_loss": mz_residual_loss,
        "intensity_residual_loss": intensity_residual_loss,
        "mz_residual_mae": (
            (generated["mz_residual"] - true_mz_residual).abs() * weights
        ).sum()
        / count,
        "intensity_residual_mae": (
            (generated["intensity_residual"] - true_intensity_residual).abs()
            * weights
        ).sum()
        / count,
        "generated_mz_residual_mean": generated_mz_residual_mean,
        "generated_mz_residual_std": generated_mz_residual_std,
        "target_mz_residual_mean": target_mz_residual_mean,
        "target_mz_residual_std": target_mz_residual_std,
        "generated_intensity_residual_mean": (
            generated_intensity_residual_mean
        ),
        "generated_intensity_residual_std": (
            generated_intensity_residual_std
        ),
        "target_intensity_residual_mean": target_intensity_residual_mean,
        "target_intensity_residual_std": target_intensity_residual_std,
        "mz_mae_da": (
            (predicted_mz - true_mz).abs() * PEAK_MZ_MAX * weights
        ).sum()
        / count,
        "intensity_mae": (
            (predicted_intensity - true_intensity).abs() * weights
        ).sum()
        / count,
        "mz_exact_bin_accuracy": exact_mz.float().sum() / count,
        "intensity_exact_bin_accuracy": (
            predicted_intensity_bins.eq(true_intensity_bins) & target
        ).float().sum()
        / count,
        "generated_mz_mean": generated_mz_mean,
        "generated_mz_std": generated_mz_std,
        "target_mz_mean": target_mz_mean,
        "target_mz_std": target_mz_std,
        "generated_intensity_mean": generated_intensity_mean,
        "generated_intensity_std": generated_intensity_std,
        "target_intensity_mean": target_intensity_mean,
        "target_intensity_std": target_intensity_std,
    }
    return reconstruction_loss, metrics, exact_mz


def build_adversarial_fake_batch(
    source: dict[str, Tensor],
    generated: dict[str, Tensor],
    exact_mz: Tensor,
) -> dict[str, Tensor]:
    target = generated["target_mask"]
    result = dict(source)
    result.update(
        {
            "peak_mz": generated["peak_mz"],
            "peak_intensity": generated["peak_intensity"],
            "true_peak_mz": source["peak_mz"],
            "true_peak_intensity": source["peak_intensity"],
            "fake_peak_mask": target,
            "generator_exact_bin_mask": exact_mz,
            "detection_mask": target,
            "student_visible_mask": source["context_mask"] | target,
        }
    )
    return result


def build_adversarial_real_batch(
    source: dict[str, Tensor],
) -> dict[str, Tensor]:
    target = source["target_masks"][:, 0] & source["peak_valid_mask"]
    result = dict(source)
    result.update(
        {
            "true_peak_mz": source["peak_mz"],
            "true_peak_intensity": source["peak_intensity"],
            "fake_peak_mask": torch.zeros_like(target),
            "generator_exact_bin_mask": torch.zeros_like(target),
            "detection_mask": target,
            "student_visible_mask": source["context_mask"] | target,
        }
    )
    return result


def _concatenate_batches(
    fake: dict[str, Tensor],
    real: dict[str, Tensor],
) -> dict[str, Tensor]:
    return {
        key: torch.cat((fake[key], real[key]), dim=0)
        for key in fake
    }


def _prepare_adversarial_batch(
    generator: DynamicPeakGenerator,
    cpu_batch: dict[str, Tensor],
    generator_device: torch.device,
    discriminator_device: torch.device,
    config: config_dict.ConfigDict,
) -> tuple[
    dict[str, Tensor],
    dict[str, Tensor],
    Tensor,
    dict[str, Tensor],
]:
    half_batch = cpu_batch["peak_mz"].shape[0] // 2
    generator_source = _to_device(
        cpu_batch,
        generator_device,
    )
    real_source = _to_device(
        _slice_batch(cpu_batch, half_batch, 2 * half_batch),
        discriminator_device,
    )
    generated = generator(generator_source)
    reconstruction_loss, generator_metrics, exact_mz = _generator_metrics(
        generated,
        generator_source,
        config,
    )
    generated_batch = build_adversarial_fake_batch(
        generator_source,
        generated,
        exact_mz,
    )
    fake_on_generator = _slice_batch(generated_batch, 0, half_batch)
    fake_on_discriminator = _to_device(
        fake_on_generator,
        discriminator_device,
    )
    detached_fake = {
        key: value.detach()
        for key, value in fake_on_discriminator.items()
    }
    discriminator_batch = _concatenate_batches(
        detached_fake,
        build_adversarial_real_batch(real_source),
    )
    return (
        fake_on_discriminator,
        discriminator_batch,
        reconstruction_loss,
        generator_metrics,
    )


def _generator_adversarial_loss(
    discriminator: FakePeakDiscriminator,
    fake_batch: dict[str, Tensor],
) -> Tensor:
    predictions = discriminator.predict(fake_batch)
    target = fake_batch["detection_mask"]
    per_peak = F.softplus(predictions["fake_logits"])
    return (per_peak * target.float()).sum() / target.float().sum()


def train_adversarial_microbatch(
    generator: DynamicPeakGenerator,
    discriminator: FakePeakDiscriminator,
    cpu_batch: dict[str, Tensor],
    generator_device: torch.device,
    discriminator_device: torch.device,
    config: config_dict.ConfigDict,
    global_step: int,
    accumulation_steps: int,
) -> dict[str, Tensor]:
    (
        fake_batch,
        discriminator_batch,
        reconstruction_loss,
        generator_metrics,
    ) = _prepare_adversarial_batch(
        generator,
        cpu_batch,
        generator_device,
        discriminator_device,
        config,
    )

    discriminator_metrics = discriminator(discriminator_batch)
    (discriminator_metrics["loss"] / accumulation_steps).backward()

    discriminator.requires_grad_(False)
    adversarial_loss = _generator_adversarial_loss(discriminator, fake_batch)
    adversarial_weight = adversarial_weight_at_step(config, global_step)
    generator_loss = (
        reconstruction_loss
        + adversarial_weight * adversarial_loss.to(generator_device)
    )
    (generator_loss / accumulation_steps).backward()
    discriminator.requires_grad_(True)

    return {
        **{
            f"discriminator/{key}": value.detach()
            for key, value in discriminator_metrics.items()
        },
        **{
            f"generator/{key}": value.detach()
            for key, value in generator_metrics.items()
        },
        "generator/adversarial_loss": adversarial_loss.detach(),
        "generator/adversarial_weight": torch.tensor(
            adversarial_weight,
            device=generator_device,
        ),
        "generator/weighted_adversarial_loss": (
            adversarial_weight * adversarial_loss.detach()
        ),
        "generator/loss": generator_loss.detach(),
    }


@torch.no_grad()
def _evaluate(
    generator: DynamicPeakGenerator,
    discriminator: FakePeakDiscriminator,
    batches: Iterable[dict[str, Tensor]],
    generator_device: torch.device,
    discriminator_device: torch.device,
    config: config_dict.ConfigDict,
    global_step: int,
    max_steps: int,
) -> dict[str, float]:
    generator.eval()
    discriminator.eval()
    totals: dict[str, float] = {}
    steps = 0
    rng_devices = [
        device
        for device in (generator_device, discriminator_device)
        if device.type == "cuda"
    ]
    with torch.random.fork_rng(devices=rng_devices):
        for cpu_batch in islice(batches, max_steps):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                (
                    fake_batch,
                    discriminator_batch,
                    reconstruction_loss,
                    generator_metrics,
                ) = _prepare_adversarial_batch(
                    generator,
                    cpu_batch,
                    generator_device,
                    discriminator_device,
                    config,
                )
                discriminator_metrics = discriminator(discriminator_batch)
                adversarial_loss = _generator_adversarial_loss(
                    discriminator,
                    fake_batch,
                )
                adversarial_weight = adversarial_weight_at_step(
                    config,
                    global_step,
                )
                generator_loss = (
                    reconstruction_loss
                    + adversarial_weight * adversarial_loss.to(generator_device)
                )
            metrics = {
                **{
                    f"discriminator/{key}": value
                    for key, value in discriminator_metrics.items()
                },
                **{
                    f"generator/{key}": value
                    for key, value in generator_metrics.items()
                },
                "generator/adversarial_loss": adversarial_loss,
                "generator/adversarial_weight": torch.tensor(
                    adversarial_weight
                ),
                "generator/weighted_adversarial_loss": (
                    adversarial_weight * adversarial_loss
                ),
                "generator/loss": generator_loss,
            }
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            steps += 1
    generator.train()
    discriminator.train()
    return {key: value / steps for key, value in totals.items()}


def _training_contract(
    config: config_dict.ConfigDict,
    resolved_generator_config: config_dict.ConfigDict,
) -> dict[str, Any]:
    data_config = GemsDataConfig.from_config(config)
    return {
        "discriminator_model": asdict(PeakSetJEPASettings.from_config(config)),
        "generator_model": asdict(
            PeakSetJEPASettings.from_config(resolved_generator_config)
        ),
        "objective": {
            key: config[key]
            for key in (
                "fake_peak_detection_loss_weight",
                "fake_peak_reconstruction_loss_weight",
                "fake_peak_intensity_reconstruction_loss_weight",
                "generator_gumbel_temperature",
                "generator_mz_loss_weight",
                "generator_intensity_loss_weight",
                "generator_adversarial_loss_weight",
                "generator_adversarial_warmup_steps",
            )
        },
        "optimization": {
            key: config[key]
            for key in (
                "batch_size",
                "gradient_accumulation_steps",
                "learning_rate",
                "min_learning_rate",
                "warmup_steps",
                "generator_learning_rate",
                "generator_min_learning_rate",
                "generator_warmup_steps",
                "b2",
                "grad_clip_norm",
                "optimizer_fused",
            )
        },
        "data_stream": {
            "drop_remainder": data_config.drop_remainder,
            "gems_hdf5_rows_per_block": data_config.gems_hdf5_rows_per_block,
            "dataloader_num_workers": data_config.dataloader_num_workers,
        },
        "masking": {
            key: config[key]
            for key in (
                "jepa_num_target_blocks",
                "jepa_mask_strategy",
                "jepa_context_fraction",
                "jepa_target_fraction",
                "jepa_block_min_len",
                "jepa_mask_lengths",
                "jepa_mask_round_from",
                "jepa_allow_target_overlap",
            )
        }
        | {
            "jepa_intensity_aware_mask_config": (
                data_config.jepa_intensity_aware_mask_config
            )
        },
        "seed": config.seed,
    }


def _save_checkpoint(
    path: StoragePath,
    generator: DynamicPeakGenerator,
    discriminator: FakePeakDiscriminator,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    generator_scheduler: Any,
    discriminator_scheduler: Any,
    generator_device: torch.device,
    discriminator_device: torch.device,
    global_step: int,
    loss: float,
    total_steps: int,
    steps_per_epoch: int,
    data_info: dict[str, Any],
    config: config_dict.ConfigDict,
    training_contract: dict[str, Any],
    wandb_run_id: str | None,
) -> None:
    save_torch_checkpoint(
        {
            "format_version": ADVERSARIAL_FAKE_PEAK_CHECKPOINT_FORMAT_VERSION,
            "training_task": "adversarial_fake_peak",
            "training_contract": training_contract,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "generator_optimizer": generator_optimizer.state_dict(),
            "discriminator_optimizer": discriminator_optimizer.state_dict(),
            "generator_scheduler": generator_scheduler.state_dict(),
            "discriminator_scheduler": discriminator_scheduler.state_dict(),
            "global_step": global_step,
            "epoch": global_step // steps_per_epoch,
            "loss": loss,
            "total_steps": total_steps,
            "steps_per_epoch": steps_per_epoch,
            "wandb_run_id": wandb_run_id,
            "torch_rng_state": torch.get_rng_state(),
            "generator_rng_state": torch.cuda.get_rng_state(generator_device),
            "discriminator_rng_state": torch.cuda.get_rng_state(
                discriminator_device
            ),
            "peak_preprocessing": peak_preprocessing_contract(config),
            "data_provenance": data_provenance_contract(data_info),
        },
        path,
    )


def train_adversarial_fake_peaks(
    config: config_dict.ConfigDict,
    workdir: str | Path,
) -> dict[str, object]:
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError(
            "adversarial fake-peak training uses one process controlling both GPUs."
        )
    finalize_config(config)
    workdir = normalize_storage_path(workdir)
    local_workdir = local_scratch_dir(workdir)
    local_workdir.mkdir(parents=True, exist_ok=True)
    storage_mkdir(workdir)
    checkpoint_path = storage_join(workdir, "checkpoints", "last.pt")

    torch.manual_seed(int(config.seed))
    generator_device = torch.device(str(config.generator_device))
    discriminator_device = torch.device(str(config.discriminator_device))
    resolved_generator_config = generator_config(config)
    generator = DynamicPeakGenerator(resolved_generator_config).to(
        generator_device
    ).train()
    discriminator = FakePeakDiscriminator(config).to(
        discriminator_device
    ).train()
    datamodule = GemsDataModule(config, seed=int(config.seed))
    total_steps = total_training_steps(config, datamodule)
    accumulation_steps = int(config.gradient_accumulation_steps)
    generator_optimizers, generator_schedulers = build_optimizers(
        resolved_generator_config,
        generator,
        total_steps,
        generator_device,
    )
    discriminator_optimizers, discriminator_schedulers = build_optimizers(
        config,
        discriminator,
        total_steps,
        discriminator_device,
    )
    generator_optimizer = generator_optimizers[0]
    generator_scheduler = generator_schedulers[0]
    discriminator_optimizer = discriminator_optimizers[0]
    discriminator_scheduler = discriminator_schedulers[0]
    training_contract = _training_contract(config, resolved_generator_config)

    global_step = 0
    if storage_exists(checkpoint_path):
        checkpoint = load_torch_checkpoint(checkpoint_path, map_location="cpu")
        if (
            checkpoint["format_version"]
            != ADVERSARIAL_FAKE_PEAK_CHECKPOINT_FORMAT_VERSION
            or checkpoint["training_task"] != "adversarial_fake_peak"
        ):
            raise ValueError(
                "Unsupported adversarial fake-peak checkpoint; use a new workdir."
            )
        if checkpoint["training_contract"] != training_contract:
            raise ValueError(
                "Adversarial training contract mismatch; use a new workdir."
            )
        if checkpoint["total_steps"] != total_steps:
            raise ValueError("Checkpoint training length does not match the config.")
        if checkpoint["steps_per_epoch"] != datamodule.train_steps:
            raise ValueError("Checkpoint epoch length does not match the dataset.")
        validate_peak_preprocessing_contract(checkpoint, config)
        validate_data_provenance_contract(checkpoint, datamodule.info)
        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        generator_optimizer.load_state_dict(checkpoint["generator_optimizer"])
        discriminator_optimizer.load_state_dict(
            checkpoint["discriminator_optimizer"]
        )
        generator_scheduler.load_state_dict(checkpoint["generator_scheduler"])
        discriminator_scheduler.load_state_dict(
            checkpoint["discriminator_scheduler"]
        )
        global_step = int(checkpoint["global_step"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        torch.cuda.set_rng_state(
            checkpoint["generator_rng_state"],
            generator_device,
        )
        torch.cuda.set_rng_state(
            checkpoint["discriminator_rng_state"],
            discriminator_device,
        )
        if checkpoint["wandb_run_id"]:
            config.wandb_resume_id = checkpoint["wandb_run_id"]

    save_config(config, workdir)
    logger = build_logger(config, local_workdir)
    wandb_run_id = getattr(logger.experiment, "id", None)
    logger.log_metrics(
        {
            "global_step": global_step,
            "model/generator_params_total": sum(
                parameter.numel() for parameter in generator.parameters()
            ),
            "model/discriminator_params_total": sum(
                parameter.numel() for parameter in discriminator.parameters()
            ),
        },
        step=global_step,
    )

    val_num_steps = validation_steps(config)
    val_metrics = _evaluate(
        generator,
        discriminator,
        datamodule.val_loader,
        generator_device,
        discriminator_device,
        config,
        global_step,
        val_num_steps,
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
    progress = tqdm(
        total=total_steps,
        initial=global_step,
        disable=bool(config.get("disable_progress_bar", False)),
    )
    started = time.perf_counter()
    epoch = global_step // datamodule.train_steps
    resume_offset = global_step % datamodule.train_steps
    while global_step < total_steps:
        train_batches = iter(
            datamodule.train_loader_for_epoch(
                epoch,
                start_batch=resume_offset,
            )
        )
        for _ in range(resume_offset, datamodule.train_steps):
            if global_step >= total_steps:
                break
            generator_optimizer.zero_grad(set_to_none=True)
            discriminator_optimizer.zero_grad(set_to_none=True)
            metric_totals: dict[str, Tensor] = {}
            data_wait_seconds = 0.0
            step_started = time.perf_counter()
            for _ in range(accumulation_steps):
                data_wait_started = time.perf_counter()
                cpu_batch = next(train_batches)
                data_wait_seconds += time.perf_counter() - data_wait_started
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    metrics = train_adversarial_microbatch(
                        generator,
                        discriminator,
                        cpu_batch,
                        generator_device,
                        discriminator_device,
                        config,
                        global_step,
                        accumulation_steps,
                    )
                for key, value in metrics.items():
                    metric_totals[key] = metric_totals.get(
                        key,
                        torch.zeros_like(value),
                    ) + value

            grad_clip_norm = float(config.get("grad_clip_norm", 0.0))
            if grad_clip_norm > 0:
                discriminator_grad_norm = nn.utils.clip_grad_norm_(
                    discriminator.parameters(),
                    grad_clip_norm,
                )
                generator_grad_norm = nn.utils.clip_grad_norm_(
                    generator.parameters(),
                    grad_clip_norm,
                )
            else:
                discriminator_grad_norm = torch.zeros(
                    (),
                    device=discriminator_device,
                )
                generator_grad_norm = torch.zeros((), device=generator_device)
            discriminator_optimizer.step()
            generator_optimizer.step()
            discriminator_scheduler.step()
            generator_scheduler.step()
            torch.cuda.synchronize(discriminator_device)
            torch.cuda.synchronize(generator_device)

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
                        "train/discriminator/learning_rate": (
                            discriminator_optimizer.param_groups[0]["lr"]
                        ),
                        "train/discriminator/grad_norm": float(
                            discriminator_grad_norm
                        ),
                        "train/generator/learning_rate": (
                            generator_optimizer.param_groups[0]["lr"]
                        ),
                        "train/generator/grad_norm": float(generator_grad_norm),
                        "run/step_seconds": step_seconds,
                        "run/spectra_per_second": (
                            int(config.batch_size) / step_seconds
                        ),
                        "run/data_wait_seconds": data_wait_seconds,
                        "run/data_wait_fraction": (
                            data_wait_seconds / step_seconds
                        ),
                    },
                    step=global_step,
                )
            if should_run_at_step(checkpoint_every, global_step):
                storage_mkdir(storage_join(workdir, "checkpoints"))
                _save_checkpoint(
                    checkpoint_path,
                    generator,
                    discriminator,
                    generator_optimizer,
                    discriminator_optimizer,
                    generator_scheduler,
                    discriminator_scheduler,
                    generator_device,
                    discriminator_device,
                    global_step,
                    (
                        float(averaged["generator/loss"])
                        + float(averaged["discriminator/loss"])
                    ),
                    total_steps,
                    datamodule.train_steps,
                    datamodule.info,
                    config,
                    training_contract,
                    wandb_run_id,
                )
            if should_run_at_step(val_every, global_step):
                val_metrics = _evaluate(
                    generator,
                    discriminator,
                    datamodule.val_loader,
                    generator_device,
                    discriminator_device,
                    config,
                    global_step,
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
        epoch += 1
        resume_offset = 0

    progress.close()
    if not should_run_at_step(val_every, global_step):
        val_metrics = _evaluate(
            generator,
            discriminator,
            datamodule.val_loader,
            generator_device,
            discriminator_device,
            config,
            global_step,
            val_num_steps,
        )
    results: dict[str, object] = {
        "run/final_global_step": float(global_step),
        "run/train_elapsed_seconds": time.perf_counter() - started,
        "run/discriminator_device": str(discriminator_device),
        "run/generator_device": str(generator_device),
        **{f"val/{key}": value for key, value in val_metrics.items()},
    }
    logger.log_metrics({"global_step": global_step, **results}, step=global_step)
    storage_mkdir(storage_join(workdir, "checkpoints"))
    _save_checkpoint(
        checkpoint_path,
        generator,
        discriminator,
        generator_optimizer,
        discriminator_optimizer,
        generator_scheduler,
        discriminator_scheduler,
        generator_device,
        discriminator_device,
        global_step,
        (
            val_metrics["generator/loss"]
            + val_metrics["discriminator/loss"]
        ),
        total_steps,
        datamodule.train_steps,
        datamodule.info,
        config,
        training_contract,
        wandb_run_id,
    )
    logger.finish()
    return results
