from __future__ import annotations

import logging
import random
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tqdm import tqdm

warnings.filterwarnings("ignore", message="Profiler function.*will be ignored")
from torch.utils.data import DataLoader
import torch._inductor.config as inductor_config

from ml_collections import config_dict

from input_pipeline import TfLightningDataModule
from models.model import PeakSetSIGReg
from utils.probing import run_attentive_probe
from utils.schedulers import learning_rate_at_step
from utils.training import build_logger, build_model_from_config

torch.set_float32_matmul_precision('high')
inductor_config.coordinate_descent_tuning = True
inductor_config.triton.unique_kernel_names = True
inductor_config.fx_graph_cache = True
inductor_config.epilogue_fusion = True
inductor_config.shape_padding = True


def _partition_params_for_muon(
    model: torch.nn.Module,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Split parameters into 2D (Muon-eligible) and non-2D (AdamW-only)."""
    muon_params = []
    adamw_params = []
    for param in model.parameters():
        if param.ndim == 2:
            muon_params.append(param)
        else:
            adamw_params.append(param)
    return muon_params, adamw_params


# ---------------------------------------------------------------------------
# CUDA stream prefetch helpers
# ---------------------------------------------------------------------------

_TRAIN_BATCH_KEYS = frozenset({
    "fused_mz", "fused_intensity", "fused_precursor_mz",
    "fused_valid_mask", "fused_masked_positions",
    "view1_masked_fraction",
})


def _record_stream(obj: Any, stream: torch.cuda.Stream) -> None:
    if isinstance(obj, torch.Tensor):
        obj.record_stream(stream)
        return
    if isinstance(obj, dict):
        for value in obj.values():
            _record_stream(value, stream)
        return
    if isinstance(obj, (list, tuple)):
        for value in obj:
            _record_stream(value, stream)


def _move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        if k not in _TRAIN_BATCH_KEYS:
            continue
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = torch.as_tensor(v, device=device)
    return out


class _BatchPrefetcher:
    """Prefetches batches to GPU using a dedicated CUDA stream."""

    def __init__(self, loader: Iterator, device: torch.device) -> None:
        self._loader = loader
        self._device = device
        self._stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        self._batch: dict[str, torch.Tensor] | None = None
        self._exhausted = False
        self._prefetch()

    def _prefetch(self) -> None:
        batch = next(self._loader, None)
        if batch is None:
            self._batch = None
            self._exhausted = True
            return
        if self._stream is not None:
            with torch.cuda.stream(self._stream):
                self._batch = _move_batch_to_device(batch, self._device)
        else:
            self._batch = _move_batch_to_device(batch, self._device)

    def next(self) -> dict[str, torch.Tensor] | None:
        if self._exhausted:
            return None
        if self._stream is not None:
            current = torch.cuda.current_stream(device=self._device)
            current.wait_stream(self._stream)
            _record_stream(self._batch, current)
        batch = self._batch
        self._prefetch()
        return batch


# ---------------------------------------------------------------------------
# Compiled training step
# ---------------------------------------------------------------------------

def _train_step_impl(
    model: PeakSetSIGReg,
    batch: dict[str, torch.Tensor],
    bcs_projection: torch.Tensor,
    optimizers: list[torch.optim.Optimizer],
    schedulers: list[torch.optim.lr_scheduler.LRScheduler],
) -> dict[str, torch.Tensor]:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        metrics = model.forward_augmented(batch, bcs_projection=bcs_projection)
    metrics["loss"].backward()
    for opt in optimizers:
        opt.step()
        opt.zero_grad(set_to_none=True)
    for sched in schedulers:
        sched.step()
    return metrics


# ---------------------------------------------------------------------------
# Optimizer / scheduler construction
# ---------------------------------------------------------------------------

def _build_optimizers(
    config: config_dict.ConfigDict,
    model: PeakSetSIGReg,
    total_steps: int,
    device: torch.device,
) -> tuple[list[torch.optim.Optimizer], list[torch.optim.lr_scheduler.LRScheduler]]:
    base_lr = float(config.learning_rate)
    warmup_steps = int(config.get("warmup_steps", 0))
    schedule_type = str(config.get("learning_rate_schedule", "cosine"))
    min_learning_rate = config.get("min_learning_rate", None)
    b2 = float(config.get("b2", 0.999))
    weight_decay = float(config.weight_decay)
    optimizer_type = str(config.get("optimizer", "adamw")).lower()
    is_cuda = device.type == "cuda"
    capturable = bool(config.get("optimizer_capturable", True)) and is_cuda
    fused_cfg = config.get("optimizer_fused", None)
    fused = (is_cuda if fused_cfg is None else bool(fused_cfg) and is_cuda)

    def _lr_for_step(step: int) -> float:
        return learning_rate_at_step(
            step,
            base_lr=base_lr,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            schedule_type=schedule_type,
            min_learning_rate=min_learning_rate,
        )

    def _make_lr_lambda(lr: float):
        def lr_lambda(step_idx: int) -> float:
            return _lr_for_step(step_idx + 1) / lr
        return lr_lambda

    if optimizer_type == "muon":
        muon_params, adamw_params = _partition_params_for_muon(model)
        muon_lr = float(config.get("muon_lr", None) or base_lr)
        adamw_lr = float(config.get("adamw_lr", None) or base_lr)
        muon_wd = float(config.get("muon_weight_decay", None) or weight_decay)

        muon_opt = torch.optim.Muon(
            muon_params,
            lr=torch.tensor(muon_lr),
            momentum=float(config.get("muon_momentum", 0.95)),
            nesterov=bool(config.get("muon_nesterov", True)),
            ns_steps=int(config.get("muon_ns_steps", 5)),
            weight_decay=muon_wd,
            adjust_lr_fn=str(config.get("muon_adjust_lr_fn", "match_rms_adamw")),
        )
        adamw_opt = torch.optim.AdamW(
            adamw_params,
            lr=torch.tensor(adamw_lr),
            betas=(0.9, b2),
            weight_decay=weight_decay,
            capturable=capturable,
            fused=fused,
        )
        muon_sched = torch.optim.lr_scheduler.LambdaLR(
            muon_opt, lr_lambda=_make_lr_lambda(muon_lr),
        )
        adamw_sched = torch.optim.lr_scheduler.LambdaLR(
            adamw_opt, lr_lambda=_make_lr_lambda(adamw_lr),
        )
        return [muon_opt, adamw_opt], [muon_sched, adamw_sched]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=torch.tensor(base_lr),
        betas=(0.9, b2),
        weight_decay=weight_decay,
        capturable=capturable,
        fused=fused,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=_make_lr_lambda(base_lr),
    )
    return [optimizer], [scheduler]


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def _save_checkpoint(
    path: Path,
    model: PeakSetSIGReg,
    optimizers: list[torch.optim.Optimizer],
    schedulers: list[torch.optim.lr_scheduler.LRScheduler],
    global_step: int,
    epoch: int,
    loss: float,
) -> None:
    torch.save({
        "model": model.state_dict(),
        "optimizers": [opt.state_dict() for opt in optimizers],
        "schedulers": [sched.state_dict() for sched in schedulers],
        "global_step": global_step,
        "epoch": epoch,
        "loss": loss,
    }, path)


def _latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    pts = sorted(checkpoint_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    if not pts:
        return None
    return pts[-1]


def _prune_checkpoints(checkpoint_dir: Path, keep_top_k: int = 5) -> None:
    """Keep the last checkpoint and the top-k by lowest loss."""
    pts = sorted(checkpoint_dir.glob("step-*.pt"), key=lambda p: p.stat().st_mtime)
    if len(pts) <= keep_top_k:
        return
    losses: list[tuple[float, Path]] = []
    for p in pts:
        ckpt = torch.load(p, map_location="cpu", weights_only=True)
        losses.append((ckpt.get("loss", float("inf")), p))
    losses.sort(key=lambda x: x[0])
    keep = {p for _, p in losses[:keep_top_k]}
    keep.add(pts[-1])  # always keep latest
    for p in pts:
        if p not in keep:
            p.unlink()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _run_validation(
    model: PeakSetSIGReg,
    compiled_forward,
    val_loader: DataLoader | None,
    device: torch.device,
    seed: int,
    epoch: int,
) -> dict[str, float]:
    """Run validation on MassSpecGym val split, returns averaged metrics."""
    model.eval()
    if val_loader is None:
        model.train()
        return {}
    sums: dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            batch = _move_batch_to_device(batch, device)
            bcs_projection = model.sample_bcs_projection(
                device=device, seed=seed + 7_000_000 + epoch * 100_000 + batch_idx,
            )
            torch.compiler.cudagraph_mark_step_begin()
            metrics = compiled_forward(batch, bcs_projection)
            bs = int(batch["fused_mz"].shape[0]) // 2
            count += bs
            for key, value in metrics.items():
                val = float(value) * bs
                sums[key] = sums.get(key, 0.0) + val
    model.train()
    return {f"msg_eval/{k}": v / count for k, v in sums.items()} if count > 0 else {}


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_and_evaluate(
    config: config_dict.ConfigDict,
    workdir: str | Path,
) -> dict[str, float]:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    seed = int(config.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    datamodule = TfLightningDataModule(config, seed=seed)
    num_epochs = int(config.num_epochs)
    steps_per_epoch = datamodule.train_steps
    total_steps = num_epochs * steps_per_epoch
    log_every_n_steps = int(config.get("log_every_n_steps", 50))
    checkpoint_every_steps = int(config.checkpoint_every_steps)

    info = datamodule.info
    config.num_peaks = info["num_peaks"]
    config.fingerprint_bits = int(info["fingerprint_bits"])

    logging.info("Training for %d epochs.", num_epochs)
    logging.info("Steps per epoch: %d", steps_per_epoch)
    logging.info("Total steps: %d", total_steps)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(config)
    model.to(device)
    model.train()

    optimizers, schedulers = _build_optimizers(config, model, total_steps, device)

    checkpoint_dir = workdir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger = build_logger(config, workdir)

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    ckpt_path = _latest_checkpoint(checkpoint_dir)
    if ckpt_path is not None:
        logging.info("Resuming from checkpoint: %s", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        for opt, state in zip(optimizers, ckpt["optimizers"]):
            opt.load_state_dict(state)
        for sched, state in zip(schedulers, ckpt["schedulers"]):
            sched.load_state_dict(state)
        global_step = ckpt["global_step"]
        start_epoch = ckpt["epoch"]
        del ckpt

    # Compile training step (forward + backward + optimizer + scheduler)
    compiled_step = torch.compile(_train_step_impl, mode="max-autotune")
    compiled_forward = torch.compile(
        model.forward_augmented, mode="max-autotune", fullgraph=True,
    )

    optimizer_type = str(config.get("optimizer", "adamw")).lower()

    train_loader = datamodule.train_loader
    val_loader = datamodule.val_loader

    for epoch in range(start_epoch, num_epochs):
        prefetcher = _BatchPrefetcher(iter(train_loader), device)

        batch_idx = 0
        pbar = tqdm(total=steps_per_epoch, desc=f"Epoch {epoch}", unit="step")
        while True:
            batch = prefetcher.next()
            if batch is None:
                break

            bcs_projection = model.sample_bcs_projection(
                device=device, seed=seed + 6_000_000 + global_step,
            )
            torch.compiler.cudagraph_mark_step_begin()
            metrics = compiled_step(model, batch, bcs_projection, optimizers, schedulers)
            global_step += 1

            loss_val = float(metrics["loss"].detach())
            pbar.set_postfix(loss=f"{loss_val:.4f}", step=global_step)
            pbar.update(1)

            if global_step % log_every_n_steps == 0:
                log_metrics = {f"train/{k}": float(v.detach()) for k, v in metrics.items()}
                if optimizer_type == "muon":
                    for opt, label in zip(optimizers, ("muon", "adamw")):
                        log_metrics[f"train/lr_{label}"] = opt.param_groups[0]["lr"]
                else:
                    log_metrics["train/learning_rate"] = optimizers[0].param_groups[0]["lr"]
                log_metrics["epoch"] = epoch
                log_metrics["global_step"] = global_step
                logger.log_metrics(log_metrics, step=global_step)

            if global_step % checkpoint_every_steps == 0:
                ckpt_name = f"step-{global_step:08d}.pt"
                _save_checkpoint(
                    checkpoint_dir / ckpt_name,
                    model, optimizers, schedulers,
                    global_step, epoch, float(metrics["loss"]),
                )
                _prune_checkpoints(checkpoint_dir, keep_top_k=5)

            batch_idx += 1
        pbar.close()

        # End-of-epoch validation (MassSpecGym val only)
        val_metrics = _run_validation(
            model, compiled_forward, val_loader, device, seed, epoch,
        )
        if val_metrics:
            logger.log_metrics(val_metrics, step=global_step)
            val_loss = val_metrics.get("msg_eval/loss", float("nan"))
            logging.info("epoch=%d val_loss=%.4f", epoch, val_loss)

    # Save final checkpoint
    _save_checkpoint(
        checkpoint_dir / "last.pt",
        model, optimizers, schedulers,
        global_step, num_epochs, float("nan"),
    )

    # Run final attentive probe
    final_probe_metrics = run_attentive_probe(
        config=config,
        datamodule=datamodule,
        model=model,
        device=device,
        loggers=(logger,),
        global_step=global_step,
    )
    for key, value in final_probe_metrics.items():
        logging.info("%s = %.6f", key, value)
    return final_probe_metrics
