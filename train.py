from __future__ import annotations

import logging
import random
import warnings
from collections import deque
from collections.abc import Iterator
from contextlib import nullcontext
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
from utils.msg_probe import run_msg_probe, should_run_msg_probe
from utils.schedulers import CapturableCosineSchedule
from utils.training import (
    build_logger,
    build_model_from_config,
    collect_model_param_summary,
    log_model_param_summary,
    model_param_summary_to_metrics,
)

torch.set_float32_matmul_precision('medium')
torch._dynamo.config.capture_scalar_outputs = True
inductor_config.coordinate_descent_tuning = True
inductor_config.triton.unique_kernel_names = True
inductor_config.fx_graph_cache = True
inductor_config.epilogue_fusion = True
inductor_config.shape_padding = True


_WD_MODULE_TOKENS = (
    "attention.",
    "feed_forward.",
    "cross_attn.",
    "pool_mha.",
)


def _is_weight_decay_target(
    param_name: str,
    param: torch.nn.Parameter,
) -> bool:
    if param.ndim < 2:
        return False
    if not param_name.endswith("weight"):
        return False
    return any(token in param_name for token in _WD_MODULE_TOKENS)


def _partition_params_for_adamw(
    model: torch.nn.Module,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _is_weight_decay_target(name, param):
            decay_params.append(param)
        else:
            no_decay_params.append(param)
    return decay_params, no_decay_params


def _partition_params_for_muon(
    model: torch.nn.Module,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Split parameters into Muon-eligible and AdamW-only, with explicit no-decay."""
    muon_decay_params = []
    muon_no_decay_params = []
    adamw_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 2:
            if _is_weight_decay_target(name, param):
                muon_decay_params.append(param)
            else:
                muon_no_decay_params.append(param)
        else:
            adamw_params.append(param)
    return muon_decay_params, muon_no_decay_params, adamw_params


# ---------------------------------------------------------------------------
# CUDA stream prefetch helpers
# ---------------------------------------------------------------------------

_TRAIN_BATCH_KEYS = frozenset({
    "peak_mz",
    "peak_intensity",
    "peak_valid_mask",
    "context_mask",
    "target_masks",
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
    """Keeps N batches prefetched to GPU using a side CUDA stream.

    ``_preload_one()`` kicks off one async H2D copy on a dedicated stream.
    ``next()`` waits for the next batch-ready event, records stream usage
    for allocator safety, then immediately refills one slot.
    """

    def __init__(
        self,
        loader: Iterator,
        device: torch.device,
        prefetch_size: int = 1,
    ) -> None:
        self._loader = loader
        self._device = device
        self._prefetch_size = prefetch_size
        self._stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        self._ready: deque[tuple[dict[str, torch.Tensor], torch.cuda.Event | None]] = deque()
        self._exhausted = False
        for _ in range(self._prefetch_size):
            self._preload_one()

    def _preload_one(self) -> None:
        if self._exhausted:
            return
        batch = next(self._loader, None)
        if batch is None:
            self._exhausted = True
            return

        if self._stream is None:
            moved = _move_batch_to_device(batch, self._device)
            ready_event = None
        else:
            with torch.cuda.stream(self._stream):
                moved = _move_batch_to_device(batch, self._device)
                ready_event = torch.cuda.Event()
                ready_event.record(self._stream)
        self._ready.append((moved, ready_event))

    def next(self) -> dict[str, torch.Tensor] | None:
        if not self._ready:
            return None

        batch, ready_event = self._ready.popleft()
        if ready_event is not None:
            current_stream = torch.cuda.current_stream(device=self._device)
            current_stream.wait_event(ready_event)
            _record_stream(batch, current_stream)
        self._preload_one()
        return batch


# ---------------------------------------------------------------------------
# Compiled training step
# ---------------------------------------------------------------------------

def _train_step_impl(
    model: PeakSetSIGReg,
    batch: dict[str, torch.Tensor],
    optimizers: list[torch.optim.Optimizer],
    schedulers: list[CapturableCosineSchedule],
    autocast_dtype: torch.dtype | None,
    grad_clip_norm: float | None,
) -> dict[str, torch.Tensor]:
    model.advance_sigreg_lambda_schedule()
    device_type = next(model.parameters()).device.type
    if autocast_dtype is None or device_type != "cuda":
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(device_type=device_type, dtype=autocast_dtype)
    with autocast_ctx:
        metrics = model.forward_augmented(batch)
    metrics["loss"].backward()
    if grad_clip_norm is not None and grad_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
    for opt in optimizers:
        opt.step()
        opt.zero_grad(set_to_none=True)
    for sched in schedulers:
        sched.step()
    return metrics


def _resolve_autocast_dtype(config: config_dict.ConfigDict) -> torch.dtype | None:
    autocast_name = str(config.get("autocast_dtype", "bf16")).lower()
    if autocast_name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if autocast_name in {"fp16", "float16", "half"}:
        return torch.float16
    if autocast_name in {"fp32", "float32", "none"}:
        return None
    raise ValueError(f"Unsupported autocast_dtype: {autocast_name}")


def _resolve_norm_type(config: config_dict.ConfigDict) -> str:
    norm_type = str(config.get("norm_type", "rmsnorm")).lower()
    if norm_type not in {"rmsnorm", "layernorm"}:
        raise ValueError(f"Unsupported norm_type: {norm_type}")
    return norm_type


# ---------------------------------------------------------------------------
# Optimizer / scheduler construction
# ---------------------------------------------------------------------------

def _build_optimizers(
    config: config_dict.ConfigDict,
    model: PeakSetSIGReg,
    total_steps: int,
    device: torch.device,
) -> tuple[list[torch.optim.Optimizer], list[CapturableCosineSchedule]]:
    base_lr = float(config.learning_rate)
    warmup_steps = int(config.get("warmup_steps", 0))
    min_learning_rate = config.get("min_learning_rate", None)
    b2 = float(config.get("b2", 0.999))
    weight_decay = float(config.weight_decay)
    optimizer_type = str(config.get("optimizer", "adamw")).lower()
    is_cuda = device.type == "cuda"
    capturable = bool(config.get("optimizer_capturable", True)) and is_cuda
    fused_cfg = config.get("optimizer_fused", None)
    fused = (is_cuda if fused_cfg is None else bool(fused_cfg) and is_cuda)

    def _make_schedule(
        optimizer: torch.optim.Optimizer, lr: float,
    ) -> CapturableCosineSchedule:
        return CapturableCosineSchedule(
            optimizer,
            base_lr=lr,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr=min_learning_rate,
            device=device,
        )

    if optimizer_type == "muon":
        muon_decay_params, muon_no_decay_params, adamw_params = _partition_params_for_muon(model)
        muon_lr = float(config.get("muon_lr", None) or base_lr)
        adamw_lr = float(config.get("adamw_lr", None) or base_lr)
        muon_wd = float(config.get("muon_weight_decay", None) or weight_decay)
        muon_param_groups: list[dict[str, Any]] = []
        if muon_decay_params:
            muon_param_groups.append({"params": muon_decay_params, "weight_decay": muon_wd})
        if muon_no_decay_params:
            muon_param_groups.append({"params": muon_no_decay_params, "weight_decay": 0.0})

        muon_opt = torch.optim.Muon(
            muon_param_groups,
            lr=torch.tensor(muon_lr),
            momentum=float(config.get("muon_momentum", 0.95)),
            nesterov=bool(config.get("muon_nesterov", True)),
            ns_steps=int(config.get("muon_ns_steps", 5)),
            weight_decay=0.0,
            adjust_lr_fn=str(config.get("muon_adjust_lr_fn", "match_rms_adamw")),
        )
        adamw_opt = torch.optim.AdamW(
            adamw_params,
            lr=torch.tensor(adamw_lr),
            betas=(0.9, b2),
            weight_decay=0.0,
            capturable=capturable,
            fused=fused,
        )
        return [muon_opt, adamw_opt], [_make_schedule(muon_opt, muon_lr), _make_schedule(adamw_opt, adamw_lr)]

    decay_params, no_decay_params = _partition_params_for_adamw(model)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=torch.tensor(base_lr),
        betas=(0.9, b2),
        weight_decay=0.0,
        capturable=capturable,
        fused=fused,
    )
    return [optimizer], [_make_schedule(optimizer, base_lr)]


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def _save_checkpoint(
    path: Path,
    model: PeakSetSIGReg,
    optimizers: list[torch.optim.Optimizer],
    schedulers: list[CapturableCosineSchedule],
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

    logging.info("Training for %d epochs.", num_epochs)
    logging.info("Steps per epoch: %d", steps_per_epoch)
    logging.info("Total steps: %d", total_steps)
    config.norm_type = _resolve_norm_type(config)
    logging.info("Norm type: %s", config.norm_type)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(config)
    model_param_summary = collect_model_param_summary(model)
    log_model_param_summary(model_param_summary)
    model_param_metrics = model_param_summary_to_metrics(model_param_summary)
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

    logger.log_metrics(model_param_metrics, step=global_step)

    # Compile training step (forward + backward + optimizer + scheduler)
    autocast_dtype = _resolve_autocast_dtype(config)
    compiled_step = torch.compile(_train_step_impl, mode="max-autotune-no-cudagraphs", fullgraph=False)


    optimizer_type = str(config.get("optimizer", "adamw")).lower()
    device_prefetch_size = int(config.get("device_prefetch_size", 1))
    _msg_probe_raw = float(config.get("msg_probe_every_n_steps", 0))
    if 0 < _msg_probe_raw <= 1:
        msg_probe_every_n_steps = max(1, int(_msg_probe_raw * steps_per_epoch))
    else:
        msg_probe_every_n_steps = int(_msg_probe_raw)
    msg_probe_cache_dir = config.get("msg_probe_cache_dir", None)
    grad_clip_norm = config.get("grad_clip_norm", None)
    if grad_clip_norm is not None:
        grad_clip_norm = float(grad_clip_norm)

    train_loader = datamodule.train_loader
    last_msg_probe_metrics: dict[str, float] = {}

    for epoch in range(start_epoch, num_epochs):
        prefetcher = _BatchPrefetcher(
            iter(train_loader), device, prefetch_size=device_prefetch_size,
        )

        pbar = tqdm(total=steps_per_epoch, desc=f"Epoch {epoch}", unit="step")
        while True:
            batch = prefetcher.next()
            if batch is None:
                break

            metrics = compiled_step(
                model,
                batch,
                optimizers,
                schedulers,
                autocast_dtype,
                grad_clip_norm,
            )
            model.update_teacher()
            global_step += 1
            pbar.update(1)

            if global_step % log_every_n_steps == 0:
                loss_val = float(metrics["loss"].detach())
                pbar.set_postfix(loss=f"{loss_val:.4f}", step=global_step)
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
                _prune_checkpoints(checkpoint_dir, keep_top_k=15)

            if should_run_msg_probe(global_step, msg_probe_every_n_steps):
                probe_metrics = run_msg_probe(
                    config=config,
                    datamodule=datamodule,
                    model=model,
                    device=device,
                    cache_dir_override=msg_probe_cache_dir,
                )
                logger.log_metrics(probe_metrics, step=global_step)
                last_msg_probe_metrics = probe_metrics
                logging.info(
                    "step=%d msg_probe(test_r2_mol_weight=%.4f test_auc_fg_mean=%.4f fg_tasks=%d)",
                    global_step,
                    probe_metrics["msg_probe/test/r2_mol_weight"],
                    probe_metrics["msg_probe/test/auc_fg_mean"],
                    int(probe_metrics["msg_probe/num_fg_tasks"]),
                )

        pbar.close()

    # Save final checkpoint
    _save_checkpoint(
        checkpoint_dir / "last.pt",
        model, optimizers, schedulers,
        global_step, num_epochs, float("nan"),
    )
    return {**last_msg_probe_metrics, **model_param_metrics}
