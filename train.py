import logging
import math
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
import torch._inductor.config as inductor_config

from ml_collections import config_dict

from input_pipeline import TfLightningDataModule
from models.model import PeakSetSIGReg
from utils.msg_probe import run_msg_probe
from utils.schedulers import CapturableCosineSchedule
from utils.training import (
    build_logger,
    build_model_from_config,
    collect_and_log_param_metrics,
    collect_runtime_norm_metrics,
)

torch.set_float32_matmul_precision("high")
torch._dynamo.config.capture_scalar_outputs = True
inductor_config.coordinate_descent_tuning = True
inductor_config.triton.unique_kernel_names = True
inductor_config.fx_graph_cache = True
inductor_config.epilogue_fusion = True
inductor_config.shape_padding = True
inductor_config.aggressive_fusion = True


def _is_weight_decay_target(name: str, param: torch.nn.Parameter) -> bool:
    return (
        param.ndim >= 2
        and name.endswith("weight")
        and any(
            t in name
            for t in ("attention.", "feed_forward.", "cross_attn.")
        )
    )


_TRAIN_BATCH_KEYS = frozenset(
    {
        "peak_mz",
        "peak_intensity",
        "peak_valid_mask",
        "context_mask",
        "target_masks",
        "precursor_mz",
    }
)


def _move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        k: v.to(device, non_blocking=True)
        if isinstance(v, torch.Tensor)
        else torch.as_tensor(v, device=device)
        for k, v in batch.items()
        if k in _TRAIN_BATCH_KEYS
    }


class _BatchPrefetcher:
    def __init__(
        self,
        loader: Iterator,
        device: torch.device,
        prefetch_size: int = 1,
    ) -> None:
        self._loader = loader
        self._device = device
        self._stream = (
            torch.cuda.Stream(device=device) if device.type == "cuda" else None
        )
        self._ready: deque = deque()
        self._exhausted = False
        for _ in range(prefetch_size):
            self._preload_one()

    def _preload_one(self) -> None:
        if self._exhausted:
            return
        if (batch := next(self._loader, None)) is None:
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
            for v in batch.values():
                v.record_stream(current_stream)
        self._preload_one()
        return batch


def _get_compiled_forward(model: PeakSetSIGReg, compile_mode: str):
    compiled = getattr(model, "_compiled_forward_augmented", None)
    if compiled is None:
        compiled = torch.compile(
            model.forward_augmented,
            mode=compile_mode,
            dynamic=False,
        )
        model._compiled_forward_augmented = compiled
    return compiled


def _get_trainable_params(model: PeakSetSIGReg) -> list[torch.nn.Parameter]:
    params = getattr(model, "_trainable_params", None)
    if params is None:
        params = [p for p in model.parameters() if p.requires_grad]
        model._trainable_params = params
    return params


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
    fused = is_cuda if fused_cfg is None else bool(fused_cfg) and is_cuda

    def _make_schedule(
        optimizer: torch.optim.Optimizer,
        lr: float,
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
        from optimizers import MuonAdamW

        muon_decay_params, muon_no_decay_params, adamw_params = [], [], []
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
        muon_lr = float(config.get("muon_lr", None) or base_lr)
        adamw_lr = float(config.get("adamw_lr", None) or base_lr)
        muon_wd = float(config.get("muon_weight_decay", None) or weight_decay)
        muon_momentum = float(config.get("muon_momentum", 0.95))
        muon_nesterov = bool(config.get("muon_nesterov", True))
        muon_adjust_lr_fn = str(config.get("muon_adjust_lr_fn", "match_rms_adamw"))
        param_groups = []
        if muon_decay_params:
            param_groups.append({
                "params": muon_decay_params,
                "name": "attn_2d",
                "optimizer": "muon",
                "lr": muon_lr,
                "momentum": muon_momentum,
                "weight_decay": muon_wd,
                "nesterov": muon_nesterov,
                "adjust_lr_fn": muon_adjust_lr_fn,
            })
        if muon_no_decay_params:
            param_groups.append({
                "params": muon_no_decay_params,
                "name": "ffn_2d",
                "optimizer": "muon",
                "lr": muon_lr,
                "momentum": muon_momentum,
                "weight_decay": 0.0,
                "nesterov": muon_nesterov,
                "adjust_lr_fn": muon_adjust_lr_fn,
            })
        if adamw_params:
            param_groups.append({
                "params": adamw_params,
                "name": "non_2d",
                "optimizer": "adamw",
                "lr": adamw_lr,
                "weight_decay": 0.0,
                "betas": (0.9, b2),
            })
        optimizer = MuonAdamW(param_groups)
        return [optimizer], [_make_schedule(optimizer, base_lr)]
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _is_weight_decay_target(name, param):
            decay_params.append(param)
        else:
            no_decay_params.append(param)
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


def _save_checkpoint(
    path: Path,
    model: PeakSetSIGReg,
    optimizers: list[torch.optim.Optimizer],
    schedulers: list[CapturableCosineSchedule],
    global_step: int,
    epoch: int,
    loss: float,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizers": [opt.state_dict() for opt in optimizers],
            "schedulers": [sched.state_dict() for sched in schedulers],
            "global_step": global_step,
            "epoch": epoch,
            "loss": loss,
        },
        path,
    )


def _prune_checkpoints(checkpoint_dir: Path, keep_top_k: int = 5) -> None:
    pts = sorted(checkpoint_dir.glob("step-*.pt"), key=lambda p: p.stat().st_mtime)
    if len(pts) <= keep_top_k:
        return
    losses: list[tuple[float, Path]] = []
    for p in pts:
        ckpt = torch.load(p, map_location="cpu", weights_only=True)
        losses.append((ckpt.get("loss", float("inf")), p))
    losses.sort(key=lambda x: x[0])
    keep = {p for _, p in losses[:keep_top_k]}
    keep.add(pts[-1])
    for p in pts:
        if p not in keep:
            p.unlink()


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
    num_epochs = float(config.num_epochs)
    steps_per_epoch = datamodule.train_steps
    total_steps = max(1, int(num_epochs * steps_per_epoch))
    loop_epochs = max(1, math.ceil(num_epochs))
    log_every_n_steps = int(config.get("log_every_n_steps", 50))
    checkpoint_every_steps = int(config.checkpoint_every_steps)
    config.num_peaks = datamodule.info["num_peaks"]
    logging.info("Training for %s epochs (%d steps).", num_epochs, total_steps)
    logging.info("Steps per epoch: %d", steps_per_epoch)
    logging.info("Total steps: %d", total_steps)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(config)
    model_param_metrics = collect_and_log_param_metrics(model)
    model.to(device).train()
    optimizers, schedulers = _build_optimizers(config, model, total_steps, device)
    checkpoint_dir = workdir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger(config, workdir)
    start_epoch, global_step = 0, 0
    if _pts := sorted(checkpoint_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime):
        logging.info("Resuming from checkpoint: %s", _pts[-1])
        ckpt = torch.load(_pts[-1], map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        for objs, key in [(optimizers, "optimizers"), (schedulers, "schedulers")]:
            for obj, state in zip(objs, ckpt[key]):
                obj.load_state_dict(state)
        global_step = ckpt["global_step"]
        start_epoch = ckpt["epoch"]
        del ckpt
    logger.log_metrics(model_param_metrics, step=global_step)
    _ac_name = str(config.get("autocast_dtype", "bf16")).lower()
    if _ac_name in {"bf16", "bfloat16"}:
        autocast_dtype = torch.bfloat16
    elif _ac_name in {"fp16", "float16", "half"}:
        autocast_dtype = torch.float16
    elif _ac_name in {"fp32", "float32", "none"}:
        autocast_dtype = None
    else:
        raise ValueError(f"Unsupported autocast_dtype: {_ac_name}")
    _compile_mode = str(config.get("compile_mode", "max-autotune"))
    if autocast_dtype is not None and device.type == "cuda":
        autocast_ctx = torch.autocast(device_type="cuda", dtype=autocast_dtype)
    else:
        autocast_ctx = nullcontext()
    optimizer_type = str(config.get("optimizer", "adamw")).lower()
    device_prefetch_size = int(config.get("device_prefetch_size", 1))
    _msg_probe_raw = float(config.get("msg_probe_every_n_steps", 0))
    if 0 < _msg_probe_raw <= 1:
        reference_steps = total_steps if num_epochs < 1 else steps_per_epoch
        msg_probe_every_n_steps = max(1, int(_msg_probe_raw * reference_steps))
    else:
        msg_probe_every_n_steps = int(_msg_probe_raw)
    _gcn = config.get("grad_clip_norm", None)
    grad_clip_norm = float(_gcn) if _gcn is not None else None
    train_loader = datamodule.train_loader
    last_msg_probe_metrics: dict[str, float] = {}
    _wandb_run = getattr(logger, "experiment", None)
    for epoch in range(start_epoch, loop_epochs):
        prefetcher = _BatchPrefetcher(
            iter(train_loader),
            device,
            prefetch_size=device_prefetch_size,
        )
        epoch_steps = min(steps_per_epoch, total_steps - global_step)
        pbar = tqdm(total=epoch_steps, desc=f"Epoch {epoch}", unit="step")
        while global_step < total_steps and (batch := prefetcher.next()) is not None:
            # Forward (compiled, separate from backward)
            torch.compiler.cudagraph_mark_step_begin()
            with autocast_ctx:
                metrics = _get_compiled_forward(model, _compile_mode)(batch)
            # Backward + optimizer (not compiled)
            metrics["loss"].backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    _get_trainable_params(model), max_norm=grad_clip_norm
                )
            runtime_norm_metrics = (
                collect_runtime_norm_metrics(model)
                if (global_step + 1) % log_every_n_steps == 0
                else None
            )
            for opt in optimizers:
                opt.step()
            for sched in schedulers:
                sched.step()
            global_step += 1
            pbar.update(1)
            if global_step % log_every_n_steps == 0:
                loss_val = float(metrics["loss"].detach())
                pbar.set_postfix(loss=f"{loss_val:.4f}", step=global_step)
                log_metrics = {
                    f"train/{k}": float(v.detach()) for k, v in metrics.items()
                }
                log_metrics["train/learning_rate"] = float(
                    optimizers[0].param_groups[0]["lr"]
                )
                if runtime_norm_metrics is not None:
                    log_metrics.update(
                        {f"train/{k}": v for k, v in runtime_norm_metrics.items()}
                    )
                log_metrics["epoch"] = epoch
                log_metrics["global_step"] = global_step
                logger.log_metrics(log_metrics, step=global_step)
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            if global_step % checkpoint_every_steps == 0:
                _save_checkpoint(
                    checkpoint_dir / f"step-{global_step:08d}.pt",
                    model,
                    optimizers,
                    schedulers,
                    global_step,
                    epoch,
                    float(metrics["loss"]),
                )
                _prune_checkpoints(checkpoint_dir, keep_top_k=15)
            if (
                msg_probe_every_n_steps > 0
                and global_step % msg_probe_every_n_steps == 0
            ):
                _probe_epoch_log: list[dict[str, float]] = []
                probe_metrics = run_msg_probe(
                    config=config,
                    model=model,
                    device=device,
                    on_epoch_end=_probe_epoch_log.append,
                )
                logger.log_metrics(probe_metrics, step=global_step)
                if _wandb_run is not None and _probe_epoch_log:
                    import wandb

                    _epochs = [
                        int(m["msg_probe_epoch"]) for m in _probe_epoch_log
                    ]
                    _curve_keys = [
                        ("r2_mean", "msg_probe/train/r2_mean", "msg_probe/test/r2_mean"),
                        ("auc_fg_mean", "msg_probe/train/auc_fg_mean", "msg_probe/test/auc_fg_mean"),
                    ]
                    for label, train_key, test_key in _curve_keys:
                        _wandb_run.log(
                            {
                                f"msg_probe_curve/{label}": wandb.plot.line_series(
                                    xs=_epochs,
                                    ys=[
                                        [m[train_key] for m in _probe_epoch_log],
                                        [m[test_key] for m in _probe_epoch_log],
                                    ],
                                    keys=["train", "test"],
                                    title=f"MSG Probe {label} (step {global_step})",
                                    xname="probe_epoch",
                                ),
                            },
                            step=global_step,
                        )
                last_msg_probe_metrics = probe_metrics
                logging.info(
                    "step=%d msg_probe(test_r2_mol_weight=%.4f test_auc_fg_mean=%.4f fg_tasks=%d)",
                    global_step,
                    probe_metrics["msg_probe/test/r2_mol_weight"],
                    probe_metrics["msg_probe/test/auc_fg_mean"],
                    int(probe_metrics["msg_probe/num_fg_tasks"]),
                )
        pbar.close()
    _save_checkpoint(
        checkpoint_dir / "last.pt",
        model,
        optimizers,
        schedulers,
        global_step,
        loop_epochs,
        float("nan"),
    )
    if last_msg_probe_metrics:
        logging.info("=== Final MSG Probe Results ===")
        for k, v in sorted(last_msg_probe_metrics.items()):
            logging.info("  %s: %.6f", k, v)
    else:
        logging.info("No MSG probe results were collected during training.")
    return {**last_msg_probe_metrics, **model_param_metrics}


if __name__ == "__main__":
    import argparse
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Train peak-set SIGReg model.")
    parser.add_argument("--config", required=True, help="Path to config file.")
    parser.add_argument("--workdir", required=True, help="Output directory.")
    args = parser.parse_args()
    from utils.training import load_config

    train_and_evaluate(
        load_config(args.config), workdir=Path(args.workdir).expanduser().resolve()
    )
