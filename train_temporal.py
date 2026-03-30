"""Temporal finetuning loop for next-frame prediction.

Loads a pretrained spatial JEPA checkpoint and finetunes with temporal
next-frame prediction objective.

Usage:
    python train_temporal.py \
        --config configs/gems_a_temporal.py \
        --workdir experiments/temporal_run \
        --pretrained_checkpoint experiments/spatial_run/checkpoints/step-100000.pt
"""

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

from input_pipeline_temporal import TemporalLightningDataModule
from models.model import PeakSetSIGReg
from utils.msg_probe import run_msg_probe
from utils.schedulers import CapturableCosineSchedule
from utils.training import (
    build_logger,
    build_model_from_config,
    collect_and_log_param_metrics,
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
        and any(t in name for t in ("attention.", "feed_forward.", "cross_attn."))
    )


_TEMPORAL_BATCH_KEYS = frozenset(
    {
        "frame_peak_mz",
        "frame_peak_intensity",
        "frame_peak_valid_mask",
        "frame_rt",
        "frame_precursor_mz",
        "next_frame_peak_mz",
        "next_frame_peak_intensity",
        "next_frame_peak_valid_mask",
        "next_frame_rt",
        "next_frame_precursor_mz",
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
        if k in _TEMPORAL_BATCH_KEYS
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


class _CUDAGraphRunner:
    """Explicit CUDA graph capture with static input buffers for training."""

    def __init__(self, compile_kwargs=None):
        self.graph = None
        self.static_inputs: dict[str, torch.Tensor] | None = None
        self.static_output = None
        self.compiled_fn = None
        self.compile_kwargs = compile_kwargs or {}

    def _warmup_and_capture(self, fn, batch):
        if self.compiled_fn is None:
            self.compiled_fn = torch.compile(fn, **self.compile_kwargs)
        self.static_inputs = {k: v.clone() for k, v in batch.items()}
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self.compiled_fn(self.static_inputs)
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_output = self.compiled_fn(self.static_inputs)

    def run(self, fn, batch):
        if self.graph is None:
            self._warmup_and_capture(fn, batch)
        else:
            for k, v in batch.items():
                self.static_inputs[k].copy_(v)
        self.graph.replay()
        return self.static_output


def _get_teacher_runner(model: PeakSetSIGReg) -> _CUDAGraphRunner:
    runner = getattr(model, "_temporal_teacher_graph_runner", None)
    if runner is None:
        runner = _CUDAGraphRunner(
            compile_kwargs={"mode": "max-autotune-no-cudagraphs", "dynamic": False}
        )
        model._temporal_teacher_graph_runner = runner
    return runner


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


def _build_temporal_optimizers(
    config: config_dict.ConfigDict,
    model: PeakSetSIGReg,
    total_steps: int,
    device: torch.device,
) -> tuple[list[torch.optim.Optimizer], list[CapturableCosineSchedule]]:
    """Build optimizer with differential LR for encoder vs temporal predictor."""
    base_lr = float(config.learning_rate)
    encoder_lr = float(
        config.get("encoder_learning_rate", config.get("encoder_finetune_lr", None))
        or base_lr
    )
    warmup_steps = int(config.get("warmup_steps", 0))
    min_learning_rate = config.get("min_learning_rate", None)
    b2 = float(config.get("b2", 0.999))
    weight_decay = float(config.weight_decay)
    is_cuda = device.type == "cuda"
    capturable = bool(config.get("optimizer_capturable", True)) and is_cuda
    fused_cfg = config.get("optimizer_fused", None)
    fused = is_cuda if fused_cfg is None else bool(fused_cfg) and is_cuda

    temporal_prefixes = (
        "temporal_predictor.",
        "temporal_rt_proj.",
        "temporal_query_token",
    )
    optimizer_type = str(config.get("optimizer", "adamw")).lower()

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
        encoder_matrix, encoder_scalar = [], []
        temporal_matrix, temporal_scalar = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            is_temporal = any(name.startswith(p) for p in temporal_prefixes)
            if param.ndim >= 2:
                (temporal_matrix if is_temporal else encoder_matrix).append(param)
            else:
                (temporal_scalar if is_temporal else encoder_scalar).append(param)

        muon_lr = float(config.get("muon_lr", None) or base_lr)
        encoder_muon_lr = float(config.get("encoder_muon_lr", None) or encoder_lr)
        adamw_lr = float(config.get("adamw_lr", None) or base_lr)
        encoder_adamw_lr = float(config.get("encoder_adamw_lr", None) or encoder_lr)
        muon_kwargs = dict(
            momentum=float(config.get("muon_momentum", 0.95)),
            nesterov=bool(config.get("muon_nesterov", True)),
            ns_steps=int(config.get("muon_ns_steps", 5)),
            weight_decay=float(config.get("muon_weight_decay", None) or weight_decay),
            adjust_lr_fn=str(config.get("muon_adjust_lr_fn", "match_rms_adamw")),
        )

        optimizers: list[torch.optim.Optimizer] = []
        schedulers: list[CapturableCosineSchedule] = []
        muon_adamw_specs: list[tuple[list, str, float]] = [
            (encoder_matrix, "muon", encoder_muon_lr),
            (temporal_matrix, "muon", muon_lr),
            (encoder_scalar, "adamw", encoder_adamw_lr),
            (temporal_scalar, "adamw", adamw_lr),
        ]
        for params, opt_type, lr in muon_adamw_specs:
            if not params:
                continue
            if opt_type == "muon":
                opt = torch.optim.Muon(
                    params, lr=torch.tensor(lr), **muon_kwargs
                )
            else:
                opt = torch.optim.AdamW(
                    params,
                    lr=torch.tensor(lr),
                    betas=(0.9, b2),
                    weight_decay=0.0,
                    capturable=capturable,
                    fused=fused,
                )
            optimizers.append(opt)
            schedulers.append(_make_schedule(opt, lr))
        return optimizers, schedulers

    # AdamW path
    encoder_decay, encoder_no_decay = [], []
    temporal_decay, temporal_no_decay = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_temporal = any(name.startswith(p) for p in temporal_prefixes)
        if _is_weight_decay_target(name, param):
            (temporal_decay if is_temporal else encoder_decay).append(param)
        else:
            (temporal_no_decay if is_temporal else encoder_no_decay).append(param)

    def _build_param_groups(
        decay_params: list[torch.nn.Parameter],
        no_decay_params: list[torch.nn.Parameter],
    ) -> list[dict[str, Any]]:
        param_groups: list[dict[str, Any]] = []
        if decay_params:
            param_groups.append(
                {
                    "params": decay_params,
                    "weight_decay": weight_decay,
                }
            )
        if no_decay_params:
            param_groups.append(
                {
                    "params": no_decay_params,
                    "weight_decay": 0.0,
                }
            )
        return param_groups

    optimizers: list[torch.optim.Optimizer] = []
    schedulers: list[CapturableCosineSchedule] = []
    optimizer_specs = [
        (
            _build_param_groups(encoder_decay, encoder_no_decay),
            encoder_lr,
        ),
        (
            _build_param_groups(temporal_decay, temporal_no_decay),
            base_lr,
        ),
    ]
    for param_groups, lr in optimizer_specs:
        if not param_groups:
            continue
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=torch.tensor(lr),
            betas=(0.9, b2),
            weight_decay=0.0,
            capturable=capturable,
            fused=fused,
        )
        optimizers.append(optimizer)
        schedulers.append(_make_schedule(optimizer, lr))
    return optimizers, schedulers


def _load_pretrained_checkpoint(
    model: PeakSetSIGReg,
    checkpoint_path: str,
) -> None:
    """Load pretrained spatial JEPA checkpoint, allowing missing temporal keys."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    sd = ckpt.get("model") or ckpt.get("state_dict") or ckpt
    # Strip "model." prefix if present
    prefixed = {
        k.removeprefix("model."): v for k, v in sd.items() if k.startswith("model.")
    }
    sd = prefixed or sd
    for key in tuple(sd):
        if key.startswith("masked_latent_readout.") or key.endswith(
            (
                "position_embedding.weight",
                "predictor_position_embedding.weight",
            )
        ):
            sd.pop(key)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # Validate only temporal keys are missing
    allowed_prefixes = (
        "temporal_predictor.",
        "temporal_rt_proj.",
        "temporal_query_token",
        "masked_latent_readout.",
    )
    allowed_suffixes = (
        "encoder.position_embedding.weight",
        "predictor_position_embedding.weight",
    )
    bad_missing = [
        k
        for k in missing
        if not any(k.startswith(p) for p in allowed_prefixes)
        and not k.endswith(allowed_suffixes)
    ]
    if bad_missing:
        raise RuntimeError(
            f"Unexpected missing keys in pretrained checkpoint: {bad_missing}"
        )
    if unexpected:
        logging.warning("Unexpected keys in checkpoint (ignored): %s", unexpected)
    logging.info(
        "Loaded pretrained checkpoint: %d keys loaded, %d temporal keys initialized fresh",
        len(sd) - len(unexpected),
        len(missing),
    )


def train_temporal(
    config: config_dict.ConfigDict,
    workdir: str | Path,
    pretrained_checkpoint: str | None = None,
) -> dict[str, float]:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    seed = int(config.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    datamodule = TemporalLightningDataModule(config, seed=seed)
    total_steps = max(1, int(datamodule.train_steps))
    log_every_n_steps = int(config.get("log_every_n_steps", 50))
    checkpoint_every_steps = int(config.checkpoint_every_steps)
    config.num_peaks = datamodule.info["num_peaks"]

    logging.info("Temporal finetuning for %d steps.", total_steps)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(config)

    # Load pretrained checkpoint (partial — temporal keys missing)
    ckpt_path = pretrained_checkpoint or config.get("pretrained_checkpoint", None)
    if ckpt_path:
        _load_pretrained_checkpoint(model, ckpt_path)
    else:
        logging.warning("No pretrained checkpoint provided — training from scratch.")

    model_param_metrics = collect_and_log_param_metrics(model)
    model.to(device).train()

    optimizers, schedulers = _build_temporal_optimizers(
        config, model, total_steps, device
    )

    checkpoint_dir = workdir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger(config, workdir)

    global_step = 0
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

    device_prefetch_size = int(config.get("device_prefetch_size", 1))
    _gcn = config.get("grad_clip_norm", None)
    grad_clip_norm = float(_gcn) if _gcn is not None else None
    optimizer_type = str(config.get("optimizer", "adamw")).lower()

    _msg_probe_raw = float(config.get("msg_probe_every_n_steps", 0))
    if 0 < _msg_probe_raw <= 1:
        msg_probe_every_n_steps = max(1, int(_msg_probe_raw * total_steps))
    else:
        msg_probe_every_n_steps = int(_msg_probe_raw)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    train_loader = datamodule.train_loader

    # Compile forward_temporal
    compiled_forward = None
    last_msg_probe_metrics: dict[str, float] = {}
    _wandb_run = getattr(logger, "experiment", None)
    prefetcher = _BatchPrefetcher(
        iter(train_loader),
        device,
        prefetch_size=device_prefetch_size,
    )
    pbar = tqdm(total=total_steps, desc="Temporal train", unit="step")

    while global_step < total_steps and (batch := prefetcher.next()) is not None:
        # Teacher embeddings (CUDA graph)
        if model.teacher_encoder is not None:
            runner = _get_teacher_runner(model)
            with autocast_ctx:
                teacher_embeddings = runner.run(
                    model.compute_next_frame_teacher_embeddings, batch
                )
        else:
            teacher_embeddings = None

        # Forward
        torch.compiler.cudagraph_mark_step_begin()
        with autocast_ctx:
            if compiled_forward is None:
                compiled_forward = torch.compile(
                    model.forward_temporal,
                    mode=_compile_mode,
                    dynamic=False,
                )
            metrics = compiled_forward(batch, teacher_embeddings)

        # Backward
        metrics["loss"].backward()
        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                trainable_params, max_norm=grad_clip_norm
            )
        for opt in optimizers:
            opt.step()
            opt.zero_grad(set_to_none=True)
        for sched in schedulers:
            sched.step()
        model.update_teacher()
        global_step += 1
        pbar.update(1)

        if global_step % log_every_n_steps == 0:
            loss_val = float(metrics["loss"].detach())
            pbar.set_postfix(loss=f"{loss_val:.4f}", step=global_step)
            log_metrics = {
                f"train/{k}": float(v.detach()) for k, v in metrics.items()
            }
            if optimizer_type == "muon":
                _muon_labels = (
                    "encoder_muon", "temporal_muon",
                    "encoder_adamw", "temporal_adamw",
                )
                for opt, label in zip(optimizers, _muon_labels):
                    log_metrics[f"train/lr_{label}"] = float(
                        opt.param_groups[0]["lr"]
                    )
            elif len(optimizers) == 2:
                log_metrics["train/encoder_learning_rate"] = float(
                    optimizers[0].param_groups[0]["lr"]
                )
                log_metrics["train/temporal_learning_rate"] = float(
                    optimizers[1].param_groups[0]["lr"]
                )
            else:
                log_metrics["train/learning_rate"] = float(
                    optimizers[0].param_groups[0]["lr"]
                )
            log_metrics["global_step"] = global_step
            logger.log_metrics(log_metrics, step=global_step)

        if global_step % checkpoint_every_steps == 0:
            _save_checkpoint(
                checkpoint_dir / f"step-{global_step:08d}.pt",
                model,
                optimizers,
                schedulers,
                global_step,
                0,
                float(metrics["loss"]),
            )
            _prune_checkpoints(checkpoint_dir, keep_top_k=15)

        if (
            msg_probe_every_n_steps > 0
            and global_step % msg_probe_every_n_steps == 0
        ):
            probe_metrics = run_msg_probe(
                config=config,
                model=model,
                device=device,
            )
            logger.log_metrics(probe_metrics, step=global_step)
            last_msg_probe_metrics = probe_metrics
            logging.info(
                "step=%d msg_probe(test_r2_mol_weight=%.4f test_auc_fg_mean=%.4f)",
                global_step,
                probe_metrics["msg_probe/test/r2_mol_weight"],
                probe_metrics["msg_probe/test/auc_fg_mean"],
            )

    pbar.close()

    _save_checkpoint(
        checkpoint_dir / "last.pt",
        model,
        optimizers,
        schedulers,
        global_step,
        0,
        float("nan"),
    )
    return {**last_msg_probe_metrics, **model_param_metrics}


if __name__ == "__main__":
    import argparse
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Temporal finetuning for peak-set JEPA."
    )
    parser.add_argument("--config", required=True, help="Path to config file.")
    parser.add_argument("--workdir", required=True, help="Output directory.")
    parser.add_argument(
        "--pretrained_checkpoint",
        default=None,
        help="Path to pretrained spatial JEPA checkpoint.",
    )
    args = parser.parse_args()

    from utils.training import load_config

    train_temporal(
        load_config(args.config),
        workdir=Path(args.workdir).expanduser().resolve(),
        pretrained_checkpoint=args.pretrained_checkpoint,
    )
