"""Profile training loop with torch.profiler (CPU + CUDA + memory).

Usage:
    uv run python scripts/profile_training.py --config configs/gems_small.py

Outputs:
    - Chrome trace: profile_out/trace.json.gz  (open in chrome://tracing)
    - Console table: top operators by CUDA/CPU time and memory
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.profiler

from spectra_learning.data.gems.datamodule import GemsNativeDataModule
from spectra_learning.training.batch import BatchPrefetcher
from spectra_learning.training.optimization import build_optimizers
from spectra_learning.training.steps import train_step_impl
from utils.training import build_model_from_config, load_config, parse_autocast_dtype

torch.set_float32_matmul_precision("high")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--warmup-steps", type=int, default=5,
                        help="Steps before profiling (for torch.compile warmup)")
    parser.add_argument("--active-steps", type=int, default=8,
                        help="Steps to profile")
    parser.add_argument("--outdir", default="profile_out")
    parser.add_argument("--no-compile", action="store_true",
                        help="Skip torch.compile to profile eager mode")
    parser.add_argument("--disable-msg-probe", action="store_true", default=True)
    args = parser.parse_args()

    config = load_config(args.config)
    # Disable wandb and msg probe for profiling
    config.enable_wandb = False
    config.msg_probe_every_n_steps = 0

    seed = int(config.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    datamodule = GemsNativeDataModule(config, seed=seed)
    steps_per_epoch = datamodule.train_steps
    total_steps = max(1, int(float(config.num_epochs) * steps_per_epoch))

    model = build_model_from_config(config)
    model.to(device).train()

    optimizers, schedulers = build_optimizers(config, model, total_steps, device)

    # Autocast dtype
    autocast_dtype = parse_autocast_dtype(config.get("autocast_dtype", "bf16"))

    grad_clip_norm = config.get("grad_clip_norm", None)
    if grad_clip_norm is not None:
        grad_clip_norm = float(grad_clip_norm)

    # Optional compile
    if not args.no_compile:
        compile_mode = str(config.get("compile_mode", "reduce-overhead"))
        log.info("Compiling with mode=%s", compile_mode)
        model.forward_augmented = torch.compile(
            model.forward_augmented,
            mode=compile_mode,
            fullgraph=True,
        )

    device_prefetch_size = int(config.get("device_prefetch_size", 1))
    train_loader = datamodule.train_loader

    # Warmup (needed for torch.compile + CUDA graphs)
    log.info("Running %d warmup steps...", args.warmup_steps)
    prefetcher = BatchPrefetcher(iter(train_loader), device, prefetch_size=device_prefetch_size)
    for i in range(args.warmup_steps):
        batch = prefetcher.next()
        if batch is None:
            break
        train_step_impl(model, batch, optimizers, schedulers, autocast_dtype, grad_clip_norm)
    if device.type == "cuda":
        torch.cuda.synchronize()
    log.info("Warmup done.")

    # Profile
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    schedule = torch.profiler.schedule(
        wait=0, warmup=1, active=args.active_steps - 1, repeat=1
    )

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(outdir)),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        for i in range(args.active_steps):
            batch = prefetcher.next()
            if batch is None:
                # wrap around
                prefetcher = BatchPrefetcher(iter(train_loader), device, prefetch_size=device_prefetch_size)
                batch = prefetcher.next()
            train_step_impl(model, batch, optimizers, schedulers, autocast_dtype, grad_clip_norm)
            if device.type == "cuda":
                torch.cuda.synchronize()
            prof.step()

    # Print summary tables
    print("\n" + "=" * 80)
    print("TOP 30 OPERATORS BY CUDA TIME (total)")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

    print("\n" + "=" * 80)
    print("TOP 30 OPERATORS BY CPU TIME (total)")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))

    print("\n" + "=" * 80)
    print("TOP 20 OPERATORS BY SELF CUDA MEMORY USAGE")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20))

    # Also group by input shape
    print("\n" + "=" * 80)
    print("TOP 20 BY CUDA TIME (grouped by input shape)")
    print("=" * 80)
    print(prof.key_averages(group_by_input_shape=True).table(
        sort_by="cuda_time_total", row_limit=20))

    log.info("TensorBoard logs saved to %s", outdir)
    log.info("View with: tensorboard --logdir %s", outdir)


if __name__ == "__main__":
    main()
