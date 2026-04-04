"""Benchmark torch.optim.Muon vs gram_newton_schulz.Muon on the real model.

Usage:
    uv run python scripts/bench_muon.py --config configs/gems_small.py
"""

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import torch

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

from input_pipeline import TfLightningDataModule
from train import _BatchPrefetcher, _train_step_impl
from utils.training import build_model_from_config, load_config

torch.set_float32_matmul_precision("high")
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def _make_std_scheduler(optimizer, config, total_steps):
    """Standard PyTorch LR scheduler (warmup + cosine) that won't block torch.compile."""
    warmup_steps = int(config.get("warmup_steps", 0))
    base_lr = float(optimizer.param_groups[0]["lr"])
    min_lr = config.get("min_learning_rate", None)
    eta_min = float(min_lr) if min_lr is not None else 0.1 * base_lr

    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=eta_min
    )
    if warmup_steps > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )
    return cosine


def _collect_param_groups(model):
    """Split model params into 2D (Muon) and scalar (AdamW) groups."""
    weight_params, scalar_params = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2:
            weight_params.append(p)
        else:
            scalar_params.append(p)
    return weight_params, scalar_params


def _collect_param_groups_kernel_aware(model):
    """Split params into kernel-compatible Muon, kernel-incompatible 2D, and scalar groups."""
    muon_params, incompatible_2d_params, scalar_params = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2:
            scalar_params.append(p)
        elif p.stride()[0] % 8 != 0 or p.stride()[-1] % 8 != 0 and p.shape[-1] % 8 != 0:
            # quack kernels require stride[0] divisible by 8
            incompatible_2d_params.append(p)
        else:
            muon_params.append(p)
    return muon_params, incompatible_2d_params, scalar_params


def build_torch_muon(config, model, total_steps, device):
    """Build torch.optim.Muon + AdamW (current production setup)."""
    base_lr = float(config.learning_rate)
    b2 = float(config.get("b2", 0.999))
    weight_decay = float(config.weight_decay)
    muon_lr = float(config.get("muon_lr", None) or base_lr)
    adamw_lr = float(config.get("adamw_lr", None) or base_lr)
    muon_momentum = float(config.get("muon_momentum", 0.95))
    muon_wd = float(config.get("muon_weight_decay", None) or weight_decay)
    warmup_steps = int(config.get("warmup_steps", 0))
    min_lr = config.get("min_learning_rate", None)

    weight_params, scalar_params = _collect_param_groups(model)

    muon_opt = torch.optim.Muon(
        weight_params,
        lr=torch.tensor(muon_lr),
        weight_decay=muon_wd,
        momentum=muon_momentum,
        nesterov=bool(config.get("muon_nesterov", True)),
        adjust_lr_fn="match_rms_adamw",
    )
    adamw_opt = torch.optim.AdamW(
        scalar_params,
        lr=torch.tensor(adamw_lr),
        betas=(0.9, b2),
        weight_decay=0.0,
    )
    muon_sched = _make_std_scheduler(muon_opt, config, total_steps)
    adamw_sched = _make_std_scheduler(adamw_opt, config, total_steps)
    return [muon_opt, adamw_opt], [muon_sched, adamw_sched]


def build_gns_muon(config, model, total_steps, device, coefficients_preset, ns_algorithm, ns_use_kernels=True):
    """Build gram_newton_schulz.Muon with built-in scalar optimizer.

    When ns_use_kernels=True, params with incompatible strides are routed to AdamW.
    """
    from gram_newton_schulz import Muon as GNSMuon

    base_lr = float(config.learning_rate)
    b2 = float(config.get("b2", 0.999))
    weight_decay = float(config.weight_decay)
    muon_lr = float(config.get("muon_lr", None) or base_lr)
    adamw_lr = float(config.get("adamw_lr", None) or base_lr)
    muon_momentum = float(config.get("muon_momentum", 0.95))
    muon_wd = float(config.get("muon_weight_decay", None) or weight_decay)
    warmup_steps = int(config.get("warmup_steps", 0))
    min_lr = config.get("min_learning_rate", None)

    if ns_use_kernels:
        muon_params, incompatible_2d, scalar_params = _collect_param_groups_kernel_aware(model)
        # Put incompatible 2D params into AdamW with normal weight decay
        adamw_param_groups = [
            {"params": scalar_params, "weight_decay": 0.0},
        ]
        if incompatible_2d:
            adamw_param_groups.append(
                {"params": incompatible_2d, "weight_decay": weight_decay},
            )
            log.info("  %d params routed to Muon, %d incompatible 2D params to AdamW, %d scalars to AdamW",
                     len(muon_params), len(incompatible_2d), len(scalar_params))
    else:
        weight_params, scalar_params = _collect_param_groups(model)
        muon_params = weight_params
        adamw_param_groups = [{"params": scalar_params, "weight_decay": 0.0}]

    adamw_opt = torch.optim.AdamW(
        adamw_param_groups,
        lr=adamw_lr,
        betas=(0.9, b2),
    )

    muon_opt = GNSMuon(
        muon_params,
        lr=muon_lr,
        weight_decay=muon_wd,
        momentum=muon_momentum,
        nesterov=bool(config.get("muon_nesterov", True)),
        adjust_lr="rms_norm",
        ns_coefficients_preset=coefficients_preset,
        ns_algorithm=ns_algorithm,
        ns_use_kernels=ns_use_kernels,
        scalar_optimizer=adamw_opt,
    )

    muon_sched = _make_std_scheduler(muon_opt, config, total_steps)
    return [muon_opt], [muon_sched]


def _make_compiled_opt_and_sched_step(optimizers, schedulers):
    """Compile optimizer + scheduler steps together per the PyTorch recipe."""
    compiled_fns = []
    for opt, sched in zip(optimizers, schedulers):
        @torch.compile(fullgraph=False)
        def _step(o=opt, s=sched):
            o.step()
            s.step()
        compiled_fns.append(_step)
    return compiled_fns


def _compiled_train_step(
    model,
    batch,
    autocast_dtype,
    grad_clip_norm,
    compiled_opt_sched_steps,
    optimizers,
):
    """Train step with compiled optimizer+scheduler steps."""
    from contextlib import nullcontext
    device_type = next(model.parameters()).device.type
    if autocast_dtype is None or device_type != "cuda":
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(device_type=device_type, dtype=autocast_dtype)
    torch.compiler.cudagraph_mark_step_begin()
    with autocast_ctx:
        metrics = model.forward_augmented(batch)
    metrics["loss"].backward()
    if grad_clip_norm is not None and grad_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
    for fn in compiled_opt_sched_steps:
        fn()
    for opt in optimizers:
        opt.zero_grad(set_to_none=True)
    return metrics


def run_benchmark(
    label: str,
    config,
    model,
    optimizers,
    schedulers,
    prefetcher_factory,
    autocast_dtype,
    grad_clip_norm,
    warmup_steps: int,
    bench_steps: int,
    device,
    compile_optimizers: bool = False,
):
    """Run warmup + timed steps, return per-step timings."""
    model.train()
    compile_mode = str(config.get("compile_mode", "reduce-overhead"))
    model.forward_augmented = torch.compile(
        model.forward_augmented, mode=compile_mode, fullgraph=False
    )

    if compile_optimizers:
        compiled_fns = _make_compiled_opt_and_sched_step(optimizers, schedulers)
        def do_step(batch):
            return _compiled_train_step(
                model, batch, autocast_dtype, grad_clip_norm, compiled_fns, optimizers,
            )
    else:
        def do_step(batch):
            return _train_step_impl(
                model, batch, optimizers, schedulers, autocast_dtype, grad_clip_norm,
            )

    prefetcher = prefetcher_factory()

    # Warmup (torch.compile + CUDA graphs)
    log.info("[%s] Warming up %d steps...", label, warmup_steps)
    for _ in range(warmup_steps):
        batch = prefetcher.next()
        if batch is None:
            prefetcher = prefetcher_factory()
            batch = prefetcher.next()
        do_step(batch)
        model.update_teacher()
    torch.cuda.synchronize()
    log.info("[%s] Warmup done.", label)

    # Timed steps
    torch.cuda.synchronize()
    step_times = []
    for i in range(bench_steps):
        batch = prefetcher.next()
        if batch is None:
            prefetcher = prefetcher_factory()
            batch = prefetcher.next()

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        do_step(batch)
        model.update_teacher()

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        step_times.append(t1 - t0)

    return step_times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--bench-steps", type=int, default=50)
    args = parser.parse_args()

    config = load_config(args.config)
    config.enable_wandb = False
    config.msg_probe_every_n_steps = 0

    seed = int(config.seed)
    device = torch.device("cuda")

    # Data
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    datamodule = TfLightningDataModule(config, seed=seed)
    config.num_peaks = datamodule.info["num_peaks"]
    steps_per_epoch = datamodule.train_steps
    total_steps = max(1, int(float(config.num_epochs) * steps_per_epoch))

    _ac = str(config.get("autocast_dtype", "bf16")).lower()
    autocast_dtype = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
                      "fp16": torch.float16, "float16": torch.float16,
                      "fp32": None, "float32": None, "none": None}.get(_ac, torch.bfloat16)
    grad_clip_norm = config.get("grad_clip_norm", None)
    if grad_clip_norm is not None:
        grad_clip_norm = float(grad_clip_norm)

    device_prefetch_size = int(config.get("device_prefetch_size", 1))
    train_loader = datamodule.train_loader

    def make_prefetcher():
        return _BatchPrefetcher(iter(train_loader), device, prefetch_size=device_prefetch_size)

    # Variants to benchmark: (label, build_fn, compile_optimizers)
    variants = [
        ("torch.optim.Muon", lambda m: build_torch_muon(config, m, total_steps, device), False),
        ("torch.optim.Muon (compiled)", lambda m: build_torch_muon(config, m, total_steps, device), True),
        ("GNS gram YOU (kern)", lambda m: build_gns_muon(config, m, total_steps, device, "YOU_COEFFICIENTS", "gram_newton_schulz", ns_use_kernels=True), False),
    ]

    results = {}
    for label, build_fn, compile_opts in variants:
        log.info("=" * 60)
        log.info("Benchmarking: %s", label)
        log.info("=" * 60)

        # Fresh model each time for fair comparison
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.empty_cache()
        # Reset dynamo caches between variants
        torch._dynamo.reset()
        model = build_model_from_config(config)
        model.to(device)

        try:
            optimizers, schedulers = build_fn(model)
        except Exception as e:
            log.warning("[%s] Failed to build optimizer: %s", label, e)
            del model
            continue

        try:
            step_times = run_benchmark(
                label, config, model, optimizers, schedulers,
                make_prefetcher, autocast_dtype, grad_clip_norm,
                args.warmup_steps, args.bench_steps, device,
                compile_optimizers=compile_opts,
            )
        except Exception as e:
            log.warning("[%s] Failed during benchmark: %s", label, e)
            del model, optimizers, schedulers
            torch.cuda.empty_cache()
            continue

        # Stats (drop first 2 timed steps as potential outliers)
        trimmed = step_times[2:]
        median_ms = np.median(trimmed) * 1000
        mean_ms = np.mean(trimmed) * 1000
        std_ms = np.std(trimmed) * 1000
        min_ms = np.min(trimmed) * 1000
        max_ms = np.max(trimmed) * 1000
        p5_ms = np.percentile(trimmed, 5) * 1000
        p95_ms = np.percentile(trimmed, 95) * 1000

        results[label] = {
            "median_ms": median_ms,
            "mean_ms": mean_ms,
            "std_ms": std_ms,
            "min_ms": min_ms,
            "max_ms": max_ms,
            "p5_ms": p5_ms,
            "p95_ms": p95_ms,
            "throughput_samples_per_sec": config.batch_size / (median_ms / 1000),
        }

        del model, optimizers, schedulers
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 90)
    print(f"{'BENCHMARK RESULTS':^90}")
    print("=" * 90)
    print(f"Model: {config.encoder_num_layers}L/{config.model_dim}D/{config.encoder_num_heads}H, "
          f"batch_size={config.batch_size}, num_peaks={config.num_peaks}")
    print(f"Steps: {args.bench_steps} (first 2 dropped), warmup: {args.warmup_steps}")
    print("-" * 90)
    print(f"{'Variant':<40} {'Median':>8} {'Mean':>8} {'Std':>6} {'P5':>8} {'P95':>8} {'Samples/s':>10}")
    print("-" * 90)
    for label, r in results.items():
        print(f"{label:<40} {r['median_ms']:>7.2f}ms {r['mean_ms']:>7.2f}ms {r['std_ms']:>5.2f}ms "
              f"{r['p5_ms']:>7.2f}ms {r['p95_ms']:>7.2f}ms {r['throughput_samples_per_sec']:>9.0f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
