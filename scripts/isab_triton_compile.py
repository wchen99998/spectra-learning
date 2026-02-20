"""Benchmark ISAB: vanilla eager vs TritonISAB vs torch.compile.

Usage:
    python scripts/isab_triton_compile.py

Compares three execution modes for a single ISAB layer:
1. Vanilla ISAB (eager, unfused flex_attention)
2. TritonISAB (hand-fused Triton kernels + custom attention)
3. torch.compile(ISAB) (auto-generated Triton via inductor)

Uses CUDA events for precise GPU kernel timing.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from networks.isab_triton import TritonISAB
from networks.set_transformer_torch import ISAB

# Config shapes (gems_a_50_mask.py)
BATCH_SIZE = 512
NUM_PEAKS = 60
MODEL_DIM = 256
NUM_HEADS = 8
NUM_KV_HEADS = 4
NUM_INDUCING_POINTS = 32
ATTENTION_MLP_MULTIPLE = 4.0

DEVICE = "cuda"
DTYPE = torch.bfloat16


def benchmark_cuda_events(
    fn, warmup: int = 100, repeat: int = 10_000, label: str = "",
) -> float:
    """Benchmark using CUDA events for precise GPU kernel timing."""
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            fn()
    torch.cuda.synchronize()

    # Create events
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]

    # Timed runs
    for i in range(repeat):
        start_events[i].record()
        with torch.no_grad():
            fn()
        end_events[i].record()
    torch.cuda.synchronize()

    # Collect timings
    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times_ms.sort()

    # Trim top/bottom 5% for stable median
    trim = max(1, repeat // 20)
    trimmed = times_ms[trim:-trim]
    mean_ms = sum(trimmed) / len(trimmed)
    median_ms = trimmed[len(trimmed) // 2]
    p5 = times_ms[trim]
    p95 = times_ms[-trim - 1]

    print(f"  [{label}]  median={median_ms:.3f}ms  mean={mean_ms:.3f}ms  "
          f"p5={p5:.3f}ms  p95={p95:.3f}ms  ({repeat} iters)")
    return median_ms


def main():
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"dtype: {DTYPE}")
    print(f"Shapes: B={BATCH_SIZE}, N={NUM_PEAKS}, D={MODEL_DIM}")
    print(f"  heads={NUM_HEADS}, kv_heads={NUM_KV_HEADS}, inducing={NUM_INDUCING_POINTS}")
    print()

    # Create vanilla ISAB
    torch.manual_seed(42)
    vanilla = ISAB(
        dim=MODEL_DIM, num_inducing_points=NUM_INDUCING_POINTS,
        n_heads=NUM_HEADS, n_kv_heads=NUM_KV_HEADS,
        attention_mlp_multiple=ATTENTION_MLP_MULTIPLE,
    ).to(DEVICE, DTYPE).eval()

    # Create TritonISAB from vanilla weights
    triton_isab = TritonISAB.from_vanilla_isab(vanilla).eval()

    # Create torch.compiled version
    class ISABWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.isab = m
        def forward(self, x):
            return self.isab(x, kv_block_mask=None, q_block_mask=None)

    compiled = torch.compile(ISABWrapper(vanilla), backend="inductor", mode="max-autotune")

    # Input
    x = torch.randn(BATCH_SIZE, NUM_PEAKS, MODEL_DIM, device=DEVICE, dtype=DTYPE)

    # Correctness check
    print("=== Correctness ===")
    with torch.no_grad():
        ref = vanilla(x, kv_block_mask=None, q_block_mask=None)
        triton_out = triton_isab(x)
        print("Compiling torch.compile version...")
        compiled_out = compiled(x)
        torch.cuda.synchronize()

    diff_triton = (ref.float() - triton_out.float()).abs()
    diff_compiled = (ref.float() - compiled_out.float()).abs()
    print(f"TritonISAB vs vanilla:  max={diff_triton.max():.2e}, mean={diff_triton.mean():.2e}")
    print(f"Compiled vs vanilla:    max={diff_compiled.max():.2e}, mean={diff_compiled.mean():.2e}")
    print()

    # Benchmark
    print("=== Benchmark (CUDA events, 10k iters, trimmed 5%) ===")
    t_vanilla = benchmark_cuda_events(
        lambda: vanilla(x, kv_block_mask=None, q_block_mask=None),
        label="vanilla (eager)",
    )
    t_triton = benchmark_cuda_events(
        lambda: triton_isab(x),
        label="TritonISAB",
    )
    t_compiled = benchmark_cuda_events(
        lambda: compiled(x),
        label="torch.compile",
    )
    print()

    print("=== Speedup (median) ===")
    print(f"  TritonISAB vs vanilla:    {t_vanilla / t_triton:.2f}x")
    print(f"  torch.compile vs vanilla: {t_vanilla / t_compiled:.2f}x")
    print(f"  TritonISAB vs torch.compile: {t_compiled / t_triton:.2f}x")


if __name__ == "__main__":
    main()
