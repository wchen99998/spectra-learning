"""Benchmark FramePairDataset / TemporalLightningDataModule throughput.

Uses ``torch.utils.benchmark.Timer`` for single-item latency and manual
timing with multiple runs for DataLoader throughput (since DataLoader
iterators are stateful and non-reentrant).

Usage:
    python scripts/bench_temporal_dataloader.py [--data-dir data/gems_grouped]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.utils.benchmark as benchmark
from torch.utils.data import DataLoader

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from input_pipeline_temporal import FramePairDataset


def _build_loader(
    ds: FramePairDataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int | None,
    persistent_workers: bool,
) -> DataLoader:
    kwargs: dict = dict(
        dataset=ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor or 2
        kwargs["persistent_workers"] = persistent_workers
    return DataLoader(**kwargs)


def _measure_throughput(
    ds: FramePairDataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    persistent_workers: bool,
    n_warmup: int = 5,
    n_batches: int = 50,
    n_runs: int = 3,
) -> dict:
    """Time n_batches across n_runs, return stats."""
    # Ensure we don't exceed dataset size
    max_batches = len(ds) // batch_size - n_warmup - 1
    n_batches = min(n_batches, max(10, max_batches))
    times = []
    for _ in range(n_runs):
        loader = _build_loader(
            ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers and num_workers > 0,
        )
        it = iter(loader)
        # Warmup
        for _ in range(n_warmup):
            next(it)

        t0 = time.perf_counter()
        for _ in range(n_batches):
            next(it)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        del loader, it

    import numpy as np

    arr = np.array(times)
    median = float(np.median(arr))
    samples = n_batches * batch_size
    return {
        "median_s": median,
        "min_s": float(arr.min()),
        "max_s": float(arr.max()),
        "ms_per_batch": median / n_batches * 1e3,
        "samples_per_s": samples / median,
        "n_runs": n_runs,
        "n_batches": n_batches,
    }


def _print_row(label: str, r: dict) -> None:
    print(
        f"  {label:<30s}  "
        f"{r['ms_per_batch']:7.1f} ms/batch  "
        f"{r['samples_per_s']:8,.0f} samples/s  "
        f"(median {r['median_s']:.2f}s, min {r['min_s']:.2f}s, max {r['max_s']:.2f}s, "
        f"{r['n_runs']} runs × {r['n_batches']} batches)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark temporal DataLoader")
    parser.add_argument("--data-dir", type=Path, default=Path("data/gems_grouped"))
    parser.add_argument("--num-peaks", type=int, default=64)
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    with (data_dir / "manifest.json").open() as f:
        manifest = json.load(f)

    ds = FramePairDataset(
        file_list=manifest["train"]["files"],
        data_dir=data_dir / "train",
        num_peaks=args.num_peaks,
    )
    print(f"Dataset: {len(ds)} experiments, num_peaks={args.num_peaks}")
    print(f"CPU cores: {os.cpu_count()}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # ---- 1) Single-item latency ----
    print("\n" + "=" * 70)
    print("1. Single __getitem__ latency")
    print("=" * 70)

    indices = list(range(200))
    t = benchmark.Timer(
        stmt="[ds[i] for i in indices]",
        globals={"ds": ds, "indices": indices},
        label="__getitem__",
        sub_label=f"200 items",
        description=f"num_peaks={args.num_peaks}",
    )
    m = t.blocked_autorange(min_run_time=5.0)
    print(m)
    print(f"  → {m.median * 1e6 / len(indices):.0f} µs / item")

    # ---- 2) num_workers sweep ----
    print("\n" + "=" * 70)
    print("2. num_workers sweep  (batch_size=256, pin_memory=False, prefetch=2)")
    print("=" * 70)

    for nw in [0, 1, 2, 4, 8, 12, 16]:
        r = _measure_throughput(
            ds,
            batch_size=256,
            num_workers=nw,
            pin_memory=False,
            prefetch_factor=2,
            persistent_workers=True,
        )
        _print_row(f"num_workers={nw}", r)

    # ---- 3) batch_size sweep ----
    print("\n" + "=" * 70)
    print("3. batch_size sweep  (num_workers=8)")
    print("=" * 70)

    for bs in [64, 128, 256, 512, 1024]:
        r = _measure_throughput(
            ds,
            batch_size=bs,
            num_workers=8,
            pin_memory=False,
            prefetch_factor=2,
            persistent_workers=True,
        )
        _print_row(f"batch_size={bs}", r)

    # ---- 4) pin_memory comparison ----
    print("\n" + "=" * 70)
    print("4. pin_memory comparison  (batch_size=256, num_workers=8)")
    print("=" * 70)

    for pin in [False, True]:
        r = _measure_throughput(
            ds,
            batch_size=256,
            num_workers=8,
            pin_memory=pin,
            prefetch_factor=2,
            persistent_workers=True,
        )
        _print_row(f"pin_memory={pin}", r)

    # ---- 5) prefetch_factor sweep ----
    print("\n" + "=" * 70)
    print("5. prefetch_factor sweep  (batch_size=256, num_workers=8)")
    print("=" * 70)

    for pf in [1, 2, 4, 8]:
        r = _measure_throughput(
            ds,
            batch_size=256,
            num_workers=8,
            pin_memory=False,
            prefetch_factor=pf,
            persistent_workers=True,
        )
        _print_row(f"prefetch_factor={pf}", r)

    # ---- 6) persistent_workers comparison ----
    print("\n" + "=" * 70)
    print("6. persistent_workers comparison  (batch_size=256, num_workers=8)")
    print("=" * 70)

    for pw in [False, True]:
        r = _measure_throughput(
            ds,
            batch_size=256,
            num_workers=8,
            pin_memory=False,
            prefetch_factor=2,
            persistent_workers=pw,
        )
        _print_row(f"persistent_workers={pw}", r)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
