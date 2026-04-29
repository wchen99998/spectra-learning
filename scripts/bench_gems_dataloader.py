"""Benchmark native GeMS DataLoader throughput.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/bench_gems_dataloader.py \
        --artifact-dir /path/to/gems-native-artifact
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler

from spectra_learning.data.gems.collate import GemsBatchCollator
from spectra_learning.data.gems.dataset import GemsMemmapDataset
from spectra_learning.data.gems.native import load_gems_native_metadata


def _parse_workers(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _build_dataset(artifact_dir: Path, split: str) -> GemsMemmapDataset:
    metadata = load_gems_native_metadata(artifact_dir)
    shard_key = f"{split}_shards"
    length_key = f"{split}_lengths"
    entries = [
        {
            "dir": str(artifact_dir / split / name),
            "length": int(length),
        }
        for name, length in zip(
            metadata[shard_key],
            metadata[length_key],
            strict=True,
        )
    ]
    return GemsMemmapDataset(entries)


def _build_loader(
    dataset: GemsMemmapDataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    persistent_workers: bool,
    warmup_batches: int,
    measure_batches: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loader_kwargs: dict = {
        "dataset": dataset,
        "batch_size": batch_size,
        "sampler": RandomSampler(
            dataset,
            replacement=True,
            num_samples=batch_size * (warmup_batches + measure_batches),
            generator=generator,
        ),
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": True,
        "collate_fn": GemsBatchCollator(
            augment=True,
            num_target_blocks=2,
            context_fraction=0.35,
            target_fraction=0.2,
            block_min_len=1,
            use_precursor_token=False,
            num_peaks=64,
            max_precursor_mz=1000.0,
            min_peak_intensity=1e-4,
            peak_drop_min_intensity=1e-4,
            peak_ordering="mz",
            precursor_peak_exclusion_window_da=0.0,
        ),
        "generator": generator,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = persistent_workers
    return DataLoader(**loader_kwargs)


def _measure_throughput(
    dataset: GemsMemmapDataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    persistent_workers: bool,
    warmup_batches: int,
    measure_batches: int,
    runs: int,
    seed: int,
) -> dict[str, float | list[float] | int]:
    times = []
    for run_idx in range(runs):
        loader = _build_loader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            warmup_batches=warmup_batches,
            measure_batches=measure_batches,
            seed=seed + run_idx,
        )
        iterator = iter(loader)
        for _ in range(warmup_batches):
            next(iterator)
        start = time.perf_counter()
        for _ in range(measure_batches):
            next(iterator)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        del iterator, loader

    median = statistics.median(times)
    samples = batch_size * measure_batches
    return {
        "median_s": median,
        "min_s": min(times),
        "max_s": max(times),
        "ms_per_batch": median / measure_batches * 1e3,
        "samples_per_s": samples / median,
        "runs": times,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark native GeMS DataLoader")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", default="0,1,2,4,8")
    parser.add_argument("--warmup-batches", type=int, default=10)
    parser.add_argument("--measure-batches", type=int, default=100)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=torch.cuda.is_available(),
    )
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    artifact_dir = args.artifact_dir.expanduser().resolve()
    dataset = _build_dataset(artifact_dir, args.split)
    workers = _parse_workers(args.num_workers)

    print(f"Artifact: {artifact_dir}")
    print(f"Split: {args.split}")
    print(f"Samples: {len(dataset):,}")
    print(f"Batch size: {args.batch_size}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Pin memory: {args.pin_memory}")
    print(f"Persistent workers: {args.persistent_workers}")
    print()

    results = {}
    for num_workers in workers:
        result = _measure_throughput(
            dataset,
            batch_size=args.batch_size,
            num_workers=num_workers,
            pin_memory=args.pin_memory,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=args.persistent_workers,
            warmup_batches=args.warmup_batches,
            measure_batches=args.measure_batches,
            runs=args.runs,
            seed=args.seed,
        )
        results[num_workers] = result

    baseline = results[workers[0]]["samples_per_s"]
    for num_workers in workers:
        result = results[num_workers]
        speedup = float(result["samples_per_s"]) / float(baseline)
        print(
            f"num_workers={num_workers:<2d}  "
            f"{float(result['ms_per_batch']):7.2f} ms/batch  "
            f"{float(result['samples_per_s']):9.0f} samples/s  "
            f"{speedup:5.2f}x vs {workers[0]}"
        )


if __name__ == "__main__":
    main()
