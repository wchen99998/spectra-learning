from __future__ import annotations

import argparse
import gc
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectra_learning.config import load_config
from spectra_learning.data.gems.grouped import GroupedGemsDataModule


def _configure(args: argparse.Namespace, workers: int):
    overrides: dict[str, Any] = {
        "dataloader_num_workers": workers,
        "dataloader_prefetch_factor": args.prefetch_factor,
        "dataloader_output_format": "numpy",
        "dataloader_pin_memory": False,
    }
    if args.artifact_dir is not None:
        overrides["artifact_dir"] = str(args.artifact_dir)
    if workers > 0:
        overrides["dataloader_multiprocessing_context"] = (
            args.multiprocessing_context
        )
    return load_config(args.config, overrides)


def _consumer(backend: str) -> Callable[[dict[str, Any]], None]:
    if backend == "jax":
        import jax

        from spectra_learning.training.pretrain_jax import numpy_batch_to_jax

        return lambda batch: jax.block_until_ready(numpy_batch_to_jax(batch))
    return lambda batch: batch["peak_mz"].shape


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return ordered[index]


def _run_once(
    args: argparse.Namespace,
    workers: int,
    repeat: int,
) -> dict[str, float]:
    config = _configure(args, workers)
    consume = _consumer(args.backend)

    setup_start = time.perf_counter()
    datamodule = GroupedGemsDataModule(
        config,
        seed=int(config.get("seed", 0)) + repeat,
        distributed_world_size=args.world_size,
        distributed_rank=args.rank,
        distributed_local_rank=args.local_rank,
    )
    setup_seconds = time.perf_counter() - setup_start

    loader = datamodule.train_loader_for_epoch(repeat)
    iterator = iter(loader)
    first_start = time.perf_counter()
    first_batch = next(iterator)
    consume(first_batch)
    first_batch_seconds = time.perf_counter() - first_start

    for _ in range(max(0, args.warmup_batches - 1)):
        consume(next(iterator))

    groups = 0
    spectra = 0
    latencies: list[float] = []
    timed_start = time.perf_counter()
    for _ in range(args.timed_batches):
        batch_start = time.perf_counter()
        batch = next(iterator)
        consume(batch)
        latencies.append(time.perf_counter() - batch_start)
        groups += int(batch["peak_mz"].shape[0])
        spectra += int(batch["peak_mz"].shape[0] * batch["peak_mz"].shape[1])
    timed_seconds = time.perf_counter() - timed_start

    local_groups_per_step = datamodule.global_batch_size / args.world_size
    optimizer_steps = groups / local_groups_per_step
    batch_bytes = sum(value.nbytes for value in first_batch.values())
    result = {
        "setup_seconds": setup_seconds,
        "first_batch_seconds": first_batch_seconds,
        "groups_per_second": groups / timed_seconds,
        "spectra_per_second": spectra / timed_seconds,
        "optimizer_steps_per_second": optimizer_steps / timed_seconds,
        "latency_p50_ms": 1_000 * _percentile(latencies, 0.50),
        "latency_p95_ms": 1_000 * _percentile(latencies, 0.95),
        "latency_max_ms": 1_000 * max(latencies),
        "batch_mib": batch_bytes / 2**20,
    }
    print(
        f"backend={args.backend} workers={workers} repeat={repeat} "
        f"rank={args.rank}/{args.world_size} "
        f"microbatch_groups={first_batch['peak_mz'].shape[0]} "
        f"spectra_per_group={first_batch['peak_mz'].shape[1]} "
        f"microbatch_mib={result['batch_mib']:.2f} "
        f"setup_s={setup_seconds:.2f} first_batch_s={first_batch_seconds:.2f} "
        f"groups/s={result['groups_per_second']:.1f} "
        f"spectra/s={result['spectra_per_second']:.1f} "
        f"max_optimizer_steps/s={result['optimizer_steps_per_second']:.3f} "
        f"latency_ms_p50={result['latency_p50_ms']:.1f} "
        f"p95={result['latency_p95_ms']:.1f} max={result['latency_max_ms']:.1f}",
        flush=True,
    )

    if hasattr(iterator, "_shutdown_workers"):
        iterator._shutdown_workers()
    for dataset in datamodule._datasets.values():
        dataset.close()
    del iterator, loader, datamodule
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark grouped GeMS JEPA dataloader throughput."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/1b_grouped_jepa.py"),
    )
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--backend", choices=("numpy", "jax"), default="numpy")
    parser.add_argument("--workers", type=int, nargs="+", default=[32])
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--multiprocessing-context", default="spawn")
    parser.add_argument("--warmup-batches", type=int, default=10)
    parser.add_argument("--timed-batches", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()

    for workers in args.workers:
        results = [
            _run_once(args, workers, repeat) for repeat in range(args.repeats)
        ]
        rates = [result["spectra_per_second"] for result in results]
        print(
            f"SUMMARY backend={args.backend} workers={workers} "
            f"mean_spectra/s={statistics.mean(rates):.1f} "
            f"min={min(rates):.1f} max={max(rates):.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
