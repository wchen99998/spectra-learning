from __future__ import annotations

import argparse
import gc
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectra_learning.config import load_config
from spectra_learning.data.gems.datamodule import GemsDataModule


def _config_get(config: Any, key: str, default: Any) -> Any:
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _configure(args: argparse.Namespace, workers: int):
    overrides = {
        "dataloader_num_workers": workers,
        "dataloader_prefetch_factor": args.prefetch_factor,
        "dataloader_persistent_workers": workers > 0,
        "peak_filtering": args.peak_filtering,
        "grouped_peak_shoulder_da": args.grouped_peak_shoulder_da,
        "grouped_peak_isotope_charges": tuple(args.grouped_peak_isotope_charges),
        "dataloader_output_format": (
            "numpy" if args.backend in {"jax", "numpy"} else "torch"
        ),
    }
    if workers > 0:
        overrides["dataloader_multiprocessing_context"] = args.multiprocessing_context
    if args.backend == "jax":
        overrides["dataloader_pin_memory"] = False
        overrides["dataloader_persistent_workers"] = False
    return load_config(args.config, overrides)


def _consume_torch(batch: dict[str, Any]) -> None:
    _ = batch["peak_mz"].shape


def _consume_numpy(batch: dict[str, Any]) -> None:
    _ = batch["peak_mz"].shape


def _run_once(args: argparse.Namespace, workers: int, repeat: int) -> float:
    config = _configure(args, workers)
    if args.backend == "jax":
        import jax

        from spectra_learning.training.pretrain_jax import numpy_batch_to_jax

    datamodule = GemsDataModule(config, seed=int(_config_get(config, "seed", 0)) + repeat)
    loader = datamodule.train_loader_for_epoch(repeat)
    iterator = iter(loader)
    if args.backend == "jax":
        consume = lambda batch: jax.block_until_ready(numpy_batch_to_jax(batch))
    elif args.backend == "torch-cuda":
        import torch

        from spectra_learning.training.batch import move_batch_to_device

        device = torch.device("cuda")

        def consume(batch: dict[str, Any]) -> None:
            move_batch_to_device(batch, device)
            torch.cuda.synchronize(device)

    else:
        consume = {
            "numpy": _consume_numpy,
            "torch": _consume_torch,
        }[args.backend]
    for _ in range(args.warmup_batches):
        consume(next(iterator))
    samples = 0
    start = time.perf_counter()
    for _ in range(args.timed_batches):
        batch = next(iterator)
        consume(batch)
        samples += int(batch["peak_mz"].shape[0])
    seconds = time.perf_counter() - start
    samples_per_second = samples / seconds
    print(
        f"backend={args.backend} workers={workers} repeat={repeat} "
        f"batch_size={datamodule.batch_size} seconds={seconds:.3f} "
        f"samples_per_s={samples_per_second:.1f}",
        flush=True,
    )
    if hasattr(iterator, "_shutdown_workers"):
        iterator._shutdown_workers()
    del iterator, loader, datamodule
    gc.collect()
    return samples_per_second


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark GeMS dataloader throughput.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--backend",
        choices=("torch", "torch-cuda", "numpy", "jax"),
        default="torch",
    )
    parser.add_argument("--peak-filtering", default="grouped")
    parser.add_argument("--grouped-peak-shoulder-da", type=float, default=0.05)
    parser.add_argument(
        "--grouped-peak-isotope-charges",
        type=int,
        nargs="+",
        default=[1, 2, 3],
    )
    parser.add_argument("--workers", type=int, nargs="+", default=[8])
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--multiprocessing-context", default="forkserver")
    parser.add_argument("--warmup-batches", type=int, default=20)
    parser.add_argument("--timed-batches", type=int, default=250)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    for workers in args.workers:
        values = [_run_once(args, workers, repeat) for repeat in range(args.repeats)]
        stdev = statistics.stdev(values) if len(values) > 1 else 0.0
        print(
            f"SUMMARY backend={args.backend} workers={workers} "
            f"mean_samples_per_s={statistics.mean(values):.1f} "
            f"stdev={stdev:.1f} min={min(values):.1f} max={max(values):.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
