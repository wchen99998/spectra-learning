from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectra_learning.config import load_config
from spectra_learning.data.gems.datamodule import GemsDataModule


def _configure(args: argparse.Namespace, workers: int):
    overrides = {
        **json.loads(args.overrides_json),
        "dataloader_num_workers": workers,
        "dataloader_prefetch_factor": args.prefetch_factor,
        "dataloader_persistent_workers": False,
        "dataloader_output_format": "numpy"
        if args.backend in {"numpy", "jax"}
        else "torch",
    }
    if args.artifact_dir is not None:
        overrides["artifact_dir"] = str(args.artifact_dir)
    if args.gradient_accumulation_steps is not None:
        overrides["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if workers > 0:
        overrides["dataloader_multiprocessing_context"] = args.multiprocessing_context
    if args.backend in {"numpy", "jax"}:
        overrides["dataloader_pin_memory"] = False
    return load_config(args.config, overrides)


def _datamodule(config, *, world_size: int, rank: int) -> GemsDataModule:
    task = str(config.get("training_task", "pretrain"))
    if task == "grouped_jepa":
        from spectra_learning.data.gems.grouped import GroupedGemsDataModule

        module_type = GroupedGemsDataModule
    elif task == "ar_spectra":
        from spectra_learning.data.ar_spectra import SpectraARGemsDataModule

        module_type = SpectraARGemsDataModule
    elif task in {"pretrain", "sjepa"}:
        module_type = GemsDataModule
    else:
        raise ValueError(f"No GeMS input pipeline for task {task!r}")
    return module_type(
        config,
        seed=int(config.seed),
        distributed_world_size=world_size,
        distributed_rank=rank,
        distributed_local_rank=0,
    )


def _latency_metrics(
    latencies: list[float] | np.ndarray, prefix: str = ""
) -> dict[str, float]:
    return {
        f"{prefix}latency_p{percentile}_ms": float(np.percentile(latencies, percentile))
        * 1000
        for percentile in (50, 95, 99)
    }


def _run_once(args: argparse.Namespace, workers: int, repeat: int) -> dict[str, Any]:
    config = _configure(args, workers)
    world_size = args.world_size or 1
    rank = (
        args.rank
        if args.rank is not None
        else int(os.environ.get("SPECTRA_GEMS_RANK", 0))
    )
    process_index = rank

    def consume(batch):
        pass

    if args.backend == "jax":
        import jax
        from spectra_learning.training.pretrain_jax import (
            _jax_data_mesh,
            configure_jax_runtime,
            initialize_jax_distributed,
            numpy_batch_to_jax,
            prepare_jax_training_config,
        )

        configure_jax_runtime(config)
        initialize_jax_distributed(config)
        prepare_jax_training_config(config)
        world_size = jax.process_count()
        process_index = jax.process_index()
        if args.world_size is not None and args.world_size != world_size:
            raise ValueError(
                "--world-size must match the initialized JAX process count"
            )
        if args.rank is None:
            rank = int(os.environ.get("SPECTRA_GEMS_RANK", process_index))
        mesh = _jax_data_mesh(config)

        def consume(batch):
            jax.block_until_ready(numpy_batch_to_jax(batch, data_mesh=mesh))

    elif args.backend == "torch-cuda":
        import torch
        from spectra_learning.training.batch import move_batch_to_device

        device = torch.device("cuda")

        def consume(batch):
            move_batch_to_device(batch, device)
            torch.cuda.synchronize(device)

    if not 0 <= rank < world_size:
        raise ValueError("Data rank must be between zero and world_size - 1")
    setup_start = time.perf_counter()
    module = _datamodule(config, world_size=world_size, rank=rank)
    setup_seconds = time.perf_counter() - setup_start
    iterator = None
    try:
        startup_start = time.perf_counter()
        loader = module.train_loader_for_epoch(repeat)
        if len(loader) < args.warmup_batches + args.timed_batches:
            raise ValueError(
                "Not enough full batches; stage more data or reduce benchmark batch counts"
            )
        iterator = iter(loader)
        first = next(iterator)
        consume(first)
        startup_seconds = time.perf_counter() - startup_start
        for _ in range(args.warmup_batches - 1):
            consume(next(iterator))
        latencies = []
        examples = 0
        start = time.perf_counter()
        for batch_index in range(args.timed_batches):
            batch_start = time.perf_counter()
            batch = next(iterator)
            consume(batch)
            latencies.append(time.perf_counter() - batch_start)
            examples += int(next(iter(batch.values())).shape[0])
            if (
                args.report_every_batches
                and (batch_index + 1) % args.report_every_batches == 0
            ):
                print(
                    json.dumps(
                        {
                            "workers": workers,
                            "timed_batches_completed": batch_index + 1,
                            "elapsed_seconds": time.perf_counter() - start,
                        }
                    ),
                    file=sys.stderr,
                    flush=True,
                )
        elapsed = time.perf_counter() - start
        spectra_per_example = (
            int(config.group_jepa_spectra_per_group)
            if config.training_task == "grouped_jepa"
            else 1
        )
        result = {
            "task": config.training_task,
            "backend": args.backend,
            "workers_per_process": workers,
            "repeat": repeat,
            "world_size": world_size,
            "process_index": process_index,
            "data_rank": rank,
            "local_microbatch_size": module.batch_size,
            "setup_seconds": setup_seconds,
            "startup_seconds": startup_seconds,
            "timed_seconds": elapsed,
            "examples_per_second": examples / elapsed,
            "spectra_per_second": examples * spectra_per_example / elapsed,
            "input_optimizer_steps_per_second": examples
            / elapsed
            / (module.global_batch_size / world_size),
            "batch_bytes": sum(
                value.nbytes
                if isinstance(value, np.ndarray)
                else value.numel() * value.element_size()
                for value in first.values()
            ),
            "parent_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024,
            "batch_latencies_seconds": latencies,
            **_latency_metrics(latencies),
        }
        if args.backend == "jax" and world_size > 1:
            from jax.experimental import multihost_utils

            per_host = np.asarray(
                multihost_utils.process_allgather(
                    np.asarray(latencies, dtype=np.float32)
                )
            ).reshape(world_size, -1)
            result.update(_latency_metrics(per_host.max(axis=0), "slowest_host_"))
            host_elapsed = np.asarray(
                multihost_utils.process_allgather(np.asarray(elapsed, dtype=np.float32))
            )
            result["pod_input_optimizer_steps_per_second"] = (
                examples
                / float(host_elapsed.max())
                / (module.global_batch_size / world_size)
            )
        print(json.dumps(result, sort_keys=True), flush=True)
        return result
    finally:
        if iterator is not None:
            shutdown = getattr(iterator, "_shutdown_workers", None)
            if shutdown is not None:
                shutdown()
        for dataset in module._datasets.values():
            dataset.close()
        gc.collect()


def _run_training_once(
    args: argparse.Namespace, workers: int, repeat: int
) -> dict[str, Any]:
    # Replay is deliberately confined to this diagnostic, never a training option.
    import tempfile
    from unittest.mock import patch

    import jax
    import jax.numpy as jnp

    from train import _train
    from spectra_learning.training import pretrain_jax

    config = _configure(args, workers)
    if config.device_backend != "jax":
        raise ValueError("Training benchmarks require a JAX config")
    config.training_max_steps = args.warmup_steps + args.timed_steps
    config.throughput_warmup_steps = args.warmup_steps
    config.num_epochs = 1
    config.enable_wandb = False
    config.checkpoint_every_steps = 0
    config.val_every_n_steps = -1
    config.msg_probe_every_n_steps = -1
    config.msg_probe_at_final_step = False
    config.jax_timing_barriers = False
    config.max_duration_hours = None
    pretrain_jax.configure_jax_runtime(config)
    pretrain_jax.initialize_jax_distributed(config)
    if args.world_size is not None and args.world_size != jax.process_count():
        raise ValueError("--world-size must match the initialized JAX process count")
    if args.rank is not None:
        os.environ["SPECTRA_GEMS_RANK"] = str(args.rank)

    class BenchmarkLoop(pretrain_jax._JaxTrainingLoop):
        cached_batch = None
        cached_stage = None

        def _save_checkpoint(self, step):
            pass

        def _next_accumulated_batch(self, loader_iter):
            if args.mode == "train-real":
                return super()._next_accumulated_batch(loader_iter)
            if self.cached_batch is None or self.cached_stage != self.mask_stage_index:
                self.cached_batch = super()._next_accumulated_batch(loader_iter)
                self.cached_stage = self.mask_stage_index
                jax.block_until_ready(self.cached_batch)
                pretrain_jax._shutdown_torch_loader_iterator(loader_iter)
            if self.cached_batch is not None and self.use_sjepa:
                return {
                    **self.cached_batch,
                    "sjepa_sigreg_step": jnp.full_like(
                        self.cached_batch["sjepa_sigreg_step"], self.global_step
                    ),
                }
            return self.cached_batch

    with tempfile.TemporaryDirectory(prefix="spectra-input-benchmark-") as workdir:
        with patch.object(pretrain_jax, "_JaxTrainingLoop", BenchmarkLoop):
            result = _train(config, workdir)
    measured_steps = int(result["run/measured_steps"])
    if measured_steps != args.timed_steps:
        raise ValueError(
            f"Expected {args.timed_steps} measured steps, got {measured_steps}; stage more data"
        )
    elapsed = float(result["run/measured_elapsed_seconds"])
    if jax.process_count() > 1:
        from jax.experimental import multihost_utils

        elapsed = float(
            np.asarray(
                multihost_utils.process_allgather(np.asarray(elapsed, dtype=np.float32))
            ).max()
        )
    result.update(
        {
            "mode": args.mode,
            "task": config.training_task,
            "repeat": repeat,
            "workers_per_process": workers,
            "process_index": jax.process_index(),
            "pod_measured_steps_per_second": measured_steps / elapsed,
        }
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark staged GeMS inputs for MAE, S-JEPA, grouped JEPA, or AR."
    )
    parser.add_argument(
        "--mode", choices=("loader", "train-real", "train-cached"), default="loader"
    )
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--timed-steps", type=int, default=100)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overrides-json", default="{}")
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument(
        "--backend", choices=("numpy", "jax", "torch", "torch-cuda"), default="numpy"
    )
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--multiprocessing-context", choices=("spawn", "forkserver"), default="spawn"
    )
    parser.add_argument("--world-size", type=int)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--warmup-batches", type=int, default=20)
    parser.add_argument("--timed-batches", type=int, default=250)
    parser.add_argument("--report-every-batches", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if (
        min(args.warmup_batches, args.timed_batches, args.repeats) < 1
        or min(args.workers) < 0
    ):
        parser.error(
            "Batch counts and repeats must be positive; workers must be nonnegative"
        )
    if args.mode != "loader" and (
        args.backend != "jax" or min(args.warmup_steps, args.timed_steps) < 1
    ):
        parser.error(
            "Training modes require --backend jax and positive warmup/timed steps"
        )
    run_once = _run_once if args.mode == "loader" else _run_training_once
    for workers in args.workers:
        for repeat in range(args.repeats):
            run_once(args, workers, repeat)


if __name__ == "__main__":
    main()
