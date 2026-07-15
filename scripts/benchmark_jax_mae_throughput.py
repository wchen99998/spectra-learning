from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectra_learning.training.jax_runtime_flags import configure_jax_tpu_xla_flags


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark synchronized JAX MAE optimizer-step throughput on real GeMS data."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--stage", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument(
        "--projection-kernel",
        choices=("xla", "pallas"),
        default=None,
    )
    parser.add_argument(
        "--encoder-projection-kernel",
        choices=("xla", "pallas"),
        default=None,
    )
    parser.add_argument(
        "--predictor-projection-kernel",
        choices=("xla", "pallas"),
        default=None,
    )
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--timed-steps", type=int, default=8)
    parser.add_argument("--mask-seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_jax_tpu_xla_flags()

    import jax
    import torch
    from flax import nnx

    from spectra_learning.config import load_config
    from spectra_learning.data.gems.datamodule import GemsDataModule
    from spectra_learning.data.gems.mask_schedule import jepa_mask_stages
    from spectra_learning.models.factory_jax import build_model_from_config
    from spectra_learning.models.fastmixer_capacity import (
        pairmixer_stage_projection_kernels,
    )
    from spectra_learning.models.settings import PeakSetJEPASettings
    from spectra_learning.training.pretrain_jax import (
        _stack_micro_batches,
        init_pure_optax_train_state,
        make_pure_accumulated_train_step,
        numpy_batch_to_jax,
        prepare_jax_training_config,
    )

    overrides = {
        "artifact_dir": str(args.artifact_dir),
        "batch_size": args.global_batch_size,
        "jax_mesh_devices": "1",
        "dataloader_num_workers": args.workers,
        "dataloader_persistent_workers": False,
        "dataloader_pin_memory": False,
        "dataloader_output_format": "numpy",
        "enable_wandb": False,
        "jax_compilation_cache_dir": str(
            args.artifact_dir / "jax_compilation_cache"
        ),
        "jax_persistent_cache_min_compile_time_secs": 0.0,
        "jax_persistent_cache_min_entry_size_bytes": 0,
        "training_max_steps": args.warmup_steps + args.timed_steps,
        "num_epochs": 1,
        "checkpoint_every_steps": 0,
        "val_every_n_steps": -1,
        "msg_probe_every_n_steps": -1,
        "msg_probe_at_final_step": False,
    }
    if args.gradient_accumulation_steps is not None:
        overrides["gradient_accumulation_steps"] = (
            args.gradient_accumulation_steps
        )
    config = load_config(args.config, overrides)
    stage = jepa_mask_stages(config)[args.stage - 1]
    config.jepa_context_fraction = stage.context_fraction
    config.jepa_target_fraction = stage.target_fraction
    if args.gradient_accumulation_steps is None:
        config.gradient_accumulation_steps = stage.gradient_accumulation_steps
    encoder_kernel, predictor_kernel = pairmixer_stage_projection_kernels(config)[
        args.stage - 1
    ]
    if args.projection_kernel is not None:
        encoder_kernel = args.projection_kernel
        predictor_kernel = args.projection_kernel
    if args.encoder_projection_kernel is not None:
        encoder_kernel = args.encoder_projection_kernel
    if args.predictor_projection_kernel is not None:
        predictor_kernel = args.predictor_projection_kernel
    config.pairmixer_encoder_projection_kernel = encoder_kernel
    config.pairmixer_predictor_projection_kernel = predictor_kernel
    prepare_jax_training_config(config)

    torch.manual_seed(args.mask_seed)
    datamodule = GemsDataModule(config, seed=int(config.seed))
    settings = PeakSetJEPASettings.from_config(config)

    build_start = time.perf_counter()
    model = build_model_from_config(config)
    params = nnx.state(model, nnx.Param)
    parameter_count = sum(int(value.size) for value in jax.tree.leaves(params))
    build_seconds = time.perf_counter() - build_start

    graphdef, trainable_params, static_state, opt_state, optimizer = (
        init_pure_optax_train_state(
            config,
            model,
            total_steps=args.warmup_steps + args.timed_steps,
        )
    )
    train_step = make_pure_accumulated_train_step(
        graphdef,
        optimizer,
        sharded=False,
        metric_reduction="mean",
    )

    loader = datamodule.train_loader_for_epoch(0)
    loader_iterator = iter(loader)

    def next_batch():
        micro_batches = [
            next(loader_iterator)
            for _ in range(int(config.gradient_accumulation_steps))
        ]
        return (
            numpy_batch_to_jax(
                _stack_micro_batches(micro_batches),
                batch_axis=1,
            ),
            micro_batches,
        )

    elapsed = []
    losses = []
    valid_counts = []
    context_counts = []
    target_counts = []
    compile_and_first_step_seconds = 0.0
    total_benchmark_steps = args.warmup_steps + args.timed_steps
    for step_index in range(total_benchmark_steps):
        batch, micro_batches = next_batch()
        for micro_batch in micro_batches:
            valid_counts.extend(micro_batch["peak_valid_mask"].sum(axis=1).tolist())
            context_counts.extend(micro_batch["context_mask"].sum(axis=1).tolist())
            target_counts.extend(
                micro_batch["target_masks"].sum(axis=(1, 2)).tolist()
            )
        step_start = time.perf_counter()
        trainable_params, opt_state, metrics = train_step(
            trainable_params,
            static_state,
            opt_state,
            batch,
        )
        jax.block_until_ready((trainable_params, opt_state, metrics))
        step_seconds = time.perf_counter() - step_start
        if step_index == 0:
            compile_and_first_step_seconds = step_seconds
        if step_index >= args.warmup_steps:
            elapsed.append(step_seconds)
            losses.append(float(metrics["loss"]))
        phase = "warmup" if step_index < args.warmup_steps else "timed"
        print(
            f"{phase} step={step_index + 1} seconds={step_seconds:.6f} "
            f"loss={float(metrics['loss']):.6f}",
            flush=True,
        )

    samples = args.timed_steps * datamodule.global_batch_size
    result = {
        "config": str(args.config),
        "artifact_dir": str(args.artifact_dir),
        "device": str(jax.devices()[0]),
        "jax_version": jax.__version__,
        "stage": args.stage,
        "context_fraction": stage.context_fraction,
        "target_fraction": stage.target_fraction,
        "encoder_projection_kernel": (
            settings.pairmixer_encoder_projection_kernel
        ),
        "predictor_projection_kernel": (
            settings.pairmixer_predictor_projection_kernel
        ),
        "encoder_tokens": settings.pairmixer_fast_encoder_max_visible_tokens,
        "predictor_tokens": settings.pairmixer_fast_max_visible_tokens,
        "parameter_count": parameter_count,
        "global_batch_size": datamodule.global_batch_size,
        "microbatch_size": datamodule.batch_size,
        "gradient_accumulation_steps": int(config.gradient_accumulation_steps),
        "warmup_steps": args.warmup_steps,
        "timed_steps": args.timed_steps,
        "samples": samples,
        "build_seconds": build_seconds,
        "compile_and_first_step_seconds": compile_and_first_step_seconds,
        "total_seconds": sum(elapsed),
        "mean_step_seconds": statistics.mean(elapsed),
        "median_step_seconds": statistics.median(elapsed),
        "min_step_seconds": min(elapsed),
        "max_step_seconds": max(elapsed),
        "steps_per_second": args.timed_steps / sum(elapsed),
        "samples_per_second": samples / sum(elapsed),
        "mean_loss": statistics.mean(losses),
        "mean_valid_peaks": statistics.mean(valid_counts),
        "mean_context_peaks": statistics.mean(context_counts),
        "mean_target_peaks": statistics.mean(target_counts),
    }
    print("BENCHMARK_RESULT " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
