import argparse
from contextlib import nullcontext
import sys
import time
from pathlib import Path

import jax
import torch
import torchax
from torchax import interop

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectra_learning.training.api import load_config
from spectra_learning.training.pretrain import seed_all
from spectra_learning.training.torchax_backend import (
    _build_optax_optimizer,
    _build_torchax_model,
    _make_train_step,
    _metric_float,
    _place_batch,
    _replicate_tree,
    _split_trainable_and_static_state,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark TorchAX teacher-JEPA step throughput.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--per-device-batch-size", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measure-steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--profile-dir", default="")
    parser.add_argument("--profile-create-perfetto-trace", action="store_true")
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--block-each-step", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_benchmark_args(args)
    cfg = load_config(args.config)
    seed_all(int(cfg.seed))
    torchax.enable_globally()
    torchax.enable_performance_mode()
    device_count = jax.device_count()
    global_batch_size = args.per_device_batch_size * device_count
    cfg.batch_size = global_batch_size
    cfg.learning_rate = args.learning_rate
    cfg.min_learning_rate = args.learning_rate
    cfg.warmup_steps = 0

    model = _build_torchax_model(cfg)
    weights, static_state = _split_trainable_and_static_state(model)
    trainable_params = sum(param.numel() for param in weights.values())
    static_params = sum(param.numel() for param in static_state.values())
    mesh = jax.make_mesh((device_count,), ("data",))
    with jax.set_mesh(mesh):
        weights = _replicate_tree(weights, mesh)
        static_state = _replicate_tree(static_state, mesh)
        optimizer = _build_optax_optimizer(
            cfg,
            weights,
            total_steps=args.warmup_steps + args.measure_steps,
        )
        opt_state = interop.call_jax(optimizer.init, weights)
        step = _make_train_step(model, optimizer)
        batch = _make_batch(
            global_batch_size,
            int(cfg.num_peaks),
            int(cfg.jepa_num_target_blocks),
        )
        batch = _place_batch(batch, mesh)
        for _ in range(args.warmup_steps):
            metrics, weights, opt_state = step(weights, static_state, opt_state, batch)
            jax.block_until_ready(interop.jax_view_elem(metrics["loss"]))
        profile_dir = Path(args.profile_dir) if args.profile_dir else None
        if profile_dir is not None:
            profile_dir.mkdir(parents=True, exist_ok=True)
        profile_context = (
            jax.profiler.trace(
                profile_dir,
                create_perfetto_trace=args.profile_create_perfetto_trace,
            )
            if profile_dir is not None
            else nullcontext()
        )
        with profile_context:
            start = time.perf_counter()
            for step_idx in range(args.measure_steps):
                annotation = (
                    jax.profiler.StepTraceAnnotation(
                        "teacher_jepa_train",
                        step_num=step_idx,
                        batch_size=global_batch_size,
                    )
                    if profile_dir is not None
                    else nullcontext()
                )
                with annotation:
                    metrics, weights, opt_state = step(
                        weights,
                        static_state,
                        opt_state,
                        batch,
                    )
                    if args.block_each_step:
                        jax.block_until_ready(interop.jax_view_elem(metrics["loss"]))
            jax.block_until_ready(interop.jax_view_elem(metrics["loss"]))
            elapsed = time.perf_counter() - start
        if profile_dir is not None and args.profile_memory:
            memory_profile_path = profile_dir / "device_memory_profile.pb"
            jax.profiler.save_device_memory_profile(memory_profile_path)

    print(f"devices={device_count}")
    print(f"per_device_batch_size={args.per_device_batch_size}")
    print(f"global_batch_size={global_batch_size}")
    print(f"warmup_steps={args.warmup_steps}")
    print(f"measure_steps={args.measure_steps}")
    print(f"elapsed_seconds={elapsed:.6f}")
    print(f"steps_per_second={args.measure_steps / elapsed:.6f}")
    print(f"samples_per_second={global_batch_size * args.measure_steps / elapsed:.6f}")
    print(f"trainable_params={trainable_params}")
    print(f"static_params={static_params}")
    print(f"loss={_metric_float(metrics['loss']):.6f}")
    print(f"masked_prediction_loss={_metric_float(metrics['masked_prediction_loss']):.6f}")
    if "pair_latent_loss" in metrics:
        print(f"pair_latent_loss={_metric_float(metrics['pair_latent_loss']):.6f}")
    if "distogram_loss" in metrics:
        print(f"distogram_loss={_metric_float(metrics['distogram_loss']):.6f}")
    if profile_dir is not None:
        print(f"profile_dir={profile_dir}")
        for artifact_path in sorted(path for path in profile_dir.rglob("*") if path.is_file()):
            print(f"profile_artifact={artifact_path}")


def _make_batch(
    batch_size: int,
    num_peaks: int,
    num_targets: int,
) -> dict[str, torch.Tensor]:
    context_count = max(1, int(num_peaks * 0.65))
    peak_mz = torch.linspace(0.01, 1.0, batch_size * num_peaks).reshape(
        batch_size,
        num_peaks,
    )
    peak_intensity = torch.linspace(1.0, 0.01, batch_size * num_peaks).reshape(
        batch_size,
        num_peaks,
    )
    peak_valid_mask = torch.ones(batch_size, num_peaks, dtype=torch.bool)
    context_mask = torch.zeros(batch_size, num_peaks, dtype=torch.bool)
    context_mask[:, :context_count] = True
    target_masks = torch.zeros(batch_size, num_targets, num_peaks, dtype=torch.bool)
    target_masks[:, :, context_count:] = True
    precursor_mz = torch.linspace(0.2, 1.0, batch_size)
    return {
        "peak_mz": peak_mz,
        "peak_intensity": peak_intensity,
        "peak_valid_mask": peak_valid_mask,
        "context_mask": context_mask,
        "target_masks": target_masks,
        "precursor_mz": precursor_mz,
    }


def _validate_benchmark_args(args: argparse.Namespace) -> None:
    if int(args.per_device_batch_size) <= 0:
        raise ValueError("--per-device-batch-size must be positive")
    if int(args.warmup_steps) < 0:
        raise ValueError("--warmup-steps must be non-negative")
    if int(args.measure_steps) <= 0:
        raise ValueError("--measure-steps must be positive")
    if float(args.learning_rate) <= 0.0:
        raise ValueError("--learning-rate must be positive")


if __name__ == "__main__":
    main()
