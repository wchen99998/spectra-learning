from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark full-model grouped JEPA JAX step variants."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/1b_grouped_jepa.py"),
    )
    parser.add_argument(
        "--mode",
        choices=("separate-ema", "fused-ema", "lookahead"),
        required=True,
    )
    parser.add_argument("--groups-per-device", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--timed-steps", type=int, default=10)
    parser.add_argument("--tiny-model", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _synthetic_batch(
    *,
    accumulation_steps: int,
    groups: int,
    spectra_per_group: int,
    num_peaks: int,
    offset: float,
) -> dict[str, np.ndarray]:
    shape = (accumulation_steps, groups, spectra_per_group, num_peaks)
    peak_index = np.arange(num_peaks, dtype=np.float32)
    peak_mz = np.broadcast_to(
        (peak_index + 1.0 + offset) / 2_000.0,
        shape,
    ).copy()
    peak_intensity = np.broadcast_to(
        ((peak_index % 13.0) + 1.0 + offset) / 14.0,
        shape,
    ).copy()
    metadata_shape = (accumulation_steps, groups, spectra_per_group)
    return {
        "peak_mz": peak_mz,
        "peak_intensity": peak_intensity,
        "peak_valid_mask": np.ones(shape, dtype=np.bool_),
        "precursor_mz": np.full(metadata_shape, 0.5 + offset / 100.0, np.float32),
        "collision_energy": np.full(metadata_shape, 0.25, np.float32),
        "charge": np.full(metadata_shape, 2.0, np.float32),
    }


def _memory_analysis(executable: Any) -> dict[str, int]:
    analysis = executable.memory_analysis()
    if analysis is None:
        return {}
    values = {
        "temp_size_in_bytes": analysis.temp_size_in_bytes,
        "argument_size_in_bytes": analysis.argument_size_in_bytes,
        "output_size_in_bytes": analysis.output_size_in_bytes,
        "alias_size_in_bytes": analysis.alias_size_in_bytes,
    }
    values["total_size_in_bytes"] = (
        values["temp_size_in_bytes"]
        + values["argument_size_in_bytes"]
        + values["output_size_in_bytes"]
        - values["alias_size_in_bytes"]
    )
    return values


def _cost_analysis(executable: Any) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in executable.cost_analysis().items()
        if key in {"flops", "transcendentals", "bytes accessed"}
    }


def _tree_bytes(tree: Any) -> int:
    return sum(value.size * value.dtype.itemsize for value in _jax().tree.leaves(tree))


def _jax():
    import jax

    return jax


def main() -> None:
    args = parse_args()

    import jax
    from flax import nnx

    from spectra_learning.config import load_config
    from spectra_learning.models.grouped_jepa_jax import (
        GROUP_JEPA_TEACHER_TARGET_AGE_KEY,
        GROUP_JEPA_TEACHER_TARGET_KEY,
        GroupedSpectrumJEPAJax,
    )
    from spectra_learning.models.settings import PeakSetJEPASettings
    from spectra_learning.training.pretrain_jax import (
        _jax_data_mesh,
        _replicate_tree_on_data_mesh,
        configure_jax_runtime,
        init_pure_optax_train_state,
        make_grouped_jepa_teacher_target_step,
        make_pure_accumulated_train_step,
        numpy_batch_to_jax,
        update_grouped_jepa_ema_state,
    )

    total_steps = args.warmup_steps + args.timed_steps
    overrides = {
        "jax_mesh_devices": "all",
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "training_max_steps": total_steps,
        "jax_compilation_cache_dir": "/tmp/grouped-jepa-jax-cache",
        "jax_persistent_cache_min_compile_time_secs": 0.0,
        "jax_persistent_cache_min_entry_size_bytes": 0,
    }
    if args.tiny_model:
        overrides["optimizer"] = "adamw"
    config = load_config(args.config, overrides)
    configure_jax_runtime(config)
    mesh = _jax_data_mesh(config)
    global_microbatch_groups = args.groups_per_device * mesh.size

    build_start = time.perf_counter()
    settings = PeakSetJEPASettings.from_config(config)
    if args.tiny_model:
        settings = PeakSetJEPASettings.create(
            settings,
            model_dim=24,
            encoder_num_layers=1,
            encoder_num_heads=4,
            feature_mlp_hidden_dim=32,
            encoder_fourier_mlp_hidden_dim=32,
            encoder_fourier_num_freqs=4,
            num_peaks=7,
            predictor_dim=24,
            masked_latent_predictor_num_layers=1,
            masked_latent_predictor_num_heads=4,
            pairmixer_pair_dim=8,
            pairmixer_pair_feature_hidden_dim=8,
            encoder_use_position_embedding=False,
            target_projector_dim=-1,
            autocast_dtype="none",
        )
    model = GroupedSpectrumJEPAJax(
        settings,
        teacher_spectra_per_group=int(config.group_jepa_teacher_spectra_per_group),
        ema_momentum=float(config.group_jepa_ema_momentum),
        rngs=nnx.Rngs(int(config.seed)),
    )
    graphdef, params, static_state, opt_state, optimizer = (
        init_pure_optax_train_state(
            config,
            model,
            total_steps=total_steps,
        )
    )
    params = _replicate_tree_on_data_mesh(params, mesh)
    static_state = _replicate_tree_on_data_mesh(static_state, mesh)
    opt_state = _replicate_tree_on_data_mesh(opt_state, mesh)
    jax.block_until_ready((params, static_state, opt_state))
    build_seconds = time.perf_counter() - build_start

    host_batches = [
        _synthetic_batch(
            accumulation_steps=args.gradient_accumulation_steps,
            groups=global_microbatch_groups,
            spectra_per_group=int(config.group_jepa_spectra_per_group),
            num_peaks=settings.num_peaks,
            offset=offset,
        )
        for offset in (0.0, 1.0)
    ]
    batches = [
        numpy_batch_to_jax(batch, data_mesh=mesh, batch_axis=1)
        for batch in host_batches
    ]

    ema_momentum = float(config.group_jepa_ema_momentum)
    compile_start = time.perf_counter()
    executable_memory: dict[str, dict[str, int]] = {}
    executable_cost: dict[str, dict[str, float]] = {}
    target_executable = None
    ema_executable = None
    if args.mode == "separate-ema":
        step = make_pure_accumulated_train_step(
            graphdef,
            optimizer,
            sharded=mesh.size > 1,
            data_mesh=mesh,
        )
        executable = step.lower(
            params,
            static_state,
            opt_state,
            batches[0],
        ).compile()
        ema_executable = update_grouped_jepa_ema_state.lower(
            params,
            static_state,
            ema_momentum,
        ).compile()
        executable_memory["train"] = _memory_analysis(executable)
        executable_memory["ema"] = _memory_analysis(ema_executable)
        executable_cost["train"] = _cost_analysis(executable)
        executable_cost["ema"] = _cost_analysis(ema_executable)
        teacher_target = None
    elif args.mode == "fused-ema":
        step = make_pure_accumulated_train_step(
            graphdef,
            optimizer,
            sharded=mesh.size > 1,
            data_mesh=mesh,
            group_jepa_ema_momentum=ema_momentum,
        )
        executable = step.lower(
            params,
            static_state,
            opt_state,
            batches[0],
        ).compile()
        executable_memory["train"] = _memory_analysis(executable)
        executable_cost["train"] = _cost_analysis(executable)
        teacher_target = None
    else:
        target_step = make_grouped_jepa_teacher_target_step(
            graphdef,
            sharded=mesh.size > 1,
            data_mesh=mesh,
        )
        target_executable = target_step.lower(
            params,
            static_state,
            batches[0],
        ).compile()
        teacher_target = target_executable(params, static_state, batches[0])
        current = {
            **batches[0],
            GROUP_JEPA_TEACHER_TARGET_KEY: teacher_target,
            GROUP_JEPA_TEACHER_TARGET_AGE_KEY: jax.numpy.zeros(
                teacher_target.shape[:2],
                dtype=jax.numpy.float32,
            ),
        }
        step = make_pure_accumulated_train_step(
            graphdef,
            optimizer,
            sharded=mesh.size > 1,
            data_mesh=mesh,
            group_jepa_ema_momentum=ema_momentum,
            group_jepa_lookahead=True,
        )
        executable = step.lower(
            params,
            static_state,
            opt_state,
            current,
            batches[1],
        ).compile()
        executable_memory["train"] = _memory_analysis(executable)
        executable_memory["initial_target"] = _memory_analysis(target_executable)
        executable_cost["train"] = _cost_analysis(executable)
        executable_cost["initial_target"] = _cost_analysis(target_executable)
    compile_seconds = time.perf_counter() - compile_start

    elapsed: list[float] = []
    losses: list[float] = []
    current_index = 0
    for step_index in range(total_steps):
        current_batch = batches[current_index]
        next_index = 1 - current_index
        step_start = time.perf_counter()
        if args.mode == "separate-ema":
            params, opt_state, metrics = executable(
                params,
                static_state,
                opt_state,
                current_batch,
            )
            static_state = ema_executable(
                params,
                static_state,
                ema_momentum,
            )
        elif args.mode == "fused-ema":
            params, static_state, opt_state, metrics = executable(
                params,
                static_state,
                opt_state,
                current_batch,
            )
        else:
            current_batch = {
                **current_batch,
                GROUP_JEPA_TEACHER_TARGET_KEY: teacher_target,
                GROUP_JEPA_TEACHER_TARGET_AGE_KEY: jax.numpy.full(
                    teacher_target.shape[:2],
                    int(step_index > 0),
                    dtype=jax.numpy.float32,
                ),
            }
            (
                params,
                static_state,
                opt_state,
                metrics,
                teacher_target,
            ) = executable(
                params,
                static_state,
                opt_state,
                current_batch,
                batches[next_index],
            )
        jax.block_until_ready(
            (params, static_state, opt_state, metrics, teacher_target)
        )
        step_seconds = time.perf_counter() - step_start
        loss = float(metrics["loss"])
        phase = "warmup" if step_index < args.warmup_steps else "timed"
        print(
            f"{phase} step={step_index + 1} seconds={step_seconds:.6f} "
            f"loss={loss:.8f}",
            flush=True,
        )
        if step_index >= args.warmup_steps:
            elapsed.append(step_seconds)
            losses.append(loss)
        current_index = next_index

    global_groups_per_step = (
        global_microbatch_groups * args.gradient_accumulation_steps
    )
    result = {
        "mode": args.mode,
        "config": str(args.config),
        "jax_version": jax.__version__,
        "devices": [str(device) for device in jax.devices()],
        "device_count": mesh.size,
        "groups_per_device_microbatch": args.groups_per_device,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "global_groups_per_step": global_groups_per_step,
        "global_spectra_per_step": (
            global_groups_per_step * int(config.group_jepa_spectra_per_group)
        ),
        "parameter_count": sum(value.size for value in jax.tree.leaves(params)),
        "trainable_state_bytes": _tree_bytes(params),
        "static_state_bytes": _tree_bytes(static_state),
        "optimizer_state_bytes": _tree_bytes(opt_state),
        "build_seconds": build_seconds,
        "compile_seconds": compile_seconds,
        "executable_memory": executable_memory,
        "executable_cost": executable_cost,
        "warmup_steps": args.warmup_steps,
        "timed_steps": args.timed_steps,
        "mean_step_seconds": statistics.mean(elapsed),
        "median_step_seconds": statistics.median(elapsed),
        "min_step_seconds": min(elapsed),
        "max_step_seconds": max(elapsed),
        "steps_per_second": args.timed_steps / sum(elapsed),
        "spectra_per_second": (
            args.timed_steps
            * global_groups_per_step
            * int(config.group_jepa_spectra_per_group)
            / sum(elapsed)
        ),
        "first_timed_loss": losses[0],
        "last_timed_loss": losses[-1],
    }
    print("RESULT " + json.dumps(result, sort_keys=True), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
