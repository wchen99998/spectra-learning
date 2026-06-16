from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from jax.experimental.serialize_executable import serialize
from jax.experimental.topologies import get_topology_desc
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from spectra_learning.config import load_config
from spectra_learning.models.factory_jax import build_model_from_config
from spectra_learning.training.pretrain_jax import (
    JAX_DATA_AXIS,
    _config_get,
    _context_encoder_pack_choices,
    _jax_train_step_compile_variants,
    build_jax_optax_transform,
    configure_jax_runtime,
    init_pure_optax_train_state,
    make_pure_accumulated_train_step,
)
from spectra_learning.training.jax_runtime_flags import jax_tpu_xla_flags_string


@dataclass(frozen=True)
class TpuCompileTarget:
    name: str
    topology_name: str
    chips_per_host_bounds: tuple[int, int, int]
    devices_per_slice: int
    vm_count: int
    num_slices: int = 1
    platform: str = "tpu"
    chip_config_name: str = "default"
    wrap: tuple[bool, bool, bool] = (False, False, False)

    @property
    def process_count(self) -> int:
        return self.vm_count * self.num_slices


TPU_COMPILE_TARGETS: dict[str, TpuCompileTarget] = {
    "ct6e-standard-8t": TpuCompileTarget(
        name="ct6e-standard-8t",
        topology_name="v6e:2x4",
        chips_per_host_bounds=(2, 4, 1),
        devices_per_slice=8,
        vm_count=1,
    ),
    "v6e-8": TpuCompileTarget(
        name="ct6e-standard-8t",
        topology_name="v6e:2x4",
        chips_per_host_bounds=(2, 4, 1),
        devices_per_slice=8,
        vm_count=1,
    ),
    "ct6e-standard-4t-v6e-8": TpuCompileTarget(
        name="ct6e-standard-4t-v6e-8",
        topology_name="v6e:2x4",
        chips_per_host_bounds=(2, 2, 1),
        devices_per_slice=8,
        vm_count=2,
    ),
    "v6e-8-multihost": TpuCompileTarget(
        name="ct6e-standard-4t-v6e-8",
        topology_name="v6e:2x4",
        chips_per_host_bounds=(2, 2, 1),
        devices_per_slice=8,
        vm_count=2,
    ),
}


@dataclass(frozen=True)
class CompileVariant:
    name: str
    pack_tokens: int | None
    full_context_fallback: bool


@dataclass(frozen=True)
class CompileSummary:
    target: dict[str, Any]
    variant: str
    abstract_state_seconds: float
    lower_seconds: float
    compile_seconds: float
    memory_analysis: dict[str, int] | None
    cost_analysis: dict[str, float] | None
    executable_file: str


def resolve_tpu_compile_target(name: str, *, num_slices: int = 1) -> TpuCompileTarget:
    target = TPU_COMPILE_TARGETS[name.lower()]
    return TpuCompileTarget(
        name=target.name,
        topology_name=target.topology_name,
        chips_per_host_bounds=target.chips_per_host_bounds,
        devices_per_slice=target.devices_per_slice,
        vm_count=target.vm_count,
        num_slices=num_slices,
        platform=target.platform,
        chip_config_name=target.chip_config_name,
        wrap=target.wrap,
    )


def build_tpu_compile_mesh(target: TpuCompileTarget) -> Mesh:
    topology = get_topology_desc(
        platform=target.platform,
        topology_name=target.topology_name,
        chip_config_name=target.chip_config_name,
        chips_per_host_bounds=target.chips_per_host_bounds,
        num_slices=target.num_slices,
        wrap=target.wrap,
    )
    return Mesh(np.asarray(topology.devices), (JAX_DATA_AXIS,))


def target_microbatch_size(config: Any, target: TpuCompileTarget) -> int:
    global_batch_size = int(_config_get(config, "batch_size", 512))
    grad_accum_steps = int(_config_get(config, "gradient_accumulation_steps", 1))
    denominator = target.process_count * grad_accum_steps
    assert global_batch_size % denominator == 0
    return global_batch_size // denominator


def compile_variants(config: Any, selector: str) -> tuple[CompileVariant, ...]:
    pack_choices = _context_encoder_pack_choices(
        config,
        int(_config_get(config, "mae_context_encoder_pack_tokens", 0)),
    )
    variants = _jax_train_step_compile_variants(
        pack_choices,
        selector=selector,
        has_default_train_step=True,
        has_full_fallback=bool(pack_choices),
    )
    return tuple(CompileVariant(*variant) for variant in variants)


def compile_jax_train_steps_for_tpu(
    config: Any,
    *,
    target: TpuCompileTarget,
    variant_selector: str = "default",
    output_dir: str | Path | None = None,
    compiler_options: dict[str, str] | None = None,
) -> list[CompileSummary]:
    configure_jax_runtime(config)
    compile_options = parse_compiler_options(jax_tpu_xla_flags_string())
    if compiler_options:
        compile_options.update(compiler_options)
    data_mesh = build_tpu_compile_mesh(target)
    variants = compile_variants(config, variant_selector)
    summaries: list[CompileSummary] = []
    output_path = None if output_dir is None else Path(output_dir)
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    for variant in variants:
        print(
            f"Preparing abstract train state for {variant.name}...",
            file=sys.stderr,
            flush=True,
        )
        abstract_start = time.perf_counter()
        graphdef, trainable_params, static_state, opt_state, optimizer = (
            _abstract_pure_train_state(
                config,
                variant=variant,
            )
        )
        abstract_state_seconds = time.perf_counter() - abstract_start
        trainable_params = _with_sharding(trainable_params, NamedSharding(data_mesh, P()))
        static_state = _with_sharding(static_state, NamedSharding(data_mesh, P()))
        opt_state = _with_sharding(opt_state, NamedSharding(data_mesh, P()))
        batch = abstract_train_batch(config, target=target, data_mesh=data_mesh)
        train_step = make_pure_accumulated_train_step(
            graphdef,
            optimizer,
            sharded=True,
            data_mesh=data_mesh,
        )
        print(f"Lowering {variant.name}...", file=sys.stderr, flush=True)
        lower_start = time.perf_counter()
        with jax.set_mesh(data_mesh):
            lowered = train_step.lower(
                trainable_params,
                static_state,
                opt_state,
                batch,
            )
        lower_seconds = time.perf_counter() - lower_start
        print(f"Compiling {variant.name}...", file=sys.stderr, flush=True)
        compile_start = time.perf_counter()
        compiled = lowered.compile(compiler_options=compile_options)
        compile_seconds = time.perf_counter() - compile_start
        print(
            f"Compiled {variant.name}: abstract={abstract_state_seconds:.2f}s "
            f"lower={lower_seconds:.2f}s compile={compile_seconds:.2f}s",
            file=sys.stderr,
            flush=True,
        )
        executable_file = ""
        if output_path is not None:
            executable_file = str(output_path / f"{variant.name}.compiled")
            serialized, _, _ = serialize(compiled)
            Path(executable_file).write_bytes(serialized)
        summaries.append(
            CompileSummary(
                target=asdict(target),
                variant=variant.name,
                abstract_state_seconds=abstract_state_seconds,
                lower_seconds=lower_seconds,
                compile_seconds=compile_seconds,
                memory_analysis=_memory_analysis_dict(compiled.memory_analysis()),
                cost_analysis=_cost_analysis_summary(compiled.cost_analysis()),
                executable_file=executable_file,
            )
        )
    return summaries


def abstract_train_batch(
    config: Any,
    *,
    target: TpuCompileTarget,
    data_mesh: Mesh,
) -> dict[str, jax.ShapeDtypeStruct]:
    grad_accum_steps = int(_config_get(config, "gradient_accumulation_steps", 1))
    microbatch_size = target_microbatch_size(config, target)
    num_peaks = int(_config_get(config, "num_peaks", 60))
    num_target_blocks = int(_config_get(config, "jepa_num_target_blocks", 2))
    batch_shape = (grad_accum_steps, microbatch_size, num_peaks)
    batch_sharding = NamedSharding(data_mesh, P(None, JAX_DATA_AXIS))
    return {
        "peak_mz": jax.ShapeDtypeStruct(
            batch_shape,
            jnp.float32,
            sharding=batch_sharding,
        ),
        "peak_intensity": jax.ShapeDtypeStruct(
            batch_shape,
            jnp.float32,
            sharding=batch_sharding,
        ),
        "peak_valid_mask": jax.ShapeDtypeStruct(
            batch_shape,
            jnp.bool_,
            sharding=batch_sharding,
        ),
        "precursor_mz": jax.ShapeDtypeStruct(
            batch_shape[:2],
            jnp.float32,
            sharding=batch_sharding,
        ),
        "context_mask": jax.ShapeDtypeStruct(
            batch_shape,
            jnp.bool_,
            sharding=batch_sharding,
        ),
        "target_masks": jax.ShapeDtypeStruct(
            (*batch_shape[:2], num_target_blocks, num_peaks),
            jnp.bool_,
            sharding=batch_sharding,
        ),
    }


def parse_compiler_options(flags: str) -> dict[str, str]:
    options: dict[str, str] = {}
    for token in flags.replace("\\", " ").split():
        assert token.startswith("--") and "=" in token
        key, value = token[2:].split("=", 1)
        assert key not in options
        options[key] = value
    return options


def _abstract_pure_train_state(
    config: Any,
    *,
    variant: CompileVariant,
) -> tuple[Any, nnx.State, nnx.State, Any, optax.GradientTransformation]:
    total_steps = int(_config_get(config, "training_max_steps", 1))

    def init_fn():
        model = build_model_from_config(config)
        if variant.pack_tokens is not None:
            model.mae_context_encoder_pack_tokens = variant.pack_tokens
        graphdef, params, static_state, opt_state, _optimizer = (
            init_pure_optax_train_state(
                config,
                model,
                total_steps=total_steps,
            )
        )
        return graphdef, params, static_state, opt_state

    graphdef, params, static_state, opt_state = nnx.eval_shape(init_fn)
    optimizer = build_jax_optax_transform(config, total_steps=total_steps)
    return graphdef, params, static_state, opt_state, optimizer


def _with_sharding(tree: Any, sharding: NamedSharding) -> Any:
    return jax.tree.map(
        lambda value: (
            jax.ShapeDtypeStruct(value.shape, value.dtype, sharding=sharding)
            if isinstance(value, jax.ShapeDtypeStruct)
            else value
        ),
        tree,
    )


def _memory_analysis_dict(value: Any) -> dict[str, int] | None:
    if value is None:
        return None
    return {
        field: int(getattr(value, field))
        for field in (
            "generated_code_size_in_bytes",
            "argument_size_in_bytes",
            "output_size_in_bytes",
            "alias_size_in_bytes",
            "temp_size_in_bytes",
            "host_generated_code_size_in_bytes",
            "host_argument_size_in_bytes",
            "host_output_size_in_bytes",
            "host_alias_size_in_bytes",
            "host_temp_size_in_bytes",
        )
    }


def _cost_analysis_summary(value: Any) -> dict[str, float] | None:
    if value is None:
        return None
    return {
        key: float(value[key])
        for key in ("flops", "bytes accessed", "transcendentals", "optimal_seconds")
        if key in value
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Shape-only AOT compile of the JAX train step for a TPU target."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--target", default="ct6e-standard-8t")
    parser.add_argument("--num-slices", type=int, default=1)
    parser.add_argument(
        "--variant",
        default="all",
        help="default, largest-pack, full, all, or pack:N",
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--summary-json", default="")
    parser.add_argument(
        "--compilation-cache-dir",
        default="",
        help="Persistent JAX compilation cache directory for repeated compiles.",
    )
    parser.add_argument("--compiler-options", default="")
    parser.add_argument("--overrides-json", default="{}")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config.update(json.loads(args.overrides_json))
    if args.compilation_cache_dir:
        config.update(
            {
                "jax_compilation_cache_dir": args.compilation_cache_dir,
                "jax_enable_compilation_cache": True,
                "jax_persistent_cache_min_compile_time_secs": 0.0,
                "jax_persistent_cache_min_entry_size_bytes": 0,
            }
        )
    target = resolve_tpu_compile_target(args.target, num_slices=args.num_slices)
    summaries = compile_jax_train_steps_for_tpu(
        config,
        target=target,
        variant_selector=args.variant,
        output_dir=args.output_dir or None,
        compiler_options=parse_compiler_options(args.compiler_options),
    )
    payload = [asdict(summary) for summary in summaries]
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
