from __future__ import annotations

import argparse
import json
import re
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
from spectra_learning.config.msg_probe import validate_msg_probe_config
from spectra_learning.data.loading import local_batch_size
from spectra_learning.models.factory_jax import build_model_from_config
from spectra_learning.probes.massspec.msg_probe_jax import (
    _extract_pair_features_jitted,
    _extract_single_features_jitted,
    _init_probe_params,
    _make_jitted_probe_predict_step,
    _make_jitted_probe_train_step,
    _uses_pair_features,
    jitted_probe_train_step_compile_fns,
)
from spectra_learning.probes.massspec.msg_settings import (
    MsgProbeTaskSpec,
    msg_probe_variants_from_config,
)
from spectra_learning.data.massspec_targets import MACCS_FINGERPRINT_BITS
from spectra_learning.training.pretrain_jax import (
    JAX_DATA_AXIS,
    _config_get,
    _context_encoder_pack_choices,
    _jax_train_step_compile_variants,
    build_jax_optax_transform,
    configure_jax_runtime,
    init_pure_optax_train_state,
    make_pure_accumulated_train_step,
    make_pure_eval_step,
)


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


def _dynamic_v6e_multihost_target(name: str) -> TpuCompileTarget | None:
    normalized = name.lower()
    match = re.fullmatch(
        r"(?:v6e-|ct6e-standard-4t-v6e-)(?P<topology>\d+x\d+)-multihost",
        normalized,
    )
    if match is None:
        match = re.fullmatch(
            r"ct6e-standard-4t-v6e-(?P<topology>\d+x\d+)",
            normalized,
        )
    if match is None:
        return None
    topology = match.group("topology")
    dims = tuple(int(part) for part in topology.split("x"))
    devices_per_slice = dims[0] * dims[1]
    if devices_per_slice <= 0 or devices_per_slice % 4 != 0:
        raise ValueError(
            f"v6e multihost topology must contain a positive multiple of 4 chips; got {topology!r}"
        )
    return TpuCompileTarget(
        name=f"ct6e-standard-4t-v6e-{topology}",
        topology_name=f"v6e:{topology}",
        chips_per_host_bounds=(2, 2, 1),
        devices_per_slice=devices_per_slice,
        vm_count=devices_per_slice // 4,
    )


@dataclass(frozen=True)
class CompileVariant:
    name: str
    pack_tokens: int | None
    full_context_fallback: bool


@dataclass(frozen=True)
class CompileSummary:
    kind: str
    target: dict[str, Any]
    variant: str
    cache_glob: str
    abstract_state_seconds: float
    lower_seconds: float
    compile_seconds: float
    memory_analysis: dict[str, int] | None
    cost_analysis: dict[str, float] | None
    executable_file: str


@dataclass(frozen=True)
class LoweredCompileResult:
    compiled: Any | None
    compile_seconds: float
    reused_persistent_cache: bool


OFFLINE_TOPOLOGY_CACHE_HIT_ERROR = (
    "PjRtCompiler must be constructed with a Client to call "
    "DeserializeLoadedExecutable"
)


def resolve_tpu_compile_target(name: str, *, num_slices: int = 1) -> TpuCompileTarget:
    target = TPU_COMPILE_TARGETS.get(name.lower()) or _dynamic_v6e_multihost_target(name)
    if target is None:
        raise KeyError(name)
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
    compile_options = build_compiler_options(compiler_options)
    data_mesh = build_tpu_compile_mesh(target)
    variants = compile_variants(config, variant_selector)
    summaries: list[CompileSummary] = []
    output_path = None if output_dir is None else Path(output_dir)
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    for variant in variants:
        cache_glob = "jit_pure_sharded_accumulated_train_step-*-cache"
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
        compile_result = compile_lowered_or_reuse_persistent_cache(
            lowered,
            compile_options,
            label=variant.name,
        )
        compiled = compile_result.compiled
        print(
            f"Finished {variant.name}: abstract={abstract_state_seconds:.2f}s "
            f"lower={lower_seconds:.2f}s compile={compile_result.compile_seconds:.2f}s "
            f"status={'cache-hit' if compile_result.reused_persistent_cache else 'compiled'}",
            file=sys.stderr,
            flush=True,
        )
        executable_file = ""
        if output_path is not None:
            executable_file = str(output_path / f"{variant.name}.compiled")
            if compiled is not None:
                write_serialized_executable(compiled, Path(executable_file))
            elif not _nonempty_file(Path(executable_file)):
                executable_file = ""
        summary = CompileSummary(
            kind="train",
            target=asdict(target),
            variant=variant.name,
            cache_glob=cache_glob,
            abstract_state_seconds=abstract_state_seconds,
            lower_seconds=lower_seconds,
            compile_seconds=compile_result.compile_seconds,
            memory_analysis=compiled_memory_analysis(compiled),
            cost_analysis=compiled_cost_analysis(compiled),
            executable_file=executable_file,
        )
        summaries.append(summary)
    if bool(_config_get(config, "jax_precompile_eval_steps", False)):
        summaries.append(
            compile_jax_eval_step_for_tpu(
                config,
                target=target,
                data_mesh=data_mesh,
                output_dir=output_path,
                compiler_options=compile_options,
            )
        )
    if bool(_config_get(config, "jax_precompile_msg_probe", False)):
        summaries.extend(
            compile_jax_msg_probe_for_tpu(
                config,
                target=target,
                output_dir=output_path,
                compiler_options=compile_options,
            )
        )
    return summaries


def compile_jax_eval_step_for_tpu(
    config: Any,
    *,
    target: TpuCompileTarget,
    data_mesh: Mesh,
    output_dir: Path | None,
    compiler_options: dict[str, str],
) -> CompileSummary:
    cache_glob = "jit_pure_sharded_eval_step-*-cache"
    print("Preparing abstract eval state...", file=sys.stderr, flush=True)
    abstract_start = time.perf_counter()
    graphdef, trainable_params, static_state, _opt_state, _optimizer = (
        _abstract_pure_train_state(
            config,
            variant=CompileVariant("eval_full", 0, True),
        )
    )
    abstract_state_seconds = time.perf_counter() - abstract_start
    trainable_params = _with_sharding(trainable_params, NamedSharding(data_mesh, P()))
    static_state = _with_sharding(static_state, NamedSharding(data_mesh, P()))
    batch = abstract_eval_batch(config, target=target, data_mesh=data_mesh)
    eval_step = make_pure_eval_step(graphdef, sharded=True, data_mesh=data_mesh)
    print("Lowering eval...", file=sys.stderr, flush=True)
    lower_start = time.perf_counter()
    with jax.set_mesh(data_mesh):
        lowered = eval_step.lower(trainable_params, static_state, batch)
    lower_seconds = time.perf_counter() - lower_start
    print("Compiling eval...", file=sys.stderr, flush=True)
    compile_result = compile_lowered_or_reuse_persistent_cache(
        lowered,
        compiler_options,
        label="eval",
    )
    compiled = compile_result.compiled
    executable_file = ""
    if output_dir is not None:
        executable_file = str(output_dir / "eval.compiled")
        if compiled is not None:
            write_serialized_executable(compiled, Path(executable_file))
        elif not _nonempty_file(Path(executable_file)):
            executable_file = ""
    summary = CompileSummary(
        kind="eval",
        target=asdict(target),
        variant="full",
        cache_glob=cache_glob,
        abstract_state_seconds=abstract_state_seconds,
        lower_seconds=lower_seconds,
        compile_seconds=compile_result.compile_seconds,
        memory_analysis=compiled_memory_analysis(compiled),
        cost_analysis=compiled_cost_analysis(compiled),
        executable_file=executable_file,
    )
    return summary


def compile_jax_msg_probe_for_tpu(
    config: Any,
    *,
    target: TpuCompileTarget,
    output_dir: Path | None,
    compiler_options: dict[str, str],
) -> list[CompileSummary]:
    validate_msg_probe_config(config)
    variants = msg_probe_variants_from_config(config)
    use_pair_features = any(_uses_pair_features(variant) for variant in variants)
    task_spec = abstract_msg_probe_task_spec(config)
    probe_batch = abstract_msg_probe_batch(config, target=target)
    feature_summary = compile_jax_msg_probe_feature_step(
        config,
        target=target,
        batch=probe_batch,
        use_pair_features=use_pair_features,
        output_dir=output_dir,
        compiler_options=compiler_options,
    )
    summaries = [feature_summary]
    features = abstract_msg_probe_features(
        config,
        probe_batch_size=int(probe_batch["peak_mz"].shape[0]),
        use_pair_features=use_pair_features,
    )
    optimizer = optax.adamw(
        learning_rate=float(_config_get(config, "msg_probe_learning_rate", 1e-3)),
        weight_decay=float(_config_get(config, "msg_probe_weight_decay", 1e-2)),
    )
    step_batch = abstract_msg_probe_step_batch(config, task_spec, target=target)
    for variant in variants:
        params = jax.eval_shape(
            lambda variant=variant: _init_probe_params(
                jax.random.PRNGKey(0),
                variant=variant,
                config=config,
                task_spec=task_spec,
            )
        )
        opt_state = jax.eval_shape(lambda params=params: optimizer.init(params))
        train_step = _make_jitted_probe_train_step(
            optimizer=optimizer,
            variant=variant,
            task_spec=task_spec,
            distributed=True,
        )
        for component_name, component in jitted_probe_train_step_compile_fns(train_step):
            args: tuple[Any, ...]
            if component_name == "probe_grad":
                args = (params, step_batch, features)
                cache_glob = "jit_grad_step-*-cache"
            elif component_name == "probe_apply":
                grads = _shape_like_tree(params)
                args = (params, opt_state, grads, step_batch, features)
                cache_glob = "jit_apply_step-*-cache"
            else:
                args = (params, opt_state, step_batch, features)
                cache_glob = "jit_local_train_step-*-cache"
            summaries.append(
                compile_jitted_probe_component(
                    component,
                    args=args,
                    target=target,
                    kind=f"msg_probe_{component_name}",
                    variant=variant,
                    cache_glob=cache_glob,
                    output_dir=output_dir,
                    compiler_options=compiler_options,
                )
            )
        predict_step = _make_jitted_probe_predict_step(
            variant=variant,
            task_spec=task_spec,
        )
        summaries.append(
            compile_jitted_probe_component(
                predict_step,
                args=(params, step_batch, features),
                target=target,
                kind="msg_probe_predict",
                variant=variant,
                cache_glob="jit_predict_step-*-cache",
                output_dir=output_dir,
                compiler_options=compiler_options,
            )
        )
    return summaries


def compile_jax_msg_probe_feature_step(
    config: Any,
    *,
    target: TpuCompileTarget,
    batch: dict[str, jax.ShapeDtypeStruct],
    use_pair_features: bool,
    output_dir: Path | None,
    compiler_options: dict[str, str],
) -> CompileSummary:
    variant = "pair" if use_pair_features else "single"
    cache_glob = (
        "jit__extract_pair_features_jitted-*-cache"
        if use_pair_features
        else "jit__extract_single_features_jitted-*-cache"
    )
    print("Preparing abstract MSG probe feature extractor...", file=sys.stderr, flush=True)
    abstract_start = time.perf_counter()
    model = nnx.eval_shape(lambda: build_model_from_config(config))
    abstract_state_seconds = time.perf_counter() - abstract_start
    feature_step = (
        _extract_pair_features_jitted
        if use_pair_features
        else _extract_single_features_jitted
    )
    print("Lowering MSG probe feature extractor...", file=sys.stderr, flush=True)
    lower_start = time.perf_counter()
    lowered = feature_step.lower(
        model,
        batch["peak_mz"],
        batch["peak_intensity"],
        batch["peak_valid_mask"],
        batch["precursor_mz"],
    )
    lower_seconds = time.perf_counter() - lower_start
    print("Compiling MSG probe feature extractor...", file=sys.stderr, flush=True)
    compile_result = compile_lowered_or_reuse_persistent_cache(
        lowered,
        compiler_options,
        label="MSG probe feature extractor",
    )
    compiled = compile_result.compiled
    executable_file = ""
    if output_dir is not None:
        stem = "msg_probe_pair_features" if use_pair_features else "msg_probe_features"
        executable_file = str(output_dir / f"{stem}.compiled")
        if compiled is not None:
            write_serialized_executable(compiled, Path(executable_file))
        elif not _nonempty_file(Path(executable_file)):
            executable_file = ""
    summary = CompileSummary(
        kind="msg_probe_features",
        target=asdict(target),
        variant=variant,
        cache_glob=cache_glob,
        abstract_state_seconds=abstract_state_seconds,
        lower_seconds=lower_seconds,
        compile_seconds=compile_result.compile_seconds,
        memory_analysis=compiled_memory_analysis(compiled),
        cost_analysis=compiled_cost_analysis(compiled),
        executable_file=executable_file,
    )
    return summary


def compile_jitted_probe_component(
    component: Any,
    *,
    args: tuple[Any, ...],
    target: TpuCompileTarget,
    kind: str,
    variant: str,
    cache_glob: str,
    output_dir: Path | None,
    compiler_options: dict[str, str],
) -> CompileSummary:
    print(f"Lowering {kind}[{variant}]...", file=sys.stderr, flush=True)
    lower_start = time.perf_counter()
    lowered = component.lower(*args)
    lower_seconds = time.perf_counter() - lower_start
    print(f"Compiling {kind}[{variant}]...", file=sys.stderr, flush=True)
    compile_result = compile_lowered_or_reuse_persistent_cache(
        lowered,
        compiler_options,
        label=f"{kind}[{variant}]",
    )
    compiled = compile_result.compiled
    executable_file = ""
    if output_dir is not None:
        executable_file = str(output_dir / f"{kind}-{variant}.compiled")
        if compiled is not None:
            write_serialized_executable(compiled, Path(executable_file))
        elif not _nonempty_file(Path(executable_file)):
            executable_file = ""
    summary = CompileSummary(
        kind=kind,
        target=asdict(target),
        variant=variant,
        cache_glob=cache_glob,
        abstract_state_seconds=0.0,
        lower_seconds=lower_seconds,
        compile_seconds=compile_result.compile_seconds,
        memory_analysis=compiled_memory_analysis(compiled),
        cost_analysis=compiled_cost_analysis(compiled),
        executable_file=executable_file,
    )
    return summary


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


def abstract_eval_batch(
    config: Any,
    *,
    target: TpuCompileTarget,
    data_mesh: Mesh,
) -> dict[str, jax.ShapeDtypeStruct]:
    batch_size = target_microbatch_size(config, target)
    num_peaks = int(_config_get(config, "num_peaks", 60))
    num_target_blocks = int(_config_get(config, "jepa_num_target_blocks", 2))
    batch_shape = (batch_size, num_peaks)
    batch_sharding = NamedSharding(data_mesh, P(JAX_DATA_AXIS))
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
            (batch_size,),
            jnp.float32,
            sharding=batch_sharding,
        ),
        "context_mask": jax.ShapeDtypeStruct(
            batch_shape,
            jnp.bool_,
            sharding=batch_sharding,
        ),
        "target_masks": jax.ShapeDtypeStruct(
            (batch_size, num_target_blocks, num_peaks),
            jnp.bool_,
            sharding=batch_sharding,
        ),
    }


def abstract_msg_probe_task_spec(config: Any) -> MsgProbeTaskSpec:
    return MsgProbeTaskSpec(
        regression_tasks=(),
        binary_tasks=(),
        maccs_bits=MACCS_FINGERPRINT_BITS,
        regression_means={},
        regression_stds={},
        fingerprint_task="maccs",
        single_pair_covariance_include_diagonal=bool(
            _config_get(
                config,
                "msg_probe_single_pair_covariance_include_diagonal",
                False,
            )
        ),
    )


def target_probe_batch_size(config: Any, target: TpuCompileTarget) -> int:
    global_probe_batch_size = int(
        _config_get(
            config,
            "msg_probe_batch_size",
            _config_get(config, "batch_size", 512),
        )
    )
    return local_batch_size(global_probe_batch_size, target.process_count)


def abstract_msg_probe_batch(
    config: Any,
    *,
    target: TpuCompileTarget,
) -> dict[str, jax.ShapeDtypeStruct]:
    batch_size = target_probe_batch_size(config, target)
    num_peaks = int(_config_get(config, "num_peaks", 60))
    batch: dict[str, jax.ShapeDtypeStruct] = {
        "peak_mz": jax.ShapeDtypeStruct((batch_size, num_peaks), jnp.float32),
        "peak_intensity": jax.ShapeDtypeStruct((batch_size, num_peaks), jnp.float32),
        "peak_valid_mask": jax.ShapeDtypeStruct((batch_size, num_peaks), jnp.bool_),
        "precursor_mz": jax.ShapeDtypeStruct((batch_size,), jnp.float32),
        "probe_valid_mol": jax.ShapeDtypeStruct((batch_size,), jnp.bool_),
        "probe_maccs": jax.ShapeDtypeStruct(
            (batch_size, MACCS_FINGERPRINT_BITS),
            jnp.int32,
        ),
    }
    return batch


def abstract_msg_probe_step_batch(
    config: Any,
    task_spec: MsgProbeTaskSpec,
    *,
    target: TpuCompileTarget,
) -> dict[str, jax.ShapeDtypeStruct]:
    batch = abstract_msg_probe_batch(config, target=target)
    keys = ["peak_valid_mask", "probe_valid_mol"]
    keys.extend(f"probe_{name}" for name in task_spec.regression_tasks)
    keys.extend(f"probe_{name}" for name in task_spec.binary_tasks)
    if task_spec.maccs_bits > 0:
        keys.append(f"probe_{task_spec.fingerprint_task}")
    return {key: batch[key] for key in keys}


def abstract_msg_probe_features(
    config: Any,
    *,
    probe_batch_size: int,
    use_pair_features: bool,
) -> Any:
    num_peaks = int(_config_get(config, "num_peaks", 60))
    token_count = num_peaks + 1
    single = jax.ShapeDtypeStruct(
        (probe_batch_size, token_count, int(_config_get(config, "model_dim", 0))),
        jnp.float32,
    )
    if not use_pair_features:
        return single
    pair_dim = int(
        _config_get(
            config,
            "pairmixer_pair_dim",
            _config_get(config, "model_dim", 0),
        )
    )
    pair = jax.ShapeDtypeStruct(
        (probe_batch_size, token_count, token_count, pair_dim),
        jnp.float32,
    )
    return single, pair


def _shape_like_tree(tree: Any) -> Any:
    return jax.tree.map(
        lambda value: jax.ShapeDtypeStruct(value.shape, value.dtype),
        tree,
    )


def parse_compiler_options(flags: str) -> dict[str, str]:
    options: dict[str, str] = {}
    for token in flags.replace("\\", " ").split():
        assert token.startswith("--") and "=" in token
        key, value = token[2:].split("=", 1)
        assert key not in options
        options[key] = value
    return options


def build_compiler_options(
    compiler_options: dict[str, str] | None,
) -> dict[str, str]:
    return dict(compiler_options or {})


def compile_lowered_or_reuse_persistent_cache(
    lowered: Any,
    compiler_options: dict[str, str],
    *,
    label: str,
) -> LoweredCompileResult:
    previous_raise_cache_errors = bool(jax.config.jax_raise_persistent_cache_errors)
    jax.config.update("jax_raise_persistent_cache_errors", True)
    compile_start = time.perf_counter()
    try:
        compiled = lowered.compile(compiler_options=compiler_options)
    except Exception as exc:
        compile_seconds = time.perf_counter() - compile_start
        if _is_offline_topology_cache_hit_error(exc):
            print(
                f"Reusing JAX persistent cache entry for {label}; "
                "offline topology compiler cannot deserialize it without a TPU Client.",
                file=sys.stderr,
                flush=True,
            )
            return LoweredCompileResult(
                compiled=None,
                compile_seconds=compile_seconds,
                reused_persistent_cache=True,
            )
        raise
    finally:
        jax.config.update(
            "jax_raise_persistent_cache_errors",
            previous_raise_cache_errors,
        )
    compile_seconds = time.perf_counter() - compile_start
    return LoweredCompileResult(
        compiled=compiled,
        compile_seconds=compile_seconds,
        reused_persistent_cache=False,
    )


def _is_offline_topology_cache_hit_error(exc: BaseException) -> bool:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if OFFLINE_TOPOLOGY_CACHE_HIT_ERROR in str(current):
            return True
        current = current.__cause__ or current.__context__
    return False


def serializable_compiled_object(compiled: Any) -> Any:
    return getattr(compiled, "compiled", compiled)


def write_serialized_executable(compiled: Any, path: Path) -> None:
    serialized, _, _ = serialize(serializable_compiled_object(compiled))
    path.write_bytes(serialized)


def compiled_memory_analysis(compiled: Any | None) -> dict[str, int] | None:
    if compiled is None:
        return None
    return _memory_analysis_dict(compiled.memory_analysis())


def compiled_cost_analysis(compiled: Any | None) -> dict[str, float] | None:
    if compiled is None:
        return None
    return _cost_analysis_summary(compiled.cost_analysis())


def _nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


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
