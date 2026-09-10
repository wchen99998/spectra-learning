"""Per-chip TPU HBM of a training step, compiled ahead of time on any host.

`jax.experimental.topologies` builds a compile-only device set for a named TPU
topology, so the real XLA TPU backend runs its buffer assignment for a slice
this machine does not have attached. `memory_analysis()` on the result is the
same number the compiler reports on the slice itself - arguments, temporaries,
outputs and the aliased donations - not a hand-rolled activation estimate.

The step is the one `pretrain_jax` runs, lowered from abstract shapes, so no
parameters or optimizer moments are ever allocated on the host.

    .venv/bin/python -m scripts.aot_tpu_memory \\
        --config configs/1p8b_fastmixer_pair_distogram_multiscale_n64_v6e128_ga4.py \\
        --chips 128 --sweep 2,4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# v6e carries 32 GiB of HBM per chip. XLA reports 31.24 GiB of that as
# available to a program - the rest is the program image and runtime scratch -
# and refuses to compile a step that needs more, which is the hard signal this
# script keys off. The reserve below is only for the soft `fits` flag.
V6E_HBM_BYTES = 32 * 1024**3
V6E_AVAILABLE_HBM_BYTES = int(31.24 * 1024**3)
DEFAULT_RUNTIME_RESERVE_BYTES = V6E_HBM_BYTES - V6E_AVAILABLE_HBM_BYTES

V6E_TOPOLOGY_BY_CHIPS = {
    4: "2x2",
    8: "2x4",
    16: "4x4",
    32: "4x8",
    64: "8x8",
    128: "8x16",
    256: "16x16",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--chips", type=int, default=128)
    parser.add_argument(
        "--sweep",
        type=str,
        required=True,
        help="Comma-separated per-chip micro-batch sizes to compile.",
    )
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument(
        "--reserve-bytes",
        type=int,
        default=DEFAULT_RUNTIME_RESERVE_BYTES,
        help="HBM held back for the runtime when deciding whether a step fits.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Config override applied before the model is built, repeatable.",
    )
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def _parse_override(item: str) -> tuple[str, object]:
    key, _, raw = item.partition("=")
    if not _:
        raise ValueError(f"--override expects KEY=VALUE, got {item!r}")
    try:
        return key, json.loads(raw)
    except json.JSONDecodeError:
        return key, raw


def main() -> None:
    args = parse_args()

    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx
    from jax.experimental import topologies
    from jax.sharding import Mesh

    from spectra_learning.config import load_config
    from spectra_learning.data.gems.collate import ACQUISITION_METADATA_KEYS
    from spectra_learning.data.gems.mask_schedule import jepa_mask_stages
    from spectra_learning.models.factory_jax import build_model_from_config
    from spectra_learning.models.fastmixer_capacity import (
        pairmixer_fast_stage_capacities,
    )
    from spectra_learning.models.spectrum_metadata import (
        MASSIVE_V2_ACQUISITION_SCHEMA,
        MASSIVE_V2_CONDITION_DIM,
    )
    from spectra_learning.training import optimizer_sharding
    from spectra_learning.training.pretrain_jax import (
        JAX_DATA_AXIS,
        build_jax_optax_transform,
        jax_shard_optimizer_axis,
        jax_zero1_mode,
        jax_zero1_settings,
        jax_zero1_shards,
        make_pure_accumulated_train_step,
        prepare_jax_training_config,
        trainable_param_filter,
    )

    if args.chips not in V6E_TOPOLOGY_BY_CHIPS:
        raise ValueError(
            f"--chips must be one of {sorted(V6E_TOPOLOGY_BY_CHIPS)}"
        )
    topology_name = f"v6e:{V6E_TOPOLOGY_BY_CHIPS[args.chips]}"
    devices = topologies.get_topology_desc(
        topology_name, platform="tpu",
    ).devices
    mesh = Mesh(np.asarray(devices).reshape(args.chips), (JAX_DATA_AXIS,))

    micro_batches = [int(value) for value in args.sweep.split(",") if value]

    overrides = {"enable_wandb": False}
    overrides.update(dict(_parse_override(item) for item in args.override))
    config = load_config(args.config, overrides)
    stages = jepa_mask_stages(config)
    if not 1 <= args.stage <= len(stages):
        raise ValueError(f"--stage must be in 1..{len(stages)}")
    if not micro_batches or any(value < 1 for value in micro_batches):
        raise ValueError("--sweep requires positive micro-batch sizes")
    stage = stages[args.stage - 1]
    encoder_tokens, predictor_tokens, target_tokens = pairmixer_fast_stage_capacities(
        config
    )[args.stage - 1]
    config.jepa_context_fraction = stage.context_fraction
    config.jepa_target_fraction = stage.target_fraction
    grad_accum = (
        args.gradient_accumulation_steps
        if args.gradient_accumulation_steps is not None
        else stage.gradient_accumulation_steps
    )
    prepare_jax_training_config(config)

    model = nnx.eval_shape(lambda: build_model_from_config(config))
    model.set_fastmixer_capacities(encoder_tokens, predictor_tokens, target_tokens)
    parameter_count = sum(
        int(np.prod(leaf.shape))
        for leaf in jax.tree.leaves(nnx.state(model, nnx.Param))
    )

    # Same split and optimizer as init_pure_optax_train_state, but the moments
    # are shaped rather than allocated: 1.8B parameters would be 7 GB of host
    # RAM per tree and nothing here needs their values.
    graphdef, params, static_state = nnx.split(model, trainable_param_filter, ...)
    params = nnx.as_pure(params)
    static_state = nnx.as_pure(static_state)
    total_steps = int(config.training_max_steps)
    zero1_shards = jax_zero1_shards(config, sharded=True, device_count=args.chips)
    optimizer = build_jax_optax_transform(
        config,
        total_steps=total_steps,
        shard_axis=jax_shard_optimizer_axis(config, sharded=True),
        shard_count=args.chips,
    )
    if zero1_shards:
        zero1_init = (
            optimizer_sharding.zero1_flat_init
            if jax_zero1_mode(config) == "flat"
            else optimizer_sharding.zero1_adam_init_sharded
        )
        opt_state = jax.eval_shape(
            lambda tree: zero1_init(
                tree,
                shards=zero1_shards,
                mesh=mesh,
                shard_axis=JAX_DATA_AXIS,
                moment_dtype=optimizer_sharding.zero1_moment_dtype(config),
            ),
            params,
        )
    else:
        opt_state = jax.eval_shape(optimizer.init, params)
    train_step = make_pure_accumulated_train_step(
        graphdef,
        optimizer,
        sharded=True,
        data_mesh=mesh,
        metric_reduction="mean",
        grad_all_reduce_dtype=jnp.dtype(
            config.get("jax_grad_all_reduce_dtype", "float32")
        ),
        zero1_shards=zero1_shards,
        zero1_mode=jax_zero1_mode(config),
        zero1_settings=(
            jax_zero1_settings(config, total_steps=total_steps)
            if zero1_shards
            else None
        ),
    )

    num_peaks = int(config.num_peaks)
    num_views = int(config.jepa_num_target_blocks)

    def batch_shapes(global_batch: int) -> dict[str, jax.ShapeDtypeStruct]:
        def sds(shape, dtype):
            return jax.ShapeDtypeStruct((grad_accum, global_batch, *shape), dtype)

        shapes = {
            "peak_mz": sds((num_peaks,), jnp.float32),
            "peak_intensity": sds((num_peaks,), jnp.float32),
            "peak_valid_mask": sds((num_peaks,), jnp.bool_),
            "context_mask": sds((num_peaks,), jnp.bool_),
            "target_masks": sds((num_views, num_peaks), jnp.bool_),
            "precursor_mz": sds((), jnp.float32),
            "collision_energy": sds((), jnp.float32),
            "charge": sds((), jnp.float32),
        }
        if config.get("encoder_metadata_schema", None) == MASSIVE_V2_ACQUISITION_SCHEMA:
            shapes["spectrum_metadata"] = sds(
                (MASSIVE_V2_CONDITION_DIM,), jnp.float32
            )
            for name in ACQUISITION_METADATA_KEYS:
                shapes[name] = sds((), jnp.float32)
        return shapes

    results = []
    for micro_batch in micro_batches:
        global_batch = micro_batch * args.chips
        budget = V6E_HBM_BYTES - args.reserve_bytes
        try:
            compiled = train_step.lower(
                params,
                static_state,
                opt_state,
                batch_shapes(global_batch),
            ).compile()
        except jax.errors.JaxRuntimeError as error:
            # The TPU compiler refuses outright rather than reporting a size,
            # which is the authoritative answer for this micro-batch.
            if not any(marker in str(error).lower() for marker in (
                "out of memory", "memory allocation", "exceeds available hbm",
            )):
                raise
            message = str(error)
            results.append(
                {
                    "micro_batch_per_chip": micro_batch,
                    "gradient_accumulation_steps": grad_accum,
                    "global_batch_size": global_batch * grad_accum,
                    "fits": False,
                    "compiler_error": message,
                }
            )
            print(
                f"micro_batch={micro_batch:>4} "
                f"global_batch={global_batch * grad_accum:>7} "
                f"OOM at compile: {message}",
                flush=True,
            )
            continue
        memory = compiled.memory_analysis()
        total = (
            memory.argument_size_in_bytes
            + memory.temp_size_in_bytes
            + memory.output_size_in_bytes
            - memory.alias_size_in_bytes
        )
        results.append(
            {
                "micro_batch_per_chip": micro_batch,
                "gradient_accumulation_steps": grad_accum,
                "global_batch_size": global_batch * grad_accum,
                "argument_bytes": memory.argument_size_in_bytes,
                "temp_bytes": memory.temp_size_in_bytes,
                "output_bytes": memory.output_size_in_bytes,
                "alias_bytes": memory.alias_size_in_bytes,
                "total_bytes": total,
                "total_gib": total / 1024**3,
                "budget_gib": budget / 1024**3,
                "fits": total <= budget,
            }
        )
        print(
            f"micro_batch={micro_batch:>4} "
            f"global_batch={global_batch * grad_accum:>7} "
            f"hbm={total / 1024**3:7.2f} GiB "
            f"(args {memory.argument_size_in_bytes / 1024**3:6.2f} "
            f"+ temp {memory.temp_size_in_bytes / 1024**3:6.2f}) "
            f"{'fits' if total <= budget else 'OOM'}",
            flush=True,
        )

    report = {
        "config": str(args.config),
        "topology": topology_name,
        "chips": args.chips,
        "overrides": args.override,
        "zero1_shards": zero1_shards,
        "parameter_count": parameter_count,
        "trainable_parameter_count": sum(leaf.size for leaf in jax.tree.leaves(params)),
        "encoder_tokens": encoder_tokens,
        "predictor_tokens": predictor_tokens,
        "target_tokens": target_tokens,
        "hbm_bytes_per_chip": V6E_HBM_BYTES,
        "runtime_reserve_bytes": args.reserve_bytes,
        "results": results,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
