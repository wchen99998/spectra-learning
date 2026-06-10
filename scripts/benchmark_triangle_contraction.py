import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PairMixer triangle contractions.")
    parser.add_argument("--devices", type=int, default=4)
    parser.add_argument("--batch-per-device", type=int, default=32)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--pair-dim", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=25)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--metrics-json", type=Path, default=None)
    return parser.parse_args()


def _precision(dtype: jnp.dtype):
    return jax.lax.Precision.DEFAULT if dtype == jnp.bfloat16 else None


def current_channel_einsum(a: jax.Array, b: jax.Array) -> jax.Array:
    precision = _precision(a.dtype)
    a_channel_major = jnp.transpose(a, (0, 3, 1, 2))
    b_channel_major = jnp.transpose(b, (0, 3, 1, 2))
    outgoing = jnp.einsum(
        "bcik,bcjk->bcij",
        a_channel_major,
        b_channel_major,
        precision=precision,
    )
    incoming = jnp.einsum(
        "bcki,bckj->bcij",
        a_channel_major,
        b_channel_major,
        precision=precision,
    )
    return jnp.transpose(outgoing + incoming, (0, 2, 3, 1))


def direct_einsum(a: jax.Array, b: jax.Array) -> jax.Array:
    precision = _precision(a.dtype)
    outgoing = jnp.einsum(
        "bikc,bjkc->bijc",
        a,
        b,
        precision=precision,
    )
    incoming = jnp.einsum(
        "bkic,bkjc->bijc",
        a,
        b,
        precision=precision,
    )
    return outgoing + incoming


def flattened_matmul(a: jax.Array, b: jax.Array) -> jax.Array:
    a_channel_major = jnp.transpose(a, (0, 3, 1, 2))
    b_channel_major = jnp.transpose(b, (0, 3, 1, 2))
    batch_size, pair_dim, tokens, _ = a_channel_major.shape
    a_flat = jnp.reshape(a_channel_major, (batch_size * pair_dim, tokens, tokens))
    b_flat = jnp.reshape(b_channel_major, (batch_size * pair_dim, tokens, tokens))
    outgoing = jnp.matmul(a_flat, jnp.swapaxes(b_flat, -1, -2))
    incoming = jnp.matmul(jnp.swapaxes(a_flat, -1, -2), b_flat)
    update = jnp.reshape(outgoing + incoming, (batch_size, pair_dim, tokens, tokens))
    return jnp.transpose(update, (0, 2, 3, 1))


def elementwise_reduce(a: jax.Array, b: jax.Array) -> jax.Array:
    outgoing = jnp.sum(a[:, :, None, :, :] * b[:, None, :, :, :], axis=3)
    incoming = jnp.sum(a[:, :, :, None, :] * b[:, :, None, :, :], axis=1)
    return outgoing + incoming


def elementwise_reduce_f32_sum(a: jax.Array, b: jax.Array) -> jax.Array:
    a_float = a.astype(jnp.float32)
    b_float = b.astype(jnp.float32)
    outgoing = jnp.sum(
        a_float[:, :, None, :, :] * b_float[:, None, :, :, :],
        axis=3,
    )
    incoming = jnp.sum(
        a_float[:, :, :, None, :] * b_float[:, :, None, :, :],
        axis=1,
    )
    return (outgoing + incoming).astype(a.dtype)


VARIANTS = {
    "current_channel_einsum": current_channel_einsum,
    "direct_einsum": direct_einsum,
    "flattened_matmul": flattened_matmul,
    "elementwise_reduce": elementwise_reduce,
    "elementwise_reduce_f32_sum": elementwise_reduce_f32_sum,
}


def _make_inputs(args: argparse.Namespace) -> tuple[jax.Array, jax.Array]:
    shape = (
        args.devices,
        args.batch_per_device,
        args.tokens,
        args.tokens,
        args.pair_dim,
    )
    key_a, key_b = jax.random.split(jax.random.PRNGKey(0))
    a = jax.random.normal(key_a, shape, dtype=jnp.float32).astype(jnp.bfloat16)
    b = jax.random.normal(key_b, shape, dtype=jnp.float32).astype(jnp.bfloat16)
    return a, b


def _benchmark_variant(
    name: str,
    fn,
    a: jax.Array,
    b: jax.Array,
    devices: list[jax.Device],
    warmup_steps: int,
    measured_steps: int,
) -> dict[str, float | str]:
    sharded_fn = jax.pmap(fn, devices=devices)
    compiled = sharded_fn(a, b)
    compiled.block_until_ready()

    for _ in range(warmup_steps):
        out = sharded_fn(a, b)
    out.block_until_ready()

    start = time.perf_counter()
    for _ in range(measured_steps):
        out = sharded_fn(a, b)
    out.block_until_ready()
    elapsed = time.perf_counter() - start
    devices, batch_per_device, tokens, _, pair_dim = a.shape
    output_elements = devices * batch_per_device * tokens * tokens * pair_dim
    contraction_flops = 4.0 * tokens * output_elements
    return {
        "variant": name,
        "elapsed_seconds": elapsed,
        "steps": float(measured_steps),
        "steps_per_second": float(measured_steps) / elapsed,
        "us_per_step": 1e6 * elapsed / float(measured_steps),
        "estimated_flops_per_step": contraction_flops,
        "estimated_tflops_per_second": contraction_flops * measured_steps / elapsed / 1e12,
    }


def main() -> None:
    args = parse_args()
    jax.config.update("jax_default_matmul_precision", "highest")
    devices = jax.devices()[: args.devices]
    a, b = _make_inputs(args)

    baseline = jax.pmap(current_channel_einsum, devices=devices)(a, b)
    baseline.block_until_ready()
    metrics = []
    for name, fn in VARIANTS.items():
        out = jax.pmap(fn, devices=devices)(a, b)
        out.block_until_ready()
        max_abs_diff = float(jnp.max(jnp.abs(out.astype(jnp.float32) - baseline.astype(jnp.float32))))
        result = _benchmark_variant(name, fn, a, b, devices, args.warmup_steps, args.steps)
        result["max_abs_diff_vs_current"] = max_abs_diff
        metrics.append(result)

    payload = {
        "devices": float(args.devices),
        "batch_per_device": float(args.batch_per_device),
        "tokens": float(args.tokens),
        "pair_dim": float(args.pair_dim),
        "dtype": "bfloat16",
        "variants": metrics,
    }
    if args.metrics_json is not None:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_json.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
