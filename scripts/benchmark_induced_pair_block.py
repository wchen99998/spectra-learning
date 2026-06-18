from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import torch
from flax import nnx

from spectra_learning.models.induced_pair import InducedPairState, InducedPairBlock
from spectra_learning.models.induced_pair_jax import (
    InducedPairBlock as InducedPairBlockJax,
)
from spectra_learning.models.induced_pair_jax import (
    InducedPairState as InducedPairStateJax,
)
from spectra_learning.models.pairmixer import PairMixerBlock
from spectra_learning.models.pairmixer_jax import PairMixerBlock as PairMixerBlockJax


def _sync_torch() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_ms(fn: Callable[[], object], *, warmup: int, steps: int) -> float:
    for _ in range(warmup):
        fn()
    _sync_torch()
    start = time.perf_counter()
    for _ in range(steps):
        fn()
    _sync_torch()
    return (time.perf_counter() - start) * 1000.0 / steps


def _benchmark_torch(args: argparse.Namespace) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bf16" and device.type == "cuda" else torch.float32
    single = torch.randn(
        args.batch_size,
        args.tokens,
        args.model_dim,
        device=device,
        dtype=dtype,
    )
    pair = torch.randn(
        args.batch_size,
        args.tokens,
        args.tokens,
        args.pair_dim,
        device=device,
        dtype=dtype,
    )
    state = InducedPairState(
        torch.randn(
            args.batch_size,
            args.inducing,
            args.model_dim,
            device=device,
            dtype=dtype,
        ),
        torch.randn(
            args.batch_size,
            args.inducing,
            args.inducing,
            args.pair_dim,
            device=device,
            dtype=dtype,
        ),
        torch.zeros(
            args.batch_size,
            args.tokens,
            args.inducing,
            device=device,
            dtype=dtype,
        ),
    )
    mask = torch.ones(args.batch_size, args.tokens, device=device, dtype=torch.bool)
    dense_block = PairMixerBlock(
        single_dim=args.model_dim,
        pair_dim=args.pair_dim,
        num_heads=args.num_heads,
        attention_mlp_multiple=args.mlp_multiple,
        norm_eps=1e-5,
        dropout=0.0,
        use_pair_bias_attention=True,
    ).to(device=device, dtype=dtype)
    induced_block = InducedPairBlock(
        single_dim=args.model_dim,
        pair_dim=args.pair_dim,
        num_heads=args.num_heads,
        attention_mlp_multiple=args.mlp_multiple,
        norm_eps=1e-5,
        dropout=0.0,
        use_pair_bias_attention=True,
    ).to(device=device, dtype=dtype)
    dense_block.eval()
    induced_block.eval()

    with torch.no_grad():
        dense_ms = _time_ms(
            lambda: dense_block(single, pair, mask, mask),
            warmup=args.warmup,
            steps=args.steps,
        )
        induced_ms = _time_ms(
            lambda: induced_block(single, state, mask, mask),
            warmup=args.warmup,
            steps=args.steps,
        )
    return {"torch_dense_ms": dense_ms, "torch_induced_ms": induced_ms}


def _benchmark_jax(args: argparse.Namespace) -> dict[str, float]:
    dtype = jnp.bfloat16 if args.dtype == "bf16" else jnp.float32
    key = jax.random.key(0)
    single_key, pair_key, inducing_key, latent_pair_key = jax.random.split(key, 4)
    single = jax.random.normal(
        single_key,
        (args.batch_size, args.tokens, args.model_dim),
        dtype=dtype,
    )
    pair = jax.random.normal(
        pair_key,
        (args.batch_size, args.tokens, args.tokens, args.pair_dim),
        dtype=dtype,
    )
    state = InducedPairStateJax(
        jax.random.normal(
            inducing_key,
            (args.batch_size, args.inducing, args.model_dim),
            dtype=dtype,
        ),
        jax.random.normal(
            latent_pair_key,
            (args.batch_size, args.inducing, args.inducing, args.pair_dim),
            dtype=dtype,
        ),
        jnp.zeros((args.batch_size, args.tokens, args.inducing), dtype=dtype),
    )
    mask = jnp.ones((args.batch_size, args.tokens), dtype=jnp.bool_)
    dense_block = PairMixerBlockJax(
        single_dim=args.model_dim,
        pair_dim=args.pair_dim,
        num_heads=args.num_heads,
        attention_mlp_multiple=args.mlp_multiple,
        norm_eps=1e-5,
        dropout=0.0,
        use_pair_bias_attention=True,
        compute_dtype=dtype,
    )
    induced_block = InducedPairBlockJax(
        single_dim=args.model_dim,
        pair_dim=args.pair_dim,
        num_heads=args.num_heads,
        attention_mlp_multiple=args.mlp_multiple,
        norm_eps=1e-5,
        dropout=0.0,
        use_pair_bias_attention=True,
        compute_dtype=dtype,
    )

    @nnx.jit
    def dense_forward(block, single, pair, mask):
        out_single, out_pair = block(single, pair, mask, mask)
        return out_single, out_pair

    @nnx.jit
    def induced_forward(block, single, state, mask):
        out_single, out_state = block(single, state, mask, mask)
        return out_single, out_state.pair

    dense_forward(dense_block, single, pair, mask)[0].block_until_ready()
    induced_forward(induced_block, single, state, mask)[0].block_until_ready()

    start = time.perf_counter()
    for _ in range(args.steps):
        dense_forward(dense_block, single, pair, mask)[0].block_until_ready()
    dense_ms = (time.perf_counter() - start) * 1000.0 / args.steps

    start = time.perf_counter()
    for _ in range(args.steps):
        induced_forward(induced_block, single, state, mask)[0].block_until_ready()
    induced_ms = (time.perf_counter() - start) * 1000.0 / args.steps
    return {"jax_dense_ms": dense_ms, "jax_induced_ms": induced_ms}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("torch", "jax", "both"), default="both")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--inducing", type=int, default=8)
    parser.add_argument("--model-dim", type=int, default=640)
    parser.add_argument("--pair-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=10)
    parser.add_argument("--mlp-multiple", type=float, default=4.0)
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    print(
        "shape "
        f"B={args.batch_size} N={args.tokens} M={args.inducing} "
        f"D={args.model_dim} P={args.pair_dim} H={args.num_heads} dtype={args.dtype}"
    )
    results: dict[str, float] = {}
    if args.backend in {"torch", "both"}:
        results.update(_benchmark_torch(args))
    if args.backend in {"jax", "both"}:
        results.update(_benchmark_jax(args))
    for key, value in results.items():
        print(f"{key}: {value:.3f} ms")
    if "torch_dense_ms" in results and "torch_induced_ms" in results:
        print(f"torch_speedup: {results['torch_dense_ms'] / results['torch_induced_ms']:.2f}x")
    if "jax_dense_ms" in results and "jax_induced_ms" in results:
        print(f"jax_speedup: {results['jax_dense_ms'] / results['jax_induced_ms']:.2f}x")


if __name__ == "__main__":
    main()
