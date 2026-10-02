"""Synchronised JAX forward+backward timing and compiler memory for pair readouts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from spectra_learning.models.pair_to_single_jax import PairSummaryToSingleReadout, PairToSingleReadout
from spectra_learning.models.pairmixer_jax import AttentionPairBias
from scripts.pair_to_single_pilot import DiagonalReadout


class Interface(nnx.Module):
    def __init__(self, arm):
        self.attention = AttentionPairBias(
            single_dim=1536, pair_dim=640, num_heads=12, norm_eps=1e-5,
            use_pair_bias=arm != "single" and arm != "readout", use_rope=False, zero_output=False,
            compute_dtype=jnp.bfloat16, rngs=nnx.Rngs(0),
        )
        cls = DiagonalReadout if arm == "bias_diagonal" else PairToSingleReadout
        if arm in {"bias_poolnorm", "bias_cls"}:
            self.bridge = PairSummaryToSingleReadout(
                1536, 640, summary="cls" if arm == "bias_cls" else "mean",
                message_dim=96 if arm == "bias_cls" else 128,
                compute_dtype=jnp.bfloat16, rngs=nnx.Rngs(1))
        else:
            self.bridge = (cls(1536, 640, compute_dtype=jnp.bfloat16,
                               rngs=nnx.Rngs(1)) if arm in {"readout", "bias_readout", "bias_diagonal"} else None)

    def __call__(self, single, pair, mask):
        result = self.attention(single, pair, mask, single.shape[1])
        if self.bridge is not None:
            if isinstance(self.bridge, PairSummaryToSingleReadout):
                cls_mask = mask & (jnp.arange(mask.shape[1]) == mask.shape[1] - 1)
                result = result + self.bridge(pair, mask, cls_mask)
            else:
                result = result + self.bridge(pair, mask)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lengths", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--arms", nargs="+", default=["single", "bias", "readout"],
                        choices=["single", "bias", "readout", "bias_readout", "bias_diagonal", "bias_poolnorm", "bias_cls"])
    args = parser.parse_args()
    rows = []
    timing_rng = np.random.default_rng(42)
    for length in args.lengths:
        single = jax.random.normal(jax.random.key(2), (args.batch_size, length, 1536), dtype=jnp.bfloat16)
        pair = jax.random.normal(jax.random.key(3), (args.batch_size, length, length, 640), dtype=jnp.bfloat16)
        mask = jnp.ones((args.batch_size, length), bool)
        cases = []
        for arm in args.arms:
            model = Interface(arm)
            graph, params = nnx.split(model)
            def loss(p, s, z):
                out = nnx.merge(graph, p)(s, z, mask).astype(jnp.float32)
                return jnp.square(out).mean()
            fn = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2)))
            compiled = fn.lower(params, single, pair).compile()
            for _ in range(3):
                jax.block_until_ready(compiled(params, single, pair))
            memory = compiled.memory_analysis()
            cases.append((compiled, params, {
                "arm": arm, "length": length, "batch_size": args.batch_size,
                "compiler_memory_bytes": int(memory.argument_size_in_bytes + memory.temp_size_in_bytes
                                              + memory.output_size_in_bytes - memory.alias_size_in_bytes),
                "temporary_bytes": memory.temp_size_in_bytes,
                "params": sum(x.size for x in jax.tree.leaves(params))}))
        # Interleave conditions to reduce temporal load bias on a shared GPU.
        times = np.zeros((args.repeats, len(cases)))
        for repeat in range(args.repeats):
            for index in timing_rng.permutation(len(cases)):
                compiled, params, _ = cases[index]
                start = time.perf_counter()
                jax.block_until_ready(compiled(params, single, pair))
                times[repeat, index] = 1000 * (time.perf_counter() - start)
        for index, (_, _, row) in enumerate(cases):
            row.update(median_ms=float(np.median(times[:, index])),
                       p90_ms=float(np.quantile(times[:, index], .9)),
                       timings_ms=times[:, index].tolist())
            rows.append(row)
            print(json.dumps(rows[-1]), flush=True)
        del cases, compiled, fn, params, model
    result = {"device": str(jax.devices()[0]), "jax_version": jax.__version__,
              "scope": "interface only, forward + parameter/input backward; no optimizer or pair trunk",
              "timing_order": "Randomly interleaved per repeat; seed 42; three warmup calls per condition",
              "rows": rows}
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
