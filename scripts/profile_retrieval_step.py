"""Time each phase of head-stage retrieval training steps (host packing vs device compute)."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax
import jax.numpy as jnp
import numpy as np

import run_retrieval_jax as runner
from train_global_glacier_jax import functions
from spectra_learning.probes.massspec import atom_flare_jax as model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--local-cache", default="")
    parser.add_argument("--matmul-precision", default="highest")
    cli = parser.parse_args()
    extra = ["--local-cache", cli.local_cache] if cli.local_cache else []
    args = runner.recipe_args(["--inputs", str(cli.inputs), "--out", "/tmp/profile_run", "--stage", "head",
                               "--matmul-precision", cli.matmul_precision, *extra])
    jax.config.update("jax_default_matmul_precision", args.matmul_precision)
    started = time.monotonic()
    data, auxiliary = runner.load_data(args)
    store, tokens, mask, fractions, _, _, truth_ids = data
    train, _, _, _, identity, bank, _, _ = auxiliary
    print("load_data seconds", round(time.monotonic() - started, 1), flush=True)
    peak, gnn, agreement, optimizer, train_step, *_ = functions(args)
    key, kp, kg, ka = jax.random.split(jax.random.PRNGKey(0), 4)
    graph = runner.graph_batch(store, truth_ids[train[:4]].tolist(), args, train=True)
    params = {"peak": peak.init(kp, jnp.zeros((1, *tokens.shape[1:])), train=False)["params"],
              "gnn": (g := gnn.init(kg, graph, train=False))["params"],
              "agreement": agreement.init(ka, jnp.zeros((1, 1)), jnp.ones((1, 1)), jnp.ones((1, 1), bool))["params"],
              "compatibility": model.QueryKeyProjection(args.d).init(
                  kp, jnp.zeros((1, args.d)), jnp.zeros((1, args.d)))["params"]}
    stats, opt_state = g["batch_stats"], optimizer.init(params)
    rng = np.random.default_rng(0)
    perm = train[rng.permutation(len(train))]
    timings = {name: [] for name in ("negatives", "graph_batch", "tokens", "device_step")}
    for step in range(cli.steps):
        rows = perm[step * args.batch:(step + 1) * args.batch]
        t0 = time.monotonic()
        ids, negative_indices, negative_valid = runner.sample_negatives(bank, rows, truth_ids, rng, args.negatives)
        t1 = time.monotonic()
        graph = runner.graph_batch(store, ids, args, train=True)
        t2 = time.monotonic()
        token_batch = np.asarray(tokens[rows], dtype=np.float32)
        t3 = time.monotonic()
        key, step_key = jax.random.split(key)
        params, stats, opt_state, loss = train_step(
            params, stats, opt_state, graph, token_batch, mask[rows],
            jax.tree.map(lambda value: value[rows], fractions), identity[rows],
            negative_indices, negative_valid, step_key)
        float(loss)
        t4 = time.monotonic()
        if step >= 3:  # exclude compilation
            for name, value in zip(timings, (t1 - t0, t2 - t1, t3 - t2, t4 - t3)):
                timings[name].append(value)
    summary = {name: round(float(np.mean(values)) * 1000, 1) for name, values in timings.items()}
    summary["serial_total_ms"] = round(sum(summary.values()), 1)

    def draw(step):
        rows = perm[step * args.batch:(step + 1) * args.batch]
        return rows, *runner.sample_negatives(bank, rows, truth_ids, rng, args.negatives)

    def pack(drawn):
        rows, ids, negative_indices, negative_valid = drawn
        return rows, runner.graph_batch(store, ids, args, train=True), np.asarray(tokens[rows], np.float32), \
            negative_indices, negative_valid

    steps = list(range(cli.steps, 3 * cli.steps))
    t0 = None
    for count, (rows, graph, token_batch, negative_indices, negative_valid) in enumerate(
            runner.prefetched(draw, pack, steps, depth=6)):
        if count == 3:
            t0 = time.monotonic()
        key, step_key = jax.random.split(key)
        params, stats, opt_state, loss = train_step(
            params, stats, opt_state, graph, token_batch, mask[rows],
            jax.tree.map(lambda value: value[rows], fractions), identity[rows],
            negative_indices, negative_valid, step_key)
        float(loss)
    summary["pipelined_ms_per_step"] = round((time.monotonic() - t0) / (len(steps) - 3) * 1000, 1)
    summary["matmul_precision"] = args.matmul_precision
    summary["graph_batch_bytes_mb"] = round(sum(v.nbytes for v in graph.values()) / 2**20, 1)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
