"""Dump per-candidate validation and test scores of one selected retrieval checkpoint.

Reads the run directory without writing to it; scores go to a separate output directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np

import run_retrieval_jax as runner
from train_global_glacier_jax import functions
from spectra_learning.probes.massspec import atom_flare_jax as model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True, help="Selected run directory holding best.msgpack")
    parser.add_argument("--encoder-adaptation", choices=("full", "frozen", "lora"), default="full")
    parser.add_argument("--stage", choices=("eval", "head"), default="eval",
                        help="head: cached-token head run (encoder not restored)")
    parser.add_argument("--checkpoints", nargs="+", default=["best.msgpack"],
                        help="Paths relative to --run; each writes to --dest/<stem> when several are given")
    parser.add_argument("--average", action="store_true",
                        help="Dump one model whose parameters and batch statistics average --checkpoints")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dest", type=Path, required=True)
    cli = parser.parse_args()
    saved_args = json.loads((cli.run / "manifest.json").read_text())["args"]
    args = runner.recipe_args(["--inputs", str(cli.inputs), "--out", str(cli.run), "--stage", cli.stage,
                               "--encoder-adaptation", cli.encoder_adaptation, "--seed", str(cli.seed),
                               "--peak-readout", saved_args.get("peak_readout", "mlp"),
                               *(["--global-term"] if saved_args.get("global_term") else []),
                               *(["--aux-spectrum", saved_args["aux_spectrum"]] if saved_args.get("aux_spectrum") else [])])
    jax.config.update("jax_default_matmul_precision", "highest")
    data, auxiliary = runner.load_data(args)
    store, tokens, _, fractions, *_, truth_ids = data
    train, *_, peak_input, ion_provenance = auxiliary
    peak, gnn, agreement, _, _, encode_molecules, encode_peaks, query_score = functions(args)
    _, kp, kg, ka = jax.random.split(jax.random.PRNGKey(args.seed), 4)
    graph = runner.graph_batch(store, truth_ids[train[:4]].tolist(), args, train=True)
    gp = gnn.init(kg, graph, train=False)
    pp = peak.init(kp, jnp.zeros((1, *tokens.shape[1:]), dtype=jnp.float32), train=False)
    ap = agreement.init(ka, jnp.zeros((1, 1)), jnp.ones((1, 1)), jnp.ones((1, 1), dtype=bool))
    params = {"peak": pp["params"], "gnn": gp["params"], "agreement": ap["params"],
              "compatibility": model.QueryKeyProjection(args.d).init(
                  jax.random.PRNGKey(args.seed), jnp.zeros((1, args.d)), jnp.zeros((1, args.d)))["params"]}
    if args.global_term:
        from train_global_glacier_jax import init_global_term
        params["agreement"] = {**params["agreement"], "global": init_global_term(args.d)}
    if args.aux_spectrum:
        from train_global_glacier_jax import init_aux_term
        params["agreement"] = {**params["agreement"], "aux": init_aux_term(
            jax.random.fold_in(ka, 1), fractions["aux"].shape[1], args.d, args.dropout)}
    stats = gp["batch_stats"]
    del gp, pp, ap
    saved = json.loads((args.out / "manifest.json").read_text())
    runner.validate_saved_run(saved, runner.input_identity(args, store, peak_input, ion_provenance), args)
    template = {"params": params, "stats": stats}
    del params, stats
    states = (serialization.from_bytes(template, (args.out / c).read_bytes()) for c in cli.checkpoints)
    if cli.average:
        states = iter([jax.tree.map(lambda *leaves: np.mean(leaves, axis=0).astype(leaves[0].dtype), *states)])
    for checkpoint, state in zip(["average"] if cli.average else cli.checkpoints, states, strict=True):
        dest = cli.dest if cli.average or len(cli.checkpoints) == 1 else cli.dest / Path(checkpoint).stem
        if cli.average:
            dest.mkdir(parents=True, exist_ok=True)
            runner.save_checkpoint(dest / "average.msgpack", state)
        dump(jax.device_put(state), dest, args, data, auxiliary, encode_molecules, encode_peaks, query_score)


def dump(state, dest, args, data, auxiliary, encode_molecules, encode_peaks, query_score) -> None:
    _, _, _, _, smiles, candidates, truth_ids = data
    _, val, test, ik14, *_ = auxiliary
    dest.mkdir(parents=True, exist_ok=True)
    recorded: list[np.ndarray] = []

    def recording_score(*arguments):
        scores = np.asarray(query_score(*arguments))
        recorded.append(scores)
        return scores

    for split, rows in (("val", val), ("test", test)):
        recorded.clear()
        result, ranks = runner.evaluate(state["params"], state["stats"], rows, data, args,
                                        encode_molecules, encode_peaks, recording_score)
        counts = ranks["candidate_counts"]
        truth_position = np.array([int(np.flatnonzero(candidates[smiles[r]] == truth_ids[r])[0]) for r in rows])
        np.savez(dest / f"{split}_scores.npz", rows=ranks["rows"], counts=counts,
                 truth_position=truth_position, inchikey14=ik14[rows],
                 scores=np.concatenate([s[:c] for s, c in zip(recorded, counts, strict=True)]).astype(np.float32),
                 ranks_stable=ranks["ranks_stable"])
        runner.write_json(dest / f"{split}_metrics.json", result)
        print(split, json.dumps(result["stable"]), flush=True)


if __name__ == "__main__":
    main()
