"""Full-pool retrieval training with structural risk and the pair field, data-parallel.

Objective. Test retrieval ranks the true molecule t among its supplied formula
pool C(t). Training uses that exact task: for each spectrum the candidate set is
{t} plus its complete training formula pool (no sampling, no in-batch
columns), with p_c = softmax_c(F(s, c) / tau). The loss is

    L = -log p_t + sum_c p_c d(c, t),        d(c, t) = 1 - Tanimoto(c, t),

the log-loss of the identity decision plus the expected structural distance
(Bayes risk) of the model's ranking distribution. The risk term rewards every
candidate the model ranks high in proportion to its structural similarity to
the truth, not only the top-1 identity; d(t, t) = 0 and d lies in [0, 1].
Tanimoto uses Morgan radius-2, 2048-bit fingerprints (RDKit defaults).

Model. The selected readout (docs/retrieval_recipe.md), optionally with the
pair field (spectra_learning/probes/massspec/pair_field_jax.py): the dual
encoder's pair states enter as the second-order term of the same
peak->fragment assignment energy. All optimisation settings are the recipe's:
head stage with frozen encoder (Adam 5e-4, 60 epochs, patience 8), then joint
stage (head 1e-5, encoder 3e-6, 8 epochs), batch 128, dropout 0.1, tau 0.05,
seed 0, selection by validation Hit@1.

Parallelism. One process per GPU (``srun`` under Slurm). The global batch of
128 spectra is split evenly across devices; gradients are summed and divided
by the global row count, BatchNorm running statistics are averaged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

STAGES = ("head", "joint", "eval")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--pair-field", action="store_true")
    parser.add_argument("--tree", action="store_true", help="Pair field with fragmentation-tree consistency")
    parser.add_argument("--fingerprints", type=Path, required=True, help="Directory with morgan2_2048.u8.npy")
    parser.add_argument("--head-checkpoint", type=Path, help="Joint stage: selected head best.msgpack")
    parser.add_argument("--eval-checkpoint", type=Path, help="Eval stage: best.msgpack to evaluate on test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--session-hours", type=float, default=0)
    parser.add_argument("--smoke", action="store_true", help="Pipeline test only: batch 8, 16 train / 8 eval rows, 1 epoch")
    cli = parser.parse_args(argv)
    paths = {k: Path(v) for k, v in json.loads(cli.inputs.read_text()).items()}
    if cli.tree and not cli.pair_field:
        raise ValueError("--tree extends the pair field")
    fixed = dict(
        d=256, hidden=496, dropout=0.1, batch=128, atom_cap=160, candidate_cap=256, tau=0.05,
        lr=5e-4 if cli.stage == "head" else 1e-5, jepa_lr=3e-6, epochs=60 if cli.stage == "head" else 8,
        patience=8, eval_batch=512, jepa_eval_batch=128, train_jepa=True, aux_spectrum=None,
        local_cache=None, lora_rank=16, lora_alpha=16.,
        encoder_adaptation="frozen" if cli.stage == "head" else "full",
    )
    if cli.smoke:
        fixed.update(batch=8, epochs=1, eval_batch=64, jepa_eval_batch=8)
    return SimpleNamespace(**vars(cli), **fixed, **paths)


def tanimoto_distance(fingerprints, truth: int, candidates: np.ndarray) -> np.ndarray:
    a = np.asarray(fingerprints[truth])
    b = np.asarray(fingerprints[candidates])
    both = np.bitwise_count(a & b).sum(-1)
    either = np.bitwise_count(a | b).sum(-1)
    return np.where(either > 0, 1.0 - both / np.maximum(either, 1), 0.0).astype(np.float32)


def main(argv=None):
    args = parse_args(argv)
    import jax
    if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        jax.distributed.initialize()
    elif int(os.environ.get("PF_NUM_PROCESSES", "1")) > 1:
        # One process per GPU without Slurm (Modal): the launcher pins each process's GPU.
        jax.distributed.initialize(os.environ["PF_COORDINATOR"], int(os.environ["PF_NUM_PROCESSES"]),
                                   int(os.environ["PF_PROCESS_ID"]))
    import jax.numpy as jnp
    import optax
    from flax import nnx, serialization
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as PS

    from atom_flare_data import batch_graphs
    from glacier_fragment_data import batch_fragments
    from run_retrieval_jax import load_data, metrics, prefetched, ranks_with_ties
    from train_global_glacier_jax import scale_by_group_learning_rate
    from trainable_jepa_peak import restore_peak
    from spectra_learning.probes.massspec import atom_flare_jax as core
    from spectra_learning.probes.massspec import glacier_ion_logmixture_jax as recipe_scoring
    from spectra_learning.probes.massspec import pair_field_jax as pf
    from spectra_learning.probes.massspec.gine_global_glacier_jax import GlobalGlacierGNN

    jax.config.update("jax_default_matmul_precision", "highest")
    pid, nproc, ndev = jax.process_index(), jax.process_count(), jax.device_count()
    if jax.local_device_count() != 1 or ndev != nproc or args.batch % ndev:
        raise ValueError(f"Expected one GPU per process and {args.batch} divisible by {ndev} devices")
    local_batch = args.batch // ndev
    main_process = pid == 0
    log = (lambda *a: print(*a, flush=True)) if main_process else (lambda *a: None)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    log(f"pair-field retrieval: stage={args.stage} pair_field={args.pair_field} devices={ndev} local_batch={local_batch}")

    data, auxiliary = load_data(args)
    store, tokens, mask, fractions, smiles, candidates, truth_ids = data
    train, val, test, ik14, identity, bank, peak_input, ion_provenance = auxiliary
    if args.smoke:
        train, val, test = train[:16], val[:8], test[:8]
    if tokens.shape[1] != mask.shape[1]:
        raise ValueError("Packed spectra and masks disagree")
    fingerprints = np.load(args.fingerprints / "morgan2_2048.u8.npy", mmap_mode="r")
    if len(fingerprints) != len(store.smiles):
        raise ValueError("Fingerprints are not aligned with the graph store")

    # Training candidate sets: truth first, then its complete formula pool.
    pools = [np.concatenate([[truth_ids[r]], bank[r]]).astype(np.int32) if truth_ids[r] >= 0 else None
             for r in range(len(smiles))]
    pool_nodes = np.array([store.counts[pools[r]].sum() for r in train])
    pool_edges = np.array([store.edge_counts[pools[r]].sum() for r in train])
    graph_capacity = local_batch * args.candidate_cap
    node_capacity = int(local_batch * pool_nodes.max())
    edge_capacity = int(local_batch * pool_edges.max())
    log(json.dumps({"train": len(train), "val": len(val), "test": len(test),
                    "mean_pool": float(np.mean([len(pools[r]) for r in train])),
                    "graph_capacity": graph_capacity, "node_capacity": node_capacity,
                    "edge_capacity": edge_capacity}))

    peak = restore_peak(args)
    frozen = nnx.to_pure_dict(peak.static) if args.encoder_adaptation == "frozen" else None
    gnn = GlobalGlacierGNN(args.hidden, args.d, args.dropout)
    compatibility = core.QueryKeyProjection(args.d)
    agreement = recipe_scoring.CrossAttentionAgreement(dim=args.d)

    def pack_rows(rows):
        """Rows (with -1 padding) -> one device's batch."""
        weight = (rows >= 0).astype(np.float32)
        rows = np.where(rows >= 0, rows, train[0])
        ids, position = [], {}
        index = np.zeros((len(rows), args.candidate_cap), np.int32)
        valid = np.zeros_like(index, dtype=bool)
        distance = np.zeros(index.shape, np.float32)
        for b, row in enumerate(rows):
            pool = pools[row]
            for k, graph in enumerate(pool.tolist()):
                if graph not in position:
                    position[graph] = len(ids)
                    ids.append(graph)
                index[b, k] = position[graph]
            valid[b, :len(pool)] = True
            distance[b, :len(pool)] = tanimoto_distance(fingerprints, pool[0], pool)
            distance[b, 0] = 0.0
        graph = batch_graphs(store, ids, graph_capacity=graph_capacity, node_capacity=node_capacity,
                             edge_capacity=edge_capacity, atom_cap=args.atom_cap)
        graph.update(batch_fragments(store.fragments, ids, graph_capacity=graph_capacity, atom_cap=args.atom_cap))
        mass = np.ones(graph_capacity, np.float32)
        mass[:len(ids)] = store.fragments.total_mass[ids]
        graph["molecular_mass"] = mass
        return {"graph": graph, "tokens": np.asarray(tokens[rows], np.float32), "pmask": mask[rows],
                "query": {k: np.asarray(v[rows]) for k, v in fractions.items()},
                "index": index, "valid": valid, "distance": distance, "weight": weight}

    # ---- parameters --------------------------------------------------------------------------
    key = jax.random.PRNGKey(args.seed)
    key, kp, kg, ka, kf = jax.random.split(key, 5)
    example = pack_rows(train[:local_batch])
    gp = gnn.init(kg, example["graph"], train=False)
    pp = peak.init(kp, jnp.zeros((1, *tokens.shape[1:]), jnp.float32), train=False)
    ap = agreement.init(ka, jnp.zeros((1, 1)), jnp.ones((1, 1)), jnp.ones((1, 1), dtype=bool))
    params = {"peak": pp["params"], "gnn": gp["params"], "agreement": ap["params"],
              "compatibility": compatibility.init(jax.random.PRNGKey(args.seed), jnp.zeros((1, args.d)),
                                                  jnp.zeros((1, args.d)))["params"]}
    if args.pair_field:
        params["pair_field"] = pf.init_pair_field(kf, args.d)
    stats = gp["batch_stats"]
    del gp, pp, ap
    if args.stage == "joint" and not args.resume:
        # Selected head (frozen-encoder stage) plus the pretrained encoder, now trainable.
        head = serialization.msgpack_restore(args.head_checkpoint.read_bytes())
        head["params"]["peak"]["encoder"] = serialization.to_state_dict(params["peak"]["encoder"])
        params = serialization.from_state_dict(params, head["params"])
        stats = serialization.from_state_dict(stats, head["stats"])
    if args.stage == "eval":
        state = serialization.msgpack_restore(args.eval_checkpoint.read_bytes())
        if not state["params"]["peak"].get("encoder"):
            # A head-stage checkpoint: its encoder is the frozen pretrained one.
            state["params"]["peak"]["encoder"] = serialization.to_state_dict(params["peak"]["encoder"])
        params = serialization.from_state_dict(params, state["params"])
        stats = serialization.from_state_dict(stats, state["stats"])

    if args.stage == "joint":
        optimizer = optax.chain(optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8, eps_root=0.0),
                                scale_by_group_learning_rate(args.jepa_lr, args.lr))
    else:
        optimizer = optax.adam(args.lr, b1=0.9, b2=0.999, eps=1e-8, eps_root=0.0)

    # ---- data-parallel training step ------------------------------------------------------------
    mesh = Mesh(np.array(jax.devices()), ("d",))
    replicated, sharded = NamedSharding(mesh, PS()), NamedSharding(mesh, PS("d"))

    def replicate(tree):
        return jax.tree.map(lambda x: jax.make_array_from_process_local_data(replicated, np.asarray(x)), tree)

    def shard(tree):
        return jax.tree.map(lambda x: jax.make_array_from_process_local_data(sharded, np.asarray(x)[None]), tree)

    def to_host(tree):
        return jax.tree.map(lambda x: np.asarray(x.addressable_data(0)) if isinstance(x, jax.Array) else x, tree)

    def encode_peaks_train(peak_params, batch, frozen_params, rng, train):
        extras = {"frozen": frozen_params} if frozen_params is not None else {}
        out = peak.apply({"params": peak_params}, batch["tokens"], train=train,
                         rngs={"dropout": rng} if train else None, **extras)
        return out if args.pair_field else (out, None)

    def local_loss(params, stats, frozen_params, batch, rng):
        peak_rng, gnn_rng = jax.random.split(rng)
        P, psi = encode_peaks_train(params["peak"], batch, frozen_params, peak_rng, True)
        (Z, gate), updated = gnn.apply({"params": params["gnn"], "batch_stats": stats}, batch["graph"],
                                       train=True, rngs={"dropout": gnn_rng}, mutable=["batch_stats"])
        P, Z = compatibility.apply({"params": params["compatibility"]}, P, Z)
        graph, index, valid = batch["graph"], batch["index"], batch["valid"]
        coordinates = {"fraction": graph["fragment_mass_frac"], "mass": graph["molecular_mass"]}
        agree = params["agreement"]
        subsets = pf.fragment_subsets(graph["member_mean"], graph["fragment_mask"])[index] if args.tree else None
        scores = pf.score_candidates(
            P, batch["pmask"], psi, Z[index], graph["fragment_mask"][index] & valid[..., None], gate[index],
            batch["query"], jax.tree.map(lambda v: v[index], coordinates), agree["log_sigma"],
            agree["log_beta"], cross_attention=agree["cross_attention"], pair_field=params.get("pair_field"),
            subsets=subsets)
        logits = scores / args.tau
        log_p = jax.nn.log_softmax(jnp.where(valid, logits, -jnp.inf), axis=-1)
        cross_entropy = -log_p[:, 0]
        risk = (jnp.exp(log_p) * jnp.where(valid, batch["distance"], 0.0)).sum(-1)
        w = batch["weight"]
        return ((cross_entropy + risk) * w).sum(), (updated["batch_stats"], (cross_entropy * w).sum(),
                                                   (risk * w).sum(), w.sum())

    def device_step(params, stats, opt_state, frozen_params, batch, rng):
        batch = jax.tree.map(lambda v: v[0], batch)
        rng = jax.random.fold_in(rng, jax.lax.axis_index("d"))
        (_, (new_stats, ce, risk, rows)), grads = jax.value_and_grad(local_loss, has_aux=True)(
            params, stats, frozen_params, batch, rng)
        rows = jax.lax.psum(rows, "d")
        grads = jax.tree.map(lambda g: jax.lax.psum(g, "d") / rows, grads)
        new_stats = jax.lax.pmean(new_stats, "d")
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, new_stats, opt_state, jax.lax.psum(ce, "d") / rows, jax.lax.psum(risk, "d") / rows

    train_step = jax.jit(jax.shard_map(
        device_step, mesh=mesh, in_specs=(PS(), PS(), PS(), PS(), PS("d"), PS()),
        out_specs=(PS(), PS(), PS(), PS(), PS()), check_vma=False), donate_argnums=(0, 2))

    # ---- evaluation (each process scores a slice of queries on its own GPU) ----------------------
    @jax.jit
    def encode_molecules(gnn_params, stats, graph, compat):
        Z, gate = gnn.apply({"params": gnn_params, "batch_stats": stats}, graph, train=False)
        Z = compatibility.apply({"params": compat}, Z, method=compatibility.project_keys)
        subsets = pf.fragment_subsets(graph["member_mean"], graph["fragment_mask"]) if args.tree else None
        return Z.astype(jnp.float16), gate, subsets

    @jax.jit
    def encode_queries(peak_params, compat, packed, frozen_params):
        P, psi = encode_peaks_train(peak_params, {"tokens": packed}, frozen_params, None, False)
        P = compatibility.apply({"params": compat}, P, method=compatibility.project_queries)
        return P.astype(jnp.float16), (None if psi is None else psi.astype(jnp.float16))

    @jax.jit
    def query_score(agree, pair_params, P, pmask, psi, Z, fmask, gate, query, coordinates, subsets):
        return pf.score_query(P.astype(jnp.float32), pmask, Z.astype(jnp.float32), fmask, gate, query,
                              coordinates, agree["log_sigma"], agree["log_beta"],
                              cross_attention=agree["cross_attention"], psi=psi, pair_field=pair_params,
                              subsets=subsets)

    def evaluate(host_params, host_stats, rows):
        local_rows = rows[pid::nproc]
        dev = jax.local_devices()[0]
        put = lambda tree: jax.device_put(tree, dev)
        p_params, p_stats = put(host_params), put(host_stats)
        frozen_local = put(frozen) if frozen is not None else None
        needed = sorted({int(g) for r in local_rows for g in candidates[smiles[r]]})
        local_ids = {g: i for i, g in enumerate(needed)}
        cap = store.fragments.metadata["f_max"]
        Zall = np.zeros((len(needed), cap, args.d), np.float16)
        Mall = np.zeros((len(needed), cap), bool)
        Fall = np.zeros((len(needed), cap), np.float32)
        Gall = np.zeros((len(needed), cap), np.float32)
        Rall = np.zeros((len(needed), cap, cap), bool) if args.tree else None
        for start in range(0, len(needed), args.eval_batch):
            ids = needed[start:start + args.eval_batch]
            graph = batch_graphs(store, ids, graph_capacity=args.eval_batch,
                                 node_capacity=args.eval_batch * args.atom_cap,
                                 edge_capacity=args.eval_batch * args.atom_cap * 3, atom_cap=args.atom_cap)
            graph.update(batch_fragments(store.fragments, ids, graph_capacity=args.eval_batch, atom_cap=args.atom_cap))
            graph["molecular_mass"] = np.ones(args.eval_batch, np.float32)
            Z, gate, subsets = encode_molecules(p_params["gnn"], p_stats, graph, p_params["compatibility"])
            n = len(ids)
            if args.tree:
                Rall[start:start + n] = np.asarray(subsets)[:n, :cap, :cap]
            Zall[start:start + n] = np.asarray(Z)[:n, :cap]
            Gall[start:start + n] = np.asarray(gate)[:n, :cap]
            Mall[start:start + n] = graph["fragment_mask"][:n, :cap]
            Fall[start:start + n] = graph["fragment_mass_frac"][:n, :cap]
        out = {"row": [], "opt": [], "stable": [], "top": [], "count": [], "tani1": [], "tani5": []}
        for start in range(0, len(local_rows), args.jepa_eval_batch):
            chunk = local_rows[start:start + args.jepa_eval_batch]
            packed = np.zeros((args.jepa_eval_batch, *tokens.shape[1:]), np.float32)
            packed[:len(chunk)] = tokens[chunk]
            P, psi = encode_queries(p_params["peak"], p_params["compatibility"], packed, frozen_local)
            P = np.asarray(P)
            psi = None if psi is None else np.asarray(psi)
            for q, row in enumerate(chunk):
                pool_ids = candidates[smiles[row]]
                truth = int(np.flatnonzero(pool_ids == truth_ids[row])[0])
                ids = [local_ids[int(g)] for g in pool_ids]
                K = args.candidate_cap
                A = np.zeros((K, cap, args.d), np.float16); A[:len(ids)] = Zall[ids]
                M = np.zeros((K, cap), bool); M[:len(ids)] = Mall[ids]
                F = np.full((K, cap), -1, np.float32); F[:len(ids)] = Fall[ids]
                G = np.full((K, cap), -np.inf, np.float32); G[:len(ids)] = Gall[ids]
                mass = np.ones(K, np.float32); mass[:len(ids)] = store.fragments.total_mass[pool_ids]
                query = {k: np.asarray(v[row]) for k, v in fractions.items()}
                R = None
                if args.tree:
                    R = np.zeros((K, cap, cap), bool); R[:len(ids)] = Rall[ids]
                scores = np.asarray(query_score(p_params["agreement"], p_params.get("pair_field"), P[q], mask[row],
                                                None if psi is None else psi[q], A, M, G, query,
                                                {"fraction": F, "mass": mass}, R))[:len(ids)]
                if not np.isfinite(scores).all():
                    raise FloatingPointError(f"Nonfinite retrieval score at row {row}")
                opt, stable, top = ranks_with_ties(scores, truth)
                order = np.argsort(-scores, kind="stable")[:5]
                similarity = 1.0 - tanimoto_distance(fingerprints, int(truth_ids[row]), pool_ids[order])
                for name, value in (("row", row), ("opt", opt), ("stable", stable), ("top", int(pool_ids[top])),
                                    ("count", len(ids)), ("tani1", similarity[0]), ("tani5", similarity.mean())):
                    out[name].append(value)
        from jax.experimental import multihost_utils
        length = int(np.ceil(len(rows) / nproc))
        local = np.full((7, length), np.nan)
        for i, name in enumerate(out):
            local[i, :len(out[name])] = out[name]
        gathered = np.asarray(multihost_utils.process_allgather(local)).reshape(nproc, 7, length)
        merged = np.concatenate([g[:, ~np.isnan(g[0])] for g in gathered], axis=1)
        order = np.argsort(merged[0], kind="stable")
        merged = merged[:, order]
        ranks = {"rows": merged[0].astype(np.int64), "ranks_optimistic": merged[1].astype(np.int64),
                 "ranks_stable": merged[2].astype(np.int64), "top_graph_ids": merged[3].astype(np.int64),
                 "candidate_counts": merged[4].astype(np.int64)}
        if not np.array_equal(ranks["rows"], np.sort(rows)):
            raise ValueError("Gathered evaluation rows are incomplete")
        result = {"optimistic": metrics(ranks["ranks_optimistic"]), "stable": metrics(ranks["ranks_stable"]),
                  "top1_tanimoto": float(merged[5].mean()), "top5_tanimoto": float(merged[6].mean())}
        return result, ranks

    def save(path, tree):
        if main_process:
            tmp = path.with_suffix(path.suffix + ".partial")
            tmp.write_bytes(serialization.to_bytes(tree))
            tmp.replace(path)

    def write_json(path, value):
        if main_process:
            path.write_text(json.dumps(value, indent=2, default=float) + "\n")

    if args.stage == "eval":
        result, ranks = evaluate(params, stats, test)
        if main_process:
            ranks.update(truth_smiles=smiles[ranks["rows"]], inchikey14=ik14[ranks["rows"]],
                         top_smiles=np.array([store.smiles[int(i)] for i in ranks["top_graph_ids"]]))
            np.savez(args.out / "test_ranks.npz", **ranks)
            result["checkpoint_sha256"] = hashlib.sha256(args.eval_checkpoint.read_bytes()).hexdigest()
            write_json(args.out / "metrics.json", result)
            log(json.dumps(result, indent=2))
        return

    manifest = {"script": "train_pairfield_retrieval.py", "args": {k: str(v) for k, v in vars(args).items()},
                "devices": ndev, "objective": "full-pool CE + expected (1 - Tanimoto) risk",
                "parameters": int(sum(np.size(x) for x in jax.tree.leaves(params)))}
    opt_state = optimizer.init(params)
    rng = np.random.default_rng(args.seed)
    history, best, best_epoch, since, first_epoch = [], -1.0, -1, 0, 0
    if args.resume:
        progress = json.loads((args.out / "progress.json").read_text())
        history, best, best_epoch, since = (progress[k] for k in ("history", "best", "best_epoch", "since"))
        state = serialization.from_bytes({"params": params, "stats": stats, "opt_state": opt_state, "key": key},
                                         (args.out / "last.msgpack").read_bytes())
        params, stats, opt_state, key = state["params"], state["stats"], state["opt_state"], state["key"]
        rng.bit_generator.state = progress["numpy_rng"]
        first_epoch = history[-1]["epoch"] + 1 if history else 0
    else:
        write_json(args.out / "manifest.json", manifest)
    if args.stage == "joint" and not args.resume:
        initial, _ = evaluate(params, stats, val)
        write_json(args.out / "initial_validation.json", initial)
        best = initial["optimistic"]["hit1"]
        save(args.out / "best.msgpack", {"params": params, "stats": stats})
        log(json.dumps({"initial_validation": initial}))

    params, stats, opt_state = replicate(params), replicate(stats), replicate(opt_state)
    frozen_global = replicate(frozen) if frozen is not None else None

    def save_progress():
        save(args.out / "last.msgpack", {"params": to_host(params), "stats": to_host(stats),
                                         "opt_state": to_host(opt_state), "key": np.asarray(key)})
        write_json(args.out / "progress.json", {"history": history, "best": best, "best_epoch": best_epoch,
                                                "since": since, "numpy_rng": rng.bit_generator.state})

    for epoch in range(first_epoch, args.epochs if since < args.patience else 0):
        epoch_start = time.monotonic()
        perm = train[rng.permutation(len(train))]
        offsets = [o for o in range(0, len(perm), args.batch) if len(perm[o:o + args.batch]) >= 4]

        def local_rows(offset):
            rows = np.full(args.batch, -1, np.int64)
            chunk = perm[offset:offset + args.batch]
            rows[:len(chunk)] = chunk
            return rows[pid * local_batch:(pid + 1) * local_batch]

        totals = np.zeros(3)
        for step, batch in enumerate(prefetched(local_rows, pack_rows, offsets, depth=6, workers=6)):
            key, step_key = jax.random.split(key)
            params, stats, opt_state, ce, risk = train_step(params, stats, opt_state, frozen_global,
                                                            shard(batch), replicate(step_key))
            if step % 50 == 0:
                ce, risk = float(ce), float(risk)
                if not np.isfinite(ce + risk):
                    raise FloatingPointError(f"Nonfinite loss at epoch {epoch} step {step}")
                totals += (ce, risk, 1)
                log(f"epoch={epoch} step={step}/{len(offsets)} ce={ce:.4f} risk={risk:.4f} "
                    f"seconds={time.monotonic() - epoch_start:.0f}")
        host_params, host_stats = to_host(params), to_host(stats)
        result, _ = evaluate(host_params, host_stats, val)
        record = {"epoch": epoch, "train_ce": totals[0] / max(totals[2], 1), "train_risk": totals[1] / max(totals[2], 1),
                  "seconds": time.monotonic() - epoch_start, "validation": result}
        history.append(record)
        if result["optimistic"]["hit1"] > best:
            best, best_epoch, since = result["optimistic"]["hit1"], epoch, 0
            save(args.out / "best.msgpack", {"params": host_params, "stats": host_stats})
        else:
            since += 1
        save_progress()
        log(json.dumps(record))
        if since >= args.patience:
            break
        elapsed = time.monotonic() - started
        if args.session_hours and elapsed + record["seconds"] > args.session_hours * 3600:
            break
    complete = since >= args.patience or len(history) >= args.epochs
    write_json(args.out / "validation_complete.json", {"complete": complete, "epochs": len(history),
                                                       "best_epoch": best_epoch, "best_validation_hit1": best})
    log("Validation selection complete" if complete else "Session checkpoint saved; resume to finish")


if __name__ == "__main__":
    main()
