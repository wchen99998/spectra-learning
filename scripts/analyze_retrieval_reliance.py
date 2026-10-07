"""How much does a trained retrieval readout rely on the JEPA tokens?

The readout's per peak/fragment compatibility is S_pf = q_p.k_f + B_pf + delta_pf:
a learned JEPA-query/fragment-key term, the fixed-shape physical ion bonus B
(the "mass-matching kernel", one learned scale), and the cross-attention
residual. Three diagnostics on one trained checkpoint:

``score``  Re-score every candidate of each evaluation spectrum with
           inference-time ablations (no retraining) and report ranks:
             full                   the checkpoint as trained
             no_pair                pair-field term removed (pair-field models)
             no_adapter             cross-attention residual removed
             no_bonus               ion bonus removed (B = 0)
             bonus_only             q = 0, no adapter, no pair term: ion bonus
                                    with the learned GNN fragment prior pi_f
             bonus_uniform_prior    as bonus_only with uniform pi_f
             jepa_global_mean       every peak query = mean query over all
                                    evaluated peaks (no spectrum information)
             jepa_spectrum_mean     every peak query = its spectrum's mean
             jepa_peak_shuffle      queries permuted among a spectrum's peaks
             pretrained_encoder     joint checkpoints only: head unchanged,
                                    tokens from the pretrained encoder
           The jepa_* variants drop the pair term (it also reads JEPA states).
           Also reports encoder token drift (joint vs pretrained).

``grad``   Gradients of the full-pool cross-entropy (truth vs its complete
           formula pool, deterministic forward) on training batches: norms per
           parameter group, gradient/parameter ratios, the gradient on the JEPA
           tokens, and cross-batch coherence |mean g| / mean |g|. Repeated with
           the ion bonus removed at the same parameters.

``drift``  Relative weight change of the encoder (joint vs pretrained).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

VARIANTS = ("full", "no_pair", "no_adapter", "no_bonus", "bonus_only", "bonus_uniform_prior",
            "jepa_global_mean", "jepa_spectrum_mean", "jepa_peak_shuffle", "pretrained_encoder")
OFF = -1e4  # log_beta that switches the ion bonus off (exp underflows to 0 in float32)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("score", "grad", "drift", "merge"), required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--fingerprints", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pair-field", action="store_true")
    parser.add_argument("--tree", action="store_true")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--grad-batches", type=int, default=4, help="Batches of 128 training spectra")
    parser.add_argument("--micro-batch", type=int, default=4)
    cli = parser.parse_args(argv)
    if cli.mode == "merge":
        return cli, None
    from train_pairfield_retrieval import parse_args as recipe_args
    recipe = ["--inputs", str(cli.inputs), "--out", str(cli.out), "--stage", "eval", "--fingerprints",
              str(cli.fingerprints), "--eval-checkpoint", str(cli.checkpoint)]
    recipe += ["--pair-field"] * cli.pair_field + ["--tree"] * cli.tree
    return cli, recipe_args(recipe)


def merge(out: Path):
    """Combine score shards into per-variant metrics."""
    shards = [dict(np.load(p)) for p in sorted(out.glob("scores_shard*.npz"))]
    data = {k: np.concatenate([s[k] for s in shards]) for k in shards[0] if not k.endswith("/scores") and k != "score_offsets"}
    variants = [k.split("/")[0] for k in data if k.endswith("/greater")]
    chance = float(np.mean(1.0 / data["count"]))
    full_hit = float(np.mean(data["full/greater"] == 0))
    result = {"queries": len(data["rows"]), "chance_hit1": chance, "variants": {}}
    for v in variants:
        greater, equal = data[f"{v}/greater"], data[f"{v}/equal"]
        hit1 = float(np.mean(greater == 0))
        result["variants"][v] = {
            "hit1": hit1, "hit1_random_ties": float(np.mean((greater == 0) / (1 + equal))),
            "hit5": float(np.mean(greater < 5)), "hit10": float(np.mean(greater < 10)),
            "mrr": float(np.mean(1.0 / (1 + greater))),
            "share_of_full_above_chance": (float(np.mean((greater == 0) / (1 + equal))) - chance) / (full_hit - chance),
            "queries_with_truth_ties": float(np.mean(equal > 0)),
            "top1_agrees_with_full": float(np.mean(data[f"{v}/top"] == data["full/top"])),
            "spearman_with_full": float(np.nanmean(data[f"{v}/spearman"])),
            "within_pool_score_std": float(np.mean(data[f"{v}/std"])),
            "gained_vs_full": int(((greater == 0) & (data["full/greater"] > 0)).sum()),
            "lost_vs_full": int(((greater > 0) & (data["full/greater"] == 0)).sum())}
    drifts = [json.loads(p.read_text()) for p in sorted(out.glob("drift_shard*.json"))]
    if drifts and drifts[0]:
        total = {k: sum(d[k] for d in drifts) for k in drifts[0]}
        result["encoder_output_drift"] = {
            "token_relative_l2": float(np.sqrt(total["token_delta_sq"] / total["token_ref_sq"])),
            "token_mean_cosine": total["token_cos_sum"] / total["peaks"],
            "query_relative_l2": float(np.sqrt(total["query_delta_sq"] / total["query_ref_sq"])),
            "query_mean_cosine": total["query_cos_sum"] / total["peaks"], "peaks": total["peaks"]}
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main(argv=None):
    cli, args = parse_args(argv)
    if cli.mode == "merge":
        return merge(cli.out)
    import jax
    import jax.numpy as jnp
    from flax import serialization

    from atom_flare_data import batch_graphs
    from glacier_fragment_data import batch_fragments
    from run_retrieval_jax import load_data, metrics
    from trainable_jepa_peak import restore_peak
    from spectra_learning.probes.massspec import atom_flare_jax as core
    from spectra_learning.probes.massspec import glacier_ion_logmixture_jax as recipe_scoring
    from spectra_learning.probes.massspec import pair_field_jax as pf
    from spectra_learning.probes.massspec.gine_global_glacier_jax import GlobalGlacierGNN

    jax.config.update("jax_default_matmul_precision", "highest")
    cli.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    log = lambda *a: print(f"[{time.monotonic() - started:7.0f}s]", *a, flush=True)

    data, auxiliary = load_data(args)
    store, tokens, mask, fractions, smiles, candidates, truth_ids = data
    train, val, test = auxiliary[:3]
    bank = auxiliary[5]
    peak = restore_peak(args)
    gnn = GlobalGlacierGNN(args.hidden, args.d, args.dropout)
    compatibility = core.QueryKeyProjection(args.d)
    agreement = recipe_scoring.CrossAttentionAgreement(dim=args.d)

    def molecule_batch(ids, capacity, nodes=None, edges=None):
        graph = batch_graphs(store, ids, graph_capacity=capacity, node_capacity=nodes or capacity * args.atom_cap,
                             edge_capacity=edges or capacity * args.atom_cap * 3, atom_cap=args.atom_cap)
        graph.update(batch_fragments(store.fragments, ids, graph_capacity=capacity, atom_cap=args.atom_cap))
        mass = np.ones(capacity, np.float32)
        mass[:len(ids)] = store.fragments.total_mass[ids]
        graph["molecular_mass"] = mass
        return graph

    key = jax.random.PRNGKey(args.seed)
    key, kp, kg, ka, kf = jax.random.split(key, 5)
    example_ids = truth_ids[train[:4]].tolist()
    gp = gnn.init(kg, molecule_batch(example_ids, 4), train=False)
    pp = peak.init(kp, jnp.zeros((1, *tokens.shape[1:]), jnp.float32), train=False)
    ap = agreement.init(ka, jnp.zeros((1, 1)), jnp.ones((1, 1)), jnp.ones((1, 1), dtype=bool))
    params = {"peak": pp["params"], "gnn": gp["params"], "agreement": ap["params"],
              "compatibility": compatibility.init(jax.random.PRNGKey(args.seed), jnp.zeros((1, args.d)),
                                                  jnp.zeros((1, args.d)))["params"]}
    if args.pair_field:
        params["pair_field"] = pf.init_pair_field(kf, args.d)
    stats = gp["batch_stats"]
    del gp, pp, ap
    pretrained = jax.tree.map(np.asarray, params["peak"]["encoder"])
    state = serialization.msgpack_restore(cli.checkpoint.read_bytes())
    joint = bool(state["params"]["peak"].get("encoder"))
    if not joint:
        state["params"]["peak"]["encoder"] = serialization.to_state_dict(params["peak"]["encoder"])
    params = serialization.from_state_dict(params, state["params"])
    stats = serialization.from_state_dict(stats, state["stats"])
    del state
    log(f"checkpoint={cli.checkpoint} joint={joint} pair_field={args.pair_field} tree={args.tree}")

    def group_norms(tree):
        """Squared L2 norms per readable parameter group."""
        out = {}
        flat = jax.tree_util.tree_flatten_with_path(tree)[0]
        for path, value in flat:
            names = [str(getattr(p, "key", getattr(p, "idx", p))) for p in path]
            top = names[0]
            if top == "peak" and names[1] == "encoder":
                # encoder/<module>[/<index>] e.g. encoder/blocks/3
                sub = names[2:4] if len(names) > 3 and names[3].isdigit() else names[2:3]
                group = "encoder/" + "/".join(sub)
            elif top == "peak":
                group = "readout"
            elif top == "agreement":
                group = "agreement/" + names[1]
            else:
                group = top
            sq = float(np.square(np.asarray(value, np.float64)).sum())
            for g in {group, top if top != "peak" else ("encoder" if names[1] == "encoder" else "readout")}:
                out[g] = out.get(g, 0.0) + sq
        return out

    if cli.mode == "drift":
        if not joint:
            raise ValueError("Drift compares a joint checkpoint's encoder with the pretrained one")
        current = jax.tree.map(np.asarray, params["peak"]["encoder"])
        delta = jax.tree.map(lambda a, b: a.astype(np.float64) - b.astype(np.float64), current, pretrained)
        dn, pn = group_norms({"peak": {"encoder": delta}}), group_norms({"peak": {"encoder": pretrained}})
        result = {g: {"relative_l2": float(np.sqrt(dn[g] / max(pn[g], 1e-30))), "param_l2": float(np.sqrt(pn[g]))}
                  for g in sorted(pn)}
        (cli.out / "weight_drift.json").write_text(json.dumps(result, indent=2) + "\n")
        log(json.dumps(result["encoder"]))
        return

    @jax.jit
    def encode_molecules(gnn_params, stats, graph, compat):
        Z, gate = gnn.apply({"params": gnn_params, "batch_stats": stats}, graph, train=False)
        Z = compatibility.apply({"params": compat}, Z, method=compatibility.project_keys)
        subsets = pf.fragment_subsets(graph["member_mean"], graph["fragment_mask"]) if args.tree else None
        return Z.astype(jnp.float16), gate, subsets

    def encoded_tuple(encoder_params, packed):
        out = peak.encode(encoder_params, packed)
        return out if isinstance(out, tuple) else (out,)

    def readout(readout_params, encoded, train=False):
        out = peak.readout.apply({"params": readout_params}, *encoded, train=train)
        return (out[0], out[1]) if args.pair_field else (out, None)

    if cli.mode == "score":
        rows = {"val": val, "test": test}[cli.split][cli.shard::cli.num_shards]
        dev = jax.local_devices()[0]
        p_params, p_stats = jax.device_put(params, dev), jax.device_put(stats, dev)

        @jax.jit
        def encode_queries(peak_params, compat, packed):
            encoded = encoded_tuple(peak_params["encoder"], packed)
            P, psi = readout(peak_params["readout"], encoded)
            P = compatibility.apply({"params": compat}, P, method=compatibility.project_queries)
            return (P.astype(jnp.float16), None if psi is None else psi.astype(jnp.float16),
                    encoded[0].astype(jnp.float32))

        @jax.jit
        def query_score(agree, pair_params, P, pmask, psi, Z, fmask, gate, query, coordinates, subsets):
            return pf.score_query(P.astype(jnp.float32), pmask, Z.astype(jnp.float32), fmask, gate, query,
                                  coordinates, agree["log_sigma"], agree["log_beta"],
                                  cross_attention=agree["cross_attention"], psi=psi, pair_field=pair_params,
                                  subsets=subsets)

        needed = sorted({int(g) for r in rows for g in candidates[smiles[r]]})
        local_ids = {g: i for i, g in enumerate(needed)}
        cap = store.fragments.metadata["f_max"]
        Zall = np.zeros((len(needed), cap, args.d), np.float16)
        Mall = np.zeros((len(needed), cap), bool)
        Fall = np.zeros((len(needed), cap), np.float32)
        Gall = np.zeros((len(needed), cap), np.float32)
        Rall = np.zeros((len(needed), cap, cap), bool) if args.tree else None
        for start in range(0, len(needed), args.eval_batch):
            ids = needed[start:start + args.eval_batch]
            graph = molecule_batch(ids, args.eval_batch)
            Z, gate, subsets = encode_molecules(p_params["gnn"], p_stats, graph, p_params["compatibility"])
            n = len(ids)
            if args.tree:
                Rall[start:start + n] = np.asarray(subsets)[:n, :cap, :cap]
            Zall[start:start + n] = np.asarray(Z)[:n, :cap]
            Gall[start:start + n] = np.asarray(gate)[:n, :cap]
            Mall[start:start + n] = graph["fragment_mask"][:n, :cap]
            Fall[start:start + n] = graph["fragment_mass_frac"][:n, :cap]
        log(f"shard {cli.shard}/{cli.num_shards}: {len(rows)} rows, {len(needed)} candidate molecules encoded")

        # Peak queries (and pair states) for the checkpoint and, for joint checkpoints, the pretrained encoder.
        def all_queries(peak_params):
            Ps, psis, raws = [], [], []
            for start in range(0, len(rows), args.jepa_eval_batch):
                chunk = rows[start:start + args.jepa_eval_batch]
                packed = np.zeros((args.jepa_eval_batch, *tokens.shape[1:]), np.float32)
                packed[:len(chunk)] = tokens[chunk]
                P, psi, raw = encode_queries(peak_params, p_params["compatibility"], packed)
                Ps.append(np.asarray(P)[:len(chunk)])
                raws.append(np.asarray(raw, np.float16)[:len(chunk)])
                if psi is not None:
                    psis.append(np.asarray(psi)[:len(chunk)])
            return np.concatenate(Ps), (np.concatenate(psis) if psis else None), np.concatenate(raws)

        P_all, psi_all, raw_all = all_queries(p_params["peak"])
        drift = None
        if joint:
            P_pre, psi_pre, raw_pre = all_queries({**p_params["peak"], "encoder": jax.device_put(pretrained, dev)})
            m = mask[rows]
            a, b = raw_all[m].astype(np.float32), raw_pre[m].astype(np.float32)
            cos = (a * b).sum(-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-12)
            pa, pb = P_all[m].astype(np.float32), P_pre[m].astype(np.float32)
            pcos = (pa * pb).sum(-1) / np.maximum(np.linalg.norm(pa, axis=-1) * np.linalg.norm(pb, axis=-1), 1e-12)
            drift = {"token_delta_sq": float(np.square(a - b).sum()), "token_ref_sq": float(np.square(b).sum()),
                     "token_cos_sum": float(cos.sum()), "query_delta_sq": float(np.square(pa - pb).sum()),
                     "query_ref_sq": float(np.square(pb).sum()), "query_cos_sum": float(pcos.sum()),
                     "peaks": int(m.sum())}
            del raw_pre
        del raw_all
        valid_peaks = mask[rows]
        global_mean = P_all[valid_peaks].astype(np.float32).mean(0)
        rng = np.random.default_rng(1000 + cli.shard)

        agree = jax.tree.map(np.asarray, params["agreement"])
        no_bonus = {**agree, "log_beta": np.float32(OFF)}
        no_adapter = {**agree, "cross_attention": {**agree["cross_attention"],
                                                     "O": np.zeros_like(agree["cross_attention"]["O"])}}
        pair_params = params.get("pair_field")
        variants = [v for v in VARIANTS if (v != "no_pair" or args.pair_field) and (v != "pretrained_encoder" or joint)]
        out = {"rows": [], "count": [], "truth": []}
        for v in variants:
            out[f"{v}/greater"], out[f"{v}/equal"], out[f"{v}/top"] = [], [], []
            out[f"{v}/spearman"], out[f"{v}/std"] = [], []
        stored_scores = {v: [] for v in variants}

        def spearman(x, y):
            rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
            if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
                return np.nan
            return float(np.corrcoef(rx, ry)[0, 1])

        for q, row in enumerate(rows):
            pool_ids = candidates[smiles[row]]
            truth = int(np.flatnonzero(pool_ids == truth_ids[row])[0])
            ids = [local_ids[int(g)] for g in pool_ids]
            K = args.candidate_cap
            A = np.zeros((K, cap, args.d), np.float16); A[:len(ids)] = Zall[ids]
            M = np.zeros((K, cap), bool); M[:len(ids)] = Mall[ids]
            F = np.full((K, cap), -1, np.float32); F[:len(ids)] = Fall[ids]
            G = np.full((K, cap), -np.inf, np.float32); G[:len(ids)] = Gall[ids]
            G_uniform = np.where(M, 0.0, -np.inf).astype(np.float32)
            mass = np.ones(K, np.float32); mass[:len(ids)] = store.fragments.total_mass[pool_ids]
            R = None
            if args.tree:
                R = np.zeros((K, cap, cap), bool); R[:len(ids)] = Rall[ids]
            query = {k: np.asarray(v[row]) for k, v in fractions.items()}
            coords = {"fraction": F, "mass": mass}
            pm = mask[row]
            P, psi = P_all[q], None if psi_all is None else psi_all[q]
            zero = np.zeros_like(P)
            spec_mean = np.where(pm[:, None], P[pm].astype(np.float32).mean(0), 0).astype(np.float16)
            glob = np.where(pm[:, None], global_mean, 0).astype(np.float16)
            shuffled = P.copy()
            idx = np.flatnonzero(pm)
            shuffled[idx] = P[rng.permutation(idx)]
            calls = {
                "full": (agree, P, psi, G),
                "no_pair": (agree, P, None, G),
                "no_adapter": (no_adapter, P, psi, G),
                "no_bonus": (no_bonus, P, psi, G),
                "bonus_only": (no_adapter, zero, None, G),
                "bonus_uniform_prior": (no_adapter, zero, None, G_uniform),
                "jepa_global_mean": (agree, glob, None, G),
                "jepa_spectrum_mean": (agree, spec_mean, None, G),
                "jepa_peak_shuffle": (agree, shuffled, None, G),
            }
            if joint:
                calls["pretrained_encoder"] = (agree, P_pre[q], None if psi_pre is None else psi_pre[q], G)
            scores = {}
            for v in variants:
                a, Pv, psiv, Gv = calls[v]
                s = np.asarray(query_score(a, pair_params, Pv, pm, psiv, A, M, Gv, query, coords, R))[:len(ids)]
                if not np.isfinite(s).all():
                    raise FloatingPointError(f"Nonfinite score at row {row} variant {v}")
                scores[v] = s.astype(np.float64)
            out["rows"].append(row); out["count"].append(len(ids)); out["truth"].append(truth)
            for v in variants:
                s = scores[v]
                out[f"{v}/greater"].append(int((s > s[truth]).sum()))
                out[f"{v}/equal"].append(int((s == s[truth]).sum()) - 1)
                out[f"{v}/top"].append(int(np.argmax(s)))
                out[f"{v}/spearman"].append(spearman(scores["full"], s))
                out[f"{v}/std"].append(float(np.std(s)))
                stored_scores[v].append(s.astype(np.float32))
            if q % 100 == 0:
                hits = {v: np.mean(np.array(out[f"{v}/greater"]) == 0) for v in ("full", "bonus_only", "no_bonus")}
                log(f"row {q}/{len(rows)} " + " ".join(f"{k}={x:.3f}" for k, x in hits.items()))
        arrays = {k: np.asarray(v) for k, v in out.items()}
        offsets = np.concatenate([[0], np.cumsum(arrays["count"])])
        for v in variants:
            arrays[f"{v}/scores"] = np.concatenate(stored_scores[v])
        arrays["score_offsets"] = offsets
        np.savez(cli.out / f"scores_shard{cli.shard:02d}.npz", **arrays)
        (cli.out / f"drift_shard{cli.shard:02d}.json").write_text(json.dumps(drift) + "\n")
        log("done")
        return

    # ---- gradient diagnostics on training batches ----------------------------------------------
    pools = {int(r): np.concatenate([[truth_ids[r]], bank[r]]).astype(np.int32) for r in train}

    def pack(rows):
        ids, position = [], {}
        index = np.zeros((len(rows), args.candidate_cap), np.int32)
        valid = np.zeros_like(index, dtype=bool)
        for b, row in enumerate(rows):
            pool = pools[int(row)]
            for k, graph in enumerate(pool.tolist()):
                if graph not in position:
                    position[graph] = len(ids)
                    ids.append(graph)
                index[b, k] = position[graph]
            valid[b, :len(pool)] = True
        capacity = len(rows) * args.candidate_cap
        nodes = int(store.counts[ids].sum()) + 1
        edges = int(store.edge_counts[ids].sum()) + 1
        # Fixed capacities avoid recompiling per batch.
        graph = molecule_batch(ids, capacity, nodes=len(rows) * args.candidate_cap * 64,
                               edges=len(rows) * args.candidate_cap * 64 * 3)
        if nodes > len(rows) * args.candidate_cap * 64 or edges > len(rows) * args.candidate_cap * 192:
            raise ValueError("Graph capacity exceeded")
        return {"graph": graph, "tokens": np.asarray(tokens[rows], np.float32), "pmask": mask[rows],
                "query": {k: np.asarray(v[rows]) for k, v in fractions.items()}, "index": index, "valid": valid}

    def head_loss(head, floats, edge, batch, bonus_on):
        encoded = (*floats, edge) if args.pair_field else floats
        P, psi = readout(head["readout"], encoded)
        Z, gate = gnn.apply({"params": head["gnn"], "batch_stats": stats}, batch["graph"], train=False)
        P, Z = compatibility.apply({"params": head["compatibility"]}, P, Z)
        graph, index, valid = batch["graph"], batch["index"], batch["valid"]
        coordinates = {"fraction": graph["fragment_mass_frac"], "mass": graph["molecular_mass"]}
        agree = head["agreement"]
        log_beta = jnp.where(bonus_on, agree["log_beta"], OFF)
        subsets = pf.fragment_subsets(graph["member_mean"], graph["fragment_mask"])[index] if args.tree else None
        scores = pf.score_candidates(
            P, batch["pmask"], psi, Z[index], graph["fragment_mask"][index] & valid[..., None], gate[index],
            batch["query"], jax.tree.map(lambda v: v[index], coordinates), agree["log_sigma"], log_beta,
            cross_attention=agree["cross_attention"], pair_field=head.get("pair_field"), subsets=subsets)
        log_p = jax.nn.log_softmax(jnp.where(valid, scores / args.tau, -jnp.inf), axis=-1)
        return -log_p[:, 0].sum(), (jnp.exp(log_p[:, 0]).sum(), scores)

    @jax.jit
    def grads(encoder, head, batch, bonus_on):
        n = 2 if args.pair_field else 1
        full = encoded_tuple(encoder, batch["tokens"])
        floats, vjp = jax.vjp(lambda e: encoded_tuple(e, batch["tokens"])[:n], encoder)
        edge = full[2] if args.pair_field else None
        (loss, (p_truth, _)), (g_head, g_floats) = jax.value_and_grad(head_loss, argnums=(0, 1), has_aux=True)(
            head, floats, edge, batch, bonus_on)
        (g_encoder,) = vjp(g_floats)
        m = batch["pmask"][..., None].astype(jnp.float32)
        token_grad_sq = (jnp.square(g_floats[0]) * m).sum()
        token_sq = (jnp.square(floats[0]) * m).sum()
        return loss, p_truth, g_encoder, g_head, token_grad_sq, token_sq

    head = {"readout": params["peak"]["readout"], "gnn": params["gnn"],
            "compatibility": params["compatibility"], "agreement": params["agreement"]}
    if args.pair_field:
        head["pair_field"] = params["pair_field"]
    encoder = params["peak"]["encoder"]
    rng = np.random.default_rng(0)
    batches = [rng.choice(train, args.batch, replace=False) for _ in range(cli.grad_batches)]
    param_sq = group_norms({"peak": {"encoder": encoder, "readout": head["readout"]},
                            **{k: v for k, v in head.items() if k != "readout"}})
    report = {"param_l2": {g: float(np.sqrt(v)) for g, v in param_sq.items()}}
    for bonus_on in (True, False):
        per_batch, loss_sum, truth_sum, tok_g, tok = [], 0.0, 0.0, 0.0, 0.0
        for b, rows in enumerate(batches):
            total = None
            for start in range(0, len(rows), cli.micro_batch):
                batch = pack(rows[start:start + cli.micro_batch])
                loss, p_truth, g_enc, g_head, tg, tn = grads(encoder, head, batch, jnp.bool_(bonus_on))
                tree = {"peak": {"encoder": g_enc, "readout": g_head["readout"]},
                        **{k: v for k, v in g_head.items() if k != "readout"}}
                tree = jax.tree.map(lambda x: np.asarray(x, np.float64), tree)
                total = tree if total is None else jax.tree.map(np.add, total, tree)
                loss_sum += float(loss); truth_sum += float(p_truth); tok_g += float(tg); tok += float(tn)
            per_batch.append(jax.tree.map(lambda x: x / len(rows), total))
            log(f"bonus_on={bonus_on} batch {b + 1}/{len(batches)} loss={loss_sum / ((b + 1) * len(rows)):.4f}")
        norms = [group_norms(g) for g in per_batch]
        mean_grad = jax.tree.map(lambda *xs: sum(xs) / len(xs), *per_batch)
        mean_norm = group_norms(mean_grad)
        groups = sorted(norms[0])
        rows_seen = len(batches) * args.batch
        report["bonus_on" if bonus_on else "bonus_off"] = {
            "loss": loss_sum / rows_seen, "mean_p_truth": truth_sum / rows_seen,
            "token_grad_rms_per_peak_ratio": float(np.sqrt(tok_g / max(tok, 1e-30))),
            "groups": {g: {"grad_l2": float(np.mean([np.sqrt(n[g]) for n in norms])),
                           "grad_over_param": float(np.mean([np.sqrt(n[g]) for n in norms]) / max(np.sqrt(param_sq[g]), 1e-30)),
                           "coherence": float(np.sqrt(mean_norm[g]) / max(np.mean([np.sqrt(n[g]) for n in norms]), 1e-30))}
                       for g in groups}}
    (cli.out / "gradients.json").write_text(json.dumps(report, indent=2) + "\n")
    log(json.dumps({k: report[k]["groups"].get("encoder") for k in ("bonus_on", "bonus_off")}))


if __name__ == "__main__":
    main()
