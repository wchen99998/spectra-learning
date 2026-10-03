"""Global+GLACIER retrieval: joint training from initialization or historical two-stage training."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np

from atom_flare_data import GraphStore, batch_graphs
from glacier_fragment_data import FragmentStore
from ion_logmixture_data import load_ion_training_inputs
from prepare_atom_flare_jax import write_json
from train_global_glacier_jax import functions
import harness_outputs
from spectra_learning.probes.massspec import atom_flare_jax as model

def metrics(ranks: np.ndarray) -> dict[str, float | int]:
    return {'n': len(ranks), **{f'hit{k}': float((ranks <= k).mean()) for k in (1, 2, 5, 10, 20)}, 'mrr': float((1.0 / ranks).mean()), 'median_rank': float(np.median(ranks)), 'mean_rank': float(np.mean(ranks))}

def ranks_with_ties(scores: np.ndarray, truth: int) -> tuple[int, int, int]:
    optimistic = 1 + int((scores > scores[truth]).sum())
    stable = optimistic + int((scores[:truth] == scores[truth]).sum())
    return (optimistic, stable, int(np.argmax(scores)))

def sample_negatives(bank: list[np.ndarray], rows: np.ndarray, truth_ids: np.ndarray, rng: np.random.Generator, count: int) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Keep repeated paired truths; deduplicate only the appended decoy graphs."""
    graph_ids = truth_ids[rows].tolist()
    indices = np.zeros((len(rows), count), dtype=np.int32)
    valid = np.zeros_like(indices, dtype=bool)
    decoy_position: dict[int, int] = {}
    for b, row in enumerate(rows):
        decoys = bank[int(row)]
        if len(decoys) > count:
            decoys = decoys[rng.choice(len(decoys), count, replace=False)]
        for k, graph_id in enumerate(decoys):
            graph_id = int(graph_id)
            if graph_id not in decoy_position:
                decoy_position[graph_id] = len(graph_ids)
                graph_ids.append(graph_id)
            indices[b, k] = decoy_position[graph_id]
            valid[b, k] = True
    return (graph_ids, indices, valid)

def graph_batch(store, ids, args, *, train):
    graph_capacity = args.batch * (args.negatives + 1) if train else args.eval_batch
    graph = batch_graphs(store, ids, graph_capacity=graph_capacity, node_capacity=args.node_capacity if train else args.eval_batch * args.atom_cap, edge_capacity=args.edge_capacity if train else args.eval_batch * args.atom_cap * 3, atom_cap=args.atom_cap)
    from glacier_fragment_data import batch_fragments
    graph.update(batch_fragments(store.fragments, ids, graph_capacity=graph_capacity, atom_cap=args.atom_cap))
    mass = np.ones(graph_capacity, dtype=np.float32)
    mass[:len(ids)] = store.fragments.total_mass[ids]
    graph['molecular_mass'] = mass
    return graph

def evaluate(params, stats, rows, data, args, encode_molecules, encode_peaks, query_score, scored=None):
    """`scored`, when a list, receives (row, candidate graph ids, scores) per query."""
    if len(rows) > args.eval_chunk:
        start = time.monotonic()
        blocks = []
        for offset in range(0, len(rows), args.eval_chunk):
            _, block = _evaluate_candidates(params, stats, rows[offset:offset + args.eval_chunk], data, args, encode_molecules, encode_peaks, query_score, scored)
            blocks.append(block)
        ranks = {key: np.concatenate([block[key] for block in blocks]) for key in blocks[0]}
        return ({'optimistic': metrics(ranks['ranks_optimistic']), 'stable': metrics(ranks['ranks_stable']), 'seconds': time.monotonic() - start}, ranks)
    return _evaluate_candidates(params, stats, rows, data, args, encode_molecules, encode_peaks, query_score, scored)

def _evaluate_candidates(params, stats, rows, data, args, encode_molecules, encode_peaks, query_score, scored=None):
    store, tokens, mask, fractions, smiles, candidates, truth_ids = data
    all_opt, all_stable, top_ids, counts = ([], [], [], [])
    start_time = time.monotonic()
    needed = sorted({int(g) for row in rows for g in candidates[smiles[row]]})
    local_ids = {g: i for i, g in enumerate(needed)}
    cap = store.fragments.metadata['f_max']
    atoms = np.empty((len(needed), cap, args.d), dtype=np.float16)
    atom_mask = np.empty((len(needed), cap), dtype=bool)
    fraction_shape = (cap,)
    fraction_dtype = np.float32
    atom_fractions = np.empty((len(needed), *fraction_shape), dtype=fraction_dtype)
    gates = np.empty((len(needed), cap), dtype=np.float32)
    for start in range(0, len(needed), args.eval_batch):
        ids = needed[start:start + args.eval_batch]
        graph = graph_batch(store, ids, args, train=False)
        encoded = encode_molecules(params['gnn'], stats, graph, params.get('compatibility'))
        encoded, gate = encoded
        gates[start:start + len(ids)] = np.asarray(gate)[:len(ids)]
        atoms[start:start + len(ids)] = np.asarray(encoded)[:len(ids), :cap]
        atom_mask[start:start + len(ids)] = graph['fragment_mask'][:len(ids), :cap]
        atom_fractions[start:start + len(ids)] = graph['fragment_mass_frac'][:len(ids), :cap]
        if start % (args.eval_batch * 100) == 0:
            print(f'encoded candidates {start + len(ids)}/{len(needed)} seconds={time.monotonic() - start_time:.1f}', flush=True)
    for offset in range(0, len(rows), args.eval_chunk):
        chunk = rows[offset:offset + args.eval_chunk]
        peak_vectors = np.empty((len(chunk), tokens.shape[1], args.d), dtype=np.float16)
        peak_batch = args.jepa_eval_batch if getattr(args, 'train_jepa', False) else args.eval_batch
        for start in range(0, len(chunk), peak_batch):
            idx = chunk[start:start + peak_batch]
            xb = np.zeros((peak_batch, *tokens.shape[1:]), dtype=np.float32)
            xb[:len(idx)] = tokens[idx]
            mb = np.zeros(xb.shape[:2], dtype=bool)
            mb[:len(idx)] = mask[idx]
            peak_vectors[start:start + len(idx)] = np.asarray(encode_peaks(params['peak'], xb, params.get('compatibility'), mb))[:len(idx)]
        for q, row in enumerate(chunk):
            pool_ids = candidates[smiles[row]]
            pos = int(np.flatnonzero(pool_ids == truth_ids[row])[0])
            ids = [local_ids[int(g)] for g in pool_ids]
            A = np.zeros((args.candidate_cap, cap, args.d), dtype=np.float16)
            M = np.zeros((args.candidate_cap, cap), dtype=bool)
            F = np.full((args.candidate_cap, *fraction_shape), -1, dtype=fraction_dtype)
            A[:len(ids)], M[:len(ids)], F[:len(ids)] = (atoms[ids], atom_mask[ids], atom_fractions[ids])
            score_extras = ()
            gate = np.full((args.candidate_cap, cap), -np.inf, dtype=np.float32)
            gate[:len(ids)] = gates[ids]
            score_extras = (gate,)
            mass = np.ones(args.candidate_cap, dtype=np.float32)
            mass[:len(ids)] = store.fragments.total_mass[pool_ids]
            coordinates = {"fraction": F, "mass": mass}
            query_coordinates = jax.tree.map(lambda value: value[row], fractions)
            scores = np.asarray(query_score(params['agreement'], peak_vectors[q], mask[row], A, M, query_coordinates, coordinates, *score_extras))[:len(ids)]
            if not np.isfinite(scores).all():
                raise FloatingPointError(f'Nonfinite retrieval score at row {row}')
            opt, stable, top = ranks_with_ties(scores, pos)
            if scored is not None:
                scored.append((int(row), pool_ids, scores))
            all_opt.append(opt)
            all_stable.append(stable)
            top_ids.append(pool_ids[top])
            counts.append(len(ids))
        print(f'eval {offset + len(chunk)}/{len(rows)} hit1={np.mean(np.array(all_opt) == 1):.4f} seconds={time.monotonic() - start_time:.1f}', flush=True)
    opt, stable = (np.asarray(all_opt), np.asarray(all_stable))
    result = {'optimistic': metrics(opt), 'stable': metrics(stable), 'seconds': time.monotonic() - start_time}
    return (result, {'rows': rows, 'ranks_optimistic': opt, 'ranks_stable': stable, 'top_graph_ids': np.array(top_ids), 'candidate_counts': np.array(counts)})

RECIPE = "global_glacier_cross_attention_joint_4602_v1"
INPUT_NAMES = ("input_root", "prepared", "fragments", "peak_fractions", "jepa_checkpoint", "jepa_metadata", "jepa_data")


def recipe_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="JSON containing the seven prepared input paths")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=("train", "head", "joint", "eval"), required=True,
                        help="train: fresh head and pretrained JEPA updated jointly from the first batch")
    parser.add_argument("--head-checkpoint", type=Path, help="Selected frozen-token head for fresh joint adaptation")
    parser.add_argument("--encoder-adaptation", choices=("full", "frozen", "lora"), default="full",
                        help="Single-stage encoder update policy; evaluation must match training")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--session-hours", type=float, default=0, help="Checkpoint between epochs before this session limit")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=.1)
    parser.add_argument("--negatives", type=int, default=8,
                        help="Formula-pool decoys per training query (head only when not 8)")
    parser.add_argument("--patience", type=int, default=8, help="Early-stopping patience in epochs")
    parser.add_argument("--local-cache", type=Path,
                        help="Copy the frozen token array to this local disk once and read it from there")
    parser.add_argument("--matmul-precision", choices=("highest", "high", "default"), default="highest",
                        help="JAX default matmul precision; 'high' uses TF32 on NVIDIA GPUs")
    parser.add_argument("--peak-readout", choices=("mlp", "attention"), default="mlp",
                        help="Head stage only: per-token MLP (baseline) or contextual attention over peak tokens")
    parser.add_argument("--global-term", action="store_true",
                        help="Head stage only: add a zero-initialized global spectrum-molecule similarity")
    parser.add_argument("--reweight-power", type=float, default=0.,
                        help="Sample training spectra with (p_val/p_train)^power over parent-mass x adduct "
                             "bins; 1 matches the validation queries' covariates (no labels), 0 disables")
    parser.add_argument("--weight-decay", type=float, default=0.,
                        help="Head stage only: decoupled AdamW decay on matrix parameters")
    parser.add_argument("--aux-spectrum", type=Path,
                        help="Directory with row-aligned features.f16.npy [N,D] and available.npy [N]: a "
                             "precomputed second spectrum embedding scored against a pooled molecule vector")
    parser.add_argument("--snapshots", action="store_true",
                        help="Head stage only: keep every epoch's parameters under snapshots/ for ensembling")
    cli = parser.parse_args(argv)
    paths = json.loads(cli.inputs.read_text())
    if set(paths) != set(INPUT_NAMES):
        raise ValueError(f"Input JSON must contain exactly {INPUT_NAMES}")
    if cli.session_hours < 0 or not np.isfinite(cli.session_hours):
        raise ValueError("session-hours must be finite and nonnegative")
    if cli.stage == "joint" and not cli.resume and cli.head_checkpoint is None:
        raise ValueError("Fresh joint adaptation requires --head-checkpoint")
    if cli.stage != "joint" and cli.head_checkpoint is not None:
        raise ValueError("--head-checkpoint is only used by joint adaptation")
    if cli.stage == "eval" and cli.resume:
        raise ValueError("Evaluation loads best.msgpack, not a resumable optimizer")
    if cli.weight_decay and cli.stage != "head":
        raise ValueError("--weight-decay is supported for head training only")
    if cli.global_term and cli.stage != "head":
        raise ValueError("--global-term is supported for cached-token head training only")
    if cli.peak_readout != "mlp" and cli.stage != "head":
        raise ValueError("--peak-readout is supported for cached-token head training only")
    if cli.negatives != 8 and cli.stage != "head":
        raise ValueError("--negatives is supported for cached-token head training only")
    if cli.snapshots and cli.stage != "head":
        raise ValueError("--snapshots stores small head-only checkpoints")
    if cli.encoder_adaptation != "full" and cli.stage not in ("train", "eval"):
        raise ValueError("Encoder adaptation policies are supported by train/eval only")
    fixed = dict(
        d=256, hidden=496, gine_layers=3,
        batch=128, negative_bank_size=0, atom_cap=160,
        # Packed-graph capacities scale with the number of graphs per batch (8 decoys: 65536/147456).
        node_capacity=65536 * (cli.negatives + 1) // 9, edge_capacity=147456 * (cli.negatives + 1) // 9, eval_batch=512,
        eval_chunk=2000, candidate_cap=256, jepa_eval_batch=128,
        score_chunk=16, tau=.05,
        train_jepa=cli.stage != "head", lr=5e-4 if cli.stage in ("train", "head") else 1e-5,
        jepa_lr=1e-4 if cli.encoder_adaptation == "lora" else 3e-6,
        lora_rank=16, lora_alpha=16., epochs=60 if cli.stage in ("train", "head") else 8,
        molecule_encoder="gine_global_glacier", peak_input="raw", similarity="qk",
        fragment_readout="ion_logmixture", fragment_cross_attention=True,
        cross_attention_dim=32, reverse_loss_weight=1.0,
    )
    return SimpleNamespace(**vars(cli), **fixed, **{k: Path(v) for k, v in paths.items()})


def local_copy(source: Path, cache: Path | None) -> Path:
    """Sequentially copy a large read-only array to local disk; random reads are then fast."""
    if cache is None:
        return source
    import shutil
    target = cache / hashlib.sha256(str(source.resolve()).encode()).hexdigest()[:16] / source.name
    if not target.exists() or target.stat().st_size != source.stat().st_size:
        target.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        partial = target.with_suffix(".partial")
        with source.open("rb") as reader, partial.open("wb") as writer:
            shutil.copyfileobj(reader, writer, length=64 << 20)
        partial.replace(target)
        print(f"copied {source} to local disk in {time.monotonic() - started:.0f}s", flush=True)
    return target


def load_raw_tokens(root: Path, cache: Path | None = None):
    metadata = json.loads((root / "embedding_metadata.json").read_text())
    tokens = np.load(local_copy(root / "covariance_embeddings.float16.npy", cache), mmap_mode="r")
    mask = np.load(root / "data/token_valid_mask.npy")
    if (metadata["embedding"] != "valid_peak_tokens" or tokens.dtype != np.float16
            or list(tokens.shape) != metadata["embedding_shape"]
            or metadata["embedding_dim"] != tokens.shape[-1]
            or metadata["embedding_dtype"] != str(tokens.dtype) or metadata["samples"] != len(tokens)
            or mask.dtype != np.bool_ or mask.shape != (len(tokens), tokens.shape[1] + 1)
            or not mask[:, -1].all() or not mask[:, :-1].any(axis=1).all()):
        raise ValueError("Expected aligned physical raw tokens and one trailing valid CLS mask slot")
    return tokens, mask[:, :-1], {"kind": "raw", "shape": list(tokens.shape),
                                  "dtype": str(tokens.dtype), "embedding_metadata": metadata,
                                  "cls_used": False}


def load_data(args):
    if args.train_jepa:
        from trainable_jepa_peak import load_spectra
        tokens, mask, peak_input = load_spectra(args.input_root, args.jepa_data, args.jepa_metadata)
        if str(args.jepa_checkpoint) != peak_input["embedding_metadata"]["checkpoint"]:
            raise ValueError("JEPA initialization must match the input metadata's encoder")
    else:
        tokens, mask, peak_input = load_raw_tokens(args.input_root, args.local_cache)
    fractions = np.load(args.peak_fractions)
    with np.load(args.prepared / "rows.npz") as rows:
        smiles, ik14 = rows["smiles"], rows["inchikey14"]
        train, val, test = rows["train"], rows["val"], rows["test"]
    if len(smiles) != len(tokens) or len(ik14) != len(tokens):
        raise ValueError("Spectrum features and molecule identities are not row-aligned")
    fractions, ion_provenance = load_ion_training_inputs(args.input_root, peak_input, mask, fractions)
    if args.aux_spectrum:
        aux = np.load(args.aux_spectrum / "features.f16.npy", mmap_mode="r")
        aux_valid = np.load(args.aux_spectrum / "available.npy")
        if aux.shape[0] != len(tokens) or aux_valid.shape != (len(tokens),) or aux_valid.dtype != np.bool_:
            raise ValueError("Auxiliary spectrum features are not row-aligned")
        fractions = {**fractions, "aux": aux, "aux_valid": aux_valid}
    store = GraphStore(args.prepared / "graphs")
    store.fragments = FragmentStore(args.fragments)
    if (store.fragments.metadata["smiles_sha256"] != store.metadata["smiles_sha256"]
            or len(store.fragments) != len(store.smiles) or store.metadata["max_nodes"] > args.atom_cap):
        raise ValueError("Fragments must align with complete, untruncated molecular graphs")
    graph_lookup = {s: i for i, s in enumerate(store.smiles)}
    truth_ids = np.array([graph_lookup.get(s, -1) for s in smiles], dtype=np.int32)
    train = train[truth_ids[train] >= 0]
    if (truth_ids[np.concatenate([val, test])] < 0).any():
        raise ValueError("Evaluation truth missing from graph store")
    pool = json.loads((args.input_root / "msg15/MassSpecGym1.5_retrieval_candidates_formula.json").read_text())
    bank = [np.empty(0, dtype=np.int32) for _ in smiles]
    for row in train:
        # Exact supplied identity/order, with support over the entire formula pool.
        bank[row] = np.array([graph_lookup[s] for s in pool.get(smiles[row], [])
                              if s != smiles[row]], dtype=np.int32)
    candidates = {truth: np.array([graph_lookup[s] for s in pool[truth]], dtype=np.int32)
                  for truth in set(smiles[np.concatenate([val, test])])}
    if any(len(ids) > args.candidate_cap for ids in candidates.values()):
        raise ValueError("Candidate capacity does not cover the complete supplied pool")
    _, identity = np.unique(ik14, return_inverse=True)
    return (store, tokens, mask, fractions, smiles, candidates, truth_ids), (
        train, val, test, ik14, identity.astype(np.int32), bank, peak_input, ion_provenance)


MASS_BINS = np.array([200., 300., 400., 500., 600., 700.])


def covariate_weights(input_root: Path, train: np.ndarray, val: np.ndarray, power: float) -> np.ndarray:
    """Tempered importance weights (p_val(bin) / p_train(bin))^power over parent-mass x adduct bins."""
    import pyarrow.csv as csv
    table = csv.read_csv(input_root / "msg15/MassSpecGym1.5.tsv", parse_options=csv.ParseOptions(delimiter="\t"),
                         convert_options=csv.ConvertOptions(include_columns=["parent_mass", "adduct"]))
    mass = np.asarray(table["parent_mass"].to_pylist(), np.float64)
    sodium = np.asarray([a == "[M+Na]+" for a in table["adduct"].to_pylist()])
    cell = np.digitize(mass, MASS_BINS) * 2 + sodium
    cells = len(MASS_BINS) * 2 + 2
    p_train = np.bincount(cell[train], minlength=cells) / len(train)
    p_val = np.bincount(cell[val], minlength=cells) / len(val)
    weights = (p_val[cell[train]] / p_train[cell[train]]) ** power
    return weights / weights.sum()


def prefetched(sequential, parallel, items, depth, workers=4):
    """Yield parallel(sequential(item)) in order.

    `sequential` runs in item order on one producer thread (e.g. RNG draws); `parallel`
    runs ahead on a thread pool (NumPy packing and memory-mapped reads release the GIL).
    """
    import queue
    import threading
    from concurrent.futures import ThreadPoolExecutor
    buffer = queue.Queue(maxsize=depth)
    stop = threading.Event()
    pool = ThreadPoolExecutor(workers)

    def produce():
        try:
            for item in items:
                if stop.is_set():
                    return
                buffer.put((True, pool.submit(parallel, sequential(item))))
            buffer.put((False, None))
        except BaseException as error:  # surfaced on the consuming thread
            buffer.put((None, error))

    thread = threading.Thread(target=produce, daemon=True)
    thread.start()
    try:
        while True:
            ok, value = buffer.get()
            if ok is None:
                raise value
            if not ok:
                return
            yield value.result()
    finally:
        stop.set()
        while thread.is_alive():
            try:
                buffer.get_nowait()
            except queue.Empty:
                thread.join(timeout=0.1)
        pool.shutdown(wait=True, cancel_futures=True)


def save_checkpoint(path, state):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(serialization.to_bytes(state))
    temporary.replace(path)


def input_identity(args, store, peak_input, ion_provenance):
    def digest(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    return {"embedding_metadata": peak_input["embedding_metadata"], "graphs": store.metadata,
            "fragments": store.fragments.metadata,
            "ion_inputs": ion_provenance["query_array_sha256"],
            "spectrum_inputs": {name: peak_input[name] for name in
                ("metadata_sha256", "prepared_spectra_sha256", "preparation_metadata_sha256") if name in peak_input},
            "rows_sha256": digest(args.prepared / "rows.npz"),
            **({"aux_spectrum": {"metadata": json.loads((args.aux_spectrum / "metadata.json").read_text()),
                                 "features_sha256": digest(args.aux_spectrum / "features.f16.npy"),
                                 "available_sha256": digest(args.aux_spectrum / "available.npy")}}
               if args.aux_spectrum else {}),
            "pool_sha256": digest(args.input_root / "msg15/MassSpecGym1.5_retrieval_candidates_formula.json")}



SELECTED_INPUT_SHA256 = {
    "rows_sha256": "4b74f8c40ef75344c9ff84826c263a18d9f71889d30ef5c3418e3699a8cf9c3d",
    "pool_sha256": "454636d2346aebb3cd87edc2efc3fa1bf91eb8e0219edff5880f02aa537ac954",
}
SELECTED_CHECKPOINT_SHA256 = "324679e57439fd3294cce1cc379f1332a197598b1a0a15fe6fa5b35c7a6f6b99"


def validate_saved_run(saved, current_identity, args):
    if saved.get("recipe") == RECIPE:
        if args.stage == "eval" and not json.loads((args.out / "validation_complete.json").read_text())["complete"]:
            raise ValueError("Complete validation selection before held-out evaluation")
        expected_stages = ("train", "joint") if args.stage == "eval" else (args.stage,)
        if saved["input_identity"] != current_identity or saved["args"]["stage"] not in expected_stages:
            raise ValueError("Checkpoint stage or prepared inputs differ from the saved run")
        if saved["args"].get("encoder_adaptation", "full") != args.encoder_adaptation:
            raise ValueError("Encoder adaptation policy differs from the saved run")
        return
    # Read-only import of the published historical checkpoint, never another trial.
    if args.stage != "eval":
        raise ValueError("Historical checkpoints are supported only for selected-checkpoint evaluation")
    with (args.out / "best.msgpack").open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != SELECTED_CHECKPOINT_SHA256:
        raise ValueError("Historical checkpoint is not the selected 46.02% model")
    if any(current_identity[name] != value for name, value in SELECTED_INPUT_SHA256.items()):
        raise ValueError("Selected checkpoint rows or candidate pool identity differs")
    for name in ("graphs", "fragments"):
        if saved[name] != current_identity[name]:
            raise ValueError(f"Selected checkpoint input identity differs: {name}")
    if (saved["peak_input"]["embedding_metadata"] != current_identity["embedding_metadata"]
            or saved["ion_inputs"]["query_array_sha256"] != current_identity["ion_inputs"]):
        raise ValueError("Selected checkpoint spectrum input identity differs")
    if any(saved["peak_input"][name] != value for name, value in current_identity["spectrum_inputs"].items()):
        raise ValueError("Selected checkpoint prepared spectrum archive differs")
    for name in ("hidden", "d", "dropout", "batch", "negatives", "atom_cap", "tau", "lr", "jepa_lr"):
        if saved["args"][name] != getattr(args, name):
            raise ValueError(f"Selected checkpoint recipe differs: {name}")


def source_hashes():
    """Bind the actual imported local model, data and training implementation."""
    import sys
    root = Path(__file__).resolve().parents[1]
    paths = {Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if (path.suffix == ".py" and path.is_relative_to(root)
                    and ".venv" not in path.parts and path.is_file()):
                paths.add(path)
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def main(argv=None):
    args = recipe_args(argv)
    session_started = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage != "eval" and not args.resume and (args.out / "manifest.json").exists():
        raise ValueError("Existing run directory: use --resume or choose a fresh --out")
    jax.config.update("jax_default_matmul_precision", args.matmul_precision)
    print(f"{RECIPE}; JAX {jax.__version__}; devices={jax.devices()}", flush=True)
    data, auxiliary = load_data(args)
    store, tokens, mask, fractions, smiles, _, truth_ids = data
    train, val, test, ik14, identity, bank, peak_input, ion_provenance = auxiliary
    funcs = functions(args)
    peak, gnn, agreement, optimizer, train_step, encode_molecules, encode_peaks, query_score = funcs
    key, kp, kg, ka = jax.random.split(jax.random.PRNGKey(args.seed), 4)
    graph = graph_batch(store, truth_ids[train[:4]].tolist(), args, train=True)
    gp = gnn.init(kg, graph, train=False)
    pp = peak.init(kp, jnp.zeros((1, *tokens.shape[1:]), dtype=jnp.float32), train=False)
    ap = agreement.init(ka, jnp.zeros((1, 1)), jnp.ones((1, 1)), jnp.ones((1, 1), dtype=bool))
    params = {"peak": pp["params"], "gnn": gp["params"], "agreement": ap["params"]}
    if args.global_term:
        from train_global_glacier_jax import init_global_term
        params["agreement"] = {**params["agreement"], "global": init_global_term(args.d)}
    if args.aux_spectrum:
        from train_global_glacier_jax import init_aux_term
        params["agreement"] = {**params["agreement"], "aux": init_aux_term(
            jax.random.fold_in(ka, 1), fractions["aux"].shape[1], args.d, args.dropout)}
    stats = gp["batch_stats"]
    params["compatibility"] = model.QueryKeyProjection(args.d).init(
        jax.random.PRNGKey(args.seed), jnp.zeros((1, args.d)), jnp.zeros((1, args.d)))["params"]
    del gp, pp, ap
    current_identity = input_identity(args, store, peak_input, ion_provenance)
    head_initialization = None
    if args.resume or args.stage == "eval":
        saved = json.loads((args.out / "manifest.json").read_text())
        validate_saved_run(saved, current_identity, args)
        head_initialization = saved.get("head_initialization")
    elif args.stage == "joint":
        from initialize_glacier_head import restore_head
        from trainable_jepa_peak import validate_initial_tokens
        peak_input["initial_token_validation"] = validate_initial_tokens(
            peak, params["peak"], tokens, mask, args.input_root, train)
        params, stats, head_initialization = restore_head(
            args.head_checkpoint, params, stats, args, peak_input, store.metadata, store.fragments.metadata)
    manifest = {"recipe": RECIPE,
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "input_identity": current_identity, "peak_input": peak_input,
                "graphs": store.metadata, "fragments": store.fragments.metadata,
                "ion_inputs": ion_provenance, "head_initialization": head_initialization,
                "train_n": len(train), "val_n": len(val), "test_n": len(test),
                "parameters": sum(x.size for x in jax.tree.leaves(params)),
                "versions": {name: importlib.metadata.version(name) for name in
                             ("jax", "jaxlib", "flax", "optax", "numpy", "rdkit")},
                "selection_metric": "validation.optimistic.hit1", "test_unit": "spectrum",
                "source_sha256": source_hashes()}
    write_json(args.out / ("evaluation_manifest.json" if args.stage == "eval" else "manifest.json"), manifest)
    rng = np.random.default_rng(args.seed)
    fold = np.load(args.input_root / "data/fold.npy", allow_pickle=True).astype(str)
    selection = json.loads((args.prepared / "selection.json").read_text())
    rng.choice(np.flatnonzero(fold == "val"), selection["val_n"], replace=False)
    rng.choice(np.flatnonzero(fold == "test"), min(6000, int((fold == "test").sum())), replace=False)
    sample_weights = (covariate_weights(args.input_root, train, val, args.reweight_power)
                      if args.reweight_power else None)
    history, best, best_epoch, since, first_epoch = [], -1., -1, 0, 0
    opt_state = None if args.stage == "eval" else optimizer.init(params)
    if args.resume or args.stage == "eval":
        progress = json.loads((args.out / "progress.json").read_text())
        history, best, best_epoch, since = (progress[k] for k in ("history", "best", "best_epoch", "since"))
        if args.resume:
            state = jax.device_put(serialization.from_bytes(
                {"params": params, "stats": stats, "opt_state": opt_state, "key": key},
                (args.out / "last.msgpack").read_bytes()))
            params, stats, opt_state, key = (state[k] for k in ("params", "stats", "opt_state", "key"))
            del state
            first_epoch = history[-1]["epoch"] + 1 if history else 0
            rng.bit_generator.state = progress["numpy_rng"]
    def save_progress():
        save_checkpoint(args.out / "last.msgpack", {
            "params": params, "stats": stats, "opt_state": opt_state, "key": key})
        write_json(args.out / "progress.json", {"history": history, "best": best,
                   "best_epoch": best_epoch, "since": since, "numpy_rng": rng.bit_generator.state})
    if args.stage == "joint" and not args.resume:
        initial, ranks = evaluate(params, stats, val, data, args, encode_molecules, encode_peaks, query_score)
        write_json(args.out / "initial_validation.json", initial)
        np.savez(args.out / "initial_validation_ranks.npz", **ranks)
        best = initial["optimistic"]["hit1"]
        save_checkpoint(args.out / "best.msgpack", {"params": params, "stats": stats})
        save_progress()
    for epoch in range(first_epoch, 0 if args.stage == "eval" or since >= args.patience else args.epochs):
        start = time.monotonic()
        perm = (rng.choice(train, len(train), p=sample_weights) if args.reweight_power
                else train[rng.permutation(len(train))])
        total_loss, seen = 0., 0

        def draw(offset):
            # The only consumer of `rng` during the epoch, called in batch order.
            rows = perm[offset:offset + args.batch]
            return rows, *sample_negatives(bank, rows, truth_ids, rng, args.negatives)

        def pack(drawn):
            rows, ids, negative_indices, negative_valid = drawn
            return (rows, graph_batch(store, ids, args, train=True), np.asarray(tokens[rows], dtype=np.float32),
                    negative_indices, negative_valid)

        offsets = [offset for offset in range(0, len(perm), args.batch) if len(perm[offset:offset + args.batch]) >= 4]
        for offset, (rows, graph, token_batch, negative_indices, negative_valid) in zip(
                offsets, prefetched(draw, pack, offsets, depth=6), strict=True):
            key, step_key = jax.random.split(key)
            params, stats, opt_state, loss = train_step(
                params, stats, opt_state, graph, token_batch, mask[rows],
                jax.tree.map(lambda value: value[rows], fractions), identity[rows],
                negative_indices, negative_valid, step_key)
            loss = float(loss)
            if not np.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch={epoch} offset={offset}")
            total_loss += loss * len(rows)
            seen += len(rows)
            if offset % (args.batch * 100) == 0:
                print(f"epoch={epoch} rows={seen}/{len(train)} loss={loss:.5f}", flush=True)
        result, _ = evaluate(params, stats, val, data, args, encode_molecules, encode_peaks, query_score)
        row = {"epoch": epoch, "train_loss": total_loss / seen,
               "seconds": time.monotonic() - start, "validation": result}
        history.append(row)
        hit = result["optimistic"]["hit1"]
        if hit > best:
            best, best_epoch, since = hit, epoch, 0
            save_checkpoint(args.out / "best.msgpack", {"params": params, "stats": stats})
        else:
            since += 1
        if args.snapshots:
            (args.out / "snapshots").mkdir(exist_ok=True)
            save_checkpoint(args.out / f"snapshots/epoch_{epoch:03d}.msgpack", {"params": params, "stats": stats})
        save_progress()
        print(json.dumps(row), flush=True)
        if since >= args.patience:
            break
        if args.session_hours and time.monotonic() - session_started + time.monotonic() - start > args.session_hours * 3600:
            break
    if args.stage != "eval":
        complete = since >= args.patience or len(history) >= args.epochs
        write_json(args.out / "validation_complete.json", {
            "complete": complete, "test_evaluated": False, "epochs": len(history),
            "best_epoch": best_epoch, "best_validation_hit1": best})
        print("Validation selection complete" if complete else "Session checkpoint saved; resume to finish", flush=True)
        return
    best_state = jax.device_put(serialization.from_bytes(
        {"params": params, "stats": stats}, (args.out / "best.msgpack").read_bytes()))
    del params, stats
    scored = []
    result, ranks = evaluate(best_state["params"], best_state["stats"], test, data, args,
                             encode_molecules, encode_peaks, query_score, scored)
    ranks.update(truth_smiles=smiles[test], inchikey14=ik14[test],
                 top_smiles=np.array([store.smiles[int(i)] for i in ranks["top_graph_ids"]]))
    np.savez(args.out / "test_ranks.npz", **ranks)
    # --- write per-candidate retrieval table for downstream harness ---
    import pyarrow.csv as csv
    query_ids = np.asarray(csv.read_csv(
        args.input_root / "msg15/MassSpecGym1.5.tsv", parse_options=csv.ParseOptions(delimiter="\t"),
        convert_options=csv.ConvertOptions(include_columns=["identifier"]))["identifier"].to_pylist())
    if len(query_ids) != len(smiles):
        raise ValueError("MassSpecGym TSV is not row-aligned with the prepared rows")
    hits_out, n_rows = harness_outputs.write_retrieval_hits(args.out, (
        (query_ids[row], smiles[row], [store.smiles[int(g)] for g in pool_ids], scores)
        for row, pool_ids, scores in scored))
    print(f"saved -> {hits_out} ({n_rows} rows)", flush=True)
    with (args.out / "best.msgpack").open("rb") as stream:
        checkpoint_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    result.update(best_epoch=best_epoch, best_validation_hit1=best, history=history,
                  checkpoint_sha256=checkpoint_sha256)
    write_json(args.out / "metrics.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
