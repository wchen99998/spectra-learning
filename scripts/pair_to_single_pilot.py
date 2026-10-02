"""Matched JAX pair->single pilots on the existing compact MAE/distogram path.

This is a small architecture experiment, not a production pretraining recipe.
Chemical labels are used only for frozen evaluation after self-supervised
training. All arms use the same examples, masks, backbone initialization, and
optimizer schedule. Run from the repository root with its .venv/bin/python.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from spectra_learning.models.model_jax import PeakSetJEPAJax
from spectra_learning.models.pair_to_single_jax import PairSummaryToSingleReadout, PairToSingleReadout
from spectra_learning.models.pairmixer_jax import (
    _apply_rope, _dot_precision, _linear_with_preferred_acc, _preferred_acc_dtype,
)


def mask_matched_donors(mask):
    """Next spectrum with exactly the same token mask, or self if unique."""
    row = jnp.arange(mask.shape[0])
    same = jnp.all(mask[:, None] == mask[None, :], axis=-1)
    offset = (row[None, :] - row[:, None]) % mask.shape[0]
    eligible = same & (offset > 0)
    donor = jnp.argmin(jnp.where(eligible, offset, mask.shape[0]), axis=1)
    return jnp.where(eligible.any(1), donor, row)


class MeanReadout(PairToSingleReadout):
    """Same output projections as the candidate; only routing is removed."""

    def pool(self, pair, mask):
        values = self.pair_norm(pair).astype(self.compute_dtype)
        mean = (values * mask[:, None, :, None]).sum(2)
        mean /= jnp.maximum(mask.sum(1)[:, None, None], 1)
        return jnp.broadcast_to(mean[:, :, None, :],
                                (*mean.shape[:2], self.head_weight.shape[0], mean.shape[-1]))


class DiagonalReadout(PairToSingleReadout):
    """Read each peak's triangle-updated self-pair at O(N) readout storage.

    The parameter layout matches the pooling arms for this controlled pilot;
    its unused score projection is retained only for that matched comparison.
    """

    def pool(self, pair, mask):
        idx = jnp.arange(pair.shape[1])
        diagonal = self.pair_norm(pair[:, idx, idx]).astype(self.compute_dtype)
        return jnp.broadcast_to(diagonal[:, :, None],
                               (*diagonal.shape[:2], self.head_weight.shape[0], diagonal.shape[-1]))


def pair_attention_weights(module, single, pair, mask, positions):
    """The existing compact attention's routing, reused for pair values."""
    batch, tokens, _ = single.shape
    qkv = _linear_with_preferred_acc(module.qkv, module.single_norm(single)).reshape(
        batch, tokens, 3, module.num_heads, module.head_dim)
    q, k, _ = jnp.moveaxis(qkv, 2, 0)
    q, k = module.q_norm(jnp.swapaxes(q, 1, 2)), module.k_norm(jnp.swapaxes(k, 1, 2))
    if module.use_rope:
        if positions is None:
            positions = jnp.broadcast_to(jnp.arange(tokens), (batch, tokens))
        q, k = _apply_rope(q, positions), _apply_rope(k, positions)
    bias = _linear_with_preferred_acc(module.pair_bias, module.pair_norm(pair))
    bias = jnp.where(mask[:, None, None, :], jnp.transpose(bias, (0, 3, 1, 2)).astype(jnp.float32), -jnp.inf)
    scores = jnp.einsum("...qd,...kd->...qk", q, k,
                        precision=_dot_precision(q.dtype),
                        preferred_element_type=_preferred_acc_dtype(q.dtype)).astype(jnp.float32)
    return jax.nn.softmax(scores / math.sqrt(module.head_dim) + bias, axis=-1).astype(q.dtype)


class SharedAttentionReadout(PairToSingleReadout):
    def __call__(self, pair, mask, attention):
        keys = mask[:, None, :, None]
        uniform = keys / jnp.maximum(keys.sum(2, keepdims=True), 1)
        weights = .5 * (jnp.transpose(attention, (0, 2, 3, 1)) + uniform)
        normalized = self.pair_norm(pair).astype(self.compute_dtype)
        pooled = jnp.einsum("bijh,bijc->bihc", weights.astype(self.compute_dtype), normalized,
                            precision=self.precision, preferred_element_type=jnp.float32).astype(self.compute_dtype)
        return self.project(pooled, mask)


class ReadoutBlock(nnx.Module):
    def __init__(self, base, bridge, shared_attention=False):
        self.base = base
        self.bridge = bridge
        self.shared_attention = shared_attention
        self.intervention = "intact"

    def fastmixer_compact_only_call(self, single, pair, mask, positions=None):
        single_input = single
        single, pair = self.base.fastmixer_compact_only_call(single, pair, mask, positions)
        if self.bridge is not None and self.intervention != "zero":
            if self.intervention == "shuffle":
                # Match the complete compact validity mask so donor padding
                # cannot masquerade as an information-content intervention.
                values = pair[mask_matched_donors(mask)]
            else:
                values = pair
            if self.shared_attention:
                attention = pair_attention_weights(self.base.single_attention, single_input, pair, mask, positions)
                single = single + self.bridge(values, mask, attention)
            elif isinstance(self.bridge, PairSummaryToSingleReadout):
                cls_mask = None if positions is None else positions == 31
                single = single + self.bridge(values, mask, cls_mask)
            else:
                single = single + self.bridge(values, mask)
        return single, pair


def build_model(arm: str, seed: int, dtype=jnp.bfloat16, *, use_cls_token: bool = True):
    if arm == "bias_cls" and not use_cls_token:
        raise ValueError("The CLS-pair readout requires CLS to be enabled")
    model = PeakSetJEPAJax(
        training_mode="mae", model_dim=128, predictor_dim=128,
        encoder_num_layers=4, encoder_num_heads=4,
        attention_mlp_multiple=8 / 3, feature_mlp_hidden_dim=256,
        encoder_fourier_mlp_hidden_dim=256, encoder_fourier_num_freqs=16,
        pairmixer_fourier_num_freqs=4, pairmixer_pair_dim=64,
        pairmixer_pair_feature_hidden_dim=128, pairmixer_block_type="fastmixer",
        pairmixer_transition_type="swiglu", encoder_use_pair_path=True,
        masked_latent_predictor_num_layers=2, masked_latent_predictor_num_heads=4,
        masked_token_input_mode="latent_token", num_peaks=31,
        encoder_use_cls_token=True, jepa_num_target_blocks=1,
        jepa_mae_mz_bin_size=2.0, mae_intensity_loss_weight=0.0,
        mae_loss_weight=0.7, distogram_loss_weight=0.3,
        target_projector_dim=-1, use_ema_teacher=False,
        autocast_dtype="bf16" if dtype == jnp.bfloat16 else "fp32",
        rngs=nnx.Rngs(seed),
    )
    if not use_cls_token:
        # Initialize with CLS first so removing its four parameters does not
        # shift the RNG stream of any shared backbone/predictor parameter.
        model.settings.encoder_use_cls_token = False
        model.encoder_use_cls_token = False
        model.encoder.use_cls_token = False
        model.encoder.cls_token = None
        model.encoder.cls_to_peak_pair_token = None
        model.encoder.peak_to_cls_pair_token = None
        model.encoder.cls_cls_pair_token = None
    extra = int(use_cls_token)
    model.set_fastmixer_capacities(19 + extra, 27 + extra, 8)
    blocks = []
    for i, base in enumerate(model.encoder.blocks):
        # Construct every backbone before altering routing to preserve all
        # shared parameters exactly across arms, including unused bias weights.
        base.single_attention.use_pair_bias = arm in {
            "bias", "bias_readout", "bias_diagonal", "bias_weighted", "bias_poolnorm", "bias_cls"}
        bridge = None
        if arm in {"mean", "readout", "bias_readout", "bias_diagonal", "bias_weighted"}:
            cls = {"mean": MeanReadout, "bias_diagonal": DiagonalReadout,
                   "bias_weighted": SharedAttentionReadout}.get(arm, PairToSingleReadout)
            bridge = cls(128, 64, num_heads=4, head_dim=16,
                         compute_dtype=dtype, rngs=nnx.Rngs(seed + 1000 + i))
        elif arm in {"bias_poolnorm", "bias_cls"}:
            bridge = PairSummaryToSingleReadout(
                128, 64, summary="cls" if arm == "bias_cls" else "mean",
                message_dim=48 if arm == "bias_cls" else 64,
                compute_dtype=dtype, rngs=nnx.Rngs(seed + 1000 + i))
        blocks.append(ReadoutBlock(base, bridge, shared_attention=arm == "bias_weighted"))
    model.encoder.blocks = nnx.List(blocks)
    return model


def mask_batch(data, indices, rng):
    batch = {k: v[indices] for k, v in data.items() if k in {
        "peak_mz", "peak_intensity", "peak_valid_mask", "precursor_mz"}}
    valid = batch["peak_valid_mask"]
    order = np.argsort(np.where(valid, rng.random(valid.shape), 2.0), axis=1)
    rank = np.argsort(order, axis=1)
    counts = valid.sum(1)
    context_n = np.maximum((counts * .6).astype(int), 1)
    target_n = np.minimum(np.maximum((counts * .25).astype(int), 1), 8)
    context = valid & (rank < context_n[:, None])
    target = valid & (rank >= context_n[:, None]) & (rank < (context_n + target_n)[:, None])
    batch["context_mask"] = context
    batch["target_masks"] = target[:, None]
    return {k: jnp.asarray(v) for k, v in batch.items()}


@nnx.jit
def step(model, optimizer, batch):
    def loss_fn(m):
        metrics = m.forward_mae(batch)
        return metrics["loss"], metrics
    (_, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return metrics


@nnx.jit
def validation(model, batch):
    return model.forward_mae(batch)


@nnx.jit
def encode(model, batch):
    num_peaks = batch["peak_mz"].shape[1]
    tokens, positions, valid = model.encoder.forward_compact(
        batch["peak_mz"], batch["peak_intensity"],
        valid_mask=batch["peak_valid_mask"],
        max_visible_tokens=num_peaks + int(model.encoder_use_cls_token),
        precursor_mz=batch["precursor_mz"])
    tokens = tokens.astype(jnp.float32)
    peaks = valid & (positions < num_peaks)
    mean = (tokens * peaks[..., None]).sum(1) / jnp.maximum(peaks.sum(1)[:, None], 1)
    if not model.encoder_use_cls_token:
        return mean
    cls = valid & (positions == num_peaks)
    return jnp.concatenate((mean, (tokens * cls[..., None]).sum(1)), axis=-1)


def prepare(args):
    import pyarrow.csv as csv
    from spectra_learning.data.massspec_targets import compute_maccs_fingerprint_bits_for_smiles

    source = np.load(args.inputs)
    data = {k: source[k] for k in ("peak_mz", "peak_intensity", "peak_valid_mask",
                                  "precursor_mz", "fold")}
    rows = csv.read_csv(args.tsv, parse_options=csv.ParseOptions(delimiter="\t"),
                       convert_options=csv.ConvertOptions(include_columns=["smiles", "inchikey"])).to_pydict()
    keys, first, mol = np.unique(np.asarray(rows["inchikey"]), return_index=True, return_inverse=True)
    fingerprints, valid = compute_maccs_fingerprint_bits_for_smiles([rows["smiles"][i] for i in first])
    if not valid.all():
        raise ValueError("Invalid molecular targets in pilot panel")
    data.update(molecule=mol, fingerprints=fingerprints, molecule_keys=keys)
    # Keep complete held-out folds, and a fixed train subset for frozen probes.
    rng = np.random.default_rng(20260930)
    data["probe_train"] = np.sort(rng.choice(np.flatnonzero(data["fold"] == "train"), 8192, replace=False))
    data["probe_val"] = np.sort(rng.choice(np.flatnonzero(data["fold"] == "val"), 4096, replace=False))
    data["probe_test"] = np.flatnonzero(data["fold"] == "test")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if np.intersect1d(mol[data["fold"] == left], mol[data["fold"] == right]).size:
            raise ValueError(f"Molecule overlap between {left} and {right}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **data)
    print(json.dumps({"rows": len(mol), "molecules": len(keys),
                      "folds": {f: int((data['fold'] == f).sum()) for f in ("train", "val", "test")}}))


def train(args):
    data = dict(np.load(args.inputs))
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    model = build_model(args.arm, args.seed, use_cls_token=not args.no_cls)
    total_params = sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param)))
    schedule = optax.warmup_cosine_decay_schedule(0, 3e-4, 100, args.steps, end_value=3e-5)
    optimizer = nnx.Optimizer(model, optax.chain(optax.clip_by_global_norm(1.0),
                              optax.adamw(schedule, b2=.95, weight_decay=.01)), wrt=nnx.Param)
    train_idx = np.flatnonzero((data["fold"] == "train") & (data["peak_valid_mask"].sum(1) >= 4))
    rng = np.random.default_rng(args.seed + 77)
    history = []
    started = time.perf_counter()
    steady_start = None
    for i in range(args.steps):
        idx = rng.choice(train_idx, args.batch_size, replace=False)
        metrics = step(model, optimizer, mask_batch(data, idx, rng))
        if i == 49:
            jax.block_until_ready(metrics)
            steady_start = time.perf_counter()
        if (i + 1) % 250 == 0 or i == 0:
            row = {k: float(v) for k, v in metrics.items() if k in {"loss", "mae_mz_loss", "distogram_loss"}}
            row.update(step=i + 1, seconds=time.perf_counter() - started)
            history.append(row)
            print(json.dumps(row), flush=True)
    jax.block_until_ready(metrics)
    train_seconds = time.perf_counter() - started
    steady_sps = (args.steps - 50) / (time.perf_counter() - steady_start) if steady_start else None
    flat = {"/".join(map(str, path)): np.asarray(value) for path, value in nnx.to_flat_state(nnx.state(model))}
    np.savez(destination / "model.npz", **flat)
    output = {"arm": args.arm, "seed": args.seed, "steps": args.steps, "batch_size": args.batch_size,
              "use_cls_token": model.encoder_use_cls_token,
              "params": total_params, "train_seconds": train_seconds,
              "steady_steps_per_second": steady_sps, "jax_version": jax.__version__,
              "device": str(jax.devices()[0]), "history": history, "validation": {}}
    val_idx = data["probe_val"][:1024]
    for intervention in ("intact", "zero", "shuffle"):
        if args.arm in {"bias", "none"} and intervention != "intact":
            continue
        for block in model.encoder.blocks:
            block.intervention = intervention
        val_rng = np.random.default_rng(42)
        values = []
        for start in range(0, len(val_idx), args.batch_size):
            m = validation(model, mask_batch(data, val_idx[start:start + args.batch_size], val_rng))
            values.append({k: float(v) for k, v in m.items() if k in {"loss", "mae_mz_loss", "distogram_loss"}})
        output["validation"][intervention] = {k: float(np.mean([r[k] for r in values])) for k in values[0]}
    for block in model.encoder.blocks:
        block.intervention = "intact"
    model.encoder.pairmixer_fast_max_visible_tokens = 31 + int(model.encoder_use_cls_token)
    indices = np.concatenate([data[f"probe_{fold}"] for fold in ("train", "val", "test")])
    arrays = []
    for start in range(0, len(indices), args.batch_size):
        idx = indices[start:start + args.batch_size]
        batch = {k: jnp.asarray(data[k][idx]) for k in ("peak_mz", "peak_intensity", "peak_valid_mask", "precursor_mz")}
        arrays.append(np.asarray(encode(model, batch)))
    np.savez(destination / "embeddings.npz", features=np.concatenate(arrays), indices=indices)
    (destination / "training.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output), flush=True)


def evaluate(args):
    from sklearn.metrics import roc_auc_score

    data = dict(np.load(args.inputs))
    destination = Path(args.output)
    stored = np.load(destination / "embeddings.npz")
    idx = stored["indices"]
    features = stored["features"].astype(np.float64)
    if args.readout == "mean":
        # CLS-enabled exports concatenate 128 peak-mean and 128 CLS channels;
        # no-CLS exports already contain only the 128 peak-mean channels.
        features = features[:, :128]
    folds = data["fold"][idx]
    y = data["fingerprints"][data["molecule"][idx]].astype(np.float64)
    tr, va, te = [folds == f for f in ("train", "val", "test")]
    means, scales = features[tr].mean(0), features[tr].std(0)
    x = (features - means) / np.maximum(scales, 1e-5)
    x = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    # Choose all hyperparameters on validation, score test exactly once.
    eligible = (y[tr].sum(0) >= 20) & ((1 - y[tr]).sum(0) >= 20)
    eligible &= (y[va].min(0) < y[va].max(0))
    gram, cross = x[tr].T @ x[tr] / tr.sum(), x[tr].T @ y[tr] / tr.sum()
    best = (-np.inf, None, None)
    for alpha in (1e-4, 1e-3, 1e-2, .1, 1., 10.):
        regularizer = np.eye(x.shape[1]) * alpha
        regularizer[-1, -1] = 0
        weights = np.linalg.solve(gram + regularizer, cross)
        auc = roc_auc_score(y[va][:, eligible], (x[va] @ weights)[:, eligible], average="macro")
        if auc > best[0]:
            best = (auc, alpha, weights)
    test_bits = eligible & (y[te].min(0) < y[te].max(0))
    predictions = x[te] @ best[2]
    test_auc = roc_auc_score(y[te][:, test_bits], predictions[:, test_bits], average="macro")
    # Untrained cosine retrieval of independent spectra from held-out molecules.
    # One fixed library spectrum per test molecule; all other spectra are queries.
    mol = data["molecule"][idx[te]]
    _, library = np.unique(mol, return_index=True)
    query = np.setdiff1d(np.arange(len(mol)), library)
    def retrieve(z):
        z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-8)
        ranks = []
        for start in range(0, len(query), 256):
            q = query[start:start + 256]
            sim = z[q] @ z[library].T
            correct = mol[q, None] == mol[library][None]
            pos = (sim * correct).sum(1)
            ranks.extend(1 + (sim > pos[:, None]).sum(1) + .5 * ((sim == pos[:, None]).sum(1) - 1))
        ranks = np.asarray(ranks)
        return {"top1": float(np.mean(ranks == 1)), "top10": float(np.mean(ranks <= 10)),
                "mrr": float(np.mean(1 / ranks)), "queries": len(query), "candidates": len(library)}
    results = {"maccs_test_auc": float(test_auc), "maccs_val_auc": float(best[0]),
               "readout": args.readout, "feature_dim": features.shape[1],
               "ridge_alpha": best[1], "evaluated_bits": int(test_bits.sum()),
               "retrieval_raw": retrieve(features[te]),
               "retrieval_centered": retrieve(features[te] - means),
               "retrieval_probe": retrieve(predictions)}
    # A separate molecule-candidate retrieval check: candidates are held-out
    # molecular MACCS vectors, not spectra or training molecules. This small
    # fingerprint retrieval protocol is not the repository's full GLACIER test.
    candidate_fp = y[te][library][:, eligible]
    candidate_fp /= np.maximum(np.linalg.norm(candidate_fp, axis=1, keepdims=True), 1e-8)
    predicted_fp = np.clip(predictions[:, eligible], 0, 1)
    predicted_fp /= np.maximum(np.linalg.norm(predicted_fp, axis=1, keepdims=True), 1e-8)
    ranks = []
    for start in range(0, len(predicted_fp), 256):
        scores = predicted_fp[start:start + 256] @ candidate_fp.T
        correct = mol[start:start + 256, None] == mol[library][None]
        positive = (scores * correct).sum(1)
        ranks.extend(1 + (scores > positive[:, None]).sum(1)
                     + .5 * ((scores == positive[:, None]).sum(1) - 1))
    ranks = np.asarray(ranks)
    results["retrieval_molecule_maccs"] = {
        "top1": float(np.mean(ranks == 1)), "top10": float(np.mean(ranks <= 10)),
        "mrr": float(np.mean(1 / ranks)), "queries": len(ranks), "candidates": len(library)}
    filename = "evaluation_mean.json" if args.readout == "mean" else "evaluation.json"
    (destination / filename).write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results), flush=True)


def restore_model(destination: Path):
    """Restore parameters and fixed Fourier state using the recorded variant."""
    metadata = json.loads((destination / "training.json").read_text())
    model = build_model(metadata["arm"], metadata["seed"],
                        use_cls_token=metadata.get("use_cls_token", True))
    saved = np.load(destination / "model.npz")
    nnx.update(model, nnx.from_flat_state(
        (path, jnp.asarray(saved["/".join(map(str, path))]))
        for path, _ in nnx.to_flat_state(nnx.state(model))
    ))
    return model


def interventions(args):
    """Re-evaluate saved models with mask-matched causal substitutions."""
    data = dict(np.load(args.inputs))
    destination = Path(args.output)
    model = restore_model(destination)
    val_idx = data["probe_val"][:1024]
    result = {}
    for intervention in ("intact", "zero", "shuffle"):
        for block in model.encoder.blocks:
            block.intervention = intervention
        rng = np.random.default_rng(42)
        rows = []
        for start in range(0, len(val_idx), args.batch_size):
            metrics = validation(model, mask_batch(data, val_idx[start:start + args.batch_size], rng))
            rows.append({k: float(v) for k, v in metrics.items()
                         if k in {"loss", "mae_mz_loss", "distogram_loss"}})
        result[intervention] = {k: float(np.mean([row[k] for row in rows])) for k in rows[0]}
    result["matching"] = "Cyclic donor with exactly equal compact token mask; singleton groups remain unchanged."
    (destination / "interventions.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "train", "evaluate", "interventions"))
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tsv")
    parser.add_argument("--arm", choices=("none", "bias", "mean", "readout", "bias_readout", "bias_diagonal", "bias_weighted", "bias_poolnorm", "bias_cls"), default="readout")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-cls", action="store_true",
                        help="Remove CLS and its pair row/column, preserving shared initialization")
    parser.add_argument("--readout", choices=("all", "mean"), default="all",
                        help="Evaluate exported features, or only their peak-mean channels")
    args = parser.parse_args()
    {"prepare": prepare, "train": train, "evaluate": evaluate,
     "interventions": interventions}[args.mode](args)


if __name__ == "__main__":
    main()
