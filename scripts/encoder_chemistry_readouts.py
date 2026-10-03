"""Small frozen-token chemistry readouts, fitted with direct JAX/Optax loops.

Both readouts normalize each token with the same affine-free LayerNorm, then use
the same linear classifier. Attention adds four learned query vectors, averages
their pooled vectors, and introduces no Q/K/V projections or wider classifier.
Its attention weights describe this readout; they are not causal explanations
of the encoder or assignments of molecular labels to peaks/fragments.

The supplied boolean mask is authoritative. Primary experiments select physical
peaks and explicitly exclude the trailing CLS token; callers may select CLS in a
separate declared experiment. Invalid tokens are zeroed before normalization so
their values and gradients cannot influence predictions. Labels enter only the
training loss and evaluation, never token inputs or attention masks.

Training uses class-balanced BCE for nonconstant train targets. Sigmoid outputs
are scores, not calibrated prevalence probabilities. Epoch and learning rate are
selected using validation macro AUROC; the test fold is evaluated only after
selection. Metrics use sklearn's tie-correct AUROC and balanced accuracy on
targets with at least min_count positives and negatives in both train and the
evaluated fold. No scaling statistics or targets are fitted on validation/test.
"""

from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


KINDS = ("mean", "attention")
FOLDS = ("train", "val", "test")


def init_readout(
    key: jax.Array, width: int, num_targets: int = 188,
    kind: str = "mean", num_queries: int = 4,
) -> dict[str, jax.Array]:
    if kind not in KINDS:
        raise ValueError(f"Unknown readout kind: {kind}")
    if min(width, num_targets, num_queries) < 1:
        raise ValueError("Readout dimensions must be positive")
    classifier_key, query_key = jax.random.split(key)
    params = {
        "weight": jax.random.normal(classifier_key, (width, num_targets)) / math.sqrt(width),
        "bias": jnp.zeros((num_targets,), dtype=jnp.float32),
    }
    if kind == "attention":
        params["queries"] = jax.random.normal(query_key, (num_queries, width))
    return params


def normalized_tokens(tokens: jax.Array, mask: jax.Array) -> jax.Array:
    """Affine-free token LayerNorm; padded NaNs/values do not enter reductions."""
    x = jnp.where(mask[..., None], tokens.astype(jnp.float32), 0.)
    centered = x - jnp.mean(x, axis=-1, keepdims=True)
    return centered * jax.lax.rsqrt(jnp.mean(centered ** 2, axis=-1, keepdims=True) + 1e-5)


def apply_readout(
    params: dict[str, jax.Array], tokens: jax.Array, mask: jax.Array, kind: str,
) -> tuple[jax.Array, jax.Array | None]:
    """Pure differentiable decoder; returns logits and optional [B,Q,T] weights.

    ``kind`` is static under JIT. For an empty mask, the pooled vector is zero
    and the prediction is the learned classifier bias; attention weights are 0.
    """
    if kind not in KINDS:
        raise ValueError(f"Unknown readout kind: {kind}")
    if tokens.ndim != 3 or mask.shape != tokens.shape[:2] or mask.dtype != jnp.bool_:
        raise ValueError("Expected tokens [B,T,D] and boolean mask [B,T]")
    if params["weight"].shape[0] != tokens.shape[-1]:
        raise ValueError("Token width differs from the fitted readout")
    x = normalized_tokens(tokens, mask)
    attention = None
    if kind == "mean":
        pooled = jnp.sum(x, axis=1) / jnp.maximum(mask.sum(axis=1, keepdims=True), 1)
    else:
        scores = jnp.einsum("btd,qd->bqt", x, params["queries"]) / math.sqrt(x.shape[-1])
        attention = jax.nn.softmax(jnp.where(mask[:, None, :], scores, -1e30), axis=-1)
        attention = jnp.where(mask[:, None, :], attention, 0.)
        pooled = jnp.einsum("bqt,btd->bqd", attention, x).mean(axis=1)
    return pooled @ params["weight"] + params["bias"], attention


def binary_metrics(
    logits: np.ndarray, labels: np.ndarray, target_names: list[str],
    *, train_positive: np.ndarray, train_negative: np.ndarray, min_count: int = 5,
) -> dict[str, Any]:
    """Tie-correct molecular-target metrics with explicit per-target support."""
    logits, labels = np.asarray(logits), np.asarray(labels)
    if logits.shape != labels.shape or logits.ndim != 2 or len(target_names) != logits.shape[1]:
        raise ValueError("Predictions, labels and target names must align")
    if not np.isfinite(logits).all():
        raise ValueError("Nonfinite readout predictions")
    positive = labels.sum(axis=0)
    negative = len(labels) - positive
    supported = ((positive >= min_count) & (negative >= min_count)
                 & (train_positive >= min_count) & (train_negative >= min_count))
    records = []
    for index, name in enumerate(target_names):
        record = {"name": name, "positive": int(positive[index]), "negative": int(negative[index]),
                  "train_positive": int(train_positive[index]), "train_negative": int(train_negative[index]),
                  "supported": bool(supported[index]), "auroc": None, "balanced_accuracy": None}
        if supported[index]:
            record["auroc"] = float(roc_auc_score(labels[:, index], logits[:, index]))
            record["balanced_accuracy"] = float(balanced_accuracy_score(labels[:, index], logits[:, index] >= 0))
        records.append(record)

    def macro(prefix: str = "") -> float | None:
        values = [r["auroc"] for r in records if r["supported"] and r["name"].startswith(prefix)]
        return float(np.mean(values)) if values else None

    return {"macro_auroc": macro(), "maccs_auroc": macro("maccs_"), "elements_auroc": macro("elem_"),
            "fg_auroc": macro("fg_"), "supported_targets": int(supported.sum()), "min_count": min_count,
            "per_target": records}


def _validate_training_data(tokens, mask, labels, folds, molecule_ids, target_names):
    if tokens.ndim != 3 or mask.shape != tokens.shape[:2] or mask.dtype != np.bool_:
        raise ValueError("Expected tokens [N,T,D] and boolean mask [N,T]")
    count = len(tokens)
    if labels.shape != (count, len(target_names)) or folds.shape != (count,) or molecule_ids.shape != (count,):
        raise ValueError("Labels, folds and molecule identities must align with token rows")
    if not np.isin(labels, (0, 1)).all() or not np.isin(folds, FOLDS).all():
        raise ValueError("Expected binary labels and train/val/test folds")
    if not mask.any(axis=1).all():
        raise ValueError("Training spectra must contain at least one selected token")
    split = {name: np.flatnonzero(folds == name) for name in FOLDS}
    if not all(len(rows) for rows in split.values()):
        raise ValueError("All three folds must be nonempty")
    groups = {name: set(molecule_ids[rows].tolist()) for name, rows in split.items()}
    for i, first in enumerate(FOLDS):
        for second in FOLDS[i + 1:]:
            if groups[first] & groups[second]:
                raise ValueError(f"Molecular identity crosses {first}/{second} folds")
    return split


def train_readout(
    tokens: np.ndarray | jax.Array, mask: np.ndarray, labels: np.ndarray,
    folds: np.ndarray, molecule_ids: np.ndarray, target_names: list[str], *,
    kind: str, seed: int = 0, learning_rates: tuple[float, ...] = (3e-4, 1e-3),
    epochs: int = 40, patience: int = 6, batch_size: int = 128,
    weight_decay: float = 1e-4, num_queries: int = 4, min_count: int = 5,
) -> dict[str, Any]:
    """Fit one seed/kind, choose LR/epoch on val, then evaluate each fold once.

    Returns ``params`` (NumPy arrays), ``predictions`` by fold, and JSON-compatible
    ``metadata``. Reuse the saved params through apply_readout for every causal
    intervention; refitting on perturbed tokens would change the endpoint.
    """
    mask, labels, folds, molecule_ids = map(np.asarray, (mask, labels, folds, molecule_ids))
    split = _validate_training_data(tokens, mask, labels, folds, molecule_ids, target_names)
    if min(epochs, patience, batch_size, min_count) < 1 or not learning_rates or min(learning_rates) <= 0:
        raise ValueError("Epochs, patience, batch size, support and learning rates must be positive")
    train_labels = labels[split["train"]].astype(np.float32)
    train_positive = train_labels.sum(axis=0)
    train_negative = len(train_labels) - train_positive
    active = (train_positive > 0) & (train_negative > 0)
    if not active.any():
        raise ValueError("No nonconstant training targets")
    positive_weight = jnp.asarray(np.where(active, len(train_labels) / (2 * np.maximum(train_positive, 1)), 0))
    negative_weight = jnp.asarray(np.where(active, len(train_labels) / (2 * np.maximum(train_negative, 1)), 0))
    x, token_mask = jax.device_put(tokens), jax.device_put(mask)
    validation_labels = labels[split["val"]]
    initial = init_readout(jax.random.key(seed), tokens.shape[-1], labels.shape[1], kind, num_queries)

    @jax.jit
    def infer(params, batch_tokens, batch_mask):
        return apply_readout(params, batch_tokens, batch_mask, kind)[0]

    def predict(params, rows):
        return np.concatenate([np.asarray(infer(params, x[batch], token_mask[batch]))
                               for batch in (rows[start:start + batch_size]
                                             for start in range(0, len(rows), batch_size))])

    def metrics(prediction, truth):
        return binary_metrics(prediction, truth, target_names, train_positive=train_positive,
                              train_negative=train_negative, min_count=min_count)

    selected, trials = None, []
    for learning_rate in learning_rates:
        optimizer = optax.adamw(learning_rate, weight_decay=weight_decay)
        params = jax.tree.map(lambda value: value.copy(), initial)
        opt_state = optimizer.init(params)

        @jax.jit
        def step(current, state, batch_tokens, batch_mask, targets):
            def loss_fn(weights):
                logits, _ = apply_readout(weights, batch_tokens, batch_mask, kind)
                weights = jnp.where(targets > 0, positive_weight, negative_weight)
                loss = optax.sigmoid_binary_cross_entropy(logits, targets) * weights
                return jnp.sum(loss) / (targets.shape[0] * active.sum())
            loss, gradient = jax.value_and_grad(loss_fn)(current)
            updates, state = optimizer.update(gradient, state, current)
            return optax.apply_updates(current, updates), state, loss

        rng = np.random.default_rng(seed)
        best_score, wait, history = -np.inf, 0, []
        for epoch in range(1, epochs + 1):
            order = rng.permutation(split["train"])
            loss_sum, seen = 0., 0
            for start in range(0, len(order), batch_size):
                rows = order[start:start + batch_size]
                params, opt_state, loss = step(params, opt_state, x[rows], token_mask[rows],
                                                jnp.asarray(labels[rows], jnp.float32))
                loss_sum += float(loss) * len(rows)
                seen += len(rows)
            validation = metrics(predict(params, split["val"]), validation_labels)
            score = validation["macro_auroc"]
            if score is None:
                raise ValueError("No validation targets meet the declared support threshold")
            history.append({"epoch": epoch, "train_balanced_bce": loss_sum / seen, "validation_macro_auroc": score})
            if not np.isfinite(loss_sum):
                raise ValueError("Nonfinite readout training loss")
            if score > best_score + 1e-6:
                best_score, wait = score, 0
                if selected is None or score > selected["validation_macro_auroc"] + 1e-6:
                    selected = {"learning_rate": float(learning_rate), "epoch": epoch,
                                "validation_macro_auroc": score,
                                "params": jax.tree.map(lambda value: np.asarray(value).copy(), params)}
            else:
                wait += 1
            if wait >= patience:
                break
        trials.append({"learning_rate": float(learning_rate), "history": history, "best_validation_macro_auroc": best_score})
    params = selected.pop("params")
    predictions, fold_metrics = {}, {}
    for name, rows in split.items():
        logits = predict(params, rows)
        predictions[name] = {"logits": logits, "probabilities": np.asarray(jax.nn.sigmoid(jnp.asarray(logits))),
                             "labels": labels[rows], "indices": rows, "molecule_ids": molecule_ids[rows]}
        fold_metrics[name] = metrics(logits, labels[rows])
    metadata = {
        "schema_version": 1, "kind": kind, "seed": seed, "width": int(tokens.shape[-1]),
        "num_targets": len(target_names), "target_names": target_names,
        "num_queries": num_queries if kind == "attention" else 0,
        "parameter_count": sum(value.size for value in params.values()),
        "normalization": {"kind": "per_token_layer_norm", "affine": False, "epsilon": 1e-5},
        "attention_pool": "equal mean of learned-query pooled vectors; shared-width linear classifier",
        "training": {"optimizer": "optax.adamw", "learning_rates": list(learning_rates), "epochs_max": epochs,
                     "patience": patience, "batch_size": batch_size, "weight_decay": weight_decay,
                     "loss": "per-target class-balanced BCE; nonconstant train targets only",
                     "selection": "validation macro AUROC over targets supported in train and validation; ties retain earlier selection",
                     "test_used_for_selection": False},
        "selection": selected, "trials": trials, "metrics": fold_metrics,
        "fold_rows": {name: len(rows) for name, rows in split.items()},
        "label_level": "molecule; no peak, atom, or fragment assignment", "causal_endpoint_requires_frozen_params": True,
    }
    return {"params": params, "predictions": predictions, "metadata": metadata}


def save_readout(result: dict[str, Any], output: Path, *, contract: dict[str, Any]) -> None:
    """Save reusable weights and row-addressable predictions with provenance."""
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "params.npz", **result["params"])
    for fold, arrays in result["predictions"].items():
        np.savez_compressed(output / f"{fold}_predictions.npz", **arrays)
    with (output / "params.npz").open("rb") as stream:
        weights_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    metadata = {**result["metadata"], "contract": contract, "complete": True,
                "weights": "params.npz", "weights_sha256": weights_sha256,
                "predictions": {fold: f"{fold}_predictions.npz" for fold in FOLDS}}
    (output / "trainedprobe.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")


def load_readout(output: Path) -> tuple[dict[str, jax.Array], dict[str, Any]]:
    metadata = json.loads((output / "trainedprobe.json").read_text())
    if not metadata["complete"] or metadata["schema_version"] != 1:
        raise ValueError("Unsupported or incomplete readout artifact")
    with (output / metadata["weights"]).open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != metadata["weights_sha256"]:
            raise ValueError("Saved readout weights differ from their recorded digest")
    with np.load(output / metadata["weights"], allow_pickle=False) as saved:
        params = {key: jnp.asarray(saved[key]) for key in saved.files}
    expected = {"weight": (metadata["width"], metadata["num_targets"]), "bias": (metadata["num_targets"],)}
    if metadata["kind"] == "attention":
        expected["queries"] = (metadata["num_queries"], metadata["width"])
    if set(params) != set(expected) or any(params[key].shape != shape for key, shape in expected.items()):
        raise ValueError("Saved readout parameters differ from their contract")
    return params, metadata
