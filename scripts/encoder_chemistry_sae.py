"""Native JAX TopK sparse autoencoder for frozen peak-token residuals.

Method sources: Gao et al. (2024), https://arxiv.org/abs/2406.04093,
and https://github.com/openai/sparse_autoencoder/blob/main/sparse_autoencoder/train.py.
This implements tied-direction initialization, TopK followed by ReLU, unit-norm
dictionary directions, projected decoder gradients, and optional AuxK recovery
of dead features. It is an adaptation, not a reproduction of their LLM study:
we use a fixed train-fitted mean and scalar RMS, equal-molecule token weights,
and a corpus-relative dead-token threshold. No per-token normalization erases
the residual magnitude. Chemical labels never enter dictionary training.

Decoder rows are dictionary columns in the mathematical convention. Serialized
params include frozen center/scale. Feature edits subtract a dictionary
contribution in ORIGINAL activation units from the ORIGINAL input, preserving
the SAE reconstruction error. Reconstruction alone is a separate baseline.
Reconstruction/sparsity do not establish monosemanticity or chemical causality.
"""
from __future__ import annotations

import hashlib
import json
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax


class SparseCode(NamedTuple):
    indices: jax.Array
    values: jax.Array


def sample_physical_tokens(valid: np.ndarray, *, maximum: int = 32, seed: int = 0) -> dict[str, np.ndarray]:
    """Uniform sampling within each molecule; each molecule has total weight 1.

    The mask contains physical positions only, so CLS has no representable slot.
    Sampling is independent of fold, chemical label, and activation values.
    """
    if valid.ndim != 2 or valid.dtype != np.bool_ or maximum < 1 or np.any(valid.sum(1) == 0):
        raise ValueError("Expected a boolean physical-token mask with nonempty molecules")
    rng = np.random.default_rng(seed)
    rows, slots, weights = [], [], []
    for row, mask in enumerate(valid):
        candidates = np.flatnonzero(mask)
        selected = np.sort(rng.choice(candidates, size=min(maximum, len(candidates)), replace=False))
        rows.extend([row] * len(selected))
        slots.extend(selected.tolist())
        weights.extend([1 / len(selected)] * len(selected))
    return {"rows": np.asarray(rows, np.int32), "slots": np.asarray(slots, np.int16),
            "weights": np.asarray(weights, np.float32)}


def fit_normalization(tokens: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if tokens.ndim != 2 or weights.shape != (len(tokens),) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Expected token matrix and positive per-token weights")
    center = np.average(tokens.astype(np.float64), axis=0, weights=weights)
    square = np.mean((tokens.astype(np.float64) - center) ** 2, axis=1)
    scale = float(np.sqrt(np.average(square, weights=weights)))
    if not np.isfinite(center).all() or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Training activations need finite nonzero variation")
    return center.astype(np.float32), np.asarray(scale, np.float32)


def init_sae(key: jax.Array, center: jax.Array, scale: jax.Array, features: int) -> dict[str, jax.Array]:
    width = center.shape[0]
    if features < 1 or center.ndim != 1:
        raise ValueError("Expected positive dictionary width and vector center")
    decoder = jax.random.normal(key, (features, width), dtype=jnp.float32)
    decoder /= jnp.linalg.norm(decoder, axis=1, keepdims=True)
    return {"encoder": decoder.copy(), "decoder": decoder, "pre_bias": jnp.zeros(width),
            "latent_bias": jnp.zeros(features), "center": jnp.asarray(center, jnp.float32),
            "scale": jnp.asarray(scale, jnp.float32)}


def _preactivations(params: dict, x: jax.Array) -> jax.Array:
    normalized = (x.astype(jnp.float32) - params["center"]) / params["scale"]
    return (normalized - params["pre_bias"]) @ params["encoder"].T + params["latent_bias"]


def _topk(preactivations: jax.Array, k: int) -> SparseCode:
    if not 1 <= k <= preactivations.shape[-1]:
        raise ValueError("Sparsity k must lie between 1 and dictionary width")
    values, indices = jax.lax.top_k(preactivations, k)
    return SparseCode(indices, jax.nn.relu(values))


def encode_sae(params: dict, x: jax.Array, k: int) -> SparseCode:
    """Encode original-unit activations; k is static, leading dimensions persist."""
    return _topk(_preactivations(params, x), k)


def dense_activations(code: SparseCode, features: int) -> jax.Array:
    indices, values = code.indices.reshape(-1, code.indices.shape[-1]), code.values.reshape(-1, code.values.shape[-1])
    dense = jnp.zeros((len(indices), features), dtype=values.dtype)
    dense = dense.at[jnp.arange(len(indices))[:, None], indices].add(values)
    return dense.reshape(*code.values.shape[:-1], features)


def decode_sae(params: dict, code: SparseCode) -> jax.Array:
    reconstructed = dense_activations(code, params["decoder"].shape[0]) @ params["decoder"] + params["pre_bias"]
    return reconstructed * params["scale"] + params["center"]


def reconstruct_sae(params: dict, x: jax.Array, k: int) -> jax.Array:
    return decode_sae(params, encode_sae(params, x, k))


def feature_amplitude(params: dict, x: jax.Array, k: int, feature: jax.Array) -> jax.Array:
    code = encode_sae(params, x, k)
    return jnp.sum(jnp.where(code.indices == feature, code.values, 0), axis=-1)


def edit_feature(params: dict, x: jax.Array, k: int, feature: jax.Array, dose: float = 1.) -> jax.Array:
    """x - dose*a_j*d_j*scale; dose0 is identity, dose1 removes one contribution."""
    amplitude = feature_amplitude(params, x, k, feature)
    return x.astype(jnp.float32) - dose * amplitude[..., None] * params["decoder"][feature] * params["scale"]


def matched_direction_control(
    params: dict, x: jax.Array, k: int, feature: jax.Array,
    direction: jax.Array, dose: float = 1.,
) -> jax.Array:
    """Same perturbation norm as feature removal along a supplied control direction.

    The direction may be a random unit vector or a random OTHER dictionary row;
    this is a direction control, not removal of that other feature's activation.
    """
    direction = direction / jnp.maximum(jnp.linalg.norm(direction), 1e-20)
    amplitude = feature_amplitude(params, x, k, feature)
    return x.astype(jnp.float32) - dose * amplitude[..., None] * direction * params["scale"]


def sae_loss(params: dict, x: jax.Array, weights: jax.Array, dead: jax.Array, *,
             k: int, aux_k: int, aux_weight: float = 1 / 32) -> tuple[jax.Array, dict]:
    """Weighted reconstruction plus AuxK on a stopped-gradient error target."""
    normalized = (x.astype(jnp.float32) - params["center"]) / params["scale"]
    pre = (normalized - params["pre_bias"]) @ params["encoder"].T + params["latent_bias"]
    code = _topk(pre, k)
    dense = dense_activations(code, params["decoder"].shape[0])
    reconstructed = dense @ params["decoder"] + params["pre_bias"]
    error = normalized - reconstructed
    mse_per_token = jnp.mean(error ** 2, axis=-1)
    mse = jnp.sum(weights * mse_per_token) / jnp.sum(weights)
    auxiliary = _topk(jnp.where(dead, pre, -jnp.inf), aux_k)
    predicted_error = dense_activations(auxiliary, params["decoder"].shape[0]) @ params["decoder"]
    target_error = jax.lax.stop_gradient(error)
    residual_scale = jnp.maximum(jnp.mean(target_error ** 2, axis=-1), 1e-8)
    aux_per_token = jnp.mean((predicted_error - target_error) ** 2, axis=-1) / residual_scale
    aux = jnp.where(dead.any(), jnp.sum(weights * aux_per_token) / jnp.sum(weights), 0.)
    fired = jnp.any(dense > 1e-3, axis=0)
    return mse + aux_weight * aux, {"mse": mse, "aux": aux, "fired": fired}


@partial(jax.jit, static_argnames=("k",))
def _evaluate_batch(params: dict, batch: jax.Array, *, k: int):
    code = encode_sae(params, batch, k)
    return decode_sae(params, code), code


def reconstruction_metrics(params: dict, tokens: np.ndarray | jax.Array, weights: np.ndarray,
                           *, k: int, batch_size: int = 1024) -> dict:
    features = params["decoder"].shape[0]
    count = np.zeros(features, np.int64)
    total_weight, error_sum, energy_sum, centered_sum, l0_sum = 0., 0., 0., 0., 0.

    center = np.asarray(params["center"])
    for first in range(0, len(tokens), batch_size):
        last = min(first + batch_size, len(tokens))
        x = np.asarray(tokens[first:last], np.float32)
        reconstruction, code = _evaluate_batch(params, jnp.asarray(x), k=k)
        values, indices = np.asarray(code.values), np.asarray(code.indices)
        w = weights[first:last]
        error_sum += float(np.sum(w * np.sum((np.asarray(reconstruction) - x) ** 2, axis=-1)))
        energy_sum += float(np.sum(w * np.sum(x ** 2, axis=-1)))
        centered_sum += float(np.sum(w * np.sum((x - center) ** 2, axis=-1)))
        l0_sum += float(np.sum(w * (values > 0).sum(axis=-1)))
        np.add.at(count, indices[values > 0], 1)
        total_weight += float(w.sum())
    return {"tokens": len(tokens), "fve_vs_train_mean": 1 - error_sum / centered_sum,
            "relative_l2": float(np.sqrt(error_sum / max(energy_sum, 1e-20))),
            "centered_relative_l2": float(np.sqrt(error_sum / centered_sum)),
            "normalized_mse": error_sum / (total_weight * tokens.shape[-1] * float(params["scale"]) ** 2),
            "mean_l0": l0_sum / total_weight, "dead_fraction_on_split": float(np.mean(count == 0)),
            "firing_counts": count}


def train_sae(
    train_tokens: np.ndarray, validation_tokens: np.ndarray, *, train_weights: np.ndarray,
    validation_weights: np.ndarray, features: int = 8192, k: int = 32, seed: int = 0,
    learning_rate: float = 3e-4, epochs: int = 40, patience: int = 5, batch_size: int = 1024,
    dead_after_tokens: int | None = None, aux_k: int | None = None,
) -> dict[str, Any]:
    """Unsupervised fit with validation-reconstruction selection; no test/labels API."""
    center, scale = fit_normalization(train_tokens, train_weights)
    if validation_tokens.ndim != 2 or validation_tokens.shape[1] != train_tokens.shape[1]:
        raise ValueError("Validation activations must match training width")
    if (validation_weights.shape != (len(validation_tokens),) or np.any(validation_weights <= 0)
            or not np.isfinite(validation_weights).all()):
        raise ValueError("Validation weights must be positive and token-aligned")
    if min(epochs, patience, batch_size, k) <= 0 or k > features:
        raise ValueError("Invalid SAE training dimensions")
    aux_k = min(features, train_tokens.shape[1] // 2) if aux_k is None else aux_k
    dead_after_tokens = 2 * len(train_tokens) if dead_after_tokens is None else dead_after_tokens
    initial = init_sae(jax.random.key(seed), center, scale, features)
    fixed = {name: initial[name] for name in ("center", "scale")}
    trainable = {name: value for name, value in initial.items() if name not in fixed}
    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adam(learning_rate, eps=6.25e-10))
    opt_state = optimizer.init(trainable)
    since_fired = jnp.zeros(features, jnp.int32)
    x, w = jax.device_put(train_tokens), jnp.asarray(train_weights, jnp.float32)

    @jax.jit
    def step(current, state, inactive, batch, weights):
        dead = inactive >= dead_after_tokens
        def loss_fn(parameters):
            return sae_loss({**parameters, **fixed}, batch, weights, dead, k=k, aux_k=aux_k)
        (loss, statistics), gradient = jax.value_and_grad(loss_fn, has_aux=True)(current)
        decoder = current["decoder"]
        parallel = jnp.sum(gradient["decoder"] * decoder, axis=1, keepdims=True)
        gradient["decoder"] = gradient["decoder"] - parallel * decoder
        updates, state = optimizer.update(gradient, state, current)
        current = optax.apply_updates(current, updates)
        current["decoder"] /= jnp.maximum(jnp.linalg.norm(current["decoder"], axis=1, keepdims=True), 1e-20)
        inactive = jnp.where(statistics["fired"], 0, inactive + batch.shape[0])
        return current, state, inactive, loss, statistics["mse"], statistics["aux"]

    best, best_score, history, wait = None, np.inf, [], 0
    rng = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(train_tokens))
        losses = []
        for first in range(0, len(order), batch_size):
            indices = order[first:first + batch_size]
            trainable, opt_state, since_fired, loss, mse, aux = step(
                trainable, opt_state, since_fired, x[indices], w[indices])
            losses.append([float(loss), float(mse), float(aux)])
        if not np.isfinite(losses).all():
            raise ValueError("Nonfinite SAE training loss")
        combined = {**trainable, **fixed}
        validation = reconstruction_metrics(combined, validation_tokens, validation_weights, k=k, batch_size=batch_size)
        score = validation["normalized_mse"]
        entry = {"epoch": epoch, "train_loss": float(np.mean(losses, axis=0)[0]),
                 "train_mse": float(np.mean(losses, axis=0)[1]), "train_aux": float(np.mean(losses, axis=0)[2]),
                 "validation_normalized_mse": score, "validation_fve_vs_train_mean": validation["fve_vs_train_mean"],
                 "validation_mean_l0": validation["mean_l0"],
                 "training_dead_fraction": float(np.mean(np.asarray(since_fired) >= dead_after_tokens))}
        history.append(entry)
        print(json.dumps({"sae_epoch": entry}), flush=True)
        if score < best_score - 1e-5:
            best_score, wait = score, 0
            best = {"params": jax.tree.map(lambda value: np.asarray(value).copy(), combined), "epoch": epoch}
        else:
            wait += 1
        if wait >= patience:
            break
    return {"params": best["params"], "initial_params": jax.tree.map(np.asarray, initial),
            "metadata": {"schema_version": 1, "features": features, "width": train_tokens.shape[1], "k": k,
                         "seed": seed, "selected_epoch": best["epoch"], "history": history,
                         "normalization": "fixed train-only weighted mean and scalar RMS; no per-token rescaling",
                         "learning_rate": learning_rate, "epochs_max": epochs, "patience": patience,
                         "batch_size": batch_size, "optimizer": "Adam eps6.25e-10; gradient norm clip1; decoder tangent projection and unit renormalization",
                         "aux_k": aux_k, "aux_weight": 1 / 32, "dead_after_tokens": dead_after_tokens,
                         "dead_rule": "No activation above1e-3 for2complete sampled training passes bydefault; adapted from paper10Mtoken threshold",
                         "selection": "minimum validation weighted reconstruction MSE only",
                         "labels_used": False, "sources": ["https://arxiv.org/abs/2406.04093",
                             "https://github.com/openai/sparse_autoencoder/blob/main/sparse_autoencoder/train.py"]}}


def dictionary_redundancy(params: dict, block: int = 512) -> dict:
    decoder = jnp.asarray(params["decoder"])
    nearest, maximum = [], []
    for first in range(0, len(decoder), block):
        last = min(first + block, len(decoder))
        similarities = np.asarray(jnp.abs(decoder[first:last] @ decoder.T)).copy()
        similarities[np.arange(last - first), np.arange(first, last)] = -1
        nearest.extend(similarities.argmax(axis=1).tolist())
        maximum.extend(similarities.max(axis=1).tolist())
    values = np.asarray(maximum)
    return {"mean_max_absolute_cosine": float(values.mean()), "p95_max_absolute_cosine": float(np.quantile(values, .95)),
            "fraction_above_0_9": float(np.mean(values > .9)), "fraction_above_0_95": float(np.mean(values > .95)),
            "nearest_feature": np.asarray(nearest, np.int32), "max_absolute_cosine": values.astype(np.float32)}


def save_sae(result: dict, output: Path, *, contract: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "params.npz", **result["params"])
    with (output / "params.npz").open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    record = {**result["metadata"], "contract": contract, "weights_sha256": digest, "complete": True}
    (output / "trained_sae.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def load_sae(output: Path) -> tuple[dict[str, jax.Array], dict]:
    record = json.loads((output / "trained_sae.json").read_text())
    with (output / "params.npz").open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != record["weights_sha256"]:
            raise ValueError("SAE weights differ from the recorded digest")
    with np.load(output / "params.npz", allow_pickle=False) as saved:
        params = {name: jnp.asarray(saved[name]) for name in saved.files}
    if not record["complete"] or params["decoder"].shape != (record["features"], record["width"]):
        raise ValueError("SAE artifact dimensions or completion state do not match")
    expected = {"decoder": (record["features"], record["width"]),
                "encoder": (record["features"], record["width"]),
                "pre_bias": (record["width"],), "latent_bias": (record["features"],),
                "center": (record["width"],), "scale": ()}
    if set(params) != set(expected) or any(params[key].shape != shape for key, shape in expected.items()):
        raise ValueError("SAE parameter schema differs from the recorded dictionary")
    if any(not np.isfinite(np.asarray(value)).all() for value in params.values()) or float(params["scale"]) <= 0:
        raise ValueError("SAE parameters must be finite with positive scale")
    if not np.allclose(np.linalg.norm(np.asarray(params["decoder"]), axis=1), 1, atol=2e-6):
        raise ValueError("SAE decoder directions must have unit norm")
    return params, record
