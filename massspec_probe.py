from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from absl import logging
from flax import nnx
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from models import mae as mae_models

_MASS_SPEC_MORGAN_FEATURE = "massspec_morgan_top16"


def _collect_latent_subset(
    model: nnx.Module,
    eval_loader: Any,
    *,
    sample_size: int,
    seed: int,
    feature_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    encode_fn = nnx.jit(lambda mdl, batch: mdl.encode(batch, train=False))
    rng = np.random.default_rng(seed)

    embeddings = None
    labels = None
    seen = 0

    for batch in iter(eval_loader):
        batch_emb = np.asarray(encode_fn(model, batch))
        batch_labels = np.asarray(batch[feature_key])

        if embeddings is None:
            embeddings = np.empty((sample_size, batch_emb.shape[1]), dtype=batch_emb.dtype)
            labels = np.empty((sample_size, batch_labels.shape[1]), dtype=batch_labels.dtype)

        for i in range(batch_emb.shape[0]):
            if seen < sample_size:
                embeddings[seen] = batch_emb[i]
                labels[seen] = batch_labels[i]
            else:
                j = rng.integers(0, seen + 1)
                if j < sample_size:
                    embeddings[j] = batch_emb[i]
                    labels[j] = batch_labels[i]
            seen += 1

        if seen >= sample_size:
            break

    count = min(seen, sample_size)
    return embeddings[:count], labels[:count]


def _plot_metrics_matrix(metrics: np.ndarray, bit_labels: list[str]) -> plt.Figure:
    fig_height = max(2.5, 0.35 * len(bit_labels))
    fig, ax = plt.subplots(figsize=(7.0, fig_height))
    im = ax.imshow(metrics, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(metrics.shape[1]))
    ax.set_xticklabels(["acc", "prec", "recall", "f1"])
    ax.set_yticks(range(len(bit_labels)))
    ax.set_yticklabels(bit_labels)

    for i in range(metrics.shape[0]):
        for j in range(metrics.shape[1]):
            ax.text(j, i, f"{metrics[i, j]:.2f}", ha="center", va="center", color="white")

    ax.set_title("MassSpecGym linear probe metrics")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def evaluate_massspec_linear_probe(
    model: nnx.Module,
    eval_loader: Any,
    *,
    writer: Any,
    step: int,
    sample_size: int = 20_000,
    seed: int = 0,
    bit_indices: list[int] | None = None,
    feature_key: str = _MASS_SPEC_MORGAN_FEATURE,
) -> None:
    logging.info("Running MassSpecGym linear probe (sample_size=%d).", sample_size)

    embeddings, labels = _collect_latent_subset(
        model,
        eval_loader,
        sample_size=sample_size,
        seed=seed,
        feature_key=feature_key,
    )

    rng = np.random.default_rng(seed)
    perm = rng.permutation(embeddings.shape[0])
    split_idx = int(0.8 * embeddings.shape[0])
    train_idx = perm[:split_idx]
    test_idx = perm[split_idx:]

    scaler = StandardScaler()
    train_emb = scaler.fit_transform(embeddings[train_idx])
    test_emb = scaler.transform(embeddings[test_idx])

    metrics = np.zeros((labels.shape[1], 4), dtype=np.float32)
    for i in range(labels.shape[1]):
        clf = SGDClassifier(loss="log_loss", max_iter=1000, random_state=seed)
        clf.fit(train_emb, labels[train_idx, i])
        pred = clf.predict(test_emb)

        metrics[i, 0] = accuracy_score(labels[test_idx, i], pred)
        metrics[i, 1] = precision_score(labels[test_idx, i], pred, zero_division=0)
        metrics[i, 2] = recall_score(labels[test_idx, i], pred, zero_division=0)
        metrics[i, 3] = f1_score(labels[test_idx, i], pred, zero_division=0)

    bit_ids = bit_indices if bit_indices is not None else list(range(labels.shape[1]))
    bit_labels = [str(bit_id) for bit_id in bit_ids]

    scalars = {
        "massspec_probe/mean_accuracy": float(metrics[:, 0].mean()),
        "massspec_probe/mean_precision": float(metrics[:, 1].mean()),
        "massspec_probe/mean_recall": float(metrics[:, 2].mean()),
        "massspec_probe/mean_f1": float(metrics[:, 3].mean()),
    }
    for i, bit_id in enumerate(bit_ids):
        scalars[f"massspec_probe/bit_{bit_id}/accuracy"] = float(metrics[i, 0])
        scalars[f"massspec_probe/bit_{bit_id}/precision"] = float(metrics[i, 1])
        scalars[f"massspec_probe/bit_{bit_id}/recall"] = float(metrics[i, 2])
        scalars[f"massspec_probe/bit_{bit_id}/f1"] = float(metrics[i, 3])

    writer.write_scalars(step, scalars)

    fig = _plot_metrics_matrix(metrics, bit_labels)
    image = mae_models.figures_to_image_array(fig)
    writer.write_images(step, {"massspec_probe/metrics_matrix": image})

    logging.info("MassSpecGym linear probe metrics: %s", scalars)
