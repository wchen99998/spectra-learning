"""Validation learning curves and fixed-endpoint evaluation for longer pilots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.metrics import roc_auc_score

from scripts.pair_to_single_pilot import evaluate


def spectral_retrieval(features: np.ndarray, molecules: np.ndarray) -> dict:
    _, library = np.unique(molecules, return_index=True)
    queries = np.setdiff1d(np.arange(len(molecules)), library)
    normalized = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
    ranks = []
    for start in range(0, len(queries), 256):
        idx = queries[start:start + 256]
        similarity = normalized[idx] @ normalized[library].T
        correct = molecules[idx, None] == molecules[library][None]
        positive = (similarity * correct).sum(1)
        ranks.extend(1 + (similarity > positive[:, None]).sum(1)
                     + .5 * ((similarity == positive[:, None]).sum(1) - 1))
    ranks = np.asarray(ranks)
    return {"top1": float(np.mean(ranks == 1)), "top10": float(np.mean(ranks <= 10)),
            "mrr": float(np.mean(1 / ranks)), "queries": len(queries), "candidates": len(library)}


def validation_scores(data: dict, stored, width: int) -> dict:
    indices = stored["indices"]
    folds = data["fold"][indices]
    train, val = folds == "train", folds == "val"
    features = stored["features"][:, :width].astype(np.float64)
    y = data["fingerprints"][data["molecule"][indices]].astype(np.float64)
    mean, scale = features[train].mean(0), np.maximum(features[train].std(0), 1e-5)
    x = (features - mean) / scale
    x = np.concatenate((x, np.ones((len(x), 1))), axis=1)
    eligible = (y[train].sum(0) >= 20) & ((1 - y[train]).sum(0) >= 20)
    eligible &= y[val].min(0) < y[val].max(0)
    gram, cross = x[train].T @ x[train] / train.sum(), x[train].T @ y[train] / train.sum()
    best_auc, best_alpha = -np.inf, None
    for alpha in (1e-4, 1e-3, 1e-2, .1, 1., 10.):
        regularizer = np.eye(width + 1) * alpha
        regularizer[-1, -1] = 0
        weights = np.linalg.solve(gram + regularizer, cross)
        auc = roc_auc_score(y[val][:, eligible], (x[val] @ weights)[:, eligible], average="macro")
        if auc > best_auc:
            best_auc, best_alpha = auc, alpha
    return {"feature_dim": width, "maccs_val_auc": float(best_auc), "ridge_alpha": best_alpha,
            "retrieval_raw": spectral_retrieval(features[val], data["molecule"][indices[val]])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    data = dict(np.load(args.inputs))
    stored = np.load(args.directory / "embeddings.npz")
    metadata = json.loads((args.directory / "training.json").read_text())
    result = {"step": metadata["steps"], "arm": metadata["arm"], "seed": metadata["seed"],
              "validation": metadata["validation"],
              **{name: validation_scores(data, stored, width)
                 for name, width in (("mean_cls", 256), ("peak_mean", 128))}}
    (args.directory / "validation_scores.json").write_text(json.dumps(result, indent=2) + "\n")
    if "test" in metadata["exported_folds"]:
        if metadata["steps"] != metadata["schedule_steps"]:
            raise ValueError("Test evaluation is restricted to the prespecified final endpoint")
        for readout in ("all", "mean"):
            evaluate(SimpleNamespace(inputs=args.inputs, output=str(args.directory), readout=readout))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
