"""Paired molecule-bootstrap samples for fixed-checkpoint chemical/retrieval metrics.

All models use identical molecule resamples. Intervals aggregated across models
are conditional on the three trained seeds, the fitted probes, and the fixed
retrieval gallery; per-seed variation must also be reported separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


class WeightedAUC:
    """Pre-sort score ties once, then compute weighted Mann-Whitney AUC."""

    def __init__(self, labels: np.ndarray, scores: np.ndarray, owner: np.ndarray):
        order = np.argsort(scores, axis=0)
        columns = np.arange(scores.shape[1])[None, :]
        values = scores[order, columns]
        self.positive = labels[order, columns].astype(bool)
        self.owner = owner[order]
        rows = np.arange(len(scores), dtype=np.int32)[:, None]
        starts = np.concatenate((np.ones((1, scores.shape[1]), bool), values[1:] != values[:-1]))
        ends = np.concatenate((values[1:] != values[:-1], np.ones((1, scores.shape[1]), bool)))
        self.start = np.maximum.accumulate(np.where(starts, rows, 0), axis=0)
        self.end = np.minimum.accumulate(np.where(ends, rows, len(scores) - 1)[::-1], axis=0)[::-1] + 1
        self.columns = columns

    def __call__(self, counts: np.ndarray) -> float:
        weights = counts[self.owner].astype(np.float64)
        positives = weights * self.positive
        negative_cumulative = np.concatenate((np.zeros((1, weights.shape[1])),
                                              np.cumsum(weights - positives, axis=0)))
        # Half the negative weight of the score's tie group contributes to AUC.
        below = (negative_cumulative[self.start, self.columns]
                 + negative_cumulative[self.end, self.columns]) * .5
        numerator = (positives * below).sum(0)
        denominator = positives.sum(0) * negative_cumulative[-1]
        valid = denominator > 0
        return float(np.mean(numerator[valid] / denominator[valid]))


def probe_predictions(data, stored, width, reference):
    indices = stored["indices"]
    train, val, test = [data["fold"][indices] == fold for fold in ("train", "val", "test")]
    features = stored["features"][:, :width].astype(np.float64)
    labels = data["fingerprints"][data["molecule"][indices]].astype(np.float64)
    mean, scale = features[train].mean(0), np.maximum(features[train].std(0), 1e-5)
    x = (features - mean) / scale
    x = np.concatenate((x, np.ones((len(x), 1))), axis=1)
    regularizer = np.eye(width + 1) * reference["ridge_alpha"]
    regularizer[-1, -1] = 0
    weights = np.linalg.solve(x[train].T @ x[train] / train.sum() + regularizer,
                              x[train].T @ labels[train] / train.sum())
    eligible = (labels[train].sum(0) >= 20) & ((1 - labels[train]).sum(0) >= 20)
    eligible &= labels[val].min(0) < labels[val].max(0)
    eligible &= labels[test].min(0) < labels[test].max(0)
    y, predictions = labels[test][:, eligible], (x[test] @ weights)[:, eligible]
    np.testing.assert_allclose(roc_auc_score(y, predictions, average="macro"),
                               reference["maccs_test_auc"], atol=1e-12, rtol=0)
    return y, predictions, features[test], data["molecule"][indices[test]], indices[test]


def retrieval_counts(features, molecules, owner):
    _, library = np.unique(molecules, return_index=True)
    queries = np.setdiff1d(np.arange(len(molecules)), library)
    features = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
    hits = []
    for start in range(0, len(queries), 256):
        idx = queries[start:start + 256]
        similarity = features[idx] @ features[library].T
        positive = (similarity * (molecules[idx, None] == molecules[library][None])).sum(1)
        hits.extend(((similarity > positive[:, None]).sum(1) == 0)
                    & ((similarity == positive[:, None]).sum(1) == 1))
    successes = np.bincount(owner[queries], weights=np.asarray(hits, dtype=float), minlength=len(library))
    totals = np.bincount(owner[queries], minlength=len(library))
    return successes, totals


def bootstrap_folder(data, folder, repetitions):
    stored = np.load(folder / "embeddings.npz")
    output, points = {}, {}
    for readout, width, filename in (("mean_cls", 256, "evaluation.json"),
                                     ("peak_mean", 128, "evaluation_mean.json")):
        started = time.perf_counter()
        reference = json.loads((folder / filename).read_text())
        y, scores, features, molecules, indices = probe_predictions(data, stored, width, reference)
        unique, owner = np.unique(molecules, return_inverse=True)
        rng = np.random.default_rng(202609301)
        draws = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=repetitions)
        auc = WeightedAUC(y, scores, owner)
        np.testing.assert_allclose(auc(np.ones(len(unique), dtype=int)), reference["maccs_test_auc"],
                                   atol=1e-12, rtol=0)
        successes, totals = retrieval_counts(features, molecules, owner)
        np.testing.assert_allclose(successes.sum() / totals.sum(), reference["retrieval_raw"]["top1"],
                                   atol=1e-12, rtol=0)
        output[f"auc_{readout}"] = np.asarray([auc(counts) for counts in draws])
        output[f"top1_{readout}"] = (draws @ successes) / (draws @ totals)
        points[readout] = {"auc": reference["maccs_test_auc"], "top1": reference["retrieval_raw"]["top1"]}
        print(json.dumps({"folder": str(folder), "readout": readout,
                          "bootstrap_seconds": time.perf_counter() - started}), flush=True)
    np.savez(folder / "molecule_bootstrap.npz", **output)
    metadata = {"replicates": repetitions, "rng_seed": 202609301, "molecules": len(unique),
                "test_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(), "points": points,
                "method": "Paired molecule resampling; fixed trained models, probes, and retrieval gallery. AUC bits with no weighted positives or negatives in a draw are omitted identically for every model."}
    (folder / "molecule_bootstrap.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--confirmation", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=500)
    args = parser.parse_args()
    bootstrap_folder(dict(np.load(args.inputs)), args.directory, args.replicates)
    bootstrap_folder(dict(np.load(args.confirmation)), args.directory / "confirmation", args.replicates)


if __name__ == "__main__":
    main()
