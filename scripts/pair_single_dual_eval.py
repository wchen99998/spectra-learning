"""Frozen-feature evaluation for ``pair_single_dual_pilot`` exports.

For each readout (single stream, pair stream, both), fit a ridge MACCS probe
on the fixed probe-train spectra, choose its penalty on probe-val, and score
the original test fold and the previously unprobed validation molecules
(``confirm``). Retrieval uses cosine similarity on train-standardized
features: one fixed library spectrum per held-out molecule, other spectra are
queries. Standardization keeps the wider stream from dominating concatenated
cosine geometry. Validation-only scores are produced for intermediate exports.

With ``--metadata`` (neutral masses from ``pair_single_dual_metadata``),
"isomer" retrieval also ranks each query only against library molecules
within 10 ppm of its neutral mass, keeping queries with at least two such
candidates: precursor mass cannot solve it, so it separates spectral
evidence from metadata shortcuts. A ``features_<step>_blind.npz`` export
(all metadata absent) is scored under ``blind``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

READOUTS = {
    "single": ("single_mean", "single_cls"),
    "pair": ("pair_mean", "pair_to_cls", "pair_from_cls"),
    "both": ("single_mean", "single_cls", "pair_mean", "pair_to_cls", "pair_from_cls"),
}
ALPHAS = (1e-3, 1e-2, .1, 1., 10., 100.)


ISOMER_PPM = 10.0


def retrieval(features, molecules, mass=None):
    _, library = np.unique(molecules, return_index=True)
    queries = np.setdiff1d(np.arange(len(molecules)), library)
    z = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
    ranks = []
    for start in range(0, len(queries), 512):
        idx = queries[start:start + 512]
        sim = z[idx] @ z[library].T
        correct = molecules[idx, None] == molecules[library][None]
        if mass is not None:
            window = np.abs(mass[idx, None] - mass[library][None]) <= ISOMER_PPM * 1e-6 * mass[idx, None]
            keep = (window.sum(1) >= 2) & (window & correct).any(1)
            sim, correct = np.where(window, sim, -np.inf)[keep], correct[keep]
        pos = (np.where(correct, sim, 0.0)).sum(1)
        ranks.append(1 + (sim > pos[:, None]).sum(1) + .5 * ((sim == pos[:, None]).sum(1) - 1))
    ranks = np.concatenate(ranks)
    return {"top1": float(np.mean(ranks == 1)), "top10": float(np.mean(ranks <= 10)),
            "mrr": float(np.mean(1 / ranks)), "queries": int(len(ranks))}


def score(data, export, readout, mass=None):
    keys = READOUTS[readout]
    if any(f"train/{k}" not in export for k in keys):
        return None
    fold = {name: np.concatenate([export[f"{name}/{k}"] for k in keys], axis=1).astype(np.float64)
            for name in ("train", "val", "test", "confirm") if f"{name}/indices" in export}
    mol = {name: data["molecule"][export[f"{name}/indices"]] for name in fold}
    y = {name: data["fingerprints"][m].astype(np.float64) for name, m in mol.items()}
    mean, std = fold["train"].mean(0), np.maximum(fold["train"].std(0), 1e-5)
    x = {name: np.concatenate(((f - mean) / std, np.ones((len(f), 1))), axis=1)
         for name, f in fold.items()}
    eligible = (y["train"].sum(0) >= 20) & ((1 - y["train"]).sum(0) >= 20)
    eligible &= y["val"].min(0) < y["val"].max(0)
    gram = x["train"].T @ x["train"] / len(x["train"])
    cross = x["train"].T @ y["train"] / len(x["train"])
    best = (-np.inf, None, None)
    for alpha in ALPHAS:
        reg = np.eye(len(gram)) * alpha
        reg[-1, -1] = 0
        w = np.linalg.solve(gram + reg, cross)
        auc = roc_auc_score(y["val"][:, eligible], (x["val"] @ w)[:, eligible], average="macro")
        if auc > best[0]:
            best = (auc, alpha, w)
    result = {"dim": int(len(mean)), "val_auc": float(best[0]), "alpha": best[1],
              "val_retrieval": retrieval(x["val"][:, :-1], mol["val"])}
    for name in ("test", "confirm"):
        if name not in x:
            continue
        bits = eligible & (y[name].min(0) < y[name].max(0))
        pred = x[name] @ best[2]
        result[f"{name}_auc"] = float(roc_auc_score(y[name][:, bits], pred[:, bits], average="macro"))
        result[f"{name}_retrieval"] = retrieval(x[name][:, :-1], mol[name])
        result[f"{name}_retrieval_raw"] = retrieval(fold[name], mol[name])
        if mass is not None:
            result[f"{name}_retrieval_isomer"] = retrieval(
                x[name][:, :-1], mol[name], mass[export[f"{name}/indices"]])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--metadata", help="pair_single_dual_metadata output (neutral masses)")
    args = parser.parse_args()
    data = dict(np.load(args.inputs))
    mass = np.load(args.metadata)["neutral_mass_da"] if args.metadata else None
    for directory in args.directories:
        training = json.loads((directory / "training.json").read_text())
        rows = []
        for checkpoint in training["checkpoints"]:
            export = np.load(directory / f"features_{checkpoint['step']}.npz")
            row = {"step": checkpoint["step"], "validation": checkpoint["validation"]}
            for readout in READOUTS:
                scores = score(data, export, readout, mass)
                if scores is not None:
                    row[readout] = scores
            blind = directory / f"features_{checkpoint['step']}_blind.npz"
            if blind.exists():
                export = np.load(blind)
                row["blind"] = {r: s for r in READOUTS
                                if (s := score(data, export, r, mass)) is not None}
            for key in ("validation_ts", "validation_blind"):
                if key in checkpoint:
                    row[key] = checkpoint[key]
            rows.append(row)
        output = {"arm": training["arm"], "seed": training["seed"], "params": training["params"],
                  "steps_per_second": training["steps_per_second"],
                  "interventions": training["interventions"], "checkpoints": rows}
        (directory / "evaluation.json").write_text(json.dumps(output, indent=2) + "\n")
        final = rows[-1]
        print(json.dumps({"dir": directory.name, **{
            r: {k: final[r][k] for k in ("val_auc", "test_auc", "confirm_auc") if k in final[r]}
            for r in READOUTS if r in final}}), flush=True)


if __name__ == "__main__":
    main()
