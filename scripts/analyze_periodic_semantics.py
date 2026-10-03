"""Paired molecule-level uncertainty and portable evidence table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.periodic_semantics_stats import macro_auc, paired_mean_interval

ARMS = ("mae_666k", "ftjepa_1190k", "periodic_student_1200k", "periodic_student_1209k")


def bootstrap_auc_delta(pred_a, pred_b, truth, repeats=500):
    """Paired molecule bootstrap. Tied predictions receive half credit."""
    valid = (truth.sum(0) > 0) & (truth.sum(0) < len(truth))
    truth = truth[:, valid]
    n, d = truth.shape
    rows = np.arange(n)[:, None]
    cols = np.arange(d)[None, :]
    prepared = []
    for prediction in (pred_a[:, valid], pred_b[:, valid]):
        order = np.argsort(prediction, axis=0, kind="stable")
        p = np.take_along_axis(prediction, order, axis=0)
        y = np.take_along_axis(truth, order, axis=0)
        new = np.vstack([np.ones((1, d), bool), p[1:] != p[:-1]])
        end = np.vstack([p[1:] != p[:-1], np.ones((1, d), bool)])
        first = np.maximum.accumulate(np.where(new, rows, 0), axis=0)
        last = np.minimum.accumulate(np.where(end, rows, n - 1)[::-1], axis=0)[::-1]
        prepared.append((order, y, first, last))
    rng = np.random.default_rng(20260929)
    deltas = []
    for _ in range(repeats):
        weights = np.bincount(rng.integers(n, size=n), minlength=n)
        values = []
        for order, y, first, last in prepared:
            w = weights[order]
            negative_cumulative = np.cumsum(w * (1 - y), axis=0)
            before = np.where(first > 0, negative_cumulative[(first - 1).clip(0), cols], 0)
            through = negative_cumulative[last, cols]
            wins = (w * y * (before + through) * .5).sum(0)
            positive = (w * y).sum(0)
            negative = n - positive
            valid_draw = (positive > 0) & (negative > 0)
            values.append(np.mean(wins[valid_draw] / (positive[valid_draw] * negative[valid_draw])))
        deltas.append(values[1] - values[0])
    return {"delta": macro_auc(pred_b, truth) - macro_auc(pred_a, truth) if valid.all() else
            macro_auc(pred_b[:, valid], truth) - macro_auc(pred_a[:, valid], truth),
            "ci95": np.quantile(deltas, [.025, .975]).tolist(), "repeats": repeats,
            "test_evaluable_bits": int(valid.sum())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/periodic_semantics_20260929"))
    parser.add_argument("--output", type=Path, default=Path("presentations/periodic_semantics/research"))
    parser.add_argument("--reuse-probe-bootstrap", action="store_true",
                        help="Reuse unchanged saved probe intervals; compute intervals for added comparisons.")
    args = parser.parse_args()
    data = {a: json.loads((args.input / f"{a}.json").read_text()) for a in ARMS}
    samples = {a: np.load(args.input / f"{a}_samples.npz") for a in ARMS}
    previous = json.loads((args.output / "summary.json").read_text()) if args.reuse_probe_bootstrap else None
    summary = {"models": data, "paired_differences": {}, "stage_differences": {}, "unit": "held-out molecule",
               "uncertainty": "paired nonparametric molecule bootstrap; percentile 95% intervals; fixed trained models; no multiple-testing correction"}
    comparisons = [
        ("paired_differences", arm, "ftjepa_1190k", arm, ("mean_final", "mean_19", "cls_final"))
        for arm in ARMS[2:]
    ] + [
        ("stage_differences", "jepa_minus_mae", "mae_666k", "ftjepa_1190k", ("mean_final",)),
        ("stage_differences", "latest_minus_mae", "mae_666k", "periodic_student_1209k", ("mean_final",)),
    ]
    for group, comparison, reference, arm, features in comparisons:
        initial = samples[reference]
        current = samples[arm]
        assert data[reference]["dataset_sha256"] == data[arm]["dataset_sha256"]
        np.testing.assert_array_equal(initial["test_binary"], current["test_binary"])
        np.testing.assert_array_equal(initial["test_regression"], current["test_regression"])
        np.testing.assert_array_equal(initial["test_molecule"], current["test_molecule"])
        effects = {"reference": reference, "comparison": arm}
        cached = previous.get(group, {}).get(comparison, {}) if previous else {}
        for feature in features:
            print("bootstrap", arm, feature, flush=True)
            key = feature + "_maccs_auc"
            if key not in cached:
                effects[key] = bootstrap_auc_delta(
                    initial[feature + "_binary_prediction"][:, :166],
                    current[feature + "_binary_prediction"][:, :166], initial["test_binary"][:, :166])
            else:
                # Existing comparisons retain their original saved predictions.
                value = macro_auc(current[feature + "_binary_prediction"][:, :166], initial["test_binary"][:, :166]) - macro_auc(initial[feature + "_binary_prediction"][:, :166], initial["test_binary"][:, :166])
                assert np.isclose(value, cached[key]["delta"], atol=1e-12)
                effects[key] = cached[key]
            for variant in ("raw", "standardized"):
                key = f"{feature}_{variant}_knn1"
                effects[key] = paired_mean_interval(initial[key], current[key])
        if "single_raw_knn1" in initial and "single_raw_knn1" in current:
            for variant in ("raw", "standardized", "whiten_0.01"):
                for kind in ("knn1", "knn_mass5"):
                    key = f"single_{variant}_{kind}"
                    keep = np.isfinite(initial[key]) & np.isfinite(current[key])
                    effects[key] = paired_mean_interval(initial[key][keep], current[key][keep])
        x, y = initial["test_mean_features"].astype(np.float64), current["test_mean_features"].astype(np.float64)
        x -= x.mean(0); y -= y.mean(0)
        effects["linear_cka_final"] = float(np.square(x.T @ y).sum() / np.sqrt(np.square(x.T @ x).sum() * np.square(y.T @ y).sum()))
        summary[group][comparison] = effects
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    rows = ["model,feature,maccs_auc,fg_auc,r2_mean,raw_knn1,standardized_knn1,raw_partial_rho,effective_rank"]
    for arm, result in data.items():
        for label, f in result["features"].items():
            r = f["ridge"]["1.0"]
            values = [arm, label, r["maccs_auc"], r["fg_auc"], r["r2_mean"], f["geometry"]["raw"]["knn1_tanimoto"], f["geometry"]["standardized"]["knn1_tanimoto"], f["geometry"]["raw"]["pair_partial_spearman"], f["collapse"]["effective_rank"]]
            rows.append(",".join(map(str, values)))
    (args.output / "metrics.csv").write_text("\n".join(rows) + "\n")
    print(json.dumps({g: summary[g] for g in ("paired_differences", "stage_differences")}, indent=2))


if __name__ == "__main__":
    main()
