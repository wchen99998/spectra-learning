"""Summarize the prespecified 15k-step, three-seed pair-to-single comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("artifacts/pair_to_single_long_20260930"))
    args = parser.parse_args()
    names = {"bias": "Pair bias", "bias_poolnorm": "Bias + mean-first values"}
    colors = {"bias": "#667583", "bias_poolnorm": "#bd762f"}
    rows, endpoints = [], []
    for arm in names:
        for seed in (0, 1, 2):
            for step in (3000, 7500, 15000):
                folder = args.directory / f"pilot_{arm}_s{seed}" / f"step_{step:06d}"
                validation = json.loads((folder / "validation_scores.json").read_text())
                training = json.loads((folder / "training.json").read_text())
                for readout, filename in (("mean_cls", "evaluation.json"), ("peak_mean", "evaluation_mean.json")):
                    loss = validation["validation"]["intact"]
                    row = {"arm": arm, "seed": seed, "step": step, "readout": readout,
                           "params": training["params"], "epochs": training["sampled_epoch_equivalents"],
                           "validation_loss": loss["loss"], "validation_mae": loss["mae_mz_loss"],
                           "validation_distogram": loss["distogram_loss"],
                           "maccs_val_auc": validation[readout]["maccs_val_auc"],
                           "val_retrieval_top1": validation[readout]["retrieval_raw"]["top1"],
                           "val_ridge_alpha": validation[readout]["ridge_alpha"]}
                    for condition in ("zero", "shuffle"):
                        metrics = validation["validation"].get(condition)
                        row[f"{condition}_delta_loss"] = metrics["loss"] - loss["loss"] if metrics else ""
                    rows.append(row)
                    if step == 15000:
                        test = json.loads((folder / filename).read_text())
                        confirmation = json.loads((folder / "confirmation" / filename).read_text())
                        endpoint = {**row, "maccs_test_auc": test["maccs_test_auc"],
                                    "test_retrieval_top1": test["retrieval_raw"]["top1"],
                                    "test_retrieval_top10": test["retrieval_raw"]["top10"],
                                    "test_retrieval_mrr": test["retrieval_raw"]["mrr"],
                                    "test_centered_top1": test["retrieval_centered"]["top1"],
                                    "test_probe_top1": test["retrieval_probe"]["top1"],
                                    "test_molecule_maccs_top1": test["retrieval_molecule_maccs"]["top1"],
                                    "confirmation_auc": confirmation["maccs_test_auc"],
                                    "confirmation_top1": confirmation["retrieval_raw"]["top1"]}
                        endpoints.append(endpoint)
    for filename, data in (("validation_curve.csv", rows), ("endpoints.csv", endpoints)):
        with (args.directory / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)

    metrics = ("validation_loss", "maccs_test_auc", "test_retrieval_top1", "confirmation_auc", "confirmation_top1")
    summary = {}
    for readout in ("mean_cls", "peak_mean"):
        summary[readout] = {}
        for metric in metrics:
            values = {arm: np.asarray([row[metric] for row in endpoints
                                      if row["arm"] == arm and row["readout"] == readout]) for arm in names}
            differences = values["bias_poolnorm"] - values["bias"]
            summary[readout][metric] = {
                **{arm: {"mean": float(value.mean()), "seed_sd": float(value.std(ddof=1)),
                         "per_seed": value.tolist()} for arm, value in values.items()},
                "paired_delta": {"mean": float(differences.mean()),
                                 "seed_sd": float(differences.std(ddof=1)),
                                 "per_seed": differences.tolist(),
                                 "improved_seeds": int(np.sum(differences < 0 if metric == "validation_loss"
                                                              else differences > 0))}}

    for readout in ("mean_cls", "peak_mean"):
        for scope, metric_names in (("", {"auc": "maccs_test_auc", "top1": "test_retrieval_top1"}),
                                   ("confirmation", {"auc": "confirmation_auc", "top1": "confirmation_top1"})):
            samples = {arm: [] for arm in names}
            identities = []
            for arm in names:
                for seed in (0, 1, 2):
                    folder = args.directory / f"pilot_{arm}_s{seed}" / "step_015000" / scope
                    samples[arm].append(dict(np.load(folder / "molecule_bootstrap.npz")))
                    metadata = json.loads((folder / "molecule_bootstrap.json").read_text())
                    identities.append((metadata["replicates"], metadata["rng_seed"], metadata["test_indices_sha256"]))
            if len(set(identities)) != 1:
                raise ValueError("Paired bootstrap requires identical molecule resamples and evaluation rows")
            for key, metric in metric_names.items():
                differences = np.mean([samples["bias_poolnorm"][seed][f"{key}_{readout}"]
                                       - samples["bias"][seed][f"{key}_{readout}"] for seed in (0, 1, 2)], axis=0)
                summary[readout][metric]["paired_delta"]["molecule_bootstrap_95_ci"] = np.quantile(
                    differences, [.025, .975]).tolist()
    (args.directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.directory / "paired_deltas.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["readout", "metric", "mean_delta", "seed_sd", "improved_seeds", "ci_low", "ci_high"])
        writer.writeheader()
        for readout, comparisons in summary.items():
            for metric, comparison in comparisons.items():
                paired = comparison["paired_delta"]
                interval = paired.get("molecule_bootstrap_95_ci", ["", ""])
                writer.writerow({"readout": readout, "metric": metric, "mean_delta": paired["mean"],
                                 "seed_sd": paired["seed_sd"], "improved_seeds": paired["improved_seeds"],
                                 "ci_low": interval[0], "ci_high": interval[1]})

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.1))
    for axis, metric, label, scale in zip(
            axes, ("validation_loss", "maccs_val_auc", "val_retrieval_top1"),
            ("Masked prediction loss ↓", "Validation MACCS AUC ↑", "Validation spectral top-1 (%) ↑"),
            (1, 1, 100), strict=True):
        for arm in names:
            for readout in (("mean_cls",) if metric == "validation_loss" else ("mean_cls", "peak_mean")):
                points = [np.asarray([row[metric] * scale for row in rows if row["arm"] == arm
                                      and row["readout"] == readout and row["step"] == step])
                          for step in (3000, 7500, 15000)]
                mean, sd = np.asarray([x.mean() for x in points]), np.asarray([x.std(ddof=1) for x in points])
                axis.plot([3, 7.5, 15], mean, color=colors[arm], marker="o",
                          linestyle="-" if readout == "mean_cls" else "--")
                axis.fill_between([3, 7.5, 15], mean - sd, mean + sd, color=colors[arm], alpha=.08)
        axis.set(xlabel="Training steps (thousands)", ylabel=label, xticks=[3, 7.5, 15])
        axis.grid(alpha=.2)
    fig.suptitle("Longer pair → single training: matched 15k schedule, three seeds", x=.015, ha="left", fontsize=13)
    fig.text(.015, .075, "Gray: pair bias. Orange: mean-first values + bias. Solid: mean + CLS. Dashed: peak mean. Shading: seed SD, not a confidence interval.", fontsize=8)
    fig.text(.015, .025, "All learning curves use validation. Test comparison uses only the prespecified 15k endpoint; 2.5M-parameter pilot, 31 peaks + CLS.", fontsize=8)
    fig.tight_layout(rect=(0, .12, 1, .92))
    fig.savefig(args.directory / "learning_curves.png", dpi=180)
    fig.savefig(args.directory / "learning_curves.svg")
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
