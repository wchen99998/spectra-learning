"""Compare cheap pair summaries with the matched bias and learned-row pilots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("artifacts/pair_to_single_20260930"))
    parser.add_argument("--directory", type=Path, default=Path("artifacts/pair_to_single_efficient_20260930"))
    args = parser.parse_args()
    names = {"bias": "Pair bias", "bias_readout": "Bias + learned row pooling",
             "bias_poolnorm": "Bias + mean then normalize", "bias_cls": "Bias + CLS pair edges"}
    colors = {"bias": "#777f88", "bias_readout": "#7865a5", "bias_poolnorm": "#cd853f", "bias_cls": "#248878"}
    rows = []
    for arm in names:
        root = args.baseline if arm in {"bias", "bias_readout"} else args.directory
        for seed in (0, 1):
            folder = root / f"pilot_{arm}_s{seed}"
            training = json.loads((folder / "training.json").read_text())
            for readout, filename in (("mean_cls", "evaluation.json"), ("peak_mean", "evaluation_mean.json")):
                evaluation = json.loads((folder / filename).read_text())
                row = {"arm": arm, "seed": seed, "readout": readout,
                       "feature_dim": 256 if readout == "mean_cls" else 128,
                       "params": training["params"], "steps": training["steps"],
                       "steps_per_second": training["steady_steps_per_second"],
                       "validation_loss": training["validation"]["intact"]["loss"],
                       "mae_loss": training["validation"]["intact"]["mae_mz_loss"],
                       "distogram_loss": training["validation"]["intact"]["distogram_loss"],
                       "maccs_test_auc": evaluation["maccs_test_auc"],
                       "maccs_val_auc": evaluation["maccs_val_auc"],
                       "ridge_alpha": evaluation["ridge_alpha"]}
                for protocol in ("retrieval_raw", "retrieval_centered", "retrieval_probe", "retrieval_molecule_maccs"):
                    row.update({f"{protocol}_{k}": v for k, v in evaluation[protocol].items()})
                for intervention in ("zero", "shuffle"):
                    m = training["validation"].get(intervention)
                    row[f"{intervention}_delta_loss"] = (
                        m["loss"] - row["validation_loss"] if m is not None else "")
                rows.append(row)
    (args.directory / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (args.directory / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    benchmark = json.loads((args.directory / "benchmark_interleaved.json").read_text())["rows"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for arm, label in names.items():
        points = sorted([r for r in benchmark if r["arm"] == arm], key=lambda r: r["length"])
        x = [r["length"] for r in points]
        axes[0].plot(x, [r["median_ms"] for r in points], marker="o", color=colors[arm], label=label)
        axes[0].fill_between(x, [np.quantile(r["timings_ms"], .1) for r in points],
                             [r["p90_ms"] for r in points], color=colors[arm], alpha=.1)
        axes[1].plot(x, [r["compiler_memory_bytes"] / 2**20 for r in points],
                     marker="o", color=colors[arm], label=label)
    axes[0].set_ylabel("Forward + backward median (ms)")
    axes[1].set_ylabel("Compiler-estimated total memory (MiB)")
    for axis in axes:
        axis.set_xlabel("Tokens, including CLS")
        axis.set_xticks([32, 128, 256])
        axis.grid(alpha=.2)
    fig.suptitle("Cheap pair → single readouts at D=1536, C=640, batch=8", x=.02, ha="left", fontsize=13)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.53, .065), ncol=2, frameon=False)
    fig.text(.02, .012, "H100 shared with another job. Interleaved timing; shading = p10–p90, not a confidence interval. Pair trunk and optimizer excluded.",
             fontsize=8, color="#56616a")
    fig.tight_layout(rect=(0, .22, 1, .92))
    fig.savefig(args.directory / "interface_cost.png", dpi=180)
    fig.savefig(args.directory / "interface_cost.svg")
    plt.close(fig)
    for readout in ("mean_cls", "peak_mean"):
        for arm in names:
            selected = [r for r in rows if r["arm"] == arm and r["readout"] == readout]
            print(json.dumps({"arm": arm, "readout": readout, **{
                k: float(np.mean([r[k] for r in selected]))
                for k in ("validation_loss", "maccs_test_auc", "retrieval_raw_top1", "steps_per_second")}}))


if __name__ == "__main__":
    main()
