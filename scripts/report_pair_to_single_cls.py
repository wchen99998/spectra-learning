"""Compare CLS removal with matched peak-mean and original concatenated readouts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("artifacts/pair_to_single_20260930"))
    parser.add_argument("--no-cls", type=Path, default=Path("artifacts/pair_to_single_no_cls_20260930"))
    args = parser.parse_args()
    arms = {"none": "No pair feedback", "bias": "Pair bias", "bias_readout": "Bias + pair values"}
    conditions = {
        "cls_concat": (args.baseline, "evaluation.json", "CLS on, mean + CLS (256D)", "#91989d"),
        "cls_mean": (args.baseline, "evaluation_mean.json", "CLS on, peak mean (128D)", "#426da4"),
        "no_cls_mean": (args.no_cls, "evaluation.json", "CLS off, peak mean (128D)", "#308879"),
    }
    rows = []
    for arm in arms:
        for seed in (0, 1):
            for condition, (root, filename, _, _) in conditions.items():
                folder = root / f"pilot_{arm}_s{seed}"
                training = json.loads((folder / "training.json").read_text())
                evaluation = json.loads((folder / filename).read_text())
                intact = training["validation"]["intact"]
                row = {"arm": arm, "seed": seed, "condition": condition,
                       "use_cls_token": training.get("use_cls_token", True),
                       "feature_dim": 256 if condition == "cls_concat" else 128,
                       "steps": training["steps"], "params": training["params"],
                       "validation_loss": intact["loss"], "mae_loss": intact["mae_mz_loss"],
                       "distogram_loss": intact["distogram_loss"],
                       "steps_per_second": training["steady_steps_per_second"],
                       "maccs_test_auc": evaluation["maccs_test_auc"],
                       "maccs_val_auc": evaluation["maccs_val_auc"],
                       "ridge_alpha": evaluation["ridge_alpha"]}
                for protocol in ("retrieval_raw", "retrieval_centered", "retrieval_probe", "retrieval_molecule_maccs"):
                    row.update({f"{protocol}_{k}": v for k, v in evaluation[protocol].items()})
                for intervention in ("zero", "shuffle"):
                    metrics = training["validation"].get(intervention)
                    row[f"{intervention}_delta_loss"] = (
                        metrics["loss"] - intact["loss"] if metrics is not None else "")
                rows.append(row)
    (args.no_cls / "cls_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (args.no_cls / "cls_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.4), sharey=True)
    specs = [("validation_loss", "Prediction loss ↓", 1),
             ("maccs_test_auc", "Chemical probe AUC ↑", 1),
             ("retrieval_raw_top1", "Spectral retrieval top-1 (%) ↑", 100)]
    for axis, (metric, title, scale) in zip(axes, specs, strict=True):
        for i, arm in enumerate(arms):
            for offset, (condition, (_, _, _, color)) in zip((-.2, 0, .2), conditions.items(), strict=True):
                selected = [r for r in rows if r["arm"] == arm and r["condition"] == condition]
                values = [r[metric] * scale for r in selected]
                axis.plot([min(values), max(values)], [i + offset] * 2, color=color, lw=1.5)
                for row, value in zip(selected, values, strict=True):
                    axis.scatter(value, i + offset, color=color,
                                 marker="o" if row["seed"] == 0 else "s", s=36, zorder=3)
        axis.set_title(title, loc="left", pad=12)
        axis.set_yticks(range(len(arms)), list(arms.values()))
        axis.grid(axis="x", color="#e2e6e9", lw=.7)
        axis.set_axisbelow(True)
        axis.tick_params(axis="y", length=0)
    axes[0].invert_yaxis()
    fig.suptitle("Removing CLS: token effect versus embedding readout effect", x=.02, ha="left", fontsize=13)
    handles = [Line2D([0], [0], color=color, marker="o", label=label)
               for _, _, label, color in conditions.values()]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.55, .07), frameon=False, ncol=3)
    fig.text(.02, .015, "3,000 matched steps; circle = seed 0, square = seed 1. Lines show seed range, not confidence intervals.",
             fontsize=9, color="#56616a")
    fig.tight_layout(rect=(0, .17, 1, .92))
    fig.savefig(args.no_cls / "cls_outcomes.png", dpi=180)
    fig.savefig(args.no_cls / "cls_outcomes.svg")
    plt.close(fig)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
