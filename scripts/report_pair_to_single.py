"""Collect the small pair->single pilots and plot the three measured outcomes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("artifacts/pair_to_single_20260930"))
    args = parser.parse_args()
    rows = []
    for path in sorted(args.directory.glob("pilot_*/evaluation.json")):
        evaluation = json.loads(path.read_text())
        training = json.loads((path.parent / "training.json").read_text())
        row = {"arm": training["arm"], "seed": training["seed"],
                     "steps": training["steps"], "params": training["params"],
                     "validation_loss": training["validation"]["intact"]["loss"],
                     "mae_loss": training["validation"]["intact"]["mae_mz_loss"],
                     "distogram_loss": training["validation"]["intact"]["distogram_loss"],
                     "steps_per_second": training["steady_steps_per_second"], **evaluation}
        interventions = path.parent / "interventions.json"
        if interventions.exists():
            row["interventions"] = json.loads(interventions.read_text())
        rows.append(row)
    (args.directory / "pilot_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    flat_rows = []
    for row in rows:
        flat = {k: v for k, v in row.items() if not isinstance(v, dict)}
        for protocol in ("retrieval_raw", "retrieval_centered", "retrieval_probe", "retrieval_molecule_maccs"):
            flat.update({f"{protocol}_{k}": v for k, v in row[protocol].items()})
        effects = row.get("interventions", {})
        for condition in ("intact", "zero", "shuffle"):
            flat[f"intervention_{condition}_loss"] = effects.get(condition, {}).get("loss", "")
        flat_rows.append(flat)
    with (args.directory / "pilot_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    arms = [arm for arm in ("none", "bias", "mean", "readout", "bias_readout", "bias_diagonal", "bias_weighted")
            if any(row["arm"] == arm for row in rows)]
    labels = {"none": "No feedback", "bias": "Bias", "mean": "Mean values",
              "readout": "Pooled values", "bias_readout": "Bias + pooled values",
              "bias_diagonal": "Bias + diagonal values", "bias_weighted": "Shared routing + values"}
    colors = {"none": "#87939b", "bias": "#395f93", "mean": "#a67c52",
              "readout": "#348c82", "bias_readout": "#7761a8", "bias_diagonal": "#bb663f", "bias_weighted": "#b44673"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.2), sharey=True)
    specs = [("validation_loss", "Masked prediction loss ↓", lambda r: r["validation_loss"]),
             ("maccs_test_auc", "Chemical probe AUC ↑", lambda r: r["maccs_test_auc"]),
             ("retrieval", "Spectral retrieval top-1 ↑", lambda r: r["retrieval_raw"]["top1"] * 100)]
    for axis, (_, title, get_value) in zip(axes, specs, strict=True):
        for i, arm in enumerate(arms):
            selected = [row for row in rows if row["arm"] == arm]
            values = [get_value(row) for row in selected]
            axis.plot([min(values), max(values)], [i, i], color=colors[arm], lw=2)
            for row, value in zip(selected, values, strict=True):
                axis.scatter(value, i, color=colors[arm], marker="o" if row["seed"] == 0 else "s", s=45, zorder=3)
        axis.set_title(title, loc="left", pad=16)
        axis.set_yticks(range(len(arms)), [labels[arm] for arm in arms])
        axis.grid(axis="x", color="#e2e6e9", lw=.7)
        axis.set_axisbelow(True)
        axis.tick_params(axis="y", length=0)
    axes[0].invert_yaxis()
    control = json.loads((args.directory / "binned_spectrum/evaluation.json").read_text())
    axes[1].axvline(control["maccs_test_auc"], color="#777777", linestyle=":", linewidth=1)
    # The raw-spectrum retrieval control is explicitly annotated rather than
    # omitted because it lies far outside the pilot embeddings' narrow range.
    axes[2].set_xlabel(f"Percent; raw spectrum baseline = {100 * control['retrieval_raw']['top1']:.2f}%")
    fig.suptitle("Pair → single: prediction, chemistry, and retrieval can move differently", x=.02, ha="left", fontsize=13)
    fig.text(.02, .015, "3,000 matched steps / arm; circle = seed 0, square = seed 1. Lines show seed range, not confidence intervals.", fontsize=9, color="#56616a")
    fig.tight_layout(rect=(0, .06, 1, .91))
    fig.savefig(args.directory / "pilot_outcomes.png", dpi=180)
    fig.savefig(args.directory / "pilot_outcomes.svg")
    plt.close(fig)
    historical = json.loads((args.directory / "historical/joint_fastmixer_1b_150k.json").read_text())
    fig, axis = plt.subplots(figsize=(7.2, 3.8))
    for name, label, color in [("single_mean", "Single mean", colors["bias"]),
                                ("single_cls", "Single CLS", colors["none"]),
                                ("pair_mean", "Pair mean", colors["bias_readout"]),
                                ("diag_mean", "Pair diagonal mean", colors["bias_diagonal"])]:
        values = [historical["layers"][str(i)][name]["auc_maccs_macro"] for i in range(1, 26)]
        axis.plot(range(1, 26), values, color=color, label=label)
    axis.set(xlabel="Encoder block", ylabel="MACCS macro AUC",
             title="Original joint checkpoint: pair chemistry is easier to read")
    axis.legend(frameon=False, ncol=2)
    axis.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(args.directory / "historical_stream_gap.png", dpi=180)
    plt.close(fig)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
