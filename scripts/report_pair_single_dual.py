"""Aggregate ``pair_single_dual_eval`` results into per-arm tables and a CSV."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = (
    ("val_mae", lambda c, r: c["validation"].get("mae_mz_loss")),
    ("val_distogram", lambda c, r: c["validation"].get("distogram_loss")),
    ("test_auc", lambda c, r: r.get("test_auc")),
    ("confirm_auc", lambda c, r: r.get("confirm_auc")),
    ("test_top1", lambda c, r: r.get("test_retrieval", {}).get("top1")),
    ("confirm_top1", lambda c, r: r.get("confirm_retrieval", {}).get("top1")),
    ("test_mrr", lambda c, r: r.get("test_retrieval", {}).get("mrr")),
    ("confirm_mrr", lambda c, r: r.get("confirm_retrieval", {}).get("mrr")),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.root.glob("*/evaluation.json")):
        result = json.loads(path.read_text())
        final = result["checkpoints"][-1]
        for readout in ("single", "pair", "both"):
            if readout not in final:
                continue
            row = {"arm": result["arm"], "seed": result["seed"], "readout": readout,
                   "step": final["step"], "params": result["params"]}
            row.update({name: fn(final, final[readout]) for name, fn in METRICS})
            rows.append(row)
    if args.csv:
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["arm"], row["readout"])].append(row)
    names = [name for name, _ in METRICS]
    print("| arm | readout | n | " + " | ".join(names) + " |")
    print("|" + "---|" * (3 + len(names)))
    for (arm, readout), group in grouped.items():
        cells = []
        for name in names:
            values = [r[name] for r in group if r[name] is not None]
            if not values:
                cells.append("")
                continue
            scale = 100 if "top1" in name else 1
            mean, sd = np.mean(values) * scale, np.std(values, ddof=1) * scale if len(values) > 1 else 0
            cells.append(f"{mean:.4f}±{sd:.4f}")
        print(f"| {arm} | {readout} | {len(group)} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
