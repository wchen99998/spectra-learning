"""Prepare complete molecular pools for the selected raw-token retrieval recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.csv as pc


def read_rows(root: Path) -> dict[str, np.ndarray]:
    table = pc.read_csv(
        root / "msg15/MassSpecGym1.5.tsv",
        parse_options=pc.ParseOptions(delimiter="\t"),
        convert_options=pc.ConvertOptions(include_columns=["smiles", "inchikey", "fold"]),
    )
    return {name: np.asarray(table[name].to_pylist(), dtype=str) for name in table.column_names}


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def prepare_graphs(root: Path, out: Path, *, workers: int = 16, val_n: int = 5000) -> None:
    """Use seed 0, all eligible formula candidates and an atom cap of 160.

    val_n permits small fixtures without exposing a CLI tuning option.
    """
    from atom_flare_data import build_graph_store

    frame = read_rows(root)
    cached_fold = np.load(root / "data/fold.npy", allow_pickle=True).astype(str)
    if not np.array_equal(frame["fold"], cached_fold):
        raise ValueError("TSV and frozen input fold alignment differ")
    train = np.flatnonzero(frame["fold"] == "train")
    val = np.flatnonzero(frame["fold"] == "val")
    val = np.sort(np.random.default_rng(0).choice(val, val_n, replace=False))
    test = np.flatnonzero(frame["fold"] == "test")
    pool = json.loads((root / "msg15/MassSpecGym1.5_retrieval_candidates_formula.json").read_text())
    needed = set(frame["smiles"][train])
    for row in train:
        truth = frame["smiles"][row]
        needed.update(s for s in pool.get(truth, []) if s != truth)
    for row in np.concatenate([val, test]):
        needed.update(pool[frame["smiles"][row]])
    print(f"Preparing {len(needed):,} unique molecular graphs", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    build_graph_store(needed, out / "graphs", max_atoms=160, workers=workers)
    np.savez(out / "rows.npz", train=train, val=val, test=test,
             smiles=frame["smiles"], inchikey14=np.array([s[:14] for s in frame["inchikey"]]))
    write_json(out / "selection.json", {
        "seed": 0, "train_n": len(train), "val_n": len(val), "test_n": len(test),
        "validation_sha256": hashlib.sha256(val.tobytes()).hexdigest(),
        "molecules_requested": len(needed), "graph_store": str(out / "graphs"),
        "negative_bank": "all pool entries excluding exact truth; sample up to 8",
        "negative_bank_size": 0,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    prepare_graphs(args.input_root, args.out, workers=args.workers)


if __name__ == "__main__":
    main()
