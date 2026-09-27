"""Compute threshold-15 MCES@1 for selected retrieval predictions.

Only saved selected-model top-1 predictions are used. No training or checkpoint
selection occurs here. Compatible historical distances are reused with provenance;
new pair distances are written incrementally so solver work can be resumed.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np


def pair_key(a: str, b: str) -> str:
    return "||".join(sorted((a, b)))


def compute_pair(key: str) -> dict:
    from myopic_mces.myopic_mces import MCES

    a, b = key.split("||")
    _, value, seconds, mode = MCES(
        a, b, threshold=15, solver="PULP_CBC_CMD",
        solver_options={"msg": False, "threads": 1}, catch_errors=False,
    )
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"Invalid MCES result for {key}: {value}, mode={mode}")
    return {"key": key, "distance": float(value), "seconds": float(seconds),
            "mode": int(mode.value if hasattr(mode, "value") else mode)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="Selected run containing run/test_ranks.npz")
    parser.add_argument("--baseline", type=Path,
                        help="Optional reference run for an aligned comparison")
    parser.add_argument("--seed-cache", type=Path,
                        help="Optional compatible JSON distances with a sibling *_provenance.json")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    out = args.root / "full_results"
    out.mkdir(parents=True, exist_ok=True)
    records = {}
    needed = set()
    ranks = {}
    # Retain historical labels for explicitly requested comparison reports.
    runs = (("baseline", args.baseline), ("global_glacier", args.root)) if args.baseline else (("selected", args.root),)
    for label, root in runs:
        with np.load(root / "run/test_ranks.npz", allow_pickle=False) as saved:
            ranks[label] = {key: saved[key] for key in saved.files}
        needed.update(pair_key(a, b) for a, b in zip(
            ranks[label]["truth_smiles"], ranks[label]["top_smiles"], strict=True) if a != b)
    if args.baseline:
        for key in ("rows", "truth_smiles", "inchikey14", "candidate_counts"):
            np.testing.assert_array_equal(ranks["baseline"][key], ranks["global_glacier"][key])
    seed_path, provenance = args.seed_cache, None
    if seed_path is not None:
        provenance_path = seed_path.with_name(seed_path.stem + "_provenance.json")
        provenance = json.loads(provenance_path.read_text())
        assert provenance["threshold"] == 15
        assert provenance["compatibility"]["myopic_mces"] == importlib.metadata.version("myopic-mces")
        assert provenance["compatibility"]["rdkit"] == importlib.metadata.version("rdkit")
        assert hashlib.sha256(seed_path.read_bytes()).hexdigest() == provenance["sha256"]
        for key, value in json.loads(seed_path.read_text()).items():
            if key in needed and np.isfinite(value) and value >= 0:
                records[key] = {"key": key, "distance": value, "mode": "historical_cache"}
    seed_count = len(records)
    journal = out / "mces_pairs.jsonl"
    if journal.exists():
        for line in journal.read_text().splitlines():
            record = json.loads(line)
            if record["key"] in records:
                assert record["distance"] == records[record["key"]]["distance"]
            records[record["key"]] = record
    todo = sorted(needed - records.keys())
    started = time.monotonic()
    protocol = {
        "threshold": 15, "solver": "PULP_CBC_CMD", "solver_options": {"msg": False, "threads": 1},
        "time_limit": None, "exact_below_threshold_lower_bound_above": True,
        "seed_cache": str(seed_path) if seed_path is not None else None, "seed_provenance": provenance,
        "cached_pairs_reused": seed_count, "unique_nonidentical_pairs": len(needed),
        "versions": {name: importlib.metadata.version(name) for name in ("myopic-mces", "pulp", "rdkit", "numpy")},
        "rank_files_sha256": {label: hashlib.sha256((root / "run/test_ranks.npz").read_bytes()).hexdigest()
                              for label, root in runs},
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "certified_threshold_cases": [
            {"pair": record["key"], "distance": record["distance"],
             "certificate": record["certificate"],
             "certificate_sha256": hashlib.sha256((out / record["certificate"]).read_bytes()).hexdigest(),
             "computation": record["computation"]}
            for record in records.values() if "certificate" in record
        ],
    }
    (out / "mces_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps({"needed": len(needed), "cached": len(records), "to_compute": len(todo)}), flush=True)
    with journal.open("a") as stream, ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(compute_pair, key) for key in todo]
        for index, future in enumerate(as_completed(futures), 1):
            record = future.result()
            records[record["key"]] = record
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            if index % 50 == 0 or index == len(todo):
                status = {"computed_this_run": index, "to_compute": len(todo),
                          "total_completed_pairs": len(records), "total_pairs": len(needed),
                          "elapsed_seconds": time.monotonic() - started}
                (out / "mces_progress.json").write_text(json.dumps(status, indent=2) + "\n")
                print(json.dumps(status), flush=True)
    arrays, summary = {}, {}
    for label, data in ranks.items():
        values = np.asarray([0.0 if a == b else records[pair_key(a, b)]["distance"]
                             for a, b in zip(data["truth_smiles"], data["top_smiles"], strict=True)])
        assert np.isfinite(values).all() and (values >= 0).all()
        arrays[label] = values
        summary[label] = {"mces_at_1": float(values.mean()), "n": len(values), "n_failed": 0}
    arrays["rows"] = ranks[runs[0][0]]["rows"]
    np.savez(out / "mces_per_query.npz", **arrays)
    (out / "mces_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
