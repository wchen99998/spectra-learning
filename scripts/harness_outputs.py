"""
harness_outputs.py
==================
Per-candidate retrieval table for the Novogaia evaluation harness, same format as
jepa-fps `scripts/harness_outputs.py` (PR #1).

Deliberately agnostic to how candidate pools are built or scored: callers hand
over, per query, the ground-truth SMILES, the candidate SMILES and their scores
(higher = better). Ranks are recomputed here with the same convention as
run_retrieval_jax.py (rank = 1 + # candidates scoring STRICTLY higher, i.e.
optimistic on ties).

Output columns (one row per query x candidate):
    query_id             MassSpecGym identifier of the query spectrum
    ground_truth_smiles  true molecule of the query
    candidate_smiles     candidate SMILES
    rank                 1-based rank of the candidate within the query's pool
"""

import csv
import gzip
from pathlib import Path

import numpy as np

HARNESS_DIR = "harness_outputs"
HARNESS_FILE = "structure_retrieval_candidates.csv.gz"
COLUMNS = ("query_id", "ground_truth_smiles", "candidate_smiles", "rank")


def write_retrieval_hits(out_base, query_results):
    """
    query_results: iterable of (query_id, ground_truth_smiles, candidate_smiles: list[str], scores: array-like)
    Writes <out_base>/harness_outputs/structure_retrieval_candidates.csv.gz, rows sorted by query then rank.
    """
    out_dir = Path(out_base) / HARNESS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / HARNESS_FILE
    n_rows = 0
    with gzip.open(out, "wt", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(COLUMNS)
        for qid, truth, cands, scores in query_results:
            scores = np.asarray(scores, dtype=np.float64)
            order = np.argsort(-scores, kind="stable")
            s_sorted = scores[order]
            # rank = 1 + #strictly-higher: position of the first element >= own score in descending order
            ranks = 1 + np.searchsorted(-s_sorted, -s_sorted, side="left")
            writer.writerows((qid, truth, cands[j], int(r)) for j, r in zip(order, ranks))
            n_rows += len(order)
    return out, n_rows
