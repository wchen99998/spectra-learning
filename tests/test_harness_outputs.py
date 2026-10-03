import csv
import gzip

from scripts.harness_outputs import write_retrieval_hits


def test_rows_sorted_by_rank_with_optimistic_ties_and_quoted_smiles(tmp_path):
    path, n_rows = write_retrieval_hits(tmp_path, [
        ("q1", "A", ["A", "B", "C", "D"], [0.5, 0.9, 0.5, 0.1]),
        ("q2", "C,C", ["C,C"], [1.0]),
    ])
    assert path == tmp_path / "harness_outputs/structure_retrieval_candidates.csv.gz"
    with gzip.open(path, "rt", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert n_rows == len(rows) == 5
    assert [(r["query_id"], r["candidate_smiles"], r["rank"]) for r in rows] == [
        ("q1", "B", "1"), ("q1", "A", "2"), ("q1", "C", "2"), ("q1", "D", "4"), ("q2", "C,C", "1")]
    assert rows[-1]["ground_truth_smiles"] == "C,C"
