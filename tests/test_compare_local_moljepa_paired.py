import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from compare_local_moljepa_paired import Reference, align_records, compare_ranks


def test_paired_hit_and_rank_changes_and_clustered_bootstrap():
    old = np.array([1, 2, 3, 1])
    new = np.array([2, 1, 2, 1])
    truths = np.array(['A', 'A', 'B', 'B'])
    result = compare_ranks(old, new, truths, repetitions=2000, seed=3)
    hit = result['hit1']
    assert (hit['gains'], hit['losses'], hit['ties']) == (1, 1, 2)
    assert hit['both_correct'] == hit['both_incorrect'] == 1
    assert hit['delta_percentage_points'] == 0
    assert result['rank']['improved'] == 2
    assert result['rank']['worsened'] == result['rank']['unchanged'] == 1
    # Each molecule has zero net Hit@1 change, while individual spectra differ.
    assert hit['bootstrap']['molecule_cluster_ci95_percentage_points'] == [0, 0]
    lower, upper = hit['bootstrap']['query_ci95_percentage_points']
    assert lower < 0 < upper
    assert result == compare_ranks(old, new, truths, repetitions=2000, seed=3)
    unchanged = compare_ranks(old, old, truths, repetitions=100, seed=3)
    assert unchanged['hit1']['bootstrap']['query_ci95_percentage_points'] == [0, 0]


def test_alignment_accepts_reordering_and_rejects_wrong_rows_counts_or_predictions():
    ref = Reference(np.array([2, 4]), np.array(['A', 'B']),
                    {'stable': np.array([1, 2]), 'optimistic': np.array([1, 2])},
                    np.array([2, 2]), {'A': ['A', 'X'], 'B': ['B', 'Y']}, {})
    records = [{'row': 4, 'stable_rank': 2, 'optimistic_rank': 2, 'prediction': 'Y', 'candidates': 2},
               {'row': 2, 'stable_rank': 1, 'optimistic_rank': 1, 'prediction': 'A', 'candidates': 2}]
    np.testing.assert_array_equal(align_records(ref, records)['stable'], [1, 2])
    for field, value in [('row', 99), ('candidates', 3), ('prediction', 'Q'), ('row', 4.5)]:
        invalid = [dict(record) for record in records]
        invalid[0][field] = value
        with pytest.raises(ValueError):
            align_records(ref, invalid)
