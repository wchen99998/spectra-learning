import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

from scripts.periodic_semantics_stats import macro_auc, nearest_mass, rank_correlation, paired_mean_interval


def test_tied_auc_matches_sklearn():
    y = np.array([[0, 1], [1, 0], [0, 1], [1, 0]])
    p = np.array([[.5, .2], [.5, .2], [.8, .2], [.8, .7]])
    assert np.isclose(macro_auc(p, y), roc_auc_score(y, p, average="macro"))
    assert macro_auc(np.ones_like(y), y) == .5


def test_mass_baseline_checks_both_neighbours_and_edges():
    train = np.array([200., 100., 300.])
    query = np.array([101., 199., 0., 400.])
    np.testing.assert_array_equal(nearest_mass(train, query), [1, 0, 1, 2])


def test_spearman_uses_average_ties():
    a, b = np.array([0, 0, 1, 2]), np.array([0, 1, 1, 2])
    assert np.isclose(rank_correlation(a, b), spearmanr(a, b).statistic)


def test_paired_interval_preserves_pairing():
    a = np.arange(20.)
    result = paired_mean_interval(a, a - .1, repeats=30)
    np.testing.assert_allclose(result['ci95'], [-.1, -.1])


def test_weighted_auc_bootstrap_matches_explicit_resampling():
    from scripts.analyze_periodic_semantics import bootstrap_auc_delta
    y = np.array([[0, 1], [1, 0], [0, 1], [1, 0], [1, 1], [0, 0]])
    a = np.array([[.2, .2], [.2, .8], [.8, .8], [.8, .5], [.5, .5], [.5, .2]])
    b = a[::-1].copy()
    rng = np.random.default_rng(20260929)
    expected = []
    for _ in range(20):
        ids = rng.integers(len(y), size=len(y))
        expected.append(macro_auc(b[ids], y[ids]) - macro_auc(a[ids], y[ids]))
    actual = bootstrap_auc_delta(a, b, y, repeats=20)
    np.testing.assert_allclose(actual['ci95'], np.quantile(expected, [.025, .975]))
