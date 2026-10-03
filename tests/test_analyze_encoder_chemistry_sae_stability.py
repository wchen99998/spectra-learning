import numpy as np
import pytest
from scipy.stats import spearmanr

from scripts.analyze_encoder_chemistry_sae_stability import align_features, normalized_rank_columns


def test_normalized_rank_correlations_match_scipy_with_ties():
    x = np.array([[0, 1, 3], [0, 2, 3], [2, 0, 3], [3, 2, 3], [4, 2, 3]])
    ranks = normalized_rank_columns(x)
    assert np.dot(ranks[:, 0], ranks[:, 1]) == pytest.approx(spearmanr(x[:, 0], x[:, 1]).statistic)
    assert np.all(ranks[:, 2] == 0)


def test_activation_alignment_is_train_only_and_distinct_from_decoder_alignment():
    x = np.array([[0, 0], [0, 1], [2, 0], [3, 2], [4, 2], [99, -99.]], float)
    y = np.column_stack([x[:, 1], x[:, 0], x[:, 0], np.ones(6)])
    left = np.array([[1., 0], [0, 1]])
    right = np.array([[1., 0], [0, 1], [.5, .5], [-1., 0]])
    first = align_features(x, y, np.arange(5), [0], left, right)
    assert first[0]['activation_alignment']['aligned_feature'] == 1  # smaller tied ID
    assert first[0]['activation_alignment']['train_spearman'] == pytest.approx(1)
    assert first[0]['decoder_alignment']['aligned_feature'] == 0
    assert first[0]['decoder_alignment']['mutual_nearest_decoder']
    x[5] = np.nan
    y[5] = np.nan
    assert align_features(x, y, np.arange(5), [0], left, right) == first


def test_constant_selected_primary_cannot_be_interpreted_as_stable():
    x = np.ones((5, 1))
    with pytest.raises(ValueError, match='no training activation variation'):
        align_features(x, np.arange(5.)[:, None], np.arange(5), [0], np.ones((1, 2)), np.ones((1, 2)))
