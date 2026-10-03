import numpy as np

from scripts.analyze_encoder_chemistry_sae_causal import holm_adjust, paired_summary, sparse_code_audit


def test_paired_bootstrap_identity_and_constant_shift():
    counts = np.random.default_rng(8).multinomial(8, np.full(8, 1 / 8), size=200)
    values = np.stack([np.zeros(8), np.ones(8) * -.25], axis=1)
    result = paired_summary(values, np.ones(8, bool), counts, ("identity", "shift"))
    assert result["metrics"]["identity"] == {"mean": 0., "ci95": [0., 0.]}
    assert result["metrics"]["shift"] == {"mean": -.25, "ci95": [-.25, -.25]}
    empty = paired_summary(values, np.zeros(8, bool), counts, ("identity", "shift"))
    assert empty["molecules"] == 0 and empty["metrics"]["identity"] is None


def test_holm_adjust_restores_input_order_and_is_monotone():
    np.testing.assert_allclose(holm_adjust([.02, .001, .3, .008]), [.04, .004, .3, .024])


def test_sparse_audit_ignores_order_but_detects_positive_threshold_flip():
    export = {"rows": np.array([0, 0, 2]), "slots": np.array([0, 2, 1]),
              "feature_indices": np.array([[1, 3], [2, 4], [3, 5]]),
              "feature_values": np.array([[2., 1.], [3., 0.], [4., 2.]])}
    valid = np.array([[False, True, False], [True, False, True]])
    indices = np.full((2, 3, 2), -1)
    values = np.zeros((2, 3, 2))
    indices[valid] = [[5, 3], [3, 1], [2, 6]]
    values[valid] = [[2., 4.], [1., 2.], [3., .1]]
    selected = np.zeros((1, 2, 3)); selected[0, 0, 1] = 4.; selected[0, 1, 0] = 1.
    saved = {"rows": np.array([2, 0]), "valid_peak_mask": valid, "sparse_code_indices": indices,
             "sparse_code_values": values, "selected_feature_activations": selected}
    result = sparse_code_audit(saved, export, [3])
    assert result["positive_membership_changed_tokens"] == 1
    assert result["selected_features"][0]["firing_membership_flips"] == 0
    np.testing.assert_allclose(result["maximum_token_code_l2_difference"], .1)
