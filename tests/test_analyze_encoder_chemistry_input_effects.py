import numpy as np

from scripts.analyze_encoder_chemistry_input_effects import contrast, ensemble


def test_unsigned_labels_have_signed_margins_and_probabilities_ensemble_before_loss():
    labels = np.array([[0, 1]], np.uint8)
    logits = np.array([[[[4., -4.], [0., 0.], [0., 0.]]]])
    result = ensemble(logits, labels)
    np.testing.assert_allclose(result["signed_margin"], [[[-4 / 3, -4 / 3]]])
    expected = ((1 / (1 + np.exp(-4))) + .5 + .5) / 3
    np.testing.assert_allclose(result["probability"][0, 0], [expected, 1 - expected])
    np.testing.assert_allclose(result["bce"][0, 0], -np.log(1 - expected))


def test_unsupported_unchanged_molecules_do_not_dilute_paired_effects():
    labels = np.ones((2, 1), np.uint8)
    logits = np.zeros((2, 2, 3, 1))
    logits[1, 0] = -2
    phase = {"metrics": ensemble(logits, labels)}
    eligible = np.array([True, False])
    result, rows = contrast(phase, 0, 1, eligible, {"all": np.array([0])})
    expected = np.logaddexp(0, 2) - np.log(2)
    assert result["eligible_molecules"] == 1
    np.testing.assert_allclose(result["groups"]["all"]["delta_bce"]["mean"], expected)
    np.testing.assert_allclose(result["groups"]["all"]["delta_bce"]["ci95"], [expected, expected])
    np.testing.assert_array_equal(rows["all"]["delta_bce"][1], 0)
