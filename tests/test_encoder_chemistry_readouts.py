from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.encoder_chemistry_readouts import (
    apply_readout, binary_metrics, init_readout, load_readout,
    normalized_tokens, save_readout, train_readout,
)


def _tokens():
    tokens = jax.random.normal(jax.random.key(5), (3, 5, 8))
    mask = jnp.asarray([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]], dtype=jnp.bool_)
    return tokens, mask


@pytest.mark.parametrize("kind", ["mean", "attention"])
def test_permutation_and_extra_padding_preserve_predictions(kind):
    tokens, mask = _tokens()
    params = init_readout(jax.random.key(0), 8, 3, kind)
    logits, weights = apply_readout(params, tokens, mask, kind)
    permutation = jnp.asarray([3, 1, 4, 0, 2])
    permuted, permuted_weights = apply_readout(params, tokens[:, permutation], mask[:, permutation], kind)
    np.testing.assert_allclose(permuted, logits, atol=3e-7, rtol=3e-7)
    padded, _ = apply_readout(params, jnp.pad(tokens, ((0, 0), (0, 4), (0, 0)), constant_values=1234),
                              jnp.pad(mask, ((0, 0), (0, 4))), kind)
    np.testing.assert_allclose(padded, logits, atol=3e-7, rtol=3e-7)
    np.testing.assert_array_equal(logits[2], params["bias"])
    if kind == "attention":
        assert weights.shape == (3, 4, 5)
        np.testing.assert_allclose(permuted_weights, weights[:, :, permutation], atol=1e-7)
        np.testing.assert_allclose(weights[:2].sum(-1), 1, atol=1e-7)
        np.testing.assert_array_equal(weights[2], 0)


@pytest.mark.parametrize("kind", ["mean", "attention"])
def test_invalid_tokens_have_no_value_or_gradient_channel(kind):
    tokens, mask = _tokens()
    params = init_readout(jax.random.key(1), 8, 2, kind)
    logits, _ = apply_readout(params, tokens, mask, kind)
    changed = jnp.where(mask[..., None], tokens, jnp.nan)
    actual, _ = apply_readout(params, changed, mask, kind)
    np.testing.assert_array_equal(actual, logits)
    gradient = jax.grad(lambda value: apply_readout(params, value, mask, kind)[0].sum())(tokens)
    np.testing.assert_array_equal(np.asarray(gradient)[~np.asarray(mask)], 0)
    assert np.linalg.norm(np.asarray(gradient)[np.asarray(mask)]) > 0


def test_uniform_query_readout_matches_mean_and_has_same_classifier_capacity():
    tokens, mask = _tokens()
    mean = init_readout(jax.random.key(2), 8, 3, "mean")
    attention = {**mean, "queries": jnp.zeros((4, 8))}
    mean_output, _ = apply_readout(mean, tokens, mask, "mean")
    attention_output, _ = apply_readout(attention, tokens, mask, "attention")
    np.testing.assert_allclose(mean_output, attention_output, atol=3e-7, rtol=3e-7)
    assert sum(x.size for x in attention.values()) - sum(x.size for x in mean.values()) == 4 * 8
    norm = normalized_tokens(tokens, mask)
    np.testing.assert_allclose(np.asarray(norm)[np.asarray(mask)].mean(-1), 0, atol=2e-7)
    np.testing.assert_allclose(normalized_tokens(tokens + 10, mask), norm, atol=3e-6)


def test_auc_ties_are_half_credit_and_unsupported_targets_are_explicit():
    labels = np.asarray([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]], np.uint8)
    logits = np.asarray([[0, -1, 0], [0, 1, 0], [0, -1, 0], [0, 1, 0]], np.float32)
    result = binary_metrics(logits, labels, ["maccs_001", "fg_amine", "elem_N"],
                            train_positive=np.asarray([5, 5, 1]), train_negative=np.asarray([5, 5, 9]), min_count=2)
    assert result["macro_auroc"] == .75
    assert result["maccs_auroc"] == .5
    assert result["fg_auroc"] == 1
    assert result["elements_auroc"] is None
    assert result["supported_targets"] == 2
    assert result["per_target"][2]["auroc"] is None


def _training_example():
    rng = np.random.default_rng(9)
    labels = np.tile(np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], np.uint8), (12, 1))
    tokens = rng.normal(0, .15, (48, 3, 8)).astype(np.float32)
    # Two chemistry signals are deliberately contained in valid tokens. Garbage
    # padding is independent and cannot improve the fitted classifier.
    tokens[:, :, 0] += (2 * labels[:, 0].astype(float) - 1)[:, None]
    tokens[:, :, 1] += (2 * labels[:, 1].astype(float) - 1)[:, None]
    mask = np.ones((48, 3), bool)
    mask[:, 2] = False
    folds = np.repeat(np.asarray(["train", "val", "test"]), [24, 12, 12])
    molecules = np.asarray([f"mol_{i}" for i in range(48)])
    return tokens, mask, labels, folds, molecules


@pytest.mark.parametrize("kind", ["mean", "attention"])
def test_training_selects_without_test_labels_and_artifact_reuses_exact_decoder(tmp_path, kind):
    tokens, mask, labels, folds, molecules = _training_example()
    options = dict(kind=kind, seed=4, learning_rates=(.03,), epochs=8, patience=3,
                   batch_size=12, min_count=2)
    result = train_readout(tokens, mask, labels, folds, molecules, ["maccs_001", "fg_amine"], **options)
    altered = labels.copy()
    altered[folds == "test"] = 1 - altered[folds == "test"]
    repeated = train_readout(tokens, mask, altered, folds, molecules, ["maccs_001", "fg_amine"], **options)
    assert result["metadata"]["selection"] == repeated["metadata"]["selection"]
    assert result["metadata"]["trials"] == repeated["metadata"]["trials"]
    for key in result["params"]:
        np.testing.assert_array_equal(result["params"][key], repeated["params"][key])
    assert result["metadata"]["metrics"]["test"]["macro_auroc"] > .8
    assert not result["metadata"]["training"]["test_used_for_selection"]
    save_readout(result, tmp_path, contract={"token_scope": "physical_only", "input_sha256": "test-fixture"})
    params, metadata = load_readout(tmp_path)
    rows = result["predictions"]["test"]["indices"]
    actual, _ = jax.jit(lambda p, x, m: apply_readout(p, x, m, kind))(params, tokens[rows], mask[rows])
    np.testing.assert_allclose(actual, result["predictions"]["test"]["logits"], atol=1e-6)
    assert metadata["contract"]["token_scope"] == "physical_only"
    assert set(np.load(tmp_path / "test_predictions.npz").files) == {
        "logits", "probabilities", "labels", "indices", "molecule_ids",
    }
    assert json.loads((tmp_path / "trainedprobe.json").read_text())["complete"]


def test_cross_fold_molecules_and_shape_mismatch_are_rejected_before_training():
    tokens, mask, labels, folds, molecules = _training_example()
    molecules[-1] = molecules[0]
    with pytest.raises(ValueError, match="Molecular identity crosses"):
        train_readout(tokens, mask, labels, folds, molecules, ["maccs_001", "fg_amine"], kind="mean")
    params = init_readout(jax.random.key(0), 8)
    with pytest.raises(ValueError, match="boolean mask"):
        apply_readout(params, jnp.asarray(tokens), jnp.asarray(mask[:, :-1]), "mean")
