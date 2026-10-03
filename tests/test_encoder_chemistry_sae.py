from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np

from scripts.encoder_chemistry_sae import (
    decode_sae, dense_activations, dictionary_redundancy, edit_feature, encode_sae,
    feature_amplitude, fit_normalization, init_sae, load_sae, matched_direction_control,
    reconstruct_sae, reconstruction_metrics, sae_loss, sample_physical_tokens, save_sae, train_sae,
)


def _known_dictionary():
    return {"center": jnp.asarray([10., -3., 5., 2.]), "scale": jnp.asarray(2.),
            "encoder": jnp.eye(4), "decoder": jnp.eye(4), "pre_bias": jnp.zeros(4),
            "latent_bias": jnp.zeros(4)}


def test_physical_sampling_is_equal_molecule_weighted_and_mask_safe():
    valid = np.asarray([[True, False, False, False, False], [True, False, True, True, True]])
    sampled = sample_physical_tokens(valid, maximum=3, seed=8)
    assert len(sampled["rows"]) == 4
    assert np.all(valid[sampled["rows"], sampled["slots"]])
    np.testing.assert_allclose(np.bincount(sampled["rows"], weights=sampled["weights"]), [1, 1])
    assert len(set(zip(sampled["rows"], sampled["slots"]))) == 4
    repeated = sample_physical_tokens(valid, maximum=3, seed=8)
    for key in sampled:
        np.testing.assert_array_equal(sampled[key], repeated[key])


def test_topk_relu_sparsity_and_original_unit_reconstruction():
    params = _known_dictionary()
    normalized = jnp.asarray([[3., 2., -1., 1.], [-3., -2., -1., -4.]])
    x = normalized * params["scale"] + params["center"]
    code = encode_sae(params, x, 2)
    dense = dense_activations(code, 4)
    np.testing.assert_array_equal(dense, [[3, 2, 0, 0], [0, 0, 0, 0]])
    assert np.all((np.asarray(code.values) > 0).sum(-1) <= 2)
    np.testing.assert_array_equal(decode_sae(params, code), dense * 2 + params["center"])


def test_feature_ablation_preserves_error_and_matched_control_norm():
    params = _known_dictionary()
    x = jnp.asarray([[16., 1., 3., 4.]])
    reconstructed = reconstruct_sae(params, x, 2)
    edited = edit_feature(params, x, 2, jnp.asarray(0), dose=1.)
    np.testing.assert_array_equal(edited, x - jnp.asarray([[6., 0., 0., 0.]]))
    code = encode_sae(params, x, 2)
    zeroed = code._replace(values=jnp.where(code.indices == 0, 0, code.values))
    np.testing.assert_array_equal(edited - decode_sae(params, zeroed), x - reconstructed)
    np.testing.assert_array_equal(edit_feature(params, x, 2, jnp.asarray(0), dose=0.), x)
    np.testing.assert_array_equal(edit_feature(params, x, 2, jnp.asarray(3)), x)
    control = matched_direction_control(params, x, 2, jnp.asarray(0), jnp.asarray([0., 3., 4., 0.]))
    np.testing.assert_allclose(np.linalg.norm(np.asarray(control - x), axis=-1),
                               np.linalg.norm(np.asarray(edited - x), axis=-1), atol=1e-6)
    np.testing.assert_array_equal(feature_amplitude(params, x, 2, jnp.asarray(0)), [3.])


def test_auxk_has_zero_effect_without_dead_features_and_trains_dead_direction():
    params = _known_dictionary()
    x = params["center"] + params["scale"] * jnp.asarray([[3., 2., 1., .5]])
    weights = jnp.ones(1)
    none_dead = jnp.zeros(4, jnp.bool_)
    loss, stats = sae_loss(params, x, weights, none_dead, k=1, aux_k=2)
    assert stats["aux"] == 0
    assert loss == stats["mse"]
    dead = jnp.asarray([False, True, True, False])
    with_aux = jax.grad(lambda p: sae_loss(p, x, weights, dead, k=1, aux_k=2)[0])(params)
    no_aux = jax.grad(lambda p: sae_loss(p, x, weights, dead, k=1, aux_k=2, aux_weight=0)[0])(params)
    # Dead dictionary directions receive a gradient toward the remaining error,
    # although these features were absent from the main TopK bottleneck.
    assert np.linalg.norm(np.asarray(with_aux["decoder"][1:3] - no_aux["decoder"][1:3])) > 0


def test_normalization_is_fixed_weighted_train_statistics_and_decoder_unit_norm():
    x = np.asarray([[0., 1.], [2., 3.], [10., 9.]], np.float32)
    weights = np.asarray([.5, .5, 1.], np.float32)
    center, scale = fit_normalization(x, weights)
    np.testing.assert_allclose(center, [5.5, 5.5])
    np.testing.assert_allclose(np.average(((x - center) / scale) ** 2, axis=0, weights=weights).mean(), 1, atol=2e-7)
    params = init_sae(jax.random.key(4), center, scale, 8)
    np.testing.assert_allclose(np.linalg.norm(params["decoder"], axis=1), 1, atol=1e-6)
    np.testing.assert_array_equal(params["encoder"], params["decoder"])


def test_small_sparse_signal_fit_improves_reconstruction_and_round_trips(tmp_path):
    rng = np.random.default_rng(11)
    atoms = np.eye(6, dtype=np.float32)
    amplitudes = rng.uniform(.5, 3., (240, 2)).astype(np.float32)
    indices = np.stack([rng.choice(6, 2, replace=False) for _ in range(240)])
    x = (atoms[indices] * amplitudes[..., None]).sum(1) + rng.normal(0, .015, (240, 6))
    train, validation = x[:192].astype(np.float32), x[192:].astype(np.float32)
    result = train_sae(train, validation, train_weights=np.ones(192), validation_weights=np.ones(48),
                       features=24, k=2, learning_rate=.01, epochs=14, patience=4, batch_size=32,
                       dead_after_tokens=192, aux_k=4)
    initial = reconstruction_metrics(result["initial_params"], validation, np.ones(48), k=2, batch_size=32)
    trained = reconstruction_metrics(result["params"], validation, np.ones(48), k=2, batch_size=32)
    assert trained["normalized_mse"] < initial["normalized_mse"] * .75
    assert trained["mean_l0"] <= 2
    np.testing.assert_allclose(np.linalg.norm(result["params"]["decoder"], axis=1), 1, atol=1e-6)
    save_sae(result, tmp_path, contract={"depth": 19, "token_scope": "physical_only"})
    restored, metadata = load_sae(tmp_path)
    np.testing.assert_allclose(reconstruct_sae(restored, jnp.asarray(validation), 2),
                               reconstruct_sae(result["params"], jnp.asarray(validation), 2), atol=1e-6)
    assert metadata["labels_used"] is False
    assert "labels" not in inspect.signature(train_sae).parameters
    assert "test_tokens" not in inspect.signature(train_sae).parameters
    redundancy = dictionary_redundancy(restored, block=8)
    assert len(redundancy["nearest_feature"]) == 24
    assert np.all(redundancy["nearest_feature"] != np.arange(24))
