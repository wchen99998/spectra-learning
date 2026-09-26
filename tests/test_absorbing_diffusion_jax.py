import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
nnx = pytest.importorskip("flax.nnx")

from spectra_learning.config import load_config
from spectra_learning.models.factory_jax import build_model_from_config
from spectra_learning.training.pretrain_jax import (
    init_pure_optax_train_state,
    jax_config_checkpoint_contract,
    make_pure_accumulated_train_step,
    trainable_param_filter,
)


REFERENCE_CONFIG = "configs/1p8b_mae_teacher_jepa.py"
DIFFUSION_CONFIG = "configs/1p8b_absorbing_diffusion.py"


def _config(**overrides):
    return load_config(
        REFERENCE_CONFIG,
        {
            "training_mode": "mae",
            "mae_objective": "absorbing_diffusion",
            "mae_loss_weight": 1.0,
            "masked_token_loss_weight": 0.0,
            "model_dim": 32,
            "predictor_dim": 32,
            "encoder_num_layers": 1,
            "encoder_num_heads": 2,
            "masked_latent_predictor_num_layers": 1,
            "masked_latent_predictor_num_heads": 2,
            "feature_mlp_hidden_dim": 32,
            "encoder_fourier_mlp_hidden_dim": 32,
            "encoder_fourier_num_freqs": 4,
            "num_peaks": 4,
            "jepa_mae_mz_bin_size": 250.0,
            "mae_gap_bucket_edges_da": (),
            "encoder_metadata_schema": "",
            "encoder_metadata_conditioning": "none",
            "activation_checkpoint_mode": "none",
            "autocast_dtype": "none",
            **overrides,
        },
    )


def _batch():
    return {
        "peak_mz": jnp.array([[0.1, 0.3, 0.55, 0.85], [0.8, 0.82, 0.9, 0.95]]),
        "peak_intensity": jnp.array([[0.2, 0.4, 0.6, 0.8], [0.9, 0.7, 0.3, 0.1]]),
        "peak_valid_mask": jnp.ones((2, 4), dtype=bool),
        "context_mask": jnp.array([[False, True, True, True], [True, False, False, False]]),
        "target_masks": jnp.array([[[True, False, False, False]], [[False, True, True, True]]]),
    }


@pytest.mark.parametrize("block_type", ["fastmixer-dense", "dense"])
@pytest.mark.parametrize("loss_only", [False, True])
def test_diffusion_averages_each_spectrum_before_averaging_batch(block_type, loss_only):
    model = build_model_from_config(_config(pairmixer_block_type=block_type))
    # Unequal target counts and unequal target difficulties distinguish the
    # diffusion expectation from the old average over every masked batch token.
    logits = np.array([-2.0, -0.5, 0.8, 2.0], dtype=np.float32)
    model.jepa_mae_mz_head.weight[...] = jnp.zeros_like(model.jepa_mae_mz_head.weight[...])
    model.jepa_mae_mz_head.bias[...] = jnp.asarray(logits)
    batch = _batch()
    log_partition = np.log(np.exp(logits).sum())
    individual_losses = log_partition - logits[[0, 3]]
    expected = individual_losses.mean()
    token_weighted = np.average(individual_losses, weights=[1, 3])
    result = model(batch, loss_only=loss_only)

    assert float(result["loss"]) == pytest.approx(float(expected), rel=1e-6)
    assert not np.isclose(float(result["loss"]), token_weighted)
    row_losses = [
        model(jax.tree.map(lambda value: value[row : row + 1], batch), loss_only=loss_only)["loss"]
        for row in range(2)
    ]
    np.testing.assert_allclose(row_losses, individual_losses, rtol=1e-6)
    assert float(result["loss"]) == pytest.approx(float(jnp.mean(jnp.stack(row_losses))), rel=1e-6)


def _compact_predictions(model, batch):
    encoded, positions, visible = model.encoder.forward_compact(
        batch["peak_mz"],
        batch["peak_intensity"],
        valid_mask=batch["peak_valid_mask"],
        visible_mask=batch["context_mask"],
        max_visible_tokens=model.pairmixer_fast_encoder_max_visible_tokens,
    )
    prediction, _, target_mask, _ = model._predict_augmented_target_outputs_fastmixer_compact(
        encoded,
        positions,
        visible,
        batch["context_mask"],
        batch["target_masks"],
        batch["peak_mz"],
    )
    return prediction, target_mask


def test_masked_peak_values_cannot_change_predictions():
    model = build_model_from_config(_config())
    batch = _batch()
    modified = {
        **batch,
        "peak_mz": jnp.where(batch["context_mask"], batch["peak_mz"], 0.45),
        "peak_intensity": jnp.where(batch["context_mask"], batch["peak_intensity"], 0.99),
    }
    before, target_mask = _compact_predictions(model, batch)
    after, changed_target_mask = _compact_predictions(model, modified)
    np.testing.assert_array_equal(changed_target_mask, target_mask)
    np.testing.assert_array_equal(np.asarray(before)[target_mask], np.asarray(after)[target_mask])


def test_clean_encoder_uses_categorical_mz_for_both_inference_entrypoints():
    model = build_model_from_config(_config())
    batch = _batch()
    # Distinct continuous spectra represent the same categorical m/z state.
    modified = {**batch, "peak_mz": batch["peak_mz"] + 0.01}
    np.testing.assert_array_equal(model.encode(batch), model.encode(modified))
    encoder_kwargs = {
        "valid_mask": batch["peak_valid_mask"],
        "visible_mask": batch["peak_valid_mask"],
    }
    before = model.encoder(batch["peak_mz"], batch["peak_intensity"], **encoder_kwargs)
    after = model.encoder(modified["peak_mz"], batch["peak_intensity"], **encoder_kwargs)
    np.testing.assert_array_equal(before, after)
    different = {**batch, "peak_mz": batch["peak_mz"].at[0, 0].set(0.35)}
    assert not np.allclose(model.encode(batch), model.encode(different))


def test_fully_masked_spectra_have_finite_loss_and_gradients():
    model = build_model_from_config(_config())
    batch = _batch()
    valid = batch["peak_valid_mask"].at[1, 3].set(False)
    batch = {
        **batch,
        "peak_valid_mask": valid,
        "context_mask": jnp.zeros_like(valid),
        "target_masks": valid[:, None],
    }
    loss, gradients = nnx.value_and_grad(lambda value: value(batch, loss_only=True)["loss"])(model)
    assert np.isfinite(float(loss))
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(gradients))
    assert any(np.any(np.asarray(leaf) != 0) for leaf in jax.tree.leaves(gradients["encoder"]))
    assert any(np.any(np.asarray(leaf) != 0) for leaf in jax.tree.leaves(gradients["masked_latent_predictor"]))


def test_jitted_accumulated_train_step_updates_encoder_and_predictor():
    model = build_model_from_config(_config(mae_mz_head="readout", jepa_mae_mz_bin_size=1000.0 / 32))
    graphdef, parameters, static_state, optimizer_state, optimizer = init_pure_optax_train_state(
        {"optimizer": "adam", "learning_rate": 1e-3, "weight_decay": 0.0}, model
    )
    before = {
        name: [np.asarray(leaf).copy() for leaf in jax.tree.leaves(parameters[name])]
        for name in ("encoder", "masked_latent_predictor", "masked_latent_readout")
    }
    train_step = make_pure_accumulated_train_step(graphdef, optimizer, sharded=False)
    batch = _batch()
    # Two one-example microbatches have different mask counts. Accumulation
    # must preserve the equally weighted spectrum objective.
    accumulated = jax.tree.map(lambda value: value[:, None], batch)
    # Compare compiled forwards: fusion changes rounding in the existing
    # high-frequency Fourier features relative to eager execution.
    batch_loss = jax.jit(lambda values, inputs: nnx.merge(graphdef, values, static_state)(inputs)["loss"])
    expected_loss = float(batch_loss(parameters, batch))
    parameters, _, metrics = train_step(parameters, static_state, optimizer_state, accumulated)
    assert float(metrics["loss"]) == pytest.approx(expected_loss, rel=1e-5)
    for name, original in before.items():
        updated = jax.tree.leaves(parameters[name])
        assert all(np.isfinite(np.asarray(leaf)).all() for leaf in updated)
        assert any(not np.array_equal(old, np.asarray(new)) for old, new in zip(original, updated)), name


def test_readout_head_uses_existing_predictor_outputs_as_categorical_logits():
    model = build_model_from_config(_config(mae_mz_head="readout", jepa_mae_mz_bin_size=1000.0 / 32))
    batch = _batch()
    prediction, target_mask = _compact_predictions(model, batch)
    assert prediction.shape[-1] == 32
    assert not jax.tree.leaves(nnx.state(model.jepa_mae_mz_head, nnx.Param))
    # Recover selected targets in their original order without using model
    # loss helpers, then calculate the categorical likelihood independently.
    mask = np.asarray(batch["target_masks"][:, 0])
    targets = np.floor(np.asarray(batch["peak_mz"]) * 32).astype(np.int32)
    log_probabilities = np.asarray(jax.nn.log_softmax(prediction, axis=-1))
    expected_rows = [
        -log_probabilities[row, np.arange(mask[row].sum()), targets[row, mask[row]]].mean()
        for row in range(2)
    ]
    assert np.asarray(target_mask).sum(axis=-1).tolist() == [1, 3]
    for loss_only in (False, True):
        result = model(batch, loss_only=loss_only)
        assert float(result["loss"]) == pytest.approx(float(np.mean(expected_rows)), rel=1e-6)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("training_mode", "jepa"),
        ("masked_token_input_mode", "mz_sentinel"),
        ("jepa_num_target_blocks", 2),
        ("encoder_use_cls_token", False),
        ("mae_mz_objective", "sparse_ordinal_ce"),
        ("mae_mz_multiscale_weight", 1.0),
        ("mae_loss_weight", 0.0),
        ("mae_intensity_loss_weight", 1.0),
        ("distogram_loss_weight", 1.0),
        ("use_ema_teacher", True),
    ],
)
def test_diffusion_rejects_incompatible_objective_settings(key, value):
    with pytest.raises(ValueError, match=f"absorbing_diffusion requires {key}"):
        build_model_from_config(_config(**{key: value}))


def test_readout_head_requires_one_category_per_existing_output_dimension():
    with pytest.raises(ValueError, match="one m/z bin per readout dimension"):
        build_model_from_config(_config(mae_mz_head="readout"))


def test_full_size_diffusion_preserves_reference_backbone_parameter_shapes():
    reference = nnx.eval_shape(lambda: build_model_from_config(load_config(REFERENCE_CONFIG)))
    diffusion = nnx.eval_shape(lambda: build_model_from_config(load_config(DIFFUSION_CONFIG)))
    reference_parameters = nnx.state(reference, trainable_param_filter)
    diffusion_parameters = nnx.state(diffusion, trainable_param_filter)
    reference_shapes = {path: leaf.shape for path, leaf in nnx.to_flat_state(reference_parameters)}
    diffusion_shapes = {path: leaf.shape for path, leaf in nnx.to_flat_state(diffusion_parameters)}
    assert diffusion_shapes == reference_shapes
    reference_count = sum(np.prod(shape) for shape in reference_shapes.values())
    diffusion_count = sum(np.prod(shape) for shape in diffusion_shapes.values())
    assert diffusion_count == reference_count == 1_777_491_456
    assert diffusion.teacher_encoder is None
    assert diffusion.jepa_mae_intensity_head is None


def test_checkpoint_contract_preserves_legacy_default_and_records_diffusion():
    config = load_config(REFERENCE_CONFIG)
    explicit_default = jax_config_checkpoint_contract(config)
    del config["mae_objective"]
    del config["mae_mz_head"]
    assert jax_config_checkpoint_contract(config) == explicit_default
    config.mae_objective = "absorbing_diffusion"
    config.mae_mz_head = "readout"
    contract = jax_config_checkpoint_contract(config)["config"]
    assert contract["mae_objective"] == "absorbing_diffusion"
    assert contract["mae_mz_head"] == "readout"
