"""The pair-branch distogram: encoder pair track in, missing distances out."""

import numpy as np
import pytest
from spectra_learning.config import load_config

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from flax import nnx  # noqa: E402

from spectra_learning.models.factory_jax import (  # noqa: E402
    build_model_from_config,
)
from spectra_learning.models.model_jax import PeakSetJEPAJax  # noqa: E402
from spectra_learning.models.pairmixer_jax import (  # noqa: E402
    PairOnlyBlock,
)
from spectra_learning.models.settings import (  # noqa: E402
    PAIR_DISTOGRAM_TRAINING_MODE,
    PeakSetJEPASettings,
    encoder_uses_pair_path,
)

CONFIG = "configs/1p8b_fastmixer_pair_distogram_multiscale_n64_v6e128_ga4.py"
PARENT = (
    "configs/1p8b_singlemixer_dense_adam_metadata_adaln_"
    "multiscale_n128_b6144_666k_d2048.py"
)
WIDTHS = (2, 4, 8, 16, 32, 64)


def _model(**overrides) -> PeakSetJEPAJax:
    kwargs = dict(
        training_mode=PAIR_DISTOGRAM_TRAINING_MODE,
        model_dim=16,
        predictor_dim=16,
        encoder_num_layers=2,
        encoder_num_heads=2,
        attention_mlp_multiple=2.0,
        feature_mlp_hidden_dim=16,
        encoder_fourier_mlp_hidden_dim=16,
        encoder_fourier_num_freqs=2,
        pairmixer_fourier_num_freqs=2,
        pairmixer_pair_dim=8,
        pairmixer_pair_feature_hidden_dim=8,
        pairmixer_block_type="fastmixer",
        masked_latent_predictor_num_layers=2,
        masked_latent_predictor_num_heads=2,
        masked_token_input_mode="mz_sentinel",
        num_peaks=8,
        encoder_use_cls_token=False,
        jepa_num_target_blocks=1,
        jepa_mae_mz_bin_size=0.25,
        distogram_mz_max=2.0,
        mae_loss_weight=0.0,
        distogram_loss_weight=1.0,
        distogram_multiscale_widths=WIDTHS,
        distogram_multiscale_weight=1.0,
        target_projector_dim=-1,
        rngs=nnx.Rngs(params=0, dropout=0),
    )
    kwargs.update(overrides)
    return PeakSetJEPAJax(**kwargs)


def _batch(seed: int = 0, batch: int = 3, peaks: int = 8) -> dict:
    rng = np.random.default_rng(seed)
    context = rng.random((batch, peaks)) < 0.6
    target = (~context) & (rng.random((batch, peaks)) < 0.9)
    target[:, -1] = ~context[:, -1]
    return {
        "peak_mz": jnp.asarray(
            np.sort(rng.random((batch, peaks)) * 2.0, axis=-1), jnp.float32
        ),
        "peak_intensity": jnp.asarray(rng.random((batch, peaks)), jnp.float32),
        "peak_valid_mask": jnp.ones((batch, peaks), bool),
        "context_mask": jnp.asarray(context),
        "target_masks": jnp.asarray(target[:, None, :]),
        "precursor_mz": jnp.asarray(rng.random(batch) * 2.0, jnp.float32),
    }


def test_the_pair_path_is_not_optional_in_this_mode():
    # The objective reads the encoder's pair track, so `encoder_use_pair_path`
    # cannot turn it off the way it can for a distogram on predictor latents.
    settings = PeakSetJEPASettings(
        training_mode=PAIR_DISTOGRAM_TRAINING_MODE,
        encoder_use_pair_path=False,
        distogram_loss_weight=0.0,
    )
    assert encoder_uses_pair_path(settings) is True


def test_encoder_and_predictor_have_no_single_stream():
    model = _model()

    assert all(isinstance(block, PairOnlyBlock) for block in model.encoder.blocks)
    assert not hasattr(model.encoder, "embedder")
    assert model.encoder.pair_embedder.single_pair_proj is None
    assert all(
        isinstance(block, PairOnlyBlock)
        for block in model.masked_latent_predictor
    )
    # The pair-only block drops the single track outright.
    block = model.masked_latent_predictor[0]
    assert not hasattr(block, "attention")
    assert not hasattr(block, "single_transition")
    assert not hasattr(block, "single_to_pair")
    assert model.pair_mask_token is not None
    # Nothing routes single-token latents anywhere, so their mask token and
    # readout are dead weight and are not built.
    assert model.latent_mask_token is None
    assert model.masked_latent_readout is None


def test_no_mae_value_heads_when_the_distogram_is_the_only_loss():
    model = _model()

    assert model.uses_mae_value_heads is False
    assert model.jepa_mae_mz_head is None
    assert model.jepa_mae_intensity_head is None
    assert model.distogram_head is not None
    assert model.distogram_head.weight.shape == (
        model.distogram_num_bins,
        model.encoder_pair_dim,
    )


def test_forward_emits_one_distogram_term_per_scale_and_nothing_else():
    model = _model()
    metrics = model(_batch())

    assert set(metrics) >= {
        "loss",
        "distogram_loss",
        "distogram_multiscale_loss",
        "distogram_objective",
        "distogram_term",
    }
    for width in WIDTHS:
        assert f"distogram_ce_{width}bins" in metrics
    assert not any(key.startswith("mae_") for key in metrics)
    np.testing.assert_allclose(
        np.asarray(metrics["loss"]), np.asarray(metrics["distogram_term"]), rtol=1e-6
    )
    assert np.isfinite(np.asarray(metrics["loss"]))
    # Coarser marginals can only be cheaper than the exact one.
    exact = float(metrics["distogram_loss"])
    previous = exact
    for width in WIDTHS:
        current = float(metrics[f"distogram_ce_{width}bins"])
        assert current <= previous + 1e-5
        previous = current


def test_the_encoder_pair_representation_cannot_see_the_masked_distances():
    """The sentinel is what stops the head reading off its own input.

    `PairFeatureEmbedder` is fed |m/z_i - m/z_j| directly, so if the masked
    peaks reached the encoder with their real m/z the objective would be a
    lookup. Move them and the encoder's pair tensor must not move at all.
    """
    model = _model()
    batch = _batch()
    target_union = jnp.any(batch["target_masks"], axis=1)

    def encoder_pair(peak_mz):
        moved = dict(batch, peak_mz=peak_mz)
        encoder_mz, encoder_intensity, visible = model._context_encoder_inputs(
            moved["peak_mz"],
            moved["peak_intensity"],
            moved["context_mask"] & moved["peak_valid_mask"],
            moved["target_masks"] & moved["peak_valid_mask"][:, None, :],
        )
        pair, _idx, _mask = model.encoder(
            encoder_mz,
            encoder_intensity,
            valid_mask=moved["peak_valid_mask"],
            visible_mask=visible,
            max_visible_tokens=model.pairmixer_fast_encoder_max_visible_tokens,
            precursor_mz=moved["precursor_mz"],
        )
        return pair

    shifted = jnp.where(target_union, batch["peak_mz"] + 0.37, batch["peak_mz"])
    assert int(jnp.sum(shifted != batch["peak_mz"])) > 0

    before = np.asarray(encoder_pair(batch["peak_mz"]))
    after = np.asarray(encoder_pair(shifted))
    np.testing.assert_array_equal(before, after)

    # ... while the labels the head is scored against do move.
    assert float(model(dict(batch, peak_mz=shifted))["loss"]) != float(
        model(batch)["loss"]
    )


def _nudge_zero_initialised_branches(model: PeakSetJEPAJax) -> None:
    """Move the zero-init residual outputs off zero, as one step would."""
    flat = nnx.to_flat_state(nnx.state(model, nnx.Param))
    key = jax.random.key(0)
    updated = []
    for index, (path, leaf) in enumerate(flat):
        name = "/".join(str(part) for part in path)
        if name.endswith(("p_out/weight", "fc3/weight", "o/weight")):
            leaf = leaf.replace(
                jax.random.normal(
                    jax.random.fold_in(key, index), leaf.shape, jnp.float32
                )
                * 0.02
            )
        updated.append((path, leaf))
    nnx.update(model, nnx.from_flat_state(updated))


def _gradients(model: PeakSetJEPAJax, batch: dict) -> dict[str, np.ndarray]:
    grads = nnx.grad(lambda m: m(batch, loss_only=True)["loss"])(model)
    return {
        "/".join(str(part) for part in path): np.abs(
            np.asarray(leaf.get_value())
        ).max()
        for path, leaf in nnx.to_flat_state(grads)
    }


def _largest(grads: dict[str, np.ndarray], prefix: str) -> float:
    selected = [value for key, value in grads.items() if key.startswith(prefix)]
    assert selected, f"no parameters under {prefix}"
    return float(max(selected))


def test_the_predictor_is_what_composes_the_missing_distances():
    model = _model()
    grads = _gradients(model, _batch())

    assert _largest(grads, "masked_latent_predictor") > 0.0
    assert _largest(grads, "distogram_head") > 0.0
    # The pair mask token is the only thing standing in for a distance the
    # encoder never saw; if it is not trained the seeded entries are frozen.
    assert _largest(grads, "pair_mask_token") > 0.0

    # The sentinel is sufficient to hide labels. Keep target pair features so
    # encoder gradients and target-specific intensity information survive.
    assert _largest(grads, "encoder") > 0.0


def test_the_distogram_trains_the_encoder_through_the_pair_track():
    model = _model()
    _nudge_zero_initialised_branches(model)
    grads = _gradients(model, _batch())

    assert _largest(grads, "encoder/blocks/0/tri_mul") > 0.0
    assert _largest(grads, "encoder/blocks/0/pair_transition") > 0.0
    assert _largest(grads, "encoder/pair_embedder") > 0.0


def test_only_target_rows_are_scored_and_all_views_are_included():
    model = _model(jepa_num_target_blocks=2, predictor_target_max_tokens=1)
    batch = _batch(batch=1)
    batch["context_mask"] = jnp.array([[True, True, True, False, False, False, False, False]])
    batch["target_masks"] = jnp.array([[
        [False, False, False, True, False, False, False, False],
        [False, False, False, False, True, False, False, False],
    ]])
    # Uniform logits make the exact CE independent of targets, so record what
    # the scorer receives to check the actual supervised rows and weights.
    captured = {}
    def score(features, targets):
        captured["targets"] = targets
        return {1: targets.astype(jnp.float32)}
    model._distogram_score = score
    loss = float(model(batch)["loss"])
    mz = np.asarray(batch["peak_mz"])[0]
    expected = np.clip(np.floor(np.abs(mz[[3, 4], None] - mz[None, :5]) * 2 / 0.25), 0, 7)
    assert captured["targets"].shape[1] == 2
    expected[0, 3] = expected[1, 4] = 0
    assert loss == pytest.approx(expected.sum() / 8)


def test_excluded_peaks_cannot_change_the_prediction():
    model = _model()
    batch = _batch(batch=1)
    batch["peak_valid_mask"] = batch["peak_valid_mask"].at[:, -1].set(False)
    moved = dict(batch,
        peak_mz=batch["peak_mz"].at[:, -1].set(100),
        peak_intensity=batch["peak_intensity"].at[:, -1].set(100),
    )
    assert float(model(batch)["loss"]) == float(model(moved)["loss"])


@pytest.mark.parametrize("empty", ["targets", "spectrum"])
def test_empty_supervision_has_zero_finite_loss_and_gradients(empty):
    model = _model()
    batch = _batch(batch=1)
    if empty == "targets":
        batch["target_masks"] = jnp.zeros_like(batch["target_masks"])
    else:
        batch["peak_valid_mask"] = jnp.zeros_like(batch["peak_valid_mask"])
    assert float(model(batch)["loss"]) == 0.0
    assert all(np.isfinite(value) and value == 0 for value in _gradients(model, batch).values())


@pytest.mark.parametrize("overrides,match", [
    ({"masked_token_input_mode": "latent_token"}, "mz_sentinel"),
    ({"distogram_loss_weight": 0}, "distogram_loss_weight"),
    ({"mae_loss_weight": 1}, "other loss weights"),
])
def test_incompatible_pair_training_settings_are_rejected(overrides, match):
    with pytest.raises(ValueError, match=match):
        _model(**overrides)


@pytest.mark.parametrize("mode", ["full", "selective"])
def test_pair_rematerialization_preserves_loss_and_gradients(mode):
    shape = dict(encoder_num_layers=5, masked_latent_predictor_num_layers=3)
    stored = _model(**shape)
    _nudge_zero_initialised_branches(stored)
    remat = _model(**shape, activation_checkpoint_mode=mode,
        activation_checkpoint_modules=("encoder", "predictor", "distogram"))
    nnx.update(remat, nnx.state(stored))
    batch = _batch(batch=1)
    np.testing.assert_allclose(float(stored(batch)["loss"]), float(remat(batch)["loss"]), rtol=1e-6)
    gradient = nnx.grad(lambda model: model(batch, loss_only=True)["loss"])
    for expected, actual in zip(
        jax.tree.leaves(gradient(stored)), jax.tree.leaves(gradient(remat)), strict=True,
    ):
        np.testing.assert_allclose(np.asarray(expected), np.asarray(actual), rtol=1e-5, atol=1e-5)


def test_config_trains_the_pair_branch_at_reference_capacity():
    config = load_config(CONFIG)
    parent = load_config(PARENT)

    assert config.training_mode == PAIR_DISTOGRAM_TRAINING_MODE
    assert config.pairmixer_block_type == "fastmixer"
    assert config.num_peaks == 63
    assert config.encoder_use_cls_token is True
    assert config.max_duration_hours == 72.0
    assert config.masked_token_input_mode == "mz_sentinel"
    assert config.mae_loss_weight == 0.0
    assert config.mae_mz_multiscale_widths == ()
    assert config.distogram_loss_weight == 1.0
    assert config.distogram_multiscale_widths == WIDTHS
    assert config.distogram_multiscale_weight == 1.0
    # 0.5 Da bins over the 1,000 Da range, the same resolution the m/z head
    # used, so the multiscale widths mean 1..32 Da.
    assert config.jepa_mae_mz_bin_size == 0.5

    # Everything the reference run fixes that is not the objective.
    for key in (
        "model_dim",
        "predictor_dim",
        "encoder_num_heads",
        "optimizer",
        "optimizer_state_dtype",
        "learning_rate",
        "training_max_steps",
    ):
        assert config[key] == parent[key], key


def test_capacity_stays_within_a_percent_of_the_reference_run():
    counts = []
    for path in (CONFIG, PARENT):
        config = load_config(path)
        model = nnx.eval_shape(lambda: build_model_from_config(config))
        counts.append(
            sum(
                int(np.prod(leaf.shape))
                for leaf in jax.tree.leaves(nnx.state(model, nnx.Param))
            )
        )
    pair_distogram, reference = counts
    assert abs(pair_distogram / reference - 1.0) < 0.01


def test_the_torch_backend_refuses_the_mode_rather_than_half_building_it():
    torch_model = pytest.importorskip("spectra_learning.models.model")

    with pytest.raises(ValueError, match="training_mode must be one of"):
        torch_model.PeakSetJEPA(training_mode=PAIR_DISTOGRAM_TRAINING_MODE)


def test_128_chip_config_uses_ga4_without_zero1():
    config = load_config(CONFIG)
    assert config.device_backend == "jax"
    assert int(config.jax_mesh_devices) == 128
    assert config.batch_size == 2048
    assert config.gradient_accumulation_steps == 4
    assert config.batch_size // (128 * 4) == 4
    assert config.jax_zero1_optimizer is False
    assert config.jax_shard_optimizer is False
    assert config.optimizer_state_dtype == "bf16"
    assert config.activation_checkpoint_mode == "full"
    assert set(config.activation_checkpoint_modules) == {"encoder", "predictor", "distogram"}
    assert config.gems_hdf5_shuffle_window_blocks == 2048
    from spectra_learning.data.gems.settings import GemsDataConfig
    settings = GemsDataConfig.from_config(config)
    assert settings.gems_hdf5_shuffle_window_blocks == 2048


def test_metadata_adaln_gates_learn_and_condition_the_pair_stream():
    from spectra_learning.models.spectrum_metadata import (
        MASSIVE_V2_ACQUISITION_SCHEMA, MASSIVE_V2_CONDITION_DIM,
    )
    model = _model(encoder_metadata_schema=MASSIVE_V2_ACQUISITION_SCHEMA,
        encoder_metadata_conditioning="adaln_zero", encoder_metadata_condition_dim=8)
    batch = dict(_batch(batch=1), spectrum_metadata=jnp.ones((1, MASSIVE_V2_CONDITION_DIM)))
    grads = _gradients(model, batch)
    assert _largest(grads, "encoder/blocks/0/adaLN_modulation") > 0
    graphdef, params, static = nnx.split(model, nnx.Param, ...)
    def loss(values):
        return nnx.merge(graphdef, values, static)(batch)["loss"]
    gradient = jax.grad(loss)(params)
    nnx.update(model, jax.tree.map(lambda p, g: p - 1e-4 * g, params, gradient))
    assert _largest(_gradients(model, batch), "encoder/metadata_embedder") > 0
    assert float(model(batch)["loss"]) != float(model(dict(batch,
        spectrum_metadata=jnp.zeros_like(batch["spectrum_metadata"])))["loss"])


def test_ga4_train_step_matches_manual_accumulation_with_bf16_adam():
    import optax
    from spectra_learning.training.pretrain_jax import (
        init_pure_optax_train_state, make_pure_accumulated_train_step,
    )
    model = _model(activation_checkpoint_mode="full",
        activation_checkpoint_modules=("encoder", "predictor", "distogram"))
    optimizer_config = {"optimizer": "adam", "learning_rate": 1e-4,
        "weight_decay": 0.0, "b2": 0.95, "optimizer_state_dtype": "bf16"}
    graphdef, params, static, opt_state, optimizer = init_pure_optax_train_state(optimizer_config, model)
    batches = [_batch(seed=i, batch=1) for i in range(4)]
    def loss(values, batch):
        return nnx.merge(graphdef, values, static)(batch, loss_only=True)["loss"]
    gradients = [jax.grad(loss)(params, batch) for batch in batches]
    average = jax.tree.map(lambda *values: sum(values) / 4, *gradients)
    updates, expected_state = optimizer.update(average, opt_state, params)
    expected_params = optax.apply_updates(params, updates)
    expected_loss = sum(float(loss(params, batch)) for batch in batches) / 4
    step = make_pure_accumulated_train_step(graphdef, optimizer, sharded=False)
    stacked = jax.tree.map(lambda *values: jnp.stack(values), *batches)
    actual_params, actual_state, metrics = step(params, static, opt_state, stacked)
    assert float(metrics["loss"]) == pytest.approx(expected_loss, rel=1e-5)
    for actual, expected in zip(jax.tree.leaves(actual_params), jax.tree.leaves(expected_params), strict=True):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=2e-6, rtol=1e-5)
    for actual, expected in zip(jax.tree.leaves(actual_state), jax.tree.leaves(expected_state), strict=True):
        np.testing.assert_allclose(np.asarray(actual, dtype=np.float32), np.asarray(expected, dtype=np.float32), atol=1e-5, rtol=0.02)


def test_cls_occupies_one_pair_position_and_never_receives_distance_supervision():
    model = _model(encoder_use_cls_token=True, predictor_target_max_tokens=1,
        pairmixer_fast_encoder_max_visible_tokens=6)
    batch = _batch(batch=1)
    batch["context_mask"] = jnp.array([[True, True, True, True, False, False, False, False]])
    batch["target_masks"] = jnp.array([[[False, False, False, False, True, False, False, False]]])
    mz, intensity, visible = model._context_encoder_inputs(
        batch["peak_mz"], batch["peak_intensity"], batch["context_mask"], batch["target_masks"])
    pair, idx, mask = model.encoder(mz, intensity, valid_mask=batch["peak_valid_mask"],
        visible_mask=visible, precursor_mz=batch["precursor_mz"])
    assert pair.shape == (1, 6, 6, 8)
    np.testing.assert_array_equal(np.asarray(idx), [[0, 1, 2, 3, 4, 8]])
    assert np.asarray(mask).all()
    assert not hasattr(model.encoder, "cls_token")
    def score(features, targets):
        assert targets.shape == (1, 1, 6)
        # Penalize the CLS column heavily: it must be excluded entirely.
        return {1: jnp.ones_like(targets, dtype=jnp.float32).at[..., -1].set(1000)}
    model._distogram_score = score
    assert float(model(batch)["loss"]) == 1.0


def test_cls_pairs_receive_gradients_through_triangle_updates():
    model = _model(encoder_use_cls_token=True)
    _nudge_zero_initialised_branches(model)
    grads = _gradients(model, _batch(batch=1))
    assert _largest(grads, "encoder/peak_to_cls_pair_token") > 0
    assert _largest(grads, "encoder/cls_to_peak_pair_token") > 0


def test_n64_config_compacts_to_context_targets_and_cls():
    from spectra_learning.models.fastmixer_capacity import pairmixer_fast_stage_capacities
    config = load_config(CONFIG)
    assert config.num_peaks + int(config.encoder_use_cls_token) == 64
    assert pairmixer_fast_stage_capacities(config) == ((55, 55, 16),)
