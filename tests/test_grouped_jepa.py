from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from ml_collections import config_dict

from spectra_learning.config import load_config
from spectra_learning.data.gems.grouped import GroupedBatchSampler
from spectra_learning.data.gems.hdf5 import _grouped_eligible_ranges
from spectra_learning.models.grouped_jepa_jax import (
    GROUP_JEPA_TEACHER_TARGET_AGE_KEY,
    GROUP_JEPA_TEACHER_TARGET_KEY,
    GroupedSpectrumJEPAJax,
    visreg_metrics,
)
from spectra_learning.models.settings import PeakSetJEPASettings
from spectra_learning.training.pretrain_jax import (
    init_pure_optax_train_state,
    make_grouped_jepa_teacher_target_step,
    make_pure_accumulated_train_step,
    trainable_param_filter,
    update_grouped_jepa_ema_state,
)
from spectra_learning.training.routing import resolve_training_route


def _small_settings() -> PeakSetJEPASettings:
    return PeakSetJEPASettings(
        training_mode="grouped_jepa",
        model_dim=24,
        encoder_num_layers=1,
        encoder_num_heads=4,
        feature_mlp_hidden_dim=32,
        encoder_fourier_mlp_hidden_dim=32,
        encoder_fourier_num_freqs=4,
        num_peaks=7,
        predictor_dim=24,
        masked_latent_predictor_num_layers=1,
        masked_latent_predictor_num_heads=4,
        pairmixer_block_type="dense",
        pairmixer_pair_dim=8,
        pairmixer_pair_feature_hidden_dim=8,
        encoder_use_position_embedding=False,
        target_projector_dim=-1,
    )


def _small_batch() -> dict[str, jax.Array]:
    values = jnp.arange(2 * 8 * 7, dtype=jnp.float32).reshape(2, 8, 7)
    return {
        "peak_mz": (values % 100) / 1000,
        "peak_intensity": (values % 7 + 1) / 7,
        "peak_valid_mask": jnp.ones((2, 8, 7), dtype=jnp.bool_),
        "precursor_mz": jnp.full((2, 8), 0.5),
        "collision_energy": jnp.full((2, 8), 0.25),
        "charge": jnp.full((2, 8), 2.0),
    }


def _stacked_small_batch(offset: float = 0.0) -> dict[str, jax.Array]:
    return jax.tree.map(
        lambda value: value[None],
        {
            key: value + offset if value.dtype == jnp.float32 else value
            for key, value in _small_batch().items()
        },
    )


def _optimizer_config() -> config_dict.ConfigDict:
    return config_dict.ConfigDict(
        {
            "optimizer": "adamw",
            "learning_rate": 1e-3,
            "min_learning_rate": 1e-5,
            "warmup_steps": 0,
            "weight_decay": 0.0,
            "b1": 0.9,
            "b2": 0.999,
            "grad_clip_norm": 0.0,
            "optimizer_state_dtype": "fp32",
        }
    )


def test_group_ranges_use_composite_key_and_only_eligible_rows(tmp_path: Path) -> None:
    path = tmp_path / "groups.hdf5"
    with h5py.File(path, "w") as file:
        file["massive_id"] = np.asarray([b"A"] * 6 + [b"B"] * 6 + [b"B"] * 4)
        file["global_group_id"] = np.asarray([1] * 6 + [1] * 6 + [2] * 4)
        file["group_id"] = np.asarray([1] * 6 + [1] * 6 + [-1] * 4)
        eligible = np.ones(16, dtype=np.bool_)
        eligible[[1, 8]] = False
        file["training_eligible"] = eligible
        starts, counts = _grouped_eligible_ranges(
            file,
            logical_start=10,
            minimum_size=5,
            scan_rows=4,
        )

    assert starts.tolist() == [10, 15]
    assert counts.tolist() == [5, 5]


def test_grouped_sampler_samples_without_replacement_and_changes_by_epoch() -> None:
    ranges = ((np.asarray([0, 20]), np.asarray([12, 10])),)
    first = list(
        GroupedBatchSampler(
            ranges,
            groups_per_batch=2,
            spectra_per_group=8,
            shuffle=True,
            seed=7,
            drop_last=True,
            epoch=0,
        )
    )[0]
    repeated = list(
        GroupedBatchSampler(
            ranges,
            groups_per_batch=2,
            spectra_per_group=8,
            shuffle=True,
            seed=7,
            drop_last=True,
            epoch=0,
        )
    )[0]
    second_epoch = list(
        GroupedBatchSampler(
            ranges,
            groups_per_batch=2,
            spectra_per_group=8,
            shuffle=True,
            seed=7,
            drop_last=True,
            epoch=1,
        )
    )[0]

    assert first == repeated
    assert first != second_epoch
    assert len(set(first[:8])) == 8
    assert len(set(first[8:])) == 8
    assert all(index < 12 for index in first[:8]) or all(
        20 <= index < 30 for index in first[:8]
    )


def test_grouped_jepa_uses_mean_teacher_cls_and_per_student_mse() -> None:
    model = GroupedSpectrumJEPAJax(
        _small_settings(),
        teacher_spectra_per_group=3,
        ema_momentum=0.9992,
        rngs=nnx.Rngs(0),
    )
    nnx.update(
        model.teacher_encoder,
        nnx.as_pure(nnx.state(model.encoder, nnx.Param)),
    )
    batch = _small_batch()
    metrics = model(batch)
    teacher = model._encode_cls(
        model.teacher_encoder,
        {key: value[:, :3] for key, value in batch.items()},
    ).astype(jnp.float32)
    student_tokens = model._encode_tokens(
        model.encoder,
        {key: value[:, 3:] for key, value in batch.items()},
    )
    student = student_tokens[..., -1, :].astype(jnp.float32)
    groups, spectra, tokens, model_dim = student_tokens.shape
    prediction = model.predictor(
        student_tokens.reshape(groups * spectra, tokens, model_dim),
        batch["peak_valid_mask"][:, 3:].reshape(groups * spectra, -1),
    ).reshape(groups, spectra, model_dim).astype(jnp.float32)
    expected = jnp.square(
        prediction - teacher.mean(axis=1)[:, None]
    ).mean()

    assert float(metrics["loss"]) == pytest.approx(float(expected), rel=1e-6)
    assert float(metrics["student_prediction_variance"]) == pytest.approx(
        float(jnp.var(prediction.reshape(-1, prediction.shape[-1]), axis=0).mean()),
        rel=1e-6,
    )
    assert float(metrics["teacher_target_age_steps"]) == 0.0


def test_grouped_jepa_stopgrad_teacher_uses_shared_encoder() -> None:
    model = GroupedSpectrumJEPAJax(
        _small_settings(),
        teacher_spectra_per_group=3,
        ema_momentum=None,
        rngs=nnx.Rngs(0),
    )
    assert model.teacher_encoder is None

    batch = _small_batch()

    def loss(peak_mz: jax.Array) -> jax.Array:
        return model({**batch, "peak_mz": peak_mz})["loss"]

    peak_mz_grad = jax.grad(loss)(batch["peak_mz"])
    np.testing.assert_allclose(peak_mz_grad[:, :3], 0.0, atol=1e-7)
    assert float(jnp.max(jnp.abs(peak_mz_grad[:, 3:]))) > 0.0


def test_visreg_has_finite_nonzero_gradient_at_collapse() -> None:
    directions = jnp.eye(4, dtype=jnp.float32)

    def loss(embeddings: jax.Array) -> jax.Array:
        return visreg_metrics(
            embeddings,
            directions,
            center_weight=1.0,
            scale_weight=1.0,
            shape_weight=1.0,
        )["visreg_loss"]

    collapsed = jnp.zeros((2, 8, 4), dtype=jnp.float32)
    gradient = jax.grad(loss)(collapsed)

    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert float(jnp.max(jnp.abs(gradient))) > 0.0


def test_grouped_jepa_combines_invariance_and_visreg_terms() -> None:
    model = GroupedSpectrumJEPAJax(
        _small_settings(),
        teacher_spectra_per_group=3,
        ema_momentum=None,
        invariance_loss_weight=0.1,
        visreg_loss_weight=0.9,
        visreg_num_projections=8,
        rngs=nnx.Rngs(0),
    )
    metrics = model(_small_batch())
    expected = 0.1 * metrics["group_jepa_loss"] + 0.9 * metrics["visreg_loss"]

    assert float(metrics["visreg_loss"]) > 0.0
    assert float(metrics["loss"]) == pytest.approx(float(expected), rel=1e-6)


def test_grouped_jepa_predictor_cross_attends_valid_encoder_tokens() -> None:
    model = GroupedSpectrumJEPAJax(
        _small_settings(),
        teacher_spectra_per_group=3,
        ema_momentum=0.9992,
        rngs=nnx.Rngs(0),
    )
    memory = jnp.arange(2 * 8 * 24, dtype=jnp.float32).reshape(2, 8, 24) / 100
    valid_mask = jnp.ones((2, 7), dtype=jnp.bool_)
    prediction = model.predictor(memory, valid_mask)
    changed_prediction = model.predictor(memory.at[:, 0].add(1), valid_mask)
    masked_prediction = model.predictor(
        memory,
        valid_mask.at[:, 0].set(False),
    )
    masked_changed_prediction = model.predictor(
        memory.at[:, 0].add(1),
        valid_mask.at[:, 0].set(False),
    )

    assert prediction.shape == (2, 24)
    assert float(jnp.max(jnp.abs(changed_prediction - prediction))) > 0
    np.testing.assert_allclose(
        masked_changed_prediction,
        masked_prediction,
        rtol=1e-6,
        atol=1e-6,
    )


def test_grouped_jepa_freezes_teacher_and_updates_post_student_ema() -> None:
    model = GroupedSpectrumJEPAJax(
        _small_settings(),
        teacher_spectra_per_group=3,
        ema_momentum=0.9992,
        rngs=nnx.Rngs(0),
    )
    _graphdef, params, static = nnx.split(model, trainable_param_filter, ...)
    params = nnx.as_pure(params)
    static = nnx.as_pure(static)
    assert set(params) == {"encoder", "predictor"}
    assert "teacher_encoder" in static

    params = jax.tree.map(lambda value: jnp.ones_like(value), params)
    static["teacher_encoder"] = jax.tree.map(
        jnp.zeros_like,
        static["teacher_encoder"],
    )
    updated = update_grouped_jepa_ema_state(params, static, 0.9992)
    teacher_values = jax.tree.leaves(updated["teacher_encoder"])
    trainable_paths = {
        path for path, _value in nnx.to_flat_state(params["encoder"])
    }
    for path, value in nnx.to_flat_state(updated["teacher_encoder"]):
        if path in trainable_paths:
            np.testing.assert_allclose(value, 0.0008, atol=1e-7)


def test_grouped_jepa_optimizes_student_predictor() -> None:
    model = GroupedSpectrumJEPAJax(
        _small_settings(),
        teacher_spectra_per_group=3,
        ema_momentum=0.9992,
        rngs=nnx.Rngs(0),
    )
    graphdef, params, static, opt_state, optimizer = init_pure_optax_train_state(
        _optimizer_config(),
        model,
        total_steps=2,
    )
    train_step = make_pure_accumulated_train_step(
        graphdef,
        optimizer,
        sharded=False,
        group_jepa_ema_momentum=0.9992,
    )
    predictor_before = [
        np.asarray(value).copy() for value in jax.tree.leaves(params["predictor"])
    ]
    updated_params, _static, _opt_state, _metrics = train_step(
        params,
        static,
        opt_state,
        _stacked_small_batch(),
    )

    changes = [
        jnp.max(jnp.abs(after - before))
        for before, after in zip(
            predictor_before,
            jax.tree.leaves(updated_params["predictor"]),
            strict=True,
        )
    ]
    assert max(float(change) for change in changes) > 0


def test_fused_grouped_jepa_ema_matches_separate_post_step_update() -> None:
    def training_state():
        model = GroupedSpectrumJEPAJax(
            _small_settings(),
            teacher_spectra_per_group=3,
            ema_momentum=0.9992,
            rngs=nnx.Rngs(0),
        )
        return init_pure_optax_train_state(
            _optimizer_config(),
            model,
            total_steps=2,
        )

    graphdef, params, static, opt_state, optimizer = training_state()
    separate_step = make_pure_accumulated_train_step(
        graphdef,
        optimizer,
        sharded=False,
    )
    separate_params, separate_opt_state, separate_metrics = separate_step(
        params,
        static,
        opt_state,
        _stacked_small_batch(),
    )
    separate_static = update_grouped_jepa_ema_state(
        separate_params,
        static,
        0.9992,
    )

    graphdef, params, static, opt_state, optimizer = training_state()
    fused_step = make_pure_accumulated_train_step(
        graphdef,
        optimizer,
        sharded=False,
        group_jepa_ema_momentum=0.9992,
    )
    fused_params, fused_static, fused_opt_state, fused_metrics = fused_step(
        params,
        static,
        opt_state,
        _stacked_small_batch(),
    )

    separate_values = jax.tree.leaves(
        (
            separate_params,
            separate_static,
            separate_opt_state,
            separate_metrics,
        )
    )
    fused_values = jax.tree.leaves(
        (fused_params, fused_static, fused_opt_state, fused_metrics)
    )
    assert len(separate_values) == len(fused_values)
    for separate, fused in zip(separate_values, fused_values, strict=True):
        np.testing.assert_allclose(separate, fused, rtol=1e-6, atol=1e-7)


def test_grouped_jepa_lookahead_returns_pre_ema_next_batch_target() -> None:
    model = GroupedSpectrumJEPAJax(
        _small_settings(),
        teacher_spectra_per_group=3,
        ema_momentum=0.9992,
        rngs=nnx.Rngs(0),
    )
    graphdef, params, static, opt_state, optimizer = (
        init_pure_optax_train_state(
            _optimizer_config(),
            model,
            total_steps=2,
        )
    )
    target_step = make_grouped_jepa_teacher_target_step(
        graphdef,
        sharded=False,
    )
    current_target = target_step(params, static, _stacked_small_batch())
    next_batch = _stacked_small_batch(0.01)
    expected_next_target = target_step(params, static, next_batch)
    current_batch = {
        **_stacked_small_batch(),
        GROUP_JEPA_TEACHER_TARGET_KEY: current_target,
        GROUP_JEPA_TEACHER_TARGET_AGE_KEY: jnp.ones(
            current_target.shape[:2],
            dtype=jnp.float32,
        ),
    }
    lookahead_step = make_pure_accumulated_train_step(
        graphdef,
        optimizer,
        sharded=False,
        group_jepa_ema_momentum=0.9992,
        group_jepa_lookahead=True,
    )
    _params, _static, _opt_state, metrics, next_target = lookahead_step(
        params,
        static,
        opt_state,
        current_batch,
        next_batch,
    )

    np.testing.assert_allclose(next_target, expected_next_target, rtol=1e-6)
    assert float(metrics["teacher_target_age_steps"]) == 1.0


def test_1b_grouped_jepa_config() -> None:
    config = load_config("configs/1b_grouped_jepa.py")
    settings = PeakSetJEPASettings.from_config(config)

    assert resolve_training_route(config) == ("grouped_jepa", "jax")
    assert config.group_jepa_spectra_per_group == 8
    assert config.group_jepa_teacher_spectra_per_group == 3
    assert config.group_jepa_groups_per_batch == 256
    assert config.group_jepa_ema_momentum == pytest.approx(0.9992)
    assert config.group_jepa_teacher_target_mode == "lookahead"
    assert config.group_jepa_init_checkpoint_path.endswith("/orbax/350000")
    assert config.optimizer == "muon"
    assert config.training_max_steps == 100_000
    assert config.warmup_steps == 5_000
    assert config.learning_rate == pytest.approx(4e-5)
    assert config.min_learning_rate == pytest.approx(4e-7)
    assert settings.pairmixer_fast_encoder_max_visible_tokens == 64


def test_1b_grouped_jepa_visreg_config() -> None:
    config = load_config("configs/1b_grouped_jepa_visreg.py")

    assert resolve_training_route(config) == ("grouped_jepa", "jax")
    assert config.group_jepa_use_ema_teacher is False
    assert config.group_jepa_ema_momentum is None
    assert config.group_jepa_teacher_target_mode == "same_step"
    assert config.group_jepa_invariance_loss_weight == pytest.approx(0.9)
    assert config.group_jepa_visreg_loss_weight == pytest.approx(0.1)
    assert config.group_jepa_visreg_num_projections == 256
    assert config.group_jepa_visreg_gather_embeddings is True
    assert config.training_max_steps == 300_000
