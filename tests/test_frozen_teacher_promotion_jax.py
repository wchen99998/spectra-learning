from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from ml_collections import config_dict

from spectra_learning.models.model_jax import PeakSetJEPAJax
from spectra_learning.training import pretrain_jax
from spectra_learning.training.checkpointing_jax import (
    build_jax_checkpoint_manager,
    jax_training_checkpoint_metadata,
    save_jax_training_state,
)


def _loop(interval: int, *, projector_dim: int = -1):
    loop = object.__new__(pretrain_jax._JaxTrainingLoop)
    loop.config = config_dict.ConfigDict({
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "optimizer_state_dtype": "bf16",
        "frozen_teacher_update_every_steps": interval,
    })
    loop.model = PeakSetJEPAJax(
        training_mode="mae_teacher_jepa", model_dim=8, predictor_dim=8,
        encoder_num_layers=1, encoder_num_heads=2,
        feature_mlp_hidden_dim=8, encoder_fourier_num_freqs=2,
        masked_latent_predictor_num_layers=1,
        masked_latent_predictor_num_heads=2, num_peaks=5,
        pairmixer_block_type="fastmixer-dense",
        target_projector_dim=projector_dim,
        masked_token_loss_weight=1.0, mae_loss_weight=0.0,
    )
    graphdef, params, static_state, opt_state, optimizer = (
        pretrain_jax.init_pure_optax_train_state(
            loop.config, loop.model, total_steps=10,
        )
    )
    loop.state = pretrain_jax._JaxTrainState(params, static_state, opt_state)
    loop.train_step = pretrain_jax.make_pure_accumulated_train_step(
        graphdef, optimizer, sharded=False,
    )
    loop.frozen_teacher_update_every_steps = interval
    loop.group_jepa_lookahead = False
    loop.group_jepa_ema_momentum = None
    loop.timing_barriers = False
    loop.compile_stall_threshold_seconds = 0.0
    loop.measured_start = None
    loop.global_step = 0
    return loop


def _batch():
    return {
        "peak_mz": jnp.array([[[0.1, 0.2, 0.3, 0.4, 0.5]]]),
        "peak_intensity": jnp.array([[[1.0, 0.2, 0.5, 0.3, 0.8]]]),
        "peak_valid_mask": jnp.ones((1, 1, 5), dtype=jnp.bool_),
        "context_mask": jnp.array([[[True, True, False, False, False]]]),
        "target_masks": jnp.array([[[[False, False, True, True, True]]]]),
        "precursor_mz": jnp.array([[0.9]]),
    }


def _host_snapshot(tree):
    return jax.tree.map(lambda value: np.array(value, copy=True), tree)


def _assert_equal(actual, expected):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for value, reference in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(value), np.asarray(reference))


def _assert_teacher_matches_student(loop):
    nnx.update(loop.model, loop.state.trainable_params, loop.state.static_state)
    student = nnx.as_pure(nnx.state(loop.model.encoder))
    teacher = loop.state.static_state["teacher_encoder"]
    _assert_equal(teacher, student)
    for target, source in zip(
        jax.tree.leaves(teacher), jax.tree.leaves(student), strict=True,
    ):
        assert target.unsafe_buffer_pointer() != source.unsafe_buffer_pointer()


@pytest.mark.parametrize("interval", [0, 2])
def test_teacher_is_frozen_between_post_update_student_promotions(interval):
    loop = _loop(interval)
    expected_teacher = _host_snapshot(loop.state.static_state["teacher_encoder"])
    batch = _batch()

    for step in range(1, 4):
        metrics, _ = loop._train_batch(batch, None)
        assert np.isfinite(np.asarray(metrics["loss"]))
        assert loop.global_step == step
        if interval and step % interval == 0:
            _assert_teacher_matches_student(loop)
            expected_teacher = _host_snapshot(
                loop.state.static_state["teacher_encoder"]
            )
        else:
            _assert_equal(
                loop.state.static_state["teacher_encoder"], expected_teacher,
            )

    # The third optimizer step donates student buffers. Its target must still
    # contain the second-step snapshot, and the student must continue learning.
    student_weight = loop.state.trainable_params["encoder"]["cls_token"]
    assert not np.array_equal(
        np.asarray(student_weight), expected_teacher["cls_token"],
    )


def test_promotion_copies_encoder_buffers_and_target_projector():
    loop = _loop(2, projector_dim=4)
    static_before = loop.state.static_state
    initial_teacher_token = np.array(
        static_before["teacher_encoder"]["cls_token"], copy=True,
    )
    promoted = pretrain_jax.promote_frozen_teacher_state(
        loop.state.trainable_params, static_before,
    )
    loop.state.static_state = promoted
    _assert_teacher_matches_student(loop)
    _assert_equal(
        promoted["teacher_target_projector"],
        loop.state.trainable_params["target_projector"],
    )
    # Fourier frequencies are excluded from the optimizer but are part of the
    # encoder snapshot just as much as its trainable weights.
    assert jax.tree.leaves(promoted["encoder"])
    _assert_equal(promoted["encoder"], static_before["encoder"])
    assert not np.array_equal(
        initial_teacher_token,
        np.asarray(promoted["teacher_encoder"]["cls_token"]),
    )


def test_checkpoint_resume_preserves_teacher_until_next_global_boundary(tmp_path):
    loop = _loop(2)
    batch = _batch()
    for _ in range(3):
        loop._train_batch(batch, None)
    expected_teacher = _host_snapshot(loop.state.static_state["teacher_encoder"])
    metadata = jax_training_checkpoint_metadata("test", {})
    manager = build_jax_checkpoint_manager(
        tmp_path, enable_async_checkpointing=False,
    )
    try:
        save_jax_training_state(
            manager, loop.global_step, loop.state.checkpoint_state(),
            metadata=metadata,
        )
        resumed = _loop(2)
        resumed.checkpoint_manager = manager
        resumed.checkpoint_metadata = metadata
        resumed.global_step = resumed._restore_checkpoint(3)
        _assert_equal(
            resumed.state.static_state["teacher_encoder"], expected_teacher,
        )
        resumed._train_batch(batch, None)
        assert resumed.global_step == 4
        _assert_teacher_matches_student(resumed)
    finally:
        manager.close()


def test_disabled_promotion_preserves_legacy_checkpoint_contract():
    legacy = config_dict.ConfigDict({"training_mode": "mae_teacher_jepa"})
    disabled = config_dict.ConfigDict({
        "training_mode": "mae_teacher_jepa",
        "frozen_teacher_update_every_steps": 0,
    })
    enabled = config_dict.ConfigDict({
        "training_mode": "mae_teacher_jepa",
        "frozen_teacher_update_every_steps": 50_000,
    })
    assert pretrain_jax.jax_config_checkpoint_contract(legacy) == (
        pretrain_jax.jax_config_checkpoint_contract(disabled)
    )
    assert pretrain_jax.jax_config_checkpoint_contract(enabled)["config"][
        "frozen_teacher_update_every_steps"
    ] == 50_000


@pytest.mark.parametrize("interval", [-1, 1.5, True])
def test_promotion_rejects_invalid_intervals(interval):
    config = config_dict.ConfigDict({
        "frozen_teacher_update_every_steps": interval,
    })
    with pytest.raises(ValueError, match="nonnegative integer"):
        pretrain_jax._validate_frozen_teacher_update_config(config, object())


def test_promotion_requires_frozen_teacher_mode():
    config = config_dict.ConfigDict({"frozen_teacher_update_every_steps": 2})
    with pytest.raises(ValueError, match="mae_teacher_jepa"):
        pretrain_jax._validate_frozen_teacher_update_config(config, object())


def test_promotion_allows_full_teacher_and_compact_student_capacities():
    loop = _loop(2)
    loop.model.encoder.pairmixer_fast_max_visible_tokens = 3
    loop.model.teacher_encoder.pairmixer_fast_max_visible_tokens = 6
    assert pretrain_jax._validate_frozen_teacher_update_config(
        loop.config, loop.model,
    ) == 2
    assert loop.model.teacher_encoder.pairmixer_fast_max_visible_tokens == 6


@pytest.mark.parametrize("mismatch", ["shape", "normalization"])
def test_promotion_rejects_incompatible_encoders(mismatch):
    loop = _loop(2)
    if mismatch == "shape":
        loop.model.teacher_encoder.cls_token = nnx.Param(jnp.zeros((9,)))
    else:
        loop.model.teacher_encoder.final_norm.eps *= 10
    with pytest.raises(ValueError, match="matching student and teacher encoders"):
        pretrain_jax._validate_frozen_teacher_update_config(loop.config, loop.model)
