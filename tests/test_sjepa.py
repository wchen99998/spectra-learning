from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx
from ml_collections import config_dict

from spectra_learning.config import load_config
from spectra_learning.data.gems.collate import SJEPABatchCollator
from spectra_learning.models.settings import PeakSetJEPASettings
from spectra_learning.models.sjepa_jax import (
    SJEPASpectrumJax,
    sample_sigreg_directions,
    sigreg_loss,
    sjepa_metrics,
)
from spectra_learning.training import pretrain_jax
from spectra_learning.training.pretrain_jax import build_jax_optax_transform
from spectra_learning.training.routing import resolve_training_route


CONFIG_PATH = (
    "configs/1b_sjepa_dense_adam_metadata_adaln_n128_b2048_2m_hd256.py"
)


def _small_settings() -> PeakSetJEPASettings:
    return PeakSetJEPASettings(
        training_mode="sjepa",
        model_dim=24,
        encoder_num_layers=1,
        encoder_num_heads=4,
        feature_mlp_hidden_dim=32,
        encoder_fourier_mlp_hidden_dim=32,
        encoder_fourier_num_freqs=4,
        num_peaks=7,
        pairmixer_block_type="dense",
        pairmixer_pair_dim=8,
        pairmixer_pair_feature_hidden_dim=8,
        encoder_use_position_embedding=False,
        target_projector_dim=-1,
    )


def _small_batch() -> dict[str, jax.Array]:
    values = jnp.arange(2 * 7, dtype=jnp.float32).reshape(2, 7)
    global_masks = jnp.asarray(
        [
            [
                [True, True, True, True, True, False, False],
                [False, True, True, True, True, True, False],
            ],
            [
                [True, True, True, True, True, False, False],
                [False, True, True, True, True, True, False],
            ],
        ]
    )
    local_masks = jnp.asarray(
        [
            [
                [True, False, False, False, False, False, False],
                [False, True, False, False, False, False, False],
                [False, False, True, False, False, False, False],
            ],
            [
                [False, False, False, True, False, False, False],
                [False, False, False, False, True, False, False],
                [False, False, False, False, False, True, False],
            ],
        ]
    )
    return {
        "peak_mz": values / 1_000,
        "peak_intensity": (values % 7 + 1) / 7,
        "peak_valid_mask": jnp.ones((2, 7), dtype=jnp.bool_),
        "precursor_mz": jnp.full((2,), 0.5),
        "sjepa_global_view_masks": global_masks,
        "sjepa_local_view_masks": local_masks,
    }


def test_sjepa_config_is_standalone_and_has_requested_training_shape() -> None:
    source = Path(CONFIG_PATH).read_text()
    config = load_config(CONFIG_PATH)

    assert "import_module" not in source
    assert "configs." not in source
    assert resolve_training_route(config) == ("sjepa", "jax")
    assert config.batch_size == 2_048
    assert config.jax_mesh_devices == "32"
    assert config.sjepa_num_global_views == 2
    assert config.sjepa_num_local_views == 3
    assert config.sjepa_global_view_fraction == pytest.approx(0.75)
    assert config.sjepa_local_view_fraction == pytest.approx(0.20)
    assert config.batch_size * (
        config.sjepa_num_global_views + config.sjepa_num_local_views
    ) == 10_240
    assert config.sjepa_sigreg_loss_weight == pytest.approx(0.1)
    assert config.optimizer == "adam"
    assert config.weight_decay == 0.0
    assert config.training_max_steps == 2_000_000
    assert config.encoder_num_heads == 6
    assert config.model_dim // config.encoder_num_heads == 256


def test_sjepa_random_views_retain_the_configured_fraction() -> None:
    valid_mask = torch.tensor(
        [
            [True] * 10,
            [True] * 5 + [False] * 5,
        ]
    )
    views = SJEPABatchCollator._sample_views(
        object.__new__(SJEPABatchCollator), valid_mask,
        num_views=3,
        fraction=0.20,
    )

    assert views.shape == (2, 3, 10)
    assert views[0].sum(dim=-1).tolist() == [2, 2, 2]
    assert views[1].sum(dim=-1).tolist() == [1, 1, 1]
    assert not bool((views & ~valid_mask[:, None]).any())


def test_sjepa_global_teacher_is_stop_gradient() -> None:
    global_cls = (
        jnp.arange(2 * 2 * 4, dtype=jnp.float32).reshape(2, 2, 4) / 10
    )
    local_cls = jnp.arange(2 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 4) / 7
    directions = jnp.eye(4, dtype=jnp.float32)

    def loss(global_embeddings, local_embeddings):
        return sjepa_metrics(
            global_embeddings,
            local_embeddings,
            directions,
            invariance_loss_weight=1.0,
            sigreg_loss_weight=0.1,
        )["loss"]

    global_grad, local_grad = jax.grad(loss, argnums=(0, 1))(
        global_cls,
        local_cls,
    )

    assert bool(jnp.all(global_grad == 0))
    assert bool(jnp.all(jnp.isfinite(local_grad)))
    assert bool(jnp.all(jnp.linalg.norm(local_grad, axis=-1) > 0))


def test_sjepa_global_teacher_receives_gradient_when_enabled() -> None:
    global_cls = (
        jnp.arange(2 * 2 * 4, dtype=jnp.float32).reshape(2, 2, 4) / 10
    )
    local_cls = jnp.arange(2 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 4) / 7
    directions = jnp.eye(4, dtype=jnp.float32)

    def loss(global_embeddings, local_embeddings):
        return sjepa_metrics(
            global_embeddings,
            local_embeddings,
            directions,
            invariance_loss_weight=1.0,
            sigreg_loss_weight=0.1,
            teacher_stop_gradient=False,
        )["loss"]

    global_grad, local_grad = jax.grad(loss, argnums=(0, 1))(
        global_cls,
        local_cls,
    )

    assert bool(jnp.all(jnp.isfinite(global_grad)))
    assert bool(jnp.all(jnp.linalg.norm(global_grad, axis=-1) > 0))
    assert bool(jnp.all(jnp.isfinite(local_grad)))
    assert bool(jnp.all(jnp.linalg.norm(local_grad, axis=-1) > 0))


def test_sjepa_invariance_uses_mean_global_cls() -> None:
    global_views = jnp.asarray(
        [[[1.0, 3.0], [5.0, 7.0]]],
        dtype=jnp.float32,
    )
    local_cls = jnp.asarray(
        [[[2.0, 4.0], [4.0, 8.0]]],
        dtype=jnp.float32,
    )
    metrics = sjepa_metrics(
        global_views,
        local_cls,
        jnp.eye(2, dtype=jnp.float32),
        invariance_loss_weight=1.0,
        sigreg_loss_weight=0.0,
    )

    global_cls = global_views.mean(axis=1)
    expected = jnp.square(global_cls[:, None, :] - local_cls).mean()
    assert float(metrics["sjepa_loss"]) == pytest.approx(float(expected))


def test_sigreg_has_finite_gradient_for_all_embeddings() -> None:
    embeddings = jnp.arange(3 * 4 * 6, dtype=jnp.float32).reshape(3, 4, 6) / 10
    directions = jnp.eye(6, dtype=jnp.float32)
    gradient = jax.grad(sigreg_loss)(embeddings, directions)

    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert bool(jnp.all(jnp.linalg.norm(gradient, axis=-1) > 0))


def test_sigreg_computes_each_view_statistic_before_averaging() -> None:
    embeddings = jnp.asarray(
        [
            [[0.0, 0.5], [1.0, 1.5], [2.0, 2.5]],
            [[3.0, 3.5], [4.0, 4.5], [5.0, 5.5]],
        ],
        dtype=jnp.float32,
    )
    directions = jnp.eye(2, dtype=jnp.float32)

    loss = sigreg_loss(embeddings, directions)
    per_view_loss = jnp.stack(
        [sigreg_loss(view, directions) for view in embeddings]
    ).mean()
    duplicated_loss = sigreg_loss(
        jnp.concatenate([embeddings, embeddings], axis=0),
        directions,
    )

    np.testing.assert_allclose(loss, per_view_loss, rtol=1e-6)
    np.testing.assert_allclose(loss, duplicated_loss, rtol=1e-6)


def test_sigreg_directions_are_deterministic_per_step_and_resampled() -> None:
    directions_0 = sample_sigreg_directions(
        24,
        8,
        seed=66,
        step=jnp.asarray(0, dtype=jnp.uint32),
    )
    directions_0_again = sample_sigreg_directions(
        24,
        8,
        seed=66,
        step=jnp.asarray(0, dtype=jnp.uint32),
    )
    directions_1 = sample_sigreg_directions(
        24,
        8,
        seed=66,
        step=jnp.asarray(1, dtype=jnp.uint32),
    )

    np.testing.assert_array_equal(directions_0, directions_0_again)
    assert not bool(jnp.all(directions_0 == directions_1))
    np.testing.assert_allclose(
        jnp.linalg.norm(directions_0, axis=0),
        jnp.ones((8,)),
        rtol=1e-6,
    )


def test_sjepa_batches_receive_the_current_optimizer_step() -> None:
    loop = object.__new__(pretrain_jax._JaxTrainingLoop)
    loop.grad_accum_steps = 2
    loop.use_sjepa = True
    loop.global_step = 17
    loop.use_sharded_step = False
    loop.data_mesh = None
    loop.timing_barriers = False
    loop.measured_start = None

    micro_batch = {
        "peak_mz": np.zeros((3, 7), dtype=np.float32),
    }
    batch = loop._next_accumulated_batch(iter([micro_batch, micro_batch]))

    np.testing.assert_array_equal(
        batch["sjepa_sigreg_step"],
        np.full((2, 3), 17, dtype=np.uint32),
    )


def test_sjepa_model_contains_only_the_shared_encoder_and_compacts_views() -> None:
    model = SJEPASpectrumJax(
        _small_settings(),
        num_global_views=2,
        num_local_views=3,
        global_view_fraction=0.75,
        local_view_fraction=0.20,
        invariance_loss_weight=1.0,
        sigreg_loss_weight=0.1,
        sigreg_num_slices=8,
        sigreg_data_axis_name=None,
        sigreg_seed=66,
        rngs=nnx.Rngs(0),
    )
    batch = _small_batch()
    metrics = model(
        {
            **batch,
            "sjepa_sigreg_step": jnp.zeros((2,), dtype=jnp.uint32),
        }
    )
    next_step_metrics = model(
        {
            **batch,
            "sjepa_sigreg_step": jnp.ones((2,), dtype=jnp.uint32),
        }
    )
    params = nnx.state(model, nnx.Param)

    assert set(params) == {"encoder"}
    assert model.global_view_max_tokens == 6
    assert model.local_view_max_tokens == 2
    assert float(metrics["global_view_fraction"]) == pytest.approx(5 / 7)
    assert float(metrics["local_view_fraction"]) == pytest.approx(1 / 7)
    assert float(metrics["loss"]) == pytest.approx(
        float(metrics["sjepa_invariance_term"] + metrics["sjepa_sigreg_term"])
    )
    assert float(metrics["sjepa_loss"]) == pytest.approx(
        float(next_step_metrics["sjepa_loss"])
    )
    assert float(metrics["sigreg_loss"]) != pytest.approx(
        float(next_step_metrics["sigreg_loss"])
    )


def test_jax_plain_adam_does_not_construct_adamw(monkeypatch) -> None:
    sentinel = object()
    calls = []

    def fake_adam(**kwargs):
        calls.append(kwargs)
        return sentinel

    def fail_adamw(**_kwargs):
        raise AssertionError("plain Adam must not construct AdamW")

    monkeypatch.setattr("spectra_learning.training.pretrain_jax.optax.adam", fake_adam)
    monkeypatch.setattr(
        "spectra_learning.training.pretrain_jax.optax.adamw",
        fail_adamw,
    )
    config = config_dict.ConfigDict(
        {
            "optimizer": "adam",
            "learning_rate": 3e-4,
            "b1": 0.9,
            "b2": 0.95,
            "weight_decay": 0.0,
        }
    )

    transform = build_jax_optax_transform(config, total_steps=100)

    assert transform is sentinel
    assert calls[0]["b1"] == pytest.approx(0.9)
    assert calls[0]["b2"] == pytest.approx(0.95)
