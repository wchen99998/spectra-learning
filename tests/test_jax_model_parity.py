from __future__ import annotations

import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx
from ml_collections import config_dict
from torch.utils.data import DataLoader

from spectra_learning.data.gems.collate import GemsBatchCollator
from spectra_learning.models.model import PeakSetJEPA
from spectra_learning.models.model_jax import PeakSetJEPAJax
from spectra_learning.models.pairmixer_jax import PairMixerBlock as JaxPairMixerBlock
from spectra_learning.models.transformer_jax import FeedForward as JaxFeedForward
from spectra_learning.training.checkpointing import save_torch_checkpoint
from spectra_learning.training.pretrain_jax import (
    build_jax_optimizer,
    jax_apply_grads,
    jax_grad_step,
    jax_sharded_apply_grads,
    jax_sharded_grad_step,
    jax_train_step,
    initialize_jax_model_from_torch_seed,
    init_pure_optax_train_state,
    make_pure_accumulated_train_step,
    numpy_batch_to_jax,
    trainable_param_filter,
)


def _small_mae_kwargs() -> dict[str, object]:
    return {
        "training_mode": "mae",
        "model_dim": 16,
        "encoder_num_layers": 1,
        "encoder_num_heads": 4,
        "attention_mlp_multiple": 2.0,
        "feature_mlp_hidden_dim": 8,
        "encoder_fourier_num_freqs": 4,
        "pairmixer_fourier_num_freqs": 2,
        "pairmixer_pair_dim": 12,
        "pairmixer_pair_feature_hidden_dim": 8,
        "masked_latent_predictor_num_layers": 1,
        "masked_latent_predictor_num_heads": 4,
        "num_peaks": 5,
        "jepa_num_target_blocks": 1,
        "distogram_loss_weight": 1.0,
        "predictor_dropout": 0.0,
        "target_projector_dim": -1,
    }


def _tiny_mae_kwargs() -> dict[str, object]:
    return {
        "training_mode": "mae",
        "model_dim": 4,
        "encoder_num_layers": 1,
        "encoder_num_heads": 1,
        "attention_mlp_multiple": 1.0,
        "feature_mlp_hidden_dim": 4,
        "encoder_fourier_num_freqs": 1,
        "pairmixer_fourier_num_freqs": 1,
        "pairmixer_pair_dim": 4,
        "pairmixer_pair_feature_hidden_dim": 4,
        "masked_latent_predictor_num_layers": 1,
        "masked_latent_predictor_num_heads": 1,
        "num_peaks": 3,
        "jepa_num_target_blocks": 1,
        "distogram_loss_weight": 0.0,
        "predictor_dropout": 0.0,
        "target_projector_dim": -1,
        "jepa_mae_mz_bin_size": 100.0,
        "jepa_mae_intensity_bin_size": 0.5,
    }


def _small_bi_dense_mae_kwargs() -> dict[str, object]:
    return {
        **_small_mae_kwargs(),
        "pairmixer_block_type": "bi-dense",
    }


def _small_jepa_kwargs(**overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "training_mode": "jepa",
        "model_dim": 16,
        "encoder_num_layers": 1,
        "encoder_num_heads": 4,
        "attention_mlp_multiple": 2.0,
        "feature_mlp_hidden_dim": 8,
        "encoder_fourier_num_freqs": 4,
        "pairmixer_fourier_num_freqs": 2,
        "pairmixer_pair_dim": 12,
        "pairmixer_pair_feature_hidden_dim": 8,
        "masked_latent_predictor_num_layers": 1,
        "masked_latent_predictor_num_heads": 4,
        "num_peaks": 5,
        "jepa_num_target_blocks": 1,
        "masked_token_loss_weight": 1.0,
        "distogram_loss_weight": 1.0,
        "latent_pair_loss_weight": 1.0,
        "latent_pair_target_normalization": "layernorm",
        "predictor_dropout": 0.0,
        "target_projector_dim": -1,
    }
    kwargs.update(overrides)
    return kwargs


def test_jax_native_dense_encoder_cls_pair_tokens_are_random_initialized():
    model = PeakSetJEPAJax(**_small_mae_kwargs())
    encoder = model.encoder

    for name in (
        "cls_to_peak_pair_token",
        "peak_to_cls_pair_token",
        "cls_cls_pair_token",
    ):
        assert not np.allclose(np.asarray(getattr(encoder, name)[...]), 0.0), name


def _sample(
    mz: list[float],
    intensity: list[float],
    precursor_mz: float,
    collision_energy: float,
    charge: float,
) -> dict[str, torch.Tensor]:
    spectra = torch.zeros(2, 128, dtype=torch.float32)
    spectra[0, : len(mz)] = torch.tensor(mz, dtype=torch.float32)
    spectra[1, : len(intensity)] = torch.tensor(intensity, dtype=torch.float32)
    return {
        "spectra": spectra,
        "precursor_mz_raw": torch.tensor(precursor_mz, dtype=torch.float32),
        "collision_energy": torch.tensor(collision_energy, dtype=torch.float32),
        "charge": torch.tensor(charge, dtype=torch.float32),
    }


def _real_pattern_batch(
    mask_strategy: str,
    *,
    num_peaks: int = 5,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(123)
    samples = [
        _sample(
            [100.0, 125.0, 150.0, 175.0, 200.0],
            [1.0, 0.8, 0.4, 0.2, 0.1],
            500.0,
            20.0,
            1.0,
        ),
        _sample([220.0, 240.0, 300.0], [0.9, 0.3, 0.2], 620.0, 35.0, 2.0),
        _sample(
            [80.0, 81.0, 120.0, 180.0, 260.0, 400.0],
            [0.5, 1.0, 0.7, 0.4, 0.2, 0.1],
            700.0,
            60.0,
            3.0,
        ),
    ]
    collator = GemsBatchCollator(
        augment=True,
        num_target_blocks=1,
        context_fraction=0.4,
        target_fraction=0.35,
        block_min_len=1,
        num_peaks=num_peaks,
        max_precursor_mz=1000.0,
        min_peak_intensity=0.0,
        peak_drop_min_intensity=0.0,
        peak_ordering="mz",
        precursor_peak_exclusion_window_da=0.0,
        mask_strategy=mask_strategy,
        mask_lengths=(1, 2, 3),
        mask_round_from=2,
    )
    loader = DataLoader(samples, batch_size=3, collate_fn=collator, num_workers=0)
    return next(iter(loader))


def _real_pattern_batch_size(
    mask_strategy: str,
    batch_size: int,
    *,
    num_peaks: int = 5,
) -> dict[str, torch.Tensor]:
    batch = _real_pattern_batch(mask_strategy, num_peaks=num_peaks)
    indices = torch.arange(batch_size) % next(iter(batch.values())).shape[0]
    return {key: value[indices] for key, value in batch.items()}


def _numpy_batch(batch: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {key: value.detach().cpu().numpy() for key, value in batch.items()}


def _jax_batch(batch: dict[str, torch.Tensor]) -> dict[str, jax.Array]:
    return numpy_batch_to_jax(_numpy_batch(batch))


def _assert_metrics_close(
    torch_metrics: dict[str, torch.Tensor],
    jax_metrics,
    *,
    atol: float = 2e-5,
) -> None:
    for key, torch_value in torch_metrics.items():
        expected = torch_value.detach().cpu().numpy()
        actual = np.asarray(jax_metrics[key])
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=atol, err_msg=key)


def _assert_jax_metrics_close(
    expected_metrics,
    actual_metrics,
    *,
    atol: float = 2e-5,
) -> None:
    for key, expected_value in expected_metrics.items():
        np.testing.assert_allclose(
            np.asarray(actual_metrics[key]),
            np.asarray(expected_value),
            rtol=1e-5,
            atol=atol,
            err_msg=key,
        )


@pytest.mark.parametrize("mask_strategy", ["contiguous", "random"])
def test_jax_mae_matches_pytorch_on_real_collated_batch(mask_strategy: str):
    torch.manual_seed(7)
    kwargs = _small_mae_kwargs()
    torch_model = PeakSetJEPA(**kwargs).eval()
    jax_model = PeakSetJEPAJax(**kwargs)
    jax_model.load_torch_state_dict(torch_model.state_dict())
    batch = _real_pattern_batch(mask_strategy)

    with torch.no_grad():
        torch_metrics = torch_model(batch)
    jax_metrics = jax_model(_jax_batch(batch))

    _assert_metrics_close(torch_metrics, jax_metrics)


def test_initialize_jax_model_from_torch_seed_matches_pytorch_forward():
    seed = 123
    kwargs = _small_mae_kwargs()
    torch.manual_seed(seed)
    torch_model = PeakSetJEPA(**kwargs).eval()
    jax_model = PeakSetJEPAJax(**kwargs)
    cfg = config_dict.ConfigDict(kwargs)
    cfg.seed = seed

    initialize_jax_model_from_torch_seed(cfg, jax_model)

    batch = _real_pattern_batch("contiguous")
    with torch.no_grad():
        torch_metrics = torch_model(batch)
    jax_metrics = jax_model(_jax_batch(batch))

    _assert_metrics_close(torch_metrics, jax_metrics)


def test_jax_bi_dense_mae_matches_pytorch_on_real_collated_batch():
    torch.manual_seed(8)
    kwargs = _small_bi_dense_mae_kwargs()
    torch_model = PeakSetJEPA(**kwargs).eval()
    jax_model = PeakSetJEPAJax(**kwargs)
    jax_model.load_torch_state_dict(torch_model.state_dict())
    batch = _real_pattern_batch("contiguous")

    with torch.no_grad():
        torch_metrics = torch_model(batch)
    jax_metrics = jax_model(_jax_batch(batch))

    _assert_metrics_close(torch_metrics, jax_metrics)


def test_jax_feedforward_pairmixer_matches_pytorch_on_real_collated_batch():
    torch.manual_seed(18)
    kwargs = {
        **_small_bi_dense_mae_kwargs(),
        "pairmixer_transition_type": "feedforward",
    }
    torch_model = PeakSetJEPA(**kwargs).eval()
    jax_model = PeakSetJEPAJax(**kwargs)
    jax_model.load_torch_state_dict(torch_model.state_dict())
    batch = _real_pattern_batch("contiguous")

    with torch.no_grad():
        torch_metrics = torch_model(batch)
    jax_metrics = jax_model(_jax_batch(batch))

    _assert_metrics_close(torch_metrics, jax_metrics)


@pytest.mark.parametrize("transition_type", ("swiglu", "feedforward"))
def test_jax_fastmixer_matches_bi_dense_on_fixed_random_masks(transition_type: str):
    torch.manual_seed(9)
    dense_kwargs = {
        **_small_bi_dense_mae_kwargs(),
        "pairmixer_block_type": "bi-dense",
        "pairmixer_transition_type": transition_type,
    }
    fast_kwargs = {
        **dense_kwargs,
        "pairmixer_block_type": "FastMixer",
        "pairmixer_fast_max_visible_tokens": 5,
    }
    torch_model = PeakSetJEPA(**dense_kwargs).eval()
    dense_model = PeakSetJEPAJax(**dense_kwargs)
    fast_model = PeakSetJEPAJax(**fast_kwargs)
    state_dict = torch_model.state_dict()
    dense_model.load_torch_state_dict(state_dict)
    fast_model.load_torch_state_dict(state_dict)
    batch = _real_pattern_batch("random")

    dense_metrics = dense_model(_jax_batch(batch))
    fast_metrics = fast_model(_jax_batch(batch))

    _assert_jax_metrics_close(dense_metrics, fast_metrics)


@pytest.mark.parametrize("transition_type", ("swiglu", "feedforward"))
def test_jax_fastmixer_dense_matches_dense_on_fixed_random_masks(transition_type: str):
    torch.manual_seed(10)
    dense_kwargs = {
        **_small_mae_kwargs(),
        "pairmixer_block_type": "dense",
        "pairmixer_transition_type": transition_type,
    }
    fast_kwargs = {
        **dense_kwargs,
        "pairmixer_block_type": "FastMixer-Dense",
        "pairmixer_fast_max_visible_tokens": 5,
    }
    torch_model = PeakSetJEPA(**dense_kwargs).eval()
    dense_model = PeakSetJEPAJax(**dense_kwargs)
    fast_model = PeakSetJEPAJax(**fast_kwargs)
    state_dict = torch_model.state_dict()
    dense_model.load_torch_state_dict(state_dict)
    fast_model.load_torch_state_dict(state_dict)
    batch = _real_pattern_batch("random")

    dense_metrics = dense_model(_jax_batch(batch))
    fast_metrics = fast_model(_jax_batch(batch))

    _assert_jax_metrics_close(dense_metrics, fast_metrics)


def test_jax_native_bi_dense_pairmixer_uses_torch_style_initialization():
    block = JaxPairMixerBlock(
        single_dim=8,
        pair_dim=6,
        num_heads=2,
        attention_mlp_multiple=2.0,
        norm_eps=1e-5,
        dropout=0.0,
        use_single_to_pair_update=True,
        rngs=nnx.Rngs(123),
    )
    single = jnp.arange(2 * 4 * 8, dtype=jnp.float32).reshape(2, 4, 8) / 17.0
    pair = jnp.arange(2 * 4 * 4 * 6, dtype=jnp.float32).reshape(2, 4, 4, 6) / 19.0
    mask = jnp.ones((2, 4), dtype=jnp.bool_)

    single_out, pair_out = block(single, pair, mask, mask)

    assert not np.allclose(np.asarray(block.tri_mul_out.p_in.weight[...]), 0.0)
    assert not np.allclose(np.asarray(block.pair_transition.fc1.weight[...]), 0.0)
    assert not np.allclose(np.asarray(block.pair_transition.fc2.weight[...]), 0.0)
    assert not np.allclose(np.asarray(block.single_to_pair_update.left.weight[...]), 0.0)
    np.testing.assert_allclose(
        np.asarray(block.pair_transition.fc3.weight[...]),
        0.0,
    )
    np.testing.assert_allclose(
        np.asarray(block.tri_mul_out.g_in.weight[...]),
        0.0,
    )
    np.testing.assert_allclose(
        np.asarray(block.tri_mul_out.g_in.bias[...]),
        1.0,
    )
    np.testing.assert_allclose(
        np.asarray(block.single_to_pair_update.gate.bias[...]),
        1.0,
    )
    assert not np.allclose(np.asarray(single_out), np.asarray(single))
    assert not np.allclose(np.asarray(pair_out), np.asarray(pair))


def test_jax_native_feedforward_pairmixer_uses_previous_transition_modules():
    block = JaxPairMixerBlock(
        single_dim=8,
        pair_dim=6,
        num_heads=2,
        attention_mlp_multiple=2.0,
        norm_eps=1e-5,
        dropout=0.0,
        use_single_to_pair_update=True,
        transition_type="feedforward",
        rngs=nnx.Rngs(123),
    )
    single = jnp.arange(2 * 4 * 8, dtype=jnp.float32).reshape(2, 4, 8) / 17.0
    pair = jnp.arange(2 * 4 * 4 * 6, dtype=jnp.float32).reshape(2, 4, 4, 6) / 19.0
    mask = jnp.ones((2, 4), dtype=jnp.bool_)

    single_out, pair_out = block(single, pair, mask, mask)

    assert isinstance(block.pair_transition, JaxFeedForward)
    assert isinstance(block.single_transition, JaxFeedForward)
    assert not np.allclose(np.asarray(block.pair_transition.w1.weight[...]), 0.0)
    assert not np.allclose(np.asarray(block.pair_transition.w2.weight[...]), 0.0)
    assert not np.allclose(np.asarray(single_out), np.asarray(single))
    assert not np.allclose(np.asarray(pair_out), np.asarray(pair))


def test_jax_model_loads_plain_pytorch_checkpoint_and_matches_output():
    torch.manual_seed(11)
    kwargs = _small_mae_kwargs()
    torch_model = PeakSetJEPA(**kwargs).eval()
    batch = _real_pattern_batch("ragged")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/step-00000009.pt"
        save_torch_checkpoint(
            {
                "model": torch_model.state_dict(),
                "optimizers": [],
                "schedulers": [],
                "grad_scaler": None,
                "global_step": 9,
                "epoch": 0,
                "loss": 0.0,
                "wandb_run_id": None,
            },
            path,
        )
        jax_model = PeakSetJEPAJax(**kwargs)
        jax_model.load_torch_checkpoint(path)

    with torch.no_grad():
        torch_metrics = torch_model(batch)
    jax_metrics = jax_model(_jax_batch(batch))

    _assert_metrics_close(torch_metrics, jax_metrics)


def test_jax_optax_train_step_updates_loaded_pytorch_weights():
    torch.manual_seed(17)
    kwargs = _small_mae_kwargs()
    torch_model = PeakSetJEPA(**kwargs).eval()
    jax_model = PeakSetJEPAJax(**kwargs)
    jax_model.load_torch_state_dict(torch_model.state_dict())
    optimizer = build_jax_optimizer(
        {"learning_rate": 1e-3, "weight_decay": 0.0, "b2": 0.95},
        jax_model,
    )
    batch = _jax_batch(_real_pattern_batch("contiguous"))
    before = np.asarray(jax_model.jepa_mae_mz_head.weight[...])

    metrics = jax_train_step(jax_model, optimizer, batch)

    after = np.asarray(jax_model.jepa_mae_mz_head.weight[...])
    assert np.isfinite(np.asarray(metrics["loss"]))
    assert not np.allclose(before, after)


@pytest.mark.parametrize("mode", ["full", "selective"])
def test_jax_activation_checkpointing_matches_uncheckpointed_update(mode: str):
    torch.manual_seed(18)
    kwargs = _small_mae_kwargs()
    torch_model = PeakSetJEPA(**kwargs).eval()
    base_model = PeakSetJEPAJax(**kwargs)
    checkpointed_model = PeakSetJEPAJax(
        **{
            **kwargs,
            "activation_checkpoint_mode": mode,
            "activation_checkpoint_every_n_layers": 1,
            "activation_checkpoint_modules": ("encoder", "predictor"),
        }
    )
    base_model.load_torch_state_dict(torch_model.state_dict())
    checkpointed_model.load_torch_state_dict(torch_model.state_dict())
    optimizer_config = {"learning_rate": 1e-3, "weight_decay": 0.0, "b2": 0.95}
    base_optimizer = build_jax_optimizer(optimizer_config, base_model)
    checkpointed_optimizer = build_jax_optimizer(optimizer_config, checkpointed_model)
    batch = _jax_batch(_real_pattern_batch("contiguous"))

    base_metrics = jax_train_step(base_model, base_optimizer, batch)
    checkpointed_metrics = jax_train_step(
        checkpointed_model,
        checkpointed_optimizer,
        batch,
    )

    np.testing.assert_allclose(
        np.asarray(checkpointed_metrics["loss"]),
        np.asarray(base_metrics["loss"]),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(checkpointed_model.jepa_mae_mz_head.weight[...]),
        np.asarray(base_model.jepa_mae_mz_head.weight[...]),
        rtol=1e-6,
        atol=1e-6,
    )


def test_jax_fastmixer_activation_checkpointing_matches_uncheckpointed_update():
    torch.manual_seed(19)
    kwargs = {
        **_small_bi_dense_mae_kwargs(),
        "encoder_num_layers": 2,
        "masked_latent_predictor_num_layers": 2,
        "pairmixer_block_type": "FastMixer",
        "pairmixer_fast_max_visible_tokens": 6,
    }
    torch_model = PeakSetJEPA(
        **{**kwargs, "pairmixer_block_type": "bi-dense"}
    ).eval()
    base_model = PeakSetJEPAJax(**kwargs)
    checkpointed_model = PeakSetJEPAJax(
        **{
            **kwargs,
            "activation_checkpoint_mode": "selective",
            "activation_checkpoint_every_n_layers": 1,
            "activation_checkpoint_modules": ("encoder", "predictor"),
        }
    )
    base_model.load_torch_state_dict(torch_model.state_dict())
    checkpointed_model.load_torch_state_dict(torch_model.state_dict())
    optimizer_config = {"learning_rate": 1e-3, "weight_decay": 0.0, "b2": 0.95}
    base_optimizer = build_jax_optimizer(optimizer_config, base_model)
    checkpointed_optimizer = build_jax_optimizer(optimizer_config, checkpointed_model)
    batch = _jax_batch(_real_pattern_batch("random"))

    base_metrics = jax_train_step(base_model, base_optimizer, batch)
    checkpointed_metrics = jax_train_step(
        checkpointed_model,
        checkpointed_optimizer,
        batch,
    )

    np.testing.assert_allclose(
        np.asarray(checkpointed_metrics["loss"]),
        np.asarray(base_metrics["loss"]),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(checkpointed_model.jepa_mae_mz_head.weight[...]),
        np.asarray(base_model.jepa_mae_mz_head.weight[...]),
        rtol=1e-6,
        atol=1e-6,
    )


def test_jax_bf16_autocast_uses_bf16_activations_and_fp32_loss():
    kwargs = {**_small_mae_kwargs(), "autocast_dtype": "bf16"}
    jax_model = PeakSetJEPAJax(**kwargs)
    batch = _jax_batch(_real_pattern_batch("contiguous"))

    encoded, pair = jax_model.encoder.forward_with_pair(
        batch["peak_mz"],
        batch["peak_intensity"],
        valid_mask=batch["peak_valid_mask"],
        visible_mask=batch["peak_valid_mask"],
        precursor_mz=batch.get("precursor_mz", None),
    )
    metrics = jax_model(batch)

    assert encoded.dtype == jnp.bfloat16
    assert pair.dtype == jnp.bfloat16
    assert metrics["loss"].dtype == jnp.float32


def test_jax_accumulated_grad_step_updates_loaded_pytorch_weights():
    import jax

    torch.manual_seed(19)
    kwargs = _small_mae_kwargs()
    torch_model = PeakSetJEPA(**kwargs).eval()
    jax_model = PeakSetJEPAJax(**kwargs)
    jax_model.load_torch_state_dict(torch_model.state_dict())
    optimizer = build_jax_optimizer(
        {"learning_rate": 1e-3, "weight_decay": 0.0, "b2": 0.95},
        jax_model,
    )
    batch = _jax_batch(_real_pattern_batch("random"))
    before = np.asarray(jax_model.jepa_mae_mz_head.weight[...])

    (_loss_0, metrics), grads_0 = jax_grad_step(jax_model, batch)
    (_loss_1, _metrics_1), grads_1 = jax_grad_step(jax_model, batch)
    grads = jax.tree.map(lambda lhs, rhs: (lhs + rhs) * 0.5, grads_0, grads_1)
    jax_apply_grads(jax_model, optimizer, grads)

    after = np.asarray(jax_model.jepa_mae_mz_head.weight[...])
    assert np.isfinite(np.asarray(metrics["loss"]))
    assert not np.allclose(before, after)


def test_jax_pure_optax_accumulated_train_step_matches_manual_accumulation():
    torch.manual_seed(22)
    kwargs = _small_mae_kwargs()
    torch_model = PeakSetJEPA(**kwargs).eval()
    manual_model = PeakSetJEPAJax(**kwargs)
    pure_model = PeakSetJEPAJax(**kwargs)
    manual_model.load_torch_state_dict(torch_model.state_dict())
    pure_model.load_torch_state_dict(torch_model.state_dict())
    optimizer_config = {"learning_rate": 1e-3, "weight_decay": 0.0, "b2": 0.95}
    manual_optimizer = build_jax_optimizer(optimizer_config, manual_model)
    batch_0 = _jax_batch(_real_pattern_batch("contiguous"))
    batch_1 = _jax_batch(_real_pattern_batch("random"))
    accumulated_batch = jax.tree.map(
        lambda lhs, rhs: jnp.stack([lhs, rhs]),
        batch_0,
        batch_1,
    )

    (_loss_0, _metrics_0), grads_0 = jax_grad_step(manual_model, batch_0)
    (_loss_1, _metrics_1), grads_1 = jax_grad_step(manual_model, batch_1)
    grads = jax.tree.map(lambda lhs, rhs: (lhs + rhs) * 0.5, grads_0, grads_1)
    jax_apply_grads(manual_model, manual_optimizer, grads)
    graphdef, trainable_params, static_state, opt_state, optimizer = (
        init_pure_optax_train_state(optimizer_config, pure_model)
    )
    pure_train_step = make_pure_accumulated_train_step(
        graphdef,
        optimizer,
        sharded=False,
    )
    trainable_params, opt_state, metrics = pure_train_step(
        trainable_params,
        static_state,
        opt_state,
        accumulated_batch,
    )
    nnx.update(pure_model, trainable_params)

    assert np.isfinite(np.asarray(metrics["loss"]))
    assert "mae_loss" in metrics
    assert "distogram_loss" in metrics
    np.testing.assert_allclose(
        np.asarray(pure_model.jepa_mae_mz_head.weight[...]),
        np.asarray(manual_model.jepa_mae_mz_head.weight[...]),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.skipif(
    os.environ.get("RUN_JAX_SHARDED_TESTS") != "1",
    reason="explicit shard_map TPU smoke test",
)
def test_jax_sharded_accumulated_grad_step_updates_on_all_devices():
    torch.manual_seed(21)
    kwargs = _tiny_mae_kwargs()
    torch_model = PeakSetJEPA(**kwargs).eval()
    jax_model = PeakSetJEPAJax(**kwargs)
    jax_model.load_torch_state_dict(torch_model.state_dict())
    optimizer = build_jax_optimizer(
        {"learning_rate": 1e-3, "weight_decay": 0.0, "b2": 0.95},
        jax_model,
    )
    batch = _jax_batch(
        _real_pattern_batch_size(
            "contiguous",
            jax.device_count(),
            num_peaks=int(kwargs["num_peaks"]),
        )
    )
    before = np.asarray(jax_model.jepa_mae_mz_head.weight[...])

    (_loss_0, metrics), grads_0 = jax_sharded_grad_step(jax_model, batch)
    (_loss_1, _metrics_1), grads_1 = jax_sharded_grad_step(jax_model, batch)
    grads = jax.tree.map(lambda lhs, rhs: (lhs + rhs) * 0.5, grads_0, grads_1)
    jax_sharded_apply_grads(jax_model, optimizer, grads)

    after = np.asarray(jax_model.jepa_mae_mz_head.weight[...])
    assert np.isfinite(np.asarray(metrics["loss"]))
    assert not np.allclose(before, after)


@pytest.mark.skipif(
    os.environ.get("RUN_JAX_SHARDED_TESTS") != "1",
    reason="explicit shard_map TPU smoke test",
)
def test_jax_sharded_pure_optax_accumulated_train_step_updates_on_all_devices():
    torch.manual_seed(22)
    kwargs = _tiny_mae_kwargs()
    torch_model = PeakSetJEPA(**kwargs).eval()
    jax_model = PeakSetJEPAJax(**kwargs)
    jax_model.load_torch_state_dict(torch_model.state_dict())
    batch = _jax_batch(
        _real_pattern_batch_size(
            "contiguous",
            jax.device_count(),
            num_peaks=int(kwargs["num_peaks"]),
        )
    )
    accumulated_batch = jax.tree.map(lambda value: jnp.stack([value, value]), batch)
    graphdef, trainable_params, static_state, opt_state, optimizer = (
        init_pure_optax_train_state(
            {"learning_rate": 1e-3, "weight_decay": 0.0, "b2": 0.95},
            jax_model,
        )
    )
    pure_train_step = make_pure_accumulated_train_step(
        graphdef,
        optimizer,
        sharded=True,
    )
    before = np.asarray(jax_model.jepa_mae_mz_head.weight[...])

    trainable_params, opt_state, metrics = pure_train_step(
        trainable_params,
        static_state,
        opt_state,
        accumulated_batch,
    )
    nnx.update(jax_model, trainable_params)

    after = np.asarray(jax_model.jepa_mae_mz_head.weight[...])
    assert np.isfinite(np.asarray(metrics["loss"]))
    assert not np.allclose(before, after)


@pytest.mark.parametrize(
    ("kwargs", "mask_strategy"),
    [
        (_small_jepa_kwargs(), "contiguous"),
        (
            _small_jepa_kwargs(
                masked_token_input_mode="mz_sentinel",
                jepa_mae_loss_weight=0.5,
                distogram_loss_weight=0.0,
                latent_pair_loss_weight=0.0,
            ),
            "random",
        ),
        (
            _small_jepa_kwargs(
                jepa_target_normalization="zscore",
                target_projector_dim=8,
                latent_pair_loss_weight=0.0,
            ),
            "ragged",
        ),
    ],
)
def test_jax_jepa_matches_pytorch_on_real_collated_batch(
    kwargs: dict[str, object],
    mask_strategy: str,
):
    torch.manual_seed(23)
    torch_model = PeakSetJEPA(**kwargs).eval()
    jax_model = PeakSetJEPAJax(**kwargs)
    jax_model.load_torch_state_dict(torch_model.state_dict())
    batch = _real_pattern_batch(mask_strategy)

    with torch.no_grad():
        torch_metrics = torch_model(batch)
    jax_metrics = jax_model(_jax_batch(batch))

    _assert_metrics_close(torch_metrics, jax_metrics)


def test_jax_contrastive_matches_pytorch_on_real_collated_batch():
    torch.manual_seed(27)
    kwargs = _small_jepa_kwargs(
        training_mode="contrastive",
        distogram_loss_weight=0.0,
        latent_pair_loss_weight=0.0,
    )
    torch_model = PeakSetJEPA(**kwargs).eval()
    jax_model = PeakSetJEPAJax(**kwargs)
    jax_model.load_torch_state_dict(torch_model.state_dict())
    batch = _real_pattern_batch("ragged")

    with torch.no_grad():
        torch_metrics = torch_model(batch)
    jax_metrics = jax_model(_jax_batch(batch))

    _assert_metrics_close(torch_metrics, jax_metrics)


def test_jax_jepa_ema_teacher_checkpoint_matches_pytorch():
    torch.manual_seed(29)
    kwargs = _small_jepa_kwargs(
        use_ema_teacher=True,
        target_projector_dim=8,
        distogram_loss_weight=0.0,
        latent_pair_loss_weight=0.0,
    )
    torch_model = PeakSetJEPA(**kwargs).eval()
    batch = _real_pattern_batch("contiguous")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/step-00000002.pt"
        save_torch_checkpoint(
            {
                "model": torch_model.state_dict(),
                "optimizers": [],
                "schedulers": [],
                "grad_scaler": None,
                "global_step": 2,
                "epoch": 0,
                "loss": 0.0,
                "wandb_run_id": None,
            },
            path,
        )
        jax_model = PeakSetJEPAJax(**kwargs)
        jax_model.load_torch_checkpoint(path)

    with torch.no_grad():
        torch_metrics = torch_model(batch)
    jax_metrics = jax_model(_jax_batch(batch))

    _assert_metrics_close(torch_metrics, jax_metrics)


def test_jax_optimizer_excludes_frozen_teacher_and_buffer_params():
    torch.manual_seed(30)
    kwargs = _small_jepa_kwargs(
        use_ema_teacher=True,
        target_projector_dim=8,
        ema_teacher_momentum_start=0.5,
        ema_teacher_momentum_final=0.9,
        ema_teacher_schedule="linear",
        distogram_loss_weight=0.0,
        latent_pair_loss_weight=0.0,
    )
    torch_model = PeakSetJEPA(**kwargs).eval()
    jax_model = PeakSetJEPAJax(**kwargs)
    jax_model.load_torch_state_dict(torch_model.state_dict())
    trainable_paths = {
        ".".join(str(part) for part in path)
        for path, _value in nnx.state(jax_model, trainable_param_filter).flat_state()
    }
    assert not any("teacher_encoder" in path for path in trainable_paths)
    assert not any("teacher_target_projector" in path for path in trainable_paths)
    assert not any("position_embedding" in path for path in trainable_paths)
    assert not any(path.endswith(".b") for path in trainable_paths)

    optimizer = build_jax_optimizer(
        {"learning_rate": 1e-3, "weight_decay": 0.5, "b2": 0.95},
        jax_model,
    )
    batch = _jax_batch(_real_pattern_batch("contiguous"))
    before_student = np.asarray(jax_model.target_projector.linear0.weight[...])
    before_teacher = np.asarray(jax_model.teacher_target_projector.linear0.weight[...])
    before_encoder_teacher = np.asarray(jax_model.teacher_encoder.cls_token[...])
    before_position = np.asarray(jax_model.encoder.position_embedding.weight[...])
    before_fourier = np.asarray(jax_model.encoder.embedder.mz_fourier.b[...])

    (_loss, metrics), grads = jax_grad_step(jax_model, batch)
    jax_apply_grads(jax_model, optimizer, grads)

    after_student = np.asarray(jax_model.target_projector.linear0.weight[...])
    after_teacher = np.asarray(jax_model.teacher_target_projector.linear0.weight[...])
    after_encoder_teacher = np.asarray(jax_model.teacher_encoder.cls_token[...])
    after_position = np.asarray(jax_model.encoder.position_embedding.weight[...])
    after_fourier = np.asarray(jax_model.encoder.embedder.mz_fourier.b[...])
    assert np.isfinite(np.asarray(metrics["loss"]))
    assert not np.allclose(before_student, after_student)
    np.testing.assert_array_equal(after_teacher, before_teacher)
    np.testing.assert_array_equal(after_encoder_teacher, before_encoder_teacher)
    np.testing.assert_array_equal(after_position, before_position)
    np.testing.assert_array_equal(after_fourier, before_fourier)

    momentum = jax_model.update_ema_teacher(step=2, total_steps=4)
    assert momentum == pytest.approx(0.7)
    expected_teacher = before_teacher * momentum + after_student * (1.0 - momentum)
    np.testing.assert_allclose(
        np.asarray(jax_model.teacher_target_projector.linear0.weight[...]),
        expected_teacher,
        rtol=1e-6,
        atol=1e-6,
    )


def test_jax_mae_teacher_jepa_matches_pytorch_with_default_teacher():
    torch.manual_seed(31)
    kwargs = _small_jepa_kwargs(
        training_mode="mae_teacher_jepa",
        target_projector_dim=8,
        jepa_mae_loss_weight=0.5,
        distogram_loss_weight=0.0,
        latent_pair_loss_weight=0.0,
    )
    torch_model = PeakSetJEPA(**kwargs).eval()
    jax_model = PeakSetJEPAJax(**kwargs)
    jax_model.load_torch_state_dict(torch_model.state_dict())
    batch = _real_pattern_batch("random")

    with torch.no_grad():
        torch_metrics = torch_model(batch)
    jax_metrics = jax_model(_jax_batch(batch))

    _assert_metrics_close(torch_metrics, jax_metrics)


def test_jax_mae_teacher_jepa_matches_pytorch_with_teacher_config():
    torch.manual_seed(37)
    with tempfile.TemporaryDirectory() as tmpdir:
        teacher_config_path = f"{tmpdir}/teacher_config.py"
        with open(teacher_config_path, "w") as f:
            f.write(
                "from ml_collections import config_dict\n"
                "def get_config():\n"
                "    cfg = config_dict.ConfigDict()\n"
                "    cfg.training_mode = 'mae'\n"
                "    cfg.model_dim = 12\n"
                "    cfg.encoder_num_layers = 1\n"
                "    cfg.encoder_num_heads = 3\n"
                "    cfg.attention_mlp_multiple = 2.0\n"
                "    cfg.feature_mlp_hidden_dim = 8\n"
                "    cfg.encoder_fourier_num_freqs = 4\n"
                "    cfg.pairmixer_fourier_num_freqs = 2\n"
                "    cfg.pairmixer_pair_dim = 10\n"
                "    cfg.pairmixer_pair_feature_hidden_dim = 8\n"
                "    cfg.num_peaks = 5\n"
                "    cfg.jepa_num_target_blocks = 1\n"
                "    cfg.target_projector_dim = -1\n"
                "    return cfg\n"
            )
        kwargs = _small_jepa_kwargs(
            training_mode="mae_teacher_jepa",
            frozen_teacher_config_path=teacher_config_path,
            target_projector_dim=-1,
            jepa_mae_loss_weight=0.0,
            distogram_loss_weight=0.0,
            latent_pair_loss_weight=0.0,
        )
        torch_model = PeakSetJEPA(**kwargs).eval()
        jax_model = PeakSetJEPAJax(**kwargs)
        jax_model.load_torch_state_dict(torch_model.state_dict())
        batch = _real_pattern_batch("contiguous")

        with torch.no_grad():
            torch_metrics = torch_model(batch)
        jax_metrics = jax_model(_jax_batch(batch))

    _assert_metrics_close(torch_metrics, jax_metrics)
