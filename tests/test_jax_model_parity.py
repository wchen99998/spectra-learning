from __future__ import annotations

import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import torch
from flax import nnx
from ml_collections import config_dict
from torch.utils.data import DataLoader

from spectra_learning.data.gems.collate import GemsBatchCollator
from spectra_learning.models.model import PeakSetJEPA
from spectra_learning.models.model_jax import PeakSetJEPAJax
from spectra_learning.models.common_jax import RMSNorm
from spectra_learning.models.pairmixer_jax import (
    PairMixerBlock as JaxPairMixerBlock,
    SingleMixerBlock as JaxSingleMixerBlock,
)
from spectra_learning.models.transformer_jax import (
    CrossAttentionBlock as JaxCrossAttentionBlock,
)
from spectra_learning.models.transformer_jax import FeedForward as JaxFeedForward
from spectra_learning.training.checkpointing import (
    load_torch_checkpoint,
    save_torch_checkpoint,
)
from spectra_learning.training.pretrain_jax import (
    _jax_data_mesh_for_device_count,
    _replicate_tree_on_data_mesh,
    build_jax_optax_transform,
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
        "distogram_loss_weight": 0.0,
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
        "distogram_loss_weight": 0.25,
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
        "distogram_loss_weight": 0.0,
        "latent_pair_loss_weight": 0.0,
        "predictor_dropout": 0.0,
        "target_projector_dim": -1,
    }
    kwargs.update(overrides)
    return kwargs


def _run_canonical_train_step(
    model: PeakSetJEPAJax,
    optimizer_config: dict[str, float],
    batch: dict[str, jax.Array],
) -> dict[str, jax.Array]:
    graphdef, trainable_params, static_state, opt_state, optimizer = (
        init_pure_optax_train_state(optimizer_config, model)
    )
    train_step = make_pure_accumulated_train_step(
        graphdef,
        optimizer,
        sharded=False,
    )
    accumulated_batch = jax.tree.map(lambda value: value[None], batch)
    trainable_params, _opt_state, metrics = train_step(
        trainable_params,
        static_state,
        opt_state,
        accumulated_batch,
    )
    nnx.update(model, trainable_params)
    return metrics


def test_jax_rmsnorm_rejects_layernorm_checkpoint():
    norm = RMSNorm(4)
    state_dict = {
        "norm.weight": torch.ones(4),
        "norm.bias": torch.zeros(4),
    }

    with pytest.raises(ValueError, match="RMSNorm requires a fresh run"):
        norm.load_torch_state_dict(state_dict, "norm")


def test_jax_native_dense_encoder_cls_pair_tokens_are_random_initialized():
    model = PeakSetJEPAJax(
        **{**_small_mae_kwargs(), "distogram_loss_weight": 0.25}
    )
    encoder = model.encoder

    for name in (
        "cls_to_peak_pair_token",
        "peak_to_cls_pair_token",
        "cls_cls_pair_token",
    ):
        assert not np.allclose(np.asarray(getattr(encoder, name)[...]), 0.0), name


def test_zero_distogram_builds_single_stream_encoder():
    model = PeakSetJEPAJax(**_small_mae_kwargs())
    encoder = model.encoder

    assert isinstance(encoder.blocks[0], JaxSingleMixerBlock)
    assert encoder.pair_embedder is None
    assert encoder.cls_to_peak_pair_token is None
    assert encoder.peak_to_cls_pair_token is None
    assert encoder.cls_cls_pair_token is None
    assert encoder.final_pair_norm is None


@pytest.mark.parametrize("use_rope", (False, True))
def test_jax_singlemixer_position_embedding_controls_rope(use_rope: bool):
    block = JaxSingleMixerBlock(
        single_dim=8,
        num_heads=2,
        attention_mlp_multiple=2.0,
        norm_eps=1e-5,
        use_rope=use_rope,
        rngs=nnx.Rngs(123),
    )
    block.single_attention.o.weight[...] = jnp.eye(8)
    single = jnp.arange(4 * 8, dtype=jnp.float32).reshape(1, 4, 8) / 17.0
    mask = jnp.ones((1, 4), dtype=jnp.bool_)

    packed = block(single, mask, jnp.array([[0, 1, 2, 3]]))
    original = block(single, mask, jnp.array([[0, 2, 4, 6]]))

    assert np.allclose(np.asarray(original), np.asarray(packed)) == (not use_rope)


def test_jax_encoder_position_embedding_setting_reaches_singlemixer_rope():
    enabled = PeakSetJEPAJax(
        **{**_small_mae_kwargs(), "encoder_use_position_embedding": True}
    )
    disabled = PeakSetJEPAJax(
        **{**_small_mae_kwargs(), "encoder_use_position_embedding": False}
    )
    pair_disabled = PeakSetJEPAJax(
        **{
            **_small_mae_kwargs(),
            "encoder_use_position_embedding": False,
            "distogram_loss_weight": 0.25,
        }
    )

    assert enabled.encoder.blocks[0].single_attention.use_rope
    assert not disabled.encoder.blocks[0].single_attention.use_rope
    assert not pair_disabled.encoder.blocks[0].single_attention.use_rope
    assert not hasattr(enabled.encoder, "position_embedding")


def test_mae_without_cls_has_peak_only_encoder_shapes_and_jax_parity():
    torch.manual_seed(7)
    kwargs = {
        **_small_mae_kwargs(),
        "encoder_use_cls_token": False,
        "distogram_loss_weight": 0.25,
    }
    torch_model = PeakSetJEPA(**kwargs).eval()
    jax_model = PeakSetJEPAJax(**kwargs)
    jax_model.load_torch_state_dict(torch_model.state_dict())
    batch = _real_pattern_batch("random")

    with torch.no_grad():
        torch_single, torch_pair = torch_model.encoder.forward_with_pair(
            batch["peak_mz"],
            batch["peak_intensity"],
            valid_mask=batch["peak_valid_mask"],
        )
        torch_metrics = torch_model(batch)
    jax_single, jax_pair = jax_model.encoder.forward_with_pair(
        jnp.asarray(batch["peak_mz"].numpy()),
        jnp.asarray(batch["peak_intensity"].numpy()),
        valid_mask=jnp.asarray(batch["peak_valid_mask"].numpy()),
    )
    jax_metrics = jax_model(_jax_batch(batch))

    assert torch_single.shape[1] == batch["peak_mz"].shape[1]
    assert torch_pair.shape[1:3] == (
        batch["peak_mz"].shape[1],
        batch["peak_mz"].shape[1],
    )
    assert jax_single.shape == torch_single.shape
    assert jax_pair.shape == torch_pair.shape
    assert "encoder.cls_token" not in torch_model.state_dict()
    assert jax_model.encoder.cls_token is None
    _assert_metrics_close(torch_metrics, jax_metrics)


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


def test_jax_tokenized_single_and_pair_mz_match_pytorch() -> None:
    kwargs = {
        **_small_mae_kwargs(),
        "encoder_mz_embedding": "token",
        "encoder_mz_token_bin_size": 0.1,
        "encoder_mz_token_embedding_dim": 8,
        "pairmixer_mz_embedding": "token",
        "pairmixer_mz_token_bin_size": 0.1,
        "pairmixer_mz_token_embedding_dim": 8,
        "distogram_loss_weight": 0.25,
    }
    torch.manual_seed(7)
    torch_model = PeakSetJEPA(**kwargs).eval()
    jax_model = PeakSetJEPAJax(**kwargs)
    jax_model.load_torch_state_dict(torch_model.state_dict())
    batch = _real_pattern_batch("random")

    with torch.no_grad():
        torch_metrics = torch_model(batch)
    jax_metrics = jax_model(_jax_batch(batch))

    _assert_metrics_close(torch_metrics, jax_metrics)
    assert torch_model.encoder.embedder.mz_features.num_tokens == 10_000
    assert torch_model.encoder.pair_embedder.mz_features.num_tokens == 10_000

    valid_mask = torch.tensor([[True, True, True, False, False]])
    intensity = torch.tensor([[1.0, 0.8, 0.4, 0.0, 0.0]])
    left_mz = torch.tensor([[0.10001, 0.20001, 0.30001, 0.0, 0.0]])
    right_mz = torch.tensor([[0.10009, 0.20009, 0.30009, 0.0, 0.0]])
    with torch.no_grad():
        left = torch_model.encoder(left_mz, intensity, valid_mask=valid_mask)
        right = torch_model.encoder(right_mz, intensity, valid_mask=valid_mask)
    torch.testing.assert_close(left, right)


def test_jax_mae_without_intensity_head_matches_pytorch():
    torch.manual_seed(29)
    kwargs = {**_small_mae_kwargs(), "mae_intensity_loss_weight": 0.0}
    torch_model = PeakSetJEPA(**kwargs).eval()
    jax_model = PeakSetJEPAJax(**kwargs)
    assert torch_model.jepa_mae_intensity_head is None
    assert jax_model.jepa_mae_intensity_head is None
    jax_model.load_torch_state_dict(torch_model.state_dict())
    batch = _real_pattern_batch("contiguous")

    with torch.no_grad():
        torch_metrics = torch_model(batch)
    jax_metrics = jax_model(_jax_batch(batch))

    _assert_metrics_close(torch_metrics, jax_metrics)
    assert float(torch_metrics["mae_intensity_loss"]) == 0.0
    assert float(jax_metrics["mae_intensity_loss"]) == 0.0
    torch.testing.assert_close(torch_metrics["mae_loss"], torch_metrics["mae_mz_loss"])
    np.testing.assert_allclose(
        np.asarray(jax_metrics["mae_loss"]),
        np.asarray(jax_metrics["mae_mz_loss"]),
        rtol=1e-6,
        atol=1e-6,
    )


def test_jax_mae_without_pair_bias_matches_pytorch():
    torch.manual_seed(29)
    kwargs = {
        **_small_mae_kwargs(),
        "pairmixer_use_pair_bias": False,
        "distogram_loss_weight": 0.25,
    }
    torch_model = PeakSetJEPA(**kwargs).eval()
    jax_model = PeakSetJEPAJax(**kwargs)
    jax_model.load_torch_state_dict(torch_model.state_dict())
    batch = _real_pattern_batch("contiguous")

    with torch.no_grad():
        torch_metrics = torch_model(batch)
    jax_metrics = jax_model(_jax_batch(batch))

    _assert_metrics_close(torch_metrics, jax_metrics)


def test_jax_fastmixer_compact_mae_without_intensity_head():
    kwargs = {
        **_small_bi_dense_mae_kwargs(),
        "pairmixer_block_type": "FastMixer",
        "pairmixer_fast_max_visible_tokens": 6,
        "mae_intensity_loss_weight": 0.0,
    }
    model = PeakSetJEPAJax(**kwargs)
    assert model.jepa_mae_intensity_head is None
    metrics = model(_jax_batch(_real_pattern_batch("random")))

    assert float(metrics["mae_intensity_loss"]) == 0.0
    assert float(metrics["mae_intensity_accuracy"]) == 0.0
    np.testing.assert_allclose(
        np.asarray(metrics["mae_loss"]),
        np.asarray(metrics["mae_mz_loss"]),
        rtol=1e-6,
        atol=1e-6,
    )


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
@pytest.mark.parametrize("distogram_loss_weight", (0.0, 0.25))
def test_jax_fastmixer_dense_matches_dense_on_fixed_random_masks(
    transition_type: str,
    distogram_loss_weight: float,
):
    torch.manual_seed(10)
    dense_kwargs = {
        **_small_mae_kwargs(),
        "pairmixer_block_type": "dense",
        "pairmixer_transition_type": transition_type,
        "distogram_loss_weight": distogram_loss_weight,
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


@pytest.mark.parametrize("distogram_loss_weight", (0.0, 0.25))
def test_jax_fastmixer_target_only_compact_path_matches_dense_loss_and_gradients(
    distogram_loss_weight: float,
):
    torch.manual_seed(11)
    dense_kwargs = _small_jepa_kwargs(
        training_mode="mae_teacher_jepa",
        pairmixer_block_type="dense",
        target_projector_dim=-1,
        jepa_mae_loss_weight=0.0,
        distogram_loss_weight=distogram_loss_weight,
        latent_pair_loss_weight=0.0,
    )
    fast_kwargs = {
        **dense_kwargs,
        "pairmixer_block_type": "FastMixer-Dense",
        "pairmixer_fast_encoder_max_visible_tokens": 4,
        "pairmixer_fast_max_visible_tokens": 6,
    }
    torch_model = PeakSetJEPA(**dense_kwargs).eval()
    dense_model = PeakSetJEPAJax(**dense_kwargs)
    fast_model = PeakSetJEPAJax(**fast_kwargs)
    state_dict = torch_model.state_dict()
    dense_model.load_torch_state_dict(state_dict)
    fast_model.load_torch_state_dict(state_dict)
    fast_model.set_fastmixer_capacities(4, 6, 5)
    batch = _jax_batch(_real_pattern_batch("random"))

    dense_metrics = dense_model(batch)
    fast_metrics = fast_model(batch)

    _assert_jax_metrics_close(dense_metrics, fast_metrics)

    def loss_fn(model: PeakSetJEPAJax) -> jax.Array:
        return model(batch, loss_only=True)["loss"]

    grad_fn = nnx.grad(
        loss_fn,
        argnums=nnx.DiffState(0, trainable_param_filter),
    )
    dense_grads = dict(nnx.to_flat_state(nnx.as_pure(grad_fn(dense_model))))
    fast_grads = dict(nnx.to_flat_state(nnx.as_pure(grad_fn(fast_model))))
    assert fast_grads.keys() == dense_grads.keys()
    for path in dense_grads:
        np.testing.assert_allclose(
            np.asarray(fast_grads[path]),
            np.asarray(dense_grads[path]),
            rtol=1e-4,
            atol=1e-4,
            err_msg=".".join(str(part) for part in path),
        )


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
    assert all(
        isinstance(norm, RMSNorm)
        for norm in (
            block.tri_mul_out_post_norm,
            block.tri_mul_in_post_norm,
            block.pair_transition_post_norm,
            block.single_to_pair_post_norm,
            block.single_attention_post_norm,
            block.single_transition_post_norm,
        )
    )
    np.testing.assert_allclose(
        np.asarray(block.pair_transition.fc3.weight[...]),
        0.0,
    )
    np.testing.assert_allclose(
        np.asarray(block.tri_mul_out.p_out.weight[...]),
        0.0,
    )
    np.testing.assert_allclose(
        np.asarray(block.tri_mul_in.p_out.weight[...]),
        0.0,
    )
    np.testing.assert_allclose(
        np.asarray(block.single_attention.o.weight[...]),
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
    np.testing.assert_allclose(np.asarray(single_out), np.asarray(single))
    assert not np.allclose(np.asarray(pair_out), np.asarray(pair))


def test_jax_fastmixer_attention_can_disable_pair_bias():
    block = JaxPairMixerBlock(
        single_dim=8,
        pair_dim=6,
        num_heads=2,
        attention_mlp_multiple=2.0,
        norm_eps=1e-5,
        dropout=0.0,
        use_pair_bias=False,
        use_fastmixer=True,
        fastmixer_max_visible_tokens=4,
        rngs=nnx.Rngs(123),
    )
    block.single_attention.o.weight[...] = jnp.eye(8)
    single = jnp.arange(2 * 4 * 8, dtype=jnp.float32).reshape(2, 4, 8) / 17.0
    pair = jnp.arange(2 * 4 * 4 * 6, dtype=jnp.float32).reshape(2, 4, 4, 6) / 19.0
    mask = jnp.ones((2, 4), dtype=jnp.bool_)

    expected = block._fast_attention_pair_bias_compact(single, pair, mask)
    actual = block._fast_attention_pair_bias_compact(single, -pair, mask)

    assert block.single_attention.pair_norm is None
    assert block.single_attention.pair_bias is None
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected))


def test_jax_model_pair_bias_flag_reaches_encoder_only():
    model = PeakSetJEPAJax(
        **{
            **_small_mae_kwargs(),
            "pairmixer_use_pair_bias": False,
            "distogram_loss_weight": 0.25,
        }
    )

    assert not model.encoder.blocks[0].single_attention.use_pair_bias
    assert model.encoder.blocks[0].single_attention.pair_bias is None
    assert isinstance(model.masked_latent_predictor[0], JaxCrossAttentionBlock)


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
        checkpoint = load_torch_checkpoint(
            path,
            map_location="cpu",
            weights_only=True,
        )
        jax_model.load_torch_state_dict(checkpoint["model"])

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
    optimizer_config = {"learning_rate": 1e-3, "weight_decay": 0.0, "b2": 0.95}
    batch = _jax_batch(_real_pattern_batch("contiguous"))
    before = np.asarray(jax_model.jepa_mae_mz_head.weight[...])

    metrics = _run_canonical_train_step(jax_model, optimizer_config, batch)

    after = np.asarray(jax_model.jepa_mae_mz_head.weight[...])
    assert np.isfinite(np.asarray(metrics["loss"]))
    assert not np.allclose(before, after)


def test_jax_selective_checkpointing_uses_transformer_dot_policy():
    from spectra_learning.models.common_jax import activation_checkpoint_policy

    assert (
        activation_checkpoint_policy("selective")
        is jax.checkpoint_policies.dots_with_no_batch_dims_saveable
    )


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
    batch = _jax_batch(_real_pattern_batch("contiguous"))

    base_metrics = _run_canonical_train_step(base_model, optimizer_config, batch)
    checkpointed_metrics = _run_canonical_train_step(
        checkpointed_model,
        optimizer_config,
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
    batch = _jax_batch(_real_pattern_batch("random"))

    base_metrics = _run_canonical_train_step(base_model, optimizer_config, batch)
    checkpointed_metrics = _run_canonical_train_step(
        checkpointed_model,
        optimizer_config,
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
    kwargs = {
        **_small_mae_kwargs(),
        "autocast_dtype": "bf16",
        "distogram_loss_weight": 0.25,
    }
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


def test_jax_pure_optax_accumulated_train_step_matches_manual_accumulation():
    torch.manual_seed(22)
    kwargs = _small_mae_kwargs()
    torch_model = PeakSetJEPA(**kwargs).eval()
    manual_model = PeakSetJEPAJax(**kwargs)
    pure_model = PeakSetJEPAJax(**kwargs)
    manual_model.load_torch_state_dict(torch_model.state_dict())
    pure_model.load_torch_state_dict(torch_model.state_dict())
    optimizer_config = {"learning_rate": 1e-3, "weight_decay": 0.0, "b2": 0.95}
    batch_0 = _jax_batch(_real_pattern_batch("contiguous"))
    batch_1 = _jax_batch(_real_pattern_batch("random"))
    accumulated_batch = jax.tree.map(
        lambda lhs, rhs: jnp.stack([lhs, rhs]),
        batch_0,
        batch_1,
    )

    def loss_fn(model: PeakSetJEPAJax, batch: dict[str, jax.Array]):
        metrics = model(batch)
        return metrics["loss"], metrics

    grad_fn = nnx.value_and_grad(
        loss_fn,
        has_aux=True,
        argnums=nnx.DiffState(0, trainable_param_filter),
    )
    (_loss_0, _metrics_0), grads_0 = grad_fn(manual_model, batch_0)
    (_loss_1, _metrics_1), grads_1 = grad_fn(manual_model, batch_1)
    grads = nnx.as_pure(
        jax.tree.map(lambda lhs, rhs: (lhs + rhs) * 0.5, grads_0, grads_1)
    )
    _graphdef, manual_params, _static_state = nnx.split(
        manual_model,
        trainable_param_filter,
        ...,
    )
    manual_params = nnx.as_pure(manual_params)
    manual_optimizer = build_jax_optax_transform(optimizer_config)
    manual_opt_state = manual_optimizer.init(manual_params)
    updates, _manual_opt_state = manual_optimizer.update(
        grads,
        manual_opt_state,
        manual_params,
    )
    nnx.update(manual_model, optax.apply_updates(manual_params, updates))
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
    assert "distogram_loss" not in metrics
    np.testing.assert_allclose(
        np.asarray(pure_model.jepa_mae_mz_head.weight[...]),
        np.asarray(manual_model.jepa_mae_mz_head.weight[...]),
        rtol=1e-6,
        atol=2e-6,
    )


def test_jax_pure_optax_train_step_logs_update_stats():
    kwargs = _tiny_mae_kwargs()
    model = PeakSetJEPAJax(**kwargs)
    optimizer_config = {"learning_rate": 1e-3, "weight_decay": 0.0, "b2": 0.95}
    graphdef, trainable_params, static_state, opt_state, optimizer = (
        init_pure_optax_train_state(optimizer_config, model)
    )
    pure_train_step = make_pure_accumulated_train_step(
        graphdef,
        optimizer,
        sharded=False,
        log_update_stats=True,
    )
    batch = _jax_batch(_real_pattern_batch("random", num_peaks=3))
    accumulated_batch = jax.tree.map(lambda value: value[None], batch)

    _trainable_params, _opt_state, metrics = pure_train_step(
        trainable_params,
        static_state,
        opt_state,
        accumulated_batch,
    )

    for key in (
        "param_l2",
        "param_rms",
        "grad_l2",
        "grad_rms",
        "update_l2",
        "update_rms",
        "update_to_param_l2",
    ):
        assert key in metrics
        assert np.isfinite(np.asarray(metrics[key]))


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
    batch = _numpy_batch(
        _real_pattern_batch_size(
            "contiguous",
            jax.local_device_count(),
            num_peaks=int(kwargs["num_peaks"]),
        )
    )
    data_mesh = _jax_data_mesh_for_device_count(jax.device_count())
    accumulated_batch = numpy_batch_to_jax(
        jax.tree.map(lambda value: np.stack([value, value]), batch),
        data_mesh=data_mesh,
        batch_axis=1,
    )
    graphdef, trainable_params, static_state, opt_state, optimizer = (
        init_pure_optax_train_state(
            {"learning_rate": 1e-3, "weight_decay": 0.0, "b2": 0.95},
            jax_model,
        )
    )
    trainable_params = _replicate_tree_on_data_mesh(trainable_params, data_mesh)
    opt_state = _replicate_tree_on_data_mesh(opt_state, data_mesh)
    static_state = _replicate_tree_on_data_mesh(static_state, data_mesh)
    pure_train_step = make_pure_accumulated_train_step(
        graphdef,
        optimizer,
        sharded=True,
        data_mesh=data_mesh,
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
                distogram_loss_weight=0.25,
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
        checkpoint = load_torch_checkpoint(
            path,
            map_location="cpu",
            weights_only=True,
        )
        jax_model.load_torch_state_dict(checkpoint["model"])

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

    optimizer_config = {"learning_rate": 1e-3, "weight_decay": 0.5, "b2": 0.95}
    batch = _jax_batch(_real_pattern_batch("contiguous"))
    before_student = np.asarray(jax_model.target_projector.linear0.weight[...])
    before_teacher = np.asarray(jax_model.teacher_target_projector.linear0.weight[...])
    before_encoder_teacher = np.asarray(jax_model.teacher_encoder.cls_token[...])
    before_fourier = np.asarray(jax_model.encoder.embedder.mz_features.b[...])

    metrics = _run_canonical_train_step(jax_model, optimizer_config, batch)

    after_student = np.asarray(jax_model.target_projector.linear0.weight[...])
    after_teacher = np.asarray(jax_model.teacher_target_projector.linear0.weight[...])
    after_encoder_teacher = np.asarray(jax_model.teacher_encoder.cls_token[...])
    after_fourier = np.asarray(jax_model.encoder.embedder.mz_features.b[...])
    assert np.isfinite(np.asarray(metrics["loss"]))
    assert not np.allclose(before_student, after_student)
    np.testing.assert_array_equal(after_teacher, before_teacher)
    np.testing.assert_array_equal(after_encoder_teacher, before_encoder_teacher)
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
