import numpy as np
import pytest
import jax.numpy as jnp
from flax import nnx
from ml_collections import config_dict

from configs.medium_pairmixer_100m_20m_mae_beta_isoflops_muon import get_config
from spectra_learning.models.fastmixer_capacity import (
    pairmixer_fast_full_visible_tokens,
    pairmixer_fast_mae_visible_tokens,
)
from spectra_learning.models.model_jax import PeakSetJEPAJax
from spectra_learning.models.pairmixer_jax import _active_indices
from spectra_learning.models.settings import PeakSetJEPASettings
from spectra_learning.probes.massspec.msg_probe_jax import (
    _full_visible_fastmixer_probe_model,
)


def test_muon_config_auto_fastmixer_capacity_matches_mask_pipeline():
    cfg = get_config()

    assert "pairmixer_fast_max_visible_tokens" not in cfg
    assert pairmixer_fast_mae_visible_tokens(cfg) == 20
    assert pairmixer_fast_full_visible_tokens(cfg) == 32
    assert PeakSetJEPASettings.from_config(cfg).pairmixer_fast_max_visible_tokens == 20


def test_fastmixer_dense_auto_capacity_matches_fastmixer():
    cfg = get_config()
    cfg.pairmixer_block_type = "FastMixer-Dense"

    assert PeakSetJEPASettings.from_config(cfg).pairmixer_fast_max_visible_tokens == 20


def test_fastmixer_capacity_rejects_explicit_config_cap():
    cfg = get_config()
    cfg.pairmixer_fast_max_visible_tokens = 19

    with pytest.raises(ValueError, match="derived from the data config"):
        PeakSetJEPASettings.from_config(cfg)


def test_fastmixer_capacity_rejects_auto_alias():
    cfg = get_config()
    cfg.pairmixer_fast_max_visible_tokens = "auto"

    with pytest.raises(ValueError, match="derived from the data config"):
        PeakSetJEPASettings.from_config(cfg)


def test_intensity_aware_auto_capacity_falls_back_to_full_visible():
    cfg = get_config()
    cfg.jepa_mask_strategy = ["intensity_aware"]

    assert pairmixer_fast_mae_visible_tokens(cfg) == 32
    assert PeakSetJEPASettings.from_config(cfg).pairmixer_fast_max_visible_tokens == 32


def test_active_indices_clamps_over_cap_to_sequence_length():
    idx, compact_mask = _active_indices(
        jnp.array([[True, False, True]], dtype=jnp.bool_),
        5,
    )

    assert idx.shape == (1, 3)
    assert compact_mask.shape == (1, 3)


def test_jax_msg_probe_uses_full_visible_fastmixer_clone():
    kwargs = {
        "training_mode": "mae",
        "model_dim": 4,
        "encoder_num_layers": 1,
        "encoder_num_heads": 1,
        "attention_mlp_multiple": 1.0,
        "feature_mlp_hidden_dim": 4,
        "encoder_use_fourier_features": False,
        "pairmixer_use_fourier_features": False,
        "pairmixer_pair_dim": 4,
        "pairmixer_pair_feature_hidden_dim": 4,
        "masked_latent_predictor_num_layers": 1,
        "masked_latent_predictor_num_heads": 1,
        "num_peaks": 4,
        "jepa_num_target_blocks": 1,
        "distogram_loss_weight": 0.0,
        "predictor_dropout": 0.0,
        "target_projector_dim": -1,
        "pairmixer_block_type": "FastMixer",
        "pairmixer_fast_max_visible_tokens": 3,
    }
    model = PeakSetJEPAJax(**kwargs, rngs=nnx.Rngs(0))
    config_values = {key: value for key, value in kwargs.items()}
    del config_values["pairmixer_fast_max_visible_tokens"]
    config = config_dict.ConfigDict({**config_values, "seed": 0})

    probe_model = _full_visible_fastmixer_probe_model(config, model)

    assert probe_model is not model
    assert probe_model.pairmixer_fast_max_visible_tokens == 5
    assert probe_model.encoder.blocks[0].fastmixer_max_visible_tokens == 5
    assert probe_model.masked_latent_predictor[0].fastmixer_max_visible_tokens == 5
    np.testing.assert_allclose(
        np.asarray(probe_model.encoder.blocks[0].tri_mul_out.p_in.weight[...]),
        np.asarray(model.encoder.blocks[0].tri_mul_out.p_in.weight[...]),
    )

    peak_mz = jnp.linspace(0.1, 0.4, 4, dtype=jnp.float32)[None, :]
    peak_intensity = jnp.ones((1, 4), dtype=jnp.float32)
    valid_mask = jnp.ones((1, 4), dtype=jnp.bool_)
    single, pair = probe_model.encoder.forward_with_pair(
        peak_mz,
        peak_intensity,
        valid_mask=valid_mask,
        visible_mask=valid_mask,
    )

    assert single.shape == (1, 5, 4)
    assert pair.shape == (1, 5, 5, 4)
