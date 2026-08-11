import importlib.util
import inspect

import numpy as np
import pytest
import jax.numpy as jnp
from flax import nnx
from ml_collections import config_dict

from spectra_learning.config import load_config
from spectra_learning.models import fastmixer_capacity
from spectra_learning.models.fastmixer_capacity import (
    pairmixer_fast_mae_encoder_visible_tokens,
    pairmixer_fast_full_visible_tokens,
    pairmixer_fast_mae_stage_visible_tokens,
    pairmixer_fast_mae_visible_tokens,
    pairmixer_fast_stage_capacities,
)
from spectra_learning.models.model_jax import PeakSetJEPAJax
from spectra_learning.models.pairmixer_jax import (
    PairMixerBlock,
    SingleMixerBlock,
    _active_indices,
)
from spectra_learning.models.settings import PeakSetJEPASettings
from spectra_learning.models.transformer_jax import CrossAttentionBlock
from spectra_learning.probes.massspec.msg_probe_jax import (
    _full_visible_fastmixer_probe_model,
)


def get_config():
    return load_config("configs/100m_pairmixer_dense_adamw.py")


def test_current_config_auto_fastmixer_capacity_matches_mask_pipeline():
    cfg = get_config()

    assert "pairmixer_fast_max_visible_tokens" not in cfg
    assert pairmixer_fast_mae_encoder_visible_tokens(cfg) == 17
    assert pairmixer_fast_mae_visible_tokens(cfg) == 41
    assert pairmixer_fast_full_visible_tokens(cfg) == 48
    settings = PeakSetJEPASettings.from_config(cfg)
    assert settings.pairmixer_fast_encoder_max_visible_tokens == 17
    assert settings.pairmixer_fast_max_visible_tokens == 41


def test_fastmixer_dense_auto_capacity_matches_fastmixer():
    cfg = get_config()
    cfg.pairmixer_block_type = "FastMixer-Dense"

    assert PeakSetJEPASettings.from_config(cfg).pairmixer_fast_max_visible_tokens == 41


def test_1b_mae_schedule_uses_three_compact_shapes():
    cfg = load_config("configs/1b_pairmixer_dense_adamw.py")

    assert cfg.device_backend == "jax"
    assert cfg.dataloader_output_format == "numpy"
    assert cfg.dataloader_pin_memory is False
    assert cfg.dataloader_multiprocessing_context == "spawn"
    assert cfg.gems_hdf5_repo_id == (
        "novogaia/massive-v2-ms2-t095-l080-sharded-10gb"
    )
    assert cfg.gems_hdf5_revision == (
        "de80d280d319f0b9a8825956b13d8dc7d9ab1eb1"
    )
    assert cfg.learning_rate == pytest.approx(3e-4)
    assert cfg.weight_decay == pytest.approx(0.1)
    assert cfg.optimizer_state_dtype == "fp32"
    assert pairmixer_fast_mae_stage_visible_tokens(cfg) == (
        (17, 41),
        (27, 41),
        (36, 41),
    )
    assert pairmixer_fast_stage_capacities(cfg) == (
        (17, 41, 24),
        (27, 41, 14),
        (36, 41, 5),
    )
    assert cfg.gradient_accumulation_steps == 8
    assert tuple(cfg.gradient_accumulation_steps_schedule) == (8, 16, 16)
    assert cfg.activation_checkpoint_mode == "none"
    assert cfg.max_duration_hours == 95.0
    assert cfg.mae_loss_weight == 1.0
    assert cfg.mae_intensity_loss_weight == 0.0
    assert cfg.distogram_loss_weight == 0.0
    settings = PeakSetJEPASettings.from_config(cfg)
    assert settings.pairmixer_fast_encoder_max_visible_tokens == 17
    assert settings.pairmixer_fast_max_visible_tokens == 41


def test_1b_mae_schedule_without_cls_uses_peak_only_compact_shapes():
    cfg = load_config(
        "configs/1b_pairmixer_dense_adamw.py",
        {"encoder_use_cls_token": False},
    )

    assert pairmixer_fast_full_visible_tokens(cfg) == 47
    assert pairmixer_fast_mae_stage_visible_tokens(cfg) == (
        (16, 40),
        (26, 40),
        (35, 40),
    )
    assert pairmixer_fast_stage_capacities(cfg) == (
        (16, 40, 24),
        (26, 40, 14),
        (35, 40, 5),
    )
    settings = PeakSetJEPASettings.from_config(cfg)
    assert not settings.encoder_use_cls_token
    assert settings.pairmixer_fast_encoder_max_visible_tokens == 16
    assert settings.pairmixer_fast_max_visible_tokens == 40


def test_encoder_and_predictor_blocks_keep_separate_fastmixer_capacities():
    model = PeakSetJEPAJax(
        training_mode="mae",
        model_dim=8,
        encoder_num_layers=1,
        encoder_num_heads=1,
        feature_mlp_hidden_dim=8,
        encoder_fourier_num_freqs=1,
        pairmixer_fourier_num_freqs=1,
        pairmixer_pair_dim=8,
        pairmixer_pair_feature_hidden_dim=8,
        pairmixer_block_type="fastmixer-dense",
        pairmixer_fast_encoder_max_visible_tokens=17,
        pairmixer_fast_max_visible_tokens=41,
        predictor_target_max_tokens=24,
        masked_latent_predictor_num_layers=1,
        masked_latent_predictor_num_heads=1,
        num_peaks=47,
        jepa_num_target_blocks=1,
        distogram_loss_weight=0.0,
        target_projector_dim=-1,
    )

    encoder_block = model.encoder.blocks[0]
    predictor_block = model.masked_latent_predictor[0]
    assert isinstance(encoder_block, SingleMixerBlock)
    assert model.encoder.pair_embedder is None
    assert model.encoder.pairmixer_fast_max_visible_tokens == 17
    assert isinstance(predictor_block, CrossAttentionBlock)
    assert model.pairmixer_fast_target_max_visible_tokens == 24

    model.set_fastmixer_capacities(27, 41, 14)

    assert model.encoder.pairmixer_fast_max_visible_tokens == 27
    assert model.pairmixer_fast_target_max_visible_tokens == 14


def test_direct_fastmixer_model_defaults_to_full_single_stream_capacity():
    model = PeakSetJEPAJax(
        training_mode="mae",
        model_dim=8,
        encoder_num_layers=1,
        encoder_num_heads=1,
        feature_mlp_hidden_dim=8,
        encoder_fourier_num_freqs=1,
        pairmixer_fourier_num_freqs=1,
        pairmixer_pair_dim=8,
        pairmixer_pair_feature_hidden_dim=8,
        pairmixer_block_type="fastmixer",
        masked_latent_predictor_num_layers=1,
        masked_latent_predictor_num_heads=1,
        num_peaks=7,
        jepa_num_target_blocks=1,
        distogram_loss_weight=0.0,
        target_projector_dim=-1,
    )

    assert model.pairmixer_fast_encoder_max_visible_tokens == 8
    assert model.encoder.pairmixer_fast_max_visible_tokens == 8
    assert isinstance(model.encoder.blocks[0], SingleMixerBlock)


def test_pairmixer_has_one_projection_path():
    cfg = load_config("configs/1b_pairmixer_dense_adamw.py")

    assert importlib.util.find_spec("spectra_learning.models.pairmixer_pallas") is None
    assert "projection_kernel" not in inspect.signature(PairMixerBlock).parameters
    assert not any(
        "projection_kernel" in name
        for name in PeakSetJEPASettings.__dataclass_fields__
    )
    assert not hasattr(
        fastmixer_capacity,
        "pairmixer_stage_projection_kernels",
    )
    assert not hasattr(PeakSetJEPAJax, "set_pairmixer_projection_kernels")
    assert not any("projection_kernel" in key for key in cfg)


@pytest.mark.parametrize(
    "removed_key",
    (
        "pairmixer_encoder_projection_kernel",
        "pairmixer_predictor_projection_kernel",
        "pairmixer_encoder_projection_kernel_schedule",
        "pairmixer_predictor_projection_kernel_schedule",
    ),
)
def test_pairmixer_rejects_removed_projection_settings(removed_key):
    cfg = config_dict.ConfigDict({removed_key: "pallas"})

    with pytest.raises(ValueError, match=f"{removed_key} has been removed"):
        PeakSetJEPASettings.from_config(cfg)


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

    assert pairmixer_fast_mae_visible_tokens(cfg) == 48
    assert PeakSetJEPASettings.from_config(cfg).pairmixer_fast_max_visible_tokens == 48


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
        "encoder_fourier_num_freqs": 1,
        "pairmixer_fourier_num_freqs": 1,
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
    assert probe_model.encoder.pairmixer_fast_max_visible_tokens == 5
    assert isinstance(probe_model.encoder.blocks[0], SingleMixerBlock)
    assert probe_model.encoder.pair_embedder is None
    assert isinstance(probe_model.masked_latent_predictor[0], CrossAttentionBlock)
    np.testing.assert_allclose(
        np.asarray(
            probe_model.encoder.blocks[0].single_attention.qkv.weight[...]
        ),
        np.asarray(model.encoder.blocks[0].single_attention.qkv.weight[...]),
    )

    peak_mz = jnp.linspace(0.1, 0.4, 4, dtype=jnp.float32)[None, :]
    peak_intensity = jnp.ones((1, 4), dtype=jnp.float32)
    valid_mask = jnp.ones((1, 4), dtype=jnp.bool_)
    single = probe_model.encoder(
        peak_mz,
        peak_intensity,
        valid_mask=valid_mask,
        visible_mask=valid_mask,
    )

    assert single.shape == (1, 5, 4)
