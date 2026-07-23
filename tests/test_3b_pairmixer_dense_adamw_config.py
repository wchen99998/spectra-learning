import math

import jax
import pytest
from flax import nnx

from spectra_learning.config import load_config
from spectra_learning.models.factory_jax import build_model_from_config


def test_3b_pairmixer_shape_is_tpu7x_mxu_aligned() -> None:
    config = load_config("configs/3b_pairmixer_dense_adamw.py")

    assert config.model_dim == 2560
    assert config.encoder_num_layers == 30
    assert config.encoder_num_heads == 20
    assert config.model_dim // config.encoder_num_heads == 128
    assert config.pairmixer_pair_dim == 768
    assert config.masked_latent_predictor_num_layers == 3
    assert config.masked_latent_predictor_num_heads == 20
    assert config.gradient_accumulation_steps == 16
    assert tuple(config.gradient_accumulation_steps_schedule) == (16, 32, 32)
    assert config.jax_mesh_devices == "16"
    assert config.max_duration_hours == pytest.approx(47.0)
    assert all(
        dimension % 256 == 0
        for dimension in (
            config.model_dim,
            config.feature_mlp_hidden_dim,
            config.encoder_fourier_mlp_hidden_dim,
            config.pairmixer_pair_dim,
            config.pairmixer_pair_feature_hidden_dim,
            config.predictor_dim,
        )
    )


def test_3b_pairmixer_has_three_billion_parameter_encoder_plus_ten_percent() -> None:
    config = load_config("configs/3b_pairmixer_dense_adamw.py")
    state = jax.eval_shape(
        lambda: nnx.state(build_model_from_config(config), nnx.Param)
    )
    counts: dict[str, int] = {}
    for path, value in nnx.to_flat_state(state):
        counts[str(path[0])] = counts.get(str(path[0]), 0) + math.prod(value.shape)

    encoder_params = counts["encoder"]
    predictor_side_params = sum(counts.values()) - encoder_params

    assert encoder_params == 3_002_501_472
    assert predictor_side_params == 306_264_746
    assert sum(counts.values()) == 3_308_766_218
    assert predictor_side_params / encoder_params == pytest.approx(0.1020032)
