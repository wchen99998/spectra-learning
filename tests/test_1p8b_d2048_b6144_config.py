"""The 1.78B / micro-batch 96 config, and what must not drift in it."""

import jax
from flax import nnx

from spectra_learning.config import load_config
from spectra_learning.models.factory_jax import build_model_from_config


CONFIG_PATH = (
    "configs/1p8b_singlemixer_dense_adam_metadata_adaln_"
    "multiscale_n128_b6144_666k_d2048.py"
)
PARENT_PATH = (
    "configs/1b_singlemixer_dense_adamw_metadata_adaln_"
    "multiscale_n128_b2048_500k_hd256.py"
)


def test_parameter_count_is_the_benchmarked_model() -> None:
    config = load_config(CONFIG_PATH)
    model = nnx.eval_shape(lambda: build_model_from_config(config))
    params = sum(
        int(v.size) for v in jax.tree.leaves(nnx.state(model, nnx.Param))
    )
    # 1.7816B is the size the v6e-16 numbers in the docstring were measured at.
    assert abs(params - 1_781_600_000) < 5_000_000
    # head_dim 256 matches the v6e MXU and is the reason num_heads moved with
    # model_dim rather than staying at 6.
    assert config.model_dim // config.encoder_num_heads == 256
    assert config.predictor_dim == config.model_dim


def test_step_shape_is_microbatch_96_on_64_chips() -> None:
    config = load_config(CONFIG_PATH)
    assert config.gradient_accumulation_steps == 1
    assert int(config.jax_mesh_devices) == 64
    assert config.batch_size == 6144
    assert config.batch_size // int(config.jax_mesh_devices) == 96


def test_learning_rate_and_token_budget() -> None:
    config = load_config(CONFIG_PATH)
    parent = load_config(PARENT_PATH)

    assert config.learning_rate == 6e-4
    assert config.min_learning_rate == config.learning_rate / 10
    assert config.learning_rate == 2 * parent.learning_rate
    assert config.batch_size == 3 * parent.batch_size
    # Warmup is intentionally left in steps, so it is 3x longer in tokens.
    assert config.warmup_steps == parent.warmup_steps

    # Same token budget as the 2M-step 1B run, within 0.5%.
    tokens = config.training_max_steps * config.batch_size
    parent_tokens = parent.training_max_steps * parent.batch_size
    assert abs(tokens / parent_tokens - 1.0) < 0.005


def test_optimizer_state_is_bf16_and_unsharded() -> None:
    config = load_config(CONFIG_PATH)
    # fp32 moments OOM at this size; bf16 also measured faster.
    assert config.optimizer_state_dtype == "bf16"
    assert config.optimizer == "adam"
    # Micro-batch 96 fits without ZeRO-1, which costs 8.9% on the step.
    assert not config.get("jax_zero1_optimizer", False)
    assert not config.jax_shard_optimizer


def test_run_name_reflects_the_changed_settings() -> None:
    name = load_config(CONFIG_PATH).run_name_suffix
    for token in (
        "-1p8b-",
        "-d2048-",
        "-h8-",
        "-feat4096-",
        "-fmlp4096-",
        "-xattnrope2048-",
        "-bs6144-",
        "-ga1-",
        "-bf16state-",
        "-lr6e-4-",
    ):
        assert token in name, token
    assert name.endswith("-666k")
    for stale in ("-d1536-", "-h6-", "-bs2048-", "-ga2-", "-fp32state-", "-lr3e-4-"):
        assert stale not in name, stale
