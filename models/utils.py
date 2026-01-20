"""Model utils."""

from typing import cast

import jax.numpy as jnp
from flax import nnx
import ml_collections

from models import mae


def get_model(config: ml_collections.ConfigDict, rngs: nnx.Rngs) -> nnx.Module:
    """Create a model factory compatible with flax.nnx."""
    if config.model_type != "mae":
        raise NotImplementedError(f"Unsupported model_type: {config.model_type}")

    spectrum_length_val = config.get("padded_length", None)
    if spectrum_length_val is None:
        spectrum_length_val = config.get("spectrum_length")
    if spectrum_length_val is None:
        data_shape_val = config.get("data_shape")
        if data_shape_val is None:
            raise ValueError("Config must provide spectrum_length or data_shape.")
        data_shape_tuple = cast(tuple[int, ...], data_shape_val)
        if len(data_shape_tuple) < 1:
            raise ValueError("data_shape must contain spectrum length.")
        spectrum_length = int(data_shape_tuple[0])
    else:
        spectrum_length = cast(int, spectrum_length_val)

    num_heads_val = config.get("num_heads", config.get("attention_heads", None))
    num_heads = cast(int | None, num_heads_val) if num_heads_val is not None else None

    num_kv_heads_val = config.get("num_kv_heads", config.get("attention_kv_heads", None))
    num_kv_heads = cast(int | None, num_kv_heads_val) if num_kv_heads_val is not None else None

    num_transformer_blocks_val = config.get("num_transformer_blocks", (4, 2))
    if isinstance(num_transformer_blocks_val, (tuple, list)):
        if len(num_transformer_blocks_val) != 2:
            raise ValueError(
                "num_transformer_blocks must be an int or a (encoder, decoder) pair"
            )
        num_transformer_blocks = (
            int(num_transformer_blocks_val[0]),
            int(num_transformer_blocks_val[1]),
        )
    else:
        num_blocks = int(num_transformer_blocks_val)
        num_transformer_blocks = (num_blocks, num_blocks)

    decoder_dim_val = config.get("decoder_dim", None)
    decoder_dim = cast(int | None, decoder_dim_val) if decoder_dim_val is not None else None
    peak_mlp_hidden_dim_val = config.get("peak_mlp_hidden_dim", None)
    peak_mlp_hidden_dim = (
        cast(int, peak_mlp_hidden_dim_val)
        if peak_mlp_hidden_dim_val is not None
        else None
    )

    return mae.MAE(
        rngs=rngs,
        spectrum_length=spectrum_length,
        mask_ratio=cast(float, config.get("mask_ratio", 0.75)),
        encoder_dim=cast(int, config.get("encoder_dim", 256)),
        decoder_dim=decoder_dim,
        num_transformer_blocks=num_transformer_blocks,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        attention_mlp_multiple=cast(float, config.get("attention_mlp_multiple", 4.0)),
        mask_top_k_peaks=cast(int | None, config.get("mask_top_k_peaks", None)),
        min_visible_peaks=cast(int, config.get("min_visible_peaks", 12)),
        always_keep_top_n=cast(int, config.get("always_keep_top_n", 3)),
        mz_window_prob=cast(float, config.get("mz_window_prob", 0.5)),
        mz_window_width=cast(tuple[float, float], config.get("mz_window_width", (50.0, 200.0))),
        intensity_log_jitter=cast(float, config.get("intensity_log_jitter", 0.0)),
        precursor_mask_prob=cast(float, config.get("precursor_mask_prob", 0.5)),
        rt_mask_prob=cast(float, config.get("rt_mask_prob", 0.5)),
        mz_min=cast(float, config.get("mz_min", 0.0)),
        mz_max=cast(float, config.get("mz_max", 1500.0)),
        num_fourier_features=cast(int, config.get("num_fourier_features", 32)),
        fourier_max_freq=cast(float, config.get("fourier_max_freq", 16.0)),
        peak_mlp_hidden_dim=peak_mlp_hidden_dim,
        mz_loss_weight=cast(float, config.get("mz_loss_weight", 1.0)),
        intensity_loss_weight=cast(float, config.get("intensity_loss_weight", 1.0)),
        repr_loss_weight=cast(float, config.get("repr_loss_weight", 1.0)),
        dtype=cast(jnp.dtype, config.get("dtype", jnp.float32)),
        param_dtype=cast(jnp.dtype, config.get("param_dtype", jnp.float32)),
        recon_loss_weight=cast(float, config.get("recon_loss_weight", 1.0)),
        aux_loss_weight=cast(float, config.get("aux_loss_weight", 1.0)),
        use_vicreg=cast(bool, config.get("use_vicreg", True)),
    )
