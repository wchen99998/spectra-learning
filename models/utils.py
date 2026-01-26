"""Model utils."""

from typing import cast

import jax.numpy as jnp
from flax import nnx
import ml_collections

from models import bert


def get_model(config: ml_collections.ConfigDict, rngs: nnx.Rngs) -> nnx.Module:
    """Create a model factory compatible with flax.nnx."""
    if config.model_type != "bert":
        raise NotImplementedError(f"Unsupported model_type: {config.model_type}")

    return bert.BERT(
        rngs=rngs,
        vocab_size=cast(int, config.vocab_size),
        max_length=cast(int, config.max_length),
        model_dim=cast(int, config.get("model_dim", 256)),
        num_layers=cast(int, config.get("num_layers", 6)),
        num_heads=cast(int, config.get("num_heads", 8)),
        num_kv_heads=cast(int | None, config.get("num_kv_heads", None)),
        attention_mlp_multiple=cast(float, config.get("attention_mlp_multiple", 4.0)),
        num_segments=cast(int, config.get("num_segments", 2)),
        mask_ratio=cast(float, config.get("mask_ratio", 0.15)),
        mask_token_id=cast(int, config.get("mask_token_id", 103)),
        pad_token_id=cast(int, config.get("pad_token_id", 0)),
        cls_token_id=cast(int, config.get("cls_token_id", 101)),
        sep_token_id=cast(int, config.get("sep_token_id", 102)),
        dtype=cast(jnp.dtype, config.get("dtype", jnp.float32)),
        param_dtype=cast(jnp.dtype, config.get("param_dtype", jnp.float32)),
    )
