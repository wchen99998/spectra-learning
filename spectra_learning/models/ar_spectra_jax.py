from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from ml_collections import config_dict

from spectra_learning.data.ar_spectra import (
    SPECTRA_AR_TARGET_KINDS,
    SpectraARTokenizer,
)
from spectra_learning.models.common_jax import (
    Array,
    Embedding,
    LayerNorm,
    Linear,
    gelu,
    resolve_jax_compute_dtype,
    scaled_dot_product_attention,
)


def _config_get(config: config_dict.ConfigDict, key: str, default: Any) -> Any:
    return config.get(key, default)


@dataclass(frozen=True)
class SpectraARTransformerJaxConfig:
    vocab_size: int
    num_token_kinds: int
    max_sequence_length: int
    pad_token_id: int
    model_dim: int = 1024
    num_layers: int = 8
    num_heads: int = 8
    mlp_multiple: float = 4.0
    rope_base: float = 10_000.0
    attention_kernel: str = "xla"
    splash_block_size: int = 128
    compute_dtype: object = jnp.bfloat16

    @classmethod
    def from_config(
        cls,
        config: config_dict.ConfigDict,
        tokenizer: SpectraARTokenizer,
    ) -> "SpectraARTransformerJaxConfig":
        return cls(
            vocab_size=tokenizer.vocab_size,
            num_token_kinds=tokenizer.num_token_kinds,
            max_sequence_length=tokenizer.sequence_length,
            pad_token_id=tokenizer.pad_token_id,
            model_dim=int(_config_get(config, "ar_model_dim", 1024)),
            num_layers=int(_config_get(config, "ar_num_layers", 8)),
            num_heads=int(_config_get(config, "ar_num_heads", 8)),
            mlp_multiple=float(_config_get(config, "ar_mlp_multiple", 4.0)),
            rope_base=float(_config_get(config, "ar_rope_base", 10_000.0)),
            attention_kernel=str(_config_get(config, "ar_attention_kernel", "xla")),
            splash_block_size=int(_config_get(config, "ar_splash_block_size", 128)),
            compute_dtype=resolve_jax_compute_dtype(
                str(_config_get(config, "autocast_dtype", "bf16"))
            ),
        )


class RotaryEmbeddingJax(nnx.Module):
    def __init__(
        self,
        head_dim: int,
        max_sequence_length: int,
        *,
        base: float,
    ) -> None:
        assert head_dim % 2 == 0
        inv_freq = 1.0 / (
            base
            ** (
                jnp.arange(0, head_dim, 2, dtype=jnp.float32)
                / float(head_dim)
            )
        )
        positions = jnp.arange(max_sequence_length, dtype=jnp.float32)
        freqs = positions[:, None] * inv_freq[None, :]
        self.cos = jnp.cos(freqs)
        self.sin = jnp.sin(freqs)

    def __call__(self, tensor: Array) -> Array:
        seq_len = tensor.shape[1]
        cos = self.cos[:seq_len].astype(tensor.dtype)[None, :, None, :]
        sin = self.sin[:seq_len].astype(tensor.dtype)[None, :, None, :]
        even = tensor[..., 0::2]
        odd = tensor[..., 1::2]
        return jnp.stack((even * cos - odd * sin, even * sin + odd * cos), axis=-1).reshape(
            tensor.shape
        )


class SpectraARCausalSelfAttentionJax(nnx.Module):
    def __init__(
        self,
        config: SpectraARTransformerJaxConfig,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        assert config.model_dim % config.num_heads == 0
        self.num_heads = config.num_heads
        self.head_dim = config.model_dim // config.num_heads
        assert self.head_dim % 2 == 0
        self.attention_kernel = config.attention_kernel
        self.splash_block_size = config.splash_block_size
        self.qkv = Linear(
            config.model_dim,
            3 * config.model_dim,
            bias=False,
            compute_dtype=config.compute_dtype,
            rngs=rngs,
        )
        self.out_proj = Linear(
            config.model_dim,
            config.model_dim,
            bias=False,
            compute_dtype=config.compute_dtype,
            rngs=rngs,
        )
        self.rope = RotaryEmbeddingJax(
            self.head_dim,
            config.max_sequence_length,
            base=config.rope_base,
        )

    def __call__(self, hidden: Array) -> Array:
        batch_size, seq_len, model_dim = hidden.shape
        qkv = self.qkv(hidden).reshape(
            batch_size,
            seq_len,
            3,
            self.num_heads,
            self.head_dim,
        )
        query, key, value = jnp.moveaxis(qkv, 2, 0)
        query = self.rope(query)
        key = self.rope(key)
        if self.attention_kernel == "splash":
            attended = splash_causal_attention(
                query,
                key,
                value,
                block_size=self.splash_block_size,
            )
        else:
            attended = jnp.swapaxes(
                scaled_dot_product_attention(
                    jnp.swapaxes(query, 1, 2),
                    jnp.swapaxes(key, 1, 2),
                    jnp.swapaxes(value, 1, 2),
                    is_causal=True,
                    implementation=self.attention_kernel,
                ),
                1,
                2,
            )
        attended = attended.reshape(batch_size, seq_len, model_dim)
        return self.out_proj(attended)


class SpectraARTransformerBlockJax(nnx.Module):
    def __init__(
        self,
        config: SpectraARTransformerJaxConfig,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        hidden_dim = int(config.model_dim * config.mlp_multiple)
        self.attention_norm = LayerNorm(config.model_dim)
        self.attention = SpectraARCausalSelfAttentionJax(config, rngs=rngs)
        self.ffn_norm = LayerNorm(config.model_dim)
        self.ffn0 = Linear(
            config.model_dim,
            hidden_dim,
            compute_dtype=config.compute_dtype,
            rngs=rngs,
        )
        self.ffn1 = Linear(
            hidden_dim,
            config.model_dim,
            compute_dtype=config.compute_dtype,
            rngs=rngs,
        )

    def __call__(self, hidden: Array) -> Array:
        hidden = hidden + self.attention(self.attention_norm(hidden))
        hidden = hidden + self.ffn1(gelu(self.ffn0(self.ffn_norm(hidden))))
        return hidden


class SpectraARTransformerJax(nnx.Module):
    def __init__(
        self,
        config: SpectraARTransformerJaxConfig,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        rngs = nnx.Rngs(0) if rngs is None else rngs
        self.config = config
        self.pad_token_id = config.pad_token_id
        self.compute_dtype = config.compute_dtype
        self.use_ema_teacher = False
        self.token_embedding = Embedding(config.vocab_size, config.model_dim)
        self.kind_embedding = Embedding(config.num_token_kinds, config.model_dim)
        self.token_embedding.weight[...] = (
            rngs.params.normal(
                (config.vocab_size, config.model_dim),
                dtype=jnp.float32,
            )
            * 0.02
        )
        self.kind_embedding.weight[...] = (
            rngs.params.normal(
                (config.num_token_kinds, config.model_dim),
                dtype=jnp.float32,
            )
            * 0.02
        )
        self.blocks = nnx.List(
            [
                SpectraARTransformerBlockJax(config, rngs=rngs)
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = LayerNorm(config.model_dim)

    def hidden_states(
        self,
        input_ids: Array,
        token_kinds: Array,
    ) -> Array:
        input_ids = input_ids.astype(jnp.int32)
        token_kinds = token_kinds.astype(jnp.int32)
        hidden = (
            self.token_embedding(input_ids).astype(self.compute_dtype)
            + self.kind_embedding(token_kinds).astype(self.compute_dtype)
        )
        for block in self.blocks:
            hidden = block(hidden)
        return self.final_norm(hidden)

    def encode_batch(self, batch: dict[str, Array]) -> Array:
        return self.hidden_states(
            batch["input_token_ids"],
            batch["input_token_kinds"],
        )

    def __call__(self, batch: dict[str, Array]) -> dict[str, Array]:
        hidden = self.encode_batch(batch)
        logits = jnp.matmul(
            hidden.astype(self.compute_dtype),
            jnp.swapaxes(
                self.token_embedding.weight[...].astype(self.compute_dtype),
                -1,
                -2,
            ),
        )
        if "target_token_ids" in batch:
            return self._loss_metrics(logits, batch)
        return {"logits": logits}

    def _loss_metrics(
        self,
        logits: Array,
        batch: dict[str, Array],
    ) -> dict[str, Array]:
        labels = batch["target_token_ids"].astype(jnp.int32)
        target_kinds = batch["target_token_kinds"].astype(jnp.int32)
        loss_mask = batch["target_loss_mask"].astype(jnp.bool_)
        log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
        token_loss = -jnp.take_along_axis(
            log_probs,
            labels[..., None],
            axis=-1,
        )[..., 0]
        denominator = jnp.maximum(jnp.sum(loss_mask), 1)
        predictions = jnp.argmax(logits, axis=-1).astype(labels.dtype)
        correct = predictions == labels
        metrics = {
            "loss": jnp.sum(token_loss * loss_mask) / denominator,
            "token_accuracy": jnp.sum(correct * loss_mask) / denominator,
            "target_tokens": denominator.astype(jnp.float32),
        }
        for kind in SPECTRA_AR_TARGET_KINDS:
            kind_mask = loss_mask & (target_kinds == int(kind))
            kind_count = jnp.sum(kind_mask)
            kind_denominator = jnp.maximum(kind_count, 1)
            kind_name = kind.name.lower()
            metrics[f"loss/{kind_name}"] = (
                jnp.sum(token_loss * kind_mask) / kind_denominator
            )
            metrics[f"token_accuracy/{kind_name}"] = (
                jnp.sum(correct * kind_mask) / kind_denominator
            )
            metrics[f"target_tokens/{kind_name}"] = kind_count.astype(jnp.float32)
        return metrics


def build_spectra_ar_model_jax_from_config(
    config: config_dict.ConfigDict,
    tokenizer: SpectraARTokenizer,
) -> SpectraARTransformerJax:
    seed = int(_config_get(config, "seed", 0))
    return SpectraARTransformerJax(
        SpectraARTransformerJaxConfig.from_config(config, tokenizer),
        rngs=nnx.Rngs(seed),
    )


def splash_causal_attention(
    query: Array,
    key: Array,
    value: Array,
    *,
    block_size: int,
) -> Array:
    batch_size, seq_len, num_heads, _head_dim = query.shape
    padded_seq_len = _ceil_multiple(seq_len, block_size)
    query = _pad_sequence_axis(query, padded_seq_len)
    key = _pad_sequence_axis(key, padded_seq_len)
    value = _pad_sequence_axis(value, padded_seq_len)
    kernel = _splash_causal_kernel(num_heads, padded_seq_len, block_size)

    def apply_one(query_row: Array, key_row: Array, value_row: Array) -> Array:
        out = kernel(
            jnp.swapaxes(query_row, 0, 1),
            jnp.swapaxes(key_row, 0, 1),
            jnp.swapaxes(value_row, 0, 1),
        )
        return jnp.swapaxes(out, 0, 1)

    out = jax.vmap(apply_one)(query, key, value)
    return out[:, :seq_len, :, :]


@cache
def _splash_causal_kernel(num_heads: int, seq_len: int, block_size: int):
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_kernel as splash,
    )
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_mask as mask_lib,
    )

    mask = mask_lib.MultiHeadMask(
        tuple(mask_lib.CausalMask((seq_len, seq_len)) for _ in range(num_heads))
    )
    block_sizes = splash.BlockSizes(
        block_q=block_size,
        block_kv=block_size,
        block_kv_compute=block_size,
        block_q_dkv=block_size,
        block_kv_dkv=block_size,
        block_kv_dkv_compute=block_size,
        block_q_dq=block_size,
        block_kv_dq=block_size,
    )
    return splash.make_splash_mha_single_device(mask, block_sizes=block_sizes)


def _ceil_multiple(value: int, multiple: int) -> int:
    return multiple * int(math.ceil(value / multiple))


def _pad_sequence_axis(value: Array, padded_seq_len: int) -> Array:
    pad_len = padded_seq_len - value.shape[1]
    return jnp.pad(value, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
