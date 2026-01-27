"""BERT-style masked language model for discrete tokens."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import optax
from clu import metrics as clu_metrics
from flax import nnx

from networks import transformer


def _resolve_attention_heads(
    dim: int,
    num_heads: int,
    num_kv_heads: int | None,
) -> tuple[int, int]:
    heads = int(num_heads)
    kv_heads = heads if num_kv_heads is None else int(num_kv_heads)
    return heads, kv_heads


def _build_transformer_blocks(
    rngs: nnx.Rngs,
    *,
    dim: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int | None,
    attention_mlp_multiple: float,
    use_rotary_embeddings: bool,
    dtype: jnp.dtype,
    param_dtype: jnp.dtype,
) -> nnx.List[transformer.TransformerBlock]:
    heads, kv_heads = _resolve_attention_heads(dim, num_heads, num_kv_heads)
    hidden_dim = int(math.ceil(dim * attention_mlp_multiple))
    blocks = []
    for _ in range(num_layers):
        blocks.append(
            transformer.TransformerBlock(
                rngs=rngs,
                dim=dim,
                n_heads=heads,
                n_kv_heads=kv_heads,
                causal=False,
                dtype=dtype,
                param_dtype=param_dtype,
                norm_eps=1e-5,
                mlp_type="swish",
                multiple_of=4,
                hidden_dim=hidden_dim,
                w_init_scale=1.0,
                use_rotary_embeddings=use_rotary_embeddings,
            )
        )
    return nnx.List(blocks)


class TransformerStack(nnx.Module):
    """Stack of transformer blocks with rotary positional encoding."""

    def __init__(
        self,
        rngs: nnx.Rngs,
        *,
        dim: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int | None,
        attention_mlp_multiple: float,
        rope_theta: float,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
    ):
        heads, kv_heads = _resolve_attention_heads(dim, num_heads, num_kv_heads)
        self.dim = int(dim)
        self.num_heads = int(heads)
        self.rope_theta = float(rope_theta)
        self.dtype = dtype
        self.blocks = _build_transformer_blocks(
            rngs=rngs,
            dim=self.dim,
            num_layers=num_layers,
            num_heads=self.num_heads,
            num_kv_heads=kv_heads,
            attention_mlp_multiple=attention_mlp_multiple,
            use_rotary_embeddings=True,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.norm = nnx.RMSNorm(
            num_features=self.dim,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        train: bool,
        attention_bias: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        seq_len = x.shape[1]
        freqs_cos, freqs_sin = transformer.precompute_freqs_cis(
            self.dim // self.num_heads,
            seq_len,
            theta=self.rope_theta,
            dtype=self.dtype,
        )
        freqs_cos = freqs_cos[:seq_len].astype(x.dtype)
        freqs_sin = freqs_sin[:seq_len].astype(x.dtype)
        for block in self.blocks:
            x = block(
                x,
                freqs_cos,
                freqs_sin,
                train=train,
                attention_bias=attention_bias,
            )
        return self.norm(x)


class BERT(nnx.Module):
    """BERT-style masked language model."""

    def __init__(
        self,
        rngs: nnx.Rngs,
        *,
        vocab_size: int,
        max_length: int,
        precursor_bins: int,
        precursor_offset: int,
        model_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        num_kv_heads: int | None = None,
        attention_mlp_multiple: float = 4.0,
        num_segments: int = 2,
        mask_ratio: float = 0.15,
        mask_token_id: int = 103,
        pad_token_id: int = 0,
        cls_token_id: int = 101,
        sep_token_id: int = 102,
        rope_theta: float = 10000.0,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
    ):
        self.vocab_size = int(vocab_size)
        self.max_length = int(max_length)
        self.precursor_bins = int(precursor_bins)
        self.precursor_offset = int(precursor_offset)
        self.model_dim = int(model_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.num_kv_heads = None if num_kv_heads is None else int(num_kv_heads)
        self.attention_mlp_multiple = float(attention_mlp_multiple)
        self.num_segments = int(num_segments)
        self.mask_ratio = float(mask_ratio)
        self.mask_token_id = int(mask_token_id)
        self.pad_token_id = int(pad_token_id)
        self.cls_token_id = int(cls_token_id)
        self.sep_token_id = int(sep_token_id)
        self.rope_theta = float(rope_theta)
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.rngs = rngs

        embed_init = nnx.with_partitioning(
            nnx.initializers.variance_scaling(1.0, "fan_in", "normal", out_axis=0),
            ("embed_vocab", "hidden"),
        )
        self.token_embed = nnx.Embed(
            num_embeddings=self.vocab_size,
            features=self.model_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            embedding_init=embed_init,
            rngs=rngs,
        )
        self.segment_embed = nnx.Embed(
            num_embeddings=self.num_segments,
            features=self.model_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            embedding_init=embed_init,
            rngs=rngs,
        )
        self.embed_norm = nnx.LayerNorm(
            num_features=self.model_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.encoder = TransformerStack(
            rngs=rngs,
            dim=self.model_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            attention_mlp_multiple=self.attention_mlp_multiple,
            rope_theta=self.rope_theta,
            dtype=dtype,
            param_dtype=param_dtype,
        )

        lm_head_init = nnx.with_partitioning(
            nnx.initializers.zeros,
            ("hidden", "vocab"),
        )
        self.lm_head = nnx.Linear(
            in_features=self.model_dim,
            out_features=self.vocab_size,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=lm_head_init,
            rngs=rngs,
        )
        self.precursor_head = nnx.Linear(
            in_features=self.model_dim,
            out_features=2 * self.precursor_bins,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=lm_head_init,
            rngs=rngs,
        )
        self.retention_head = nnx.Linear(
            in_features=self.model_dim,
            out_features=2,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=lm_head_init,
            rngs=rngs,
        )

    def _make_attention_bias(self, allow: jnp.ndarray, *, dtype: jnp.dtype) -> jnp.ndarray:
        neg = jnp.asarray(-1e9, dtype=dtype)
        zero = jnp.asarray(0.0, dtype=dtype)
        bias = jnp.where(allow, zero, neg)
        return bias[:, None, None, :]

    def _mask_tokens(
        self,
        token_ids: jnp.ndarray,
        attention_mask: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        maskable = attention_mask & (token_ids != self.cls_token_id)
        seq_len = token_ids.shape[1]
        mask_count = int(self.mask_ratio * seq_len)
        scores = jax.random.uniform(self.rngs.mask(), token_ids.shape, dtype=jnp.float32)
        scores = jnp.where(maskable, scores, -1.0)
        _, mask_idx = jax.lax.top_k(scores, mask_count)
        batch_idx = jnp.arange(token_ids.shape[0])[:, None]
        mask = jnp.zeros_like(maskable)
        mask = mask.at[batch_idx, mask_idx].set(True)
        masked_tokens = jnp.where(mask, self.mask_token_id, token_ids)
        return masked_tokens, mask, mask_idx

    def _embed_inputs(
        self,
        token_ids: jnp.ndarray,
        segment_ids: jnp.ndarray,
    ) -> jnp.ndarray:
        tok = self.token_embed(token_ids)
        seg = self.segment_embed(segment_ids)
        return self.embed_norm(tok + seg)

    def _masked_cross_entropy(
        self,
        logits: jnp.ndarray,
        labels: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> jnp.ndarray:
        per_token = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        mask_f = mask.astype(per_token.dtype)
        return jnp.sum(per_token * mask_f) / jnp.sum(mask_f)

    def _masked_accuracy(
        self,
        logits: jnp.ndarray,
        labels: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> jnp.ndarray:
        pred = jnp.argmax(logits, axis=-1)
        correct = (pred == labels).astype(jnp.float32)
        mask_f = mask.astype(jnp.float32)
        return jnp.sum(correct * mask_f) / jnp.sum(mask_f)

    def _precursor_metrics(
        self,
        cls_state: jnp.ndarray,
        precursor_tokens: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        logits = self.precursor_head(cls_state).reshape(
            cls_state.shape[0], 2, self.precursor_bins
        )
        labels = precursor_tokens - self.precursor_offset
        labels = labels.astype(jnp.int32)
        per_token = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        loss = jnp.mean(per_token)
        pred = jnp.argmax(logits, axis=-1)
        acc = jnp.mean((pred == labels).astype(jnp.float32))
        return loss, acc

    def _retention_metrics(
        self,
        cls_state: jnp.ndarray,
        rt: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        logits = self.retention_head(cls_state)
        labels = jnp.where(rt[:, 0] < rt[:, 1], 0, 1).astype(jnp.int32)
        loss = jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        )
        pred = jnp.argmax(logits, axis=-1)
        acc = jnp.mean((pred == labels).astype(jnp.float32))
        return loss, acc

    def __call__(
        self,
        batch: dict[str, jnp.ndarray],
        *,
        train: bool = True,
        apply_mask: bool | None = None,
    ) -> dict[str, jnp.ndarray]:
        if apply_mask is None:
            apply_mask = train

        token_ids = jnp.asarray(batch["token_ids"])
        segment_ids = jnp.asarray(batch["segment_ids"])
        attention_mask = token_ids != self.pad_token_id

        if apply_mask:
            masked_tokens = jnp.asarray(batch["masked_token_ids"])
            mask = jnp.asarray(batch["mlm_mask"])
        else:
            masked_tokens = token_ids
            mask = jnp.zeros_like(attention_mask, dtype=jnp.bool_)

        x = self._embed_inputs(masked_tokens, segment_ids)
        attention_bias = self._make_attention_bias(attention_mask, dtype=x.dtype)
        encoded = self.encoder(x, train=train, attention_bias=attention_bias)
        logits = self.lm_head(encoded)

        if apply_mask:
            mlm_loss = self._masked_cross_entropy(logits, token_ids, mask)
            token_accuracy = self._masked_accuracy(logits, token_ids, mask)
        else:
            mlm_loss = jnp.asarray(0.0, dtype=logits.dtype)
            token_accuracy = jnp.asarray(0.0, dtype=jnp.float32)
        mask_ratio_actual = jnp.mean(mask.astype(jnp.float32))
        cls_state = encoded[:, 0, :]
        precursor_tokens = jnp.asarray(batch["precursor_mz"])
        rt = jnp.asarray(batch["rt"])
        precursor_loss, precursor_accuracy = self._precursor_metrics(
            cls_state, precursor_tokens
        )
        retention_loss, retention_accuracy = self._retention_metrics(cls_state, rt)
        loss = mlm_loss + precursor_loss + retention_loss

        return {
            "loss": loss,
            "token_accuracy": token_accuracy,
            "mask_ratio_actual": mask_ratio_actual,
            "precursor_loss": precursor_loss,
            "precursor_accuracy": precursor_accuracy,
            "retention_loss": retention_loss,
            "retention_accuracy": retention_accuracy,
        }

    def encode(self, batch: dict[str, jnp.ndarray], *, train: bool = False) -> jnp.ndarray:
        token_ids = jnp.asarray(batch["token_ids"])
        segment_ids = jnp.asarray(batch["segment_ids"])
        attention_mask = token_ids != self.pad_token_id

        x = self._embed_inputs(token_ids, segment_ids)
        attention_bias = self._make_attention_bias(attention_mask, dtype=x.dtype)
        encoded = self.encoder(x, train=train, attention_bias=attention_bias)
        return encoded[:, 0, :]

    def compute_loss(self, batch: dict[str, jax.Array], *, train: bool = False):
        metrics = self(batch, train=train, apply_mask=train)
        loss = metrics["loss"]
        return loss, metrics


def _create_metrics_class():
    return clu_metrics.Collection.create(
        loss=clu_metrics.Average.from_output("loss"),
        token_accuracy=clu_metrics.Average.from_output("token_accuracy"),
        mask_ratio_actual=clu_metrics.Average.from_output("mask_ratio_actual"),
        precursor_loss=clu_metrics.Average.from_output("precursor_loss"),
        precursor_accuracy=clu_metrics.Average.from_output("precursor_accuracy"),
        retention_loss=clu_metrics.Average.from_output("retention_loss"),
        retention_accuracy=clu_metrics.Average.from_output("retention_accuracy"),
    )


Metrics = _create_metrics_class()
TrainMetrics = Metrics
EvalMetrics = Metrics


def get_train_metrics_class():
    return TrainMetrics


def create_train_metrics():
    return TrainMetrics.empty()


def create_eval_metrics():
    return EvalMetrics.empty()
