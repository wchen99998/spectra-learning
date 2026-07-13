from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

import train
from spectra_learning.data.ar_spectra import SpectraARTokenizer, SpectraARTokenizerConfig
from spectra_learning.data.spectra import DEFAULT_MAX_PRECURSOR_MZ, PEAK_MZ_MAX
import spectra_learning.models.ar_spectra_jax as ar_spectra_jax
from spectra_learning.models.ar_spectra_jax import (
    RotaryEmbeddingJax,
    SpectraARCausalSelfAttentionJax,
    SpectraARTransformerJax,
    SpectraARTransformerJaxConfig,
    pack_qk_projection_rows,
)
from spectra_learning.models.causal_attention_pallas import pallas_causal_attention
from spectra_learning.models.common_jax import scaled_dot_product_attention


def _batch() -> dict[str, torch.Tensor]:
    return {
        "peak_mz": torch.tensor(
            [[138.75, 938.50, 466.25, 0.0]],
            dtype=torch.float32,
        )
        / PEAK_MZ_MAX,
        "peak_intensity": torch.tensor([[0.10, 1.00, 0.50, 0.0]], dtype=torch.float32),
        "peak_valid_mask": torch.tensor([[True, True, True, False]]),
        "precursor_mz": torch.tensor([512.50], dtype=torch.float32)
        / DEFAULT_MAX_PRECURSOR_MZ,
        "collision_energy": torch.tensor([0.35], dtype=torch.float32),
        "charge": torch.tensor([2.0], dtype=torch.float32),
    }


def _jax_batch(batch: dict[str, torch.Tensor]) -> dict[str, jax.Array]:
    return {
        key: jnp.asarray(value.detach().cpu().numpy())
        for key, value in batch.items()
    }


def test_jax_scaled_dot_product_attention_matches_manual_causal_attention() -> None:
    key = jax.random.PRNGKey(0)
    q, k, v = (
        jax.random.normal(subkey, (2, 4, 7, 8), dtype=jnp.float32)
        for subkey in jax.random.split(key, 3)
    )

    actual = scaled_dot_product_attention(q, k, v, is_causal=True)
    scores = jnp.einsum("...qd,...kd->...qk", q, k) / np.sqrt(q.shape[-1])
    causal = jnp.tril(jnp.ones((q.shape[-2], k.shape[-2]), dtype=jnp.bool_))
    scores = jnp.where(causal, scores, -jnp.inf)
    expected = jnp.einsum("...qk,...kd->...qd", jax.nn.softmax(scores, axis=-1), v)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-5)


def test_packed_qk_projection_matches_interleaved_rope_coordinates() -> None:
    model_dim = 16
    num_heads = 2
    head_dim = model_dim // num_heads
    hidden = jax.random.normal(jax.random.key(1), (2, 5, model_dim))
    weight = jax.random.normal(jax.random.key(2), (3 * model_dim, model_dim))
    query_scale = 0.5
    projected = jnp.matmul(hidden, weight.T).reshape(
        2,
        5,
        3,
        num_heads,
        head_dim,
    )
    packed_projected = jnp.matmul(
        hidden,
        pack_qk_projection_rows(
            weight,
            num_heads=num_heads,
            query_scale=query_scale,
        ).T,
    ).reshape(2, 5, 3, num_heads, head_dim)
    rope = RotaryEmbeddingJax(head_dim, 5, base=10_000.0)

    for index, scale in ((0, query_scale), (1, 1.0)):
        interleaved = rope(projected[:, :, index] * scale)
        expected_split = jnp.concatenate(
            (interleaved[..., 0::2], interleaved[..., 1::2]),
            axis=-1,
        )
        actual_split = rope.split_half(packed_projected[:, :, index])
        np.testing.assert_allclose(
            np.asarray(actual_split),
            np.asarray(expected_split),
            atol=2e-5,
            rtol=2e-5,
        )
    np.testing.assert_allclose(
        np.asarray(packed_projected[:, :, 2]),
        np.asarray(projected[:, :, 2]),
        atol=2e-5,
        rtol=2e-5,
    )


@pytest.mark.skipif(jax.default_backend() != "tpu", reason="Pallas TPU kernel")
@pytest.mark.parametrize("sequence_length", (128, 136))
def test_pallas_causal_attention_forward_and_backward_match_xla(
    sequence_length: int,
) -> None:
    shape = (1, 8, sequence_length, 128)
    query, key, value, output_gradient = (
        jax.random.normal(subkey, shape, dtype=jnp.bfloat16)
        for subkey in jax.random.split(jax.random.key(3), 4)
    )

    def loss(attention, query, key, value):
        output = attention(query, key, value)
        objective = jnp.sum(
            output.astype(jnp.float32) * output_gradient.astype(jnp.float32)
        )
        return objective, output

    pallas_fn = jax.jit(
        jax.value_and_grad(
            lambda q, k, v: loss(pallas_causal_attention, q, k, v),
            argnums=(0, 1, 2),
            has_aux=True,
        )
    )
    xla_fn = jax.jit(
        jax.value_and_grad(
            lambda q, k, v: loss(
                lambda query, key, value: scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    is_causal=True,
                ),
                q,
                k,
                v,
            ),
            argnums=(0, 1, 2),
            has_aux=True,
        )
    )
    (_pallas_loss, pallas_output), pallas_grads = pallas_fn(query, key, value)
    (_xla_loss, xla_output), xla_grads = xla_fn(query, key, value)

    for actual, expected in zip(
        (pallas_output, *pallas_grads),
        (xla_output, *xla_grads),
        strict=True,
    ):
        actual_np = np.asarray(actual, dtype=np.float32)
        assert np.isfinite(actual_np).all()
        np.testing.assert_allclose(
            actual_np,
            np.asarray(expected, dtype=np.float32),
            atol=4e-2,
            rtol=3e-2,
        )


def test_jax_ar_splash_attention_flattens_sequence_before_heads(monkeypatch) -> None:
    def fake_splash_attention(query, key, value, *, block_size):
        del key, value, block_size
        return query

    monkeypatch.setattr(
        ar_spectra_jax,
        "splash_causal_attention",
        fake_splash_attention,
    )
    config = SpectraARTransformerJaxConfig(
        vocab_size=16,
        num_token_kinds=4,
        max_sequence_length=3,
        pad_token_id=0,
        model_dim=4,
        num_layers=1,
        num_heads=2,
        mlp_multiple=2.0,
        attention_kernel="splash",
        compute_dtype=jnp.float32,
    )
    attention = SpectraARCausalSelfAttentionJax(config, rngs=ar_spectra_jax.nnx.Rngs(0))
    attention.qkv.weight[...] = jnp.concatenate(
        [jnp.eye(4, dtype=jnp.float32)] * 3,
        axis=0,
    )
    attention.out_proj.weight[...] = jnp.eye(4, dtype=jnp.float32)
    hidden = jnp.arange(12, dtype=jnp.float32).reshape(1, 3, 4)

    actual = attention(hidden)
    expected = attention.rope(hidden.reshape(1, 3, 2, 2)).reshape(1, 3, 4)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-6)


def test_jax_ar_transformer_forward_returns_finite_loss() -> None:
    tokenizer = SpectraARTokenizer(SpectraARTokenizerConfig(max_num_peaks=4))
    tokenized = tokenizer.tokenize_batch(_batch())
    model = SpectraARTransformerJax(
        SpectraARTransformerJaxConfig(
            vocab_size=tokenizer.vocab_size,
            num_token_kinds=tokenizer.num_token_kinds,
            max_sequence_length=tokenizer.sequence_length,
            pad_token_id=tokenizer.pad_token_id,
            model_dim=32,
            num_layers=1,
            num_heads=4,
            mlp_multiple=2.0,
            compute_dtype=jnp.float32,
        )
    )

    assert model.use_ema_teacher is False
    train_output = model(_jax_batch(tokenized))
    inference_batch = {
        key: value
        for key, value in tokenized.items()
        if not key.startswith("target_")
    }
    inference_output = model(_jax_batch(inference_batch))

    assert inference_output["logits"].shape == (
        1,
        tokenizer.sequence_length - 1,
        tokenizer.vocab_size,
    )
    assert "logits" not in train_output
    assert np.isfinite(np.asarray(train_output["loss"]))
    assert float(np.asarray(train_output["target_tokens"])) == 19.0
    assert "token_accuracy/fragment_mz_level_0" in train_output


def test_train_routes_ar_spectra_jax_backend(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_train(config, workdir):
        calls.append((config, workdir))
        return {"run/device_backend": "jax", "run/training_task": "ar_spectra"}

    import spectra_learning.training.ar_spectra_jax as ar_spectra_jax

    monkeypatch.setattr(
        ar_spectra_jax,
        "train_and_evaluate_ar_spectra_jax",
        fake_train,
    )
    cfg = {
        "training_task": "ar_spectra",
        "device_backend": "jax",
    }

    assert train._train(cfg, tmp_path)["run/training_task"] == "ar_spectra"
    assert calls == [(cfg, tmp_path)]
