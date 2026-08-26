import math

import numpy as np
from flax import nnx
from ml_collections import config_dict

from spectra_learning.models.encoder_jax import MetadataEmbedder
from spectra_learning.models.peak_features_jax import MzTokenEmbedding
from spectra_learning.training import pretrain_jax


def test_jax_pretraining_keeps_native_model_initialization(
    monkeypatch,
    tmp_path,
) -> None:
    captured = {}

    def fake_train(config, workdir, *, task):
        del config, workdir
        captured["initialize_model"] = task.initialize_model
        return {}

    monkeypatch.setattr(pretrain_jax, "train_and_evaluate_jax_task", fake_train)

    pretrain_jax.train_and_evaluate_jax(config_dict.ConfigDict(), tmp_path)

    assert captured["initialize_model"] is None


def test_jax_metadata_embedder_uses_xavier_normal() -> None:
    condition_dim = 1_024
    embedder = MetadataEmbedder(
        condition_dim,
        compute_dtype=np.float32,
        rngs=nnx.Rngs(7),
    )

    linear0 = np.asarray(embedder.linear0.weight[...])
    linear2 = np.asarray(embedder.linear2.weight[...])
    np.testing.assert_allclose(
        linear0.std(),
        math.sqrt(2.0 / (28 + condition_dim)),
        rtol=0.05,
    )
    np.testing.assert_allclose(
        linear2.std(),
        math.sqrt(2.0 / (2 * condition_dim)),
        rtol=0.05,
    )
    np.testing.assert_array_equal(
        np.asarray(embedder.linear0.bias[...]),
        np.zeros(condition_dim, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(embedder.linear2.bias[...]),
        np.zeros(condition_dim, dtype=np.float32),
    )


def test_jax_mz_token_embedding_is_random_initialized() -> None:
    embedding = MzTokenEmbedding(
        mz_scale=100.0,
        bin_size=0.1,
        embedding_dim=32,
        rngs=nnx.Rngs(11),
    )

    weight = np.asarray(embedding.embedding.weight[...])
    assert not np.allclose(weight, 0.0)
    np.testing.assert_allclose(weight.std(), 0.02, rtol=0.05)
