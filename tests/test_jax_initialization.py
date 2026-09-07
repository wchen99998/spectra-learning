import math
import gc
import weakref

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from ml_collections import config_dict

from spectra_learning.models.encoder_jax import MetadataEmbedder
from spectra_learning.models.peak_features_jax import MzTokenEmbedding
from spectra_learning.models.model_jax import PeakSetJEPAJax
from spectra_learning.training import pretrain_jax


def _frozen_teacher_loop(*, mask_schedule: bool):
    loop = object.__new__(pretrain_jax._JaxTrainingLoop)
    loop.config = config_dict.ConfigDict({
        "optimizer": "adam", "learning_rate": 1e-3,
        "optimizer_state_dtype": "bf16",
        "frozen_teacher_checkpoint_path": "unused-in-test",
    })
    loop.model = PeakSetJEPAJax(
        training_mode="mae_teacher_jepa", model_dim=8, predictor_dim=8,
        encoder_num_layers=1, encoder_num_heads=2,
        feature_mlp_hidden_dim=8, encoder_fourier_num_freqs=2,
        masked_latent_predictor_num_layers=1,
        masked_latent_predictor_num_heads=2, num_peaks=5,
        pairmixer_block_type="fastmixer-dense", target_projector_dim=-1,
        masked_token_loss_weight=1.0, mae_loss_weight=0.0,
    )
    loop.total_steps = 10
    loop.use_sharded_step = True
    loop.data_parallel_devices = jax.device_count()
    loop.data_mesh = pretrain_jax._jax_data_mesh_for_device_count(jax.device_count())
    loop.metric_reduction = "mean"
    loop.group_jepa_ema_momentum = None
    loop.group_jepa_lookahead = False
    loop.use_mask_schedule = mask_schedule
    loop.mask_capacities = ((3, 5, 2),) if mask_schedule else ()
    loop.resume_step = 0
    return loop


@pytest.mark.parametrize("mask_schedule", (False, True))
def test_initialization_releases_original_buffers_before_replicating_moments(
    monkeypatch, mask_schedule: bool,
) -> None:
    loop = _frozen_teacher_loop(mask_schedule=mask_schedule)
    original_refs = []
    placements = 0

    def copy_to_mesh(tree, mesh):
        nonlocal placements
        del mesh
        if placements < 2:
            original_refs.extend(weakref.ref(x) for x in jax.tree.leaves(tree))
        else:
            gc.collect()
            assert all(ref() is None for ref in original_refs)
        placements += 1
        # Force distinct buffers even on one CPU device, as multihost
        # replication does when copying initialized arrays onto the mesh.
        return jax.tree.map(lambda x: jnp.array(x, copy=True), tree)

    monkeypatch.setattr(pretrain_jax, "_replicate_tree_on_data_mesh", copy_to_mesh)
    loop._initialize_train_state()
    assert placements == 3


def test_restored_frozen_teacher_replaces_model_buffers(monkeypatch) -> None:
    loop = _frozen_teacher_loop(mask_schedule=True)
    loop.resume_step = None
    restored_refs = []

    def restore_teacher(path, state):
        del path
        restored = jax.tree.map(lambda x: x + 1, state)
        restored_refs.extend(weakref.ref(x) for x in jax.tree.leaves(restored))
        return restored

    monkeypatch.setattr(pretrain_jax, "restore_frozen_teacher_encoder", restore_teacher)
    state, *_ = loop._initialize_train_state()
    model_teacher = nnx.as_pure(nnx.state(loop.model.teacher_encoder))
    for model_value, state_value, ref in zip(
        jax.tree.leaves(model_teacher),
        jax.tree.leaves(state.static_state["teacher_encoder"]),
        restored_refs,
        strict=True,
    ):
        assert model_value is state_value is ref()


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
