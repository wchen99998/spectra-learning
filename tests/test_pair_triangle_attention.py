"""Triangle attention on compact pair tensors and alternating pair-only stacks."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
torch = pytest.importorskip("torch")

from flax import nnx  # noqa: E402

from spectra_learning.models.model_jax import PeakSetJEPAJax  # noqa: E402
from spectra_learning.models.pairmixer import (  # noqa: E402
    TriangleAttention as TorchTriangleAttention,
)
from spectra_learning.models.pairmixer_jax import (  # noqa: E402
    PairOnlyBlock,
    TriangleAttention,
)
from spectra_learning.models.settings import PAIR_DISTOGRAM_TRAINING_MODE  # noqa: E402


def _attention(ending: bool, dtype=jnp.float32):
    reference = TorchTriangleAttention(
        8, num_heads=2, ending=ending, norm_eps=1e-5,
    )
    # Set every parameter explicitly so the reference does not depend on the
    # global Torch RNG, and exercise the learned triangle bias and gates.
    rng = np.random.default_rng(3)
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.copy_(torch.from_numpy(
                rng.normal(0, 0.3, parameter.shape).astype(np.float32)
            ))
        reference.norm.weight.add_(1)
    attention = TriangleAttention(
        8, num_heads=2, ending=ending, norm_eps=1e-5,
        compute_dtype=dtype, rngs=nnx.Rngs(0),
    )
    state = reference.state_dict()
    for name in ("norm", "qkv", "bias", "g", "o"):
        getattr(attention, name).load_torch_state_dict(state, name)
    return attention, reference


@pytest.mark.parametrize("ending", [False, True])
def test_triangle_attention_matches_torch_forward_and_gradients(ending):
    attention, reference = _attention(ending)
    rng = np.random.default_rng(5)
    values = rng.normal(size=(2, 4, 4, 8)).astype(np.float32)
    cotangent = rng.normal(size=values.shape).astype(np.float32)
    visible = np.array([[True, False, True, True], [False, True, True, False]])
    mask = visible[:, :, None] & visible[:, None, :]

    torch_input = torch.tensor(values, requires_grad=True)
    expected = reference(torch_input, torch.tensor(visible), torch.tensor(mask))
    (expected * torch.tensor(cotangent)).sum().backward()
    graphdef, params = nnx.split(attention)

    def loss(parameters, pair):
        return jnp.sum(nnx.merge(graphdef, parameters)(pair, jnp.asarray(mask)) * cotangent)

    param_grads, input_grads = jax.grad(loss, argnums=(0, 1))(params, jnp.asarray(values))
    actual = attention(jnp.asarray(values), jnp.asarray(mask))
    np.testing.assert_allclose(actual, expected.detach().numpy(), atol=2e-6, rtol=2e-5)
    np.testing.assert_allclose(input_grads, torch_input.grad.numpy(), atol=3e-6, rtol=3e-5)
    torch_grads = dict(reference.named_parameters())
    for path, gradient in nnx.to_flat_state(param_grads):
        name = ".".join(str(part) for part in path)
        np.testing.assert_allclose(
            gradient.get_value(), torch_grads[name].grad.numpy(),
            atol=5e-6, rtol=5e-5, err_msg=name,
        )


@pytest.mark.parametrize("ending", [False, True])
def test_packed_attention_matches_noncontiguous_padding_and_empty_spectra(ending):
    attention, _ = _attention(ending)
    values = jnp.asarray(np.random.default_rng(7).normal(size=(2, 5, 5, 8)), jnp.float32)
    visible = jnp.array([[True, False, True, False, True], [False] * 5])
    mask = visible[:, :, None] & visible[:, None, :]
    positions = jnp.array([0, 2, 4])
    packed = jnp.take(jnp.take(values[:1], positions, axis=1), positions, axis=2)
    expected = attention(packed, jnp.ones((1, 3, 3), bool))
    actual = attention(values, mask)
    compact_actual = jnp.take(jnp.take(actual[:1], positions, axis=1), positions, axis=2)
    np.testing.assert_allclose(compact_actual, expected, atol=2e-6, rtol=2e-5)
    np.testing.assert_array_equal(np.asarray(actual)[~np.asarray(mask)], 0)

    changed_padding = jnp.where(mask[..., None], values, values * 100 + 10)
    np.testing.assert_allclose(attention(changed_padding, mask), actual, atol=2e-6, rtol=2e-5)
    gradients = jax.grad(lambda pair: jnp.square(attention(pair, mask)).sum())(values)
    assert np.isfinite(np.asarray(gradients)).all()
    np.testing.assert_array_equal(np.asarray(gradients)[~np.asarray(mask)], 0)


@pytest.mark.parametrize("ending", [False, True])
def test_bfloat16_triangle_attention_has_finite_nonzero_gradients(ending):
    attention, _ = _attention(ending, jnp.bfloat16)
    values = jnp.asarray(np.random.default_rng(9).normal(size=(2, 4, 4, 8)), jnp.bfloat16)
    visible = jnp.array([[True, False, True, True], [False] * 4])
    mask = visible[:, :, None] & visible[:, None, :]
    output = attention(values, mask)
    assert output.dtype == jnp.bfloat16
    assert np.isfinite(np.asarray(output, dtype=np.float32)).all()
    np.testing.assert_array_equal(np.asarray(output[1], dtype=np.float32), 0)
    gradients = nnx.grad(
        lambda module: jnp.square(module(values, mask).astype(jnp.float32)).sum()
    )(attention)
    for gradient in jax.tree.leaves(gradients):
        assert np.isfinite(np.asarray(gradient)).all()
    assert float(jnp.linalg.norm(gradients.qkv.weight[...])) > 0
    assert float(jnp.linalg.norm(gradients.bias.weight[...])) > 0


def _model(**overrides) -> PeakSetJEPAJax:
    kwargs = dict(
        training_mode=PAIR_DISTOGRAM_TRAINING_MODE,
        model_dim=8,
        predictor_dim=8,
        encoder_num_layers=2,
        encoder_num_heads=2,
        attention_mlp_multiple=2.0,
        feature_mlp_hidden_dim=8,
        encoder_fourier_mlp_hidden_dim=8,
        encoder_fourier_num_freqs=2,
        pairmixer_fourier_num_freqs=2,
        pairmixer_pair_dim=8,
        pairmixer_pair_feature_hidden_dim=8,
        pairmixer_block_type="fastmixer",
        pairmixer_triangle_attention_every_n_layers=2,
        pairmixer_triangle_attention_num_heads=2,
        pairmixer_fast_encoder_max_visible_tokens=3,
        pairmixer_fast_max_visible_tokens=5,
        predictor_target_max_tokens=2,
        masked_latent_predictor_num_layers=2,
        masked_latent_predictor_num_heads=2,
        masked_token_input_mode="latent_token",
        num_peaks=5,
        encoder_use_cls_token=True,
        jepa_num_target_blocks=1,
        jepa_mae_mz_bin_size=0.25,
        distogram_mz_max=2.0,
        mae_loss_weight=0.0,
        distogram_loss_weight=1.0,
        target_projector_dim=-1,
        rngs=nnx.Rngs(params=0, dropout=0),
    )
    kwargs.update(overrides)
    return PeakSetJEPAJax(**kwargs)


def test_attention_replaces_every_second_encoder_and_predictor_block():
    model = nnx.eval_shape(lambda: _model(
        encoder_num_layers=5, masked_latent_predictor_num_layers=3,
    ))
    for blocks in (model.encoder.blocks, model.masked_latent_predictor):
        for index, block in enumerate(blocks, start=1):
            assert isinstance(block, PairOnlyBlock)
            attention = index % 2 == 0
            assert block.use_triangle_attention is attention
            assert hasattr(block, "tri_attn_start") is attention
            assert hasattr(block, "tri_attn_end") is attention
            assert hasattr(block, "tri_mul_out") is not attention
            assert hasattr(block, "tri_mul_in") is not attention


def test_attention_metadata_gates_start_at_identity_and_learn():
    block = PairOnlyBlock(
        pair_dim=8, attention_mlp_multiple=2.0, norm_eps=1e-5,
        condition_dim=4, triangle_attention_num_heads=2, rngs=nnx.Rngs(0),
    )
    pair = jnp.asarray(np.random.default_rng(11).normal(size=(1, 3, 3, 8)), jnp.float32)
    visible = jnp.array([[True, False, True]])
    condition = jnp.ones((1, 4))
    mask = visible[:, :, None] & visible[:, None, :]
    np.testing.assert_array_equal(block(pair, visible, condition), pair * mask[..., None])

    gradients = nnx.grad(lambda module: module(pair, visible, condition).sum())(block)
    modulation_grad = gradients.adaLN_modulation.weight[...]
    # Each attention direction must give its metadata gate a learning signal.
    for gate_index in (2, 5):
        assert float(jnp.linalg.norm(modulation_grad[gate_index * 8:(gate_index + 1) * 8])) > 0
    block.adaLN_modulation.weight[...] -= 1e-3 * modulation_grad
    conditioned = block(pair, visible, condition)
    unconditioned = block(pair, visible, jnp.zeros_like(condition))
    assert not np.allclose(conditioned, unconditioned)
    np.testing.assert_array_equal(np.asarray(conditioned)[~np.asarray(mask)], 0)


def _batch() -> dict[str, jax.Array]:
    return {
        "peak_mz": jnp.array([[0.1, 0.3, 0.6, 1.0, 1.5]]),
        "peak_intensity": jnp.array([[0.8, 0.4, 0.5, 0.7, 0.2]]),
        "peak_valid_mask": jnp.ones((1, 5), bool),
        "context_mask": jnp.array([[True, False, False, True, False]]),
        "target_masks": jnp.array([[[False, True, False, False, True]]]),
        "precursor_mz": jnp.array([1.8]),
    }


def _exercise_residual_branches(model: PeakSetJEPAJax) -> None:
    # Let the attention stack read the context instead of testing only the
    # zero-initialized residual identity.
    flat = nnx.to_flat_state(nnx.state(model, nnx.Param))
    updated = []
    rng = np.random.default_rng(13)
    for path, leaf in flat:
        if path[-2:] in (("o", "weight"), ("p_out", "weight"), ("fc3", "weight")):
            leaf = leaf.replace(jnp.asarray(rng.normal(0, 0.02, leaf.shape), jnp.float32))
        updated.append((path, leaf))
    nnx.update(model, nnx.from_flat_state(updated))


def test_alternating_attention_does_not_receive_target_masses_or_intensities(monkeypatch):
    model = _model()
    _exercise_residual_branches(model)
    batch = _batch()
    captured = []
    score = model._distogram_score

    def capture_score(features, targets):
        captured.append(np.asarray(features))
        return score(features, targets)

    monkeypatch.setattr(model, "_distogram_score", capture_score)
    original_loss = float(model(batch)["loss"])
    target = jnp.any(batch["target_masks"], axis=1)
    changed = dict(batch,
        peak_mz=jnp.where(target, batch["peak_mz"] + 0.37, batch["peak_mz"]),
        peak_intensity=jnp.where(target, 100.0, batch["peak_intensity"]),
    )
    changed_loss = float(model(changed)["loss"])
    np.testing.assert_array_equal(captured[0], captured[1])
    assert original_loss != changed_loss
    intensity_loss = float(model(dict(batch, peak_intensity=changed["peak_intensity"]))["loss"])
    assert intensity_loss == original_loss


def test_alternating_pair_model_rematerialization_and_compiled_accumulation():
    batch = _batch()
    results = []
    for mode in ("none", "full", "selective"):
        model = _model(
            activation_checkpoint_mode=mode,
            activation_checkpoint_modules=("encoder", "predictor", "distogram"),
        )
        _exercise_residual_branches(model)
        results.append(nnx.value_and_grad(lambda module: module(batch, loss_only=True)["loss"])(model))
    loss, gradients = results[0]
    for remat_loss, remat_gradients in results[1:]:
        np.testing.assert_allclose(remat_loss, loss, atol=1e-6, rtol=1e-5)
        for actual, expected in zip(jax.tree.leaves(remat_gradients), jax.tree.leaves(gradients), strict=True):
            assert np.isfinite(np.asarray(actual)).all()
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=1e-4)
    for prefix in ("encoder", "masked_latent_predictor"):
        attention_grads = [
            gradient.get_value()
            for path, gradient in nnx.to_flat_state(gradients)
            if path[0] == prefix and "tri_attn_start" in path and path[-2:] == ("qkv", "weight")
        ]
        assert len(attention_grads) == 1
        assert float(jnp.linalg.norm(attention_grads[0])) > 0

    # The production train step JITs the accumulation scan and the rematted
    # alternating stack together. Two identical microbatches must produce the
    # same update as the full gradient already checked above.
    import optax
    from spectra_learning.training.pretrain_jax import (
        init_pure_optax_train_state,
        make_pure_accumulated_train_step,
        trainable_param_filter,
    )
    graphdef, params, static, state, optimizer = init_pure_optax_train_state(
        {"optimizer": "adam", "learning_rate": 1e-4, "weight_decay": 0.0,
         "b2": 0.95, "optimizer_state_dtype": "bf16"}, model,
    )
    trainable_gradients = nnx.as_pure(nnx.filter_state(remat_gradients, trainable_param_filter))
    updates, _ = optimizer.update(trainable_gradients, state, params)
    expected_params = optax.apply_updates(params, updates)
    step = make_pure_accumulated_train_step(graphdef, optimizer, sharded=False)
    accumulated = jax.tree.map(lambda value: jnp.stack((value, value)), batch)
    actual_params, _, metrics = step(params, static, state, accumulated)
    np.testing.assert_allclose(metrics["loss"], loss, atol=1e-6, rtol=1e-5)
    for actual, expected in zip(jax.tree.leaves(actual_params), jax.tree.leaves(expected_params), strict=True):
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=1e-5)
