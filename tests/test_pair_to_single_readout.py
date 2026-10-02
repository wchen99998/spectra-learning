"""Information flow, factorization, and compact-axis contracts of pair readout."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
nnx = pytest.importorskip("flax.nnx")
pytest.importorskip("torch")

from spectra_learning.models.pair_to_single_jax import PairToSingleReadout


def _model(dtype=jnp.float32):
    return PairToSingleReadout(16, 8, num_heads=2, head_dim=4,
                              compute_dtype=dtype, rngs=nnx.Rngs(3))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_padding_empty_rows_and_gradients(dtype):
    if dtype == jnp.bfloat16 and jax.default_backend() == "cpu":
        pytest.skip("JAX 0.11 CPU DotThunk does not support BF16 operands with FP32 output")
    model = _model(dtype)
    pair = jax.random.normal(jax.random.key(1), (2, 5, 5, 8)).astype(dtype)
    mask = jnp.array([[True, False, True, False, True], [False] * 5])
    valid = mask[:, :, None] & mask[:, None, :]
    actual = model(pair, mask)
    changed = jnp.where(valid[..., None], pair, pair * 30 + 50)
    np.testing.assert_allclose(model(changed, mask).astype(jnp.float32),
                               actual.astype(jnp.float32), atol=1e-6)
    np.testing.assert_array_equal(np.asarray(actual)[~np.asarray(mask)], 0)
    gradient = jax.grad(lambda z: model(z, mask).astype(jnp.float32)[..., 0].sum())(pair)
    assert np.isfinite(np.asarray(gradient, dtype=np.float32)).all()
    np.testing.assert_array_equal(np.asarray(gradient)[~np.asarray(valid)], 0)
    assert float(jnp.linalg.norm(gradient.astype(jnp.float32))) > 0


def test_equivariance_and_compact_padding():
    model = _model()
    pair = jax.random.normal(jax.random.key(2), (1, 5, 5, 8))
    mask = jnp.array([[True, False, True, False, True]])
    idx = jnp.array([4, 0, 2])
    compact = pair[:, idx][:, :, idx]
    np.testing.assert_allclose(model(compact, jnp.ones((1, 3), bool)),
                               model(pair, mask)[:, idx], atol=2e-6)


def test_project_after_pool_matches_explicit_edge_values_and_gradients():
    model = _model()
    pair = jax.random.normal(jax.random.key(5), (2, 4, 4, 8))
    mask = jnp.array([[True, False, True, True], [True] * 4])

    def reference(z):
        z = model.pair_norm(z)
        keys = mask[:, None, :, None]
        logits = jnp.where(keys, model.score(z), jnp.finfo(jnp.float32).min)
        weights = .5 * (jax.nn.softmax(logits, axis=2) * keys
                        + keys / keys.sum(axis=2, keepdims=True))
        # Intentionally expensive reference: project every edge into every head.
        values = jnp.einsum("bijc,hcr->bijhr", z, model.head_weight[...])
        messages = jnp.einsum("bijh,bijhr->bihr", weights, values).reshape(2, 4, -1)
        return model.output_norm(model.out(messages)) * model.scale[...] * mask[..., None]

    np.testing.assert_allclose(model(pair, mask), reference(pair), atol=2e-6)
    cotangent = jax.random.normal(jax.random.key(6), (2, 4, 16))
    actual_grad = jax.grad(lambda z: (model(z, mask) * cotangent).sum())(pair)
    expected_grad = jax.grad(lambda z: (reference(z) * cotangent).sum())(pair)
    np.testing.assert_allclose(actual_grad, expected_grad, atol=2e-6, rtol=2e-5)


def test_pair_content_survives_identical_single_values():
    model = _model()
    pair = jax.random.normal(jax.random.key(7), (2, 4, 4, 8))
    mask = jnp.ones((2, 4), bool)
    values = jnp.ones((2, 4, 16))

    def bias_only(z):
        weights = jax.nn.softmax(z[..., 0], axis=-1)
        return jnp.einsum("bij,bjc->bic", weights, values)

    cotangent = jax.random.normal(jax.random.key(8), values.shape)
    old_grad = jax.grad(lambda z: (bias_only(z) * cotangent).sum())(pair)
    new_grad = jax.grad(lambda z: (model(z, mask) * cotangent).sum())(pair)
    assert float(jnp.linalg.norm(old_grad)) < 1e-6
    assert float(jnp.linalg.norm(new_grad)) > 1e-3
    assert not np.allclose(model(pair, mask), model(pair[::-1], mask))


def test_shuffle_intervention_preserves_padding_and_only_keeps_singletons():
    from scripts.pair_to_single_pilot import mask_matched_donors

    mask = jnp.array([[1, 1, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [1, 1, 0]], bool)
    donors = mask_matched_donors(mask)
    np.testing.assert_array_equal(donors, [2, 1, 4, 3, 0])
    np.testing.assert_array_equal(mask[donors], mask)


def test_updated_diagonal_transfers_off_diagonal_relations():
    from scripts.pair_to_single_pilot import DiagonalReadout
    from spectra_learning.models.pairmixer_jax import PairOnlyBlock

    block = PairOnlyBlock(pair_dim=8, attention_mlp_multiple=2, norm_eps=1e-5,
                          rngs=nnx.Rngs(11))
    # Learned triangle output projections must be active; their default zero
    # initialization deliberately suppresses triangle updates at step zero.
    block.tri_mul_out.p_out.weight[...] = jnp.eye(8)
    block.tri_mul_in.p_out.weight[...] = jnp.eye(8)
    readout = DiagonalReadout(16, 8, num_heads=2, head_dim=4, rngs=nnx.Rngs(12))
    pair = jax.random.normal(jax.random.key(13), (1, 4, 4, 8))
    mask = jnp.ones((1, 4), bool)
    cotangent = jax.random.normal(jax.random.key(14), (1, 4, 16))
    gradient = jax.grad(lambda z: (readout(block(z, mask), mask) * cotangent).sum())(pair)
    off_diagonal = ~np.eye(4, dtype=bool)
    assert np.linalg.norm(np.asarray(gradient)[0, off_diagonal]) > 1e-3


def test_shared_routing_matches_existing_attention_with_rope_and_padding():
    from scripts.pair_to_single_pilot import pair_attention_weights
    from spectra_learning.models.pairmixer_jax import PairMixerBlock, _linear_with_preferred_acc

    block = PairMixerBlock(single_dim=16, pair_dim=8, num_heads=2,
                           attention_mlp_multiple=2, norm_eps=1e-5, dropout=0,
                           use_fastmixer=True, fastmixer_max_visible_tokens=4, rngs=nnx.Rngs(15))
    module = block.single_attention
    module.o.weight[...] = jnp.eye(16)
    single = jax.random.normal(jax.random.key(16), (2, 4, 16))
    pair = jax.random.normal(jax.random.key(17), (2, 4, 4, 8))
    mask = jnp.array([[True, False, True, True], [False, True, True, True]])
    positions = jnp.array([[1, 4, 9, 31], [3, 7, 8, 31]])

    def reconstructed(z):
        attention = pair_attention_weights(module, single, z, mask, positions)
        normalized = module.single_norm(single)
        qkv = _linear_with_preferred_acc(module.qkv, normalized).reshape(2, 4, 3, 2, 8)
        values = jnp.transpose(qkv[:, :, 2], (0, 2, 1, 3))
        result = jnp.einsum("bhij,bhjd->bhid", attention, values)
        result = jnp.swapaxes(result, 1, 2).reshape(2, 4, 16)
        result *= jax.nn.sigmoid(_linear_with_preferred_acc(module.g, normalized))
        return _linear_with_preferred_acc(module.o, result) * mask[..., None]

    original = lambda z: block._fast_attention_pair_bias_compact(single, z, mask, positions)
    np.testing.assert_allclose(reconstructed(pair), original(pair), atol=2e-6, rtol=2e-5)
    cotangent = jax.random.normal(jax.random.key(18), (2, 4, 16))
    np.testing.assert_allclose(jax.grad(lambda z: (reconstructed(z) * cotangent).sum())(pair),
                               jax.grad(lambda z: (original(z) * cotangent).sum())(pair),
                               atol=2e-6, rtol=2e-5)


def test_no_cls_removes_token_and_pairs_without_changing_shared_initialization():
    from scripts.pair_to_single_pilot import build_model, encode

    enabled = build_model("bias_readout", 0, jnp.float32)
    disabled = build_model("bias_readout", 0, jnp.float32, use_cls_token=False)
    enabled_state = dict(nnx.to_flat_state(nnx.state(enabled)))
    disabled_state = dict(nnx.to_flat_state(nnx.state(disabled)))
    assert set(enabled_state) - set(disabled_state) == {
        ("encoder", name) for name in (
            "cls_token", "cls_to_peak_pair_token", "peak_to_cls_pair_token", "cls_cls_pair_token")
    }
    for path, value in disabled_state.items():
        np.testing.assert_array_equal(value, enabled_state[path])
    assert not disabled.settings.encoder_use_cls_token
    assert not disabled.encoder_use_cls_token
    assert not disabled.encoder.use_cls_token
    valid = jnp.array([[True] * 31, [True, False] * 15 + [True]])
    batch = {"peak_mz": jnp.broadcast_to(jnp.linspace(.01, .8, 31), (2, 31)),
             "peak_intensity": jnp.ones((2, 31)), "peak_valid_mask": valid,
             "precursor_mz": jnp.ones(2)}
    tokens, pair, positions, mask = disabled.encoder.forward_with_pair_compact(
        batch["peak_mz"], batch["peak_intensity"], valid_mask=valid,
        precursor_mz=batch["precursor_mz"], max_visible_tokens=31)
    assert tokens.shape == (2, 31, 128)
    assert pair.shape == (2, 31, 31, 64)
    np.testing.assert_array_equal(mask.sum(1), valid.sum(1))
    assert np.all(np.asarray(positions)[np.asarray(mask)] < 31)
    expected = (tokens * mask[..., None]).sum(1) / mask.sum(1)[:, None]
    # Compare the readout in the same eager execution mode as its reference;
    # fused Fourier-feature evaluation can differ numerically from eager JAX.
    with jax.disable_jit():
        actual = encode(disabled, batch)
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
