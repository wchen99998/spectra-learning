"""Masked summaries and direct relational information in cheap pair readouts."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
nnx = pytest.importorskip("flax.nnx")
pytest.importorskip("torch")

from spectra_learning.models.pair_to_single_jax import PairSummaryToSingleReadout


@pytest.mark.parametrize("summary", ["mean", "cls"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_padding_empty_spectra_and_information_gradient(summary, dtype):
    model = PairSummaryToSingleReadout(16, 8, summary=summary, message_dim=8,
                                      compute_dtype=dtype, rngs=nnx.Rngs(20))
    pair = jax.random.normal(jax.random.key(21), (2, 5, 5, 8)).astype(dtype)
    mask = jnp.array([[True, False, True, True, False], [False] * 5])
    cls_mask = jnp.array([[False, False, True, False, False], [False] * 5])
    valid = mask[:, :, None] & mask[:, None, :]
    actual = model(pair, mask, cls_mask)
    changed = jnp.where(valid[..., None], pair, pair * 20 + 100)
    np.testing.assert_allclose(model(changed, mask, cls_mask).astype(jnp.float32),
                               actual.astype(jnp.float32), atol=1e-6)
    np.testing.assert_array_equal(np.asarray(actual)[~np.asarray(mask)], 0)
    cotangent = jax.random.normal(jax.random.key(22), actual.shape)
    gradient = jax.grad(lambda z: (model(z, mask, cls_mask) * cotangent).sum())(pair)
    assert np.isfinite(np.asarray(gradient, dtype=np.float32)).all()
    np.testing.assert_array_equal(np.asarray(gradient)[~np.asarray(valid)], 0)
    assert np.linalg.norm(np.asarray(gradient, dtype=np.float32)) > 1e-3


@pytest.mark.parametrize("summary", ["mean", "cls"])
def test_compact_permutation_equivalence_and_cls_position(summary):
    model = PairSummaryToSingleReadout(16, 8, summary=summary, message_dim=8, rngs=nnx.Rngs(23))
    pair = jax.random.normal(jax.random.key(24), (2, 5, 5, 8))
    mask = jnp.array([[True, False, True, False, True]] * 2)
    cls_mask = jnp.array([[False, False, True, False, False], [True, False, False, False, False]])
    idx = jnp.array([4, 0, 2])
    compact = pair[:, idx][:, :, idx]
    np.testing.assert_allclose(model(compact, mask[:, idx], cls_mask[:, idx]),
                               model(pair, mask, cls_mask)[:, idx], atol=2e-6)
    if summary == "cls":
        np.testing.assert_array_equal(model(pair, mask, jnp.zeros_like(mask)), 0)


def test_cls_edges_receive_off_edge_content_from_triangle_updates():
    from spectra_learning.models.pairmixer_jax import PairOnlyBlock

    block = PairOnlyBlock(pair_dim=8, attention_mlp_multiple=2, norm_eps=1e-5,
                          rngs=nnx.Rngs(25))
    block.tri_mul_out.p_out.weight[...] = jnp.eye(8)
    block.tri_mul_in.p_out.weight[...] = jnp.eye(8)
    readout = PairSummaryToSingleReadout(16, 8, summary="cls", message_dim=8, rngs=nnx.Rngs(26))
    pair = jax.random.normal(jax.random.key(27), (1, 4, 4, 8))
    mask = jnp.ones((1, 4), bool)
    cls_mask = jnp.array([[False, False, False, True]])
    cotangent = jax.random.normal(jax.random.key(28), (1, 4, 16))
    direct = jax.grad(lambda z: (readout(z, mask, cls_mask) * cotangent).sum())(pair)
    through_trunk = jax.grad(lambda z: (readout(block(z, mask), mask, cls_mask) * cotangent).sum())(pair)
    np.testing.assert_array_equal(np.asarray(direct)[:, :3, :3], 0)
    assert np.linalg.norm(np.asarray(through_trunk)[:, :3, :3]) > 1e-3


def test_mean_reduction_matches_dot_product_forward_and_backward():
    model = PairSummaryToSingleReadout(16, 8, message_dim=8, rngs=nnx.Rngs(29))
    pair = jax.random.normal(jax.random.key(30), (2, 5, 5, 8))
    mask = jnp.array([[True, False, True, True, False], [True] * 5])

    def dot_reference(z):
        pooled = jnp.einsum("bijc,bj->bic", z, mask.astype(jnp.float32))
        pooled /= mask.sum(1)[:, None, None]
        update = model.output_norm(model.out(model.value(model.pair_norm(pooled))))
        return update * model.scale[...] * mask[..., None]

    np.testing.assert_allclose(model(pair, mask), dot_reference(pair), atol=2e-6)
    cotangent = jax.random.normal(jax.random.key(31), (2, 5, 16))
    np.testing.assert_allclose(jax.grad(lambda z: (model(z, mask) * cotangent).sum())(pair),
                               jax.grad(lambda z: (dot_reference(z) * cotangent).sum())(pair),
                               atol=2e-6, rtol=2e-5)
