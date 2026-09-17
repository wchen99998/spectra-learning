import jax.numpy as jnp
import numpy as np

from spectra_learning.probes.massspec.pair_flare import (
    pair_peak_statistics,
    restore_peak_order,
)


def test_directed_means_exclude_self_cls_and_padding():
    pair = np.full((1, 5, 5, 2), np.nan, np.float32)
    pair[0, 0, 0] = [2, 3]
    pair[0, 2, 2] = [5, 7]
    pair[0, 0, 2] = [11, 13]
    pair[0, 2, 0] = [17, 19]
    pair[0, 4, 4] = [23, 29]
    peak, cls = pair_peak_statistics(jnp.asarray(pair), jnp.array([[1, 0, 1, 0, 1]], bool))
    np.testing.assert_array_equal(peak[0, 0], [2, 3, 11, 13, 17, 19])
    np.testing.assert_array_equal(peak[0, 2], [5, 7, 17, 19, 11, 13])
    np.testing.assert_array_equal(peak[0, [1, 3]], 0)
    np.testing.assert_array_equal(cls[0], [23, 29, 23, 29, 23, 29])


def test_peak_permutation_equivariance_and_projection_commutation():
    rng = np.random.default_rng(17)
    pair = jnp.asarray(rng.normal(size=(2, 6, 6, 4)).astype(np.float32))
    valid = jnp.array([[1, 1, 0, 1, 1, 1], [1, 0, 1, 0, 1, 1]], bool)
    order = np.array([3, 0, 4, 2, 1, 5])
    peak, cls = pair_peak_statistics(pair, valid)
    changed, changed_cls = pair_peak_statistics(pair[:, order][:, :, order], valid[:, order])
    np.testing.assert_allclose(changed, peak[:, order[:-1]], atol=1e-6)
    np.testing.assert_allclose(changed_cls, cls)
    projection = jnp.asarray(rng.normal(size=(4, 3)).astype(np.float32))
    projected, _ = pair_peak_statistics(pair @ projection, valid)
    expected = (peak.reshape(2, 5, 3, 4) @ projection).reshape(2, 5, 9)
    np.testing.assert_allclose(projected, expected, atol=1e-6)


def test_no_neighbors_have_zero_messages():
    pair = jnp.ones((2, 4, 4, 2))
    valid = jnp.array([[1, 0, 0, 1], [0, 0, 0, 1]], bool)
    peak, cls = pair_peak_statistics(pair, valid)
    np.testing.assert_array_equal(peak[:, :, 2:], 0)
    np.testing.assert_array_equal(peak[1], 0)
    assert np.isfinite(peak).all() and np.isfinite(cls).all()


def test_scatter_restores_gaps_without_padded_index_overwrite():
    values = jnp.array([[[1., 2.], [3., 4.], [99., 99.]]])
    result = restore_peak_order(values, jnp.array([[2, 0, 0]]), jnp.array([[1, 1, 0]], bool), 4)
    np.testing.assert_array_equal(result[0], [[3, 4], [0, 0], [1, 2], [0, 0]])
