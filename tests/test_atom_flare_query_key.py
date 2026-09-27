"""Bilinear and normalized compatibility values, gradients and masking."""

from flax import linen as nn, serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from spectra_learning.probes.massspec.atom_flare_jax import (
    QueryKeyProjection,
)


def test_query_key_identity_and_asymmetric_bilinear_gradients():
    rng = np.random.default_rng(27)
    x = rng.normal(size=(3, 5)).astype(np.float32)
    a = rng.normal(size=(4, 5)).astype(np.float32)
    weights = rng.normal(size=(3, 4)).astype(np.float32)
    module = QueryKeyProjection(5)
    initial = module.init(jax.random.key(0), x, a)["params"]
    q, k = module.apply({"params": initial}, x, a)
    np.testing.assert_array_equal(q, x)
    np.testing.assert_array_equal(k, a)

    params = {name: jnp.asarray(rng.normal(size=(5, 5)), jnp.float32) for name in ("query", "key")}

    def loss(p):
        q, k = module.apply({"params": p}, x, a)
        return (weights * (q @ k.T)).sum(), q @ k.T

    (_, actual), grads = jax.value_and_grad(loss, has_aux=True)(params)
    wq, wk = np.asarray(params["query"]), np.asarray(params["key"])
    metric = wq @ wk.T
    assert not np.allclose(metric, metric.T)
    np.testing.assert_allclose(actual, x @ metric @ a.T, rtol=3e-5, atol=2e-5)
    np.testing.assert_allclose(grads["query"], x.T @ weights @ a @ wk, rtol=3e-5, atol=2e-5)
    np.testing.assert_allclose(grads["key"], a.T @ weights.T @ x @ wq, rtol=3e-5, atol=2e-5)
    np.testing.assert_array_equal(module.apply({"params": params}, x, method=module.project_queries), x @ wq)
    np.testing.assert_array_equal(module.apply({"params": params}, a, method=module.project_keys), a @ wk)






def test_default_projection_preserves_legacy_parameters_outputs_and_rng():
    class LegacyProjection(nn.Module):
        dim: int

        def setup(self):
            self.query = self.param("query", lambda key: jnp.eye(self.dim, dtype=jnp.float32))
            self.key = self.param("key", lambda key: jnp.eye(self.dim, dtype=jnp.float32))

        def __call__(self, peaks, atoms):
            return peaks.astype(jnp.float32) @ self.query, atoms.astype(jnp.float32) @ self.key

    class Container(nn.Module):
        projection: nn.Module

        @nn.compact
        def __call__(self, peaks, atoms):
            before = self.param("before", nn.initializers.normal(), (3,))
            q, k = self.projection(peaks, atoms)
            after = self.param("after", nn.initializers.normal(), (3,))
            return q, k, before, after, jax.random.uniform(self.make_rng("dropout"), (3,))

    peaks = jnp.array([[2, -1, 4]], jnp.float32)
    atoms = jnp.array([[0, 3, -2], [1, 0, 0]], jnp.float32)
    rngs = {"params": jax.random.key(19), "dropout": jax.random.key(20)}
    legacy = Container(LegacyProjection(3))
    expected_output, expected_variables = legacy.init_with_output(rngs, peaks, atoms)
    for projection in (QueryKeyProjection(3),):
        output, variables = Container(projection).init_with_output(rngs, peaks, atoms)
        assert serialization.to_bytes(variables) == serialization.to_bytes(expected_variables)
        for actual, expected in zip(output, expected_output):
            np.testing.assert_array_equal(actual, expected)




