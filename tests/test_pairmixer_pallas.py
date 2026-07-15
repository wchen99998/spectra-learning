import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spectra_learning.models.pairmixer_pallas import (
    _triangle_input_projections_reference,
    triangle_input_projections,
)


pytestmark = pytest.mark.skipif(
    jax.devices()[0].platform != "tpu",
    reason="PairMixer Pallas kernels require a TPU",
)


def _projection_inputs():
    keys = iter(jax.random.split(jax.random.key(0), 5))
    batch_size = 2
    num_tokens = 9
    pair_dim = 128
    return (
        jax.random.normal(
            next(keys),
            (batch_size, num_tokens, num_tokens, pair_dim),
            dtype=jnp.bfloat16,
        ),
        jnp.ones((pair_dim,), dtype=jnp.float32),
        jnp.zeros((pair_dim,), dtype=jnp.float32),
        jax.random.normal(
            next(keys),
            (2 * pair_dim, pair_dim),
            dtype=jnp.float32,
        )
        * 0.02,
        jnp.zeros((2 * pair_dim,), dtype=jnp.float32),
        jax.random.normal(
            next(keys),
            (2 * pair_dim, pair_dim),
            dtype=jnp.float32,
        )
        * 0.02,
        jnp.ones((2 * pair_dim,), dtype=jnp.float32),
        jax.random.normal(
            next(keys),
            (pair_dim, pair_dim),
            dtype=jnp.float32,
        )
        * 0.02,
        jnp.ones((pair_dim,), dtype=jnp.float32),
    )


def _pallas_projection(*args):
    return triangle_input_projections(
        *args,
        norm_eps=1e-5,
        kernel_role="test",
    )


def _reference_projection(*args):
    return _triangle_input_projections_reference(
        *args,
        norm_eps=1e-5,
    )


def _projection_value_and_grads(function, *args):
    def loss(*values):
        projected, output_gate = function(*values)
        return projected.astype(jnp.float32).mean() + output_gate.astype(
            jnp.float32
        ).mean()

    return jax.value_and_grad(loss, argnums=tuple(range(len(args))))(*args)


def test_pallas_triangle_projection_forward_and_backward_match_xla():
    inputs = _projection_inputs()
    pallas_outputs = jax.jit(_pallas_projection)(*inputs)
    reference_outputs = jax.jit(_reference_projection)(*inputs)

    for actual, expected in zip(pallas_outputs, reference_outputs, strict=True):
        np.testing.assert_allclose(
            np.asarray(actual, dtype=np.float32),
            np.asarray(expected, dtype=np.float32),
            atol=4e-3,
            rtol=2e-2,
        )

    pallas_value, pallas_grads = jax.jit(
        lambda *args: _projection_value_and_grads(_pallas_projection, *args)
    )(*inputs)
    reference_value, reference_grads = jax.jit(
        lambda *args: _projection_value_and_grads(_reference_projection, *args)
    )(*inputs)
    np.testing.assert_allclose(
        np.asarray(pallas_value),
        np.asarray(reference_value),
        atol=1e-4,
        rtol=1e-4,
    )
    for actual, expected in zip(pallas_grads, reference_grads, strict=True):
        np.testing.assert_allclose(
            np.asarray(actual, dtype=np.float32),
            np.asarray(expected, dtype=np.float32),
            atol=2e-6,
            rtol=2e-3,
        )
