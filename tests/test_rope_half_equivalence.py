"""The contiguous rotate-half RoPE must reproduce the old interleaved RoPE.

The interleaved implementation strided the minor-most axis, which TPU lowered to
pad/slice chains costing 51% of a v6e forward+backward pass. Switching to
contiguous halves is only safe because the two forms differ by a fixed
permutation of each head's channels, so an existing checkpoint stays valid after
the q/k projection rows are permuted.
"""

from __future__ import annotations

import unittest

import jax.numpy as jnp
import numpy as np

from spectra_learning.models.pairmixer_jax import (
    _apply_rope,
    permute_fused_qkv_rows_for_rope_half,
    rope_interleaved_to_half_permutation,
)


def _interleaved_rope(tensor: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """The implementation this codebase used before the contiguous rewrite."""
    head_dim = tensor.shape[-1]
    inv_freq = 1.0 / (
        10_000.0 ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim)
    )
    angles = positions[..., None] * inv_freq
    cos = np.cos(angles)[:, None]
    sin = np.sin(angles)[:, None]
    even = tensor[..., 0::2]
    odd = tensor[..., 1::2]
    return np.stack(
        (even * cos - odd * sin, even * sin + odd * cos),
        axis=-1,
    ).reshape(tensor.shape)


class RopeHalfEquivalenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(0)

    def test_permuted_input_matches_interleaved_output(self) -> None:
        for head_dim in (64, 128, 256):
            with self.subTest(head_dim=head_dim):
                x = self.rng.normal(size=(2, 3, 7, head_dim)).astype(np.float32)
                positions = self.rng.integers(0, 50, size=(2, 7)).astype(np.float32)
                permutation = rope_interleaved_to_half_permutation(head_dim)

                interleaved = _interleaved_rope(x, positions)
                contiguous = np.asarray(
                    _apply_rope(jnp.asarray(x[..., permutation]), jnp.asarray(positions))
                )
                np.testing.assert_allclose(
                    contiguous,
                    interleaved[..., permutation],
                    rtol=1e-4,
                    atol=1e-4,
                )

    def test_attention_scores_are_unchanged(self) -> None:
        head_dim = 128
        q = self.rng.normal(size=(2, 3, 7, head_dim)).astype(np.float32)
        k = self.rng.normal(size=(2, 3, 7, head_dim)).astype(np.float32)
        positions = self.rng.integers(0, 50, size=(2, 7)).astype(np.float32)
        permutation = rope_interleaved_to_half_permutation(head_dim)

        old = np.einsum(
            "bhsd,bhtd->bhst",
            _interleaved_rope(q, positions),
            _interleaved_rope(k, positions),
        )
        new = np.einsum(
            "bhsd,bhtd->bhst",
            np.asarray(_apply_rope(jnp.asarray(q[..., permutation]), jnp.asarray(positions))),
            np.asarray(_apply_rope(jnp.asarray(k[..., permutation]), jnp.asarray(positions))),
        )
        np.testing.assert_allclose(new, old, rtol=1e-4, atol=1e-3)

    def test_fused_qkv_permutation_touches_q_and_k_only(self) -> None:
        num_heads, head_dim = 4, 64
        single_dim = num_heads * head_dim
        weight = self.rng.normal(size=(3 * single_dim, single_dim)).astype(np.float32)

        converted = permute_fused_qkv_rows_for_rope_half(weight, num_heads=num_heads)

        original = weight.reshape(3, num_heads, head_dim, single_dim)
        result = converted.reshape(3, num_heads, head_dim, single_dim)
        permutation = rope_interleaved_to_half_permutation(head_dim)
        for component in (0, 1):
            np.testing.assert_array_equal(
                result[component], original[component][:, permutation]
            )
        np.testing.assert_array_equal(result[2], original[2])

    def test_fused_qkv_permutation_preserves_attention_scores(self) -> None:
        num_heads, head_dim, tokens = 4, 64, 5
        single_dim = num_heads * head_dim
        weight = self.rng.normal(size=(3 * single_dim, single_dim)).astype(np.float32)
        x = self.rng.normal(size=(2, tokens, single_dim)).astype(np.float32)
        positions = self.rng.integers(0, 50, size=(2, tokens)).astype(np.float32)

        def scores(w: np.ndarray, rope) -> np.ndarray:
            qkv = (x @ w.T).reshape(2, tokens, 3, num_heads, head_dim)
            q = np.swapaxes(qkv[:, :, 0], 1, 2)
            k = np.swapaxes(qkv[:, :, 1], 1, 2)
            return np.einsum("bhsd,bhtd->bhst", rope(q, positions), rope(k, positions))

        old = scores(weight, _interleaved_rope)
        new = scores(
            permute_fused_qkv_rows_for_rope_half(weight, num_heads=num_heads),
            lambda t, p: np.asarray(_apply_rope(jnp.asarray(t), jnp.asarray(p))),
        )
        np.testing.assert_allclose(new, old, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
