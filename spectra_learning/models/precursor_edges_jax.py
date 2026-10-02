"""Acquisition metadata as a precursor node on the joint encoder's pair axis.

With ``encoder_metadata_conditioning="precursor_edge"`` the CLS token stands
for the precursor ion. Its edges to and from each visible peak i receive

    MLP([Fourier(L_i), L_i / 1000, m_i / P, present]) + W c,

where P is the precursor m/z, L_i = max(P - m_i, 0) the neutral loss in Da and
c the embedded ``massive_v2_acquisition_v1`` condition vector. Both output
layers start at zero, so the encoder is unchanged at initialization, and a
dropped precursor zeroes every m/z feature. Among the injection sites
compared in docs/pair_single_dual_20261001.md (Section 11), this was the only
one that improved masked prediction and chemistry probes without costing
retrieval; additive token or FiLM conditioning separates spectra of one
molecule by collision energy and instrument.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from spectra_learning.data.spectra import PEAK_MZ_MAX
from spectra_learning.models.common_jax import Linear
from spectra_learning.models.peak_features_jax import FourierFeatures

Array = jax.Array

# massive_v2_acquisition_v1 columns (spectrum_metadata.py).
PRECURSOR_COLUMN = 0
PRECURSOR_PRESENT_COLUMN = 1


class PrecursorEdgeEmbedder(nnx.Module):
    def __init__(
        self,
        pair_dim: int,
        condition_dim: int,
        *,
        precursor_mz_scale: float,
        num_freqs: int = 16,
        compute_dtype: object = jnp.float32,
        rngs: nnx.Rngs,
    ) -> None:
        self.pair_dim = pair_dim
        self.precursor_mz_scale = precursor_mz_scale
        self.loss_fourier = FourierFeatures(x_min=1e-2, x_max=PEAK_MZ_MAX, num_freqs=num_freqs)
        self.edge_in = Linear(
            self.loss_fourier.num_features() + 3, pair_dim,
            compute_dtype=compute_dtype, rngs=rngs,
        )
        self.edge_out = Linear(
            pair_dim, 2 * pair_dim, init="zeros", compute_dtype=compute_dtype, rngs=rngs,
        )
        self.condition = Linear(
            condition_dim, 2 * pair_dim, init="zeros", compute_dtype=compute_dtype, rngs=rngs,
        )

    def __call__(
        self,
        pair: Array,
        peak_mz: Array,
        spectrum_metadata: Array,
        condition: Array,
        peak_visible_mask: Array,
    ) -> Array:
        """Add edges to ``pair`` [B, N+1, N+1, C], whose last token is CLS."""
        num_peaks = peak_mz.shape[1]
        metadata = spectrum_metadata.astype(jnp.float32)
        present = metadata[:, PRECURSOR_PRESENT_COLUMN:PRECURSOR_PRESENT_COLUMN + 1]
        precursor = metadata[:, PRECURSOR_COLUMN:PRECURSOR_COLUMN + 1] * self.precursor_mz_scale
        mz = peak_mz.astype(jnp.float32) * PEAK_MZ_MAX
        loss = jnp.maximum(precursor - mz, 0.0) * present
        features = jnp.concatenate([
            self.loss_fourier(loss[..., None]) * present[..., None],
            (loss / PEAK_MZ_MAX)[..., None],
            (mz / jnp.maximum(precursor, 1.0) * present)[..., None],
            jnp.broadcast_to(present[..., None], loss.shape + (1,)),
        ], axis=-1)
        edge = self.edge_out(jax.nn.silu(self.edge_in(features)))
        edge = edge + self.condition(condition)[:, None]
        edge = edge * peak_visible_mask[..., None].astype(edge.dtype)
        edge = edge.astype(pair.dtype)
        pair = pair.at[:, :num_peaks, num_peaks].add(edge[..., :self.pair_dim])
        return pair.at[:, num_peaks, :num_peaks].add(edge[..., self.pair_dim:])
