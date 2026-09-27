"""Selected precise-ion/logmixture readout with fragment cross-attention.

One candidate-conditioned compatibility matrix feeds a uniform peak logmixture
and gated reverse MaxSim. Physical query inputs use Da for m/z and adduct mass;
unsupported adducts use retained-mass fractions.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.scipy.special import logsumexp

from spectra_learning.probes.massspec.atom_flare_jax import NEG


H_MONO = 1.00782503207
PROTON_MASS = H_MONO - 0.000548579909065
WIDTH_DA = 0.01
HYDROGEN_SHIFTS = 2
PROTON_PRODUCT_WEIGHT = 0.5
BETA_SCALE = 1.5
ALPHA = 0.65
TEMPERATURE = 0.1
CROSS_ATTENTION_DIM = 32


def _initialize_cross_attention(key, dim):
    attention_dim = CROSS_ATTENTION_DIM
    keys = jax.random.split(key, 5)
    glorot = jax.nn.initializers.glorot_uniform()
    shapes = {"U": (dim, attention_dim), "V": (dim, attention_dim),
              "A_q": (attention_dim, attention_dim), "A_c": (attention_dim, attention_dim),
              "A_x": (attention_dim, attention_dim)}
    params = {name: glorot(subkey, shape, jnp.float32)
              for (name, shape), subkey in zip(shapes.items(), keys, strict=True)}
    params.update(b=jnp.zeros(attention_dim, jnp.float32),
                  O=jnp.zeros((attention_dim, attention_dim), jnp.float32))
    return params


class CrossAttentionAgreement(nn.Module):
    """Zero-output 32-dimensional candidate-conditioned attention adapter."""

    dim: int = 256
    sigma0: float = 0.01
    beta0: float = 0.5

    def setup(self):
        self.log_sigma = self.param("log_sigma", lambda key: jnp.float32(math.log(self.sigma0)))
        self.log_beta = self.param("log_beta", lambda key: jnp.float32(math.log(self.beta0)))
        self.cross_attention = self.param(
            "cross_attention", _initialize_cross_attention, self.dim,
        )

    def __call__(self, fp, ff, fmask):
        delta = jnp.abs(fp[:, None, :, None] - ff[None, :, None, :])
        bonus = jnp.exp(self.log_beta) * jnp.exp(-jnp.square(delta / jnp.exp(self.log_sigma)))
        return jnp.where(fmask[None, :, None, :], bonus, 0.0)


@jax.custom_vjp
def _peak_max(values):
    return jnp.max(values, axis=-2)


def _peak_max_fwd(values):
    # JAX 0.11 CPU can compare differently rounded, recomputed similarities in
    # reduce_max's derivative and lose a unique winner. Sorting materializes
    # the values used to identify every tie. Only differentiation pays this
    # cost; inference retains the ordinary maximum above.
    order = jnp.broadcast_to(jnp.arange(values.shape[-2])[None, :, None], values.shape)
    ordered, order = jax.lax.sort_key_val(values, order, dimension=-2)
    winners = ordered == ordered[:, -1:, :]
    sorted_weights = winners.astype(values.dtype) / winners.sum(axis=-2, keepdims=True)
    weights = jnp.zeros_like(values).at[
        jnp.arange(values.shape[0])[:, None, None], order,
        jnp.arange(values.shape[-1])[None, None, :],
    ].set(sorted_weights)
    return jnp.max(values, axis=-2), weights


def _peak_max_bwd(weights, cotangent):
    return (weights * cotangent[:, None, :],)


_peak_max.defvjp(_peak_max_fwd, _peak_max_bwd)


def _ion_kernel(mz, fraction, mass, adduct_mass):
    predicted = mass[:, None] * fraction + adduct_mass
    delta = mz[None, :, None] - predicted[:, None, :]
    shift = jnp.clip(jnp.rint(delta / H_MONO), -HYDROGEN_SHIFTS, HYDROGEN_SHIFTS)
    return jnp.exp(-jnp.square((delta - shift * H_MONO) / WIDTH_DA))


def score_query(P, pmask, Z, fmask, gate, query: dict[str, jax.Array],
                candidates: dict[str, jax.Array], log_sigma, log_beta, *, cross_attention):
    """Score one query against candidates, returning ``[candidates]``.

    Empty padding returns finite NEG with zero parameter gradients. The learned
    cross-attention residual uses physical compatibility and fragment priors;
    its zero-initialized output matrix preserves the initial base score.
    """
    P = jnp.where(pmask[:, None], P.astype(jnp.float32), 0.0)
    Z = jnp.where(fmask[..., None], Z.astype(jnp.float32), 0.0)
    fp = jnp.where(pmask, query["fraction"], 0.0)
    mz = jnp.where(pmask, query["mz"], 0.0)
    ff = jnp.where(fmask, candidates["fraction"], 0.0)
    active = fmask.any(axis=-1)
    mass = jnp.where(active, candidates["mass"], 1.0)
    adduct_mass = jnp.where(query["eligible"], query["adduct_mass"], 0.0)
    original = jnp.exp(-jnp.square((fp[None, :, None] - ff[:, None, :]) / jnp.exp(log_sigma)))
    parent = _ion_kernel(mz, ff, mass, adduct_mass)
    proton = _ion_kernel(mz, ff, mass, PROTON_MASS)
    kernel = jnp.where(query["eligible"], jnp.maximum(parent, PROTON_PRODUCT_WEIGHT * proton), original)
    bonus = BETA_SCALE * (jnp.exp(log_beta) * kernel)
    sim = jnp.einsum("pd,cfd->cpf", P, Z) + jnp.where(fmask[:, None, :], bonus, 0.0)

    # A private dummy slot makes both reductions differentiable for empty
    # padding. It never contributes to a real candidate or the returned score.
    safe_fragments = fmask | (~active[:, None] & (jnp.arange(fmask.shape[-1]) == 0))
    safe_peaks = pmask | (~pmask.any() & (jnp.arange(pmask.shape[-1]) == 0))
    safe_gate = jnp.where(fmask, gate, 0.0)
    logits = jnp.where(safe_fragments, safe_gate, -jnp.inf)
    log_prior = jax.nn.log_softmax(logits, axis=-1)
    adapter = jax.tree.map(lambda value: value.astype(jnp.float32), cross_attention)

    def rms_norm(value):
        return value / jnp.sqrt(jnp.mean(jnp.square(value), axis=-1, keepdims=True) + 1e-6)

    u = rms_norm(P @ adapter["U"])
    v = rms_norm(Z @ adapter["V"])
    # Fixed attention temperature 1.0; the final readout retains 0.1.
    attention = jax.nn.softmax(sim + log_prior[:, None, :], axis=-1)
    context = jnp.einsum("cpf,cfr->cpr", attention, v)
    hidden = jnp.tanh((u @ adapter["A_q"])[None, :, :]
                      + context @ adapter["A_c"]
                      + (u[None, :, :] * context) @ adapter["A_x"]
                      + adapter["b"])
    hidden = jnp.where(active[:, None, None] & pmask[None, :, None], hidden, 0.0)
    residual = jnp.einsum("cpr,cfr->cpf", hidden @ adapter["O"], v) / jnp.sqrt(jnp.float32(CROSS_ATTENTION_DIM))
    sim = sim + jnp.where(pmask[None, :, None] & fmask[:, None, :], residual, 0.0)
    forward_peaks = TEMPERATURE * logsumexp(sim / TEMPERATURE + log_prior[:, None, :], axis=-1)
    forward = jnp.where(pmask[None, :], forward_peaks, 0.0).sum(axis=-1) / jnp.maximum(pmask.sum(), 1)
    reverse_fragments = _peak_max(jnp.where(safe_peaks[None, :, None], sim, -jnp.inf))
    reverse = (jnp.exp(log_prior) * reverse_fragments).sum(axis=-1)
    result = ALPHA * forward + (1.0 - ALPHA) * reverse
    return jnp.where(active & pmask.any(), result, NEG)


def score_matrix(P, pmask, Z, fmask, gate, query: dict[str, jax.Array],
                 candidates: dict[str, jax.Array], log_sigma, log_beta, chunk_size=16,
                 *, cross_attention):
    """Cross scores [queries,candidates], evaluated in query chunks."""
    def row(inputs):
        peaks, mask, metadata = inputs
        return score_query(peaks, mask, Z, fmask, gate, metadata, candidates, log_sigma, log_beta,
                           cross_attention=cross_attention)

    # Keep the reverse maximum and its operands from the same forward pass.
    # Rematerializing this row produced nonfinite reduce_max gradients when
    # composed with the molecular encoder (JAX 0.11 CPU), despite finite loss.
    return jax.lax.map(row, (P, pmask, query), batch_size=chunk_size)


def score_hardneg(P, pmask, Z, fmask, gate, query: dict[str, jax.Array],
                  candidates: dict[str, jax.Array], log_sigma, log_beta, chunk_size=16,
                  *, cross_attention):
    """Per-query scores [queries,negatives]; Z has [queries,negatives,F,D]."""
    def row(inputs):
        peaks, mask, fragments, fragment_mask, fragment_gate, metadata, molecular = inputs
        return score_query(peaks, mask, fragments, fragment_mask, fragment_gate,
                           metadata, molecular, log_sigma, log_beta,
                           cross_attention=cross_attention)

    return jax.lax.map(row, (P, pmask, Z, fmask, gate, query, candidates),
                       batch_size=chunk_size)
