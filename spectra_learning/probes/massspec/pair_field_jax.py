"""Second-order (pair) term of the peak->fragment assignment energy.

The selected readout scores a spectrum s against a molecule G through a
per-peak assignment model: peak i takes fragment f with prior pi_f and
compatibility S_if, and the forward term is the average per-peak log partition

    forward = (1/N) sum_i T log sum_f pi_f exp(S_if / T),   T = 0.1.

The dual encoder also produces pair states z_ij. They define a pairwise
energy Phi_ij(f, g) between the assignments of two peaks, i.e. a pairwise
Markov random field over assignments a_i:

    E(a) = sum_i S_{i a_i} + sum_{(i,j)} Phi_ij(a_i, a_j).

Its log partition, under the mean-field family q(a) = prod_i q_i(a_i) started
from the unary posteriors q_i(f) = softmax_f(S_if / T + log pi_f), expands to

    T log Z ~= sum_i T log Z_i + sum_{(i,j)} E_{q_i q_j}[Phi_ij],

so the readout's forward term gains exactly

    delta = (1/N) sum_{(i,j)} sum_{f,g} q_i(f) q_j(g) Phi_ij(f, g).

Edges are directed parent -> child: a heavier peak i (higher m/z slot) to
each lighter peak j, plus the precursor (CLS token) to every peak. The
precursor is the intact molecule, so its assignment is fixed to the intact
anchor fragment (always slot 0). Phi is a trilinear form between a learned
projection of the pair state and the two fragments' keys,

    Phi_ij(f, g) = <psi_ij, (k_f W_parent) * (k_g W_child)>,

so the expectation factorizes into per-peak expected fragment vectors and
costs O(N^2 r) per candidate instead of O(N^2 F^2). psi starts at zero,
making the initial score identical to the single-stream readout.

Fragmentation-tree consistency (``tree``). In MS/MS a lighter ion j can be a
product of a heavier ion i only if its fragment lies inside the parent's
fragment. Phi therefore also carries rho_ij R_fg with R_fg = 1[g subset f]
(GLACIER atom memberships; every fragment lies inside the intact anchor), and
rho_ij read from the same pair state. Its mean-field expectation is
rho_ij q_i^T R q_j, O(N F^2 + N^2 F) per candidate.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.scipy.special import logsumexp

from spectra_learning.probes.massspec.atom_flare_jax import NEG, PeakEncoder
from spectra_learning.probes.massspec.glacier_ion_logmixture_jax import (
    ALPHA, BETA_SCALE, CROSS_ATTENTION_DIM, PROTON_MASS, PROTON_PRODUCT_WEIGHT, TEMPERATURE,
    _ion_kernel, _peak_max,
)

PAIR_DIM = CROSS_ATTENTION_DIM


def pair_edge_mask(peak_mask: jax.Array) -> jax.Array:
    """[..., N + 1, N] parent->child edges: heavier peak -> lighter peak, CLS -> every peak."""
    n = peak_mask.shape[-1]
    lower = jnp.tril(jnp.ones((n, n), bool), k=-1)
    peaks = peak_mask[..., :, None] & peak_mask[..., None, :] & lower
    cls = jnp.broadcast_to(peak_mask[..., None, :], (*peak_mask.shape[:-1], 1, n))
    return jnp.concatenate([peaks, cls], axis=-2)


class PairFieldPeakEncoder(nn.Module):
    """Per-peak queries (the unchanged PeakEncoder) and zero-initialized edge projections."""

    out_dim: int = 256
    dropout: float = 0.1
    pair_dim: int = PAIR_DIM
    tree: bool = False

    @nn.compact
    def __call__(self, tokens, pair, edge_mask, *, train: bool = False):
        queries = PeakEncoder(self.out_dim, self.dropout, name="peak")(tokens, train=train)
        pair = nn.LayerNorm(epsilon=1e-5, use_fast_variance=False, name="pair_norm")(pair.astype(jnp.float32))
        width = self.pair_dim + int(self.tree)  # last channel: rho_ij for the tree term
        psi = nn.Dense(width, use_bias=False, kernel_init=nn.initializers.zeros, name="psi")(pair)
        psi = psi * edge_mask[..., None]
        return queries, psi


def fragment_subsets(member_mean: jax.Array, fmask: jax.Array) -> jax.Array:
    """[..., F, F] R_fg = 1[atoms(g) subset atoms(f)] for valid fragments f, g."""
    members = (member_mean > 0).astype(jnp.float32)
    outside = jnp.einsum("...ga,...fa->...fg", members, 1.0 - members)
    return (outside == 0) & fmask[..., :, None] & fmask[..., None, :]


def init_pair_field(key, dim: int) -> dict:
    """Fragment role projections; Glorot like the cross-attention adapter."""
    glorot = jax.nn.initializers.glorot_uniform()
    parent, child = jax.random.split(key)
    return {"parent": glorot(parent, (dim, PAIR_DIM), jnp.float32),
            "child": glorot(child, (dim, PAIR_DIM), jnp.float32)}


def score_query(P, pmask, Z, fmask, gate, query, candidates, log_sigma, log_beta, *,
                cross_attention, psi=None, pair_field=None, subsets=None):
    """The selected readout (glacier_ion_logmixture_jax.score_query) plus the pair term.

    ``psi`` is [N + 1, N, r] (parent rows, the last one the precursor; child
    columns), zero off the edge mask. Without ``psi`` the score is identical
    to the single-stream readout.
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

    safe_fragments = fmask | (~active[:, None] & (jnp.arange(fmask.shape[-1]) == 0))
    safe_peaks = pmask | (~pmask.any() & (jnp.arange(pmask.shape[-1]) == 0))
    logits = jnp.where(safe_fragments, jnp.where(fmask, gate, 0.0), -jnp.inf)
    log_prior = jax.nn.log_softmax(logits, axis=-1)
    adapter = jax.tree.map(lambda value: value.astype(jnp.float32), cross_attention)

    def rms_norm(value):
        return value / jnp.sqrt(jnp.mean(jnp.square(value), axis=-1, keepdims=True) + 1e-6)

    u = rms_norm(P @ adapter["U"])
    v = rms_norm(Z @ adapter["V"])
    attention = jax.nn.softmax(sim + log_prior[:, None, :], axis=-1)
    context = jnp.einsum("cpf,cfr->cpr", attention, v)
    hidden = jnp.tanh((u @ adapter["A_q"])[None, :, :] + context @ adapter["A_c"]
                      + (u[None, :, :] * context) @ adapter["A_x"] + adapter["b"])
    hidden = jnp.where(active[:, None, None] & pmask[None, :, None], hidden, 0.0)
    residual = jnp.einsum("cpr,cfr->cpf", hidden @ adapter["O"], v) / jnp.sqrt(jnp.float32(CROSS_ATTENTION_DIM))
    sim = sim + jnp.where(pmask[None, :, None] & fmask[:, None, :], residual, 0.0)
    unary = sim / TEMPERATURE + log_prior[:, None, :]
    forward_peaks = TEMPERATURE * logsumexp(unary, axis=-1)
    peaks = jnp.maximum(pmask.sum(), 1)
    forward = jnp.where(pmask[None, :], forward_peaks, 0.0).sum(axis=-1) / peaks
    if psi is not None:
        parent_keys = Z @ pair_field["parent"].astype(jnp.float32)
        child_keys = Z @ pair_field["child"].astype(jnp.float32)
        psi = psi.astype(jnp.float32)
        anchor = jnp.broadcast_to(jax.nn.one_hot(0, fmask.shape[-1]), (fmask.shape[0], 1, fmask.shape[-1]))
        R = None if subsets is None else subsets.astype(jnp.float32)

        def pair_energy(q):
            """E_q[sum_edges Phi_ij] per candidate for factorized assignments q [C, N, F]."""
            parents = jnp.concatenate([jnp.einsum("cpf,cfr->cpr", q, parent_keys), parent_keys[:, :1]], axis=1)
            children = jnp.einsum("cpf,cfr->cpr", q, child_keys)  # precursor = intact anchor (row N)
            energy = jnp.einsum("ijr,cir,cjr->c", psi[..., :PAIR_DIM], parents, children)
            if R is not None:
                contained = jnp.einsum("cif,cfg->cig", jnp.concatenate([q, anchor], axis=1), R)
                energy = energy + jnp.einsum("ij,cig,cjg->c", psi[..., PAIR_DIM], contained, q)
            return energy

        posterior = jnp.where(pmask[None, :, None], jax.nn.softmax(unary, axis=-1), 0.0)
        forward = forward + pair_energy(posterior) / peaks
    reverse_fragments = _peak_max(jnp.where(safe_peaks[None, :, None], sim, -jnp.inf))
    reverse = (jnp.exp(log_prior) * reverse_fragments).sum(axis=-1)
    result = ALPHA * forward + (1.0 - ALPHA) * reverse
    return jnp.where(active & pmask.any(), result, NEG)


def score_candidates(P, pmask, psi, Z, fmask, gate, query, candidates, log_sigma, log_beta,
                     *, cross_attention, pair_field, chunk_size=1, subsets=None):
    """Per-query candidate scores [queries, K]; Z has [queries, K, F, D]; ``psi`` may be None."""

    def row(inputs):
        peaks, mask, fragments, fragment_mask, fragment_gate, metadata, molecular, *extra = inputs
        return score_query(peaks, mask, fragments, fragment_mask, fragment_gate, metadata, molecular,
                           log_sigma, log_beta, cross_attention=cross_attention,
                           psi=extra[0] if psi is not None else None, pair_field=pair_field,
                           subsets=extra[-1] if subsets is not None else None)

    inputs = ((P, pmask, Z, fmask, gate, query, candidates) + (() if psi is None else (psi,))
              + (() if subsets is None else (subsets,)))
    return jax.lax.map(row, inputs, batch_size=chunk_size)
