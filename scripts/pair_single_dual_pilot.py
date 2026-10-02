"""Joint single/pair pilots where the pair stream receives the distogram gradient.

The historical joint MAE path discards the encoder's pair tensor: its
predictor reads only single tokens, so pairs learn only through their
attention-bias influence on singles. This pilot attaches the pair-only
distogram predictor directly to the joint encoder's pair output (``dual``
objective) and compares pair->single interfaces on top of it, against
single-only, pair-only and the historical joint recipe. All arms share the
same examples, masks and optimizer schedule; chemical labels are used only by
frozen downstream probes. Run from the repository root.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from scripts import pair_single_dual_metadata as metadata_lib
from scripts.pair_to_single_pilot import (
    SharedAttentionReadout, mask_batch, mask_matched_donors, pair_attention_weights,
)
from spectra_learning.models.common_jax import Linear, RMSNorm
from spectra_learning.models.model_jax import PeakSetJEPAJax
from spectra_learning.models.pair_to_single_jax import PairSummaryToSingleReadout
from spectra_learning.models.pairmixer_jax import (
    _active_indices, _gather_pair, _gather_single, run_pair_only_blocks,
)
from spectra_learning.models.peak_features_jax import FourierFeatures
from spectra_learning.models.spectrum_metadata import MASSIVE_V2_ACQUISITION_SCHEMA

NUM_PEAKS = 31
PEAK_MZ_SCALE = 1000.0  # panel m/z values are divided by PEAK_MZ_MAX
CAPACITIES = (20, 28, 8)  # encoder (with CLS), predictor (with CLS), targets
FEATURE_FOLDS = ("train", "val")

# objective: "old" = 0.7 MAE + 0.3 distogram from single predictor latents
# (historical joint recipe); "dual" = 0.7 MAE on singles + 0.3 distogram from
# the pair predictor on encoder pairs; "dualplus" = dual plus the historical
# 0.3 single-latent distogram; "pair" = pair-only distogram.
ARMS = {
    "single_only": ("old", "none"),
    "joint_bias": ("old", "bias"),
    "pair_only": ("pair", None),
    "dual_none": ("dual", "none"),
    "dual_bias": ("dual", "bias"),
    "dual_bias_mean": ("dual", "bias_mean"),
    "dual_bias_ipa": ("dual", "bias_ipa"),
    "dual_bias_gated": ("dual", "bias_gated"),
    "dual_gated": ("dual", "gated"),
    "dual_bias_mean_gated": ("dual", "bias_mean_gated"),
    "dualplus_bias": ("dualplus", "bias"),
    "dualplus_bias_gated": ("dualplus", "bias_gated"),
}


class EdgeGatedMessage(nnx.Module):
    """Channel-wise pair-gated neighbor messages plus mean-first pair values.

    m_i = mean_j sigmoid(G z_ij) * (V s_j) + P mean_j RMSNorm(z_ij). Unlike the
    attention bias (one scalar per edge and head), each edge gates ``rank``
    message channels, and the second term transports pair content even when
    neighbor singles are identical.
    """

    def __init__(self, single_dim: int, pair_dim: int, *, rank: int = 64,
                 residual_scale: float = 0.1, compute_dtype=jnp.float32,
                 rngs: nnx.Rngs):
        self.compute_dtype = compute_dtype
        self.pair_norm = RMSNorm(pair_dim, affine=False)
        self.single_norm = RMSNorm(single_dim, affine=False)
        self.gate = Linear(pair_dim, rank, compute_dtype=compute_dtype, rngs=rngs)
        self.value = Linear(single_dim, rank, bias=False, compute_dtype=compute_dtype, rngs=rngs)
        self.pair_value = Linear(pair_dim, rank, bias=False, compute_dtype=compute_dtype, rngs=rngs)
        self.out = Linear(rank, single_dim, bias=False, compute_dtype=compute_dtype, rngs=rngs)
        self.output_norm = RMSNorm(single_dim, affine=False)
        self.scale = nnx.Param(jnp.full((single_dim,), residual_scale, jnp.float32))

    def __call__(self, single, pair, mask):
        keys = mask[:, None, :, None]
        count = jnp.maximum(mask.sum(1), 1)[:, None, None].astype(jnp.float32)
        normalized = self.pair_norm(pair).astype(self.compute_dtype)
        gates = jax.nn.sigmoid(self.gate(normalized).astype(jnp.float32))
        values = self.value(self.single_norm(single)).astype(jnp.float32)
        gated = jnp.sum(jnp.where(keys, gates * values[:, None], 0), axis=2) / count
        mean_pair = jnp.sum(jnp.where(keys, normalized, 0), axis=2, dtype=jnp.float32) / count
        message = gated + self.pair_value(mean_pair.astype(self.compute_dtype)).astype(jnp.float32)
        update = self.output_norm(self.out(message.astype(self.compute_dtype)))
        return update * self.scale[...].astype(update.dtype) * mask[..., None]


class BridgedBlock(nnx.Module):
    """Joint block followed by optional pair->single residual bridges."""

    def __init__(self, base, bridges: dict):
        self.base = base
        self.bridges = nnx.Dict(bridges)
        self.intervention = "intact"

    def fastmixer_compact_only_call(self, single, pair, mask, positions=None):
        single_input = single
        single, pair = self.base.fastmixer_compact_only_call(single, pair, mask, positions)
        if self.intervention == "zero":
            return single, pair
        values = pair[mask_matched_donors(mask)] if self.intervention == "shuffle" else pair
        update = 0.0
        for name, bridge in self.bridges.items():
            if name == "ipa":
                attention = pair_attention_weights(
                    self.base.single_attention, single_input, pair, mask, positions)
                update = update + bridge(values, mask, attention)
            elif name == "gated":
                update = update + bridge(single, values, mask)
            else:
                update = update + bridge(values, mask)
        return single + update, pair


# Joint-encoder metadata injection sites. "token" is the production additive
# mode (all single tokens and CLS, before the pair embedder reads singles);
# "final" is what production AdaLN-Zero reduces to on the fastmixer path (a
# modulation of the final single norm only); "film_*" modulate a stream at
# every block input; "cls_edge" makes CLS a precursor node whose edges carry
# each peak's neutral loss.
META_SITES = ("token", "cls", "pair_in", "film_single", "film_pair", "cls_edge", "final")


class MetadataInjection(nnx.Module):
    """Metadata embedding with zero-initialised heads, one per injection site.

    Every head starts at zero, so at initialisation each site leaves the
    encoder exactly as without metadata; arms differ only in where the
    condition can enter.
    """

    def __init__(self, in_dim: int, sites: tuple[str, ...], *, single_dim: int, pair_dim: int,
                 layers: int, dim: int = 256, compute_dtype=jnp.bfloat16, rngs: nnx.Rngs):
        unknown = set(sites) - set(META_SITES)
        if unknown:
            raise ValueError(f"unknown metadata sites {sorted(unknown)}")
        self.sites = tuple(sites)
        self.pair_dim = pair_dim
        self.embed = nnx.List([Linear(in_dim, dim, compute_dtype=compute_dtype, rngs=rngs),
                               Linear(dim, dim, compute_dtype=compute_dtype, rngs=rngs)])

        def head(out, site):
            return (Linear(dim, out, init="zeros", compute_dtype=compute_dtype, rngs=rngs)
                    if site in sites else None)

        self.token = head(single_dim, "token")
        self.cls = head(single_dim, "cls")
        self.pair_in = head(pair_dim, "pair_in")
        self.final = head(2 * single_dim, "final")
        self.film_single = (nnx.List([head(2 * single_dim, "film_single") for _ in range(layers)])
                            if "film_single" in sites else None)
        self.film_pair = (nnx.List([head(2 * pair_dim, "film_pair") for _ in range(layers)])
                          if "film_pair" in sites else None)
        if "cls_edge" in sites:
            self.loss_fourier = FourierFeatures(x_min=1e-2, x_max=PEAK_MZ_SCALE, num_freqs=16)
            width = self.loss_fourier.num_features() + 3
            self.edge = nnx.List([
                Linear(width, pair_dim, compute_dtype=compute_dtype, rngs=rngs),
                Linear(pair_dim, 2 * pair_dim, init="zeros", compute_dtype=compute_dtype, rngs=rngs)])
            self.edge_condition = head(2 * pair_dim, "cls_edge")
        else:
            self.loss_fourier = self.edge = self.edge_condition = None

    def condition(self, metadata):
        return self.embed[1](jax.nn.silu(self.embed[0](metadata)))

    def cls_edges(self, pair, peak_mz, metadata, c, visible):
        """Add neutral-loss features to peak->CLS and CLS->peak edges."""
        n = peak_mz.shape[1]
        present = metadata[:, 1:2].astype(jnp.float32)
        precursor = metadata[:, 0:1].astype(jnp.float32) * PEAK_MZ_SCALE
        mz = peak_mz.astype(jnp.float32) * PEAK_MZ_SCALE
        loss = jnp.maximum(precursor - mz, 0.0) * present
        features = jnp.concatenate([
            self.loss_fourier(loss[..., None]) * present[..., None],
            (loss / PEAK_MZ_SCALE)[..., None],
            (mz / jnp.maximum(precursor, 1.0) * present)[..., None],
            jnp.broadcast_to(present[..., None], loss.shape + (1,)),
        ], axis=-1)
        edge = self.edge[1](jax.nn.silu(self.edge[0](features.astype(self.edge[0].compute_dtype))))
        edge = (edge + self.edge_condition(c)[:, None]) * visible[..., None].astype(edge.dtype)
        edge = edge.astype(pair.dtype)
        pair = pair.at[:, :n, n].add(edge[..., :self.pair_dim])
        return pair.at[:, n, :n].add(edge[..., self.pair_dim:])


def _film(x, head, c, mask):
    scale, shift = jnp.split(head(c).astype(x.dtype), 2, axis=-1)
    extra = (1,) * (x.ndim - 2)
    scale = scale.reshape(scale.shape[0], *extra, scale.shape[-1])
    shift = shift.reshape(shift.shape[0], *extra, shift.shape[-1])
    return (x * (1 + scale) + shift) * mask[..., None].astype(x.dtype)


class RegisterTokens(nnx.Module):
    """Learned register tokens for the joint encoder.

    Registers are appended after compaction, so the encoder axis grows from
    the peak+CLS capacity T to T + R. They take RoPE positions after CLS, have
    learned pair rows and columns (z[i, r], z[r, j], and a shared z[r, r']
    for register-register edges), and join every block. Like ViT registers
    they are dropped before the predictors and readouts, so they can only act
    as scratch space for the peaks and CLS.
    """

    def __init__(self, count: int, single_dim: int, pair_dim: int, *, rngs: nnx.Rngs):
        def normal(shape):
            return nnx.Param(rngs.params.normal(shape, dtype=jnp.float32) * 0.02)

        self.count = count
        self.single = normal((count, single_dim))
        self.to_register = normal((count, pair_dim))
        self.from_register = normal((count, pair_dim))
        self.between = normal((pair_dim,))

    def append(self, x, z, idx, mask, first_position):
        batch, tokens = mask.shape
        count, width = self.count, z.shape[-1]
        x = jnp.concatenate([x, jnp.broadcast_to(
            self.single[...].astype(x.dtype), (batch, count, x.shape[-1]))], axis=1)
        column = jnp.broadcast_to(self.to_register[...].astype(z.dtype), (batch, tokens, count, width))
        row = jnp.broadcast_to(self.from_register[...].astype(z.dtype)[:, None],
                               (batch, count, tokens, width))
        corner = jnp.broadcast_to(self.between[...].astype(z.dtype), (batch, count, count, width))
        z = jnp.concatenate([jnp.concatenate([z, column], axis=2),
                             jnp.concatenate([row, corner], axis=2)], axis=1)
        positions = first_position + jnp.arange(count, dtype=idx.dtype)
        idx = jnp.concatenate([idx, jnp.broadcast_to(positions, (batch, count))], axis=1)
        mask = jnp.concatenate([mask, jnp.ones((batch, count), jnp.bool_)], axis=1)
        return x, z, idx, mask


def encode_joint(encoder, meta, peak_mz, peak_intensity, *, valid_mask, visible_mask,
                 max_visible_tokens, precursor_mz, metadata=None, registers=None):
    """``forward_with_pair_compact`` with metadata injection sites and registers.

    With ``meta`` or ``metadata`` None and no registers this is the same
    computation as the encoder's own compact forward (checked in the tests).
    ``precursor_mz`` None drops the precursor from the pair features
    (reference mass = largest peak). Register slots are removed from the
    returned tensors.
    """
    c = None if meta is None or metadata is None else meta.condition(metadata)
    use = (lambda name: c is not None and name in meta.sites)
    x, peak_visible_mask, _ = encoder._embed_peaks(
        peak_mz, peak_intensity, valid_mask, visible_mask, None)
    cls_extra = None
    if use("token"):
        token = meta.token(c).astype(x.dtype)
        x = x + token[:, None]
        cls_extra = token
    if use("cls"):
        cls = meta.cls(c).astype(x.dtype)
        cls_extra = cls if cls_extra is None else cls_extra + cls
    z = encoder.pair_embedder(encoder._quantize_mz(peak_mz), peak_intensity, x,
                              peak_visible_mask, precursor_mz=precursor_mz)
    x = encoder._append_cls_token(x, cls_extra)
    z = encoder._append_cls_pair_tokens(z)
    if use("pair_in"):
        z = z + meta.pair_in(c).astype(z.dtype)[:, None, None]
    if use("cls_edge"):
        z = meta.cls_edges(z, peak_mz, metadata, c, peak_visible_mask)
    token_visible_mask = encoder._append_cls_mask(peak_visible_mask)
    idx, mask = _active_indices(token_visible_mask, max_visible_tokens
                                or encoder.pairmixer_fast_max_visible_tokens)
    pair_mask = mask[:, :, None] & mask[:, None, :]
    x = _gather_single(x, idx)
    x = x * mask[..., None].astype(x.dtype)
    z = _gather_pair(z, idx)
    if use("pair_in") or use("cls_edge"):
        z = z * pair_mask[..., None].astype(z.dtype)
    capacity = mask.shape[1]
    if registers is not None:
        x, z, idx, mask = registers.append(x, z, idx, mask, peak_mz.shape[1] + 1)
        pair_mask = mask[:, :, None] & mask[:, None, :]
    for layer, block in enumerate(encoder.blocks):
        if use("film_single"):
            x = _film(x, meta.film_single[layer], c, mask)
        if use("film_pair"):
            z = _film(z, meta.film_pair[layer], c, pair_mask)
        x, z = block.fastmixer_compact_only_call(x, z, mask, idx)
    x = encoder._finalize(x, None)
    if use("final"):
        x = _film(x, meta.final, c, mask)
    x = x * mask[..., None].astype(x.dtype)
    if encoder.final_pair_norm is not None:
        z = encoder.final_pair_norm(z)
    z = z * pair_mask[..., None].astype(z.dtype)
    return x[:, :capacity], z[:, :capacity, :capacity], idx[:, :capacity], mask[:, :capacity]


def settings(args, *, mode: str, distogram_weight: float, mae_weight: float, seed: int,
             native_metadata: bool = False):
    width = args.pair_dim if mode == "pair_distogram" else args.single_dim
    # Production pair-only conditioning: AdaLN-Zero in every pair block.
    metadata = dict(encoder_metadata_schema=MASSIVE_V2_ACQUISITION_SCHEMA,
                    encoder_metadata_conditioning="adaln_zero",
                    encoder_metadata_condition_dim=args.meta_dim) if native_metadata else {}
    return dict(**metadata,
        training_mode=mode, model_dim=width, predictor_dim=width,
        encoder_num_layers=args.layers, encoder_num_heads=args.heads,
        attention_mlp_multiple=8 / 3, feature_mlp_hidden_dim=2 * args.single_dim,
        encoder_fourier_mlp_hidden_dim=2 * args.single_dim, encoder_fourier_num_freqs=16,
        pairmixer_fourier_num_freqs=16, pairmixer_pair_dim=args.pair_dim,
        pairmixer_pair_feature_hidden_dim=2 * args.pair_dim,
        pairmixer_block_type="fastmixer", pairmixer_transition_type="swiglu",
        encoder_use_pair_path=True, encoder_apply_final_pair_norm=True,
        masked_latent_predictor_num_layers=args.predictor_layers,
        masked_latent_predictor_num_heads=args.heads,
        masked_token_input_mode="latent_token", num_peaks=NUM_PEAKS,
        encoder_use_cls_token=True, jepa_num_target_blocks=1,
        jepa_mae_mz_bin_size=args.mz_bin_size, mae_intensity_loss_weight=0.0,
        mae_loss_weight=mae_weight, jepa_mae_loss_weight=0.0, masked_token_loss_weight=0.0,
        distogram_loss_weight=distogram_weight, predictor_apply_final_norm=True,
        target_projector_dim=-1, use_ema_teacher=False,
        autocast_dtype="bf16", rngs=nnx.Rngs(seed),
    )


class PilotModel(nnx.Module):
    def __init__(self, arm: str, seed: int, args):
        objective, interface = ARMS[arm]
        self.objective = objective
        fields = metadata_lib.parse_fields(args.meta_fields)
        sites = tuple(x for x in args.meta_sites.split("+") if x)
        native = sites == ("native",)
        if sites and not fields:
            raise ValueError("metadata sites need --meta-fields")
        if native and (objective != "pair" or "adduct" in fields):
            raise ValueError("native metadata is the pair-only AdaLN-Zero path (28-value schema)")
        if sites and not native and objective not in {"dual", "dualplus"}:
            raise ValueError("joint metadata sites are implemented for the dual objectives")
        if args.pair_precursor == "off" and objective not in {"dual", "dualplus"}:
            raise ValueError("--pair-precursor off is implemented for the dual objectives")
        self.use_precursor = args.pair_precursor == "on"
        heads = tuple(x for x in args.offset_heads.split("+") if x)
        if set(heads) - {"mz", "dist"}:
            raise ValueError("--offset-heads is a '+'-joined subset of ('mz', 'dist')")
        if heads and objective not in {"dual", "pair"}:
            raise ValueError("offset heads are implemented for the dual and pair-only objectives")
        if "mz" in heads and objective == "pair":
            raise ValueError("pair-only has no m/z head")
        if objective != "pair":
            self.joint = PeakSetJEPAJax(**settings(
                args, mode="mae", seed=seed, mae_weight=0.7,
                distogram_weight=0.0 if objective == "dual" else 0.3))
            self.joint.set_fastmixer_capacities(*CAPACITIES)
        if objective != "old":
            self.pair = PeakSetJEPAJax(**settings(
                args, mode="pair_distogram", seed=seed + 500, mae_weight=0.0,
                distogram_weight=1.0 if objective == "pair" else 0.3, native_metadata=native))
            self.pair.set_fastmixer_capacities(*CAPACITIES)
            if objective in {"dual", "dualplus"}:
                # Only the pair predictor and distogram head are used; the joint
                # encoder supplies the pair tensor.
                self.pair.encoder = None
        if objective != "old":
            self.offsets = OffsetHeads(
                heads, args.offset_weight, single_dim=args.single_dim, pair_dim=args.pair_dim,
                mz_bins=self.joint.jepa_mae_num_mz_bins if objective != "pair" else 1,
                distance_bins=self.pair.distogram_num_bins, compute_dtype=jnp.bfloat16,
                rngs=nnx.Rngs(seed + 4000)) if heads else None
        if args.registers and objective not in {"dual", "dualplus"}:
            raise ValueError("registers are implemented for the dual objectives")
        if objective == "pair":
            self.meta = None
            self.registers = None
            return
        dtype = jnp.bfloat16
        blocks = []
        for i, base in enumerate(self.joint.encoder.blocks):
            base.single_attention.use_pair_bias = interface.startswith("bias")
            rngs = nnx.Rngs(seed + 1000 + i)
            bridges = {}
            if "mean" in interface:
                bridges["mean"] = PairSummaryToSingleReadout(
                    args.single_dim, args.pair_dim, summary="mean",
                    message_dim=args.bridge_rank, compute_dtype=dtype, rngs=rngs)
            if "ipa" in interface:
                bridges["ipa"] = SharedAttentionReadout(
                    args.single_dim, args.pair_dim, num_heads=args.heads,
                    head_dim=args.bridge_rank // args.heads, compute_dtype=dtype, rngs=rngs)
            if "gated" in interface:
                bridges["gated"] = EdgeGatedMessage(
                    args.single_dim, args.pair_dim, rank=args.bridge_rank,
                    compute_dtype=dtype, rngs=rngs)
            if not base.single_attention.use_pair_bias:
                base.single_attention.pair_bias = None
                base.single_attention.pair_norm = None
            blocks.append(BridgedBlock(base, bridges))
        self.joint.encoder.blocks = nnx.List(blocks)
        # Separate RNG stream: the encoder, bridges and heads initialise
        # exactly as in the no-metadata arm.
        self.meta = MetadataInjection(
            metadata_lib.condition_dim(fields), sites, single_dim=args.single_dim,
            pair_dim=args.pair_dim, layers=args.layers, dim=args.meta_dim,
            compute_dtype=dtype, rngs=nnx.Rngs(seed + 3000)) if sites else None
        self.registers = RegisterTokens(
            args.registers, args.single_dim, args.pair_dim,
            rngs=nnx.Rngs(seed + 5000)) if args.registers else None

    def set_intervention(self, name: str):
        if self.objective != "pair":
            for block in self.joint.encoder.blocks:
                block.intervention = name

    def loss(self, batch):
        if self.objective == "old":
            return self.joint.forward_mae(batch)
        if self.objective == "pair":
            if self.offsets is None:
                return self.pair.forward_pair_distogram(batch)
            loss, metrics, offset = pair_forward(self.pair, batch, self.offsets.dist)
            return {**metrics, "loss": loss + self.offsets.weight * offset}
        return dual_forward(self.joint, self.pair, batch, self.meta, self.use_precursor,
                            self.offsets if self.objective == "dual" else None, self.registers)

    def encode(self, batch):
        """Per-spectrum readout pieces from the fully visible spectrum."""
        valid = batch["peak_valid_mask"]
        common = dict(valid_mask=valid, visible_mask=valid, max_visible_tokens=NUM_PEAKS + 1)
        metadata = batch.get("spectrum_metadata")
        out = {}
        if self.objective == "pair":
            pair, idx, mask = self.pair.encoder(
                batch["peak_mz"], batch["peak_intensity"], **common,
                precursor_mz=batch["precursor_mz"], spectrum_metadata=metadata)
        else:
            single, pair, idx, mask = encode_joint(
                self.joint.encoder, self.meta, batch["peak_mz"], batch["peak_intensity"], **common,
                precursor_mz=batch["precursor_mz"] if self.use_precursor else None,
                metadata=metadata, registers=self.registers)
            single = single.astype(jnp.float32)
            peaks = mask & (idx < NUM_PEAKS)
            cls = mask & (idx == NUM_PEAKS)
            out["single_mean"] = masked_mean(single, peaks)
            out["single_cls"] = (single * cls[..., None]).sum(1)
        pair = pair.astype(jnp.float32)
        peaks = mask & (idx < NUM_PEAKS)
        cls = (mask & (idx == NUM_PEAKS)).astype(jnp.float32)
        edges = peaks[:, :, None] & peaks[:, None, :]
        count = jnp.maximum(edges.sum((1, 2)), 1)[:, None]
        out["pair_mean"] = (pair * edges[..., None]).sum((1, 2)) / count
        to_cls = jnp.einsum("bijc,bj->bic", pair, cls)
        from_cls = jnp.einsum("bijc,bi->bjc", pair, cls)
        out["pair_to_cls"] = masked_mean(to_cls, peaks)
        out["pair_from_cls"] = masked_mean(from_cls, peaks)
        return out


def masked_mean(x, mask):
    return (x * mask[..., None]).sum(1) / jnp.maximum(mask.sum(1), 1)[:, None]


class BinOffsetHead(nnx.Module):
    """Regresses where the true value sits inside its bin.

    For a value x and bin width Delta, the bin is k = floor(x / Delta) and the
    target is u = x / Delta - k in [0, 1). With per-bin weights on shared
    features, u_k(h) = sigmoid(w_k . g(h) + b_k); only the scored bin's row is
    gathered, so the cost does not grow with the number of bins. Zero rows
    start every prediction at the bin centre.
    """

    def __init__(self, dim: int, num_bins: int, *, compute_dtype, rngs: nnx.Rngs):
        self.hidden = Linear(dim, dim, compute_dtype=compute_dtype, rngs=rngs)
        self.weight = nnx.Param(jnp.zeros((num_bins, dim), jnp.float32))
        self.bias = nnx.Param(jnp.zeros((num_bins,), jnp.float32))

    def __call__(self, features, bins):
        shared = jax.nn.silu(self.hidden(features)).astype(jnp.float32)
        logit = (shared * self.weight[...][bins]).sum(-1) + self.bias[...][bins]
        return jax.nn.sigmoid(logit)


def bin_offsets(value, bin_size, num_bins):
    """(clipped bin, within-bin position, in-range mask) for values in Da."""
    scaled = value.astype(jnp.float32) / bin_size
    floor = jnp.floor(scaled)
    return (jnp.clip(floor.astype(jnp.int32), 0, num_bins - 1), scaled - floor,
            floor <= num_bins - 1)


class OffsetHeads(nnx.Module):
    def __init__(self, heads: tuple[str, ...], weight: float, *, single_dim, pair_dim,
                 mz_bins, distance_bins, compute_dtype, rngs: nnx.Rngs):
        self.weight = weight
        self.mz = (BinOffsetHead(single_dim, mz_bins, compute_dtype=compute_dtype, rngs=rngs)
                   if "mz" in heads else None)
        self.dist = (BinOffsetHead(pair_dim, distance_bins, compute_dtype=compute_dtype, rngs=rngs)
                     if "dist" in heads else None)


def _weighted_mean(values, mask):
    weights = mask.astype(jnp.float32)
    return (values * weights).sum() / jnp.maximum(weights.sum(), 1.0)


def pair_distogram_with_offsets(model: PeakSetJEPAJax, pair, idx, compact_token_mask,
                                target_union, peak_mz, head=None):
    """``_distogram_loss_pair_branch`` that can also regress within-bin distance.

    Same computation and losses as the library method (checked in the tests);
    with ``head`` it adds ``distogram_offset_loss``, the L1 error of the
    within-bin position of each supervised distance, in bin units.
    """
    predicted, idx, compact_token_mask = model._pair_predictor_inputs(
        pair, idx, compact_token_mask, target_union)
    num_cls = int(model.encoder_use_cls_token)
    slot_is_peak = compact_token_mask & (idx < peak_mz.shape[1])
    slot_is_target = jnp.take_along_axis(
        jnp.pad(target_union, ((0, 0), (0, num_cls))), idx, axis=1) & slot_is_peak
    slot_mz = jnp.take_along_axis(jnp.pad(peak_mz, ((0, 0), (0, num_cls))), idx, axis=1)
    predicted = run_pair_only_blocks(
        model.masked_latent_predictor, predicted, compact_token_mask, None,
        checkpoint_mode=(model.activation_checkpoint_mode
                         if "predictor" in model.activation_checkpoint_modules else "none"),
        checkpoint_every_n=model.activation_checkpoint_every_n_layers)
    if model.predictor_pair_norm is not None:
        predicted = model.predictor_pair_norm(predicted)
    target_slots, target_slot_mask = _active_indices(slot_is_target, min(
        predicted.shape[1],
        model.jepa_num_target_blocks * model.pairmixer_fast_target_max_visible_tokens))
    batch_size, num_tokens, _, pair_dim = predicted.shape
    rows_pair = _gather_single(
        predicted.reshape(batch_size, num_tokens, num_tokens * pair_dim), target_slots,
    ).reshape(batch_size, target_slots.shape[1], num_tokens, pair_dim)
    row_mz = jnp.take_along_axis(slot_mz, target_slots, axis=1)
    row_positions = jnp.take_along_axis(idx, target_slots, axis=1)
    pair_mask = (target_slot_mask[:, :, None] & slot_is_peak[:, None, :]
                 & (row_positions[:, :, None] != idx[:, None, :]))
    distance = jnp.abs(row_mz[:, :, None] - slot_mz[:, None, :]) * model.distogram_mz_max
    targets, position, in_range = bin_offsets(
        distance, model.jepa_mae_mz_bin_size, model.distogram_num_bins)
    features = model._distogram_pair_latents(rows_pair)
    rows = model._distogram_score(features, targets)
    losses = {("distogram_loss" if width == 1 else f"distogram_ce_{width}bins"):
              _weighted_mean(row, pair_mask) for width, row in rows.items()}
    if head is not None:
        losses["distogram_offset_loss"] = _weighted_mean(
            jnp.abs(head(features, targets) - position), pair_mask & in_range)
    return losses


def mz_offset_metrics(joint: PeakSetJEPAJax, predicted, target_mz, target_mask, head=None):
    """Within-bin m/z regression loss, and m/z error in mDa where the argmax bin is right.

    Whole-target mean errors are dominated by misclassified bins (often tens
    of Da away), so the mDa errors are taken over correctly binned targets:
    ``mz_center_error_mda`` puts the prediction at the bin centre and
    ``mz_abs_error_mda`` at the regressed position. ``mz_bin_accuracy`` is
    the fraction of targets in that set.
    """
    bin_size = joint.jepa_mae_mz_bin_size
    mz_da = target_mz.astype(jnp.float32) * joint.jepa_mae_mz_max
    bins, position, in_range = bin_offsets(mz_da, bin_size, joint.jepa_mae_num_mz_bins)
    mask = target_mask & in_range
    argmax = jnp.argmax(joint.jepa_mae_mz_head(predicted), axis=-1)
    correct = mask & (argmax == bins)
    metrics = {"mz_bin_accuracy": _weighted_mean(correct.astype(jnp.float32), mask),
               "mz_center_error_mda": 1000 * bin_size * _weighted_mean(jnp.abs(0.5 - position), correct)}
    if head is not None:
        metrics["mz_offset_loss"] = _weighted_mean(jnp.abs(head(predicted, bins) - position), mask)
        metrics["mz_abs_error_mda"] = 1000 * bin_size * _weighted_mean(
            jnp.abs(jax.lax.stop_gradient(head(predicted, argmax)) - position), correct)
    return metrics


def pair_forward(pair_model: PeakSetJEPAJax, batch, head=None):
    """``forward_pair_distogram`` (loss terms only) with an optional offset head."""
    from spectra_learning.models.spectrum_metadata import jax_spectrum_metadata_from_batch

    valid = batch["peak_valid_mask"]
    target_union = jnp.any(batch["target_masks"] & valid[:, None, :], axis=1)
    context_mask = batch["context_mask"] & valid & ~target_union
    pair, idx, mask = pair_model.encoder(
        batch["peak_mz"], batch["peak_intensity"], valid_mask=valid, visible_mask=context_mask,
        max_visible_tokens=pair_model.pairmixer_fast_encoder_max_visible_tokens,
        precursor_mz=batch.get("precursor_mz"),
        spectrum_metadata=jax_spectrum_metadata_from_batch(
            batch, pair_model.encoder_metadata_schema))
    losses = pair_distogram_with_offsets(pair_model, pair, idx, mask, target_union,
                                         batch["peak_mz"], head)
    offset = losses.pop("distogram_offset_loss", None)
    loss, metrics = pair_model._distogram_metrics(losses, pair)
    if offset is not None:
        metrics["distogram_offset_loss"] = offset
    return loss, metrics, offset


def dual_forward(joint: PeakSetJEPAJax, pair_model: PeakSetJEPAJax, batch, meta=None,
                 use_precursor=True, offsets=None, registers=None):
    """One context encoding; MAE from singles, distogram from encoder pairs."""
    peak_mz, peak_intensity = batch["peak_mz"], batch["peak_intensity"]
    valid = batch["peak_valid_mask"]
    target_masks = batch["target_masks"] & valid[:, None, :]
    target_union = jnp.any(target_masks, axis=1)
    context_mask = batch["context_mask"] & valid & ~target_union
    single, pair, idx, compact_mask = encode_joint(
        joint.encoder, meta, peak_mz, peak_intensity, valid_mask=valid, visible_mask=context_mask,
        max_visible_tokens=joint.pairmixer_fast_encoder_max_visible_tokens,
        precursor_mz=batch["precursor_mz"] if use_precursor else None,
        metadata=batch.get("spectrum_metadata"), registers=registers)
    predicted, target_positions, target_mask, single_distogram = (
        joint._predict_augmented_target_outputs_fastmixer_compact(
            single, idx, compact_mask, context_mask, target_masks, peak_mz))
    flat_mz, flat_intensity = joint._flatten_peak_values_for_target_views(
        peak_mz, peak_intensity, target_masks.shape[1])
    peak_idx = jnp.minimum(target_positions, peak_mz.shape[1] - 1)
    target_mz = jnp.take_along_axis(flat_mz, peak_idx, axis=1)
    mae_term, mae_metrics = joint._mae_metrics_compact(
        predicted, target_mz,
        jnp.take_along_axis(flat_intensity, peak_idx, axis=1), target_mask,
        reference=single, compute_accuracy=False)
    mz_head = None if offsets is None else offsets.mz
    dist_head = None if offsets is None else offsets.dist
    mae_metrics.update(mz_offset_metrics(joint, predicted, target_mz, target_mask, mz_head))
    distogram = pair_distogram_with_offsets(
        pair_model, pair, idx, compact_mask, target_union, peak_mz, dist_head)
    dist_offset = distogram.pop("distogram_offset_loss", None)
    distogram_term, distogram_metrics = pair_model._distogram_metrics(distogram, pair)
    loss = mae_term + distogram_term
    if mz_head is not None:
        loss = loss + offsets.weight * mae_metrics["mz_offset_loss"]
    if dist_offset is not None:
        distogram_metrics["distogram_offset_loss"] = dist_offset
        loss = loss + offsets.weight * dist_offset
    if single_distogram is not None:
        # dualplus: keep the historical distogram on single predictor latents.
        single_term, single_metrics = joint._distogram_metrics(single_distogram, single)
        loss = loss + single_term
        distogram_metrics["single_distogram_loss"] = single_metrics["distogram_loss"]
    return {"loss": loss, **mae_metrics, **distogram_metrics}


LOGGED = ("loss", "mae_mz_loss", "distogram_loss", "single_distogram_loss",
          "mz_offset_loss", "distogram_offset_loss", "mz_bin_accuracy", "mz_center_error_mda",
          "mz_abs_error_mda")


@nnx.jit
def train_step(model, optimizer, batch):
    def loss_fn(m):
        metrics = m.loss(batch)
        return metrics["loss"], metrics
    (_, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return {k: v for k, v in metrics.items() if k in LOGGED}


@nnx.jit
def eval_step(model, batch):
    return {k: v for k, v in model.loss(batch).items() if k in LOGGED}


@nnx.jit
def encode_step(model, batch):
    return model.encode(batch)


ISOTOPE_SPACINGS_DA = (1.003355, 2.006710)  # 13C and 13C2, singly charged
# The 13C spacing sits 0.0045 Da from the H-atom spacing (1.007825), which is
# twice as frequent in MassSpecGym peak sets; a wider window conflates them.
ISOTOPE_TOLERANCE_DA = 0.002


def isotope_adjacency(peak_mz: np.ndarray, valid: np.ndarray) -> np.ndarray:
    mz = peak_mz.astype(np.float64) * PEAK_MZ_SCALE
    gap = np.abs(mz[:, None, :] - mz[:, :, None])
    near = np.zeros(gap.shape, bool)
    for spacing in ISOTOPE_SPACINGS_DA:
        near |= np.abs(gap - spacing) <= ISOTOPE_TOLERANCE_DA
    return near & valid[:, None, :] & valid[:, :, None]


def thompson_span_targets(intensity, valid, isotopes, rng, *, concentration=4.0):
    """Thompson-span targets with isotope co-masking for m/z-sorted peaks.

    Anchors are visited in order of s_i ~ Beta(1 + k I_i, 1 + k (1 - I_i)), so
    intense peaks are preferred but not deterministic. Each anchor adds a
    contiguous 3-4 peak m/z span (trimmed to the remaining budget) plus the
    isotope closure of that span, which may exceed the budget up to ``cap``.
    The budget matches the random scheme: 25% of valid peaks, at most 8.
    """
    n = int(valid.sum())  # valid peaks form a prefix
    budget = min(max(int(n * .25), 1), 8)
    level = intensity[:n] / max(float(intensity[:n].max()), 1e-12)
    scores = rng.beta(1 + concentration * level, 1 + concentration * (1 - level))
    target = np.zeros(len(valid), bool)
    for anchor in np.argsort(-scores):
        room = budget - int(target.sum())
        if room <= 0:
            break
        length = min(int(rng.integers(3, 5)), room)
        stop = min(anchor + length, n)
        group = np.zeros_like(target)
        group[max(stop - length, 0):stop] = True
        for _ in range(2):  # closure over M+1 / M+2 chains
            group |= isotopes[group].any(0)
        group &= ~target
        if target.sum() + group.sum() <= min(8, n - 1):
            target |= group
    return target


def mask_batch_ts(data, indices, rng):
    """Same contract and context fraction as ``mask_batch``, Thompson-span targets."""
    batch = {k: data[k][indices] for k in ("peak_mz", "peak_intensity", "peak_valid_mask", "precursor_mz")}
    valid = batch["peak_valid_mask"]
    isotopes = isotope_adjacency(batch["peak_mz"], valid)
    target = np.stack([thompson_span_targets(batch["peak_intensity"][b], valid[b], isotopes[b], rng)
                       for b in range(len(indices))])
    counts = valid.sum(1)
    context_n = np.maximum(np.minimum((counts * .6).astype(int), counts - target.sum(1)), 1)
    free = valid & ~target
    order = np.argsort(np.where(free, rng.random(valid.shape), 2.0), axis=1)
    context = free & (np.argsort(order, axis=1) < context_n[:, None])
    batch["context_mask"] = context
    batch["target_masks"] = target[:, None]
    return {k: jnp.asarray(v) for k, v in batch.items()}


MASKINGS = {"random": mask_batch, "ts_iso": mask_batch_ts}


class MetadataRows:
    """Condition vectors per panel row: dropped-out for training, full or blind for evaluation."""

    def __init__(self, path, precursor_mz, fields, dropout, seed):
        self.full = metadata_lib.condition_vectors(dict(np.load(path)), precursor_mz, fields)
        self.blind = metadata_lib.blind(self.full)
        self.dropout = dropout
        self.rng = np.random.default_rng(seed + 99)  # never touches the mask stream

    def attach(self, batch, indices, mode):
        if mode == "train":
            rows = metadata_lib.drop_fields(self.full[indices], self.dropout, self.rng)
        else:
            rows = (self.full if mode == "full" else self.blind)[indices]
        return {**batch, "spectrum_metadata": jnp.asarray(rows)}


def _attach(metadata, batch, indices, mode):
    return batch if metadata is None else metadata.attach(batch, indices, mode)


def validation_losses(model, data, indices, batch_size, masking=mask_batch, metadata=None,
                      mode="full"):
    rng = np.random.default_rng(42)
    rows = []
    for start in range(0, len(indices), batch_size):
        chunk = indices[start:start + batch_size]
        batch = _attach(metadata, masking(data, chunk, rng), chunk, mode)
        rows.append({k: float(v) for k, v in eval_step(model, batch).items()})
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def export_features(model, data, indices, batch_size, metadata=None, mode="full"):
    pieces = []
    for start in range(0, len(indices), batch_size):
        idx = indices[start:start + batch_size]
        batch = _attach(metadata, {k: jnp.asarray(data[k][idx]) for k in
                                   ("peak_mz", "peak_intensity", "peak_valid_mask", "precursor_mz")},
                        idx, mode)
        pieces.append({k: np.asarray(v) for k, v in encode_step(model, batch).items()})
    return {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}


def save_export(path: Path, model, data, groups: dict, batch_size, metadata=None, mode="full"):
    arrays = {}
    for name, indices in groups.items():
        for key, value in export_features(model, data, indices, batch_size, metadata, mode).items():
            arrays[f"{name}/{key}"] = value.astype(np.float32)
        arrays[f"{name}/indices"] = indices
    np.savez(path, **arrays)


def train(args):
    data = dict(np.load(args.inputs))
    confirmation = np.load(args.confirmation)["probe_test"]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model = PilotModel(args.arm, args.seed, args)
    fields = metadata_lib.parse_fields(args.meta_fields)
    metadata = (MetadataRows(args.metadata, data["precursor_mz"], fields, args.meta_dropout, args.seed)
                if args.meta_sites else None)
    params = nnx.state(model, nnx.Param)
    total_params = sum(x.size for x in jax.tree.leaves(params))
    schedule = optax.warmup_cosine_decay_schedule(
        0, args.lr, args.warmup, args.steps, end_value=args.lr * 0.1)
    optimizer = nnx.Optimizer(model, optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(schedule, b2=.95, weight_decay=.01)), wrt=nnx.Param)
    train_idx = np.flatnonzero((data["fold"] == "train") & (data["peak_valid_mask"].sum(1) >= 4))
    val_idx = data["probe_val"]
    probe_groups = {f: data[f"probe_{f}"] for f in FEATURE_FOLDS}
    rng = np.random.default_rng(args.seed + 77)
    history, checkpoints = [], []
    checkpoint_steps = {int(args.steps * f) for f in (0.25, 0.5, 0.75, 1.0)}
    started = time.perf_counter()
    for i in range(1, args.steps + 1):
        idx = rng.choice(train_idx, args.batch_size, replace=False)
        batch = _attach(metadata, MASKINGS[args.masking](data, idx, rng), idx, "train")
        metrics = train_step(model, optimizer, batch)
        if i % 500 == 0 or i == 1:
            row = {k: float(v) for k, v in metrics.items()}
            row.update(step=i, seconds=time.perf_counter() - started)
            history.append(row)
            print(json.dumps(row), flush=True)
        if i in checkpoint_steps:
            # "validation" always uses the random masks, so every arm is
            # compared on the same task; "validation_ts" is the harder task.
            entry = {"step": i,
                     "validation": validation_losses(model, data, val_idx, 256, metadata=metadata),
                     "validation_ts": validation_losses(model, data, val_idx, 256, mask_batch_ts,
                                                        metadata=metadata)}
            if metadata is not None:
                # Same masks with every metadata field absent: reliance on metadata.
                entry["validation_blind"] = validation_losses(
                    model, data, val_idx, 256, metadata=metadata, mode="blind")
            groups = dict(probe_groups)
            if i == args.steps:
                groups.update(test=data["probe_test"], confirm=confirmation)
            save_export(out / f"features_{i}.npz", model, data, groups, 256, metadata)
            if metadata is not None and i == args.steps:
                save_export(out / f"features_{i}_blind.npz", model, data, groups, 256,
                            metadata, "blind")
            checkpoints.append(entry)
            print(json.dumps(entry), flush=True)
    train_seconds = time.perf_counter() - started
    final = {}
    interface = ARMS[args.arm][1] or ""
    if any(name in interface for name in ("mean", "ipa", "gated")):
        for name in ("zero", "shuffle"):
            model.set_intervention(name)
            final[name] = validation_losses(model, data, val_idx, 256, metadata=metadata)
        model.set_intervention("intact")
    flat = {"/".join(map(str, p)): np.asarray(v) for p, v in nnx.to_flat_state(nnx.state(model))}
    np.savez(out / "model.npz", **flat)
    result = {"arm": args.arm, "seed": args.seed, "args": vars(args), "params": int(total_params),
              "train_seconds": train_seconds, "steps_per_second": args.steps / train_seconds,
              "device": str(jax.devices()[0]), "jax_version": jax.__version__,
              "history": history, "checkpoints": checkpoints, "interventions": final}
    (out / "training.json").write_text(json.dumps(result, indent=2) + "\n")


def add_model_args(parser):
    parser.add_argument("--single-dim", type=int, default=256)
    parser.add_argument("--pair-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--predictor-layers", type=int, default=2)
    parser.add_argument("--bridge-rank", type=int, default=64)
    parser.add_argument("--mz-bin-size", type=float, default=2.0,
                        help="Da per class for both the masked m/z and distogram heads")
    parser.add_argument("--metadata", help="per-row metadata from pair_single_dual_metadata")
    parser.add_argument("--meta-fields", default="",
                        help=f"'+'-joined subset of {metadata_lib.FIELDS}")
    parser.add_argument("--meta-sites", default="",
                        help=f"'+'-joined subset of {META_SITES}, or 'native' for pair-only")
    parser.add_argument("--meta-dim", type=int, default=256)
    parser.add_argument("--meta-dropout", type=float, default=0.1,
                        help="per-field training dropout (production uses 0.1)")
    parser.add_argument("--offset-heads", default="",
                        help="'+'-joined subset of ('mz', 'dist'): regress the within-bin position")
    parser.add_argument("--offset-weight", type=float, default=1.0,
                        help="weight of each within-bin L1 term (bin units)")
    parser.add_argument("--registers", type=int, default=0,
                        help="learned register tokens appended to the joint encoder axis")
    parser.add_argument("--pair-precursor", choices=("on", "off"), default="on",
                        help="off: pair features use the largest peak as reference mass")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--confirmation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--masking", choices=tuple(MASKINGS), default="random")
    add_model_args(parser)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
