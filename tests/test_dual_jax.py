"""Joint single+pair ``dual`` training mode and peak-count stages."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from spectra_learning.models.model_jax import PeakSetJEPAJax
from spectra_learning.models.pair_to_single_jax import PairToSingleBridge
from spectra_learning.models.spectrum_metadata import (
    MASSIVE_V2_ACQUISITION_SCHEMA,
    MASSIVE_V2_CONDITION_DIM,
)


def _dual_kwargs(num_peaks: int = 7) -> dict[str, object]:
    return {
        "training_mode": "dual",
        "model_dim": 8,
        "predictor_dim": 8,
        "encoder_num_layers": 2,
        "encoder_num_heads": 2,
        "attention_mlp_multiple": 1.0,
        "predictor_mlp_multiple": 1.0,
        "feature_mlp_hidden_dim": 8,
        "encoder_fourier_num_freqs": 2,
        "pairmixer_fourier_num_freqs": 2,
        "pairmixer_pair_dim": 8,
        "pairmixer_pair_feature_hidden_dim": 8,
        "pairmixer_block_type": "fastmixer",
        "encoder_apply_final_pair_norm": True,
        "masked_latent_predictor_num_layers": 1,
        "masked_latent_predictor_num_heads": 2,
        "pair_predictor_num_layers": 2,
        "masked_token_input_mode": "latent_token",
        "num_peaks": num_peaks,
        "jepa_num_target_blocks": 1,
        "jepa_mae_mz_bin_size": 50.0,
        "mae_loss_weight": 0.7,
        "mae_intensity_loss_weight": 0.0,
        "distogram_loss_weight": 0.3,
        "predictor_apply_final_norm": True,
        "target_projector_dim": -1,
        "pair_to_single_rank": 4,
        "encoder_metadata_schema": MASSIVE_V2_ACQUISITION_SCHEMA,
        "encoder_metadata_conditioning": "precursor_edge",
        "encoder_metadata_condition_dim": 8,
    }


def _metadata(precursor_mz: np.ndarray) -> np.ndarray:
    metadata = np.zeros((len(precursor_mz), MASSIVE_V2_CONDITION_DIM), np.float32)
    metadata[:, 0] = precursor_mz
    metadata[:, 1] = 1.0
    metadata[:, [12, 15, 21]] = 1.0  # unknown polarity, acquisition, instrument
    return metadata


def _batch(num_peaks: int, *, batch_size: int = 3, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    valid = np.ones((batch_size, num_peaks), bool)
    valid[0, num_peaks - 2:] = False
    mz = np.sort(rng.uniform(0.05, 0.5, (batch_size, num_peaks)), 1) * valid
    context = np.zeros_like(valid)
    targets = np.zeros((batch_size, 1, num_peaks), bool)
    for row in range(batch_size):
        order = rng.permutation(np.flatnonzero(valid[row]))
        targets[row, 0, order[:2]] = True
        context[row, order[2:5]] = True
    precursor = np.full(batch_size, 0.6, np.float32)
    metadata = _metadata(precursor)
    metadata[-1, :2] = 0.0  # dropped precursor
    return {
        "peak_mz": mz.astype(np.float32),
        "peak_intensity": (rng.uniform(0.1, 1.0, mz.shape) * valid).astype(np.float32),
        "peak_valid_mask": valid,
        "context_mask": context,
        "target_masks": targets,
        "precursor_mz": precursor,
        "spectrum_metadata": metadata,
    }


def _perturb(model: nnx.Module, scale: float = 0.2) -> None:
    """Move zero-initialized heads off zero so every path carries signal."""
    state = nnx.state(model, nnx.Param)
    for index, (_, value) in enumerate(nnx.to_flat_state(state)):
        value[...] = value[...] + scale * jax.random.normal(
            jax.random.key(index), value.shape
        )
    nnx.update(model, state)


def test_bridge_ignores_padding_and_reads_pair_content():
    bridge = PairToSingleBridge(6, 4, rank=3, rngs=nnx.Rngs(0))
    single = jax.random.normal(jax.random.key(1), (2, 5, 6))
    pair = jax.random.normal(jax.random.key(2), (2, 5, 5, 4))
    mask = jnp.array([[True, True, False, True, False], [True] * 5])
    valid = mask[:, :, None] & mask[:, None, :]
    update = bridge(single, pair, mask)

    np.testing.assert_array_equal(np.asarray(update)[~np.asarray(mask)], 0)
    np.testing.assert_allclose(
        bridge(jnp.where(mask[..., None], single, 50.0),
               jnp.where(valid[..., None], pair, -30.0), mask),
        update,
        atol=1e-6,
    )
    gradient = jax.grad(lambda z: (bridge(single, z, mask) ** 2).sum())(pair)
    np.testing.assert_array_equal(np.asarray(gradient)[~np.asarray(valid)], 0)
    assert np.linalg.norm(np.asarray(gradient)) > 1e-3


def test_precursor_edges_touch_only_cls_edges_and_start_at_zero():
    model = PeakSetJEPAJax(rngs=nnx.Rngs(0), **_dual_kwargs())
    encoder = model.encoder
    batch = {k: jnp.asarray(v) for k, v in _batch(7).items()}
    args = (batch["peak_mz"], batch["peak_intensity"])
    x, visible, _ = encoder._embed_peaks(*args, batch["peak_valid_mask"], None, None)

    def pairs(metadata):
        return encoder._embed_pairs(*args, x, visible, batch["precursor_mz"], metadata)

    without = pairs(None)
    np.testing.assert_array_equal(pairs(batch["spectrum_metadata"]), without)

    _perturb(encoder.precursor_edges)
    delta = np.asarray(pairs(batch["spectrum_metadata"]) - without)
    cls = 7
    changed = np.zeros(delta.shape[:3], bool)
    changed[:, :cls, cls] = changed[:, cls, :cls] = True
    changed &= np.concatenate([np.asarray(visible), np.zeros((3, 1), bool)], 1)[:, :, None] | (
        np.concatenate([np.asarray(visible), np.zeros((3, 1), bool)], 1)[:, None, :]
    )
    assert np.abs(delta[changed]).min() > 0
    np.testing.assert_array_equal(delta[~changed], 0)
    # A dropped precursor leaves only the condition term: equal for every peak.
    dropped = delta[-1, :cls, cls][np.asarray(visible)[-1]]
    np.testing.assert_allclose(dropped, np.broadcast_to(dropped[:1], dropped.shape), atol=1e-6)


def test_dual_trains_every_path():
    model = PeakSetJEPAJax(rngs=nnx.Rngs(0), **_dual_kwargs())
    model.set_fastmixer_capacities(6, 8, 3)
    _perturb(model.encoder.precursor_edges)
    batch = {k: jnp.asarray(v) for k, v in _batch(7).items()}
    metrics = model(batch)

    assert model.encoder.blocks[0].pair_to_single is not None
    assert model.latent_mask_token is not None and model.pair_mask_token is not None
    for name in ("loss", "mae_mz_loss", "distogram_loss"):
        assert np.isfinite(float(metrics[name]))
    np.testing.assert_allclose(
        float(metrics["loss"]),
        float(metrics["mae_term"] + metrics["distogram_term"]),
        rtol=1e-6,
    )
    gradient = nnx.grad(lambda m: m(batch, loss_only=True)["loss"])(model)
    flat = {
        "/".join(map(str, path)): float(jnp.abs(value[...]).sum())
        for path, value in nnx.to_flat_state(gradient)
    }
    for name in (
        "pair_to_single", "precursor_edges", "pair_predictor", "pair_mask_token",
        "distogram_head", "masked_latent_predictor", "jepa_mae_mz_head", "pair_embedder",
    ):
        assert sum(value for key, value in flat.items() if name in key) > 0, name


def test_dual_encoder_paths_agree():
    """Probes use the scattered-single path; training uses the compact one."""
    model = PeakSetJEPAJax(rngs=nnx.Rngs(0), **_dual_kwargs())
    _perturb(model.encoder)
    batch = {k: jnp.asarray(v) for k, v in _batch(7).items()}
    common = dict(
        valid_mask=batch["peak_valid_mask"],
        precursor_mz=batch["precursor_mz"],
        spectrum_metadata=batch["spectrum_metadata"],
    )
    full_single, _ = model.encoder.forward_with_pair(
        batch["peak_mz"], batch["peak_intensity"], **common
    )
    compact_single, _, idx, mask = model.encoder.forward_with_pair_compact(
        batch["peak_mz"], batch["peak_intensity"], visible_mask=batch["peak_valid_mask"],
        max_visible_tokens=8, **common,
    )
    gathered = jnp.take_along_axis(full_single, idx[..., None], axis=1)
    np.testing.assert_allclose(
        np.asarray(gathered)[np.asarray(mask)],
        np.asarray(compact_single)[np.asarray(mask)],
        atol=2e-5,
    )
