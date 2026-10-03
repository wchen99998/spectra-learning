"""Extraction checks for the native absorbing-diffusion encoder.

These small CPU checks complement (and do not replace) full-checkpoint bf16
parity on the device and exact batch shape used for the chemistry atlas.
"""
from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from scripts.encoder_chemistry_interventions import identity_interventions, trace_encoder
from spectra_learning.config import load_config
from spectra_learning.models.model_jax import build_peak_set_encoder_jax
from spectra_learning.models.settings import PeakSetJEPASettings
from spectra_learning.models.spectrum_metadata import jax_spectrum_metadata_from_batch


def _example(dtype):
    settings = PeakSetJEPASettings.from_config(load_config("configs/1p8b_absorbing_diffusion.py"))
    settings = replace(
        settings, model_dim=32, encoder_num_layers=2, encoder_num_heads=2,
        feature_mlp_hidden_dim=32, encoder_fourier_mlp_hidden_dim=32,
        encoder_fourier_mlp_num_layers=2, encoder_fourier_num_freqs=4,
        encoder_metadata_condition_dim=8, num_peaks=5,
        activation_checkpoint_mode="none",
    )
    encoder = build_peak_set_encoder_jax(
        settings, compute_dtype=dtype, pairmixer_block_type=settings.pairmixer_block_type,
        pairmixer_fast_encoder_max_visible_tokens=None, rngs=nnx.Rngs(4),
    )
    # Exercise nonzero trained-like branches rather than zero-initialized AdaLN.
    for index, block in enumerate(encoder.blocks):
        for offset, module in enumerate((block.single_attention.o, block.single_transition.fc3)):
            module.weight[...] = .1 * jax.random.normal(
                jax.random.key(100 + 2 * index + offset), module.weight.shape)
        block.adaLN_modulation.weight[...] = .08 * jax.random.normal(
            jax.random.key(200 + index), block.adaLN_modulation.weight.shape)
        block.adaLN_modulation.bias[...] = jnp.linspace(-.2, .4, 6 * 32)
    encoder.final_adaLN_modulation.weight[...] = .1 * jax.random.normal(
        jax.random.key(300), encoder.final_adaLN_modulation.weight.shape)
    encoder.final_adaLN_modulation.bias[...] = jnp.linspace(-.1, .2, 2 * 32)
    normalized_width = settings.jepa_mae_mz_bin_size / 1000
    batch = {
        "peak_mz": (jnp.array([[80., 180., 400., 900., 1500.], [30., 90., 300., 1000., 1900.]]) + .2) * normalized_width,
        "peak_intensity": jnp.array([[1., .4, .7, .2, .1], [.5, 1., .3, .2, .4]]),
        "peak_valid_mask": jnp.array([[True, True, True, False, False], [True] * 5]),
        "precursor_mz": jnp.array([.55, .95]),
        "collision_energy": jnp.array([.2, .4]),
        "collision_energy_present": jnp.ones(2),
        "charge": jnp.ones(2),
        "charge_present": jnp.ones(2),
        "polarity_id": jnp.array([1, 1]),
        "instrument_family_id": jnp.array([1, 2]),
    }
    metadata = jax_spectrum_metadata_from_batch(batch, settings.encoder_metadata_schema)
    assert metadata.shape == (2, 28)
    return encoder, batch, metadata, normalized_width


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_diffusion_trace_preserves_native_quantization_and_raw_states(dtype):
    encoder, batch, metadata, width = _example(dtype)
    mz, intensity, valid = (batch[key] for key in ("peak_mz", "peak_intensity", "peak_valid_mask"))
    controls = identity_interventions(encoder, 2, 5)
    trace = trace_encoder(encoder, mz, intensity, valid_mask=valid,
                          spectrum_metadata=metadata, interventions=controls)
    native = encoder(mz, intensity, valid_mask=valid, visible_mask=valid,
                     spectrum_metadata=metadata)
    np.testing.assert_array_equal(trace.final, native)
    for depth in range(3):
        np.testing.assert_array_equal(trace.raw[depth], encoder(
            mz, intensity, valid_mask=valid, visible_mask=valid,
            spectrum_metadata=metadata, num_blocks=depth))
    assert trace.final.dtype == dtype
    assert trace.final.shape == (2, 6, 32)  # All physical slots plus trailing CLS.
    shifted = trace_encoder(encoder, mz + .1 * width, intensity, valid_mask=valid,
                            spectrum_metadata=metadata, interventions=controls)
    np.testing.assert_array_equal(shifted.raw, trace.raw)
    np.testing.assert_array_equal(shifted.final, trace.final)
    crossed = trace_encoder(encoder, mz.at[0, 0].add(width), intensity, valid_mask=valid,
                            spectrum_metadata=metadata, interventions=controls)
    assert np.linalg.norm(np.asarray(crossed.final[0], float) - np.asarray(trace.final[0], float)) > 1e-4
    np.testing.assert_array_equal(crossed.final[1], trace.final[1])
    np.testing.assert_array_equal(trace.final_normalized, encoder.final_norm(trace.raw[-1]))


def test_diffusion_padding_cannot_leak_or_receive_physical_token_patches():
    encoder, batch, metadata, _ = _example(jnp.float32)
    mz, intensity, valid = (batch[key] for key in ("peak_mz", "peak_intensity", "peak_valid_mask"))
    controls = identity_interventions(encoder, 2, 5)
    clean = trace_encoder(encoder, mz, intensity, valid_mask=valid,
                          spectrum_metadata=metadata, interventions=controls)
    dirty = trace_encoder(encoder, jnp.where(valid, mz, .9999), jnp.where(valid, intensity, 123.),
                          valid_mask=valid, spectrum_metadata=metadata,
                          interventions=controls._replace(
                              patch_layer=jnp.asarray(1, jnp.int32),
                              patch_values=jnp.full_like(clean.raw[1], 1000.),
                              patch_mask=~clean.token_mask))
    np.testing.assert_array_equal(np.asarray(dirty.final)[clean.token_mask],
                                  np.asarray(clean.final)[clean.token_mask])
    identity = trace_encoder(encoder, mz, intensity, valid_mask=valid,
                             spectrum_metadata=metadata,
                             interventions=controls._replace(
                                 patch_layer=jnp.asarray(1, jnp.int32),
                                 patch_values=clean.raw[1], patch_mask=clean.token_mask))
    np.testing.assert_array_equal(identity.final, clean.final)
