from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from scripts.encoder_chemistry_corruptions import ENCODER_KEYS
from scripts.encoder_chemistry_interventions import identity_interventions, trace_encoder
from scripts.modal_encoder_chemistry_input_effects import (
    METADATA_KEYS, metadata_path_patch, observed_repeat_arms, require_bitwise,
)
from spectra_learning.models.encoder_jax import PeakSetEncoder
from spectra_learning.models.peak_features_jax import PeakFeatureEmbedder


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_raw_final_block_factorial_diagonals_recover_native_and_only_cross_final_condition(dtype):
    rngs = nnx.Rngs(73)
    model = PeakSetEncoder(
        model_dim=16,
        embedder=PeakFeatureEmbedder(model_dim=16, hidden_dim=12, fourier_num_freqs=4,
                                   compute_dtype=dtype, rngs=rngs),
        num_layers=2, num_heads=2, attention_mlp_multiple=2, num_peaks=3,
        use_pair_path=False, use_cls_token=True, metadata_conditioning="adaln_zero",
        metadata_condition_dim=8, compute_dtype=dtype, rngs=rngs,
    )
    for index, block in enumerate(model.blocks):
        block.adaLN_modulation.weight[...] = .2 * jax.random.normal(
            jax.random.key(index), block.adaLN_modulation.weight.shape)
        block.adaLN_modulation.bias[...] = jnp.linspace(-.2, .4, 96)
    model.final_adaLN_modulation.weight[...] = .3 * jax.random.normal(
        jax.random.key(31), model.final_adaLN_modulation.weight.shape)
    mz = jnp.array([[.05, .1, .2], [.08, .13, .3]], jnp.float32)
    intensity = jnp.array([[1., .3, .7], [.5, 1., .4]], jnp.float32)
    valid = jnp.array([[True, True, False], [True, True, True]])
    anchor_metadata = jnp.linspace(0., 1., 56).reshape(2, 28)
    donor_metadata = anchor_metadata.at[:, 1].add(.7)
    traces, affines = [], []
    for condition in (anchor_metadata, donor_metadata):
        trace = trace_encoder(model, mz, intensity, valid_mask=valid, spectrum_metadata=condition)
        traces.append(trace)
        embedding = model._metadata_embedding(condition, dtype)
        affines.append(tuple(jnp.split(model.final_adaLN_modulation(embedding), 2, axis=-1)))
    controls = identity_interventions(model, 2, 3)
    anchor_blocks = metadata_path_patch(controls, traces[0].raw[2], valid, depth=2)
    donor_blocks = metadata_path_patch(controls, traces[1].raw[2], valid, depth=2)
    np.testing.assert_array_equal(anchor_blocks.patch_mask[:, -1], True)
    np.testing.assert_array_equal(anchor_blocks.patch_mask[:, :-1], valid)
    outputs = []
    for condition, patch in ((anchor_metadata, anchor_blocks), (donor_metadata, anchor_blocks),
                              (anchor_metadata, donor_blocks), (donor_metadata, donor_blocks)):
        outputs.append(trace_encoder(model, mz, intensity, valid_mask=valid,
            spectrum_metadata=condition, interventions=patch))
    token_mask = np.asarray(anchor_blocks.patch_mask)
    np.testing.assert_array_equal(outputs[0].final, traces[0].final)
    np.testing.assert_array_equal(outputs[3].final, traces[1].final)
    # Crossed paths preserve donor/anchor raw block states, including CLS, then
    # execute native final normalization and affine with the receiving metadata.
    np.testing.assert_array_equal(np.asarray(outputs[1].raw[-1])[token_mask],
                                  np.asarray(traces[0].raw[-1])[token_mask])
    np.testing.assert_array_equal(np.asarray(outputs[2].raw[-1])[token_mask],
                                  np.asarray(traces[1].raw[-1])[token_mask])
    shift, scale = affines[1]
    expected = traces[0].final_normalized * (1 + scale[:, None]) + shift[:, None]
    np.testing.assert_array_equal(np.asarray(outputs[1].final)[token_mask], np.asarray(expected)[token_mask])
    assert np.linalg.norm(np.asarray(outputs[1].final - outputs[0].final, np.float32)) > 1e-4


def test_observed_repeat_factorial_preserves_each_physical_mask_and_unsupported_rows():
    data = {
        "peak_mz": np.array([[.1, .2, .3], [.2, .3, .4]], np.float32),
        "peak_intensity": np.array([[1., .3, 0.], [.4, 1., .5]], np.float32),
        "peak_valid_mask": np.array([[True, True, False], [True, True, True]]),
        "precursor_mz": np.array([.4, .5], np.float32),
        "collision_energy": np.array([.2, .3], np.float32),
        "collision_energy_present": np.ones(2, np.float32),
        "charge": np.ones(2, np.float32), "charge_present": np.ones(2, np.float32),
        "polarity_id": np.ones(2, np.int64), "instrument_family_id": np.ones(2, np.int64),
        "computed_inchikey": np.array(["molecule_a", "molecule_b"]),
        "canonical_smiles": np.array(["CCO", "CCC"]),
        "instrument_type": np.array(["Orbitrap", "Orbitrap"]),
        "adduct": np.array(["[M+H]+", "[M+H]+"]),
        "collision_energy_raw": np.array([20., 30.]), "rows": np.array([100, 101]),
        "chemistry_labels_never_passed": np.ones((2, 188)),
    }
    repeats = {key: value[[0]].copy() for key, value in data.items()}
    repeats.update(primary_index=np.array([0]), rows=np.array([200]),
                   collision_energy_raw=np.array([40.]), collision_energy=np.array([.4], np.float32),
                   precursor_mz=np.array([.401], np.float32),
                   peak_mz=np.array([[.08, .15, .25]], np.float32),
                   peak_intensity=np.array([[1., .8, .4]], np.float32),
                   peak_valid_mask=np.array([[True, True, True]]))
    original = {key: value.copy() for key, value in data.items()}
    arms, supported, facts = observed_repeat_arms(data, np.array([0, 1]), repeats)
    np.testing.assert_array_equal(supported, [True, False])
    assert facts["repeat_indices"] == [0, -1]
    for arm in arms:
        assert set(arm) == set(ENCODER_KEYS)
        for key in ENCODER_KEYS:
            np.testing.assert_array_equal(arm[key][1], data[key][1])
    for key in METADATA_KEYS:
        np.testing.assert_array_equal(arms[0][key], arms[3][key])
        np.testing.assert_array_equal(arms[1][key], arms[2][key])
    for key in ("peak_mz", "peak_intensity", "peak_valid_mask"):
        np.testing.assert_array_equal(arms[0][key], arms[1][key])
        np.testing.assert_array_equal(arms[2][key], arms[3][key])
    for key, value in original.items():
        np.testing.assert_array_equal(data[key], value)


def test_parity_guard_rejects_small_valid_token_change_but_ignores_padding():
    expected = np.ones((2, 3, 4), np.float32)
    valid = np.array([[True, False, True], [True, True, True]])
    changed = expected.copy()
    changed[0, 1] = 100.
    assert require_bitwise(changed, expected, valid, "padding")["bitwise_equal"]
    changed[1, 0, 0] += 1e-5
    with pytest.raises(ValueError, match="not bitwise equal"):
        require_bitwise(changed, expected, valid, "small valid change")
