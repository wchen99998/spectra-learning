from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from scripts.encoder_chemistry_interventions import identity_interventions, trace_encoder
from spectra_learning.models.encoder_jax import PeakSetEncoder
from spectra_learning.models.peak_features_jax import PeakFeatureEmbedder


def _example(dtype=jnp.float32, conditioned=True):
    rngs = nnx.Rngs(19)
    encoder = PeakSetEncoder(
        model_dim=16,
        embedder=PeakFeatureEmbedder(
            model_dim=16, hidden_dim=12, fourier_num_freqs=4,
            compute_dtype=dtype, rngs=rngs,
        ),
        num_layers=3,
        num_heads=2,
        attention_mlp_multiple=2,
        num_peaks=5,
        use_pair_path=False,
        use_cls_token=True,
        metadata_conditioning="adaln_zero" if conditioned else "additive",
        metadata_condition_dim=8,
        compute_dtype=dtype,
        rngs=rngs,
    )
    # Fresh AdaLN gates/output projections can be zero; exercise trained-like,
    # nonzero residual branches and nontrivial metadata modulation instead.
    for i, block in enumerate(encoder.blocks):
        for offset, module in enumerate(
            (block.single_attention.o, block.single_transition.fc3)
        ):
            module.weight[...] = .1 * jax.random.normal(
                jax.random.key(100 + 2 * i + offset), module.weight.shape
            )
        if conditioned:
            block.adaLN_modulation.weight[...] = .08 * jax.random.normal(
                jax.random.key(200 + i), block.adaLN_modulation.weight.shape
            )
            block.adaLN_modulation.bias[...] = jnp.linspace(-.2, .4, 96)
    if conditioned:
        encoder.final_adaLN_modulation.weight[...] = .1 * jax.random.normal(
            jax.random.key(300), encoder.final_adaLN_modulation.weight.shape
        )
        encoder.final_adaLN_modulation.bias[...] = jnp.linspace(-.1, .2, 32)
    mz = jnp.asarray([[.03, .07, .13, .21, .32], [.02, .05, .09, .18, .27]])
    intensity = jnp.asarray([[1., .4, .7, .2, .1], [.5, 1., .3, .2, .4]])
    valid = jnp.asarray([[True, True, True, False, False], [True] * 5])
    metadata_width = 28 if conditioned else 2
    metadata = jnp.linspace(-.1, .9, 2 * metadata_width).reshape(2, metadata_width)
    return encoder, mz, intensity, valid, metadata


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("conditioned", [False, True])
def test_noop_matches_native_and_distinguishes_raw_from_final(dtype, conditioned):
    encoder, mz, intensity, valid, metadata = _example(dtype, conditioned)
    trace = trace_encoder(
        encoder, mz, intensity, valid_mask=valid, spectrum_metadata=metadata,
        capture_attention=True,
    )
    assert trace.final.dtype == dtype
    assert trace.raw.dtype == dtype
    actual = encoder(mz, intensity, valid_mask=valid, spectrum_metadata=metadata)
    np.testing.assert_array_equal(np.asarray(trace.final), np.asarray(actual))
    for layer in range(4):
        expected = encoder(
            mz, intensity, valid_mask=valid, spectrum_metadata=metadata,
            num_blocks=layer,
        )
        np.testing.assert_array_equal(np.asarray(trace.raw[layer]), np.asarray(expected))
    assert not np.allclose(np.asarray(trace.raw[-1], float), np.asarray(trace.final, float))
    np.testing.assert_array_equal(
        np.asarray(trace.final_normalized), np.asarray(encoder.final_norm(trace.raw[-1]))
    )
    assert trace.attention_patterns.shape == (3, 2, 2, 6, 6)
    np.testing.assert_allclose(np.asarray(trace.attention_patterns.sum(-1)), 1, atol=2e-7)
    np.testing.assert_array_equal(np.asarray(trace.attention_patterns[:, 0, :, :, 3:5]), 0)


def test_identity_patch_at_every_depth_preserves_outputs():
    encoder, mz, intensity, valid, metadata = _example()
    control = identity_interventions(encoder, 2, 5)
    clean = trace_encoder(encoder, mz, intensity, valid_mask=valid, spectrum_metadata=metadata)
    for layer in range(4):
        intervention = control._replace(
            patch_layer=jnp.asarray(layer), patch_values=clean.raw[layer],
            patch_mask=jnp.ones((2, 6), dtype=jnp.bool_),
        )
        patched = trace_encoder(
            encoder, mz, intensity, valid_mask=valid, spectrum_metadata=metadata,
            interventions=intervention,
        )
        np.testing.assert_array_equal(patched.final, clean.final)


def test_selected_token_patch_is_local_then_propagates_through_later_attention():
    encoder, mz, intensity, valid, metadata = _example()
    control = identity_interventions(encoder, 2, 5)
    clean = trace_encoder(encoder, mz, intensity, valid_mask=valid, spectrum_metadata=metadata)
    selector = jnp.zeros((2, 6), dtype=jnp.bool_).at[0, 1].set(True)
    replacement = clean.raw[1].at[0, 1].add(jnp.linspace(-2., 3., 16))
    patched = trace_encoder(
        encoder, mz, intensity, valid_mask=valid, spectrum_metadata=metadata,
        interventions=control._replace(
            patch_layer=jnp.asarray(1), patch_values=replacement, patch_mask=selector,
        ),
    )
    np.testing.assert_array_equal(patched.raw[0], clean.raw[0])
    np.testing.assert_array_equal(np.asarray(patched.raw[1])[~np.asarray(selector)],
                                  np.asarray(clean.raw[1])[~np.asarray(selector)])
    np.testing.assert_array_equal(patched.raw[1, 0, 1], replacement[0, 1])
    assert np.linalg.norm(np.asarray(patched.final[0, 0] - clean.final[0, 0])) > 1e-4
    np.testing.assert_array_equal(patched.final[1], clean.final[1])
    np.testing.assert_array_equal(patched.token_mask, clean.token_mask)


def test_ablation_targets_requested_branch_and_head():
    encoder, mz, intensity, valid, metadata = _example()
    control = identity_interventions(encoder, 2, 5)
    clean = trace_encoder(encoder, mz, intensity, valid_mask=valid, spectrum_metadata=metadata)
    for field, trace_field in (("attention_scale", "attention_update"), ("mlp_scale", "mlp_update")):
        intervention = control._replace(**{field: getattr(control, field).at[1].set(0)})
        ablated = trace_encoder(
            encoder, mz, intensity, valid_mask=valid, spectrum_metadata=metadata,
            interventions=intervention,
        )
        np.testing.assert_array_equal(ablated.raw[:2], clean.raw[:2])
        np.testing.assert_array_equal(getattr(ablated, trace_field)[1], 0)
        assert np.linalg.norm(np.asarray(ablated.final - clean.final)) > 1e-4
    one_head = trace_encoder(
        encoder, mz, intensity, valid_mask=valid, spectrum_metadata=metadata,
        interventions=control._replace(head_scale=control.head_scale.at[1, 0].set(0)),
    )
    np.testing.assert_array_equal(one_head.raw[:2], clean.raw[:2])
    assert np.linalg.norm(np.asarray(one_head.final - clean.final)) > 1e-4


def test_invalid_keys_cannot_leak_or_receive_patches():
    encoder, mz, intensity, valid, metadata = _example()
    control = identity_interventions(encoder, 2, 5)
    clean = trace_encoder(encoder, mz, intensity, valid_mask=valid, spectrum_metadata=metadata)
    changed = trace_encoder(
        encoder, jnp.where(valid, mz, .987), jnp.where(valid, intensity, 50.),
        valid_mask=valid, spectrum_metadata=metadata,
    )
    np.testing.assert_array_equal(np.asarray(changed.final)[clean.token_mask],
                                  np.asarray(clean.final)[clean.token_mask])
    patched = trace_encoder(
        encoder, mz, intensity, valid_mask=valid, spectrum_metadata=metadata,
        interventions=control._replace(
            patch_layer=jnp.asarray(1), patch_values=jnp.full((2, 6, 16), 100.),
            patch_mask=~clean.token_mask,
        ),
    )
    np.testing.assert_array_equal(patched.final, clean.final)


def test_visible_mask_and_original_positions_are_preserved():
    encoder, mz, intensity, valid, metadata = _example()
    visible = valid.at[1, 1].set(False)
    trace = trace_encoder(
        encoder, mz, intensity, valid_mask=valid, visible_mask=visible,
        spectrum_metadata=metadata,
    )
    expected = encoder(
        mz, intensity, valid_mask=valid, visible_mask=visible,
        spectrum_metadata=metadata,
    )
    np.testing.assert_array_equal(trace.final, expected)
    np.testing.assert_array_equal(trace.token_mask[:, :-1], visible)
    assert trace.raw.shape == (4, 2, 6, 16)


def test_final_modulation_removal_retains_norm_and_does_not_change_blocks():
    encoder, mz, intensity, valid, metadata = _example()
    control = identity_interventions(encoder, 2, 5)
    clean = trace_encoder(encoder, mz, intensity, valid_mask=valid, spectrum_metadata=metadata)
    changed = trace_encoder(
        encoder, mz, intensity, valid_mask=valid, spectrum_metadata=metadata,
        interventions=control._replace(final_modulation_scale=jnp.asarray(0.)),
    )
    np.testing.assert_array_equal(changed.raw, clean.raw)
    np.testing.assert_array_equal(changed.final, clean.final_normalized)
    assert np.linalg.norm(np.asarray(changed.final - clean.final)) > 1e-4


def test_block_condition_control_retains_bias_and_only_changes_selected_depth():
    encoder, mz, intensity, valid, metadata = _example()
    control = identity_interventions(encoder, 2, 5)
    clean = trace_encoder(encoder, mz, intensity, valid_mask=valid, spectrum_metadata=metadata)
    changed = trace_encoder(
        encoder, mz, intensity, valid_mask=valid, spectrum_metadata=metadata,
        interventions=control._replace(
            block_condition_scale=control.block_condition_scale.at[1].set(0),
        ),
    )
    expected = encoder.blocks[1](
        clean.raw[1], clean.token_mask, condition=jnp.zeros((2, 8)),
    )
    np.testing.assert_array_equal(changed.raw[:2], clean.raw[:2])
    np.testing.assert_array_equal(changed.raw[2], expected)
    assert np.linalg.norm(np.asarray(changed.final - clean.final)) > 1e-4


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_jit_uses_dynamic_selectors_without_retracing_and_matches_native(dtype):
    encoder, mz, intensity, valid, metadata = _example(dtype)
    traces = []

    @nnx.jit
    def run(model, controls):
        traces.append(1)
        return trace_encoder(
            model, mz, intensity, valid_mask=valid, spectrum_metadata=metadata,
            interventions=controls,
        ).final

    @nnx.jit
    def native(model):
        return model(mz, intensity, valid_mask=valid, spectrum_metadata=metadata)

    @nnx.jit
    def snapshot(model, controls):
        return trace_encoder(
            model, mz, intensity, valid_mask=valid, spectrum_metadata=metadata,
            interventions=controls,
        ).raw

    control = identity_interventions(encoder, 2, 5)
    clean = run(encoder, control)
    expected = np.asarray(native(encoder), np.float32)
    actual = np.asarray(clean, np.float32)
    if dtype == jnp.bfloat16:
        # Dynamic multipliers change XLA fusion and bf16 rounding. Match the
        # repository's historical export gate, not an assertion of bitwise JIT
        # equivalence. The actual checkpoint/device needs its own calibration.
        relative_l2 = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
        assert relative_l2 < .005, relative_l2
    else:
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
    assert clean.dtype == dtype
    trace_count = len(traces)
    for layer in (0, 1, 2):
        result = run(encoder, control._replace(mlp_scale=control.mlp_scale.at[layer].set(0)))
        result.block_until_ready()
        assert np.linalg.norm(np.asarray(result - clean)) > 1e-4
    cached_raw = snapshot(encoder, control)
    for layer in (0, 1, 3):
        identity_patch = control._replace(
            patch_layer=jnp.asarray(layer, dtype=jnp.int32), patch_values=cached_raw[layer],
            patch_mask=jnp.ones((2, 6), dtype=jnp.bool_),
        )
        patched = np.asarray(run(encoder, identity_patch), np.float32)
        if dtype == jnp.bfloat16:
            relative_l2 = np.linalg.norm(patched - actual) / np.linalg.norm(actual)
            assert relative_l2 < .005, relative_l2
        else:
            np.testing.assert_allclose(patched, actual, atol=2e-6, rtol=2e-6)
    assert len(traces) == trace_count


def test_api_accepts_spectrum_inputs_only_and_rejects_misaligned_patches():
    parameters = set(inspect.signature(trace_encoder).parameters)
    assert parameters == {
        "encoder", "peak_mz", "peak_intensity", "valid_mask", "spectrum_metadata",
        "visible_mask", "interventions", "capture_attention",
    }
    encoder, mz, intensity, valid, metadata = _example()
    controls = identity_interventions(encoder, 2, 5)._replace(
        patch_mask=jnp.zeros((2, 5), dtype=jnp.bool_)
    )
    with pytest.raises(ValueError, match="patch_mask must have shape"):
        trace_encoder(
            encoder, mz, intensity, valid_mask=valid, spectrum_metadata=metadata,
            interventions=controls,
        )
