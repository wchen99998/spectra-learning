from __future__ import annotations

import copy

import jax.numpy as jnp
import numpy as np
import pytest

from scripts.modal_encoder_chemistry_sae_causal import (
    direction_edit, feature_variants, patch_dose, physical_patch,
    random_directions, validate_selection,
)


def selection_fixture():
    contract = {"encoder": "parent_1190k", "depth": 19, "k": 32, "seed": 0,
                "params_sha256": "dictionary-a", "test_used_for_selection": False,
                "selection_splits": ["train", "val"], "panel_sha256": {"inputs.npz": "inputs-a"}}
    selection = {"contract": contract, "records": [
        {"feature": 2, "random_control_feature": 5, "concept": "fg_amide", "concept_index": 10},
    ]}
    targets = [f"target{i}" for i in range(188)]
    targets[176] = "fg_amide"
    expected = {"encoder": "parent_1190k", "params_sha256": "dictionary-a",
                "panel_sha256": {"inputs.npz": "inputs-a"}, "number_features": 8, "target_names": targets}
    return selection, expected


def test_selection_is_dictionary_specific_and_rejects_test_selection_or_target_mismatch():
    selection, expected = selection_fixture()
    assert validate_selection(selection, **expected) == selection["records"]
    for key, value in (("encoder", "periodic_1209k"), ("params_sha256", "dictionary-b"),
                       ("test_used_for_selection", True), ("depth", 27), ("k", 64), ("seed", 1)):
        altered = copy.deepcopy(selection)
        altered["contract"][key] = value
        with pytest.raises(ValueError, match="another dictionary"):
            validate_selection(altered, **expected)
    altered = copy.deepcopy(selection)
    altered["contract"]["selection_splits"] = ["train", "test"]
    with pytest.raises(ValueError, match="validation only"):
        validate_selection(altered, **expected)
    altered = copy.deepcopy(selection)
    altered["records"][0]["concept_index"] = 9
    with pytest.raises(ValueError, match="decoder target"):
        validate_selection(altered, **expected)
    altered = copy.deepcopy(selection)
    altered["records"][0]["random_control_feature"] = 2
    with pytest.raises(ValueError, match="distinct dictionary"):
        validate_selection(altered, **expected)
    altered = copy.deepcopy(selection)
    altered["records"].append(altered["records"][0].copy())
    with pytest.raises(ValueError, match="duplicate"):
        validate_selection(altered, **expected)


def test_patch_preserves_cls_invalid_tokens_and_native_bfloat16_dtype():
    original = jnp.arange(2 * 4 * 6, dtype=jnp.float32).reshape(2, 4, 6).astype(jnp.bfloat16)
    valid = jnp.array([[True, False, True], [False, True, True]])
    replacement = original[:, :-1].astype(jnp.float32) + .51
    patched, mask = physical_patch(original, replacement, valid)
    assert patched.dtype == original.dtype
    np.testing.assert_array_equal(mask[:, -1], False)
    np.testing.assert_array_equal(mask[:, :-1], valid)
    np.testing.assert_array_equal(np.asarray(patched)[~np.asarray(mask)], np.asarray(original)[~np.asarray(mask)])
    np.testing.assert_array_equal(np.asarray(patched[:, :-1])[np.asarray(valid)],
                                  np.asarray(replacement.astype(jnp.bfloat16))[np.asarray(valid)])
    absolute, relative = patch_dose(original, patched, valid)
    assert (absolute > 0).all() and (relative > 0).all()
    sham, sham_mask = physical_patch(original, original[:, :-1], valid)
    np.testing.assert_array_equal(sham, original)
    np.testing.assert_array_equal(sham_mask, mask)
    np.testing.assert_array_equal(patch_dose(original, sham, valid)[0], 0)


def test_direction_controls_share_activation_amplitude_without_reconstructing_original():
    x = jnp.array([[[2., -3., 1.], [5., 2., -1.], [7., 1., 4.]]])
    amplitudes = jnp.array([[.2, 0., .7]])
    directions = (jnp.array([1., 0., 0.]), jnp.array([0., -1., 0.]),
                  jnp.array([0., 0., 1.]))
    for dose in (.5, 1.):
        norms = []
        for direction in directions:
            edited = direction_edit(x, amplitudes, direction, dose)
            np.testing.assert_array_equal(edited[:, 1], x[:, 1])
            norms.append(np.linalg.norm(np.asarray(edited - x), axis=-1))
        for norm in norms:
            np.testing.assert_allclose(norm, np.asarray(amplitudes) * dose, atol=3e-7)
    np.testing.assert_array_equal(direction_edit(x, amplitudes, directions[0], 0.), x)


def test_variants_and_random_controls_are_prespecified_without_test_data():
    selection, _ = selection_fixture()
    variants = feature_variants(selection["records"])
    assert len(variants) == 9
    assert [variant["kind"] for variant in variants[:3]] == ["identity", "sham", "reconstruction"]
    assert len({variant["name"] for variant in variants}) == len(variants)
    decoder = np.eye(8, 16, dtype=np.float32)
    first = random_directions("parent_1190k", [2, 4], decoder)
    reordered = random_directions("parent_1190k", [4, 2], decoder)
    np.testing.assert_array_equal(first[2], reordered[2])
    np.testing.assert_allclose(np.linalg.norm(first[2]), 1, atol=1e-6)
    np.testing.assert_allclose(np.dot(first[2], decoder[2]), 0, atol=1e-7)
    assert not np.array_equal(first[2], random_directions("periodic_1209k", [2], decoder)[2])
