"""Residual-preserving SAE feature interventions on native raw block-19 tokens.

Features and firing-rate controls must be frozen using train/validation only.
This runner evaluates a selected dictionary independently for each checkpoint;
feature numbers never imply correspondence across dictionaries.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import modal
import numpy as np

from scripts.modal_encoder_chemistry import (
    LOCAL, PROJECT, ROOT, batch_inputs, error_metrics, image, provenance,
    restore, sha256, volume, write_json,
)
from scripts.modal_encoder_chemistry_causal import load_endpoints, make_functions, selected_rows

app = modal.App("spectra-native-chemistry-sae-causal")
DEPTH, K, SAE_SEED = 19, 32, 0
COHORT_SEED = 20260930


def validate_selection(selection, *, encoder, params_sha256, panel_sha256,
                       number_features, target_names):
    """Reject stale, cross-dictionary, test-selected or mislabeled features."""
    contract = selection["contract"]
    expected = {"encoder": encoder, "depth": DEPTH, "k": K, "seed": SAE_SEED,
                "params_sha256": params_sha256, "test_used_for_selection": False}
    if any(contract.get(key) != value for key, value in expected.items()):
        raise ValueError("Chemical feature selection belongs to another dictionary or selection protocol")
    if set(contract["selection_splits"]) != {"train", "val"}:
        raise ValueError("SAE feature selection must use train and validation only")
    if any(contract["panel_sha256"].get(key) != value for key, value in panel_sha256.items()):
        raise ValueError("Feature selection panel differs from the causal panel")
    records = selection["records"]
    if not 1 <= len(records) <= 12:
        raise ValueError("Necessity is supported only for 1–12 train/validation-selected features")
    features = []
    for record in records:
        feature, control = record["feature"], record["random_control_feature"]
        if (type(feature) is not int or type(control) is not int
            or not 0 <= feature < number_features or not 0 <= control < number_features
            or feature == control):
            raise ValueError("Selected and control feature IDs must be distinct dictionary indices")
        index = record["concept_index"]
        if type(index) is not int or not 0 <= index < 22 or target_names[166 + index] != record["concept"]:
            raise ValueError("Selected concept does not map to the frozen decoder target")
        features.append(feature)
    if len(features) != len(set(features)):
        raise ValueError("Select each dictionary feature once; duplicate feature IDs are ambiguous")
    return records


def physical_patch(original, replacement, valid):
    """Return native-dtype patch values; CLS and padding remain bitwise untouched."""
    import jax.numpy as jnp
    if original.shape[:2] != (valid.shape[0], valid.shape[1] + 1):
        raise ValueError("Expected physical peak slots followed by exactly one CLS token")
    if replacement.shape != original[:, :-1].shape or valid.dtype != jnp.bool_:
        raise ValueError("Replacement shape or physical validity dtype differs from the native slots")
    mask = jnp.concatenate((valid, jnp.zeros((len(valid), 1), jnp.bool_)), axis=1)
    proposed = original.at[:, :-1].set(replacement.astype(original.dtype))
    return jnp.where(mask[..., None], proposed, original), mask


def direction_edit(original_peaks, activation_original_units, direction, dose):
    """Subtract a supplied unit direction with the same target activation dose."""
    import jax.numpy as jnp
    return original_peaks.astype(jnp.float32) - (
        jnp.asarray(dose, jnp.float32) * activation_original_units[..., None]
        * direction.astype(jnp.float32)
    )


def feature_variants(records):
    result = [
        {"name": "identity", "kind": "identity"},
        {"name": "raw19_physical_sham", "kind": "sham"},
        {"name": "full_sae_reconstruction", "kind": "reconstruction"},
    ]
    for record in records:
        for dose in (.5, 1.):
            for kind in ("feature", "matched_feature_direction", "random_unit_direction"):
                result.append({"name": f"feature{record['feature']}_{kind}_dose{dose:g}",
                    "kind": kind, "feature": record["feature"], "dose": dose,
                    "concept": record["concept"], "concept_index": record["concept_index"],
                    "control_feature": record["random_control_feature"]})
    return result


def random_directions(encoder, features, decoder):
    """Fixed unit controls orthogonal to target rows, independent of test outcomes."""
    name_seed = int.from_bytes(hashlib.sha256(encoder.encode()).digest()[:4], "little")
    result = {}
    for feature in features:
        rng = np.random.default_rng(np.random.SeedSequence([COHORT_SEED, name_seed, feature]))
        direction = rng.standard_normal(decoder.shape[1])
        target = np.asarray(decoder[feature], np.float64)
        target /= np.linalg.norm(target)
        direction -= np.dot(direction, target) * target
        if np.linalg.norm(direction) == 0:
            raise ValueError("A random orthogonal control requires a nonzero orthogonal subspace")
        result[feature] = (direction / np.linalg.norm(direction)).astype(np.float32)
    return result


def patch_dose(original, patched, valid):
    """Realized original-unit L2 dose after casting the patch to native bf16."""
    delta = (np.asarray(patched[:, :-1], np.float64)
             - np.asarray(original[:, :-1], np.float64)) * np.asarray(valid)[..., None]
    absolute = np.sqrt(np.square(delta).sum((1, 2)))
    baseline = np.asarray(original[:, :-1], np.float64) * np.asarray(valid)[..., None]
    relative = absolute / np.maximum(np.sqrt(np.square(baseline).sum((1, 2))), 1e-20)
    return absolute.astype(np.float32), relative.astype(np.float32)


def exact_parity(actual, expected, valid, description):
    result = error_metrics(actual, expected, valid)
    if not result["finite"] or not result["bitwise_equal"]:
        raise ValueError(f"{description} failed bitwise parity: {result}")
    return result


@app.function(image=image, gpu="H100", cpu=8, memory=65536, timeout=21600,
              volumes={"/results": volume})
def evaluate(name: str, fold: str = "test", size: int = 512) -> dict:
    os.chdir(PROJECT)
    import jax
    import jax.numpy as jnp
    import ml_dtypes
    from scripts.encoder_chemistry_interventions import identity_interventions
    from scripts.encoder_chemistry_sae import decode_sae, encode_sae, load_sae

    if (fold, size) not in (("test", 512), ("val", 128)):
        raise ValueError("Prespecified SAE causal cohorts are test512 or optional validation128")
    source = ROOT / name
    sae_directory = source / "sae" / f"depth{DEPTH}" / f"k{K}" / f"seed{SAE_SEED}"
    output = source / "causal" / f"sae_depth{DEPTH}_k{K}_seed{SAE_SEED}_{fold}_{size}"
    if (output / "manifest.json").exists():
        raise FileExistsError(f"SAE causal result already exists: {output}")
    jax.config.update("jax_default_matmul_precision", "highest")
    readouts, endpoints = load_endpoints(name)
    sae, sae_record = load_sae(sae_directory)
    extraction = json.loads((source / "extraction.json").read_text())
    extraction_hash = sha256(source / "extraction.json")
    panel_hashes = {filename: sha256(ROOT / "panel" / filename)
                    for filename in ("inputs.npz", "labels.npz", "manifest.json")}
    contract = sae_record["contract"]
    required = {"encoder": name, "depth": DEPTH, "k": K, "seed": SAE_SEED,
                "extraction_sha256": extraction_hash,
                "token_source_sha256": extraction["files"]["tokens_bf16"]["sha256"]}
    if any(contract.get(key) != value for key, value in required.items()):
        raise ValueError("SAE dictionary differs from the frozen raw19 token contract")
    if any(contract["panel_sha256"].get(key) != value for key, value in panel_hashes.items()):
        raise ValueError("SAE panel differs from the causal panel")
    if sae_record["k"] != K or sae_record["seed"] != SAE_SEED or sae_record["labels_used"]:
        raise ValueError("Expected the prespecified unsupervised k32/seed0 SAE")
    if contract["source_sha256"]["encoder_chemistry_sae.py"] != sha256(PROJECT / "scripts/encoder_chemistry_sae.py"):
        raise ValueError("SAE encoding/edit implementation changed since fitting")
    decoder = np.asarray(sae["decoder"], np.float32)
    norms = np.linalg.norm(decoder, axis=1)
    if (not all(np.isfinite(np.asarray(value)).all() for value in sae.values())
        or float(sae["scale"]) <= 0 or not np.allclose(norms, 1., atol=1e-5, rtol=1e-5)):
        raise ValueError("SAE needs finite parameters, positive normalization scale and unit decoder rows")
    selection_file = sae_directory / "chemical_features.json"
    selection = json.loads(selection_file.read_text())
    records = validate_selection(selection, encoder=name, params_sha256=sae_record["weights_sha256"],
        panel_sha256=panel_hashes, number_features=sae_record["features"], target_names=endpoints[0]["target_names"])
    with np.load(ROOT / "panel/inputs.npz", allow_pickle=False) as archive:
        data = dict(archive)
    selection_contract = selection["contract"]
    for filename, digest in selection_contract["panel_sha256"].items():
        if sha256(ROOT / "panel" / filename) != digest:
            raise ValueError(f"Feature selection panel changed: {filename}")
    for filename_key, hash_key in (("metadata_filename", "trained_sae_sha256"),
                                    ("molecule_max_filename", "molecule_max_sha256")):
        if sha256(sae_directory / selection_contract[filename_key]) != selection_contract[hash_key]:
            raise ValueError("Dictionary metadata/activation summary changed after feature selection")
    if hashlib.sha256(data["molecule"].tobytes()).hexdigest() != selection_contract["molecule_ids_sha256"]:
        raise ValueError("Feature selection molecule identities differ from the causal panel")
    rows = selected_rows(data, fold, size)
    encoder, config = restore(name)
    current = provenance(name, config)
    if config != extraction["config"] or current["inputs_sha256"] != extraction["inputs_sha256"]:
        raise ValueError("Native encoder/input configuration changed since token extraction")
    for path, digest in extraction["source_sha256"].items():
        if path != "scripts/modal_encoder_chemistry.py" and current["source_sha256"][path] != digest:
            raise ValueError(f"Native source differs from the token extraction: {path}")
    token_file = source / "tokens_bf16.npy"
    if token_file.stat().st_size != extraction["files"]["tokens_bf16"]["bytes"]:
        raise ValueError("Token cache size differs from its pinned extraction")
    cache = np.load(token_file, mmap_mode="r", allow_pickle=False)
    if cache.dtype != np.uint16 or list(cache.shape) != extraction["shapes"]["tokens_bf16"]:
        raise ValueError("Unexpected native token storage schema")
    forward, score = make_functions(config["encoder_metadata_schema"])
    batch_size = 8
    controls = identity_interventions(encoder, batch_size, 127)
    variants = feature_variants(records)
    feature_ids = [record["feature"] for record in records]
    feature_position = {feature: index for index, feature in enumerate(feature_ids)}
    directions = random_directions(name, feature_ids, decoder)
    # Unit rows differ from norm1 only by float32 normalization rounding. Match
    # control directions to the target row's measured norm before native cast.
    control_directions = {}
    for selected in records:
        feature = selected["feature"]
        other = decoder[selected["random_control_feature"]]
        control_directions[feature] = {
            "feature": jnp.asarray(decoder[feature]),
            "matched_feature_direction": jnp.asarray(other / np.linalg.norm(other) * norms[feature]),
            "random_unit_direction": jnp.asarray(directions[feature] * norms[feature]),
        }

    @jax.jit
    def encode_and_reconstruct(x):
        code = encode_sae(sae, x, K)
        return code, decode_sae(sae, code)

    @jax.jit
    def edit(x, amplitude, direction, dose, valid):
        proposed = direction_edit(x[:, :-1], amplitude, direction, dose)
        return physical_patch(x, proposed, valid)

    logits = np.empty((len(variants), size, 3, 188), np.float32)
    actual_l2 = np.zeros((len(variants), size), np.float32)
    relative_l2 = np.zeros_like(actual_l2)
    requested_l2 = np.zeros_like(actual_l2)
    supported = np.ones((len(variants), size), bool)
    activations = np.zeros((len(records), size, 127), np.float32)
    code_indices = np.full((size, 127, K), -1, np.int32)
    code_values = np.zeros((size, 127, K), np.float32)
    started = time.monotonic()
    cache_parity, sham_parity = [], []
    with np.load(source / "pilot_tokens.npz", allow_pickle=False) as pilot:
        pilot_trace = forward(encoder, batch_inputs(data, pilot["rows"], batch_size), controls)
        pilot_parity = exact_parity(pilot_trace.final, pilot["native"], pilot["valid"], "SAE runner native pilot")
    for start in range(0, size, batch_size):
        ids = rows[start:start + batch_size]
        inputs = batch_inputs(data, ids, batch_size)
        clean = forward(encoder, inputs, controls)
        valid = inputs["peak_valid_mask"]
        token_valid = np.concatenate((valid, np.ones((batch_size, 1), bool)), axis=1)
        original = clean.raw[DEPTH]
        cache_parity.append({
            "raw19": exact_parity(original, cache[extraction["token_depths"].index(DEPTH), ids].view(ml_dtypes.bfloat16),
                                   token_valid, "SAE raw19 cache"),
            "final27": exact_parity(clean.final, cache[extraction["token_depths"].index(27), ids].view(ml_dtypes.bfloat16),
                                    token_valid, "SAE frozen readout cache"),
        })
        code, reconstructed = encode_and_reconstruct(original[:, :-1])
        indices, values = np.asarray(code.indices), np.asarray(code.values)
        code_indices[start:start + batch_size] = np.where(valid[..., None], indices, -1)
        code_values[start:start + batch_size] = np.where(valid[..., None], values, 0)
        amplitudes = {}
        for feature in feature_ids:
            amplitude = np.sum(np.where(indices == feature, values, 0), axis=-1) * valid
            activations[feature_position[feature], start:start + batch_size] = amplitude
            amplitudes[feature] = jnp.asarray(amplitude * float(sae["scale"]))
        for index, variant in enumerate(variants):
            kind = variant["kind"]
            if kind == "identity":
                tokens = clean.final
            else:
                if kind == "sham":
                    patched, patch_mask = physical_patch(original, original[:, :-1], jnp.asarray(valid))
                elif kind == "reconstruction":
                    patched, patch_mask = physical_patch(original, reconstructed, jnp.asarray(valid))
                    reconstruction_delta = (np.asarray(reconstructed, np.float64)
                        - np.asarray(original[:, :-1], np.float64)) * valid[..., None]
                    requested_l2[index, start:start + batch_size] = np.sqrt(
                        np.square(reconstruction_delta).sum((1, 2)))
                else:
                    feature = variant["feature"]
                    dose = jnp.asarray(variant["dose"], jnp.float32)
                    patched, patch_mask = edit(original, amplitudes[feature], control_directions[feature][kind], dose,
                                               jnp.asarray(valid))
                    requested_l2[index, start:start + batch_size] = (
                        variant["dose"] * norms[feature]
                        * np.linalg.norm(np.asarray(amplitudes[feature]), axis=-1))
                    supported[index, start:start + batch_size] = np.any(
                        np.asarray(amplitudes[feature]) > 0, axis=-1)
                absolute, relative = patch_dose(original, patched, valid)
                actual_l2[index, start:start + batch_size] = absolute
                relative_l2[index, start:start + batch_size] = relative
                intervention = controls._replace(patch_layer=jnp.asarray(DEPTH, jnp.int32),
                    patch_values=patched, patch_mask=patch_mask)
                tokens = forward(encoder, inputs, intervention).final
                if kind == "sham":
                    sham_parity.append(exact_parity(tokens, clean.final, token_valid, "Raw19 physical-token sham"))
            value = np.asarray(score(readouts, tokens, valid), np.float32).transpose(1, 0, 2)
            if not np.isfinite(value).all():
                raise ValueError("SAE intervention produced nonfinite frozen-readout logits")
            logits[index, start:start + batch_size] = value
        print(name, "SAE interventions", fold, start + batch_size, size,
              "seconds", round(time.monotonic() - started, 1), flush=True)
    np.testing.assert_array_equal(logits[0], logits[1], err_msg="Raw19 identity patch must preserve frozen logits")
    activation_statistics = []
    physical_mask = data["peak_valid_mask"][rows]
    for index, selected in enumerate(records):
        valid_values = activations[index][physical_mask]
        positive = valid_values[valid_values > 0]
        activation_statistics.append({
            "feature": selected["feature"], "active_physical_tokens": int((valid_values > 0).sum()),
            "physical_tokens": int(len(valid_values)), "active_molecules": int((activations[index] > 0).any(1).sum()),
            "positive_quantiles": ({str(q): float(np.quantile(positive, q)) for q in (.5, .9, .99)} if len(positive) else {}),
            "mean_activation_all_physical_tokens": float(valid_values.mean()),
            "units": "SAE activation before scalar conversion to original residual units",
        })
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "predictions.npz", logits=logits, rows=rows, source_rows=data["rows"][rows],
        variants=np.asarray([variant["name"] for variant in variants]),
        actual_l2_dose=actual_l2, actual_relative_l2_dose=relative_l2, requested_l2_dose=requested_l2,
        supported=supported,
        selected_feature_ids=np.asarray(feature_ids), selected_feature_activations=activations,
        sparse_code_indices=code_indices, sparse_code_values=code_values, valid_peak_mask=physical_mask,
        random_directions=np.stack([directions[feature] for feature in feature_ids]))
    result = {**current, "complete": True, "fold": fold, "rows": rows.tolist(), "cohort_seed": COHORT_SEED,
        "depth": DEPTH, "k": K, "sae_seed": SAE_SEED, "variants": variants,
        "selection": selection, "selection_sha256": sha256(selection_file),
        "sae_params_sha256": sae_record["weights_sha256"], "sae_metadata": sae_record,
        "sae_metadata_sha256": sha256(sae_directory / "trained_sae.json"),
        "readout_weights_sha256": [endpoint["weights_sha256"] for endpoint in endpoints],
        "extraction_sha256": extraction_hash,
        "token_source_sha256": extraction["files"]["tokens_bf16"]["sha256"],
        "panel_sha256": panel_hashes, "baseline_pilot_parity": pilot_parity,
        "baseline_cache_parity": cache_parity, "raw19_sham_parity": sham_parity,
        "activation_statistics": activation_statistics,
        "source_hashes": {filename: sha256(PROJECT / "scripts" / filename) for filename in (
            "modal_encoder_chemistry_sae_causal.py", "modal_encoder_chemistry_causal.py",
            "encoder_chemistry_sae.py", "encoder_chemistry_interventions.py", "encoder_chemistry_readouts.py")},
        "patch_scope": "Valid physical peak slots only at raw19; CLS and padding unchanged; original reconstruction residual preserved for feature/direction edits.",
        "dose_contract": "All controls use exactly the target-feature activation amplitude and target decoder norm before bf16 casting; realized original-unit L2 is recorded after casting.",
        "random_direction_protocol": "Gaussian direction projected orthogonal to the target decoder row and unit-normalized; seeded by 20260930, SHA256-derived checkpoint-name integer, dictionary-local feature ID; no test tuning.",
        "control_direction_cosines": [{"feature": item["feature"],
            "matched_feature": float(np.dot(decoder[item["feature"]], decoder[item["random_control_feature"]])
                / (norms[item["feature"]] * norms[item["random_control_feature"]])),
            "random_orthogonal": float(np.dot(decoder[item["feature"]], directions[item["feature"]]) / norms[item["feature"]])}
            for item in records],
        "quality_policy": "No arbitrary reconstruction threshold is imposed. Full reconstruction is a separate damage baseline; poor quality qualifies feature interpretation. Association alone is not a mechanism.",
        "causal_scope": "Effects on three frozen token readouts under computational dictionary-direction removal; does not assert chemically realizable molecule edits or intrinsic semantic logits.",
        "seconds": time.monotonic() - started, "predictions_sha256": sha256(output / "predictions.npz")}
    write_json(output / "manifest.json", result)
    volume.commit()
    return result


@app.local_entrypoint()
def main(name: str = "parent_1190k", fold: str = "test", size: int = 512):
    result = evaluate.remote(name, fold, size)
    suffix = f"sae_depth{DEPTH}_k{K}_seed{SAE_SEED}_{fold}_{size}"
    output = LOCAL / name / "causal" / suffix
    write_json(output / "manifest.json", result)
    remote = str((ROOT / name / "causal" / suffix / "predictions.npz").relative_to("/results"))
    with (output / "predictions.npz").open("wb") as stream:
        for chunk in volume.read_file(remote):
            stream.write(chunk)
    if sha256(output / "predictions.npz") != result["predictions_sha256"]:
        raise ValueError("Downloaded SAE causal predictions differ from their recorded hash")
    print(json.dumps({"path": str(output), "complete": True}), flush=True)
