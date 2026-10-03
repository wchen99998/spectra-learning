"""Fixed input interventions and observed-CE controls for frozen native readouts.

Launch only after the full-trace, batch-eight cache and three final attentive
readouts are complete. Every encoder invocation returns the entire EncoderTrace.
Readouts run separately; metadata factorization swaps raw25 states while keeping
final normalization and the final affine inside the native encoder graph.
"""
from __future__ import annotations

import json
import os
import time

import modal
import numpy as np

from scripts.encoder_chemistry_corruptions import (
    DEFINITIONS, ENCODER_KEYS, corrupt_inputs, observed_ce_donors,
)
from scripts.modal_encoder_chemistry import (
    LOCAL, PROJECT, ROOT, batch_inputs, error_metrics, image, metadata,
    provenance, restore, sha256, volume, write_json,
)

app = modal.App("spectra-native-chemistry-input-effects")
METADATA_KEYS = tuple(key for key in ENCODER_KEYS if not key.startswith("peak_"))
REPEAT_VARIANTS = (
    "anchor_spectrum_anchor_metadata", "anchor_spectrum_repeat_metadata",
    "repeat_spectrum_repeat_metadata", "repeat_spectrum_anchor_metadata",
)
FACTORIAL_VARIANTS = (
    "anchor_blocks_anchor_final", "anchor_blocks_donor_final",
    "donor_blocks_anchor_final", "donor_blocks_donor_final",
)


def observed_repeat_arms(data, rows, repeats):
    """Four measured-spectrum/metadata combinations; unsupported rows stay clean."""
    donors, facts = observed_ce_donors(data, rows, repeats)
    supported = donors >= 0
    anchor = {key: np.array(data[key][rows], copy=True) for key in ENCODER_KEYS}
    alternate = {key: value.copy() for key, value in anchor.items()}
    for key in ENCODER_KEYS:
        alternate[key][supported] = repeats[key][donors[supported]]
    anchor_repeat = {key: value.copy() for key, value in anchor.items()}
    repeat_anchor = {key: value.copy() for key, value in alternate.items()}
    for key in METADATA_KEYS:
        anchor_repeat[key] = alternate[key].copy()
        repeat_anchor[key] = anchor[key].copy()
    facts.update(
        repeat_indices=donors.tolist(),
        metadata_fields=list(METADATA_KEYS),
        changed_metadata_fields={key: np.any(
            (anchor[key] != alternate[key]).reshape(len(rows), -1), axis=1,
        ).tolist() for key in METADATA_KEYS},
        interpretation="Observed alternate spectra plus metadata-only hybrids; no molecule counterfactual is asserted.",
    )
    return (anchor, anchor_repeat, alternate, repeat_anchor), supported, facts


def metadata_path_patch(controls, raw_final_block, physical_valid, *, depth=25):
    """Swap completed block states; native final norm/affine still run unchanged."""
    import jax.numpy as jnp
    mask = jnp.concatenate((physical_valid, jnp.ones((len(physical_valid), 1), jnp.bool_)), axis=1)
    if raw_final_block.shape[:2] != mask.shape:
        raise ValueError("Metadata path patches require physical token slots plus the trailing CLS")
    return controls._replace(patch_layer=jnp.asarray(depth, jnp.int32),
                             patch_values=raw_final_block, patch_mask=mask)


def make_functions(schema):
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from scripts.encoder_chemistry_interventions import trace_encoder
    from scripts.encoder_chemistry_readouts import apply_readout

    @nnx.jit
    def traced(encoder, inputs, controls):
        return trace_encoder(
            encoder, inputs["peak_mz"], inputs["peak_intensity"],
            valid_mask=inputs["peak_valid_mask"], visible_mask=inputs["peak_valid_mask"],
            spectrum_metadata=metadata(inputs, schema), interventions=controls,
        )

    @jax.jit
    def score(params, tokens, valid):
        physical_mask = jnp.concatenate((valid, jnp.zeros((len(valid), 1), jnp.bool_)), axis=1)
        return jax.vmap(lambda p: apply_readout(p, tokens, physical_mask, "attention")[0])(params)

    return traced, score


def require_bitwise(actual, expected, valid, description):
    result = error_metrics(actual, expected, valid)
    if not result["finite"] or not result["bitwise_equal"]:
        raise ValueError(f"{description} is not bitwise equal: {result}")
    return result


def load_endpoints(name, extraction):
    """Pin every saved decoder to this exact input and token extraction contract."""
    import jax
    import jax.numpy as jnp
    from scripts.encoder_chemistry_readouts import load_readout
    source = ROOT / name
    token_file = source / "tokens_bf16.npy"
    extraction_hash = sha256(source / "extraction.json")
    params, records = [], []
    for seed in (0, 1, 2):
        p, info = load_readout(source / "readouts/depth27/attention" / f"seed{seed}")
        contract = info["contract"]
        expected = {
            "encoder": name, "checkpoint": extraction["checkpoint"], "depth": 27,
            "token_scope": "physical_peaks_only", "token_source": str(token_file),
            "token_source_sha256": extraction["files"]["tokens_bf16"]["sha256"],
            "extraction_sha256": extraction_hash,
        }
        if any(contract.get(key) != value for key, value in expected.items()):
            raise ValueError("Readout token/checkpoint contract differs from the completed extraction")
        if info["seed"] != seed or info["kind"] != "attention" or len(info["target_names"]) != 188:
            raise ValueError("Expected three prespecified attentive 188-target readouts")
        if contract["panel_sha256"]["inputs.npz"] != extraction["inputs_sha256"]:
            raise ValueError("Readout inputs differ from the encoder extraction")
        for filename, digest in contract["source_sha256"].items():
            if sha256(PROJECT / "scripts" / filename) != digest:
                raise ValueError(f"Readout source changed since fitting: {filename}")
        if records and info["target_names"] != records[0]["target_names"]:
            raise ValueError("Readout targets differ across seeds")
        params.append(p)
        records.append(info)
    return jax.tree.map(lambda *values: jnp.stack(values), *params), records


def _score_numpy(score, params, tokens, valid, length):
    result = np.asarray(score(params, tokens, valid), np.float32)[:, :length].transpose(1, 0, 2)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite fixed-readout logits")
    return result


@app.function(image=image, gpu="H100", cpu=8, memory=65536, timeout=14400,
              volumes={"/results": volume})
def evaluate(name: str, phase: str = "corruptions") -> dict:
    """Evaluate one explicit phase; keep expensive GPU launch under caller control."""
    os.chdir(PROJECT)
    import jax
    import ml_dtypes
    from scripts.encoder_chemistry_interventions import identity_interventions

    if phase not in ("corruptions", "observed_repeats", "metadata_factorial"):
        raise ValueError("Unknown input-effect phase")
    if len(DEFINITIONS) != 32:
        raise ValueError("The prespecified corruption family must contain exactly 32 variants")
    source = ROOT / name
    output = source / "causal" / f"input_{phase}"
    if (output / "manifest.json").exists():
        raise FileExistsError(f"Input-effect result already exists: {output}")
    extraction = json.loads((source / "extraction.json").read_text())
    if not extraction["complete"] or 27 not in extraction["token_depths"]:
        raise ValueError("A completed final-token extraction is required")
    with np.load(ROOT / "panel/inputs.npz", allow_pickle=False) as archive:
        data = dict(archive)
    with np.load(ROOT / "panel/repeat_inputs.npz", allow_pickle=False) as archive:
        repeats = dict(archive)
    with np.load(ROOT / "panel/weak_losses.npz", allow_pickle=False) as archive:
        losses = dict(archive)
    rows = np.flatnonzero(data["fold"] == "test")
    if len(rows) != 1024:
        raise ValueError("This study prespecifies all 1024 test molecules")
    jax.config.update("jax_default_matmul_precision", "highest")
    params, endpoints = load_endpoints(name, extraction)
    encoder, config = restore(name)
    current = provenance(name, config)
    if current["inputs_sha256"] != extraction["inputs_sha256"] or config != extraction["config"]:
        raise ValueError("Current model/input configuration differs from extraction")
    for path, digest in extraction["source_sha256"].items():
        if path != "scripts/modal_encoder_chemistry.py" and current["source_sha256"][path] != digest:
            raise ValueError(f"Native encoder/trace source changed since extraction: {path}")
    tokens_path = source / "tokens_bf16.npy"
    if tokens_path.stat().st_size != extraction["files"]["tokens_bf16"]["bytes"]:
        raise ValueError("Current token file size differs from its extraction contract")
    cache = np.load(tokens_path, mmap_mode="r", allow_pickle=False)
    if list(cache.shape) != extraction["shapes"]["tokens_bf16"] or cache.dtype != np.uint16:
        raise ValueError("Unexpected final-token storage schema")
    batch_size = 8
    controls = identity_interventions(encoder, batch_size, 127)
    traced, score = make_functions(config["encoder_metadata_schema"])
    output.mkdir(parents=True, exist_ok=True)
    record = {
        **current, "phase": phase, "complete": False, "rows": rows.tolist(),
        "batch_size": batch_size, "encoder_graph": "full EncoderTrace with dynamic identity controls",
        "extraction_sha256": sha256(source / "extraction.json"),
        "token_source_sha256": extraction["files"]["tokens_bf16"]["sha256"],
        "readout_weights_sha256": [info["weights_sha256"] for info in endpoints],
        "target_names": endpoints[0]["target_names"],
        "input_effect_sources": {filename: sha256(PROJECT / "scripts" / filename) for filename in (
            "modal_encoder_chemistry_input_effects.py", "encoder_chemistry_corruptions.py",
            "encoder_chemistry_readouts.py",
        )},
        "panel_sha256": {filename: sha256(ROOT / "panel" / filename) for filename in (
            "inputs.npz", "repeat_inputs.npz", "weak_losses.npz",
        )},
        "scope": "Effects on three fixed chemistry readouts of physical tokens; labels never enter the encoder; these are not molecular counterfactuals.",
    }
    started = time.monotonic()
    with np.load(source / "pilot_tokens.npz", allow_pickle=False) as pilot:
        pilot_rows = pilot["rows"]
        if len(pilot_rows) != batch_size:
            raise ValueError("Full-graph parity requires the fixed eight-row pilot")
        inputs = batch_inputs(data, pilot_rows, batch_size)
        trace = traced(encoder, inputs, controls)
        valid = np.concatenate((inputs["peak_valid_mask"], np.ones((batch_size, 1), bool)), axis=1)
        record["pilot_parity"] = {
            "native": require_bitwise(trace.final, pilot["native"], valid, "Input forward versus native pilot"),
            "instrumented": require_bitwise(trace.final, pilot["instrumented"], valid, "Input forward versus full-trace pilot"),
        }
    baseline = np.empty((len(rows), 3, 188), np.float32)
    cache_parity = []
    for start in range(0, len(rows), batch_size):
        ids = rows[start:start + batch_size]
        inputs = batch_inputs(data, ids, batch_size)
        trace = traced(encoder, inputs, controls)
        valid = np.concatenate((inputs["peak_valid_mask"], np.ones((batch_size, 1), bool)), axis=1)
        target = cache[extraction["token_depths"].index(27), ids].view(ml_dtypes.bfloat16)
        cache_parity.append(require_bitwise(trace.final, target, valid, "Input forward versus fitted token cache"))
        baseline[start:start + len(ids)] = _score_numpy(score, params, trace.final, inputs["peak_valid_mask"], len(ids))
    record["baseline_cache_parity"] = cache_parity
    print(name, phase, "all 1024 baseline cache rows bitwise equal", flush=True)
    if phase == "corruptions":
        labels = ["identity", *DEFINITIONS]
        logits = np.empty((len(labels), len(rows), 3, 188), np.float32)
        supported = np.zeros((len(labels), len(rows)), bool)
        changed = np.zeros_like(supported)
        targets = np.zeros((len(labels), len(rows), 127), bool)
        logits[0], supported[0] = baseline, True
        interventions = []
        for variant, label in enumerate(labels[1:], start=1):
            modified, target, facts = corrupt_inputs(data, rows, name=label, repeats=repeats, weak_losses=losses)
            supported[variant] = facts["supported_rows"]
            changed[variant] = facts["changed_rows"]
            targets[variant] = target
            for start in range(0, len(rows), batch_size):
                positions = np.arange(start, min(start + batch_size, len(rows)))
                inputs = batch_inputs(modified, positions, batch_size)
                trace = traced(encoder, inputs, controls)
                logits[variant, positions] = _score_numpy(score, params, trace.final, inputs["peak_valid_mask"], len(positions))
            np.testing.assert_array_equal(logits[variant, ~supported[variant]], baseline[~supported[variant]],
                                          err_msg="Unsupported unchanged corruption rows must be exact shams")
            interventions.append(facts)
            print(name, phase, label, "supported", int(supported[variant].sum()),
                  "seconds", round(time.monotonic() - started, 1), flush=True)
        np.savez_compressed(output / "predictions.npz", logits=logits, rows=rows,
            source_rows=data["rows"][rows], variants=np.asarray(labels), supported=supported,
            changed=changed, target_peak_mask=targets)
        record["interventions"] = interventions
    elif phase == "observed_repeats":
        arms, supported, facts = observed_repeat_arms(data, rows, repeats)
        logits = np.empty((4, len(rows), 3, 188), np.float32)
        logits[0] = baseline
        valid_peaks = np.stack([arm["peak_valid_mask"] for arm in arms])
        for variant in range(1, 4):
            for start in range(0, len(rows), batch_size):
                positions = np.arange(start, min(start + batch_size, len(rows)))
                inputs = batch_inputs(arms[variant], positions, batch_size)
                trace = traced(encoder, inputs, controls)
                logits[variant, positions] = _score_numpy(score, params, trace.final, inputs["peak_valid_mask"], len(positions))
            np.testing.assert_array_equal(logits[variant, ~supported], baseline[~supported])
        np.savez_compressed(output / "predictions.npz", logits=logits, rows=rows,
            source_rows=data["rows"][rows], variants=np.asarray(REPEAT_VARIANTS), supported=supported,
            donor_repeat_indices=np.asarray(facts["repeat_indices"]), valid_peak_mask=valid_peaks)
        record["repeat_matching"] = facts
    else:
        if len(encoder.blocks) != 25:
            raise ValueError("The metadata path factorial is pinned to 25 native encoder blocks")
        donor_inputs, _, facts = corrupt_inputs(data, rows, name="observed_ce_metadata", repeats=repeats)
        supported = np.asarray(facts["supported_rows"], bool)
        logits = np.empty((4, len(rows), 3, 188), np.float32)
        diagonal_shams = []
        for start in range(0, len(rows), batch_size):
            ids = rows[start:start + batch_size]
            positions = np.arange(start, start + len(ids))
            anchor = batch_inputs(data, ids, batch_size)
            donor = batch_inputs(donor_inputs, positions, batch_size)
            anchor_trace = traced(encoder, anchor, controls)
            donor_trace = traced(encoder, donor, controls)
            # Metadata takes effect throughout the blocks and through a final
            # affine. Swap raw25 so final normalization and the selected affine
            # remain inside the identical, unpruned native encoder graph.
            anchor_blocks = metadata_path_patch(controls, anchor_trace.raw[25], anchor["peak_valid_mask"])
            donor_blocks = metadata_path_patch(controls, donor_trace.raw[25], donor["peak_valid_mask"])
            anchor_self = traced(encoder, anchor, anchor_blocks)
            donor_self = traced(encoder, donor, donor_blocks)
            valid = np.concatenate((anchor["peak_valid_mask"], np.ones((batch_size, 1), bool)), axis=1)
            comparisons = {
                "anchor_raw25_self_patch": error_metrics(anchor_self.final, anchor_trace.final, valid),
                "donor_raw25_self_patch": error_metrics(donor_self.final, donor_trace.final, valid),
            }
            diagonal_shams.append(comparisons)
            # Save measured failure before stopping; no silent tolerance changes.
            if not all(item["finite"] and item["bitwise_equal"] for item in
                       comparisons.values()):
                record.update(factorization_passed=False, diagonal_shams=diagonal_shams,
                              failed_panel_rows=ids.tolist())
                write_json(output / "factorization_failure.json", record)
                volume.commit()
                raise ValueError("Raw25 metadata-path self-patch is not bitwise equal to its full-trace diagonal")
            anchor_blocks_donor_final = traced(encoder, donor, anchor_blocks)
            donor_blocks_anchor_final = traced(encoder, anchor, donor_blocks)
            combined = (anchor_trace.final, anchor_blocks_donor_final.final,
                        donor_blocks_anchor_final.final, donor_trace.final)
            for variant in range(4):
                logits[variant, positions] = _score_numpy(score, params, combined[variant], anchor["peak_valid_mask"], len(ids))
        np.testing.assert_array_equal(logits[0], baseline)
        for variant in range(1, 4):
            np.testing.assert_array_equal(logits[variant, ~supported], baseline[~supported])
        np.savez_compressed(output / "predictions.npz", logits=logits, rows=rows,
            source_rows=data["rows"][rows], variants=np.asarray(FACTORIAL_VARIANTS), supported=supported,
            donor_repeat_indices=np.asarray(facts["facts"]["repeat_indices"]),
            target_peak_mask=np.zeros((len(rows), 127), bool))
        record.update(repeat_matching=facts, diagonal_shams=diagonal_shams,
                      factorization_passed=True,
                      factorial_scope="One fixed physical spectrum; observed anchor/donor CE changes only. Raw25 activation patches swap all valid physical and CLS states between CE contexts. Native final normalization and final affine remain inside the identical full-trace graph; diagonal self-patches must be bitwise exact.")
    record.update(complete=True, seconds=time.monotonic() - started,
                  predictions_sha256=sha256(output / "predictions.npz"))
    write_json(output / "manifest.json", record)
    volume.commit()
    return record


@app.local_entrypoint()
def main(name: str = "parent_1190k", phase: str = "corruptions"):
    result = evaluate.remote(name, phase)
    destination = LOCAL / name / "causal" / f"input_{phase}"
    write_json(destination / "manifest.json", result)
    remote = str((ROOT / name / "causal" / f"input_{phase}" / "predictions.npz").relative_to("/results"))
    with (destination / "predictions.npz").open("wb") as stream:
        for chunk in volume.read_file(remote):
            stream.write(chunk)
    if sha256(destination / "predictions.npz") != result["predictions_sha256"]:
        raise ValueError("Downloaded input-effect predictions differ from their recorded hash")
    print(json.dumps({"path": str(destination), "phase": phase, "complete": True}), flush=True)
