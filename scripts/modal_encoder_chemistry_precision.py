"""Bounded native BF16 batch-shape/trace-output diagnostic; no training."""
from __future__ import annotations

import json
import os
import time

import modal

from scripts.modal_encoder_chemistry import (
    LOCAL, PROJECT, ROOT, TOKEN_DEPTHS, batch_inputs, error_metrics, image,
    metadata, restore, volume, write_json,
)

app = modal.App("spectra-encoder-chemistry-precision")


@app.function(image=image, gpu="H100", cpu=8, memory=65536, timeout=1800,
              volumes={"/results": volume})
def diagnose(name: str = "periodic_1209k") -> dict:
    os.chdir(PROJECT)
    import jax
    import jax.numpy as jnp
    import ml_dtypes
    import numpy as np
    from flax import nnx
    from scripts.encoder_chemistry_interventions import identity_interventions, trace_encoder

    jax.config.update("jax_default_matmul_precision", "highest")
    encoder, config = restore(name)
    schema = config["encoder_metadata_schema"]
    source = ROOT / name
    data = np.load(ROOT / "panel/inputs.npz", allow_pickle=False)
    with np.load(source / "pilot_tokens.npz", allow_pickle=False) as pilot:
        rows, expected, pilot_trace, valid = pilot["rows"], pilot["native"], pilot["instrumented"], pilot["valid"]
    batches = {size: batch_inputs(data, rows, size) for size in (8, 16)}

    @nnx.jit
    def native(enc, batch):
        return enc(batch["peak_mz"], batch["peak_intensity"], valid_mask=batch["peak_valid_mask"],
                   visible_mask=batch["peak_valid_mask"], precursor_mz=batch["precursor_mz"],
                   spectrum_metadata=metadata(batch, schema))

    @nnx.jit
    def full(enc, batch, controls):
        return trace_encoder(enc, batch["peak_mz"], batch["peak_intensity"],
                             valid_mask=batch["peak_valid_mask"], visible_mask=batch["peak_valid_mask"],
                             spectrum_metadata=metadata(batch, schema), interventions=controls)

    @nnx.jit
    def reduced(enc, batch):
        # Reproduce the original cache extractor's static identity controls and
        # retained output graph, including its summaries and lossless bf16 bits.
        trace = trace_encoder(enc, batch["peak_mz"], batch["peak_intensity"],
                              valid_mask=batch["peak_valid_mask"], visible_mask=batch["peak_valid_mask"],
                              spectrum_metadata=metadata(batch, schema))
        states = jnp.concatenate([trace.raw, trace.final_normalized[None], trace.final[None]], axis=0)
        peaks = states[:, :, :-1].astype(jnp.float32)
        mask = batch["peak_valid_mask"][None, :, :, None]
        mean = (peaks * mask).sum(2) / jnp.maximum(mask.sum(2), 1)
        maximum = jnp.max(jnp.where(mask, peaks, -jnp.inf), axis=2)
        tokens = jax.lax.bitcast_convert_type(states[jnp.array(TOKEN_DEPTHS)].astype(jnp.bfloat16), jnp.uint16)
        def rms(value):
            return jnp.sqrt(((value.astype(jnp.float32)[:, :, :-1] ** 2) * mask).sum((2, 3))
                            / jnp.maximum(mask.sum((2, 3)) * config["model_dim"], 1))
        return mean, maximum, states[:, :, -1].astype(jnp.float32), tokens, rms(trace.attention_update), rms(trace.mlp_update)

    outcomes, tensors = {}, {"pilot_native": expected, "pilot_trace": pilot_trace}
    for size in (8, 16):
        for kind in ("native", "full_dynamic", "reduced_static"):
            start = time.monotonic()
            if kind == "native":
                final = native(encoder, batches[size])
            elif kind == "full_dynamic":
                result = full(encoder, batches[size], identity_interventions(encoder, size, 127))
                final = result.final
            else:
                result = reduced(encoder, batches[size])
                final = np.asarray(result[3][-1]).view(ml_dtypes.bfloat16)
            final = np.asarray(final, np.float32)[:len(rows)]
            key = f"{kind}_b{size}"
            tensors[key] = final
            outcomes[key] = {"seconds": time.monotonic() - start,
                             "vs_pilot_native": error_metrics(final, expected, valid),
                             "vs_pilot_trace": error_metrics(final, pilot_trace, valid)}
            print(key, outcomes[key], flush=True)
    record = {"name": name, "rows": rows.tolist(), "device": str(jax.devices()[0]), "outcomes": outcomes,
              "batch16_policy": "identical first8 pilot rows plus8 empty padded rows; mask includes CLS"}
    write_json(source / "precision_diagnostic.json", record)
    np.savez_compressed(source / "precision_diagnostic_tokens.npz", **tensors, valid=valid, rows=rows)
    volume.commit()
    return record


@app.local_entrypoint()
def main(name: str = "periodic_1209k"):
    call = diagnose.spawn(name)
    write_json(LOCAL / f"precision_diagnostic_launch_{name}.json", {"name": name, "call_id": call.object_id})
    print(json.dumps({"name": name, "call_id": call.object_id}), flush=True)
    result = call.get()
    write_json(LOCAL / name / "precision_diagnostic.json", result)
