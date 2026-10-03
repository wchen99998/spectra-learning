"""Localize native JAX / existing Torch-export inference differences on eight rows.

This diagnostic preserves both production inference paths and writes only its
new backend_parity result directory. It does not refit or modify either model.
"""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import time

import modal

from scripts.modal_encoder_chemistry import (
    EXPORTS, LOCAL, PROJECT, ROOT, batch_inputs, error_metrics, export_volume,
    image, metadata, provenance, restore, sha256, volume, write_json,
)


app = modal.App("spectra-encoder-backend-parity")


@app.function(image=image, gpu="H100", cpu=8, memory=65536, timeout=3600,
              volumes={"/results": volume, "/exports": export_volume})
def diagnose(name: str = "parent_1190k") -> dict:
    os.chdir(PROJECT)
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx
    from scripts.encoder_chemistry_interventions import identity_interventions, trace_encoder

    started = time.monotonic()
    destination = ROOT / "backend_parity" / name
    destination.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_default_matmul_precision", "highest")
    encoder, config = restore(name)
    with np.load(ROOT / "panel/inputs.npz") as archive:
        batch = batch_inputs(archive, np.arange(4096, 4104), 8)
    schema = config["encoder_metadata_schema"]
    controls = identity_interventions(encoder, 8, 127)

    @nnx.jit
    def native(enc, inputs):
        return enc(inputs["peak_mz"], inputs["peak_intensity"],
                   valid_mask=inputs["peak_valid_mask"], visible_mask=inputs["peak_valid_mask"],
                   precursor_mz=inputs["precursor_mz"], spectrum_metadata=metadata(inputs, schema))

    @nnx.jit
    def snapshot(enc, inputs, interventions):
        spectrum_metadata = metadata(inputs, schema)
        traced = trace_encoder(enc, inputs["peak_mz"], inputs["peak_intensity"],
                               valid_mask=inputs["peak_valid_mask"],
                               visible_mask=inputs["peak_valid_mask"],
                               spectrum_metadata=spectrum_metadata, interventions=interventions)
        states = jnp.concatenate([traced.raw, traced.final_normalized[None], traced.final[None]])
        embedded = enc.embedder(enc._quantize_mz(inputs["peak_mz"]), inputs["peak_intensity"])
        condition_input = spectrum_metadata.astype(embedded.dtype)
        condition = enc._metadata_embedding(spectrum_metadata, embedded.dtype)
        return states, spectrum_metadata, condition_input, condition, embedded

    expected = np.asarray(native(encoder, batch), np.float32)
    states, meta_raw, condition_input, condition, embedded = snapshot(encoder, batch, controls)
    native_dtypes = {"states": str(states.dtype), "metadata_raw": str(meta_raw.dtype),
                     "metadata_embedder_input": str(condition_input.dtype),
                     "metadata_embedding": str(condition.dtype), "peak_embedder": str(embedded.dtype)}
    native_states = np.asarray(states, np.float32)
    native_metadata = np.asarray(meta_raw, np.float32)
    native_condition_input = np.asarray(condition_input, np.float32)
    native_condition = np.asarray(condition, np.float32)
    native_embedded = np.asarray(embedded, np.float32)
    valid = np.concatenate([batch["peak_valid_mask"], np.ones((8, 1), bool)], axis=1)
    trace_parity = error_metrics(native_states[-1], expected, valid)
    if not trace_parity["finite"] or trace_parity["relative_l2"] >= .01:
        raise ValueError(f"Native snapshot differs from native encoder: {trace_parity}")
    record = {**provenance(name, config), "rows": list(range(4096, 4104)),
              "purpose": "Layerwise localization of backend difference; no model modifications",
              "source_sha256_diagnostic": sha256(PROJECT / "scripts/modal_encoder_backend_parity.py"),
              "native_trace_parity": trace_parity, "native_dtypes": native_dtypes,
              "layer_labels": [*map(str, range(26)), "final_norm", "final"],
              "valid_peak_counts": batch["peak_valid_mask"].sum(1).tolist(),
              "exact_tensor_parity": "Already verified by preceding actual-checkpoint pilot; not repeated"}
    print(name, "native snapshot", trace_parity, native_dtypes, flush=True)
    # Preserve the JAX state on host, then release its weights before loading Torch.
    del encoder, states, meta_raw, condition_input, condition, embedded, controls
    jax.clear_caches()
    gc.collect()

    import torch
    from ml_collections import ConfigDict
    from spectra_learning.models.factory import build_model_from_config
    from spectra_learning.models.spectrum_metadata import torch_spectrum_metadata_from_batch

    torch.cuda.reset_peak_memory_stats()
    exported = torch.load(Path("/exports") / EXPORTS[name] / "encoder.pt",
                          mmap=True, weights_only=True, map_location="cpu")
    with torch.device("meta"):
        model = build_model_from_config(ConfigDict(config))
    torch_encoder = model.encoder
    del model
    torch_encoder.load_state_dict(exported, assign=True)
    del exported
    torch_encoder = torch_encoder.eval().cuda()
    torch_batch = {key: torch.from_numpy(value).cuda() for key, value in batch.items()}
    torch_metadata = torch_spectrum_metadata_from_batch(torch_batch, schema=schema)
    captures = {}
    torch_dtypes = {}

    def remember(label, value):
        torch_dtypes[label] = str(value.dtype)
        captures[label] = value.detach().float().cpu().numpy()

    hooks = [
        torch_encoder.embedder.register_forward_hook(lambda _m, _a, output: remember("peak_embedder", output)),
        torch_encoder.metadata_embedder.register_forward_pre_hook(lambda _m, args: remember("metadata_embedder_input", args[0])),
        torch_encoder.metadata_embedder.register_forward_hook(lambda _m, _a, output: remember("metadata_embedding", output)),
        torch_encoder.blocks[0].register_forward_pre_hook(lambda _m, args: remember("0", args[0])),
        torch_encoder.final_norm.register_forward_hook(lambda _m, _a, output: remember("final_norm", output)),
    ]
    for index, block in enumerate(torch_encoder.blocks, start=1):
        hooks.append(block.register_forward_hook(
            lambda _m, _a, output, index=index: remember(str(index), output)))
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = torch_encoder(torch_batch["peak_mz"], torch_batch["peak_intensity"],
                                   valid_mask=torch_batch["peak_valid_mask"],
                                   visible_mask=torch_batch["peak_valid_mask"],
                                   precursor_mz=torch_batch["precursor_mz"],
                                   spectrum_metadata=torch_metadata)
        remember("final", output)
    finally:
        for hook in hooks:
            hook.remove()
    torch_states = np.stack([captures[label] for label in record["layer_labels"]])
    record["torch_dtypes"] = torch_dtypes
    record["torch_peak_allocated_gpu_bytes"] = torch.cuda.max_memory_allocated()
    record["metadata_raw"] = error_metrics(torch_metadata.float().cpu().numpy(), native_metadata, np.ones(8, bool))
    record["metadata_embedder_input"] = error_metrics(captures["metadata_embedder_input"], native_condition_input, np.ones(8, bool))
    record["metadata_embedding"] = error_metrics(captures["metadata_embedding"], native_condition, np.ones(8, bool))
    record["peak_embedder"] = error_metrics(captures["peak_embedder"], native_embedded, batch["peak_valid_mask"])
    record["layers"] = []
    for index, label in enumerate(record["layer_labels"]):
        current = {"depth": index, "label": label,
                   "physical_peaks": error_metrics(torch_states[index, :, :-1], native_states[index, :, :-1], batch["peak_valid_mask"]),
                   "cls": error_metrics(torch_states[index, :, -1], native_states[index, :, -1], np.ones(8, bool)),
                   "all_valid": error_metrics(torch_states[index], native_states[index], valid)}
        record["layers"].append(current)
    record["interpretation"] = (
        "This is actual bf16 native JAX versus existing Torch autocast inference, not a float32 mathematical equivalence test. "
        "Torch disables autocast inside its complete peak embedder, retaining float32 residual streams; native JAX uses bf16. "
        "Identical checkpoint tensors therefore need not produce identical numerical functions. No tolerance is relaxed here."
    )
    np.savez_compressed(destination / "layer_states.npz", native=native_states, torch=torch_states,
                        native_direct_final=expected, valid=valid, rows=np.arange(4096, 4104),
                        native_metadata=native_metadata, torch_metadata=torch_metadata.float().cpu().numpy(),
                        native_metadata_embedding=native_condition,
                        torch_metadata_embedding=captures["metadata_embedding"],
                        native_metadata_embedder_input=native_condition_input,
                        torch_metadata_embedder_input=captures["metadata_embedder_input"])
    record["files"] = {"layer_states.npz": {"sha256": sha256(destination / "layer_states.npz"),
                                              "bytes": (destination / "layer_states.npz").stat().st_size}}
    record["seconds"] = time.monotonic() - started
    write_json(destination / "diagnostic.json", record)
    volume.commit()
    print(name, "backend diagnostic", record["layers"][-1], "seconds", record["seconds"], flush=True)
    return record


@app.local_entrypoint()
def main(name: str = "parent_1190k"):
    result = diagnose.remote(name)
    directory = LOCAL / "backend_parity" / name
    write_json(directory / "diagnostic.json", result)
    source = str((ROOT / "backend_parity" / name / "layer_states.npz").relative_to("/results"))
    with (directory / "layer_states.npz").open("wb") as stream:
        for chunk in volume.read_file(source):
            stream.write(chunk)
    if sha256(directory / "layer_states.npz") != result["files"]["layer_states.npz"]["sha256"]:
        raise ValueError("Downloaded backend parity cache hash mismatch")
    print(json.dumps({"path": str(directory), "final": result["layers"][-1]}, indent=2))
