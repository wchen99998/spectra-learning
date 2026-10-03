"""Native JAX chemistry study: pinned checkpoint staging, parity, layer/token atlas.

Writes only the new encoder_chemistry_20260930 directory on the existing volume.
Existing checkpoint copies and Torch exports are read without modification.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import modal

LOCAL = Path("artifacts/encoder_chemistry_20260930")
PROJECT = Path("/root/spectra-learning")
ROOT = Path("/results/encoder_chemistry_20260930")
CHECKPOINTS = {
    "parent_1190k": Path("/results/salt_mechanisms_20260919/checkpoints/1190000"),
    "periodic_1209k": ROOT / "checkpoints/1209000",
    "diffusion_831k": ROOT / "checkpoints/diffusion_831000",
}
CONFIG_FILES = {
    "parent_1190k": "parent_config.json",
    "periodic_1209k": "periodic_config.json",
    "diffusion_831k": "diffusion_extension/config.json",
}
EXPORTS = {"parent_1190k": "ftjepa_1190k", "periodic_1209k": "periodic_student_1209k"}
INPUT_NAMES = ("peak_mz", "peak_intensity", "peak_valid_mask", "precursor_mz",
               "collision_energy", "collision_energy_present", "charge", "charge_present",
               "polarity_id", "instrument_family_id")
TOKEN_DEPTHS = (0, 5, 10, 13, 19, 25, 26, 27)


def ignore_project_file(path: Path) -> bool:
    parts = path.absolute().relative_to(Path.cwd().absolute()).parts
    return parts[0] not in {"configs", "scripts", "spectra_learning", "pyproject.toml"} or "__pycache__" in parts


def snapshot_code() -> str:
    """Freeze source before Modal watches it; independent study agents may edit."""
    if not modal.is_local():
        return str(PROJECT)
    destination = Path(tempfile.mkdtemp(prefix="encoder-chemistry-source-"))
    for directory in ("configs", "scripts", "spectra_learning"):
        shutil.copytree(directory, destination / directory,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2("pyproject.toml", destination / "pyproject.toml")
    return str(destination)


image = (
    modal.Image.debian_slim(python_version="3.14")
    .apt_install("libgomp1")
    .pip_install_from_pyproject("pyproject.toml")
    .uv_pip_install("torch==2.13.0", index_url="https://download.pytorch.org/whl/cu130",
                    extra_index_url="https://pypi.org/simple")
    .uv_pip_install("jax[cuda13]==0.11.0", "flax==0.12.8", "optax==0.2.8", "orbax-checkpoint==0.12.4")
    .add_local_dir(snapshot_code(), str(PROJECT), copy=True)
    .env({"PYTHONPATH": str(PROJECT), "XLA_PYTHON_CLIENT_PREALLOCATE": "false"})
)
volume = modal.Volume.from_name("spectra-massspecgym-maccs-probe")
export_volume = modal.Volume.from_name("spectra-layerwise-semantic-probe")
app = modal.App("spectra-native-encoder-chemistry")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@app.function(image=image, cpu=8, memory=16384, timeout=3600, volumes={"/results": volume})
def stage_latest(access_token: str, name: str = "periodic_1209k") -> dict:
    from concurrent.futures import ThreadPoolExecutor
    from google.cloud import storage
    from google.oauth2.credentials import Credentials

    selection_file = {"periodic_1209k": "endpoint_selection.json",
                      "diffusion_831k": "diffusion_extension/endpoint_selection.json"}[name]
    selection = json.loads((ROOT / selection_file).read_text())
    uri = selection["uri"]
    bucket_name, prefix = uri.removeprefix("gs://").split("/", 1)
    destination = CHECKPOINTS[name]
    completion = destination / "copy_manifest.json"
    if completion.exists():
        record = json.loads(completion.read_text())
        if record["objects"] != selection["objects"]:
            raise ValueError("Existing checkpoint copy belongs to another object inventory")
        return record
    client = storage.Client(project="metal-repeater-411410", credentials=Credentials(access_token))
    bucket = client.bucket(bucket_name)

    def copy_one(item):
        relative = item["name"]
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        blob = bucket.blob(prefix + "/" + relative, generation=int(item["generation"]))
        blob.download_to_filename(target, if_generation_match=int(item["generation"]), checksum="crc32c")
        if target.stat().st_size != int(item["size"]):
            raise ValueError(f"Checkpoint object size mismatch: {relative}")
        return relative

    with ThreadPoolExecutor(max_workers=8) as executor:
        copied = list(executor.map(copy_one, selection["objects"]))
    record = {"uri": uri, "objects": selection["objects"], "copied": len(copied), "verified": True}
    write_json(completion, record)
    volume.commit()
    print(f"Verified checkpoint copy: {len(copied)} objects", flush=True)
    return record


def restore(name: str):
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from spectra_learning.models.model_jax import build_peak_set_encoder_jax
    from spectra_learning.models.settings import PeakSetJEPASettings
    from spectra_learning.training.checkpointing_jax import restore_jax_encoder_state

    config_file = CONFIG_FILES[name]
    config = json.loads((ROOT / config_file).read_text())
    settings = PeakSetJEPASettings.from_config(config)
    encoder = nnx.eval_shape(lambda: build_peak_set_encoder_jax(
        settings, compute_dtype=jnp.bfloat16, pairmixer_block_type=settings.pairmixer_block_type,
        pairmixer_fast_encoder_max_visible_tokens=None, rngs=nnx.Rngs(int(config["seed"]))))
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    template = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=sharding),
                            nnx.as_pure(nnx.state(encoder)))
    nnx.update(encoder, restore_jax_encoder_state(CHECKPOINTS[name], template))
    encoder.eval()
    return encoder, config


def batch_inputs(data, rows, size):
    import numpy as np
    result = {}
    for key in INPUT_NAMES:
        value = np.zeros((size, *data[key].shape[1:]), data[key].dtype)
        value[:len(rows)] = data[key][rows]
        result[key] = value
    return result


def metadata(batch, schema):
    from spectra_learning.models.spectrum_metadata import jax_spectrum_metadata_from_batch
    return jax_spectrum_metadata_from_batch(batch, schema=schema)


def error_metrics(actual, expected, valid):
    import numpy as np
    a = np.asarray(actual, np.float32)[valid]
    b = np.asarray(expected, np.float32)[valid]
    delta = a - b
    cosine = (a * b).sum(-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-20)
    return {"relative_l2": float(np.linalg.norm(delta) / max(float(np.linalg.norm(b)), 1e-20)),
            "max_absolute": float(np.abs(delta).max()), "rms": float(np.sqrt(np.mean(delta ** 2))),
            "median_token_cosine": float(np.median(cosine)), "minimum_token_cosine": float(cosine.min()),
            "bitwise_equal": bool(np.array_equal(a, b)), "finite": bool(np.isfinite(a).all())}


def provenance(name, config):
    import importlib.metadata
    files = ["scripts/modal_encoder_chemistry.py", "scripts/encoder_chemistry_interventions.py",
             "spectra_learning/models/model_jax.py", "spectra_learning/models/settings.py",
             "spectra_learning/models/peak_features_jax.py", "spectra_learning/models/spectrum_metadata.py",
             "spectra_learning/models/encoder_jax.py", "spectra_learning/models/pairmixer_jax.py",
             "spectra_learning/models/common_jax.py", "spectra_learning/training/checkpointing_jax.py"]
    return {"name": name, "checkpoint": str(CHECKPOINTS[name]), "config": config,
            "inputs_sha256": sha256(ROOT / "panel/inputs.npz"),
            "source_sha256": {f: sha256(PROJECT / f) for f in files},
            "versions": {p: importlib.metadata.version(p) for p in ("jax", "jaxlib", "flax", "optax", "orbax-checkpoint", "numpy", "torch")}}


@app.function(image=image, gpu="H100", cpu=8, memory=65536, timeout=7200,
              volumes={"/results": volume, "/exports": export_volume})
def pilot(name: str, compare_export: bool = True) -> dict:
    os.chdir(PROJECT)
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx
    from scripts.encoder_chemistry_interventions import identity_interventions, trace_encoder

    start = time.monotonic()
    jax.config.update("jax_default_matmul_precision", "highest")
    encoder, config = restore(name)
    data = np.load(ROOT / "panel/inputs.npz")
    rows = np.arange(4096, 4104)
    batch = batch_inputs(data, rows, 8)
    schema = config["encoder_metadata_schema"]

    @nnx.jit
    def native(enc, b):
        return enc(b["peak_mz"], b["peak_intensity"], valid_mask=b["peak_valid_mask"],
                   visible_mask=b["peak_valid_mask"], precursor_mz=b["precursor_mz"], spectrum_metadata=metadata(b, schema))

    @nnx.jit
    def instrumented(enc, b, controls):
        return trace_encoder(enc, b["peak_mz"], b["peak_intensity"], valid_mask=b["peak_valid_mask"],
                             visible_mask=b["peak_valid_mask"], spectrum_metadata=metadata(b, schema), interventions=controls)

    controls = identity_interventions(encoder, 8, 127)
    expected = np.asarray(native(encoder, batch), np.float32)
    print(name, "native compiled", flush=True)
    traced = instrumented(encoder, batch, controls)
    baseline = np.asarray(traced.final, np.float32)
    valid = np.asarray(traced.token_mask)
    record = {**provenance(name, config), "rows": rows.tolist(), "device": str(jax.devices()[0]),
              "native_vs_instrumented": error_metrics(baseline, expected, valid),
              "tolerance_relative_l2": .01,
              "tolerance_rationale": "Numerical gate only, not evidence of causal effect. All effects must also exceed measured sham-patch noise.",
              "identity_patches": []}
    for depth in range(len(encoder.blocks) + 1):
        patch = controls._replace(patch_layer=jnp.asarray(depth, jnp.int32), patch_values=traced.raw[depth],
                                  patch_mask=traced.token_mask)
        actual = instrumented(encoder, batch, patch).final
        record["identity_patches"].append({"depth": depth, **error_metrics(actual, baseline, valid)})
    record["max_sham_relative_l2"] = max(x["relative_l2"] for x in record["identity_patches"])
    record["native_parity_passed"] = (record["native_vs_instrumented"]["relative_l2"] < .01
                         and record["native_vs_instrumented"]["finite"]
                         and record["max_sham_relative_l2"] < .01)
    record["passed"] = False
    record["complete"] = False
    output = ROOT / name
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "pilot_tokens.npz", native=expected, instrumented=baseline, rows=rows, valid=valid)
    write_json(output / "pilot.json", record)
    volume.commit()
    print(name, "native trace parity", record["native_vs_instrumented"], "sham", record["max_sham_relative_l2"], flush=True)
    if not record["native_parity_passed"]:
        raise ValueError("Instrumentation failed the actual-checkpoint numerical parity gate")
    if compare_export:
        import torch
        from spectra_learning.training.checkpointing_jax import torch_encoder_state_dict_from_jax
        exported = torch.load(Path("/exports") / EXPORTS[name] / "encoder.pt", mmap=True, weights_only=True)
        converted = torch_encoder_state_dict_from_jax(nnx.as_pure(nnx.state(encoder)))
        differences = []
        if exported.keys() != converted.keys():
            raise ValueError("Native/staged export tensor names differ")
        for key in exported:
            if not torch.equal(exported[key], converted[key]):
                differences.append({"key": key, "max_abs": float((exported[key] - converted[key]).abs().max())})
        record["export_weights"] = {"count": len(exported), "exactly_equal": not differences,
                                     "differences": differences}
        del converted, encoder, traced
        jax.clear_caches()
        import gc
        gc.collect()
        from ml_collections import ConfigDict
        from spectra_learning.models.factory import build_model_from_config
        from spectra_learning.models.spectrum_metadata import torch_spectrum_metadata_from_batch
        with torch.device("meta"):
            model = build_model_from_config(ConfigDict(config))
        torch_encoder = model.encoder
        del model
        torch_encoder.load_state_dict(exported, assign=True)
        del exported
        torch_encoder = torch_encoder.eval().cuda()
        torch_batch = {k: torch.from_numpy(v).cuda() for k, v in batch.items()}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            tokens = torch_encoder(torch_batch["peak_mz"], torch_batch["peak_intensity"],
                valid_mask=torch_batch["peak_valid_mask"], visible_mask=torch_batch["peak_valid_mask"],
                precursor_mz=torch_batch["precursor_mz"],
                spectrum_metadata=torch_spectrum_metadata_from_batch(torch_batch, schema=schema))
        record["torch_vs_native"] = error_metrics(tokens.float().cpu().numpy(), expected, valid)
        record["torch_output_equivalent_at_1pct_relative_l2"] = record["torch_vs_native"]["relative_l2"] < .01
    record["passed"] = record["native_parity_passed"] and (
        not compare_export or record["export_weights"]["exactly_equal"])
    record["complete"] = True
    record["seconds"] = time.monotonic() - start
    write_json(output / "pilot.json", record)
    volume.commit()
    return record


@app.function(image=image, gpu="H100", cpu=8, memory=65536, timeout=10800,
              volumes={"/results": volume})
def extract(name: str, batch_size: int = 8, token_depths: tuple[int, ...] = TOKEN_DEPTHS) -> dict:
    os.chdir(PROJECT)
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx
    from scripts.encoder_chemistry_interventions import identity_interventions, trace_encoder

    if 27 not in token_depths or len(set(token_depths)) != len(token_depths) or any(
        depth not in range(28) for depth in token_depths
    ):
        raise ValueError("Token depths must be unique states in [0, 27], including final state 27")
    output = ROOT / name
    pilot_record = json.loads((output / "pilot.json").read_text())
    # Older completed pilots precede the explicit complete field; their duration
    # and finished Torch-output record distinguish them from the partial write.
    completed = pilot_record.get("complete", "seconds" in pilot_record and "torch_vs_native" in pilot_record)
    if not completed or not pilot_record["passed"]:
        raise ValueError("A passing actual-checkpoint parity pilot is required")
    jax.config.update("jax_default_matmul_precision", "highest")
    encoder, config = restore(name)
    current = provenance(name, config)
    if current["config"] != pilot_record["config"] or current["inputs_sha256"] != pilot_record["inputs_sha256"]:
        raise ValueError("Pilot and extraction config/input differ")
    for path, digest in pilot_record["source_sha256"].items():
        if path != "scripts/modal_encoder_chemistry.py" and current["source_sha256"][path] != digest:
            raise ValueError(f"Pilot and extraction implementation differ: {path}")
    data = np.load(ROOT / "panel/inputs.npz")
    count = len(data["peak_mz"])
    dim = config["model_dim"]
    schema = config["encoder_metadata_schema"]
    if batch_size != 8:
        raise ValueError("The verified native graph uses batch size 8")
    controls = identity_interventions(encoder, batch_size, 127)

    @nnx.jit
    def encode(enc, b, intervention):
        return trace_encoder(enc, b["peak_mz"], b["peak_intensity"], valid_mask=b["peak_valid_mask"],
                             visible_mask=b["peak_valid_mask"], spectrum_metadata=metadata(b, schema), interventions=intervention)

    reference = np.load(output / "pilot_tokens.npz")
    pilot_actual = encode(encoder, batch_inputs(data, reference["rows"], batch_size), controls).final
    preflight = error_metrics(pilot_actual, reference["native"], reference["valid"])
    if not preflight["bitwise_equal"]:
        write_json(output / "exact_graph_preflight_failure.json", preflight)
        volume.commit()
        raise ValueError("Full-trace batch-8 graph is not bitwise equal to the native pilot")

    @jax.jit
    def summarize(t, b):
        # This is a SEPARATE compiled call after the full trace has returned.
        # It cannot fuse pooling into or change the encoder's verified graph.
        states = jnp.concatenate([t.raw, t.final_normalized[None], t.final[None]], axis=0)
        peaks = states[:, :, :-1].astype(jnp.float32)
        mask = b["peak_valid_mask"][None, :, :, None]
        mean = (peaks * mask).sum(2) / jnp.maximum(mask.sum(2), 1)
        maximum = jnp.max(jnp.where(mask, peaks, -jnp.inf), axis=2)
        # Lossless storage of bf16 bit patterns; conversion is explicit downstream.
        tokens = jax.lax.bitcast_convert_type(states[jnp.array(token_depths)], jnp.uint16)
        rms = lambda x: jnp.sqrt(((x.astype(jnp.float32)[:, :, :-1] ** 2) * mask).sum((2, 3))
                                / jnp.maximum(mask.sum((2, 3)) * dim, 1))
        return mean, maximum, states[:, :, -1].astype(jnp.float32), tokens, rms(t.attention_update), rms(t.mlp_update)

    shapes = {"mean": (28, count, dim), "max": (28, count, dim), "cls": (28, count, dim),
              "tokens_bf16": (len(token_depths), count, 128, dim),
              "attention_rms": (25, count), "mlp_rms": (25, count)}
    arrays = {key: np.lib.format.open_memmap(output / f"{key}.npy", mode="w+", shape=shape,
                dtype=np.uint16 if key == "tokens_bf16" else np.float32) for key, shape in shapes.items()}
    started = time.monotonic()
    for first in range(0, count, batch_size):
        last = min(first + batch_size, count)
        batch = batch_inputs(data, np.arange(first, last), batch_size)
        values = summarize(encode(encoder, batch, controls), batch)
        for (key, dest), value in zip(arrays.items(), values, strict=True):
            dest[:, first:last] = np.asarray(value)[:, :last-first]
        if first % (32 * batch_size) == 0 or last == count:
            print(name, "extracted", last, count, "seconds", round(time.monotonic() - started, 1), flush=True)
    for dest in arrays.values():
        dest.flush()
    import ml_dtypes
    stored = arrays["tokens_bf16"][token_depths.index(27), reference["rows"]].view(ml_dtypes.bfloat16).astype(np.float32)
    extraction_parity = error_metrics(stored, reference["native"], reference["valid"])
    if not extraction_parity["bitwise_equal"]:
        write_json(output / "extraction_parity_failure.json", extraction_parity)
        volume.commit()
        raise ValueError("Extracted token cache exceeds the native pilot numerical gate")
    record = {**provenance(name, config), "complete": True, "rows": count,
              "token_depths": list(token_depths), "state_depths": [*range(26), "final_norm", "final"],
              "bf16_storage": "uint16 bit patterns, lossless; view with ml_dtypes.bfloat16",
              "shapes": {k: list(v) for k, v in shapes.items()}, "seconds": time.monotonic() - started,
              "extraction_vs_native_pilot": extraction_parity,
              "trace_mode": "full dynamic trace graph, batch8, separately compiled summaries; exact pilot parity",
              "files": {k: {"bytes": (output / f"{k}.npy").stat().st_size,
                              "sha256": sha256(output / f"{k}.npy")} for k in shapes}}
    write_json(output / "extraction.json", record)
    volume.commit()
    return record


@app.function(image=image, cpu=2, memory=4096, timeout=600, volumes={"/results": volume})
def archive_initial_extraction(name: str) -> dict:
    """Preserve the numerical pilot extraction and its dependent analyses."""
    directory = ROOT / name
    archive = directory / "initial_b16"
    archive.mkdir(exist_ok=False)
    names = ["mean.npy", "max.npy", "cls.npy", "tokens_bf16.npy", "attention_rms.npy", "mlp_rms.npy",
             "extraction.json", "cache_parity.json", "extraction_parity_failure.json", "atlas", "readouts"]
    moved = []
    for filename in names:
        path = directory / filename
        if path.exists():
            path.rename(archive / filename)
            moved.append(filename)
    record = {"name": name, "moved": moved, "reason": "Batch16/reduced graph produced numerical differences from the exact native batch8 pilot; preserved as provisional diagnostics."}
    write_json(archive / "archive_manifest.json", record)
    volume.commit()
    return record


@app.local_entrypoint()
def main(phase: str = "pilot", name: str = "both"):
    names = ["parent_1190k", "periodic_1209k"] if name == "both" else [name]
    if phase == "pilot":
        with volume.batch_upload(force=True) as upload:
            for filename in ("periodic_config.json", "parent_config.json", "endpoint_selection.json"):
                upload.put_file(LOCAL / filename, str((ROOT / filename).relative_to("/results")))
            for filename in ("inputs.npz", "labels.npz", "matched_controls.npz", "weak_losses.npz", "repeat_inputs.npz", "manifest.json"):
                upload.put_file(LOCAL / "panel" / filename, str((ROOT / "panel" / filename).relative_to("/results")))
        calls = {}
        if "parent_1190k" in names:
            calls["parent_1190k"] = pilot.spawn("parent_1190k")
        if "periodic_1209k" in names:
            import google.auth
            from google.auth.transport.requests import Request
            credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/devstorage.read_only"])
            credentials.refresh(Request())
            staged = stage_latest.remote(credentials.token)
            print("Pinned copy ready:", staged["copied"], "objects")
            calls["periodic_1209k"] = pilot.spawn("periodic_1209k")
        for key, call in calls.items():
            result = call.get()
            write_json(LOCAL / key / "pilot.json", result)
            print(key, "pilot", result["passed"])
    elif phase == "extract":
        calls = {key: extract.spawn(key) for key in names}
        for key, call in calls.items():
            result = call.get()
            write_json(LOCAL / key / "extraction.json", result)
            print(key, "extraction complete", result["seconds"])
    elif phase == "archive":
        for key in names:
            print(archive_initial_extraction.remote(key))
    else:
        raise ValueError(f"Unknown phase: {phase}")
