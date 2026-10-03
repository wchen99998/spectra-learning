"""Export full MSG1.5 pair tensors in the verified single-stream row order."""
from __future__ import annotations

import io
import json
from pathlib import Path
import time

import modal

from scripts.modal_export_jepa_fps_embeddings import (
    GCP_PROJECT, INPUT_TSV, _configure_google_credentials, _gcs_blob,
    _prepare_massspecgym_inputs, _sha256, _write_npy,
)

RUN_ID = "pair279m-attn-n32-b2048-v6e32-2d-20260910-rootcause"
STEP = 730000
ROOT = "gs://metal-repeater-411410-spectra-checkpoints"
CHECKPOINT = f"{ROOT}/skypilot/{RUN_ID}/checkpoints/orbax/{STEP}"
REFERENCE = f"{ROOT}/jepa-harness-files/msg15_jepa_embeddings_1p8b_step666000"
OUTPUT = f"{ROOT}/jepa-harness-files/msg15_pair_embeddings_pair279m_attn_step{STEP}"
KEYS = ("peak_mz", "peak_intensity", "peak_valid_mask", "precursor_mz",
        "collision_energy", "collision_energy_present", "charge", "charge_present",
        "polarity_id", "instrument_family_id")
app = modal.App("spectra-msg15-pair279m-export")
if modal.is_local():
    from google.auth import _cloud_sdk
    gcp_secret = modal.Secret.from_dict({
        "GOOGLE_APPLICATION_CREDENTIALS_JSON": Path(
            _cloud_sdk.get_application_default_credentials_path()).read_text()
    })
else:
    gcp_secret = modal.Secret.from_dict({})
image = (
    modal.Image.debian_slim(python_version="3.14")
    .apt_install("libgomp1")
    .pip_install_from_pyproject("pyproject.toml")
    .uv_pip_install("torch==2.13.0", index_url="https://download.pytorch.org/whl/cpu",
                    extra_index_url="https://pypi.org/simple")
    .uv_pip_install("jax[cuda13]==0.11.0", "flax==0.12.8", "orbax-checkpoint==0.12.4")
    .add_local_dir("spectra_learning", "/root/spectra-learning/spectra_learning", copy=True)
    .add_local_dir("scripts", "/root/spectra-learning/scripts", copy=True)
    .add_local_file("pyproject.toml", "/root/spectra-learning/pyproject.toml", copy=True)
    .env({"PYTHONPATH": "/root/spectra-learning", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"})
)


@app.function(image=image, gpu="H100", cpu=8, memory=32768, timeout=21600,
              max_containers=8, secrets=[gcp_secret])
def export_shard(start: int, stop: int, batch_size: int = 32,
                 smoke: bool = False) -> dict:
    _configure_google_credentials()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx
    from google.cloud import storage
    from ml_collections import config_dict
    from spectra_learning.models.encoder_jax import PairOnlyEncoder
    from spectra_learning.models.settings import PeakSetJEPASettings
    from spectra_learning.models.spectrum_metadata import jax_spectrum_metadata_from_batch
    from spectra_learning.training.checkpointing_jax import restore_jax_encoder_state

    started = time.monotonic()
    client = storage.Client(project=GCP_PROJECT)
    config = config_dict.ConfigDict(json.loads(_gcs_blob(client,
        f"{CHECKPOINT}/metadata/metadata").download_as_text())["task_contract"]["config"])
    config.activation_checkpoint_mode = "none"
    encoder = PairOnlyEncoder(PeakSetJEPASettings.from_config(config),
                             compute_dtype=jnp.bfloat16, rngs=nnx.Rngs(config.seed))
    template = nnx.as_pure(nnx.state(encoder))
    nnx.update(encoder, restore_jax_encoder_state(CHECKPOINT, template))
    del template
    print(f"Restored checkpoint {STEP} for rows [{start}, {stop})", flush=True)
    local = Path("/tmp/msg15-pair-inputs.npz")
    _gcs_blob(client, f"{OUTPUT}/prepared_inputs.npz").download_to_filename(local)
    with np.load(local) as archive:
        data = {key: archive[key][start:stop] for key in KEYS}
    n = stop - start

    @nnx.jit
    def encode(model, batch):
        pair, indices, valid = model(
            batch["peak_mz"], batch["peak_intensity"],
            valid_mask=batch["peak_valid_mask"], visible_mask=batch["peak_valid_mask"],
            max_visible_tokens=32, precursor_mz=batch["precursor_mz"],
            spectrum_metadata=jax_spectrum_metadata_from_batch(
                batch, schema=config.encoder_metadata_schema))
        pair = pair.astype(jnp.float16)
        return pair, indices, valid

    def get_batch(offset):
        take = min(batch_size, n - offset)
        batch = {key: jnp.asarray(np.pad(value[offset:offset + take],
                 ((0, batch_size - take),) + ((0, 0),) * (value.ndim - 1)))
                 for key, value in data.items()}
        pair, idx, valid = jax.device_get(encode(encoder, batch))
        pair, idx, valid = pair[:take], idx[:take], valid[:take]
        if pair.shape != (take, 32, 32, 640) or not np.isfinite(pair).all():
            raise ValueError("Invalid pair tensor shape or non-finite values")
        if not np.array_equal(idx, np.broadcast_to(np.arange(32), idx.shape)):
            raise ValueError("Encoder changed token order")
        expected_mask = np.concatenate([data["peak_valid_mask"][offset:offset + take],
                                       np.ones((take, 1), dtype=bool)], axis=1)
        if not np.array_equal(valid, expected_mask):
            raise ValueError("Encoder dropped or reordered peaks")
        return pair

    first = get_batch(0)
    print(f"Compiled and verified batch in {time.monotonic() - started:.1f}s", flush=True)
    if smoke:
        repeated = get_batch(0)
        np.testing.assert_array_equal(first, repeated)
        # Tail batching must preserve a spectrum's result.
        single = {key: jnp.asarray(np.pad(value[:1],
                  ((0, batch_size - 1),) + ((0, 0),) * (value.ndim - 1)))
                  for key, value in data.items()}
        alone = np.asarray(jax.device_get(encode(encoder, single)[0]))[:1]
        np.testing.assert_array_equal(first[:1], alone)
        return {"smoke_passed": True, "shape": list(first.shape),
                "max_abs_batch_difference": float(np.abs(first[:1] - alone).max()),
                "seconds": time.monotonic() - started}

    uri = f"{OUTPUT}/_parts/{start:06d}-{stop:06d}.bin"
    blob = _gcs_blob(client, uri)
    import hashlib
    digest = hashlib.sha256()
    with blob.open("wb", chunk_size=64 * 1024 * 1024, timeout=300,
                   if_generation_match=0) as stream:
        for offset in range(0, n, batch_size):
            pair = first if offset == 0 else get_batch(offset)
            payload = pair.tobytes(order="C")
            digest.update(payload)
            stream.write(payload)
            if offset % (batch_size * 100) == 0:
                print(f"rows [{start}, {stop}): {min(offset + batch_size, n)}/{n}; "
                      f"{time.monotonic() - started:.0f}s", flush=True)
    blob.reload()
    if blob.size != n * 32 * 32 * 640 * 2:
        raise ValueError("Uploaded shard size mismatch")
    report = {"start": start, "stop": stop, "uri": uri,
              "generation": int(blob.generation), "size": blob.size,
              "crc32c": blob.crc32c, "sha256": digest.hexdigest(),
              "seconds": time.monotonic() - started,
              "jax_version": jax.__version__, "device": str(jax.devices()[0])}
    _gcs_blob(client, uri + ".json").upload_from_string(json.dumps(report),
                                                       if_generation_match=0)
    return report


def prepare() -> dict:
    import numpy as np
    import pyarrow.csv as csv
    from google.cloud import storage
    from ml_collections import config_dict

    client = storage.Client(project=GCP_PROJECT)
    reference_blob = _gcs_blob(client, f"{REFERENCE}/embedding_metadata.json")
    reference_blob.reload()
    reference = json.loads(reference_blob.download_as_text())
    if reference["status"] != "complete" or reference["input_tsv"] != INPUT_TSV:
        raise ValueError("Reference is not a complete export of this TSV")
    config_raw = json.loads(_gcs_blob(client,
        f"{CHECKPOINT}/metadata/metadata").download_as_text())
    if not _gcs_blob(client, f"{CHECKPOINT}/commit_success.txt").exists():
        raise ValueError("Checkpoint is not committed")
    config = config_dict.ConfigDict(config_raw["task_contract"]["config"])
    if config.num_peaks != 31 or config.model_dim != 640:
        raise ValueError("Unexpected checkpoint configuration")
    local = Path("artifacts/msg15_pair279m_step730000")
    local.mkdir(parents=True, exist_ok=True)
    tsv = local / "MassSpecGym1.5.tsv"
    source = _gcs_blob(client, INPUT_TSV)
    source.reload()
    source.download_to_filename(tsv, if_generation_match=int(source.generation))
    if _sha256(tsv) != reference["input_tsv_sha256"]:
        raise ValueError("TSV differs from the single-stream export input")
    data = _prepare_massspecgym_inputs(tsv, config)
    count = len(data["peak_mz"])
    if count != reference["samples"]:
        raise ValueError("Row count differs from reference")
    for fold in ("train", "val", "test"):
        if int(np.count_nonzero(data["fold"] == fold)) != reference[f"{fold}_samples"]:
            raise ValueError("Fold counts differ from reference")
    np.savez(local / "prepared_inputs.npz", **data)
    _gcs_blob(client, f"{OUTPUT}/prepared_inputs.npz").upload_from_filename(
        local / "prepared_inputs.npz", if_generation_match=0)
    mask = np.concatenate([data["peak_valid_mask"], np.ones((count, 1), bool)], axis=1)
    for name, values in {"token_valid_mask": mask, "source_rows": np.arange(count),
                         "peak_mz": data["peak_mz"],
                         "peak_intensity": data["peak_intensity"]}.items():
        _write_npy(client, f"{OUTPUT}/{name}.npy", values)
    table = csv.read_csv(tsv, parse_options=csv.ParseOptions(delimiter="\t"),
                        convert_options=csv.ConvertOptions(include_columns=["identifier", "fold"]))
    csv.write_csv(table, local / "rows.tsv", write_options=csv.WriteOptions(delimiter="\t"))
    _gcs_blob(client, f"{OUTPUT}/rows.tsv").upload_from_filename(local / "rows.tsv",
                                                               if_generation_match=0)
    metadata = {"status": "prepared", "run_id": RUN_ID,
                "run_url": f"https://wandb.ai/iclac/jepa-finalrun/runs/{RUN_ID}",
                "checkpoint": CHECKPOINT, "checkpoint_step": STEP,
                "encoder": "student", "encoder_runtime": "JAX", "compute_dtype": "bfloat16",
                "input_tsv": INPUT_TSV, "input_tsv_generation": int(source.generation),
                "input_tsv_sha256": reference["input_tsv_sha256"],
                "reference_embeddings": REFERENCE,
                "reference_metadata_generation": int(reference_blob.generation),
                "row_order": "Original TSV order, identical to reference embeddings; no filtering",
                "samples": count, "embedding_shape": [count, 32, 32, 640],
                "embedding_dtype": "float16", "embedding_uri": f"{OUTPUT}/pairs_f16.npy",
                "peak_ordering": "mz", "peak_slots": 31, "cls_token_index": 31,
                "token_layout": "31 peak slots followed by CLS on both pair axes",
                "pair_valid_mask": "token_valid_mask[:, :, None] & token_valid_mask[:, None, :]",
                "valid_mask_uri": f"{OUTPUT}/token_valid_mask.npy",
                "peak_mz_units": "m/z divided by 1000",
                "inference_max_visible_tokens": 32,
                **{f"{fold}_samples": reference[f"{fold}_samples"] for fold in ("train", "val", "test")}}
    (local / "preparation.json").write_text(json.dumps(metadata, indent=2))
    _gcs_blob(client, f"{OUTPUT}/checkpoint_metadata.json").upload_from_string(
        json.dumps(config_raw, indent=2), if_generation_match=0)
    _gcs_blob(client, f"{OUTPUT}/export_status.json").upload_from_string(json.dumps(metadata, indent=2))
    return metadata


def finalize(reports: list[dict], metadata: dict) -> dict:
    import numpy as np
    from google.cloud import storage

    client = storage.Client(project=GCP_PROJECT)
    reports = sorted(reports, key=lambda report: report["start"])
    cursor = 0
    for report in reports:
        if report["start"] != cursor or report["stop"] <= cursor:
            raise ValueError("Shards have a gap or overlap")
        cursor = report["stop"]
    if cursor != metadata["samples"]:
        raise ValueError("Shards do not cover all reference rows")
    header = io.BytesIO()
    np.lib.format.write_array_header_2_0(header, {
        "descr": "<f2", "fortran_order": False, "shape": tuple(metadata["embedding_shape"])})
    header_blob = _gcs_blob(client, f"{OUTPUT}/_parts/header.bin")
    header_blob.upload_from_string(header.getvalue(), if_generation_match=0)
    header_blob.reload()
    sources = [header_blob]
    for report in reports:
        blob = _gcs_blob(client, report["uri"])
        blob.reload(if_generation_match=report["generation"])
        if blob.size != report["size"] or blob.crc32c != report["crc32c"]:
            raise ValueError("Shard changed before composition")
        sources.append(blob)
    result = _gcs_blob(client, metadata["embedding_uri"])
    result.compose(sources, if_generation_match=0,
                   if_source_generation_match=[int(blob.generation) for blob in sources], timeout=300)
    result.reload()
    expected_bytes = len(header.getvalue()) + sum(report["size"] for report in reports)
    if result.size != expected_bytes:
        raise ValueError("Final NPY size mismatch")
    actual_header = result.download_as_bytes(start=0, end=len(header.getvalue()) - 1)
    if actual_header != header.getvalue():
        raise ValueError("Final NPY header mismatch")
    # Verify both ends of every row interval after server-side composition.
    offset = len(header.getvalue())
    for blob in sources[1:]:
        for relative in (0, blob.size - 4096):
            expected = blob.download_as_bytes(start=relative, end=relative + 4095)
            actual = result.download_as_bytes(start=offset + relative, end=offset + relative + 4095)
            if actual != expected:
                raise ValueError("Composed shard boundary mismatch")
        offset += blob.size
    metadata.update(status="complete", embedding_size=int(result.size),
                    embedding_crc32c=result.crc32c, embedding_generation=int(result.generation),
                    shards=reports, validation="Input SHA-256, row/fold counts, every batch finite, "
                    "token indices/masks, contiguous shard ranges, NPY header/size and all shard boundaries")
    _gcs_blob(client, f"{OUTPUT}/embedding_metadata.json").upload_from_string(
        json.dumps(metadata, indent=2), if_generation_match=0)
    _gcs_blob(client, f"{OUTPUT}/export_status.json").upload_from_string(json.dumps(metadata, indent=2))
    for blob in sources:
        blob.delete(if_generation_match=int(blob.generation))
    return metadata


@app.local_entrypoint()
def main(smoke: bool = False, prepare_only: bool = False, batch_size: int = 32):
    local = Path("artifacts/msg15_pair279m_step730000")
    if prepare_only:
        print(json.dumps(prepare(), indent=2))
        return
    metadata = json.loads((local / "preparation.json").read_text())
    if smoke:
        print(json.dumps(export_shard.remote(0, 128, batch_size, True), indent=2))
        return
    count = metadata["samples"]
    boundaries = [count * index // 8 for index in range(9)]
    reports = list(export_shard.starmap(
        [(start, stop, batch_size) for start, stop in zip(boundaries, boundaries[1:])]))
    (local / "shards.json").write_text(json.dumps(reports, indent=2))
    result = finalize(reports, metadata)
    (local / "embedding_metadata.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
