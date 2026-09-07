from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import modal


RUN_ID = "1p8b-sm-hd256-adam-v6e64-b6144-ga1-666k-3d-mr6-20260828-024511"
CHECKPOINT_STEP = 666_000
CHECKPOINT = (
    "gs://metal-repeater-411410-spectra-checkpoints/skypilot/"
    f"{RUN_ID}/checkpoints/orbax/{CHECKPOINT_STEP}"
)
INPUT_TSV = (
    "gs://metal-repeater-411410-spectra-checkpoints/"
    "jepa-harness-files/msg15/MassSpecGym1.5.tsv"
)
OUTPUT_PREFIX = (
    "gs://metal-repeater-411410-spectra-checkpoints/jepa-harness-files/"
    "msg15_jepa_embeddings_1p8b_step666000"
)
PROJECT_ROOT = Path("/root/spectra-learning")
GCP_PROJECT = "metal-repeater-411410"


def _ignore_project_file(path: Path) -> bool:
    parts = path.absolute().relative_to(Path.cwd().absolute()).parts
    return parts[0] in {".git", ".venv", ".pytest_cache", "artifacts", "data"} or (
        "__pycache__" in parts
    )


image = (
    modal.Image.debian_slim(python_version="3.14")
    .apt_install("libgomp1")
    .pip_install_from_pyproject("pyproject.toml")
    .uv_pip_install(
        "torch==2.13.0",
        index_url="https://download.pytorch.org/whl/cu130",
        extra_index_url="https://pypi.org/simple",
    )
    .uv_pip_install(
        "jax[cuda13]>=0.11.0",
        "flax>=0.12.8",
        "orbax-checkpoint>=0.12.4",
    )
    .add_local_dir(
        ".",
        str(PROJECT_ROOT),
        copy=True,
        ignore=_ignore_project_file,
    )
    .env(
        {
            "PYTHONPATH": str(PROJECT_ROOT),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
        }
    )
)

app = modal.App("spectra-jepa-fps-embedding-export")
gcp_secret = modal.Secret.from_name("spectra-probe-gcp-adc")


def _configure_google_credentials() -> None:
    if os.environ.get("GCP_ACCESS_TOKEN") is not None:
        return
    path = Path("/tmp/gcp-adc.json")
    credentials_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if credentials_json is not None:
        path.write_text(credentials_json)
    else:
        keys = (
            "client_id",
            "client_email",
            "client_secret",
            "private_key",
            "private_key_id",
            "refresh_token",
            "token_uri",
            "type",
            "quota_project_id",
            "universe_domain",
            "account",
        )
        path.write_text(
            json.dumps({key: os.environ[key] for key in keys if key in os.environ})
        )
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(path)


def _gcs_blob(client: Any, uri: str) -> Any:
    bucket_name, object_name = uri.removeprefix("gs://").split("/", 1)
    return client.bucket(bucket_name).blob(object_name)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_massspecgym_inputs(tsv_path: Path, config: Any) -> dict[str, Any]:
    import numpy as np
    import pyarrow.csv as csv

    from spectra_learning.data.spectra import (
        COLLISION_ENERGY_MAX,
        preprocess_peak_batch_numpy,
    )
    from spectra_learning.models.spectrum_metadata import (
        instrument_family_id,
        polarity_id,
    )

    columns = [
        "mzs",
        "intensities",
        "precursor_mz",
        "collision_energy",
        "adduct",
        "instrument_type",
        "fold",
    ]
    table = csv.read_csv(
        tsv_path,
        parse_options=csv.ParseOptions(delimiter="\t"),
        convert_options=csv.ConvertOptions(include_columns=columns),
    )
    rows = table.to_pydict()
    count = table.num_rows
    num_peaks = int(config.num_peaks)
    peak_mz = np.empty((count, num_peaks), dtype=np.float32)
    peak_intensity = np.empty_like(peak_mz)
    peak_valid_mask = np.empty_like(peak_mz, dtype=bool)
    precursor_mz = np.asarray(rows["precursor_mz"], dtype=np.float32)

    chunk_size = 4096
    for start in range(0, count, chunk_size):
        end = min(start + chunk_size, count)
        mz_lists = [
            np.fromstring(value, sep=",", dtype=np.float32)
            for value in rows["mzs"][start:end]
        ]
        intensity_lists = [
            np.fromstring(value, sep=",", dtype=np.float32)
            for value in rows["intensities"][start:end]
        ]
        width = max(len(value) for value in mz_lists)
        spectra = np.zeros((end - start, 2, width), dtype=np.float32)
        for row_idx, (mz, intensity) in enumerate(
            zip(mz_lists, intensity_lists, strict=True)
        ):
            spectra[row_idx, 0, : len(mz)] = mz
            spectra[row_idx, 1, : len(intensity)] = intensity
        batch = preprocess_peak_batch_numpy(
            spectra,
            precursor_mz[start:end],
            num_peaks=num_peaks,
            peak_drop_min_intensity=float(config.peak_drop_min_intensity),
            peak_ordering=str(config.peak_ordering),
            max_precursor_mz=float(config.max_precursor_mz),
            precursor_peak_exclusion_window_da=float(
                config.precursor_peak_exclusion_window_da
            ),
            min_peak_intensity=float(config.min_peak_intensity),
        )
        peak_mz[start:end] = batch["peak_mz"]
        peak_intensity[start:end] = batch["peak_intensity"]
        peak_valid_mask[start:end] = batch["peak_valid_mask"]
        precursor_mz[start:end] = batch["precursor_mz"]

    collision_energy_raw = np.asarray(
        [0.0 if value is None else value for value in rows["collision_energy"]],
        dtype=np.float32,
    )
    acquisition_metadata = {
        "collision_energy": (
            np.clip(collision_energy_raw, 0.0, COLLISION_ENERGY_MAX)
            / COLLISION_ENERGY_MAX
        ).astype(np.float32),
        "collision_energy_present": np.asarray(
            [value is not None and value > 0.0 for value in rows["collision_energy"]],
            dtype=np.float32,
        ),
        "charge": np.ones(count, dtype=np.float32),
        "charge_present": np.ones(count, dtype=np.float32),
        "polarity_id": np.asarray(
            [polarity_id(value) for value in rows["adduct"]],
            dtype=np.int64,
        ),
        "instrument_family_id": np.asarray(
            [instrument_family_id(value) for value in rows["instrument_type"]],
            dtype=np.int64,
        ),
    }
    fold = np.asarray(rows["fold"], dtype=str)
    return {
        "peak_mz": peak_mz,
        "peak_intensity": peak_intensity,
        "peak_valid_mask": peak_valid_mask,
        "precursor_mz": precursor_mz,
        **acquisition_metadata,
        "fold": fold,
    }


def _write_npy(client: Any, uri: str, array: Any) -> Any:
    import numpy as np

    blob = _gcs_blob(client, uri)
    header = {
        "descr": np.lib.format.dtype_to_descr(array.dtype),
        "fortran_order": False,
        "shape": array.shape,
    }
    with blob.open("wb", chunk_size=64 * 1024 * 1024, timeout=300) as handle:
        np.lib.format.write_array_header_2_0(handle, header)
        handle.write(array.tobytes(order="C"))
    blob.reload()
    return blob


@app.function(
    image=image,
    gpu="H100",
    cpu=16,
    memory=65536,
    timeout=6 * 60 * 60,
    secrets=[gcp_secret],
)
def export_embeddings(
    batch_size: int = 128,
    checkpoint: str = CHECKPOINT,
    input_tsv: str = INPUT_TSV,
    output_prefix: str = OUTPUT_PREFIX,
) -> dict[str, Any]:
    _configure_google_credentials()

    import jax
    import numpy as np
    import torch
    from flax import nnx
    from google.cloud import storage
    from ml_collections import config_dict

    from spectra_learning.models.factory import build_model_from_config as build_torch_model
    from spectra_learning.models.factory_jax import build_model_from_config as build_jax_model
    from spectra_learning.models.spectrum_metadata import (
        torch_spectrum_metadata_from_batch,
    )
    from spectra_learning.training.checkpointing_jax import (
        restore_jax_encoder_state,
        torch_encoder_state_dict_from_jax,
    )

    started = time.time()
    client = storage.Client(project=GCP_PROJECT)
    metadata_blob = _gcs_blob(client, f"{checkpoint}/metadata/metadata")
    checkpoint_metadata = json.loads(metadata_blob.download_as_text())
    config = config_dict.ConfigDict(checkpoint_metadata["task_contract"]["config"])

    tsv_path = Path("/tmp/MassSpecGym1.5.tsv")
    _gcs_blob(client, input_tsv).download_to_filename(tsv_path)
    data = _prepare_massspecgym_inputs(tsv_path, config)
    preprocessing_seconds = time.time() - started

    jax_model = build_jax_model(config)
    encoder_template = nnx.as_pure(nnx.state(jax_model.encoder))
    del jax_model
    restored_encoder = restore_jax_encoder_state(checkpoint, encoder_template)
    torch_state = torch_encoder_state_dict_from_jax(restored_encoder)
    del restored_encoder, encoder_template
    jax.clear_caches()
    gc.collect()

    with torch.device("meta"):
        torch_model = build_torch_model(config)
    encoder = torch_model.encoder
    del torch_model
    encoder.load_state_dict(torch_state, assign=True)
    del torch_state
    encoder = encoder.eval().to("cuda")

    count = len(data["peak_mz"])
    num_peak_tokens = int(config.num_peaks)
    num_tokens = num_peak_tokens + 1
    embedding_dim = int(config.model_dim)
    embedding_shape = (count, num_tokens, embedding_dim)
    tokens_uri = f"{output_prefix}/tokens_f16.npy"
    tokens_blob = _gcs_blob(client, tokens_uri)
    header = {
        "descr": np.lib.format.dtype_to_descr(np.dtype(np.float16)),
        "fortran_order": False,
        "shape": embedding_shape,
    }

    extraction_started = time.time()
    with tokens_blob.open(
        "wb",
        chunk_size=64 * 1024 * 1024,
        timeout=300,
    ) as handle:
        np.lib.format.write_array_header_2_0(handle, header)
        with torch.inference_mode():
            for start in range(0, count, batch_size):
                end = min(start + batch_size, count)
                batch = {
                    name: torch.from_numpy(data[name][start:end]).to("cuda")
                    for name in (
                        "precursor_mz",
                        "collision_energy",
                        "collision_energy_present",
                        "charge",
                        "charge_present",
                        "polarity_id",
                        "instrument_family_id",
                    )
                }
                spectrum_metadata = torch_spectrum_metadata_from_batch(
                    batch,
                    schema=str(config.encoder_metadata_schema),
                )
                peak_mz = torch.from_numpy(data["peak_mz"][start:end]).to("cuda")
                peak_intensity = torch.from_numpy(
                    data["peak_intensity"][start:end]
                ).to("cuda")
                peak_valid_mask = torch.from_numpy(
                    data["peak_valid_mask"][start:end]
                ).to("cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    tokens = encoder(
                        peak_mz,
                        peak_intensity,
                        valid_mask=peak_valid_mask,
                        visible_mask=peak_valid_mask,
                        precursor_mz=batch["precursor_mz"],
                        spectrum_metadata=spectrum_metadata,
                    )
                handle.write(tokens.to(torch.float16).cpu().numpy().tobytes(order="C"))
                if start == 0 or (start // batch_size) % 100 == 0:
                    print(f"embedded {end}/{count}", flush=True)

    extraction_seconds = time.time() - extraction_started
    tokens_blob.reload()
    token_valid_mask = np.concatenate(
        [
            data["peak_valid_mask"],
            np.ones((count, 1), dtype=bool),
        ],
        axis=1,
    )
    mask_uri = f"{output_prefix}/token_valid_mask.npy"
    mask_blob = _write_npy(client, mask_uri, token_valid_mask)

    metadata = {
        "status": "complete",
        "run_id": RUN_ID,
        "run_url": f"https://wandb.ai/iclac/jepa-finalrun/runs/{RUN_ID}",
        "checkpoint": checkpoint,
        "checkpoint_step": CHECKPOINT_STEP,
        "encoder": "student",
        "encoder_runtime": "PyTorch",
        "torch_version": torch.__version__,
        "checkpoint_reader": "Orbax",
        "jax_version": jax.__version__,
        "device": torch.cuda.get_device_name(),
        "autocast_dtype": "bfloat16",
        "input_tsv": input_tsv,
        "input_tsv_sha256": _sha256(tsv_path),
        "samples": count,
        "train_samples": int(np.count_nonzero(data["fold"] == "train")),
        "val_samples": int(np.count_nonzero(data["fold"] == "val")),
        "test_samples": int(np.count_nonzero(data["fold"] == "test")),
        "peak_ordering": str(config.peak_ordering),
        "peak_slots": num_peak_tokens,
        "cls_token_index": num_peak_tokens,
        "token_layout": "127 peak slots followed by CLS",
        "embedding_uri": tokens_uri,
        "embedding_shape": list(embedding_shape),
        "embedding_dtype": "float16",
        "embedding_size": int(tokens_blob.size),
        "embedding_crc32c": tokens_blob.crc32c,
        "valid_mask_uri": mask_uri,
        "valid_mask_shape": list(token_valid_mask.shape),
        "valid_mask_dtype": "bool",
        "valid_mask_size": int(mask_blob.size),
        "valid_mask_crc32c": mask_blob.crc32c,
        "batch_size": batch_size,
        "preprocessing_seconds": preprocessing_seconds,
        "extraction_and_upload_seconds": extraction_seconds,
        "total_seconds": time.time() - started,
    }
    _gcs_blob(client, f"{output_prefix}/embedding_metadata.json").upload_from_string(
        json.dumps(metadata, indent=2, sort_keys=True),
        content_type="application/json",
    )
    return metadata


@app.local_entrypoint()
def main(
    batch_size: int = 128,
    checkpoint: str = CHECKPOINT,
    input_tsv: str = INPUT_TSV,
    output_prefix: str = OUTPUT_PREFIX,
) -> None:
    print(
        json.dumps(
            export_embeddings.remote(
                batch_size=batch_size,
                checkpoint=checkpoint,
                input_tsv=input_tsv,
                output_prefix=output_prefix,
            ),
            indent=2,
            sort_keys=True,
        )
    )
