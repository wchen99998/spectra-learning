"""Export full JEPA token arrays (127 peak slots + CLS) for the Novogaia evaluation harness.

Restores the original pretrained student, optionally overwrites its trainable
parameters with the jointly adapted encoder stored in a retrieval checkpoint
(`best.msgpack`, params["peak"]["encoder"]), and writes under --out:

    tokens_f16.npy           float16 (N, 128, D); CLS at index 127
    token_valid_mask.npy     bool (N, 128); CLS slot always valid
    embedding_metadata.json  harness contract (cls_token_index) and provenance

Rows follow the MassSpecGym 1.5 TSV order, as in the JEPA training exports.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flax import nnx, serialization
import jax
import numpy as np

from scripts.export_retrieval_tokens import BATCH_SIZE, INPUT_NAMES, build_token_encoder, padded_inputs, restore_encoder


def restore_adapted(encoder, checkpoint: Path):
    """Replace the trainable leaves exactly as TrainableJEPAPeak splits them."""
    graphdef, parameters, static = nnx.split(
        encoder, lambda path, value: isinstance(value, nnx.Param) and path[-1] != "b", ...)
    saved = serialization.msgpack_restore(checkpoint.read_bytes())["params"]["peak"]["encoder"]
    target = nnx.to_pure_dict(parameters)
    restored = serialization.from_state_dict(target, saved)
    nnx.replace_by_pure_dict(parameters, jax.tree.map(np.asarray, restored))
    return nnx.merge(graphdef, parameters, static)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="Retrieval input JSON (jepa_checkpoint, jepa_metadata, jepa_data, input_root)")
    parser.add_argument("--retrieval-checkpoint", type=Path, help="best.msgpack whose adapted encoder replaces the pretrained student")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--extra-metadata", type=json.loads, default={}, help="JSON object merged into embedding_metadata.json")
    args = parser.parse_args(argv)
    paths = {k: Path(v) for k, v in json.loads(args.inputs.read_text()).items()}
    args.out.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_default_matmul_precision", "highest")

    config = json.loads(paths["jepa_metadata"].read_text())["task_contract"]["config"]
    with np.load(paths["jepa_data"] / "prepared_massspecgym.npz", allow_pickle=False) as saved:
        data = {name: saved[name] for name in INPUT_NAMES}
    mask = data["peak_valid_mask"]
    np.testing.assert_array_equal(mask, np.load(paths["input_root"] / "data/token_valid_mask.npy")[:, :-1])

    encoder = restore_encoder(config, paths["jepa_checkpoint"])
    checkpoint_sha256 = None
    if args.retrieval_checkpoint is not None:
        with args.retrieval_checkpoint.open("rb") as stream:
            checkpoint_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        encoder = restore_adapted(encoder, args.retrieval_checkpoint)
    encode = build_token_encoder(config["encoder_metadata_schema"])

    n, slots, dim = len(mask), mask.shape[1] + 1, int(config["model_dim"])
    shape = (n, slots, dim)
    started = time.monotonic()
    with (args.out / "tokens_f16.npy.partial").open("wb") as stream:
        np.lib.format.write_array_header_1_0(stream, {"descr": "<f2", "fortran_order": False, "shape": shape})
        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)
            values = np.asarray(encode(encoder, *padded_inputs(data, np.arange(start, end))), dtype=np.float16)[:end - start]
            if values.shape[1:] != shape[1:] or not np.isfinite(values).all():
                raise ValueError(f"Unexpected token batch at row {start}: {values.shape}")
            stream.write(values.tobytes())
            if (start // BATCH_SIZE + 1) % 64 == 0 or end == n:
                print(json.dumps({"rows": end, "total": n, "seconds": time.monotonic() - started}), flush=True)
    (args.out / "tokens_f16.npy.partial").replace(args.out / "tokens_f16.npy")
    np.save(args.out / "token_valid_mask.npy", np.concatenate([mask, np.ones((n, 1), bool)], axis=1))

    # Drift of the exported peak tokens from the cached pretrained export on a few rows.
    cached = paths["input_root"] / "covariance_embeddings.float16.npy"
    drift = None
    if cached.exists():
        rows = np.arange(8)
        exported = np.load(args.out / "tokens_f16.npy", mmap_mode="r")[rows, :-1].astype(np.float32)
        reference = np.load(cached, mmap_mode="r")[rows].astype(np.float32)
        valid = mask[rows]
        drift = float(np.linalg.norm(exported[valid] - reference[valid]) / np.linalg.norm(reference[valid]))
    fold = np.load(paths["input_root"] / "data/fold.npy", allow_pickle=True).astype(str)
    metadata = {
        "cls_token_index": slots - 1, "token_layout": f"{slots - 1} peak slots followed by CLS",
        "peak_slots": slots - 1, "peak_ordering": config["peak_ordering"],
        "embedding_shape": list(shape), "embedding_dtype": "float16",
        "valid_mask_shape": [n, slots], "valid_mask_dtype": "bool",
        "samples": n, **{f"{name}_samples": int((fold == name).sum()) for name in ("train", "val", "test")},
        "encoder": "student" if args.retrieval_checkpoint is None else "retrieval_joint_adapted_student",
        "encoder_runtime": "JAX", "jax_version": jax.__version__, "batch_size": BATCH_SIZE,
        "pretrained_checkpoint": str(paths["jepa_checkpoint"]),
        "retrieval_checkpoint": None if args.retrieval_checkpoint is None else str(args.retrieval_checkpoint),
        "retrieval_checkpoint_sha256": checkpoint_sha256,
        "relative_l2_vs_pretrained_cache_first8": drift,
        "input_tsv_sha256": json.loads((paths["input_root"] / "embedding_metadata.json").read_text())["tsv_sha256"],
        "device": jax.devices()[0].device_kind, "extraction_seconds": time.monotonic() - started,
        "status": "complete", **args.extra_metadata,
    }
    (args.out / "embedding_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
