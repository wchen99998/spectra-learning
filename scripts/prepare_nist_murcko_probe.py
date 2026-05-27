"""Build and upload the prepared NIST Murcko probe artifact.

The training runtime expects ``probe_dataset="nist-murcko"`` to download native
probe shards from ``nist_murcko_probe/`` in the HF dataset repo. This script is
the one-time Parquet-to-native-shard path used to create that subdirectory.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import HfApi, snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from spectra_learning.data.spectra import NUM_PEAKS_INPUT
from spectra_learning.probes.massspec.data import (
    NIST_MURCKO_ARTIFACT_FORMAT,
    NIST_MURCKO_HF_REPO,
    NIST_MURCKO_METADATA_VERSION,
    NIST_MURCKO_PREPARED_SUBDIR,
    _filter_encode_and_write,
    _normalize_spectra_intensity,
)

NIST_MURCKO_SPLIT_DIR = "hr_msms_nist_dreams_embeddings_murcko_split"

log = logging.getLogger(__name__)


def _spectra_from_peak_lists(
    mz_lists: list[list[float]],
    intensity_lists: list[list[float]],
) -> np.ndarray:
    spectra = np.zeros((len(mz_lists), 2, NUM_PEAKS_INPUT), dtype=np.float32)
    for i, (mz, intensity) in enumerate(zip(mz_lists, intensity_lists, strict=True)):
        n = min(len(mz), NUM_PEAKS_INPUT)
        spectra[i, 0, :n] = np.asarray(mz[:n], dtype=np.float32)
        spectra[i, 1, :n] = np.asarray(intensity[:n], dtype=np.float32)
    return _normalize_spectra_intensity(spectra)


def _collision_energy_from_metadata(
    metadata_json: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    collision_energy = np.zeros(len(metadata_json), dtype=np.float32)
    collision_energy_present = np.zeros(len(metadata_json), dtype=np.int32)
    for i, raw in enumerate(metadata_json):
        metadata = json.loads(raw)
        value = str(metadata.get("COLLISIONENERGY", "")).strip()
        match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", value)
        if match is not None and value.lower() != "nan":
            collision_energy[i] = float(match.group(0))
            collision_energy_present[i] = 1
    return collision_energy, collision_energy_present


def _load_nist_murcko_parquet_split(
    parquet_path: Path,
    split_name: str,
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    table = pq.read_table(
        parquet_path,
        columns=[
            "dreams_embedding",
            "spectrum_mz",
            "spectrum_intensity",
            "metadata_json",
            "precursor_mz",
            "adduct",
            "smiles",
        ],
    )
    rows = table.to_pydict()
    metadata = [json.loads(raw) for raw in rows["metadata_json"]]
    collision_energy, collision_energy_present = _collision_energy_from_metadata(
        rows["metadata_json"]
    )
    n = len(rows["smiles"])
    return {
        "spectra": _spectra_from_peak_lists(
            rows["spectrum_mz"],
            rows["spectrum_intensity"],
        ),
        "precursor": np.asarray(rows["precursor_mz"], dtype=np.float32),
        "fold": np.repeat(split_name, n),
        "smiles": np.asarray(rows["smiles"], dtype=str),
        "adduct": np.asarray(
            [value or "unknown" for value in rows["adduct"]],
            dtype=str,
        ),
        "instrument_type": np.asarray(
            [item.get("INSTRUMENTTYPE", "") or "unknown" for item in metadata],
            dtype=str,
        ),
        "collision_energy": collision_energy,
        "collision_energy_present": collision_energy_present,
        "dreams_embedding": np.asarray(rows["dreams_embedding"], dtype=np.float32),
    }


def _load_nist_murcko_parquet_splits(source_dir: Path) -> dict[str, Any]:
    payloads = [
        _load_nist_murcko_parquet_split(source_dir / f"{split}.parquet", split)
        for split in ("train", "val", "test")
    ]
    keys = payloads[0].keys()
    return {
        key: np.concatenate([payload[key] for payload in payloads], axis=0)
        for key in keys
    }


def build_nist_murcko_probe_artifact(
    source_dir: Path,
    output_dir: Path,
    *,
    max_precursor_mz: float,
    num_shards: int,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = _filter_encode_and_write(
        **_load_nist_murcko_parquet_splits(source_dir),
        output_dir=output_dir,
        num_shards=num_shards,
        max_precursor_mz=max_precursor_mz,
        metadata_version=NIST_MURCKO_METADATA_VERSION,
        write_pairwise_alignment=False,
    )
    metadata["artifact_format"] = NIST_MURCKO_ARTIFACT_FORMAT
    if extra_metadata is not None:
        metadata.update(extra_metadata)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def _download_source_splits(
    *,
    repo_id: str,
    revision: str,
    split_dir: str,
    work_dir: Path,
) -> Path:
    source_root = work_dir / "source"
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=source_root,
        allow_patterns=[
            f"{split_dir}/metadata.json",
            f"{split_dir}/train.parquet",
            f"{split_dir}/val.parquet",
            f"{split_dir}/test.parquet",
        ],
    )
    return source_root / split_dir


def main() -> None:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    parser = argparse.ArgumentParser(
        description="Build and upload prepared NIST Murcko probe shards."
    )
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--source-hf-repo-id", default=NIST_MURCKO_HF_REPO)
    parser.add_argument("--source-hf-revision", default="main")
    parser.add_argument("--source-split-dir", default=NIST_MURCKO_SPLIT_DIR)
    parser.add_argument("--hf-repo-id", default=NIST_MURCKO_HF_REPO)
    parser.add_argument("--hf-revision", default="main")
    parser.add_argument("--hf-subdir", default=NIST_MURCKO_PREPARED_SUBDIR)
    parser.add_argument("--hf-private", action="store_true")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--max-precursor-mz", type=float, default=1000.0)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--skip-upload", action="store_true")
    args = parser.parse_args()

    work_dir = args.work_dir.expanduser().resolve()
    staging_root = work_dir / "artifact"
    artifact_dir = staging_root / args.hf_subdir.strip("/")
    if staging_root.exists():
        shutil.rmtree(staging_root)

    if args.source_dir is not None:
        source_dir = args.source_dir.expanduser().resolve()
        source_metadata = {
            "parquet_split_dir": str(source_dir),
        }
    else:
        source_dir = _download_source_splits(
            repo_id=args.source_hf_repo_id,
            revision=args.source_hf_revision,
            split_dir=args.source_split_dir,
            work_dir=work_dir,
        )
        source_metadata = {
            "parquet_repo_id": args.source_hf_repo_id,
            "parquet_revision": args.source_hf_revision,
            "parquet_split_dir": args.source_split_dir,
        }

    metadata = build_nist_murcko_probe_artifact(
        source_dir,
        artifact_dir,
        max_precursor_mz=args.max_precursor_mz,
        num_shards=args.num_shards,
        extra_metadata=source_metadata,
    )
    log.info(
        "Prepared artifact stats: train=%d val=%d test=%d",
        metadata["train_size"],
        metadata["val_size"],
        metadata["test_size"],
    )

    if args.skip_upload:
        log.info("--skip-upload set; staged artifact left at %s", staging_root)
        return

    api = HfApi()
    api.create_repo(
        args.hf_repo_id,
        repo_type="dataset",
        exist_ok=True,
        private=args.hf_private,
    )
    log.info("Uploading prepared artifact %s -> %s", staging_root, args.hf_repo_id)
    api.upload_large_folder(
        repo_id=args.hf_repo_id,
        folder_path=staging_root,
        repo_type="dataset",
        revision=args.hf_revision,
    )
    log.info(
        "Uploaded prepared NIST Murcko artifact to https://huggingface.co/datasets/%s/tree/%s/%s",
        args.hf_repo_id,
        args.hf_revision,
        args.hf_subdir.strip("/"),
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
