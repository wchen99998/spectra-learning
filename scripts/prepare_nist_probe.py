"""Download NIST hr_msms MGF from GCS, build the prepared probe artifact, upload to HF.

End-to-end pipeline for making ``hr_msms_nist.mgf`` consumable by the online
MSG probe (``probe_dataset="nist-full"``). The script writes the intermediate
HDF5, materializes the fully prepared shard artifact expected by
``MassSpecProbeData.from_config()``, writes the fixed Morgan-balanced pair set
used for online covariance/Tanimoto alignment plots, and uploads that artifact
to a dataset repo.
Enable it at train time by setting::

    cfg.probe_dataset = "nist-full"
    cfg.nist_full_probe_repo_id = "<hf-owner>/hr_msms_nist_probe"
    cfg.nist_full_probe_revision = "main"
    cfg.nist_full_probe_train_samples = 4000
    cfg.nist_full_probe_test_samples = 1000
    cfg.nist_full_probe_num_repeats = 3

Select the fingerprint target used by the online probe with::

    cfg.msg_probe_fingerprint = "maccs"   # 166 bits, default
    cfg.msg_probe_fingerprint = "morgan"  # radius 2, 4096 bits
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import Any, cast

from huggingface_hub import HfApi, hf_hub_download

from spectra_learning.probes.massspec.data import build_nist_full_probe_artifact
from spectra_learning.probes.massspec.nist_hdf5 import build_nist_probe_hdf5

log = logging.getLogger(__name__)


def _download_from_gcs(gcs_uri: str, credentials_path: Path, work_dir: Path) -> Path:
    from google.cloud import storage

    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got {gcs_uri!r}")
    without_scheme = gcs_uri[len("gs://") :]
    bucket_name, _, blob_path = without_scheme.partition("/")
    if not blob_path:
        raise ValueError(f"GCS URI missing object key: {gcs_uri}")

    download_path = work_dir / "source" / Path(blob_path).name
    download_path.parent.mkdir(parents=True, exist_ok=True)
    if download_path.exists() and download_path.stat().st_size > 0:
        log.info("Reusing cached GCS download at %s", download_path)
        return download_path
    log.info("Downloading %s -> %s", gcs_uri, download_path)
    client = storage.Client.from_service_account_json(str(credentials_path))
    client.bucket(bucket_name).blob(blob_path).download_to_filename(str(download_path))
    return download_path


def _download_from_hf(repo_id: str, filename: str, work_dir: Path) -> Path:
    source_dir = work_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=str(source_dir),
    )
    return Path(path)


def main() -> None:
    from rdkit import RDLogger

    cast(Any, RDLogger).DisableLog("rdApp.*")
    parser = argparse.ArgumentParser(
        description="Build + upload the prepared NIST full probe artifact."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--source-mgf-path", type=Path)
    source_group.add_argument("--source-hdf5-path", type=Path)
    source_group.add_argument("--source-hf-repo-id")
    source_group.add_argument("--gcs-uri", help="gs://bucket/path/to.mgf")
    parser.add_argument(
        "--gcs-credentials",
        type=Path,
        default=Path("key.json"),
        help="GCP service account JSON (used with --gcs-uri).",
    )
    parser.add_argument(
        "--source-hf-filename",
        default="hr_msms_nist.hdf5",
        help="Dataset filename used with --source-hf-repo-id.",
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--output-filename",
        default="hr_msms_nist.hdf5",
        help="Intermediate HDF5 filename when building from MGF.",
    )
    parser.add_argument("--hf-repo-id", required=True, help="e.g. cjim8889/hr_msms_nist_probe")
    parser.add_argument("--hf-private", action="store_true")
    parser.add_argument("--hf-revision", default="main")
    parser.add_argument("--num-peaks-input", type=int, default=128)
    parser.add_argument("--max-precursor-mz", type=float, default=1000.0)
    parser.add_argument("--min-precursor-mz", type=float, default=1.0)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--compression", default="gzip",
                        help="HDF5 compression: 'gzip', 'lzf', or 'none'.")
    parser.add_argument("--compression-opts", type=int, default=4)
    parser.add_argument("--skip-upload", action="store_true")
    args = parser.parse_args()

    work_dir = args.work_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    source_dir = work_dir / "source"
    artifact_dir = work_dir / "artifact"
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)

    if args.source_mgf_path is not None:
        mgf_path = args.source_mgf_path.expanduser().resolve()
        source_dir.mkdir(parents=True, exist_ok=True)
        output_path = source_dir / args.output_filename
        compression = None if args.compression.lower() == "none" else args.compression
        stats = build_nist_probe_hdf5(
            mgf_path,
            output_path,
            num_peaks_input=args.num_peaks_input,
            max_precursor_mz=args.max_precursor_mz,
            min_precursor_mz=args.min_precursor_mz,
            compression=compression,
            compression_opts=args.compression_opts if compression == "gzip" else None,
        )
        log.info("Intermediate HDF5 stats: %s", stats)
        hdf5_path = output_path
    elif args.source_hdf5_path is not None:
        hdf5_path = args.source_hdf5_path.expanduser().resolve()
    elif args.source_hf_repo_id is not None:
        hdf5_path = _download_from_hf(
            args.source_hf_repo_id,
            args.source_hf_filename,
            work_dir,
        )
    else:
        mgf_path = _download_from_gcs(args.gcs_uri, args.gcs_credentials, work_dir)
        source_dir.mkdir(parents=True, exist_ok=True)
        output_path = source_dir / args.output_filename
        compression = None if args.compression.lower() == "none" else args.compression
        stats = build_nist_probe_hdf5(
            mgf_path,
            output_path,
            num_peaks_input=args.num_peaks_input,
            max_precursor_mz=args.max_precursor_mz,
            min_precursor_mz=args.min_precursor_mz,
            compression=compression,
            compression_opts=args.compression_opts if compression == "gzip" else None,
        )
        log.info("Intermediate HDF5 stats: %s", stats)
        hdf5_path = output_path
    metadata = build_nist_full_probe_artifact(
        hdf5_path,
        artifact_dir,
        max_precursor_mz=args.max_precursor_mz,
        num_shards=args.num_shards,
    )
    log.info(
        "Prepared artifact stats: train=%d val=%d test=%d",
        metadata["train_size"],
        metadata["val_size"],
        metadata["test_size"],
    )

    if args.skip_upload:
        log.info("--skip-upload set; artifact left at %s", artifact_dir)
        return

    api = HfApi()
    api.create_repo(
        args.hf_repo_id,
        repo_type="dataset",
        exist_ok=True,
        private=args.hf_private,
    )
    log.info("Uploading prepared artifact %s -> %s", artifact_dir, args.hf_repo_id)
    api.upload_large_folder(
        repo_id=args.hf_repo_id,
        folder_path=artifact_dir,
        repo_type="dataset",
        revision=args.hf_revision,
    )
    log.info(
        "Uploaded prepared NIST full probe artifact to https://huggingface.co/datasets/%s",
        args.hf_repo_id,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
