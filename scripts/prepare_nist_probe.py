"""Download NIST hr_msms MGF from GCS, build the probe HDF5, upload to HF.

End-to-end pipeline for making ``hr_msms_nist.mgf`` consumable by the online
MSG probe (``probe_dataset="nist-full"``). The produced HDF5 matches the schema
that :mod:`utils.massspec_probe_data` expects; enable it at train time by
setting::

    cfg.probe_dataset = "nist-full"
    cfg.nist_full_hdf5_repo_id = "<hf-owner>/hr_msms_nist_probe"
    cfg.nist_full_hdf5_filename = "hr_msms_nist.hdf5"
    cfg.nist_full_probe_train_samples = 4000
    cfg.nist_full_probe_test_samples = 1000
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from huggingface_hub import HfApi, upload_file

from utils.nist_probe_hdf5 import build_nist_probe_hdf5

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build + upload full NIST hr_msms HDF5 for the MSG probe."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--source-mgf-path", type=Path)
    source_group.add_argument("--gcs-uri", help="gs://bucket/path/to.mgf")
    parser.add_argument(
        "--gcs-credentials",
        type=Path,
        default=Path("key.json"),
        help="GCP service account JSON (used with --gcs-uri).",
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--output-filename",
        default="hr_msms_nist.hdf5",
        help="Filename stored in the HF dataset repo.",
    )
    parser.add_argument("--hf-repo-id", required=True, help="e.g. cjim8889/hr_msms_nist_probe")
    parser.add_argument("--hf-private", action="store_true")
    parser.add_argument("--hf-revision", default="main")
    parser.add_argument("--hf-path-in-repo", default=None,
                        help="Path in the HF repo (default: same as --output-filename)")
    parser.add_argument("--num-peaks-input", type=int, default=128)
    parser.add_argument("--max-precursor-mz", type=float, default=1000.0)
    parser.add_argument("--min-precursor-mz", type=float, default=1.0)
    parser.add_argument("--compression", default="gzip",
                        help="HDF5 compression: 'gzip', 'lzf', or 'none'.")
    parser.add_argument("--compression-opts", type=int, default=4)
    parser.add_argument("--skip-upload", action="store_true")
    args = parser.parse_args()

    work_dir = args.work_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.source_mgf_path is not None:
        mgf_path = args.source_mgf_path.expanduser().resolve()
    else:
        mgf_path = _download_from_gcs(args.gcs_uri, args.gcs_credentials, work_dir)

    output_path = work_dir / args.output_filename
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
    log.info("HDF5 stats: %s", stats)

    if args.skip_upload:
        log.info("--skip-upload set; artifact left at %s", output_path)
        return

    path_in_repo = args.hf_path_in_repo or args.output_filename
    api = HfApi()
    api.create_repo(
        args.hf_repo_id,
        repo_type="dataset",
        exist_ok=True,
        private=args.hf_private,
    )
    log.info("Uploading %s -> %s:%s", output_path, args.hf_repo_id, path_in_repo)
    upload_file(
        path_or_fileobj=str(output_path),
        path_in_repo=path_in_repo,
        repo_id=args.hf_repo_id,
        repo_type="dataset",
        revision=args.hf_revision,
    )
    log.info(
        "Uploaded to https://huggingface.co/datasets/%s (file %s)",
        args.hf_repo_id,
        path_in_repo,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
