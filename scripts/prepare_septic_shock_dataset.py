from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huggingface_hub import HfApi

from spectra_learning.data.septic_shock import (
    SEPTIC_SHOCK_RAW_BYTES,
    SEPTIC_SHOCK_RAW_MD5,
    SEPTIC_SHOCK_RAW_URL,
    SEPTIC_SHOCK_SPLIT_SEED,
    build_septic_shock_peaklist_artifact,
    prepare_septic_shock_dataset,
)

DEFAULT_HF_REPO_ID = "cjim8889/msms_evaluation_100ktrain_20260615"
DEFAULT_HF_SUBDIR = "septic_shock_st003189_raw_peaklist_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare Metabolomics Workbench ST003189 septic-shock metadata and "
            "optionally download/extract the raw mzXML archive."
        )
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/septic_shock_st003189"),
    )
    parser.add_argument("--split-seed", type=int, default=SEPTIC_SHOCK_SPLIT_SEED)
    parser.add_argument(
        "--download-raw",
        action="store_true",
        help="Download and verify the 2.5 GB ST003189 raw mzXML archive.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refresh downloaded metadata/raw files and re-extract the archive.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Download metadata only and print the raw archive URL/checksum.",
    )
    parser.add_argument(
        "--build-peaklist-artifact",
        action="store_true",
        help="Convert extracted mzXML files to the project-native raw peaklist artifact.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Output directory for the raw peaklist artifact.",
    )
    parser.add_argument(
        "--ms-level",
        type=int,
        default=None,
        help="Optional mzXML msLevel filter. Omit to include all scans.",
    )
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--hf-repo-id", default=DEFAULT_HF_REPO_ID)
    parser.add_argument("--hf-subdir", default=DEFAULT_HF_SUBDIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = prepare_septic_shock_dataset(
        args.cache_dir,
        download_raw=bool(args.download_raw and not args.dry_run),
        split_seed=int(args.split_seed),
        force=bool(args.force),
    )
    summary = {
        "cache_dir": str(args.cache_dir.expanduser().resolve()),
        "num_samples": int(metadata["num_samples"]),
        "label_counts": metadata["label_counts"],
        "split_counts": metadata["split_counts"],
        "raw_url": SEPTIC_SHOCK_RAW_URL,
        "raw_archive_bytes": SEPTIC_SHOCK_RAW_BYTES,
        "raw_archive_md5": SEPTIC_SHOCK_RAW_MD5,
        "downloaded_raw": bool(metadata["downloaded_raw"]),
    }
    if args.build_peaklist_artifact:
        artifact_dir = (
            args.artifact_dir
            if args.artifact_dir is not None
            else args.cache_dir / "artifact"
        )
        artifact_metadata = build_septic_shock_peaklist_artifact(
            cache_dir=args.cache_dir,
            output_dir=artifact_dir,
            ms_level=args.ms_level,
            split_seed=int(args.split_seed),
        )
        summary["artifact_dir"] = str(artifact_dir.expanduser().resolve())
        summary["artifact_format"] = artifact_metadata["artifact_format"]
        summary["artifact_split_counts"] = artifact_metadata["split_counts"]
        summary["artifact_num_scans"] = {
            split: int(artifact_metadata[f"{split}_num_scans"])
            for split in ("train", "val", "test")
        }
        if args.upload:
            api = HfApi()
            api.upload_folder(
                repo_id=args.hf_repo_id,
                repo_type="dataset",
                folder_path=artifact_dir,
                path_in_repo=args.hf_subdir.strip("/"),
                commit_message=(
                    "Add ST003189 septic-shock raw peaklist evaluation artifact"
                ),
            )
            summary["uploaded_to"] = (
                f"https://huggingface.co/datasets/{args.hf_repo_id}/tree/main/"
                f"{args.hf_subdir.strip('/')}"
            )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
