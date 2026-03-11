from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve

from huggingface_hub import HfApi

from utils.gems_tfrecords import build_gems_tfrecord_artifact

log = logging.getLogger(__name__)


def _download_source(url: str, work_dir: Path) -> Path:
    parsed = urlparse(url)
    filename = Path(parsed.path).name or "source.hdf5"
    download_path = work_dir / "source" / filename
    download_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading source HDF5 from %s", url)
    urlretrieve(url, download_path)
    return download_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and upload canonical GeMS TFRecords."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--source-hdf5-path", type=Path)
    source_group.add_argument("--source-url")
    parser.add_argument("--hf-repo-id", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--hf-revision", default="main")
    parser.add_argument("--max-precursor-mz", type=float, default=1000.0)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    work_dir = args.work_dir.expanduser().resolve()
    artifact_dir = work_dir / "artifact"
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)

    if args.source_hdf5_path is not None:
        hdf5_path = args.source_hdf5_path.expanduser().resolve()
        source_path = str(hdf5_path)
        source_url = None
    else:
        hdf5_path = _download_source(args.source_url, work_dir)
        source_path = None
        source_url = args.source_url

    build_gems_tfrecord_artifact(
        hdf5_path=hdf5_path,
        output_dir=artifact_dir,
        max_precursor_mz=args.max_precursor_mz,
        num_workers=None if args.num_workers <= 0 else args.num_workers,
        source_path=source_path,
        source_url=source_url,
    )

    api = HfApi()
    api.create_repo(args.hf_repo_id, repo_type="dataset", exist_ok=True)
    api.upload_large_folder(
        repo_id=args.hf_repo_id,
        folder_path=artifact_dir,
        repo_type="dataset",
        revision=args.hf_revision,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
