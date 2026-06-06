from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huggingface_hub import HfApi, hf_hub_download

from spectra_learning.data.gems.native import build_gems_native_artifact

log = logging.getLogger(__name__)

GEMS_SOURCE_REPO_ID = "roman-bushuiev/GeMS"
GEMS_B_SOURCE_FILENAME = "data/GeMS_B/GeMS_B.hdf5"


def _download_source(url: str, work_dir: Path) -> Path:
    parsed = urlparse(url)
    filename = Path(parsed.path).name or "source.hdf5"
    download_path = work_dir / "source" / filename
    download_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading source HDF5 from %s", url)
    urlretrieve(url, download_path)
    return download_path


def _download_hf_source(
    *,
    repo_id: str,
    filename: str,
    revision: str,
    work_dir: Path,
) -> Path:
    log.info("Downloading source HDF5 from %s@%s:%s", repo_id, revision, filename)
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            revision=revision,
            local_dir=work_dir / "source",
        )
    )


def _hf_source_url(*, repo_id: str, filename: str, revision: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{filename}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and upload GeMS native PyTorch shards."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--source-hdf5-path", type=Path)
    source_group.add_argument("--source-url")
    source_group.add_argument("--source-hf-filename")
    source_group.add_argument("--source-gems-b", action="store_true")
    parser.add_argument("--source-hf-repo-id", default=GEMS_SOURCE_REPO_ID)
    parser.add_argument("--source-hf-revision", default="main")
    parser.add_argument("--hf-repo-id", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--hf-revision", default="main")
    parser.add_argument("--max-precursor-mz", type=float, default=1000.0)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--upload-num-workers", type=int, default=8)
    args = parser.parse_args()

    work_dir = args.work_dir.expanduser().resolve()
    artifact_dir = work_dir / "artifact"
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)

    if args.source_hdf5_path is not None:
        hdf5_path = args.source_hdf5_path.expanduser().resolve()
        source_path = str(hdf5_path)
        source_url = None
    elif args.source_url is not None:
        hdf5_path = _download_source(args.source_url, work_dir)
        source_path = None
        source_url = args.source_url
    else:
        source_hf_filename = (
            GEMS_B_SOURCE_FILENAME if args.source_gems_b else args.source_hf_filename
        )
        hdf5_path = _download_hf_source(
            repo_id=args.source_hf_repo_id,
            filename=source_hf_filename,
            revision=args.source_hf_revision,
            work_dir=work_dir,
        )
        source_path = None
        source_url = _hf_source_url(
            repo_id=args.source_hf_repo_id,
            filename=source_hf_filename,
            revision=args.source_hf_revision,
        )

    build_gems_native_artifact(
        hdf5_path=hdf5_path,
        output_dir=artifact_dir,
        max_precursor_mz=args.max_precursor_mz,
        num_shards=args.num_shards,
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
        num_workers=args.upload_num_workers,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
