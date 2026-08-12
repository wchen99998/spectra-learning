from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
from pathlib import Path
from typing import Any

import h5py
from huggingface_hub import snapshot_download

from spectra_learning.data.gems.eligibility import (
    massive_v2_eligibility_contract,
    massive_v2_training_eligibility_numpy,
)
from spectra_learning.data.gems.prepare_massive_v2 import (
    CONVERSION_VERSION,
    SCALAR_CHUNK_ROWS,
    dataset_card,
    upload_artifact,
    validate_artifact,
)

SOURCE_REPO_ID = "novogaia/massive-v2-ms2-t095-l080-sharded-10gb"
SOURCE_REVISION = "4de48add4e687f6ea561dc6ec74f8984ad8aebe0"
DESTINATION_REPO_ID = (
    "novogaia/massive-v2-ms2-t095-l080-sharded-10gb-repaired-staging"
)
DEFAULT_WORK_DIR = Path("/mnt/data/massive-v2-ms2-usable-peaks-v5")
REPAIR_VERSION = "recompute_training_eligible_from_spectra_v1"


def _json_write(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_artifact(
    work_dir: Path,
    *,
    repo_id: str,
    revision: str,
    workers: int,
) -> Path:
    artifact_dir = work_dir / "artifact"
    marker = work_dir / "download_complete"
    if not marker.exists():
        snapshot_download(
            repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=artifact_dir,
            allow_patterns=[
                "manifest.json",
                "README.md",
                "train/*.hdf5",
                "validation/*.hdf5",
            ],
            max_workers=workers,
        )
        marker.touch()
    return artifact_dir / "manifest.json"


def _repair_shard(
    args: tuple[Path, dict[str, Any], Path],
) -> dict[str, Any]:
    artifact_dir, shard, state_dir = args
    state_path = state_dir / (str(shard["path"]).replace("/", "--") + ".json")
    if state_path.exists():
        return json.loads(state_path.read_text())

    path = artifact_dir / shard["path"]
    with h5py.File(path, "r+") as file:
        spectrum = file["spectrum"]
        training_eligible = file["training_eligible"]
        eligible_rows = 0
        for start in range(0, len(spectrum), SCALAR_CHUNK_ROWS):
            stop = min(start + SCALAR_CHUNK_ROWS, len(spectrum))
            eligible = massive_v2_training_eligibility_numpy(
                spectrum[start:stop],
                file["precursor_mz"][start:stop],
                file["RT"][start:stop],
            )
            training_eligible[start:stop] = eligible
            eligible_rows += int(eligible.sum())
        file.flush()
        chunk_rows = int(spectrum.chunks[0])

    record = {
        "path": str(shard["path"]),
        "rows": int(shard["rows"]),
        "eligible_rows": eligible_rows,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "chunk_rows": chunk_rows,
    }
    _json_write(state_path, record)
    return record


def repair_local_artifact(
    source_manifest_path: Path,
    *,
    workers: int,
    source_repo_id: str = SOURCE_REPO_ID,
    source_revision: str = SOURCE_REVISION,
) -> Path:
    artifact_dir = source_manifest_path.parent
    source_manifest = json.loads(source_manifest_path.read_text())
    state_dir = artifact_dir.parent / f"{artifact_dir.name}_repair_state"
    state_dir.mkdir(exist_ok=True)
    tasks = [
        (artifact_dir, shard, state_dir)
        for split in ("train", "validation")
        for shard in source_manifest["splits"][split]["shards"]
    ]
    if workers == 1:
        records = [_repair_shard(task) for task in tasks]
    else:
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            records = pool.map(_repair_shard, tasks, chunksize=1)
    records_by_path = {record["path"]: record for record in records}

    manifest = source_manifest
    manifest["conversion"] = {
        **manifest["conversion"],
        "version": CONVERSION_VERSION,
        "eligibility_repair_version": REPAIR_VERSION,
        "eligibility_repaired_from_repo_id": source_repo_id,
        "eligibility_repaired_from_revision": source_revision,
    }
    manifest["eligibility"] = massive_v2_eligibility_contract()
    for split in ("train", "validation"):
        shards = [
            records_by_path[str(shard["path"])]
            for shard in manifest["splits"][split]["shards"]
        ]
        manifest["splits"][split]["shards"] = shards
        manifest["splits"][split]["eligible_rows"] = sum(
            shard["eligible_rows"] for shard in shards
        )
    manifest["eligible_rows"] = sum(
        manifest["splits"][split]["eligible_rows"]
        for split in ("train", "validation")
    )
    _json_write(source_manifest_path, manifest)
    (artifact_dir / "README.md").write_text(dataset_card(manifest))
    return source_manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute MassIVE v2 training eligibility from usable spectra."
        )
    )
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--source-repo-id", default=SOURCE_REPO_ID)
    parser.add_argument("--source-revision", default=SOURCE_REVISION)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument(
        "--destination-repo-id",
        default=DESTINATION_REPO_ID,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = download_artifact(
        args.work_dir,
        repo_id=args.source_repo_id,
        revision=args.source_revision,
        workers=args.download_workers,
    )
    repair_local_artifact(
        manifest_path,
        workers=args.workers,
        source_repo_id=args.source_repo_id,
        source_revision=args.source_revision,
    )
    validate_artifact(manifest_path, workers=args.workers)
    if args.upload:
        print(upload_artifact(manifest_path.parent, args.destination_repo_id))


if __name__ == "__main__":
    main()
