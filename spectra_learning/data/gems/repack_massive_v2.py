from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
from pathlib import Path
from typing import Any

import h5py

from spectra_learning.data.gems.prepare_massive_v2 import (
    dataset_card,
    upload_artifact,
    validate_artifact,
)

DEFAULT_SOURCE_MANIFEST = Path(
    "/mnt/tg-go-nvme/massive-v2-conversion-v3/output/manifest.json"
)
DEFAULT_WORK_DIR = Path("/mnt/tg-go-nvme/massive-v2-repack-v3-10gb")
DEFAULT_DESTINATION_REPO_ID = (
    "novogaia/massive-v2-ms2-t095-l080-sharded-10gb"
)
DEFAULT_TARGET_BYTES = 10_000_000_000
SPECTRUM_COPY_ROWS = 16_384
SCALAR_COPY_ROWS = 1_048_576


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


def plan_shard_groups(
    shards: list[dict[str, Any]],
    target_bytes: int,
) -> list[list[dict[str, Any]]]:
    group_count = max(1, round(sum(shard["bytes"] for shard in shards) / target_bytes))
    cumulative = []
    running = 0
    for shard in shards:
        running += int(shard["bytes"])
        cumulative.append(running)
    boundaries = []
    previous = 0
    for group_index in range(1, group_count):
        desired = running * group_index / group_count
        first = previous + 1
        last = len(shards) - (group_count - group_index)
        boundary = min(
            range(first, last + 1),
            key=lambda index: abs(cumulative[index - 1] - desired),
        )
        boundaries.append(boundary)
        previous = boundary
    starts = [0, *boundaries]
    stops = [*boundaries, len(shards)]
    return [
        shards[start:stop]
        for start, stop in zip(starts, stops, strict=True)
    ]


def _create_output_datasets(
    output: h5py.File,
    source: h5py.File,
    names: list[str],
    rows: int,
) -> None:
    for name in names:
        dataset = source[name]
        output.create_dataset(
            name,
            shape=(rows, *dataset.shape[1:]),
            dtype=dataset.dtype,
            chunks=dataset.chunks,
            compression=dataset.compression,
            compression_opts=dataset.compression_opts,
            shuffle=dataset.shuffle,
        )


def _copy_dataset(
    source: h5py.Dataset,
    destination: h5py.Dataset,
    destination_start: int,
) -> None:
    copy_rows = (
        SPECTRUM_COPY_ROWS
        if source.name == "/spectrum"
        else SCALAR_COPY_ROWS
    )
    for start in range(0, len(source), copy_rows):
        stop = min(start + copy_rows, len(source))
        destination[
            destination_start + start : destination_start + stop
        ] = source[start:stop]


def repack_group(
    source_root: Path,
    output_root: Path,
    split: str,
    output_index: int,
    shards: list[dict[str, Any]],
    dataset_names: list[str],
) -> dict[str, Any]:
    relative_path = f"{split}/shard_{output_index:05d}.hdf5"
    output_path = output_root / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = sum(int(shard["rows"]) for shard in shards)
    eligible_rows = sum(int(shard["eligible_rows"]) for shard in shards)
    if not output_path.exists():
        temporary_path = output_path.with_suffix(".partial.hdf5")
        temporary_path.unlink(missing_ok=True)
        first_path = source_root / shards[0]["path"]
        with (
            h5py.File(first_path, "r") as first,
            h5py.File(temporary_path, "w") as output,
        ):
            _create_output_datasets(
                output,
                first,
                dataset_names,
                rows,
            )
            destination_start = 0
            for shard in shards:
                with h5py.File(source_root / shard["path"], "r") as source:
                    for name in dataset_names:
                        _copy_dataset(
                            source[name],
                            output[name],
                            destination_start,
                        )
                    destination_start += int(shard["rows"])
        temporary_path.replace(output_path)
    with h5py.File(output_path, "r") as output:
        chunk_rows = int(output["spectrum"].chunks[0])
    return {
        "path": relative_path,
        "rows": rows,
        "eligible_rows": eligible_rows,
        "bytes": output_path.stat().st_size,
        "sha256": _sha256(output_path),
        "chunk_rows": chunk_rows,
    }


def _repack_group(args: tuple[Any, ...]) -> dict[str, Any]:
    return repack_group(*args)


def repack_artifact(
    source_manifest_path: Path,
    work_dir: Path,
    *,
    target_bytes: int = DEFAULT_TARGET_BYTES,
    workers: int = 4,
) -> Path:
    source_manifest = json.loads(source_manifest_path.read_text())
    output_dir = work_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        return manifest_path
    plans = {
        split: plan_shard_groups(
            source_manifest["splits"][split]["shards"],
            target_bytes,
        )
        for split in ("train", "validation")
    }
    _json_write(
        work_dir / "plan.json",
        {
            "source_manifest": str(source_manifest_path),
            "target_bytes": target_bytes,
            "groups": {
                split: [
                    [shard["path"] for shard in group]
                    for group in groups
                ]
                for split, groups in plans.items()
            },
        },
    )
    tasks = [
        (
            source_manifest_path.parent,
            output_dir,
            split,
            index,
            group,
            source_manifest["row_aligned_datasets"],
        )
        for split, groups in plans.items()
        for index, group in enumerate(groups)
    ]
    if workers == 1:
        records = [_repack_group(task) for task in tasks]
    else:
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            records = pool.map(_repack_group, tasks)
    by_split = {
        split: [
            record
            for record in records
            if record["path"].startswith(f"{split}/")
        ]
        for split in ("train", "validation")
    }
    manifest = dict(source_manifest)
    manifest["conversion"] = {
        **source_manifest["conversion"],
        "target_shard_bytes": target_bytes,
        "repacked_from_manifest_sha256": _sha256(source_manifest_path),
    }
    manifest["conversion"].pop("target_rows_per_shard", None)
    for split in ("train", "validation"):
        manifest["splits"][split]["shards"] = by_split[split]
    _json_write(manifest_path, manifest)
    (output_dir / "README.md").write_text(
        dataset_card(manifest)
        + f"\nFiles are repacked toward a soft {target_bytes / 1e9:g} GB "
        "target while preserving group-safe boundaries.\n"
    )
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge validated MassIVE v2 shards toward a byte target."
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=DEFAULT_SOURCE_MANIFEST,
    )
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument(
        "--target-bytes",
        type=int,
        default=DEFAULT_TARGET_BYTES,
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument(
        "--destination-repo-id",
        default=DEFAULT_DESTINATION_REPO_ID,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = repack_artifact(
        args.source_manifest,
        args.work_dir,
        target_bytes=args.target_bytes,
        workers=args.workers,
    )
    validate_artifact(manifest_path, workers=args.workers)
    if args.upload:
        print(upload_artifact(manifest_path.parent, args.destination_repo_id))


if __name__ == "__main__":
    main()
