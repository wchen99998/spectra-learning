"""Mirror local Orbax checkpoints to GCS, and seed a local workdir from GCS.

Training writes checkpoints only to local (Lustre) workdirs under one root;
this script, run on a login node, uploads them, because compute-node egress
can collapse to KB/s and Orbax async saves straight to GCS then time out and
crash the run. Orbax marks a GCS checkpoint complete with
``commit_success.txt``, so ``push`` writes those files last and readers that
check for them (Orbax itself) never see a partial upload. ``push`` never
deletes a complete GCS checkpoint; it only clears stale objects out of an
incomplete one before uploading into it.

  push: for every run directory under --local-root, upload each finalized
        step missing on GCS to --remote-root/<run>, and mirror the run's
        wandb directory (``--watch`` repeats).
  pull: seed a workdir without local steps from the newest complete GCS
        step, so a run can move from a GCS workdir to a local one.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import time
from pathlib import Path

from google.cloud import storage
from google.cloud.storage.retry import DEFAULT_RETRY

COMMIT_FILENAME = "commit_success.txt"
ORBAX_SUBDIR = Path("checkpoints") / "orbax"
TOP_LEVEL_FILES = ("config.json",)
WANDB_SUBDIR = "wandb"


def parse_gcs_url(url: str) -> tuple[str, str]:
    if not url.startswith("gs://"):
        raise ValueError(f"expected a gs:// URL, got {url!r}")
    bucket, _, prefix = url[len("gs://") :].partition("/")
    return bucket, prefix.strip("/")


def local_steps(local_dir: Path) -> list[int]:
    """Finalized local steps: Orbax renames the tmp dir to the bare step."""
    orbax_dir = local_dir / ORBAX_SUBDIR
    if not orbax_dir.is_dir():
        return []
    return sorted(int(p.name) for p in orbax_dir.iterdir() if p.name.isdigit())


def remote_step_files(
    bucket: storage.Bucket, prefix: str
) -> dict[int, dict[str, int]]:
    """{step: {relative path: size}} for every step object under the run."""
    orbax_prefix = f"{prefix}/{ORBAX_SUBDIR.as_posix()}/"
    steps: dict[int, dict[str, int]] = {}
    for blob in bucket.client.list_blobs(bucket, prefix=orbax_prefix):
        step, _, rel = blob.name[len(orbax_prefix) :].partition("/")
        if step.isdigit() and rel:
            steps.setdefault(int(step), {})[rel] = blob.size
    return steps


def is_complete(files: dict[str, int]) -> bool:
    return COMMIT_FILENAME in files


def _run_parallel(fn, items, workers: int) -> None:
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        for future in [pool.submit(fn, *item) for item in items]:
            future.result()


def push_step(
    bucket: storage.Bucket,
    prefix: str,
    step_dir: Path,
    remote_files: dict[str, int],
    workers: int,
) -> None:
    step_prefix = f"{prefix}/{ORBAX_SUBDIR.as_posix()}/{step_dir.name}"
    local_files = {
        p.relative_to(step_dir).as_posix(): p.stat().st_size
        for p in step_dir.rglob("*")
        if p.is_file()
    }
    # Commit markers copied from a GCS-seeded step are rewritten last.
    commits = sorted(
        (rel for rel in local_files if rel.split("/")[-1] == COMMIT_FILENAME),
        key=lambda rel: rel.count("/"),
        reverse=True,
    )
    data = {rel: size for rel, size in local_files.items() if rel not in commits}
    stale = [rel for rel in remote_files if rel not in data]
    _run_parallel(
        lambda rel: bucket.blob(f"{step_prefix}/{rel}").delete(),
        [(rel,) for rel in stale],
        workers,
    )
    todo = [rel for rel, size in data.items() if remote_files.get(rel) != size]
    _run_parallel(
        lambda rel: bucket.blob(f"{step_prefix}/{rel}").upload_from_filename(
            str(step_dir / rel), checksum="crc32c", retry=DEFAULT_RETRY
        ),
        [(rel,) for rel in todo],
        workers,
    )
    uploaded = {
        blob.name[len(step_prefix) + 1 :]: blob.size
        for blob in bucket.client.list_blobs(bucket, prefix=f"{step_prefix}/")
    }
    missing = {rel for rel, size in data.items() if uploaded.get(rel) != size}
    if missing:
        raise RuntimeError(f"{step_prefix}: {len(missing)} files did not verify")
    # Item markers (state/, metadata/) before the step marker Orbax checks.
    item_dirs = sorted({rel.split("/")[0] for rel in data if "/" in rel})
    markers = [f"{item}/{COMMIT_FILENAME}" for item in item_dirs] + [COMMIT_FILENAME]
    for rel in markers:
        bucket.blob(f"{step_prefix}/{rel}").upload_from_string(
            f"Checkpoint commit was successful to gs://{bucket.name}/{step_prefix}",
            retry=DEFAULT_RETRY,
        )


def push_wandb(
    bucket: storage.Bucket, prefix: str, local_dir: Path, workers: int
) -> None:
    """Mirror the run's wandb files (online and offline runs) to GCS.

    Files still being written are re-uploaded whenever their size changes.
    """
    wandb_dir = local_dir / WANDB_SUBDIR
    if not wandb_dir.is_dir():
        return
    remote_prefix = f"{prefix}/{WANDB_SUBDIR}/"
    remote = {
        blob.name[len(remote_prefix) :]: blob.size
        for blob in bucket.client.list_blobs(bucket, prefix=remote_prefix)
    }
    todo = []
    # os.walk skips the `latest-run` symlink, which only duplicates a run.
    for root, _, names in os.walk(wandb_dir):
        for name in names:
            path = Path(root) / name
            if path.is_symlink():
                continue
            rel = path.relative_to(wandb_dir).as_posix()
            if remote.get(rel) != path.stat().st_size:
                todo.append(rel)
    _run_parallel(
        lambda rel: bucket.blob(remote_prefix + rel).upload_from_filename(
            str(wandb_dir / rel), retry=DEFAULT_RETRY
        ),
        [(rel,) for rel in todo],
        workers,
    )


def push(
    bucket: storage.Bucket,
    prefix: str,
    local_dir: Path,
    *,
    settle_seconds: float,
    workers: int,
) -> list[int]:
    """Upload finalized local steps that are not complete on GCS."""
    for name in TOP_LEVEL_FILES:
        path = local_dir / name
        if path.is_file():
            bucket.blob(f"{prefix}/{name}").upload_from_filename(
                str(path), retry=DEFAULT_RETRY
            )
    push_wandb(bucket, prefix, local_dir, workers)
    remote = remote_step_files(bucket, prefix)
    pushed = []
    for step in local_steps(local_dir):
        if is_complete(remote.get(step, {})):
            continue
        step_dir = local_dir / ORBAX_SUBDIR / str(step)
        files = [p.stat() for p in step_dir.rglob("*") if p.is_file()]
        if time.time() - max(f.st_mtime for f in files) < settle_seconds:
            continue
        start = time.monotonic()
        size = sum(f.st_size for f in files)
        push_step(bucket, prefix, step_dir, remote.get(step, {}), workers)
        elapsed = time.monotonic() - start
        logging.info(
            "pushed %s step %d (%.2f GiB) in %.0f s (%.0f MiB/s)",
            local_dir.name,
            step,
            size / 2**30,
            elapsed,
            size / 2**20 / max(elapsed, 1e-9),
        )
        pushed.append(step)
    return pushed


def pull(
    bucket: storage.Bucket, prefix: str, local_dir: Path, *, workers: int
) -> int | None:
    """Download the newest complete GCS step into a workdir without steps."""
    if local_steps(local_dir):
        return None
    remote = remote_step_files(bucket, prefix)
    complete = [step for step, files in remote.items() if is_complete(files)]
    if not complete:
        return None
    step = max(complete)
    step_prefix = f"{prefix}/{ORBAX_SUBDIR.as_posix()}/{step}"
    tmp_dir = local_dir / ORBAX_SUBDIR / f"{step}.gcs-pull-tmp"
    start = time.monotonic()
    rels = list(remote[step])
    for rel in rels:
        (tmp_dir / rel).parent.mkdir(parents=True, exist_ok=True)
    _run_parallel(
        lambda rel: bucket.blob(f"{step_prefix}/{rel}").download_to_filename(
            str(tmp_dir / rel), checksum="crc32c"
        ),
        [(rel,) for rel in rels],
        workers,
    )
    for rel, size in remote[step].items():
        if (tmp_dir / rel).stat().st_size != size:
            raise RuntimeError(f"{tmp_dir / rel}: size mismatch after download")
    tmp_dir.rename(local_dir / ORBAX_SUBDIR / str(step))
    size = sum(remote[step].values())
    elapsed = time.monotonic() - start
    logging.info(
        "pulled step %d (%.2f GiB) in %.0f s (%.0f MiB/s)",
        step,
        size / 2**30,
        elapsed,
        size / 2**20 / max(elapsed, 1e-9),
    )
    return step


def run_dirs(local_root: Path) -> list[Path]:
    return sorted(p for p in local_root.iterdir() if (p / ORBAX_SUBDIR).is_dir())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    push_parser = sub.add_parser("push")
    push_parser.add_argument("--local-root", type=Path, required=True)
    push_parser.add_argument("--remote-root", required=True, help="gs://bucket/prefix")
    push_parser.add_argument(
        "--settle-seconds",
        type=float,
        default=120.0,
        help="skip steps whose newest file is younger than this",
    )
    push_parser.add_argument(
        "--watch", type=float, default=0.0, help="seconds between passes"
    )
    pull_parser = sub.add_parser("pull")
    pull_parser.add_argument("--local", type=Path, required=True, help="workdir")
    pull_parser.add_argument("--remote", required=True, help="gs:// workdir")
    for p in (push_parser, pull_parser):
        p.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.getLogger("google").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    if args.mode == "pull":
        bucket_name, prefix = parse_gcs_url(args.remote)
        bucket = storage.Client().bucket(bucket_name)
        pull(bucket, prefix, args.local, workers=args.workers)
        return
    bucket_name, root_prefix = parse_gcs_url(args.remote_root)
    bucket = storage.Client().bucket(bucket_name)
    while True:
        for run_dir in run_dirs(args.local_root):
            try:
                push(
                    bucket,
                    f"{root_prefix}/{run_dir.name}",
                    run_dir,
                    settle_seconds=args.settle_seconds,
                    workers=args.workers,
                )
            except Exception:
                # The step stays incomplete on GCS; the next pass retries it.
                if not args.watch:
                    raise
                logging.exception("push of %s failed; retrying next pass", run_dir)
        if not args.watch:
            return
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
