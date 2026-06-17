from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


MANIFEST_NAME = ".spectra_aot_manifest.json"
MANIFEST_VERSION = 2
CACHE_FILE_GLOB = "jit_*-cache"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _normalise_json(value: str) -> str:
    return json.dumps(json.loads(value), sort_keys=True, separators=(",", ":"))


def _relative_to_repo(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _source_files(repo_root: Path, config: str) -> list[Path]:
    paths = [
        repo_root / "pyproject.toml",
        repo_root / "uv.lock",
        repo_root / config,
        repo_root / "scripts" / "compile_jax_train_step.py",
        repo_root / "spectra_learning" / "training" / "pretrain_jax.py",
        repo_root / "spectra_learning" / "training" / "tpu_compile.py",
        repo_root / "spectra_learning" / "models",
        repo_root / "spectra_learning" / "data" / "gems",
    ]
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(p for p in path.rglob("*.py") if p.is_file()))
    return sorted(set(files), key=lambda p: _relative_to_repo(p, repo_root))


def fingerprint_payload(*, config: str, overrides_json: str) -> dict[str, Any]:
    repo_root = _repo_root()
    normalized_overrides = _normalise_json(overrides_json)
    files = _source_files(repo_root, config)
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "version": MANIFEST_VERSION,
                "config": config,
                "overrides_json": normalized_overrides,
                "source_files": [_relative_to_repo(path, repo_root) for path in files],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    for path in files:
        rel = _relative_to_repo(path, repo_root)
        digest.update(b"\0path\0")
        digest.update(rel.encode())
        digest.update(b"\0content\0")
        digest.update(path.read_bytes())
    return {
        "version": MANIFEST_VERSION,
        "digest": digest.hexdigest(),
        "config": config,
        "overrides_json": normalized_overrides,
        "source_files": [_relative_to_repo(path, repo_root) for path in files],
    }


def cache_files(cache_dir: Path, pattern: str = CACHE_FILE_GLOB) -> list[Path]:
    if not cache_dir.is_dir():
        return []
    return sorted(
        path
        for path in cache_dir.glob(pattern)
        if path.is_file() and path.stat().st_size > 0
    )


def expected_cache_counts(summary_json: str | None) -> dict[str, int] | None:
    if not summary_json:
        return None
    path = Path(summary_json)
    if not path.is_file() or path.stat().st_size == 0:
        return None
    payload = json.loads(path.read_text())
    if not isinstance(payload, list) or not payload:
        return None
    counts: dict[str, int] = {}
    for entry in payload:
        if not isinstance(entry, dict):
            return None
        cache_glob = entry.get("cache_glob")
        if not isinstance(cache_glob, str) or not cache_glob:
            return None
        counts[cache_glob] = counts.get(cache_glob, 0) + 1
    return counts


def manifest_path(cache_dir: Path) -> Path:
    return cache_dir / MANIFEST_NAME


def read_manifest(cache_dir: Path) -> dict[str, Any] | None:
    path = manifest_path(cache_dir)
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def is_ready(
    *,
    cache_dir: Path,
    config: str,
    overrides_json: str,
    summary_json: str | None = None,
) -> tuple[bool, str]:
    manifest = read_manifest(cache_dir)
    if manifest is None:
        return False, f"missing cache manifest {manifest_path(cache_dir)}"
    current = fingerprint_payload(config=config, overrides_json=overrides_json)
    if manifest.get("version") != MANIFEST_VERSION:
        return False, "cache manifest version mismatch"
    if manifest.get("digest") != current["digest"]:
        return False, "cache manifest digest does not match current sources/config"
    expected = expected_cache_counts(summary_json)
    if expected is None:
        raw_expected = manifest.get("expected_cache_counts", {})
        expected = {
            str(pattern): int(count)
            for pattern, count in raw_expected.items()
        }
    if not expected:
        return False, "missing expected cache counts"
    observed: dict[str, int] = {}
    for pattern, count in expected.items():
        if count <= 0:
            return False, f"invalid expected cache count for {pattern}: {count}"
        actual = len(cache_files(cache_dir, pattern))
        observed[pattern] = actual
        if actual < count:
            return False, f"found {actual} cache files for {pattern}, expected {count}"
    return True, f"cache ready with counts {observed}"


def write_manifest(
    *,
    cache_dir: Path,
    config: str,
    overrides_json: str,
    summary_json: str,
) -> dict[str, Any]:
    expected = expected_cache_counts(summary_json)
    if expected is None:
        raise SystemExit(f"summary JSON is missing or invalid: {summary_json}")
    current = fingerprint_payload(config=config, overrides_json=overrides_json)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        **current,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "expected_cache_counts": expected,
        "cache_counts": {
            pattern: len(cache_files(cache_dir, pattern))
            for pattern in expected
        },
    }
    manifest_path(cache_dir).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def _must_upload(path: Path, *, cache_dir: Path, force: bool) -> bool:
    if force:
        return True
    rel = path.relative_to(cache_dir).as_posix()
    if rel == MANIFEST_NAME:
        return True
    return path.match(CACHE_FILE_GLOB)


def upload_cache(*, cache_dir: Path, gcs_uri: str, force: bool = False) -> dict[str, int]:
    if not gcs_uri.startswith("gs://"):
        raise SystemExit(f"gcs URI must start with gs://, got {gcs_uri!r}")
    try:
        from google.cloud import storage
    except ImportError as exc:
        raise SystemExit("google-cloud-storage is required for cache upload") from exc

    bucket_name, _, prefix = gcs_uri[5:].rstrip("/").partition("/")
    if not bucket_name or not prefix:
        raise SystemExit(f"gcs URI must include bucket and prefix, got {gcs_uri!r}")
    client = storage.Client(project=os.environ.get("GOOGLE_CLOUD_PROJECT") or None)
    bucket = client.bucket(bucket_name)
    uploaded = 0
    skipped = 0
    bytes_uploaded = 0
    for path in sorted(p for p in cache_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(cache_dir).as_posix()
        blob = bucket.blob(f"{prefix}/{rel}")
        size = path.stat().st_size
        try:
            blob.reload()
            if int(blob.size or 0) == size and not _must_upload(
                path,
                cache_dir=cache_dir,
                force=force,
            ):
                skipped += 1
                continue
        except Exception:
            pass
        blob.upload_from_filename(str(path))
        uploaded += 1
        bytes_uploaded += size
    return {"uploaded": uploaded, "skipped": skipped, "bytes": bytes_uploaded}


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage Spectra JAX AOT cache state.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--cache-dir", required=True)
        subparser.add_argument("--config", required=True)
        subparser.add_argument("--overrides-json", required=True)

    ready = subparsers.add_parser("ready")
    add_common(ready)
    ready.add_argument("--summary-json", default="")

    write = subparsers.add_parser("write-manifest")
    add_common(write)
    write.add_argument("--summary-json", required=True)

    fingerprint = subparsers.add_parser("fingerprint")
    fingerprint.add_argument("--config", required=True)
    fingerprint.add_argument("--overrides-json", required=True)

    upload = subparsers.add_parser("upload")
    upload.add_argument("--cache-dir", required=True)
    upload.add_argument("--gcs-uri", required=True)
    upload.add_argument("--force", action="store_true")

    args = parser.parse_args()
    if args.command == "ready":
        ok, reason = is_ready(
            cache_dir=Path(args.cache_dir),
            config=args.config,
            overrides_json=args.overrides_json,
            summary_json=args.summary_json or None,
        )
        print(reason, file=sys.stderr)
        raise SystemExit(0 if ok else 1)
    if args.command == "write-manifest":
        manifest = write_manifest(
            cache_dir=Path(args.cache_dir),
            config=args.config,
            overrides_json=args.overrides_json,
            summary_json=args.summary_json,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return
    if args.command == "fingerprint":
        print(
            json.dumps(
                fingerprint_payload(
                    config=args.config,
                    overrides_json=args.overrides_json,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "upload":
        print(
            json.dumps(
                upload_cache(
                    cache_dir=Path(args.cache_dir),
                    gcs_uri=args.gcs_uri,
                    force=args.force,
                ),
                sort_keys=True,
            )
        )
        return
    raise AssertionError(args.command)


if __name__ == "__main__":
    main()
