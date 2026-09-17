"""Copy a pinned GCS checkpoint and verify every object before local restoration."""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path

import google_crc32c


def verify_object(root: Path, entry: dict) -> None:
    path = root / entry["name"]
    if path.stat().st_size != entry["size_bytes"]:
        raise ValueError(f"Checkpoint object size mismatch: {entry['name']}")
    checksum = google_crc32c.Checksum()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            checksum.update(block)
    if base64.b64encode(checksum.digest()).decode() != entry["crc32c"]:
        raise ValueError(f"Checkpoint object CRC32C mismatch: {entry['name']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = args.destination.resolve()
    manifest_path = root / "checkpoint_copy.json"
    if not args.verify_only:
        if args.selection is None:
            parser.error("--selection is required when downloading")
        from google.cloud import storage

        selection = json.loads(args.selection.read_text())
        bucket_name, prefix = selection["snapshot_uri"].removeprefix("gs://").split("/", 1)
        bucket = storage.Client().bucket(bucket_name)
        root.mkdir(parents=True, exist_ok=True)

        def download(entry):
            path = root / entry["name"]
            if path.exists():
                verify_object(root, entry)
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            generation = int(entry["snapshot_generation"])
            blob = bucket.blob(f"{prefix}/{entry['name']}", generation=generation)
            blob.reload(if_generation_match=generation)
            if blob.size != entry["size_bytes"] or blob.crc32c != entry["crc32c"]:
                raise ValueError(f"Pinned GCS object metadata mismatch: {entry['name']}")
            temporary = path.with_name(path.name + ".partial")
            blob.download_to_filename(temporary, if_generation_match=generation, checksum="crc32c")
            temporary.replace(path)
            print(f"Downloaded {entry['name']} ({entry['size_bytes']} bytes)", flush=True)

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(download, selection["objects"]))
        manifest = {"snapshot_uri": selection["snapshot_uri"], "objects": selection["objects"],
                    "all_checks_passed": False}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    manifest = json.loads(manifest_path.read_text())
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda entry: verify_object(root, entry), manifest["objects"]))
    manifest.update(all_checks_passed=True, verified_at_utc=datetime.now(timezone.utc).isoformat(),
                    verified_bytes=sum(entry["size_bytes"] for entry in manifest["objects"]))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: value for key, value in manifest.items() if key != "objects"}), flush=True)


if __name__ == "__main__":
    main()
