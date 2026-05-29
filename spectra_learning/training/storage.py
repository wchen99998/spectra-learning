from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from fsspec.core import url_to_fs


StoragePath = str | Path


@dataclass(frozen=True)
class StorageFile:
    path: StoragePath
    name: str
    mtime: float


def is_remote_path(path: StoragePath) -> bool:
    raw = str(path)
    return "://" in raw and not raw.startswith("file://")


def normalize_storage_path(path: StoragePath) -> StoragePath:
    if is_remote_path(path):
        return str(path).rstrip("/")
    return Path(path).expanduser().resolve()


def storage_join(path: StoragePath, *parts: str | Path) -> StoragePath:
    if is_remote_path(path):
        suffix = "/".join(str(part).strip("/") for part in parts)
        return f"{str(path).rstrip('/')}/{suffix}" if suffix else str(path).rstrip("/")
    return Path(path).joinpath(*(str(part) for part in parts))


def storage_name(path: StoragePath) -> str:
    if is_remote_path(path):
        return str(path).rstrip("/").rsplit("/", 1)[-1]
    return Path(path).name


def storage_parent(path: StoragePath) -> StoragePath:
    if is_remote_path(path):
        raw = str(path).rstrip("/")
        scheme, rest = raw.split("://", 1)
        bucket, _, key = rest.partition("/")
        if not key or "/" not in key:
            return f"{scheme}://{bucket}"
        return f"{scheme}://{bucket}/{key.rsplit('/', 1)[0]}"
    return Path(path).parent


def storage_with_name(path: StoragePath, name: str) -> StoragePath:
    return storage_join(storage_parent(path), name)


def storage_with_suffix(path: StoragePath, suffix: str) -> StoragePath:
    return storage_with_name(path, Path(storage_name(path)).with_suffix(suffix).name)


def storage_mkdir(path: StoragePath) -> None:
    if is_remote_path(path):
        return
    Path(path).expanduser().mkdir(parents=True, exist_ok=True)


def storage_exists(path: StoragePath) -> bool:
    if is_remote_path(path):
        fs, fs_path = url_to_fs(str(path))
        return fs.exists(fs_path)
    return Path(path).expanduser().exists()


def storage_delete(path: StoragePath) -> None:
    if is_remote_path(path):
        fs, fs_path = url_to_fs(str(path))
        fs.rm(fs_path)
        return
    Path(path).expanduser().unlink()


def storage_delete_if_exists(path: StoragePath) -> None:
    if storage_exists(path):
        storage_delete(path)


def local_scratch_dir(path: StoragePath) -> Path:
    if not is_remote_path(path):
        return Path(path).expanduser().resolve()
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:16]
    root = Path(os.environ.get("SPECTRA_SCRATCH_DIR", "/tmp/spectra-learning"))
    return root / digest


def local_cache_path(path: StoragePath) -> Path:
    if not is_remote_path(path):
        return Path(path).expanduser().resolve()
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:16]
    root = Path(os.environ.get("SPECTRA_SCRATCH_DIR", "/tmp/spectra-learning"))
    return root / "files" / digest / storage_name(path)


def local_path_for_read(path: StoragePath) -> Path:
    if not is_remote_path(path):
        return Path(path).expanduser().resolve()
    local_path = local_cache_path(path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    fs, fs_path = url_to_fs(str(path))
    fs.get_file(fs_path, str(local_path))
    return local_path


def upload_local_file(local_path: Path, path: StoragePath) -> None:
    if not is_remote_path(path):
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        Path(local_path).replace(target)
        return
    fs, fs_path = url_to_fs(str(path))
    fs.put_file(str(local_path), fs_path)


def write_text(path: StoragePath, text: str) -> None:
    if is_remote_path(path):
        fs, fs_path = url_to_fs(str(path))
        with fs.open(fs_path, "wt") as f:
            f.write(text)
        return
    local_path = Path(path).expanduser()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(text)


def read_text(path: StoragePath) -> str:
    if is_remote_path(path):
        fs, fs_path = url_to_fs(str(path))
        with fs.open(fs_path, "rt") as f:
            return f.read()
    return Path(path).expanduser().read_text()


def list_storage_files(directory: StoragePath, *, recursive: bool = False) -> list[StorageFile]:
    if is_remote_path(directory):
        return _list_remote_files(directory, recursive=recursive)
    root = Path(directory).expanduser()
    paths = root.rglob("*") if recursive else root.glob("*")
    return [
        StorageFile(path=path, name=path.name, mtime=path.stat().st_mtime)
        for path in paths
        if path.is_file()
    ]


def _list_remote_files(directory: StoragePath, *, recursive: bool) -> list[StorageFile]:
    raw = str(directory).rstrip("/")
    scheme = raw.split("://", 1)[0]
    fs, fs_path = url_to_fs(raw)
    if recursive:
        entries = fs.find(fs_path, withdirs=False, detail=True)
    else:
        entries = fs.find(fs_path, maxdepth=1, withdirs=False, detail=True)
    items = entries.items()
    return [
        StorageFile(
            path=f"{scheme}://{name.lstrip('/')}",
            name=name.rstrip("/").rsplit("/", 1)[-1],
            mtime=_info_mtime(info),
        )
        for name, info in items
        if info.get("type", "file") == "file"
    ]


def _info_mtime(info: dict[str, Any]) -> float:
    value = info.get("mtime", info.get("updated", info.get("created", 0.0)))
    if isinstance(value, datetime):
        return value.timestamp()
    return float(value)
