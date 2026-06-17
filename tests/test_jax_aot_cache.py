from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from scripts import jax_aot_cache


class _FakeBlob:
    def __init__(self, size: int | None) -> None:
        self.size = size
        self.uploads = 0

    def reload(self) -> None:
        if self.size is None:
            raise RuntimeError("blob does not exist")

    def upload_from_filename(self, filename: str) -> None:
        self.uploads += 1
        self.size = Path(filename).stat().st_size


class _FakeBucket:
    def __init__(self, existing_sizes: dict[str, int]) -> None:
        self._existing_sizes = existing_sizes
        self.blobs: dict[str, _FakeBlob] = {}

    def blob(self, name: str) -> _FakeBlob:
        if name not in self.blobs:
            self.blobs[name] = _FakeBlob(self._existing_sizes.get(name))
        return self.blobs[name]


def _install_fake_storage(monkeypatch, bucket: _FakeBucket) -> None:
    google = ModuleType("google")
    cloud = ModuleType("google.cloud")
    storage = SimpleNamespace(
        Client=lambda project=None: SimpleNamespace(bucket=lambda name: bucket)
    )
    cloud.storage = storage
    google.cloud = cloud
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.cloud", cloud)
    monkeypatch.setitem(sys.modules, "google.cloud.storage", storage)


def test_upload_cache_replaces_manifest_and_train_step_when_sizes_match(
    tmp_path,
    monkeypatch,
):
    manifest = tmp_path / jax_aot_cache.MANIFEST_NAME
    train_step = tmp_path / "jit_pure_sharded_accumulated_train_step-pack20-cache"
    aux = tmp_path / "unrelated-cache-entry"
    manifest.write_text("{}\n")
    train_step.write_text("compiled executable")
    aux.write_text("auxiliary cache")
    existing = {
        f"cache/{path.name}": path.stat().st_size
        for path in (manifest, train_step, aux)
    }
    bucket = _FakeBucket(existing)
    _install_fake_storage(monkeypatch, bucket)

    result = jax_aot_cache.upload_cache(
        cache_dir=tmp_path,
        gcs_uri="gs://spectra-bucket/cache",
    )

    assert result == {
        "uploaded": 2,
        "skipped": 1,
        "bytes": manifest.stat().st_size + train_step.stat().st_size,
    }
    assert bucket.blob(f"cache/{manifest.name}").uploads == 1
    assert bucket.blob(f"cache/{train_step.name}").uploads == 1
    assert bucket.blob(f"cache/{aux.name}").uploads == 0


def test_upload_cache_force_replaces_all_same_size_objects(tmp_path, monkeypatch):
    first = tmp_path / jax_aot_cache.MANIFEST_NAME
    second = tmp_path / "auxiliary-cache-entry"
    first.write_text("{}\n")
    second.write_text("auxiliary cache")
    existing = {f"cache/{path.name}": path.stat().st_size for path in (first, second)}
    bucket = _FakeBucket(existing)
    _install_fake_storage(monkeypatch, bucket)

    result = jax_aot_cache.upload_cache(
        cache_dir=tmp_path,
        gcs_uri="gs://spectra-bucket/cache",
        force=True,
    )

    assert result["uploaded"] == 2
    assert result["skipped"] == 0
    assert bucket.blob(f"cache/{first.name}").uploads == 1
    assert bucket.blob(f"cache/{second.name}").uploads == 1
