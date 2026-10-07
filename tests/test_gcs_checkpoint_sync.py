import importlib.util
import os
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gcs_checkpoint_sync.py"
spec = importlib.util.spec_from_file_location("gcs_checkpoint_sync", SCRIPT)
sync = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)


class FakeBlob:
    def __init__(self, bucket, name):
        self.bucket, self.name = bucket, name

    @property
    def size(self):
        return len(self.bucket.objects[self.name])

    def upload_from_filename(self, filename, **kwargs):
        self.bucket.objects[self.name] = Path(filename).read_bytes()
        self.bucket.writes.append(self.name)

    def upload_from_string(self, data, **kwargs):
        self.bucket.objects[self.name] = data.encode()
        self.bucket.writes.append(self.name)

    def download_to_filename(self, filename, **kwargs):
        Path(filename).write_bytes(self.bucket.objects[self.name])

    def delete(self, **kwargs):
        del self.bucket.objects[self.name]


class FakeBucket:
    name = "bucket"

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.writes: list[str] = []
        self.client = self

    def blob(self, name):
        return FakeBlob(self, name)

    def list_blobs(self, bucket, prefix):
        return [FakeBlob(self, n) for n in sorted(self.objects) if n.startswith(prefix)]


def write_step(workdir: Path, step: int, payload: bytes = b"data", *, age: float = 600):
    step_dir = workdir / "checkpoints" / "orbax" / str(step)
    files = {
        "_CHECKPOINT_METADATA": b"{}",
        "metadata/metadata": b"{}",
        "state/manifest.ocdbt": b"m",
        "state/ocdbt.process_0/d/abc": payload,
    }
    for rel, content in files.items():
        (step_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        (step_dir / rel).write_bytes(content)
        os.utime(step_dir / rel, (time.time() - age,) * 2)
    return step_dir


def push(bucket, workdir, settle=120.0):
    return sync.push(bucket, "runs/r", workdir, settle_seconds=settle, workers=2)


def test_push_uploads_finalized_steps_with_commit_markers_last(tmp_path):
    bucket = FakeBucket()
    write_step(tmp_path, 10)
    (tmp_path / "checkpoints" / "orbax" / "20.orbax-checkpoint-tmp").mkdir()
    (tmp_path / "config.json").write_text("{}")

    assert push(bucket, tmp_path) == [10]

    step = "runs/r/checkpoints/orbax/10"
    assert bucket.objects[f"{step}/state/ocdbt.process_0/d/abc"] == b"data"
    assert bucket.writes[-3:] == [
        f"{step}/metadata/commit_success.txt",
        f"{step}/state/commit_success.txt",
        f"{step}/commit_success.txt",
    ]
    assert "runs/r/config.json" in bucket.objects
    assert push(bucket, tmp_path) == []


def test_push_waits_for_recent_steps(tmp_path):
    bucket = FakeBucket()
    write_step(tmp_path, 10, age=0)
    assert push(bucket, tmp_path) == []
    assert push(bucket, tmp_path, settle=0.0) == [10]


def test_push_replaces_stale_partial_but_keeps_complete_steps(tmp_path):
    bucket = FakeBucket()
    write_step(tmp_path, 10, b"old")
    push(bucket, tmp_path)
    complete = dict(bucket.objects)
    # An earlier, aborted writer left a partial step 20 with other files.
    bucket.objects["runs/r/checkpoints/orbax/20/state/ocdbt.process_0/d/zzz"] = b"x"
    write_step(tmp_path, 20, b"new")

    assert push(bucket, tmp_path) == [20]
    assert "runs/r/checkpoints/orbax/20/state/ocdbt.process_0/d/zzz" not in bucket.objects
    assert all(bucket.objects[name] == data for name, data in complete.items())


def test_push_raises_when_upload_does_not_verify(tmp_path, monkeypatch):
    bucket = FakeBucket()
    write_step(tmp_path, 10)
    monkeypatch.setattr(FakeBlob, "upload_from_filename", lambda self, *a, **k: None)
    with pytest.raises(RuntimeError, match="did not verify"):
        push(bucket, tmp_path)
    assert "runs/r/checkpoints/orbax/10/commit_success.txt" not in bucket.objects


def test_pull_seeds_newest_complete_step_into_empty_workdir(tmp_path):
    bucket = FakeBucket()
    source = tmp_path / "source"
    write_step(source, 10)
    write_step(source, 20, b"twenty")
    push(bucket, source)
    bucket.objects["runs/r/checkpoints/orbax/30/state/x"] = b"partial"

    dest = tmp_path / "dest"
    assert sync.pull(bucket, "runs/r", dest, workers=2) == 20
    pulled = dest / "checkpoints" / "orbax" / "20"
    assert (pulled / "state/ocdbt.process_0/d/abc").read_bytes() == b"twenty"
    assert (pulled / "commit_success.txt").exists()
    assert sync.local_steps(dest) == [20]
    assert sync.pull(bucket, "runs/r", dest, workers=2) is None
    # A workdir with steps is left alone even when GCS is ahead.
    write_step(source, 40)
    push(bucket, source)
    assert sync.pull(bucket, "runs/r", dest, workers=2) is None
    # The seeded step is already complete on GCS, so push leaves it alone.
    bucket.writes.clear()
    assert push(bucket, dest) == []


def test_run_dirs_lists_only_workdirs_with_checkpoints(tmp_path):
    write_step(tmp_path / "run-a", 10)
    (tmp_path / "run-b").mkdir()
    (tmp_path / "notes.txt").write_text("")
    assert [p.name for p in sync.run_dirs(tmp_path)] == ["run-a"]


def test_push_mirrors_wandb_files_and_reuploads_grown_ones(tmp_path):
    bucket = FakeBucket()
    run_file = tmp_path / "wandb" / "run-1-r" / "run-r.wandb"
    run_file.parent.mkdir(parents=True)
    run_file.write_bytes(b"ab")
    (tmp_path / "wandb" / "latest-run").symlink_to(run_file.parent)

    push(bucket, tmp_path)
    assert bucket.objects == {"runs/r/wandb/run-1-r/run-r.wandb": b"ab"}

    bucket.writes.clear()
    push(bucket, tmp_path)
    assert bucket.writes == []
    run_file.write_bytes(b"abcd")
    push(bucket, tmp_path)
    assert bucket.objects["runs/r/wandb/run-1-r/run-r.wandb"] == b"abcd"
