from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from spectra_learning.data import murcko_dreams


def test_dreams_auxiliary_records_checkpoint_and_source_commit(
    monkeypatch,
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "artifact"
    subdir = murcko_dreams.NIST_MURCKO_PREPARED_SUBDIR
    subdir_path = artifact_dir / subdir
    subdir_path.mkdir(parents=True)
    (subdir_path / "metadata.json").write_text(
        json.dumps(
            {
                "splits": ["train", "val", "test"],
                "train_files": ["train.parquet"],
                "val_files": ["val.parquet"],
                "test_files": ["test.parquet"],
            }
        )
    )
    (artifact_dir / "metadata.json").write_text(
        json.dumps({"datasets": {"nist": {"subdir": subdir}}})
    )
    checkpoint = tmp_path / "embedding_model.ckpt"
    checkpoint.write_bytes(b"checkpoint-weights")

    monkeypatch.setattr(murcko_dreams, "_install_dreams_path", lambda _: None)
    monkeypatch.setattr(
        murcko_dreams,
        "_load_model",
        lambda checkpoint, *, device, n_highest_peaks: (
            object(),
            object(),
            object(),
        ),
    )
    monkeypatch.setattr(
        murcko_dreams,
        "_git_head_commit",
        lambda _: "0123456789abcdef",
    )

    def fake_write_split_auxiliary(**kwargs):
        split = kwargs["split"]
        output_path = (
            kwargs["subdir_path"]
            / "auxiliary"
            / "dreams"
            / f"{split}-part-00000.npz"
        )
        np.savez_compressed(
            output_path,
            spectrum_index=np.asarray([0], dtype=np.int64),
            dreams_embedding=np.zeros((1, 3), dtype=np.float32),
            dreams_embedding_valid=np.asarray([True]),
        )
        return (
            [f"auxiliary/dreams/{split}-part-00000.npz"],
            [1],
            1,
            0,
            {},
            [output_path],
        )

    monkeypatch.setattr(
        murcko_dreams,
        "_write_split_auxiliary",
        fake_write_split_auxiliary,
    )

    murcko_dreams.build_murcko_dreams_auxiliary(
        artifact_dir=artifact_dir,
        dreams_root=tmp_path / "Dreams",
        checkpoint=checkpoint,
        subdirs=[subdir],
        device="cpu",
    )

    expected_checkpoint = {
        "path": str(checkpoint.resolve()),
        "bytes": checkpoint.stat().st_size,
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    metadata = json.loads((subdir_path / "metadata.json").read_text())
    assert metadata["dreams_checkpoint"] == expected_checkpoint
    assert metadata["dreams_source_git_commit"] == "0123456789abcdef"
    top_metadata = json.loads((artifact_dir / "metadata.json").read_text())
    assert top_metadata["datasets"]["nist"]["dreams_checkpoint"] == expected_checkpoint
    assert (
        top_metadata["datasets"]["nist"]["dreams_source_git_commit"]
        == "0123456789abcdef"
    )
