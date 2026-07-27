"""Build row-aligned DreaMS embedding auxiliary files for Murcko Parquet artifacts."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm import tqdm

from spectra_learning.data.mgf import file_source_manifest
from spectra_learning.data.murcko import (
    NIST_MURCKO_PREPARED_SUBDIR,
)

DEFAULT_ARTIFACT_DIR = Path("data/prepared/nist_murcko_mh_lsh/artifact")
DEFAULT_DREAMS_ROOT = Path("/home/wuhao/Dreams")
DEFAULT_DREAMS_CHECKPOINT = (
    DEFAULT_DREAMS_ROOT / "dreams/models/pretrained/embedding_model.ckpt"
)
DEFAULT_SUBDIRS = (NIST_MURCKO_PREPARED_SUBDIR,)
DEFAULT_N_HIGHEST_PEAKS = 100
DEFAULT_BATCH_SIZE = 256

log = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate row-aligned DreaMS embedding auxiliary NPZ files."
    )
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument(
        "--subdir",
        action="append",
        default=None,
        help="Prepared dataset subdir to process. Defaults to NIST.",
    )
    parser.add_argument("--dreams-root", type=Path, default=DEFAULT_DREAMS_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_DREAMS_CHECKPOINT)
    parser.add_argument("--n-highest-peaks", type=int, default=DEFAULT_N_HIGHEST_PEAKS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--hf-repo-id", default="")
    parser.add_argument("--hf-revision", default="main")
    parser.add_argument("--hf-private", action="store_true")
    return parser.parse_args()


def _install_dreams_path(dreams_root: Path) -> None:
    sys.path.insert(0, str(dreams_root.expanduser().resolve()))


def _git_head_commit(source_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(source_root.expanduser().resolve()), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _parse_charge(row: dict[str, Any]) -> int | None:
    raw = None
    metadata_json = row.get("metadata_json")
    if metadata_json:
        metadata = json.loads(metadata_json)
        raw = metadata.get("charge") or metadata.get("CHARGE") or metadata.get("Charge")
    if raw in (None, ""):
        return None
    raw = str(raw).strip().rstrip("+")
    return int(raw)


def _row_spectrum(row: dict[str, Any]) -> np.ndarray:
    return np.stack(
        [
            np.asarray(row["spectrum_mz"], dtype=np.float32),
            np.asarray(row["spectrum_intensity"], dtype=np.float32),
        ],
        axis=0,
    )


def _validate_rows(
    rows: list[dict[str, Any]],
    *,
    dformat: Any,
    n_highest_peaks: int,
) -> tuple[np.ndarray, dict[str, int]]:
    su = importlib.import_module("dreams.utils.spectra")

    valid = np.zeros(len(rows), dtype=bool)
    problem_counts: Counter[str] = Counter()
    for idx, row in enumerate(rows):
        spec = su.trim_peak_list(_row_spectrum(row), n_highest=n_highest_peaks)
        problem = dformat.val_spec(
            spec,
            float(row["precursor_mz"]),
            charge=_parse_charge(row),
            return_problems=True,
        )
        valid[idx] = problem == "All checks passed"
        if not valid[idx]:
            problem_counts[str(problem)] += 1
    return valid, dict(problem_counts)


def _compute_embeddings(
    rows: list[dict[str, Any]],
    *,
    model: torch.nn.Module,
    spec_preproc: Any,
    batch_size: int,
) -> np.ndarray:
    du = importlib.import_module("dreams.utils.data")
    spectrum_key = importlib.import_module("dreams.definitions").SPECTRUM

    spectra = [_row_spectrum(row) for row in rows]
    precursors = [float(row["precursor_mz"]) for row in rows]
    dataset = du.RawSpectraDataset(spectra, precursors, spec_preproc)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )
    predictions = []
    for batch in tqdm(dataloader, desc="DreaMS embeddings", leave=False):
        spectrum = batch[spectrum_key].to(device=model.device, dtype=model.dtype)
        with torch.inference_mode():
            predictions.append(model(spectrum).detach().cpu().to(torch.float32).numpy())
    return np.concatenate(predictions, axis=0).astype(np.float32, copy=False)


def _load_model(
    checkpoint: Path,
    *,
    device: str,
    n_highest_peaks: int,
) -> tuple[torch.nn.Module, Any, Any]:
    du = importlib.import_module("dreams.utils.data")
    dformats = importlib.import_module("dreams.utils.dformats")
    pre_trained_model = importlib.import_module("dreams.api").PreTrainedModel
    contrastive_head = importlib.import_module(
        "dreams.models.heads.heads"
    ).ContrastiveHead

    dformat = dformats.DataFormatA()
    checkpoint = checkpoint.expanduser().resolve()
    model_ckpt = pre_trained_model.from_ckpt(
        checkpoint,
        contrastive_head,
        n_highest_peaks,
    )
    model_ckpt.model.to(torch.device(device))
    spec_preproc = du.SpectrumPreprocessor(
        dformat=dformat,
        n_highest_peaks=model_ckpt.n_highest_peaks,
    )
    return model_ckpt.model, spec_preproc, dformat


def _split_names(metadata: dict[str, Any]) -> list[str]:
    return [str(split) for split in metadata.get("splits", ("train", "val", "test"))]


def _write_split_auxiliary(
    *,
    subdir_path: Path,
    split: str,
    filenames: list[str],
    model: torch.nn.Module,
    spec_preproc: Any,
    dformat: Any,
    batch_size: int,
    n_highest_peaks: int,
) -> tuple[list[str], list[int], int, int, dict[str, int], list[Path]]:
    auxiliary_dir = subdir_path / "auxiliary" / "dreams"
    auxiliary_files: list[str] = []
    auxiliary_lengths: list[int] = []
    problem_counts: Counter[str] = Counter()
    uploaded_paths: list[Path] = []
    valid_total = 0
    row_total = 0
    for part_idx, filename in enumerate(filenames):
        rows = pq.read_table(subdir_path / filename).to_pylist()
        valid, split_problem_counts = _validate_rows(
            rows,
            dformat=dformat,
            n_highest_peaks=n_highest_peaks,
        )
        embeddings = _compute_embeddings(
            rows,
            model=model,
            spec_preproc=spec_preproc,
            batch_size=batch_size,
        )
        spectrum_index = np.asarray(
            [int(row["spectrum_index"]) for row in rows],
            dtype=np.int64,
        )
        assert embeddings.shape[0] == len(rows)
        output_name = f"{split}-part-{part_idx:05d}.npz"
        output_path = auxiliary_dir / output_name
        np.savez_compressed(
            output_path,
            spectrum_index=spectrum_index,
            dreams_embedding=embeddings,
            dreams_embedding_valid=valid,
        )
        rel_name = f"auxiliary/dreams/{output_name}"
        auxiliary_files.append(rel_name)
        auxiliary_lengths.append(len(rows))
        uploaded_paths.append(output_path)
        valid_total += int(valid.sum())
        row_total += len(rows)
        problem_counts.update(split_problem_counts)
    return (
        auxiliary_files,
        auxiliary_lengths,
        valid_total,
        row_total - valid_total,
        dict(problem_counts),
        uploaded_paths,
    )


def _update_top_metadata(
    artifact_dir: Path,
    *,
    subdir: str,
    metadata: dict[str, Any],
) -> Path | None:
    metadata_path = artifact_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    top_metadata = json.loads(metadata_path.read_text())
    for dataset_metadata in top_metadata.get("datasets", {}).values():
        if dataset_metadata.get("subdir") == subdir:
            dataset_metadata["dreams_dim"] = metadata["dreams_dim"]
            dataset_metadata["dreams_auxiliary_available"] = True
            dataset_metadata["dreams_auxiliary_format"] = metadata[
                "dreams_auxiliary_format"
            ]
            dataset_metadata["dreams_checkpoint"] = metadata["dreams_checkpoint"]
            dataset_metadata["dreams_source_git_commit"] = metadata[
                "dreams_source_git_commit"
            ]
            dataset_metadata["dreams_valid_counts"] = metadata["dreams_valid_counts"]
            dataset_metadata["dreams_invalid_counts"] = metadata[
                "dreams_invalid_counts"
            ]
    metadata_path.write_text(json.dumps(top_metadata, indent=2, sort_keys=True))
    return metadata_path


def _build_subdir(
    *,
    artifact_dir: Path,
    subdir: str,
    model: torch.nn.Module,
    spec_preproc: Any,
    dformat: Any,
    batch_size: int,
    n_highest_peaks: int,
    checkpoint_source: dict[str, Any],
    dreams_source_git_commit: str | None,
) -> list[Path]:
    subdir = subdir.strip("/")
    subdir_path = artifact_dir / subdir
    metadata_path = subdir_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    auxiliary_dir = subdir_path / "auxiliary" / "dreams"
    if auxiliary_dir.exists():
        shutil.rmtree(auxiliary_dir)
    auxiliary_dir.mkdir(parents=True, exist_ok=True)

    dreams_files: dict[str, list[str]] = {}
    dreams_lengths: dict[str, list[int]] = {}
    dreams_valid_counts: dict[str, int] = {}
    dreams_invalid_counts: dict[str, int] = {}
    dreams_problem_counts: dict[str, dict[str, int]] = {}
    for split in _split_names(metadata):
        log.info("%s/%s: building DreaMS auxiliary", subdir, split)
        files, lengths, valid_count, invalid_count, problems, _paths = (
            _write_split_auxiliary(
                subdir_path=subdir_path,
                split=split,
                filenames=[str(name) for name in metadata[f"{split}_files"]],
                model=model,
                spec_preproc=spec_preproc,
                dformat=dformat,
                batch_size=batch_size,
                n_highest_peaks=n_highest_peaks,
            )
        )
        dreams_files[split] = files
        dreams_lengths[split] = lengths
        dreams_valid_counts[split] = valid_count
        dreams_invalid_counts[split] = invalid_count
        dreams_problem_counts[split] = problems

    first_embedding = np.load(subdir_path / dreams_files[_split_names(metadata)[0]][0])[
        "dreams_embedding"
    ]
    metadata.update(
        {
            "dreams_dim": int(first_embedding.shape[1]),
            "dreams_auxiliary_available": True,
            "dreams_auxiliary_format": "dreams_npz_v1",
            "dreams_auxiliary_files": dreams_files,
            "dreams_auxiliary_lengths": dreams_lengths,
            "dreams_valid_counts": dreams_valid_counts,
            "dreams_invalid_counts": dreams_invalid_counts,
            "dreams_validation_problem_counts": dreams_problem_counts,
            "dreams_embedding_model": "embedding_model.ckpt",
            "dreams_checkpoint": checkpoint_source,
            "dreams_source_git_commit": dreams_source_git_commit,
            "dreams_data_format": "DataFormatA",
            "dreams_n_highest_peaks": int(n_highest_peaks),
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    local_upload_paths = [
        subdir_path / path for paths in dreams_files.values() for path in paths
    ]
    local_upload_paths.append(metadata_path)
    top_metadata_path = _update_top_metadata(
        artifact_dir,
        subdir=subdir,
        metadata=metadata,
    )
    if top_metadata_path is not None:
        local_upload_paths.append(top_metadata_path)
    return local_upload_paths


def _upload_paths(
    *,
    artifact_dir: Path,
    repo_id: str,
    revision: str,
    private: bool,
    paths: list[Path],
) -> None:
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=private)
    for path in paths:
        api.upload_file(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            path_or_fileobj=str(path),
            path_in_repo=str(path.relative_to(artifact_dir)),
        )


def build_murcko_dreams_auxiliary(
    *,
    artifact_dir: Path,
    dreams_root: Path = DEFAULT_DREAMS_ROOT,
    checkpoint: Path = DEFAULT_DREAMS_CHECKPOINT,
    subdirs: list[str] | None = None,
    n_highest_peaks: int = DEFAULT_N_HIGHEST_PEAKS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    upload: bool = False,
    hf_repo_id: str = "",
    hf_revision: str = "main",
    hf_private: bool = False,
) -> list[Path]:
    _install_dreams_path(dreams_root)
    artifact_dir = artifact_dir.expanduser().resolve()
    checkpoint = checkpoint.expanduser().resolve()
    checkpoint_source = file_source_manifest(checkpoint)
    dreams_source_git_commit = _git_head_commit(dreams_root)
    subdirs = subdirs if subdirs is not None else list(DEFAULT_SUBDIRS)
    model, spec_preproc, dformat = _load_model(
        checkpoint,
        device=device,
        n_highest_peaks=n_highest_peaks,
    )
    upload_paths: list[Path] = []
    for subdir in subdirs:
        upload_paths.extend(
            _build_subdir(
                artifact_dir=artifact_dir,
                subdir=subdir,
                model=model,
                spec_preproc=spec_preproc,
                dformat=dformat,
                batch_size=batch_size,
                n_highest_peaks=n_highest_peaks,
                checkpoint_source=checkpoint_source,
                dreams_source_git_commit=dreams_source_git_commit,
            )
        )
    if upload:
        if not hf_repo_id:
            raise ValueError("--hf-repo-id is required with --upload")
        _upload_paths(
            artifact_dir=artifact_dir,
            repo_id=hf_repo_id,
            revision=hf_revision,
            private=hf_private,
            paths=upload_paths,
        )
    return upload_paths


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    build_murcko_dreams_auxiliary(
        artifact_dir=args.artifact_dir,
        dreams_root=args.dreams_root,
        checkpoint=args.checkpoint,
        subdirs=args.subdir,
        n_highest_peaks=args.n_highest_peaks,
        batch_size=args.batch_size,
        device=args.device,
        upload=args.upload,
        hf_repo_id=args.hf_repo_id,
        hf_revision=args.hf_revision,
        hf_private=args.hf_private,
    )


if __name__ == "__main__":
    main()
