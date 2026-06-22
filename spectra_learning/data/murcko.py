"""Build Murcko-split raw-MGF Parquet datasets and upload them to Hugging Face."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, NamedTuple, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from huggingface_hub import HfApi, snapshot_download
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from spectra_learning.data.gems.collate import GemsBatchCollator
from spectra_learning.data.gems.conversion import format_batch
from spectra_learning.data.loading import (
    loader_sampler,
    local_batch_size,
    subset_for_max_samples,
)
from spectra_learning.data.repositories import MSMS_EVALUATION_HF_REPO
from spectra_learning.data.massspec_targets import (
    MACCS_FINGERPRINT_BITS,
    MORGAN_PROBE_FINGERPRINT_BITS,
    MORGAN_PROBE_FINGERPRINT_RADIUS,
    REGRESSION_TARGET_KEYS,
)
from spectra_learning.data.mgf import _to_float, iter_mgf
from spectra_learning.data.spectra import (
    DEFAULT_GROUPED_PEAK_ISOTOPE_CHARGES,
    DEFAULT_GROUPED_PEAK_SHOULDER_DA,
    DEFAULT_PEAK_FILTERING,
    NUM_PEAKS_INPUT,
)

log = logging.getLogger(__name__)
_MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_PROBE_FINGERPRINT_RADIUS,
    fpSize=MORGAN_PROBE_FINGERPRINT_BITS,
)

DEFAULT_NIST_MGF_URI = "gs://main-novogaia-bucket/MS/Datasets_with_structure/nist20/hr_msms_nist.mgf"
DEFAULT_LOCAL_NIST_MGF_PATH = Path("data/raw/hr_msms_nist.mgf")
DEFAULT_MCEBIO_MGF_PATH = Path(
    "data/massive_msv000094528/source/20240411_mcebio_library_pos_all_lib_MS2.mgf"
)
NIST_MURCKO_METADATA_VERSION = 2
NIST_MURCKO_HF_REPO = MSMS_EVALUATION_HF_REPO
NIST_DISJOINT_PROBE_RETRIEVAL_HF_REPO = (
    "wchen99998/msms_nist_disjoint_probe_retrieval_20260622"
)
NIST_MURCKO_PREPARED_SUBDIR = "nist_murcko_probe"
MCEBIO_MURCKO_PREPARED_SUBDIR = "mcebio_murcko_probe"
NIST_DISJOINT_ONLINE_PROBE_SUBDIR = "nist_100k_online_probe"
NIST_DISJOINT_RETRIEVAL_POOL_SUBDIR = "nist_retrieval_pool"
NIST_10PPM_RETRIEVAL_SUBDIR = "nist_same_inchi14_10ppm_retrieval"
NIST_MCES_RETRIEVAL_SUBDIR = "nist_mces_analog_retrieval"
NIST_MURCKO_ARTIFACT_FORMAT = "nist_murcko_parquet_v2"
NIST_DISJOINT_PROBE_RETRIEVAL_ARTIFACT_FORMAT = (
    "nist_disjoint_probe_retrieval_collection_v1"
)
RAW_SUBDIR = "raw"
SPLITS = ("train", "val", "test")
STANDALONE_SPLIT = "all"


def _write_nist_disjoint_probe_retrieval_readme(
    path: Path,
    *,
    metadata: Mapping[str, Any],
) -> None:
    """Write the Hugging Face dataset card for the fixed benchmark artifact."""
    online = metadata["online_probe"]
    retrieval = metadata["retrieval_pool"]
    same_inchi = metadata["same_inchi14_10ppm"]
    mces = metadata["mces_analog"]
    disjointness = metadata["disjointness"]
    text = f"""---
pretty_name: NIST MS/MS Murcko-Disjoint Probe and Retrieval Benchmark
task_categories:
- feature-extraction
tags:
- mass-spectrometry
- msms
- nist
- murcko
- molecular-retrieval
---

# NIST MS/MS Murcko-Disjoint Probe and Retrieval Benchmark

This dataset packages one fixed online-probe task and two fixed retrieval-task
pair tables derived from the raw NIST high-resolution MS/MS MGF staged in
`{metadata["source_raw_file"]}`.

## Layout

- `{online["subdir"]}/`: train/val/test Parquets built from an exact
  {online["selected_spectra_before_split_processing"]:,}-spectrum sample without
  replacement, then processed by the existing online-probe Murcko split logic.
  Produced rows: train={online["train_size"]:,}, val={online["val_size"]:,},
  test={online["test_size"]:,}.
- `{retrieval["subdir"]}/all.parquet`: the retrieval spectrum pool after
  excluding every Murcko histogram key selected for the online probe.
  Rows: {retrieval["all_size"]:,}.
- `{same_inchi["subdir"]}/pairs.parquet`: balanced 10 ppm same-InChI14 AUROC
  pairs. Positive pairs: {same_inchi["positive_pairs"]:,}. Negative pairs:
  {same_inchi["negative_pairs"]:,}.
- `{mces["subdir"]}/pairs.parquet`: MCES analog-search pairs with precomputed
  MCES distances and binary labels for thresholds 0 through
  {mces["mces_threshold"]}. Pairs: {mces["num_pairs"]:,}.

## Disjointness

The online-probe and retrieval-pool spectra are disjoint by Murcko histogram,
not merely by row id. Selected online-probe Murcko histogram keys:
{disjointness["online_probe_selected_murcko_hist_keys"]:,}. Retrieval-pool
Murcko histogram keys: {disjointness["retrieval_pool_murcko_hist_keys"]:,}.
The recorded overlap count is
{disjointness["online_probe_retrieval_murcko_hist_overlap"]}.

See `metadata.json` and each subdirectory `metadata.json` for the full schema,
sampling parameters, and pair-generation metadata. Downstream use of the staged
NIST source data should comply with the original NIST terms.
"""
    path.write_text(text, encoding="utf-8")
DEFAULT_NIST_ALLOWED_ADDUCTS = ("[M+H]+",)
DEFAULT_NIST_SPLIT_SIZE_CAPS = {"train": 100_000, "val": 25_000, "test": 25_000}
DEFAULT_ONLINE_PROBE_SAMPLE_SIZE = 100_000
DEFAULT_10PPM_RETRIEVAL_PAIRS_PER_CLASS = 100_000
DEFAULT_10PPM_RETRIEVAL_PPM = 10.0
DEFAULT_MCES_RETRIEVAL_PAIRS = 1_000
DEFAULT_MCES_RETRIEVAL_BIN_SIZE = 0.025
DEFAULT_RETRIEVAL_ADDUCT = "[M+H]+"
MCES_THRESHOLD = 7
MCES_REPORTED_THRESHOLDS = tuple(range(MCES_THRESHOLD + 1))
SAME_INCHI_10PPM_PAIR_SAMPLING = "same_inchi14_10ppm_balanced_binary_pairs_v1"
MCES_ANALOG_PAIR_SAMPLING = "morgan_tanimoto_balanced_mces_pairs_v1"
SPECTRAL_LSH_MZ_BIN_WIDTH = 0.05
SPECTRAL_LSH_INTENSITY_BINS = 10
SPECTRAL_LSH_NUM_HASHES = 32
SPECTRAL_LSH_BAND_SIZE = 4
_SPECTRAL_LSH_PRIME = np.uint64(4_294_967_311)
_SPECTRAL_LSH_RNG = np.random.default_rng(1729)
_SPECTRAL_LSH_A = _SPECTRAL_LSH_RNG.integers(
    1,
    int(_SPECTRAL_LSH_PRIME),
    size=SPECTRAL_LSH_NUM_HASHES,
    dtype=np.uint64,
)
_SPECTRAL_LSH_B = _SPECTRAL_LSH_RNG.integers(
    0,
    int(_SPECTRAL_LSH_PRIME),
    size=SPECTRAL_LSH_NUM_HASHES,
    dtype=np.uint64,
)

CHEMICAL_PROPERTY_COLUMNS = (
    *REGRESSION_TARGET_KEYS,
    "tpsa",
    "num_hbd",
    "num_hba",
    "num_rotatable_bonds",
    "fraction_csp3",
    "formal_charge",
    "num_aromatic_rings",
)

METADATA_COLUMNS = (
    "name",
    "title",
    "pepmass",
    "charge",
    "precursortype",
    "adduct",
    "collisionenergy",
    "collision energy",
    "instrument",
    "instrumenttype",
    "instrument_type",
    "ionmode",
    "spectrumtype",
    "formula",
    "exactmass",
    "inchikey",
    "smiles",
    "rtinseconds",
    "db",
    "db_ref",
    "splash",
    "comment",
    "peakannotations",
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    source: str
    subdir: str


@dataclass(frozen=True)
class FirstPassRow:
    spectrum_index: int
    canonical_smiles: str
    murcko_hist_key: str
    murcko_hist_json: str


@dataclass(frozen=True)
class FullRow:
    row: dict[str, Any]
    morgan: np.ndarray


@dataclass(frozen=True)
class RetrievalPoolRow:
    """Minimal row metadata needed to write fixed retrieval pair artifacts."""

    row_index: int
    spectrum_index: int
    canonical_smiles: str
    precursor_mz: float
    adduct: str
    inchi14: str
    murcko_hist_key: str


class MurckoFluorineData(NamedTuple):
    metadata: dict[str, Any]
    root: Path
    batch_size: int
    num_peaks: int
    max_precursor_mz: float
    min_peak_intensity: float
    peak_drop_min_intensity: float
    peak_filtering: str
    grouped_peak_shoulder_da: float
    grouped_peak_isotope_charges: tuple[int, ...]
    peak_ordering: str
    precursor_peak_exclusion_window_da: float


def _probe_metadata_valid(
    output_dir: Path,
    expected_version: int,
    max_precursor_mz: float,
    expected_metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text())
    if int(metadata.get("metadata_version", 0)) != expected_version:
        _raise_invalid_murcko_artifact(output_dir, "metadata_version mismatch")
    if float(metadata.get("max_precursor_mz", float("inf"))) != max_precursor_mz:
        _raise_invalid_murcko_artifact(output_dir, "max_precursor_mz mismatch")
    if expected_metadata is not None:
        for key, value in expected_metadata.items():
            if metadata.get(key) != value:
                _raise_invalid_murcko_artifact(output_dir, f"{key} mismatch")
    storage_format = str(metadata.get("storage_format", "native"))
    for split in ("train", "val", "test"):
        filenames = metadata.get(f"{split}_files", [])
        if not filenames:
            _raise_invalid_murcko_artifact(
                output_dir,
                f"missing {split}_files metadata",
            )
        if storage_format == "parquet":
            if not all(
                (output_dir / name).exists()
                for name in filenames
            ):
                _raise_invalid_murcko_artifact(output_dir, f"missing {split} files")
        elif not all(
            (output_dir / split / name).exists()
            for name in filenames
        ):
            _raise_invalid_murcko_artifact(output_dir, f"missing {split} files")
    return metadata


def _raise_invalid_murcko_artifact(output_dir: Path, reason: str) -> None:
    raise ValueError(
        f"Invalid Murcko probe artifact in {output_dir}: {reason}. "
        "Delete the artifact directory and rebuild or download it again."
    )


def _coordinate_distributed_download(distributed_world_size: int) -> bool:
    return (
        distributed_world_size > 1
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    )


def _snapshot_download_local_rank_zero(
    *,
    repo_id: str,
    repo_type: str,
    revision: str,
    local_dir: Path,
    allow_patterns: list[str],
    distributed_world_size: int,
    distributed_local_rank: int,
) -> None:
    coordinated = _coordinate_distributed_download(distributed_world_size)
    if not coordinated or distributed_local_rank == 0:
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            local_dir=local_dir,
            allow_patterns=allow_patterns,
        )
    if coordinated:
        torch.distributed.barrier()


def _auxiliary_files_available(
    output_dir: Path,
    metadata: dict[str, Any],
    auxiliary_name: str,
) -> bool:
    return bool(metadata.get(f"{auxiliary_name}_auxiliary_available", False)) and all(
        (output_dir / name).exists()
        for names in metadata.get(f"{auxiliary_name}_auxiliary_files", {}).values()
        for name in names
    )


def ensure_nist_murcko_probe_downloaded(
    output_dir: Path,
    *,
    max_precursor_mz: float,
    repo_id: str = NIST_MURCKO_HF_REPO,
    revision: str = "main",
    subdir: str = NIST_MURCKO_PREPARED_SUBDIR,
    include_morgan: bool = False,
    include_dreams: bool = False,
    distributed_world_size: int = 1,
    distributed_rank: int = 0,
    distributed_local_rank: int | None = None,
) -> dict[str, Any]:
    distributed_local_rank = (
        distributed_rank if distributed_local_rank is None else distributed_local_rank
    )
    cached = _probe_metadata_valid(
        output_dir,
        NIST_MURCKO_METADATA_VERSION,
        max_precursor_mz,
        expected_metadata={"artifact_format": NIST_MURCKO_ARTIFACT_FORMAT},
    )
    if cached is not None:
        if include_morgan and not _auxiliary_files_available(
            output_dir,
            cached,
            "morgan",
        ):
            _raise_invalid_murcko_artifact(
                output_dir,
                "missing Morgan auxiliary files",
            )
        if include_dreams and not _auxiliary_files_available(
            output_dir,
            cached,
            "dreams",
        ):
            _raise_invalid_murcko_artifact(
                output_dir,
                "missing DreaMS auxiliary files",
            )
        if _coordinate_distributed_download(distributed_world_size):
            torch.distributed.barrier()
        return cached
    subdir = subdir.strip("/")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    allow_patterns = [
        f"{subdir}/metadata.json",
        f"{subdir}/train.parquet",
        f"{subdir}/val.parquet",
        f"{subdir}/test.parquet",
    ]
    if include_morgan:
        allow_patterns.append(f"{subdir}/auxiliary/morgan/*")
    if include_dreams:
        allow_patterns.append(f"{subdir}/auxiliary/dreams/*")
    _snapshot_download_local_rank_zero(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=output_dir.parent,
        allow_patterns=allow_patterns,
        distributed_world_size=distributed_world_size,
        distributed_local_rank=distributed_local_rank,
    )
    metadata = _probe_metadata_valid(
        output_dir,
        NIST_MURCKO_METADATA_VERSION,
        max_precursor_mz,
        expected_metadata={"artifact_format": NIST_MURCKO_ARTIFACT_FORMAT},
    )
    if metadata is None:
        raise FileNotFoundError(f"Invalid NIST Murcko probe artifact in {output_dir}")
    if include_morgan and not _auxiliary_files_available(output_dir, metadata, "morgan"):
        raise FileNotFoundError(f"Missing NIST Murcko Morgan auxiliary files in {output_dir}")
    if include_dreams and not _auxiliary_files_available(output_dir, metadata, "dreams"):
        raise FileNotFoundError(f"Missing NIST Murcko DreaMS auxiliary files in {output_dir}")
    return metadata


def ensure_mcebio_murcko_probe_downloaded(
    cache_dir: Path,
    *,
    repo_id: str = NIST_MURCKO_HF_REPO,
    revision: str = "main",
    subdir: str = MCEBIO_MURCKO_PREPARED_SUBDIR,
    include_morgan: bool = False,
    include_dreams: bool = False,
    distributed_world_size: int = 1,
    distributed_rank: int = 0,
    distributed_local_rank: int | None = None,
) -> dict[str, Any]:
    distributed_local_rank = (
        distributed_rank if distributed_local_rank is None else distributed_local_rank
    )
    subdir = subdir.strip("/")
    cache_dir.mkdir(parents=True, exist_ok=True)
    allow_patterns = [
        f"{subdir}/metadata.json",
        f"{subdir}/all.parquet",
    ]
    if include_morgan:
        allow_patterns.append(f"{subdir}/auxiliary/morgan/*")
    if include_dreams:
        allow_patterns.append(f"{subdir}/auxiliary/dreams/*")
    cached = _read_murcko_subdir_metadata(
        cache_dir,
        subdir,
        required_splits=("all",),
    )
    needs_download = cached is None or (
        include_morgan
        and not _murcko_subdir_auxiliary_available(
            cache_dir,
            subdir,
            cached,
            "morgan",
        )
    ) or (
        include_dreams
        and not _murcko_subdir_auxiliary_available(
            cache_dir,
            subdir,
            cached,
            "dreams",
        )
    )
    if needs_download:
        _snapshot_download_local_rank_zero(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=cache_dir,
            allow_patterns=allow_patterns,
            distributed_world_size=distributed_world_size,
            distributed_local_rank=distributed_local_rank,
        )
    elif _coordinate_distributed_download(distributed_world_size):
        torch.distributed.barrier()
    metadata = cast(
        dict[str, Any],
        _read_murcko_subdir_metadata(
            cache_dir,
            subdir,
            required_splits=("all",),
        ),
    )
    if include_morgan and not _murcko_subdir_auxiliary_available(
        cache_dir,
        subdir,
        metadata,
        "morgan",
    ):
        raise FileNotFoundError(f"Missing MCEBIO Morgan auxiliary files in {cache_dir / subdir}")
    if include_dreams and not _murcko_subdir_auxiliary_available(
        cache_dir,
        subdir,
        metadata,
        "dreams",
    ):
        raise FileNotFoundError(f"Missing MCEBIO DreaMS auxiliary files in {cache_dir / subdir}")
    return metadata


def _read_murcko_subdir_metadata(
    cache_dir: Path,
    subdir: str,
    *,
    required_splits: tuple[str, ...],
) -> dict[str, Any] | None:
    metadata_path = cache_dir / subdir / "metadata.json"
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text())
    for split in required_splits:
        filenames = metadata.get(f"{split}_files", [])
        if not filenames:
            _raise_invalid_murcko_artifact(
                cache_dir / subdir,
                f"missing {split}_files metadata",
            )
        for filename in filenames:
            if not (cache_dir / subdir / filename).exists():
                _raise_invalid_murcko_artifact(
                    cache_dir / subdir,
                    f"missing {filename}",
                )
    return metadata


def _murcko_fluorine_split_metadata(
    source_metadata: dict[str, Any],
    *,
    subdir: str,
    source_split: str,
    target_split: str,
    include_morgan: bool,
    include_dreams: bool,
) -> dict[str, Any]:
    metadata = {
        f"{target_split}_files": [
            f"{subdir}/{filename}" for filename in source_metadata[f"{source_split}_files"]
        ],
        f"{target_split}_lengths": [
            int(value) for value in source_metadata[f"{source_split}_lengths"]
        ],
        f"{target_split}_size": int(source_metadata[f"{source_split}_size"]),
        f"{target_split}_positive": int(source_metadata.get(f"{source_split}_positive", 0)),
    }
    morgan_files = source_metadata.get("morgan_auxiliary_files", {}).get(source_split, [])
    if include_morgan and morgan_files:
        metadata[f"{target_split}_morgan_files"] = [
            f"{subdir}/{filename}" for filename in morgan_files
        ]
        metadata[f"{target_split}_morgan_lengths"] = [
            int(value)
            for value in source_metadata.get("morgan_auxiliary_lengths", {}).get(
                source_split,
                [],
            )
        ]
    dreams_files = source_metadata.get("dreams_auxiliary_files", {}).get(source_split, [])
    if include_dreams and dreams_files:
        metadata[f"{target_split}_dreams_files"] = [
            f"{subdir}/{filename}" for filename in dreams_files
        ]
        metadata[f"{target_split}_dreams_lengths"] = [
            int(value)
            for value in source_metadata.get("dreams_auxiliary_lengths", {}).get(
                source_split,
                [],
            )
        ]
    return metadata


def _murcko_subdir_auxiliary_available(
    cache_dir: Path,
    subdir: str,
    metadata: dict[str, Any] | None,
    auxiliary_name: str,
) -> bool:
    if metadata is None:
        return False
    if _auxiliary_files_available(cache_dir / subdir, metadata, auxiliary_name):
        return True
    _raise_invalid_murcko_artifact(
        cache_dir / subdir,
        f"missing {auxiliary_name} auxiliary files",
    )


def _merge_vocabularies(*vocabs: dict[str, int]) -> dict[str, int]:
    values = sorted({value for vocab in vocabs for value in vocab})
    return {value: idx for idx, value in enumerate(values)}


def ensure_murcko_fluorine_data_downloaded(
    cache_dir: Path,
    *,
    repo_id: str = NIST_MURCKO_HF_REPO,
    revision: str = "main",
    train_subdir: str = NIST_MURCKO_PREPARED_SUBDIR,
    test_subdir: str = MCEBIO_MURCKO_PREPARED_SUBDIR,
    include_morgan: bool = False,
    include_dreams: bool = False,
    distributed_world_size: int = 1,
    distributed_rank: int = 0,
    distributed_local_rank: int | None = None,
) -> dict[str, Any]:
    distributed_local_rank = (
        distributed_rank if distributed_local_rank is None else distributed_local_rank
    )
    train_subdir = train_subdir.strip("/")
    test_subdir = test_subdir.strip("/")
    cache_dir.mkdir(parents=True, exist_ok=True)
    allow_patterns = [
        f"{train_subdir}/metadata.json",
        f"{train_subdir}/train.parquet",
        f"{train_subdir}/val.parquet",
        f"{test_subdir}/metadata.json",
        f"{test_subdir}/all.parquet",
    ]
    if include_morgan:
        allow_patterns.extend(
            [
                f"{train_subdir}/auxiliary/morgan/*",
                f"{test_subdir}/auxiliary/morgan/*",
            ]
        )
    if include_dreams:
        allow_patterns.extend(
            [
                f"{train_subdir}/auxiliary/dreams/*",
                f"{test_subdir}/auxiliary/dreams/*",
            ]
        )
    train_cached = _read_murcko_subdir_metadata(
        cache_dir,
        train_subdir,
        required_splits=("train", "val"),
    )
    test_cached = _read_murcko_subdir_metadata(
        cache_dir,
        test_subdir,
        required_splits=("all",),
    )
    needs_download = train_cached is None or test_cached is None or (
        include_morgan
        and (
            not _murcko_subdir_auxiliary_available(
                cache_dir,
                train_subdir,
                train_cached,
                "morgan",
            )
            or not _murcko_subdir_auxiliary_available(
                cache_dir,
                test_subdir,
                test_cached,
                "morgan",
            )
        )
    ) or (
        include_dreams
        and (
            not _murcko_subdir_auxiliary_available(
                cache_dir,
                train_subdir,
                train_cached,
                "dreams",
            )
            or not _murcko_subdir_auxiliary_available(
                cache_dir,
                test_subdir,
                test_cached,
                "dreams",
            )
        )
    )
    if needs_download:
        _snapshot_download_local_rank_zero(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=cache_dir,
            allow_patterns=allow_patterns,
            distributed_world_size=distributed_world_size,
            distributed_local_rank=distributed_local_rank,
        )
    elif _coordinate_distributed_download(distributed_world_size):
        torch.distributed.barrier()
    train_metadata = cast(
        dict[str, Any],
        _read_murcko_subdir_metadata(
            cache_dir,
            train_subdir,
            required_splits=("train", "val"),
        ),
    )
    test_metadata = cast(
        dict[str, Any],
        _read_murcko_subdir_metadata(
            cache_dir,
            test_subdir,
            required_splits=("all",),
        ),
    )
    if include_morgan:
        if not _murcko_subdir_auxiliary_available(
            cache_dir,
            train_subdir,
            train_metadata,
            "morgan",
        ):
            raise FileNotFoundError(f"Missing NIST Murcko Morgan auxiliary files in {cache_dir / train_subdir}")
        if not _murcko_subdir_auxiliary_available(
            cache_dir,
            test_subdir,
            test_metadata,
            "morgan",
        ):
            raise FileNotFoundError(f"Missing MCEBIO Morgan auxiliary files in {cache_dir / test_subdir}")
    if include_dreams:
        if not _murcko_subdir_auxiliary_available(
            cache_dir,
            train_subdir,
            train_metadata,
            "dreams",
        ):
            raise FileNotFoundError(f"Missing NIST Murcko DreaMS auxiliary files in {cache_dir / train_subdir}")
        if not _murcko_subdir_auxiliary_available(
            cache_dir,
            test_subdir,
            test_metadata,
            "dreams",
        ):
            raise FileNotFoundError(f"Missing MCEBIO DreaMS auxiliary files in {cache_dir / test_subdir}")
    metadata: dict[str, Any] = {
        "metadata_version": 1,
        "storage_format": "parquet",
        "repo_id": repo_id,
        "revision": revision,
        "train_subdir": train_subdir,
        "test_subdir": test_subdir,
        "adduct_vocab": _merge_vocabularies(
            train_metadata.get("adduct_vocab", {"unknown": 0}),
            test_metadata.get("adduct_vocab", {"unknown": 0}),
        ),
        "instrument_type_vocab": _merge_vocabularies(
            train_metadata.get("instrument_type_vocab", {"unknown": 0}),
            test_metadata.get("instrument_type_vocab", {"unknown": 0}),
        ),
        "probe_maccs_bits": int(train_metadata.get("probe_maccs_bits", MACCS_FINGERPRINT_BITS)),
        "probe_morgan_bits": int(train_metadata.get("probe_morgan_bits", MORGAN_PROBE_FINGERPRINT_BITS)) if include_morgan else 0,
        "probe_morgan_radius": int(train_metadata.get("probe_morgan_radius", MORGAN_PROBE_FINGERPRINT_RADIUS)),
        "morgan_auxiliary_available": bool(
            include_morgan
            and train_metadata.get("morgan_auxiliary_available", False)
            and test_metadata.get("morgan_auxiliary_available", False)
        ),
        "dreams_dim": int(train_metadata.get("dreams_dim", 0)),
        "dreams_auxiliary_available": bool(
            include_dreams
            and train_metadata.get("dreams_auxiliary_available", False)
            and test_metadata.get("dreams_auxiliary_available", False)
        ),
    }
    metadata.update(
        _murcko_fluorine_split_metadata(
            train_metadata,
            subdir=train_subdir,
            source_split="train",
            target_split="train",
            include_morgan=include_morgan,
            include_dreams=include_dreams,
        )
    )
    metadata.update(
        _murcko_fluorine_split_metadata(
            train_metadata,
            subdir=train_subdir,
            source_split="val",
            target_split="val",
            include_morgan=include_morgan,
            include_dreams=include_dreams,
        )
    )
    metadata.update(
        _murcko_fluorine_split_metadata(
            test_metadata,
            subdir=test_subdir,
            source_split="all",
            target_split="test",
            include_morgan=include_morgan,
            include_dreams=include_dreams,
        )
    )
    return metadata


def build_murcko_fluorine_data(
    *,
    cache_dir: Path,
    batch_size: int,
    num_peaks: int,
    max_precursor_mz: float,
    min_peak_intensity: float,
    peak_drop_min_intensity: float,
    peak_ordering: str,
    precursor_peak_exclusion_window_da: float,
    peak_filtering: str = DEFAULT_PEAK_FILTERING,
    grouped_peak_shoulder_da: float = DEFAULT_GROUPED_PEAK_SHOULDER_DA,
    grouped_peak_isotope_charges: tuple[int, ...] = (
        DEFAULT_GROUPED_PEAK_ISOTOPE_CHARGES
    ),
    repo_id: str = NIST_MURCKO_HF_REPO,
    revision: str = "main",
    train_subdir: str = NIST_MURCKO_PREPARED_SUBDIR,
    test_subdir: str = MCEBIO_MURCKO_PREPARED_SUBDIR,
    include_dreams: bool = False,
    distributed_world_size: int = 1,
    distributed_rank: int = 0,
    distributed_local_rank: int | None = None,
) -> MurckoFluorineData:
    metadata = ensure_murcko_fluorine_data_downloaded(
        cache_dir,
        repo_id=repo_id,
        revision=revision,
        train_subdir=train_subdir,
        test_subdir=test_subdir,
        include_dreams=include_dreams,
        distributed_world_size=distributed_world_size,
        distributed_rank=distributed_rank,
        distributed_local_rank=distributed_local_rank,
    )
    return MurckoFluorineData(
        metadata=metadata,
        root=cache_dir,
        batch_size=batch_size,
        num_peaks=num_peaks,
        max_precursor_mz=max_precursor_mz,
        min_peak_intensity=min_peak_intensity,
        peak_drop_min_intensity=peak_drop_min_intensity,
        peak_filtering=peak_filtering,
        grouped_peak_shoulder_da=grouped_peak_shoulder_da,
        grouped_peak_isotope_charges=grouped_peak_isotope_charges,
        peak_ordering=peak_ordering,
        precursor_peak_exclusion_window_da=precursor_peak_exclusion_window_da,
    )


def _normalize_spectra_intensity(spectra: np.ndarray) -> np.ndarray:
    max_int = spectra[:, 1].max(axis=1, keepdims=True)
    np.divide(spectra[:, 1], np.maximum(max_int, 1e-8), out=spectra[:, 1])
    return spectra


def _spectra_from_peak_lists(
    mz_lists: list[list[float]],
    intensity_lists: list[list[float]],
) -> np.ndarray:
    spectra = np.zeros((len(mz_lists), 2, NUM_PEAKS_INPUT), dtype=np.float32)
    for i, (mz, intensity) in enumerate(zip(mz_lists, intensity_lists, strict=True)):
        n = min(len(mz), NUM_PEAKS_INPUT)
        spectra[i, 0, :n] = np.asarray(mz[:n], dtype=np.float32)
        spectra[i, 1, :n] = np.asarray(intensity[:n], dtype=np.float32)
    return _normalize_spectra_intensity(spectra)


class _MurckoFluorineParquetDataset(Dataset):
    def __init__(self, entries: list[dict[str, Any]]) -> None:
        self._entries = [
            {
                "path": Path(entry["path"]),
                "length": int(entry["length"]),
                "dreams_files": [Path(path) for path in entry.get("dreams_files", [])],
            }
            for entry in entries
        ]
        lengths = np.asarray([entry["length"] for entry in self._entries], dtype=np.int64)
        self._starts = np.zeros(len(lengths) + 1, dtype=np.int64)
        np.cumsum(lengths, out=self._starts[1:])
        self._arrays: list[dict[str, np.ndarray]] | None = None

    def __len__(self) -> int:
        return int(self._starts[-1])

    def _load_entry(self, entry: dict[str, Any]) -> dict[str, np.ndarray]:
        import pyarrow.parquet as pq

        table = pq.read_table(Path(entry["path"]))
        rows = table.to_pydict()
        arrays: dict[str, np.ndarray] = {
            "spectra": _spectra_from_peak_lists(
                rows["spectrum_mz"],
                rows["spectrum_intensity"],
            ),
            "precursor_mz_raw": np.asarray(rows["precursor_mz"], dtype=np.float32),
            "label": np.asarray(rows["has_fluorine"], dtype=np.float32),
        }
        if "dreams_embedding" in rows:
            arrays["dreams_embedding"] = np.asarray(
                rows["dreams_embedding"],
                dtype=np.float32,
            )
        dreams_files = entry.get("dreams_files", [])
        if dreams_files:
            dreams_payloads = [
                np.load(Path(path), allow_pickle=False) for path in dreams_files
            ]
            arrays["dreams_embedding"] = np.concatenate(
                [
                    payload["dreams_embedding"].astype(np.float32)
                    for payload in dreams_payloads
                ],
                axis=0,
            )
            arrays["dreams_embedding_valid"] = np.concatenate(
                [
                    payload["dreams_embedding_valid"].astype(bool)
                    for payload in dreams_payloads
                ],
                axis=0,
            )
            spectrum_index = np.concatenate(
                [
                    payload["spectrum_index"].astype(np.int64)
                    for payload in dreams_payloads
                ],
                axis=0,
            )
            assert arrays["dreams_embedding"].shape[0] == len(rows["precursor_mz"])
            if "spectrum_index" in rows:
                assert np.array_equal(
                    spectrum_index,
                    np.asarray(rows["spectrum_index"], dtype=np.int64),
                )
        return arrays

    def _ensure_arrays(self) -> list[dict[str, np.ndarray]]:
        if self._arrays is not None:
            return self._arrays
        self._arrays = [self._load_entry(entry) for entry in self._entries]
        return self._arrays

    def __getitem__(self, index: int) -> dict[str, Any]:
        arrays_by_entry = self._ensure_arrays()
        entry_idx = int(np.searchsorted(self._starts, index, side="right") - 1)
        local_idx = index - int(self._starts[entry_idx])
        arrays = arrays_by_entry[entry_idx]
        sample = {
            "spectra": arrays["spectra"][local_idx].copy(),
            "precursor_mz_raw": np.float32(arrays["precursor_mz_raw"][local_idx]),
            "label": np.float32(arrays["label"][local_idx]),
            "row_idx": np.int64(index),
        }
        if "dreams_embedding" in arrays:
            sample["dreams_embedding"] = arrays["dreams_embedding"][local_idx].copy()
        if "dreams_embedding_valid" in arrays:
            sample["dreams_embedding_valid"] = bool(
                arrays["dreams_embedding_valid"][local_idx]
            )
        return sample


class _MurckoFluorineCollator:
    def __init__(
        self,
        *,
        num_peaks: int,
        max_precursor_mz: float,
        min_peak_intensity: float,
        peak_drop_min_intensity: float,
        peak_ordering: str,
        precursor_peak_exclusion_window_da: float,
        dreams_only: bool,
        output_format: str,
        peak_filtering: str = DEFAULT_PEAK_FILTERING,
        grouped_peak_shoulder_da: float = DEFAULT_GROUPED_PEAK_SHOULDER_DA,
        grouped_peak_isotope_charges: tuple[int, ...] = (
            DEFAULT_GROUPED_PEAK_ISOTOPE_CHARGES
        ),
    ) -> None:
        self.dreams_only = dreams_only
        self.output_format = output_format
        self.peak_collator = GemsBatchCollator(
            augment=False,
            num_target_blocks=0,
            context_fraction=0.0,
            target_fraction=0.0,
            block_min_len=1,
            num_peaks=num_peaks,
            max_precursor_mz=max_precursor_mz,
            min_peak_intensity=min_peak_intensity,
            peak_drop_min_intensity=peak_drop_min_intensity,
            peak_ordering=peak_ordering,
            precursor_peak_exclusion_window_da=precursor_peak_exclusion_window_da,
            peak_filtering=peak_filtering,
            grouped_peak_shoulder_da=grouped_peak_shoulder_da,
            grouped_peak_isotope_charges=grouped_peak_isotope_charges,
            output_format="torch",
        )

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        if self.dreams_only:
            batch = {
                "dreams_embedding": torch.stack(
                    [
                        torch.as_tensor(sample["dreams_embedding"], dtype=torch.float32)
                        for sample in samples
                    ],
                    dim=0,
                ),
                "label": torch.as_tensor(
                    [sample["label"] for sample in samples],
                    dtype=torch.float32,
                ),
                "row_idx": torch.as_tensor(
                    [sample["row_idx"] for sample in samples],
                    dtype=torch.long,
                ),
            }
            if "dreams_embedding_valid" in samples[0]:
                batch["dreams_embedding_valid"] = torch.as_tensor(
                    [sample["dreams_embedding_valid"] for sample in samples],
                    dtype=torch.bool,
                )
            return format_batch(batch, self.output_format)
        batch = self.peak_collator(samples)
        if "dreams_embedding" in samples[0]:
            batch["dreams_embedding"] = torch.stack(
                [
                    torch.as_tensor(sample["dreams_embedding"], dtype=torch.float32)
                    for sample in samples
                ],
                dim=0,
            )
        if "dreams_embedding_valid" in samples[0]:
            batch["dreams_embedding_valid"] = torch.as_tensor(
                [sample["dreams_embedding_valid"] for sample in samples],
                dtype=torch.bool,
            )
        batch["label"] = torch.as_tensor(
            [sample["label"] for sample in samples],
            dtype=torch.float32,
        )
        batch["row_idx"] = torch.as_tensor(
            [sample["row_idx"] for sample in samples],
            dtype=torch.long,
        )
        return format_batch(batch, self.output_format)


def build_murcko_fluorine_loader(
    data: MurckoFluorineData,
    split: str,
    *,
    shuffle: bool,
    seed: int,
    max_samples: int | None,
    dreams_only: bool = False,
    drop_last: bool = False,
    distributed_world_size: int = 1,
    distributed_rank: int = 0,
    num_workers: int = 0,
    output_format: str = "torch",
) -> DataLoader:
    split_files = data.metadata[f"{split}_files"]
    split_lengths = data.metadata[f"{split}_lengths"]
    dreams_names = [
        data.root / name for name in data.metadata.get(f"{split}_dreams_files", [])
    ]
    split_dreams_files = [dreams_names] if dreams_names else [[] for _ in split_files]
    dataset: Dataset = _MurckoFluorineParquetDataset(
        [
            {"path": data.root / path, "length": length, "dreams_files": dreams_files}
            for path, length, dreams_files in zip(
                split_files,
                split_lengths,
                split_dreams_files,
                strict=True,
            )
        ]
    )
    dataset, shuffle = subset_for_max_samples(
        dataset,
        max_samples=max_samples,
        shuffle=shuffle,
        seed=seed,
    )
    sampler, loader_shuffle = loader_sampler(
        dataset,
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
        distributed_world_size=distributed_world_size,
        distributed_rank=distributed_rank,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": local_batch_size(data.batch_size, distributed_world_size),
        "shuffle": loader_shuffle,
        "sampler": sampler,
        "drop_last": drop_last,
        "num_workers": num_workers,
        "collate_fn": _MurckoFluorineCollator(
            num_peaks=data.num_peaks,
            max_precursor_mz=data.max_precursor_mz,
            min_peak_intensity=data.min_peak_intensity,
            peak_drop_min_intensity=data.peak_drop_min_intensity,
            peak_ordering=data.peak_ordering,
            precursor_peak_exclusion_window_da=data.precursor_peak_exclusion_window_da,
            peak_filtering=data.peak_filtering,
            grouped_peak_shoulder_da=data.grouped_peak_shoulder_da,
            grouped_peak_isotope_charges=data.grouped_peak_isotope_charges,
            dreams_only=dreams_only,
            output_format=output_format,
        ),
        "generator": generator,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    return DataLoader(**loader_kwargs)


def _multirings(mol: Chem.Mol) -> list[set[int]]:
    bond_rings = [set(ring) for ring in mol.GetRingInfo().BondRings()]
    rings_bonds: list[set[int]] = []
    for i, ring_i in enumerate(bond_rings):
        ring = ring_i
        adjacent = [
            ring_j
            for j, ring_j in enumerate(bond_rings)
            if i != j and len(ring_i.intersection(ring_j)) > 1
        ]
        if adjacent:
            ring = ring_i.union(*adjacent)
        if ring not in rings_bonds:
            rings_bonds.append(ring)

    rings_atoms = []
    for ring_bonds in rings_bonds:
        atoms = set()
        for bond_idx in ring_bonds:
            bond = mol.GetBondWithIdx(bond_idx)
            atoms.add(bond.GetBeginAtomIdx())
            atoms.add(bond.GetEndAtomIdx())
        rings_atoms.append(atoms)
    return rings_atoms


def _break_rings(mol: Chem.Mol, ring_size: int = 3) -> Chem.Mol:
    def get_ring() -> list[int] | None:
        for ring in mol.GetRingInfo().BondRings():
            if len(ring) == ring_size:
                return list(set(ring))
        return None

    ring = get_ring()
    while ring is not None:
        bonds = mol.GetRingInfo().BondRings()
        degrees = [sum(bond_idx in bond for bond in bonds) for bond_idx in ring]
        remove_bond = mol.GetBonds()[int(np.argmin(degrees))]
        editable = Chem.EditableMol(mol)
        editable.RemoveBond(remove_bond.GetBeginAtomIdx(), remove_bond.GetEndAtomIdx())
        mol = editable.GetMol()
        mol.ClearComputedProps()
        mol.UpdatePropertyCache()
        Chem.GetSymmSSSR(mol)
        ring = get_ring()
    return mol


def _murcko_hist(mol: Chem.Mol) -> dict[str, int]:
    # Adapted from DreaMS murcko_hist (MIT), keeping its ring/linker histogram.
    scaffold = MurckoScaffold.GetScaffoldForMol(_break_rings(mol, ring_size=3))
    link_atoms = set()
    for bond in scaffold.GetBonds():
        if bond.GetBeginAtom().GetDegree() < 2 or bond.GetEndAtom().GetDegree() < 2:
            continue
        if not bond.IsInRing():
            link_atoms.update([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

    rings_atoms = _multirings(scaffold)
    if not rings_atoms:
        return {}
    nums_adj_rings = np.zeros(len(rings_atoms), dtype=int)
    nums_adj_links = np.zeros(len(rings_atoms), dtype=int)
    for i, ring_i in enumerate(rings_atoms):
        for j, ring_j in enumerate(rings_atoms):
            if i != j:
                nums_adj_rings[i] += len(ring_i.intersection(ring_j)) // 2
        nums_adj_links[i] = len(ring_i.intersection(link_atoms))
    pairs, counts = np.unique(
        np.stack([nums_adj_rings, nums_adj_links]).T,
        axis=0,
        return_counts=True,
    )
    return {
        "_".join(str(int(value)) for value in pair): int(count)
        for pair, count in zip(pairs, counts, strict=True)
    }


def _murcko_hists_dist(left: dict[str, int], right: dict[str, int]) -> int:
    return sum(abs(left.get(key, 0) - right.get(key, 0)) for key in set(left) | set(right))


def _are_sub_hists(left: dict[str, int], right: dict[str, int], k: int = 3, d: int = 4) -> bool:
    if min(sum(left.values()), sum(right.values())) <= k:
        return left == right
    return _murcko_hists_dist(left, right) <= d


def _hist_key(hist: dict[str, int]) -> str:
    return json.dumps(hist, sort_keys=True, separators=(",", ":"))


def _metadata_json(record: dict[str, Any]) -> str:
    return json.dumps(
        {column.upper(): str(record.get(column, "")) for column in METADATA_COLUMNS},
        sort_keys=True,
    )


def _record_value(record: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = str(record.get(key, "")).strip()
        if value:
            return value
    return ""


def _collision_energy(record: dict[str, Any]) -> tuple[float, int]:
    raw = _record_value(record, "collisionenergy", "collision energy")
    if not raw or raw.lower() == "nan":
        return 0.0, 0
    token = ""
    for char in raw:
        if char.isdigit() or char in ".-+":
            token += char
        elif token:
            break
    if not token:
        return 0.0, 0
    return float(token), 1


def _precursor_mz(record: dict[str, Any]) -> float:
    pepmass = str(record.get("pepmass", ""))
    return _to_float(pepmass.split()[0]) if pepmass else float("nan")


def _record_adduct(record: dict[str, Any]) -> str:
    return _record_value(record, "precursortype", "adduct") or "unknown"


def _record_matches_adducts(
    record: dict[str, Any],
    allowed_adducts: tuple[str, ...] | None,
) -> bool:
    return allowed_adducts is None or _record_adduct(record) in allowed_adducts


def _top_peaks(record: dict[str, Any], num_peaks_input: int) -> tuple[list[float], list[float]]:
    mz = record["peak_mz"].astype(np.float32, copy=False)
    intensity = record["peak_intensity"].astype(np.float32, copy=False)
    if mz.size > num_peaks_input:
        keep = np.argpartition(-intensity, num_peaks_input - 1)[:num_peaks_input]
        keep = keep[np.argsort(mz[keep])]
        mz = mz[keep]
        intensity = intensity[keep]
    return mz.astype(np.float32, copy=False).tolist(), intensity.astype(np.float32, copy=False).tolist()


def _maccs_bits(mol: Chem.Mol) -> np.ndarray:
    full = np.zeros(MACCS_FINGERPRINT_BITS + 1, dtype=np.int8)
    DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), full)
    return full[1:]


def _morgan_bits(mol: Chem.Mol) -> np.ndarray:
    bits = np.zeros(MORGAN_PROBE_FINGERPRINT_BITS, dtype=np.int8)
    fp = _MORGAN_GENERATOR.GetFingerprint(mol)
    DataStructs.ConvertToNumpyArray(fp, bits)
    return bits


def _spectral_tokens(row: dict[str, Any]) -> set[int]:
    mz = np.asarray(row["spectrum_mz"], dtype=np.float32)
    intensity = np.asarray(row["spectrum_intensity"], dtype=np.float32)
    keep = intensity > 0
    mz = mz[keep]
    intensity = intensity[keep]
    normalized = intensity / intensity.max()
    mz_bins = np.rint(mz / SPECTRAL_LSH_MZ_BIN_WIDTH).astype(np.int64)
    intensity_bins = np.minimum(
        (normalized * SPECTRAL_LSH_INTENSITY_BINS).astype(np.int64),
        SPECTRAL_LSH_INTENSITY_BINS - 1,
    )
    return set((mz_bins * SPECTRAL_LSH_INTENSITY_BINS + intensity_bins).tolist())


def _minhash_signature(tokens: set[int]) -> tuple[int, ...]:
    token_array = np.asarray(list(tokens), dtype=np.uint64)
    hashes = (
        _SPECTRAL_LSH_A[:, None] * token_array[None, :] + _SPECTRAL_LSH_B[:, None]
    )
    hashes %= _SPECTRAL_LSH_PRIME
    return tuple(int(value) for value in hashes.min(axis=1).tolist())


def _jaccard(left: set[int], right: set[int]) -> float:
    return len(left & right) / len(left | right)


class SpectralLshThinner:
    def __init__(
        self,
        *,
        splits: tuple[str, ...],
        threshold: float,
        split_size_caps: dict[str, int],
        unique_smiles_by_split: dict[str, set[str]],
    ) -> None:
        self.threshold = threshold
        self.split_size_caps = split_size_caps
        self.unique_smiles_by_split = unique_smiles_by_split
        self.pre_lsh_counts: Counter[str] = Counter()
        self.removed_counts: Counter[str] = Counter()
        self.kept_counts: Counter[str] = Counter()
        self.kept_smiles = {split: set() for split in splits}
        self.tokens_by_split: dict[str, list[set[int]]] = {split: [] for split in splits}
        self.buckets: dict[str, dict[tuple[int, tuple[int, ...]], list[int]]] = {
            split: defaultdict(list) for split in splits
        }

    def keep(self, split: str, row: dict[str, Any]) -> bool:
        self.pre_lsh_counts[split] += 1
        canonical = row["canonical_smiles"]
        tokens = _spectral_tokens(row)
        cap = self.split_size_caps.get(split)
        if canonical in self.kept_smiles[split]:
            remaining_unseen = len(
                self.unique_smiles_by_split[split] - self.kept_smiles[split]
            )
            if cap is not None and self.kept_counts[split] + 1 + remaining_unseen > cap:
                self.removed_counts[split] += 1
                return False
            if self._has_match(split, tokens):
                self.removed_counts[split] += 1
                return False
        self._add(split, canonical, tokens)
        return True

    def _has_match(self, split: str, tokens: set[int]) -> bool:
        signature = _minhash_signature(tokens)
        candidates: set[int] = set()
        for band_idx in range(SPECTRAL_LSH_NUM_HASHES // SPECTRAL_LSH_BAND_SIZE):
            start = band_idx * SPECTRAL_LSH_BAND_SIZE
            key = (band_idx, signature[start : start + SPECTRAL_LSH_BAND_SIZE])
            candidates.update(self.buckets[split].get(key, ()))
        return any(
            _jaccard(tokens, self.tokens_by_split[split][candidate]) >= self.threshold
            for candidate in candidates
        )

    def _add(self, split: str, canonical: str, tokens: set[int]) -> None:
        spectrum_id = len(self.tokens_by_split[split])
        self.tokens_by_split[split].append(tokens)
        signature = _minhash_signature(tokens)
        for band_idx in range(SPECTRAL_LSH_NUM_HASHES // SPECTRAL_LSH_BAND_SIZE):
            start = band_idx * SPECTRAL_LSH_BAND_SIZE
            key = (band_idx, signature[start : start + SPECTRAL_LSH_BAND_SIZE])
            self.buckets[split][key].append(spectrum_id)
        self.kept_smiles[split].add(canonical)
        self.kept_counts[split] += 1


def _chemical_properties(mol: Chem.Mol) -> dict[str, float]:
    return {
        "mol_weight": float(Descriptors.ExactMolWt(mol)),
        "logp": float(Descriptors.MolLogP(mol)),
        "num_heavy_atoms": float(mol.GetNumHeavyAtoms()),
        "num_rings": float(rdMolDescriptors.CalcNumRings(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "num_hbd": float(rdMolDescriptors.CalcNumHBD(mol)),
        "num_hba": float(rdMolDescriptors.CalcNumHBA(mol)),
        "num_rotatable_bonds": float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "fraction_csp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "formal_charge": float(Chem.GetFormalCharge(mol)),
        "num_aromatic_rings": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
    }


def _mol_from_record(record: dict[str, Any]) -> tuple[Chem.Mol, str] | None:
    smiles = str(record.get("smiles", "")).strip()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    canonical = Chem.MolToSmiles(mol, canonical=True)
    return mol, canonical


@lru_cache(maxsize=500_000)
def _inchi14_from_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES in retrieval pool: {smiles!r}")
    inchikey = Chem.MolToInchiKey(mol)
    if not inchikey:
        raise ValueError(f"Could not compute InChIKey for retrieval-pool SMILES {smiles!r}")
    return inchikey.split("-")[0]


def _first_pass_task(
    payload: tuple[int, dict[str, Any], float, float, tuple[str, ...] | None],
) -> FirstPassRow | None:
    spectrum_index, record, min_precursor_mz, max_precursor_mz, allowed_adducts = payload
    if not _record_matches_adducts(record, allowed_adducts):
        return None
    precursor = _precursor_mz(record)
    if not math.isfinite(precursor) or precursor < min_precursor_mz or precursor > max_precursor_mz:
        return None
    if record["peak_mz"].size == 0 or not np.any(record["peak_intensity"] > 0):
        return None
    mol_payload = _mol_from_record(record)
    if mol_payload is None:
        return None
    mol, canonical = mol_payload
    hist = _murcko_hist(mol)
    return FirstPassRow(
        spectrum_index=int(spectrum_index),
        canonical_smiles=canonical,
        murcko_hist_key=_hist_key(hist),
        murcko_hist_json=json.dumps(hist, sort_keys=True),
    )


def _full_pass_task(
    payload: tuple[int, dict[str, Any], float, float, int, tuple[str, ...] | None],
) -> FullRow | None:
    spectrum_index, record, min_precursor_mz, max_precursor_mz, num_peaks_input, allowed_adducts = payload
    if not _record_matches_adducts(record, allowed_adducts):
        return None
    precursor = _precursor_mz(record)
    if not math.isfinite(precursor) or precursor < min_precursor_mz or precursor > max_precursor_mz:
        return None
    if record["peak_mz"].size == 0 or not np.any(record["peak_intensity"] > 0):
        return None
    mol_payload = _mol_from_record(record)
    if mol_payload is None:
        return None
    mol, canonical = mol_payload
    peak_mz, peak_intensity = _top_peaks(record, num_peaks_input)
    hist = _murcko_hist(mol)
    collision_energy, collision_energy_present = _collision_energy(record)
    atom_symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
    row: dict[str, Any] = {
        "spectrum_index": spectrum_index,
        "precursor_mz": float(precursor),
        "num_peaks": len(peak_mz),
        "spectrum_mz": peak_mz,
        "spectrum_intensity": peak_intensity,
        "smiles": str(record.get("smiles", "")).strip(),
        "canonical_smiles": canonical,
        "adduct": _record_adduct(record),
        "instrument_type": _record_value(record, "instrumenttype", "instrument_type")
        or "unknown",
        "collision_energy": collision_energy,
        "collision_energy_present": collision_energy_present,
        "has_fluorine": "F" in atom_symbols,
        "has_sulfur": "S" in atom_symbols,
        "maccs_166": _maccs_bits(mol),
        "murcko_hist_key": _hist_key(hist),
        "murcko_hist_json": json.dumps(hist, sort_keys=True),
        "metadata_json": _metadata_json(record),
    }
    row.update(_chemical_properties(mol))
    return FullRow(row=row, morgan=_morgan_bits(mol))


def _batched_mgf_tasks(
    mgf_path: Path,
    *,
    min_precursor_mz: float,
    max_precursor_mz: float,
    batch_size: int,
) -> Iterable[list[tuple[int, dict[str, Any], float, float]]]:
    batch = []
    for idx, record in enumerate(iter_mgf(mgf_path)):
        batch.append((idx, record, min_precursor_mz, max_precursor_mz))
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _select_holdout_hist_keys(
    *,
    available_keys: set[str],
    hist_counts: dict[str, int],
    hist_by_key: dict[str, dict[str, int]],
    target_size: int,
    seed: int,
    min_remaining_keys: int,
) -> set[str]:
    rng = np.random.default_rng(seed)
    keys = list(available_keys)
    rng.shuffle(keys)
    keys.sort(key=lambda key: hist_counts[key])
    selected: set[str] = set()
    selected_size = 0
    for key in keys:
        if key not in available_keys or key in selected:
            continue
        related = {
            other
            for other in available_keys
            if other not in selected and _are_sub_hists(hist_by_key[key], hist_by_key[other])
        }
        if not related:
            related = {key}
        if len(available_keys - selected - related) < min_remaining_keys:
            continue
        selected.update(related)
        selected_size += sum(hist_counts[other] for other in related)
        if selected_size >= target_size:
            break
    return selected


def _build_fold_map(
    rows: list[FirstPassRow],
    *,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    canonical_to_hist: dict[str, str] = {}
    hist_by_key: dict[str, dict[str, int]] = {}
    hist_counts: dict[str, int] = defaultdict(int)
    canonical_counts: Counter[str] = Counter()
    for row in rows:
        canonical_to_hist[row.canonical_smiles] = row.murcko_hist_key
        hist_by_key[row.murcko_hist_key] = json.loads(row.murcko_hist_json)
        hist_counts[row.murcko_hist_key] += 1
        canonical_counts[row.canonical_smiles] += 1

    total = sum(hist_counts.values())
    available = set(hist_counts)
    test_keys = _select_holdout_hist_keys(
        available_keys=available,
        hist_counts=hist_counts,
        hist_by_key=hist_by_key,
        target_size=max(1, int(round(total * test_frac))),
        seed=seed + 1,
        min_remaining_keys=2 if len(available) > 2 else 1,
    )
    available -= test_keys
    val_keys = _select_holdout_hist_keys(
        available_keys=available,
        hist_counts=hist_counts,
        hist_by_key=hist_by_key,
        target_size=max(1, int(round(total * val_frac))),
        seed=seed + 2,
        min_remaining_keys=1 if len(available) > 1 else 0,
    )
    train_keys = set(hist_counts) - test_keys - val_keys
    if not train_keys and (val_keys or test_keys):
        donor_keys = val_keys if val_keys else test_keys
        donor = max(donor_keys, key=lambda key: hist_counts[key])
        donor_keys.remove(donor)

    fold_by_smiles = {}
    for canonical, hist_key in canonical_to_hist.items():
        if hist_key in test_keys:
            fold_by_smiles[canonical] = "test"
        elif hist_key in val_keys:
            fold_by_smiles[canonical] = "val"
        else:
            fold_by_smiles[canonical] = "train"

    split_counts = Counter()
    for canonical, count in canonical_counts.items():
        split_counts[fold_by_smiles[canonical]] += count
    metadata = {
        "split_seed": seed,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "num_unique_smiles": len(canonical_counts),
        "num_murcko_histograms": len(hist_counts),
        "murcko_hist_split_counts": dict(split_counts),
        "murcko_hist_train_keys": len(set(hist_counts) - test_keys - val_keys),
        "murcko_hist_val_keys": len(val_keys),
        "murcko_hist_test_keys": len(test_keys),
    }
    return fold_by_smiles, metadata


def _build_single_split_fold_map(
    rows: list[FirstPassRow],
    *,
    split: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    canonical_to_hist: dict[str, str] = {}
    hist_counts: dict[str, int] = defaultdict(int)
    canonical_counts: Counter[str] = Counter()
    for row in rows:
        canonical_to_hist[row.canonical_smiles] = row.murcko_hist_key
        hist_counts[row.murcko_hist_key] += 1
        canonical_counts[row.canonical_smiles] += 1

    fold_by_smiles = {canonical: split for canonical in canonical_to_hist}
    split_counts = Counter({split: sum(canonical_counts.values())})
    metadata = {
        "split_seed": None,
        "val_frac": 0.0,
        "test_frac": 1.0 if split == "test" else 0.0,
        "num_unique_smiles": len(canonical_counts),
        "num_murcko_histograms": len(hist_counts),
        "murcko_hist_split_counts": dict(split_counts),
    }
    if split in SPLITS:
        metadata.update(
            {
                "murcko_hist_train_keys": len(hist_counts) if split == "train" else 0,
                "murcko_hist_val_keys": len(hist_counts) if split == "val" else 0,
                "murcko_hist_test_keys": len(hist_counts) if split == "test" else 0,
            }
        )
    else:
        metadata[f"murcko_hist_{split}_keys"] = len(hist_counts)
    return fold_by_smiles, metadata


def _fixed_size_int8_array(values: list[np.ndarray], width: int) -> pa.Array:
    if not values:
        return pa.array([], type=pa.list_(pa.int8(), width))
    matrix = np.stack(values, axis=0).astype(np.int8, copy=False)
    flat = pa.array(matrix.reshape(-1), type=pa.int8())
    return pa.FixedSizeListArray.from_arrays(flat, width)


def _rows_to_table(rows: list[dict[str, Any]]) -> pa.Table:
    names = [
        "spectrum_index",
        "fold",
        "precursor_mz",
        "num_peaks",
        "spectrum_mz",
        "spectrum_intensity",
        "smiles",
        "canonical_smiles",
        "adduct",
        "instrument_type",
        "collision_energy",
        "collision_energy_present",
        "has_fluorine",
        "has_sulfur",
    ]
    arrays: list[pa.Array] = [
        pa.array([row["spectrum_index"] for row in rows], type=pa.int64()),
        pa.array([row["fold"] for row in rows], type=pa.string()),
        pa.array([row["precursor_mz"] for row in rows], type=pa.float32()),
        pa.array([row["num_peaks"] for row in rows], type=pa.int32()),
        pa.array([row["spectrum_mz"] for row in rows], type=pa.list_(pa.float32())),
        pa.array([row["spectrum_intensity"] for row in rows], type=pa.list_(pa.float32())),
        pa.array([row["smiles"] for row in rows], type=pa.string()),
        pa.array([row["canonical_smiles"] for row in rows], type=pa.string()),
        pa.array([row["adduct"] for row in rows], type=pa.string()),
        pa.array([row["instrument_type"] for row in rows], type=pa.string()),
        pa.array([row["collision_energy"] for row in rows], type=pa.float32()),
        pa.array([row["collision_energy_present"] for row in rows], type=pa.int32()),
        pa.array([row["has_fluorine"] for row in rows], type=pa.bool_()),
        pa.array([row["has_sulfur"] for row in rows], type=pa.bool_()),
    ]
    for column in CHEMICAL_PROPERTY_COLUMNS:
        names.append(column)
        arrays.append(pa.array([row[column] for row in rows], type=pa.float32()))
    names.extend(["maccs_166", "murcko_hist_key", "murcko_hist_json", "metadata_json"])
    arrays.extend(
        [
            _fixed_size_int8_array([row["maccs_166"] for row in rows], MACCS_FINGERPRINT_BITS),
            pa.array([row["murcko_hist_key"] for row in rows], type=pa.string()),
            pa.array([row["murcko_hist_json"] for row in rows], type=pa.string()),
            pa.array([row["metadata_json"] for row in rows], type=pa.string()),
        ]
    )
    return pa.Table.from_arrays(arrays, names=names)


def _download_gcs_uri(gcs_uri: str, output_dir: Path, credentials: Path | None) -> Path:
    from google.cloud import storage

    without_scheme = gcs_uri[len("gs://") :]
    bucket_name, _, blob_path = without_scheme.partition("/")
    output_path = output_dir / Path(blob_path).name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if credentials is None:
        client = storage.Client()
    else:
        client = storage.Client.from_service_account_json(str(credentials))
    client.bucket(bucket_name).blob(blob_path).download_to_filename(str(output_path))
    return output_path


def _stage_raw_mgf(source: str, raw_dir: Path, credentials: Path | None) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    if source.startswith("gs://"):
        log.info("Downloading raw MGF %s", source)
        return _download_gcs_uri(source, raw_dir, credentials)
    source_path = Path(source).expanduser().resolve()
    output_path = raw_dir / source_path.name
    shutil.copy2(source_path, output_path)
    return output_path


def _first_pass(
    mgf_path: Path,
    *,
    min_precursor_mz: float,
    max_precursor_mz: float,
    allowed_adducts: tuple[str, ...] | None,
    num_workers: int,
    batch_size: int,
) -> list[FirstPassRow]:
    rows: list[FirstPassRow] = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for batch in _batched_mgf_tasks(
            mgf_path,
            min_precursor_mz=min_precursor_mz,
            max_precursor_mz=max_precursor_mz,
            batch_size=batch_size,
        ):
            full_batch = [
                (spectrum_index, record, min_mz, max_mz, allowed_adducts)
                for spectrum_index, record, min_mz, max_mz in batch
            ]
            for row in tqdm(
                executor.map(_first_pass_task, full_batch, chunksize=max(1, batch_size // num_workers)),
                total=len(full_batch),
                desc=f"{mgf_path.name} first pass",
                leave=False,
            ):
                if row is not None:
                    rows.append(row)
    return rows


def _flush_split(
    *,
    split: str,
    output_dir: Path,
    rows: list[dict[str, Any]],
    morgans: list[np.ndarray],
    writers: dict[str, pq.ParquetWriter],
    morgan_files: dict[str, list[str]],
    morgan_lengths: dict[str, list[int]],
) -> None:
    if not rows:
        return
    table = _rows_to_table(rows)
    parquet_path = output_dir / f"{split}.parquet"
    if split not in writers:
        writers[split] = pq.ParquetWriter(parquet_path, table.schema, compression="zstd")
    writers[split].write_table(table)

    morgan_dir = output_dir / "auxiliary" / "morgan"
    morgan_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{split}-part-{len(morgan_files[split]):05d}.npz"
    np.savez_compressed(
        morgan_dir / filename,
        spectrum_index=np.asarray([row["spectrum_index"] for row in rows], dtype=np.int64),
        morgan=np.stack(morgans, axis=0).astype(np.int8, copy=False),
    )
    morgan_files[split].append(f"auxiliary/morgan/{filename}")
    morgan_lengths[split].append(len(rows))
    rows.clear()
    morgans.clear()


def _unique_smiles_by_split(
    fold_by_smiles: dict[str, str],
    splits: tuple[str, ...],
) -> dict[str, set[str]]:
    unique_smiles = {split: set() for split in splits}
    for canonical, split in fold_by_smiles.items():
        unique_smiles[split].add(canonical)
    return unique_smiles


def _normalize_split_size_caps(
    split_size_caps: dict[str, int] | None,
    unique_smiles_by_split: dict[str, set[str]],
    splits: tuple[str, ...],
) -> dict[str, int] | None:
    if split_size_caps is None:
        return None
    caps = {split: int(split_size_caps[split]) for split in splits if split in split_size_caps}
    for split, cap in caps.items():
        unique_count = len(unique_smiles_by_split[split])
        if cap < unique_count:
            raise ValueError(
                f"{split} cap {cap} is smaller than {unique_count} unique canonical SMILES"
            )
    return caps


def _select_disjoint_probe_indices(
    rows: list[FirstPassRow],
    *,
    target_size: int,
    seed: int,
) -> tuple[set[int], set[str], dict[str, Any]]:
    """Select exact probe spectra and reserve their Murcko histograms.

    The selected spectra are sampled without replacement from a set of Murcko
    histogram keys chosen with the same related-histogram grouping used by the
    existing Murcko split. Any non-selected spectra sharing those histogram keys
    are deliberately excluded from the retrieval pool so the pool is chemically
    disjoint from the online-probe sample by Murcko histogram.
    """
    if target_size <= 0:
        raise ValueError(f"target_size must be positive, got {target_size}.")
    if len(rows) < target_size:
        raise ValueError(
            f"Cannot sample {target_size} probe spectra from only {len(rows)} eligible rows."
        )

    hist_by_key: dict[str, dict[str, int]] = {}
    hist_counts: dict[str, int] = defaultdict(int)
    spectrum_indices_by_hist: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        hist_by_key[row.murcko_hist_key] = json.loads(row.murcko_hist_json)
        hist_counts[row.murcko_hist_key] += 1
        spectrum_indices_by_hist[row.murcko_hist_key].append(int(row.spectrum_index))

    available = set(hist_counts)
    if len(available) < 2:
        raise ValueError(
            "Disjoint probe/retrieval construction requires at least two Murcko histogram keys."
        )
    selected_hist_keys = _select_holdout_hist_keys(
        available_keys=available,
        hist_counts=hist_counts,
        hist_by_key=hist_by_key,
        target_size=target_size,
        seed=seed,
        min_remaining_keys=1,
    )
    if not selected_hist_keys:
        raise ValueError("Failed to select any Murcko histogram keys for the probe sample.")

    candidate_indices = np.asarray(
        sorted(
            spectrum_index
            for key in selected_hist_keys
            for spectrum_index in spectrum_indices_by_hist[key]
        ),
        dtype=np.int64,
    )
    if len(candidate_indices) < target_size:
        raise ValueError(
            "Selected Murcko histogram keys contain too few spectra for the requested "
            f"probe sample: {len(candidate_indices)} < {target_size}."
        )

    rng = np.random.default_rng(seed + 101)
    selected_positions = rng.choice(
        np.arange(len(candidate_indices), dtype=np.int64),
        size=target_size,
        replace=False,
    )
    selected_indices = {int(value) for value in candidate_indices[selected_positions]}
    retrieval_hist_keys = available - selected_hist_keys
    retrieval_candidate_count = sum(hist_counts[key] for key in retrieval_hist_keys)
    selected_hist_candidate_count = int(len(candidate_indices))
    metadata = {
        "sampling": "murcko_hist_disjoint_without_replacement_v1",
        "seed": int(seed),
        "target_spectra": int(target_size),
        "selected_spectra": int(len(selected_indices)),
        "total_eligible_spectra": int(len(rows)),
        "total_murcko_hist_keys": int(len(available)),
        "selected_murcko_hist_keys": int(len(selected_hist_keys)),
        "retrieval_murcko_hist_keys": int(len(retrieval_hist_keys)),
        "selected_hist_candidate_spectra": selected_hist_candidate_count,
        "retrieval_candidate_spectra": int(retrieval_candidate_count),
        "dropped_selected_hist_spectra": int(selected_hist_candidate_count - target_size),
        "murcko_hist_disjoint_from_retrieval": True,
        "selected_murcko_hist_key_values": sorted(selected_hist_keys),
    }
    return selected_indices, selected_hist_keys, metadata


def _build_subset_murcko_mgf_dataset(
    *,
    mgf_path: Path,
    output_dir: Path,
    source_uri: str,
    fold_by_smiles: dict[str, str],
    split_metadata: dict[str, Any],
    active_splits: tuple[str, ...],
    min_precursor_mz: float,
    max_precursor_mz: float,
    num_peaks_input: int,
    num_workers: int,
    batch_size: int,
    parquet_batch_size: int,
    allowed_adducts: tuple[str, ...] | None,
    split_size_caps: dict[str, int] | None = None,
    spectral_lsh_threshold: float = 0.90,
    include_spectrum_indices: set[int] | None = None,
    include_murcko_hist_keys: set[str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
    collect_retrieval_rows: bool = False,
) -> tuple[dict[str, Any], list[RetrievalPoolRow]]:
    unique_smiles_by_split = _unique_smiles_by_split(fold_by_smiles, active_splits)
    normalized_split_size_caps = _normalize_split_size_caps(
        split_size_caps,
        unique_smiles_by_split,
        active_splits,
    )
    lsh_thinner = (
        SpectralLshThinner(
            splits=active_splits,
            threshold=spectral_lsh_threshold,
            split_size_caps=normalized_split_size_caps,
            unique_smiles_by_split=unique_smiles_by_split,
        )
        if normalized_split_size_caps is not None
        else None
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    writers: dict[str, pq.ParquetWriter] = {}
    buffers = {split: [] for split in active_splits}
    morgan_buffers = {split: [] for split in active_splits}
    morgan_files: dict[str, list[str]] = {split: [] for split in active_splits}
    morgan_lengths: dict[str, list[int]] = {split: [] for split in active_splits}
    split_counts: Counter[str] = Counter()
    pre_lsh_split_counts: Counter[str] = Counter()
    fluorine_counts: Counter[str] = Counter()
    sulfur_counts: Counter[str] = Counter()
    adducts: set[str] = set()
    instruments: set[str] = set()
    retrieval_rows: list[RetrievalPoolRow] = []
    retrieval_row_index = 0

    try:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for batch in _batched_mgf_tasks(
                mgf_path,
                min_precursor_mz=min_precursor_mz,
                max_precursor_mz=max_precursor_mz,
                batch_size=batch_size,
            ):
                full_batch = [
                    (
                        spectrum_index,
                        record,
                        min_mz,
                        max_mz,
                        num_peaks_input,
                        allowed_adducts,
                    )
                    for spectrum_index, record, min_mz, max_mz in batch
                    if include_spectrum_indices is None
                    or int(spectrum_index) in include_spectrum_indices
                ]
                if not full_batch:
                    continue
                for item in tqdm(
                    executor.map(
                        _full_pass_task,
                        full_batch,
                        chunksize=max(1, batch_size // num_workers),
                    ),
                    total=len(full_batch),
                    desc=f"{mgf_path.name} write subset",
                    leave=False,
                ):
                    if item is None:
                        continue
                    row = item.row
                    if (
                        include_murcko_hist_keys is not None
                        and row["murcko_hist_key"] not in include_murcko_hist_keys
                    ):
                        continue
                    split = fold_by_smiles.get(row["canonical_smiles"])
                    if split not in active_splits:
                        continue
                    row["fold"] = split
                    if lsh_thinner is None:
                        pre_lsh_split_counts[split] += 1
                    elif not lsh_thinner.keep(split, row):
                        continue
                    if collect_retrieval_rows:
                        retrieval_rows.append(
                            RetrievalPoolRow(
                                row_index=retrieval_row_index,
                                spectrum_index=int(row["spectrum_index"]),
                                canonical_smiles=str(row["canonical_smiles"]),
                                precursor_mz=float(row["precursor_mz"]),
                                adduct=str(row["adduct"]),
                                inchi14=_inchi14_from_smiles(str(row["canonical_smiles"])),
                                murcko_hist_key=str(row["murcko_hist_key"]),
                            )
                        )
                        retrieval_row_index += 1
                    buffers[split].append(row)
                    morgan_buffers[split].append(item.morgan)
                    split_counts[split] += 1
                    fluorine_counts[split] += int(row["has_fluorine"])
                    sulfur_counts[split] += int(row["has_sulfur"])
                    adducts.add(str(row["adduct"]))
                    instruments.add(str(row["instrument_type"]))
                    if len(buffers[split]) >= parquet_batch_size:
                        _flush_split(
                            split=split,
                            output_dir=output_dir,
                            rows=buffers[split],
                            morgans=morgan_buffers[split],
                            writers=writers,
                            morgan_files=morgan_files,
                            morgan_lengths=morgan_lengths,
                        )
        for split in active_splits:
            _flush_split(
                split=split,
                output_dir=output_dir,
                rows=buffers[split],
                morgans=morgan_buffers[split],
                writers=writers,
                morgan_files=morgan_files,
                morgan_lengths=morgan_lengths,
            )
    finally:
        for writer in writers.values():
            writer.close()

    adduct_vocab = {value: idx for idx, value in enumerate(sorted(adducts))}
    instrument_type_vocab = {value: idx for idx, value in enumerate(sorted(instruments))}
    if lsh_thinner is not None:
        pre_lsh_split_counts = lsh_thinner.pre_lsh_counts
        lsh_removed_counts = lsh_thinner.removed_counts
    else:
        lsh_removed_counts = Counter()
    metadata: dict[str, Any] = {
        "metadata_version": NIST_MURCKO_METADATA_VERSION,
        "artifact_format": NIST_MURCKO_ARTIFACT_FORMAT,
        "storage_format": "parquet",
        "source_uri": source_uri,
        "source_raw_file": f"{RAW_SUBDIR}/{mgf_path.name}",
        "splits": list(active_splits),
        "num_peaks_input": num_peaks_input,
        "min_precursor_mz": min_precursor_mz,
        "max_precursor_mz": max_precursor_mz,
        "allowed_adducts": list(allowed_adducts) if allowed_adducts is not None else None,
        "adduct_vocab": adduct_vocab,
        "instrument_type_vocab": instrument_type_vocab,
        "dreams_dim": 0,
        "chemical_property_columns": list(CHEMICAL_PROPERTY_COLUMNS),
        "probe_regression_target_keys": list(REGRESSION_TARGET_KEYS),
        "probe_maccs_bits": MACCS_FINGERPRINT_BITS,
        "probe_maccs_column": "maccs_166",
        "probe_morgan_bits": MORGAN_PROBE_FINGERPRINT_BITS,
        "probe_morgan_radius": MORGAN_PROBE_FINGERPRINT_RADIUS,
        "morgan_auxiliary_available": True,
        "morgan_auxiliary_files": morgan_files,
        "morgan_auxiliary_lengths": morgan_lengths,
        "pairwise_alignment_available": False,
        "pairwise_alignment_num_pairs": 0,
        "pairwise_alignment_num_endpoints": 0,
        "spectral_lsh_enabled": lsh_thinner is not None,
        "spectral_lsh_threshold": spectral_lsh_threshold,
        "split_size_caps": normalized_split_size_caps,
        "pre_lsh_split_sizes": {
            split: pre_lsh_split_counts[split] for split in active_splits
        },
        "lsh_removed_by_split": {
            split: lsh_removed_counts[split] for split in active_splits
        },
        "unique_smiles_by_split": {
            split: len(unique_smiles_by_split[split]) for split in active_splits
        },
    }
    metadata.update(split_metadata)
    if extra_metadata is not None:
        metadata.update(extra_metadata)
    for split in active_splits:
        metadata[f"{split}_files"] = [f"{split}.parquet"] if split_counts[split] else []
        metadata[f"{split}_lengths"] = [split_counts[split]] if split_counts[split] else []
        metadata[f"{split}_size"] = split_counts[split]
        metadata[f"{split}_positive"] = fluorine_counts[split]
        metadata[f"{split}_sulfur_positive"] = sulfur_counts[split]

    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata, retrieval_rows


def _sample_mz_window_balanced_inchi_pairs(
    precursor_mz: np.ndarray,
    inchi14: list[str],
    *,
    pairs_per_class: int,
    ppm: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Sample fixed 10 ppm same-InChI positives and hard negatives.

    Positives share the same InChIKey first block. Negatives have different
    first blocks but are still inside the same precursor-m/z tolerance, making
    them the hard negatives used for the AUROC retrieval task.
    """
    precursor_mz = np.asarray(precursor_mz, dtype=np.float64)
    if len(precursor_mz) != len(inchi14):
        raise ValueError(f"precursor/InChI length mismatch: {len(precursor_mz)} vs {len(inchi14)}")
    if pairs_per_class < 0:
        raise ValueError(f"pairs_per_class must be non-negative, got {pairs_per_class}.")
    if ppm <= 0.0:
        raise ValueError(f"ppm must be positive, got {ppm}.")

    rng = np.random.default_rng(seed)
    finite_rows = np.flatnonzero(np.isfinite(precursor_mz) & (precursor_mz > 0.0)).astype(np.int64)
    order = finite_rows[np.argsort(precursor_mz[finite_rows], kind="mergesort")]
    sorted_mz = precursor_mz[order]
    key_arr = np.asarray(inchi14, dtype=object)

    positive_groups: dict[str, list[tuple[int, int]]] = {}
    negative_groups: dict[str, list[tuple[int, int]]] = {}
    for sorted_pos, anchor in enumerate(order):
        anchor = int(anchor)
        mz = float(sorted_mz[sorted_pos])
        delta = mz * ppm * 1e-6
        hi = np.searchsorted(sorted_mz, mz + delta, side="right")
        for candidate in order[sorted_pos + 1 : hi]:
            candidate = int(candidate)
            pair = _ordered_pair(anchor, int(candidate))
            if key_arr[anchor] == key_arr[candidate]:
                positive_groups.setdefault(str(key_arr[anchor]), []).append(pair)
            else:
                negative_groups.setdefault(str(key_arr[anchor]), []).append(pair)
                negative_groups.setdefault(str(key_arr[candidate]), []).append(pair)

    metadata: dict[str, Any] = {
        "sampling": SAME_INCHI_10PPM_PAIR_SAMPLING,
        "target_pairs_per_class": int(pairs_per_class),
        "ppm": float(ppm),
        "finite_precursor_mz_spectra": int(len(order)),
        "positive_candidate_pairs": int(sum(len(pairs) for pairs in positive_groups.values())),
        "negative_candidate_pairs_before_dedup": int(sum(len(pairs) for pairs in negative_groups.values())),
        "negative_candidate_pairs": int(
            len({pair for pairs in negative_groups.values() for pair in pairs})
        ),
        "positive_inchi14_groups": int(len(positive_groups)),
        "negative_inchi14_groups": int(len(negative_groups)),
    }
    if pairs_per_class == 0:
        metadata.update(
            {
                "final_pairs": 0,
                "positive_pairs": 0,
                "negative_pairs": 0,
                "duplicate_pairs": 0,
            }
        )
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.int8), metadata

    positive_pairs, positive_selected_groups = _sample_pair_groups_round_robin(
        positive_groups,
        target_pairs=pairs_per_class,
        rng=rng,
    )
    negative_pairs, negative_selected_groups = _sample_pair_groups_round_robin(
        negative_groups,
        target_pairs=pairs_per_class,
        rng=rng,
    )
    pairs = np.concatenate([positive_pairs, negative_pairs], axis=0)
    labels = np.concatenate(
        [
            np.ones(len(positive_pairs), dtype=np.int8),
            np.zeros(len(negative_pairs), dtype=np.int8),
        ],
        axis=0,
    )
    duplicate_pairs = len(pairs) - len({_ordered_pair(int(i), int(j)) for i, j in pairs})
    metadata.update(
        {
            "final_pairs": int(len(pairs)),
            "positive_pairs": int(len(positive_pairs)),
            "negative_pairs": int(len(negative_pairs)),
            "selected_positive_inchi14_groups": int(len(positive_selected_groups)),
            "selected_negative_inchi14_groups": int(len(negative_selected_groups)),
            "duplicate_pairs": int(duplicate_pairs),
        }
    )
    if len(positive_pairs) != pairs_per_class or len(negative_pairs) != pairs_per_class:
        raise ValueError(
            "Could not satisfy 10 ppm same-InChI balanced pair target: "
            f"requested {pairs_per_class} per class, selected {len(positive_pairs)} "
            f"positives and {len(negative_pairs)} negatives."
        )
    if duplicate_pairs:
        raise ValueError(f"10 ppm same-InChI sampler selected {duplicate_pairs} duplicate pairs.")
    return pairs, labels, metadata


def _sample_pair_groups_round_robin(
    groups: dict[str, list[tuple[int, int]]],
    *,
    target_pairs: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, set[str]]:
    key_order = sorted(groups)
    if not key_order:
        return np.empty((0, 2), dtype=np.int64), set()
    key_order = [key_order[int(i)] for i in rng.permutation(len(key_order))]
    shuffled_groups: dict[str, list[tuple[int, int]]] = {}
    for key in key_order:
        pair_list = groups[key]
        shuffled_groups[key] = [pair_list[int(i)] for i in rng.permutation(len(pair_list))]

    cursors = {key: 0 for key in key_order}
    active_keys = list(key_order)
    selected: list[tuple[int, int]] = []
    selected_groups: set[str] = set()
    used_pairs: set[tuple[int, int]] = set()
    while len(selected) < target_pairs and active_keys:
        made_progress = False
        for key in list(active_keys):
            pairs = shuffled_groups[key]
            pair = None
            while cursors[key] < len(pairs):
                candidate_pair = pairs[cursors[key]]
                cursors[key] += 1
                if candidate_pair not in used_pairs:
                    pair = candidate_pair
                    break
            if pair is None:
                active_keys.remove(key)
                continue
            selected.append(pair)
            selected_groups.add(key)
            used_pairs.add(pair)
            made_progress = True
            if len(selected) == target_pairs:
                break
        if not made_progress:
            break
    return np.asarray(selected, dtype=np.int64).reshape(-1, 2), selected_groups


def _ordered_pair(i: int, j: int) -> tuple[int, int]:
    i = int(i)
    j = int(j)
    return (i, j) if i < j else (j, i)


def _retrieval_pair_base_columns(
    rows: list[RetrievalPoolRow],
    pairs: np.ndarray,
) -> dict[str, pa.Array]:
    left = pairs[:, 0].astype(np.int64, copy=False) if len(pairs) else np.empty(0, dtype=np.int64)
    right = pairs[:, 1].astype(np.int64, copy=False) if len(pairs) else np.empty(0, dtype=np.int64)
    left_rows = [rows[int(idx)] for idx in left]
    right_rows = [rows[int(idx)] for idx in right]
    return {
        "pair_index": pa.array(np.arange(len(pairs), dtype=np.int64), type=pa.int64()),
        "left_row": pa.array(left, type=pa.int64()),
        "right_row": pa.array(right, type=pa.int64()),
        "left_spectrum_index": pa.array([row.spectrum_index for row in left_rows], type=pa.int64()),
        "right_spectrum_index": pa.array([row.spectrum_index for row in right_rows], type=pa.int64()),
        "left_smiles": pa.array([row.canonical_smiles for row in left_rows], type=pa.string()),
        "right_smiles": pa.array([row.canonical_smiles for row in right_rows], type=pa.string()),
        "left_inchi14": pa.array([row.inchi14 for row in left_rows], type=pa.string()),
        "right_inchi14": pa.array([row.inchi14 for row in right_rows], type=pa.string()),
        "left_precursor_mz": pa.array([row.precursor_mz for row in left_rows], type=pa.float32()),
        "right_precursor_mz": pa.array([row.precursor_mz for row in right_rows], type=pa.float32()),
    }


def _write_same_inchi_retrieval_dataset(
    *,
    output_dir: Path,
    retrieval_rows: list[RetrievalPoolRow],
    pairs_per_class: int,
    ppm: float,
    adduct: str,
    seed: int,
    retrieval_pool_subdir: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_rows = [row for row in retrieval_rows if row.adduct == adduct]
    selection = np.asarray([row.row_index for row in selected_rows], dtype=np.int64)
    precursor = np.asarray([row.precursor_mz for row in selected_rows], dtype=np.float64)
    inchi14 = [row.inchi14 for row in selected_rows]
    local_pairs, labels, sampling_metadata = _sample_mz_window_balanced_inchi_pairs(
        precursor,
        inchi14,
        pairs_per_class=pairs_per_class,
        ppm=ppm,
        seed=seed,
    )
    pairs = selection[local_pairs] if len(local_pairs) else local_pairs
    columns = _retrieval_pair_base_columns(retrieval_rows, pairs)
    columns["label"] = pa.array(labels.astype(np.int8, copy=False), type=pa.int8())
    pq.write_table(pa.table(columns), output_dir / "pairs.parquet", compression="zstd")
    metadata = {
        "metadata_version": 1,
        "artifact_format": "same_inchi14_10ppm_retrieval_pairs_v1",
        "retrieval_pool_subdir": retrieval_pool_subdir,
        "pairs_file": "pairs.parquet",
        "selection_adduct": adduct,
        "num_selected_spectra": int(len(selection)),
        "num_pairs": int(len(labels)),
        "positive_pairs": int(labels.sum()) if len(labels) else 0,
        "negative_pairs": int((labels == 0).sum()) if len(labels) else 0,
        "sampling_metadata": sampling_metadata,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata


def _sample_balanced_morgan_pairs_for_retrieval(
    retrieval_rows: list[RetrievalPoolRow],
    *,
    num_pairs: int,
    bin_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if num_pairs <= 0:
        metadata = {
            "sampling": MCES_ANALOG_PAIR_SAMPLING,
            "target_pairs": int(num_pairs),
            "final_pairs": 0,
            "bin_size": float(bin_size),
            "num_bins": 0,
        }
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.float32), metadata
    if not (0.0 < bin_size <= 1.0):
        raise ValueError(f"bin_size must be in (0, 1], got {bin_size}.")
    n_bins_float = 1.0 / bin_size
    n_bins = int(round(n_bins_float))
    if not np.isclose(n_bins_float, n_bins):
        raise ValueError(f"bin_size must evenly divide 1.0, got {bin_size}.")
    target_per_bin = num_pairs // n_bins
    if target_per_bin < 1:
        raise ValueError(
            f"num_pairs={num_pairs} is too small for {n_bins} Tanimoto bins; "
            f"request at least {n_bins} pairs or use a wider bin."
        )
    target_pairs = target_per_bin * n_bins

    rng = np.random.default_rng(seed)
    row_groups: dict[str, list[int]] = defaultdict(list)
    for row in retrieval_rows:
        row_groups[row.canonical_smiles].append(int(row.row_index))
    canonical_smiles = sorted(row_groups)
    if len(canonical_smiles) < 2:
        raise ValueError("MCES retrieval pair sampling requires at least two unique SMILES.")
    representative_rows = np.asarray(
        [
            row_groups[smiles][int(rng.integers(0, len(row_groups[smiles])))]
            for smiles in canonical_smiles
        ],
        dtype=np.int64,
    )
    fps = [
        _MORGAN_GENERATOR.GetFingerprint(Chem.MolFromSmiles(smiles))
        for smiles in canonical_smiles
    ]
    all_rep = np.arange(len(fps), dtype=np.int64)
    pairs_by_bin: list[list[tuple[int, int, float]]] = [[] for _ in range(n_bins)]
    used_pairs: set[tuple[int, int]] = set()

    for i in rng.permutation(len(fps)):
        i = int(i)
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fps[i], fps), dtype=np.float32)
        bin_ids = np.ceil(sims / bin_size).astype(np.int16) - 1
        for bin_idx in range(n_bins):
            need = target_per_bin - len(pairs_by_bin[bin_idx])
            if need <= 0:
                continue
            candidates = all_rep[(bin_ids == bin_idx) & (all_rep != i)]
            if not candidates.size:
                continue
            candidates = candidates[rng.permutation(len(candidates))]
            for candidate in candidates:
                pair = _ordered_pair(i, int(candidate))
                if pair in used_pairs:
                    continue
                used_pairs.add(pair)
                pairs_by_bin[bin_idx].append((pair[0], pair[1], float(sims[int(candidate)])))
                need -= 1
                if need == 0:
                    break
        if all(len(bucket) >= target_per_bin for bucket in pairs_by_bin):
            break

    if any(len(bucket) < target_per_bin for bucket in pairs_by_bin):
        counts = [len(bucket) for bucket in pairs_by_bin]
        raise ValueError(
            "Could not satisfy balanced Morgan-Tanimoto MCES pair sampling target. "
            f"Need {target_per_bin} per bin, got {counts}."
        )

    rep_pairs = [pair for bucket in pairs_by_bin for pair in bucket[:target_per_bin]]
    pairs = np.asarray(
        [
            [representative_rows[left], representative_rows[right]]
            for left, right, _ in rep_pairs
        ],
        dtype=np.int64,
    )
    tanimoto = np.asarray([score for _, _, score in rep_pairs], dtype=np.float32)
    metadata = {
        "sampling": MCES_ANALOG_PAIR_SAMPLING,
        "seed": int(seed),
        "target_pairs": int(num_pairs),
        "final_pairs": int(len(pairs)),
        "bin_size": float(bin_size),
        "num_bins": int(n_bins),
        "target_pairs_per_bin": int(target_per_bin),
        "bin_counts": [int(len(bucket[:target_per_bin])) for bucket in pairs_by_bin],
        "unique_smiles": int(len(canonical_smiles)),
        "representative_spectra": int(len(representative_rows)),
        "morgan_bits": int(MORGAN_PROBE_FINGERPRINT_BITS),
        "morgan_radius": int(MORGAN_PROBE_FINGERPRINT_RADIUS),
    }
    return pairs, tanimoto, metadata


def _compute_mces_values(
    retrieval_rows: list[RetrievalPoolRow],
    pairs: np.ndarray,
    *,
    workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if workers <= 0:
        raise ValueError(f"mces_workers must be positive, got {workers}.")
    if len(pairs) == 0:
        return (
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int16),
        )
    jobs = [
        (
            pair_idx,
            retrieval_rows[int(left)].canonical_smiles,
            retrieval_rows[int(right)].canonical_smiles,
        )
        for pair_idx, (left, right) in enumerate(pairs)
    ]
    values = np.empty(len(jobs), dtype=np.float32)
    seconds = np.empty(len(jobs), dtype=np.float32)
    modes = np.empty(len(jobs), dtype=np.int16)
    if workers == 1:
        results = map(_compute_mces_one, jobs)
    else:
        max_workers = min(workers, len(jobs))
        chunksize = max(1, len(jobs) // (max_workers * 4))
        executor = ProcessPoolExecutor(max_workers=max_workers)
        results = executor.map(_compute_mces_one, jobs, chunksize=chunksize)
    try:
        for pair_idx, value, elapsed, mode in results:
            values[pair_idx] = value
            seconds[pair_idx] = elapsed
            modes[pair_idx] = mode
    finally:
        if workers != 1:
            executor.shutdown(wait=True)
    return values, seconds, modes


def _compute_mces_one(job: tuple[int, str, str]) -> tuple[int, float, float, int]:
    try:
        from myopic_mces.myopic_mces import MCES
        from pulp.apis.coin_api import pulp_cbc_path
    except ImportError as exc:
        raise ImportError(
            "MCES retrieval artifact generation requires myopic-mces and pulp. "
            "Install those packages or run this builder in the spectra-benchmarking environment."
        ) from exc

    pair_idx, left_smiles, right_smiles = job
    start = time.perf_counter()
    _idx, value, elapsed, mode = MCES(
        left_smiles,
        right_smiles,
        threshold=MCES_THRESHOLD,
        i=pair_idx,
        solver="COIN_CMD",
        solver_options={"msg": False, "path": pulp_cbc_path},
        catch_errors=False,
    )
    mode_value = mode.value if hasattr(mode, "value") else mode
    return pair_idx, float(value), float(elapsed or (time.perf_counter() - start)), int(mode_value)


def _write_mces_retrieval_dataset(
    *,
    output_dir: Path,
    retrieval_rows: list[RetrievalPoolRow],
    num_pairs: int,
    bin_size: float,
    seed: int,
    workers: int,
    retrieval_pool_subdir: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs, tanimoto, sampling_metadata = _sample_balanced_morgan_pairs_for_retrieval(
        retrieval_rows,
        num_pairs=num_pairs,
        bin_size=bin_size,
        seed=seed,
    )
    mces_values, mces_seconds, mces_modes = _compute_mces_values(
        retrieval_rows,
        pairs,
        workers=workers,
    )
    if np.any(~np.isfinite(mces_values)):
        bad = int(np.flatnonzero(~np.isfinite(mces_values))[0])
        raise ValueError(f"MCES returned a non-finite distance at pair {bad}.")
    if np.any(mces_values < 0):
        bad = int(np.flatnonzero(mces_values < 0)[0])
        raise ValueError(f"MCES returned a negative distance at pair {bad}: {mces_values[bad]}")

    columns = _retrieval_pair_base_columns(retrieval_rows, pairs)
    columns["morgan_tanimoto"] = pa.array(tanimoto.astype(np.float32, copy=False), type=pa.float32())
    columns["mces"] = pa.array(mces_values.astype(np.float32, copy=False), type=pa.float32())
    columns["mces_seconds"] = pa.array(mces_seconds.astype(np.float32, copy=False), type=pa.float32())
    columns["mces_compute_mode"] = pa.array(mces_modes.astype(np.int16, copy=False), type=pa.int16())
    for threshold in MCES_REPORTED_THRESHOLDS:
        columns[f"mces_le_{threshold}"] = pa.array(
            (mces_values <= threshold).astype(np.int8),
            type=pa.int8(),
        )
    pq.write_table(pa.table(columns), output_dir / "pairs.parquet", compression="zstd")
    metadata = {
        "metadata_version": 1,
        "artifact_format": "mces_analog_retrieval_pairs_v1",
        "retrieval_pool_subdir": retrieval_pool_subdir,
        "pairs_file": "pairs.parquet",
        "num_pairs": int(len(pairs)),
        "mces_threshold": int(MCES_THRESHOLD),
        "reported_thresholds": [int(value) for value in MCES_REPORTED_THRESHOLDS],
        "num_workers": int(workers),
        "compute_mode_counts": {
            str(int(mode)): int(count)
            for mode, count in zip(*np.unique(mces_modes, return_counts=True), strict=True)
        },
        "mean_seconds_per_pair": float(np.mean(mces_seconds)) if len(mces_seconds) else 0.0,
        "max_seconds_per_pair": float(np.max(mces_seconds)) if len(mces_seconds) else 0.0,
        "sampling_metadata": sampling_metadata,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata


def build_murcko_mgf_dataset(
    *,
    mgf_path: Path,
    output_dir: Path,
    source_uri: str,
    val_frac: float,
    test_frac: float,
    seed: int,
    min_precursor_mz: float,
    max_precursor_mz: float,
    num_peaks_input: int,
    num_workers: int,
    batch_size: int,
    parquet_batch_size: int,
    single_split: str | None = None,
    allowed_adducts: tuple[str, ...] | None = None,
    split_size_caps: dict[str, int] | None = None,
    spectral_lsh_threshold: float = 0.90,
) -> dict[str, Any]:
    first_rows = _first_pass(
        mgf_path,
        min_precursor_mz=min_precursor_mz,
        max_precursor_mz=max_precursor_mz,
        allowed_adducts=allowed_adducts,
        num_workers=num_workers,
        batch_size=batch_size,
    )
    if single_split is None:
        fold_by_smiles, split_metadata = _build_fold_map(
            first_rows,
            val_frac=val_frac,
            test_frac=test_frac,
            seed=seed,
        )
    else:
        fold_by_smiles, split_metadata = _build_single_split_fold_map(
            first_rows,
            split=single_split,
        )
    active_splits = (single_split,) if single_split is not None else SPLITS
    unique_smiles_by_split = _unique_smiles_by_split(fold_by_smiles, active_splits)
    normalized_split_size_caps = _normalize_split_size_caps(
        split_size_caps,
        unique_smiles_by_split,
        active_splits,
    )
    lsh_thinner = (
        SpectralLshThinner(
            splits=active_splits,
            threshold=spectral_lsh_threshold,
            split_size_caps=normalized_split_size_caps,
            unique_smiles_by_split=unique_smiles_by_split,
        )
        if normalized_split_size_caps is not None
        else None
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    writers: dict[str, pq.ParquetWriter] = {}
    buffers = {split: [] for split in active_splits}
    morgan_buffers = {split: [] for split in active_splits}
    morgan_files: dict[str, list[str]] = {split: [] for split in active_splits}
    morgan_lengths: dict[str, list[int]] = {split: [] for split in active_splits}
    split_counts: Counter[str] = Counter()
    pre_lsh_split_counts: Counter[str] = Counter()
    fluorine_counts: Counter[str] = Counter()
    sulfur_counts: Counter[str] = Counter()
    adducts: set[str] = set()
    instruments: set[str] = set()

    try:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for batch in _batched_mgf_tasks(
                mgf_path,
                min_precursor_mz=min_precursor_mz,
                max_precursor_mz=max_precursor_mz,
                batch_size=batch_size,
            ):
                full_batch = [
                    (
                        spectrum_index,
                        record,
                        min_mz,
                        max_mz,
                        num_peaks_input,
                        allowed_adducts,
                    )
                    for spectrum_index, record, min_mz, max_mz in batch
                ]
                for item in tqdm(
                    executor.map(
                        _full_pass_task,
                        full_batch,
                        chunksize=max(1, batch_size // num_workers),
                    ),
                    total=len(full_batch),
                    desc=f"{mgf_path.name} write splits",
                    leave=False,
                ):
                    if item is None:
                        continue
                    row = item.row
                    split = fold_by_smiles[row["canonical_smiles"]]
                    row["fold"] = split
                    if lsh_thinner is None:
                        pre_lsh_split_counts[split] += 1
                    elif not lsh_thinner.keep(split, row):
                        continue
                    buffers[split].append(row)
                    morgan_buffers[split].append(item.morgan)
                    split_counts[split] += 1
                    fluorine_counts[split] += int(row["has_fluorine"])
                    sulfur_counts[split] += int(row["has_sulfur"])
                    adducts.add(str(row["adduct"]))
                    instruments.add(str(row["instrument_type"]))
                    if len(buffers[split]) >= parquet_batch_size:
                        _flush_split(
                            split=split,
                            output_dir=output_dir,
                            rows=buffers[split],
                            morgans=morgan_buffers[split],
                            writers=writers,
                            morgan_files=morgan_files,
                            morgan_lengths=morgan_lengths,
                        )
        for split in active_splits:
            _flush_split(
                split=split,
                output_dir=output_dir,
                rows=buffers[split],
                morgans=morgan_buffers[split],
                writers=writers,
                morgan_files=morgan_files,
                morgan_lengths=morgan_lengths,
            )
    finally:
        for writer in writers.values():
            writer.close()

    adduct_vocab = {value: idx for idx, value in enumerate(sorted(adducts))}
    instrument_type_vocab = {value: idx for idx, value in enumerate(sorted(instruments))}
    if lsh_thinner is not None:
        pre_lsh_split_counts = lsh_thinner.pre_lsh_counts
        lsh_removed_counts = lsh_thinner.removed_counts
    else:
        lsh_removed_counts = Counter()
    metadata: dict[str, Any] = {
        "metadata_version": NIST_MURCKO_METADATA_VERSION,
        "artifact_format": NIST_MURCKO_ARTIFACT_FORMAT,
        "storage_format": "parquet",
        "source_uri": source_uri,
        "source_raw_file": f"{RAW_SUBDIR}/{mgf_path.name}",
        "splits": list(active_splits),
        "num_peaks_input": num_peaks_input,
        "min_precursor_mz": min_precursor_mz,
        "max_precursor_mz": max_precursor_mz,
        "allowed_adducts": list(allowed_adducts) if allowed_adducts is not None else None,
        "adduct_vocab": adduct_vocab,
        "instrument_type_vocab": instrument_type_vocab,
        "dreams_dim": 0,
        "chemical_property_columns": list(CHEMICAL_PROPERTY_COLUMNS),
        "probe_regression_target_keys": list(REGRESSION_TARGET_KEYS),
        "probe_maccs_bits": MACCS_FINGERPRINT_BITS,
        "probe_maccs_column": "maccs_166",
        "probe_morgan_bits": MORGAN_PROBE_FINGERPRINT_BITS,
        "probe_morgan_radius": MORGAN_PROBE_FINGERPRINT_RADIUS,
        "morgan_auxiliary_available": True,
        "morgan_auxiliary_files": morgan_files,
        "morgan_auxiliary_lengths": morgan_lengths,
        "pairwise_alignment_available": False,
        "pairwise_alignment_num_pairs": 0,
        "pairwise_alignment_num_endpoints": 0,
        "spectral_lsh_enabled": lsh_thinner is not None,
        "spectral_lsh_threshold": spectral_lsh_threshold,
        "split_size_caps": normalized_split_size_caps,
        "pre_lsh_split_sizes": {
            split: pre_lsh_split_counts[split] for split in active_splits
        },
        "lsh_removed_by_split": {
            split: lsh_removed_counts[split] for split in active_splits
        },
        "unique_smiles_by_split": {
            split: len(unique_smiles_by_split[split]) for split in active_splits
        },
    }
    metadata.update(split_metadata)
    for split in active_splits:
        metadata[f"{split}_files"] = [f"{split}.parquet"] if split_counts[split] else []
        metadata[f"{split}_lengths"] = [split_counts[split]] if split_counts[split] else []
        metadata[f"{split}_size"] = split_counts[split]
        metadata[f"{split}_positive"] = fluorine_counts[split]
        metadata[f"{split}_sulfur_positive"] = sulfur_counts[split]

    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    log.info(
        "%s: %s",
        output_dir.name,
        " ".join(f"{split}={metadata[f'{split}_size']}" for split in active_splits),
    )
    return metadata


def _build_dataset_specs(
    *,
    nist_mgf: str,
    mcebio_mgf: str | None,
    nist_subdir: str,
    mcebio_subdir: str,
) -> list[DatasetSpec]:
    specs = [
        DatasetSpec(
            name="nist",
            source=nist_mgf,
            subdir=nist_subdir,
        )
    ]
    if mcebio_mgf:
        specs.append(
            DatasetSpec(
                name="mcebio",
                source=mcebio_mgf,
                subdir=mcebio_subdir,
            )
        )
    return specs


def prepare_murcko_mgf_collection(
    *,
    nist_mgf: str = DEFAULT_NIST_MGF_URI,
    mcebio_mgf: str | None = str(DEFAULT_MCEBIO_MGF_PATH),
    nist_subdir: str = NIST_MURCKO_PREPARED_SUBDIR,
    mcebio_subdir: str = MCEBIO_MURCKO_PREPARED_SUBDIR,
    gcs_credentials: Path | None = None,
    work_dir: Path,
    hf_repo_id: str = NIST_MURCKO_HF_REPO,
    hf_revision: str = "main",
    hf_private: bool = False,
    upload: bool = True,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    seed: int = 42,
    min_precursor_mz: float = 1.0,
    max_precursor_mz: float = 1000.0,
    num_peaks_input: int = NUM_PEAKS_INPUT,
    num_workers: int = os.cpu_count() or 1,
    batch_size: int = 2048,
    parquet_batch_size: int = 50_000,
    nist_allowed_adducts: tuple[str, ...] | None = DEFAULT_NIST_ALLOWED_ADDUCTS,
    nist_split_size_caps: dict[str, int] | None = DEFAULT_NIST_SPLIT_SIZE_CAPS,
    spectral_lsh_threshold: float = 0.90,
    build_dreams_auxiliary: bool = False,
    dreams_root: Path = Path("/home/wuhao/Dreams"),
    dreams_checkpoint: Path | None = None,
    dreams_subdirs: list[str] | None = None,
    dreams_n_highest_peaks: int = 100,
    dreams_batch_size: int = 256,
    dreams_device: str | None = None,
) -> dict[str, Any]:
    work_dir = work_dir.expanduser().resolve()
    staging_root = work_dir / "artifact"
    raw_dir = staging_root / RAW_SUBDIR
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True, exist_ok=True)

    top_metadata: dict[str, Any] = {
        "metadata_version": NIST_MURCKO_METADATA_VERSION,
        "artifact_format": "murcko_mgf_dataset_collection_v1",
        "datasets": {},
    }
    for spec in _build_dataset_specs(
        nist_mgf=nist_mgf,
        mcebio_mgf=mcebio_mgf,
        nist_subdir=nist_subdir,
        mcebio_subdir=mcebio_subdir,
    ):
        raw_mgf = _stage_raw_mgf(spec.source, raw_dir, gcs_credentials)
        allowed_adducts = nist_allowed_adducts if spec.name == "nist" else None
        split_size_caps = nist_split_size_caps if spec.name == "nist" else None
        metadata = build_murcko_mgf_dataset(
            mgf_path=raw_mgf,
            output_dir=staging_root / spec.subdir.strip("/"),
            source_uri=spec.source,
            val_frac=val_frac,
            test_frac=test_frac,
            seed=seed,
            min_precursor_mz=min_precursor_mz,
            max_precursor_mz=max_precursor_mz,
            num_peaks_input=num_peaks_input,
            num_workers=num_workers,
            batch_size=batch_size,
            parquet_batch_size=parquet_batch_size,
            single_split=STANDALONE_SPLIT if spec.name == "mcebio" else None,
            allowed_adducts=allowed_adducts,
            split_size_caps=split_size_caps,
            spectral_lsh_threshold=spectral_lsh_threshold,
        )
        top_metadata["datasets"][spec.name] = {
            "subdir": spec.subdir.strip("/"),
            "source_raw_file": metadata["source_raw_file"],
            "splits": metadata["splits"],
        }
        for split in metadata["splits"]:
            top_metadata["datasets"][spec.name][f"{split}_size"] = metadata[
                f"{split}_size"
            ]
    (staging_root / "metadata.json").write_text(
        json.dumps(top_metadata, indent=2, sort_keys=True)
    )

    if build_dreams_auxiliary:
        from spectra_learning.data.murcko_dreams import build_murcko_dreams_auxiliary

        checkpoint = (
            dreams_checkpoint
            if dreams_checkpoint is not None
            else dreams_root / "dreams/models/pretrained/embedding_model.ckpt"
        )
        dreams_paths = build_murcko_dreams_auxiliary(
            artifact_dir=staging_root,
            dreams_root=dreams_root,
            checkpoint=checkpoint,
            subdirs=dreams_subdirs,
            n_highest_peaks=dreams_n_highest_peaks,
            batch_size=dreams_batch_size,
            device=dreams_device
            if dreams_device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu"),
            upload=False,
        )
        top_metadata = json.loads((staging_root / "metadata.json").read_text())
        top_metadata["dreams_auxiliary_built"] = True
        top_metadata["dreams_auxiliary_paths"] = [
            str(path.relative_to(staging_root)) for path in dreams_paths
        ]
        (staging_root / "metadata.json").write_text(
            json.dumps(top_metadata, indent=2, sort_keys=True)
        )

    if upload:
        api = HfApi()
        api.create_repo(
            hf_repo_id,
            repo_type="dataset",
            exist_ok=True,
            private=hf_private,
        )
        log.info("Uploading %s -> %s", staging_root, hf_repo_id)
        api.upload_large_folder(
            repo_id=hf_repo_id,
            folder_path=staging_root,
            repo_type="dataset",
            revision=hf_revision,
        )
        log.info(
            "Uploaded Murcko MGF dataset to https://huggingface.co/datasets/%s",
            hf_repo_id,
        )
    else:
        log.info("upload disabled; staged dataset left at %s", staging_root)
    top_metadata["artifact_dir"] = str(staging_root)
    return top_metadata


def prepare_nist_disjoint_probe_retrieval_collection(
    *,
    nist_mgf: str = str(DEFAULT_LOCAL_NIST_MGF_PATH),
    online_probe_subdir: str = NIST_DISJOINT_ONLINE_PROBE_SUBDIR,
    retrieval_pool_subdir: str = NIST_DISJOINT_RETRIEVAL_POOL_SUBDIR,
    same_inchi_subdir: str = NIST_10PPM_RETRIEVAL_SUBDIR,
    mces_subdir: str = NIST_MCES_RETRIEVAL_SUBDIR,
    gcs_credentials: Path | None = None,
    work_dir: Path,
    hf_repo_id: str = NIST_DISJOINT_PROBE_RETRIEVAL_HF_REPO,
    hf_revision: str = "main",
    hf_private: bool = False,
    upload: bool = True,
    online_probe_size: int = DEFAULT_ONLINE_PROBE_SAMPLE_SIZE,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    seed: int = 42,
    min_precursor_mz: float = 1.0,
    max_precursor_mz: float = 1000.0,
    num_peaks_input: int = NUM_PEAKS_INPUT,
    num_workers: int = os.cpu_count() or 1,
    batch_size: int = 2048,
    parquet_batch_size: int = 50_000,
    allowed_adducts: tuple[str, ...] | None = DEFAULT_NIST_ALLOWED_ADDUCTS,
    online_split_size_caps: dict[str, int] | None = DEFAULT_NIST_SPLIT_SIZE_CAPS,
    spectral_lsh_threshold: float = 0.90,
    same_inchi_pairs_per_class: int = DEFAULT_10PPM_RETRIEVAL_PAIRS_PER_CLASS,
    same_inchi_ppm: float = DEFAULT_10PPM_RETRIEVAL_PPM,
    same_inchi_adduct: str = DEFAULT_RETRIEVAL_ADDUCT,
    mces_pairs: int = DEFAULT_MCES_RETRIEVAL_PAIRS,
    mces_tanimoto_bin_size: float = DEFAULT_MCES_RETRIEVAL_BIN_SIZE,
    mces_workers: int = os.cpu_count() or 1,
) -> dict[str, Any]:
    """Build the fixed NIST benchmark collection requested for probing/retrieval.

    The collection has three task artifacts:

    - an online-probe Murcko split built from an exact 100k spectrum sample;
    - a fixed 10 ppm same-InChI binary retrieval pair table from the remaining
      Murcko-disjoint spectra;
    - a fixed MCES analog-search pair table with precomputed MCES distances.

    The latter two retrieval pair tables intentionally may reuse spectra and
    may overlap each other. The only enforced disjointness is between the
    online-probe Murcko histogram keys and the retrieval pool histogram keys.
    """
    work_dir = work_dir.expanduser().resolve()
    staging_root = work_dir / "artifact"
    raw_dir = staging_root / RAW_SUBDIR
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True, exist_ok=True)

    raw_mgf = _stage_raw_mgf(nist_mgf, raw_dir, gcs_credentials)
    first_rows = _first_pass(
        raw_mgf,
        min_precursor_mz=min_precursor_mz,
        max_precursor_mz=max_precursor_mz,
        allowed_adducts=allowed_adducts,
        num_workers=num_workers,
        batch_size=batch_size,
    )
    probe_indices, probe_hist_keys, probe_selection_metadata = _select_disjoint_probe_indices(
        first_rows,
        target_size=online_probe_size,
        seed=seed,
    )
    probe_first_rows = [
        row for row in first_rows if int(row.spectrum_index) in probe_indices
    ]
    retrieval_hist_keys = {
        row.murcko_hist_key for row in first_rows if row.murcko_hist_key not in probe_hist_keys
    }
    retrieval_first_rows = [
        row for row in first_rows if row.murcko_hist_key in retrieval_hist_keys
    ]
    if len(probe_first_rows) != online_probe_size:
        raise ValueError(
            f"Internal probe selection mismatch: expected {online_probe_size}, got {len(probe_first_rows)}."
        )
    if not retrieval_first_rows:
        raise ValueError("Murcko-disjoint retrieval pool is empty.")

    probe_fold_by_smiles, probe_split_metadata = _build_fold_map(
        probe_first_rows,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
    )
    online_metadata, _ = _build_subset_murcko_mgf_dataset(
        mgf_path=raw_mgf,
        output_dir=staging_root / online_probe_subdir.strip("/"),
        source_uri=nist_mgf,
        fold_by_smiles=probe_fold_by_smiles,
        split_metadata=probe_split_metadata,
        active_splits=SPLITS,
        min_precursor_mz=min_precursor_mz,
        max_precursor_mz=max_precursor_mz,
        num_peaks_input=num_peaks_input,
        num_workers=num_workers,
        batch_size=batch_size,
        parquet_batch_size=parquet_batch_size,
        allowed_adducts=allowed_adducts,
        split_size_caps=online_split_size_caps,
        spectral_lsh_threshold=spectral_lsh_threshold,
        include_spectrum_indices=probe_indices,
        extra_metadata={
            "subset_role": "online_probe",
            "selection_without_replacement": True,
            "online_probe_selection": probe_selection_metadata,
        },
    )

    retrieval_fold_by_smiles, retrieval_split_metadata = _build_single_split_fold_map(
        retrieval_first_rows,
        split=STANDALONE_SPLIT,
    )
    retrieval_metadata, retrieval_rows = _build_subset_murcko_mgf_dataset(
        mgf_path=raw_mgf,
        output_dir=staging_root / retrieval_pool_subdir.strip("/"),
        source_uri=nist_mgf,
        fold_by_smiles=retrieval_fold_by_smiles,
        split_metadata=retrieval_split_metadata,
        active_splits=(STANDALONE_SPLIT,),
        min_precursor_mz=min_precursor_mz,
        max_precursor_mz=max_precursor_mz,
        num_peaks_input=num_peaks_input,
        num_workers=num_workers,
        batch_size=batch_size,
        parquet_batch_size=parquet_batch_size,
        allowed_adducts=allowed_adducts,
        split_size_caps=None,
        spectral_lsh_threshold=spectral_lsh_threshold,
        include_murcko_hist_keys=retrieval_hist_keys,
        extra_metadata={
            "subset_role": "retrieval_pool",
            "murcko_hist_disjoint_from_online_probe": True,
            "excluded_online_probe_murcko_hist_keys": len(probe_hist_keys),
        },
        collect_retrieval_rows=True,
    )

    retrieval_pool_hist_keys = {row.murcko_hist_key for row in retrieval_rows}
    overlap = probe_hist_keys & retrieval_pool_hist_keys
    if overlap:
        raise ValueError(
            "Retrieval pool is not Murcko-disjoint from online probe; overlapping keys: "
            + ", ".join(sorted(overlap)[:8])
        )

    same_inchi_metadata = _write_same_inchi_retrieval_dataset(
        output_dir=staging_root / same_inchi_subdir.strip("/"),
        retrieval_rows=retrieval_rows,
        pairs_per_class=same_inchi_pairs_per_class,
        ppm=same_inchi_ppm,
        adduct=same_inchi_adduct,
        seed=seed + 11,
        retrieval_pool_subdir=retrieval_pool_subdir.strip("/"),
    )
    mces_metadata = _write_mces_retrieval_dataset(
        output_dir=staging_root / mces_subdir.strip("/"),
        retrieval_rows=retrieval_rows,
        num_pairs=mces_pairs,
        bin_size=mces_tanimoto_bin_size,
        seed=seed + 23,
        workers=mces_workers,
        retrieval_pool_subdir=retrieval_pool_subdir.strip("/"),
    )

    top_metadata: dict[str, Any] = {
        "metadata_version": 1,
        "artifact_format": NIST_DISJOINT_PROBE_RETRIEVAL_ARTIFACT_FORMAT,
        "hf_repo_id": hf_repo_id,
        "hf_revision": hf_revision,
        "source_uri": nist_mgf,
        "source_raw_file": f"{RAW_SUBDIR}/{raw_mgf.name}",
        "seed": int(seed),
        "allowed_adducts": list(allowed_adducts) if allowed_adducts is not None else None,
        "min_precursor_mz": float(min_precursor_mz),
        "max_precursor_mz": float(max_precursor_mz),
        "online_probe": {
            "subdir": online_probe_subdir.strip("/"),
            "train_size": int(online_metadata.get("train_size", 0)),
            "val_size": int(online_metadata.get("val_size", 0)),
            "test_size": int(online_metadata.get("test_size", 0)),
            "selected_spectra_before_split_processing": int(online_probe_size),
        },
        "retrieval_pool": {
            "subdir": retrieval_pool_subdir.strip("/"),
            "all_size": int(retrieval_metadata.get("all_size", 0)),
            "murcko_hist_keys": int(len(retrieval_pool_hist_keys)),
        },
        "same_inchi14_10ppm": {
            "subdir": same_inchi_subdir.strip("/"),
            "num_pairs": int(same_inchi_metadata["num_pairs"]),
            "positive_pairs": int(same_inchi_metadata["positive_pairs"]),
            "negative_pairs": int(same_inchi_metadata["negative_pairs"]),
            "ppm": float(same_inchi_ppm),
            "selection_adduct": same_inchi_adduct,
        },
        "mces_analog": {
            "subdir": mces_subdir.strip("/"),
            "num_pairs": int(mces_metadata["num_pairs"]),
            "mces_threshold": int(MCES_THRESHOLD),
            "num_workers": int(mces_metadata.get("num_workers", mces_workers)),
        },
        "disjointness": {
            "online_probe_selected_murcko_hist_keys": int(len(probe_hist_keys)),
            "retrieval_pool_murcko_hist_keys": int(len(retrieval_pool_hist_keys)),
            "online_probe_retrieval_murcko_hist_overlap": int(len(overlap)),
            "selected_murcko_hist_key_values": sorted(probe_hist_keys),
        },
    }
    (staging_root / "metadata.json").write_text(
        json.dumps(top_metadata, indent=2, sort_keys=True)
    )
    _write_nist_disjoint_probe_retrieval_readme(
        staging_root / "README.md",
        metadata=top_metadata,
    )

    if upload:
        api = HfApi()
        api.create_repo(
            hf_repo_id,
            repo_type="dataset",
            exist_ok=True,
            private=hf_private,
        )
        log.info("Uploading %s -> %s", staging_root, hf_repo_id)
        api.upload_large_folder(
            repo_id=hf_repo_id,
            folder_path=staging_root,
            repo_type="dataset",
            revision=hf_revision,
        )
        log.info(
            "Uploaded disjoint NIST probe/retrieval collection to https://huggingface.co/datasets/%s",
            hf_repo_id,
        )
    else:
        log.info("upload disabled; staged dataset left at %s", staging_root)
    top_metadata["artifact_dir"] = str(staging_root)
    return top_metadata


def main() -> None:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    parser = argparse.ArgumentParser(
        description=(
            "Build Murcko-split NIST and standalone MCEBIO Parquet datasets from raw MGF."
        )
    )
    parser.add_argument("--nist-mgf", default=DEFAULT_NIST_MGF_URI)
    parser.add_argument("--mcebio-mgf", default=str(DEFAULT_MCEBIO_MGF_PATH))
    parser.add_argument("--nist-subdir", default=NIST_MURCKO_PREPARED_SUBDIR)
    parser.add_argument("--mcebio-subdir", default=MCEBIO_MURCKO_PREPARED_SUBDIR)
    parser.add_argument("--gcs-credentials", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--hf-repo-id", default=NIST_MURCKO_HF_REPO)
    parser.add_argument("--hf-revision", default="main")
    parser.add_argument("--hf-private", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--test-frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-precursor-mz", type=float, default=1.0)
    parser.add_argument("--max-precursor-mz", type=float, default=1000.0)
    parser.add_argument("--num-peaks-input", type=int, default=NUM_PEAKS_INPUT)
    parser.add_argument("--num-workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--parquet-batch-size", type=int, default=50_000)
    parser.add_argument(
        "--nist-allowed-adducts",
        nargs="*",
        default=list(DEFAULT_NIST_ALLOWED_ADDUCTS),
        help="Allowed NIST PRECURSORTYPE values. Pass the flag with no values to disable filtering.",
    )
    parser.add_argument(
        "--nist-target-train-size",
        type=int,
        default=DEFAULT_NIST_SPLIT_SIZE_CAPS["train"],
    )
    parser.add_argument(
        "--nist-target-val-size",
        type=int,
        default=DEFAULT_NIST_SPLIT_SIZE_CAPS["val"],
    )
    parser.add_argument(
        "--nist-target-test-size",
        type=int,
        default=DEFAULT_NIST_SPLIT_SIZE_CAPS["test"],
    )
    parser.add_argument("--spectral-lsh-threshold", type=float, default=0.90)
    parser.add_argument("--build-dreams-auxiliary", action="store_true")
    parser.add_argument("--dreams-root", type=Path, default=Path("/home/wuhao/Dreams"))
    parser.add_argument("--dreams-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--dreams-subdir",
        action="append",
        default=None,
        help="Prepared subdir for DreaMS auxiliary generation. Defaults to NIST and MCEBIO.",
    )
    parser.add_argument("--dreams-n-highest-peaks", type=int, default=100)
    parser.add_argument("--dreams-batch-size", type=int, default=256)
    parser.add_argument("--dreams-device", default=None)
    parser.add_argument(
        "--build-disjoint-probe-retrieval",
        action="store_true",
        help=(
            "Build the fixed NIST collection with a 100k Murcko-disjoint online "
            "probe sample, a retrieval pool, 10 ppm pair labels, and MCES pairs."
        ),
    )
    parser.add_argument("--online-probe-subdir", default=NIST_DISJOINT_ONLINE_PROBE_SUBDIR)
    parser.add_argument("--retrieval-pool-subdir", default=NIST_DISJOINT_RETRIEVAL_POOL_SUBDIR)
    parser.add_argument("--same-inchi-subdir", default=NIST_10PPM_RETRIEVAL_SUBDIR)
    parser.add_argument("--mces-subdir", default=NIST_MCES_RETRIEVAL_SUBDIR)
    parser.add_argument("--online-probe-size", type=int, default=DEFAULT_ONLINE_PROBE_SAMPLE_SIZE)
    parser.add_argument(
        "--same-inchi-pairs-per-class",
        type=int,
        default=DEFAULT_10PPM_RETRIEVAL_PAIRS_PER_CLASS,
    )
    parser.add_argument("--same-inchi-ppm", type=float, default=DEFAULT_10PPM_RETRIEVAL_PPM)
    parser.add_argument("--same-inchi-adduct", default=DEFAULT_RETRIEVAL_ADDUCT)
    parser.add_argument("--mces-pairs", type=int, default=DEFAULT_MCES_RETRIEVAL_PAIRS)
    parser.add_argument(
        "--mces-tanimoto-bin-size",
        type=float,
        default=DEFAULT_MCES_RETRIEVAL_BIN_SIZE,
    )
    parser.add_argument("--mces-workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()
    allowed_adducts = (
        tuple(args.nist_allowed_adducts) if args.nist_allowed_adducts else None
    )
    if args.build_disjoint_probe_retrieval:
        nist_mgf = args.nist_mgf
        if nist_mgf == DEFAULT_NIST_MGF_URI:
            nist_mgf = str(DEFAULT_LOCAL_NIST_MGF_PATH)
        prepare_nist_disjoint_probe_retrieval_collection(
            nist_mgf=nist_mgf,
            online_probe_subdir=args.online_probe_subdir,
            retrieval_pool_subdir=args.retrieval_pool_subdir,
            same_inchi_subdir=args.same_inchi_subdir,
            mces_subdir=args.mces_subdir,
            gcs_credentials=args.gcs_credentials,
            work_dir=args.work_dir,
            hf_repo_id=args.hf_repo_id,
            hf_revision=args.hf_revision,
            hf_private=args.hf_private,
            upload=not args.skip_upload,
            online_probe_size=args.online_probe_size,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            seed=args.seed,
            min_precursor_mz=args.min_precursor_mz,
            max_precursor_mz=args.max_precursor_mz,
            num_peaks_input=args.num_peaks_input,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            parquet_batch_size=args.parquet_batch_size,
            allowed_adducts=allowed_adducts,
            online_split_size_caps={
                "train": args.nist_target_train_size,
                "val": args.nist_target_val_size,
                "test": args.nist_target_test_size,
            },
            spectral_lsh_threshold=args.spectral_lsh_threshold,
            same_inchi_pairs_per_class=args.same_inchi_pairs_per_class,
            same_inchi_ppm=args.same_inchi_ppm,
            same_inchi_adduct=args.same_inchi_adduct,
            mces_pairs=args.mces_pairs,
            mces_tanimoto_bin_size=args.mces_tanimoto_bin_size,
            mces_workers=args.mces_workers,
        )
        return
    prepare_murcko_mgf_collection(
        nist_mgf=args.nist_mgf,
        mcebio_mgf=args.mcebio_mgf,
        nist_subdir=args.nist_subdir,
        mcebio_subdir=args.mcebio_subdir,
        gcs_credentials=args.gcs_credentials,
        work_dir=args.work_dir,
        hf_repo_id=args.hf_repo_id,
        hf_revision=args.hf_revision,
        hf_private=args.hf_private,
        upload=not args.skip_upload,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        min_precursor_mz=args.min_precursor_mz,
        max_precursor_mz=args.max_precursor_mz,
        num_peaks_input=args.num_peaks_input,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        parquet_batch_size=args.parquet_batch_size,
        nist_allowed_adducts=allowed_adducts,
        nist_split_size_caps={
            "train": args.nist_target_train_size,
            "val": args.nist_target_val_size,
            "test": args.nist_target_test_size,
        },
        spectral_lsh_threshold=args.spectral_lsh_threshold,
        build_dreams_auxiliary=args.build_dreams_auxiliary,
        dreams_root=args.dreams_root,
        dreams_checkpoint=args.dreams_checkpoint,
        dreams_subdirs=args.dreams_subdir,
        dreams_n_highest_peaks=args.dreams_n_highest_peaks,
        dreams_batch_size=args.dreams_batch_size,
        dreams_device=args.dreams_device,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
