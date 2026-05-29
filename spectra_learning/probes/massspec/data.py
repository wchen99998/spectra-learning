import csv
import json
import logging
import math
from pathlib import Path
from typing import Any, NamedTuple, cast

import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from ml_collections import config_dict
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler

from spectra_learning.probes.massspec.targets import (
    MACCS_FINGERPRINT_BITS,
    MORGAN_PROBE_FINGERPRINT_BITS,
    MORGAN_PROBE_FINGERPRINT_RADIUS,
    REGRESSION_TARGET_KEYS,
    build_maccs_targets_for_rows,
    build_morgan_targets_for_rows,
    build_probe_targets_for_rows,
)
from spectra_learning.data.spectra import (
    DEFAULT_MAX_PRECURSOR_MZ,
    DEFAULT_MIN_PEAK_INTENSITY,
    NUM_PEAKS_INPUT,
    preprocess_peak_batch_torch,
)

logger = logging.getLogger(__name__)

_DEFAULT_BATCH_SIZE = 512
_DEFAULT_SHUFFLE_BUFFER = 10_000
_DEFAULT_ARTIFACT_DIR = Path("data/gems_artifacts")
_DEFAULT_MASSSPEC_NUM_SHARDS = 4
_DEFAULT_PRECURSOR_PEAK_EXCLUSION_WINDOW_DA = 0.0
_NUM_PEAKS_OUTPUT = 60
_FINGERPRINT_BITS = 1024
_FINGERPRINT_RADIUS = 2
_METADATA_FILENAME = "metadata.json"

MASSSPEC_HF_REPO = "roman-bushuiev/MassSpecGym"
MASSSPEC_TSV_PATH = "data/MassSpecGym.tsv"
MASSSPEC_METADATA_VERSION = 4

NIST20_METADATA_VERSION = 5
NIST20_HF_REPO = "roman-bushuiev/GeMS"
NIST20_HF_FILENAME = (
    "data/DreaMS_Atlas/nist20_mona_clean_merged_spectra_dreams_hidden_nist20.hdf5"
)
_NIST20_SPLIT_SEED = 42
_NIST20_TRAIN_FRAC = 0.70
_NIST20_VAL_FRAC = 0.15
NIST_FULL_METADATA_VERSION = 3
NIST_FULL_HF_FILENAME = "hr_msms_nist.hdf5"
NIST_FULL_ARTIFACT_FORMAT = "nist_full_probe_v3"
NIST_FULL_PAIRWISE_ALIGNMENT_FILENAME = "morgan_tanimoto_balanced_pairs.npz"
NIST_FULL_PAIRWISE_ALIGNMENT_NUM_PAIRS = 20_000
NIST_FULL_PAIRWISE_ALIGNMENT_BIN_SIZE = 0.025
NIST_FULL_PAIRWISE_ALIGNMENT_SEED = 66
NIST_MURCKO_METADATA_VERSION = 2
NIST_MURCKO_HF_REPO = "cjim8889/hr_msms_nist_mcebio_murcko_20260529"
NIST_MURCKO_PREPARED_SUBDIR = "nist_murcko_probe"
MCEBIO_MURCKO_PREPARED_SUBDIR = "mcebio_murcko_probe"
NIST_MURCKO_ARTIFACT_FORMAT = "nist_murcko_parquet_v2"

MONA_A_METADATA_VERSION = 3
MONA_A_HF_REPO = "roman-bushuiev/GeMS"
MONA_A_HF_FILENAME = (
    "data/auxiliary/MoNA_A_Murcko_split_neighbours_[M+H]+_0.05Da.pkl"
)


def _config_get(config: config_dict.ConfigDict, key: str, default: Any) -> Any:
    return config.get(key, default)


def _download_hf_file(repo_id: str, filename: str, local_dir: Path) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=str(local_dir),
    )
    return Path(path)


def _coordinate_distributed_download(distributed_world_size: int) -> bool:
    return (
        distributed_world_size > 1
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    )


def _snapshot_download_rank_zero(
    *,
    repo_id: str,
    repo_type: str,
    revision: str,
    local_dir: Path,
    allow_patterns: list[str],
    distributed_world_size: int,
    distributed_rank: int,
) -> None:
    coordinated = _coordinate_distributed_download(distributed_world_size)
    if not coordinated or distributed_rank == 0:
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            local_dir=local_dir,
            allow_patterns=allow_patterns,
        )
    if coordinated:
        torch.distributed.barrier()


def download_massspec_tsv(output_dir: Path) -> Path:
    return _download_hf_file(MASSSPEC_HF_REPO, MASSSPEC_TSV_PATH, output_dir / "data")


def _normalize_spectra_intensity(spectra: np.ndarray) -> np.ndarray:
    max_int = spectra[:, 1].max(axis=1, keepdims=True)
    np.divide(spectra[:, 1], np.maximum(max_int, 1e-8), out=spectra[:, 1])
    return spectra


def _load_massspec_tsv(tsv_path: Path) -> dict[str, Any]:
    spectra, precursor, fold, smiles = [], [], [], []
    adduct, instrument_type = [], []
    collision_energy, collision_energy_present = [], []
    with tsv_path.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            mz = np.fromstring(row["mzs"], sep=",", dtype=np.float32)
            intensity = np.fromstring(row["intensities"], sep=",", dtype=np.float32)
            if mz.size > NUM_PEAKS_INPUT:
                idx = np.argpartition(intensity, -NUM_PEAKS_INPUT)[-NUM_PEAKS_INPUT:]
                idx = idx[np.argsort(intensity[idx])[::-1]]
                mz = mz[idx]
                intensity = intensity[idx]
            elif mz.size < NUM_PEAKS_INPUT:
                pad = NUM_PEAKS_INPUT - mz.size
                mz = np.pad(mz, (0, pad))
                intensity = np.pad(intensity, (0, pad))
            spectra.append(np.stack([mz, intensity], axis=0))
            precursor.append(float(row["precursor_mz"]))
            fold.append(row["fold"])
            smiles.append(row["smiles"])
            adduct.append(row["adduct"] or "unknown")
            instrument_type.append(row["instrument_type"] or "unknown")
            ce = row["collision_energy"]
            collision_energy.append(float(ce) if ce else 0.0)
            collision_energy_present.append(1 if ce else 0)
    return {
        "spectra": _normalize_spectra_intensity(np.stack(spectra, axis=0)),
        "precursor": np.asarray(precursor, dtype=np.float32),
        "fold": np.asarray(fold),
        "smiles": np.asarray(smiles, dtype=str),
        "adduct": np.asarray(adduct, dtype=str),
        "instrument_type": np.asarray(instrument_type, dtype=str),
        "collision_energy": np.asarray(collision_energy, dtype=np.float32),
        "collision_energy_present": np.asarray(
            collision_energy_present, dtype=np.int32
        ),
        "dreams_embedding": None,
    }


def _load_nist20_hdf5(hdf5_path: Path) -> dict[str, Any]:
    import h5py
    from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey

    with h5py.File(str(hdf5_path), "r") as f:
        raw_spectra = f["spectrum"][:]
        raw_precursor = f["precursor_mz"][:]
        raw_smiles = f["smiles"][:].astype(str)
        raw_adduct = f["adduct"][:].astype(str)
        raw_dreams = (
            f["DreaMS_embedding"][:].astype(np.float32)
            if "DreaMS_embedding" in f
            else None
        )
    valid_indices, inchikey_14 = [], []
    for i, smi in enumerate(raw_smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        ik = InchiToInchiKey(MolToInchi(mol))
        if ik is None:
            continue
        valid_indices.append(i)
        inchikey_14.append(ik[:14])
    idx = np.asarray(valid_indices)
    keys = np.asarray(inchikey_14, dtype=str)
    logger.info("NIST20 HDF5: %d / %d spectra have valid SMILES", len(idx), len(raw_smiles))
    unique_keys = sorted(set(keys.tolist()))
    rng = np.random.RandomState(_NIST20_SPLIT_SEED)
    rng.shuffle(unique_keys)
    n_keys = len(unique_keys)
    n_train = int(n_keys * _NIST20_TRAIN_FRAC)
    n_val = int(n_keys * _NIST20_VAL_FRAC)
    train_keys = set(unique_keys[:n_train])
    val_keys = set(unique_keys[n_train : n_train + n_val])
    fold = np.where(
        np.isin(keys, list(train_keys)),
        "train",
        np.where(np.isin(keys, list(val_keys)), "val", "test"),
    )
    n_valid = len(idx)
    return {
        "spectra": _normalize_spectra_intensity(raw_spectra[idx].astype(np.float32)),
        "precursor": raw_precursor[idx].astype(np.float32),
        "fold": fold,
        "smiles": raw_smiles[idx].astype(str),
        "adduct": raw_adduct[idx].astype(str),
        "instrument_type": np.repeat("unknown", n_valid).astype(str),
        "collision_energy": np.zeros(n_valid, dtype=np.float32),
        "collision_energy_present": np.zeros(n_valid, dtype=np.int32),
        "dreams_embedding": raw_dreams[idx] if raw_dreams is not None else None,
    }


def _load_mona_a_pkl(pkl_path: Path) -> dict[str, Any]:
    import pickle
    import pandas as pd

    class _SafeUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            try:
                return super().find_class(module, name)
            except (ImportError, AttributeError):
                class Stub:
                    def __init__(self, *a, **kw):
                        pass

                    def __setstate__(self, state):
                        self.__dict__.update(
                            state if isinstance(state, dict) else {"_state": state}
                        )

                Stub.__name__ = name
                return Stub

    with open(pkl_path, "rb") as f:
        raw = _SafeUnpickler(f).load()
    df = pd.DataFrame.__new__(pd.DataFrame)
    df.__dict__.update(raw.__dict__)
    n = len(df)
    return {
        "spectra": _normalize_spectra_intensity(
            np.stack(list(df["PARSED PEAKS"].to_numpy())).astype(np.float32)
        ),
        "precursor": df["PRECURSOR M/Z"].to_numpy().astype(np.float32),
        "fold": np.where(df["val"].to_numpy(), "test", "train"),
        "smiles": df["SMILES"].to_numpy().astype(str),
        "adduct": df["PRECURSOR TYPE"].fillna("unknown").to_numpy().astype(str),
        "instrument_type": df["INSTRUMENT TYPE"].fillna("unknown").to_numpy().astype(str),
        "collision_energy": np.zeros(n, dtype=np.float32),
        "collision_energy_present": np.zeros(n, dtype=np.int32),
        "dreams_embedding": None,
    }


def _compute_morgan_fingerprints(smiles: np.ndarray) -> np.ndarray:
    fps = np.zeros((len(smiles), _FINGERPRINT_BITS), dtype=np.int8)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(  # type: ignore[attr-defined]
            mol,
            _FINGERPRINT_RADIUS,
            nBits=_FINGERPRINT_BITS,
        )
        DataStructs.ConvertToNumpyArray(fp, fps[i])
    return fps


def _representative_canonical_smiles(
    smiles: np.ndarray,
    valid_mol: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    raw_reps: dict[str, int] = {}
    for idx, (smi, valid) in enumerate(zip(smiles, valid_mol, strict=True)):
        if not bool(valid):
            continue
        raw = str(smi)
        if raw not in raw_reps:
            raw_reps[raw] = idx

    reps: dict[str, int] = {}
    for raw, idx in raw_reps.items():
        mol = Chem.MolFromSmiles(raw)
        canonical = Chem.MolToSmiles(mol)
        if canonical not in reps:
            reps[canonical] = idx
    canonical_smiles = list(reps.keys())
    return (
        np.asarray([reps[smi] for smi in canonical_smiles], dtype=np.int64),
        canonical_smiles,
    )


def _sample_balanced_morgan_pairs(
    smiles: np.ndarray,
    valid_mol: np.ndarray,
    *,
    num_pairs: int = NIST_FULL_PAIRWISE_ALIGNMENT_NUM_PAIRS,
    bin_size: float = NIST_FULL_PAIRWISE_ALIGNMENT_BIN_SIZE,
    seed: int = NIST_FULL_PAIRWISE_ALIGNMENT_SEED,
) -> dict[str, np.ndarray] | None:
    rep_indices, rep_smiles = _representative_canonical_smiles(smiles, valid_mol)
    if len(rep_smiles) < 2:
        return None

    fps = [
        AllChem.GetMorganFingerprintAsBitVect(  # type: ignore[attr-defined]
            Chem.MolFromSmiles(smi),
            radius=MORGAN_PROBE_FINGERPRINT_RADIUS,
            nBits=MORGAN_PROBE_FINGERPRINT_BITS,
        )
        for smi in rep_smiles
    ]
    n_bins = round(1.0 / bin_size)
    target_per_bin = num_pairs // n_bins
    rng = np.random.default_rng(seed)
    all_rep = np.arange(len(fps), dtype=np.int64)
    pairs_by_bin: list[list[tuple[int, int, float]]] = [[] for _ in range(n_bins)]

    for i in rng.permutation(len(fps)):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fps[int(i)], fps), dtype=np.float32)
        bin_ids = np.ceil(sims / bin_size).astype(np.int16) - 1
        for bin_idx in range(n_bins):
            need = target_per_bin - len(pairs_by_bin[bin_idx])
            if need <= 0:
                continue
            candidates = all_rep[(bin_ids == bin_idx) & (all_rep != int(i))]
            if candidates.size:
                take = rng.choice(
                    candidates,
                    size=min(need, candidates.size),
                    replace=False,
                )
                pairs_by_bin[bin_idx].extend(
                    (int(i), int(j), float(sims[j])) for j in take
                )
        if all(len(bucket) >= target_per_bin for bucket in pairs_by_bin):
            break

    pairs = [pair for bucket in pairs_by_bin for pair in bucket[:target_per_bin]]
    if len(pairs) != target_per_bin * n_bins:
        return None

    left_index = rep_indices[[pair[0] for pair in pairs]]
    right_index = rep_indices[[pair[1] for pair in pairs]]
    endpoint_index, inverse = np.unique(
        np.concatenate([left_index, right_index]),
        return_inverse=True,
    )
    return {
        "left_index": left_index.astype(np.int64),
        "right_index": right_index.astype(np.int64),
        "left_endpoint": inverse[: len(left_index)].astype(np.int64),
        "right_endpoint": inverse[len(left_index) :].astype(np.int64),
        "endpoint_index": endpoint_index.astype(np.int64),
        "tanimoto": np.asarray([pair[2] for pair in pairs], dtype=np.float32),
        "bin_counts": np.asarray(
            [len(bucket[:target_per_bin]) for bucket in pairs_by_bin],
            dtype=np.int32,
        ),
    }


def _write_pairwise_alignment_artifact(
    *,
    output_dir: Path,
    smiles: np.ndarray,
    valid_mol: np.ndarray,
) -> dict[str, Any]:
    payload = _sample_balanced_morgan_pairs(smiles, valid_mol)
    if payload is None:
        return {
            "pairwise_alignment_available": False,
            "pairwise_alignment_num_pairs": 0,
        }
    np.savez_compressed(
        output_dir / NIST_FULL_PAIRWISE_ALIGNMENT_FILENAME,
        **payload,
        bin_size=np.asarray(NIST_FULL_PAIRWISE_ALIGNMENT_BIN_SIZE, dtype=np.float32),
        seed=np.asarray(NIST_FULL_PAIRWISE_ALIGNMENT_SEED, dtype=np.int64),
    )
    return {
        "pairwise_alignment_available": True,
        "pairwise_alignment_file": NIST_FULL_PAIRWISE_ALIGNMENT_FILENAME,
        "pairwise_alignment_num_pairs": len(payload["tanimoto"]),
        "pairwise_alignment_num_endpoints": len(payload["endpoint_index"]),
        "pairwise_alignment_bin_size": NIST_FULL_PAIRWISE_ALIGNMENT_BIN_SIZE,
        "pairwise_alignment_seed": int(NIST_FULL_PAIRWISE_ALIGNMENT_SEED),
        "pairwise_alignment_fingerprint": "morgan",
        "pairwise_alignment_morgan_bits": int(MORGAN_PROBE_FINGERPRINT_BITS),
        "pairwise_alignment_morgan_radius": int(MORGAN_PROBE_FINGERPRINT_RADIUS),
    }


def _encode_categorical_ids(values: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    normalized = np.asarray([str(v) or "unknown" for v in values], dtype=str)
    categories = ["unknown"] + sorted(set(normalized.tolist()) - {"unknown"})
    vocab = {category: i for i, category in enumerate(categories)}
    return np.asarray([vocab[v] for v in normalized], dtype=np.int32), vocab


def _write_probe_native_shards(
    split_payload: dict[str, np.ndarray],
    output_path: Path,
    num_shards: int,
) -> tuple[list[str], list[int]]:
    n = len(split_payload["spectra"])
    num_shards = max(1, min(num_shards, n))
    shard_size = math.ceil(n / num_shards)
    output_path.mkdir(parents=True, exist_ok=True)
    shard_names, shard_lengths = [], []
    for shard_id in range(num_shards):
        start = shard_id * shard_size
        end = min(start + shard_size, n)
        if start >= end:
            break
        shard_name = f"shard-{shard_id:05d}-of-{num_shards:05d}"
        shard_dir = output_path / shard_name
        shard_dir.mkdir(parents=True, exist_ok=True)
        for key, value in split_payload.items():
            np.save(shard_dir / f"{key}.npy", value[start:end])
        shard_names.append(shard_name)
        shard_lengths.append(end - start)
    return shard_names, shard_lengths


def _filter_encode_and_write(
    *,
    spectra: np.ndarray,
    precursor: np.ndarray,
    fold: np.ndarray,
    smiles: np.ndarray,
    adduct: np.ndarray,
    instrument_type: np.ndarray,
    collision_energy: np.ndarray,
    collision_energy_present: np.ndarray,
    dreams_embedding: np.ndarray | None = None,
    output_dir: Path,
    num_shards: int,
    max_precursor_mz: float,
    metadata_version: int,
    write_pairwise_alignment: bool = True,
) -> dict[str, Any]:
    keep = np.isfinite(precursor) & (precursor <= max_precursor_mz)
    spectra = spectra[keep]
    precursor = precursor[keep]
    fold = fold[keep]
    smiles = smiles[keep]
    adduct = adduct[keep]
    instrument_type = instrument_type[keep]
    collision_energy = collision_energy[keep]
    collision_energy_present = collision_energy_present[keep]
    if dreams_embedding is not None:
        dreams_embedding = dreams_embedding[keep]
    fingerprints = _compute_morgan_fingerprints(smiles)
    adduct_id, adduct_vocab = _encode_categorical_ids(adduct)
    instrument_type_id, instrument_type_vocab = _encode_categorical_ids(instrument_type)
    probe_mol_props, _, probe_valid_mol = build_probe_targets_for_rows(smiles)
    probe_maccs, probe_maccs_valid = build_maccs_targets_for_rows(smiles)
    probe_morgan, probe_morgan_valid = build_morgan_targets_for_rows(smiles)
    probe_valid_mol &= probe_maccs_valid & probe_morgan_valid
    metadata: dict[str, Any] = {
        "metadata_version": metadata_version,
        "max_precursor_mz": max_precursor_mz,
        "adduct_vocab": adduct_vocab,
        "instrument_type_vocab": instrument_type_vocab,
        "dreams_dim": (
            int(dreams_embedding.shape[1]) if dreams_embedding is not None else 0
        ),
        "probe_maccs_bits": MACCS_FINGERPRINT_BITS,
        "probe_morgan_bits": MORGAN_PROBE_FINGERPRINT_BITS,
        "probe_morgan_radius": MORGAN_PROBE_FINGERPRINT_RADIUS,
    }
    split_ordered_smiles, split_ordered_valid = [], []
    for split_name in ("train", "val", "test"):
        split_mask = fold == split_name
        payload = {
            "spectra": spectra[split_mask].astype(np.float32),
            "precursor_mz_raw": precursor[split_mask].astype(np.float32),
            "fingerprint": fingerprints[split_mask].astype(np.int8),
            "smiles": smiles[split_mask].astype(str),
            "adduct_id": adduct_id[split_mask].astype(np.int32),
            "instrument_type_id": instrument_type_id[split_mask].astype(np.int32),
            "collision_energy": collision_energy[split_mask].astype(np.float32),
            "collision_energy_present": collision_energy_present[split_mask].astype(
                np.int32
            ),
            "probe_valid_mol": probe_valid_mol[split_mask].astype(bool),
            "probe_maccs": probe_maccs[split_mask].astype(np.int8),
            "probe_morgan": probe_morgan[split_mask].astype(np.int8),
        }
        for name in REGRESSION_TARGET_KEYS:
            payload[f"probe_{name}"] = probe_mol_props[name][split_mask].astype(
                np.float32
            )
        if dreams_embedding is not None:
            payload["dreams_embedding"] = dreams_embedding[split_mask].astype(np.float32)
        shard_names, shard_lengths = _write_probe_native_shards(
            payload,
            output_dir / split_name,
            max(1, num_shards // 2 if split_name == "train" else num_shards // 4),
        )
        metadata[f"{split_name}_files"] = shard_names
        metadata[f"{split_name}_lengths"] = shard_lengths
        metadata[f"{split_name}_size"] = np.count_nonzero(split_mask)
        split_ordered_smiles.append(smiles[split_mask].astype(str))
        split_ordered_valid.append(probe_valid_mol[split_mask].astype(bool))
    if write_pairwise_alignment:
        metadata.update(
            _write_pairwise_alignment_artifact(
                output_dir=output_dir,
                smiles=np.concatenate(split_ordered_smiles),
                valid_mol=np.concatenate(split_ordered_valid),
            )
        )
    else:
        metadata.update(
            {
                "pairwise_alignment_available": False,
                "pairwise_alignment_num_pairs": 0,
                "pairwise_alignment_num_endpoints": 0,
            }
        )
    return metadata


def _probe_metadata_valid(
    output_dir: Path,
    expected_version: int,
    max_precursor_mz: float,
    expected_metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    metadata_path = output_dir / _METADATA_FILENAME
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text())
    if int(metadata.get("metadata_version", 0)) != expected_version:
        return None
    if float(metadata.get("max_precursor_mz", float("inf"))) != max_precursor_mz:
        return None
    if expected_metadata is not None:
        for key, value in expected_metadata.items():
            if metadata.get(key) != value:
                return None
    storage_format = str(metadata.get("storage_format", "native"))
    for split in ("train", "val", "test"):
        if storage_format == "parquet":
            if not all((output_dir / name).exists() for name in metadata.get(f"{split}_files", [])):
                return None
        elif not all(
            (output_dir / split / name).exists()
            for name in metadata.get(f"{split}_files", [])
        ):
            return None
    return metadata


def ensure_massspec_probe_prepared(
    output_dir: Path,
    *,
    max_precursor_mz: float,
    num_shards: int = _DEFAULT_MASSSPEC_NUM_SHARDS,
) -> dict[str, Any]:
    cached = _probe_metadata_valid(
        output_dir, MASSSPEC_METADATA_VERSION, max_precursor_mz
    )
    if cached is not None:
        return cached
    output_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = download_massspec_tsv(output_dir)
    metadata = _filter_encode_and_write(
        **_load_massspec_tsv(tsv_path),
        output_dir=output_dir,
        num_shards=num_shards,
        max_precursor_mz=max_precursor_mz,
        metadata_version=MASSSPEC_METADATA_VERSION,
    )
    (output_dir / _METADATA_FILENAME).write_text(json.dumps(metadata, indent=2))
    return metadata


def ensure_nist20_probe_prepared(
    output_dir: Path,
    *,
    max_precursor_mz: float,
    cache_dir: Path,
    num_shards: int = _DEFAULT_MASSSPEC_NUM_SHARDS,
    hdf5_repo_id: str = NIST20_HF_REPO,
    hdf5_filename: str = NIST20_HF_FILENAME,
) -> dict[str, Any]:
    cached = _probe_metadata_valid(
        output_dir,
        NIST20_METADATA_VERSION,
        max_precursor_mz,
        expected_metadata={
            "hdf5_repo_id": hdf5_repo_id,
            "hdf5_filename": hdf5_filename,
        },
    )
    if cached is not None:
        return cached
    hdf5_path = cache_dir / hdf5_filename
    if not hdf5_path.exists():
        hdf5_path = _download_hf_file(hdf5_repo_id, hdf5_filename, cache_dir)
    metadata = _filter_encode_and_write(
        **_load_nist20_hdf5(hdf5_path),
        output_dir=output_dir,
        num_shards=num_shards,
        max_precursor_mz=max_precursor_mz,
        metadata_version=NIST20_METADATA_VERSION,
    )
    metadata["hdf5_repo_id"] = hdf5_repo_id
    metadata["hdf5_filename"] = hdf5_filename
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / _METADATA_FILENAME).write_text(json.dumps(metadata, indent=2))
    return metadata


def build_nist_full_probe_artifact(
    hdf5_path: Path,
    output_dir: Path,
    *,
    max_precursor_mz: float,
    num_shards: int = _DEFAULT_MASSSPEC_NUM_SHARDS,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = _filter_encode_and_write(
        **_load_nist20_hdf5(hdf5_path),
        output_dir=output_dir,
        num_shards=num_shards,
        max_precursor_mz=max_precursor_mz,
        metadata_version=NIST_FULL_METADATA_VERSION,
    )
    metadata["artifact_format"] = NIST_FULL_ARTIFACT_FORMAT
    metadata["source_hdf5_filename"] = hdf5_path.name
    if extra_metadata is not None:
        metadata.update(extra_metadata)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / _METADATA_FILENAME).write_text(json.dumps(metadata, indent=2))
    return metadata


def ensure_nist_full_probe_prepared(
    output_dir: Path,
    *,
    max_precursor_mz: float,
    cache_dir: Path,
    hdf5_repo_id: str,
    hdf5_filename: str = NIST_FULL_HF_FILENAME,
    num_shards: int = _DEFAULT_MASSSPEC_NUM_SHARDS,
    use_cache: bool = True,
) -> dict[str, Any]:
    if use_cache:
        cached = _probe_metadata_valid(
            output_dir,
            NIST_FULL_METADATA_VERSION,
            max_precursor_mz,
            expected_metadata={
                "artifact_format": NIST_FULL_ARTIFACT_FORMAT,
                "hdf5_repo_id": hdf5_repo_id,
                "hdf5_filename": hdf5_filename,
            },
        )
        if cached is not None:
            return cached
    hdf5_path = cache_dir / hdf5_filename
    if not hdf5_path.exists():
        hdf5_path = _download_hf_file(hdf5_repo_id, hdf5_filename, cache_dir)
    return build_nist_full_probe_artifact(
        hdf5_path,
        output_dir,
        max_precursor_mz=max_precursor_mz,
        num_shards=num_shards,
        extra_metadata={
            "hdf5_repo_id": hdf5_repo_id,
            "hdf5_filename": hdf5_filename,
        },
    )


def ensure_nist_full_probe_downloaded(
    output_dir: Path,
    *,
    max_precursor_mz: float,
    repo_id: str,
    revision: str = "main",
) -> dict[str, Any]:
    cached = _probe_metadata_valid(
        output_dir,
        NIST_FULL_METADATA_VERSION,
        max_precursor_mz,
        expected_metadata={"artifact_format": NIST_FULL_ARTIFACT_FORMAT},
    )
    if cached is not None:
        return cached
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=output_dir,
        allow_patterns=[
            _METADATA_FILENAME,
            NIST_FULL_PAIRWISE_ALIGNMENT_FILENAME,
            "train/*",
            "val/*",
            "test/*",
        ],
    )
    metadata = _probe_metadata_valid(
        output_dir,
        NIST_FULL_METADATA_VERSION,
        max_precursor_mz,
        expected_metadata={"artifact_format": NIST_FULL_ARTIFACT_FORMAT},
    )
    if metadata is None:
        raise FileNotFoundError(f"Invalid NIST full probe artifact in {output_dir}")
    return metadata


def ensure_nist_murcko_probe_downloaded(
    output_dir: Path,
    *,
    max_precursor_mz: float,
    repo_id: str = NIST_MURCKO_HF_REPO,
    revision: str = "main",
    subdir: str = NIST_MURCKO_PREPARED_SUBDIR,
    include_morgan: bool = False,
    distributed_world_size: int = 1,
    distributed_rank: int = 0,
) -> dict[str, Any]:
    cached = _probe_metadata_valid(
        output_dir,
        NIST_MURCKO_METADATA_VERSION,
        max_precursor_mz,
        expected_metadata={"artifact_format": NIST_MURCKO_ARTIFACT_FORMAT},
    )
    if cached is not None and (
        not include_morgan
        or bool(cached.get("morgan_auxiliary_available", False))
        and all(
            (output_dir / name).exists()
            for names in cached.get("morgan_auxiliary_files", {}).values()
            for name in names
        )
    ):
        if _coordinate_distributed_download(distributed_world_size):
            torch.distributed.barrier()
        return cached
    subdir = subdir.strip("/")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    allow_patterns = [
        f"{subdir}/{_METADATA_FILENAME}",
        f"{subdir}/train.parquet",
        f"{subdir}/val.parquet",
        f"{subdir}/test.parquet",
    ]
    if include_morgan:
        allow_patterns.append(f"{subdir}/auxiliary/morgan/*")
    _snapshot_download_rank_zero(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=output_dir.parent,
        allow_patterns=allow_patterns,
        distributed_world_size=distributed_world_size,
        distributed_rank=distributed_rank,
    )
    metadata = _probe_metadata_valid(
        output_dir,
        NIST_MURCKO_METADATA_VERSION,
        max_precursor_mz,
        expected_metadata={"artifact_format": NIST_MURCKO_ARTIFACT_FORMAT},
    )
    if metadata is None:
        raise FileNotFoundError(f"Invalid NIST Murcko probe artifact in {output_dir}")
    return metadata


class MurckoFluorineData(NamedTuple):
    metadata: dict[str, Any]
    root: Path
    batch_size: int
    num_peaks: int
    max_precursor_mz: float
    min_peak_intensity: float
    peak_drop_min_intensity: float
    peak_ordering: str
    precursor_peak_exclusion_window_da: float


def _read_murcko_subdir_metadata(
    cache_dir: Path,
    subdir: str,
    *,
    required_splits: tuple[str, ...],
) -> dict[str, Any] | None:
    metadata_path = cache_dir / subdir / _METADATA_FILENAME
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text())
    for split in required_splits:
        for filename in metadata.get(f"{split}_files", []):
            if not (cache_dir / subdir / filename).exists():
                return None
    return metadata


def _murcko_fluorine_split_metadata(
    source_metadata: dict[str, Any],
    *,
    subdir: str,
    source_split: str,
    target_split: str,
) -> dict[str, Any]:
    return {
        f"{target_split}_files": [
            f"{subdir}/{filename}" for filename in source_metadata[f"{source_split}_files"]
        ],
        f"{target_split}_lengths": [
            int(value) for value in source_metadata[f"{source_split}_lengths"]
        ],
        f"{target_split}_size": int(source_metadata[f"{source_split}_size"]),
        f"{target_split}_positive": int(source_metadata.get(f"{source_split}_positive", 0)),
    }


def ensure_murcko_fluorine_data_downloaded(
    cache_dir: Path,
    *,
    repo_id: str = NIST_MURCKO_HF_REPO,
    revision: str = "main",
    train_subdir: str = NIST_MURCKO_PREPARED_SUBDIR,
    test_subdir: str = MCEBIO_MURCKO_PREPARED_SUBDIR,
    distributed_world_size: int = 1,
    distributed_rank: int = 0,
) -> dict[str, Any]:
    train_subdir = train_subdir.strip("/")
    test_subdir = test_subdir.strip("/")
    cache_dir.mkdir(parents=True, exist_ok=True)
    allow_patterns = [
        f"{train_subdir}/{_METADATA_FILENAME}",
        f"{train_subdir}/train.parquet",
        f"{train_subdir}/val.parquet",
        f"{test_subdir}/{_METADATA_FILENAME}",
        f"{test_subdir}/test.parquet",
    ]
    needs_download = (
        _read_murcko_subdir_metadata(
            cache_dir,
            train_subdir,
            required_splits=("train", "val"),
        )
        is None
        or _read_murcko_subdir_metadata(
            cache_dir,
            test_subdir,
            required_splits=("test",),
        )
        is None
    )
    if needs_download:
        _snapshot_download_rank_zero(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=cache_dir,
            allow_patterns=allow_patterns,
            distributed_world_size=distributed_world_size,
            distributed_rank=distributed_rank,
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
            required_splits=("test",),
        ),
    )
    metadata: dict[str, Any] = {
        "metadata_version": 1,
        "storage_format": "parquet",
        "repo_id": repo_id,
        "revision": revision,
        "train_subdir": train_subdir,
        "test_subdir": test_subdir,
        "dreams_dim": int(train_metadata.get("dreams_dim", 0)),
    }
    metadata.update(
        _murcko_fluorine_split_metadata(
            train_metadata,
            subdir=train_subdir,
            source_split="train",
            target_split="train",
        )
    )
    metadata.update(
        _murcko_fluorine_split_metadata(
            train_metadata,
            subdir=train_subdir,
            source_split="val",
            target_split="val",
        )
    )
    metadata.update(
        _murcko_fluorine_split_metadata(
            test_metadata,
            subdir=test_subdir,
            source_split="test",
            target_split="test",
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
    repo_id: str = NIST_MURCKO_HF_REPO,
    revision: str = "main",
    train_subdir: str = NIST_MURCKO_PREPARED_SUBDIR,
    test_subdir: str = MCEBIO_MURCKO_PREPARED_SUBDIR,
    distributed_world_size: int = 1,
    distributed_rank: int = 0,
) -> MurckoFluorineData:
    metadata = ensure_murcko_fluorine_data_downloaded(
        cache_dir,
        repo_id=repo_id,
        revision=revision,
        train_subdir=train_subdir,
        test_subdir=test_subdir,
        distributed_world_size=distributed_world_size,
        distributed_rank=distributed_rank,
    )
    return MurckoFluorineData(
        metadata=metadata,
        root=cache_dir,
        batch_size=batch_size,
        num_peaks=num_peaks,
        max_precursor_mz=max_precursor_mz,
        min_peak_intensity=min_peak_intensity,
        peak_drop_min_intensity=peak_drop_min_intensity,
        peak_ordering=peak_ordering,
        precursor_peak_exclusion_window_da=precursor_peak_exclusion_window_da,
    )


def ensure_mona_a_probe_prepared(
    output_dir: Path,
    *,
    max_precursor_mz: float,
    cache_dir: Path,
    num_shards: int = _DEFAULT_MASSSPEC_NUM_SHARDS,
) -> dict[str, Any]:
    cached = _probe_metadata_valid(output_dir, MONA_A_METADATA_VERSION, max_precursor_mz)
    if cached is not None:
        return cached
    pkl_path = cache_dir / MONA_A_HF_FILENAME
    if not pkl_path.exists():
        pkl_path = _download_hf_file(MONA_A_HF_REPO, MONA_A_HF_FILENAME, cache_dir)
    metadata = _filter_encode_and_write(
        **_load_mona_a_pkl(pkl_path),
        output_dir=output_dir,
        num_shards=num_shards,
        max_precursor_mz=max_precursor_mz,
        metadata_version=MONA_A_METADATA_VERSION,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / _METADATA_FILENAME).write_text(json.dumps(metadata, indent=2))
    return metadata


class _ProbeMemmapDataset(Dataset):
    def __init__(self, shard_entries: list[dict[str, Any]]) -> None:
        self._shard_entries = [
            {"dir": Path(entry["dir"]), "length": int(entry["length"])}
            for entry in shard_entries
        ]
        lengths = np.asarray(
            [entry["length"] for entry in self._shard_entries], dtype=np.int64
        )
        self._starts = np.zeros(len(lengths) + 1, dtype=np.int64)
        np.cumsum(lengths, out=self._starts[1:])
        self._arrays: list[dict[str, np.ndarray]] | None = None

    def __len__(self) -> int:
        return int(self._starts[-1])

    def _ensure_arrays(self) -> list[dict[str, np.ndarray]]:
        if self._arrays is not None:
            return self._arrays
        keys = [
            "spectra",
            "precursor_mz_raw",
            "fingerprint",
            "smiles",
            "adduct_id",
            "instrument_type_id",
            "collision_energy",
            "collision_energy_present",
            "probe_valid_mol",
            "probe_maccs",
            "probe_morgan",
            *[f"probe_{name}" for name in REGRESSION_TARGET_KEYS],
            "dreams_embedding",
        ]
        arrays_by_shard: list[dict[str, np.ndarray]] = []
        for entry in self._shard_entries:
            shard_dir = entry["dir"]
            arrays: dict[str, np.ndarray] = {}
            for key in keys:
                path = shard_dir / f"{key}.npy"
                if path.exists():
                    arrays[key] = np.load(path, mmap_mode="r")
            arrays_by_shard.append(arrays)
        self._arrays = arrays_by_shard
        return arrays_by_shard

    def __getitem__(self, index: int) -> dict[str, Any]:
        arrays_by_shard = self._ensure_arrays()
        index = index
        shard_idx = int(np.searchsorted(self._starts, index, side="right") - 1)
        local_idx = index - int(self._starts[shard_idx])
        arrays = arrays_by_shard[shard_idx]
        sample: dict[str, Any] = {}
        for key, value in arrays.items():
            item = value[local_idx]
            if isinstance(item, np.ndarray):
                sample[key] = torch.from_numpy(item.copy())
            elif np.isscalar(item):
                if value.dtype.kind in {"U", "S"}:
                    sample[key] = str(item)
                elif value.dtype == np.bool_:
                    sample[key] = bool(item)
                elif value.dtype.kind in {"i", "u"}:
                    sample[key] = int(cast(Any, item))
                else:
                    sample[key] = float(cast(Any, item))
            else:
                sample[key] = item
        return sample


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


class _ProbeParquetDataset(Dataset):
    def __init__(
        self,
        entries: list[dict[str, Any]],
        *,
        adduct_vocab: dict[str, int],
        instrument_type_vocab: dict[str, int],
    ) -> None:
        self._entries = [
            {
                "path": Path(entry["path"]),
                "length": int(entry["length"]),
                "morgan_files": [Path(path) for path in entry.get("morgan_files", [])],
            }
            for entry in entries
        ]
        lengths = np.asarray([entry["length"] for entry in self._entries], dtype=np.int64)
        self._starts = np.zeros(len(lengths) + 1, dtype=np.int64)
        np.cumsum(lengths, out=self._starts[1:])
        self._adduct_vocab = adduct_vocab
        self._instrument_type_vocab = instrument_type_vocab
        self._arrays: list[dict[str, np.ndarray]] | None = None

    def __len__(self) -> int:
        return int(self._starts[-1])

    def _load_entry(self, entry: dict[str, Any]) -> dict[str, np.ndarray]:
        import pyarrow.parquet as pq

        path = Path(entry["path"])
        table = pq.read_table(path)
        rows = table.to_pydict()
        n = len(rows["precursor_mz"])
        arrays: dict[str, np.ndarray] = {
            "spectra": _spectra_from_peak_lists(
                rows["spectrum_mz"],
                rows["spectrum_intensity"],
            ),
            "precursor_mz_raw": np.asarray(rows["precursor_mz"], dtype=np.float32),
            "fingerprint": np.zeros((n, _FINGERPRINT_BITS), dtype=np.int8),
            "smiles": np.asarray(rows["canonical_smiles"], dtype=str),
            "adduct_id": np.asarray(
                [self._adduct_vocab[value] for value in rows["adduct"]],
                dtype=np.int32,
            ),
            "instrument_type_id": np.asarray(
                [self._instrument_type_vocab[value] for value in rows["instrument_type"]],
                dtype=np.int32,
            ),
            "collision_energy": np.asarray(rows["collision_energy"], dtype=np.float32),
            "collision_energy_present": np.asarray(
                rows["collision_energy_present"],
                dtype=np.int32,
            ),
            "probe_valid_mol": np.ones(n, dtype=bool),
            "probe_maccs": np.asarray(rows["maccs_166"], dtype=np.int8),
        }
        for name in REGRESSION_TARGET_KEYS:
            arrays[f"probe_{name}"] = np.asarray(rows[name], dtype=np.float32)
        morgan_files = entry.get("morgan_files", [])
        if morgan_files:
            arrays["probe_morgan"] = np.concatenate(
                [
                    np.load(Path(path), allow_pickle=False)["morgan"].astype(np.int8)
                    for path in morgan_files
                ],
                axis=0,
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
        sample: dict[str, Any] = {}
        for key, value in arrays.items():
            item = value[local_idx]
            if isinstance(item, np.ndarray):
                sample[key] = torch.from_numpy(item.copy())
            elif np.isscalar(item):
                if value.dtype.kind in {"U", "S"}:
                    sample[key] = str(item)
                elif value.dtype == np.bool_:
                    sample[key] = bool(item)
                elif value.dtype.kind in {"i", "u"}:
                    sample[key] = int(cast(Any, item))
                else:
                    sample[key] = float(cast(Any, item))
            else:
                sample[key] = item
        return sample


class _MurckoFluorineParquetDataset(Dataset):
    def __init__(self, entries: list[dict[str, Any]]) -> None:
        self._entries = [
            {"path": Path(entry["path"]), "length": int(entry["length"])}
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
        return arrays

    def _ensure_arrays(self) -> list[dict[str, np.ndarray]]:
        if self._arrays is not None:
            return self._arrays
        self._arrays = [self._load_entry(entry) for entry in self._entries]
        return self._arrays

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        arrays_by_entry = self._ensure_arrays()
        entry_idx = int(np.searchsorted(self._starts, index, side="right") - 1)
        local_idx = index - int(self._starts[entry_idx])
        arrays = arrays_by_entry[entry_idx]
        sample = {
            "spectra": torch.from_numpy(arrays["spectra"][local_idx].copy()),
            "precursor_mz_raw": torch.tensor(
                float(arrays["precursor_mz_raw"][local_idx]),
                dtype=torch.float32,
            ),
            "label": torch.tensor(float(arrays["label"][local_idx]), dtype=torch.float32),
            "row_idx": torch.tensor(index, dtype=torch.long),
        }
        if "dreams_embedding" in arrays:
            sample["dreams_embedding"] = torch.from_numpy(
                arrays["dreams_embedding"][local_idx].copy()
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
    ) -> None:
        self.num_peaks = num_peaks
        self.max_precursor_mz = max_precursor_mz
        self.min_peak_intensity = min_peak_intensity
        self.peak_drop_min_intensity = peak_drop_min_intensity
        self.peak_ordering = peak_ordering
        self.precursor_peak_exclusion_window_da = precursor_peak_exclusion_window_da
        self.dreams_only = dreams_only

    def __call__(self, samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        if self.dreams_only:
            return {
                "dreams_embedding": torch.stack(
                    [sample["dreams_embedding"] for sample in samples],
                    dim=0,
                ).to(torch.float32),
                "label": torch.stack([sample["label"] for sample in samples]).to(
                    torch.float32
                ),
                "row_idx": torch.stack([sample["row_idx"] for sample in samples]).to(
                    torch.long
                ),
            }
        spectra = torch.stack([sample["spectra"] for sample in samples], dim=0)
        precursor_raw = torch.stack([sample["precursor_mz_raw"] for sample in samples])
        batch = preprocess_peak_batch_torch(
            spectra[:, 0, :],
            spectra[:, 1, :],
            precursor_raw,
            num_peaks=self.num_peaks,
            peak_drop_min_intensity=self.peak_drop_min_intensity,
            peak_ordering=self.peak_ordering,
            max_precursor_mz=self.max_precursor_mz,
            precursor_peak_exclusion_window_da=self.precursor_peak_exclusion_window_da,
            min_peak_intensity=self.min_peak_intensity,
        )
        if "dreams_embedding" in samples[0]:
            batch["dreams_embedding"] = torch.stack(
                [sample["dreams_embedding"] for sample in samples],
                dim=0,
            ).to(torch.float32)
        batch["label"] = torch.stack([sample["label"] for sample in samples]).to(
            torch.float32
        )
        batch["row_idx"] = torch.stack([sample["row_idx"] for sample in samples]).to(
            torch.long
        )
        return batch


def _subset_for_max_samples(
    dataset: Dataset,
    *,
    max_samples: int | None,
    shuffle: bool,
    seed: int,
) -> tuple[Dataset, bool]:
    if max_samples is None:
        return dataset, shuffle
    n = min(len(dataset), max_samples)
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=generator)[:n].tolist()
    else:
        indices = list(range(n))
    return Subset(dataset, [int(idx) for idx in indices]), False


def _loader_sampler(
    dataset: Dataset,
    *,
    shuffle: bool,
    seed: int,
    drop_last: bool,
    distributed_world_size: int,
    distributed_rank: int,
) -> tuple[DistributedSampler | None, bool]:
    if distributed_world_size <= 1:
        return None, shuffle
    return (
        DistributedSampler(
            dataset,
            num_replicas=distributed_world_size,
            rank=distributed_rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        ),
        False,
    )


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
) -> DataLoader:
    dataset: Dataset = _MurckoFluorineParquetDataset(
        [
            {"path": data.root / path, "length": length}
            for path, length in zip(
                data.metadata[f"{split}_files"],
                data.metadata[f"{split}_lengths"],
                strict=True,
            )
        ]
    )
    dataset, shuffle = _subset_for_max_samples(
        dataset,
        max_samples=max_samples,
        shuffle=shuffle,
        seed=seed,
    )
    sampler, loader_shuffle = _loader_sampler(
        dataset,
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
        distributed_world_size=distributed_world_size,
        distributed_rank=distributed_rank,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=probe_local_batch_size(data.batch_size, distributed_world_size),
        shuffle=loader_shuffle,
        sampler=sampler,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=_MurckoFluorineCollator(
            num_peaks=data.num_peaks,
            max_precursor_mz=data.max_precursor_mz,
            min_peak_intensity=data.min_peak_intensity,
            peak_drop_min_intensity=data.peak_drop_min_intensity,
            peak_ordering=data.peak_ordering,
            precursor_peak_exclusion_window_da=data.precursor_peak_exclusion_window_da,
            dreams_only=dreams_only,
        ),
        generator=generator,
    )


class _ProbeBatchCollator:
    def __init__(
        self,
        *,
        num_peaks: int,
        max_precursor_mz: float,
        min_peak_intensity: float,
        peak_drop_min_intensity: float,
        peak_ordering: str,
        precursor_peak_exclusion_window_da: float,
    ) -> None:
        self.num_peaks = num_peaks
        self.max_precursor_mz = max_precursor_mz
        self.min_peak_intensity = min_peak_intensity
        self.peak_drop_min_intensity = peak_drop_min_intensity
        self.peak_ordering = peak_ordering
        self.precursor_peak_exclusion_window_da = precursor_peak_exclusion_window_da

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        spectra = torch.stack([sample["spectra"] for sample in samples], dim=0)
        precursor_raw = torch.tensor(
            [float(sample["precursor_mz_raw"]) for sample in samples],
            dtype=torch.float32,
        )
        batch: dict[str, Any] = preprocess_peak_batch_torch(
            spectra[:, 0, :],
            spectra[:, 1, :],
            precursor_raw,
            num_peaks=self.num_peaks,
            peak_drop_min_intensity=self.peak_drop_min_intensity,
            peak_ordering=self.peak_ordering,
            max_precursor_mz=self.max_precursor_mz,
            precursor_peak_exclusion_window_da=self.precursor_peak_exclusion_window_da,
            min_peak_intensity=self.min_peak_intensity,
        )
        batch["fingerprint"] = torch.stack(
            [sample["fingerprint"] for sample in samples], dim=0
        ).to(torch.int32)
        batch["smiles"] = [str(sample["smiles"]) for sample in samples]
        batch["adduct_id"] = torch.tensor(
            [int(sample["adduct_id"]) for sample in samples], dtype=torch.int32
        )
        batch["instrument_type_id"] = torch.tensor(
            [int(sample["instrument_type_id"]) for sample in samples], dtype=torch.int32
        )
        batch["collision_energy"] = torch.tensor(
            [float(sample["collision_energy"]) for sample in samples],
            dtype=torch.float32,
        )
        batch["collision_energy_present"] = torch.tensor(
            [int(sample["collision_energy_present"]) for sample in samples],
            dtype=torch.int32,
        )
        batch["probe_valid_mol"] = torch.tensor(
            [bool(sample["probe_valid_mol"]) for sample in samples],
            dtype=torch.bool,
        )
        batch["probe_maccs"] = torch.stack(
            [sample["probe_maccs"] for sample in samples], dim=0
        ).to(torch.int32)
        if "probe_morgan" in samples[0]:
            batch["probe_morgan"] = torch.stack(
                [sample["probe_morgan"] for sample in samples], dim=0
            ).to(torch.int32)
        for name in REGRESSION_TARGET_KEYS:
            batch[f"probe_{name}"] = torch.tensor(
                [float(sample[f"probe_{name}"]) for sample in samples],
                dtype=torch.float32,
            )
        if "dreams_embedding" in samples[0]:
            batch["dreams_embedding"] = torch.stack(
                [sample["dreams_embedding"] for sample in samples], dim=0
            ).to(torch.float32)
        return batch


class _LoaderAdapter:
    def __init__(self, loader: DataLoader) -> None:
        self._loader = loader

    def __iter__(self):
        return iter(self._loader)

    def __len__(self) -> int:
        return len(self._loader)

    def as_numpy_iterator(self):
        for batch in self._loader:
            yield {
                key: value.detach().cpu().numpy()
                if isinstance(value, torch.Tensor)
                else value
                for key, value in batch.items()
            }


def probe_local_batch_size(global_batch_size: int, distributed_world_size: int) -> int:
    if distributed_world_size <= 1:
        return global_batch_size
    assert global_batch_size % distributed_world_size == 0
    return global_batch_size // distributed_world_size


class MassSpecProbeData(NamedTuple):
    info: dict[str, Any]
    storage_format: str
    train_files: list[str]
    train_lengths: list[int]
    train_morgan_files: list[str]
    val_files: list[str]
    val_lengths: list[int]
    val_morgan_files: list[str]
    test_files: list[str]
    test_lengths: list[int]
    test_morgan_files: list[str]
    batch_size: int
    shuffle_buffer: int
    max_precursor_mz: float
    min_peak_intensity: float
    peak_drop_min_intensity: float
    peak_ordering: str
    num_peaks: int
    dreams_dim: int
    precursor_peak_exclusion_window_da: float
    pairwise_alignment_path: str

    @classmethod
    def from_config(
        cls,
        config: config_dict.ConfigDict,
        *,
        distributed_world_size: int = 1,
        distributed_rank: int = 0,
    ) -> "MassSpecProbeData":
        artifact_root = (
            Path(_config_get(config, "artifact_dir", str(_DEFAULT_ARTIFACT_DIR)))
            .expanduser()
            .resolve()
        )
        max_precursor_mz = float(
            _config_get(config, "max_precursor_mz", DEFAULT_MAX_PRECURSOR_MZ)
        )
        include_morgan = (
            str(
                _config_get(
                    config,
                    "msg_probe_fingerprint",
                    _config_get(config, "msg_probe_fingerprint_type", "maccs"),
                )
            ).lower()
            == "morgan"
            or int(_config_get(config, "msg_probe_pairwise_alignment_num_pairs", 0)) > 0
        )
        murcko_subdir = str(
            _config_get(
                config,
                "nist_murcko_probe_hf_subdir",
                NIST_MURCKO_PREPARED_SUBDIR,
            )
        ).strip("/")
        output_dir = artifact_root / murcko_subdir
        metadata = ensure_nist_murcko_probe_downloaded(
            output_dir,
            max_precursor_mz=max_precursor_mz,
            repo_id=str(
                _config_get(config, "nist_murcko_probe_repo_id", NIST_MURCKO_HF_REPO)
            ),
            revision=str(_config_get(config, "nist_murcko_probe_revision", "main")),
            subdir=murcko_subdir,
            include_morgan=include_morgan,
            distributed_world_size=distributed_world_size,
            distributed_rank=distributed_rank,
        )
        adduct_vocab = metadata.get("adduct_vocab", {"unknown": 0})
        instrument_type_vocab = metadata.get("instrument_type_vocab", {"unknown": 0})
        storage_format = str(metadata.get("storage_format", "native"))
        morgan_files = metadata.get("morgan_auxiliary_files", {}) if include_morgan else {}
        info = {
            "massspec_train_size": int(metadata.get("train_size", 0)),
            "massspec_val_size": int(metadata.get("val_size", 0)),
            "massspec_test_size": int(metadata.get("test_size", 0)),
            "massspec_metadata_version": int(metadata.get("metadata_version", 0)),
            "massspec_adduct_vocab": adduct_vocab,
            "massspec_instrument_type_vocab": instrument_type_vocab,
            "massspec_adduct_vocab_size": len(adduct_vocab),
            "massspec_instrument_type_vocab_size": len(instrument_type_vocab),
            "fingerprint_bits": _FINGERPRINT_BITS,
            "probe_maccs_bits": int(metadata.get("probe_maccs_bits", 0)),
            "probe_morgan_bits": int(metadata.get("probe_morgan_bits", 0))
            if include_morgan
            else 0,
            "probe_morgan_radius": int(metadata.get("probe_morgan_radius", 0)),
            "pairwise_alignment_available": bool(
                metadata.get("pairwise_alignment_available", False)
            ),
            "pairwise_alignment_num_pairs": int(
                metadata.get("pairwise_alignment_num_pairs", 0)
            ),
            "pairwise_alignment_num_endpoints": int(
                metadata.get("pairwise_alignment_num_endpoints", 0)
            ),
        }
        pairwise_file = str(metadata.get("pairwise_alignment_file", ""))
        pairwise_alignment_path = str(output_dir / pairwise_file) if pairwise_file else ""
        if storage_format == "parquet":
            train_files = [str(output_dir / name) for name in metadata["train_files"]]
            val_files = [str(output_dir / name) for name in metadata["val_files"]]
            test_files = [str(output_dir / name) for name in metadata["test_files"]]
        else:
            train_files = [
                str(output_dir / "train" / name) for name in metadata["train_files"]
            ]
            val_files = [str(output_dir / "val" / name) for name in metadata["val_files"]]
            test_files = [
                str(output_dir / "test" / name) for name in metadata["test_files"]
            ]
        return cls(
            info=info,
            storage_format=storage_format,
            train_files=train_files,
            train_lengths=[int(v) for v in metadata["train_lengths"]],
            train_morgan_files=[
                str(output_dir / name) for name in morgan_files.get("train", [])
            ],
            val_files=val_files,
            val_lengths=[int(v) for v in metadata["val_lengths"]],
            val_morgan_files=[
                str(output_dir / name) for name in morgan_files.get("val", [])
            ],
            test_files=test_files,
            test_lengths=[int(v) for v in metadata["test_lengths"]],
            test_morgan_files=[
                str(output_dir / name) for name in morgan_files.get("test", [])
            ],
            batch_size=int(
                _config_get(
                    config,
                    "msg_probe_batch_size",
                    _config_get(config, "batch_size", _DEFAULT_BATCH_SIZE),
                )
            ),
            shuffle_buffer=int(
                _config_get(config, "shuffle_buffer", _DEFAULT_SHUFFLE_BUFFER)
            ),
            max_precursor_mz=max_precursor_mz,
            min_peak_intensity=float(
                _config_get(config, "min_peak_intensity", DEFAULT_MIN_PEAK_INTENSITY)
            ),
            peak_drop_min_intensity=float(
                _config_get(
                    config,
                    "peak_drop_min_intensity",
                    _config_get(config, "min_peak_intensity", DEFAULT_MIN_PEAK_INTENSITY),
                )
            ),
            peak_ordering=str(_config_get(config, "peak_ordering", "mz")),
            num_peaks=int(_config_get(config, "num_peaks", _NUM_PEAKS_OUTPUT)),
            dreams_dim=int(metadata.get("dreams_dim", 0)),
            precursor_peak_exclusion_window_da=float(
                _config_get(
                    config,
                    "precursor_peak_exclusion_window_da",
                    _DEFAULT_PRECURSOR_PEAK_EXCLUSION_WINDOW_DA,
                )
            ),
            pairwise_alignment_path=pairwise_alignment_path,
        )

    def build_dataset(
        self,
        split: str,
        *,
        seed: int = 0,
        peak_ordering: str | None = None,
        shuffle: bool = False,
        drop_remainder: bool = True,
        num_parallel_reads: int | None = None,
        max_samples: int | None = None,
        distributed_world_size: int = 1,
        distributed_rank: int = 0,
        pad_distributed: bool = False,
    ):
        split_files = {
            "massspec_train": self.train_files,
            "massspec_val": self.val_files,
            "massspec_test": self.test_files,
            "train": self.train_files,
            "val": self.val_files,
            "test": self.test_files,
            "all": self.train_files + self.val_files + self.test_files,
        }[split]
        split_lengths = {
            "massspec_train": self.train_lengths,
            "massspec_val": self.val_lengths,
            "massspec_test": self.test_lengths,
            "train": self.train_lengths,
            "val": self.val_lengths,
            "test": self.test_lengths,
            "all": self.train_lengths + self.val_lengths + self.test_lengths,
        }[split]
        split_morgan_files = {
            "massspec_train": [self.train_morgan_files] if self.train_files else [],
            "massspec_val": [self.val_morgan_files] if self.val_files else [],
            "massspec_test": [self.test_morgan_files] if self.test_files else [],
            "train": [self.train_morgan_files] if self.train_files else [],
            "val": [self.val_morgan_files] if self.val_files else [],
            "test": [self.test_morgan_files] if self.test_files else [],
            "all": (
                ([self.train_morgan_files] if self.train_files else [])
                + ([self.val_morgan_files] if self.val_files else [])
                + ([self.test_morgan_files] if self.test_files else [])
            ),
        }[split]
        if self.storage_format == "parquet":
            dataset = _ProbeParquetDataset(
                [
                    {"path": path, "length": length, "morgan_files": morgan_files}
                    for path, length, morgan_files in zip(
                        split_files,
                        split_lengths,
                        split_morgan_files,
                        strict=True,
                    )
                ],
                adduct_vocab=self.info["massspec_adduct_vocab"],
                instrument_type_vocab=self.info["massspec_instrument_type_vocab"],
            )
        else:
            dataset = _ProbeMemmapDataset(
                [
                    {"dir": path, "length": length}
                    for path, length in zip(split_files, split_lengths, strict=True)
                ]
            )
        dataset, shuffle = _subset_for_max_samples(
            dataset,
            max_samples=max_samples,
            shuffle=shuffle,
            seed=seed,
        )
        sampler, loader_shuffle = _loader_sampler(
            dataset,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_remainder,
            distributed_world_size=distributed_world_size,
            distributed_rank=distributed_rank,
        )
        generator = torch.Generator()
        generator.manual_seed(seed)
        batch_size = probe_local_batch_size(self.batch_size, distributed_world_size)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=loader_shuffle,
            sampler=sampler,
            drop_last=drop_remainder,
            num_workers=0,
            collate_fn=_ProbeBatchCollator(
                num_peaks=self.num_peaks,
                max_precursor_mz=self.max_precursor_mz,
                min_peak_intensity=self.min_peak_intensity,
                peak_drop_min_intensity=self.peak_drop_min_intensity,
                peak_ordering=peak_ordering or self.peak_ordering,
                precursor_peak_exclusion_window_da=self.precursor_peak_exclusion_window_da,
            ),
            generator=generator,
        )
        return _LoaderAdapter(loader)

    def build_indexed_dataset(
        self,
        split: str,
        indices: np.ndarray,
        *,
        peak_ordering: str | None = None,
        drop_remainder: bool = False,
        distributed_world_size: int = 1,
    ):
        split_files = {
            "massspec_train": self.train_files,
            "massspec_val": self.val_files,
            "massspec_test": self.test_files,
            "train": self.train_files,
            "val": self.val_files,
            "test": self.test_files,
            "all": self.train_files + self.val_files + self.test_files,
        }[split]
        split_lengths = {
            "massspec_train": self.train_lengths,
            "massspec_val": self.val_lengths,
            "massspec_test": self.test_lengths,
            "train": self.train_lengths,
            "val": self.val_lengths,
            "test": self.test_lengths,
            "all": self.train_lengths + self.val_lengths + self.test_lengths,
        }[split]
        split_morgan_files = {
            "massspec_train": [self.train_morgan_files] if self.train_files else [],
            "massspec_val": [self.val_morgan_files] if self.val_files else [],
            "massspec_test": [self.test_morgan_files] if self.test_files else [],
            "train": [self.train_morgan_files] if self.train_files else [],
            "val": [self.val_morgan_files] if self.val_files else [],
            "test": [self.test_morgan_files] if self.test_files else [],
            "all": (
                ([self.train_morgan_files] if self.train_files else [])
                + ([self.val_morgan_files] if self.val_files else [])
                + ([self.test_morgan_files] if self.test_files else [])
            ),
        }[split]
        if self.storage_format == "parquet":
            dataset = _ProbeParquetDataset(
                [
                    {"path": path, "length": length, "morgan_files": morgan_files}
                    for path, length, morgan_files in zip(
                        split_files,
                        split_lengths,
                        split_morgan_files,
                        strict=True,
                    )
                ],
                adduct_vocab=self.info["massspec_adduct_vocab"],
                instrument_type_vocab=self.info["massspec_instrument_type_vocab"],
            )
        else:
            dataset = _ProbeMemmapDataset(
                [
                    {"dir": path, "length": length}
                    for path, length in zip(split_files, split_lengths, strict=True)
                ]
            )
        batch_size = probe_local_batch_size(self.batch_size, distributed_world_size)
        loader = DataLoader(
            Subset(dataset, [int(idx) for idx in indices]),
            batch_size=batch_size,
            shuffle=False,
            drop_last=drop_remainder,
            num_workers=0,
            collate_fn=_ProbeBatchCollator(
                num_peaks=self.num_peaks,
                max_precursor_mz=self.max_precursor_mz,
                min_peak_intensity=self.min_peak_intensity,
                peak_drop_min_intensity=self.peak_drop_min_intensity,
                peak_ordering=peak_ordering or self.peak_ordering,
                precursor_peak_exclusion_window_da=self.precursor_peak_exclusion_window_da,
            ),
        )
        return _LoaderAdapter(loader)
