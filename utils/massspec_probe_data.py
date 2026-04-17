import csv
import json
import logging
import math
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from ml_collections import config_dict
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from torch.utils.data import DataLoader, Dataset

from input_pipeline import _prepend_precursor_token_torch
from utils.massspec_probe_targets import (
    MACCS_FINGERPRINT_BITS,
    REGRESSION_TARGET_KEYS,
    build_maccs_targets_for_rows,
    build_probe_targets_for_rows,
)
from utils.spectra_preprocessing import (
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
NIST_FULL_METADATA_VERSION = 1
NIST_FULL_HF_FILENAME = "hr_msms_nist.hdf5"
NIST_FULL_ARTIFACT_FORMAT = "nist_full_probe_v1"

MONA_A_METADATA_VERSION = 3
MONA_A_HF_REPO = "roman-bushuiev/GeMS"
MONA_A_HF_FILENAME = (
    "data/auxiliary/MoNA_A_Murcko_split_neighbours_[M+H]+_0.05Da.pkl"
)


def _download_hf_file(repo_id: str, filename: str, local_dir: Path) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=str(local_dir),
    )
    return Path(path)


def download_massspec_tsv(output_dir: Path) -> Path:
    return _download_hf_file(MASSSPEC_HF_REPO, MASSSPEC_TSV_PATH, output_dir / "data")


def _normalize_spectra_intensity(spectra: np.ndarray) -> np.ndarray:
    max_int = spectra[:, 1].max(axis=1, keepdims=True)
    np.divide(spectra[:, 1], np.maximum(max_int, 1e-8), out=spectra[:, 1])
    return spectra


def _load_massspec_tsv(tsv_path: Path) -> dict[str, np.ndarray]:
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


def _load_nist20_hdf5(hdf5_path: Path) -> dict[str, np.ndarray]:
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


def _load_mona_a_pkl(pkl_path: Path) -> dict[str, np.ndarray]:
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
            np.stack(df["PARSED PEAKS"].to_numpy()).astype(np.float32)
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
        fp = AllChem.GetMorganFingerprintAsBitVect(  # type: ignore[attr-defined]
            mol,
            _FINGERPRINT_RADIUS,
            nBits=_FINGERPRINT_BITS,
        )
        DataStructs.ConvertToNumpyArray(fp, fps[i])
    return fps


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
    num_shards = max(1, min(int(num_shards), n))
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
) -> dict[str, Any]:
    keep = np.isfinite(precursor) & (precursor <= float(max_precursor_mz))
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
    probe_valid_mol &= probe_maccs_valid
    metadata: dict[str, Any] = {
        "metadata_version": metadata_version,
        "max_precursor_mz": float(max_precursor_mz),
        "adduct_vocab": adduct_vocab,
        "instrument_type_vocab": instrument_type_vocab,
        "dreams_dim": int(dreams_embedding.shape[1]) if dreams_embedding is not None else 0,
        "probe_maccs_bits": MACCS_FINGERPRINT_BITS,
    }
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
        metadata[f"{split_name}_size"] = int(np.count_nonzero(split_mask))
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
    if float(metadata.get("max_precursor_mz", float("inf"))) != float(max_precursor_mz):
        return None
    if expected_metadata is not None:
        for key, value in expected_metadata.items():
            if metadata.get(key) != value:
                return None
    for split in ("train", "val", "test"):
        if not all(
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
        allow_patterns=[_METADATA_FILENAME, "train/*", "val/*", "test/*"],
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

    def _ensure_arrays(self) -> None:
        if self._arrays is not None:
            return
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
            *[f"probe_{name}" for name in REGRESSION_TARGET_KEYS],
            "dreams_embedding",
        ]
        self._arrays = []
        for entry in self._shard_entries:
            shard_dir = entry["dir"]
            arrays = {}
            for key in keys:
                path = shard_dir / f"{key}.npy"
                if path.exists():
                    arrays[key] = np.load(path, mmap_mode="r")
            self._arrays.append(arrays)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        self._ensure_arrays()
        assert self._arrays is not None
        idx = int(idx)
        shard_idx = int(np.searchsorted(self._starts, idx, side="right") - 1)
        local_idx = idx - int(self._starts[shard_idx])
        arrays = self._arrays[shard_idx]
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
                    sample[key] = int(item)
                else:
                    sample[key] = float(item)
            else:
                sample[key] = item
        return sample


class _ProbeBatchCollator:
    def __init__(
        self,
        *,
        num_peaks: int,
        max_precursor_mz: float,
        min_peak_intensity: float,
        peak_drop_min_intensity: float,
        peak_ordering: str,
        use_precursor_token: bool,
        precursor_peak_exclusion_window_da: float,
    ) -> None:
        self.num_peaks = int(num_peaks)
        self.max_precursor_mz = float(max_precursor_mz)
        self.min_peak_intensity = float(min_peak_intensity)
        self.peak_drop_min_intensity = float(peak_drop_min_intensity)
        self.peak_ordering = str(peak_ordering)
        self.use_precursor_token = bool(use_precursor_token)
        self.precursor_peak_exclusion_window_da = float(
            precursor_peak_exclusion_window_da
        )

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
        for name in REGRESSION_TARGET_KEYS:
            batch[f"probe_{name}"] = torch.tensor(
                [float(sample[f"probe_{name}"]) for sample in samples],
                dtype=torch.float32,
            )
        if "dreams_embedding" in samples[0]:
            batch["dreams_embedding"] = torch.stack(
                [sample["dreams_embedding"] for sample in samples], dim=0
            ).to(torch.float32)
        if self.use_precursor_token:
            batch = _prepend_precursor_token_torch(batch)
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


class MassSpecProbeData(NamedTuple):
    info: dict[str, Any]
    train_files: list[str]
    train_lengths: list[int]
    val_files: list[str]
    val_lengths: list[int]
    test_files: list[str]
    test_lengths: list[int]
    batch_size: int
    shuffle_buffer: int
    max_precursor_mz: float
    min_peak_intensity: float
    peak_drop_min_intensity: float
    peak_ordering: str
    num_peaks: int
    use_precursor_token: bool
    dreams_dim: int
    precursor_peak_exclusion_window_da: float

    @classmethod
    def from_config(cls, config: config_dict.ConfigDict) -> "MassSpecProbeData":
        artifact_root = (
            Path(config.get("artifact_dir", str(_DEFAULT_ARTIFACT_DIR)))
            .expanduser()
            .resolve()
        )
        max_precursor_mz = float(
            config.get("max_precursor_mz", DEFAULT_MAX_PRECURSOR_MZ)
        )
        probe_dataset = str(config.get("probe_dataset", "massspec"))
        if probe_dataset == "nist20":
            output_dir = artifact_root / "nist20_probe"
            metadata = ensure_nist20_probe_prepared(
                output_dir,
                max_precursor_mz=max_precursor_mz,
                cache_dir=artifact_root,
                hdf5_repo_id=str(config.get("nist20_hdf5_repo_id", NIST20_HF_REPO)),
                hdf5_filename=str(config.get("nist20_hdf5_filename", NIST20_HF_FILENAME)),
            )
        elif probe_dataset == "nist-full":
            output_dir = artifact_root / "nist_full_probe"
            metadata = ensure_nist_full_probe_downloaded(
                output_dir,
                max_precursor_mz=max_precursor_mz,
                repo_id=str(
                    config.get(
                        "nist_full_probe_repo_id",
                        config.get(
                            "nist_full_hdf5_repo_id",
                            config.get("nist20_hdf5_repo_id", NIST20_HF_REPO),
                        ),
                    )
                ),
                revision=str(config.get("nist_full_probe_revision", "main")),
            )
        elif probe_dataset == "mona_a":
            output_dir = artifact_root / "mona_a_probe"
            metadata = ensure_mona_a_probe_prepared(
                output_dir,
                max_precursor_mz=max_precursor_mz,
                cache_dir=artifact_root,
            )
        else:
            output_dir = artifact_root / "massspec_probe"
            metadata = ensure_massspec_probe_prepared(
                output_dir,
                max_precursor_mz=max_precursor_mz,
            )
        adduct_vocab = metadata.get("adduct_vocab", {"unknown": 0})
        instrument_type_vocab = metadata.get("instrument_type_vocab", {"unknown": 0})
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
        }
        return cls(
            info=info,
            train_files=[str(output_dir / "train" / name) for name in metadata["train_files"]],
            train_lengths=[int(v) for v in metadata["train_lengths"]],
            val_files=[str(output_dir / "val" / name) for name in metadata["val_files"]],
            val_lengths=[int(v) for v in metadata["val_lengths"]],
            test_files=[str(output_dir / "test" / name) for name in metadata["test_files"]],
            test_lengths=[int(v) for v in metadata["test_lengths"]],
            batch_size=int(
                config.get(
                    "msg_probe_batch_size",
                    config.get("batch_size", _DEFAULT_BATCH_SIZE),
                )
            ),
            shuffle_buffer=int(config.get("shuffle_buffer", _DEFAULT_SHUFFLE_BUFFER)),
            max_precursor_mz=max_precursor_mz,
            min_peak_intensity=float(
                config.get("min_peak_intensity", DEFAULT_MIN_PEAK_INTENSITY)
            ),
            peak_drop_min_intensity=float(
                config.get(
                    "peak_drop_min_intensity",
                    config.get("min_peak_intensity", DEFAULT_MIN_PEAK_INTENSITY),
                )
            ),
            peak_ordering=str(config.get("peak_ordering", "mz")),
            num_peaks=int(config.get("num_peaks", _NUM_PEAKS_OUTPUT)),
            use_precursor_token=bool(config.get("use_precursor_token", False)),
            dreams_dim=int(metadata.get("dreams_dim", 0)),
            precursor_peak_exclusion_window_da=float(
                config.get(
                    "precursor_peak_exclusion_window_da",
                    _DEFAULT_PRECURSOR_PEAK_EXCLUSION_WINDOW_DA,
                )
            ),
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
    ):
        split_files = {
            "massspec_train": self.train_files,
            "massspec_val": self.val_files,
            "massspec_test": self.test_files,
            "train": self.train_files,
            "val": self.val_files,
            "test": self.test_files,
        }[split]
        split_lengths = {
            "massspec_train": self.train_lengths,
            "massspec_val": self.val_lengths,
            "massspec_test": self.test_lengths,
            "train": self.train_lengths,
            "val": self.val_lengths,
            "test": self.test_lengths,
        }[split]
        dataset = _ProbeMemmapDataset(
            [
                {"dir": path, "length": length}
                for path, length in zip(split_files, split_lengths, strict=True)
            ]
        )
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=bool(shuffle),
            drop_last=bool(drop_remainder),
            num_workers=0,
            collate_fn=_ProbeBatchCollator(
                num_peaks=self.num_peaks,
                max_precursor_mz=self.max_precursor_mz,
                min_peak_intensity=self.min_peak_intensity,
                peak_drop_min_intensity=self.peak_drop_min_intensity,
                peak_ordering=peak_ordering or self.peak_ordering,
                use_precursor_token=self.use_precursor_token,
                precursor_peak_exclusion_window_da=self.precursor_peak_exclusion_window_da,
            ),
            generator=generator,
        )
        return _LoaderAdapter(loader)
