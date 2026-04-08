import csv
import json
import logging
import math
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import tensorflow as tf
from huggingface_hub import hf_hub_download
from ml_collections import config_dict
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from input_pipeline import _prepend_precursor_token_tf, apply_peak_transforms_tf
from utils.massspec_probe_targets import (
    FG_SMARTS,
    REGRESSION_TARGET_KEYS,
    build_probe_targets_for_rows,
)

logger = logging.getLogger(__name__)

tf.config.set_visible_devices([], "GPU")

_DEFAULT_BATCH_SIZE = 512
_DEFAULT_SHUFFLE_BUFFER = 10_000
_DEFAULT_TFRECORD_DIR = Path("data/gems_peaklist_tfrecord")
_DEFAULT_TFRECORD_BUFFER_SIZE = 250_000
_DEFAULT_MASSSPEC_NUM_SHARDS = 4
_DEFAULT_MAX_PRECURSOR_MZ = 1000.0
_DEFAULT_MIN_PEAK_INTENSITY = 1e-4
_NUM_PEAKS_INPUT = 128
_NUM_PEAKS_OUTPUT = 60
_FINGERPRINT_BITS = 1024
_FINGERPRINT_RADIUS = 2
_PEAK_MZ_MIN = 20.0
_PEAK_MZ_MAX = 1000.0
_METADATA_FILENAME = "metadata.json"


MASSSPEC_HF_REPO = "roman-bushuiev/MassSpecGym"
MASSSPEC_TSV_PATH = "data/MassSpecGym.tsv"
MASSSPEC_METADATA_VERSION = 2

NIST20_METADATA_VERSION = 3
NIST20_HF_REPO = "roman-bushuiev/GeMS"
NIST20_HF_FILENAME = (
    "data/DreaMS_Atlas/nist20_mona_clean_merged_spectra_dreams_hidden_nist20.hdf5"
)
_NIST20_SPLIT_SEED = 42
_NIST20_TRAIN_FRAC = 0.70
_NIST20_VAL_FRAC = 0.15

MONA_A_METADATA_VERSION = 1
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


def _normalize_spectra_intensity(spectra: np.ndarray) -> np.ndarray:
    """Per-spectrum max-normalize intensities (row 1) to [0, 1] in-place."""
    max_int = spectra[:, 1].max(axis=1, keepdims=True)  # (N, 1)
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
            if mz.size > _NUM_PEAKS_INPUT:
                idx = np.argpartition(intensity, -_NUM_PEAKS_INPUT)[-_NUM_PEAKS_INPUT:]
                idx = idx[np.argsort(intensity[idx])[::-1]]
                mz = mz[idx]
                intensity = intensity[idx]
            elif mz.size < _NUM_PEAKS_INPUT:
                pad = _NUM_PEAKS_INPUT - mz.size
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
        "retention": np.full(len(spectra), 392.3146, dtype=np.float32),
        "precursor": np.asarray(precursor, dtype=np.float32),
        "fold": np.asarray(fold),
        "smiles": np.asarray(smiles),
        "adduct": np.asarray(adduct),
        "instrument_type": np.asarray(instrument_type),
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
        if mol is None or (ik := InchiToInchiKey(MolToInchi(mol))) is None:
            continue
        valid_indices.append(i)
        inchikey_14.append(ik[:14])
    idx = np.asarray(valid_indices)
    inchikey_14_arr = np.asarray(inchikey_14)
    logger.info(
        "NIST20 HDF5: %d / %d spectra have valid SMILES", len(idx), len(raw_smiles)
    )
    unique_keys = sorted(set(inchikey_14_arr.tolist()))
    rng = np.random.RandomState(_NIST20_SPLIT_SEED)
    rng.shuffle(unique_keys)
    n_keys = len(unique_keys)
    n_train = int(n_keys * _NIST20_TRAIN_FRAC)
    n_val = int(n_keys * _NIST20_VAL_FRAC)
    train_keys = set(unique_keys[:n_train])
    val_keys = set(unique_keys[n_train : n_train + n_val])
    fold_arr = np.where(
        np.isin(inchikey_14_arr, list(train_keys)),
        "train",
        np.where(np.isin(inchikey_14_arr, list(val_keys)), "val", "test"),
    )
    n_valid = len(idx)
    logger.info(
        "NIST20 split: %d train, %d val, %d test across %d unique molecules",
        np.count_nonzero(fold_arr == "train"),
        np.count_nonzero(fold_arr == "val"),
        np.count_nonzero(fold_arr == "test"),
        n_keys,
    )
    return {
        "spectra": _normalize_spectra_intensity(raw_spectra[idx].astype(np.float32)),
        "retention": np.zeros(n_valid, dtype=np.float32),
        "precursor": raw_precursor[idx].astype(np.float32),
        "fold": fold_arr,
        "smiles": raw_smiles[idx],
        "adduct": raw_adduct[idx],
        "instrument_type": np.full(n_valid, "unknown", dtype=object),
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
    spectra = _normalize_spectra_intensity(
        np.stack(df["PARSED PEAKS"].to_numpy()).astype(np.float32)
    )  # (N, 2, 128)
    precursor = df["PRECURSOR M/Z"].to_numpy().astype(np.float32)
    smiles = df["SMILES"].to_numpy().astype(str)
    adduct = df["PRECURSOR TYPE"].fillna("unknown").to_numpy().astype(str)
    instrument_type = df["INSTRUMENT TYPE"].fillna("unknown").to_numpy().astype(str)

    ce_raw = df["COLLISION ENERGY"].to_numpy()
    collision_energy = np.zeros(n, dtype=np.float32)
    collision_energy_present = np.zeros(n, dtype=np.int32)
    for i, ce in enumerate(ce_raw):
        ce_str = str(ce).strip() if ce is not None else ""
        if ce_str and ce_str.lower() not in ("nan", "none", ""):
            try:
                collision_energy[i] = float(ce_str.split()[0])
                collision_energy_present[i] = 1
            except (ValueError, IndexError):
                pass

    # Murcko split: val=True -> test, val=False -> train
    fold = np.where(df["val"].to_numpy(), "test", "train")

    return {
        "spectra": spectra,
        "retention": np.zeros(n, dtype=np.float32),
        "precursor": precursor,
        "fold": fold,
        "smiles": smiles,
        "adduct": adduct,
        "instrument_type": instrument_type,
        "collision_energy": collision_energy,
        "collision_energy_present": collision_energy_present,
        "dreams_embedding": None,
    }


def _compute_morgan_fingerprints(smiles: np.ndarray) -> np.ndarray:
    fps = np.zeros((len(smiles), _FINGERPRINT_BITS), dtype=np.int8)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(s))
        fp = AllChem.GetMorganFingerprintAsBitVect(  # type: ignore[attr-defined]
            mol,
            _FINGERPRINT_RADIUS,
            nBits=_FINGERPRINT_BITS,
        )
        DataStructs.ConvertToNumpyArray(fp, fps[i])
    return fps


def _encode_categorical_ids(values: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    normalized = np.asarray([str(v) or "unknown" for v in values], dtype=object)
    categories = ["unknown"] + sorted(set(normalized.tolist()) - {"unknown"})
    vocab = {category: i for i, category in enumerate(categories)}
    return np.asarray([vocab[v] for v in normalized], dtype=np.int32), vocab


def _float_feat(v) -> tf.train.Feature:
    return tf.train.Feature(float_list=tf.train.FloatList(value=v))


def _int64_feat(v) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=v))


def _bytes_feat(v) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=v))


def _write_tfrecords_with_fingerprint(
    spectra: np.ndarray,
    retention: np.ndarray,
    precursor: np.ndarray,
    fingerprint: np.ndarray,
    smiles: np.ndarray,
    adduct_id: np.ndarray,
    instrument_type_id: np.ndarray,
    collision_energy: np.ndarray,
    collision_energy_present: np.ndarray,
    probe_mol_props: dict[str, np.ndarray],
    probe_fg_binary: dict[str, np.ndarray],
    probe_valid_mol: np.ndarray,
    output_path: Path,
    num_shards: int,
    dreams_embedding: np.ndarray | None = None,
) -> tuple[list[str], list[int]]:
    n = len(spectra)
    num_shards = max(1, min(num_shards, n))
    shard_size = math.ceil(n / num_shards)
    output_path.mkdir(parents=True, exist_ok=True)
    files, lengths = [], []
    for shard_id in range(num_shards):
        if (start := shard_id * shard_size) >= (end := min(start + shard_size, n)):
            break
        shard_file = output_path / f"shard-{shard_id:05d}-of-{num_shards:05d}.tfrecord"
        options = tf.io.TFRecordOptions(compression_type="GZIP")
        with tf.io.TFRecordWriter(str(shard_file), options=options) as writer:  # type: ignore[attr-defined]
            for i in range(start, end):
                feat = {
                    "mz": _float_feat(spectra[i, 0].astype(np.float32)),
                    "intensity": _float_feat(spectra[i, 1].astype(np.float32)),
                    "rt": _float_feat([retention[i]]),
                    "precursor_mz": _float_feat([precursor[i]]),
                    "fingerprint": _int64_feat(fingerprint[i].astype(np.int64)),
                    "smiles": _bytes_feat([str(smiles[i]).encode("utf-8")]),
                    "adduct_id": _int64_feat([int(adduct_id[i])]),
                    "instrument_type_id": _int64_feat([int(instrument_type_id[i])]),
                    "collision_energy": _float_feat([float(collision_energy[i])]),
                    "collision_energy_present": _int64_feat(
                        [int(collision_energy_present[i])]
                    ),
                    "probe_valid_mol": _int64_feat([int(probe_valid_mol[i])]),
                }
                if dreams_embedding is not None:
                    feat["dreams_embedding"] = _float_feat(dreams_embedding[i])
                for name in REGRESSION_TARGET_KEYS:
                    feat[f"probe_{name}"] = _float_feat(
                        [float(probe_mol_props[name][i])]
                    )
                for name in FG_SMARTS:
                    feat[f"probe_fg_{name}"] = _int64_feat(
                        [int(probe_fg_binary[name][i])]
                    )
                example = tf.train.Example(features=tf.train.Features(feature=feat))
                writer.write(example.SerializeToString())
        files.append(shard_file.name)
        lengths.append(end - start)
    return files, lengths


def _filter_encode_and_write(
    *,
    spectra: np.ndarray,
    retention: np.ndarray,
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
    fingerprints = _compute_morgan_fingerprints(smiles)
    adduct_id, adduct_vocab = _encode_categorical_ids(adduct)
    instrument_type_id, instrument_type_vocab = _encode_categorical_ids(instrument_type)
    keep = np.isfinite(precursor) & (precursor <= float(max_precursor_mz))
    spectra = spectra[keep]
    retention = retention[keep]
    precursor = precursor[keep]
    fold = fold[keep]
    smiles = smiles[keep]
    fingerprints = fingerprints[keep]
    adduct_id = adduct_id[keep]
    instrument_type_id = instrument_type_id[keep]
    collision_energy = collision_energy[keep]
    collision_energy_present = collision_energy_present[keep]
    if dreams_embedding is not None:
        dreams_embedding = dreams_embedding[keep]
    probe_mol_props, probe_fg_binary, probe_valid_mol = build_probe_targets_for_rows(
        smiles
    )
    result: dict[str, Any] = {
        "metadata_version": metadata_version,
        "adduct_vocab": adduct_vocab,
        "instrument_type_vocab": instrument_type_vocab,
        "max_precursor_mz": float(max_precursor_mz),
        "dreams_dim": int(dreams_embedding.shape[1]) if dreams_embedding is not None else 0,
    }
    s2, s4 = num_shards // 2, num_shards // 4
    for split_name, split_shards in [("train", s2), ("val", s4), ("test", s4)]:
        mask = fold == split_name
        files, lengths = _write_tfrecords_with_fingerprint(
            spectra[mask],
            retention[mask],
            precursor[mask],
            fingerprints[mask],
            smiles[mask],
            adduct_id[mask],
            instrument_type_id[mask],
            collision_energy[mask],
            collision_energy_present[mask],
            {k: v[mask] for k, v in probe_mol_props.items()},
            {k: v[mask] for k, v in probe_fg_binary.items()},
            probe_valid_mol[mask],
            output_dir / split_name,
            max(1, split_shards),
            dreams_embedding=dreams_embedding[mask] if dreams_embedding is not None else None,
        )
        result[f"{split_name}_files"] = files
        result[f"{split_name}_lengths"] = lengths
        result[f"{split_name}_size"] = int(np.count_nonzero(mask))
    return result


def _probe_metadata_valid(
    output_dir: Path,
    expected_version: int,
    max_precursor_mz: float,
) -> dict[str, Any] | None:
    if not (output_dir / _METADATA_FILENAME).exists():
        return None
    with (output_dir / _METADATA_FILENAME).open() as f:
        metadata = json.load(f)
    if int(metadata.get("metadata_version", 0)) != expected_version:
        return None
    if float(metadata.get("max_precursor_mz", float("inf"))) != float(max_precursor_mz):
        return None
    for split in ("train", "val", "test"):
        if not all(
            (output_dir / split / fn).exists()
            for fn in metadata.get(f"{split}_files", [])
        ):
            return None
    if "adduct_vocab" not in metadata or "instrument_type_vocab" not in metadata:
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
        logger.info("Found existing MassSpec probe TFRecords at %s", output_dir)
        return cached
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading MassSpecGym TSV...")
    tsv_path = _download_hf_file(MASSSPEC_HF_REPO, MASSSPEC_TSV_PATH, output_dir)
    logger.info("Loading MassSpecGym data...")
    metadata = _filter_encode_and_write(
        **_load_massspec_tsv(tsv_path),
        output_dir=output_dir,
        num_shards=num_shards,
        max_precursor_mz=max_precursor_mz,
        metadata_version=MASSSPEC_METADATA_VERSION,
    )
    with (output_dir / _METADATA_FILENAME).open("w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved MassSpec probe metadata to %s", output_dir / _METADATA_FILENAME)
    return metadata


def ensure_nist20_probe_prepared(
    output_dir: Path,
    *,
    max_precursor_mz: float,
    data_dir: Path,
    num_shards: int = _DEFAULT_MASSSPEC_NUM_SHARDS,
) -> dict[str, Any]:
    cached = _probe_metadata_valid(
        output_dir, NIST20_METADATA_VERSION, max_precursor_mz
    )
    if cached is not None:
        logger.info("Found existing NIST20 probe TFRecords at %s", output_dir)
        return cached
    output_dir.mkdir(parents=True, exist_ok=True)
    hdf5_path = data_dir / NIST20_HF_FILENAME
    if not hdf5_path.exists():
        logger.info("NIST20 HDF5 not found locally, downloading from HuggingFace...")
        hdf5_path = _download_hf_file(NIST20_HF_REPO, NIST20_HF_FILENAME, data_dir)
    logger.info("Loading NIST20+MoNA HDF5 from %s ...", hdf5_path)
    metadata = _filter_encode_and_write(
        **_load_nist20_hdf5(hdf5_path),
        output_dir=output_dir,
        num_shards=num_shards,
        max_precursor_mz=max_precursor_mz,
        metadata_version=NIST20_METADATA_VERSION,
    )
    with (output_dir / _METADATA_FILENAME).open("w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved NIST20 probe metadata to %s", output_dir / _METADATA_FILENAME)
    return metadata


def ensure_mona_a_probe_prepared(
    output_dir: Path,
    *,
    max_precursor_mz: float,
    data_dir: Path,
    num_shards: int = _DEFAULT_MASSSPEC_NUM_SHARDS,
) -> dict[str, Any]:
    cached = _probe_metadata_valid(output_dir, MONA_A_METADATA_VERSION, max_precursor_mz)
    if cached is not None:
        logger.info("Found existing MoNA-A probe TFRecords at %s", output_dir)
        return cached
    output_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = data_dir / MONA_A_HF_FILENAME
    if not pkl_path.exists():
        logger.info("MoNA-A pkl not found locally, downloading from HuggingFace...")
        pkl_path = _download_hf_file(MONA_A_HF_REPO, MONA_A_HF_FILENAME, data_dir)
    logger.info("Loading MoNA-A pkl from %s ...", pkl_path)
    metadata = _filter_encode_and_write(
        **_load_mona_a_pkl(pkl_path),
        output_dir=output_dir,
        num_shards=num_shards,
        max_precursor_mz=max_precursor_mz,
        metadata_version=MONA_A_METADATA_VERSION,
    )
    with (output_dir / _METADATA_FILENAME).open("w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved MoNA-A probe metadata to %s", output_dir / _METADATA_FILENAME)
    return metadata


def _parse_probe_batch(
    *,
    max_precursor_mz: float,
    min_peak_intensity: float,
    peak_ordering: str,
    num_peaks: int = _NUM_PEAKS_OUTPUT,
    dreams_dim: int = 0,
):
    peak_mz_min = tf.constant(_PEAK_MZ_MIN, tf.float32)
    peak_mz_max = tf.constant(_PEAK_MZ_MAX, tf.float32)
    min_int = tf.constant(min_peak_intensity, tf.float32)
    max_prec = tf.constant(max_precursor_mz, tf.float32)
    feature_spec = {
        "mz": tf.io.FixedLenFeature([_NUM_PEAKS_INPUT], tf.float32),
        "intensity": tf.io.FixedLenFeature([_NUM_PEAKS_INPUT], tf.float32),
        "rt": tf.io.FixedLenFeature([1], tf.float32),
        "precursor_mz": tf.io.FixedLenFeature([1], tf.float32),
        "fingerprint": tf.io.FixedLenFeature([_FINGERPRINT_BITS], tf.int64),
        "smiles": tf.io.FixedLenFeature([], tf.string),
        "adduct_id": tf.io.FixedLenFeature([1], tf.int64),
        "instrument_type_id": tf.io.FixedLenFeature([1], tf.int64),
        "collision_energy": tf.io.FixedLenFeature([1], tf.float32),
        "collision_energy_present": tf.io.FixedLenFeature([1], tf.int64),
        "probe_valid_mol": tf.io.FixedLenFeature([1], tf.int64),
    }
    for name in REGRESSION_TARGET_KEYS:
        feature_spec[f"probe_{name}"] = tf.io.FixedLenFeature([1], tf.float32)
    for name in FG_SMARTS:
        feature_spec[f"probe_fg_{name}"] = tf.io.FixedLenFeature([1], tf.int64)
    if dreams_dim > 0:
        feature_spec["dreams_embedding"] = tf.io.FixedLenFeature(
            [dreams_dim], tf.float32
        )

    @tf.function
    def transform(serialized_batch: tf.Tensor) -> dict[str, tf.Tensor]:
        parsed = tf.io.parse_example(serialized_batch, feature_spec)
        batch = apply_peak_transforms_tf(
            parsed["mz"],
            parsed["intensity"],
            parsed["precursor_mz"][:, 0],
            peak_mz_min=peak_mz_min,
            peak_mz_max=peak_mz_max,
            min_int=min_int,
            max_prec=max_prec,
            num_peaks=num_peaks,
            peak_ordering=peak_ordering,
        )
        batch["rt"] = parsed["rt"][:, 0]
        batch["fingerprint"] = tf.cast(parsed["fingerprint"], tf.int32)
        batch["smiles"] = parsed["smiles"]
        batch["adduct_id"] = tf.cast(parsed["adduct_id"][:, 0], tf.int32)
        batch["instrument_type_id"] = tf.cast(parsed["instrument_type_id"][:, 0], tf.int32)
        batch["collision_energy"] = parsed["collision_energy"][:, 0]
        batch["collision_energy_present"] = tf.cast(
            parsed["collision_energy_present"][:, 0], tf.int32
        )
        batch["probe_valid_mol"] = tf.cast(parsed["probe_valid_mol"][:, 0], tf.bool)
        for name in REGRESSION_TARGET_KEYS:
            batch[f"probe_{name}"] = parsed[f"probe_{name}"][:, 0]
        for name in FG_SMARTS:
            batch[f"probe_fg_{name}"] = tf.cast(
                parsed[f"probe_fg_{name}"][:, 0], tf.int32
            )
        if dreams_dim > 0:
            batch["dreams_embedding"] = parsed["dreams_embedding"]
        return batch

    return transform


def _build_probe_dataset(
    filenames: list[str],
    *,
    batch_size: int,
    shuffle_buffer: int,
    seed: int,
    drop_remainder: bool,
    tfrecord_buffer_size: int,
    max_precursor_mz: float,
    min_peak_intensity: float,
    peak_ordering: str,
    num_peaks: int = _NUM_PEAKS_OUTPUT,
    use_precursor_token: bool = False,
    dreams_dim: int = 0,
    num_parallel_reads: int = tf.data.AUTOTUNE,
) -> tf.data.Dataset:
    ds = tf.data.TFRecordDataset(
        filenames,
        compression_type="GZIP",
        buffer_size=int(tfrecord_buffer_size),
        num_parallel_reads=num_parallel_reads,
    )
    if shuffle_buffer > 0:
        ds = ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.map(
        _parse_probe_batch(
            max_precursor_mz=max_precursor_mz,
            min_peak_intensity=min_peak_intensity,
            peak_ordering=peak_ordering,
            num_peaks=num_peaks,
            dreams_dim=dreams_dim,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if use_precursor_token:
        ds = ds.map(_prepend_precursor_token_tf, num_parallel_calls=tf.data.AUTOTUNE)
    options = tf.data.Options()
    options.deterministic = True
    ds = ds.with_options(options)
    return ds.prefetch(tf.data.AUTOTUNE)


class MassSpecProbeData(NamedTuple):
    info: dict[str, Any]
    train_files: list[str]
    val_files: list[str]
    test_files: list[str]
    batch_size: int
    shuffle_buffer: int
    tfrecord_buffer_size: int
    max_precursor_mz: float
    min_peak_intensity: float
    peak_ordering: str
    num_peaks: int
    use_precursor_token: bool
    dreams_dim: int

    @classmethod
    def from_config(cls, config: config_dict.ConfigDict) -> "MassSpecProbeData":
        tfrecord_base = (
            Path(config.get("tfrecord_dir", str(_DEFAULT_TFRECORD_DIR)))
            .expanduser()
            .resolve()
        )
        max_precursor_mz = float(
            config.get("max_precursor_mz", _DEFAULT_MAX_PRECURSOR_MZ)
        )
        probe_dataset = str(config.get("probe_dataset", "massspec"))
        if probe_dataset == "nist20":
            output_dir = tfrecord_base / "nist20_probe"
            metadata = ensure_nist20_probe_prepared(
                output_dir,
                max_precursor_mz=max_precursor_mz,
                data_dir=tfrecord_base,
            )
        elif probe_dataset == "mona_a":
            output_dir = tfrecord_base / "mona_a_probe"
            metadata = ensure_mona_a_probe_prepared(
                output_dir,
                max_precursor_mz=max_precursor_mz,
                data_dir=tfrecord_base,
            )
        else:
            output_dir = tfrecord_base / "massspec_probe"
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
        split_files = {
            s: [str(output_dir / s / fn) for fn in metadata.get(f"{s}_files", [])]
            for s in ("train", "val", "test")
        }
        return cls(
            info=info,
            train_files=split_files["train"],
            val_files=split_files["val"],
            test_files=split_files["test"],
            batch_size=int(config.get("batch_size", _DEFAULT_BATCH_SIZE)),
            shuffle_buffer=int(config.get("shuffle_buffer", _DEFAULT_SHUFFLE_BUFFER)),
            tfrecord_buffer_size=int(
                config.get("tfrecord_buffer_size", _DEFAULT_TFRECORD_BUFFER_SIZE)
            ),
            max_precursor_mz=max_precursor_mz,
            min_peak_intensity=float(
                config.get("min_peak_intensity", _DEFAULT_MIN_PEAK_INTENSITY)
            ),
            peak_ordering=str(config.get("peak_ordering", "intensity")),
            num_peaks=int(config.get("num_peaks", _NUM_PEAKS_OUTPUT)),
            use_precursor_token=bool(config.get("use_precursor_token", False)),
            dreams_dim=int(metadata.get("dreams_dim", 0)),
        )

    def build_dataset(
        self,
        split: str,
        *,
        seed: int,
        peak_ordering: str | None = None,
        shuffle: bool = False,
        drop_remainder: bool = True,
        num_parallel_reads: int = tf.data.AUTOTUNE,
    ) -> tf.data.Dataset:
        files = {
            "massspec_train": self.train_files,
            "massspec_val": self.val_files,
            "massspec_test": self.test_files,
        }[split]
        return _build_probe_dataset(
            files,
            batch_size=self.batch_size,
            shuffle_buffer=self.shuffle_buffer if shuffle else 0,
            seed=seed,
            drop_remainder=drop_remainder,
            tfrecord_buffer_size=self.tfrecord_buffer_size,
            max_precursor_mz=self.max_precursor_mz,
            min_peak_intensity=self.min_peak_intensity,
            peak_ordering=peak_ordering or self.peak_ordering,
            num_peaks=self.num_peaks,
            use_precursor_token=self.use_precursor_token,
            dreams_dim=self.dreams_dim,
            num_parallel_reads=num_parallel_reads,
        )
