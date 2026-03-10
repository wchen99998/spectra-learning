from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from huggingface_hub import hf_hub_download
from ml_collections import config_dict
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import AllChem
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


def _prepend_precursor_token_probe_tf(batch: dict) -> dict:
    """Prepend a precursor token at position 0 in the probe TF pipeline.

    Mirrors ``input_pipeline._prepend_precursor_token_tf`` but without
    ``context_mask``/``target_masks`` handling (probe batches don't have these).
    """
    peak_mz = batch["peak_mz"]  # [B, N]
    peak_intensity = batch["peak_intensity"]  # [B, N]
    peak_valid_mask = batch["peak_valid_mask"]  # [B, N]
    precursor_mz = batch["precursor_mz"]  # [B]
    B = tf.shape(peak_mz)[0]

    pre_mz = tf.expand_dims(precursor_mz, 1)  # [B, 1]
    pre_int = tf.fill([B, 1], -1.0)  # sentinel
    pre_valid = tf.ones([B, 1], dtype=tf.bool)

    out = dict(batch)
    out["peak_mz"] = tf.concat([pre_mz, peak_mz], axis=1)
    out["peak_intensity"] = tf.concat([pre_int, peak_intensity], axis=1)
    out["peak_valid_mask"] = tf.concat([pre_valid, peak_valid_mask], axis=1)

    del out["precursor_mz"]
    return out


MASSSPEC_HF_REPO = "roman-bushuiev/MassSpecGym"
MASSSPEC_TSV_PATH = "data/MassSpecGym.tsv"
MASSSPEC_METADATA_VERSION = 2

NIST20_METADATA_VERSION = 1
NIST20_HF_REPO = "roman-bushuiev/GeMS"
NIST20_HF_FILENAME = (
    "data/DreaMS_Atlas/nist20_mona_clean_merged_spectra_dreams_hidden_nist20.hdf5"
)
_NIST20_SPLIT_SEED = 42
_NIST20_TRAIN_FRAC = 0.70
_NIST20_VAL_FRAC = 0.15


def _download_hf_file(repo_id: str, filename: str, local_dir: Path) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=str(local_dir),
    )
    return Path(path)


def download_massspec_tsv(cache_dir: Path) -> Path:
    return _download_hf_file(MASSSPEC_HF_REPO, MASSSPEC_TSV_PATH, cache_dir)


def _ensure_nist20_hdf5(data_dir: Path) -> Path:
    local_path = data_dir / NIST20_HF_FILENAME
    if local_path.exists():
        return local_path
    logger.info("NIST20 HDF5 not found locally, downloading from HuggingFace...")
    return _download_hf_file(NIST20_HF_REPO, NIST20_HF_FILENAME, data_dir)


def _load_massspec_tsv(
    tsv_path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    spectra: list[np.ndarray] = []
    precursor: list[float] = []
    fold: list[str] = []
    smiles: list[str] = []
    adduct: list[str] = []
    instrument_type: list[str] = []
    collision_energy: list[float] = []
    collision_energy_present: list[int] = []

    with tsv_path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
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
            adduct.append(row["adduct"] if row["adduct"] else "unknown")
            instrument_type.append(
                row["instrument_type"] if row["instrument_type"] else "unknown"
            )
            if row["collision_energy"] == "":
                collision_energy.append(0.0)
                collision_energy_present.append(0)
            else:
                collision_energy.append(float(row["collision_energy"]))
                collision_energy_present.append(1)

    spectra_array = np.stack(spectra, axis=0)
    retention = np.full(len(spectra_array), 392.3146, dtype=np.float32)
    precursor_array = np.asarray(precursor, dtype=np.float32)
    fold_array = np.asarray(fold)
    smiles_array = np.asarray(smiles)
    adduct_array = np.asarray(adduct)
    instrument_type_array = np.asarray(instrument_type)
    collision_energy_array = np.asarray(collision_energy, dtype=np.float32)
    collision_energy_present_array = np.asarray(
        collision_energy_present, dtype=np.int32
    )
    return (
        spectra_array,
        retention,
        precursor_array,
        fold_array,
        smiles_array,
        adduct_array,
        instrument_type_array,
        collision_energy_array,
        collision_energy_present_array,
    )


def _load_nist20_hdf5(
    hdf5_path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    import h5py
    from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey

    with h5py.File(str(hdf5_path), "r") as f:
        raw_spectra = f["spectrum"][:]  # [N, 2, 128] float64
        raw_precursor = f["precursor_mz"][:]  # [N] float
        raw_smiles = f["smiles"][:].astype(str)  # [N] str
        raw_adduct = f["adduct"][:].astype(str)  # [N] str

    # Filter invalid SMILES and compute InChIKey connectivity layer
    valid_indices: list[int] = []
    inchikey_14: list[str] = []
    for i, smi in enumerate(raw_smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        inchi = MolToInchi(mol)
        if inchi is None:
            continue
        ik = InchiToInchiKey(inchi)
        if ik is None:
            continue
        valid_indices.append(i)
        inchikey_14.append(ik[:14])

    idx = np.asarray(valid_indices)
    inchikey_14_arr = np.asarray(inchikey_14)
    logger.info(
        "NIST20 HDF5: %d / %d spectra have valid SMILES", len(idx), len(raw_smiles)
    )

    # InChIKey-based train/val/test split
    unique_keys = sorted(set(inchikey_14_arr.tolist()))
    rng = np.random.RandomState(_NIST20_SPLIT_SEED)
    rng.shuffle(unique_keys)
    n_keys = len(unique_keys)
    n_train = int(n_keys * _NIST20_TRAIN_FRAC)
    n_val = int(n_keys * _NIST20_VAL_FRAC)
    train_keys = set(unique_keys[:n_train])
    val_keys = set(unique_keys[n_train : n_train + n_val])

    fold_list: list[str] = []
    for ik in inchikey_14_arr:
        if ik in train_keys:
            fold_list.append("train")
        elif ik in val_keys:
            fold_list.append("val")
        else:
            fold_list.append("test")

    n_valid = len(idx)
    spectra = raw_spectra[idx].astype(np.float32)  # [N_valid, 2, 128]
    precursor = raw_precursor[idx].astype(np.float32)
    smiles_arr = raw_smiles[idx]
    adduct_arr = raw_adduct[idx]

    retention = np.zeros(n_valid, dtype=np.float32)
    fold_arr = np.asarray(fold_list)
    instrument_type_arr = np.full(n_valid, "unknown", dtype=object)
    collision_energy_arr = np.zeros(n_valid, dtype=np.float32)
    collision_energy_present_arr = np.zeros(n_valid, dtype=np.int32)

    logger.info(
        "NIST20 split: %d train, %d val, %d test across %d unique molecules",
        np.count_nonzero(fold_arr == "train"),
        np.count_nonzero(fold_arr == "val"),
        np.count_nonzero(fold_arr == "test"),
        n_keys,
    )

    return (
        spectra,
        retention,
        precursor,
        fold_arr,
        smiles_arr,
        adduct_arr,
        instrument_type_arr,
        collision_energy_arr,
        collision_energy_present_arr,
    )


def _compute_morgan_fingerprints(smiles: np.ndarray) -> np.ndarray:
    fps = np.zeros((len(smiles), _FINGERPRINT_BITS), dtype=np.int8)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(s))
        fp = AllChem.GetMorganFingerprintAsBitVect(  # type: ignore[attr-defined]
            mol,
            _FINGERPRINT_RADIUS,
            nBits=_FINGERPRINT_BITS,
        )
        arr = np.zeros((_FINGERPRINT_BITS,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps[i] = arr
    return fps


def _encode_categorical_ids(values: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    normalized = np.asarray(
        [str(v) if str(v) != "" else "unknown" for v in values],
        dtype=object,
    )
    categories = sorted(set(normalized.tolist()))
    if "unknown" in categories:
        categories = ["unknown"] + [c for c in categories if c != "unknown"]
    else:
        categories = ["unknown"] + categories
    vocab = {category: i for i, category in enumerate(categories)}
    ids = np.asarray([vocab[str(v)] for v in normalized], dtype=np.int32)
    return ids, vocab


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
    desc: str,
) -> tuple[list[str], list[int]]:
    n = len(spectra)
    num_shards = max(1, min(num_shards, n))
    shard_size = math.ceil(n / num_shards)

    output_path.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    lengths: list[int] = []

    for shard_id in range(num_shards):
        start = shard_id * shard_size
        end = min(start + shard_size, n)
        if start >= end:
            break

        shard_file = output_path / f"shard-{shard_id:05d}-of-{num_shards:05d}.tfrecord"
        options = tf.io.TFRecordOptions(compression_type="GZIP")

        with tf.io.TFRecordWriter(str(shard_file), options=options) as writer:  # type: ignore[attr-defined]
            for i in range(start, end):
                mz = spectra[i, 0].astype(np.float32)
                intensity = spectra[i, 1].astype(np.float32)
                fp = fingerprint[i].astype(np.int64)
                probe_features = {
                    f"probe_{name}": tf.train.Feature(
                        float_list=tf.train.FloatList(
                            value=[float(probe_mol_props[name][i])]
                        )
                    )
                    for name in REGRESSION_TARGET_KEYS
                }
                probe_features.update(
                    {
                        f"probe_fg_{name}": tf.train.Feature(
                            int64_list=tf.train.Int64List(
                                value=[int(probe_fg_binary[name][i])]
                            )
                        )
                        for name in FG_SMARTS
                    }
                )
                probe_features["probe_valid_mol"] = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[int(probe_valid_mol[i])])
                )
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "mz": tf.train.Feature(
                                float_list=tf.train.FloatList(value=mz)
                            ),
                            "intensity": tf.train.Feature(
                                float_list=tf.train.FloatList(value=intensity)
                            ),
                            "rt": tf.train.Feature(
                                float_list=tf.train.FloatList(value=[retention[i]])
                            ),
                            "precursor_mz": tf.train.Feature(
                                float_list=tf.train.FloatList(value=[precursor[i]])
                            ),
                            "fingerprint": tf.train.Feature(
                                int64_list=tf.train.Int64List(value=fp)
                            ),
                            "smiles": tf.train.Feature(
                                bytes_list=tf.train.BytesList(
                                    value=[str(smiles[i]).encode("utf-8")]
                                )
                            ),
                            "adduct_id": tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[int(adduct_id[i])])
                            ),
                            "instrument_type_id": tf.train.Feature(
                                int64_list=tf.train.Int64List(
                                    value=[int(instrument_type_id[i])]
                                )
                            ),
                            "collision_energy": tf.train.Feature(
                                float_list=tf.train.FloatList(
                                    value=[float(collision_energy[i])]
                                )
                            ),
                            "collision_energy_present": tf.train.Feature(
                                int64_list=tf.train.Int64List(
                                    value=[int(collision_energy_present[i])]
                                )
                            ),
                            **probe_features,
                        }
                    )
                )
                writer.write(example.SerializeToString())

        files.append(shard_file.name)
        lengths.append(end - start)

    return files, lengths


def _process_massspec_probe_data(
    output_dir: Path,
    num_shards: int,
    *,
    max_precursor_mz: float,
) -> dict[str, Any]:
    logger.info("Downloading MassSpecGym TSV...")
    tsv_path = download_massspec_tsv(output_dir)

    logger.info("Loading MassSpecGym data...")
    (
        spectra,
        retention,
        precursor,
        fold,
        smiles,
        adduct,
        instrument_type,
        collision_energy,
        collision_energy_present,
    ) = _load_massspec_tsv(tsv_path)
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
    probe_mol_props, probe_fg_binary, probe_valid_mol = build_probe_targets_for_rows(
        smiles
    )

    splits = [
        ("train", num_shards // 2),
        ("val", num_shards // 4),
        ("test", num_shards // 4),
    ]
    result: dict[str, Any] = {
        "metadata_version": MASSSPEC_METADATA_VERSION,
        "adduct_vocab": adduct_vocab,
        "instrument_type_vocab": instrument_type_vocab,
        "max_precursor_mz": float(max_precursor_mz),
    }
    for split_name, split_shards in splits:
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
            desc=f"MassSpec {split_name.capitalize()}",
        )
        result[f"{split_name}_files"] = files
        result[f"{split_name}_lengths"] = lengths
        result[f"{split_name}_size"] = int(np.count_nonzero(mask))
    return result


def _process_nist20_probe_data(
    output_dir: Path,
    num_shards: int,
    *,
    max_precursor_mz: float,
    data_dir: Path,
) -> dict[str, Any]:
    hdf5_path = _ensure_nist20_hdf5(data_dir)
    logger.info("Loading NIST20+MoNA HDF5 from %s ...", hdf5_path)
    (
        spectra,
        retention,
        precursor,
        fold,
        smiles,
        adduct,
        instrument_type,
        collision_energy,
        collision_energy_present,
    ) = _load_nist20_hdf5(hdf5_path)
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
    probe_mol_props, probe_fg_binary, probe_valid_mol = build_probe_targets_for_rows(
        smiles
    )

    splits = [
        ("train", num_shards // 2),
        ("val", num_shards // 4),
        ("test", num_shards // 4),
    ]
    result: dict[str, Any] = {
        "metadata_version": NIST20_METADATA_VERSION,
        "adduct_vocab": adduct_vocab,
        "instrument_type_vocab": instrument_type_vocab,
        "max_precursor_mz": float(max_precursor_mz),
    }
    for split_name, split_shards in splits:
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
            desc=f"NIST20 {split_name.capitalize()}",
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
    metadata_path = output_dir / _METADATA_FILENAME
    if not metadata_path.exists():
        return None
    with metadata_path.open() as f:
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
    metadata = _process_massspec_probe_data(
        output_dir, num_shards, max_precursor_mz=max_precursor_mz
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
    metadata = _process_nist20_probe_data(
        output_dir, num_shards, max_precursor_mz=max_precursor_mz, data_dir=data_dir
    )
    with (output_dir / _METADATA_FILENAME).open("w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved NIST20 probe metadata to %s", output_dir / _METADATA_FILENAME)
    return metadata


def _parse_probe_batch(
    *,
    max_precursor_mz: float,
    min_peak_intensity: float,
    peak_ordering: str,
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

    @tf.function
    def transform(serialized_batch: tf.Tensor) -> dict[str, tf.Tensor]:
        parsed = tf.io.parse_example(serialized_batch, feature_spec)
        mz = parsed["mz"]
        intensity = parsed["intensity"]
        rt = parsed["rt"][:, 0]
        precursor_mz_val = parsed["precursor_mz"][:, 0]

        keep = (mz >= peak_mz_min) & (mz <= peak_mz_max)
        mz = tf.where(keep, mz, 0.0)
        intensity = tf.where(keep, intensity, 0.0)

        keep2 = intensity >= min_int
        mz = tf.where(keep2, mz, 0.0)
        intensity = tf.where(keep2, intensity, 0.0)

        values, indices = tf.math.top_k(intensity, k=_NUM_PEAKS_OUTPUT, sorted=True)
        intensity = values
        mz = tf.gather(mz, indices, batch_dims=1)

        max_intensity = tf.reduce_max(intensity, axis=1, keepdims=True)
        max_intensity = tf.maximum(max_intensity, 1e-8)
        intensity = intensity / max_intensity

        valid = intensity > 0
        if peak_ordering == "mz":
            sort_key = tf.where(valid, mz, tf.fill(tf.shape(mz), float("inf")))
            sorted_idx = tf.argsort(
                sort_key, axis=1, direction="ASCENDING", stable=True
            )
        else:
            sort_key = tf.where(
                valid,
                intensity,
                tf.fill(tf.shape(intensity), float("-inf")),
            )
            sorted_idx = tf.argsort(
                sort_key,
                axis=1,
                direction="DESCENDING",
                stable=True,
            )
        mz = tf.gather(mz, sorted_idx, batch_dims=1)
        intensity = tf.gather(intensity, sorted_idx, batch_dims=1)
        valid = tf.gather(valid, sorted_idx, batch_dims=1)
        mz = tf.where(valid, mz, 0.0)
        intensity = tf.where(valid, intensity, 0.0)

        batch = {
            "peak_mz": mz / peak_mz_max,
            "peak_intensity": tf.where(valid, intensity, 0.0),
            "peak_valid_mask": valid,
            "precursor_mz": tf.clip_by_value(precursor_mz_val, 0.0, max_prec)
            / max_prec,
            "rt": rt,
            "mz": mz,
            "intensity": intensity,
            "fingerprint": tf.cast(parsed["fingerprint"], tf.int32),
            "smiles": parsed["smiles"],
            "adduct_id": tf.cast(parsed["adduct_id"][:, 0], tf.int32),
            "instrument_type_id": tf.cast(parsed["instrument_type_id"][:, 0], tf.int32),
            "collision_energy": parsed["collision_energy"][:, 0],
            "collision_energy_present": tf.cast(
                parsed["collision_energy_present"][:, 0],
                tf.int32,
            ),
            "probe_valid_mol": tf.cast(parsed["probe_valid_mol"][:, 0], tf.bool),
        }
        for name in REGRESSION_TARGET_KEYS:
            batch[f"probe_{name}"] = parsed[f"probe_{name}"][:, 0]
        for name in FG_SMARTS:
            batch[f"probe_fg_{name}"] = tf.cast(
                parsed[f"probe_fg_{name}"][:, 0], tf.int32
            )
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
    use_precursor_token: bool = False,
    num_parallel_reads: int | None = None,
) -> tf.data.Dataset:
    if num_parallel_reads is None:
        num_parallel_reads = tf.data.AUTOTUNE
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
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if use_precursor_token:
        ds = ds.map(
            _prepend_precursor_token_probe_tf, num_parallel_calls=tf.data.AUTOTUNE
        )
    options = tf.data.Options()
    options.deterministic = True
    ds = ds.with_options(options)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


@dataclass(slots=True)
class MassSpecProbeData:
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
    use_precursor_token: bool

    @classmethod
    def from_config(cls, config: config_dict.ConfigDict) -> "MassSpecProbeData":
        probe_dataset = str(config.get("probe_dataset", "massspec"))
        tfrecord_base = (
            Path(config.get("tfrecord_dir", str(_DEFAULT_TFRECORD_DIR)))
            .expanduser()
            .resolve()
        )
        max_precursor_mz = float(
            config.get("max_precursor_mz", _DEFAULT_MAX_PRECURSOR_MZ)
        )

        if probe_dataset == "nist20":
            output_dir = tfrecord_base / "nist20_probe"
            metadata = ensure_nist20_probe_prepared(
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

        info = {
            "massspec_train_size": int(metadata.get("train_size", 0)),
            "massspec_val_size": int(metadata.get("val_size", 0)),
            "massspec_test_size": int(metadata.get("test_size", 0)),
            "massspec_metadata_version": int(metadata.get("metadata_version", 0)),
            "massspec_adduct_vocab": metadata.get("adduct_vocab", {"unknown": 0}),
            "massspec_instrument_type_vocab": metadata.get(
                "instrument_type_vocab",
                {"unknown": 0},
            ),
            "massspec_adduct_vocab_size": len(
                metadata.get("adduct_vocab", {"unknown": 0})
            ),
            "massspec_instrument_type_vocab_size": len(
                metadata.get("instrument_type_vocab", {"unknown": 0})
            ),
            "fingerprint_bits": _FINGERPRINT_BITS,
        }
        return cls(
            info=info,
            train_files=[
                str(output_dir / "train" / fn) for fn in metadata.get("train_files", [])
            ],
            val_files=[
                str(output_dir / "val" / fn) for fn in metadata.get("val_files", [])
            ],
            test_files=[
                str(output_dir / "test" / fn) for fn in metadata.get("test_files", [])
            ],
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
            use_precursor_token=bool(config.get("use_precursor_token", False)),
        )

    def build_dataset(
        self,
        split: str,
        *,
        seed: int,
        peak_ordering: str | None = None,
        shuffle: bool = False,
        drop_remainder: bool = True,
        num_parallel_reads: int | None = None,
    ) -> tf.data.Dataset:
        files = {
            "massspec_train": self.train_files,
            "massspec_val": self.val_files,
            "massspec_test": self.test_files,
        }[split]
        if peak_ordering is None:
            peak_ordering = self.peak_ordering
        return _build_probe_dataset(
            files,
            batch_size=self.batch_size,
            shuffle_buffer=self.shuffle_buffer if shuffle else 0,
            seed=seed,
            drop_remainder=drop_remainder,
            tfrecord_buffer_size=self.tfrecord_buffer_size,
            max_precursor_mz=self.max_precursor_mz,
            min_peak_intensity=self.min_peak_intensity,
            peak_ordering=peak_ordering,
            use_precursor_token=self.use_precursor_token,
            num_parallel_reads=num_parallel_reads,
        )
