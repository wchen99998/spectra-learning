"""Build Murcko-split raw-MGF Parquet datasets and upload them to Hugging Face."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from spectra_learning.data.spectra import NUM_PEAKS_INPUT
from spectra_learning.probes.massspec.data import (
    NIST_MURCKO_ARTIFACT_FORMAT,
    NIST_MURCKO_HF_REPO,
    NIST_MURCKO_METADATA_VERSION,
    NIST_MURCKO_PREPARED_SUBDIR,
)
from spectra_learning.probes.massspec.nist_hdf5 import _to_float, iter_mgf
from spectra_learning.probes.massspec.targets import (
    MACCS_FINGERPRINT_BITS,
    MORGAN_PROBE_FINGERPRINT_BITS,
    MORGAN_PROBE_FINGERPRINT_RADIUS,
    REGRESSION_TARGET_KEYS,
)

log = logging.getLogger(__name__)
_MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_PROBE_FINGERPRINT_RADIUS,
    fpSize=MORGAN_PROBE_FINGERPRINT_BITS,
)

DEFAULT_NIST_MGF_URI = "gs://main-novogaia-bucket/MS/Datasets_with_structure/nist20/hr_msms_nist.mgf"
DEFAULT_MCEBIO_MGF_PATH = Path(
    "data/massive_msv000094528/source/20240411_mcebio_library_pos_all_lib_MS2.mgf"
)
MCEBIO_MURCKO_PREPARED_SUBDIR = "mcebio_murcko_probe"
RAW_SUBDIR = "raw"
SPLITS = ("train", "val", "test")
STANDALONE_SPLIT = "all"
DEFAULT_NIST_ALLOWED_ADDUCTS = ("[M+H]+",)
DEFAULT_NIST_SPLIT_SIZE_CAPS = {"train": 100_000, "val": 25_000, "test": 25_000}
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
    canonical_smiles: str
    murcko_hist_key: str
    murcko_hist_json: str


@dataclass(frozen=True)
class FullRow:
    row: dict[str, Any]
    morgan: np.ndarray


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


def _first_pass_task(
    payload: tuple[int, dict[str, Any], float, float, tuple[str, ...] | None],
) -> FirstPassRow | None:
    _, record, min_precursor_mz, max_precursor_mz, allowed_adducts = payload
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


def _build_dataset_specs(args: argparse.Namespace) -> list[DatasetSpec]:
    specs = [
        DatasetSpec(
            name="nist",
            source=args.nist_mgf,
            subdir=args.nist_subdir,
        )
    ]
    if args.mcebio_mgf:
        specs.append(
            DatasetSpec(
                name="mcebio",
                source=args.mcebio_mgf,
                subdir=args.mcebio_subdir,
            )
        )
    return specs


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
    args = parser.parse_args()

    work_dir = args.work_dir.expanduser().resolve()
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
    for spec in _build_dataset_specs(args):
        raw_mgf = _stage_raw_mgf(spec.source, raw_dir, args.gcs_credentials)
        allowed_adducts = (
            tuple(args.nist_allowed_adducts)
            if spec.name == "nist" and args.nist_allowed_adducts
            else None
        )
        split_size_caps = (
            {
                "train": args.nist_target_train_size,
                "val": args.nist_target_val_size,
                "test": args.nist_target_test_size,
            }
            if spec.name == "nist"
            else None
        )
        metadata = build_murcko_mgf_dataset(
            mgf_path=raw_mgf,
            output_dir=staging_root / spec.subdir.strip("/"),
            source_uri=spec.source,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            seed=args.seed,
            min_precursor_mz=args.min_precursor_mz,
            max_precursor_mz=args.max_precursor_mz,
            num_peaks_input=args.num_peaks_input,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            parquet_batch_size=args.parquet_batch_size,
            single_split=STANDALONE_SPLIT if spec.name == "mcebio" else None,
            allowed_adducts=allowed_adducts,
            split_size_caps=split_size_caps,
            spectral_lsh_threshold=args.spectral_lsh_threshold,
        )
        top_metadata["datasets"][spec.name] = {
            "subdir": spec.subdir.strip("/"),
            "source_raw_file": metadata["source_raw_file"],
            "splits": metadata["splits"],
        }
        for split in metadata["splits"]:
            top_metadata["datasets"][spec.name][f"{split}_size"] = metadata[f"{split}_size"]
    (staging_root / "metadata.json").write_text(json.dumps(top_metadata, indent=2, sort_keys=True))

    if args.skip_upload:
        log.info("--skip-upload set; staged dataset left at %s", staging_root)
        return

    api = HfApi()
    api.create_repo(
        args.hf_repo_id,
        repo_type="dataset",
        exist_ok=True,
        private=args.hf_private,
    )
    log.info("Uploading %s -> %s", staging_root, args.hf_repo_id)
    api.upload_large_folder(
        repo_id=args.hf_repo_id,
        folder_path=staging_root,
        repo_type="dataset",
        revision=args.hf_revision,
    )
    log.info("Uploaded Murcko MGF dataset to https://huggingface.co/datasets/%s", args.hf_repo_id)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
