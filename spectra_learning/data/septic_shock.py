from __future__ import annotations

import base64
import hashlib
import json
import re
import urllib.request
import zipfile
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, NamedTuple
import xml.etree.ElementTree as ET

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

from spectra_learning.data.gems.collate import GemsBatchCollator
from spectra_learning.data.spectra import (
    DEFAULT_GROUPED_PEAK_ISOTOPE_CHARGES,
    DEFAULT_GROUPED_PEAK_SHOULDER_DA,
    DEFAULT_MAX_PRECURSOR_MZ,
    DEFAULT_MIN_PEAK_INTENSITY,
    DEFAULT_PEAK_FILTERING,
    DEFAULT_PRECURSOR_PEAK_EXCLUSION_WINDOW_DA,
    NUM_PEAKS_INPUT,
)

SEPTIC_SHOCK_METADATA_VERSION = 1
SEPTIC_SHOCK_STUDY_ID = "ST003189"
SEPTIC_SHOCK_ANALYSIS_ID = "AN005237"
SEPTIC_SHOCK_MWTAB_URL = (
    "https://www.metabolomicsworkbench.org/data/study_textformat_view.php?"
    f"ANALYSIS_ID={SEPTIC_SHOCK_ANALYSIS_ID}&STUDY_ID={SEPTIC_SHOCK_STUDY_ID}"
)
SEPTIC_SHOCK_RESULTS_URL = (
    "https://www.metabolomicsworkbench.org/studydownload/"
    f"{SEPTIC_SHOCK_STUDY_ID}_{SEPTIC_SHOCK_ANALYSIS_ID}_Results.txt"
)
SEPTIC_SHOCK_RAW_URL = (
    "https://www.metabolomicsworkbench.org/studydownload/"
    f"{SEPTIC_SHOCK_STUDY_ID}_Rawdata.zip"
)
SEPTIC_SHOCK_RAW_FILENAME = f"{SEPTIC_SHOCK_STUDY_ID}_Rawdata.zip"
SEPTIC_SHOCK_MWTAB_FILENAME = (
    f"{SEPTIC_SHOCK_STUDY_ID}_{SEPTIC_SHOCK_ANALYSIS_ID}_mwtab.txt"
)
SEPTIC_SHOCK_RAW_BYTES = 2_544_920_290
SEPTIC_SHOCK_RAW_MD5 = "521ac9c60f48f22782ff1a385c396983"
SEPTIC_SHOCK_SPLIT_SEED = 42
SEPTIC_SHOCK_TEST_SIZE = 0.30
SEPTIC_SHOCK_VAL_SIZE_OF_DEV = 0.20
SEPTIC_SHOCK_POSITIVE_LABEL = "Septic shock"
SEPTIC_SHOCK_NEGATIVE_LABEL = "Non-septic shock"
SEPTIC_SHOCK_ARTIFACT_FORMAT = "raw_peaklist_v1"
SEPTIC_SHOCK_TASK = "septic_shock_st003189"

USE_PYTEOMICS_MZXML = True


class SepticShockData(NamedTuple):
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
    ms_level: int | None


def parse_septic_shock_mwtab(text: str) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) < 5 or parts[0].strip() != "SUBJECT_SAMPLE_FACTORS":
            continue
        label_match = re.search(r"Subject_ID:([^|]+)", parts[3])
        raw_match = re.search(r"RAW_FILE_NAME\(Raw file name\)=([^;\t]+)", parts[4])
        label_name = label_match.group(1).strip()
        raw_file_name = raw_match.group(1).strip()
        samples.append(
            {
                "sample_index": len(samples),
                "sample_id": parts[2].strip(),
                "raw_file_name": raw_file_name,
                "label_name": label_name,
                "label": 1 if label_name == SEPTIC_SHOCK_POSITIVE_LABEL else 0,
            }
        )
    return samples


def make_septic_shock_splits(
    samples: list[dict[str, Any]],
    *,
    split_seed: int = SEPTIC_SHOCK_SPLIT_SEED,
) -> dict[str, list[int]]:
    labels = np.asarray([sample["label"] for sample in samples], dtype=np.int64)
    indices = np.arange(len(samples), dtype=np.int64)
    dev_idx, test_idx = train_test_split(
        indices,
        test_size=SEPTIC_SHOCK_TEST_SIZE,
        random_state=split_seed,
        stratify=labels,
    )
    train_idx, val_idx = train_test_split(
        dev_idx,
        test_size=SEPTIC_SHOCK_VAL_SIZE_OF_DEV,
        random_state=split_seed,
        stratify=labels[dev_idx],
    )
    return {
        "train": sorted(int(idx) for idx in train_idx),
        "val": sorted(int(idx) for idx in val_idx),
        "test": sorted(int(idx) for idx in test_idx),
    }


def build_septic_shock_sample_records(
    samples: list[dict[str, Any]],
    *,
    split_seed: int = SEPTIC_SHOCK_SPLIT_SEED,
    raw_subdir: str = "raw/mzxml",
) -> list[dict[str, Any]]:
    splits = make_septic_shock_splits(samples, split_seed=split_seed)
    split_by_index = {
        sample_index: split
        for split, indices in splits.items()
        for sample_index in indices
    }
    records: list[dict[str, Any]] = []
    for sample in samples:
        record = dict(sample)
        record["split"] = split_by_index[int(sample["sample_index"])]
        record["mzxml_path"] = sample.get(
            "mzxml_path",
            f"{raw_subdir}/{sample['raw_file_name']}",
        )
        records.append(record)
    return records


def septic_shock_split_counts(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        split_records = [record for record in records if record["split"] == split]
        positive = sum(int(record["label"]) for record in split_records)
        counts[split] = {
            "size": len(split_records),
            "positive": positive,
            "negative": len(split_records) - positive,
        }
    return counts


def _download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, output_path.open("wb") as f:
        while chunk := response.read(1024 * 1024):
            f.write(chunk)


def _file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _find_extracted_mzxml(raw_dir: Path, raw_file_name: str) -> Path:
    matches = sorted((raw_dir / "mzxml").rglob(raw_file_name))
    return matches[0]


def prepare_septic_shock_dataset(
    cache_dir: Path,
    *,
    download_raw: bool = False,
    split_seed: int = SEPTIC_SHOCK_SPLIT_SEED,
    force: bool = False,
) -> dict[str, Any]:
    cache_dir = cache_dir.expanduser().resolve()
    raw_dir = cache_dir / "raw"
    mwtab_path = raw_dir / SEPTIC_SHOCK_MWTAB_FILENAME
    if force or not mwtab_path.exists():
        _download_file(SEPTIC_SHOCK_MWTAB_URL, mwtab_path)
    samples = parse_septic_shock_mwtab(mwtab_path.read_text())
    records = build_septic_shock_sample_records(samples, split_seed=split_seed)

    archive_path = raw_dir / SEPTIC_SHOCK_RAW_FILENAME
    if download_raw:
        if (
            force
            or not archive_path.exists()
            or archive_path.stat().st_size != SEPTIC_SHOCK_RAW_BYTES
        ):
            _download_file(SEPTIC_SHOCK_RAW_URL, archive_path)
        raw_size = archive_path.stat().st_size
        raw_md5 = _file_md5(archive_path)
        if raw_size != SEPTIC_SHOCK_RAW_BYTES:
            raise ValueError(f"Expected {SEPTIC_SHOCK_RAW_BYTES} bytes, got {raw_size}")
        if raw_md5 != SEPTIC_SHOCK_RAW_MD5:
            raise ValueError(f"Expected MD5 {SEPTIC_SHOCK_RAW_MD5}, got {raw_md5}")
        extract_dir = raw_dir / "mzxml"
        extract_dir.mkdir(parents=True, exist_ok=True)
        if force or len(list(extract_dir.rglob("*.mzXML"))) < len(records):
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(extract_dir)
        for record in records:
            path = _find_extracted_mzxml(raw_dir, str(record["raw_file_name"]))
            record["mzxml_path"] = str(path.relative_to(cache_dir))

    split_counts = septic_shock_split_counts(records)
    label_counts = Counter(record["label_name"] for record in records)
    metadata: dict[str, Any] = {
        "metadata_version": SEPTIC_SHOCK_METADATA_VERSION,
        "study_id": SEPTIC_SHOCK_STUDY_ID,
        "analysis_id": SEPTIC_SHOCK_ANALYSIS_ID,
        "mwtab_url": SEPTIC_SHOCK_MWTAB_URL,
        "results_url": SEPTIC_SHOCK_RESULTS_URL,
        "raw_url": SEPTIC_SHOCK_RAW_URL,
        "raw_archive_filename": SEPTIC_SHOCK_RAW_FILENAME,
        "raw_archive_bytes": SEPTIC_SHOCK_RAW_BYTES,
        "raw_archive_md5": SEPTIC_SHOCK_RAW_MD5,
        "downloaded_raw": bool(download_raw),
        "split_seed": int(split_seed),
        "test_size": SEPTIC_SHOCK_TEST_SIZE,
        "val_size_of_dev": SEPTIC_SHOCK_VAL_SIZE_OF_DEV,
        "num_samples": len(records),
        "label_counts": dict(label_counts),
        "split_counts": split_counts,
        "samples": records,
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        split_records = [record for record in records if record["split"] == split]
        (cache_dir / f"{split}.json").write_text(json.dumps(split_records, indent=2))
    (cache_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def load_septic_shock_metadata(cache_dir: Path) -> dict[str, Any]:
    return json.loads((cache_dir / "metadata.json").read_text())


def _write_peaklist_shard(
    output_dir: Path,
    *,
    split: str,
    spectra: np.ndarray,
    precursor_mz: np.ndarray,
    sample_index: np.ndarray,
) -> tuple[str, int]:
    shard_name = "shard-00000-of-00001"
    shard_dir = output_dir / split / shard_name
    shard_dir.mkdir(parents=True, exist_ok=True)
    np.save(shard_dir / "spectra.npy", spectra.astype(np.float32, copy=False))
    np.save(
        shard_dir / "precursor_mz_raw.npy",
        precursor_mz.astype(np.float32, copy=False),
    )
    np.save(shard_dir / "sample_index.npy", sample_index.astype(np.int64, copy=False))
    return shard_name, int(spectra.shape[0])


def _artifact_readme(metadata: dict[str, Any]) -> str:
    counts = metadata["split_counts"]
    return "\n".join(
        [
            "# ST003189 Septic-Shock Raw Peaklist Artifact",
            "",
            "This subdirectory contains a project-native raw peaklist artifact for",
            "Metabolomics Workbench ST003189 / PR001985.",
            "",
            f"- Artifact format: `{metadata['artifact_format']}`",
            f"- Input peak slots per scan: `{metadata['num_peaks_input']}`",
            "- Model-facing preprocessing is intentionally not baked into the files.",
            "  Use `build_septic_shock_loader` so filtering, normalization, ordering,",
            "  precursor clipping, grouped peak filtering, and padding match the rest",
            "  of the project.",
            f"- Train samples: {counts['train']['size']} ({counts['train']['positive']} septic shock)",
            f"- Val samples: {counts['val']['size']} ({counts['val']['positive']} septic shock)",
            f"- Test samples: {counts['test']['size']} ({counts['test']['positive']} septic shock)",
            "",
            "Each split has one shard with `spectra.npy`, `precursor_mz_raw.npy`,",
            "and `sample_index.npy`. Labels and sample-level scan ranges are in",
            "`metadata.json`.",
            "",
        ]
    )


def build_septic_shock_peaklist_artifact(
    *,
    cache_dir: Path,
    output_dir: Path,
    ms_level: int | None = None,
    split_seed: int = SEPTIC_SHOCK_SPLIT_SEED,
) -> dict[str, Any]:
    cache_dir = cache_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    source_metadata = load_septic_shock_metadata(cache_dir)
    records = build_septic_shock_sample_records(
        source_metadata["samples"],
        split_seed=split_seed,
    )
    metadata_records: list[dict[str, Any]] = []
    split_metadata: dict[str, Any] = {}
    split_counts = septic_shock_split_counts(records)

    for split in ("train", "val", "test"):
        split_records = [record for record in records if record["split"] == split]
        spectra_chunks, precursor_chunks, sample_chunks = [], [], []
        scan_start = 0
        for record in split_records:
            spectra, precursor_mz = read_mzxml_sample_spectra(
                cache_dir / record["mzxml_path"],
                ms_level=ms_level,
            )
            artifact_record = dict(record)
            artifact_record["scan_start"] = int(scan_start)
            artifact_record["scan_count"] = int(spectra.shape[0])
            artifact_record["mzxml_path"] = str(record["mzxml_path"])
            metadata_records.append(artifact_record)
            spectra_chunks.append(spectra)
            precursor_chunks.append(precursor_mz)
            sample_chunks.append(
                np.full(spectra.shape[0], int(record["sample_index"]), dtype=np.int64)
            )
            scan_start += int(spectra.shape[0])

        split_spectra = np.concatenate(spectra_chunks, axis=0)
        split_precursor = np.concatenate(precursor_chunks, axis=0)
        split_sample_index = np.concatenate(sample_chunks, axis=0)
        shard_name, shard_length = _write_peaklist_shard(
            output_dir,
            split=split,
            spectra=split_spectra,
            precursor_mz=split_precursor,
            sample_index=split_sample_index,
        )
        split_metadata[f"{split}_shards"] = [shard_name]
        split_metadata[f"{split}_scan_lengths"] = [shard_length]
        split_metadata[f"{split}_num_scans"] = shard_length
        split_metadata[f"{split}_size"] = split_counts[split]["size"]
        split_metadata[f"{split}_positive"] = split_counts[split]["positive"]
        split_metadata[f"{split}_negative"] = split_counts[split]["negative"]

    label_counts = Counter(record["label_name"] for record in records)
    metadata: dict[str, Any] = {
        "metadata_version": SEPTIC_SHOCK_METADATA_VERSION,
        "task": SEPTIC_SHOCK_TASK,
        "artifact_format": SEPTIC_SHOCK_ARTIFACT_FORMAT,
        "num_peaks_input": NUM_PEAKS_INPUT,
        "study_id": SEPTIC_SHOCK_STUDY_ID,
        "analysis_id": SEPTIC_SHOCK_ANALYSIS_ID,
        "mwtab_url": SEPTIC_SHOCK_MWTAB_URL,
        "results_url": SEPTIC_SHOCK_RESULTS_URL,
        "raw_url": SEPTIC_SHOCK_RAW_URL,
        "raw_archive_bytes": SEPTIC_SHOCK_RAW_BYTES,
        "raw_archive_md5": SEPTIC_SHOCK_RAW_MD5,
        "split_seed": int(split_seed),
        "test_size": SEPTIC_SHOCK_TEST_SIZE,
        "val_size_of_dev": SEPTIC_SHOCK_VAL_SIZE_OF_DEV,
        "ms_level": ms_level,
        "num_samples": len(records),
        "label_counts": dict(label_counts),
        "split_counts": split_counts,
        "samples": metadata_records,
    }
    metadata.update(split_metadata)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (output_dir / "README.md").write_text(_artifact_readme(metadata))
    return metadata


def _strip_namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _decode_mzxml_peaks(peaks: ET.Element) -> tuple[np.ndarray, np.ndarray]:
    precision = int(peaks.attrib.get("precision", "32"))
    dtype = ">f4" if precision == 32 else ">f8"
    payload = base64.b64decode((peaks.text or "").strip())
    if peaks.attrib.get("compressionType", "none") == "zlib":
        payload = zlib.decompress(payload)
    values = np.frombuffer(payload, dtype=dtype).astype(np.float32)
    pairs = values.reshape(-1, 2)
    pair_order = peaks.attrib.get("pairOrder", "m/z-int")
    if pair_order == "int-m/z":
        return pairs[:, 1], pairs[:, 0]
    return pairs[:, 0], pairs[:, 1]


def _precursor_from_xml_scan(scan: ET.Element) -> float:
    for child in scan:
        if _strip_namespace(child.tag) == "precursorMz":
            return float((child.text or "0").strip())
    return 0.0


def _read_mzxml_scan_arrays_xml(
    path: Path,
    *,
    ms_level: int | None,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    root = ET.parse(path).getroot()
    scans: list[tuple[np.ndarray, np.ndarray, float]] = []
    for scan in root.iter():
        if _strip_namespace(scan.tag) != "scan":
            continue
        scan_ms_level = int(scan.attrib.get("msLevel", "1"))
        if ms_level is not None and scan_ms_level != ms_level:
            continue
        peaks = next(
            child for child in scan if _strip_namespace(child.tag) == "peaks"
        )
        mz, intensity = _decode_mzxml_peaks(peaks)
        scans.append((mz, intensity, _precursor_from_xml_scan(scan)))
    return scans


def _scan_precursor_mz(scan: dict[str, Any]) -> float:
    precursor = scan.get("precursorMz", [])
    if isinstance(precursor, list) and precursor:
        first = precursor[0]
        if isinstance(first, dict):
            return float(first.get("precursorMz", first.get("precursor m/z", 0.0)))
        return float(first)
    return 0.0


def _read_mzxml_scan_arrays_pyteomics(
    path: Path,
    *,
    ms_level: int | None,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    from pyteomics import mzxml

    scans: list[tuple[np.ndarray, np.ndarray, float]] = []
    for scan in mzxml.read(str(path)):
        scan_ms_level = int(scan.get("msLevel", scan.get("ms level", 1)))
        if ms_level is not None and scan_ms_level != ms_level:
            continue
        scans.append(
            (
                np.asarray(scan["m/z array"], dtype=np.float32),
                np.asarray(scan["intensity array"], dtype=np.float32),
                _scan_precursor_mz(scan),
            )
        )
    return scans


def _read_mzxml_scan_arrays(
    path: Path,
    *,
    ms_level: int | None,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    if USE_PYTEOMICS_MZXML:
        import importlib.util

        if importlib.util.find_spec("pyteomics") is not None:
            return _read_mzxml_scan_arrays_pyteomics(path, ms_level=ms_level)
    return _read_mzxml_scan_arrays_xml(path, ms_level=ms_level)


def _spectra_from_scan_arrays(
    scan_arrays: list[tuple[np.ndarray, np.ndarray, float]],
) -> tuple[np.ndarray, np.ndarray]:
    spectra = np.zeros((len(scan_arrays), 2, NUM_PEAKS_INPUT), dtype=np.float32)
    precursor_mz = np.zeros(len(scan_arrays), dtype=np.float32)
    for i, (mz, intensity, precursor) in enumerate(scan_arrays):
        mz = np.asarray(mz, dtype=np.float32)
        intensity = np.asarray(intensity, dtype=np.float32)
        finite = np.isfinite(mz) & np.isfinite(intensity) & (intensity > 0)
        mz = mz[finite]
        intensity = intensity[finite]
        if mz.size > NUM_PEAKS_INPUT:
            idx = np.argpartition(intensity, -NUM_PEAKS_INPUT)[-NUM_PEAKS_INPUT:]
            idx = idx[np.argsort(intensity[idx])[::-1]]
            mz = mz[idx]
            intensity = intensity[idx]
        n = min(mz.size, NUM_PEAKS_INPUT)
        spectra[i, 0, :n] = mz[:n]
        spectra[i, 1, :n] = intensity[:n]
        precursor_mz[i] = np.float32(precursor)
    return spectra, precursor_mz


def read_mzxml_sample_spectra(
    path: Path,
    *,
    ms_level: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    return _spectra_from_scan_arrays(
        _read_mzxml_scan_arrays(path, ms_level=ms_level)
    )


class SepticShockMzXMLDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        *,
        root: Path,
        ms_level: int | None,
    ) -> None:
        self.records = records
        self.root = root
        self.ms_level = ms_level

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        spectra, precursor_mz = read_mzxml_sample_spectra(
            self.root / record["mzxml_path"],
            ms_level=self.ms_level,
        )
        return {
            "spectra": torch.from_numpy(spectra),
            "precursor_mz_raw": torch.from_numpy(precursor_mz),
            "label": torch.tensor(float(record["label"]), dtype=torch.float32),
            "sample_index": torch.tensor(int(record["sample_index"]), dtype=torch.long),
            "sample_id": str(record["sample_id"]),
            "raw_file_name": str(record["raw_file_name"]),
        }


class SepticShockPeaklistDataset(Dataset):
    def __init__(
        self,
        data: SepticShockData,
        split: str,
    ) -> None:
        self.records = [
            record for record in data.metadata["samples"] if record["split"] == split
        ]
        arrays = []
        for shard_name in data.metadata[f"{split}_shards"]:
            shard_dir = data.root / split / shard_name
            arrays.append(
                {
                    "spectra": np.load(shard_dir / "spectra.npy", mmap_mode="r"),
                    "precursor_mz_raw": np.load(
                        shard_dir / "precursor_mz_raw.npy",
                        mmap_mode="r",
                    ),
                    "sample_index": np.load(
                        shard_dir / "sample_index.npy",
                        mmap_mode="r",
                    ),
                }
            )
        self._arrays = arrays
        self._positions = [
            [
                (entry_idx, np.flatnonzero(arrays[entry_idx]["sample_index"] == int(record["sample_index"])))
                for entry_idx in range(len(arrays))
            ]
            for record in self.records
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        spectra = np.concatenate(
            [
                self._arrays[entry_idx]["spectra"][positions]
                for entry_idx, positions in self._positions[index]
                if positions.size
            ],
            axis=0,
        )
        precursor_mz = np.concatenate(
            [
                self._arrays[entry_idx]["precursor_mz_raw"][positions]
                for entry_idx, positions in self._positions[index]
                if positions.size
            ],
            axis=0,
        )
        return {
            "spectra": torch.from_numpy(spectra.astype(np.float32, copy=True)),
            "precursor_mz_raw": torch.from_numpy(
                precursor_mz.astype(np.float32, copy=True)
            ),
            "label": torch.tensor(float(record["label"]), dtype=torch.float32),
            "sample_index": torch.tensor(int(record["sample_index"]), dtype=torch.long),
            "sample_id": str(record["sample_id"]),
            "raw_file_name": str(record["raw_file_name"]),
        }


class SepticShockCollator:
    def __init__(
        self,
        *,
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
    ) -> None:
        self.num_peaks = num_peaks
        self.max_precursor_mz = max_precursor_mz
        self.min_peak_intensity = min_peak_intensity
        self.peak_drop_min_intensity = peak_drop_min_intensity
        self.peak_ordering = peak_ordering
        self.precursor_peak_exclusion_window_da = precursor_peak_exclusion_window_da
        self.peak_filtering = peak_filtering
        self.grouped_peak_shoulder_da = grouped_peak_shoulder_da
        self.grouped_peak_isotope_charges = grouped_peak_isotope_charges
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
        )

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        spectrum_counts = torch.tensor(
            [sample["spectra"].shape[0] for sample in samples],
            dtype=torch.long,
        )
        peak_samples = [
            {
                "spectra": sample["spectra"][scan_idx],
                "precursor_mz_raw": sample["precursor_mz_raw"][scan_idx],
            }
            for sample in samples
            for scan_idx in range(int(sample["spectra"].shape[0]))
        ]
        batch = self.peak_collator(peak_samples)
        sample_positions = [
            torch.full((int(count),), i, dtype=torch.long)
            for i, count in enumerate(spectrum_counts.tolist())
        ]
        batch["spectrum_sample_index"] = torch.cat(sample_positions, dim=0)
        batch["spectrum_count"] = spectrum_counts
        batch["sample_index"] = torch.stack([sample["sample_index"] for sample in samples])
        batch["label"] = torch.stack([sample["label"] for sample in samples]).to(
            torch.float32
        )
        batch["sample_id"] = [sample["sample_id"] for sample in samples]
        batch["raw_file_name"] = [sample["raw_file_name"] for sample in samples]
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


def build_septic_shock_data(
    *,
    cache_dir: Path,
    batch_size: int,
    num_peaks: int,
    max_precursor_mz: float = DEFAULT_MAX_PRECURSOR_MZ,
    min_peak_intensity: float = DEFAULT_MIN_PEAK_INTENSITY,
    peak_drop_min_intensity: float = DEFAULT_MIN_PEAK_INTENSITY,
    peak_filtering: str = DEFAULT_PEAK_FILTERING,
    grouped_peak_shoulder_da: float = DEFAULT_GROUPED_PEAK_SHOULDER_DA,
    grouped_peak_isotope_charges: tuple[int, ...] = (
        DEFAULT_GROUPED_PEAK_ISOTOPE_CHARGES
    ),
    peak_ordering: str = "mz",
    precursor_peak_exclusion_window_da: float = (
        DEFAULT_PRECURSOR_PEAK_EXCLUSION_WINDOW_DA
    ),
    ms_level: int | None = None,
    prepare: bool = False,
    download_raw: bool = False,
    split_seed: int = SEPTIC_SHOCK_SPLIT_SEED,
) -> SepticShockData:
    cache_dir = cache_dir.expanduser().resolve()
    metadata = (
        prepare_septic_shock_dataset(
            cache_dir,
            download_raw=download_raw,
            split_seed=split_seed,
        )
        if prepare
        else load_septic_shock_metadata(cache_dir)
    )
    return SepticShockData(
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
        ms_level=ms_level,
    )


def build_septic_shock_loader(
    data: SepticShockData,
    split: str,
    *,
    shuffle: bool,
    seed: int,
    max_samples: int | None = None,
    drop_last: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    if data.metadata.get("artifact_format") == SEPTIC_SHOCK_ARTIFACT_FORMAT:
        dataset: Dataset = SepticShockPeaklistDataset(data, split)
    else:
        records = [
            record for record in data.metadata["samples"] if record["split"] == split
        ]
        dataset = SepticShockMzXMLDataset(
            records,
            root=data.root,
            ms_level=data.ms_level,
        )
    dataset, shuffle = _subset_for_max_samples(
        dataset,
        max_samples=max_samples,
        shuffle=shuffle,
        seed=seed,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": data.batch_size,
        "shuffle": shuffle,
        "drop_last": drop_last,
        "num_workers": num_workers,
        "collate_fn": SepticShockCollator(
            num_peaks=data.num_peaks,
            max_precursor_mz=data.max_precursor_mz,
            min_peak_intensity=data.min_peak_intensity,
            peak_drop_min_intensity=data.peak_drop_min_intensity,
            peak_ordering=data.peak_ordering,
            precursor_peak_exclusion_window_da=data.precursor_peak_exclusion_window_da,
            peak_filtering=data.peak_filtering,
            grouped_peak_shoulder_da=data.grouped_peak_shoulder_da,
            grouped_peak_isotope_charges=data.grouped_peak_isotope_charges,
        ),
        "generator": generator,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    return DataLoader(**loader_kwargs)
