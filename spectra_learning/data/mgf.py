"""Build a NIST hr_msms HDF5 matching the DreaMS Atlas schema consumed by the
offline NIST full probe artifact builder.

The probe expects, at minimum, these datasets:

``spectrum``       : (N, 2, 128) float — row 0 = m/z, row 1 = intensity (raw;
                     data loaders normalize each spectrum by its maximum)
``precursor_mz``   : (N,)        float
``smiles``         : (N,)        variable-length strings
``adduct``         : (N,)        variable-length strings
``DreaMS_embedding`` (optional): (N, D) float

We parse ``hr_msms_nist.mgf`` (MS2, with SMILES), pack peaks to the top-128 by
intensity, validate SMILES via RDKit, and write a GZIP-compressed HDF5.
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from rdkit import Chem
from tqdm import tqdm

from spectra_learning.data.spectra import (
    DEFAULT_MAX_PRECURSOR_MZ,
    DEFAULT_MIN_PRECURSOR_MZ,
    NUM_PEAKS_INPUT,
)

log = logging.getLogger(__name__)


def file_source_manifest(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _to_float(value: str | None) -> float:
    if value is None:
        return float("nan")
    value = value.strip()
    if not value or value.lower() == "nan":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def iter_mgf(path: Path) -> Iterator[dict[str, Any]]:
    """Stream an MGF file, yielding one dict per ``BEGIN IONS`` block."""
    current: dict[str, Any] | None = None
    peak_mz: list[float] = []
    peak_int: list[float] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line == "BEGIN IONS":
                current = {}
                peak_mz = []
                peak_int = []
                continue
            if line == "END IONS":
                if current is not None:
                    current["peak_mz"] = np.asarray(peak_mz, dtype=np.float64)
                    current["peak_intensity"] = np.asarray(peak_int, dtype=np.float64)
                    yield current
                current = None
                continue
            if current is None:
                continue
            if "=" in line and not line[0].isdigit():
                key, _, value = line.partition("=")
                current[key.strip().lower()] = value.strip()
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    mz_val = float(parts[0])
                    int_val = float(parts[1])
                except ValueError:
                    continue
                peak_mz.append(mz_val)
                peak_int.append(int_val)


def _extract_record(
    record: dict[str, Any],
    *,
    num_peaks_input: int,
    max_precursor_mz: float,
    min_precursor_mz: float,
) -> tuple[np.ndarray, float, str, str] | None:
    """Return (peaks[2, num_peaks_input], precursor_mz, smiles, adduct) or None."""
    smiles = (record.get("smiles") or "").strip()
    if not smiles:
        return None
    if Chem.MolFromSmiles(smiles) is None:
        return None

    pepmass = record.get("pepmass") or ""
    precursor = _to_float(pepmass.split()[0]) if pepmass else float("nan")
    if not math.isfinite(precursor):
        return None
    if precursor < min_precursor_mz or precursor > max_precursor_mz:
        return None

    peak_mz = record["peak_mz"]
    peak_int = record["peak_intensity"]
    if peak_mz.size == 0 or not np.any(peak_int > 0):
        return None

    if peak_mz.size > num_peaks_input:
        top = np.argpartition(-peak_int, num_peaks_input)[:num_peaks_input]
        peak_mz = peak_mz[top]
        peak_int = peak_int[top]

    mz_buf = np.zeros(num_peaks_input, dtype=np.float32)
    int_buf = np.zeros(num_peaks_input, dtype=np.float32)
    mz_buf[: peak_mz.size] = peak_mz
    int_buf[: peak_int.size] = peak_int

    adduct = (record.get("precursortype") or "").strip() or "unknown"
    return np.stack([mz_buf, int_buf], axis=0), precursor, smiles, adduct


def build_nist_probe_hdf5(
    mgf_path: Path,
    output_path: Path,
    *,
    num_peaks_input: int = NUM_PEAKS_INPUT,
    max_precursor_mz: float = DEFAULT_MAX_PRECURSOR_MZ,
    min_precursor_mz: float = DEFAULT_MIN_PRECURSOR_MZ,
    compression: str | None = "gzip",
    compression_opts: int | None = 4,
) -> dict[str, Any]:
    """Parse ``mgf_path`` and write an HDF5 in the DreaMS-Atlas NIST20 schema."""
    source = file_source_manifest(mgf_path)
    spectra_parts: list[np.ndarray] = []
    precursor_parts: list[float] = []
    smiles_parts: list[str] = []
    adduct_parts: list[str] = []
    n_raw: int = 0

    for record in tqdm(iter_mgf(mgf_path), desc="Parsing MGF", unit="spec"):
        n_raw += 1
        out = _extract_record(
            record,
            num_peaks_input=num_peaks_input,
            max_precursor_mz=max_precursor_mz,
            min_precursor_mz=min_precursor_mz,
        )
        if out is None:
            continue
        peaks, precursor, smiles, adduct = out
        spectra_parts.append(peaks)
        precursor_parts.append(precursor)
        smiles_parts.append(smiles)
        adduct_parts.append(adduct)

    n_kept = len(spectra_parts)
    if n_kept == 0:
        raise ValueError(f"No usable spectra in {mgf_path}")
    log.info(
        "Parsed %d MGF records, kept %d (%.1f%%)",
        n_raw,
        n_kept,
        100.0 * n_kept / max(1, n_raw),
    )

    spectrum = np.stack(spectra_parts, axis=0)
    precursor_mz = np.asarray(precursor_parts, dtype=np.float32)
    smiles_arr = np.asarray(smiles_parts, dtype=object)
    adduct_arr = np.asarray(adduct_parts, dtype=object)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    str_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(str(output_path), "w") as fh:
        kw: dict[str, Any] = {}
        if compression:
            kw["compression"] = compression
            if compression_opts is not None:
                kw["compression_opts"] = compression_opts
        fh.create_dataset("spectrum", data=spectrum, **kw)
        fh.create_dataset("precursor_mz", data=precursor_mz, **kw)
        fh.create_dataset("smiles", data=smiles_arr, dtype=str_dtype, **kw)
        fh.create_dataset("adduct", data=adduct_arr, dtype=str_dtype, **kw)
        fh.attrs["source_path"] = source["path"]
        fh.attrs["source_bytes"] = source["bytes"]
        fh.attrs["source_sha256"] = source["sha256"]
        fh.attrs["num_peaks_input"] = num_peaks_input
        fh.attrs["min_precursor_mz"] = min_precursor_mz
        fh.attrs["max_precursor_mz"] = max_precursor_mz

    log.info(
        "Wrote %s (%d spectra, %.1f MB)",
        output_path,
        n_kept,
        output_path.stat().st_size / 1e6,
    )
    return {
        "path": str(output_path),
        "num_spectra": n_kept,
        "num_raw": n_raw,
        "size_bytes": output_path.stat().st_size,
        "source": source,
        "num_peaks_input": num_peaks_input,
        "min_precursor_mz": min_precursor_mz,
        "max_precursor_mz": max_precursor_mz,
    }
