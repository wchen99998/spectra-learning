"""Audit observed ion metadata for exactly the rows of a frozen retrieval cache.

No inference or filtering is performed. Prepared charge flags are reported
separately from charge explicitly supported by the source TSV's adduct strings.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pyarrow.csv as csv


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def distribution(values: np.ndarray) -> dict[str, int]:
    labels, counts = np.unique(values, return_counts=True)
    return {str(label): int(count) for label, count in zip(labels, counts, strict=True)}


def parse_adduct_charge(adduct: str) -> tuple[int, bool]:
    """Read an explicit ion charge suffix; do not infer absent charge."""
    match = re.fullmatch(r"\[.+\]\s*(\d*)([+-])", adduct.strip())
    if match is None:
        return 0, False
    magnitude = int(match[1] or "1")
    if not 1 <= magnitude <= np.iinfo(np.int16).max:
        raise ValueError(f"Invalid explicit adduct charge: {adduct}")
    return magnitude * (1 if match[2] == "+" else -1), True


def mass_scales(embedding: dict) -> tuple[float, float, dict]:
    """Read the two distinct normalization scales from their authoritative sources."""
    source = Path(__file__).resolve().parents[1] / "spectra_learning/data/spectra.py"
    constants = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.Assign) and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "PEAK_MZ_MAX"
    }
    peak_scale = float(constants["PEAK_MZ_MAX"])
    config_path = Path(embedding["config_metadata_path"])
    config = json.loads(config_path.read_text())["task_contract"]["config"]
    precursor_scale = float(config["max_precursor_mz"])
    if not np.isfinite([peak_scale, precursor_scale]).all() or min(peak_scale, precursor_scale) <= 0:
        raise ValueError("Invalid source m/z normalization scales")
    return peak_scale, precursor_scale, {
        "peak_mz_scale_da": peak_scale, "precursor_mz_scale_da": precursor_scale,
        "peak_scale_source": f"{source}:PEAK_MZ_MAX", "preprocessing_source_sha256": sha256(source),
        "precursor_scale_source": f"{config_path}:task_contract.config.max_precursor_mz",
        "config_metadata_sha256": sha256(config_path),
        "prepared_peak_mz_units": "dimensionless: original peak m/z divided by PEAK_MZ_MAX",
        "prepared_precursor_mz_units": "dimensionless: precursor m/z clipped to [0,max_precursor_mz] then divided by max_precursor_mz",
        "peak_mz_da_source": "Prepared peak_mz multiplied by authoritative PEAK_MZ_MAX",
        "precursor_mz_da_source": "Original verified TSV precursor_mz; normalization checked against prepared array",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cache_metadata_path = args.cache / "metadata.json"
    cache = json.loads(cache_metadata_path.read_text())
    if cache.get("schema") != "frozen_global_glacier_retrieval_v1" or not cache.get("complete"):
        raise ValueError("A completed frozen Global + GLACIER retrieval cache is required")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"Metadata output must be empty: {args.out}")
    rows = np.load(args.cache / "rows.npy")
    if hashlib.sha256(rows.tobytes()).hexdigest() != cache["rows_sha256"]:
        raise ValueError("Retrieval row identity differs from cache provenance")
    embedding = cache["peak_input"]["embedding_metadata"]
    peak_scale, precursor_scale, units = mass_scales(embedding)
    data_root = Path(embedding["data_root"])
    preparation_path = data_root / "preparation_metadata.json"
    preparation = json.loads(preparation_path.read_text())
    if preparation["tsv_sha256"] != embedding["tsv_sha256"]:
        raise ValueError("Prepared ion metadata differs from the approved frozen-token TSV")
    prepared_path = data_root / "prepared_massspecgym.npz"
    arrays = {"rows": rows}
    with np.load(prepared_path) as prepared:
        np.testing.assert_array_equal(
            prepared["peak_valid_mask"][rows], np.load(args.cache / "peak_mask.npy"),
        )
        for name in ("charge", "charge_present", "polarity_id", "precursor_mz"):
            values = prepared[name][rows]
            if values.shape != rows.shape or not np.isfinite(values).all():
                raise ValueError(f"Invalid selected prepared {name}")
            arrays[name] = values
        peak_mz = prepared["peak_mz"][rows].astype(np.float32)
        if peak_mz.shape != np.load(args.cache / "peak_mask.npy").shape or not np.isfinite(peak_mz).all():
            raise ValueError("Invalid selected prepared peak_mz")
        peak_cache = args.cache / "peak_mz.npy"
        if peak_cache.exists():
            np.testing.assert_array_equal(peak_mz, np.load(peak_cache))
        arrays["peak_mz_da"] = peak_mz * np.float32(peak_scale)
        precursor_cache = args.cache / "precursor_mz.npy"
        if precursor_cache.exists():
            np.testing.assert_array_equal(arrays["precursor_mz"], np.load(precursor_cache))

    input_root = Path(cache["peak_input"]["path"]).parent
    tsv_path = input_root / "msg15/MassSpecGym1.5.tsv"
    adduct_info: dict = {"available": False, "path": str(tsv_path)}
    if not tsv_path.is_file():
        raise FileNotFoundError(f"Source TSV required to verify physical m/z recovery: {tsv_path}")
    else:
        digest = sha256(tsv_path)
        if digest != embedding["tsv_sha256"]:
            raise ValueError("Raw adduct TSV differs from frozen embedding provenance")
        adduct_info["tsv_sha256"] = digest
        with csv.open_csv(tsv_path, parse_options=csv.ParseOptions(delimiter="\t")) as reader:
            columns = reader.schema.names
        if "precursor_mz" not in columns:
            raise ValueError("Source TSV requires precursor_mz to verify m/z normalization")
        table = csv.read_csv(
            tsv_path, parse_options=csv.ParseOptions(delimiter="\t"),
            convert_options=csv.ConvertOptions(include_columns=[
                "precursor_mz", *(["adduct"] if "adduct" in columns else []),
            ]),
        )
        if table.num_rows != embedding["samples"]:
            raise ValueError("Raw ion TSV row count differs from the frozen token source")
        raw_precursor = np.asarray(table["precursor_mz"].take(rows), np.float32)
        if raw_precursor.shape != rows.shape or not np.isfinite(raw_precursor).all() or np.any(raw_precursor <= 0):
            raise ValueError("Raw TSV precursor_mz must be finite and positive")
        expected = np.clip(raw_precursor, 0, precursor_scale) / np.float32(precursor_scale)
        np.testing.assert_allclose(arrays["precursor_mz"], expected, rtol=0, atol=2e-7)
        reconstructed = arrays["precursor_mz"] * np.float32(precursor_scale)
        unclipped = raw_precursor <= precursor_scale
        np.testing.assert_allclose(reconstructed[unclipped], raw_precursor[unclipped], rtol=2e-7, atol=1e-4)
        arrays["precursor_mz_da"] = raw_precursor
        units.update(
            tsv_sha256=digest, normalization_verified_against_tsv=True,
            clipped_precursor_rows=int((~unclipped).sum()),
            maximum_unclipped_reconstruction_error_da=float(np.max(
                np.abs(reconstructed[unclipped] - raw_precursor[unclipped]), initial=0,
            )),
        )
        if "adduct" not in columns:
            adduct_info["reason"] = "Source TSV has no adduct column"
        else:
            values = table["adduct"].take(rows).to_pylist()
            arrays["adduct"] = np.asarray(["" if value is None else value for value in values], dtype=str)
            arrays["adduct_present"] = np.asarray([bool(value.strip()) for value in arrays["adduct"]])
            parsed = [parse_adduct_charge(value) for value in arrays["adduct"]]
            arrays["adduct_charge"] = np.asarray([value[0] for value in parsed], np.int16)
            arrays["adduct_charge_present"] = np.asarray([value[1] for value in parsed], bool)
            arrays["positive_single_charge_known"] = arrays["adduct_charge_present"] & (arrays["adduct_charge"] == 1)
            arrays["protonated_monomer_known"] = np.asarray([
                re.fullmatch(r"\[m\+h\](?:1)?\+", value.lower().replace(" ", "")) is not None
                for value in arrays["adduct"]
            ], bool)
            adduct_info.update(available=True, distribution=distribution(arrays["adduct"]),
                               missing_count=int((~arrays["adduct_present"]).sum()),
                               explicit_charge_count=int(arrays["adduct_charge_present"].sum()))

    flags_positive_single = ((arrays["charge_present"] > 0) & (arrays["charge"] == 1)
                             & (arrays["polarity_id"] == 1))
    summary = {
        "schema": "global_glacier_mass_metadata_audit_v2", "complete": True,
        "cache": str(args.cache), "cache_metadata_sha256": sha256(cache_metadata_path),
        "checkpoint_sha256": cache["checkpoint_sha256"], "split": cache["split"],
        "rows_sha256": cache["rows_sha256"], "n": len(rows), "excluded_rows": 0,
        "prepared": str(prepared_path), "preparation_metadata_sha256": sha256(preparation_path),
        "prepared_distributions": {name: distribution(arrays[name]) for name in (
            "charge", "charge_present", "polarity_id",
        )},
        "polarity_codes": {"0": "unknown", "1": "positive", "2": "negative"},
        "prepared_positive_single_charge_count": int(flags_positive_single.sum()),
        "prepared_positive_single_charge_fraction": float(flags_positive_single.mean()),
        "prepared_charge_provenance": (
            "scripts/modal_massspecgym_maccs_probe.py:_massspecgym_acquisition_metadata "
            "assigns charge=1 and charge_present=1 for every row; these flags alone do not establish observed charge"
        ),
        "raw_adduct": adduct_info,
        "units": units,
        "positive_single_charge_known_count": None,
        "positive_single_charge_known_fraction": None,
        "protonated_monomer_known_count": None,
        "protonated_monomer_known_fraction": None,
        "known_charge_definition": "Explicit raw adduct suffix ]+ or ]1+; absent/unparsed charge stays unknown",
        "precursor_mz_min": float(arrays["precursor_mz"].min()),
        "precursor_mz_max": float(arrays["precursor_mz"].max()),
        "precursor_mz_da_min": float(arrays["precursor_mz_da"].min()),
        "precursor_mz_da_max": float(arrays["precursor_mz_da"].max()),
        "inference_performed": False,
    }
    if adduct_info["available"]:
        for name in ("positive_single_charge_known", "protonated_monomer_known"):
            summary[f"{name}_count"] = int(arrays[name].sum())
            summary[f"{name}_fraction"] = float(arrays[name].mean())
        known = arrays["adduct_charge_present"]
        prepared_sign = np.where(arrays["polarity_id"] == 1, 1, np.where(arrays["polarity_id"] == 2, -1, 0))
        summary["explicit_vs_prepared_charge_disagreement_count"] = int(np.count_nonzero(
            known & ((np.abs(arrays["adduct_charge"]) != arrays["charge"])
                     | (np.sign(arrays["adduct_charge"]) != prepared_sign)),
        ))
    args.out.mkdir(parents=True, exist_ok=True)
    output_path = args.out / "metadata.npz"
    np.savez(output_path, **arrays)
    for name in ("peak_mz_da", "precursor_mz_da"):
        path = args.out / f"{name}.npy"
        np.save(path, arrays[name])
        units[f"{name}_sha256"] = sha256(path)
    summary["metadata_sha256"] = sha256(output_path)
    summary["script_sha256"] = sha256(Path(__file__))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
