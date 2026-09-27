"""Verified physical ion inputs for the trainable precise-mass readout."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.csv as csv

from inspect_global_glacier_mass_metadata import mass_scales, sha256

PROTON_MASS = 1.00782503207 - 0.000548579909065


def load_ion_training_inputs(
    input_root: Path, peak_input: dict, mask: np.ndarray, fractions: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict]:
    """Load all source rows without reading tokens, molecular labels, or ranks.

    Physical m/z uses the original prepared peak order and the authoritative
    peak normalization scale. Legacy fractions are preserved for the unsupported
    adduct fallback; they are never reconstructed from the physical-Da values.
    Trainable JEPA inputs must carry the exact file identities from load_spectra.
    """
    embedding_path = input_root / "embedding_metadata.json"
    embedding = json.loads(embedding_path.read_text())
    trainable_jepa = peak_input["kind"] == "trainable_jepa"
    if peak_input["kind"] not in ("raw", "trainable_jepa") or embedding != peak_input["embedding_metadata"]:
        raise ValueError("Ion inputs must use the same raw-token embedding metadata")
    if (mask.dtype != np.bool_ or mask.ndim != 2 or mask.shape[0] != embedding["samples"]
            or tuple(embedding["embedding_shape"][:2]) != mask.shape
            or fractions.shape != mask.shape or not np.isfinite(fractions).all()):
        raise ValueError("Peak masks and fractions must align with every frozen embedding row")
    token_mask_path = input_root / "data/token_valid_mask.npy"
    token_mask = np.load(token_mask_path, allow_pickle=False)
    if (token_mask.dtype != np.bool_ or token_mask.shape != (len(mask), mask.shape[1] + 1)
            or not token_mask[:, -1].all()):
        raise ValueError("Expected original physical peak masks with one valid trailing CLS slot")
    np.testing.assert_array_equal(token_mask[:, :-1], mask, err_msg="Frozen physical peak slots changed")
    peak_scale, precursor_scale, units = mass_scales(embedding)
    data_root = Path(embedding["data_root"])
    if trainable_jepa:
        required = {"data_root", "metadata_path", "metadata_sha256",
                    "prepared_spectra_sha256", "preparation_metadata_sha256"}
        if not required.issubset(peak_input):
            raise ValueError("Trainable JEPA physical input provenance is incomplete")
        if Path(peak_input["data_root"]).resolve() != data_root.resolve():
            raise ValueError("Trainable JEPA and ion inputs must use the same prepared data root")
        if Path(peak_input["metadata_path"]).resolve() != Path(embedding["config_metadata_path"]).resolve():
            raise ValueError("Trainable JEPA and ion inputs must use the same encoder metadata path")
        if peak_input["metadata_sha256"] != units["config_metadata_sha256"]:
            raise ValueError("Trainable JEPA encoder metadata changed before loading ion inputs")
    preparation_path = data_root / "preparation_metadata.json"
    preparation = json.loads(preparation_path.read_text())
    if preparation["tsv_sha256"] != embedding["tsv_sha256"]:
        raise ValueError("Prepared spectra and frozen tokens have different source TSVs")
    prepared_path = data_root / "prepared_massspecgym.npz"
    with np.load(prepared_path, allow_pickle=False) as prepared:
        np.testing.assert_array_equal(prepared["peak_valid_mask"], mask,
                                      err_msg="Prepared physical peak slots changed")
        normalized_mz = np.asarray(prepared["peak_mz"], np.float32)
        normalized_precursor = np.asarray(prepared["precursor_mz"], np.float32)
    if (normalized_mz.shape != mask.shape or normalized_precursor.shape != (len(mask),)
            or not np.isfinite(normalized_mz).all() or not np.isfinite(normalized_precursor).all()):
        raise ValueError("Prepared physical mass arrays must be finite and aligned with every row")

    tsv_path = input_root / "msg15/MassSpecGym1.5.tsv"
    tsv_digest = sha256(tsv_path)
    if tsv_digest != embedding["tsv_sha256"]:
        raise ValueError("Raw ion TSV differs from frozen embedding provenance")
    table = csv.read_csv(
        tsv_path, parse_options=csv.ParseOptions(delimiter="\t"),
        convert_options=csv.ConvertOptions(include_columns=["precursor_mz", "adduct"]),
    )
    if table.num_rows != len(mask):
        raise ValueError("Raw ion TSV row count differs from the frozen token source")
    raw_precursor = np.asarray(table["precursor_mz"], np.float32)
    if not np.isfinite(raw_precursor).all() or np.any(raw_precursor <= 0):
        raise ValueError("Raw precursor m/z must be finite and positive")
    expected = np.clip(raw_precursor, 0, precursor_scale) / np.float32(precursor_scale)
    np.testing.assert_allclose(normalized_precursor, expected, rtol=0, atol=2e-7,
                               err_msg="Prepared precursor normalization differs from the source TSV")
    unclipped = raw_precursor <= precursor_scale
    reconstructed = normalized_precursor * np.float32(precursor_scale)
    np.testing.assert_allclose(reconstructed[unclipped], raw_precursor[unclipped], rtol=2e-7, atol=1e-4)
    adduct = np.asarray(["" if value is None else value for value in table["adduct"].to_pylist()], dtype=str)
    eligible = np.isin(adduct, ["[M+H]+", "[M+Na]+"])
    adduct_mass = np.where(adduct == "[M+Na]+", 22.989769282 - .000548579909065,
                           PROTON_MASS).astype(np.float32)
    query = {
        "fraction": np.asarray(fractions, np.float32),
        "mz": normalized_mz * np.float32(peak_scale),
        "eligible": eligible,
        "adduct_mass": adduct_mass,
    }
    units.update(normalization_verified_against_tsv=True,
                 clipped_precursor_rows=int((~unclipped).sum()),
                 maximum_unclipped_reconstruction_error_da=float(np.max(
                     np.abs(reconstructed[unclipped] - raw_precursor[unclipped]), initial=0)))
    source_paths = {"embedding_metadata": embedding_path, "preparation_metadata": preparation_path,
                    "prepared_spectra": prepared_path, "token_mask": token_mask_path, "source_tsv": tsv_path,
                    "loader": Path(__file__),
                    "mass_scale_helper": Path(__file__).with_name("inspect_global_glacier_mass_metadata.py"),
                    "ion_mass_constants": Path(__file__)}
    if trainable_jepa:
        source_paths["encoder_metadata"] = Path(peak_input["metadata_path"])
        source_paths["trainable_peak_loader"] = Path(__file__).with_name("trainable_jepa_peak.py")
    provenance = {
        "schema": "ion_logmixture_training_inputs_v1", "rows": len(mask), "physical_peaks": mask.shape[1],
        "source_paths": {key: str(path) for key, path in source_paths.items()},
        "source_sha256": {key: sha256(path) for key, path in source_paths.items()},
        "query_array_sha256": {key: hashlib.sha256(memoryview(np.ascontiguousarray(value))).hexdigest()
                               for key, value in query.items()},
        "units": units, "physical_masks_exactly_verified": True,
        "physical_peak_order": "Unchanged column order of the original embedding's prepared_massspecgym.npz",
        "eligible_exact_adducts": ["[M+H]+", "[M+Na]+"], "eligible_rows": int(eligible.sum()),
        "unsupported_rows": int((~eligible).sum()),
        "unsupported_adduct_mass": "Finite proton mass; ignored by the original-fraction fallback",
        "legacy_fraction_preserved": True, "tokens_read": False, "molecular_labels_read": False,
    }
    if trainable_jepa:
        for name in ("prepared_spectra", "preparation_metadata"):
            if peak_input[f"{name}_sha256"] != provenance["source_sha256"][name]:
                raise ValueError(f"Trainable JEPA {name} changed before loading ion inputs")
        provenance["trainable_jepa_input_verified"] = True
    return query, provenance
