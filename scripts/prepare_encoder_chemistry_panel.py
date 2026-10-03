"""Pin molecule-level chemistry and acquisition controls for encoder interventions.

Labels describe molecules, not observed peak fragments. Matched molecules are
observational controls, never molecular counterfactuals. This script does not fit
or tune a model and writes test examples only to the declared evaluation panel.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.csv as csv
from rdkit import Chem, rdBase
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from spectra_learning.data.massspec_targets import FG_SMARTS
from spectra_learning.data.spectra import PEAK_MZ_MAX

ELEMENTS = ("N", "O", "F", "P", "S", "Cl", "Br")
FOLDS = ("train", "val", "test")
LOSS_COMPOSITIONS = {"H2O": {"H": 2, "O": 1}, "NH3": {"N": 1, "H": 3},
                     "CO": {"C": 1, "O": 1}, "CO2": {"C": 1, "O": 2},
                     "HCN": {"H": 1, "C": 1, "N": 1}, "H2S": {"H": 2, "S": 1}}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verify_identities(smiles: np.ndarray, source: np.ndarray,
                      fold: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    """Exclude whole connectivity groups with identity conflicts or fold leakage."""
    cache = {}
    with rdBase.BlockLogs():
        for smi in np.unique(smiles):
            mol = Chem.MolFromSmiles(str(smi))
            cache[smi] = ("", "") if mol is None else (
                Chem.MolToInchiKey(mol), Chem.MolToSmiles(mol, isomericSmiles=True))
    full = np.asarray([cache[s][0] for s in smiles])
    canonical = np.asarray([cache[s][1] for s in smiles])
    connectivity = np.asarray([key.split("-")[0] for key in full])
    source_connectivity = np.asarray([key.split("-")[0] for key in source])
    invalid = full == ""
    mismatch = connectivity != source_connectivity
    fold_keys = {f: set(connectivity[(fold == f) & ~invalid]) for f in FOLDS}
    overlap = {f"{a}/{b}": sorted(fold_keys[a] & fold_keys[b])
               for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    leaked = set().union(*(set(v) for v in overlap.values()))
    # A mislabeled source group must not contribute another apparently good row.
    bad_source = set(source_connectivity[invalid | mismatch])
    bad_computed = set(connectivity[mismatch & ~invalid]) | leaked
    eligible = (~invalid & ~np.isin(source_connectivity, sorted(bad_source))
                & ~np.isin(connectivity, sorted(bad_computed)))
    arrays = {"computed_inchikey": full, "canonical_smiles": canonical,
              "molecule": connectivity, "identity_eligible": eligible}
    audit = {"unparseable_or_missing_inchikey_rows": int(invalid.sum()),
             "source_connectivity_mismatch_rows": int(mismatch.sum()),
             "crossfold_computed_connectivities": overlap,
             "excluded_source_connectivities": sorted(bad_source),
             "excluded_computed_connectivities": sorted(bad_computed - {""}),
             "excluded_total_rows": int((~eligible).sum()),
             "eligible_rows_by_fold": {f: int(np.sum(eligible & (fold == f))) for f in FOLDS},
             "policy": "Exclude the entire source and computed connectivity groups implicated in any source/SMILES identity mismatch; exclude computed connectivities present in multiple official folds."}
    return arrays, audit


def select_primary_rows(identity: dict[str, np.ndarray], fold: np.ndarray,
                        eligible: np.ndarray, sizes: dict[str, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = []
    for split in FOLDS:
        groups: dict[str, list[int]] = defaultdict(list)
        for row in np.flatnonzero(eligible & (fold == split)):
            groups[str(identity["molecule"][row])].append(int(row))
        keys = sorted(groups)
        size = sizes[split]
        if size < 1 or size > len(keys):
            raise ValueError(f"Requested {size} {split} molecules, available {len(keys)}")
        chosen = rng.choice(len(keys), size=size, replace=False)
        selected.extend(int(rng.choice(groups[keys[i]])) for i in chosen)
    return np.asarray(selected, np.int64)


def repeated_rows(primary: np.ndarray, identity: dict[str, np.ndarray],
                  eligible: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Retain all alternate spectra of the same full identity and canonical graph."""
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in np.flatnonzero(eligible):
        groups[(str(identity["computed_inchikey"][row]),
                str(identity["canonical_smiles"][row]))].append(int(row))
    rows, primary_index = [], []
    for index, source_row in enumerate(primary):
        key = (str(identity["computed_inchikey"][source_row]),
               str(identity["canonical_smiles"][source_row]))
        for row in groups[key]:
            if row != source_row:
                rows.append(row)
                primary_index.append(index)
    return np.asarray(rows, np.int64), np.asarray(primary_index, np.int32)


def chemical_labels(smiles: np.ndarray) -> dict[str, np.ndarray]:
    patterns = {name: Chem.MolFromSmarts(s) for name, s in FG_SMARTS.items()}
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024,
                                                           includeChirality=False)
    values: dict[str, list] = defaultdict(list)
    for smi in smiles:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            raise ValueError(f"Invalid selected molecule: {smi}")
        counts = Counter(atom.GetSymbol() for atom in mol.GetAtoms())
        element_counts = [counts[e] for e in ELEMENTS]
        group_counts = [len(mol.GetSubstructMatches(p)) for p in patterns.values()]
        values["element_counts"].append(element_counts)
        values["functional_group_counts"].append(group_counts)
        values["binary"].append(np.asarray(element_counts + group_counts) > 0)
        values["maccs"].append(np.asarray(MACCSkeys.GenMACCSKeys(mol), np.uint8)[1:])
        values["morgan"].append(generator.GetFingerprintAsNumPy(mol))
        values["formula"].append(rdMolDescriptors.CalcMolFormula(mol))
        values["exact_mass"].append(rdMolDescriptors.CalcExactMolWt(mol))
        values["scaffold"].append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False))
        values["heavy_atom_count"].append(mol.GetNumHeavyAtoms())
    result = {k: np.asarray(v) for k, v in values.items()}
    for name in ("binary", "maccs", "morgan"):
        result[name] = result[name].astype(np.uint8)
    for name in ("element_counts", "functional_group_counts", "heavy_atom_count"):
        result[name] = result[name].astype(np.int16)
    result["binary_names"] = np.asarray([f"elem_{e}" for e in ELEMENTS]
                                         + [f"fg_{name}" for name in FG_SMARTS])
    result["element_names"] = np.asarray(ELEMENTS)
    result["functional_group_names"] = np.asarray(list(FG_SMARTS))
    return result


def matched_controls(data: dict[str, np.ndarray], binary: np.ndarray,
                     *, mass_caliper: float = 5.0, energy_caliper: float = 10.0,
                     count_caliper: int = 5) -> tuple[dict[str, np.ndarray], dict]:
    """Deterministic nearest eligible donor, with explicit unmatched rows (-1).

    Matching uses exact fold, instrument and adduct; raw precursor mass,
    collision-energy presence/value, and actual processed peak count. It does
    not consult representations, predictions, or intervention outcomes.
    """
    if mass_caliper <= 0 or energy_caliper <= 0 or count_caliper < 0:
        raise ValueError("Mass/energy calipers must be positive and count caliper nonnegative")
    n = len(data["fold"])
    count = data["peak_valid_mask"].sum(1)
    energy = data["collision_energy_raw"]
    energy_present = data["collision_energy_present"].astype(bool)
    mass = data["precursor_mz_raw"]
    groups: dict[tuple[str, str, str, bool], list[int]] = defaultdict(list)
    for i in range(n):
        groups[(str(data["fold"][i]), str(data["instrument_type"][i]),
                str(data["adduct"][i]), bool(energy_present[i]))].append(i)
    donor = np.full(n, -1, np.int32)
    opposite = np.full(binary.shape, -1, np.int32)
    eligible_count = np.zeros(n, np.int32)
    opposite_count = np.zeros(binary.shape, np.int32)
    for members in groups.values():
        candidates = np.asarray(members)
        for i in members:
            dm = np.abs(mass[candidates] - mass[i])
            de = np.abs(energy[candidates] - energy[i]) if energy_present[i] else np.zeros(len(candidates))
            dc = np.abs(count[candidates] - count[i])
            keep = ((data["molecule"][candidates] != data["molecule"][i])
                    & (dm <= mass_caliper) & (de <= energy_caliper) & (dc <= count_caliper))
            possible = candidates[keep]
            score = dm[keep] / mass_caliper + de[keep] / energy_caliper + dc[keep] / max(1, count_caliper)
            # Source row breaks equal scores; no dependence on panel array order.
            ordered = possible[np.lexsort((data["rows"][possible], score))]
            eligible_count[i] = len(ordered)
            if len(ordered):
                donor[i] = ordered[0]
                differing = binary[ordered] != binary[i]
                opposite_count[i] = differing.sum(0)
                for label in np.flatnonzero(differing.any(0)):
                    opposite[i, label] = ordered[np.flatnonzero(differing[:, label])[0]]
    arrays = {"donor": donor, "opposite_concept_donor": opposite,
              "eligible_donor_count": eligible_count,
              "eligible_opposite_concept_donor_count": opposite_count}
    audits = {}
    for name, selected in (("nuisance", donor[:, None]), ("opposite_concept", opposite)):
        anchor, column = np.where(selected >= 0)
        chosen = selected[anchor, column]
        gap_mass = np.abs(mass[anchor] - mass[chosen])
        gap_count = np.abs(count[anchor] - count[chosen])
        present = energy_present[anchor]
        gap_energy = np.abs(energy[anchor[present]] - energy[chosen[present]])
        def gap_summary(gap):
            return {"mean": float(gap.mean()), "max": float(gap.max())} if len(gap) else None
        audits[name] = {"matched_edges": len(anchor), "unique_donors": len(np.unique(chosen)),
                        "unmatched_entries": int((selected < 0).sum()),
                        "mass_gap_da": gap_summary(gap_mass),
                        "peak_count_gap": gap_summary(gap_count),
                        "collision_energy_gap_when_present": gap_summary(gap_energy),
                        "per_fold_matched_anchors": {f: int(np.sum((data["fold"] == f)
                            & np.any(selected >= 0, axis=1))) for f in FOLDS}}
    return arrays, {"mass_caliper_da": mass_caliper, "collision_energy_caliper": energy_caliper,
                    "processed_peak_count_caliper": count_caliper,
                    "exact_matches": ["fold", "instrument_type", "adduct", "collision_energy_present"],
                    "tie_break": "normalized L1 nuisance distance, then original source row",
                    "donor_reuse": "allowed; use molecule-clustered inference and inspect balance",
                    "interpretation": "Observational matched controls; changing molecular identity also changes other chemistry. Not a molecular counterfactual or proof of chemical causality.",
                    "audits": audits}


def weak_loss_flags(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    table = Chem.GetPeriodicTable()
    masses = np.asarray([sum(table.GetMostCommonIsotopeMass(e) * n for e, n in counts.items())
                         for counts in LOSS_COMPOSITIONS.values()])
    mz = data["peak_mz"].astype(np.float64) * PEAK_MZ_MAX
    observed = data["precursor_mz_raw"][:, None].astype(np.float64) - mz
    tolerance = np.maximum(0.01, mz * 20e-6)
    eligible = data["adduct"] == "[M+H]+"
    match = (np.abs(observed[:, :, None] - masses) <= tolerance[:, :, None])
    match &= data["peak_valid_mask"][:, :, None] & eligible[:, None, None]
    return {"peak_matches": match, "spectrum_matches": match.any(1),
            "eligible_protonated_spectrum": eligible,
            "loss_names": np.asarray(list(LOSS_COMPOSITIONS)), "loss_masses_da": masses}


def prepare(tsv: Path, config_path: Path, output: Path, *, train_size: int = 4096,
            val_size: int = 1024, test_size: int = 1024, seed: int = 20260930) -> dict:
    from ml_collections import ConfigDict
    from scripts.modal_export_jepa_fps_embeddings import _prepare_massspecgym_inputs

    config = json.loads(config_path.read_text())
    if config["num_peaks"] != 127:
        raise ValueError("This panel pins the original 127-physical-peak encoder input")
    table = csv.read_csv(tsv, parse_options=csv.ParseOptions(delimiter="\t"))
    columns = ("identifier", "inchikey", "smiles", "formula", "instrument_type", "adduct", "fold")
    source = {k: np.asarray(table[k].to_pylist(), dtype=str) for k in columns}
    if not np.isin(source["fold"], FOLDS).all():
        raise ValueError("Unknown source fold")
    source["precursor_mz_raw"] = np.asarray(table["precursor_mz"], np.float64)
    source["collision_energy_raw"] = np.asarray([np.nan if x is None else x
        for x in table["collision_energy"].to_pylist()], np.float64)
    print("Auditing SMILES identities and official fold separation", flush=True)
    identity, identity_audit = verify_identities(source["smiles"], source["inchikey"], source["fold"])
    print("Preprocessing source spectra using native encoder inputs", flush=True)
    native = _prepare_massspecgym_inputs(tsv, ConfigDict(config))
    eligible = identity["identity_eligible"] & (native["peak_valid_mask"].sum(1) >= 5)
    eligible &= np.isfinite(source["precursor_mz_raw"]) & (source["precursor_mz_raw"] > 0)
    sizes = dict(zip(FOLDS, (train_size, val_size, test_size), strict=True))
    primary = select_primary_rows(identity, source["fold"], eligible, sizes, seed)
    repeats, repeat_primary = repeated_rows(primary, identity, eligible)

    def subset(rows):
        arrays = {k: v[rows] for k, v in native.items()}
        arrays.update({k: v[rows] for k, v in source.items()})
        arrays["source_inchikey"] = arrays.pop("inchikey")
        arrays["source_formula"] = arrays.pop("formula")
        arrays.update({k: v[rows] for k, v in identity.items() if k != "identity_eligible"})
        arrays["rows"] = rows
        return arrays

    data, repeat_data = subset(primary), subset(repeats)
    labels = chemical_labels(data["smiles"])
    for k in ("formula", "exact_mass", "scaffold", "heavy_atom_count"):
        data[k] = labels[k]
        repeat_data[k] = labels[k][repeat_primary]
    # Empty Murcko strings denote acyclic molecules; do not call them novel scaffolds.
    train_scaffolds = set(data["scaffold"][data["fold"] == "train"]) - {""}
    data["unseen_nonempty_train_scaffold"] = ((data["scaffold"] != "")
        & ~np.isin(data["scaffold"], sorted(train_scaffolds)))
    repeat_data["primary_index"] = repeat_primary
    repeat_data["primary_source_row"] = primary[repeat_primary]
    controls, matching_audit = matched_controls(data, labels["binary"])
    controls["binary_names"] = labels["binary_names"]
    controls["rows"] = primary
    loss = weak_loss_flags(data)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "inputs.npz", **data)
    np.savez_compressed(output / "labels.npz", **labels, rows=primary, fold=data["fold"], molecule=data["molecule"])
    np.savez_compressed(output / "repeat_inputs.npz", **repeat_data)
    np.savez_compressed(output / "matched_controls.npz", **controls)
    np.savez_compressed(output / "weak_losses.npz", **loss, rows=primary)
    np.savez_compressed(output / "source_identity.npz", **identity, eligible=eligible,
                        source_inchikey=source["inchikey"], fold=source["fold"])
    write_json(output / "identity_audit.json", identity_audit)
    prevalence = {}
    for split in FOLDS:
        selected = data["fold"] == split
        prevalence[split] = {str(name): {"positive": int(labels["binary"][selected, j].sum()),
                            "negative": int((1 - labels["binary"][selected, j]).sum()),
                            "matched_opposite_donors": int(np.sum(controls["opposite_concept_donor"][selected, j] >= 0))}
                            for j, name in enumerate(labels["binary_names"])}
    write_json(output / "labels.json", {
        "rdkit_version": rdBase.rdkitVersion, "functional_groups": FG_SMARTS,
        "element_order": ELEMENTS, "fingerprints": {"maccs": "166 bits; unused bit0 omitted",
        "morgan": {"radius": 2, "bits": 1024, "include_chirality": False}},
        "label_level": "Molecular graph labels derived from source SMILES; no token/fragment/atom-to-peak assignments",
        "existing_definitions": "aromatic_ring SMARTS is a benzene ring; hydroxyl includes acidic hydroxyl; concepts overlap",
        "prevalence": prevalence})
    write_json(output / "matching_audit.json", matching_audit)
    write_json(output / "weak_losses.json", {
        "interpretation": "Only mass-consistent theoretical neutral-loss flags, not measured fragment identity, atom correspondence, or verified fragmentation mechanism.",
        "eligible": "[M+H]+ only; native valid physical peak slots",
        "mass_difference": "raw precursor m/z minus processed physical peak m/z",
        "tolerance": "max(0.01 Da, 20 ppm of fragment m/z); fixed before any model outcomes",
        "loss_compositions": LOSS_COMPOSITIONS, "per_fold_counts": {f: dict(zip(loss["loss_names"].tolist(),
            loss["spectrum_matches"][data["fold"] == f].sum(0).tolist(), strict=True)) for f in FOLDS}})
    repeats_per = np.bincount(repeat_primary, minlength=len(primary))
    manifest = {
        "schema": "encoder_chemistry_panel_v1", "seed": seed, "dataset": str(tsv.resolve()),
        "dataset_sha256": sha256(tsv), "config": str(config_path.resolve()), "config_sha256": sha256(config_path),
        "script_sha256": sha256(Path(__file__)), "native_preparation_sha256": sha256(Path(__file__).with_name("modal_export_jepa_fps_embeddings.py")),
        "native_peak_preprocessing_sha256": sha256(Path(__file__).parents[1] / "spectra_learning/data/spectra.py"),
        "source_spectra": len(eligible), "identity_audit": identity_audit,
        "selection": "Uniform without replacement over sorted eligible computed molecular connectivities within official folds; uniform seeded representative spectrum within each selected group. Selection never uses model outcomes or concept prevalence.",
        "minimum_processed_peaks": 5,
        "primary_molecules": sizes,
        "eligible_molecules": {f: len(np.unique(identity["molecule"][eligible & (source["fold"] == f)])) for f in FOLDS},
        "repeated_acquisitions": {f: {"alternate_spectra": int(np.sum(repeat_data["fold"] == f)),
            "primary_with_repeats": int(np.sum((data["fold"] == f) & (repeats_per > 0)))} for f in FOLDS},
        "repeat_identity": "Same full SMILES-derived InChIKey AND same canonical isomeric SMILES; every alternate eligible source spectrum retained. Not independent molecules.",
        "test_policy": "No normalization, readout fitting, feature selection, threshold selection, or stopping decisions on test. Label-based opposite-concept donors are fixed intervention controls, not tuning data.",
        "exposure": "Official test has been inspected in prior repository benchmarks; this is a fixed new sample, not an untouched dataset. Unlabeled pretraining overlap is unknown.",
        "scaffold_policy": "Official folds unchanged. Report nonempty Murcko scaffolds absent from selected train separately; acyclic empty scaffolds are not counted as unseen.",
        "unseen_nonempty_train_scaffold_counts": {f: int(np.sum((data["fold"] == f) & data["unseen_nonempty_train_scaffold"])) for f in FOLDS},
        "physical_peak_mz_scale": PEAK_MZ_MAX,
        "normalized_precursor_mz_scale": config["max_precursor_mz"],
        "nuisance_matching": matching_audit,
        "peak_fragment_ground_truth_available": False,
    }
    manifest["files"] = {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
                         for p in sorted(output.iterdir()) if p.is_file() and p.name != "manifest.json"}
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"primary_molecules": sizes, "repeats": manifest["repeated_acquisitions"],
                      "matching": matching_audit["audits"], "output": str(output)}, indent=2), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv", type=Path, default=Path("data/massspecgym/data/MassSpecGym1.5.tsv"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/encoder_chemistry_20260930/panel"))
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--val-size", type=int, default=1024)
    parser.add_argument("--test-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()
    prepare(args.tsv, args.config, args.output, train_size=args.train_size,
            val_size=args.val_size, test_size=args.test_size, seed=args.seed)


if __name__ == "__main__":
    main()
