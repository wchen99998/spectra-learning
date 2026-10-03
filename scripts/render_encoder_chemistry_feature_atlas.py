"""Illustrative SAE feature cards after frozen chemical confirmation, not new evidence."""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw

from scripts.analyze_encoder_chemistry_sae import load_panel, rank_correlation, sha256, write_json
from spectra_learning.data.massspec_targets import FG_SMARTS

# These narrative examples deliberately include a failure and a nuisance feature.
# They are selected after held-out confirmation and never enter any model fit.
CARDS = {
    "parent_1190k": [("halide", 738, "Replicated halide association"),
                     ("sulfonyl", 1347, "Modest sulfonyl association"),
                     ("nitrile", 6482, "Failed nitrile generalization"),
                     ("hydroxyl", 3678, "Hydroxyl association with peak-count dependence")],
    "periodic_1209k": [("halide", 7363, "Replicated halide association"),
                       ("sulfonyl", 3055, "Candidate sulfonyl association; seed variation"),
                       ("nitrile", 6208, "Weak nitrile association with collision-energy dependence"),
                       ("hydroxyl", 1823, "Hydroxyl association with peak-count dependence")],
}


def physical_activations(sparse: dict, feature: int, shape: tuple[int, int]) -> np.ndarray:
    if len(sparse["rows"]) != len(sparse["slots"]):
        raise ValueError("Sparse physical-token coordinates differ")
    index, rank = np.nonzero(sparse["feature_indices"] == feature)
    rows, slots = sparse["rows"][index], sparse["slots"][index]
    coordinates = rows.astype(np.int64)*shape[1] + slots
    if len(np.unique(coordinates)) != len(coordinates):
        raise ValueError("Feature occurs twice in one token's TopK export")
    if (rows < 0).any() or (rows >= shape[0]).any() or (slots < 0).any() or (slots >= shape[1]).any():
        raise ValueError("SAE export contains nonphysical or out-of-panel slots")
    values = np.zeros(shape, np.float32)
    values[rows, slots] = sparse["feature_values"][index, rank]
    return values


def select_examples(scores: np.ndarray, labels: np.ndarray, test: np.ndarray) -> list[dict]:
    examples = []
    for positive in (True, False):
        eligible = test[labels[test].astype(bool) == positive]
        if not len(eligible):
            raise ValueError("Illustrative class has no test support")
        # Deterministic tie break on panel row, never a visual/manual swap.
        chosen = int(eligible[np.argmax(scores[eligible])])
        examples.append({"panel_row": chosen, "molecular_concept_present": positive,
                         "criterion": "Highest test molecular feature-max within this true label class; deterministic first-panel-row tie break"})
    return examples


def render_card(record: dict, panel: dict, activations: np.ndarray, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13, 7.7), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=(1.1, 2.5))
    maximum = max(float(activations[e["panel_row"]].max()) for e in record["examples"])
    for position, example in enumerate(record["examples"]):
        row = example["panel_row"]
        structure = fig.add_subplot(grid[position, 0])
        mol = Chem.MolFromSmiles(str(panel["canonical_smiles"][row]))
        structure.imshow(Draw.MolToImage(mol, size=(650, 360), highlightAtoms=example["concept_atoms"]))
        structure.axis("off")
        true_class = "Concept present" if example["molecular_concept_present"] else "Concept absent · false positive"
        if not example["above_frozen_threshold"] and not example["molecular_concept_present"]:
            true_class = "Concept absent · top negative activation"
        structure.set_title(true_class, loc="left", fontsize=11)
        lower = (f"Panel row {row} · {example['formula']}\n"
                 f"{example['instrument']} · {example['adduct']} · CE {example['raw_ce']}\n"
                 f"Precursor {example['precursor_mz']:.4f} · {example['valid_peaks']} input peaks")
        structure.text(0, -.02, lower, transform=structure.transAxes, fontsize=9, va="top")
        sub = grid[position, 1].subgridspec(2, 1, height_ratios=(1.4, 1), hspace=.06)
        intensity_axis = fig.add_subplot(sub[0])
        activation_axis = fig.add_subplot(sub[1], sharex=intensity_axis)
        valid = panel["peak_valid_mask"][row].astype(bool)
        mz = panel["peak_mz"][row, valid] * 1000
        intensity = panel["peak_intensity"][row, valid]
        feature = activations[row, valid]
        intensity_axis.vlines(mz, 0, intensity, color="#84939d", linewidth=1.1)
        active = feature > 0
        intensity_axis.vlines(mz[active], 0, intensity[active], color="#c36844", linewidth=1.5)
        activation_axis.vlines(mz, 0, feature, color="#c36844", linewidth=1.5)
        intensity_axis.set(ylabel="Input intensity", ylim=(0, max(1., intensity.max())*1.2))
        activation_axis.set(xlabel="Measured peak m/z (native selected physical slots)", ylabel="SAE activation", ylim=(0, maximum*1.35))
        intensity_axis.tick_params(labelbottom=False)
        for axis in (intensity_axis, activation_axis):
            axis.spines[["top", "right"]].set_visible(False)
            axis.set_xlim(max(0., float(mz.min())-10), max(float(mz.max()), example["precursor_mz"])+10)
        # Label only the maximum to keep closely spaced measured peaks legible;
        # the atlas JSON retains every slot, m/z and activation.
        for peak in sorted(example["peaks"], key=lambda p: -p["activation"])[:1]:
            if peak["activation"] <= 0:
                continue
            activation_axis.annotate(f"{peak['mz']:.2f}", (peak["mz"], peak["activation"]),
                                     xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
    name = record["arm"].replace("parent_1190k", "Parent").replace("periodic_1209k", "Latest")
    fig.suptitle(f"{name} · {record['concept']} · feature {record['feature']}\n{record['description']}", fontsize=15)
    for extension in ("png", "pdf"):
        fig.savefig(output.with_suffix("." + extension), dpi=180)
    plt.close(fig)


def create(root: Path) -> dict:
    output = root / "feature_atlas"
    output.mkdir(exist_ok=True)
    panel, labels, _ = load_panel(root / "panel")
    test = np.flatnonzero(panel["fold"] == "test")
    report = {"schema": "post_confirmation_sae_feature_atlas_v1", "illustrative_only": True,
        "examples_selected_after_test_confirmation": True,
        "policy": "Four narrative feature types per checkpoint: replicated halide, sulfonyl candidate, failed/weak nitrile, count-associated hydroxyl. Show deterministic highest-activating test molecule among true positives and negatives. No model or causal selection changes; example pairs are not nuisance-matched.",
        "boundaries": "Molecular SMARTS highlighting identifies graph motifs only. Orange spectrum peaks carry nonzero SAE feature activation, not validated chemical fragments. Native selected measured peaks and transformed model-input intensities are shown. Maxima and physical-slot activation consistency are checked exactly.",
        "source_sha256": sha256(Path(__file__)), "cards": []}
    sections = []
    for arm, choices in CARDS.items():
        directory = root / arm / "sae/depth19/k32/seed0"
        selection_path = directory / "chemical_features.json"
        confirmation_path = directory / "chemical_features_test_confirmation.json"
        confirmation = json.loads(confirmation_path.read_text())
        dictionary = json.loads((directory / "trained_sae.json").read_text())
        if confirmation["selection_sha256"] != sha256(selection_path):
            raise ValueError("Confirmation and frozen selection differ")
        sparse_path = directory / "sparse_tokens.npz"
        if sha256(sparse_path) != dictionary["artifact_files"][sparse_path.name]["sha256"]:
            raise ValueError("Sparse token export differs from completed dictionary")
        with np.load(sparse_path) as data:
            sparse = {key: data[key] for key in data.files}
        if not np.all(panel["peak_valid_mask"][sparse["rows"], sparse["slots"]]):
            raise ValueError("SAE export contains an invalid peak")
        maxima = np.load(directory / "molecule_max.npy", mmap_mode="r")
        for concept, feature, description in choices:
            confirmed = next(r for r in confirmation["families"]["sae"] if r["feature"] == feature)
            if confirmed["concept"] != "fg_"+concept:
                raise ValueError("Narrative feature concept differs from frozen discovery")
            activation = physical_activations(sparse, feature, panel["peak_valid_mask"].shape)
            np.testing.assert_array_equal(activation.max(1), maxima[:, feature])
            column = confirmed["concept_index"]
            examples = select_examples(maxima[:, feature], labels["binary"][:, column], test)
            for example in examples:
                row = example["panel_row"]
                mol = Chem.MolFromSmiles(str(panel["canonical_smiles"][row]))
                matches = mol.GetSubstructMatches(Chem.MolFromSmarts(FG_SMARTS[concept]))
                atoms = sorted({atom for group in matches for atom in group})
                if bool(atoms) != example["molecular_concept_present"]:
                    raise ValueError("Example graph SMARTS disagrees with fixed panel label")
                slots = np.flatnonzero(panel["peak_valid_mask"][row])
                example.update(source_row=int(panel["rows"][row]), identifier=str(panel["identifier"][row]),
                    molecule=str(panel["molecule"][row]), smiles=str(panel["canonical_smiles"][row]),
                    formula=str(panel["formula"][row]), concept_atoms=atoms, concept_atom_matches=[list(m) for m in matches],
                    above_frozen_threshold=bool(maxima[row,feature] > confirmed["threshold"]),
                    maximum_activation=float(maxima[row,feature]), instrument=str(panel["instrument_type"][row]),
                    adduct=str(panel["adduct"][row]), raw_ce=float(panel["collision_energy_raw"][row]) if panel["collision_energy_present"][row] else None,
                    precursor_mz=float(panel["precursor_mz_raw"][row]), valid_peaks=len(slots),
                    peaks=[{"physical_slot":int(slot), "mz":float(panel["peak_mz"][row,slot]*1000),
                            "input_intensity":float(panel["peak_intensity"][row,slot]), "activation":float(activation[row,slot])} for slot in slots])
            valid = panel["peak_valid_mask"][test].astype(bool)
            flat = activation[test][valid]
            profiles = {"test_physical_tokens": int(valid.sum()), "positive_activation_tokens": int((flat > 0).sum()),
                "token_weighted_spearman_mz": rank_correlation(flat, (panel["peak_mz"][test]*1000)[valid]),
                "token_weighted_spearman_input_intensity": rank_correlation(flat, panel["peak_intensity"][test][valid]),
                "interpretation": "Descriptive physical-token weighting gives molecules with more peaks greater weight; no fragment identity inference"}
            stability = {}
            for control in ("k32_seed1", "k64_seed0"):
                control_result = json.loads((directory / "stability" / f"{control}_test_confirmation.json").read_text())
                stability[control] = next(r for r in control_result["records"] if r["feature"] == feature)["activation_alignment"]
            card = {"arm": arm, "concept": concept, "feature": feature, "description": description,
                "selection_sha256": sha256(selection_path), "confirmation_sha256": sha256(confirmation_path),
                "test_metrics": confirmed["test"], "matched_donor_ordering": confirmed["test_matched_donor_ordering"],
                "nuisance_profile": confirmed["test_nuisance_profile"], "nuisance_flags": confirmed["test_nuisance_flags"],
                "train_val_nuisance_flags": confirmed["nuisance_flags"], "token_profile": profiles,
                "stability": stability, "examples": examples}
            filename = f"{arm}_{concept}_f{feature}"
            render_card(card, panel, activation, output / filename)
            image_data = base64.b64encode((output / (filename+".png")).read_bytes()).decode()
            report["cards"].append(card)
            scores = card["test_metrics"]; donor = card["matched_donor_ordering"]
            nuisance = card["nuisance_profile"]["spearman"]
            summary = (f"Test AUC {scores['auroc']:.3f}; AP {scores['average_precision']:.3f}; precision {scores['precision']:.3f}, "
                       f"recall {scores['recall']:.3f}. Matched donor ordering {donor['accuracy']:.3f} over {donor['directed_pairs']} directed pairs (donors reused).")
            nuisance_text = "; ".join(f"{name.replace('_',' ')} ρ={value:.3f}" for name,value in nuisance.items() if value is not None)
            stability_text = "; ".join(f"{name}: held-out activation ρ={value['test_spearman']:.3f}, chemical AUC={value['test']['auroc']:.3f}" for name,value in stability.items())
            sections.append(f'<section id="{filename}"><h2>{html.escape(arm)} · {html.escape(description)} · f{feature}</h2><p>{summary}</p>'
                f'<p class="muted">{html.escape(nuisance_text)}<br>{html.escape(stability_text)}</p>'
                f'<img alt="Measured spectrum, SAE activation per physical peak, and molecular structure for positive and negative examples" src="data:image/png;base64,{image_data}">'
                '<details><summary>Exact molecule identifiers, physical slots, activations and nuisance statistics</summary>'
                f'<pre>{html.escape(json.dumps(card, indent=2))}</pre></details></section>')
            print("Rendered", filename, flush=True)
    write_json(output / "selection.json", report)
    document = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Token SAE chemistry atlas</title><style>
body{font:16px/1.55 system-ui,sans-serif;max-width:1160px;margin:40px auto;padding:0 24px;color:#172733;background:#fafbf9}
h1{font-size:36px;line-height:1.2}h2{font-size:22px}section{border-top:1px solid #bac4c8;padding:26px 0;margin-top:32px}
.muted{color:#526570;font-size:14px}img{width:100%;height:auto;background:white}pre{font:12px/1.4 monospace;white-space:pre-wrap;max-height:500px;overflow:auto;background:#eef2f2;padding:18px}summary{cursor:pointer}strong{font-weight:650}
</style><h1>Token SAE chemistry atlas</h1><p>These cards show chemically associated, ambiguous and failed features from each encoder. The encoder supplies native physical-peak tokens; a TopK SAE at depth 19 (8192 features, k=32) supplies the orange activations.</p>
<p><strong>Illustration after frozen confirmation:</strong> feature types were chosen to explain the results, then the highest-activating test molecule from each true label class was selected deterministically. These examples are not additional unseen evidence and do not alter the frozen feature or causal lists.</p>
<p>Orange peaks indicate SAE activation. Highlighted atoms indicate a SMARTS motif in the known molecular graph. They are <strong>not measured peak-to-fragment annotations</strong>. Spectra show selected measured m/z values and the transformed intensities seen by the model. Example pairs are not nuisance-matched; cohort-level matched ordering and nuisance correlations accompany every card.</p>
'''+"\n".join(sections)+"</html>"
    (root / "feature_atlas.html").write_text(document)
    return report


if __name__ == "__main__":
    create(Path("artifacts/encoder_chemistry_20260930"))
