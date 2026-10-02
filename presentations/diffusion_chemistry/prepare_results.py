"""Populate the supplement only from completed diffusion comparison artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
STUDY = REPO / "artifacts/encoder_chemistry_20260930"
EXTENSION = STUDY / "diffusion_extension"
ANALYSIS = EXTENSION / "diffusion_831k"
ARM = "diffusion_831k"
BASELINES = ("parent_1190k", "periodic_1209k")


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tex(value: str) -> str:
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "_": r"\_",
                    "#": r"\#", "{": r"\{", "}": r"\}", "$": r"\$"}
    return "".join(replacements.get(character, character) for character in value)


def format_ci(record: dict) -> str:
    return f"{record['estimate']:+.4f} [{record['ci95'][0]:+.4f}, {record['ci95'][1]:+.4f}]"


def main() -> None:
    snapshot = load(ROOT / "research/plot_inputs.json")
    readout, sae, provenance = (snapshot[key] for key in ("readout", "sae", "provenance"))
    pilot, extraction = snapshot["pilot"], snapshot["extraction"]
    if not pilot["native_vs_instrumented"]["bitwise_equal"] or pilot["max_sham_relative_l2"] != 0:
        raise ValueError("The diffusion native graph is not qualified")
    if not extraction["extraction_vs_native_pilot"]["bitwise_equal"]:
        raise ValueError("The diffusion token cache is not qualified")
    if readout["test_molecules"] != 1024 or readout["native_state"] != 27 or readout["seeds"] != [0, 1, 2]:
        raise ValueError("The completed readout differs from the report protocol")
    if readout["arms"] != [*BASELINES, ARM] or sae["diffusion_arm"] != ARM:
        raise ValueError("The report must describe the pinned three-arm comparison")
    figures = ("final_token_auroc", "paired_readout_differences", "sae_quality_and_associations")

    def metric(encoder: str, kind: str = "attention", group: str = "MACCS") -> dict:
        return readout["models"][f"{encoder}/depth27/{kind}"]["groups"][group]["ensemble"]

    def contrast(encoder: str) -> dict:
        return readout["contrasts"][f"{ARM}_minus_{encoder}_attention"]["groups"]["MACCS"]["ensemble"]

    primary = metric(ARM)
    differences = {name: contrast(name) for name in BASELINES}
    if all(record["ci95"][0] > 0 for record in differences.values()):
        verdict = "Diffusion scores higher than both JEPA checkpoints on the shared final-token chemical readout."
        headline = "Diffusion leads the final-token chemical readout"
    elif all(record["ci95"][1] < 0 for record in differences.values()):
        verdict = "Diffusion scores lower than both JEPA checkpoints on the shared final-token chemical readout."
        headline = "Diffusion trails the final-token chemical readout"
    else:
        verdict = "The final-token readout gives a checkpoint-specific comparison; paired intervals qualify the differences."
        headline = "Token-aware chemistry differs across the three endpoints"
    primary_sae = next(record for record in sae["quality"]
                       if (record["encoder"], record["k"], record["seed"]) == (ARM, 32, 0))
    fve = primary_sae["quality"]["test"]["fve_vs_train_mean"]
    relative_l2 = primary_sae["quality"]["test"]["relative_l2"]
    reference_l2 = {name: next(record["quality"]["test"]["relative_l2"] for record in sae["quality"]
                              if (record["encoder"], record["k"], record["seed"]) == (name, 32, 0))
                    for name in BASELINES}
    selection_count = len(sae["frozen_primary_records"])
    chemical_records = [record for concept in sae["chemical_confirmation"][ARM]["families"]["sae"].values()
                        for record in concept["records"]]
    example = next(record for record in chemical_records if record["feature"] == 4682)
    candidate_summary = (f"The strongest hydroxyl candidate (f4682) has test AUROC {example['test']['auroc']:.3f}, "
                         f"but also mass correlation {example['test_nuisance_profile']['spearman']['exact_mass']:.3f}; "
                         f"strict matched-pair ordering is {example['test_matched_donor_ordering']['accuracy']:.3f}.")
    controls = readout["contrasts"][f"attention_minus_mean_{ARM}"]["groups"]["MACCS"]["ensemble"]
    attention_takeaway = f"Diffusion attention minus mean control: {controls['estimate']:+.4f} MACCS AUROC."
    if controls["ci95"][0] <= 0 <= controls["ci95"][1]:
        attention_takeaway = f"Diffusion attention gains {controls['estimate']:+.4f} MACCS AUROC over mean; its paired interval includes zero."
    macros = {
        "DiffusionVerdict": verdict,
        "DiffusionPrimaryNumber": f"{primary['estimate']:.4f}",
        "DiffusionTakeaway": "A matched token-level evaluation; training exposure and native input quantization still differ.",
        "ReadoutTitle": headline,
        "ReadoutTakeaway": attention_takeaway,
        "MaccsTargetCount": str(readout["groups"]["MACCS"]["targets"]),
        "FunctionalTargetCount": str(readout["groups"]["functional_groups"]["targets"]),
        "BootstrapCount": f"{readout['bootstrap_replicates']:,}",
        "BootstrapDefined": str(primary["valid_bootstrap_replicates"]),
        "PairedTitle": "Paired molecules separate the checkpoint differences",
        "PairedTakeaway": f"Attention differences: parent {differences[BASELINES[0]]['estimate']:+.4f}; periodic {differences[BASELINES[1]]['estimate']:+.4f} MACCS AUROC.",
        "SaeTitle": "Sparse reconstruction and chemical recognition are separate",
        "SaeTakeaway": f"Diffusion k=32 seed0: {100*fve:.1f}% held-out variance explained; {selection_count} chemical candidates frozen before test.",
        "ChemicalCandidateSummary": candidate_summary,
        "RelativeLTwoNote": (f"original-unit relative L2 is {relative_l2:.3f} for diffusion versus "
                              f"{reference_l2[BASELINES[0]]:.3f} / {reference_l2[BASELINES[1]]:.3f} for parent / periodic."),
        "ClosingTakeaway": "Token-level readouts answer chemical accessibility; SAE structure adds a separate, qualified view.",
    }
    research = ROOT / "research"
    research.mkdir(exist_ok=True)
    (research / "results.tex").write_text("% Generated from completed artifacts by prepare_results.py.\n" +
        "".join(f"\\newcommand{{\\{key}}}{{{tex(value)}}}\n" for key, value in macros.items()))

    index = {
        "status": "populated_from_completed_artifacts", "primary_protocol": readout["ensemble"],
        "target_support": readout["support_policy"], "bootstrap_protocol": readout["bootstrap_policy"],
        "sae_protocol": sae["selection_policy"], "native_tokens": "Physical peaks only; CLS/padding excluded",
        "claims": {
            "slide1_diffusion_attention_MACCS": primary,
            "slide3_attention_minus_mean_MACCS": controls,
            "slide4_paired_attention_MACCS": differences,
            "slide5_primary_diffusion_SAE_test_quality": primary_sae["quality"]["test"],
            "slide5_frozen_candidates": selection_count,
            "slide7_illustrative_strongest_hydroxyl": {
                "selection": "Descriptive strongest test association among the already frozen hydroxyl candidates; not an independent confirmatory case",
                "feature": 4682, "test_auroc": example["test"]["auroc"],
                "mass_spearman": example["test_nuisance_profile"]["spearman"]["exact_mass"],
                "matched_ordering": example["test_matched_donor_ordering"],
            },
            "slide6_nominal_presentations": provenance["training_exposure"],
        },
        "figure_sources": {stem: str((ANALYSIS / (stem + "_slide.pdf")).relative_to(REPO)) for stem in figures},
        "source_files": snapshot["source_files"],
        "compact_inputs_sha256": digest(ROOT / "research/plot_inputs.json"),
        "plotting_source_sha256": digest(ROOT / "make_figures.py"),
        "limits": ["Descriptive checkpoint comparison, not a controlled objective experiment",
                   "Scores are not calibrated molecular probabilities",
                   "No diffusion feature ablation or mechanistic sufficiency claim",
                   "No measured token-to-fragment assignments", "One pretrained encoder per recipe"],
    }
    (research / "evidence_index.json").write_text(json.dumps(index, indent=2, allow_nan=False) + "\n")
    plan = load(research / "report_plan.json")
    plan["status"] = "completed_measurements_populated"
    plan["result_build_guard"] = "prepare_results.py requires completed result JSON and native/cache parity before writing results.tex"
    (research / "report_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    notes = ["# Diffusion supplement: speaker notes", "",
             f"Primary final-token MACCS AUROC: {primary['estimate']:.6f}, 95% descriptive interval {primary['ci95']}.", "",
             "The protocol preserves physical peak tokens and fits the same downstream attention and mean-control budgets on each frozen native encoder.", "",
             "## Paired final-token differences", ""]
    notes.extend(f"- Diffusion minus {name}: {format_ci(record)}." for name, record in differences.items())
    notes += ["", "All intervals resample molecules jointly across checkpoints. They do not include pretraining-seed variation and are not multiplicity adjusted.", "",
              "## Sparse decomposition", "",
              f"Raw state 19, k=32, seed 0: FVE versus the training mean is {fve:.6f}. {selection_count} candidates were frozen on training/validation data before test confirmation. All three fitted dictionaries per encoder are shown, including sparsity and seed controls.", "",
              f"This is not uniformly superior reconstruction: original-unit relative L2 is {relative_l2:.6f} for diffusion, versus {reference_l2[BASELINES[0]]:.6f} for parent and {reference_l2[BASELINES[1]]:.6f} for periodic. Different representation means and variance scales change the denominators. The slide specifically claims higher explained centered variance, not lower error under every normalization.", "",
              "Chemical-association points average all frozen candidates for a concept; absence remains a gap. Feature IDs belong to their own dictionary. Better reconstruction does not by itself establish better chemistry; no diffusion feature intervention was run.", "",
              candidate_summary, "",
              "The f4682 summary is a descriptive illustration of the strongest test-associated hydroxyl candidate among the already frozen list. It was not used to tune the feature list or inference protocol. Matched donor pairs reuse molecules and are not independent counterfactual observations.", "",
              "## Training and provenance", "",
              "All three W&B histories start at global step zero. The periodic student was freshly trained; the parent is its teacher initialization, not its own weight initialization. Nominal presentations multiply global step by global batch 1,024, with accumulation already included. These are not unique spectra counts.", "",
              "The dataset revision, manifest hash and shard-plan hash are shared. Diffusion has a different objective, shorter exposure, no frozen teacher and native m/z midpoint quantization at 0.48828125 Da. This is an endpoint comparison, not a clean causal estimate of the diffusion loss.", "",
              "Only train.py is archived as historical W&B source. The current native forward source is separately hashed and its actual remote execution is attested; exact native graph and cache parity are verified.", "",
              "## Experiment entry points", "",
              "The presentation build uses only frozen compact inputs. The original experiment is separately orchestrated by `scripts/run_encoder_chemistry_diffusion.py` in three phases:", "",
              "```sh", ".venv/bin/python -m scripts.run_encoder_chemistry_diffusion prepare",
              ".venv/bin/python -m scripts.run_encoder_chemistry_diffusion fit",
              ".venv/bin/python -m scripts.run_encoder_chemistry_diffusion download", "```", "",
              "Preparation requires authenticated pinned checkpoint access and is protected once the committed native cache exists. Fitting schedules the prescribed GPU readouts and SAEs; downloading retrieves their saved outputs. Rebuilding this PDF invokes none of these experiment phases."]
    (ROOT / "diffusion_chemistry_notes.md").write_text("\n".join(notes) + "\n")
    print("Populated seven-slide diffusion supplement from completed results.")


if __name__ == "__main__":
    main()
