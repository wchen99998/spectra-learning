"""Paired molecular analysis of saved native input effects; never reruns models."""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

ARMS = ("parent_1190k", "periodic_1209k")
PHASES = ("corruptions", "observed_repeats", "metadata_factorial")
SEED, BOOTSTRAPS = 20260930, 2000
GENERIC = ("attenuate_top20", "attenuate_random20", "remove_top20", "remove_random20",
           "permute_intensity", "mz_jitter_0p01da", "mz_shift_plus_0p1da", "observed_ce_metadata")
LOSS_CONTEXTS = {
    "H2O": ("elem_O", "fg_hydroxyl", "fg_carboxyl"),
    "NH3": ("elem_N", "fg_amine"),
    "CO": ("elem_O", "fg_carboxyl", "fg_amide", "fg_ester", "fg_ketone", "fg_aldehyde"),
    "CO2": ("elem_O", "fg_carboxyl"), "HCN": ("elem_N", "fg_nitrile"),
    "H2S": ("elem_S", "fg_thiol"),
}
COLORS = {ARMS[0]: "#3a6186", ARMS[1]: "#b06436"}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def clean_json(value):
    if isinstance(value, dict): return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray): return clean_json(value.tolist())
    if isinstance(value, np.generic): return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def write_json(path, record):
    Path(path).write_text(json.dumps(clean_json(record), indent=2, allow_nan=False) + "\n")


@lru_cache(maxsize=4)
def bootstrap_counts(n):
    return np.random.default_rng(SEED).multinomial(n, np.full(n, 1 / n), BOOTSTRAPS).astype(np.float32)


def summarize(values, *, interval=True):
    values = np.asarray(values, np.float64)
    if not len(values): return {"n": 0, "mean": None, "ci95": None}
    result = {"n": len(values), "mean": float(values.mean()), "median": float(np.median(values))}
    if interval:
        samples = bootstrap_counts(len(values)) @ values / len(values)
        result["ci95"] = np.quantile(samples, [.025, .975]).tolist()
    return result


def ensemble(logits, labels):
    """Average probabilities across heads; cast unsigned labels before margins."""
    logits = np.asarray(logits, np.float64)
    y = np.asarray(labels, np.float64)
    p = expit(logits).mean(axis=2)
    z = logits.mean(axis=2)
    epsilon = np.finfo(np.float64).eps
    return {
        "probability": p, "logit": z,
        "bce": -(y[None] * np.log(np.clip(p, epsilon, 1 - epsilon))
                 + (1 - y[None]) * np.log(np.clip(1 - p, epsilon, 1 - epsilon))),
        "signed_margin": (2 * y[None] - 1) * z,
        "true_label_probability": y[None] * p + (1 - y[None]) * (1 - p),
        "brier": (p - y[None]) ** 2,
        "accuracy_at_half": ((p >= .5) == y[None]).astype(float),
    }


def output_cosine(a, b):
    return np.sum(a * b, axis=-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-20)


def load_phase(root, name, phase, panel, labels, target_names):
    path = root / name / "causal" / f"input_{phase}"
    manifest = json.loads((path / "manifest.json").read_text())
    if not manifest["complete"] or digest(path / "predictions.npz") != manifest["predictions_sha256"]:
        raise ValueError("Input result is incomplete or differs from its recorded hash")
    if manifest["target_names"] != target_names:
        raise ValueError("Chemical target ordering differs from saved readout logits")
    for filename, checksum in manifest["panel_sha256"].items():
        if digest(root / "panel" / filename) != checksum:
            raise ValueError(f"Intervention panel changed: {filename}")
    with np.load(path / "predictions.npz", allow_pickle=False) as saved:
        data = dict(saved)
    rows = data["rows"]
    np.testing.assert_array_equal(rows, np.flatnonzero(panel["fold"] == "test"))
    np.testing.assert_array_equal(data["source_rows"], panel["rows"][rows])
    if len(np.unique(panel["molecule"][rows])) != len(rows):
        raise ValueError("Bootstrap cohort must contain unique molecules")
    if data["logits"].shape != (len(data["variants"]), len(rows), 3, 188) or not np.isfinite(data["logits"]).all():
        raise ValueError("Expected finite [variant,molecule,seed,target] logits")
    if phase == "metadata_factorial":
        if not manifest["factorization_passed"] or not all(
            item["bitwise_equal"] for batch in manifest["diagonal_shams"] for item in batch.values()):
            raise ValueError("Metadata factorization lacks exact native diagonal shams")
    return {"arrays": data, "manifest": manifest, "metrics": ensemble(data["logits"], labels[rows])}


def target_changes(metrics, labels, names, left, right, supported):
    p = metrics["probability"]
    y = labels[supported].astype(bool)
    before, after = p[left, supported] >= .5, p[right, supported] >= .5
    output = []
    for column, name in enumerate(names):
        positive = y[:, column]
        output.append({"target": name, "positive_molecules": int(positive.sum()),
            "negative_molecules": int((~positive).sum()),
            "delta_probability": float((p[right, supported, column] - p[left, supported, column]).mean()),
            "delta_bce": float((metrics["bce"][right, supported, column] - metrics["bce"][left, supported, column]).mean()),
            "delta_signed_margin": float((metrics["signed_margin"][right, supported, column] - metrics["signed_margin"][left, supported, column]).mean()),
            "correct_to_wrong": int(((before[:, column] == positive) & (after[:, column] != positive)).sum()),
            "wrong_to_correct": int(((before[:, column] != positive) & (after[:, column] == positive)).sum()),
            "false_positive_before": int((before[:, column] & ~positive).sum()),
            "false_positive_after": int((after[:, column] & ~positive).sum()),
            "false_negative_before": int((~before[:, column] & positive).sum()),
            "false_negative_after": int((~after[:, column] & positive).sum())})
    return output


def contrast(phase, left, right, support, groups, *, interval=True):
    m = phase["metrics"]
    output = {"eligible_molecules": int(support.sum()), "groups": {}}
    rows = {}
    for group, columns in groups.items():
        row = {}
        for metric in ("bce", "signed_margin", "true_label_probability", "brier", "accuracy_at_half"):
            difference = (m[metric][right] - m[metric][left])[:, columns].mean(axis=-1)
            row["delta_" + metric] = difference
        a, b = m["logit"][left][:, columns], m["logit"][right][:, columns]
        row["readout_logit_cosine"] = output_cosine(a, b)
        row["probability_l1_drift"] = np.abs(m["probability"][right] - m["probability"][left])[:, columns].mean(-1)
        output["groups"][group] = {key: summarize(value[support], interval=interval) for key, value in row.items()}
        rows[group] = row
    return output, rows


def analyze_arm(name, phases, panel, labels, groups, names):
    out = {"corruptions": {}, "weak_losses": {}, "observed_repeats": {}, "metadata_factorial": {}}
    row_effects = {}
    corruption = phases["corruptions"]
    arrays = corruption["arrays"]
    variants = arrays["variants"].tolist()
    y = labels[arrays["rows"]]
    for index, variant in enumerate(variants[1:], start=1):
        support = arrays["supported"][index]
        result, rows = contrast(corruption, 0, index, support, groups, interval=variant in GENERIC)
        result["changed_molecules"] = int(arrays["changed"][index, support].sum())
        result["targeted_peaks"] = int(arrays["target_peak_mask"][index, support].sum())
        if variant in GENERIC:
            result["target_changes"] = target_changes(corruption["metrics"], y, names, 0, index, support)
            row_effects[variant] = {"support": support, **rows["trained_targets"]}
        out["corruptions"][variant] = result
    for loss, contexts in LOSS_CONTEXTS.items():
        out["weak_losses"][loss] = {}
        for mode in ("attenuate", "remove"):
            target, control = [variants.index(f"{mode}_loss_{prefix}{loss}") for prefix in ("", "matched_")]
            support = arrays["supported"][target] & arrays["supported"][control]
            np.testing.assert_array_equal(arrays["supported"][target], arrays["supported"][control])
            np.testing.assert_array_equal(arrays["target_peak_mask"][target].sum(1), arrays["target_peak_mask"][control].sum(1))
            if np.any(arrays["target_peak_mask"][target] & arrays["target_peak_mask"][control]):
                raise ValueError("Weak-loss target/control peaks overlap")
            result, _ = contrast(corruption, control, target, support, groups, interval=False)
            result["inference"] = "Exploratory point estimates only; no significance claim for rare matched cohorts"
            result["interpretation"] = "Theoretical precursor-neutral-loss mass consistency, not an observed fragment assignment"
            result["molecular_contexts"] = {}
            for context in contexts:
                column = names.index(context)
                for present in (True, False):
                    cohort = support & (y[:, column] == present)
                    prefix = context + ("_present" if present else "_absent")
                    result["molecular_contexts"][prefix] = {
                        "n": int(cohort.sum()), "at_least_5": bool(cohort.sum() >= 5),
                        "delta_target_bce": summarize((corruption["metrics"]["bce"][target, :, column]
                            - corruption["metrics"]["bce"][control, :, column])[cohort], interval=False),
                        "delta_target_margin": summarize((corruption["metrics"]["signed_margin"][target, :, column]
                            - corruption["metrics"]["signed_margin"][control, :, column])[cohort], interval=False),
                    }
            out["weak_losses"][loss][mode] = result
    repeats = phases["observed_repeats"]
    support = repeats["arrays"]["supported"]
    repeat_pairs = {"metadata_only_on_anchor": (0, 1), "observed_repeat_own_metadata": (0, 2),
                    "repeat_with_anchor_metadata": (0, 3), "own_vs_anchor_metadata_on_repeat": (3, 2)}
    for title, (left, right) in repeat_pairs.items():
        result, rows = contrast(repeats, left, right, support, groups)
        result["target_changes"] = target_changes(repeats["metrics"], y, names, left, right, support)
        out["observed_repeats"][title] = result
        row_effects[title] = {"support": support, **rows["trained_targets"]}
    factorial = phases["metadata_factorial"]
    support = factorial["arrays"]["supported"]
    comparisons = {"final_affine_only": (0, 1), "blocks_only": (0, 2), "both_paths": (0, 3)}
    for title, (left, right) in comparisons.items():
        result, rows = contrast(factorial, left, right, support, groups)
        result["target_changes"] = target_changes(factorial["metrics"], y, names, left, right, support)
        out["metadata_factorial"][title] = result
        row_effects[title] = {"support": support, **rows["trained_targets"]}
    interaction = {}
    for group, columns in groups.items():
        interaction[group] = {}
        for metric in ("bce", "signed_margin", "true_label_probability"):
            value = factorial["metrics"][metric]
            delta = (value[3] - value[2] - value[1] + value[0])[:, columns].mean(-1)
            interaction[group][metric] = summarize(delta[support])
            if group == "trained_targets": row_effects["interaction_" + metric] = {"support": support, "effect": delta}
    out["metadata_factorial"]["interaction"] = interaction
    # Native token parity is separately bitwise. Independent GPU readout scoring
    # can differ by float32 rounding; measure it, and pair effects within jobs.
    scoring_checks = {}
    def scoring_error(left, right):
        delta = np.asarray(left, np.float64) - np.asarray(right, np.float64)
        return {"bitwise_equal": bool(np.array_equal(left, right)),
                "max_absolute_logit_error": float(np.abs(delta).max()),
                "rms_logit_error": float(np.sqrt(np.square(delta).mean()))}
    for phase_name, phase in phases.items():
        np.testing.assert_array_equal(phase["arrays"]["rows"], arrays["rows"])
        if phase["manifest"]["readout_weights_sha256"] != corruption["manifest"]["readout_weights_sha256"]:
            raise ValueError("Readout weights differ between intervention phases")
        scoring_checks[phase_name + "_baseline"] = scoring_error(phase["arrays"]["logits"][0], arrays["logits"][0])
    scoring_checks["ce_factorial_vs_repeat"] = scoring_error(factorial["arrays"]["logits"][3], repeats["arrays"]["logits"][1])
    scoring_checks["ce_corruption_vs_repeat"] = scoring_error(arrays["logits"][variants.index("observed_ce_metadata")], repeats["arrays"]["logits"][1])
    out["cross_phase_readout_scoring_checks"] = scoring_checks
    out["baseline"] = {group: {metric: float(corruption["metrics"][metric][0][:, columns].mean())
        for metric in ("bce", "signed_margin", "true_label_probability", "brier", "accuracy_at_half")}
        for group, columns in groups.items()}
    return out, row_effects


def case_summaries(root, all_phases, panel, labels, names, groups):
    candidates = json.loads((root / "panel/case_candidates.json").read_text())
    if candidates["model_outcomes_used"]: raise ValueError("Cases must be selected before model outcomes")
    for filename, checksum in candidates["source_sha256"].items():
        if digest(root / "panel" / filename) != checksum: raise ValueError("Prespecified case source changed")
    records = []
    for case in candidates["candidate_cases"]:
        record = {key: case[key] for key in ("case_id", "category", "fold", "purpose", "limitations")}
        record["concept"] = case.get("concept")
        record["spectra"] = [{key: spectrum[key] for key in (
            "array", "panel_index", "primary_index", "identifier", "canonical_smiles",
            "positive_concepts", "processed_peak_count", "collision_energy")} for spectrum in case["spectra"]]
        record["checkpoints"] = {}
        if case["fold"] != "test":
            record["availability"] = "No intervention output: these prespecified molecules are outside the test sweep"
            records.append(record); continue
        for name in ARMS:
            phases = all_phases[name]
            arrays = phases["corruptions"]["arrays"]
            position = {int(row): i for i, row in enumerate(arrays["rows"])}
            entries = []
            for spectrum in case["spectra"]:
                row = spectrum["primary_index"]
                index = position[row]
                if spectrum["array"] == "repeat_inputs.npz":
                    repeat = phases["observed_repeats"]
                    actual = int(repeat["arrays"]["donor_repeat_indices"][index])
                    if actual != spectrum["panel_index"]:
                        entries.append({"available": False, "reason": "Prespecified repeat differs from sweep donor; no substitution",
                                        "requested_repeat_index": spectrum["panel_index"], "sweep_repeat_index": actual})
                        continue
                    metric, variant = repeat["metrics"], 2
                else:
                    metric, variant = phases["corruptions"]["metrics"], 0
                p = metric["probability"][variant, index]
                entry = {"available": True, "panel_row": row,
                    "bce_trained_targets": float(metric["bce"][variant, index, groups["trained_targets"]].mean()),
                    "true_label_probability": float(metric["true_label_probability"][variant, index, groups["trained_targets"]].mean()),
                    "known_molecular_labels": {target: bool(labels[row, column]) for column, target in enumerate(names[166:], 166)},
                    "binary_target_probabilities": {target: float(p[column]) for column, target in enumerate(names[166:], 166)}}
                if case.get("concept"):
                    column = names.index(case["concept"])
                    entry["prespecified_concept_probability"] = float(p[column])
                    entry["prespecified_concept_present"] = bool(labels[row, column])
                if spectrum["array"] == "inputs.npz":
                    entry["generic_effects"] = {variant_name: float(
                        (phases["corruptions"]["metrics"]["bce"][arrays["variants"].tolist().index(variant_name), index]
                         - metric["bce"][0, index])[groups["trained_targets"]].mean())
                        for variant_name in ("remove_top20", "remove_random20", "observed_ce_metadata")
                        if arrays["supported"][arrays["variants"].tolist().index(variant_name), index]}
                entries.append(entry)
            record["checkpoints"][name] = entries
        record["paired_case_summary"] = {}
        for name, entries in record["checkpoints"].items():
            if len(entries) != 2 or not all(entry["available"] for entry in entries):
                continue
            if case.get("concept"):
                positive = next((entry for entry in entries if entry["prespecified_concept_present"]), None)
                negative = next((entry for entry in entries if not entry["prespecified_concept_present"]), None)
                if positive is not None and negative is not None:
                    gap = positive["prespecified_concept_probability"] - negative["prespecified_concept_probability"]
                    record["paired_case_summary"][name] = {"positive_minus_negative_concept_probability": gap,
                        "correct_concept_order": gap > 0}
            elif case["spectra"][1]["array"] == "repeat_inputs.npz":
                record["paired_case_summary"][name] = {"repeat_minus_anchor_bce":
                    entries[1]["bce_trained_targets"] - entries[0]["bce_trained_targets"]}
            else:
                differences = []
                for target in entries[0]["known_molecular_labels"]:
                    a, b = [entry["known_molecular_labels"][target] for entry in entries]
                    if a == b: continue
                    gap = (entries[0]["binary_target_probabilities"][target]
                           - entries[1]["binary_target_probabilities"][target]) * (1 if a else -1)
                    differences.append({"target": target, "positive_minus_negative_probability": gap,
                                        "correct_order": gap > 0})
                record["paired_case_summary"][name] = {"differing_molecular_labels": differences}
        records.append(record)
    return {"selection_sha256": digest(root / "panel/case_candidates.json"),
            "model_outcomes_used_for_case_selection": False, "cases": records,
            "selection_note": "All32 prespecified cases retained, including unavailable pairs and design failures; figure uses all test matched-concept pairs plus the test coverage failure."}


def figures(output, report, cases):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    files = []
    def save(fig, stem):
        fig.savefig(output / (stem + ".png"), dpi=180, bbox_inches="tight")
        fig.savefig(output / (stem + ".pdf"), bbox_inches="tight")
        plt.close(fig); files.extend([stem + ".png", stem + ".pdf"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    labels = [name.replace("_", " ") for name in GENERIC]
    for offset, name in enumerate(ARMS):
        arm = report["checkpoints"][name]
        for axis, metric in zip(axes, ("delta_bce", "probability_l1_drift")):
            records = [arm["corruptions"][v]["groups"]["trained_targets"][metric] for v in GENERIC]
            mean = np.array([r["mean"] for r in records]); ci = np.array([r["ci95"] for r in records])
            axis.errorbar(mean, np.arange(len(labels)) + (offset - .5) * .19,
                xerr=np.maximum(np.stack((mean-ci[:,0], ci[:,1]-mean)), 0), fmt="o", color=COLORS[name],
                markersize=4, capsize=2, label=name)
        axes[0].set_yticks(np.arange(len(labels)), labels)
        axes[1].set_yticks(np.arange(len(labels)), labels)
    axes[0].axvline(0, color=".7", lw=.8); axes[0].set_xlabel("Change in ensemble BCE (higher = worse)")
    axes[1].set_xlabel("Mean absolute probability change")
    for axis in axes: axis.invert_yaxis(); axis.grid(axis="x", alpha=.18)
    axes[0].legend(frameon=False); fig.suptitle("Native token-readout response to fixed input interventions\n177 trained targets; paired molecule bootstrap 95% intervals")
    save(fig, "input_robustness")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    comparisons = ("metadata_only_on_anchor", "observed_repeat_own_metadata", "repeat_with_anchor_metadata", "own_vs_anchor_metadata_on_repeat")
    for offset, name in enumerate(ARMS):
        arm = report["checkpoints"][name]
        for axis, metric in zip(axes, ("delta_bce", "probability_l1_drift")):
            records = [arm["observed_repeats"][v]["groups"]["trained_targets"][metric] for v in comparisons]
            mean = np.array([r["mean"] for r in records]); ci=np.array([r["ci95"] for r in records])
            axis.errorbar(np.arange(4)+(offset-.5)*.16, mean, yerr=np.maximum(np.stack((mean-ci[:,0],ci[:,1]-mean)),0),
                          fmt="o", capsize=3, color=COLORS[name], label=name)
    for axis in axes:
        axis.set_xticks(range(4), ["CE swap\non anchor", "Observed repeat\nwith own CE", "Repeat with\nanchor CE", "Own CE vs anchor CE\non repeat"])
        axis.axhline(0,color=".7",lw=.8);axis.grid(axis="y",alpha=.18)
    axes[0].set_ylabel("Change in ensemble BCE");axes[1].set_ylabel("Absolute probability drift")
    axes[0].legend(frameon=False);fig.suptitle("Same-molecule observed CE pairs (689 eligible molecules)\nPrediction stability is not embedding invariance")
    save(fig, "observed_ce_pairs")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    titles=("final_affine_only","blocks_only","both_paths")
    for offset,name in enumerate(ARMS):
        for axis,metric in zip(axes,("delta_bce","delta_signed_margin")):
            records=[report["checkpoints"][name]["metadata_factorial"][v]["groups"]["trained_targets"][metric]for v in titles]
            mean=np.array([r["mean"]for r in records]);ci=np.array([r["ci95"]for r in records])
            axis.errorbar(np.arange(3)+(offset-.5)*.16,mean,yerr=np.maximum(np.stack((mean-ci[:,0],ci[:,1]-mean)),0),fmt="o",capsize=3,color=COLORS[name],label=name)
    for axis in axes:
        axis.set_xticks(range(3),["Final affine only","Blocks only","Both paths"]);axis.axhline(0,color=".7",lw=.8);axis.grid(axis="y",alpha=.18)
    axes[0].set_ylabel("Change in ensemble BCE (higher = worse)")
    axes[1].set_ylabel("Change in signed correct-label logit margin")
    axes[0].legend(frameon=False);fig.suptitle("Observed CE path decomposition on a fixed physical spectrum\nRaw25 swaps; native diagonal shams match bitwise")
    save(fig,"metadata_paths")
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True,constrained_layout=True)
    losses=list(LOSS_CONTEXTS)
    for axis,mode in zip(axes,("attenuate","remove")):
        for offset,name in enumerate(ARMS):
            points=[report["checkpoints"][name]["weak_losses"][loss][mode]["groups"]["trained_targets"]["delta_bce"]["mean"] for loss in losses]
            axis.scatter(np.arange(6)+(offset-.5)*.16,points,s=30,color=COLORS[name],label=name)
        counts=[report["checkpoints"][ARMS[0]]["weak_losses"][loss][mode]["eligible_molecules"]for loss in losses]
        axis.set_xticks(range(6),[loss+f"\nn={count}"for loss,count in zip(losses,counts)])
        axis.axhline(0,color=".6",lw=.8);axis.grid(axis="y",alpha=.18);axis.set_title(mode.capitalize())
    axes[0].set_ylabel("BCE: theoretical-loss peaks minus matched control peaks")
    axes[0].legend(frameon=False);fig.suptitle("Weak neutral-loss comparisons: matched eligible cohorts only\nRare samples; exploratory point estimates without significance claims")
    save(fig,"weak_loss_matched_controls")
    names=report["definitions"]["primary_targets"]
    binary=[name for name in names if not name.startswith("maccs_")]
    values=[]
    for name in ARMS:
        for path in titles:
            records={row["target"]:row for row in report["checkpoints"][name]["metadata_factorial"][path]["target_changes"]}
            values.append([records[target]["delta_signed_margin"]for target in binary])
    values=np.asarray(values).T;limit=float(np.abs(values).max())
    fig,axis=plt.subplots(figsize=(9,7),constrained_layout=True)
    im=axis.imshow(values,aspect="auto",cmap="RdBu",vmin=-limit,vmax=limit)
    reference={row["target"]:row for row in report["checkpoints"][ARMS[0]]["metadata_factorial"]["both_paths"]["target_changes"]}
    axis.set_yticks(range(len(binary)),[target+f" [{reference[target]['positive_molecules']}+]"for target in binary])
    axis.set_xticks(range(6),["Parent\nfinal","Parent\nblocks","Parent\nboth","Latest\nfinal","Latest\nblocks","Latest\nboth"])
    axis.axvline(2.5,color="white",lw=2)
    fig.colorbar(im,ax=axis,label="Change in signed correct-label logit margin")
    axis.set_title("Molecular-label responses to observed CE paths\nSame 689 molecules; brackets show positive-label counts; positive margin favors truth")
    save(fig,"metadata_chemical_target_responses")
    selected=[c for c in cases["cases"] if c["fold"]=="test" and c["category"] in ("matched_concept","coverage_limitation")]
    fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True,constrained_layout=True)
    for axis,name in zip(axes,ARMS):
        for index,case in enumerate(selected):
            for member,entry in enumerate(case["checkpoints"][name]):
                axis.scatter(index+(member-.5)*.16,entry["prespecified_concept_probability"],
                    marker="o" if entry["prespecified_concept_present"] else "x",s=45,color=COLORS[name])
        axis.axhline(.5,color=".6",ls=":");axis.set_ylim(-.03,1.03);axis.set_ylabel(name+"\nConcept probability");axis.grid(axis="y",alpha=.15)
    axes[-1].set_xticks(range(len(selected)),[c["case_id"]+"\n"+c["concept"].removeprefix("fg_")for c in selected])
    fig.suptitle("Prespecified test cases: matched concept pairs and coverage failure\nCircle = label present; cross = label absent; all eligible cases shown")
    save(fig,"prespecified_chemical_cases")
    return files


def concise_summary(report, cases):
    output={"primary_targets":177,"eligible_generic_molecules":1024,"eligible_ce_molecules":689,
        "metric":"Ensemble-probability BCE over train-supported targets; paired molecule bootstrap with 2,000 resamples",
        "checkpoints":{},"paired_latest_minus_parent":{},
        "limitations":report["limitations"],"figures":report["figures"]}
    for name,arm in report["checkpoints"].items():
        output["checkpoints"][name]={"baseline":arm["baseline"]["trained_targets"],
            "corruption_delta_bce":{key:arm["corruptions"][key]["groups"]["trained_targets"]["delta_bce"]for key in GENERIC},
            "observed_repeat":{key:{metric:value for metric,value in item["groups"]["trained_targets"].items()
                if metric in ("delta_bce","probability_l1_drift","readout_logit_cosine")}
                for key,item in arm["observed_repeats"].items()},
            "metadata_path_delta_bce":{key:arm["metadata_factorial"][key]["groups"]["trained_targets"]["delta_bce"]
                for key in ("final_affine_only","blocks_only","both_paths")},
            "metadata_bce_interaction":arm["metadata_factorial"]["interaction"]["trained_targets"]["bce"],
            "weak_loss_target_minus_control_bce":{loss:{mode:{"n":item["eligible_molecules"],
                "mean":item["groups"]["trained_targets"]["delta_bce"]["mean"]}for mode,item in modes.items()}
                for loss,modes in arm["weak_losses"].items()},
            "cross_job_readout_rounding":arm["cross_phase_readout_scoring_checks"]}
    output["paired_latest_minus_parent"]={key:value.get("delta_bce",value)for key,value in report["between_checkpoint_effect_differences"].items()}
    output["prespecified_test_case_summaries"]=[{key:case.get(key)for key in (
        "case_id","category","concept","paired_case_summary")}for case in cases["cases"]if case["fold"]=="test"]
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path("artifacts/encoder_chemistry_20260930"))
    args=parser.parse_args();root=args.root;output=root/"input_effects_analysis";output.mkdir(exist_ok=True)
    with np.load(root/"panel/inputs.npz",allow_pickle=False)as saved:panel=dict(saved)
    with np.load(root/"panel/labels.npz",allow_pickle=False)as saved:label_data=dict(saved)
    np.testing.assert_array_equal(panel["rows"],label_data["rows"])
    labels=np.concatenate((label_data["maccs"],label_data["binary"]),axis=1).astype(np.float64)
    names=[f"maccs_{i:03d}"for i in range(1,167)]+label_data["binary_names"].tolist()
    train=labels[panel["fold"]=="train"];supported=(train.sum(0)>0)&(train.sum(0)<len(train))
    groups={"trained_targets":np.flatnonzero(supported),"trained_maccs":np.flatnonzero(supported[:166]),
            "elements":np.arange(166,173),"functional_groups":np.arange(173,188),"all188":np.arange(188)}
    phases={name:{phase:load_phase(root,name,phase,panel,labels,names)for phase in PHASES}for name in ARMS}
    report={"schema":"encoder_chemistry_input_effects_analysis_v1","source_sha256":digest(__file__),
        "checkpoints":{},"between_checkpoint_effect_differences":{},
        "bootstrap":{"replicates":BOOTSTRAPS,"seed":SEED,"unit":"molecule; paired within all checkpoint/variant contrasts","head_seeds":"Three fixed readouts ensembled before resampling; seeds are not independent molecules"},
        "definitions":{"probability":"Arithmetic mean of sigmoid(logits) across three saved attentive readouts",
            "signed_margin":"Mean seed logit multiplied by (2*float(label)-1); larger means greater correct-label margin",
            "primary_targets":names if supported.all() else [names[i]for i in groups["trained_targets"]],
            "excluded_train_constant_targets":[names[i]for i in np.flatnonzero(~supported)],
            "primary_cohort":"Only supported paired molecules for each intervention; unsupported unchanged rows excluded",
            "cosine":"Cosine of mean-readout-logit vectors, not native embedding cosine",
            "bce":"Unweighted binary cross entropy of ensemble probabilities; readouts were class-balanced during fitting, so absolute calibration is not intrinsic encoder chemistry",
            "interaction":"f(blocksB,finalB)-f(blocksB,finalA)-f(blocksA,finalB)+f(blocksA,finalA)",
            "rare_losses":"Point estimates only, no significance claim; context groups with fewer than5 molecules explicitly marked"},
        "limitations":["These endpoints are three frozen token-attentive chemical readouts, not intrinsic chemical logits.",
            "Only saved logits are available for repeats; native embedding invariance cannot be recovered from these artifacts.",
            "Synthetic peak/mass interventions can be out of distribution. Stability alone does not establish semantic quality.",
            "Molecular labels are unchanged across repeat acquisitions and perturbations; theoretical neutral-loss matches are not observed fragments.",
            "Intervals are pointwise paired molecule bootstrap intervals, without familywise multiple-comparison claims.",
            "Same-molecule observed repeats preserve full identity/instrument/adduct but also change the physical spectrum; metadata-only hybrids do not recreate the alternate acquisition."],
        "artifact_hashes":{name:{phase:{"predictions_sha256":item["manifest"]["predictions_sha256"],
            "manifest_sha256":digest(root/name/"causal"/f"input_{phase}"/"manifest.json")}for phase,item in data.items()}for name,data in phases.items()}}
    all_rows={}
    for name in ARMS:
        report["checkpoints"][name],all_rows[name]=analyze_arm(name,phases[name],panel,labels,groups,names)
    for variant in all_rows[ARMS[0]]:
        left,right=all_rows[ARMS[0]][variant],all_rows[ARMS[1]][variant]
        np.testing.assert_array_equal(left["support"],right["support"])
        cohort=left["support"]
        report["between_checkpoint_effect_differences"][variant]={metric:summarize((right[metric]-left[metric])[cohort])
            for metric in left if metric!="support"}
    cases=case_summaries(root,phases,panel,labels,names,groups)
    write_json(output/"prespecified_cases.json",cases)
    report["figures"]=figures(output,report,cases)
    write_json(output/"results.json",report)
    write_json(output/"summary.json",concise_summary(report,cases))
    np.savez_compressed(output/"paired_row_effects.npz",**{name+"/"+variant+"/"+metric:value
        for name,variants in all_rows.items()for variant,metrics in variants.items()for metric,value in metrics.items()})
    print(json.dumps({"output":str(output),"trained_targets":len(groups["trained_targets"]),"figures":report["figures"]}),flush=True)


if __name__=="__main__":main()
