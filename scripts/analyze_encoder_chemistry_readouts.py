"""Paired, tie-correct chemistry readout comparisons on a pinned molecule panel.

Fetch only completed small prediction artifacts when explicitly requested. This
analysis never fits or changes a readout. Balanced-BCE-trained sigmoid outputs
are uncalibrated scores, not molecular prevalence probabilities.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np

SEEDS = (0, 1, 2)
GROUPS = {"all": "", "MACCS": "maccs_", "elements": "elem_", "functional_groups": "fg_"}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def weighted_auroc(scores: np.ndarray, labels: np.ndarray,
                   weights: np.ndarray | None = None) -> np.ndarray:
    """Exact weighted Mann–Whitney AUROC, giving tied scores half credit.

    Weights may be [N] or [replicates,N]. Integer multinomial weights give the
    same result as explicit duplicated-molecule bootstrap samples. A replicate
    with no positive or no negative weight returns NaN, never chance AUROC.
    """
    scores, labels = np.asarray(scores), np.asarray(labels)
    if scores.ndim != 1 or labels.shape != scores.shape or not np.isin(labels, (0, 1)).all():
        raise ValueError("Expected aligned one-dimensional scores and binary labels")
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite prediction score")
    weights = np.ones((1, len(scores)), np.float64) if weights is None else np.atleast_2d(weights)
    if weights.shape[1] != len(scores) or np.any(weights < 0) or not np.isfinite(weights).all():
        raise ValueError("Weights must be finite, nonnegative and aligned with molecules")
    if not len(scores):
        return np.full(len(weights), np.nan)
    order = np.argsort(scores, kind="stable")
    sorted_scores, sorted_labels = scores[order], labels[order]
    starts = np.r_[0, 1 + np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1])]
    sampled = weights[:, order].astype(np.float64)
    positive = np.add.reduceat(sampled * sorted_labels, starts, axis=1)
    negative = np.add.reduceat(sampled * (1 - sorted_labels), starts, axis=1)
    numerator = (positive * (np.cumsum(negative, axis=1) - .5 * negative)).sum(1)
    denominator = positive.sum(1) * negative.sum(1)
    return np.divide(numerator, denominator, out=np.full(len(weights), np.nan), where=denominator > 0)


def auc_matrix(scores: np.ndarray, labels: np.ndarray, weights: np.ndarray | None = None,
               support: np.ndarray | None = None) -> np.ndarray:
    if scores.shape != labels.shape or scores.ndim != 2:
        raise ValueError("Predictions and labels must be aligned molecule-by-target arrays")
    support = np.ones(scores.shape[1], bool) if support is None else np.asarray(support, bool)
    count = 1 if weights is None else len(np.atleast_2d(weights))
    result = np.full((count, scores.shape[1]), np.nan)
    for column in np.flatnonzero(support):
        result[:, column] = weighted_auroc(scores[:, column], labels[:, column], weights)
    return result


def fixed_macro(values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Keep the target set fixed; an undefined target invalidates that replicate."""
    selected = values[..., targets]
    if selected.shape[-1] == 0:
        return np.full(values.shape[:-1], np.nan)
    return selected.mean(axis=-1)


def interval(values: np.ndarray, point: float) -> dict:
    finite = np.isfinite(values)
    return {"estimate": float(point), "ci95": np.quantile(values[finite], [.025, .975]).tolist() if finite.any() else None,
            "valid_bootstrap_replicates": int(finite.sum()),
            "undefined_bootstrap_replicates": int((~finite).sum()),
            "interval_type": "paired molecule percentile bootstrap; conditional on defined fixed-target statistic"}


def balanced_score_losses(logits: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    """Balanced BCE/Brier on sigmoid scores; no probability-calibration claim."""
    logits = np.asarray(logits, np.float64)
    scores = np.exp(-np.logaddexp(0, -logits))
    positive, negative = labels.sum(0), (1 - labels).sum(0)
    result = {}
    for name, positive_loss, negative_loss in (
        ("balanced_bce", np.logaddexp(0, -logits), np.logaddexp(0, logits)),
        ("balanced_brier", (1 - scores) ** 2, scores ** 2),
    ):
        pos = np.divide((positive_loss * labels).sum(0), positive,
                        out=np.full(logits.shape[1], np.nan), where=positive > 0)
        neg = np.divide((negative_loss * (1 - labels)).sum(0), negative,
                        out=np.full(logits.shape[1], np.nan), where=negative > 0)
        result[name] = .5 * (pos + neg)
    return result


def fetch_readouts(local: Path, remote_root: str, arms: list[str], depths: list[int],
                   volume_name: str = "spectra-massspecgym-maccs-probe") -> dict:
    """Download completed small files, refusing to overwrite differing content."""
    import modal
    volume = modal.Volume.from_name(volume_name)
    combinations = [(arm, depth, kind, seed) for arm in arms for depth in depths
                    for kind in ("mean", "attention") for seed in SEEDS]

    def download(combination):
        arm, depth, kind, seed = combination
        relative = Path(arm) / "readouts" / f"depth{depth}" / kind / f"seed{seed}"
        remote = str(Path(remote_root) / relative)
        content = b"".join(volume.read_file(remote + "/trainedprobe.json"))
        metadata = json.loads(content)
        if not metadata.get("complete"):
            raise ValueError(f"Incomplete readout: {remote}")
        files = {"trainedprobe.json": content}
        for name in ("test_predictions.npz", "val_predictions.npz"):
            files[name] = b"".join(volume.read_file(remote + "/" + name))
        records = {}
        for name, content in files.items():
            path = local / relative / name
            if path.exists():
                if path.read_bytes() != content:
                    raise ValueError(f"Existing local artifact differs; refusing overwrite: {path}")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(path.suffix + ".partial")
                temporary.write_bytes(content)
                temporary.replace(path)
            records[str(path)] = hashlib.sha256(content).hexdigest()
        return records

    with ThreadPoolExecutor(max_workers=4) as executor:
        gathered = list(executor.map(download, combinations))
    return {key: value for records in gathered for key, value in records.items()}


def load_predictions(root: Path, arms: list[str], depths: list[int], panel: dict,
                     labels: np.ndarray, names: list[str], panel_hashes: dict) -> tuple[dict, dict]:
    predictions, provenance = {}, {}
    test_rows = np.flatnonzero(panel["fold"] == "test")
    for arm in arms:
        for depth in depths:
            for kind in ("mean", "attention"):
                seed_logits = []
                for seed in SEEDS:
                    source = root / arm / "readouts" / f"depth{depth}" / kind / f"seed{seed}"
                    metadata = json.loads((source / "trainedprobe.json").read_text())
                    contract = metadata["contract"]
                    if (not metadata["complete"] or metadata["kind"] != kind or metadata["seed"] != seed
                            or metadata["target_names"] != names or contract["depth"] != depth
                            or contract["encoder"] != arm or contract["token_scope"] != "physical_peaks_only"
                            or metadata["training"]["test_used_for_selection"]):
                        raise ValueError(f"Readout contract differs: {source}")
                    parity = contract["cache_final_parity"]
                    if any(not parity[key]["finite"] or not parity[key]["bitwise_equal"]
                           for key in ("cache_vs_pilot_trace", "cache_vs_pilot_native")):
                        raise ValueError(f"Readout uses a cache without exact native/pilot parity: {source}")
                    for filename, checksum in panel_hashes.items():
                        if contract["panel_sha256"][filename] != checksum:
                            raise ValueError(f"Panel source mismatch for {source}: {filename}")
                    for fold in ("test", "val"):
                        rows = np.flatnonzero(panel["fold"] == fold)
                        with np.load(source / f"{fold}_predictions.npz", allow_pickle=False) as values:
                            np.testing.assert_array_equal(values["indices"], rows)
                            np.testing.assert_array_equal(values["molecule_ids"], panel["molecule"][rows])
                            np.testing.assert_array_equal(values["labels"], labels[rows])
                            if not np.isfinite(values["logits"]).all():
                                raise ValueError(f"Nonfinite logits: {source}")
                            if fold == "test":
                                seed_logits.append(values["logits"].astype(np.float64))
                    provenance[str(source)] = {"selection": metadata["selection"],
                        "contract": contract, "parameter_count": metadata["parameter_count"],
                        "files": {name: digest(source / name) for name in (
                            "trainedprobe.json", "test_predictions.npz", "val_predictions.npz")}}
                predictions[(arm, depth, kind)] = np.stack(seed_logits)
    if len(np.unique(panel["molecule"][test_rows])) != len(test_rows):
        raise ValueError("Bootstrap requires one primary spectrum per independent molecule")
    return predictions, provenance


def matched_margin_metrics(scores: np.ndarray, labels: np.ndarray, names: list[str],
                           test_rows: np.ndarray, controls: dict,
                           support: np.ndarray) -> dict:
    """Descriptive strict-donor rank accuracy; ties receive half credit.

    Donors can recur and reciprocal comparisons can duplicate edges, so no
    independent-edge binomial interval or unclustered significance test is used.
    """
    position = {int(row): i for i, row in enumerate(test_rows)}
    records = []
    for j, name in enumerate(controls["binary_names"]):
        column = names.index(str(name))
        candidate = controls["opposite_concept_donor"][test_rows, j]
        anchor = np.flatnonzero(candidate >= 0)
        donor = np.asarray([position[int(v)] for v in candidate[anchor]], np.int64)
        if not len(anchor):
            records.append({"name": str(name), "directed_pairs": 0, "accuracy": None})
            continue
        if np.any(labels[anchor, column] == labels[donor, column]):
            raise ValueError("Matched opposite-concept labels do not differ")
        signed = (scores[anchor, column] - scores[donor, column]) * (2 * labels[anchor, column].astype(float) - 1)
        correct = (signed > 0).astype(float) + .5 * (signed == 0)
        positive = labels[anchor, column] > 0
        balanced = .5 * (correct[positive].mean() + correct[~positive].mean()) if positive.any() and (~positive).any() else np.nan
        records.append({"name": str(name), "global_target_supported": bool(support[column]),
            "directed_pairs": len(anchor), "unique_undirected_pairs": len({tuple(sorted((int(a), int(b)))) for a, b in zip(anchor, donor, strict=True)}),
            "positive_anchors": int(positive.sum()), "negative_anchors": int((~positive).sum()),
            "unique_donors": len(np.unique(donor)), "accuracy": float(correct.mean()),
            "balanced_anchor_accuracy": balanced, "ties": int((signed == 0).sum())})
    return {"interpretation": "Correct opposite-label score ordering on fixed nuisance-matched donors; descriptive and observational, not isolated molecular-concept causality. Donor reuse prevents independent-pair intervals.",
            "per_concept": records}


def analyze(root: Path, panel_path: Path, arms: list[str], depths: list[int],
            *, bootstrap: int = 1000, seed: int = 20260930) -> dict:
    if len(arms) != 2 or bootstrap < 1000:
        raise ValueError("Exactly two checkpoint arms and at least 1000 paired resamples are required")
    with np.load(panel_path / "inputs.npz") as saved:
        panel = {key: saved[key] for key in saved.files}
    with np.load(panel_path / "labels.npz") as saved:
        np.testing.assert_array_equal(saved["rows"], panel["rows"])
        labels = np.concatenate([saved["maccs"], saved["binary"]], axis=1)
        names = [f"maccs_{i:03d}" for i in range(1, 167)] + saved["binary_names"].tolist()
    with np.load(panel_path / "matched_controls.npz") as saved:
        controls = {key: saved[key] for key in saved.files}
        np.testing.assert_array_equal(controls["rows"], panel["rows"])
    panel_hashes = {filename: digest(panel_path / filename) for filename in (
        "inputs.npz", "labels.npz", "manifest.json", "matched_controls.npz")}
    predictions, provenance = load_predictions(root, arms, depths, panel, labels, names, panel_hashes)
    train = panel["fold"] == "train"
    test_rows = np.flatnonzero(panel["fold"] == "test")
    y = labels[test_rows]
    trpositive, trnegative = labels[train].sum(0), (1 - labels[train]).sum(0)
    raw_support = (trpositive >= 5) & (trnegative >= 5) & (y.sum(0) >= 5) & ((1-y).sum(0) >= 5)
    report = {"schema": "encoder_chemistry_readout_comparison_v1", "arms": arms, "depths": depths,
        "seeds": list(SEEDS), "bootstrap_replicates": bootstrap, "bootstrap_seed": seed,
        "primary_endpoint": "Macro AUROC on the fixed shared targets with >=5 positives AND negatives in train AND evaluated test cohort; exact half-credit for ties.",
        "ensemble_definition": "Arithmetic mean of the three seed logits, then compute metrics. This differs from averaging the three seed metrics. Neither selects a test-best seed.",
        "loss_interpretation": "Balanced BCE and balanced Brier use sigmoid-transformed scores from class-balanced training; no claim of calibrated chemical prevalence probabilities. Reported as descriptive secondary metrics.",
        "bootstrap_policy": "Identical multinomial molecule counts for every arm/readout/seed in each cohort. Targets stay fixed. If any fixed target lacks a class in a resample, that group statistic is undefined and the entire replicate is omitted for that group; omitted counts are reported. Per-target intervals omit only their own undefined replicates. No bootstrap p-values.",
        "multiplicity": "Intervals are descriptive and unadjusted. Per-concept and subgroup contrasts are exploratory; no familywise significant-discovery claim or bootstrap p-values.",
        "scope_of_uncertainty": "Molecule sampling conditional on these fitted readouts, three fixed probe seeds, and two fixed pretrained checkpoints. Does not include pretraining-run variation or model-selection uncertainty.",
        "scaffold_policy": "Unseen means a nonempty Murcko scaffold absent from selected training panel, not all encoder pretraining. Subgroup target support is fixed separately using train and that subgroup; group target sets can differ from full test.",
        "panel_hashes": panel_hashes, "provenance": provenance, "cohorts": {},
        "matched_controls": {}}
    boot_saved = {}
    for scope, subset in (("test", np.ones(len(y), bool)),
                           ("unseen_nonempty_train_scaffold", panel["unseen_nonempty_train_scaffold"][test_rows])):
        truth = y[subset]
        positive, negative = truth.sum(0), (1-truth).sum(0)
        supported = raw_support & (positive >= 5) & (negative >= 5)
        rng = np.random.default_rng(seed + (scope != "test"))
        counts = rng.multinomial(len(truth), np.full(len(truth), 1 / len(truth)), size=bootstrap).astype(np.int16)
        groups = {g: supported & np.asarray([n.startswith(prefix) for n in names]) for g, prefix in GROUPS.items()}
        records = {"molecules": len(truth), "target_names": names,
            "target_support": [{"name": n, "train_positive": int(trpositive[i]),
               "train_negative": int(trnegative[i]), "positive": int(positive[i]), "negative": int(negative[i]),
               "supported": bool(supported[i]), "few_test_positives_or_negatives": bool(min(positive[i],negative[i]) < 20)} for i,n in enumerate(names)],
            "groups": {g: {"targets": int(mask.sum()), "names": [n for n,k in zip(names,mask,strict=True) if k]} for g,mask in groups.items()},
            "models": {}, "contrasts": {}}
        boot, point = {}, {}
        for key, logits in predictions.items():
            arm, depth, kind = key
            name = f"{arm}/depth{depth}/{kind}"
            values = logits[:, subset]
            seed_point = np.stack([auc_matrix(value,truth,support=supported)[0] for value in values])
            seed_boot = np.stack([auc_matrix(value,truth,counts,supported) for value in values])
            ensemble = values.mean(0)
            point[name] = {"ensemble": auc_matrix(ensemble,truth,support=supported)[0],
                           "seed_mean_metric": seed_point.mean(0)}
            boot[name] = {"ensemble": auc_matrix(ensemble,truth,counts,supported),
                          "seed_mean_metric": seed_boot.mean(0)}
            losses = balanced_score_losses(ensemble,truth)
            seed_losses = [balanced_score_losses(value,truth) for value in values]
            model = {"seed_auroc": {str(s): seed_point[i] for i,s in enumerate(SEEDS)},
                "per_target": [{"name": n, "supported": bool(supported[i]),
                    "ensemble_auroc": interval(boot[name]["ensemble"][:,i],point[name]["ensemble"][i]),
                    "seed_mean_auroc": interval(boot[name]["seed_mean_metric"][:,i],point[name]["seed_mean_metric"][i]),
                    **{loss: value[i] for loss,value in losses.items()}}
                    for i,n in enumerate(names)], "groups": {}}
            for group, mask in groups.items():
                model["groups"][group] = {
                    method: interval(fixed_macro(samples,mask),float(fixed_macro(point[name][method],mask)))
                    for method,samples in boot[name].items()}
                model["groups"][group].update(
                    seed_metrics=[float(fixed_macro(value,mask)) for value in seed_point],
                    ensemble_balanced_bce=float(fixed_macro(losses["balanced_bce"],mask)),
                    ensemble_balanced_brier=float(fixed_macro(losses["balanced_brier"],mask)),
                    seed_mean_balanced_bce=float(np.mean([fixed_macro(v["balanced_bce"],mask) for v in seed_losses])),
                    seed_mean_balanced_brier=float(np.mean([fixed_macro(v["balanced_brier"],mask) for v in seed_losses])))
            records["models"][name] = model
            boot_saved[f"{scope}/{name}/ensemble"] = boot[name]["ensemble"].astype(np.float32)
            boot_saved[f"{scope}/{name}/seed_mean_metric"] = boot[name]["seed_mean_metric"].astype(np.float32)
            if scope == "test":
                report["matched_controls"][name] = matched_margin_metrics(ensemble,truth,names,test_rows,controls,supported)
            print(f"Analyzed {scope} {name}",flush=True)
        comparisons = [(f"checkpoint_depth{depth}_{kind}", f"{arms[1]}/depth{depth}/{kind}",f"{arms[0]}/depth{depth}/{kind}")
                       for depth in depths for kind in ("mean","attention")]
        comparisons += [(f"attention_minus_mean_{arm}_depth{depth}",f"{arm}/depth{depth}/attention",f"{arm}/depth{depth}/mean")
                        for arm in arms for depth in depths]
        for name,left,right in comparisons:
            record = {"left":left,"right":right,"direction":"left minus right; positive favors left AUROC","groups":{},"per_target":[]}
            for group,mask in groups.items():
                record["groups"][group] = {method:interval(
                    fixed_macro(boot[left][method]-boot[right][method],mask),
                    float(fixed_macro(point[left][method]-point[right][method],mask)))
                    for method in ("ensemble","seed_mean_metric")}
            for i,n in enumerate(names):
                record["per_target"].append({"name":n,"supported":bool(supported[i]),
                    **{method:interval(boot[left][method][:,i]-boot[right][method][:,i],
                        point[left][method][i]-point[right][method][i])for method in ("ensemble","seed_mean_metric")}})
            records["contrasts"][name] = record
        report["cohorts"][scope] = records
    output = root / "readout_comparison"
    output.mkdir(parents=True,exist_ok=True)
    report["analysis_source_sha256"] = digest(Path(__file__))
    (output/"comparison.json").write_text(json.dumps(json_value(report),indent=2,allow_nan=False)+"\n")
    np.savez_compressed(output/"bootstrap_auroc.npz",**boot_saved)
    make_figures(report,output)
    return report


def make_figures(report:dict,output:Path)->None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    colors={report["arms"][0]:"#697b8f",report["arms"][1]:"#d37a40"}
    test=report["cohorts"]["test"]
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout="constrained")
    for ax,group in zip(axes,("MACCS","elements","functional_groups"),strict=True):
        keys=[]
        for depth in report["depths"]:
            for kind in ("mean","attention"):
                keys.append((depth,kind))
        for offset,arm in zip((-.12,.12),report["arms"],strict=True):
            for i,(depth,kind) in enumerate(keys):
                value=test["models"][f"{arm}/depth{depth}/{kind}"]["groups"][group]["ensemble"]
                lo,hi=value["ci95"];point=value["estimate"]
                ax.plot([i+offset,i+offset],[lo,hi],color=colors[arm],lw=1.6)
                ax.scatter(i+offset,point,color=colors[arm],s=35,label=arm if i==0 else None)
        ax.set_xticks(range(len(keys)),[f"L{d}\n{k}" for d,k in keys]);ax.set_title(f"{group} · {test['groups'][group]['targets']} targets")
        ax.set_ylabel("Ensemble AUROC");ax.axhline(.5,color="#dddddd",lw=1);ax.set_ylim(.45,1)
    axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle("Native token readouts · fixed molecule test panel",fontsize=14)
    for extension in ("png","pdf"):fig.savefig(output/f"readout_auroc.{extension}",dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,5),layout="constrained")
    contrasts=list(test["contrasts"])
    for ax,group in zip(axes,("MACCS","elements","functional_groups"),strict=True):
        for i,name in enumerate(contrasts):
            record=test["contrasts"][name]["groups"][group]["ensemble"]
            lo,hi=record["ci95"];estimate=record["estimate"]
            color="#286f69" if estimate>=0 else "#a35847"
            ax.plot([lo,hi],[i,i],color=color,lw=1.5);ax.scatter(estimate,i,color=color,s=25)
        ax.axvline(0,color="#888888",lw=.8);ax.set_title(group);ax.set_xlabel("Paired AUROC difference")
        ax.set_yticks(range(len(contrasts)),[name.replace("checkpoint_","periodic − parent ").replace("attention_minus_mean_","attention − mean ") for name in contrasts] if ax is axes[0] else [])
        ax.invert_yaxis()
    fig.suptitle("Paired molecule bootstrap · 95% percentile intervals",fontsize=13)
    for extension in ("png","pdf"):fig.savefig(output/f"paired_contrasts.{extension}",dpi=180)
    plt.close(fig)


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path("artifacts/encoder_chemistry_20260930"))
    parser.add_argument("--panel",type=Path,default=Path("artifacts/encoder_chemistry_20260930/panel"))
    parser.add_argument("--arms",nargs=2,default=["parent_1190k","periodic_1209k"])
    parser.add_argument("--depths",nargs="+",type=int,default=[27,19])
    parser.add_argument("--bootstrap",type=int,default=1000)
    parser.add_argument("--seed",type=int,default=20260930)
    parser.add_argument("--fetch",action="store_true")
    parser.add_argument("--fetch-only",action="store_true")
    parser.add_argument("--remote-root",default="/encoder_chemistry_20260930")
    args=parser.parse_args()
    if args.fetch or args.fetch_only:
        fetched=fetch_readouts(args.root,args.remote_root,args.arms,args.depths)
        print(f"Verified {len(fetched)} small remote artifacts",flush=True)
    if not args.fetch_only:
        analyze(args.root,args.panel,args.arms,args.depths,bootstrap=args.bootstrap,seed=args.seed)


if __name__=="__main__":main()
