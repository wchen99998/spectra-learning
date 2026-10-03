"""Compare a pinned diffusion encoder with the two qualified JEPA checkpoints.

This extension reuses the original molecule panel, exact native-cache gates,
readout metric definitions, and train/validation-only SAE discovery protocol.
It fits no models and writes no legacy study or presentation outputs.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np

from scripts.analyze_encoder_chemistry_readouts import (
    SEEDS, auc_matrix, balanced_score_losses, digest, fetch_readouts,
    fixed_macro, interval, load_predictions, matched_margin_metrics, weighted_auroc,
)
from scripts.analyze_encoder_chemistry_sae import (
    CONCEPTS, analyze_retrieval, confirm, discover, load_panel, write_json,
)
from scripts.analyze_encoder_chemistry_sae_stability import analyze as analyze_stability
from scripts.analyze_encoder_chemistry_matched_geometry import candidate_banks, retrieve

BASELINES = ("parent_1190k", "periodic_1209k")
CONFIGURATIONS = ((32, 0), (32, 1), (64, 0))
GROUPS = {"MACCS": "maccs_", "functional_groups": "fg_", "elements": "elem_"}
DEFAULT_ROOT = Path("artifacts/encoder_chemistry_20260930")


def configure_plots() -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    for style in ("Regular","Bold","Italic","BoldItalic"):
        font_manager.fontManager.addfont(str(Path(__file__).resolve().parents[1]/"presentations/shared/assets"/f"LiberationSans-{style}.ttf"))
    matplotlib.rcParams.update({"font.family":"Liberation Sans","pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none"})


def summarize_readouts(predictions: dict, labels: np.ndarray, train_labels: np.ndarray,
                       names: list[str], arm: str, *, bootstrap: int = 1000,
                       seed: int = 20260930) -> tuple[dict, dict]:
    """Three-arm paired comparison, with shared fixed targets and molecule counts."""
    if bootstrap < 1000:
        raise ValueError("At least1000 paired molecule bootstrap resamples are required")
    arms = (*BASELINES, arm)
    expected = {(name, 27, kind) for name in arms for kind in ("mean", "attention")}
    if set(predictions) != expected or arm in BASELINES:
        raise ValueError("Expected one new diffusion arm and both original final-state readouts")
    n = len(labels)
    train_positive, positive = train_labels.sum(0), labels.sum(0)
    support = ((train_positive >= 5) & (len(train_labels)-train_positive >= 5)
               & (positive >= 5) & (n-positive >= 5))
    groups = {group: support & np.asarray([name.startswith(prefix) for name in names])
              for group, prefix in GROUPS.items()}
    counts = np.random.default_rng(seed).multinomial(n, np.full(n, 1/n), size=bootstrap).astype(np.int16)
    report = {
        "schema": "diffusion_final_token_readout_comparison_v1", "arms": list(arms),
        "test_molecules": n, "native_state": 27, "token_scope": "valid physical peaks, excluding CLS",
        "primary_endpoint": "Final physical-token attention MACCS macro AUROC",
        "secondary_endpoints": "Mean-token linear control, functional groups, elements, and per-concept metrics",
        "seeds": list(SEEDS), "bootstrap_replicates": bootstrap, "bootstrap_seed": seed,
        "ensemble": "Mean of three seed logits before AUROC; separately report mean of three seed metrics. No test-selected seed.",
        "support_policy": "Same fixed targets for all encoders/readouts: at least5 positives and5 negatives in train AND test. Exact half-credit for tied scores.",
        "bootstrap_policy": "Shared multinomial molecule counts across all three encoders, both readouts and all seeds. A macro replicate is undefined if any fixed target loses a class; omit the entire group replicate and report its count. Targets never vary across bootstrap replicates.",
        "uncertainty_scope": "Conditional on three fixed pretrained checkpoints and fitted readouts. Descriptive unadjusted percentile intervals; no p-values, no pretraining-run uncertainty, no upstream-exposure claim.",
        "causal_scope": "Checkpoint comparison, not an isolated training-objective experiment. Training duration, initialization, optimization and native input quantization can differ; molecular labels and downstream panel/readout budgets are shared.",
        "score_losses": "Balanced BCE/Brier use sigmoid scores from class-balanced readouts; they are not calibrated prevalence probabilities.",
        "target_support": [{"name": name, "train_positive": int(train_positive[i]),
            "train_negative": int(len(train_labels)-train_positive[i]), "test_positive": int(positive[i]),
            "test_negative": int(n-positive[i]), "supported": bool(support[i]),
            "few_test_examples": bool(min(positive[i],n-positive[i]) < 20)} for i,name in enumerate(names)],
        "groups": {g: {"targets": int(mask.sum()), "names": [name for name,keep in zip(names,mask,strict=True) if keep]}
                   for g,mask in groups.items()}, "models": {}, "contrasts": {},
    }
    points, samples = {}, {}
    for (encoder, depth, kind), logits in predictions.items():
        if logits.shape != (3, *labels.shape) or not np.isfinite(logits).all():
            raise ValueError("Expected finite three-seed molecule-by-target logits")
        key = f"{encoder}/depth{depth}/{kind}"
        seed_point = np.stack([auc_matrix(value,labels,support=support)[0] for value in logits])
        seed_boot = np.stack([auc_matrix(value,labels,counts,support) for value in logits])
        ensemble = logits.mean(0)
        points[key] = {"ensemble": auc_matrix(ensemble,labels,support=support)[0],
                       "seed_mean_metric": seed_point.mean(0)}
        samples[key] = {"ensemble": auc_matrix(ensemble,labels,counts,support),
                        "seed_mean_metric": seed_boot.mean(0)}
        losses = balanced_score_losses(ensemble,labels)
        model = {"groups": {}, "per_target": []}
        for group,mask in groups.items():
            model["groups"][group] = {method: interval(fixed_macro(value,mask),float(fixed_macro(points[key][method],mask)))
                                      for method,value in samples[key].items()}
            model["groups"][group].update(seed_metrics=[float(fixed_macro(v,mask)) for v in seed_point],
                ensemble_balanced_bce=float(fixed_macro(losses["balanced_bce"],mask)),
                ensemble_balanced_brier=float(fixed_macro(losses["balanced_brier"],mask)))
        for i,name in enumerate(names):
            model["per_target"].append({"name": name, "supported": bool(support[i]),
                **{method: interval(samples[key][method][:,i],points[key][method][i]) for method in samples[key]},
                "balanced_bce": losses["balanced_bce"][i], "balanced_brier": losses["balanced_brier"][i]})
        report["models"][key] = model
        print("Compared final token readout", key, flush=True)
    contrasts = [(f"{arm}_minus_{baseline}_{kind}",f"{arm}/depth27/{kind}",f"{baseline}/depth27/{kind}")
                 for baseline in BASELINES for kind in ("attention","mean")]
    contrasts += [(f"attention_minus_mean_{encoder}",f"{encoder}/depth27/attention",f"{encoder}/depth27/mean")
                  for encoder in arms]
    for name,left,right in contrasts:
        result = {"left": left, "right": right, "direction": "left minus right; positive favors left AUROC",
                  "groups": {}, "per_target": []}
        for group,mask in groups.items():
            result["groups"][group] = {method: interval(fixed_macro(samples[left][method]-samples[right][method],mask),
                float(fixed_macro(points[left][method]-points[right][method],mask))) for method in samples[left]}
        for i,target in enumerate(names):
            result["per_target"].append({"name": target,"supported":bool(support[i]),
                **{method: interval(samples[left][method][:,i]-samples[right][method][:,i],
                                    points[left][method][i]-points[right][method][i]) for method in samples[left]}})
        report["contrasts"][name] = result
    return report, {key+"/"+method: values.astype(np.float32) for key,methods in samples.items() for method,values in methods.items()}


def compare_readouts(root: Path, arm: str, output: Path, *, bootstrap: int = 1000) -> dict:
    panel, labels, controls = load_panel(root/"panel")
    targets = np.concatenate([labels["maccs"],labels["binary"]],axis=1)
    names = [f"maccs_{i:03d}" for i in range(1,167)]+labels["binary_names"].tolist()
    hashes = {name:digest(root/"panel"/name) for name in ("inputs.npz","labels.npz","manifest.json","matched_controls.npz")}
    predictions, provenance = load_predictions(root,[*BASELINES,arm],[27],panel,targets,names,hashes)
    test_rows = np.flatnonzero(panel["fold"]=="test")
    report, sampled = summarize_readouts(predictions,targets[test_rows],targets[panel["fold"]=="train"],names,arm,bootstrap=bootstrap)
    support = np.asarray([r["supported"] for r in report["target_support"]])
    report.update(panel_sha256=hashes,provenance=provenance,analysis_source_sha256=digest(Path(__file__)),
        matched_nuisance_ordering={f"{name}/depth27/{kind}": matched_margin_metrics(values.mean(0),targets[test_rows],names,test_rows,controls,support)
            for (name,_,kind),values in predictions.items()})
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/"readout_comparison.json",report)
    np.savez_compressed(output/"readout_bootstrap.npz",**sampled)
    plot_readouts(report,arm,output)
    return report


def plot_readouts(report: dict, arm: str, output: Path) -> None:
    configure_plots()
    import matplotlib.pyplot as plt
    colors=("#345D7A","#B25D3A","#187E74")
    fig,axes=plt.subplots(1,2,figsize=(10,4.3),layout="constrained")
    for axis,group in zip(axes,("MACCS","functional_groups"),strict=True):
        for i,encoder in enumerate((*BASELINES,arm)):
            rows=[report["models"][f"{encoder}/depth27/{kind}"]["groups"][group]["ensemble"] for kind in ("mean","attention")]
            values=np.array([r["estimate"] for r in rows]);ci=np.array([r["ci95"] for r in rows])
            axis.errorbar(np.arange(2)+(i-1)*.16,values,yerr=np.maximum(0,np.stack([values-ci[:,0],ci[:,1]-values])),
                          fmt="o",capsize=3,color=colors[i],label=encoder.replace("_"," "))
        axis.set(xticks=[0,1],xticklabels=["Mean control","Token attention"],ylabel="Final physical-token macro AUROC",title=group.replace("_"," "))
        axis.spines[["top","right"]].set_visible(False)
    axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle("Pinned diffusion extension · identical molecule panel and readout protocol")
    for extension in ("png","pdf"):fig.savefig(output/f"final_token_auroc.{extension}",dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.1),layout="constrained")
    keys=[f"{arm}_minus_{baseline}_{kind}" for baseline in BASELINES for kind in ("attention","mean")]
    labels=[f"vs {baseline.replace('_',' ')} · {kind}" for baseline in BASELINES for kind in ("attention","mean")]
    for axis,group in zip(axes,("MACCS","functional_groups"),strict=True):
        rows=[report["contrasts"][key]["groups"][group]["ensemble"] for key in keys]
        values=np.array([r["estimate"] for r in rows]);ci=np.array([r["ci95"] for r in rows])
        axis.errorbar(values,np.arange(len(keys)),xerr=np.maximum(0,np.stack([values-ci[:,0],ci[:,1]-values])),fmt="o",capsize=3,color=colors[2])
        axis.set(yticks=range(len(keys)),yticklabels=labels,xlabel="Diffusion minus comparison AUROC",title=group.replace("_"," "))
        axis.invert_yaxis();axis.axvline(0,color=".6",lw=.7);axis.spines[["top","right"]].set_visible(False)
    fig.suptitle("Paired molecule differences · 95% percentile intervals")
    for extension in ("png","pdf"):fig.savefig(output/f"paired_readout_differences.{extension}",dpi=180)
    plt.close(fig)
    plot_readout_slides(report,arm,output)


def plot_readout_slides(report: dict, arm: str, output: Path) -> None:
    """Small vector panels with explicit legends supplied by the report/deck."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    colors=("#345D7A","#B25D3A","#187E74")
    with plt.rc_context({"font.size":8,"axes.titlesize":9,"axes.labelsize":8,"legend.fontsize":7.5}):
        fig,axes=plt.subplots(1,2,figsize=(143/25.4,47.5/25.4),layout="constrained")
        for axis,group in zip(axes,("MACCS","functional_groups"),strict=True):
            for i,(encoder,label) in enumerate(zip((*BASELINES,arm),("Parent 1190k","Latent 1209k",arm.replace("diffusion_","Diffusion ")),strict=True)):
                rows=[report["models"][f"{encoder}/depth27/{kind}"]["groups"][group]["ensemble"] for kind in ("mean","attention")]
                y=np.array([r["estimate"] for r in rows]);ci=np.array([r["ci95"] for r in rows]).T
                axis.errorbar(np.arange(2)+(i-1)*.16,y,yerr=np.maximum(0,np.stack([y-ci[0],ci[1]-y])),
                    fmt="o",ms=3.5,capsize=2,color=colors[i],label=label)
            axis.set(xticks=[0,1],xticklabels=["Mean","Token attention"],title="MACCS" if group=="MACCS" else "Functional groups",ylabel="AUROC")
            axis.yaxis.set_major_locator(MaxNLocator(4));axis.spines[["top","right"]].set_visible(False)
        fig.legend(*axes[0].get_legend_handles_labels(),loc="outside upper center",ncol=3,frameon=False)
        fig.savefig(output/"final_token_auroc_slide.pdf");plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(143/25.4,47.5/25.4),layout="constrained",sharey=True)
        keys=[f"{arm}_minus_{baseline}_{kind}" for baseline in BASELINES for kind in ("attention","mean")]
        labels=[f"{base} · {kind}" for base in ("Parent","Latent") for kind in ("attention","mean")]
        for axis,group in zip(axes,("MACCS","functional_groups"),strict=True):
            rows=[report["contrasts"][key]["groups"][group]["ensemble"] for key in keys]
            y=np.array([r["estimate"] for r in rows]);ci=np.array([r["ci95"] for r in rows]).T
            axis.errorbar(y,np.arange(4),xerr=np.maximum(0,np.stack([y-ci[0],ci[1]-y])),fmt="o",ms=3.5,capsize=2,color=colors[2])
            axis.set(yticks=range(4),yticklabels=labels,xlabel="Diffusion − comparison AUROC",title="MACCS" if group=="MACCS" else "Functional groups")
            axis.xaxis.set_major_locator(MaxNLocator(4));axis.axvline(0,color=".6",lw=.7);axis.spines[["top","right"]].set_visible(False)
        axes[0].invert_yaxis()
        fig.savefig(output/"paired_readout_differences_slide.pdf");plt.close(fig)


def stage_immutable_file(source: Path, destination: Path) -> None:
    """New-arm analysis view only; never overwrite differing prior artifacts."""
    if destination.exists():
        if digest(source)!=digest(destination):
            raise ValueError(f"Analysis view differs from source; use a new pinned arm: {destination}")
        return
    destination.parent.mkdir(parents=True,exist_ok=True)
    if source.suffix==".json":shutil.copyfile(source,destination)
    else:os.link(source,destination)


def validate_sae_contract(metadata: dict, arm: str, k: int, seed: int, panel_hashes: dict) -> None:
    contract=metadata["contract"]
    if not metadata["complete"] or (contract["encoder"],contract["depth"],contract["features"],contract["k"],contract["seed"])!=(arm,19,8192,k,seed):
        raise ValueError("SAE dictionary differs from fixed diffusion comparison budget")
    for key in ("native","trace"):
        if not contract["cache_final_parity"][key]["finite"] or not contract["cache_final_parity"][key]["bitwise_equal"]:
            raise ValueError("SAE source is not bitwise-qualified native extraction")
    if contract["panel_sha256"] != panel_hashes:
        raise ValueError("SAE training/export panel differs from the current chemistry panel")


def analyze_new_sae(root: Path, arm: str, output: Path) -> dict:
    """Run the unchanged primary selector in an extension-only dictionary view."""
    from scripts.encoder_chemistry_sae import load_sae
    panel_hashes = {name:digest(root/"panel"/name) for name in ("inputs.npz","labels.npz","manifest.json","matched_controls.npz")}
    records=[]
    for k,seed in CONFIGURATIONS:
        relative=Path("sae")/"depth19"/f"k{k}"/f"seed{seed}"
        source=root/arm/relative;destination=output/relative
        metadata=json.loads((source/"trained_sae.json").read_text())
        contract=metadata["contract"]
        validate_sae_contract(metadata,arm,k,seed,panel_hashes)
        filenames=["trained_sae.json","params.npz","molecule_max.npy"]
        if (k,seed)==(32,0):filenames += ["random_dictionary_molecule_max.npy","dense_channel_molecule_max.npy"]
        for name in filenames:
            if name!="trained_sae.json":
                checksum=metadata["weights_sha256"] if name=="params.npz" else metadata["artifact_files"][name]["sha256"]
                if digest(source/name)!=checksum:raise ValueError(f"Incomplete or changed SAE source: {source/name}")
            stage_immutable_file(source/name,destination/name)
        params, _ = load_sae(destination)
        del params
        records.append({"encoder":arm,"k":k,"seed":seed,"quality":metadata["quality"],
            "dictionary_redundancy":metadata["dictionary_redundancy"],"selected_epoch":metadata["selected_epoch"],
            "trained_sae_sha256":digest(source/"trained_sae.json"),"weights_sha256":metadata["weights_sha256"],
            "contract":contract})
    primary=output/"sae/depth19/k32/seed0"
    if not(primary/"chemical_features.json").exists():discover(primary,root/"panel")
    confirmation=confirm(primary,root/"panel")
    geometry=analyze_retrieval(primary,root/"panel")
    stability={f"k{k}_seed{seed}":analyze_stability(primary,output/"sae/depth19"/f"k{k}"/f"seed{seed}",root/"panel")
               for k,seed in CONFIGURATIONS if (k,seed)!=(32,0)}
    legacy=json.loads((root/"sae_reconstruction_comparison.json").read_text())
    all_quality=[*legacy["results"],*records]
    selection=json.loads((primary/"chemical_features.json").read_text())
    comparisons={}
    for encoder in (*BASELINES,arm):
        directory=primary if encoder==arm else root/encoder/"sae/depth19/k32/seed0"
        if encoder==arm:record=confirmation
        else:record=json.loads((directory/"chemical_features_test_confirmation.json").read_text())
        comparisons[encoder]={"selection_sha256":record["selection_sha256"],
            "families":{family:{concept:{"selected_features":len(rows:= [r for r in features if r["concept"]==concept]),
                "mean_test_auroc":float(np.mean([r["test"]["auroc"] for r in rows])) if rows else None,
                "records":rows} for concept in CONCEPTS} for family,features in record["families"].items()}}
    report={"schema":"diffusion_sae_comparison_v1","diffusion_arm":arm,"dictionary_budget":{"depth":19,"features":8192,"configurations":[list(c)for c in CONFIGURATIONS]},
        "quality":all_quality,"primary_selection_sha256":digest(primary/"chemical_features.json"),
        "selection_counts":selection["selection_counts"],"frozen_primary_records":selection["records"],
        "chemical_confirmation":comparisons,"control_stability":stability,"label_free_geometry":geometry,
        "paired_comparison":compare_sae_chemistry(root,arm,output),
        "matched_nuisance_geometry":compare_matched_geometry(root,arm,output),
        "reconstruction_interpretation":"FVE is relative to residuals centered at the training mean; raw relative L2 uses uncentered residual norm. Both are reported in each quality record. Different activation means and variance scales make higher FVE insufficient to claim uniformly better reconstruction across encoders.",
        "selection_policy":"Unchanged original selector: train/validation only, up to2 features each for6 fixed concepts, no test filtering. Matched8192 random dictionary and2048 dense-channel/fixed-SAE-subset controls.",
        "scope":"Associations, reconstruction and seed/k stability are distinct evidence. Feature IDs are dictionary-specific. No new SAE causal intervention is implied by this extension; raw peaks are not fragment annotations.",
        "analysis_source_sha256":digest(Path(__file__))}
    write_json(output/"sae_comparison.json",report)
    plot_sae(report,arm,output)
    return report


def compare_sae_chemistry(root: Path, arm: str, output: Path) -> dict:
    """Reuse fixed selected features and retrieval rows; no new selection or fits."""
    panel, labels, _ = load_panel(root/"panel")
    test = np.flatnonzero(panel["fold"]=="test")
    counts = np.random.default_rng(20260930).multinomial(len(test),np.full(len(test),1/len(test)),size=1000)
    directories = {name: (output if name==arm else root/name)/"sae/depth19/k32/seed0"
                   for name in (*BASELINES,arm)}
    points, samples, geometries = {}, {}, {}
    result = {"bootstrap": {"replicates":1000,"seed":20260930,"paired_unit":"test molecule"},
        "scope":"Conditional on frozen train/validation-selected features and fitted dictionaries. Mean AUROC across all selected features per concept, never test-best feature. Missing concepts remain missing; dictionary IDs are not aligned across encoders. Descriptive unadjusted intervals.",
        "feature_models":{},"feature_contrasts":{},"geometry_contrasts":{}}
    for name,directory in directories.items():
        selection = json.loads((directory/"chemical_features.json").read_text())
        confirmation = json.loads((directory/"chemical_features_test_confirmation.json").read_text())
        if confirmation["selection_sha256"] != digest(directory/"chemical_features.json"):
            raise ValueError("Test confirmation differs from frozen feature selection")
        with np.load(directory/"chemical_geometry_rows.npz",allow_pickle=False) as data:
            geometries[name] = {key:data[key] for key in data.files}
        result["feature_models"][name] = {}
        for family,records in confirmation["families"].items():
            source = "sae" if family=="sae_equal_dense_search_budget" else family
            metadata = selection["baseline_files"][source]
            path = Path(metadata["path"])
            if digest(path)!=metadata["sha256"]:
                raise ValueError("Feature activations changed after frozen selection")
            features = np.load(path,mmap_mode="r")
            concepts, per_feature = {}, []
            for record in records:
                target = labels["binary"][test,record["concept_index"]]
                boot = weighted_auroc(features[test,record["feature"]],target,counts)
                point = record["test"]["auroc"]
                per_feature.append({"feature":record["feature"],"concept":record["concept"],"auroc":interval(boot,point)})
                concepts.setdefault(record["concept"],[]).append((point,boot))
            grouped = {}
            for concept in CONCEPTS:
                selected = concepts.get(concept,[])
                grouped[concept] = {"selected_features":len(selected),"auroc":None}
                if selected:
                    key = name,family,concept
                    points[key] = float(np.mean([r[0] for r in selected]))
                    samples[key] = np.mean([r[1] for r in selected],axis=0)
                    grouped[concept]["auroc"] = interval(samples[key],points[key])
            result["feature_models"][name][family] = {"concepts":grouped,"per_feature":per_feature}
    for baseline in BASELINES:
        contrast = arm+"_minus_"+baseline
        result["feature_contrasts"][contrast] = {}
        for family in result["feature_models"][arm]:
            common = {}
            for concept in CONCEPTS:
                left,right = (arm,family,concept),(baseline,family,concept)
                if left in samples and right in samples:
                    common[concept] = interval(samples[left]-samples[right],points[left]-points[right])
            result["feature_contrasts"][contrast][family] = common
        result["geometry_contrasts"][contrast] = {}
        for representation in ("sae","random_dictionary","dense_channel","sae_equal_dense_search_budget"):
            metrics = {}
            for name in (arm,baseline):
                np.testing.assert_array_equal(geometries[name][representation+"/test_rows"],test)
            for metric in ("morgan_tanimoto","functional_group_jaccard","precursor_mass_gap_da"):
                key = representation+"/"+metric
                a,b = geometries[arm][key],geometries[baseline][key]
                valid = np.isfinite(a)&np.isfinite(b)
                difference = a[valid]-b[valid]
                weights = counts[:,valid]
                denominator = weights.sum(1)
                bootstrap = np.divide(weights@difference,denominator,out=np.full(len(counts),np.nan),where=denominator>0)
                metrics[metric] = {"paired_molecules":int(valid.sum()),
                    "difference":interval(bootstrap,float(difference.mean())) if valid.any() else None}
            result["geometry_contrasts"][contrast][representation] = metrics
    return result


def compare_matched_geometry(root: Path, arm: str, output: Path) -> dict:
    """Extend saved observational controls with identical fixed candidate banks."""
    panel, labels, _ = load_panel(root/"panel")
    test,banks = candidate_banks(panel)
    source = root/"sae_comparison/matched_geometry_rows.npz"
    with np.load(source,allow_pickle=False) as values:
        data={key:values[key] for key in values.files}
    np.testing.assert_array_equal(test,data["test_rows"])
    np.testing.assert_array_equal([len(bank) for bank in banks],data["candidate_counts"])
    common=data["common"].copy()
    selection=json.loads((output/"sae/depth19/k32/seed0/chemical_features.json").read_text())
    for representation,metadata in selection["baseline_files"].items():
        path=Path(metadata["path"])
        if digest(path)!=metadata["sha256"]:
            raise ValueError("Features changed after frozen selection")
        rows=retrieve(np.load(path,mmap_mode="r"),test,banks,panel,labels)
        common &= np.isfinite(rows["morgan_tanimoto"])
        data.update({arm+"/"+representation+"/"+key:value for key,value in rows.items()})
    n=int(common.sum())
    if n==0:raise ValueError("No common nuisance-matched molecules")
    counts=np.random.default_rng(20260930).multinomial(n,np.full(n,1/n),size=1000)
    report={"support":{"test_molecules":len(test),"common_molecules":n,
        "no_candidate":int((data["candidate_counts"]==0).sum()),
        "one_candidate":int((data["candidate_counts"]==1).sum()),
        "at_least_two_candidates":int((data["candidate_counts"]>=2).sum())},
        "policy":"Identical same-instrument/adduct/CE-presence train candidate banks; <=5 Da precursor, <=10 raw CE, <=5 processed peak-count differences; >=2 candidates. Unchanged fixed calipers, no label filtering. All three encoders and all three representations share the eligible test cohort. Restricted-support observational comparison, not a population or causal claim.",
        "bootstrap":{"replicates":1000,"seed":20260930,"paired_unit":"common eligible test molecule"},
        "baseline_rows_sha256":digest(source),"models":{},"contrasts":{}}
    for representation in ("sae","random_dictionary","dense_channel"):
        for encoder in (*BASELINES,arm):
            key=encoder+"/"+representation
            report["models"][key]={}
            for metric in ("morgan_tanimoto","functional_group_jaccard","precursor_mass_gap_da"):
                values=data[key+"/"+metric][common]
                report["models"][key][metric]=interval(counts@values/n,float(values.mean()))
        for baseline in BASELINES:
            contrast=arm+"_minus_"+baseline
            report["contrasts"].setdefault(contrast,{})[representation]={}
            for metric in ("morgan_tanimoto","functional_group_jaccard","precursor_mass_gap_da"):
                values=(data[arm+"/"+representation+"/"+metric]-data[baseline+"/"+representation+"/"+metric])[common]
                report["contrasts"][contrast][representation][metric]=interval(counts@values/n,float(values.mean()))
    data["common"]=common
    np.savez_compressed(output/"matched_geometry_rows.npz",**data)
    write_json(output/"matched_geometry_comparison.json",report)
    return report


def plot_sae(report: dict, arm: str, output: Path) -> None:
    configure_plots()
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout="constrained")
    colors=("#345D7A","#B25D3A","#187E74")
    for i,encoder in enumerate((*BASELINES,arm)):
        fit=[next(r for r in report["quality"] if(r["encoder"],r["k"],r["seed"])==(encoder,k,s)) for k,s in CONFIGURATIONS]
        axes[0].plot(np.arange(3)+(i-1)*.1,[r["quality"]["test"]["fve_vs_train_mean"] for r in fit],"o-",color=colors[i],label=encoder.replace("_"," "))
        values=report["chemical_confirmation"][encoder]["families"]["sae"]
        axes[1].plot(np.arange(len(CONCEPTS))+(i-1)*.1,[values[c]["mean_test_auroc"] if values[c]["selected_features"] else np.nan for c in CONCEPTS],"o",color=colors[i])
    axes[0].set(xticks=range(3),xticklabels=["k32 seed0","k32 seed1","k64 seed0"],ylabel="Held-out centered FVE",title="Equal dictionary budgets · centered reconstruction")
    axes[1].set(xticks=range(len(CONCEPTS)),xticklabels=[c.removeprefix("fg_") for c in CONCEPTS],ylabel="Mean held-out selected-feature AUROC",title="All fixed selected features; missing shown as gaps")
    axes[1].tick_params(axis="x",rotation=25);axes[1].axhline(.5,color=".6",lw=.7)
    axes[0].legend(frameon=False,fontsize=8)
    for axis in axes:axis.spines[["top","right"]].set_visible(False)
    fig.suptitle("Diffusion SAE extension · point estimates; no test-best feature selection")
    for extension in ("png","pdf"):fig.savefig(output/f"sae_quality_and_associations.{extension}",dpi=180)
    plt.close(fig)
    with plt.rc_context({"font.size":8,"axes.titlesize":9,"axes.labelsize":8,"legend.fontsize":7.5}):
        fig,axes=plt.subplots(1,2,figsize=(143/25.4,47.5/25.4),layout="constrained")
        for i,(encoder,label) in enumerate(zip((*BASELINES,arm),("Parent 1190k","Latent 1209k",arm.replace("diffusion_","Diffusion ")),strict=True)):
            fit=[next(r for r in report["quality"] if(r["encoder"],r["k"],r["seed"])==(encoder,k,s)) for k,s in CONFIGURATIONS]
            axes[0].plot(np.arange(3)+(i-1)*.08,[r["quality"]["test"]["fve_vs_train_mean"] for r in fit],"o-",ms=3.5,lw=.8,color=colors[i],label=label)
            values=report["paired_comparison"]["feature_models"][encoder]["sae"]["concepts"]
            for j,concept in enumerate(CONCEPTS):
                stat=values[concept]["auroc"]
                if stat is None:continue
                y=stat["estimate"];low,high=stat["ci95"]
                axes[1].errorbar(j+(i-1)*.2,y,yerr=[[max(0,y-low)],[max(0,high-y)]],fmt="o",ms=2.7,capsize=1.5,color=colors[i])
        axes[0].set(xticks=range(3),xticklabels=["k32 s0","k32 s1","k64 s0"],ylabel="Held-out centered FVE",title="Centered reconstruction")
        axes[1].set(xticks=range(len(CONCEPTS)),xticklabels=["Amide","Amine","OH","Halide","Nitrile","Sulfonyl"],ylabel="Test AUROC",title="Fixed feature candidates")
        axes[1].tick_params(axis="x",rotation=35,labelsize=7)
        axes[1].axhline(.5,color=".6",lw=.7)
        for axis in axes:axis.spines[["top","right"]].set_visible(False)
        fig.legend(*axes[0].get_legend_handles_labels(),loc="outside upper center",ncol=3,frameon=False)
        fig.savefig(output/"sae_quality_and_associations_slide.pdf");plt.close(fig)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm",required=True,help="Pinned new arm directory name, e.g. diffusion_1234k")
    parser.add_argument("--root",type=Path,default=DEFAULT_ROOT)
    parser.add_argument("--mode",choices=("readouts","sae","all"),default="all")
    parser.add_argument("--fetch-readouts",action="store_true")
    parser.add_argument("--remote-root",default="/encoder_chemistry_20260930")
    parser.add_argument("--bootstrap",type=int,default=1000)
    args=parser.parse_args()
    if args.arm in BASELINES or Path(args.arm).name!=args.arm:
        raise ValueError("Use one new pinned diffusion arm name")
    output=args.root/"diffusion_extension"/args.arm
    output.mkdir(parents=True,exist_ok=True)
    if args.fetch_readouts:
        fetch_readouts(args.root,args.remote_root,[args.arm],[27])
    if args.mode in("readouts","all"):
        compare_readouts(args.root,args.arm,output,bootstrap=args.bootstrap)
    if args.mode in("sae","all"):
        analyze_new_sae(args.root,args.arm,output)


if __name__=="__main__":main()
