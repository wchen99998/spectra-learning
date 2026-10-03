"""Paired molecule-level summaries for frozen-decoder intervention experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score


def analyze_predictions(path: Path, panel: Path, *, resamples: int = 2000) -> dict:
    saved = np.load(path, allow_pickle=False)
    rows = saved["rows"]
    labels = np.load(panel / "labels.npz", allow_pickle=False)
    data = np.load(panel / "inputs.npz", allow_pickle=False)
    all_truth = np.concatenate([labels["maccs"], labels["binary"]], axis=1).astype(float)
    truth = all_truth[rows]
    logits = saved["logits"]
    if logits.ndim != 4 or logits.shape[1:] != (len(rows), 3, 188):
        raise ValueError("Expected [intervention,molecule,seed,target] frozen readout logits")
    variants = saved["variants"].astype(str)
    if variants[0] not in ("identity", "clean"):
        raise ValueError("The first intervention must be the unmodified baseline")
    if not np.isfinite(logits).all() or len(np.unique(data["molecule"][rows])) != len(rows):
        raise ValueError("Need finite logits and one independent row per molecular identity")
    ensemble = expit(logits).mean(2)
    positive, negative = truth.sum(0), len(rows)-truth.sum(0)
    training = all_truth[data["fold"] == "train"]
    supported = (positive >= 5) & (negative >= 5) & (training.sum(0) >= 5) & ((1-training).sum(0) >= 5)
    target_names = [f"maccs_{i:03d}" for i in range(1,167)] + labels["binary_names"].tolist()
    groups = {"MACCS":np.arange(166),"elements":np.arange(166,173),"functional_groups":np.arange(173,188),
              "all":np.arange(188)}
    clipped = np.clip(ensemble,1e-7,1-1e-7)
    bce = -truth[None]*np.log(clipped)-(1-truth[None])*np.log1p(-clipped)
    # This expresses a class-balanced risk as an average over independent molecules.
    # Fixed empirical class weights are shared by every arm and every resample.
    weight = np.where(truth>0,len(rows)/(2*np.maximum(positive,1)),len(rows)/(2*np.maximum(negative,1)))
    rng=np.random.default_rng(20260930)
    boot=rng.integers(0,len(rows),size=(resamples,len(rows)))
    results=[]
    for index,name in enumerate(variants):
        record={"variant":str(name),"groups":{},"per_target":[]}
        for group,indices in groups.items():
            indices=indices[supported[indices]]
            if not len(indices):
                record["groups"][group]=None
                continue
            losses=(bce[index][:,indices]*weight[:,indices]).mean(1)
            reference=(bce[0][:,indices]*weight[:,indices]).mean(1)
            delta=losses-reference
            resampled=delta[boot].mean(1)
            auc=np.mean([roc_auc_score(truth[:,j],ensemble[index,:,j]) for j in indices])
            baseline_auc=np.mean([roc_auc_score(truth[:,j],ensemble[0,:,j]) for j in indices])
            record["groups"][group]={"targets":len(indices),"balanced_bce":float(losses.mean()),
                "delta_balanced_bce":float(delta.mean()),"delta_bce_ci95":np.quantile(resampled,[.025,.975]).tolist(),
                "auroc":float(auc),"delta_auroc":float(auc-baseline_auc),
                "interpretation":"Positive BCE change or negative AUROC change indicates worse fixed-decoder chemistry."}
        for j,target in enumerate(target_names):
            delta=(bce[index,:,j]-bce[0,:,j])*weight[:,j]
            record["per_target"].append({"target":target,"supported":bool(supported[j]),
                "positive":int(positive[j]),"negative":int(negative[j]),
                "signed_logit_change":float(((logits[index,:,:,j]-logits[0,:,:,j])*(2*truth[:,j,None]-1)).mean()),
                "delta_balanced_bce":float(delta.mean()) if supported[j] else None,
                "auroc":float(roc_auc_score(truth[:,j],ensemble[index,:,j])) if supported[j] else None})
        results.append(record)
    return {"source":str(path),"rows":rows.tolist(),"molecules":len(rows),"folds":np.unique(data["fold"][rows]).tolist(),
        "seeds":[0,1,2],"ensemble":"Mean of three sigmoid score vectors; balanced-loss outputs are not calibrated probabilities.",
        "baseline":str(variants[0]),"results":results,"bootstrap":{"resamples":resamples,"seed":20260930,
            "unit":"molecule","interval":"paired percentile, fixed empirical class weights","multiplicity":"Unadjusted descriptive intervals; not simultaneous family-wise guarantees.",
            "scope":"Conditional on the three fitted decoder seeds and the fixed benchmark cohort; not training-recipe uncertainty."}}


def select_head_layers(summary: dict, *, count: int = 3) -> dict:
    if summary["folds"] != ["val"]:
        raise ValueError("Head layers may only be selected on validation intervention effects")
    candidates=[]
    for record in summary["results"]:
        name=record["variant"]
        if name.startswith("attention_block") and name.endswith("_scale0"):
            layer=int(name.removeprefix("attention_block").removesuffix("_scale0"))
            candidates.append((record["groups"]["all"]["delta_balanced_bce"],layer))
    selected=sorted(candidates,reverse=True)[:count]
    return {"layers":[layer for _,layer in selected],"selection":"Largest validation increase in supported-target balanced BCE after attention-branch ablation",
            "scores":[value for value,_ in selected],"test_used":False,"source":summary["source"]}


def make_figure(summary: dict, destination: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    for prefix,label,color in (("attention_block","Attention","#286f9b"),("mlp_block","MLP","#bd5a30"),("metadata_block","Conditioning","#4f865e")):
        entries=[r for r in summary["results"] if r["variant"].startswith(prefix)]
        if not entries:
            continue
        layer=[int(r["variant"].removeprefix(prefix).removesuffix("_scale0")) for r in entries]
        for ax,group in zip(axes,("MACCS","functional_groups")):
            y=[r["groups"][group]["delta_balanced_bce"] for r in entries]
            lower=[r["groups"][group]["delta_bce_ci95"][0] for r in entries]
            upper=[r["groups"][group]["delta_bce_ci95"][1] for r in entries]
            ax.plot(layer,y,label=label,color=color,lw=1.5)
            ax.fill_between(layer,lower,upper,color=color,alpha=.12)
            ax.axhline(0,color="#777777",lw=.7)
            ax.set(xlabel="Encoder block",ylabel="Change in balanced BCE",title=group.replace("_"," "))
            ax.spines[["top","right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle(f"Frozen token-readout response to ablation · {summary['molecules']} molecules")
    fig.savefig(destination,dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions",type=Path)
    parser.add_argument("--panel",type=Path,default=Path("artifacts/encoder_chemistry_20260930/panel"))
    parser.add_argument("--resamples",type=int,default=2000)
    args=parser.parse_args()
    summary=analyze_predictions(args.predictions,args.panel,resamples=args.resamples)
    output=args.predictions.parent
    (output/"analysis.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    if summary["folds"]==["val"] and any(r["variant"].startswith("attention_block") for r in summary["results"]):
        (output/"selected_head_layers.json").write_text(json.dumps(select_head_layers(summary),indent=2)+"\n")
    if any(r["variant"].startswith("attention_block") for r in summary["results"]):
        make_figure(summary,output/"component_effects.png")


if __name__=="__main__":
    main()
