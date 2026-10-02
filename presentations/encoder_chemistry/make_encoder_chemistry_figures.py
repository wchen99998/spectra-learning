"""Slide and publication vector artwork from frozen encoder-chemistry results.

The first build freezes a compact evidence subset. Subsequent builds use only
research/plot_inputs.json and shared fonts; --refresh-inputs explicitly refreezes
the source artifacts. No model fitting or new statistical estimation occurs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
ART = REPO / "artifacts/encoder_chemistry_20260930"
OUT = ROOT / "figures"
DATA = ROOT / "research"
ARMS = ("parent_1190k", "periodic_1209k")
NAMES = ("Parent", "Latest")
COLORS = ("#345D7A", "#B25D3A")
CONTROLS = ("#167B75", "#866393")
CONCEPTS = ("fg_amide", "fg_hydroxyl", "fg_halide", "fg_nitrile", "fg_sulfonyl")
GENERIC = ("attenuate_top20", "attenuate_random20", "remove_top20", "remove_random20",
           "permute_intensity", "mz_jitter_0p01da", "mz_shift_plus_0p1da")
LEGENDS = {
 "token_attention": "Frozen token-attention readouts on1024 disjoint test molecules; arithmetic mean of3 seed logits. Panels a,b show macro AUROC with1000 paired molecule bootstrap95% intervals. Panel c is latest minus parent, paired on identical molecules. Fixed shared145MACCS and13functional-group targets (≥5 positives/negatives in train/test); rare-target undefined resamples are excluded with the target set fixed. Depth19 is raw; final is native state27. Scores are chemical accessibility under this readout, not fragment annotations.",
 "token_geometry": "Existing descriptive native physical-token geometry over6144 panel molecules. Effective rank uses a shared256-dimensional random projection and at most8 fixed quantile physical peaks per molecule. Input m/z R² uses7953 held-out sampled tokens with train/validation split by molecule. States26/27 are final norm/output. Geometry and input reconstruction are not direct chemical semantics; no confidence intervals are claimed.",
 "sae_reconstruction": "Panel a: raw19 TopK SAE held-out fraction of variance explained relative to the training mean; equal molecule weighting,8192dictionary features, three configurations per encoder. Seed1 changes initialization, training order and sampled physical tokens; k64/seed0 shares the k32/seed0 sampled tokens. Individual fits are points, not independent repeated experiments. Panel b: full SAE reconstruction replacement changes frozen-readout balanced BCE on512 test molecules;2000 paired molecule bootstrap95% intervals. High reconstruction quality does not imply chemical interpretability or harmless replacement.",
 "chemical_features": "Held-out1024-molecule AUROC averaged over both fixed train/validation-selected SAE features per concept. Feature IDs were frozen before test confirmation; no test-best feature is chosen. Error bars are1000 paired molecule bootstrap95% intervals, conditional on dictionary and selected features. Nitrile failures retained; amine had no qualifying feature and is not plotted. Sparse-feature associations are not monosemanticity or measured fragment identity.",
 "feature_stability": "For each primary feature, align a control-dictionary feature by highest training molecule-max Spearman without chemical labels, then measure held-out test activation Spearman. Points average the two frozen primary features per concept; no uncertainty interval or independent-run population claim. Seed1 and k64/seed0 have distinct perturbations described in reconstruction legend. Latest hydroxyl primary features share one aligned control feature in both controls; replication need not imply two independent mechanisms or chemistry.",
 "sae_causal": "All10 frozen features per encoder, dose1 error-preserving SAE direction edits on512 test molecules. Points show target-logit effect of feature ablation minus a control with the same REQUESTED per-token edit L2 before bfloat16 rounding: firing/RMS-matched dictionary direction or random orthogonal direction. Realized norms match approximately; some median mismatches are3–4% and latest-checkpoint tails reach7–8%. Negative means less target evidence than control.2000 paired molecule bootstrap95% intervals are unadjusted; the40-test family is documented in original analysis. Off-target effects, nuisance features and failed nitrile associations remain; no claim of isolated chemical mechanisms or molecule counterfactuals.",
 "chemical_neighbors": "Label-free cosine nearest training molecule using molecular maxima over8192 token-SAE features. Panels a,b show1024-query full-bank and472-query matched-bank chemical similarities; the matched bank requires≥2candidates with identical instrument/adduct/CEpresence and≤5Da precursor,10rawCE,5peak-count gaps. Panel c shows paired latest-minus-parent Morgan differences with1000 molecule bootstrap95% intervals. Labels only score neighbors. The matched subset has404queries with no candidate and148 with only one; support restriction and residual confounding limit inference.",
 "component_heads": "Panel a: test balanced-BCE change after ablating the validation-top-ranked attention, MLP or metadata branch for each encoder; selection used256validation molecules, confirmation512test molecules. Bars are2000 paired molecule bootstrap95% intervals. Panels b,c show all8 individual heads in each of3validation-selected attention blocks; colors are point estimates with a common scale. Branch and head perturbation sizes differ, so magnitude is not a normalized causal ranking. Block IDs are1-based; head IDs are0-based. Negative BCE change is improvement.",
 "activation_rescue": "After attenuating the highest-intensity20% of physical peaks to10%, replace clean raw activations at the affected slots or an equal number of uncorrupted random slots, for every native raw depth0–25. Curves show selected-minus-random logit-MSE reduction and balanced-BCE recovery on256test molecules; shaded bands are2000 paired molecule bootstrap95% intervals. Output-distance rescue is stronger than chemical-loss evidence; no test-selected best-layer claim. Full-state restoration matches clean predictions bitwise.",
 "input_robustness": "1024test molecules,177train-supported chemical targets, three frozen attentive heads ensembled in score space. Panels show unweighted BCE change and absolute sigmoid-score drift with2000 paired molecule bootstrap95% intervals. Attenuation retains slots/validity and multiplies selected intensity by0.1; removal changes validity; random controls match counts. m/z jitter±0.01Da and shift+0.1Da are artificial/OOD perturbations, preserving physical slots. These score-based metrics are not calibrated probabilities or embedding invariance.",
 "metadata_effects": "689same-molecule observed-repeat pairs with matching instrument/adduct and changed measured collision energy. Panel a compares CE-only metadata replacement, observed spectrum+ownCE, and observed spectrum+anchorCE against the anchor. Panel b changes measured CE on a fixed spectrum through final affine, internal blocks, or both; native diagonal shams are bitwise exact. Unweighted ensemble BCE across177targets;2000 paired molecule bootstrap95% intervals. Measured repeats can change spectra; metadata replacement is a computational intervention, not a molecule counterfactual.",
 "halide_case": "Illustrative latest-encoder raw19 SAE feature7363, from the already-frozen feature atlas. The positive molecule and false positive are exactly the highest-activating test examples in their true-label classes, selected after confirmation, not new unseen evidence or a matched pair. Structures highlight molecular halogen atoms only. Grey lines are native selected measured peaks with transformed input intensity; orange lines are actual SAE activation at exact physical peak slots. Highest-activation peak annotated; no peak-to-fragment ground truth is asserted. Cohort testAUROC.675, matched donor ordering.649; counterexamples show imperfect specificity."
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stat(value: dict) -> dict:
    return {"mean": float(value.get("mean", value.get("estimate"))), "ci95": value.get("ci95")}


def freeze_inputs() -> dict:
    sources = {}
    def read(relative):
        path = ART / relative
        sources[str(path.relative_to(REPO))] = sha(path)
        return json.loads(path.read_text())
    result = {"schema": "encoder_chemistry_presentation_plot_inputs_v1", "source_hashes": sources,
              "legends": LEGENDS, "checkpoint_names": dict(zip(ARMS, NAMES)), "plots": {}}
    plots = result["plots"]
    data = read("readout_comparison/comparison.json")["cohorts"]["test"]
    plots["token_attention"] = {
        "models": {arm: {str(depth): {g: stat(data["models"][f"{arm}/depth{depth}/attention"]["groups"][g]["ensemble"])
                  for g in ("MACCS", "functional_groups")} for depth in (19,27)} for arm in ARMS},
        "contrasts": {str(depth): {g: stat(data["contrasts"][f"checkpoint_depth{depth}_attention"]["groups"][g]["ensemble"])
                      for g in ("MACCS", "functional_groups")} for depth in (19,27)}}
    plots["token_geometry"] = {arm: read(f"{arm}/atlas.json")["token_geometry"] for arm in ARMS}
    causal = read("sae_causal_analysis/sae_causal_analysis.json")
    plots["sae_reconstruction"] = {"fits": read("sae_reconstruction_comparison.json")["results"],
        "replacement": {arm["encoder"]: {g: stat(arm["reconstruction"][g]["metrics"]["balanced_bce_change"])
                                        for g in ("MACCS","functional_groups","all")} for arm in causal["arms"]}}
    associations = read("sae_comparison/comparison.json")
    plots["chemical_features"] = {arm: associations["feature_models"][arm]["families"]["sae"]["concepts"] for arm in ARMS}
    plots["feature_stability"] = {arm: {config: read(f"{arm}/sae/depth19/k32/seed0/stability/{config}_test_confirmation.json")["records"]
                                        for config in ("k32_seed1", "k64_seed0")} for arm in ARMS}
    plots["sae_causal"] = {arm["encoder"]: [{"feature": f["feature"], "concept": f["concept"],
        "effects": {kind: stat(f["doses"]["1.0"]["feature_minus_control"][kind]["all"]["metrics"]["target_logit_change"])
                    for kind in ("matched_feature_direction", "random_unit_direction")}} for f in arm["features"]] for arm in causal["arms"]}
    matched = read("sae_comparison/matched_geometry.json")
    plots["chemical_neighbors"] = {
        "models": {arm: {"full": {metric: stat({"mean": r["mean"], "ci95": r["ci95"]})
            for metric,r in associations["geometry_models"][arm]["representations"]["sae"]["metrics"].items()},
            "matched": {metric: stat(r["absolute"]) for metric,r in matched["models"][arm+"/sae"].items()}} for arm in ARMS},
        "contrasts": {"full": {metric: stat(r["difference"]) for metric,r in associations["geometry_checkpoint_contrasts"]["sae"].items()},
                      "matched": {metric: stat(r) for metric,r in matched["checkpoint_contrasts"]["sae"].items()}},
        "matched_support": matched["support"]}
    plots["component_heads"] = read("component_analysis/summary.json")["arms"]
    plots["activation_rescue"] = read("patching_analysis/results.json")["arms"]
    inputs = read("input_effects_analysis/results.json")["checkpoints"]
    plots["input_robustness"] = {arm: {key: {metric: stat(inputs[arm]["corruptions"][key]["groups"]["trained_targets"][metric])
        for metric in ("delta_bce", "probability_l1_drift")} for key in GENERIC} for arm in ARMS}
    plots["metadata_effects"] = {arm: {"repeat": {key: stat(inputs[arm]["observed_repeats"][key]["groups"]["trained_targets"]["delta_bce"])
        for key in ("metadata_only_on_anchor", "observed_repeat_own_metadata", "repeat_with_anchor_metadata")},
        "paths": {key: stat(inputs[arm]["metadata_factorial"][key]["groups"]["trained_targets"]["delta_bce"])
                  for key in ("final_affine_only", "blocks_only", "both_paths")}} for arm in ARMS}
    atlas = read("feature_atlas/selection.json")
    plots["halide_case"] = next(c for c in atlas["cards"] if c["arm"]==ARMS[1] and c["feature"]==7363)
    # Freeze editable molecular SVG as well, so a source-bundle rebuild needs no RDKit.
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    for example in plots["halide_case"]["examples"]:
        drawer = rdMolDraw2D.MolDraw2DSVG(320,320,-1,-1,True)
        drawer.drawOptions().fixedFontSize = 25
        drawer.drawOptions().bondLineWidth = 1.4
        drawer.DrawMolecule(Chem.MolFromSmiles(example["smiles"]), highlightAtoms=example["concept_atoms"])
        drawer.FinishDrawing()
        example["structure_svg"] = drawer.GetDrawingText().replace("font-family:sans-serif", "font-family:Liberation Sans")
    DATA.mkdir(exist_ok=True)
    (DATA/"plot_inputs.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    return result


def setup(slide: bool):
    size = 9 if slide else 6.5
    plt.rcParams.update({"font.family":"Liberation Sans", "font.size":size, "axes.labelsize":size,
        "axes.titlesize":10 if slide else 7, "legend.fontsize":size, "xtick.labelsize":size,
        "ytick.labelsize":size, "axes.spines.top":False, "axes.spines.right":False,
        "axes.linewidth":.55, "lines.linewidth":.85, "xtick.major.width":.5, "ytick.major.width":.5,
        "pdf.fonttype":42, "ps.fonttype":42, "svg.fonttype":"none", "savefig.facecolor":"white"})


def panels(n=2, slide=False, ratios=None):
    fig, axes = plt.subplots(1,n,figsize=((143 if slide else 183)/25.4,(47 if slide else 60)/25.4),
                            layout="constrained",squeeze=False,width_ratios=ratios)
    return fig, axes[0]


def title(axis, index, label):
    axis.set_title(f"{chr(97+index)}  {label}",loc="left",pad=4)


def points(axis, records, positions, color, label=None, horizontal=False, marker="o"):
    means=np.array([r["mean"] for r in records]); ci=np.array([r["ci95"] for r in records])
    errors=np.maximum(0,np.stack([means-ci[:,0],ci[:,1]-means]))
    args={"fmt":marker,"color":color,"label":label,"markersize":3.2,"capsize":1.6,"elinewidth":.7,"markeredgewidth":.7}
    if horizontal:axis.errorbar(means,positions,xerr=errors,**args)
    else:axis.errorbar(positions,means,yerr=errors,**args)


def legend(fig, axes, ncol=2):
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:fig.legend(handles,labels,loc="outside lower center",ncol=ncol,frameon=False,
                          columnspacing=1.2,handlelength=1.3,borderpad=.05,labelspacing=.1)


def plot_token_attention(data,slide):
    fig,axes=panels(3,slide)
    for j,g in enumerate(("MACCS","functional_groups")):
        for i,arm in enumerate(ARMS):
            points(axes[j],[data["models"][arm][str(d)][g]for d in (19,27)],np.arange(2)+(i-.5)*.12,COLORS[i],NAMES[i])
        axes[j].set(xticks=[0,1],xticklabels=["Raw19","Final"],ylabel="Macro AUROC",xlim=(-.4,1.4))
        title(axes[j],j,"MACCS"if j==0 else "Functional groups")
    records=[data["contrasts"][str(d)][g] for d in (19,27)for g in ("MACCS","functional_groups")]
    points(axes[2],records,np.arange(4),COLORS[1],horizontal=True)
    axes[2].set(yticks=range(4),yticklabels=["19 MACCS","19 FG","Final MACCS","Final FG"],xlabel="Δ AUROC")
    axes[2].invert_yaxis();axes[2].axvline(0,color=".6",lw=.5);axes[2].xaxis.set_major_locator(MaxNLocator(3))
    title(axes[2],2,"Latest − parent");legend(fig,axes);return fig,[]


def plot_token_geometry(data,slide):
    fig,axes=panels(3,slide)
    for j,(key,label)in enumerate((("projected_token_effective_rank","Projected rank"),("median_within_spectrum_variance_fraction","Token spread"),("input_linear_r2_mz","Input m/z R²"))):
        for i,arm in enumerate(ARMS):
            rows=data[arm];axes[j].plot(range(len(rows)),[r[key]for r in rows],"o-",color=COLORS[i],markersize=2.6,label=NAMES[i])
        axes[j].set(xticks=[0,2,4,6,7],xticklabels=["0","10","19","N","O"],xlabel="Selected native state")
        title(axes[j],j,label)
    axes[0].set_ylabel("Effective rank (≤256)");axes[1].set_ylabel("Variance fraction");axes[2].set_ylabel("Held-out token R²")
    legend(fig,axes);return fig,[]


def plot_sae_reconstruction(data,slide):
    fig,axes=panels(2,slide)
    configs=((32,0),(32,1),(64,0))
    for i,arm in enumerate(ARMS):
        rows=[next(r for r in data["fits"]if(r["encoder"],r["k"],r["seed"])==(arm,k,s))for k,s in configs]
        axes[0].plot(np.arange(3)+(i-.5)*.12,[r["quality"]["test"]["fve_vs_train_mean"]for r in rows],"o-",color=COLORS[i],label=NAMES[i],markersize=3.5)
        points(axes[1],[data["replacement"][arm][g]for g in("MACCS","functional_groups","all")],np.arange(3)+(i-.5)*.12,COLORS[i],NAMES[i])
    axes[0].set(xticks=range(3),xticklabels=["k32\nseed0","k32\nseed1","k64\nseed0"],ylabel="Held-out FVE",ylim=(.68,.95))
    axes[1].set(xticks=range(3),xticklabels=["MACCS","FG","All"],ylabel="Balanced BCE change")
    axes[1].axhline(0,color=".6",lw=.5);title(axes[0],0,"Six dictionary fits");title(axes[1],1,"Full reconstruction damage")
    legend(fig,axes);return fig,[]


def plot_chemical_features(data,slide):
    fig,axes=panels(1,slide);axis=axes[0]
    for i,arm in enumerate(ARMS):points(axis,[stat(data[arm][c]["auroc"])for c in CONCEPTS],np.arange(5)+(i-.5)*.13,COLORS[i],NAMES[i])
    axis.set(xticks=range(5),xticklabels=[c.removeprefix("fg_")for c in CONCEPTS],ylabel="Test AUROC",ylim=(.44,.74))
    axis.axhline(.5,color=".6",lw=.5,ls="--");title(axis,0,"Both frozen features per concept; failures retained")
    legend(fig,axes);return fig,[]


def plot_feature_stability(data,slide):
    fig,axes=panels(2,slide)
    for i,arm in enumerate(ARMS):
        for j,config in enumerate(("k32_seed1","k64_seed0")):
            values=[np.mean([r["activation_alignment"]["test_spearman"]for r in data[arm][config]if r["concept"]==c])for c in CONCEPTS]
            axes[i].plot(range(5),values,"o-"if j==0 else "s--",color=CONTROLS[j],label="Seed1"if j==0 else "k64",markersize=3)
        axes[i].set(xticks=range(5),xticklabels=["Amide","OH","Halide","CN","SO₂"],ylim=(.3,1),ylabel="Activation Spearman ρ")
        title(axes[i],i,NAMES[i]+": fixed-feature replication")
    legend(fig,axes);return fig,[]


def plot_sae_causal(data,slide):
    fig,axes=panels(2,slide)
    for i,arm in enumerate(ARMS):
        rows=data[arm]
        for j,kind in enumerate(("matched_feature_direction","random_unit_direction")):
            points(axes[i],[r["effects"][kind]for r in rows],np.arange(10)+(j-.5)*.17,CONTROLS[j],"Matched direction"if j==0 else "Orthogonal direction",horizontal=True,marker="o"if j==0 else "s")
        labels=[r["concept"].removeprefix("fg_")+" "+str(r["feature"])for r in rows]
        axes[i].set(yticks=range(10),yticklabels=labels,xlabel="Target logit: feature − control")
        axes[i].invert_yaxis();axes[i].axvline(0,color=".6",lw=.5);axes[i].xaxis.set_major_locator(MaxNLocator(3));title(axes[i],i,NAMES[i])
    legend(fig,axes);return fig,[]


def plot_chemical_neighbors(data,slide):
    fig,axes=panels(3,slide)
    for j,(metric,name)in enumerate((("morgan_tanimoto","Morgan Tanimoto"),("functional_group_jaccard","Functional-group overlap"))):
        for i,arm in enumerate(ARMS):points(axes[j],[data["models"][arm][pool][metric]for pool in("full","matched")],np.arange(2)+(i-.5)*.12,COLORS[i],NAMES[i])
        axes[j].set(xticks=range(2),xticklabels=["Full\nn=1024","Matched\nn=472"],ylabel="Chemical similarity")
        title(axes[j],j,name)
    points(axes[2],[data["contrasts"][p]["morgan_tanimoto"]for p in("full","matched")],np.arange(2),COLORS[1],horizontal=True)
    axes[2].set(yticks=range(2),yticklabels=["Full","Matched"],xlabel="Δ Morgan Tanimoto",ylim=(1.5,-.5))
    axes[2].axvline(0,color=".6",lw=.5);axes[2].xaxis.set_major_locator(MaxNLocator(3));title(axes[2],2,"Paired difference")
    legend(fig,axes);return fig,[]


def plot_component_heads(data,slide):
    fig,axes=panels(3,slide,ratios=[1.1,1,1]);kinds=("attention","mlp","metadata")
    matrices=[];blocks=[]
    for i,arm in enumerate(ARMS):
        d=data[arm];condition={r["variant"]:r for r in d["conditions"]}
        rows=[condition[f"{k}_block{d['validation_selection']['branch_layers'][k][0]}_scale0"]for k in kinds]
        points(axes[0],[{"mean":r["delta"],"ci95":r["ci95"]}for r in rows],np.arange(3)+(i-.5)*.14,COLORS[i],NAMES[i],horizontal=True)
        layer=d["validation_selection"]["branch_layers"]["attention"];blocks.append(layer)
        matrices.append(np.array([[condition[f"head_block{b}_head{h}_scale0"]["delta"]for h in range(8)]for b in layer]))
    axes[0].set(yticks=range(3),yticklabels=["Attention","MLP","Metadata"],xlabel="Balanced BCE change",ylim=(2.5,-.5));axes[0].axvline(0,color=".6",lw=.5)
    title(axes[0],0,"Validation-first branch")
    limit=max(np.abs(m).max()for m in matrices)
    for i in range(2):
        im=axes[i+1].pcolormesh(np.arange(9)-.5,np.arange(4)-.5,matrices[i],cmap="RdBu_r",vmin=-limit,vmax=limit,shading="flat",rasterized=False)
        axes[i+1].set(xticks=range(8),yticks=range(3),yticklabels=[f"B{b}"for b in blocks[i]],xlabel="Ablated head",ylim=(2.5,-.5))
        title(axes[i+1],i+1,NAMES[i]+" heads")
    colorbar=fig.colorbar(im,ax=axes[1:],label="Δ balanced BCE",shrink=.8,pad=.025,ticks=[-limit,0,limit],format="%.3f")
    colorbar.solids.set_rasterized(False)
    legend(fig,axes);return fig,[]


def plot_activation_rescue(data,slide):
    fig,axes=panels(2,slide)
    for j,(metric,title_text)in enumerate((("selected_minus_random_mse_reduction","Logit-distance rescue"),("selected_minus_random_bce_recovery","Chemical-loss rescue"))):
        for i,arm in enumerate(ARMS):
            rows=data[arm]["depths"];x=[r["depth"]for r in rows];values=[r[metric]["mean"]for r in rows];ci=np.array([r[metric]["ci95"]for r in rows])
            axes[j].plot(x,values,color=COLORS[i],label=NAMES[i]);axes[j].fill_between(x,ci[:,0],ci[:,1],color=COLORS[i],alpha=.13,lw=0)
        axes[j].set(xlabel="Restored raw state",ylabel="Logit MSE reduction"if j==0 else "Balanced BCE recovery",xticks=[0,5,10,15,20,25])
        axes[j].axhline(0,color=".6",lw=.5);title(axes[j],j,title_text)
    legend(fig,axes);return fig,[]


def plot_input_robustness(data,slide):
    fig,axes=panels(2,slide)
    labels=["Top20% atten.","Random atten.","Top20% remove","Random remove","Intensity permute","m/z ±0.01Th*","m/z +0.1Th*"]
    for j,metric in enumerate(("delta_bce","probability_l1_drift")):
        for i,arm in enumerate(ARMS):points(axes[j],[data[arm][key][metric]for key in GENERIC],np.arange(7)+(i-.5)*.14,COLORS[i],NAMES[i],horizontal=True)
        axes[j].set(yticks=range(7),yticklabels=labels if j==0 else [""]*7,xlabel="Unweighted BCE change"if j==0 else "Absolute sigmoid-score drift",ylim=(6.6,-.6))
        axes[j].axvline(0,color=".6",lw=.5);title(axes[j],j,"Chemical readout loss"if j==0 else "Prediction response")
    legend(fig,axes);return fig,[]


def plot_metadata_effects(data,slide):
    fig,axes=panels(2,slide)
    for j,kind in enumerate(("repeat","paths")):
        for i,arm in enumerate(ARMS):points(axes[j],list(data[arm][kind].values()),np.arange(3)+(i-.5)*.12,COLORS[i],NAMES[i])
        labels=["CE only","Repeat +\nown CE","Repeat +\nanchor CE"]if j==0 else ["Final affine","Blocks","Both"]
        axes[j].set(xticks=range(3),xticklabels=labels,ylabel="Unweighted BCE change")
        axes[j].axhline(0,color=".6",lw=.5);title(axes[j],j,"Observed CE pairs (n=689)"if j==0 else "CE through native pathways")
    legend(fig,axes);return fig,[]


def plot_halide_case(data,slide):
    fig=plt.figure(figsize=((143 if slide else 183)/25.4,(47 if slide else 60)/25.4),layout="constrained")
    outer=fig.add_gridspec(1,2,wspace=.04);overlays=[]
    max_activation=max(e["maximum_activation"]for e in data["examples"])
    for i,e in enumerate(data["examples"]):
        grid=outer[i].subgridspec(1,2,width_ratios=[1,1.05],wspace=.01)
        mol=fig.add_subplot(grid[0]);mol.axis("off");overlays.append((mol,e["structure_svg"]))
        mol.set_title(("a  Concept present"if i==0 else "b  False positive"),loc="left",pad=3)
        mol.text(.02,.01,f"CE {e['raw_ce']:.0f}\n{e['valid_peaks']} peaks",transform=mol.transAxes,va="bottom",fontsize=8 if slide else 6)
        sub=grid[1].subgridspec(2,1,hspace=.08);top=fig.add_subplot(sub[0]);bottom=fig.add_subplot(sub[1],sharex=top)
        mz=np.array([p["mz"]for p in e["peaks"]]);intensity=np.array([p["input_intensity"]for p in e["peaks"]]);activation=np.array([p["activation"]for p in e["peaks"]])
        top.vlines(mz,0,intensity,color="#71818B",lw=.6);bottom.vlines(mz,0,activation,color=COLORS[1],lw=.7)
        top.set(ylabel="Input",ylim=(0,1.08));top.tick_params(labelbottom=False);bottom.set(ylabel="SAE",xlabel="m/z (Th)",ylim=(0,max_activation*1.3))
        best=int(np.argmax(activation));bottom.annotate(f"{mz[best]:.1f}",(mz[best],activation[best]),xytext=(0,3),textcoords="offset points",ha="center",fontsize=8 if slide else 6)
        bottom.xaxis.set_major_locator(MaxNLocator(3));top.yaxis.set_major_locator(MaxNLocator(2));bottom.yaxis.set_major_locator(MaxNLocator(2))
    return fig,overlays


BUILDERS={name:globals()["plot_"+name]for name in LEGENDS}


def save(fig,overlays,name,slide):
    suffix="_slide"if slide else "";stem=OUT/(name+suffix)
    fig.canvas.draw()
    fig.savefig(stem.with_suffix(".svg"),transparent=slide)
    if overlays:
        ET.register_namespace("","http://www.w3.org/2000/svg")
        tree=ET.parse(stem.with_suffix(".svg"));root=tree.getroot()
        width,height=fig.get_size_inches()*72
        for axis,svg in overlays:
            position=axis.get_position();element=ET.fromstring(svg)
            # Reserve title/metadata clearance; preserve the molecular square.
            side=min(position.width*width,position.height*height*.72)
            element.set("x",str(position.x0*width));element.set("y",str((1-position.y1)*height+position.height*height*.06))
            element.set("width",str(side));element.set("height",str(side));element.set("viewBox","0 0 320 320")
            root.append(element)
        tree.write(stem.with_suffix(".svg"),encoding="unicode")
        subprocess.run(["rsvg-convert","--format=pdf","--output",str(stem.with_suffix(".pdf")),str(stem.with_suffix(".svg"))],check=True)
    else:fig.savefig(stem.with_suffix(".pdf"),transparent=slide)
    # Previews are rasterized from the final vector PDF so overlay checks agree.
    import pymupdf
    with pymupdf.open(stem.with_suffix(".pdf")) as doc:
        doc[0].get_pixmap(matrix=pymupdf.Matrix(2.5,2.5),alpha=False).save(stem.with_suffix(".png"))
    plt.close(fig)


def write_summary(data):
    summary={"schema":"encoder_chemistry_presentation_summary_v1","source_hashes":data["source_hashes"],
        "plot_inputs_sha256":sha(DATA/"plot_inputs.json"),"figure_source_sha256":sha(Path(__file__)),
        "production":{"slide_mm":[143,47],"standalone_mm":[183,60],"slide_font_pt":9,"standalone_font_pt":6.5,
                      "fonts":"Liberation Sans; embedded TrueType; no Type3; SVG text editable"},
        "figures":[{"stem":name,"slide_pdf":f"figures/{name}_slide.pdf","standalone_pdf":f"figures/{name}.pdf", "legend":legend}for name,legend in data["legends"].items()]}
    (DATA/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    with(DATA/"metrics.csv").open("w",newline="")as stream:
        writer=csv.writer(stream);writer.writerow(["plot_data_path","value"])
        def rows(value,path):
            if isinstance(value,dict):
                for k,v in value.items():rows(v,path+"/"+k)
            elif isinstance(value,list):
                for i,v in enumerate(value):rows(v,path+"/"+str(i))
            elif isinstance(value,(int,float))and not isinstance(value,bool):writer.writerow([path,value])
        rows(data["plots"],"")
    readouts=data["plots"]["token_attention"]
    numbers={"ECCParentAttentionMACCS":readouts["models"][ARMS[0]]["19"]["MACCS"]["mean"],
             "ECCLatestAttentionMACCS":readouts["models"][ARMS[1]]["19"]["MACCS"]["mean"],
             "ECCAttentionMACCSDelta":readouts["contrasts"]["19"]["MACCS"]["mean"],
             "ECCParentAttentionFG":readouts["models"][ARMS[0]]["19"]["functional_groups"]["mean"],
             "ECCLatestAttentionFG":readouts["models"][ARMS[1]]["19"]["functional_groups"]["mean"]}
    (DATA/"numbers.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v:.3f}}}\n"for k,v in numbers.items()))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--refresh-inputs",action="store_true");parser.add_argument("--only",choices=list(LEGENDS));args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    for name in("Regular","Bold"):font_manager.fontManager.addfont(str(ROOT.parent/"shared/assets"/f"LiberationSans-{name}.ttf"))
    data=freeze_inputs()if args.refresh_inputs or not(DATA/"plot_inputs.json").exists()else json.loads((DATA/"plot_inputs.json").read_text())
    for name in([args.only]if args.only else BUILDERS):
        for slide in(False,True):
            setup(slide);fig,overlays=BUILDERS[name](data["plots"][name],slide);save(fig,overlays,name,slide)
        print("Rendered",name,flush=True)
    write_summary(data)


if __name__=="__main__":main()
