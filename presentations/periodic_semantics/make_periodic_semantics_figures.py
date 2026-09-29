"""Vector figures for the periodic-teacher investigation; no synthetic results."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "research"
OUT = ROOT / "figures"
ARMS = ("ftjepa_1190k", "periodic_student_1200k", "periodic_student_1209k")
LABELS = ("Initial teacher", "Promoted teacher", "Latest student")
COLORS = ("#0C0C0C", "#8A9398", "#007C83")


def setup():
    font_manager.fontManager.addfont(str(ROOT.parent / "shared/assets/LiberationSans-Regular.ttf"))
    font_manager.fontManager.addfont(str(ROOT.parent / "shared/assets/LiberationSans-Bold.ttf"))
    plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 7,
        "axes.titlesize": 8, "axes.labelsize": 7, "legend.fontsize": 6.3,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": .5, "lines.linewidth": .9,
        "xtick.major.width": .5, "ytick.major.width": .5, "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white"})
    OUT.mkdir(parents=True, exist_ok=True)


def panels(n=2, height=68):
    fig, ax = plt.subplots(1, n, figsize=(183/25.4, height/25.4), layout="constrained", squeeze=False)
    for i, a in enumerate(ax[0]):
        a.set_title(chr(97+i), loc="left", fontweight="bold")
    return fig, ax[0]


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.svg")
    fig.savefig(OUT / f"{name}_slide.pdf", transparent=True)
    plt.close(fig)


def grouped(ax, values, categories, ylabel, ylim=None):
    x = np.arange(len(categories))
    for j, (v, color, label) in enumerate(zip(values, COLORS, LABELS)):
        ax.plot(x+(j-1)*.15, v, "o", color=color, label=label, markersize=3.7)
    ax.set_xticks(x, categories)
    ax.set_ylabel(ylabel)
    if ylim: ax.set_ylim(*ylim)


def main():
    setup()
    summary = json.loads((DATA / "summary.json").read_text())
    models = summary["models"]
    finals = [models[a]["features"]["mean_final"] for a in ARMS]
    history = json.loads((DATA / "training_history.json").read_text())
    fig, ax = panels()
    for group, color in (("train", COLORS[2]), ("val", COLORS[0])):
        points = [r for r in history[group] if r.get(group + "/loss") is not None]
        x = np.array([r["global_step"] for r in points]) / 1e6
        y = np.array([r[group + "/loss"] for r in points])
        ax[0].plot(x, y, color=color, label=group, linewidth=.65)
        keep = x >= 1.08
        ax[1].plot(x[keep], y[keep], color=color, label=group, linewidth=.65)
    for a in ax:
        a.set_xlabel("Optimizer steps (millions)"); a.set_ylabel("Masked latent MSE")
    ax[0].set_yscale("log"); ax[0].legend(frameon=False)
    for x in (1.1, 1.15, 1.2): ax[1].axvline(x,color="#A4A9AC",linestyle="--",linewidth=.5)
    ax[1].set_title("b  Dashed lines: scheduled promotions",loc="left")
    save(fig,"training_loss")

    fig, ax = panels()
    legacy_names = ("random_init", "mae_666k", *ARMS)
    for name, label, color, style in zip(legacy_names, ("Random", "MAE ancestor", *LABELS),
                                      ("#B0B0B0", "#B88746", *COLORS), (":", "--", "-", "--", "-")):
        old = json.loads((DATA / f"legacy_{name}.json").read_text())["layers"]
        layers = ["embed", *map(str, range(1,26)), "final"]
        for a, metric in zip(ax, ("auc_maccs_macro", "knn_top1_tanimoto")):
            a.plot(range(27), [old[l]["mean"][metric] for l in layers], style, color=color,label=label)
    ax[0].set_ylabel("MACCS macro AUROC"); ax[1].set_ylabel("Nearest-neighbor Morgan Tanimoto")
    for a in ax: a.set_xlabel("Encoder depth"); a.set_xticks([0,5,10,15,20,26],["Embed","5","10","15","20","Final"])
    ax[0].legend(frameon=False,ncol=2)
    save(fig,"legacy_depth")

    fig, ax = panels(3)
    for a, metric, title in zip(ax,("maccs_auc","fg_auc","r2_mean"),("MACCS macro AUROC","Functional-group AUROC","Mean property R²")):
        grouped(a,[[f["ridge"]["1.0"][metric]] for f in finals],["Final mean"],title)
        for j,f in enumerate(finals):
            value=f["ridge"]["1.0"][metric]
            a.annotate(f"{value:.3f}",((j-1)*.15,value),xytext=(0,7),textcoords="offset points",ha="center",fontsize=6)
        a.margins(y=.2)
    ax[0].legend(frameon=False,loc="center right")
    save(fig,"chemical_probes")

    fig, ax = panels(3)
    variants=("raw","centered","standardized","whiten_0.01")
    for a,metric,title in zip(ax,("knn1_tanimoto","pair_partial_spearman","knn1_mass_window_5Da"),
                            ("Nearest-neighbor Tanimoto","Partial Spearman ρ","Tanimoto within ±5 Th")):
        for f,color,label in zip(finals,COLORS,LABELS):
            a.plot(range(4),[f["geometry"][v][metric] for v in variants],"o-",color=color,label=label,markersize=3)
        a.set_xticks(range(4),["Raw","Center","Scale","Whiten"],rotation=20)
        a.set_ylabel(title)
    ax[0].legend(frameon=False)
    save(fig,"geometry")

    fig, ax = panels(3)
    for a,metric,title in zip(ax,("effective_rank","pc1_fraction","mean_pair_cosine"),
                             ("Effective rank / 2,048 dimensions","Leading-PC variance fraction","Mean raw pair cosine")):
        grouped(a,[[models[arm]["features"][pool]["collapse"][metric] for pool in ("mean_final", "cls_final")] for arm in ARMS],
                ["Mean", "CLS"], title)
    ax[0].legend(frameon=False,loc="best")
    save(fig,"covariance")

    fig, ax = panels(3)
    perturb=("drop20","intensity_jitter","metadata_zero")
    for a,metric,title in zip(ax[:2],("mean_centered_cosine","self_retrieval_top1"),
                             ("Centered original–perturbed cosine","Self-retrieval accuracy")):
        grouped(a,[[models[arm]["robustness"][p][metric] for p in perturb] for arm in ARMS],
                ["Drop 20%","Jitter","Zero metadata"],title)
        a.tick_params(axis="x",rotation=20)
    grouped(ax[2],[[models[a]["features"][f]["replicate_retrieval"]["top1"] for f in ("mean_final","cls_final")] for a in ARMS],
            ["Mean","CLS"],"Repeat-spectrum retrieval accuracy",(0,1))
    ax[0].legend(frameon=False,loc="best")
    save(fig,"robustness")

    fig, ax = panels()
    for a,metric,title in zip(ax,("maccs_auc","r2_mean"),("MACCS macro AUROC","Mean property R²")):
        for f,color,label in zip(finals,COLORS,LABELS):
            values=[f["ridge"][str(v)] for v in (.01,.1,1.)]
            a.plot([v["train_molecules"] for v in values],[v[metric] for v in values],"o-",color=color,label=label,markersize=3)
        a.set_xscale("log");a.set_xlabel("Labeled training molecules");a.set_ylabel(title)
    ax[0].legend(frameon=False)
    save(fig,"label_efficiency")

    fig, ax = panels()
    for i,method in enumerate(("mean_linear","attentive")):
        for arm,color,label in zip((ARMS[0],ARMS[2]),(COLORS[0],COLORS[2]),(LABELS[0],LABELS[2])):
            old=json.loads((DATA/f"legacy_attentive_{arm}.json").read_text())["layers"]
            vals=np.array([[v["test"]["auc_maccs_macro"] for v in old[l][method]] for l in ("13","19","final")])
            ax[i].plot(range(3),vals.mean(1),"o-",color=color,label=label,markersize=3)
            for seed in range(2):ax[i].scatter(np.arange(3)+(seed-.5)*.05,vals[:,seed],color=color,marker="_",s=25,linewidths=.7)
        ax[i].set_xticks(range(3),["13","19","Final"]);ax[i].set_xlabel("Readout depth")
        ax[i].set_ylabel("MACCS macro AUROC");ax[i].set_title(chr(97+i)+"  "+("Mean + linear" if i==0 else "Learned attention"),loc="left")
        ax[i].set_ylim(.775,.81)
    ax[0].legend(frameon=False)
    save(fig,"attentive")

    fig, ax = panels()
    effects=summary["paired_differences"][ARMS[2]]
    for a,keys,labels in ((ax[0],[f+"_maccs_auc" for f in ("mean_final","mean_19","cls_final")],["Final mean","Block 19 mean","Final CLS"]),
                          (ax[1],[f"mean_final_{v}_knn1" for v in ("raw","standardized")],["Raw final mean","Scaled final mean"])):
        a.axvline(0,color="#999999",linestyle="--",linewidth=.5)
        for j,key in enumerate(keys):
            item=effects[key]; lo,hi=item["ci95"];d=item["delta"]
            a.errorbar(d,j,xerr=[[d-lo],[hi-d]],fmt="o",color=COLORS[2],markersize=4,capsize=2)
        a.set_yticks(range(len(keys)),labels);a.set_xlabel("Latest student minus initial teacher")
    ax[0].set_title("a  Δ MACCS AUROC (95% CI)",loc="left");ax[1].set_title("b  Δ nearest-neighbor Tanimoto (95% CI)",loc="left")
    save(fig,"paired_uncertainty")

    fig, ax = panels(1)
    props=list(finals[0]["ridge"]["1.0"]["r2"])
    grouped(ax[0],[[f["ridge"]["1.0"]["r2"][p] for p in props] for f in finals],
            ["logP","Rings","TPSA","Fraction sp³","Aromatic rings","H donors","H acceptors"],"Property R²")
    ax[0].legend(frameon=False,ncol=3)
    save(fig,"properties")

    fig, ax = panels()
    for a,metric,title in zip(ax,("pair_spearman","pair_partial_spearman"),("Spearman ρ with Morgan Tanimoto","Partial Spearman ρ")):
        grouped(a,[[models[arm]["features"][f]["geometry"][v][metric] for f,v in (("mean_final","raw"),("cls_final","raw"),("cls_final","standardized"))] for arm in ARMS],
                ["Raw mean","Raw CLS","Scaled CLS"],title)
    ax[0].legend(frameon=False)
    save(fig,"cls_exception")

    fig, ax = panels()
    variants=("raw", "standardized", "whiten_0.01")
    for a,key,title in zip(ax,("knn1_tanimoto","knn1_mass_window_5Da"),
                          ("Single-spectrum neighbor Tanimoto","Single-spectrum Tanimoto within ±5 Th")):
        for arm,color,label in zip(ARMS,COLORS,LABELS):
            a.plot(range(3),[models[arm]["single_spectrum"][v][key] for v in variants],"o-",color=color,label=label,markersize=3)
        a.set_xticks(range(3),["Raw", "Standardized", "Whitened"]);a.set_ylabel(title)
    ax[0].legend(frameon=False)
    save(fig,"single_spectrum")


if __name__ == "__main__":
    main()
