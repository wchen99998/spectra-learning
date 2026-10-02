"""Regenerate presentation figures from frozen measured inputs, without model fits.

Plot functions copied from scripts/analyze_encoder_chemistry_diffusion.py
SHA256 2f029c7b6d37d3d320a6152fe4b4f6bfcb118161295ec1b20d8f43089db76a2b. Only the font-path root changes.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
BASELINES = ("parent_1190k", "periodic_1209k")
CONFIGURATIONS = ((32, 0), (32, 1), (64, 0))
CONCEPTS = ("fg_amide", "fg_amine", "fg_hydroxyl", "fg_halide", "fg_nitrile", "fg_sulfonyl")

def configure_plots() -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    for style in ("Regular","Bold","Italic","BoldItalic"):
        font_manager.fontManager.addfont(str(ROOT.parent/"shared/assets"/f"LiberationSans-{style}.ttf"))
    matplotlib.rcParams.update({"font.family":"Liberation Sans","pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none"})

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

if __name__ == "__main__":
    data = json.loads((ROOT / "research/plot_inputs.json").read_text())
    plot_readouts(data["readout"], "diffusion_831k", ROOT / "figures")
    plot_sae(data["sae"], "diffusion_831k", ROOT / "figures")
