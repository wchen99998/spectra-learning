"""Render existing native layer metrics; no model fitting or test-based selection."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.analyze_encoder_chemistry_sae import sha256, write_json

ARMS = ("parent_1190k", "periodic_1209k")
POOLS = ("mean", "max", "cls")
METRICS = ("maccs_auroc", "element_auroc", "functional_group_auroc")
TITLES = ("MACCS fingerprints", "Elements", "Functional groups")
COLORS = ("#3c6e8f", "#bd6849")


def validation_best(records: list[dict], metric: str) -> dict:
    """Choose only by validation AUROC; fixed pool/depth order resolves ties."""
    eligible = [r for r in records if r["validation"][metric] is not None]
    if not eligible:
        raise ValueError("No supported validation target group")
    return min(eligible, key=lambda r: (-r["validation"][metric], POOLS.index(r["pool"]), r["depth"]))


def summarize(root: Path, atlases: dict) -> dict:
    report = {"schema": "native_layer_atlas_summary_v1", "supplemental": True,
        "scope": "Existing pooled projected ridge readouts and descriptive native token geometry; no new fits. Pooled performance is not a bound on token-level chemistry or an SAE result.",
        "projection": {"dimensions": 256, "seed": 20260930, "shared_across_checkpoints": True},
        "ridge_penalty_selection": "Existing per-target validation MSE over penalties 0.1,1,10,100,1000",
        "best_layer_selection": "Highest validation macro AUROC for the named target group, independently per pool; overall selection also chooses pool on validation. Ties use fixed mean/max/CLS order then shallower depth. Test scores are evaluated only after this selection.",
        "curve_interpretation": "All-depth test curves are descriptive; stars identify validation-selected depths, not test-best depths. No confidence intervals or multiple-comparison discoveries inferred from these curves.",
        "depth_mapping": {"0_to_25": "raw native states", "26": "final normalization", "27": "final encoder output"},
        "target_support_rule": "At least20 positives and negatives in train, at least5 each in validation and test; fixed shared targets. This supplemental rule differs from the primary token-readout comparison.",
        "token_geometry_scope": "Descriptive statistics across all6144 panel molecules; covariance effective rank after a shared256D projection over at most8 fixed quantile peak slots per molecule. No learned chemical labels in geometry. Spectrum norms/variance/cosines use all valid physical peaks.",
        "input_probe_scope": "m/z and transformed input intensity decoded by existing ridge models, train/validation by molecule and test7953 sampled physical tokens. Input reconstruction is not chemical fragment interpretation.",
        "source_sha256": sha256(Path(__file__)), "arms": {}}
    support = np.asarray(atlases[ARMS[0]]["target_support"], bool)
    report["target_counts"] = {"MACCS": int(support[:166].sum()), "elements": int(support[166:173].sum()),
                               "functional_groups": int(support[173:].sum())}
    for arm, atlas in atlases.items():
        selected = {}
        for pool in POOLS:
            records = [r for r in atlas["records"] if r["pool"] == pool]
            selected[pool] = {}
            for metric in METRICS:
                chosen = validation_best(records, metric)
                selected[pool][metric] = {"depth": chosen["depth"], "validation_auroc": chosen["validation"][metric],
                    "test_auroc": chosen["test"][metric],
                    "test_minus_nuisance_baseline": chosen["test"][metric] - atlas["nuisance_baseline"]["test"][metric]}
        overall = {}
        for metric in METRICS:
            chosen = validation_best(atlas["records"], metric)
            overall[metric] = {"pool": chosen["pool"], **selected[chosen["pool"]][metric]}
        report["arms"][arm] = {"atlas_sha256": sha256(root / arm / "atlas.json"),
            "bitwise_native_cache_parity": atlas["cache_parity"]["bitwise_equal"],
            "validation_selected_by_pool": selected, "validation_selected_overall": overall,
            "nuisance_only_baseline": {"features": atlas["nuisance_baseline"]["features"],
                "test_auroc": {metric: atlas["nuisance_baseline"]["test"][metric] for metric in METRICS}},
            "selected_depth_token_metrics": atlas["token_geometry"]}
    return report


def render(root: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    atlases = {arm: json.loads((root / arm / "atlas.json").read_text()) for arm in ARMS}
    for arm, atlas in atlases.items():
        if not atlas["cache_parity"]["bitwise_equal"] or not atlas["cache_parity"]["finite"]:
            raise ValueError("Atlas does not use qualified exact native caches")
        if {(r["pool"], r["depth"]) for r in atlas["records"]} != {(p, d) for p in POOLS for d in range(28)}:
            raise ValueError("Expected every native state for mean/max/CLS")
        if atlas["extraction_manifest"] != json.loads((root / arm / "extraction.json").read_text()):
            raise ValueError("Atlas extraction manifest differs from current qualified extraction")
    if atlases[ARMS[0]]["target_support"] != atlases[ARMS[1]]["target_support"]:
        raise ValueError("Checkpoint atlas support differs")
    report = summarize(root, atlases)
    output = root / "layer_atlas"
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", report)
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True, sharey="col", layout="constrained")
    for row, pool in enumerate(POOLS):
        for column, (metric, title) in enumerate(zip(METRICS, TITLES, strict=True)):
            axis = axes[row, column]
            baseline = atlases[ARMS[0]]["nuisance_baseline"]["test"][metric]
            for i, arm in enumerate(ARMS):
                records = sorted((r for r in atlases[arm]["records"] if r["pool"] == pool), key=lambda r: r["depth"])
                axis.plot([r["depth"] for r in records], [r["test"][metric] for r in records], color=COLORS[i], linewidth=1.9)
                best = report["arms"][arm]["validation_selected_by_pool"][pool][metric]
                axis.scatter([best["depth"]], [best["test_auroc"]], marker="*", s=105, color=COLORS[i], edgecolor="white", linewidth=.6, zorder=5)
            axis.axhline(baseline, color="#777777", linestyle="--", linewidth=1.1)
            axis.axvline(25.5, color=".85", linewidth=.8)
            axis.set_xticks([0, 5, 10, 15, 20, 25, 27])
            axis.spines[["top", "right"]].set_visible(False)
            axis.set_xlim(-.4, 27.4)
            if row == 0:
                axis.set_title(title)
            if column == 0:
                axis.set_ylabel(f"{pool.upper()} pooling\nTest macro AUROC")
            if row == 2:
                axis.set_xlabel("Native state index")
    handles = [Line2D([0], [0], color=COLORS[0], lw=2, label="Parent"),
               Line2D([0], [0], color=COLORS[1], lw=2, label="Latest"),
               Line2D([0], [0], color=".45", lw=1.1, ls="--", label="Nuisance-only baseline"),
               Line2D([0], [0], color=".3", marker="*", linestyle="none", markersize=9, label="Validation-selected layer")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("Supplemental pooled ridge atlas · shared 256D projection\nNative states 0–25; 26 final norm; 27 encoder output", fontsize=15)
    for extension in ("png", "pdf"):
        fig.savefig(figures / f"pooled_chemical_readouts.{extension}", dpi=180)
    plt.close(fig)
    panels = [("projected_token_effective_rank", "Projected token effective rank", "Effective rank (≤256)"),
              ("median_within_spectrum_variance_fraction", "Within-spectrum heterogeneity", "Median variance / total energy"),
              ("mean_valid_peak_cosine", "Within-spectrum token similarity", "Mean valid-peak pair cosine"),
              ("input_linear_r2_mz", "Linear m/z reconstruction", "Held-out token R²"),
              ("input_linear_r2_intensity", "Linear intensity reconstruction", "Held-out token R²"),
              ("median_token_l2", "Token magnitude", "Median spectrum mean token L2")]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    depths = [r["depth"] for r in atlases[ARMS[0]]["token_geometry"]]
    for axis, (metric, title, ylabel) in zip(axes.flat, panels, strict=True):
        for i, arm in enumerate(ARMS):
            records = atlases[arm]["token_geometry"]
            if [r["depth"] for r in records] != depths:
                raise ValueError("Token geometry states differ between checkpoints")
            axis.plot(np.arange(len(depths)), [r[metric] for r in records], marker="o", markersize=4,
                      color=COLORS[i], linewidth=1.8, label="Parent" if i == 0 else "Latest")
        axis.set(xticks=np.arange(len(depths)), xticklabels=["Norm" if d == 26 else "Out" if d == 27 else str(d) for d in depths],
                 title=title, ylabel=ylabel, xlabel="Selected native state (categorical spacing)")
        axis.spines[["top", "right"]].set_visible(False)
        if metric == "median_token_l2":
            axis.set_yscale("log")
        if metric.startswith("input_linear_r2"):
            axis.set_ylim(.45, 1.025)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Native physical-token atlas · geometry is descriptive; input decoding is not fragment chemistry\nGeometry: 6144 molecules · input R²: 7953 held-out sampled physical tokens", fontsize=14)
    for extension in ("png", "pdf"):
        fig.savefig(figures / f"native_token_geometry.{extension}", dpi=180)
    plt.close(fig)
    return report


if __name__ == "__main__":
    report = render(Path("artifacts/encoder_chemistry_20260930"))
    for arm in ARMS:
        print(arm, json.dumps(report["arms"][arm]["validation_selected_overall"]))
