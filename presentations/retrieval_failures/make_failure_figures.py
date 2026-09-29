"""Render the measured charts for the retrieval failure-analysis deck.

Every value is read from results/retrieval_failure_analysis_20260929/*.json, the SWA score
logs in results/retrieval_ensemble_20260927/, or the validation-only fragment bounds there.
Charts are vector PDFs with live Liberation Sans text (fonttype 42), sized to their slide
slots, with transparent backgrounds for the Novogaia canvas. Molecule drawings for the
example slide are emitted as TikZ from RDKit 2D coordinates (vector, no raster).
"""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
ANALYSIS = REPO / "results/retrieval_failure_analysis_20260929"
ENSEMBLE = REPO / "results/retrieval_ensemble_20260927"
OUT = ROOT / "figures"
INK, MUTED, LINE, SOFT = "#0C0C0C", "#63676B", "#C6C9CC", "#E7E9EB"
CM = 1 / 2.54


def setup():
    for name in ("Regular", "Bold"):
        font_manager.fontManager.addfont(ROOT.parent / f"shared/assets/LiberationSans-{name}.ttf")
    plt.rcParams.update({
        "font.family": "Liberation Sans", "font.size": 7, "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK, "axes.linewidth": .5,
        "xtick.major.width": .5, "ytick.major.width": .5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "figure.facecolor": "none", "axes.facecolor": "none", "savefig.transparent": True,
        "legend.frameon": False, "legend.fontsize": 6.5, "axes.titlesize": 7.5, "axes.titleweight": "bold",
        "axes.titlelocation": "left"})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


def interval_plot(ax, labels, entries, val_entries=None, reference=None):
    x = np.arange(len(labels))
    means = np.array([e["mean"] for e in entries]) * 100
    err = np.array([[e["mean"] - e["lo"], e["hi"] - e["mean"]] for e in entries]).T * 100
    ax.bar(x, means, width=.62, color=INK, zorder=2)
    ax.errorbar(x, means, yerr=err, fmt="none", ecolor=MUTED, elinewidth=.7, capsize=1.6, zorder=3)
    if val_entries is not None:
        ax.scatter(x, [e["mean"] * 100 for e in val_entries], s=9, marker="o", facecolor="white",
                   edgecolor=INK, linewidth=.6, zorder=4, label="validation")
    if reference is not None:
        ax.axhline(reference, color=MUTED, lw=.6, ls=(0, (3, 2)), zorder=1)
    ax.set_xticks(x, [f"{label}\n{e['n']:,}" for label, e in zip(labels, entries)])
    ax.set_ylim(0, 90)
    ax.set_yticks([0, 25, 50, 75])


def fig_strata(a):
    t, v = a["test"]["strata"], a["val"]["strata"]
    overall = a["test"]["overall"]["ensemble"]["mean"] * 100
    fig, axes = plt.subplots(1, 4, figsize=(14.4 * CM, 4.7 * CM), gridspec_kw={"width_ratios": [6, 2, 2, 5]})
    groups = [
        ("Parent mass (Da)", ["mass 200-300", "mass 300-400", "mass 400-500", "mass 500-600", "mass 600-700", "mass 700-5000"],
         ["200–\n300", "300–\n400", "400–\n500", "500–\n600", "600–\n700", ">700\n"]),
        ("Adduct", ["[M+H]+", "[M+Na]+"], ["[M+H]$^+$\n", "[M+Na]$^+$\n"]),
        ("Instrument", ["instrument Orbitrap", "instrument QTOF"], ["Orbitrap\n", "QTOF\n"]),
        ("Peaks per spectrum", ["peaks 1-5", "peaks 6-10", "peaks 11-20", "peaks 21-40", "peaks 41-1000"],
         ["1–5\n", "6–10\n", "11–20\n", "21–40\n", ">40\n"])]
    for ax, (title, keys, labels) in zip(axes, groups):
        interval_plot(ax, labels, [t[k] for k in keys], [v[k] for k in keys], overall)
        ax.set_title(title)
        ax.tick_params(axis="x", labelsize=6)
    axes[0].set_ylabel("Test Hit@1 (%)")
    for ax in axes[1:]:
        ax.set_yticklabels([])
    axes[0].text(5.4, overall + 2, f"all {overall:.1f}%", ha="right", fontsize=5.8, color=MUTED)
    axes[3].legend(loc="upper left", handletextpad=.2, borderaxespad=0)
    fig.subplots_adjust(left=.068, right=.995, bottom=.25, top=.88, wspace=.12)
    save(fig, "strata")


def fig_shift(a):
    d = a["distribution"]
    labels = ["<200", "200–300", "300–400", "400–500", "500–600", "600–700", ">700"]
    fig, axes = plt.subplots(1, 2, figsize=(14.4 * CM, 3.9 * CM), gridspec_kw={"width_ratios": [3, 1.35]})
    x = np.arange(len(labels))
    for offset, split, color in ((-.2, "train", LINE), (.2, "test", INK)):
        axes[0].bar(x + offset, np.array(d[split]["mass_bins"]) * 100, width=.38, color=color, label=split)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Spectra (%)")
    axes[0].set_title("Parent-mass distribution (Da)")
    axes[0].legend(loc="upper right")
    shares = [("[M+Na]$^+$", "sodium"), ("QTOF", "qtof")]
    x2 = np.arange(len(shares))
    for offset, split, color in ((-.2, "train", LINE), (.2, "test", INK)):
        axes[1].bar(x2 + offset, [d[split][k] * 100 for _, k in shares], width=.38, color=color)
    axes[1].set_xticks(x2, [s for s, _ in shares])
    axes[1].set_title("Acquisition share")
    axes[1].set_ylim(0, 30)
    fig.subplots_adjust(left=.07, right=.995, bottom=.15, top=.87, wspace=.28)
    save(fig, "shift")


def fig_forest(a):
    names = list(a["test"]["logistic"])
    labels = ["Parent mass (+100 Da)", "[M+Na]$^+$ adduct", "QTOF instrument", "Peaks (×2)", "Pool size (×2)",
              "Train similarity (+0.1)", "Max decoy similarity (+0.1)", "Rotatable bonds (+5)"]
    fig, ax = plt.subplots(figsize=(8.3 * CM, 5.0 * CM))
    y = np.arange(len(names))[::-1]
    for split, marker, face, dy in (("test", "s", INK, .12), ("val", "o", "white", -.12)):
        e = a[split]["logistic"]
        ors = np.array([e[k]["odds_ratio"] for k in names])
        lo = np.array([e[k]["lo"] for k in names])
        hi = np.array([e[k]["hi"] for k in names])
        ax.errorbar(ors, y + dy, xerr=[ors - lo, hi - ors], fmt=marker, ms=3.2, mfc=face, mec=INK, mew=.6,
                    ecolor=MUTED, elinewidth=.7, capsize=1.4, label=split)
    ax.axvline(1, color=MUTED, lw=.6, ls=(0, (3, 2)))
    ax.set_xscale("log")
    ax.set_xticks([.4, .6, 1, 1.5, 2], ["0.4", "0.6", "1", "1.5", "2"])
    ax.minorticks_off()
    ax.set_yticks(y, labels)
    ax.set_xlabel("Odds ratio of a correct top-1 (log scale)")
    ax.legend(loc="upper left", handletextpad=.2, bbox_to_anchor=(0, 1.02))
    fig.subplots_adjust(left=.40, right=.98, bottom=.19, top=.98)
    save(fig, "forest")


def fig_decoy(a):
    t, v = a["test"]["strata"], a["val"]["strata"]
    keys = ["max decoy similarity 0.0-0.5", "max decoy similarity 0.5-0.7", "max decoy similarity 0.7-0.9",
            "max decoy similarity 0.9-1.0"]
    fig, axes = plt.subplots(1, 2, figsize=(14.4 * CM, 4.4 * CM), gridspec_kw={"width_ratios": [1, 1]})
    interval_plot(axes[0], ["<0.5", "0.5–0.7", "0.7–0.9", "≥0.9"], [t[k] for k in keys], [v[k] for k in keys],
                  a["test"]["overall"]["ensemble"]["mean"] * 100)
    axes[0].set_title("Hit@1 by most similar decoy in the pool")
    axes[0].set_xlabel("Max Tanimoto(truth, decoy), Morgan r=2")
    axes[0].set_ylabel("Test Hit@1 (%)")
    axes[0].legend(loc="upper right")
    keys = ["train similarity 0.0-0.4", "train similarity 0.4-0.6", "train similarity 0.6-0.8", "train similarity 0.8-1.0"]
    interval_plot(axes[1], ["<0.4", "0.4–0.6", "0.6–0.8", "≥0.8"], [t[k] for k in keys], [v[k] for k in keys],
                  a["test"]["overall"]["ensemble"]["mean"] * 100)
    axes[1].set_title("Hit@1 by nearest training molecule")
    axes[1].set_xlabel("Max Tanimoto(truth, train molecule)")
    axes[1].set_yticklabels([])
    fig.subplots_adjust(left=.075, right=.995, bottom=.3, top=.88, wspace=.08)
    save(fig, "decoy")


def fig_misses(a):
    m = a["test"]["misses"]
    fig, axes = plt.subplots(1, 2, figsize=(14.4 * CM, 4.4 * CM), gridspec_kw={"width_ratios": [1, 1.5]})
    labels = ["2", "3–5", "6–20", ">20"]
    values = [m["truth_rank_share"][k] * 100 for k in labels[:1] + ["3-5", "6-20", ">20"]]
    axes[0].bar(labels, values, color=INK, width=.62)
    for i, value in enumerate(values):
        axes[0].text(i, value + 1.2, f"{value:.0f}%", ha="center", fontsize=6.5)
    axes[0].set_ylim(0, 42)
    axes[0].set_ylabel("Test misses (%)")
    axes[0].set_xlabel("Rank of the true molecule")
    axes[0].set_title("Where the truth lands on a miss")
    hist = np.array(m["top1_truth_tanimoto_hist"]) / m["count"] * 100
    edges = np.linspace(0, 1, 11)
    axes[1].bar(edges[:-1] + .05, hist, width=.092, color=[INK if e >= .7 - 1e-9 else LINE for e in edges[:-1]])
    axes[1].set_xlim(0, 1.02)
    axes[1].set_xticks(edges[::2], [f"{e:.1f}" for e in edges[::2]])
    axes[1].set_xlabel("Tanimoto(predicted top-1, truth), Morgan r=2")
    axes[1].set_title("How close the wrong answer is")
    axes[1].text(.985, max(hist) * .98, f"≥0.7: {m['top1_tanimoto_ge_0.7'] * 100:.0f}% of misses\nsame Murcko scaffold: "
                 f"{m['top1_same_murcko_scaffold'] * 100:.0f}%", ha="right", va="top", fontsize=6.3)
    fig.subplots_adjust(left=.075, right=.975, bottom=.22, top=.88, wspace=.22)
    save(fig, "misses")


def fig_fragment(a):
    f = a["test"]["fragment_evidence"]
    keys = ["<=500 Da", ">500 Da", "[M+H]+", "[M+Na]+", "QTOF"]
    labels = ["≤500 Da", ">500 Da", "[M+H]$^+$", "[M+Na]$^+$", "QTOF"]
    fig, axes = plt.subplots(1, 2, figsize=(14.4 * CM, 4.4 * CM), gridspec_kw={"width_ratios": [1.6, 1]})
    x = np.arange(len(keys))
    for offset, field, color, label in ((-.27, "truth_explained_hits", LINE, "truth, hits"),
                                        (0, "truth_explained_misses", MUTED, "truth, misses"),
                                        (.27, "top1_explained_misses", INK, "wrong top-1, misses")):
        axes[0].bar(x + offset, [f[k][field] * 100 for k in keys], width=.26, color=color, label=label)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Intensity explained (%)")
    axes[0].set_title("GLACIER fragment ions explain the spectrum")
    axes[0].legend(loc="upper right", ncol=3, columnspacing=.8, handlelength=1, bbox_to_anchor=(1, 1.02))
    values = [f[k]["misses_top1_explains_at_least_truth"] * 100 for k in keys]
    axes[1].bar(x, values, color=INK, width=.62)
    for i, value in enumerate(values):
        axes[1].text(i, value + 1.5, f"{value:.0f}", ha="center", fontsize=6.3)
    axes[1].set_xticks(x, labels, fontsize=6)
    axes[1].set_ylim(0, 100)
    axes[1].set_title("Wrong top-1 ≥ truth (% of misses)", fontsize=7)
    fig.subplots_adjust(left=.075, right=.995, bottom=.15, top=.85, wspace=.2)
    save(fig, "fragment")


def fig_bounds(a):
    cap = a["proposal_cap_bound"]
    caps = cap["caps"]
    labels = ["32", "64", "128", "256", "all"]
    fig, axes = plt.subplots(1, 2, figsize=(14.4 * CM, 4.0 * CM))
    for group, style in (("all", "-"), ("mass>500", (0, (3, 1.6)))):
        axes[0].plot(range(len(caps)), [cap[group][str(c)]["truth_coverage"] * 100 for c in caps], color=INK, lw=1,
                     ls=style, marker="o", ms=2.6, label="all queries" if group == "all" else ">500 Da")
    axes[0].set_xticks(range(len(caps)), labels)
    axes[0].set_ylim(30, 70)
    axes[0].set_xlabel("GLACIER proposals kept per molecule")
    axes[0].set_ylabel("Truth intensity explained (%)")
    axes[0].set_title("More proposals: little new coverage")
    axes[0].legend(loc="lower right")
    for field, style, label in (("coverage_hit1", (0, (3, 1.6)), "coverage ranker"),
                                ("weighted_hit1", "-", "plausibility-weighted ranker")):
        axes[1].plot(range(len(caps)), [cap["all"][str(c)][field] * 100 for c in caps], color=INK, lw=1, ls=style,
                     marker="o", ms=2.6, label=label)
    enum = a["enumeration_bound"]["all"]["explained_intensity_ranker_hit1"]["enumerated"] * 100
    axes[1].axhline(enum, color=MUTED, lw=.6, ls=(0, (1, 1.5)))
    axes[1].text(4, enum + .6, f"exhaustive 3-cut enumeration {enum:.1f}%", ha="right", fontsize=6, color=MUTED)
    axes[1].set_xticks(range(len(caps)), labels)
    axes[1].set_xlabel("GLACIER proposals kept per molecule")
    axes[1].set_ylabel("Fragment-only Hit@1 (%)")
    axes[1].set_title("…and no better ranking")
    axes[1].text(0, 25, "plausibility-weighted", fontsize=6.3)
    axes[1].text(4, 11.6, "coverage only", fontsize=6.3, ha="right")
    axes[1].set_ylim(0, 31)
    fig.subplots_adjust(left=.065, right=.995, bottom=.22, top=.88, wspace=.26)
    save(fig, "bounds")


def swa_hit1(member):
    text = (ENSEMBLE / f"swa_{member}.log").read_text()
    values = {}
    for split in ("val", "test"):
        match = re.search(rf"^{split} (\{{.*\}})", text, re.M)
        values[split] = json.loads(match.group(1))["hit1"] * 100
    return values


def fig_interventions():
    runs = [("wd05", "Weight decay 0.05"), ("wd05_do2", "Weight decay + dropout 0.2"), ("do3", "Dropout 0.3"),
            ("attn", "Attention readout"), ("glob", "Global spectrum–molecule term"), ("attn_glob", "Attention + global"),
            ("L19", "Encoder layer 19 tokens"), ("L13", "Encoder layer 13 tokens"), ("neg24", "24 pool negatives"),
            ("neg24_wd05_do2", "24 negatives + wd + dropout")]
    control = swa_hit1("pt407000")
    deltas = {name: {s: swa_hit1(f"pt407000_{key}")[s] - control[s] for s in control} for key, name in runs}
    fig, ax = plt.subplots(figsize=(8.6 * CM, 5.3 * CM))
    y = np.arange(len(runs))[::-1]
    ax.barh(y + .17, [deltas[n]["val"] for _, n in runs], height=.32, color=LINE, label="validation")
    ax.barh(y - .17, [deltas[n]["test"] for _, n in runs], height=.32, color=INK, label="test")
    ax.axvline(0, color=INK, lw=.5)
    ax.set_yticks(y, [n for _, n in runs], fontsize=6.3)
    ax.set_xlim(-2.2, 2.2)
    ax.set_xlabel("Δ Hit@1 vs control (percentage points)")
    ax.legend(loc="lower right", handlelength=1)
    fig.subplots_adjust(left=.40, right=.98, bottom=.17, top=.99)
    save(fig, "interventions")
    return control, deltas


def fig_agreement(a, d):
    agree = a["test"]["member_agreement"]
    fig, axes = plt.subplots(1, 2, figsize=(14.4 * CM, 3.6 * CM), gridspec_kw={"width_ratios": [1, 1.25]})
    parts = [("all 15 correct", agree["all_members_hit"], INK), ("members disagree", agree["split"], MUTED),
             ("all 15 wrong", agree["all_members_miss"], LINE)]
    left = 0
    for label, share, color in parts:
        axes[0].barh(0, share * 100, left=left, color=color, height=.5)
        axes[0].text(left + share * 50, 0, f"{label}\n{share * 100:.0f}%", ha="center", va="center", fontsize=6.3,
                     color="white" if color != LINE else INK)
        left += share * 100
    axes[0].set_xlim(0, 100)
    axes[0].set_yticks([])
    axes[0].spines["left"].set_visible(False)
    axes[0].set_xlabel("Test spectra (%)")
    axes[0].set_title("Agreement of the 15 ensemble members")
    loss = d["loss_share"]
    keys = [">500 Da", "[M+Na]+", "QTOF", "<=5 peaks", "max decoy similarity >=0.7"]
    labels = [">500 Da", "[M+Na]$^+$", "QTOF", "≤5 peaks", "decoy sim. ≥0.7"]
    x = np.arange(len(keys))
    axes[1].bar(x, [loss[k]["counterfactual_hit1"] * 100 for k in keys], color=LINE, width=.62,
                label="if stratum matched the rest")
    axes[1].axhline(d["hit1"] * 100, color=INK, lw=.8, label=f"current {d['hit1'] * 100:.1f}%")
    axes[1].axhline(50, color=MUTED, lw=.6, ls=(0, (3, 2)))
    axes[1].text(2, 50.6, "50%", fontsize=6, color=MUTED, ha="center")
    axes[1].set_xticks(x, labels, fontsize=6)
    axes[1].set_ylim(40, 72)
    axes[1].set_ylabel("Test Hit@1 (%)")
    axes[1].set_title("Counterfactual overall Hit@1")
    axes[1].legend(loc="upper left", handlelength=1)
    fig.subplots_adjust(left=.02, right=.995, bottom=.24, top=.86, wspace=.18)
    save(fig, "agreement")


def tikz_molecule(smiles, name, width, height):
    """TikZ bonds/atom labels from RDKit 2D coordinates, fitted and centred in a width x height cm box."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smiles)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    AllChem.Compute2DCoords(mol)
    xy = np.array([[p.x, p.y] for p in (mol.GetConformer().GetAtomPosition(i) for i in range(mol.GetNumAtoms()))])
    xy -= xy.min(0)
    scale = min(width / max(xy[:, 0].max(), 1e-6), height / max(xy[:, 1].max(), 1e-6))
    xy *= scale
    xy += (np.array([width, height]) - xy.max(0)) / 2
    lines = [f"% {name}: {smiles}", "\\begin{scope}[line width=.45pt,line cap=round]"]
    labelled = {a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() != "C"}
    shrink = .12 * scale / .8
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        p, q = xy[i].copy(), xy[j].copy()
        direction = (q - p) / np.linalg.norm(q - p)
        if i in labelled:
            p = p + direction * min(shrink, .35 * np.linalg.norm(q - p))
        if j in labelled:
            q = q - direction * min(shrink, .35 * np.linalg.norm(q - p))
        lines.append(f"\\draw ({p[0]:.3f},{p[1]:.3f})--({q[0]:.3f},{q[1]:.3f});")
        if bond.GetBondTypeAsDouble() >= 2:
            normal = np.array([-direction[1], direction[0]]) * .09 * scale / .8
            a, b = p + normal + direction * .1 * scale / .8, q + normal - direction * .1 * scale / .8
            lines.append(f"\\draw ({a[0]:.3f},{a[1]:.3f})--({b[0]:.3f},{b[1]:.3f});")
    for atom in mol.GetAtoms():
        if atom.GetIdx() in labelled:
            label = atom.GetSymbol() + ("H" if atom.GetTotalNumHs() == 1 else f"H$_{atom.GetTotalNumHs()}$" if atom.GetTotalNumHs() > 1 else "")
            x, y = xy[atom.GetIdx()]
            lines.append(f"\\node[inner sep=.2pt,font=\\figurefont\\fontsize{{4.6}}{{5}}\\selectfont] at ({x:.3f},{y:.3f}) {{{label}}};")
    lines.append("\\end{scope}")
    return "\n".join(lines), scale


def molecules():
    examples = json.loads((ANALYSIS / "examples.json").read_text())
    for key, example in examples.items():
        slug = key.replace(" ", "_").replace("-", "_")
        for role in ("truth", "top1"):
            code, scale = tikz_molecule(example[f"{role}_smiles"], f"{key} {role}", 4.3, 1.62)
            (OUT / f"mol_{slug}_{role}.tex").write_text(f"% fitted in 4.3 x 1.7 cm; {scale:.3f} cm per RDKit unit\n{code}\n")


def pct(value, digits=1):
    return f"{value * 100:.{digits}f}"


def write_numbers(a, d, memo, interventions):
    """LaTeX macros for every number quoted on the slides (no hand transcription)."""
    t, v = a["test"], a["val"]
    st, tw, lg, fe = t["strata"], t["two_way"], t["logistic"], t["fragment_evidence"]
    cap, cond, enum = a["proposal_cap_bound"], a["conditioned_bound"], a["enumeration_bound"]
    dist = a["distribution"]
    ens = t["overall"]["ensemble"]
    n = {
        "TestHit": pct(ens["mean"], 2), "TestLo": pct(ens["lo"]), "TestHi": pct(ens["hi"]),
        "ValHit": pct(v["overall"]["ensemble"]["mean"], 2), "SingleHit": pct(t["overall"]["single"]["mean"], 2),
        "TestN": f"{t['n']:,}", "TestMol": f"{ens['molecules']:,}", "ValN": f"{v['n']:,}",
        "ValMol": f"{v['overall']['ensemble']['molecules']:,}", "Misses": f"{t['misses']['count']:,}",
        "ToFifty": f"{d['spectra_to_50pct']:,}", "Random": pct(t["random_hit1"]),
        "MassLow": pct(st["mass 300-400"]["mean"]), "MassHigh": pct(d["loss_share"][">500 Da"]["hit1"]),
        "MassTop": pct(st["mass 700-5000"]["mean"]), "Proton": pct(st["[M+H]+"]["mean"]),
        "Sodium": pct(st["[M+Na]+"]["mean"]), "Orbitrap": pct(st["instrument Orbitrap"]["mean"]),
        "Qtof": pct(st["instrument QTOF"]["mean"]), "FewPeaks": pct(st["peaks 1-5"]["mean"]),
        "ManyPeaks": pct(st["peaks 21-40"]["mean"]),
        "SmallProton": pct(tw["<=500 Da, [M+H]+"]["mean"]), "SmallSodium": pct(tw["<=500 Da, [M+Na]+"]["mean"]),
        "LargeProton": pct(tw[">500 Da, [M+H]+"]["mean"]), "LargeSodium": pct(tw[">500 Da, [M+Na]+"]["mean"]),
        "SmallOrbi": pct(tw["<=500 Da, Orbitrap"]["mean"]), "SmallQtof": pct(tw["<=500 Da, QTOF"]["mean"]),
        "LargeOrbi": pct(tw[">500 Da, Orbitrap"]["mean"]), "LargeQtof": pct(tw[">500 Da, QTOF"]["mean"]),
        "SmallProtonN": f"{tw['<=500 Da, [M+H]+']['n']:,}", "SmallSodiumN": f"{tw['<=500 Da, [M+Na]+']['n']:,}",
        "LargeProtonN": f"{tw['>500 Da, [M+H]+']['n']:,}", "LargeSodiumN": f"{tw['>500 Da, [M+Na]+']['n']:,}",
        "SmallOrbiN": f"{tw['<=500 Da, Orbitrap']['n']:,}", "SmallQtofN": f"{tw['<=500 Da, QTOF']['n']:,}",
        "LargeOrbiN": f"{tw['>500 Da, Orbitrap']['n']:,}", "LargeQtofN": f"{tw['>500 Da, QTOF']['n']:,}",
        "OrMass": f"{lg['parent mass (per 100 Da)']['odds_ratio']:.2f}", "OrSodium": f"{lg['[M+Na]+']['odds_ratio']:.2f}",
        "OrQtof": f"{lg['QTOF']['odds_ratio']:.2f}", "OrPeaks": f"{lg['log2 peaks']['odds_ratio']:.2f}",
        "OrDecoy": f"{lg['max decoy similarity (per 0.1)']['odds_ratio']:.2f}",
        "OrTrain": f"{lg['train similarity (per 0.1)']['odds_ratio']:.2f}",
        "OrPool": f"{lg['log2 pool size']['odds_ratio']:.2f}",
        "TrainMedian": f"{dist['train']['median_mass']:.0f}", "TestMedian": f"{dist['test']['median_mass']:.0f}",
        "TrainLarge": pct(sum(dist["train"]["mass_bins"][4:])), "TestLarge": pct(sum(dist["test"]["mass_bins"][4:])),
        "TrainSodium": pct(dist["train"]["sodium"]), "TestSodium": pct(dist["test"]["sodium"]),
        "TrainQtof": pct(dist["train"]["qtof"]), "TestQtof": pct(dist["test"]["qtof"]),
        "DecoyLow": pct(st["max decoy similarity 0.0-0.5"]["mean"]), "DecoyMid": pct(st["max decoy similarity 0.7-0.9"]["mean"]),
        "DecoyTop": pct(st["max decoy similarity 0.9-1.0"]["mean"]),
        "DecoyShare": pct(d["loss_share"]["max decoy similarity >=0.7"]["spectra_share"]),
        "DecoyMissShare": pct(d["loss_share"]["max decoy similarity >=0.7"]["miss_share"]),
        "TrainSimTop": pct(st["train similarity 0.8-1.0"]["mean"]),
        "TrainSimNoDecoy": pct(d["train_vs_decoy_similarity"]["hit1_train_ge_0.8_by_decoy"]["decoy<0.7"]),
        "TrainSimNoDecoyN": f"{d['train_vs_decoy_similarity']['hit1_train_ge_0.8_by_decoy']['n_decoy<0.7']:,}",
        "TrainSimDecoy": pct(d["train_vs_decoy_similarity"]["hit1_train_ge_0.8_by_decoy"]["decoy>=0.7"]),
        "TrainSimDecoyN": f"{d['train_vs_decoy_similarity']['hit1_train_ge_0.8_by_decoy']['n_decoy>=0.7']:,}",
        "MemoMiss": pct(memo["test"]["share_of_misses_where_top1_is_train"]),
        "MemoPools": pct(memo["test"]["pools_with_a_train_decoy"]),
        "RankTwo": pct(t["misses"]["truth_rank_share"]["2"], 0),
        "RankFive": pct(t["misses"]["truth_rank_share"]["2"] + t["misses"]["truth_rank_share"]["3-5"], 0),
        "RankFar": pct(t["misses"]["truth_rank_share"][">20"], 0),
        "NearIsomer": pct(t["misses"]["top1_tanimoto_ge_0.7"], 0), "SameScaffold": pct(t["misses"]["top1_same_murcko_scaffold"], 0),
        "TopSimMedian": f"{t['misses']['top1_truth_tanimoto_median']:.2f}",
        "FragHits": pct(fe["all"]["truth_explained_hits"], 0), "FragMissTruth": pct(fe["all"]["truth_explained_misses"], 0),
        "FragMissTop": pct(fe["all"]["top1_explained_misses"], 0),
        "FragIndist": pct(fe["all"]["misses_top1_explains_at_least_truth"], 0),
        "FragStrict": pct(d["fragment_split_of_misses"]["top1_strictly_more"], 0),
        "FragTie": pct(d["fragment_split_of_misses"]["tie"], 0),
        "FragZero": pct(d["fragment_split_of_misses"]["tie_both_zero"], 0),
        "FragSodiumHits": pct(fe["[M+Na]+"]["truth_explained_hits"], 0),
        "FragLargeHits": pct(fe[">500 Da"]["truth_explained_hits"], 0),
        "CapCov": pct(cap["all"]["128"]["truth_coverage"]), "CapCovAll": pct(cap["all"]["100000"]["truth_coverage"]),
        "CapLargeCov": pct(cap["mass>500"]["128"]["truth_coverage"]), "CapLargeCovAll": pct(cap["mass>500"]["100000"]["truth_coverage"]),
        "CapWeighted": pct(cap["all"]["128"]["weighted_hit1"]), "CapWeightedAll": pct(cap["all"]["100000"]["weighted_hit1"]),
        "CapCoverage": pct(cap["all"]["128"]["coverage_hit1"]), "CapCoverageAll": pct(cap["all"]["100000"]["coverage_hit1"]),
        "EnumHit": pct(enum["all"]["explained_intensity_ranker_hit1"]["enumerated"]),
        "EnumCov": pct(enum["all"]["explained_intensity_truth"]["enumerated"]),
        "CondNaN": f"{cond['na_orbitrap']['n']}", "CondNaFixed": pct(cond["na_orbitrap"]["fixed"]["truth_coverage"]),
        "CondNaMatched": pct(cond["na_orbitrap"]["matched"]["truth_coverage"]),
        "CondNaWFixed": pct(cond["na_orbitrap"]["fixed"]["weighted_hit1"]),
        "CondNaWMatched": pct(cond["na_orbitrap"]["matched"]["weighted_hit1"]),
        "CondQN": f"{cond['h_qtof']['n']}", "CondQWFixed": pct(cond["h_qtof"]["fixed"]["weighted_hit1"]),
        "CondQWMatched": pct(cond["h_qtof"]["matched"]["weighted_hit1"]),
        "AllHit": pct(t["member_agreement"]["all_members_hit"], 0), "AllMiss": pct(t["member_agreement"]["all_members_miss"], 0),
        "Disagree": pct(t["member_agreement"]["split"], 0), "Oracle": pct(t["member_agreement"]["oracle_any_member_hit1"], 0),
        "CfLarge": pct(d["loss_share"][">500 Da"]["counterfactual_hit1"]), "CfSodium": pct(d["loss_share"]["[M+Na]+"]["counterfactual_hit1"]),
        "CfQtof": pct(d["loss_share"]["QTOF"]["counterfactual_hit1"]), "CfPeaks": pct(d["loss_share"]["<=5 peaks"]["counterfactual_hit1"]),
        "CfDecoy": pct(d["loss_share"]["max decoy similarity >=0.7"]["counterfactual_hit1"]),
        "LargeMissShare": pct(d["loss_share"][">500 Da"]["miss_share"], 0),
        "LargeShare": pct(d["loss_share"][">500 Da"]["spectra_share"], 0),
        "ModeIsomer": pct(d["miss_modes"]["near-isomer top-1 (Tanimoto >= 0.7)"], 0),
        "ModePeaks": pct(d["miss_modes"]["<= 5 peaks"], 0),
        "ModeFrag": pct(d["miss_modes"]["top-1 explains >= truth intensity"], 0),
        "ModeAny": pct(d["miss_modes"]["any of the three"], 0), "ModeNone": pct(d["miss_modes"]["none of the three"], 0),
        "ControlVal": f"{interventions[0]['val']:.2f}", "ControlTest": f"{interventions[0]['test']:.2f}",
        "BestTestDelta": f"{max(x['test'] for x in interventions[1].values()):+.2f}",
    }
    lines = ["% Generated by make_failure_figures.py from results/retrieval_failure_analysis_20260929; do not edit."]
    lines += [f"\\newcommand{{\\fn{k}}}{{{value}}}" for k, value in n.items()]
    (OUT / "numbers.tex").write_text("\n".join(lines) + "\n")


def main():
    setup()
    a = json.loads((ANALYSIS / "analysis.json").read_text())
    d = json.loads((ANALYSIS / "decomposition.json").read_text())
    fig_strata(a)
    fig_shift(a)
    fig_forest(a)
    fig_decoy(a)
    fig_misses(a)
    fig_fragment(a)
    fig_bounds(a)
    control, deltas = fig_interventions()
    fig_agreement(a, d)
    molecules()
    memo = json.loads((ANALYSIS / "memorization.json").read_text())
    write_numbers(a, d, memo, (control, deltas))
    (OUT / "interventions.json").write_text(json.dumps({"control": control, "deltas": deltas}, indent=2) + "\n")
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
