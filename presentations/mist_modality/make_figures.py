"""Build the MIST-modality slide plots from the measured JSON inputs in research/."""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


HERE = Path(__file__).resolve().parent
RESEARCH = HERE / "research"
OUT = HERE / "figures"
INK, MUTED, TEAL, OCHRE, LINE = "#0C0C0C", "#63676B", "#087F8C", "#A86B17", "#C6C9CC"
HEADS = [("control", "Control (no MIST)", INK, "o"), ("mist_h0fp", "MIST h0 + fp", TEAL, "D"),
         ("mist_h0", "MIST h0", OCHRE, "s"), ("mist_fp", "MIST fp", MUTED, "^")]


def load(name: str) -> dict:
    return json.loads((RESEARCH / f"{name}.json").read_text())


def style_axis(ax: plt.Axes, grid_y: bool = False) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=3)
    ax.tick_params(axis="y", labelcolor=INK)
    ax.set_axisbelow(True)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", facecolor="#F6F7F8")
    plt.close(fig)


def quality() -> None:
    """MIST fingerprint agreement with the true Morgan fingerprint, by MassSpecGym fold."""
    q = load("fusion_sel4602")["mist_quality"]
    folds = [("train_sample", "Train fold\n(20,000 sampled)"), ("val", "Validation\n(5,000 subset)"),
             ("test", "Test\n(17,556)")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.5), sharey=True)
    for ax, key, title in zip(axes, ["tanimoto_mean", "cosine_mean"],
                              ["Tanimoto (bits at p ≥ 0.187) ↑", "Cosine (probabilities) ↑"], strict=True):
        values = [q[f][key] for f, _ in folds]
        colors = [OCHRE, TEAL, TEAL]
        ax.barh(range(3), values, height=0.5, color=colors)
        for i, v in enumerate(values):
            ax.annotate(f"{v:.3f}", (v, i), xytext=(6, 0), textcoords="offset points", va="center", fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_title(title, loc="left", fontsize=11, pad=12)
        ax.set_xlabel("Mean over spectra (dimensionless)")
        style_axis(ax)
    axes[0].set_yticks(range(3), [label for _, label in folds])
    axes[0].invert_yaxis()
    fig.subplots_adjust(left=0.14, right=0.97, bottom=0.18, top=0.86, wspace=0.18)
    save(fig, "quality")


def fusion_curve() -> None:
    """Validation Hit@1 of z(base) + w z(MIST) for the three similarity definitions."""
    report = load("fusion_sel4602")
    curves = report["val_curves"]
    weights = curves["weights"]
    fig, ax = plt.subplots(figsize=(11.5, 3.6))
    for kind, label, color, style in (("bernoulli", "Bernoulli log-likelihood", TEAL, "-"),
                                      ("tanimoto", "Tanimoto", OCHRE, "--"), ("cosine", "Cosine", MUTED, ":")):
        ax.plot(weights, [100 * v for v in curves[kind]], color=color, linestyle=style, linewidth=1.6, label=label)
    w = report["selected"]["weight"]
    best = 100 * report["fusion"]["val"]["hit1"]
    ax.scatter([w], [best], color=TEAL, s=45, zorder=4)
    ax.annotate(f"selected: w = {w:g}, {best:.2f}%", (w, best), xytext=(10, 6), textcoords="offset points",
                fontsize=10, color=TEAL)
    ax.axhline(100 * report["base"]["val"]["hit1"], color=INK, linewidth=0.8)
    ax.annotate(f"base model alone {100 * report['base']['val']['hit1']:.2f}%", (5.0, 100 * report["base"]["val"]["hit1"]),
                xytext=(0, -13), textcoords="offset points", ha="right", fontsize=10)
    ax.set_xscale("symlog", linthresh=0.2, linscale=1.6)
    ax.set_xticks([0, 0.05, 0.1, 0.2, 0.5, 1, 2, 5], ["0", "0.05", "0.1", "0.2", "0.5", "1", "2", "5"])
    ax.set_xlim(0, 5.2)
    ax.set_ylim(44, 60)
    ax.set_xlabel("MIST weight w (symmetric-log axis; linear below 0.2)")
    ax.set_ylabel("Validation Hit@1 (%)")
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    style_axis(ax)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.18, top=0.95)
    save(fig, "fusion_curve")


def hit_at_k() -> None:
    """Held-out test Hit@k for the base model, MIST alone and the validation-selected fusion."""
    report = load("fusion_sel4602")
    series = [("Base model (44.10% checkpoint)", report["base"]["test"], MUTED),
              ("MIST alone (Bernoulli)", report["mist_only"]["bernoulli"]["test"], OCHRE),
              ("Fusion, w = 0.9", report["fusion"]["test"], TEAL)]
    ks = [1, 5, 10, 20]
    fig, ax = plt.subplots(figsize=(11.5, 3.5))
    width = 0.26
    for j, (label, metrics, color) in enumerate(series):
        xs = [i + (j - 1) * width for i in range(len(ks))]
        values = [100 * metrics[f"hit{k}"] for k in ks]
        ax.bar(xs, values, width=width * 0.92, color=color, label=label)
        for x, v in zip(xs, values):
            ax.annotate(f"{v:.1f}", (x, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(range(len(ks)), [f"Hit@{k}" for k in ks])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Test spectra (%)")
    ax.legend(frameon=False, fontsize=10, loc="upper left", ncols=3, bbox_to_anchor=(0, 1.12))
    style_axis(ax)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.1, top=0.86)
    save(fig, "hit_at_k")


def head_curves() -> None:
    """Per-epoch validation Hit@1 and training loss of the four head-stage runs."""
    runs = load("head_curves")["runs"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6))
    for key, label, color, marker in HEADS:
        epochs = runs[f"head_{key}_s0"]
        xs = [e["epoch"] for e in epochs]
        axes[0].plot(xs, [100 * e["val_hit1"] for e in epochs], color=color, marker=marker, markersize=3.5,
                     linewidth=1.2, label=label)
        axes[1].plot(xs, [e["train_loss"] for e in epochs], color=color, marker=marker, markersize=3.5,
                     linewidth=1.2, label=label)
    axes[0].axhline(45.04, color=LINE, linewidth=1.0, linestyle="--")
    axes[0].annotate("original source head 45.04%", (24, 45.04), xytext=(0, -12), textcoords="offset points",
                     ha="right", fontsize=9, color=MUTED)
    axes[0].set_ylim(38, 46)
    axes[0].set_ylabel("Validation Hit@1 (%)")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Mean training InfoNCE loss (log scale)")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_xlim(-0.5, 24.5)
        style_axis(ax)
    axes[1].legend(frameon=False, fontsize=10, loc="upper right")
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.17, top=0.95, wspace=0.25)
    save(fig, "head_curves")


def head_fusion() -> None:
    """Test Hit@1 of each head-stage run alone and after the same MIST score fusion."""
    rows = [("Selected joint\n(44.10% model)", load("fusion_sel4602"))]
    rows += [(label.replace(" (no MIST)", "\n(no MIST)"), load(f"fusion_head_{key}")) for key, label, _, _ in HEADS]
    fig, ax = plt.subplots(figsize=(11.5, 3.5))
    for i, (label, report) in enumerate(rows):
        base, fused = 100 * report["base"]["test"]["hit1"], 100 * report["fusion"]["test"]["hit1"]
        ax.plot([base, fused], [i, i], color=LINE, linewidth=2.2, zorder=1)
        ax.scatter([base], [i], color=MUTED, s=40, zorder=2)
        ax.scatter([fused], [i], color=TEAL if i < 2 else OCHRE, marker="D", s=44, zorder=2)
        ax.annotate(f"{base:.2f}", (base, i), xytext=(-8, 0), textcoords="offset points", ha="right",
                    va="center", fontsize=10, color=MUTED)
        ax.annotate(f"{fused:.2f}  (w = {report['selected']['weight']:g})", (fused, i), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=10)
    ax.set_yticks(range(len(rows)), [label for label, _ in rows])
    ax.invert_yaxis()
    ax.set_xlim(36, 66)
    ax.set_xlabel("Test Hit@1 (%); circle = model alone, diamond = + MIST score fusion")
    style_axis(ax)
    ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.17, top=0.97)
    save(fig, "head_fusion")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    font_manager.fontManager.addfont(HERE.parent / "shared/assets/LiberationSans-Regular.ttf")
    plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 11,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.facecolor": "#F6F7F8", "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK})
    for plot in (quality, fusion_curve, hit_at_k, head_curves, head_fusion):
        plot()
    print(f"Built 5 vector figures from {RESEARCH.relative_to(HERE.parents[1])}")


if __name__ == "__main__":
    main()
