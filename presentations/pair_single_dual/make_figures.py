"""Build slide figures directly from the dated worklog's result tables."""

from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "docs/pair_single_dual_20261001.md"
OUT = HERE / "figures"
INK, MUTED, TEAL, OCHRE = "#0C0C0C", "#63676B", "#087F8C", "#A86B17"


def table(text: str, heading: str) -> list[list[str]]:
    """Read the first pipe table after an exact heading; carry blank arm cells."""
    section = text.split(heading, 1)[1]
    block = re.search(r"(?m)^\|.*(?:\n\|.*)*", section).group()
    rows = []
    previous_arm = ""
    for line in block.splitlines()[2:]:
        cells = [cell.strip().replace("**", "") for cell in line.strip("|").split("|")]
        if not cells[0]:
            cells[0] = previous_arm
        previous_arm = cells[0]
        rows.append(cells)
    return rows


def number(cell: str) -> float:
    return float(re.search(r"[−-]?\d+(?:\.\d+)?", cell).group().replace("−", "-"))


def style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="y", length=0, pad=7)
    ax.tick_params(axis="x", colors=MUTED, length=3)
    ax.set_axisbelow(True)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", facecolor="#F6F7F8")
    plt.close(fig)


def wave_one(text: str) -> None:
    rows = table(text, "### Per-arm means (± SD over 3 seeds)")
    labels = ["Single only", "Old joint", "Pair only", "Dual / no feedback",
              "Dual / bias", "Dual / bias + mean", "Dual / bias + IPA", "Dual / bias + gated"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)
    for ax, col, title, bounds in zip(
        axes, [3, 4, 5], ["Single MACCS AUC ↑", "Pair MACCS AUC ↑", "Single retrieval top-1 (%) ↑"],
        [(0.63, 0.69), (0.63, 0.72), (20, 29)], strict=True,
    ):
        for i, row in enumerate(rows):
            if row[col] == "—":
                continue
            color = TEAL if i == 6 else (INK if i >= 3 else MUTED)
            value = number(row[col])
            ax.scatter(value, i, color=color, marker="D" if i == 6 else "o", s=35, zorder=3)
            ax.annotate(f"{value:.3f}" if col != 5 else f"{value:.2f}",
                        (value, i), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=10, color=color)
        ax.set_title(title, loc="left", fontsize=10, pad=14)
        ax.set_xlim(*bounds)
        ax.set_xlabel("Test fold; mean across 3 seeds", fontsize=10)
        style_axis(ax)
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].invert_yaxis()
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.17, top=0.85, wspace=0.3)
    save(fig, "wave1")


def capacity(text: str) -> None:
    rows = table(text, "**Pair-only capacity controls (20k).**")
    labels = ["C128 / 6 layers", "C128 / 12 layers", "C128 / 18 layers",
              "C192 / 12 layers", "C256 / 6 layers"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.4), sharey=True)
    for ax, col, title, bounds in zip(
        axes, [1, 2], ["Pair-distogram CE ↓", "Pair MACCS AUC (test) ↑"],
        [(2.21, 2.30), (0.68, 0.725)], strict=True,
    ):
        for i, row in enumerate(rows):
            value = number(row[col])
            color = TEAL if i == 4 else MUTED
            ax.scatter(value, i, color=color, s=45)
            ax.annotate(f"{value:.3f}", (value, i), xytext=(7, 0),
                        textcoords="offset points", va="center", fontsize=10, color=color)
        ax.set_title(title, loc="left", fontsize=11, pad=15)
        ax.set_xlim(*bounds)
        ax.set_xlabel("Seed 0; 20k steps", fontsize=10)
        style_axis(ax)
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].invert_yaxis()
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.18, top=0.84, wspace=0.25)
    save(fig, "capacity")


def bin_size(text: str) -> None:
    rows = table(text, "### Results (seed 0, 20k steps, D256/C128)")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.5))
    settings = [("single_only", "Single only", MUTED, "o"),
                ("joint_bias", "Old joint", INK, "s"),
                ("dual_bias_mean", "Dual / bias + mean", TEAL, "D"),
                ("pair_only", "Pair only", OCHRE, "^")]
    for ax, col, title in zip(axes, [4, 5], ["Single MACCS AUC ↑", "Pair MACCS AUC ↑"], strict=True):
        for arm, label, color, marker in settings:
            selected = [r for r in rows if r[0] == arm and r[col] != "—"]
            if not selected:
                continue
            xs = [["2", "0.2", "0.1"].index(r[1]) for r in selected]
            ax.plot(xs, [number(r[col]) for r in selected], label=label, color=color,
                    marker=marker, linewidth=1.4, markersize=5)
        ax.set_title(title, loc="left", fontsize=11, pad=14)
        ax.set_xticks([0, 1, 2], ["2 Da", "0.2 Da", "0.1 Da"])
        ax.set_xlim(-0.15, 2.15)
        ax.set_ylim(0.64, 0.755)
        ax.set_ylabel("Test AUC")
        ax.set_xlabel("Bin width (categorical spacing)")
        style_axis(ax)
        ax.legend(frameon=False, fontsize=10, loc="lower right")
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.2, top=0.86, wspace=0.27)
    save(fig, "bins")


def masking(text: str) -> None:
    rows = table(text, "### Results (seed 0, 20k steps; deltas versus random-mask seed 0)")
    labels = ["Single only", "Old joint", "Pair only", "Dual / no feedback",
              "Dual / bias", "Dual / bias + mean", "Dual / bias + IPA", "Dual / bias + gated"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6), sharey=True)
    for ax, col, title, bounds in zip(
        axes, [1, 3], ["Change in single MACCS AUC", "Change in single top-1 (pp)"],
        [(-0.032, 0.005), (-4.5, 4)], strict=True,
    ):
        ax.axvline(0, color=MUTED, linewidth=0.7)
        for i, row in enumerate(rows):
            if row[col] == "—":
                continue
            value = number(row[col].split("(", 1)[1])
            color = TEAL if value > 0 else OCHRE
            ax.barh(i, value, height=0.46, color=color)
        ax.set_title(title, loc="left", fontsize=11, pad=14)
        ax.set_xlim(*bounds)
        ax.set_xlabel("Span + isotope minus random; test fold")
        style_axis(ax)
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].invert_yaxis()
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.18, top=0.85, wspace=0.24)
    save(fig, "masking")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    font_manager.fontManager.addfont(HERE.parent / "shared/assets/LiberationSans-Regular.ttf")
    plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 11,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.facecolor": "#F6F7F8", "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK})
    text = SOURCE.read_text()
    for plot in (wave_one, capacity, bin_size, masking):
        plot(text)
    print(f"Built 4 vector figures from {SOURCE.relative_to(HERE.parents[1])}")


if __name__ == "__main__":
    main()
