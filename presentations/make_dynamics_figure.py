"""Render an exact analytic illustration of different learning speeds.

All modes start at the same learned fraction rho, rather than the same
absolute amplitude. The 89 x 43 mm artwork uses live 5-7 pt text and
0.25-1 pt vector strokes. Slide assets are transparent; journal exports
are white. No measured transformer training data are represented.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
FIGURES = ROOT / "figures"
BACKGROUND = "#FFFFFF"
INK = "#000000"
RHO = 0.015
ETA = 1.0
SNAPSHOT_TIME = 6.0
MODES = [
    (1.0, "-", "Stronger link"),
    (0.55, (0, (5, 2.3)), "Intermediate"),
    (0.2, (0, (1, 2.0)), "Weaker link"),
]


def learned_fraction(
    time: np.ndarray, sigma: float, rho: float = RHO, eta: float = ETA
) -> np.ndarray:
    """Return u(t)/sigma for du/dt = 2 eta u (sigma-u), u(0) = rho*sigma.

    Under aligned modes, balanced factorization and equal learning rates,
    the two-layer linear model gives this exact logistic solution.
    """
    return 1.0 / (1.0 + (1.0 / rho - 1.0) * np.exp(-2.0 * eta * sigma * time))


def main() -> None:
    font_path = ASSETS / "LiberationSans-Regular.ttf"
    font_manager.fontManager.addfont(font_path)
    font_family = font_manager.FontProperties(fname=font_path).get_name()

    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.size": 6,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )

    figure, ax = plt.subplots(figsize=(89 / 25.4, 43 / 25.4), facecolor=BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    figure.subplots_adjust(left=0.14, right=0.975, bottom=0.35, top=0.82)

    time = np.linspace(0.0, 20.0, 801)
    for sigma, linestyle, label in MODES:
        ax.plot(
            time,
            learned_fraction(time, sigma),
            color=INK,
            linewidth=0.85,
            linestyle=linestyle,
            label=label,
            solid_capstyle="round",
            dash_capstyle="round",
        )

    # A single comparison guide is distinct from background gridlines.
    ax.axvline(SNAPSHOT_TIME, color="#777777", linewidth=0.5, linestyle=(0, (3, 3)))
    ax.text(6.7, 0.45, "t = 6", fontsize=5.5, color=INK, ha="left", va="center")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.28),
        ncol=3,
        frameon=False,
        fontsize=5.5,
        handlelength=1.8,
        handletextpad=0.45,
        columnspacing=1.0,
        borderaxespad=0,
    )

    ax.set_xlim(0.0, 20.0)
    ax.set_ylim(0.0, 1.03)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_yticks([0, 0.5, 1.0], labels=["0", "0.5", "1"])
    ax.set_xlabel("Optimization time (a.u.)", labelpad=3, fontsize=6.5)
    ax.set_ylabel("Fraction learned (u/σ)", labelpad=5, fontsize=6.5)
    ax.tick_params(axis="both", labelsize=6, length=2.5, width=0.5, pad=2.5)
    ax.spines[["top", "right"]].set_visible(False)

    figure.text(
        0.14,
        0.085,
        "Analytic illustration · not measured transformer curves",
        fontsize=5.25,
        color=INK,
        ha="left",
    )
    figure.text(
        0.14,
        0.018,
        "σ = 1, 0.55, 0.2; ρ = 0.015; η = 1",
        fontsize=5.25,
        color=INK,
        ha="left",
    )

    ASSETS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg"):
        figure.savefig(ASSETS / f"dynamics-modes.{suffix}", transparent=True)
        figure.savefig(
            FIGURES / f"fig-09-dynamics.{suffix}",
            facecolor=BACKGROUND,
            transparent=False,
        )
    plt.close(figure)


if __name__ == "__main__":
    main()
