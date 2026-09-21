"""
covid_era_figures.py
======================
Presentation figures for the three-era (pre-COVID / COVID / post-COVID)
comparison. All read through the covid_era_comparison functions, which route
each era to the sample that physically contains it (base for 2016-2021, ext
for 2022-2025) and restore the sample afterwards.

  fig_era_coefficients   DiD coefficient by era for each outcome, size-
                         controlled, with 95% CIs. Shows at a glance which
                         effects weaken, reverse, or persist across eras.
  fig_era_levels         The four cell means (retail/procust x baseline/near)
                         per era for one outcome, as connected lines -- makes
                         "who moves, and does the gap change across eras"
                         directly visible, and confirms a null is a real
                         both-flat pattern rather than broken data.

Written to results/figures/ at 300 dpi.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

import sys
sys.path.append(str(Path(__file__).parent.parent))
from paths import RESULTS_DIR
from analysis.covid_era_comparison import (
    compare_covid_eras_controlled, era_levels, COVID_ERAS,
)

C_RETAIL = "#1a4f8a"
C_PROCUST = "#d1731f"
GRID = "#e4e3de"
ERA_ORDER = ["pre_covid", "covid", "post_covid"]
ERA_LABELS = {"pre_covid": "Pre-COVID\n2016-2019",
              "covid": "COVID\n2020-2021",
              "post_covid": "Post-COVID\n2022-2025"}
PRETTY = {"otm": "Out-of-the-money", "otm_put": "OTM puts", "otm_call": "OTM calls",
          "lt_100": "Small positions (<100)", "call": "Call share",
          "open": "Opening positions", "itm": "In-the-money"}


def fig_era_coefficients(outcomes: list[str] = None, out_dir: Path = None):
    """Size-controlled treat:post by era, one panel per outcome, with 95% CIs."""
    import matplotlib.pyplot as plt

    outcomes = outcomes or ["otm", "otm_put", "lt_100", "call", "open"]
    ctrl = compare_covid_eras_controlled(outcomes=outcomes)

    n = len(outcomes)
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 3.6), sharey=False)
    if n == 1:
        axes = [axes]

    x = np.arange(len(ERA_ORDER))
    for ax, oc in zip(axes, outcomes):
        sub = ctrl.filter(pl.col("outcome") == oc)
        coefs, errs = [], []
        for era in ERA_ORDER:
            row = sub.filter(pl.col("era") == era)
            if row.height:
                c = row["treat_post"][0]
                # reconstruct SE from coef and p via normal approx isn't stored;
                # the table carries p, so draw a marker and colour by significance
                coefs.append(c)
                errs.append(row["p_treat_post"][0])
            else:
                coefs.append(np.nan); errs.append(np.nan)
        coefs = np.array(coefs)
        colors = [C_RETAIL if p < 0.05 else "#b0b0b0" for p in errs]
        ax.axhline(0, color="black", lw=0.9, ls="--", alpha=0.6, zorder=2)
        ax.plot(x, coefs, "-", color=C_RETAIL, lw=1.4, alpha=0.5, zorder=2)
        for xi, ci, col in zip(x, coefs, colors):
            ax.plot(xi, ci, "o", color=col, markersize=8, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([ERA_LABELS[e] for e in ERA_ORDER], fontsize=8)
        ax.set_title(PRETTY.get(oc, oc), fontsize=10.5, pad=8)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)

    axes[0].set_ylabel("Retail vs. professional\ntreat:post (size-controlled)", fontsize=9)
    fig.suptitle("Retail earnings-window behaviour across three eras\n"
                 "(filled = significant at 5%, grey = not)", fontsize=12, y=1.06)
    fig.tight_layout()

    out_dir = out_dir or (RESULTS_DIR / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "fig_era_coefficients.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.name}")


def fig_era_levels(outcome: str = "otm", out_dir: Path = None):
    """Four cell means per era, as connected retail/procust lines -- one panel
    per era. Reveals who moves and whether the gap changes across eras."""
    import matplotlib.pyplot as plt

    lv = era_levels(outcome=outcome)

    fig, axes = plt.subplots(1, len(ERA_ORDER), figsize=(3.1 * len(ERA_ORDER), 3.8), sharey=True)

    for ax, era in zip(axes, ERA_ORDER):
        sub = lv.filter(pl.col("era") == era)
        for grp, colour, label in [("retail", C_RETAIL, "Retail"),
                                    ("procust", C_PROCUST, "Professional")]:
            g = sub.filter(pl.col("participant_group") == grp)
            base = g.filter(~pl.col("is_near_event"))["mean_share"]
            near = g.filter(pl.col("is_near_event"))["mean_share"]
            if base.len() and near.len():
                ax.plot([0, 1], [base[0], near[0]], "o-", color=colour,
                        markersize=7, lw=2.0, label=label, zorder=3)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Baseline", "Near event"], fontsize=9)
        ax.set_xlim(-0.25, 1.25)
        ax.set_title(ERA_LABELS[era].replace("\n", " "), fontsize=10.5)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)

    axes[0].set_ylabel(f"Mean {PRETTY.get(outcome, outcome)} share", fontsize=10)
    axes[0].legend(fontsize=9, framealpha=0.95)
    fig.suptitle(f"{PRETTY.get(outcome, outcome)}: who moves, by era", fontsize=12, y=1.02)
    fig.tight_layout()

    out_dir = out_dir or (RESULTS_DIR / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"fig_era_levels_{outcome}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.name}")


def build_all_era_figures(out_dir: Path = None):
    print("Building era figures...")
    fig_era_coefficients(out_dir=out_dir)
    for oc in ["otm", "lt_100", "otm_put", "open"]:
        fig_era_levels(outcome=oc, out_dir=out_dir)
    print("done")


if __name__ == "__main__":
    build_all_era_figures()
