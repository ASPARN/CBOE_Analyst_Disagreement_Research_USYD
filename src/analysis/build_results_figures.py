"""
build_results_figures.py
==========================
Generates presentation-quality figures from the regression results, saved to
results/figures/ at 300 dpi for print.

Three figures, each answering a question a table answers less clearly:

  fig1_did_coefficients   Coefficient plot of the main difference-in-differences
                          estimates with 95% confidence intervals, full panel
                          against balanced panel. Shows magnitude, precision and
                          robustness in one view -- particularly the outcomes
                          where the two samples disagree.

  fig2_dispersion_quartiles  The central contrast in the findings: analyst
                          dispersion affects contract selection as a THRESHOLD
                          (nothing until the top quartile) but trading volume as
                          a GRADIENT (smooth and monotonic). Two panels side by
                          side make this immediately visible; a linear
                          coefficient hides it.

  fig3_levels             Mean shares by participant group and period. A DiD
                          coefficient cannot show which group moved, and in
                          several of these results the answer is the
                          professionals rather than the retail investors.

Usage from Jupyter:
    from analysis.build_results_figures import build_all_figures
    build_all_figures()

Usage from a terminal (run from the repo root):
    python src/analysis/build_results_figures.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

import sys
sys.path.append(str(Path(__file__).parent.parent))
from paths import RESULTS_DIR
from analysis.event_window_profile import (
    build_diff_in_diff_panel, run_diff_in_diff, add_market_cap, build_balanced_panel,
)

# colourblind-safe; blue for retail, orange for professional customers
C_RETAIL = "#1a4f8a"
C_PROCUST = "#d1731f"
C_FULL = "#4a90c4"
C_BAL = "#1a4f8a"
GRID = "#e4e3de"

OUTCOMES = ["otm", "otm_put", "otm_call", "itm", "lt_100", "call", "open"]

PRETTY = {
    "otm": "Out-of-the-money",
    "otm_put": "OTM puts",
    "otm_call": "OTM calls",
    "itm": "In-the-money",
    "lt_100": "Small positions (<100)",
    "call": "Call share",
    "open": "Opening positions",
}


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def fig1_did_coefficients(outcomes: list[str] = None, out_dir: Path = None):
    import matplotlib.pyplot as plt

    outcomes = outcomes or OUTCOMES
    rows = []
    for oc in outcomes:
        full = build_diff_in_diff_panel(outcome=oc, verbose=False)
        bal = build_balanced_panel(full, verbose=False)
        for label, panel in [("Full panel", full), ("Balanced panel", bal)]:
            m = run_diff_in_diff(panel, cluster_by="ticker")
            coef = m.params["treat:post"]
            se = m.bse["treat:post"]
            rows.append({
                "outcome": PRETTY.get(oc, oc), "sample": label,
                "coef": coef, "lo": coef - 1.96 * se, "hi": coef + 1.96 * se,
                "sig": m.pvalues["treat:post"] < 0.05,
            })
    df = pl.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    labels = [PRETTY.get(o, o) for o in outcomes]
    y = np.arange(len(labels))
    offset = 0.18

    for i, (sample, colour, dy) in enumerate(
        [("Full panel", C_FULL, offset), ("Balanced panel", C_BAL, -offset)]
    ):
        sub = df.filter(pl.col("sample") == sample)
        sub = sub.join(pl.DataFrame({"outcome": labels, "_ord": y}), on="outcome").sort("_ord")
        yy = sub["_ord"].to_numpy() + dy
        c = sub["coef"].to_numpy()
        err = np.vstack([c - sub["lo"].to_numpy(), sub["hi"].to_numpy() - c])
        ax.errorbar(c, yy, xerr=err, fmt="o", color=colour, markersize=5,
                    capsize=3, elinewidth=1.4, label=sample, zorder=3)

    ax.axvline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.6, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Retail vs. professional difference-in-differences estimate\n"
                  "(share of volume; 95% confidence intervals)", fontsize=10)
    ax.set_title("Difference-in-differences estimates by outcome",
                 fontsize=11.5, pad=12)
    ax.legend(fontsize=9, framealpha=0.95, loc="lower right")
    _style(ax)
    fig.tight_layout()

    out_dir = out_dir or (RESULTS_DIR / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "fig1_did_coefficients.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.name}")
    return df


def fig2_dispersion_quartiles(outcome: str = "otm", out_dir: Path = None):
    import matplotlib.pyplot as plt
    import statsmodels.formula.api as smf

    panel = add_market_cap(build_diff_in_diff_panel(outcome=outcome, verbose=False), verbose=False)

    results = {}
    for dep in ["share", "log_volume"]:
        df = (
            panel.filter(pl.col("is_near_event") & (pl.col("avg_daily_vol") > 0))
            .with_columns(
                pl.col("dispersion_scaled")
                .qcut(4, labels=["Q1", "Q2", "Q3", "Q4"], allow_duplicates=True)
                .alias("disp_q")
            )
            .to_pandas()
            .dropna(subset=["log_mktcap", "dispersion_scaled"])
        )
        df["treat"] = (df["participant_group"] == "retail").astype(int)
        df["log_mktcap_z"] = (df["log_mktcap"] - df["log_mktcap"].mean()) / df["log_mktcap"].std()
        df["y"] = np.log(df["avg_daily_vol"]) if dep == "log_volume" else df["share"]
        m = smf.ols("y ~ C(disp_q) * treat + log_mktcap_z * treat", data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df["resolved_ticker"]}
        )
        pts = [(0.0, 0.0)]  # Q1 is the reference category
        for q in ["Q2", "Q3", "Q4"]:
            term = f"C(disp_q)[T.{q}]:treat"
            pts.append((m.params[term], 1.96 * m.bse[term]) if term in m.params.index else (np.nan, np.nan))
        results[dep] = pts

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    x = np.arange(4)
    titles = [
        f"Contract selection ({PRETTY.get(outcome, outcome)} share)",
        "Trading volume (log)",
    ]

    for ax, dep, title in zip(axes, ["share", "log_volume"], titles):
        pts = results[dep]
        c = np.array([p[0] for p in pts])
        e = np.array([p[1] for p in pts])
        ok = ~np.isnan(c)
        if (~ok).any():
            missing = [f"Q{i+1}" for i in range(4) if not ok[i]]
            print(f"  note: {dep} missing quartile(s) {missing} -- not enough distinct values to bin")
        ax.errorbar(x[ok], c[ok], yerr=e[ok], fmt="o-", color=C_RETAIL, markersize=6,
                    capsize=4, elinewidth=1.4, linewidth=1.6, zorder=3)
        ax.axhline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.6, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(["Q1\n(lowest)", "Q2", "Q3", "Q4\n(highest)"], fontsize=9)
        ax.set_xlabel("Analyst forecast dispersion quartile", fontsize=10)
        ax.set_title(title, fontsize=10.5, pad=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Retail vs. professional gap\n(relative to Q1, size-controlled)", fontsize=10)
    fig.suptitle("Retail–professional gap by analyst dispersion quartile",
                 fontsize=12, y=1.02)
    fig.tight_layout()

    out_dir = out_dir or (RESULTS_DIR / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "fig2_dispersion_quartiles.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.name}")
    return results


def fig3_levels(outcomes: list[str] = None, out_dir: Path = None):
    import matplotlib.pyplot as plt

    outcomes = outcomes or ["otm", "otm_put", "lt_100", "open"]
    fig, axes = plt.subplots(1, len(outcomes), figsize=(3.1 * len(outcomes), 4.0), sharey=False)
    if len(outcomes) == 1:
        axes = [axes]

    for ax, oc in zip(axes, outcomes):
        panel = build_balanced_panel(
            build_diff_in_diff_panel(outcome=oc, verbose=False), verbose=False
        )
        lv = (
            panel.group_by(["participant_group", "is_near_event"])
            .agg(pl.col("share").mean().alias("m"))
        )
        for grp, colour, label in [("retail", C_RETAIL, "Retail"), ("procust", C_PROCUST, "Professional")]:
            base = lv.filter((pl.col("participant_group") == grp) & (~pl.col("is_near_event")))["m"][0]
            near = lv.filter((pl.col("participant_group") == grp) & (pl.col("is_near_event")))["m"][0]
            ax.plot([0, 1], [base, near], "o-", color=colour, markersize=7,
                    linewidth=2.0, label=label, zorder=3)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Baseline", "Near event"], fontsize=9)
        ax.set_xlim(-0.25, 1.25)
        ax.set_title(PRETTY.get(oc, oc), fontsize=10.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Mean share of volume", fontsize=10)
    axes[0].legend(fontsize=9, framealpha=0.95)
    fig.suptitle("Who actually moves? Levels behind the difference-in-differences estimates",
                 fontsize=12, y=1.02)
    fig.tight_layout()

    out_dir = out_dir or (RESULTS_DIR / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "fig3_levels.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.name}")


def build_all_figures(out_dir: Path = None):
    out_dir = out_dir or (RESULTS_DIR / "figures")
    print("Building figures...")
    fig1_did_coefficients(out_dir=out_dir)
    fig2_dispersion_quartiles(out_dir=out_dir)
    fig3_levels(out_dir=out_dir)
    print(f"\nAll figures written to {out_dir}")


if __name__ == "__main__":
    build_all_figures()
