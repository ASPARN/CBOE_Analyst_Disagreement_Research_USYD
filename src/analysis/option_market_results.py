"""
option_market_results.py
==========================
Writes every reported table and figure for the option-market tests
(Tests 1-3, notebook B4) to results/, so nothing reported in the thesis is
copied from notebook output -- the same rule as build_results_tables and
build_results_figures.

Tables (results/, CSV)
    table7_informed_trading.csv             Test 1: CAR and SUE on net put flow,
                                            both samples, pooled and by era
    table7b_informed_trading_robustness.csv Test 1: flow definitions, size
                                            terciles, professional presence
    table8_option_returns.csv               Test 2: midpoint returns, all measures
    table9_option_returns_costs.csv         Test 2: dw_buy and dw_net at three cost
                                            levels, all contracts and OTM
    table10_loss_decomposition.csv          Test 3: components, by group and era
    table11_right_direction.csv             Test 3: right direction, still lost

Figures (results/figures/, PNG, 300 dpi)
    fig_informed_flow_by_era.png        Test 1 coefficients by era, with 95% CIs
    fig_option_return_distribution.png  the lottery shape: most positions lose,
                                        a long right tail lifts the mean
    fig_option_returns_after_costs.png  midpoint vs half vs full spread
    fig_option_loss_decomposition.png   where the typical loss comes from
    fig_option_right_direction.png      right direction, still lost

Usage (end of B4, or from the repository root):
    from analysis.option_market_results import build_option_market_results
    build_option_market_results(panel, contracts)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.append(str(Path(__file__).parent.parent))
from paths import RESULTS_DIR
from analysis.informed_trading import ERAS, informed_trading_table
from analysis.option_returns import event_option_returns, option_returns_table, spread_comparison
from analysis.option_decomposition import (decomposition_panel, event_decomposition,
                                           decomposition_table, right_direction_summary, LABELS)

# house style, matching covid_era_figures
C_RETAIL = "#1a4f8a"
C_PROCUST = "#d1731f"
GRID = "#e4e3de"
ERA_ORDER = ["pooled", "pre_covid", "covid", "post_covid"]
ERA_LABELS = {"pooled": "Pooled\n2016–2025", "pre_covid": "Pre-COVID\n2016–2019",
              "covid": "COVID\n2020–2021", "post_covid": "Post-COVID\n2022–2025"}
GROUP_STYLE = {"retail": (C_RETAIL, "Retail"), "procust": (C_PROCUST, "Professional")}
COST_LABELS = {0.0: "Midpoint", 0.5: "Half spread", 1.0: "Full spread"}


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def _pct(ax, axis="y"):
    from matplotlib.ticker import PercentFormatter
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(PercentFormatter(1.0, decimals=0))


def _save(fig, name, fig_dir):
    import matplotlib.pyplot as plt
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}")


def _write(df, name, tables_dir):
    tables_dir.mkdir(parents=True, exist_ok=True)
    df.write_csv(tables_dir / name, float_precision=6)
    print(f"  wrote {name}  ({df.height} rows)")


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def informed_trading_robustness(panel: pl.DataFrame) -> pl.DataFrame:
    """Test 1 robustness on the retail-only sample, stacked with a 'check'
    column: flow definitions, firm-size terciles, professional presence."""
    parts = [informed_trading_table(panel, dvs=("car01",), flows=("net_put", "pp_ratio", "net_bearish"),
                                    samples=("retail",))
             .with_columns(pl.lit("flow_measure").alias("check"), pl.col("flow").alias("subset"))]
    sized = panel.with_columns(pl.col("log_mktcap").qcut(3, labels=["small", "mid", "large"]).alias("_size"))
    for s in ("small", "mid", "large"):
        parts.append(informed_trading_table(sized.filter(pl.col("_size") == s), dvs=("car01",), samples=("retail",))
                     .with_columns(pl.lit("size_tercile").alias("check"), pl.lit(s).alias("subset")))
    for label, sub in (("with_pro", panel.filter(pl.col("procust_net_put").is_not_null())),
                       ("no_pro", panel.filter(pl.col("procust_net_put").is_null()))):
        parts.append(informed_trading_table(sub, dvs=("car01",), samples=("retail",))
                     .with_columns(pl.lit("professional_presence").alias("check"), pl.lit(label).alias("subset")))
    out = pl.concat(parts, how="diagonal_relaxed")
    lead = ["check", "subset", "dv", "flow", "sample", "era"]
    return out.select(lead + [c for c in out.columns if c not in lead])


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def fig_informed_flow_by_era(t1: pl.DataFrame, fig_dir: Path):
    """Test 1 coefficients (bp per 1 SD of net put flow) by era with 95% CIs.
    Filled markers are significant at 5%, hollow are not."""
    import matplotlib.pyplot as plt
    t = t1.filter((pl.col("dv") == "car01") & (pl.col("flow") == "net_put"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    x = np.arange(len(ERA_ORDER))
    panels = [("retail", "Retail flow only\n(every event with retail trading)", ["retail"]),
              ("both", "Both groups' flow\n(events where professionals also trade)", ["retail", "procust"])]
    for ax, (sample, title, groups) in zip(axes, panels):
        sub = t.filter(pl.col("sample") == sample)
        for j, g in enumerate(groups):
            colour, label = GROUP_STYLE[g]
            off = (j - (len(groups) - 1) / 2) * 0.16
            for i, era in enumerate(ERA_ORDER):
                r = sub.filter(pl.col("era") == era)
                if r.height == 0 or r[f"b_{g}"][0] is None:
                    continue
                b, se, p = r[f"b_{g}"][0] * 1e4, r[f"se_{g}"][0] * 1e4, r[f"p_{g}"][0]
                ax.errorbar(x[i] + off, b, yerr=1.96 * se, fmt="o", color=colour, ms=7, capsize=3,
                            mfc=colour if p < 0.05 else "white", mew=1.6, lw=1.4, zorder=3)
        ax.axhline(0, color="black", lw=0.9, ls="--", alpha=0.6, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels([ERA_LABELS[e] for e in ERA_ORDER], fontsize=8.5)
        ax.set_title(title, fontsize=10.5)
        _style(ax)
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([], [], marker="o", ls="", color=GROUP_STYLE[g][0], ms=7, label=GROUP_STYLE[g][1])
                           for g in groups], fontsize=9, frameon=False, loc="lower left")
    axes[0].set_ylabel("Change in CAR[0,+1] per 1 SD\nof net put flow (basis points)", fontsize=9.5)
    fig.suptitle("Pre-announcement net put flow and the announcement return, by era\n"
                 "(95% confidence intervals; filled = significant at 5%, hollow = not)", fontsize=11.5, y=1.04)
    fig.tight_layout()
    _save(fig, "fig_informed_flow_by_era.png", fig_dir)


def fig_option_return_distribution(er: pl.DataFrame, fig_dir: Path):
    """Distribution of event-level dollar-weighted returns on options bought
    (midpoint), with the median and the (winsorised) mean marked."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    series = {g: er[f"{g}_dw_buy"].drop_nulls().to_numpy() for g in GROUP_STYLE}
    top = max(np.quantile(v, 0.99) for v in series.values() if len(v))
    top = float(min(max(top, 1.0), 4.0))
    bins = np.linspace(-1.0, top, 70)
    for ax, (g, v) in zip(axes, series.items()):
        colour, label = GROUP_STYLE[g]
        lo, hi = np.quantile(v, [0.01, 0.99])
        mean_w, med = np.clip(v, lo, hi).mean(), np.median(v)
        ax.hist(np.clip(v, -1.0, top), bins=bins, density=True, color=colour, alpha=0.75, edgecolor="white", lw=0.3)
        above = float((v > top).mean())
        if above > 0:
            ax.annotate(f"{above:.1%} of events\nabove +{top:.0%}\n(shown in last bar)",
                        xy=(top, 0), xytext=(-8, 38), textcoords="offset points", ha="right", fontsize=8,
                        arrowprops={"arrowstyle": "-", "color": "grey", "lw": 0.8})
        ax.axvline(med, color="black", lw=1.6, label=f"Median {med:+.1%}")
        ax.axvline(mean_w, color="black", lw=1.6, ls="--", label=f"Mean {mean_w:+.1%}")
        ax.axvline(0, color="grey", lw=0.8, alpha=0.6)
        ax.set_title(f"{label} (n = {len(v):,} firm-events)", fontsize=10.5, loc="left")
        ax.set_ylabel("Density", fontsize=9.5)
        ax.legend(fontsize=9, frameon=False)
        _style(ax)
    _pct(axes[1], "x")
    axes[1].set_xlabel(f"Return per dollar of premium on options bought, day −1 to +1 "
                       f"(midpoint; right tail shown to {top:.0%})", fontsize=9.5)
    fig.suptitle("Distribution of returns on options bought across the announcement", fontsize=12, y=1.0)
    fig.tight_layout()
    _save(fig, "fig_option_return_distribution.png", fig_dir)


def fig_option_returns_after_costs(t9: pl.DataFrame, fig_dir: Path):
    """dw_buy by cost level, all contracts and OTM: bars = mean (95% CI),
    diamonds = median."""
    import matplotlib.pyplot as plt
    t = t9.filter((pl.col("era") == "pooled") & (pl.col("measure") == "dw_buy"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    costs = sorted(t["cost_frac"].unique().to_list())
    x = np.arange(len(costs))
    w = 0.36
    for ax, contracts in zip(axes, ["all", "otm"]):
        sub = t.filter(pl.col("contracts") == contracts)
        for j, g in enumerate(GROUP_STYLE):
            colour, label = GROUP_STYLE[g]
            rows = [sub.filter(pl.col("cost_frac") == c) for c in costs]
            means = [r[f"{g}_mean"][0] for r in rows]
            ses = [r[f"{g}_se"][0] for r in rows]
            meds = [r[f"{g}_median"][0] for r in rows]
            pos = x + (j - 0.5) * w
            ax.bar(pos, means, w, color=colour, alpha=0.85, label=f"{label} mean",
                   yerr=[1.96 * s for s in ses], capsize=3, error_kw={"lw": 1})
            ax.scatter(pos, meds, marker="D", s=36, color="white", edgecolor=colour, lw=1.6, zorder=4,
                       label=f"{label} median")
        ax.axhline(0, color="black", lw=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([COST_LABELS.get(c, f"{c:g} of half-spread") for c in costs], fontsize=9)
        ax.set_title("All contracts" if contracts == "all" else "Out-of-the-money contracts", fontsize=10.5)
        _pct(ax)
        _style(ax)
    axes[0].set_ylabel("Return per dollar of premium,\noptions bought, day −1 to +1", fontsize=9.5)
    h, l = axes[0].get_legend_handles_labels()
    order = [l.index(x) for x in ("Retail mean", "Professional mean", "Retail median", "Professional median")]
    axes[0].legend([h[i] for i in order], [l[i] for i in order], fontsize=8.5, frameon=False,
                   loc="lower left", ncol=2)
    fig.suptitle("Returns on options bought across the announcement, before and after trading costs",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, "fig_option_returns_after_costs.png", fig_dir)


def fig_option_loss_decomposition(t10: pl.DataFrame, fig_dir: Path):
    """Components of the midpoint return on options bought: means (which add up
    to the total) and medians (the typical position), by group."""
    import matplotlib.pyplot as plt
    t = t10.filter(pl.col("era") == "pooled")
    comps = [LABELS[k] for k in ("delta_term", "gamma_term", "vega_term", "theta_term", "residual", "dP")]
    short = {LABELS["delta_term"]: "Direction\n(delta)", LABELS["gamma_term"]: "Size of move\n(gamma)",
             LABELS["vega_term"]: "IV crush\n(vega)", LABELS["theta_term"]: "Time decay\n(theta)",
             LABELS["residual"]: "Residual", LABELS["dP"]: "Total"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    y = np.arange(len(comps))[::-1]
    h = 0.36
    for ax, stat, title in [(axes[0], "mean", "Mean (components add up to the total)"),
                            (axes[1], "median", "Median (the typical position)")]:
        for j, g in enumerate(GROUP_STYLE):
            colour, label = GROUP_STYLE[g]
            vals, errs = [], []
            for comp in comps:
                r = t.filter((pl.col("group") == g) & (pl.col("component") == comp))
                vals.append(r[stat][0] if r.height else np.nan)
                errs.append(1.96 * r["se"][0] if (r.height and stat == "mean") else 0)
            pos = y + (0.5 - j) * h
            ax.barh(pos, vals, h, color=colour, alpha=0.85, label=label,
                    xerr=errs if stat == "mean" else None, capsize=2.5, error_kw={"lw": 0.9})
        ax.axvline(0, color="black", lw=0.9)
        ax.axhspan(y[-1] - 0.5, y[-1] + 0.5, color=GRID, alpha=0.5, zorder=0)   # shade the total row
        ax.set_yticks(y)
        ax.set_yticklabels([short[c] for c in comps], fontsize=9)
        ax.set_title(title, fontsize=10.5)
        _pct(ax, "x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=9, frameon=False, loc="upper left")
    fig.supxlabel("Share of premium paid, options bought, day −1 to +1 (midpoint)", fontsize=9.5)
    fig.suptitle("Decomposition of the midpoint price change on options bought", fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, "fig_option_loss_decomposition.png", fig_dir)


def fig_option_right_direction(t11: pl.DataFrame, fig_dir: Path):
    """Share of volume called correctly, and of those, the share that still
    lost and the share where the IV change alone outweighed the gain."""
    import matplotlib.pyplot as plt
    t = t11.filter(pl.col("era") == "pooled")
    metrics = [("share_right", "Direction\ncalled correctly"),
               ("right_but_lost", "Correct calls that\nstill lost money"),
               ("iv_outweighed", "Correct calls where the IV\ncrush outweighed the gain")]
    vals_all = [v for m, _ in metrics for v in t[m].to_list() if v is not None]
    top = min(1.0, (max(vals_all) if vals_all else 0.6) + 0.12)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    x = np.arange(len(metrics))
    w = 0.36
    for ax, contracts in zip(axes, ["all", "otm"]):
        sub = t.filter(pl.col("contracts") == contracts)
        for j, g in enumerate(GROUP_STYLE):
            colour, label = GROUP_STYLE[g]
            r = sub.filter(pl.col("group") == g)
            if r.height == 0:
                continue
            vals = [r[m][0] for m, _ in metrics]
            pos = x + (j - 0.5) * w
            bars = ax.bar(pos, vals, w, color=colour, alpha=0.85, label=label if contracts == "all" else None)
            for b_, v in zip(bars, vals):
                if v is not None:
                    ax.text(b_.get_x() + b_.get_width() / 2, v + 0.012, f"{v:.0%}", ha="center", fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in metrics], fontsize=9)
        ax.set_ylim(0, top)
        ax.set_title("All contracts" if contracts == "all" else "Out-of-the-money contracts", fontsize=10.5)
        _pct(ax)
        _style(ax)
    axes[0].set_ylabel("Share of contracts bought\n(second and third bars: share of correct calls)", fontsize=9)
    fig.suptitle("Directional accuracy and outcomes of correctly called positions", fontsize=12, y=1.06)
    fig.legend(fontsize=9, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    _save(fig, "fig_option_right_direction.png", fig_dir)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def build_option_market_results(panel: pl.DataFrame, contracts: pl.DataFrame,
                                d: pl.DataFrame | None = None, out_dir: Path | None = None) -> dict:
    """Compute and write every Test 1-3 table and figure. `panel` is the Test 1
    event panel (build_event_panel), `contracts` the saved contract panel, and
    `d` the decomposition panel if already computed. Returns the tables."""
    tables_dir = out_dir or RESULTS_DIR
    fig_dir = tables_dir / "figures"
    print("Building option-market results...")

    t7 = informed_trading_table(panel)
    t7b = informed_trading_robustness(panel)
    er = event_option_returns(contracts)
    t8 = option_returns_table(er)
    t9 = pl.concat([
        spread_comparison(contracts, measures=("dw_buy", "dw_net")).with_columns(pl.lit("all").alias("contracts")),
        spread_comparison(contracts, measures=("dw_buy", "dw_net"), moneyness="otm")
        .with_columns(pl.lit("otm").alias("contracts"))])
    t9 = t9.select(["contracts"] + [c for c in t9.columns if c != "contracts"])
    d = d if d is not None else decomposition_panel(contracts, verbose=False)
    t10 = decomposition_table(event_decomposition(d))
    t11 = pl.concat([right_direction_summary(d).with_columns(pl.lit("all").alias("contracts")),
                     right_direction_summary(d, moneyness="otm").with_columns(pl.lit("otm").alias("contracts"))])
    t11 = t11.select(["contracts"] + [c for c in t11.columns if c != "contracts"])

    for df, name in [(t7, "table7_informed_trading.csv"), (t7b, "table7b_informed_trading_robustness.csv"),
                     (t8, "table8_option_returns.csv"), (t9, "table9_option_returns_costs.csv"),
                     (t10, "table10_loss_decomposition.csv"), (t11, "table11_right_direction.csv")]:
        _write(df, name, tables_dir)

    fig_informed_flow_by_era(t7, fig_dir)
    fig_option_return_distribution(er, fig_dir)
    fig_option_returns_after_costs(t9, fig_dir)
    fig_option_loss_decomposition(t10, fig_dir)
    fig_option_right_direction(t11, fig_dir)
    print("done")
    return {"table7": t7, "table7b": t7b, "table8": t8, "table9": t9, "table10": t10, "table11": t11}


if __name__ == "__main__":
    from analysis.informed_trading import build_event_panel
    from analysis.option_returns import PANEL_PATH
    build_option_market_results(build_event_panel(verbose=False), pl.read_parquet(PANEL_PATH))
