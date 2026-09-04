"""
build_results_tables.py
=========================
Regenerates every results table for the thesis from the underlying data, and
writes each to results/ as CSV. Nothing here is exploratory -- this is the
final, citable output, kept separate from the investigative work in
B1_retail_activity_analysis.ipynb so that the numbers appearing in the
write-up can always be reproduced from a single command.

Tables produced:
  table1_main_did          Binary difference-in-differences across all
                           outcomes, on the full and balanced panels
  table2_levels            The underlying shares behind those coefficients,
                           by participant group and period
  table3_dispersion        Continuous dispersion regressions, with and
                           without a firm-size control
  table4_dispersion_quartiles  Dispersion quartile dummies -- the honest
                           specification, since the relationship is a
                           threshold for composition and a gradient for volume
  table5_sample            Sample construction and selection diagnostics

Usage from Jupyter:
    from analysis.build_results_tables import build_all_tables
    tables = build_all_tables()

Usage from a terminal (run from the repo root):
    python src/analysis/build_results_tables.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

import sys
sys.path.append(str(Path(__file__).parent.parent))
from paths import RESULTS_DIR
from analysis.event_window_profile import (
    build_diff_in_diff_panel, run_diff_in_diff, run_dispersion_regression,
    add_market_cap, build_balanced_panel,
)

OUTCOMES = ["otm", "otm_put", "otm_call", "itm", "lt_100", "call", "open"]


def table1_main_did(outcomes: list[str] = None, cluster_by: str = "ticker") -> pl.DataFrame:
    """Binary DiD on both panels. The treat:post coefficient answers 'does
    retail shift differently from professional customers', not merely 'does
    retail shift' -- a distinction that matters, since several outcomes are
    driven by professionals moving rather than retail."""
    outcomes = outcomes or OUTCOMES
    rows = []
    for oc in outcomes:
        full = build_diff_in_diff_panel(outcome=oc, verbose=False)
        bal = build_balanced_panel(full, verbose=False)
        m_f = run_diff_in_diff(full, cluster_by=cluster_by)
        m_b = run_diff_in_diff(bal, cluster_by=cluster_by)
        rows.append({
            "outcome": oc,
            "full_coef": m_f.params["treat:post"],
            "full_p": m_f.pvalues["treat:post"],
            "full_n": int(m_f.nobs),
            "balanced_coef": m_b.params["treat:post"],
            "balanced_p": m_b.pvalues["treat:post"],
            "balanced_n": int(m_b.nobs),
        })
    return pl.DataFrame(rows)


def table2_levels(outcomes: list[str] = None, balanced: bool = True) -> pl.DataFrame:
    """Mean share by participant group and period. A DiD coefficient alone
    cannot show which group moved, and in several cases here the answer is
    'the professionals did', so the levels belong alongside every result."""
    outcomes = outcomes or OUTCOMES
    frames = []
    for oc in outcomes:
        panel = build_diff_in_diff_panel(outcome=oc, verbose=False)
        if balanced:
            panel = build_balanced_panel(panel, verbose=False)
        lv = (
            panel.group_by(["participant_group", "is_near_event"])
            .agg(pl.col("share").mean().alias("mean_share"), pl.len().alias("n"))
            .with_columns(pl.lit(oc).alias("outcome"))
        )
        frames.append(lv)
    return (
        pl.concat(frames)
        .select(["outcome", "participant_group", "is_near_event", "mean_share", "n"])
        .sort(["outcome", "participant_group", "is_near_event"])
    )


def table3_dispersion(outcomes: list[str] = None, cluster_by: str = "ticker") -> pl.DataFrame:
    """Continuous dispersion, with and without a firm-size control.

    Dispersion is strongly negatively correlated with firm size, so an
    apparent dispersion effect can be size in disguise. Where did_ctrl
    collapses while size_did is significant, size was doing the work."""
    outcomes = outcomes or OUTCOMES
    rows = []
    for oc in outcomes:
        panel = build_diff_in_diff_panel(outcome=oc, verbose=False)
        m_base = run_dispersion_regression(panel, spec="triple", cluster_by=cluster_by)
        pmc = add_market_cap(panel, verbose=False)
        m_ctrl = run_dispersion_regression(
            pmc, spec="triple", cluster_by=cluster_by, controls=["log_mktcap"], verbose=False
        )
        rows.append({
            "outcome": oc,
            "did_base": m_base.params["dispersion:treat:post"],
            "p_base": m_base.pvalues["dispersion:treat:post"],
            "did_ctrl": m_ctrl.params["dispersion:treat:post"],
            "p_ctrl": m_ctrl.pvalues["dispersion:treat:post"],
            "cross_ctrl": m_ctrl.params["dispersion:treat"],
            "p_cross": m_ctrl.pvalues["dispersion:treat"],
            "size_did": m_ctrl.params["log_mktcap:treat:post"],
            "p_size": m_ctrl.pvalues["log_mktcap:treat:post"],
        })
    return pl.DataFrame(rows)


def table4_dispersion_quartiles(
    outcome: str = "otm", cluster_by: str = "ticker"
) -> pl.DataFrame:
    """Dispersion quartile dummies rather than a linear term, for both the
    composition outcome and volume.

    The linear specification implies a smooth gradient. For contract
    selection that is misleading -- the effect is confined to the top
    quartile -- while for volume the gradient is real. Quartile dummies let
    the data show which."""
    import statsmodels.formula.api as smf

    panel = add_market_cap(build_diff_in_diff_panel(outcome=outcome, verbose=False), verbose=False)
    rows = []

    for dep_label, dep in [("share", "share"), ("log_volume", "log_volume")]:
        for period_label, near in [("near_event", True), ("baseline", False)]:
            df = (
                panel.filter(pl.col("is_near_event") == near)
                .filter(pl.col("avg_daily_vol") > 0)
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
            for q in ["Q2", "Q3", "Q4"]:
                term = f"C(disp_q)[T.{q}]:treat"
                if term not in m.params.index:
                    continue  # quartile absent from this subsample
                rows.append({
                    "dependent": dep_label,
                    "period": period_label,
                    "quartile_vs_Q1": q,
                    "coef": m.params[term],
                    "p_value": m.pvalues[term],
                    "n": int(m.nobs),
                })
    return pl.DataFrame(rows)


def table5_sample(outcome: str = "otm") -> pl.DataFrame:
    """Sample construction and the selection diagnostic. Documents how many
    firm-events survive each stage, and the size difference between events
    present in both periods and those present in baseline only."""
    panel = add_market_cap(build_diff_in_diff_panel(outcome=outcome, verbose=False), verbose=False)
    bal = build_balanced_panel(panel, verbose=False)

    df = panel.filter(pl.col("log_mktcap").is_not_null() & (pl.col("avg_daily_vol") > 0))
    presence = (
        df.group_by(["resolved_ticker", "ANNDATS_ACT", "participant_group"])
        .agg(
            pl.col("is_near_event").any().alias("has_near"),
            (~pl.col("is_near_event")).any().alias("has_base"),
            pl.col("log_mktcap").first().alias("log_mktcap"),
        )
        .with_columns(
            pl.when(pl.col("has_near") & pl.col("has_base")).then(pl.lit("both"))
            .when(pl.col("has_base")).then(pl.lit("baseline_only"))
            .otherwise(pl.lit("near_only")).alias("presence")
        )
        .group_by("presence")
        .agg(pl.col("log_mktcap").mean().alias("mean_log_mktcap"), pl.len().alias("n"))
        .sort("presence")
    )

    summary = pl.DataFrame([
        {"stage": "panel rows (full)", "value": float(panel.height)},
        {"stage": "panel rows (balanced)", "value": float(bal.height)},
        {"stage": "balanced retention rate", "value": bal.height / panel.height},
        {"stage": "distinct firm-events (full)",
         "value": float(panel.select(["resolved_ticker", "ANNDATS_ACT"]).unique().height)},
        {"stage": "distinct tickers (full)",
         "value": float(panel["resolved_ticker"].n_unique())},
    ])
    return {"summary": summary, "selection": presence}


def build_all_tables(out_dir: Path = None, save: bool = True) -> dict:
    out_dir = out_dir or RESULTS_DIR
    if save:
        out_dir.mkdir(parents=True, exist_ok=True)

    tables = {}
    print("Building table 1 (main DiD)...")
    tables["table1_main_did"] = table1_main_did()
    print("Building table 2 (levels)...")
    tables["table2_levels"] = table2_levels()
    print("Building table 3 (dispersion, size-controlled)...")
    tables["table3_dispersion"] = table3_dispersion()
    print("Building table 4 (dispersion quartiles)...")
    tables["table4_dispersion_quartiles"] = table4_dispersion_quartiles()
    print("Building table 5 (sample diagnostics)...")
    t5 = table5_sample()
    tables["table5_sample_summary"] = t5["summary"]
    tables["table5_sample_selection"] = t5["selection"]

    if save:
        for name, tbl in tables.items():
            path = out_dir / f"{name}.csv"
            tbl.write_csv(path)
            print(f"  wrote {path.name} ({tbl.height} rows)")
        print(f"\nAll tables written to {out_dir}")
    return tables


if __name__ == "__main__":
    build_all_tables()
