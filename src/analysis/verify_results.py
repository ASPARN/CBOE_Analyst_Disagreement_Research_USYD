"""
verify_results.py
===================
Recomputes every headline number reported in the thesis, from the current
state of the data, and prints them alongside the values that were reported.
Any discrepancy means a result was computed against a superseded version of
an intermediate file and needs revisiting.

Why this exists. The pipeline was rebuilt several times during the project:
daily_retail gained size tiers, then professional-customer columns, then an
Int64 fix for volume overflow, then transaction counts; dispersion_events
gained prior earnings volatility, then winsorisation of the underlying
surprise. Adding columns does not change existing values, so results should
be stable across those rebuilds -- but that is an assumption worth testing
rather than trusting, particularly where analysis notebooks contain rebuild
cells that ran after results in the same notebook were produced.

Run this from a clean kernel after running A1 end to end. Every line should
show MATCH.

Usage (from the repo root):
    python src/analysis/verify_results.py
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

import sys
sys.path.append(str(Path(__file__).parent.parent))
from paths import DATA_DIR, IBES_DIR
from analysis.event_window_profile import (
    build_diff_in_diff_panel, run_diff_in_diff, run_dispersion_regression,
    add_market_cap, build_balanced_panel, _match_events_to_daily,
)
from analysis.build_results_tables import PRIMARY_SAMPLE_START

TOL = 5e-4  # tolerance on coefficients; anything larger is a real difference


def _check(label: str, got, expected, tol: float = TOL, integer: bool = False):
    if got is None:
        print(f"  {label:46s} MISSING (expected {expected})")
        return False
    ok = (got == expected) if integer else (abs(got - expected) <= tol)
    fmt = f"{got:,}" if integer else f"{got:+.4f}"
    exp = f"{expected:,}" if integer else f"{expected:+.4f}"
    print(f"  {label:46s} {fmt:>12s}  expected {exp:>12s}  {'MATCH' if ok else '*** DIFFERS ***'}")
    return ok


def verify() -> bool:
    results = []
    print("=" * 78)
    print("VERIFYING HEADLINE RESULTS")
    print("=" * 78)

    # ---- 1. sample construction -------------------------------------------
    print("\n1. Sample construction (full period)")
    _, n_matched, n_total = _match_events_to_daily(window=30)
    results.append(_check("firm-events in dispersion panel", n_total, 163_010, integer=True))
    results.append(_check("matched to CBOE trading data", n_matched, 97_270, integer=True))

    # ---- 2. binary DiD, restricted balanced panel -------------------------
    print(f"\n2. Binary difference-in-differences (balanced panel, {PRIMARY_SAMPLE_START} onward)")
    expected_did = {
        "otm": 0.0388, "otm_put": 0.0226, "otm_call": 0.0162,
        "itm": -0.0170, "lt_100": -0.0133, "call": -0.0189, "open": 0.0184,
    }
    panels = {}
    for oc, exp in expected_did.items():
        p = build_diff_in_diff_panel(outcome=oc, verbose=False, date_from=PRIMARY_SAMPLE_START)
        panels[oc] = p
        m = run_diff_in_diff(build_balanced_panel(p, verbose=False), cluster_by="ticker")
        results.append(_check(f"{oc} treat:post", m.params["treat:post"], exp))

    # ---- 3. dispersion on volume ------------------------------------------
    print("\n3. Analyst dispersion, cross-sectional (log volume, size-controlled)")
    pmc = add_market_cap(panels["otm"], verbose=False)
    m = run_dispersion_regression(
        pmc, spec="triple", outcome_var="log_volume", cluster_by="ticker",
        controls=["log_mktcap"], verbose=False,
    )
    results.append(_check("dispersion:treat", m.params["dispersion:treat"], 0.0387))
    results.append(_check("log_mktcap:treat:post", m.params["log_mktcap:treat:post"], 0.4947))

    # ---- 4. prior earnings volatility -------------------------------------
    print("\n4. Prior earnings volatility (OTM share, size-controlled)")
    for var, exp in [("dispersion_scaled", -0.0038), ("earnings_volatility", -0.0134)]:
        m = run_dispersion_regression(
            pmc, spec="triple", cluster_by="ticker", controls=["log_mktcap"],
            uncertainty_var=var, verbose=False,
        )
        results.append(_check(f"{var} triple", m.params["dispersion:treat:post"], exp))

    # ---- 5. the 2015 classification break ---------------------------------
    print("\n5. Professional-customer classification break (2014 vs 2015)")
    daily = pl.concat([
        pl.read_parquet(f)
        for f in sorted((DATA_DIR / "cboe_daily_retail").glob("daily_retail_*.parquet"))
    ])
    cov = (
        daily.with_columns(pl.col("quote_date").dt.year().alias("year"))
        .group_by("year")
        .agg(
            (pl.col("procust_vol_total") > 0).sum().alias("active"),
            pl.len().alias("days"),
        )
        .with_columns((pl.col("active") / pl.col("days")).alias("coverage"))
        .sort("year")
    )
    for yr, exp in [(2014, 0.2439), (2015, 0.1491)]:
        got = cov.filter(pl.col("year") == yr)["coverage"]
        results.append(_check(f"procust coverage {yr}", got[0] if got.len() else None, exp, tol=1e-3))

    # ---- 6. small-trade proxy recall --------------------------------------
    print("\n6. Small-trade proxy recall")
    rec = (
        daily.with_columns(pl.col("quote_date").dt.year().alias("year"))
        .group_by("year")
        .agg(
            pl.col("retail_vol_lt_100").sum().alias("small"),
            pl.col("retail_vol_total").sum().alias("total"),
        )
        .with_columns((pl.col("small") / pl.col("total")).alias("recall"))
        .sort("year")
    )
    for yr, exp in [(2011, 0.5018), (2022, 0.6710)]:
        got = rec.filter(pl.col("year") == yr)["recall"]
        results.append(_check(f"recall {yr}", got[0] if got.len() else None, exp, tol=1e-3))

    print("\n" + "=" * 78)
    n_ok, n = sum(results), len(results)
    if n_ok == n:
        print(f"ALL {n} CHECKS MATCH -- results are reproducible from current data.")
    else:
        print(f"{n - n_ok} of {n} CHECKS DIFFER -- investigate before relying on these numbers.")
    print("=" * 78)
    return n_ok == n


if __name__ == "__main__":
    sys.exit(0 if verify() else 1)
