"""
check_moneyness_coverage.py
=============================
Diagnostic, not part of the pipeline: works out whether the option volume
that could NOT be assigned a moneyness classification (no matching CRSP
spot price) actually matters for the analysis.

Roughly 9% of CBOE option-rows have no CRSP price. That number is
misleading on its own, because CRSP's daily stock file covers securities,
not indices -- so index products (^SPX, ^VIX, ^XEO, ^XSP and similar) are
unpriceable by construction. Those products also have no earnings
announcements, so they were never part of the firm-event sample in the
first place.

The number that actually matters is therefore not "what share of all CBOE
volume is unpriced" but "what share of volume ON EVENT-SAMPLE TICKERS is
unpriced". This reports both, plus the tickers carrying the most unpriced
volume so the cause can be seen directly rather than assumed.

Usage:
    from analysis.check_moneyness_coverage import check_coverage
    check_coverage()
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

import sys
sys.path.append(str(Path(__file__).parent.parent))
from paths import DATA_DIR
from analysis.event_window_profile import _match_events_to_daily

MONEYNESS_DIR = DATA_DIR / "cboe_daily_moneyness"


def check_coverage(window: int = 30) -> pl.DataFrame:
    files = sorted(MONEYNESS_DIR.glob("daily_moneyness_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No moneyness files in {MONEYNESS_DIR}. Run build_moneyness.py first.")
    mny = pl.concat([pl.read_parquet(f) for f in files])

    classified_cols = [c for c in mny.columns if c.startswith("retail_vol_") and "unknown" not in c]
    unknown_cols = [c for c in mny.columns if c.startswith("retail_vol_") and "unknown" in c]

    mny = mny.with_columns(
        pl.sum_horizontal(classified_cols).alias("_classified"),
        pl.sum_horizontal(unknown_cols).alias("_unknown"),
    )

    tot_c = mny["_classified"].sum()
    tot_u = mny["_unknown"].sum()
    print("=== ALL CBOE tickers ===")
    print(f"  Classified retail volume: {tot_c:,}")
    print(f"  Unclassified (no spot):   {tot_u:,}  ({tot_u / (tot_c + tot_u):.1%})\n")

    print("=== Tickers carrying the most unclassified volume ===")
    top = (
        mny.group_by("underlying_symbol")
        .agg(pl.col("_unknown").sum().alias("unknown_vol"))
        .filter(pl.col("unknown_vol") > 0)
        .sort("unknown_vol", descending=True)
        .head(15)
    )
    print(top)

    # restrict to tickers that actually appear in the matched event sample
    matched, n_matched, n_total = _match_events_to_daily(window=window)
    event_tickers = matched.select("resolved_ticker").unique()
    print(f"\nEvent-sample tickers: {event_tickers.height:,}")

    in_sample = mny.join(
        event_tickers, left_on="underlying_symbol", right_on="resolved_ticker", how="inner"
    )
    ev_c = in_sample["_classified"].sum()
    ev_u = in_sample["_unknown"].sum()
    print("\n=== EVENT-SAMPLE tickers only (the number that matters) ===")
    print(f"  Classified retail volume: {ev_c:,}")
    print(f"  Unclassified (no spot):   {ev_u:,}  ({ev_u / (ev_c + ev_u):.1%})")

    return top


if __name__ == "__main__":
    check_coverage()
