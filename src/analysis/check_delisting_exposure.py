"""
check_delisting_exposure.py
=============================
Diagnostic, not part of the pipeline: estimates how much of the ticker
universe needing spot prices is likely to be DELISTED by the end of the
sample period, and therefore at risk of being unavailable (or wrongly
priced) from a free, survivorship-biased price source such as yfinance.

Why this matters for this project specifically: delisting is not random.
Companies disappear because they were acquired, taken private, or went
bankrupt -- exactly the high-uncertainty situations where analyst forecast
dispersion is elevated and this research is most interesting. Silently
losing them biases the sample against the hypothesis being tested, so the
size of that exposure is worth measuring BEFORE choosing a price source.

Method: a ticker whose last CBOE options activity falls well before the end
of the dataset is very likely to have stopped trading, rather than merely
being illiquid on the final day. The staleness threshold is configurable;
the default of 60 trading days is deliberately conservative, so this
under-counts rather than over-counts.

Usage:
    from analysis.check_delisting_exposure import check_delisting_exposure
    summary = check_delisting_exposure()
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

import sys
sys.path.append(str(Path(__file__).parent.parent))
from paths import DATA_DIR
from analysis.event_window_profile import _match_events_to_daily, _load_daily_retail


def check_delisting_exposure(
    stale_days: int = 60,
    window: int = 30,
    events_path: Path = None,
) -> pl.DataFrame:
    matched, n_matched, n_total = _match_events_to_daily(window=window, events_path=events_path)

    needed_tickers = matched.select("resolved_ticker").unique()
    print(f"Tickers needing spot prices: {needed_tickers.height:,}\n")

    daily = _load_daily_retail()
    sample_end = daily["quote_date"].max()
    print(f"Dataset ends: {sample_end}")

    last_seen = (
        daily.group_by("underlying_symbol")
        .agg(pl.col("quote_date").max().alias("last_active"))
    )

    # count distinct trading days in the dataset after each ticker's last
    # appearance -- trading days, not calendar days, so holidays don't distort it
    all_trading_days = daily.select("quote_date").unique().sort("quote_date")
    n_total_days = all_trading_days.height

    day_rank = all_trading_days.with_columns(
        pl.int_range(pl.len()).alias("day_rank")
    )
    last_seen = last_seen.join(
        day_rank, left_on="last_active", right_on="quote_date", how="left"
    ).with_columns(
        (n_total_days - 1 - pl.col("day_rank")).alias("trading_days_since_last_active")
    )

    result = needed_tickers.join(
        last_seen, left_on="resolved_ticker", right_on="underlying_symbol", how="left"
    ).with_columns(
        (pl.col("trading_days_since_last_active") >= stale_days).alias("likely_delisted")
    )

    n_delisted = result.filter(pl.col("likely_delisted")).height
    n_total_tickers = result.height
    print(
        f"Likely delisted (no CBOE activity in final {stale_days}+ trading days): "
        f"{n_delisted:,} of {n_total_tickers:,} ({n_delisted / n_total_tickers:.1%})\n"
    )

    # how many EVENTS sit on those at-risk tickers -- the number that actually
    # matters, since that's the analysis sample at stake
    distinct_events = matched.select(["resolved_ticker", "ANNDATS_ACT"]).unique()
    at_risk = result.filter(pl.col("likely_delisted")).select("resolved_ticker")
    n_events_at_risk = distinct_events.join(at_risk, on="resolved_ticker", how="inner").height
    print(
        f"Firm-events on likely-delisted tickers: {n_events_at_risk:,} of "
        f"{distinct_events.height:,} ({n_events_at_risk / distinct_events.height:.1%})\n"
    )

    print("=== Sensitivity to the staleness threshold ===")
    for threshold in [20, 60, 125, 250]:
        n = result.filter(pl.col("trading_days_since_last_active") >= threshold).height
        label = {20: "~1 month", 60: "~3 months", 125: "~6 months", 250: "~1 year"}[threshold]
        print(f"  >= {threshold:>3} trading days ({label:>9}): {n:>5,} tickers ({n / n_total_tickers:5.1%})")

    return result.sort("trading_days_since_last_active", descending=True)


if __name__ == "__main__":
    check_delisting_exposure()
