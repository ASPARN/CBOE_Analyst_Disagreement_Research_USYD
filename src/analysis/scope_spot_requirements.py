"""
scope_spot_requirements.py
============================
Diagnostic, not part of the pipeline: works out exactly which
(ticker, date-range) combinations actually need spot price data, before any
of it gets downloaded.

Moneyness (OTM vs ITM) is only needed for option-days that fall inside an
event window -- so this is deliberately NOT "every ticker in the CBOE data
for the full 2011-2022 range". Scoping it properly first avoids fetching
years of irrelevant history for thousands of tickers, and sidesteps much of
the delisting problem by only requesting names during periods when they
were genuinely trading.

IMPORTANT: this imports _match_events_to_daily from event_window_profile
rather than reimplementing the matching logic. An earlier version of this
script duplicated that logic by hand and produced materially different
(understated) counts as a result. Any script that needs "which events
matched" must call the shared function, so there is exactly one definition
of matching in the project.

Usage:
    from analysis.scope_spot_requirements import scope_requirements
    per_ticker = scope_requirements(window=30)
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

import sys
sys.path.append(str(Path(__file__).parent.parent))
from analysis.event_window_profile import _match_events_to_daily


def scope_requirements(window: int = 30, events_path: Path = None) -> pl.DataFrame:
    matched, n_matched, n_total = _match_events_to_daily(window=window, events_path=events_path)

    print(f"Total events: {n_total:,}")
    print(f"Events matched to CBOE trading data: {n_matched:,}")

    # matched is exploded to one row per (event, relative day); collapse back
    # to distinct events before computing per-ticker ranges
    events = matched.select(["resolved_ticker", "ANNDATS_ACT"]).unique()
    print(f"Distinct matched (ticker, announcement) pairs: {events.height:,}")
    print(f"Unique tickers needing spot prices: {events['resolved_ticker'].n_unique():,}\n")

    # per-ticker date range actually needed: earliest event minus the window,
    # latest event plus the window. Calendar-day padding is deliberately
    # generous -- a trading-day window never spans more calendar days than this.
    pad = window * 2
    per_ticker = (
        events.group_by("resolved_ticker")
        .agg(
            pl.col("ANNDATS_ACT").min().alias("first_event"),
            pl.col("ANNDATS_ACT").max().alias("last_event"),
            pl.len().alias("n_events"),
        )
        .with_columns(
            (pl.col("first_event") - pl.duration(days=pad)).alias("fetch_start"),
            (pl.col("last_event") + pl.duration(days=pad)).alias("fetch_end"),
        )
        .with_columns(
            (pl.col("fetch_end") - pl.col("fetch_start")).dt.total_days().alias("span_days")
        )
        .sort("resolved_ticker")
    )

    total_span = per_ticker["span_days"].sum()
    overall_days = (events["ANNDATS_ACT"].max() - events["ANNDATS_ACT"].min()).days + 2 * pad
    naive_span = per_ticker.height * overall_days

    print("=== Per-ticker fetch span (calendar days) ===")
    print(per_ticker["span_days"].describe())
    print()
    print(f"Sum of needed calendar-day spans:          {total_span:,}")
    print(f"If instead fetching full range per ticker: {naive_span:,}")
    if naive_span:
        print(f"Scoped fetch is {total_span / naive_span:.1%} of the naive approach")

    return per_ticker


if __name__ == "__main__":
    scope_requirements()
