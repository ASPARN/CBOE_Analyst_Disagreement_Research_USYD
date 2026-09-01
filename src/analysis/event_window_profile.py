"""
event_window_profile.py
=========================
Diagnostic tool, not part of the main pipeline: builds an empirical profile
of retail trading volume relative to earnings announcement dates, averaged
across a sample of firm-events. Used to inform the actual window-length
choice (currently arbitrary: "1 month prior", "2 weeks prior", etc.) with
real data rather than a round number picked by feel.

For each sampled firm-event, this looks up the announced ticker's OWN
trading-day calendar (not calendar days -- weekends/holidays don't count),
finds the announcement date's position in it, and pulls retail volume at
each relative trading-day offset around that position. Averaging this
across many events reveals whether there's a real inflection point where
retail activity visibly picks up.

Usage from Jupyter:
    from event_window_profile import build_event_profile
    profile = build_event_profile(n_sample=1000, window=30)
"""

from __future__ import annotations

import random
from pathlib import Path

import polars as pl

import sys
sys.path.append(str(Path(__file__).parent.parent))
from paths import DATA_DIR, IBES_DIR

DAILY_RETAIL_DIR = DATA_DIR / "cboe_daily_retail"


def _load_daily_retail() -> pl.DataFrame:
    files = sorted(DAILY_RETAIL_DIR.glob("daily_retail_*.parquet"))
    if not files:
        return pl.DataFrame(schema={"underlying_symbol": pl.Utf8, "quote_date": pl.Date, "retail_vol_total": pl.Int64})
    daily = pl.concat([pl.read_parquet(f) for f in files])

    # join in the moneyness breakdown if build_moneyness.py has been run.
    # Both tables are keyed on (underlying_symbol, quote_date) with identical
    # row counts, so this is a clean 1:1 join. Kept optional so the rest of
    # the analysis still works before moneyness exists.
    mny_dir = DATA_DIR / "cboe_daily_moneyness"
    mny_files = sorted(mny_dir.glob("daily_moneyness_*.parquet"))
    if mny_files:
        mny = pl.concat([pl.read_parquet(f) for f in mny_files])
        daily = daily.join(mny, on=["underlying_symbol", "quote_date"], how="left")

    return daily


def _resolve_tickers(oftic_values: list[str], cboe_tickers: set) -> pl.DataFrame:
    """CBOE's share-class ticker separator ('.' vs '/') appears to have
    changed convention partway through the study period -- some tickers
    (e.g. BRK.B / BRK.A) exist under BOTH spellings in the combined CBOE
    data, some only under '/', and IBES consistently uses '.'. Try the
    ticker as given first; if that's not in CBOE's data, try swapping the
    separator before giving up."""
    resolved = []
    for t in oftic_values:
        if t in cboe_tickers:
            resolved.append(t)
        elif "." in t and t.replace(".", "/") in cboe_tickers:
            resolved.append(t.replace(".", "/"))
        else:
            resolved.append(t)
    return pl.DataFrame({"OFTIC": oftic_values, "resolved_ticker": resolved})


def _match_events_to_daily(
    n_sample: int = None,
    window: int = 30,
    seed: int = 42,
    events_path: Path = None,
) -> tuple[pl.DataFrame, int, int]:
    """Shared core: resolves tickers, anchors each event to its own trading-
    day calendar, and pulls every composition column (not just total volume)
    at each relative trading-day offset. Both build_event_profile and
    build_composition_comparison aggregate this same matched data
    differently, so they're guaranteed to agree on which events/days
    actually matched."""
    events_path = events_path or (IBES_DIR / "dispersion_events.parquet")
    events = pl.read_parquet(events_path).select(["OFTIC", "ANNDATS_ACT"]).filter(
        pl.col("OFTIC").is_not_null()
    )

    if n_sample is not None and events.height > n_sample:
        random.seed(seed)
        idx = random.sample(range(events.height), n_sample)
        events = events[idx]

    daily = (
        _load_daily_retail()
        .sort(["underlying_symbol", "quote_date"])
        .with_columns(pl.int_range(pl.len()).over("underlying_symbol").alias("day_idx"))
    )

    cboe_ticker_set = set(daily["underlying_symbol"].unique().to_list())
    ticker_map = _resolve_tickers(events["OFTIC"].unique().to_list(), cboe_ticker_set)
    events = events.join(ticker_map, on="OFTIC")

    anchors = events.join(
        daily.select(["underlying_symbol", "quote_date", "day_idx"]),
        left_on=["resolved_ticker", "ANNDATS_ACT"],
        right_on=["underlying_symbol", "quote_date"],
        how="inner",
    )
    n_matched = anchors.height
    n_total = events.height

    rel_days = list(range(-window, window + 1))
    exploded = (
        anchors.with_columns(pl.lit(rel_days).alias("rel_day"))
        .explode("rel_day")
        .with_columns((pl.col("day_idx") + pl.col("rel_day")).alias("target_idx"))
    )

    composition_cols = [
        "retail_vol_total", "retail_vol_lt_100", "retail_vol_100_199",
        "retail_vol_gt_199", "retail_vol_call", "retail_vol_put",
        "procust_vol_total", "procust_vol_lt_100", "procust_vol_100_199",
        "procust_vol_gt_199", "procust_vol_call", "procust_vol_put",
    ]
    # moneyness columns are present only if build_moneyness.py has been run
    mny_cols = [c for c in daily.columns if "_otm_" in c or "_itm_" in c or "_atm_" in c]
    composition_cols += mny_cols

    matched = exploded.join(
        daily.select(["underlying_symbol", "day_idx"] + composition_cols),
        left_on=["resolved_ticker", "target_idx"],
        right_on=["underlying_symbol", "day_idx"],
        how="inner",
    )

    # combined OTM / ITM totals across calls and puts -- this is DV2's
    # numerator and denominator in the research design
    for group in ["retail", "procust"]:
        for mny in ["otm", "itm", "atm"]:
            parts = [c for c in mny_cols if c.startswith(f"{group}_vol_{mny}_")]
            if parts:
                matched = matched.with_columns(
                    pl.sum_horizontal(parts).alias(f"{group}_vol_{mny}")
                )

    return matched, n_matched, n_total


def build_event_profile(
    n_sample: int = None,
    window: int = 30,
    seed: int = 42,
    events_path: Path = None,
) -> pl.DataFrame:
    """Vectorized: no per-event Python loop, so this scales cleanly from a
    thousand-event sample up to the full firm-event panel. n_sample=None
    (the default) uses every event; pass a number for a quick exploratory
    subset instead."""
    matched, n_matched, n_total = _match_events_to_daily(n_sample, window, seed, events_path)

    profile = (
        matched.group_by("rel_day")
        .agg(
            pl.col("retail_vol_total").mean().alias("mean_retail_vol"),
            pl.len().alias("n_events"),
        )
        .sort("rel_day")
    )

    print(f"Matched {n_matched:,} of {n_total:,} events to CBOE trading data")
    return profile


def build_composition_comparison(
    near_event_days: range = range(-1, 5),
    n_sample: int = None,
    window: int = 30,
    seed: int = 42,
    events_path: Path = None,
) -> pl.DataFrame:
    """Compares activity COMPOSITION -- position size, call/put split --
    between the near-announcement window (default: -1 to +4 trading days,
    based on where the volume profile actually showed elevation) and every
    other day in the +-window range, for BOTH retail (cust_) and
    professional customers (procust_) side by side. Answers not just
    whether retail shifts toward smaller, more lottery-like positions
    around earnings, but whether that shift is retail-specific or a
    general pattern professional customers show too."""
    matched, n_matched, n_total = _match_events_to_daily(n_sample, window, seed, events_path)

    near_event_set = set(near_event_days)
    matched = matched.with_columns(
        pl.col("rel_day").is_in(near_event_set).alias("is_near_event")
    )

    rows = []
    for group in ["retail", "procust"]:
        summary = matched.group_by("is_near_event").agg(
            pl.col(f"{group}_vol_total").sum().alias("vol_total"),
            pl.col(f"{group}_vol_lt_100").sum().alias("vol_lt_100"),
            pl.col(f"{group}_vol_100_199").sum().alias("vol_100_199"),
            pl.col(f"{group}_vol_gt_199").sum().alias("vol_gt_199"),
            pl.col(f"{group}_vol_call").sum().alias("vol_call"),
            pl.col(f"{group}_vol_put").sum().alias("vol_put"),
            pl.len().alias("n_obs"),
        ).with_columns(pl.lit(group).alias("participant_group"))
        rows.append(summary)

    combined = pl.concat(rows)
    combined = combined.with_columns(
        (pl.col("vol_lt_100") / pl.col("vol_total")).alias("share_lt_100"),
        (pl.col("vol_100_199") / pl.col("vol_total")).alias("share_100_199"),
        (pl.col("vol_gt_199") / pl.col("vol_total")).alias("share_gt_199"),
        (pl.col("vol_call") / pl.col("vol_total")).alias("share_call"),
        (pl.col("vol_put") / pl.col("vol_total")).alias("share_put"),
    ).sort("participant_group", "is_near_event")

    print(f"Matched {n_matched:,} of {n_total:,} events to CBOE trading data")
    return combined.select(
        "participant_group", "is_near_event", "n_obs",
        "share_lt_100", "share_100_199", "share_gt_199", "share_call", "share_put",
    )


def build_diff_in_diff_panel(
    outcome: str = "call",
    near_event_days: range = range(-1, 5),
    n_sample: int = None,
    window: int = 30,
    seed: int = 42,
    events_path: Path = None,
) -> pl.DataFrame:
    """Builds a firm-event level panel for a difference-in-differences test.
    For each event, computes EACH participant group's outcome share
    separately within its own near-event days and its own baseline days --
    two 'periods' per group, per event -- ready for an OLS regression with
    a group x period interaction term. outcome is one of: 'lt_100',
    '100_199', 'gt_199', 'call', 'put'."""
    matched, n_matched, n_total = _match_events_to_daily(n_sample, window, seed, events_path)

    matched = matched.with_columns(
        pl.col("rel_day").is_in(set(near_event_days)).alias("is_near_event")
    )

    rows = []
    for group in ["retail", "procust"]:
        per_event = (
            matched.group_by(["resolved_ticker", "ANNDATS_ACT", "is_near_event"])
            .agg(
                pl.col(f"{group}_vol_{outcome}").sum().alias("outcome_vol"),
                pl.col(f"{group}_vol_total").sum().alias("total_vol"),
            )
            .filter(pl.col("total_vol") > 0)  # can't compute a share with zero volume
            .with_columns(
                (pl.col("outcome_vol") / pl.col("total_vol")).alias("share"),
                pl.lit(group).alias("participant_group"),
            )
        )
        rows.append(per_event)

    panel = pl.concat(rows).select(
        "resolved_ticker", "ANNDATS_ACT", "participant_group", "is_near_event", "share", "total_vol"
    )
    print(f"Matched {n_matched:,} of {n_total:,} events to CBOE trading data")
    print(f"Panel has {panel.height:,} rows (up to 2 periods x 2 groups per event)")
    return panel


def run_diff_in_diff(panel: pl.DataFrame, cluster_by: str = "event"):
    """Runs share ~ treat * post via OLS. cluster_by controls what standard
    errors are clustered on: 'event' (default) treats each firm-event as
    the correlated unit; 'ticker' instead treats each COMPANY as the
    correlated unit, acknowledging that the same firm's repeated quarterly
    events likely correlate with each other too, not just the two group
    rows within a single event. A result that stays significant under both
    choices is considerably more robust than one that only holds under one."""
    import statsmodels.formula.api as smf

    df = panel.to_pandas()
    df["treat"] = (df["participant_group"] == "retail").astype(int)
    df["post"] = df["is_near_event"].astype(int)

    if cluster_by == "event":
        groups = df["resolved_ticker"].astype(str) + "_" + df["ANNDATS_ACT"].astype(str)
    elif cluster_by == "ticker":
        groups = df["resolved_ticker"]
    else:
        raise ValueError("cluster_by must be 'event' or 'ticker'")

    model = smf.ols("share ~ treat * post", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": groups}
    )
    return model


def compare_weighting_schemes(outcome: str = "lt_100", near_event_days: range = range(-1, 5)) -> None:
    """Computes the near-event shift for both retail and procust under TWO
    weighting schemes, from the exact same matched data in a single call --
    volume-weighted (every event-day counted by its own volume, matching
    build_composition_comparison) and equal-weighted-per-event (every
    firm-event counted once regardless of size, matching
    build_diff_in_diff_panel). Run this when the two functions seem to
    disagree, to rule out comparing numbers computed at different points in
    the pipeline's history rather than a genuine weighting effect."""
    matched, n_matched, n_total = _match_events_to_daily()
    matched = matched.with_columns(
        pl.col("rel_day").is_in(set(near_event_days)).alias("is_near_event")
    )
    print(f"Matched {n_matched:,} of {n_total:,} events to CBOE trading data\n")

    print("=== Volume-weighted (every event-day counted by its own volume) ===")
    for group in ["retail", "procust"]:
        vw = (
            matched.group_by("is_near_event")
            .agg(
                pl.col(f"{group}_vol_{outcome}").sum().alias("outcome"),
                pl.col(f"{group}_vol_total").sum().alias("total"),
            )
            .with_columns((pl.col("outcome") / pl.col("total")).alias("share"))
            .sort("is_near_event")
        )
        shift = vw.filter(pl.col("is_near_event"))["share"][0] - vw.filter(~pl.col("is_near_event"))["share"][0]
        print(f"  {group}: shift = {shift:+.4f}")

    print("\n=== Equal-weighted per event (every firm-event counted once) ===")
    for group in ["retail", "procust"]:
        per_event = (
            matched.group_by(["resolved_ticker", "ANNDATS_ACT", "is_near_event"])
            .agg(
                pl.col(f"{group}_vol_{outcome}").sum().alias("outcome"),
                pl.col(f"{group}_vol_total").sum().alias("total"),
            )
            .filter(pl.col("total") > 0)
            .with_columns((pl.col("outcome") / pl.col("total")).alias("share"))
        )
        ew = per_event.group_by("is_near_event").agg(pl.col("share").mean().alias("mean_share")).sort("is_near_event")
        shift = ew.filter(pl.col("is_near_event"))["mean_share"][0] - ew.filter(~pl.col("is_near_event"))["mean_share"][0]
        print(f"  {group}: shift = {shift:+.4f}")


def compare_top_n_tickers(outcome: str = "lt_100", near_event_days: range = range(-1, 5), top_n: int = 10) -> None:
    """Isolates the top_n highest-volume tickers specifically and compares
    their retail/procust shift against everyone else -- sharper than
    compare_weighting_schemes's implicit whole-distribution weighting, for
    when a genuine divergence is driven by a small handful of extreme names
    rather than 'the upper half' broadly."""
    matched, n_matched, n_total = _match_events_to_daily()
    matched = matched.with_columns(
        pl.col("rel_day").is_in(set(near_event_days)).alias("is_near_event")
    )

    ticker_vol = (
        matched.group_by("resolved_ticker")
        .agg((pl.col("retail_vol_total") + pl.col("procust_vol_total")).sum().alias("ticker_total_vol"))
        .sort("ticker_total_vol", descending=True)
    )
    top_tickers = set(ticker_vol.head(top_n)["resolved_ticker"].to_list())
    print(f"Top {top_n} tickers by volume: {sorted(top_tickers)}\n")

    matched = matched.with_columns(pl.col("resolved_ticker").is_in(top_tickers).alias("is_top_n"))

    for is_top in [True, False]:
        label = f"TOP {top_n}" if is_top else "EVERYONE ELSE"
        sub = matched.filter(pl.col("is_top_n") == is_top)
        print(f"--- {label} ---")
        for group in ["retail", "procust"]:
            per_event = (
                sub.group_by(["resolved_ticker", "ANNDATS_ACT", "is_near_event"])
                .agg(
                    pl.col(f"{group}_vol_{outcome}").sum().alias("outcome"),
                    pl.col(f"{group}_vol_total").sum().alias("total"),
                )
                .filter(pl.col("total") > 0)
                .with_columns((pl.col("outcome") / pl.col("total")).alias("share"))
            )
            ew = per_event.group_by("is_near_event").agg(pl.col("share").mean().alias("mean_share")).sort("is_near_event")
            if ew.height == 2:
                shift = ew.filter(pl.col("is_near_event"))["mean_share"][0] - ew.filter(~pl.col("is_near_event"))["mean_share"][0]
                print(f"  {group}: shift = {shift:+.4f}")
        print()


if __name__ == "__main__":
    profile = build_event_profile()
    print(profile)
