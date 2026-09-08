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
    date_from: str = None,
    date_to: str = None,
) -> tuple[pl.DataFrame, int, int]:
    """Shared core: resolves tickers, anchors each event to its own trading-
    day calendar, and pulls every composition column (not just total volume)
    at each relative trading-day offset. Both build_event_profile and
    build_composition_comparison aggregate this same matched data
    differently, so they're guaranteed to agree on which events/days
    actually matched."""
    events_path = events_path or (IBES_DIR / "dispersion_events.parquet")
    _ev_cols = ["OFTIC", "ANNDATS_ACT"]
    # dispersion_scaled is the continuous regressor; carried through here so
    # downstream regressions can use it instead of the binary near-event flag
    _avail = pl.scan_parquet(events_path).collect_schema().names()
    if "dispersion_scaled" in _avail:
        _ev_cols.append("dispersion_scaled")
    # prior earnings volatility -- the second uncertainty measure in RQ1
    if "earnings_volatility" in _avail:
        _ev_cols.append("earnings_volatility")
    events = pl.read_parquet(events_path).select(_ev_cols).filter(
        pl.col("OFTIC").is_not_null()
    )

    # optional era restriction, for comparing against studies covering a
    # narrower period than this sample. Bryzgalova, Pavlova & Sikorskaya
    # (2023) cover Nov 2019 - Jun 2021; this sample runs 2011 - May 2022,
    # so their entire window is roughly 15% of these data.
    if date_from is not None:
        events = events.filter(pl.col("ANNDATS_ACT") >= pl.lit(date_from).str.to_date())
    if date_to is not None:
        events = events.filter(pl.col("ANNDATS_ACT") <= pl.lit(date_to).str.to_date())

    # an era window can legitimately contain no events (e.g. a year outside
    # the CBOE sample). Exit cleanly rather than letting an empty frame reach
    # the joins, where the null-typed key column raises a schema error.
    if events.height == 0:
        empty = pl.DataFrame(schema={"resolved_ticker": pl.Utf8, "ANNDATS_ACT": pl.Date,
                                     "rel_day": pl.Int64})
        return empty, 0, 0

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
        "retail_vol_open", "retail_vol_close",
        "procust_vol_total", "procust_vol_lt_100", "procust_vol_100_199",
        "procust_vol_gt_199", "procust_vol_call", "procust_vol_put",
        "procust_vol_open", "procust_vol_close",
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
    date_from: str = None,
    date_to: str = None,
) -> pl.DataFrame:
    """Vectorized: no per-event Python loop, so this scales cleanly from a
    thousand-event sample up to the full firm-event panel. n_sample=None
    (the default) uses every event; pass a number for a quick exploratory
    subset instead."""
    matched, n_matched, n_total = _match_events_to_daily(
        n_sample, window, seed, events_path, date_from, date_to
    )
    if matched.height == 0:
        print("No events in this window; returning empty profile.")
        return pl.DataFrame(schema={"rel_day": pl.Int64, "mean_retail_vol": pl.Float64,
                                    "n_events": pl.UInt32})

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
    verbose: bool = True,
    date_from: str = None,
    date_to: str = None,
) -> pl.DataFrame:
    """Builds a firm-event level panel for a difference-in-differences test.
    For each event, computes EACH participant group's outcome share
    separately within its own near-event days and its own baseline days --
    two 'periods' per group, per event -- ready for an OLS regression with
    a group x period interaction term. outcome is one of: 'lt_100',
    '100_199', 'gt_199', 'call', 'put'."""
    matched, n_matched, n_total = _match_events_to_daily(
        n_sample, window, seed, events_path, date_from, date_to
    )

    if matched.height == 0:
        if verbose:
            print("No events in this window; returning empty panel.")
        return pl.DataFrame(schema={
            "resolved_ticker": pl.Utf8, "ANNDATS_ACT": pl.Date,
            "participant_group": pl.Utf8, "is_near_event": pl.Boolean,
            "share": pl.Float64, "total_vol": pl.Int64, "n_days": pl.UInt32,
            "avg_daily_vol": pl.Float64,
        })

    matched = matched.with_columns(
        pl.col("rel_day").is_in(set(near_event_days)).alias("is_near_event")
    )

    rows = []
    for group in ["retail", "procust"]:
        agg = [
            pl.col(f"{group}_vol_{outcome}").sum().alias("outcome_vol"),
            pl.col(f"{group}_vol_total").sum().alias("total_vol"),
            pl.len().alias("n_days"),
        ]
        if "dispersion_scaled" in matched.columns:
            agg.append(pl.col("dispersion_scaled").first().alias("dispersion_scaled"))
        if "earnings_volatility" in matched.columns:
            agg.append(pl.col("earnings_volatility").first().alias("earnings_volatility"))
        per_event = (
            matched.group_by(["resolved_ticker", "ANNDATS_ACT", "is_near_event"])
            .agg(agg)
            .filter(pl.col("total_vol") > 0)  # can't compute a share with zero volume
            .with_columns(
                (pl.col("outcome_vol") / pl.col("total_vol")).alias("share"),
                pl.lit(group).alias("participant_group"),
            )
        )
        rows.append(per_event)

    panel = pl.concat(rows)
    keep = ["resolved_ticker", "ANNDATS_ACT", "participant_group", "is_near_event",
            "share", "total_vol", "n_days"]
    if "dispersion_scaled" in panel.columns:
        keep.append("dispersion_scaled")
    if "earnings_volatility" in panel.columns:
        keep.append("earnings_volatility")
    panel = panel.select(keep).with_columns(
        # average daily volume, so near-event (6 days) and baseline (~55 days)
        # periods are directly comparable rather than differing by day count
        (pl.col("total_vol") / pl.col("n_days")).alias("avg_daily_vol")
    )
    if verbose:
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


def run_dispersion_regression(
    panel: pl.DataFrame,
    spec: str = "triple",
    outcome_var: str = "share",
    cluster_by: str = "ticker",
    standardize: bool = True,
    controls: list[str] = None,
    verbose: bool = True,
    uncertainty_var: str = "dispersion_scaled",
):
    """Replaces the binary near-event flag with the CONTINUOUS analyst
    forecast dispersion measure -- closer to what the research questions
    actually ask ("does higher dispersion predict different behaviour")
    than "is this near an announcement".

    spec:
      "triple"      share ~ dispersion * treat * post, on the full panel.
                    The dispersion:treat:post coefficient is the headline
                    result: does retail's near-event SHIFT grow with
                    dispersion, relative to professional customers?
      "near_only"   share ~ dispersion * treat, restricted to near-event
                    rows. Simpler to interpret: within earnings windows,
                    does the retail/procust gap widen with dispersion?

    outcome_var:
      "share"       composition outcome (whatever the panel was built for)
      "log_volume"  ln(average daily volume) -- for RQ1, which is about
                    trading volume rather than contract composition

    standardize: z-scores dispersion so coefficients read "per standard
    deviation of dispersion" rather than per raw unit, which matters
    because the raw measure is winsorised at 1.85-2.4 and a one-unit
    change is therefore enormous relative to its actual spread.
    """
    import numpy as np
    import statsmodels.formula.api as smf

    if uncertainty_var not in panel.columns:
        raise ValueError(
            f"Panel has no {uncertainty_var} column. Rebuild dispersion_events.parquet "
            f"and the panel with versions that carry it through."
        )

    df = panel.to_pandas().dropna(subset=[uncertainty_var])
    df["treat"] = (df["participant_group"] == "retail").astype(int)
    df["post"] = df["is_near_event"].astype(int)

    if outcome_var == "log_volume":
        df = df[df["avg_daily_vol"] > 0].copy()
        df["y"] = np.log(df["avg_daily_vol"])
    elif outcome_var == "share":
        df["y"] = df["share"]
    else:
        raise ValueError("outcome_var must be 'share' or 'log_volume'")

    # named "dispersion" in the formula regardless of which measure is used,
    # so coefficient names stay stable across specifications
    df["dispersion"] = df[uncertainty_var]
    if standardize:
        df["dispersion"] = (df["dispersion"] - df["dispersion"].mean()) / df["dispersion"].std()

    if spec == "triple":
        formula = "y ~ dispersion * treat * post"
    elif spec == "near_only":
        df = df[df["post"] == 1].copy()
        formula = "y ~ dispersion * treat"
    else:
        raise ValueError("spec must be 'triple' or 'near_only'")

    # Controls are added as full interaction terms, not just level shifts.
    # Adding "log_mktcap" alone would only absorb size LEVELS; the question
    # is whether the dispersion effect survives once size is allowed its own
    # event response, so "log_mktcap * treat * post" is the meaningful test.
    if controls:
        missing = [c for c in controls if c not in df.columns]
        if missing:
            raise ValueError(f"Control column(s) not in panel: {missing}. Did you run add_market_cap()?")
        before = len(df)
        df = df.dropna(subset=controls)
        if len(df) < before and verbose:
            print(f"  Dropped {before - len(df):,} rows missing control values ({len(df):,} remain)")
        for c in controls:
            if standardize and df[c].std() > 0:
                df[c] = (df[c] - df[c].mean()) / df[c].std()
            formula += f" + {c} * treat" + (" * post" if spec == "triple" else "")

    if cluster_by == "event":
        groups = df["resolved_ticker"].astype(str) + "_" + df["ANNDATS_ACT"].astype(str)
    elif cluster_by == "ticker":
        groups = df["resolved_ticker"]
    else:
        raise ValueError("cluster_by must be 'event' or 'ticker'")

    return smf.ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": groups})


_CRSP_MKTCAP_CACHE = None


def _crsp_mktcap_frame() -> pl.DataFrame:
    """Loads and caches the CRSP market cap lookup. Cached because running a
    robustness table across several outcomes calls add_market_cap once per
    outcome, and re-reading the full CRSP parquet each time is wasteful."""
    global _CRSP_MKTCAP_CACHE
    if _CRSP_MKTCAP_CACHE is None:
        from paths import CRSP_DIR
        crsp_path = CRSP_DIR / "crsp_daily.parquet"
        if not crsp_path.exists():
            raise FileNotFoundError(f"CRSP parquet not found at {crsp_path}. Run ingest_crsp.py first.")
        _CRSP_MKTCAP_CACHE = (
            pl.scan_parquet(crsp_path)
            .select(["Ticker", "DlyCalDt", "DlyCap"])
            .filter(pl.col("DlyCap").is_not_null() & (pl.col("DlyCap") > 0))
            .unique(subset=["Ticker", "DlyCalDt"])
            .collect()
            .sort(["Ticker", "DlyCalDt"])
        )
    return _CRSP_MKTCAP_CACHE


def add_market_cap(panel: pl.DataFrame, max_lookback_days: int = 10, verbose: bool = True) -> pl.DataFrame:
    """Joins CRSP market capitalisation onto a firm-event panel, as of each
    event's announcement date, and adds log_mktcap.

    Motivation: analyst dispersion is systematically LOWER for large,
    heavily-covered firms, and ten mega-cap tickers are already known to
    behave oppositely to the rest of the universe on position size. Any
    apparent "dispersion effect" could therefore partly be a size effect in
    disguise. Controlling for size properly is the way to tell them apart.

    DlyCap is in thousands of dollars and is heavily right-skewed, so the
    log is what enters the regression.

    Announcements can fall on non-trading days (a company reporting after
    Friday's close, say), so this takes the most recent CRSP observation
    within max_lookback_days rather than requiring an exact date match.
    """
    crsp = _crsp_mktcap_frame()

    # join_asof needs both sides sorted on the ordering key
    panel_sorted = panel.sort(["resolved_ticker", "ANNDATS_ACT"])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = panel_sorted.join_asof(
            crsp,
            left_on="ANNDATS_ACT",
            right_on="DlyCalDt",
            by_left="resolved_ticker",
            by_right="Ticker",
            strategy="backward",
            tolerance=f"{max_lookback_days}d",
        ).with_columns(pl.col("DlyCap").log().alias("log_mktcap"))

    n_matched = out.filter(pl.col("log_mktcap").is_not_null()).height
    if verbose:
        print(f"Market cap matched for {n_matched:,} of {out.height:,} panel rows ({n_matched/out.height:.1%})")
    return out


def exclude_top_n_tickers(panel: pl.DataFrame, n: int = 10) -> pl.DataFrame:
    """Drops the n highest-volume tickers from a panel. Those names are known
    to behave oppositely to the rest of the universe and carry enough volume
    to dominate aggregates, so re-running without them is a basic robustness
    check rather than a cosmetic one."""
    ticker_vol = (
        panel.group_by("resolved_ticker")
        .agg(pl.col("total_vol").sum().alias("v"))
        .sort("v", descending=True)
    )
    top = ticker_vol.head(n)["resolved_ticker"].to_list()
    print(f"Excluding top {n} tickers by volume: {sorted(top)}")
    out = panel.filter(~pl.col("resolved_ticker").is_in(top))
    print(f"  {panel.height:,} -> {out.height:,} panel rows")
    return out


def dispersion_robustness_table(
    outcomes: list[str] = None,
    outcome_var: str = "share",
    cluster_by: str = "ticker",
    near_event_days: range = range(-1, 5),
) -> pl.DataFrame:
    """Runs the dispersion regression with and without a market-cap control
    across several outcomes, and reports the coefficients side by side.

    The point of the comparison: analyst dispersion is systematically lower
    for large firms, so an apparent dispersion effect can be firm size in
    disguise. If dispersion:treat:post survives the log_mktcap * treat * post
    control, the event effect is real; if it collapses while
    log_mktcap:treat:post is significant, size was doing the work.

    Reports three coefficients per outcome:
      did_base    dispersion:treat:post with no size control
      did_ctrl    dispersion:treat:post with the size control
      cross_ctrl  dispersion:treat with the size control -- the CROSS-SECTIONAL
                  relationship (does the retail/professional gap vary with
                  dispersion at all), which is a separate question from the
                  event response and may survive when the event effect does not
    """
    outcomes = outcomes or ["otm", "otm_put", "lt_100", "call", "open"]
    rows = []

    for oc in outcomes:
        panel = build_diff_in_diff_panel(
            outcome=oc, near_event_days=near_event_days, verbose=False
        )
        m_base = run_dispersion_regression(
            panel, spec="triple", outcome_var=outcome_var, cluster_by=cluster_by
        )
        panel_mc = add_market_cap(panel, verbose=False)
        m_ctrl = run_dispersion_regression(
            panel_mc, spec="triple", outcome_var=outcome_var,
            cluster_by=cluster_by, controls=["log_mktcap"], verbose=False
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


def investigate_size_reversal(panel_mc: pl.DataFrame) -> dict:
    """Characterises the near-event reversal in the firm-size interaction:
    retail's volume advantage over professional customers SHRINKS with firm
    size in ordinary conditions but GROWS with it around earnings.

    Runs three checks, because a reversal in an interaction coefficient can
    have three quite different causes:

      1. SELECTION. The near-event window is ~6 trading days against ~55 for
         baseline, so a firm-event needs volume in a much shorter window to
         appear at all. If small, illiquid firms drop out of the near-event
         sample disproportionately, the "reversal" is sample composition
         rather than behaviour. Compares the size distribution of events
         present in both periods against those present in baseline only.

      2. SHAPE. Whether the size effect is a smooth gradient or a threshold,
         via size-quartile dummies rather than a linear term.

      3. WHO MOVES. A DiD-style coefficient says nothing about which group
         changed. Reports each group's actual near-event volume lift within
         each size quartile, so the mechanism is visible directly.

    Requires a panel that has been through add_market_cap().
    """
    import numpy as np

    if "log_mktcap" not in panel_mc.columns:
        raise ValueError("Panel has no log_mktcap. Run add_market_cap() first.")

    df = panel_mc.filter(
        pl.col("log_mktcap").is_not_null() & (pl.col("avg_daily_vol") > 0)
    ).with_columns(
        pl.col("log_mktcap")
        .qcut(4, labels=["S1_small", "S2", "S3", "S4_large"], allow_duplicates=True)
        .alias("size_q"),
        pl.col("avg_daily_vol").log().alias("log_vol"),
    )

    out = {}

    # --- 1. selection ------------------------------------------------------
    periods_per_event = (
        df.group_by(["resolved_ticker", "ANNDATS_ACT", "participant_group"])
        .agg(
            pl.col("is_near_event").any().alias("has_near"),
            (~pl.col("is_near_event")).any().alias("has_base"),
            pl.col("log_mktcap").first().alias("log_mktcap"),
        )
    )
    sel = (
        periods_per_event.with_columns(
            pl.when(pl.col("has_near") & pl.col("has_base")).then(pl.lit("both"))
            .when(pl.col("has_base")).then(pl.lit("baseline_only"))
            .otherwise(pl.lit("near_only"))
            .alias("presence")
        )
        .group_by("presence")
        .agg(
            pl.col("log_mktcap").mean().alias("mean_log_mktcap"),
            pl.len().alias("n"),
        )
        .sort("presence")
    )
    out["selection"] = sel

    # --- 2. shape: size quartiles, retail vs procust, by period ------------
    shape = (
        df.group_by(["size_q", "participant_group", "is_near_event"])
        .agg(pl.col("log_vol").mean().alias("mean_log_vol"), pl.len().alias("n"))
        .sort(["size_q", "participant_group", "is_near_event"])
    )
    out["by_size_quartile"] = shape

    # --- 3. who moves: near-event lift per group per size quartile ---------
    lift = (
        df.group_by(["size_q", "participant_group", "is_near_event"])
        .agg(pl.col("log_vol").mean().alias("m"))
        .pivot(values="m", index=["size_q", "participant_group"], on="is_near_event")
        .rename({"true": "near", "false": "base"})
        .with_columns((pl.col("near") - pl.col("base")).alias("near_event_lift"))
        .sort(["size_q", "participant_group"])
    )
    out["near_event_lift"] = lift

    return out


def build_balanced_panel(panel: pl.DataFrame, verbose: bool = True) -> pl.DataFrame:
    """Restricts a panel to firm-events where ALL FOUR cells exist: both
    participant groups, in both the baseline and near-event periods.

    Why this matters: a panel row is only created when that group traded in
    that period. The near-event window is ~6 trading days against ~55 for
    baseline, and professional customers trade sparsely in small stocks, so
    procust rows drop out of the near-event window at a rate that falls
    monotonically with firm size (54% in the smallest size quartile, 14% in
    the largest). Estimates from the unbalanced panel are therefore partly
    conditioned on "professionals happened to trade", which biases anything
    that interacts with firm size.

    The balanced panel removes that conditioning, at the cost of dropping
    roughly 42% of observations -- disproportionately small firms. Neither
    sample is the correct one; they describe slightly different universes,
    and results are reported for both."""
    complete = (
        panel.filter(pl.col("avg_daily_vol") > 0)
        .group_by(["resolved_ticker", "ANNDATS_ACT"])
        .agg(pl.len().alias("n_cells"))
        .filter(pl.col("n_cells") == 4)
        .select(["resolved_ticker", "ANNDATS_ACT"])
    )
    out = panel.join(complete, on=["resolved_ticker", "ANNDATS_ACT"], how="inner")
    if verbose:
        print(f"Balanced panel: {out.height:,} rows (from {panel.height:,}, "
              f"{out.height / panel.height:.1%} retained)")
    return out


# Bryzgalova, Pavlova & Sikorskaya (2023) characterise retail options trading
# between November 2019 and June 2021. This sample runs 2011 - May 2022, so
# their entire window is a minority of these data and sits squarely inside
# the retail trading boom. Splitting on it tests whether findings that differ
# from theirs are driven by measurement or simply by period.
ERAS = {
    "pre_boom": ("2011-01-01", "2019-10-31"),
    "bpz_window": ("2019-11-01", "2021-06-30"),
    "post_boom": ("2021-07-01", "2022-05-16"),
}


def compare_eras_profile(window: int = 30, eras: dict = None) -> pl.DataFrame:
    """Runs the relative-day volume profile separately for each era.

    The motivating question: this sample shows no two-week pre-announcement
    buildup in retail volume, while Bryzgalova et al. report one. If the
    buildup appears inside their Nov 2019 - Jun 2021 window and not outside
    it, the discrepancy is period rather than method -- and the pattern
    documented in the literature is era-specific rather than structural.

    Returns one row per (era, rel_day) with mean retail volume normalised by
    that era's own baseline, so eras with very different absolute volumes
    remain comparable.
    """
    eras = eras or ERAS
    frames = []
    for name, (d0, d1) in eras.items():
        prof = build_event_profile(window=window, date_from=d0, date_to=d1)
        # normalise by the era's own far-from-event baseline (|rel_day| > 10)
        baseline = (
            prof.filter(pl.col("rel_day").abs() > 10)["mean_retail_vol"].mean()
        )
        frames.append(
            prof.with_columns(
                pl.lit(name).alias("era"),
                (pl.col("mean_retail_vol") / baseline).alias("vol_vs_baseline"),
            )
        )
    return pl.concat(frames).select(
        ["era", "rel_day", "mean_retail_vol", "vol_vs_baseline", "n_events"]
    )


def compare_eras_did(
    outcomes: list[str] = None, eras: dict = None, cluster_by: str = "ticker"
) -> pl.DataFrame:
    """Re-runs the binary difference-in-differences separately per era, so
    findings can be checked against the period a comparison study covers."""
    outcomes = outcomes or ["otm", "otm_put", "lt_100", "call", "open"]
    eras = eras or ERAS
    rows = []
    for name, (d0, d1) in eras.items():
        for oc in outcomes:
            panel = build_diff_in_diff_panel(
                outcome=oc, verbose=False, date_from=d0, date_to=d1
            )
            if panel.height < 100:
                print(f"  skipping {name}/{oc}: only {panel.height} panel rows")
                continue
            m = run_diff_in_diff(panel, cluster_by=cluster_by)
            rows.append({
                "era": name,
                "outcome": oc,
                "coef": m.params["treat:post"],
                "p_value": m.pvalues["treat:post"],
                "n_obs": int(m.nobs),
            })
    return pl.DataFrame(rows)


def yearly_eras(start: int = 2011, end: int = 2022, end_date: str = "2022-05-16") -> dict:
    """Builds an era dict with one entry per calendar year, for use with
    compare_eras_did and compare_eras_profile.

    Three coarse era bins can show that something changed but not whether it
    was a sharp break or a gradual drift, and that distinction matters for
    interpretation -- a break around a specific event (zero-commission
    trading going industry-wide in late 2019) implies a different mechanism
    from a steady trend. The final year is truncated at end_date since the
    CBOE sample stops mid-May 2022."""
    eras = {}
    for y in range(start, end + 1):
        first = f"{y}-01-01"
        last = end_date if y == end else f"{y}-12-31"
        eras[str(y)] = (first, last)
    return eras


def summarise_eras_profile(window: int = 30, eras: dict = None) -> pl.DataFrame:
    """Condenses the relative-day volume profile to a few statistics per era,
    so many eras can be compared at once without reading a wide table.

    pre_window_mean covers days -10 to -3, the region where a two-week
    pre-announcement buildup would appear if there were one; day0_ratio and
    day1_ratio capture the announcement spike itself. All are expressed
    relative to that era's own far-from-event baseline (|rel_day| > 10)."""
    eras = eras or ERAS
    rows = []
    for name, (d0, d1) in eras.items():
        prof = build_event_profile(window=window, date_from=d0, date_to=d1)
        baseline = prof.filter(pl.col("rel_day").abs() > 10)["mean_retail_vol"].mean()
        if not baseline:
            continue
        norm = prof.with_columns((pl.col("mean_retail_vol") / baseline).alias("r"))

        def _at(lo, hi):
            sub = norm.filter(pl.col("rel_day").is_between(lo, hi))["r"]
            return float(sub.mean()) if sub.len() else None

        rows.append({
            "era": name,
            "pre_window_mean": _at(-10, -3),
            "day_minus1": _at(-1, -1),
            "day0_ratio": _at(0, 0),
            "day1_ratio": _at(1, 1),
            "post_window_mean": _at(2, 5),
            "n_events_day0": int(norm.filter(pl.col("rel_day") == 0)["n_events"][0]),
        })
    return pl.DataFrame(rows)


def decompose_break_by_year(
    outcome: str = "otm", eras: dict = None, window: int = 30
) -> pl.DataFrame:
    """Decomposes a year-by-year difference-in-differences series into its
    four underlying levels, so a change in the coefficient can be attributed
    to the group that actually moved.

    Motivation: the OTM coefficient reverses sign between 2015 and 2016, but
    a DiD coefficient is a difference of differences -- it cannot say whether
    retail changed, professional customers changed, or the event response of
    one of them changed. This returns, per year and participant group, the
    mean outcome share in the baseline and near-event periods, plus the
    within-group event shift and each group's mean daily volume.

    Volume is included because a compositional explanation (who is being
    classified as a customer) and a behavioural one (how customers trade)
    have different signatures: a compositional shift should show up in
    relative volumes and baseline levels, a behavioural one mainly in the
    event shift."""
    eras = eras or yearly_eras()
    rows = []
    for name, (d0, d1) in eras.items():
        panel = build_diff_in_diff_panel(
            outcome=outcome, verbose=False, window=window, date_from=d0, date_to=d1
        )
        if panel.height < 100:
            continue
        agg = (
            panel.group_by(["participant_group", "is_near_event"])
            .agg(
                pl.col("share").mean().alias("mean_share"),
                pl.col("avg_daily_vol").mean().alias("mean_daily_vol"),
                pl.len().alias("n"),
            )
        )
        for grp in ["retail", "procust"]:
            base = agg.filter((pl.col("participant_group") == grp) & (~pl.col("is_near_event")))
            near = agg.filter((pl.col("participant_group") == grp) & (pl.col("is_near_event")))
            if base.height == 0 or near.height == 0:
                continue
            rows.append({
                "era": name,
                "participant_group": grp,
                "baseline_share": base["mean_share"][0],
                "near_event_share": near["mean_share"][0],
                "event_shift": near["mean_share"][0] - base["mean_share"][0],
                "baseline_daily_vol": base["mean_daily_vol"][0],
                "n_baseline": int(base["n"][0]),
                "n_near": int(near["n"][0]),
            })
    return pl.DataFrame(rows).sort(["era", "participant_group"])
