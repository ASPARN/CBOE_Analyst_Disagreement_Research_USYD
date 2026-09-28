"""
informed_trading.py
=====================
Test 1: does pre-announcement option flow predict the announcement return?

    CAR[0,+1]_j = a + b1 RetailNetPutFlow_j[-2,-1] + b2 ProNetPutFlow_j[-2,-1] + X'g + e_j

If retail put buying carries private information about the announcement
(Pan & Poteshman 2006), b1 < 0: net put buying before the news predicts a
negative announcement return. If it is uninformed or lottery-driven, b1 is
indistinguishable from zero (or positive). Professional flow enters the same
regression, so the test also asks which group -- if either -- trades on
information.

Design decisions, each made explicit here
-----------------------------------------
Effective day 0.
    IBES dates the announcement, ANNTIMS_ACT times it. An announcement at or
    after 16:00 ET reaches the market the NEXT trading day, so day 0 moves
    forward one day; otherwise day 0 is the announcement date (or the next
    trading day, if the date is not one). Getting this wrong would put
    post-news trading inside the "pre-event" flow window and start the CAR a
    day early.

Signed flow.
    Net put buying = put buys - put sells (opening + closing), from the
    signed daily flow table (build_daily_flow). Scaled by the group's total
    option volume in the window, so it lies in [-1, 1] and is comparable
    across firms. Two alternatives are available for robustness: the
    Pan-Poteshman opening-buy put ratio and net bearish flow (net put buying
    minus net call buying).

Event matching.
    Events are matched to CBOE tickers with event_window_profile's
    _resolve_tickers -- the same rule the rest of the thesis uses -- and to
    CRSP PERMNOs by 8-digit CUSIP, falling back to ticker.

Returns.
    CAR[0,+1] is the sum of daily market-adjusted returns (firm DlyRet minus
    the constructed value-weighted market return) over days 0 and +1.
    Earnings surprise (surprise_scaled) is available as an alternative
    dependent variable.

The CBOE gap.
    CBOE data is missing 2022-05-17 to 2022-07-31. Zero volume on those days
    would look like "no flow" rather than "no data", so any event whose flow
    window touches a day absent from the flow tables is dropped.

Inference.
    OLS with year-quarter fixed effects; standard errors clustered two ways,
    by firm and by day-0 date, because earnings announcements bunch in time
    and market-wide shocks correlate returns across firms on the same day.
    The dependent variable and flows are winsorised at 1/99 within each
    estimation sample; flows are standardised, so coefficients read as the
    change in CAR for a one-standard-deviation change in flow.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.append(str(Path(__file__).parent.parent))
from paths import DATA_DIR, IBES_DIR, CRSP_DIR
from analysis.event_window_profile import _resolve_tickers

RET_DIR = CRSP_DIR / "returns"
FLOW_DIRS = [DATA_DIR / "cboe_daily_flow", DATA_DIR / "cboe_daily_flow_ext"]
EVENT_FILES = [IBES_DIR / "dispersion_events.parquet", IBES_DIR / "dispersion_events_ext.parquet"]

GROUPS = ("retail", "procust")
ACTIONS = ("open_buy", "close_buy", "open_sell", "close_sell")

ERAS = {
    "pre_covid":  ("2016-01-01", "2019-12-31"),
    "covid":      ("2020-01-01", "2021-12-31"),
    "post_covid": ("2022-08-01", "2025-12-31"),
}

FLOW_WINDOW = (-2, -1)    # pre-announcement flow days, relative to day 0
CAR_WINDOW = (0, 1)       # announcement return days
PRE_RET_WINDOW = (-20, -3)  # pre-event return control (momentum / reversal)


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

def _calendar() -> pl.Series:
    """Market trading days, from the constructed market-return series."""
    return (pl.read_parquet(RET_DIR / "market_vw_constructed.parquet", columns=["DlyCalDt"])
            ["DlyCalDt"].cast(pl.Date).sort())


def _load_events(date_from: str, date_to: str) -> pl.DataFrame:
    want = ["OFTIC", "TICKER", "CUSIP", "FPEDATS", "ANNDATS_ACT", "ANNTIMS_ACT",
            "surprise_scaled", "dispersion_scaled", "earnings_volatility", "NUMEST"]
    frames = []
    for f in EVENT_FILES:
        if f.exists():
            lf = pl.scan_parquet(f)
            have = lf.collect_schema().names()
            frames.append(lf.select([c for c in want if c in have]).collect())
    ev = pl.concat(frames, how="diagonal_relaxed").unique()
    return ev.filter(
        pl.col("OFTIC").is_not_null()
        & (pl.col("ANNDATS_ACT") >= pl.lit(date_from).str.to_date())
        & (pl.col("ANNDATS_ACT") <= pl.lit(date_to).str.to_date())
    )


def _load_flow() -> pl.DataFrame:
    frames = [pl.scan_parquet(d / "daily_flow_*.parquet") for d in FLOW_DIRS
              if d.exists() and any(d.glob("daily_flow_*.parquet"))]
    if not frames:
        raise FileNotFoundError("No signed flow tables found -- run build_daily_flow first.")
    return (pl.concat(frames).with_columns(pl.col("quote_date").cast(pl.Date)).collect()
            .unique(subset=["underlying_symbol", "quote_date"], keep="last"))


def _crsp_link() -> pl.DataFrame:
    """(key, PERMNO, first date, last date) ranges from the local CRSP files,
    for both CUSIP (8-digit) and ticker keys."""
    frames = []
    for name in ("crsp_daily.parquet", "crsp_daily_ext.parquet"):
        f = CRSP_DIR / name
        if f.exists():
            frames.append(pl.scan_parquet(f).select(
                pl.col("PERMNO").cast(pl.Int64), pl.col("DlyCalDt").cast(pl.Date),
                pl.col("CUSIP").cast(pl.Utf8).str.slice(0, 8).alias("cusip8"),
                pl.col("Ticker").cast(pl.Utf8).alias("ticker")))
    base = pl.concat(frames)
    out = []
    for key in ("cusip8", "ticker"):
        out.append(base.filter(pl.col(key).is_not_null())
                   .group_by(key, "PERMNO")
                   .agg(pl.col("DlyCalDt").min().alias("d0"), pl.col("DlyCalDt").max().alias("d1"))
                   .rename({key: "key"})
                   .with_columns(pl.lit(key).alias("key_type"))
                   .collect())
    return pl.concat(out)


def _link_permno(ev: pl.DataFrame, link: pl.DataFrame) -> pl.DataFrame:
    """PERMNO for each event: CUSIP first, ticker as fallback, each requiring
    the event date to fall within the PERMNO's range for that key (with 30
    days' tolerance at the edges)."""
    tol = pl.duration(days=30)

    def _match(key_col: str, key_type: str) -> pl.DataFrame:
        cand = (ev.select("_eid", "ANNDATS_ACT", pl.col(key_col).alias("key"))
                .join(link.filter(pl.col("key_type") == key_type), on="key", how="inner")
                .filter((pl.col("ANNDATS_ACT") >= pl.col("d0") - tol)
                        & (pl.col("ANNDATS_ACT") <= pl.col("d1") + tol))
                .sort("_eid", "d1", descending=[False, True])
                .unique(subset="_eid", keep="first"))
        return cand.select("_eid", "PERMNO", pl.lit(key_type).alias("link_via"))

    by_cusip = _match("CUSIP", "cusip8")
    rest = ev.join(by_cusip.select("_eid"), on="_eid", how="anti")
    by_ticker = _match("OFTIC", "ticker").join(rest.select("_eid"), on="_eid", how="semi")
    return ev.join(pl.concat([by_cusip, by_ticker]), on="_eid", how="left")


# ---------------------------------------------------------------------------
# event panel
# ---------------------------------------------------------------------------

def _minutes_after_midnight(col: pl.Expr, dtype) -> pl.Expr:
    """Announcement time as minutes after midnight, parsed numerically.

    IBES times can arrive with or without a leading zero on the hour
    ('7:00:00' vs '07:00:00'), or as a Time type. Comparing them as text is
    wrong -- '7:00:00' sorts after '16:00:00' -- so the hour and minute are
    extracted as integers instead."""
    if dtype == pl.Time:
        return col.dt.hour().cast(pl.Int32) * 60 + col.dt.minute().cast(pl.Int32)
    txt = col.cast(pl.Utf8).str.strip_chars()
    h = txt.str.extract(r"^(\d{1,2}):", 1).cast(pl.Int32, strict=False)
    m = txt.str.extract(r"^\d{1,2}:(\d{1,2})", 1).cast(pl.Int32, strict=False)
    return h * 60 + m.fill_null(0)


def _day0_position(ev: pl.DataFrame, cal: pl.Series) -> pl.DataFrame:
    """Calendar position of the effective day 0, and the timing classification.

    before_open   (before 09:30)          day 0 = announcement date
    during_market (09:30 to 15:59)        day 0 = announcement date
    after_close   (16:00 or later)        day 0 = next trading day
    unknown       (missing, or 00:00)     day 0 = announcement date
    If the announcement date is not a trading day, day 0 is the next one
    regardless of time."""
    pos = cal.search_sorted(ev["ANNDATS_ACT"], side="left")   # first trading day >= date
    ev = ev.with_columns(pl.Series("_pos", pos).cast(pl.Int64))
    is_trading_day = pl.col("ANNDATS_ACT").is_in(cal.to_list())
    mins = _minutes_after_midnight(pl.col("ANNTIMS_ACT"), ev.schema["ANNTIMS_ACT"])
    timing = (pl.when(mins.is_null() | (mins == 0)).then(pl.lit("unknown"))
              .when(mins >= 16 * 60).then(pl.lit("after_close"))
              .when(mins < 9 * 60 + 30).then(pl.lit("before_open"))
              .otherwise(pl.lit("during_market")))
    return ev.with_columns(timing.alias("timing")).with_columns(
        pl.when(is_trading_day & (pl.col("timing") == "after_close"))
        .then(pl.col("_pos") + 1).otherwise(pl.col("_pos")).alias("pos0"))


def build_event_panel(date_from: str = "2016-01-01", date_to: str = "2025-12-31",
                      verbose: bool = True) -> pl.DataFrame:
    """One row per firm-event: pre-announcement flow for both groups, CAR[0,+1],
    SUE, controls and era."""
    cal = _calendar()
    n_cal = len(cal)
    ev = _load_events(date_from, date_to).with_row_index("_eid")
    n0 = ev.height

    # --- timing -> effective day 0 -------------------------------------------
    ev = _day0_position(ev, cal)
    earliest = min(FLOW_WINDOW[0], PRE_RET_WINDOW[0])
    ev = ev.filter((pl.col("pos0") + earliest >= 0) & (pl.col("pos0") + CAR_WINDOW[1] < n_cal))
    ev = ev.with_columns(cal.gather(ev["pos0"]).alias("day0"))

    # --- flow over [-2,-1] ----------------------------------------------------
    flow = _load_flow()
    covered = flow["quote_date"].unique().to_list()
    tmap = _resolve_tickers(ev["OFTIC"].unique().to_list(),
                            set(flow["underlying_symbol"].unique().to_list()))
    ev = ev.join(tmap, on="OFTIC", how="left")

    flow_offsets = pl.DataFrame({"k": list(range(FLOW_WINDOW[0], FLOW_WINDOW[1] + 1))})
    win = ev.select("_eid", "resolved_ticker", "pos0").join(flow_offsets, how="cross")
    win = win.with_columns(cal.gather(win["pos0"] + win["k"]).alias("quote_date"))
    # events whose flow window touches a day with no CBOE data at all (the 2022 gap)
    gap = win.filter(~pl.col("quote_date").is_in(covered)).select("_eid").unique()
    n_gap = gap.height
    win = win.join(gap, on="_eid", how="anti")

    fcols = [c for c in flow.columns if c not in ("underlying_symbol", "quote_date")]
    fw = (win.join(flow, left_on=["resolved_ticker", "quote_date"],
                   right_on=["underlying_symbol", "quote_date"], how="left")
          .with_columns([pl.col(c).fill_null(0) for c in fcols])
          .group_by("_eid").agg([pl.col(c).sum() for c in fcols]))

    measures = []
    for g in GROUPS:
        buys = lambda cp: pl.col(f"{g}_{cp}_open_buy") + pl.col(f"{g}_{cp}_close_buy")
        sells = lambda cp: pl.col(f"{g}_{cp}_open_sell") + pl.col(f"{g}_{cp}_close_sell")
        total = pl.sum_horizontal([pl.col(f"{g}_{cp}_{a}") for cp in ("call", "put") for a in ACTIONS])
        denom = pl.when(total > 0).then(total).otherwise(None)
        ob = pl.col(f"{g}_put_open_buy") + pl.col(f"{g}_call_open_buy")
        measures += [
            total.alias(f"{g}_vol"),
            ((buys("put") - sells("put")) / denom).alias(f"{g}_net_put"),
            (((buys("put") - sells("put")) - (buys("call") - sells("call"))) / denom).alias(f"{g}_net_bearish"),
            (pl.col(f"{g}_put_open_buy") / pl.when(ob > 0).then(ob).otherwise(None)).alias(f"{g}_pp_ratio"),
        ]
    fw = fw.with_columns(measures).select(["_eid"] + [f"{g}_{m}" for g in GROUPS
                                                     for m in ("vol", "net_put", "net_bearish", "pp_ratio")])
    ev = ev.join(fw, on="_eid", how="inner")
    n_flow = ev.height

    # --- PERMNO, returns, CAR, controls ---------------------------------------
    ev = _link_permno(ev, _crsp_link())
    n_linked = ev.filter(pl.col("PERMNO").is_not_null()).height
    ev = ev.filter(pl.col("PERMNO").is_not_null())

    ret_offsets = pl.DataFrame({"k": list(range(PRE_RET_WINDOW[0], CAR_WINDOW[1] + 1))})
    rw = ev.select("_eid", "PERMNO", "pos0").join(ret_offsets, how="cross")
    rw = rw.with_columns(cal.gather(rw["pos0"] + rw["k"]).alias("DlyCalDt"))

    permnos = ev["PERMNO"].unique().to_list()
    rets = (pl.scan_parquet(RET_DIR / "crsp_returns_*.parquet")
            .select(pl.col("PERMNO").cast(pl.Int64), pl.col("DlyCalDt").cast(pl.Date), "DlyRet")
            .filter(pl.col("PERMNO").is_in(permnos)).collect())
    mkt = (pl.read_parquet(RET_DIR / "market_vw_constructed.parquet")
           .select(pl.col("DlyCalDt").cast(pl.Date), pl.col("mkt_vw_all").alias("mkt")))
    caps = []
    for name in ("crsp_daily.parquet", "crsp_daily_ext.parquet"):
        f = CRSP_DIR / name
        if f.exists():
            caps.append(pl.scan_parquet(f).select(
                pl.col("PERMNO").cast(pl.Int64), pl.col("DlyCalDt").cast(pl.Date), "DlyCap")
                .filter(pl.col("PERMNO").is_in(permnos)))
    caps = pl.concat(caps).collect().unique(subset=["PERMNO", "DlyCalDt"], keep="last")

    rw = (rw.join(rets, on=["PERMNO", "DlyCalDt"], how="left")
            .join(mkt, on="DlyCalDt", how="left")
            .join(caps, on=["PERMNO", "DlyCalDt"], how="left")
            .with_columns((pl.col("DlyRet") - pl.col("mkt")).alias("ar")))

    in_car = pl.col("k").is_between(*CAR_WINDOW)
    in_pre = pl.col("k").is_between(*PRE_RET_WINDOW)
    agg = rw.group_by("_eid").agg(
        pl.col("ar").filter(in_car).sum().alias("car01"),
        pl.col("ar").filter(in_car).count().alias("_n_car"),
        pl.col("ar").filter(in_pre).sum().alias("pre_car"),
        pl.col("ar").filter(in_pre).count().alias("_n_pre"),
        pl.col("DlyCap").filter(in_pre).drop_nulls().last().alias("_cap"),
    )
    n_car_days = CAR_WINDOW[1] - CAR_WINDOW[0] + 1
    agg = agg.with_columns(
        pl.when(pl.col("_n_car") == n_car_days).then(pl.col("car01")).otherwise(None).alias("car01"),
        pl.when(pl.col("_n_pre") >= 10).then(pl.col("pre_car")).otherwise(None).alias("pre_car"),
        pl.when(pl.col("_cap") > 0).then(pl.col("_cap").log()).otherwise(None).alias("log_mktcap"),
    ).drop("_n_car", "_n_pre", "_cap")
    ev = ev.join(agg, on="_eid", how="left")

    # --- era and fixed-effect keys ---------------------------------------------
    era = pl.lit(None, dtype=pl.Utf8)
    for name, (a, b) in ERAS.items():
        era = pl.when(pl.col("day0").is_between(pl.lit(a).str.to_date(), pl.lit(b).str.to_date())) \
                .then(pl.lit(name)).otherwise(era)
    ev = ev.with_columns(
        era.alias("era"),
        (pl.col("day0").dt.year().cast(pl.Utf8) + "Q" + pl.col("day0").dt.quarter().cast(pl.Utf8)).alias("yq"),
        (pl.col("retail_vol") + pl.col("procust_vol") + 1).log().alias("log_opt_vol"),
    )

    if verbose:
        n_car = ev.filter(pl.col("car01").is_not_null()).height
        tim = ev["timing"].value_counts().sort("count", descending=True)
        print(f"Events {date_from} .. {date_to}: {n0:,}")
        print(f"  dropped {n_gap:,} whose flow window falls in a CBOE data gap")
        print(f"  with a complete flow window: {n_flow:,}  |  linked to a PERMNO: {n_linked:,}")
        print(f"  with a complete CAR[0,+1]: {n_car:,}")
        print("  announcement timing:", ", ".join(f"{r['timing']} {r['count']:,}" for r in tim.iter_rows(named=True)))
        share_bo = ev.filter(pl.col("timing") == "before_open").height / max(ev.height, 1)
        if share_bo < 0.15:
            print(f"  WARNING: only {share_bo:.1%} of announcements classed as before-open; US firms "
                  "typically split roughly evenly. Check the ANNTIMS_ACT format before trusting results.")
        for g in GROUPS:
            n = ev.filter(pl.col(f"{g}_net_put").is_not_null()).height
            print(f"  events with {g} option volume in [-2,-1]: {n:,}")
    return ev


# ---------------------------------------------------------------------------
# regressions
# ---------------------------------------------------------------------------

def _winsorise(s, p=0.01):
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lo, hi)


def run_informed_trading(panel: pl.DataFrame, dv: str = "car01", flow: str = "net_put",
                         sample: str = "both", era: str = None,
                         controls: tuple = ("log_mktcap", "pre_car", "dispersion_scaled", "log_opt_vol"),
                         fixed_effects: bool = True):
    """One regression. sample='both' needs both groups' flow (Andrew's
    specification); sample='retail' uses retail flow alone on every event
    with retail volume."""
    import statsmodels.formula.api as smf

    groups = ["retail", "procust"] if sample == "both" else ["retail"]
    fcols = [f"{g}_{flow}" for g in groups]
    df = panel if era is None else panel.filter(pl.col("era") == era)
    df = df.select(["PERMNO", "day0", "yq", dv] + fcols + list(controls)).drop_nulls().to_pandas()
    if len(df) < 50:
        return None

    df[dv] = _winsorise(df[dv])
    for c in fcols:
        w = _winsorise(df[c])
        df[f"{c}_z"] = (w - w.mean()) / w.std()
    used = []
    for c in controls:
        df[c] = _winsorise(df[c])
        if df[c].nunique() > 1:
            used.append(c)
        else:
            print(f"  note: control '{c}' has no variation in this sample ({era or 'pooled'}) -- dropped")

    rhs = " + ".join([f"{c}_z" for c in fcols] + used + (["C(yq)"] if fixed_effects else []))
    g_firm = df["PERMNO"].astype("category").cat.codes.to_numpy()
    g_date = df["day0"].astype("category").cat.codes.to_numpy()
    return smf.ols(f"{dv} ~ {rhs}", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": np.column_stack([g_firm, g_date])})


def informed_trading_table(panel: pl.DataFrame, dvs=("car01", "surprise_scaled"),
                           flows=("net_put",), samples=("both", "retail")) -> pl.DataFrame:
    """Every combination of dependent variable, flow measure, sample and era
    (plus the pooled 2016-2025 sample), as one tidy table."""
    rows = []
    for dv in dvs:
        for flow in flows:
            for sample in samples:
                for era in [None] + list(ERAS):
                    m = run_informed_trading(panel, dv=dv, flow=flow, sample=sample, era=era)
                    if m is None:
                        continue
                    row = {"dv": dv, "flow": flow, "sample": sample,
                           "era": era or "pooled", "n": int(m.nobs)}
                    for g in (["retail", "procust"] if sample == "both" else ["retail"]):
                        term = f"{g}_{flow}_z"
                        row[f"b_{g}"] = m.params[term]
                        row[f"se_{g}"] = m.bse[term]
                        row[f"p_{g}"] = m.pvalues[term]
                    rows.append(row)
    order = {"pooled": 0, **{e: i + 1 for i, e in enumerate(ERAS)}}
    return (pl.DataFrame(rows, infer_schema_length=None)
            .with_columns(pl.col("era").replace_strict(order).alias("_o"))
            .sort("dv", "flow", "sample", "_o").drop("_o"))
