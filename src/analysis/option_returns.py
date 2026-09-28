"""
option_returns.py
===================
Test 2: what did the options retail and professional customers traded
before an earnings announcement actually earn across it?

For every contract a group traded on day -1 (the last close before the news
reaches the market), the holding-period return to day +1 is

    R_c = ( V_c,+1 - P_c,-1 ) / P_c,-1

where P_c,-1 is the OptionMetrics bid-ask midpoint at 15:59 ET on day -1 and
V_c,+1 is the midpoint at day +1 -- or, for a contract that expires inside
the window, its payoff at expiry (dropping those would remove exactly the
short-dated weeklies retail favours). Day 0 is the timing-corrected
effective announcement day from Test 1 (informed_trading).

Group returns per firm-event
----------------------------
Selling a put earns the opposite of the put's return, so a single volume-
weighted return that treats every trade as a purchase would mix the two
sides. Each measure is therefore explicit about whose position it describes:

  vw_open_buy   volume-weighted mean return of contracts the group OPENED
                long -- the lottery-demand trade
  vw_buy        volume-weighted mean return of everything the group bought
                (Andrew's specification)
  dw_buy        dollar-weighted: total gain / total premium on what the
                group bought -- the actual return per dollar invested, and
                not dominated by a few very cheap contracts
  dw_sell       dollar-weighted return of the contracts the group SOLD (the
                seller earned the negative of this)
  dw_net        return on the group's net position (buys minus sells) per
                dollar of gross premium -- what the group actually earned

By default returns are midpoint to midpoint. event_option_returns(cost_frac=)
charges a fraction of the quoted half-spread on every trade (0 = midpoint,
1 = buy at the ask, sell at the bid), and spread_comparison() reports the
table at several cost levels. The spread matters most for cheap
out-of-the-money contracts. (The measure definitions below are superseded by
event_option_returns' docstring, which covers the cost-adjusted versions.)

Contract matching
-----------------
CBOE contracts (underlying, expiry, strike, call/put) are matched to
OptionMetrics quotes on the event's secid and entry date, with the
OptionMetrics strike divided by 1000. Standard-settlement contracts only
(ss_flag == '0'). The share of each group's day -1 volume that could be
priced is reported, per event, as coverage.

The contract-level panel is saved (option_contract_panel.parquet) with IVs
and Greeks at both ends, for the Test 3 decomposition.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.append(str(Path(__file__).parent.parent))
from paths import DATA_DIR
from analysis.informed_trading import _calendar, ERAS

OM_DIR = DATA_DIR / "optionmetrics"
CBOE_DIRS = [DATA_DIR / "cboe_parquet", DATA_DIR / "cboe_parquet_ext"]
PANEL_PATH = OM_DIR / "option_contract_panel.parquet"

GROUPS = {"cust": "retail", "procust": "procust"}
TIERS = ("lt_100", "100_199", "gt_199")
ACTIONS = ("open_buy", "close_buy", "open_sell", "close_sell")
ATM_BAND = 0.02

OPT_COLS = ["secid", "date", "optionid", "exdate", "cp_flag", "strike_price", "best_bid",
            "best_offer", "impl_volatility", "delta", "gamma", "vega", "theta",
            "ss_flag", "expiry_indicator"]


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------

def _years(dates: pl.Series) -> list[int]:
    return sorted(dates.dt.year().unique().to_list())


def _load_om_options(keys: pl.DataFrame) -> pl.DataFrame:
    """OptionMetrics quotes for the given (secid, date) pairs."""
    frames = []
    for y in _years(keys["date"]):
        d = OM_DIR / f"opprcd_{y}"
        if not d.exists():
            continue
        lf = pl.scan_parquet(d / "*.parquet")
        have = set(lf.collect_schema().names())
        frames.append(lf.select([c for c in OPT_COLS if c in have]).with_columns(
            pl.col("secid").cast(pl.Int64), pl.col("date").cast(pl.Date), pl.col("exdate").cast(pl.Date),
            pl.col("strike_price").cast(pl.Float64), pl.col("ss_flag").cast(pl.Utf8))
            .join(keys.lazy(), on=["secid", "date"], how="semi"))
    if not frames:
        raise FileNotFoundError("No OptionMetrics option files found -- run pull_optionmetrics first.")
    return pl.concat(frames, how="diagonal_relaxed").collect()


def _load_om_spot(keys: pl.DataFrame) -> pl.DataFrame:
    """Underlying closing price (absolute value: a negative close in secprd is
    a bid-ask midpoint on a no-trade day)."""
    frames = []
    for y in _years(keys["date"]):
        d = OM_DIR / f"secprd_{y}"
        if d.exists():
            frames.append(pl.scan_parquet(d / "*.parquet")
                          .select(pl.col("secid").cast(pl.Int64), pl.col("date").cast(pl.Date),
                                  pl.col("close").cast(pl.Float64).abs().alias("spot"))
                          .join(keys.lazy(), on=["secid", "date"], how="semi"))
    return pl.concat(frames).collect().unique(subset=["secid", "date"], keep="last")


def _load_cboe_contracts(keys: pl.DataFrame) -> pl.DataFrame:
    """Contract-level retail and professional volume by side, for the given
    (underlying_symbol, quote_date) pairs, summed across size tiers."""
    needed_vol = [f"{g}_{t}_{a}_vol" for g in GROUPS for t in TIERS for a in ACTIONS]
    years = set(_years(keys["quote_date"]))
    frames = []
    for d in CBOE_DIRS:
        for f in sorted(d.glob("*.parquet")):
            try:
                if int(f.stem.split("_")[-1]) not in years:
                    continue
            except ValueError:
                pass
            frames.append(
                pl.scan_parquet(f)
                .select(["underlying_symbol", "quote_date", "expiration_date", "strike_price", "call_put_flag"] + needed_vol)
                .with_columns(pl.col("quote_date").cast(pl.Date), pl.col("expiration_date").cast(pl.Date))
                .join(keys.lazy(), on=["underlying_symbol", "quote_date"], how="semi"))
    raw = pl.concat(frames).collect()
    aggs = []
    for rg, g in GROUPS.items():
        for a in ACTIONS:
            aggs.append(pl.sum_horizontal(pl.col(f"{rg}_{t}_{a}_vol").cast(pl.Int64).fill_null(0) for t in TIERS)
                        .sum().alias(f"{g}_{a}"))
    return (raw.with_columns(pl.col("strike_price").cast(pl.Float64).round(3).alias("strike"),
                             pl.col("call_put_flag").str.strip_chars().str.to_uppercase().alias("cp_flag"))
            .filter(pl.col("cp_flag").is_in(["C", "P"]))
            .group_by("underlying_symbol", "quote_date", pl.col("expiration_date").alias("exdate"), "strike", "cp_flag")
            .agg(aggs))


# ---------------------------------------------------------------------------
# contract panel
# ---------------------------------------------------------------------------

def build_option_panel(panel: pl.DataFrame, entry_offset: int = -1, exit_offset: int = 1,
                       min_price: float = 0.05, save: bool = True, verbose: bool = True) -> pl.DataFrame:
    """One row per (event, contract traded on the entry day), with group
    volumes by side, entry and exit prices, holding return, moneyness, and
    IV/Greeks at both ends.

    `panel` is the Test 1 event panel from informed_trading.build_event_panel.
    `min_price` drops contracts whose entry midpoint is below it (default 5
    cents): percentage returns on near-zero quotes are not meaningful."""
    cal = _calendar()
    n_cal = len(cal)

    pairs = pl.read_parquet(OM_DIR / "event_pairs.parquet", columns=["OFTIC", "ANNDATS_ACT", "secid"])
    secid_map = (pairs.with_columns(pl.col("secid").cast(pl.Int64), pl.col("ANNDATS_ACT").cast(pl.Date))
                 .unique(subset=["OFTIC", "ANNDATS_ACT"], keep="first"))

    ev = (panel.select("_eid", "OFTIC", "resolved_ticker", "ANNDATS_ACT", "pos0", "day0", "era",
                       "PERMNO", "timing")
          .filter((pl.col("pos0") + entry_offset >= 0) & (pl.col("pos0") + exit_offset < n_cal)))
    ev = ev.with_columns(cal.gather(ev["pos0"] + entry_offset).alias("entry_date"),
                         cal.gather(ev["pos0"] + exit_offset).alias("exit_date"))
    ev = ev.join(secid_map, on=["OFTIC", "ANNDATS_ACT"], how="inner")
    n_ev = ev.height

    # last date OptionMetrics covers: events exiting after it cannot be priced
    om_last = max(pl.scan_parquet(d / "*.parquet").select(pl.col("date").cast(pl.Date).max()).collect().item()
                  for d in OM_DIR.glob("opprcd_*") if d.is_dir())
    n_beyond = ev.filter(pl.col("exit_date") > om_last).height
    ev = ev.filter(pl.col("exit_date") <= om_last)

    # --- CBOE contracts traded on the entry day ------------------------------
    cboe = _load_cboe_contracts(ev.select(pl.col("resolved_ticker").alias("underlying_symbol"),
                                          pl.col("entry_date").alias("quote_date")).unique())
    con = ev.select("_eid", "secid", "entry_date", "exit_date",
                    pl.col("resolved_ticker").alias("underlying_symbol")).join(
        cboe, left_on=["underlying_symbol", "entry_date"], right_on=["underlying_symbol", "quote_date"], how="inner")

    # --- OptionMetrics quotes and spot at entry, exit, and expiry ------------
    exp_keys = con.select("secid", pl.col("exdate").alias("date"))
    keys = pl.concat([ev.select("secid", pl.col("entry_date").alias("date")),
                      ev.select("secid", pl.col("exit_date").alias("date")), exp_keys]).unique()
    om = _load_om_options(keys.filter(pl.col("date") <= om_last))
    om = om.with_columns((pl.col("strike_price") / 1000).round(3).alias("strike"),
                         ((pl.col("best_bid") + pl.col("best_offer")) / 2).alias("mid"),
                         ((pl.col("best_bid") >= 0) & (pl.col("best_offer") > 0)
                          & (pl.col("best_offer") >= pl.col("best_bid"))).alias("quote_ok"))
    spot = _load_om_spot(keys)

    greek_cols = [c for c in ("impl_volatility", "delta", "gamma", "vega", "theta") if c in om.columns]
    entry_q = om.select(["secid", pl.col("date").alias("entry_date"), "exdate", "strike", "cp_flag",
                         "optionid", "ss_flag", "quote_ok", pl.col("mid").alias("mid_entry"),
                         pl.col("best_bid").alias("bid_entry"), pl.col("best_offer").alias("ask_entry"), *greek_cols]
                        + (["expiry_indicator"] if "expiry_indicator" in om.columns else []))
    entry_q = entry_q.rename({c: f"{c}_entry" for c in greek_cols})
    exit_q = om.filter(pl.col("quote_ok")).select(
        "secid", "optionid", pl.col("date").alias("exit_date"), pl.col("mid").alias("mid_exit"),
        pl.col("best_bid").alias("bid_exit"), pl.col("best_offer").alias("ask_exit"),
        *[pl.col(c).alias(f"{c}_exit") for c in greek_cols])

    con = con.join(entry_q, on=["secid", "entry_date", "exdate", "strike", "cp_flag"], how="left")
    con = con.with_columns(
        (pl.col("optionid").is_not_null() & (pl.col("ss_flag") == "0") & pl.col("quote_ok")
         & (pl.col("mid_entry") >= min_price) & (pl.col("exdate") > pl.col("entry_date"))).alias("_priceable"))
    con = con.join(exit_q, on=["secid", "optionid", "exit_date"], how="left")
    con = (con.join(spot.rename({"date": "entry_date", "spot": "spot_entry"}), on=["secid", "entry_date"], how="left")
              .join(spot.rename({"date": "exit_date", "spot": "spot_exit"}), on=["secid", "exit_date"], how="left")
              .join(spot.rename({"date": "exdate", "spot": "spot_expiry"}), on=["secid", "exdate"], how="left"))

    intrinsic = (pl.when(pl.col("cp_flag") == "C").then((pl.col("spot_expiry") - pl.col("strike")).clip(lower_bound=0))
                 .otherwise((pl.col("strike") - pl.col("spot_expiry")).clip(lower_bound=0)))
    expired = pl.col("exdate") < pl.col("exit_date")
    con = con.with_columns(
        pl.when(~pl.col("_priceable")).then(None)
        .when(pl.col("mid_exit").is_not_null()).then(pl.col("mid_exit"))
        .when(expired).then(intrinsic)
        .otherwise(None).alias("value_exit"),
        pl.when(pl.col("_priceable") & pl.col("mid_exit").is_null() & expired)
        .then(True).otherwise(False).alias("valued_at_expiry"))
    con = con.with_columns(
        (pl.col("value_exit") - pl.col("mid_entry")).alias("dP"),
        (pl.col("value_exit") / pl.col("mid_entry") - 1).alias("ret"),
        (pl.col("strike") / pl.col("spot_entry")).log().alias("log_mny"))
    lm = pl.col("log_mny")
    otm = pl.when(pl.col("cp_flag") == "C").then(lm > ATM_BAND).otherwise(lm < -ATM_BAND)
    itm = pl.when(pl.col("cp_flag") == "C").then(lm < -ATM_BAND).otherwise(lm > ATM_BAND)
    con = con.with_columns(
        pl.when(lm.is_null()).then(pl.lit("unknown"))
        .when(lm.abs() <= ATM_BAND).then(pl.lit("atm"))
        .when(otm).then(pl.lit("otm")).when(itm).then(pl.lit("itm")).otherwise(pl.lit("unknown"))
        .alias("moneyness"))

    con = con.join(ev.select("_eid", "OFTIC", "ANNDATS_ACT", "day0", "era", "PERMNO", "timing"),
                   on="_eid", how="left")

    if verbose:
        priced = con.filter(pl.col("ret").is_not_null())
        print(f"Events with a secid and a valid window: {n_ev:,}  "
              f"(dropped {n_beyond:,} exiting after OptionMetrics ends on {om_last})")
        print(f"  events with CBOE contracts on the entry day: {con['_eid'].n_unique():,}")
        print(f"  contract-events: {con.height:,}  |  priced entry-to-exit: {priced.height:,} "
              f"({priced.height / max(con.height, 1):.1%})")
        print(f"  of which valued at payoff (expired inside the window): {priced['valued_at_expiry'].sum():,}")
        for g in GROUPS.values():
            b = pl.col(f"{g}_open_buy") + pl.col(f"{g}_close_buy")
            tot = con.select(b.sum()).item()
            got = priced.select(b.sum()).item()
            print(f"  {g} buy volume priced: {got / max(tot, 1):.1%} of {tot:,} contracts")
    if save:
        OM_DIR.mkdir(parents=True, exist_ok=True)
        con.write_parquet(PANEL_PATH, compression="zstd")
    return con


# ---------------------------------------------------------------------------
# event-level returns and the Test 2 table
# ---------------------------------------------------------------------------

def event_option_returns(contracts: pl.DataFrame, moneyness: str | None = None,
                         cost_frac: float = 0.0) -> pl.DataFrame:
    """Per firm-event, each group's option returns, from the group's own side.

    cost_frac (0 to 1) is the fraction of the quoted half-spread paid on each
    trade: buys execute at mid + cost_frac * half-spread, sells at mid -
    cost_frac * half-spread. 0 = midpoint to midpoint, 1 = buy at the ask and
    sell at the bid. Contracts settled at expiry pay no exit spread.

      vw_open_buy  volume-weighted mean return on contracts opened long
      vw_buy       volume-weighted mean return on everything bought
      dw_buy       dollar-weighted return on everything bought
      dw_short     dollar-weighted return to the group on contracts it SOLD
                   (receives the sell price, pays the buy price to close)
      dw_net       dollar-weighted return on the net position (longs where
                   the group net-bought a contract, shorts where it net-sold)
      dw_sell      cost-free return of the contracts sold, from the holder's
                   side (kept for comparison; at cost_frac 0 it is -dw_short)

    `moneyness` restricts to 'otm', 'atm' or 'itm' contracts."""
    if not 0 <= cost_frac <= 1:
        raise ValueError("cost_frac must be between 0 and 1")
    c = contracts.filter(pl.col("ret").is_not_null())
    if moneyness is not None:
        c = c.filter(pl.col("moneyness") == moneyness)

    if cost_frac > 0:
        missing = [x for x in ("bid_entry", "ask_entry", "bid_exit", "ask_exit") if x not in c.columns]
        if missing:
            raise KeyError(f"Contract panel lacks {missing}; rebuild it with build_option_panel().")
        half_in = (pl.col("ask_entry") - pl.col("bid_entry")) / 2
        half_out = (pl.when(pl.col("valued_at_expiry")).then(0.0)
                    .otherwise((pl.col("ask_exit") - pl.col("bid_exit")) / 2))
    else:
        half_in = half_out = pl.lit(0.0)

    c = c.with_columns(
        (pl.col("mid_entry") + cost_frac * half_in).alias("_buy_in"),
        (pl.col("mid_entry") - cost_frac * half_in).alias("_sell_in"),
        (pl.col("value_exit") - cost_frac * half_out).alias("_long_out"),
        (pl.col("value_exit") + cost_frac * half_out).alias("_short_out"),
    ).with_columns(
        (pl.col("_long_out") - pl.col("_buy_in")).alias("_long_dP"),
        (pl.col("_long_out") / pl.col("_buy_in") - 1).alias("_long_ret"),
        (pl.col("_sell_in") - pl.col("_short_out")).alias("_short_pnl"),
    )

    ratio = lambda num, den: pl.when(den > 0).then(num / den).otherwise(None)
    aggs = []
    for g in GROUPS.values():
        ob = pl.col(f"{g}_open_buy").cast(pl.Float64)
        b = ob + pl.col(f"{g}_close_buy")
        s = pl.col(f"{g}_open_sell").cast(pl.Float64) + pl.col(f"{g}_close_sell")
        n = b - s
        nl, ns = n.clip(lower_bound=0), (-n).clip(lower_bound=0)
        aggs += [
            ratio((ob * pl.col("_long_ret")).sum(), ob.sum()).alias(f"{g}_vw_open_buy"),
            ratio((b * pl.col("_long_ret")).sum(), b.sum()).alias(f"{g}_vw_buy"),
            ratio((b * pl.col("_long_dP")).sum(), (b * pl.col("_buy_in")).sum()).alias(f"{g}_dw_buy"),
            ratio((s * pl.col("_short_pnl")).sum(), (s * pl.col("_sell_in")).sum()).alias(f"{g}_dw_short"),
            ratio((nl * pl.col("_long_dP") + ns * pl.col("_short_pnl")).sum(),
                  (nl * pl.col("_buy_in") + ns * pl.col("_sell_in")).sum()).alias(f"{g}_dw_net"),
            ratio((s * pl.col("dP")).sum(), (s * pl.col("mid_entry")).sum()).alias(f"{g}_dw_sell"),
            b.sum().alias(f"{g}_buy_vol"),
        ]
    return (c.group_by("_eid", "OFTIC", "ANNDATS_ACT", "day0", "era", "PERMNO").agg(aggs)
            .with_columns(pl.lit(cost_frac).alias("cost_frac"))
            .sort("day0"))


MEASURES = ("vw_open_buy", "vw_buy", "dw_buy", "dw_short", "dw_net")


def _mean_test(df, col):
    """Mean of col with two-way (firm, date) clustered SE; col winsorised 1/99."""
    import statsmodels.formula.api as smf
    d = df.select("PERMNO", "day0", col).drop_nulls().to_pandas()
    if len(d) < 30:
        return None
    lo, hi = d[col].quantile(0.01), d[col].quantile(0.99)
    d["y"] = d[col].clip(lo, hi)
    g = np.column_stack([d["PERMNO"].astype("category").cat.codes, d["day0"].astype("category").cat.codes])
    m = smf.ols("y ~ 1", data=d).fit(cov_type="cluster", cov_kwds={"groups": g})
    return {"n": int(m.nobs), "mean": m.params["Intercept"], "se": m.bse["Intercept"],
            "p": m.pvalues["Intercept"], "median": float(np.median(d[col]))}


def option_returns_table(event_returns: pl.DataFrame, measures=MEASURES) -> pl.DataFrame:
    """For each measure and era: retail mean, professional mean, and the
    within-event difference (retail minus professional, events where both
    traded), each with firm-and-date clustered standard errors."""
    rows = []
    for meas in measures:
        for era in [None] + list(ERAS):
            df = event_returns if era is None else event_returns.filter(pl.col("era") == era)
            df = df.with_columns((pl.col(f"retail_{meas}") - pl.col(f"procust_{meas}")).alias("_diff"))
            r = {"measure": meas, "era": era or "pooled"}
            for label, col in (("retail", f"retail_{meas}"), ("procust", f"procust_{meas}"), ("diff", "_diff")):
                t = _mean_test(df, col)
                for k in ("n", "mean", "se", "p", "median"):
                    if label != "diff" or k != "median":
                        r[f"{label}_{k}"] = None if t is None else t[k]
            rows.append(r)
    return pl.DataFrame(rows, infer_schema_length=None)


def spread_comparison(contracts: pl.DataFrame, cost_fracs=(0.0, 0.5, 1.0),
                      measures=("dw_buy", "dw_net"), moneyness: str | None = None) -> pl.DataFrame:
    """option_returns_table at several cost levels, stacked, so the effect of
    the bid-ask spread on each group's returns can be read off directly."""
    out = []
    for cf in cost_fracs:
        er = event_option_returns(contracts, moneyness=moneyness, cost_frac=cf)
        out.append(option_returns_table(er, measures=measures).with_columns(pl.lit(cf).alias("cost_frac")))
    return pl.concat(out).select(["cost_frac"] + [c for c in out[0].columns if c != "cost_frac"])
