"""
sector_analysis.py
====================
Tests 1-3 by industry, organised around three hypotheses stated BEFORE any
sector result was seen. They are the confirmatory tests; every other sector
comparison in this module is descriptive and should be reported as such.

H1  Technology shows more speculative retail demand, because it receives
    the most media attention. Measures: retail share of customer option
    volume, retail out-of-the-money share, retail short-dated (<= 7 days)
    share, and retail's return on options bought (midpoint and half-spread).
    Prediction: tech coefficient > 0 for demand measures, < 0 for returns.
    Mechanism check: the tech coefficient should shrink once pre-announcement
    attention (abnormal trading volume) is controlled for.

H2  Earnings are a minor event for pharma & biotech, whose largest news is
    regulatory (FDA decisions, trial readouts). Predictions: a smaller
    announcement move relative to normal volatility, a smaller IV crush in
    near-the-money options, and weaker information in pre-announcement flow
    (flow x biotech interaction of the opposite sign to the flow effect).

H3  Flow is more informative in energy & industrials, whose earnings are
    more forecastable from public data (commodity prices, input costs,
    backlogs). Prediction: flow x energy_industrials interaction < 0 for
    announcement returns and surprises, especially for professional flow.

Design
------
- Sector differences are tested DIRECTLY -- dummies and flow x sector
  interactions in one regression -- never by comparing separate per-sector
  regressions (a significant result in one group and an insignificant one in
  another does not show that the two differ).
- Every test controls for firm size (log market cap) and year-quarter fixed
  effects; standard errors are clustered by firm and by announcement date.
- Tests pool 2016-2025; eras are a secondary check, since small sectors
  thin out quickly once split.

Sectors (from CRSP NAICS, as of each announcement date)
-------------------------------------------------------
Rules are applied in order, so specific codes win over broad prefixes:
  Pharma & biotech  3254 (pharma manufacturing), 541711 / 541714 (biotech R&D)
  Energy            211, 2121, 213111, 213112, 324, 486
  Technology        334, 5112 / 51321 (software, pre/post 2022 NAICS), 518,
                    5191 / 51929, 5415
  Utilities         22
  Financials        52, 531
  Consumer & retail 311-316, 44, 45, 71, 72
  Industrials       remaining 21, 23, 31-33, 42, 48, 49
  Other             everything else;  Unknown = no NAICS
Borderline assignments worth knowing: e-commerce (NAICS 45/454) is consumer &
retail, carmakers (3361) are industrials, medical instruments (3345) are
technology, and media and telecom (512-517) are other.
sector_check() prints the most frequent firms per sector to verify the rules.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.append(str(Path(__file__).parent.parent))
from paths import CRSP_DIR, RESULTS_DIR
from analysis.informed_trading import _calendar, _winsorise, ERAS
from analysis.option_returns import event_option_returns

RET_DIR = CRSP_DIR / "returns"
KEY = ["OFTIC", "ANNDATS_ACT"]

SECTOR_RULES = [
    ("Pharma & biotech", ["3254", "541711", "541714"]),
    ("Energy", ["211", "2121", "213111", "213112", "324", "486"]),
    ("Technology", ["334", "5112", "51321", "518", "5191", "51929", "5415"]),
    ("Utilities", ["22"]),
    ("Financials", ["52", "531"]),
    ("Consumer & retail", ["311", "312", "313", "314", "315", "316", "44", "45", "71", "72"]),
    ("Industrials", ["21", "23", "31", "32", "33", "42", "48", "49"]),
]
CONTROLS = ["log_mktcap", "pre_car", "dispersion_scaled", "log_opt_vol"]


# ---------------------------------------------------------------------------
# sector assignment
# ---------------------------------------------------------------------------

def sector_expr(naics: pl.Expr) -> pl.Expr:
    code = naics.cast(pl.Utf8).str.strip_chars()
    expr = pl.when(code.is_null() | (code == "") | (code == "0")).then(pl.lit("Unknown"))
    for sector, prefixes in SECTOR_RULES:
        expr = expr.when(pl.any_horizontal([code.str.starts_with(p) for p in prefixes])).then(pl.lit(sector))
    return expr.otherwise(pl.lit("Other"))


def _crsp(cols: list[str], permnos: list[int]) -> pl.DataFrame:
    frames = []
    for name in ("crsp_daily.parquet", "crsp_daily_ext.parquet"):
        f = CRSP_DIR / name
        if f.exists():
            frames.append(pl.scan_parquet(f).select(
                [pl.col("PERMNO").cast(pl.Int64), pl.col("DlyCalDt").cast(pl.Date)] + [pl.col(c) for c in cols])
                .filter(pl.col("PERMNO").is_in(permnos)))
    return pl.concat(frames).collect().unique(subset=["PERMNO", "DlyCalDt"], keep="last")


def attach_sector(ev: pl.DataFrame) -> pl.DataFrame:
    """NAICS and sector for each event, as of the announcement date (the
    firm's classification can change over time)."""
    permnos = ev["PERMNO"].drop_nulls().unique().to_list()
    n = (_crsp(["NAICS"], permnos)
         .with_columns(pl.col("NAICS").cast(pl.Utf8).str.strip_chars().alias("NAICS"))
         .sort("PERMNO", "DlyCalDt"))
    # keep only the dates where a firm's code changes
    changes = (n.filter((pl.col("NAICS") != pl.col("NAICS").shift(1).over("PERMNO"))
                        | pl.col("NAICS").shift(1).over("PERMNO").is_null())
               .rename({"DlyCalDt": "_from"}))
    first = changes.group_by("PERMNO").agg(pl.col("NAICS").first().alias("_first"))
    out = (ev.sort("day0")
           .join_asof(changes.sort("_from"), left_on="day0", right_on="_from", by="PERMNO",
                      strategy="backward", check_sortedness=False)
           .join(first, on="PERMNO", how="left")
           .with_columns(pl.coalesce("NAICS", "_first").alias("NAICS"))
           .drop("_from", "_first"))
    return out.with_columns(sector_expr(pl.col("NAICS")).alias("sector"))


def sector_check(sp: pl.DataFrame, top: int = 8) -> pl.DataFrame:
    """Most frequent firms in each sector -- a quick check that the NAICS
    rules put familiar names where they belong."""
    return (sp.group_by("sector", "OFTIC").len().sort("len", descending=True)
            .group_by("sector", maintain_order=True)
            .agg(pl.len().alias("_n"), pl.col("OFTIC").head(top).str.join(", ").alias("top_firms"))
            .drop("_n")
            .join(sp.group_by("sector").len().rename({"len": "events"}), on="sector")
            .sort("events", descending=True)
            .select("sector", "events", "top_firms"))


# ---------------------------------------------------------------------------
# event-level measures
# ---------------------------------------------------------------------------

def _attention_and_volatility(ev: pl.DataFrame) -> pl.DataFrame:
    """Abnormal pre-announcement volume (attention proxy) and pre-event
    abnormal-return volatility, from CRSP.
        abn_vol  = log( mean volume, days -3..-1 / mean volume, days -60..-11 )
        pre_sd   = standard deviation of daily abnormal returns, days -60..-11
    Both need at least 30 baseline days."""
    cal = _calendar()
    base = (ev.select(KEY + ["PERMNO", "pos0"]).drop_nulls()
            .join(pl.DataFrame({"k": list(range(-60, 0))}), how="cross")
            .filter(pl.col("pos0") + pl.col("k") >= 0))
    base = base.with_columns(cal.gather(base["pos0"] + base["k"]).alias("DlyCalDt"))
    permnos = base["PERMNO"].unique().to_list()
    vol = _crsp(["DlyVol"], permnos)
    rets = (pl.scan_parquet(RET_DIR / "crsp_returns_*.parquet")
            .select(pl.col("PERMNO").cast(pl.Int64), pl.col("DlyCalDt").cast(pl.Date), "DlyRet")
            .filter(pl.col("PERMNO").is_in(permnos)).collect())
    mkt = (pl.read_parquet(RET_DIR / "market_vw_constructed.parquet")
           .select(pl.col("DlyCalDt").cast(pl.Date), pl.col("mkt_vw_all").alias("mkt")))
    w = (base.join(vol, on=["PERMNO", "DlyCalDt"], how="left")
             .join(rets, on=["PERMNO", "DlyCalDt"], how="left")
             .join(mkt, on="DlyCalDt", how="left")
             .with_columns((pl.col("DlyRet") - pl.col("mkt")).alias("ar")))
    recent, baseline = pl.col("k").is_between(-3, -1), pl.col("k").is_between(-60, -11)
    agg = w.group_by(KEY).agg(
        pl.col("DlyVol").filter(recent).mean().alias("_v_recent"),
        pl.col("DlyVol").filter(baseline).mean().alias("_v_base"),
        pl.col("DlyVol").filter(baseline).count().alias("_nv"),
        pl.col("ar").filter(baseline).std().alias("_sd"),
        pl.col("ar").filter(baseline).count().alias("_na"),
    )
    return agg.select(
        *KEY,
        pl.when((pl.col("_nv") >= 30) & (pl.col("_v_base") > 0) & (pl.col("_v_recent") > 0))
        .then((pl.col("_v_recent") / pl.col("_v_base")).log()).otherwise(None).alias("abn_vol"),
        pl.when(pl.col("_na") >= 30).then(pl.col("_sd")).otherwise(None).alias("pre_sd"),
    )


def _option_measures(contracts: pl.DataFrame) -> pl.DataFrame:
    """Per event, from the contract panel: retail demand shape, retail
    returns, and the near-the-money IV crush."""
    c = contracts.with_columns(pl.col("ANNDATS_ACT").cast(pl.Date))
    b = (pl.col("retail_open_buy") + pl.col("retail_close_buy")).cast(pl.Float64)
    known = c.filter(pl.col("moneyness") != "unknown")
    demand = known.group_by(KEY).agg(
        pl.when(b.sum() > 0).then((b * (pl.col("moneyness") == "otm")).sum() / b.sum())
        .otherwise(None).alias("retail_otm_share"),
        pl.when(b.sum() > 0).then(
            (b * ((pl.col("exdate") - pl.col("entry_date")).dt.total_days() <= 7)).sum() / b.sum())
        .otherwise(None).alias("retail_short_share"),
    )
    crush = (c.filter(~pl.col("valued_at_expiry").fill_null(True)
                      & (pl.col("impl_volatility_entry") > 0) & (pl.col("impl_volatility_exit") > 0)
                      & (pl.col("log_mny").abs() <= 0.05))
             .with_columns((pl.col("impl_volatility_exit") - pl.col("impl_volatility_entry")).alias("_ds"))
             .group_by(KEY).agg(
                 pl.col("_ds").median().alias("atm_iv_change"),
                 (pl.col("_ds") / pl.col("impl_volatility_entry")).median().alias("atm_iv_change_rel")))
    r0 = event_option_returns(c, cost_frac=0.0).select(KEY + [pl.col("retail_dw_buy").alias("retail_ret_mid")])
    r5 = event_option_returns(c, cost_frac=0.5).select(KEY + [pl.col("retail_dw_buy").alias("retail_ret_half")])
    out = demand.join(crush, on=KEY, how="full", coalesce=True)
    for r in (r0, r5):
        out = out.join(r.with_columns(pl.col("ANNDATS_ACT").cast(pl.Date)), on=KEY, how="full", coalesce=True)
    return out


def build_sector_panel(panel: pl.DataFrame, contracts: pl.DataFrame | None = None,
                       verbose: bool = True) -> pl.DataFrame:
    """One row per firm-event: the Test 1 panel plus sector, attention,
    pre-event volatility, the announcement move relative to normal
    volatility, and (if `contracts` is given) the option measures.
    Events are keyed by (OFTIC, ANNDATS_ACT), which is stable across runs."""
    ev = (panel.with_columns(pl.col("ANNDATS_ACT").cast(pl.Date))
          .unique(subset=KEY, keep="first")
          .filter(pl.col("PERMNO").is_not_null()))
    ev = attach_sector(ev)
    ev = ev.join(_attention_and_volatility(ev), on=KEY, how="left")
    ev = ev.with_columns(
        pl.col("car01").abs().alias("abs_car"),
        (pl.col("car01").abs() / (pl.col("pre_sd") * np.sqrt(2))).alias("event_ratio"),
        (pl.col("retail_vol") / (pl.col("retail_vol") + pl.col("procust_vol"))).alias("retail_share"),
    )
    if contracts is not None:
        ev = ev.join(_option_measures(contracts), on=KEY, how="left")
    ev = ev.with_columns(
        (pl.col("sector") == "Technology").cast(pl.Int8).alias("tech"),
        (pl.col("sector") == "Pharma & biotech").cast(pl.Int8).alias("biotech"),
        pl.col("sector").is_in(["Energy", "Industrials"]).cast(pl.Int8).alias("energy_industrials"),
    )
    if verbose:
        n_unknown = ev.filter(pl.col("sector") == "Unknown").height
        print(f"Sector panel: {ev.height:,} firm-events; {n_unknown:,} without a NAICS code")
        print(f"  attention proxy available: {ev['abn_vol'].drop_nulls().len():,}  |  "
              f"event ratio available: {ev['event_ratio'].drop_nulls().len():,}")
        if contracts is not None:
            print(f"  with option measures: {ev['retail_ret_mid'].drop_nulls().len():,}")
    return ev


# ---------------------------------------------------------------------------
# regressions
# ---------------------------------------------------------------------------

def _fit(df: pl.DataFrame, y: str, terms: list[str], continuous: list[str]):
    """OLS of y on terms + year-quarter FE, two-way clustered (firm, date).
    y and the listed continuous regressors are winsorised at 1/99."""
    import statsmodels.formula.api as smf
    cols = list(dict.fromkeys(["PERMNO", "day0", "yq", y] + [t for t in terms if ":" not in t] + continuous))
    d = df.select(cols).drop_nulls().to_pandas()
    if len(d) < 100 or d[y].nunique() <= 1:
        return None, len(d)          # too few events, or an outcome with no variation
    d[y] = _winsorise(d[y])
    for c in continuous:
        if c != y:
            d[c] = _winsorise(d[c])
    g1 = d["PERMNO"].astype("category").cat.codes.to_numpy()
    g2 = d["day0"].astype("category").cat.codes.to_numpy()
    rhs = " + ".join(terms + ["C(yq)"])
    m = smf.ols(f"{y} ~ {rhs}", data=d).fit(cov_type="cluster", cov_kwds={"groups": np.column_stack([g1, g2])})
    return m, int(m.nobs)


def _row(hyp, test, y, term, m, n, **extra):
    if m is None or term not in m.params.index:
        return {"hypothesis": hyp, "test": test, "outcome": y, "term": term, "n": n,
                "coef": None, "se": None, "p": None, **extra}
    return {"hypothesis": hyp, "test": test, "outcome": y, "term": term, "n": n,
            "coef": m.params[term], "se": m.bse[term], "p": m.pvalues[term], **extra}


H1_OUTCOMES = ["retail_share", "retail_otm_share", "retail_short_share", "retail_ret_mid", "retail_ret_half"]
H2_OUTCOMES = ["event_ratio", "abs_car", "atm_iv_change", "atm_iv_change_rel"]


def test_h1_tech(sp: pl.DataFrame) -> pl.DataFrame:
    """Tech dummy on each speculative-demand measure, with a size control;
    then the same sample with the attention proxy added. If attention drives
    tech's excess speculation, the tech coefficient shrinks in the second."""
    rows = []
    for y in [o for o in H1_OUTCOMES if o in sp.columns]:
        sub = sp.drop_nulls([y, "log_mktcap", "abn_vol"])      # same sample for both fits
        m1, n1 = _fit(sub, y, ["tech", "log_mktcap"], [y, "log_mktcap"])
        m2, n2 = _fit(sub, y, ["tech", "log_mktcap", "abn_vol"], [y, "log_mktcap", "abn_vol"])
        rows.append(_row("H1", "tech, size-controlled", y, "tech", m1, n1))
        rows.append(_row("H1", "tech, + attention", y, "tech", m2, n2))
        rows.append(_row("H1", "tech, + attention", y, "abn_vol", m2, n2))
    return pl.DataFrame(rows)


def test_h2_biotech(sp: pl.DataFrame) -> pl.DataFrame:
    """Biotech dummy on the size of the announcement event: the move relative
    to normal volatility, the absolute move, and the near-the-money IV
    change (a smaller crush = a less negative change = a positive coefficient)."""
    rows = []
    for y in [o for o in H2_OUTCOMES if o in sp.columns]:
        m, n = _fit(sp, y, ["biotech", "log_mktcap"], [y, "log_mktcap"])
        rows.append(_row("H2", "biotech, size-controlled", y, "biotech", m, n))
    return pl.DataFrame(rows)


def test_flow_interactions(sp: pl.DataFrame, flags=("energy_industrials", "biotech"),
                           dvs=("car01", "surprise_scaled"), groups=("retail", "procust")) -> pl.DataFrame:
    """Does pre-announcement flow predict the announcement differently in a
    sector?  dv ~ flow_z + flow_z:flag + flag + controls + FE, estimated
    separately for retail flow (events with retail flow) and professional
    flow (events with professional flow). The interaction is the test."""
    hyp = {"energy_industrials": "H3", "biotech": "H2"}
    rows = []
    for g in groups:
        for flag in flags:
            for dv in dvs:
                d = sp.drop_nulls([f"{g}_net_put", dv] + CONTROLS)
                if d.height < 100:
                    continue
                w = _winsorise(d[f"{g}_net_put"].to_pandas())
                d = d.with_columns(pl.Series("flow_z", ((w - w.mean()) / w.std()).to_numpy()))
                m, n = _fit(d, dv, ["flow_z", f"flow_z:{flag}", flag] + CONTROLS, [dv] + CONTROLS)
                for term in ("flow_z", f"flow_z:{flag}"):
                    rows.append(_row(hyp.get(flag, "descriptive"), f"{g} flow x {flag}", dv, term, m, n, group=g))
    return pl.DataFrame(rows, infer_schema_length=None)


def sector_profile(sp: pl.DataFrame) -> pl.DataFrame:
    """Descriptive: medians of the key measures by sector (not a test)."""
    meas = [c for c in ["log_mktcap", "retail_share", "retail_otm_share", "retail_short_share",
                        "retail_ret_mid", "retail_ret_half", "event_ratio", "atm_iv_change_rel", "abn_vol"]
            if c in sp.columns]
    return (sp.group_by("sector")
            .agg(pl.len().alias("events"), *[pl.col(m).median().alias(m) for m in meas])
            .sort("events", descending=True))


def run_sector_tests(sp: pl.DataFrame) -> pl.DataFrame:
    return pl.concat([test_h1_tech(sp), test_h2_biotech(sp), test_flow_interactions(sp)],
                     how="diagonal_relaxed")


def write_sector_results(sp: pl.DataFrame, out_dir: Path | None = None) -> dict:
    out_dir = out_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    prof, tests, check = sector_profile(sp), run_sector_tests(sp), sector_check(sp)
    prof.write_csv(out_dir / "table12_sector_profile.csv", float_precision=6)
    tests.write_csv(out_dir / "table13_sector_hypotheses.csv", float_precision=6)
    check.write_csv(out_dir / "table12b_sector_membership.csv")
    print("  wrote table12_sector_profile.csv, table12b_sector_membership.csv, table13_sector_hypotheses.csv")
    return {"profile": prof, "tests": tests, "check": check}
