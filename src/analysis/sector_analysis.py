"""
sector_analysis.py
====================
Tests 1-3 by industry, organised around hypotheses stated BEFORE any sector
result was seen. The first version (commit "Add sector analysis (B5)") stated
three hypotheses; they were revised to the set below, still before any
result was seen. These are the confirmatory tests; everything else in this
module is exploratory or descriptive and is labelled as such.

H1 (primary)  Retail option flow is more informative about earnings
    announcements in knowledge-intensive industries -- those whose workforce
    is more technical -- because their employees and others with industry
    expertise understand announcements better. Knowledge intensity is the
    industry's STEM employment share (BLS OEWS; build_knowledge_intensity).
    Prediction: in  CAR ~ flow + flow x KI + KI + flow x attention + ...
    the flow x KI interaction is NEGATIVE (net put buying predicts lower
    returns more strongly as KI rises), net of attention. Variants: among
    low-attention events; with abnormal flow (relative to the firm's own
    previous events); with short-dated OTM opening buys (the instrument
    informed traders favour); excluding pharma & biotech. Plus a tail test:
    do the most extreme abnormal-flow events call the direction of the move
    more often in high-KI industries?
    The data identify informed trading, not insider trading: CBOE records
    that retail flow predicted the announcement, not who traded or why.

H2 (secondary, the confound)  Attention dilutes flow informativeness:
    heavily covered firms attract uninformed traders whose volume masks any
    informed signal. Prediction: flow x attention interaction POSITIVE (the
    negative flow effect weakens as attention rises).

H3 (kept, reframed)  Earnings are a minor event for pharma & biotech, whose
    largest news is regulatory. Prediction: a smaller announcement move
    relative to normal volatility, a milder near-the-money IV crush. This can
    coexist with H1: biotech's informed trading may sit around regulatory
    events an earnings study cannot see, which is why H1 is also run
    excluding biotech.

Exploratory  flow x energy & industrials (the original H3), and tech's
    speculative demand (the original H1), reported as descriptive.

Design
------
- Sector differences are tested DIRECTLY -- interactions in one regression --
  never by comparing separate per-sector regressions.
- Every test controls for firm size and year-quarter fixed effects; standard
  errors are clustered by firm and by announcement date. KI and attention
  enter as z-scores, so an interaction reads as the change in the flow
  effect per one-standard-deviation change in KI (or attention).
- Tests pool 2016-2025; eras are a secondary check.

Sectors (from CRSP NAICS, as of each announcement date) are used for the
biotech, exploratory and descriptive parts; rules are applied in order so
specific codes win over broad prefixes:
  Pharma & biotech  3254 (pharma manufacturing), 541711 / 541714 (biotech R&D)
  Energy            211, 2121, 213111, 213112, 324, 486
  Technology        334, 5112 / 51321 (software, pre/post 2022 NAICS), 518,
                    5191 / 51929, 5415
  Utilities         22
  Financials        52, 531
  Consumer & retail 311-316, 44, 45, 71, 72
  Industrials       remaining 21, 23, 31-33, 42, 48, 49
  Other             everything else;  Unknown = no NAICS
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.append(str(Path(__file__).parent.parent))
from paths import CRSP_DIR, DATA_DIR, RESULTS_DIR
from analysis.informed_trading import _calendar, _winsorise, ERAS
from analysis.option_returns import event_option_returns

RET_DIR = CRSP_DIR / "returns"
STEM_PATH = DATA_DIR / "external" / "oews" / "stem_share.parquet"
KEY = ["OFTIC", "ANNDATS_ACT"]

# 4-digit NAICS codes that changed in the 2022 revision (information sector),
# translated before matching firms' codes to the 2022-based OEWS industries
NAICS_2022_CROSSWALK = {"5111": "5131", "5112": "5132", "5151": "5161", "5152": "5162",
                        "5173": "5171", "5191": "5192"}

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


def attach_knowledge_intensity(ev: pl.DataFrame, stem_path: Path = STEM_PATH) -> pl.DataFrame:
    """STEM employment share of each firm's industry. Matched at the 4-digit
    NAICS level where OEWS publishes it (after translating codes changed in
    the 2022 revision), else 3-digit, else sector. `ki_level` records which."""
    if not stem_path.exists():
        print(f"  note: {stem_path.name} not found -- run build_knowledge_intensity first; KI tests skipped")
        return ev.with_columns(pl.lit(None, dtype=pl.Float64).alias("stem_share"),
                               pl.lit(None, dtype=pl.Int64).alias("ki_level"))
    stem = pl.read_parquet(stem_path)
    code = pl.col("NAICS").cast(pl.Utf8).str.strip_chars()
    d4 = code.str.slice(0, 4).replace(NAICS_2022_CROSSWALK)
    ev = ev.with_columns(d4.alias("_k4"), code.str.slice(0, 3).alias("_k3"), code.str.slice(0, 2).alias("_k2"))
    for lvl in (4, 3, 2):
        m = stem.filter(pl.col("level") == lvl).select(
            pl.col("naics_key").alias(f"_k{lvl}"), pl.col("stem_share").alias(f"_s{lvl}"))
        ev = ev.join(m, on=f"_k{lvl}", how="left")
    ev = ev.with_columns(
        pl.coalesce("_s4", "_s3", "_s2").alias("stem_share"),
        pl.when(pl.col("_s4").is_not_null()).then(4).when(pl.col("_s3").is_not_null()).then(3)
        .when(pl.col("_s2").is_not_null()).then(2).otherwise(None).alias("ki_level"))
    return ev.drop("_k4", "_k3", "_k2", "_s4", "_s3", "_s2")


def _abnormal_flow(ev: pl.DataFrame, col: str, min_prior: int = 4) -> pl.Expr:
    """Flow relative to the firm's own PREVIOUS events: (x - mean) / sd over
    events before this one, requiring at least `min_prior`. Only earlier
    events enter, so no future information is used."""
    x = pl.col(col)
    valid = x.is_not_null().cast(pl.Float64)
    xf = x.fill_null(0.0)
    n_prev = (valid.cum_sum() - valid).over("PERMNO")
    s_prev = (xf.cum_sum() - xf).over("PERMNO")
    q_prev = ((xf ** 2).cum_sum() - xf ** 2).over("PERMNO")
    mean = s_prev / n_prev
    var = q_prev / n_prev - mean ** 2
    return (pl.when((n_prev >= min_prior) & (var > 1e-12) & x.is_not_null())
            .then((x - mean) / var.sqrt()).otherwise(None))


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
    # short-dated OTM opening buys, puts minus calls, over retail's total day -1 volume:
    # the instrument informed traders favour (maximum leverage)
    ob = pl.col("retail_open_buy").cast(pl.Float64)
    tot = sum(pl.col(f"retail_{a}").cast(pl.Float64) for a in ("open_buy", "close_buy", "open_sell", "close_sell"))
    s_otm = (pl.col("moneyness") == "otm") & ((pl.col("exdate") - pl.col("entry_date")).dt.total_days() <= 14)
    sotm = known.group_by(KEY).agg(
        pl.when(tot.sum() > 0).then(
            ((ob * (s_otm & (pl.col("cp_flag") == "P"))).sum() - (ob * (s_otm & (pl.col("cp_flag") == "C"))).sum())
            / tot.sum()).otherwise(None).alias("retail_sotm_flow"))
    crush = (c.filter(~pl.col("valued_at_expiry").fill_null(True)
                      & (pl.col("impl_volatility_entry") > 0) & (pl.col("impl_volatility_exit") > 0)
                      & (pl.col("log_mny").abs() <= 0.05))
             .with_columns((pl.col("impl_volatility_exit") - pl.col("impl_volatility_entry")).alias("_ds"))
             .group_by(KEY).agg(
                 pl.col("_ds").median().alias("atm_iv_change"),
                 (pl.col("_ds") / pl.col("impl_volatility_entry")).median().alias("atm_iv_change_rel")))
    r0 = event_option_returns(c, cost_frac=0.0).select(KEY + [pl.col("retail_dw_buy").alias("retail_ret_mid")])
    r5 = event_option_returns(c, cost_frac=0.5).select(KEY + [pl.col("retail_dw_buy").alias("retail_ret_half")])
    out = demand.join(crush, on=KEY, how="full", coalesce=True).join(sotm, on=KEY, how="full", coalesce=True)
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
    ev = attach_knowledge_intensity(ev)
    ev = ev.join(_attention_and_volatility(ev), on=KEY, how="left")
    ev = ev.with_columns(
        pl.col("car01").abs().alias("abs_car"),
        (pl.col("car01").abs() / (pl.col("pre_sd") * np.sqrt(2))).alias("event_ratio"),
        (pl.col("retail_vol") / (pl.col("retail_vol") + pl.col("procust_vol"))).alias("retail_share"),
    )
    if contracts is not None:
        ev = ev.join(_option_measures(contracts), on=KEY, how="left")
    ev = ev.sort("PERMNO", "day0").with_columns(
        _abnormal_flow(ev, "retail_net_put").alias("retail_abn_flow"),
        _abnormal_flow(ev, "procust_net_put").alias("procust_abn_flow"))
    ev = ev.with_columns(
        (pl.col("sector") == "Technology").cast(pl.Int8).alias("tech"),
        (pl.col("sector") == "Pharma & biotech").cast(pl.Int8).alias("biotech"),
        pl.col("sector").is_in(["Energy", "Industrials"]).cast(pl.Int8).alias("energy_industrials"),
    )
    if verbose:
        n_unknown = ev.filter(pl.col("sector") == "Unknown").height
        print(f"Sector panel: {ev.height:,} firm-events; {n_unknown:,} without a NAICS code")
        if ev["stem_share"].drop_nulls().len():
            lv = ev.group_by("ki_level").len().sort("ki_level", descending=True, nulls_last=True)
            print("  knowledge intensity matched: " + ", ".join(
                f"{n:,} at {'no match' if l is None else str(l) + '-digit' if l > 2 else 'sector level'}"
                for l, n in lv.iter_rows()))
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


def test_speculative_demand(sp: pl.DataFrame) -> pl.DataFrame:
    """Tech dummy on each speculative-demand measure, with a size control;
    then the same sample with the attention proxy added. If attention drives
    tech's excess speculation, the tech coefficient shrinks in the second."""
    rows = []
    for y in [o for o in H1_OUTCOMES if o in sp.columns]:
        sub = sp.drop_nulls([y, "log_mktcap", "abn_vol"])      # same sample for both fits
        m1, n1 = _fit(sub, y, ["tech", "log_mktcap"], [y, "log_mktcap"])
        m2, n2 = _fit(sub, y, ["tech", "log_mktcap", "abn_vol"], [y, "log_mktcap", "abn_vol"])
        rows.append(_row("descriptive", "tech, size-controlled", y, "tech", m1, n1))
        rows.append(_row("descriptive", "tech, + attention", y, "tech", m2, n2))
        rows.append(_row("descriptive", "tech, + attention", y, "abn_vol", m2, n2))
    return pl.DataFrame(rows)


def test_biotech_minor_event(sp: pl.DataFrame) -> pl.DataFrame:
    """Biotech dummy on the size of the announcement event: the move relative
    to normal volatility, the absolute move, and the near-the-money IV
    change (a smaller crush = a less negative change = a positive coefficient)."""
    rows = []
    for y in [o for o in H2_OUTCOMES if o in sp.columns]:
        m, n = _fit(sp, y, ["biotech", "log_mktcap"], [y, "log_mktcap"])
        rows.append(_row("H3", "biotech, size-controlled", y, "biotech", m, n))
    return pl.DataFrame(rows)


def test_flow_interactions(sp: pl.DataFrame, flags=("energy_industrials", "biotech"),
                           dvs=("car01", "surprise_scaled"), groups=("retail", "procust")) -> pl.DataFrame:
    """Does pre-announcement flow predict the announcement differently in a
    sector?  dv ~ flow_z + flow_z:flag + flag + controls + FE, estimated
    separately for retail flow (events with retail flow) and professional
    flow (events with professional flow). The interaction is the test."""
    hyp = {"energy_industrials": "exploratory", "biotech": "exploratory"}
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


def _z(s: pl.Series) -> pl.Series:
    w = _winsorise(s.to_pandas())
    return pl.Series(((w - w.mean()) / w.std()).to_numpy())


def _ki_regression(sp, flow_col, dv, hyp, test, with_attention=True, restrict=None, group="retail"):
    """dv ~ flow_z + flow_z:ki_z + ki_z [+ flow_z:attn_z + attn_z] + controls + FE."""
    need = [flow_col, dv, "stem_share"] + CONTROLS + (["abn_vol"] if with_attention else [])
    d = sp if restrict is None else sp.filter(restrict)
    d = d.drop_nulls(need)
    if d.height < 200:
        return []
    d = d.with_columns(_z(d[flow_col]).alias("flow_z"), _z(d["stem_share"]).alias("ki_z"))
    terms = ["flow_z", "flow_z:ki_z", "ki_z"]
    if with_attention:
        d = d.with_columns(_z(d["abn_vol"]).alias("attn_z"))
        terms += ["flow_z:attn_z", "attn_z"]
    m, n = _fit(d, dv, terms + CONTROLS, [dv] + CONTROLS)
    keep = ["flow_z", "flow_z:ki_z"] + (["flow_z:attn_z"] if with_attention else [])
    return [_row(hyp, test, dv, t, m, n, group=group, flow=flow_col) for t in keep]


def test_knowledge_informativeness(sp: pl.DataFrame, dvs=("car01", "surprise_scaled")) -> pl.DataFrame:
    """H1 and H2. The primary specification, then the four H1 variants, for
    retail flow; professional flow as a comparison."""
    if sp["stem_share"].drop_nulls().len() == 0:
        return pl.DataFrame()
    med_attn = sp["abn_vol"].median()
    rows = []
    for dv in dvs:
        rows += _ki_regression(sp, "retail_net_put", dv, "H1/H2", "primary: flow x KI, flow x attention")
        rows += _ki_regression(sp, "retail_net_put", dv, "H1", "low-attention events", with_attention=False,
                               restrict=pl.col("abn_vol") <= med_attn)
        rows += _ki_regression(sp, "retail_abn_flow", dv, "H1", "abnormal flow (vs firm's past)")
        if "retail_sotm_flow" in sp.columns:
            rows += _ki_regression(sp, "retail_sotm_flow", dv, "H1", "short-dated OTM opening buys")
        rows += _ki_regression(sp, "retail_net_put", dv, "H1", "excluding pharma & biotech",
                               restrict=pl.col("sector") != "Pharma & biotech")
        rows += _ki_regression(sp, "procust_net_put", dv, "comparison", "professional flow", group="procust")
    return pl.DataFrame(rows, infer_schema_length=None)


def tail_test(sp: pl.DataFrame, flow_col: str = "retail_abn_flow", tail: float = 0.02) -> pl.DataFrame:
    """Among the events with the most extreme abnormal flow (top `tail` by
    absolute value), how often does the flow call the direction of the
    announcement return (net put buying -> negative CAR)? Compared between
    high-KI industries (top tercile of STEM share) and the rest. Chance is 50%."""
    from scipy.stats import binomtest
    from statsmodels.stats.proportion import proportions_ztest
    d = sp.drop_nulls([flow_col, "car01", "stem_share"]).filter(pl.col("car01") != 0)
    if d.height < 100:
        return pl.DataFrame()
    cut = d[flow_col].abs().quantile(1 - tail)
    t = d.filter(pl.col(flow_col).abs() >= cut).with_columns(
        (pl.col("car01").sign() == -pl.col(flow_col).sign()).alias("hit"))
    hi_cut = d["stem_share"].quantile(2 / 3)
    rows = []
    groups = {"high KI (top tercile)": t.filter(pl.col("stem_share") >= hi_cut),
              "other industries": t.filter(pl.col("stem_share") < hi_cut), "all": t}
    for label, g in groups.items():
        k, n = int(g["hit"].sum()), g.height
        rows.append({"group": label, "events": n, "hits": k, "hit_rate": k / n if n else None,
                     "p_vs_50pct": binomtest(k, n, 0.5).pvalue if n else None})
    a, b = groups["high KI (top tercile)"], groups["other industries"]
    if a.height and b.height:
        _, p = proportions_ztest([int(a["hit"].sum()), int(b["hit"].sum())], [a.height, b.height])
        rows.append({"group": "difference (high KI - other)", "events": a.height + b.height, "hits": None,
                     "hit_rate": rows[0]["hit_rate"] - rows[1]["hit_rate"], "p_vs_50pct": p})
    return pl.DataFrame(rows).with_columns(pl.lit(flow_col).alias("flow"), pl.lit(tail).alias("tail"))


def sector_profile(sp: pl.DataFrame) -> pl.DataFrame:
    """Descriptive: medians of the key measures by sector (not a test)."""
    meas = [c for c in ["stem_share", "log_mktcap", "retail_share", "retail_otm_share", "retail_short_share",
                        "retail_ret_mid", "retail_ret_half", "event_ratio", "atm_iv_change_rel", "abn_vol"]
            if c in sp.columns]
    return (sp.group_by("sector")
            .agg(pl.len().alias("events"), *[pl.col(m).median().alias(m) for m in meas])
            .sort("events", descending=True))


def run_sector_tests(sp: pl.DataFrame) -> pl.DataFrame:
    parts = [test_knowledge_informativeness(sp), test_biotech_minor_event(sp),
             test_flow_interactions(sp), test_speculative_demand(sp)]
    return pl.concat([p for p in parts if p.height], how="diagonal_relaxed")


def write_sector_results(sp: pl.DataFrame, out_dir: Path | None = None) -> dict:
    out_dir = out_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    prof, tests, check = sector_profile(sp), run_sector_tests(sp), sector_check(sp)
    tails = pl.concat([t for t in (tail_test(sp, tail=0.01), tail_test(sp, tail=0.02)) if t.height],
                      how="diagonal_relaxed") if sp["stem_share"].drop_nulls().len() else pl.DataFrame()
    prof.write_csv(out_dir / "table12_sector_profile.csv", float_precision=6)
    check.write_csv(out_dir / "table12b_sector_membership.csv")
    tests.write_csv(out_dir / "table13_sector_hypotheses.csv", float_precision=6)
    if tails.height:
        tails.write_csv(out_dir / "table13b_tail_test.csv", float_precision=6)
    print("  wrote table12_sector_profile.csv, table12b_sector_membership.csv, table13_sector_hypotheses.csv"
          + (", table13b_tail_test.csv" if tails.height else ""))
    return {"profile": prof, "tests": tests, "check": check, "tails": tails}
