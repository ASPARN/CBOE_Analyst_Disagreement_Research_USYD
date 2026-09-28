"""
build_market_return.py
========================
Constructs a daily value-weighted market return from CRSP stock returns and
lagged market capitalisation, for the market-adjusted announcement returns
in Test 1.

Why construct it rather than use CRSP's own vwretd:
CRSP's legacy index table (crsp.dsi) stopped updating at 2024-12-31 when the
legacy format was retired, but the post-COVID era runs through 2025. Rather
than splice a different series in for 2025 -- a methodology change in the
middle of an era -- this builds one consistent series for every year and
validates it against vwretd wherever vwretd exists.

Method:
    r_mkt,t = sum_i( cap_i,t-1 * r_i,t ) / sum_i( cap_i,t-1 )
using each security's previous trading day's DlyCap as its weight, over
securities with a return on day t and a positive lagged cap.

Two versions are built:
    mkt_vw_all     every security with a return and a lagged cap
    mkt_vw_common  common stocks only (SecurityType EQTY, ShareType NS)
validate_market_return() compares both against vwretd, so the choice
between them is made on evidence rather than assumed.

Inputs:  data/crsp/crsp_daily.parquet, crsp_daily_ext.parquet  (caps)
         data/crsp/returns/crsp_returns_<year>.parquet           (returns)
         data/crsp/returns/crsp_market_daily.parquet             (vwretd)
Output:  data/crsp/returns/market_vw_constructed.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.append(str(Path(__file__).parent.parent))
from paths import CRSP_DIR

RET_DIR = CRSP_DIR / "returns"
OUT_PATH = RET_DIR / "market_vw_constructed.parquet"


def _load_caps() -> pl.LazyFrame:
    frames = []
    for name in ("crsp_daily.parquet", "crsp_daily_ext.parquet"):
        f = CRSP_DIR / name
        if not f.exists():
            continue
        lf = pl.scan_parquet(f)
        have = set(lf.collect_schema().names())
        cols = ["PERMNO", "DlyCalDt", "DlyCap"] + [c for c in ("SecurityType", "ShareType") if c in have]
        frames.append(lf.select(cols))
    if not frames:
        raise FileNotFoundError("No CRSP price parquet found for market caps.")
    return (pl.concat(frames, how="diagonal_relaxed")
            .with_columns(pl.col("PERMNO").cast(pl.Int64), pl.col("DlyCalDt").cast(pl.Date))
            # base and ext overlap Jan-Jul 2022; keep one row per security-day
            .unique(subset=["PERMNO", "DlyCalDt"], keep="last"))


def build_market_return(out_path: Path = OUT_PATH) -> pl.DataFrame:
    caps = _load_caps()
    have = set(caps.collect_schema().names())

    rets = (pl.scan_parquet(RET_DIR / "crsp_returns_*.parquet")
            .select(pl.col("PERMNO").cast(pl.Int64), pl.col("DlyCalDt").cast(pl.Date), "DlyRet"))

    panel = (caps.sort("PERMNO", "DlyCalDt")
             .with_columns(pl.col("DlyCap").shift(1).over("PERMNO").alias("cap_lag"))
             .join(rets, on=["PERMNO", "DlyCalDt"], how="inner")
             .filter(pl.col("DlyRet").is_not_null() & (pl.col("cap_lag") > 0)))

    if {"SecurityType", "ShareType"} <= have:
        panel = panel.with_columns(
            ((pl.col("SecurityType") == "EQTY") & (pl.col("ShareType") == "NS")).alias("_common"))
    else:
        panel = panel.with_columns(pl.lit(None, dtype=pl.Boolean).alias("_common"))

    def _vw(mask: pl.Expr, name: str) -> list[pl.Expr]:
        w = pl.when(mask).then(pl.col("cap_lag")).otherwise(None)
        return [
            ((w * pl.col("DlyRet")).sum() / w.sum()).alias(name),
            w.count().alias(f"n_{name.removeprefix('mkt_vw_')}"),
        ]

    mkt = (panel.group_by("DlyCalDt")
           .agg(_vw(pl.lit(True), "mkt_vw_all") + _vw(pl.col("_common").fill_null(False), "mkt_vw_common"))
           .sort("DlyCalDt")
           .collect())

    # A day where the common-stock filter matched nothing yields NaN; make it null.
    mkt = mkt.with_columns(pl.col("mkt_vw_common").fill_nan(None))
    mkt.write_parquet(out_path, compression="zstd")
    print(f"Constructed market return: {mkt.height:,} days, "
          f"{mkt['DlyCalDt'].min()} to {mkt['DlyCalDt'].max()}")
    print(f"  median securities per day: all {mkt['n_all'].median():,.0f}, "
          f"common {mkt['n_common'].median():,.0f}")
    if "SecurityType" in have:
        types = (caps.select("SecurityType", "ShareType").collect()
                 .group_by("SecurityType", "ShareType").len().sort("len", descending=True).head(6))
        print("  most common SecurityType/ShareType pairs in CRSP (sanity check on the filter):")
        print(types)
    return mkt


def validate_market_return(constructed: pl.DataFrame = None) -> pl.DataFrame:
    """Compare both constructed series against CRSP vwretd, by year."""
    constructed = constructed if constructed is not None else pl.read_parquet(OUT_PATH)
    ref = pl.read_parquet(RET_DIR / "crsp_market_daily.parquet").select("DlyCalDt", "vwretd")
    both = constructed.join(ref, on="DlyCalDt", how="inner").drop_nulls(["vwretd"])

    rows = []
    for col in ("mkt_vw_all", "mkt_vw_common"):
        d = both.drop_nulls([col])
        if d.height == 0:
            continue
        yearly = (d.with_columns(pl.col("DlyCalDt").dt.year().alias("year"))
                  .group_by("year")
                  .agg(pl.corr(col, "vwretd").alias("corr"),
                       ((pl.col(col) - pl.col("vwretd")).abs().mean() * 1e4).alias("mean_abs_diff_bp"))
                  .with_columns(pl.lit(col).alias("series")))
        rows.append(yearly)
        overall = d.select(pl.corr(col, "vwretd")).item()
        mad = d.select((pl.col(col) - pl.col("vwretd")).abs().mean()).item() * 1e4
        print(f"{col:15s} vs vwretd: corr {overall:.5f}, mean |diff| {mad:.2f} bp/day "
              f"over {d.height:,} days")
    return pl.concat(rows).sort("series", "year")


if __name__ == "__main__":
    validate_market_return(build_market_return())
