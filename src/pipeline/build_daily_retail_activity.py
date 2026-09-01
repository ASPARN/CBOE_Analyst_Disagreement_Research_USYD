"""
build_daily_retail_activity.py
================================
Collapses the CBOE option-contract-level data (one row per strike, per
expiration, per day) down to one row per (underlying_symbol, quote_date),
summing trading volume for BOTH retail customers (cust_) and professional
customers (procust_) -- the two most directly comparable CBOE participant
categories, since they share the identical "customer" market role and
differ specifically in sophistication/capital, not market function. This
is the granularity almost every downstream question actually needs --
"how much activity was there in ticker X on date Y, by participant type" --
rather than the much larger option-level detail.

Volume uses the *_vol columns (total contracts traded), not *_qty (number
of separate transactions) -- confirmed by direct inspection that these are
genuinely different measures, not duplicates, with _vol always the larger
of the two.

Produces, per (ticker, day), one set of columns per participant group
(retail_vol_* for cust_, procust_vol_* for procust_):
  {group}_vol_total    -- all volume (open + close, calls + puts)
  {group}_vol_open     -- volume opening new positions
  {group}_vol_close    -- volume closing existing positions
  {group}_vol_call     -- volume in call contracts
  {group}_vol_put      -- volume in put contracts
  {group}_vol_lt_100   -- volume from trades sized under 100 contracts
  {group}_vol_100_199  -- volume from trades sized 100-199 contracts
  {group}_vol_gt_199   -- volume from trades sized over 199 contracts

The retail_vol_* names are kept exactly as before (not renamed to
cust_vol_*) so every existing script that already reads them keeps working
unchanged; procust_vol_* is purely additive.

Usage from Jupyter:
    from build_daily_retail_activity import build_daily_retail_activity
    build_daily_retail_activity()

Usage from a terminal (run from the repo root):
    python src/build_daily_retail_activity.py
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

import sys
sys.path.append(str(Path(__file__).parent.parent))
from paths import PARQUET_DIR, DATA_DIR

# maps the output column prefix to the CBOE column prefix it's built from.
# "retail" keeps its established name for backward compatibility; "procust"
# is the new comparison group.
PARTICIPANT_GROUPS = {
    "retail": "cust_",
    "procust": "procust_",
}


def _tier_columns(cols: list[str], prefix: str) -> dict[str, list[str]]:
    """Given the full CBOE column list and a participant prefix (e.g.
    'cust_' or 'procust_'), group that participant's *_vol columns by
    open/close and by contract-size tier. .startswith(prefix) is used
    deliberately rather than substring matching -- 'cust_' is itself a
    substring of 'procust_', so a naive `in` check would incorrectly pull
    procust_ columns into the cust_ group."""
    vol_cols = [c for c in cols if c.startswith(prefix) and c.endswith("_vol")]
    return {
        "total": vol_cols,
        "open": [c for c in vol_cols if "_open_" in c],
        "close": [c for c in vol_cols if "_close_" in c],
        "lt_100": [c for c in vol_cols if c.startswith(f"{prefix}lt_100")],
        "100_199": [c for c in vol_cols if c.startswith(f"{prefix}100_199")],
        "gt_199": [c for c in vol_cols if c.startswith(f"{prefix}gt_199")],
    }


def build_daily_retail_activity(
    cboe_dir: Path = PARQUET_DIR,
    out_dir: Path = None,
) -> pl.DataFrame:
    out_dir = out_dir or (DATA_DIR / "cboe_daily_retail")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(cboe_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No CBOE parquet files found in {cboe_dir.resolve()}")

    all_daily = []
    for f in files:
        lf = pl.scan_parquet(f)
        cols = lf.collect_schema().names()

        row_exprs = []
        agg_exprs = []
        for group_name, cboe_prefix in PARTICIPANT_GROUPS.items():
            tiers = _tier_columns(cols, cboe_prefix)
            row_exprs += [
                pl.sum_horizontal(tiers["total"]).cast(pl.Int64).alias(f"_row_{group_name}_total"),
                pl.sum_horizontal(tiers["open"]).cast(pl.Int64).alias(f"_row_{group_name}_open"),
                pl.sum_horizontal(tiers["close"]).cast(pl.Int64).alias(f"_row_{group_name}_close"),
                pl.sum_horizontal(tiers["lt_100"]).cast(pl.Int64).alias(f"_row_{group_name}_lt_100"),
                pl.sum_horizontal(tiers["100_199"]).cast(pl.Int64).alias(f"_row_{group_name}_100_199"),
                pl.sum_horizontal(tiers["gt_199"]).cast(pl.Int64).alias(f"_row_{group_name}_gt_199"),
            ]
            agg_exprs += [
                pl.col(f"_row_{group_name}_total").sum().alias(f"{group_name}_vol_total"),
                pl.col(f"_row_{group_name}_open").sum().alias(f"{group_name}_vol_open"),
                pl.col(f"_row_{group_name}_close").sum().alias(f"{group_name}_vol_close"),
                pl.col(f"_row_{group_name}_total").filter(pl.col("call_put_flag") == "C").sum().alias(f"{group_name}_vol_call"),
                pl.col(f"_row_{group_name}_total").filter(pl.col("call_put_flag") == "P").sum().alias(f"{group_name}_vol_put"),
                pl.col(f"_row_{group_name}_lt_100").sum().alias(f"{group_name}_vol_lt_100"),
                pl.col(f"_row_{group_name}_100_199").sum().alias(f"{group_name}_vol_100_199"),
                pl.col(f"_row_{group_name}_gt_199").sum().alias(f"{group_name}_vol_gt_199"),
            ]

        daily = (
            lf.with_columns(row_exprs)
            .group_by(["underlying_symbol", "quote_date"])
            .agg(agg_exprs)
            .collect()
        )

        out_path = out_dir / f"daily_retail_{f.stem.split('_')[-1]}.parquet"
        daily.write_parquet(out_path, compression="zstd")
        print(f"{f.name}: -> {daily.height:,} ticker-day rows -> {out_path.name}")
        all_daily.append(daily)

    combined = pl.concat(all_daily)
    print(f"\n{combined.height:,} total ticker-day rows across {len(files)} year(s)")
    print(f"Unique tickers: {combined['underlying_symbol'].n_unique():,}")
    return combined


if __name__ == "__main__":
    build_daily_retail_activity()