"""
build_daily_flow.py
=====================
Daily SIGNED option flow for retail and professional customers, for the
informed-trading test (Test 1: does pre-announcement put buying predict the
announcement return?).

Why a separate table rather than new columns on cboe_daily_retail:
the daily retail table underpins every verified result in the thesis and
verify_results is calibrated against it. This table adds the one thing that
table collapsed away -- the buy/sell side of each trade -- without touching
anything already signed off.

Why the side matters: selling a put is a bullish trade and buying one is
bearish, so a put share that mixes buyers and writers has no clean
directional sign. Pan & Poteshman (2006) use opening BUY volume for exactly
this reason.

Output, one row per (underlying_symbol, quote_date), 16 flow columns:

    {group}_{cp}_{action}
      group  : retail (CBOE cust_), procust (CBOE procust_)
      cp     : call, put
      action : open_buy, close_buy, open_sell, close_sell

Each is contract volume summed across the three size tiers (lt_100,
100_199, gt_199). Every flow measure the analysis might want is derivable:
  net put buying      = put_open_buy + put_close_buy - put_open_sell - put_close_sell
  Pan-Poteshman ratio = put_open_buy / (put_open_buy + call_open_buy)
  net bearish flow    = net put buying - net call buying

Volumes are summed as Int64 (an Int32 sum overflowed in an earlier version
of the daily retail build).

Usage (base and extended samples, same function):
    build_daily_flow()                                         # base
    build_daily_flow(cboe_dir=DATA_DIR / "cboe_parquet_ext",
                     out_dir=DATA_DIR / "cboe_daily_flow_ext")  # extension
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import polars as pl

sys.path.append(str(Path(__file__).parent.parent))
from paths import DATA_DIR, PARQUET_DIR

GROUPS = {"cust": "retail", "procust": "procust"}
TIERS = ("lt_100", "100_199", "gt_199")
ACTIONS = ("open_buy", "close_buy", "open_sell", "close_sell")
CP = {"C": "call", "P": "put"}
KEYS = ["underlying_symbol", "quote_date"]


def flow_columns() -> list[str]:
    return [f"{g}_{cp}_{a}" for g in GROUPS.values() for cp in CP.values() for a in ACTIONS]


def _flow_exprs() -> list[pl.Expr]:
    exprs = []
    for raw_g, g in GROUPS.items():
        for raw_cp, cp in CP.items():
            is_cp = pl.col("_cp") == raw_cp
            for a in ACTIONS:
                tier_sum = pl.sum_horizontal(
                    pl.col(f"{raw_g}_{t}_{a}_vol").cast(pl.Int64).fill_null(0) for t in TIERS
                )
                exprs.append(pl.when(is_cp).then(tier_sum).otherwise(0).sum().alias(f"{g}_{cp}_{a}"))
    return exprs


def build_daily_flow(cboe_dir: Path = PARQUET_DIR, out_dir: Path = None) -> pl.DataFrame:
    out_dir = out_dir or (DATA_DIR / "cboe_daily_flow")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(cboe_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No CBOE parquet files found in {cboe_dir.resolve()}")

    needed = KEYS + ["call_put_flag"] + [
        f"{rg}_{t}_{a}_vol" for rg in GROUPS for t in TIERS for a in ACTIONS
    ]

    summaries = []
    for f in files:
        t0 = time.time()
        lf = pl.scan_parquet(f)
        have = set(lf.collect_schema().names())
        missing = [c for c in needed if c not in have]
        if missing:
            raise KeyError(f"{f.name} is missing columns needed for signed flow: {missing}")

        lf = lf.select(needed).with_columns(
            pl.col("call_put_flag").str.strip_chars().str.to_uppercase().alias("_cp")
        )
        n_other = lf.filter(~pl.col("_cp").is_in(list(CP))).select(pl.len()).collect().item()

        daily = lf.group_by(KEYS).agg(_flow_exprs()).sort(KEYS).collect()

        out_path = out_dir / f"daily_flow_{f.stem.split('_')[-1]}.parquet"
        daily.write_parquet(out_path, compression="zstd")
        note = f" | {n_other:,} rows with an unrecognised call/put flag excluded" if n_other else ""
        print(f"{f.name}: -> {daily.height:,} ticker-days -> {out_path.name} "
              f"({time.time() - t0:.1f}s){note}")
        summaries.append(daily)

    combined = pl.concat(summaries)
    print(f"\n{combined.height:,} ticker-days across {len(files)} file(s); "
          f"{combined['underlying_symbol'].n_unique():,} tickers")
    return combined


if __name__ == "__main__":
    build_daily_flow()
