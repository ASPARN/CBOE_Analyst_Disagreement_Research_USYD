"""
build_moneyness.py
====================
Joins CRSP daily spot prices to the CBOE option-contract-level data and
classifies every traded contract as out-of-the-money (OTM), in-the-money
(ITM), or at-the-money (ATM), then aggregates retail and professional-
customer volume by that classification down to one row per
(underlying_symbol, quote_date).

This produces the raw material for DV2 in the research design -- the
OTM-to-ITM ratio of contracts traded by retail customers -- which is the
measure most directly tied to the lottery-preference framing: a
far-out-of-the-money call is the lottery ticket.

Moneyness must be computed at the CONTRACT level (each strike against that
day's spot), not from the pre-aggregated daily table, so this reads the
full option-level CBOE parquet files. It processes them one year at a time
to keep memory bounded.

Classification uses log moneyness, m = ln(strike / spot):
  call:  m >  atm_band -> OTM      m < -atm_band -> ITM
  put:   m < -atm_band -> OTM      m >  atm_band -> ITM
  either: |m| <= atm_band -> ATM
Log rather than raw ratio so that "10% out of the money" means the same
thing for a $10 stock and a $1,000 stock.

Notes:
  - Joins on (ticker, date). CRSP's Ticker is as-of-that-date, so a company
    that changed ticker mid-sample matches correctly on each side of the
    change without any special handling.
  - Applies the same '.' vs '/' share-class resolution used elsewhere in
    the project (IBES/CBOE use '.', CRSP may use either).
  - Contract-days with no CRSP price cannot be classified and are counted
    and reported rather than silently dropped.
  - DlyPrcFlg is retained per ticker-day so downstream work can exclude
    days where the spot is a bid-ask midpoint rather than a real trade.

Usage from Jupyter:
    from pipeline.build_moneyness import build_moneyness
    build_moneyness()

Usage from a terminal (run from the repo root):
    python src/pipeline/build_moneyness.py
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

import sys
sys.path.append(str(Path(__file__).parent.parent))
from paths import PARQUET_DIR, CRSP_DIR, DATA_DIR

ATM_BAND = 0.02  # |ln(strike/spot)| within this counts as at-the-money


def _load_crsp_prices() -> pl.DataFrame:
    path = CRSP_DIR / "crsp_daily.parquet"
    if not path.exists():
        raise FileNotFoundError(f"CRSP parquet not found at {path}. Run ingest_crsp.py first.")
    return (
        pl.scan_parquet(path)
        .select(["Ticker", "DlyCalDt", "DlyPrc", "DlyPrcFlg"])
        .filter(pl.col("Ticker").is_not_null() & pl.col("DlyPrc").is_not_null() & (pl.col("DlyPrc") > 0))
        .unique(subset=["Ticker", "DlyCalDt"])  # guard against a ticker appearing twice on one date
        .collect()
    )


def _resolve_symbols(symbols: list[str], crsp_tickers: set) -> pl.DataFrame:
    """CBOE/IBES write share classes as 'BRK.B'; CRSP may use either that or
    'BRK/B'. Try as-given, then the swapped separator."""
    resolved = []
    for s in symbols:
        if s in crsp_tickers:
            resolved.append(s)
        elif "." in s and s.replace(".", "/") in crsp_tickers:
            resolved.append(s.replace(".", "/"))
        elif "/" in s and s.replace("/", ".") in crsp_tickers:
            resolved.append(s.replace("/", "."))
        else:
            resolved.append(s)
    return pl.DataFrame({"underlying_symbol": symbols, "crsp_ticker": resolved})


def build_moneyness(
    cboe_dir: Path = PARQUET_DIR,
    out_dir: Path = None,
    atm_band: float = ATM_BAND,
) -> pl.DataFrame:
    out_dir = out_dir or (DATA_DIR / "cboe_daily_moneyness")
    out_dir.mkdir(parents=True, exist_ok=True)

    crsp = _load_crsp_prices()
    crsp_tickers = set(crsp["Ticker"].unique().to_list())
    print(f"CRSP price rows: {crsp.height:,}  |  unique tickers: {len(crsp_tickers):,}\n")

    files = sorted(cboe_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No CBOE parquet files found in {cboe_dir.resolve()}")

    all_daily = []
    total_rows = total_matched = 0

    for f in files:
        lf = pl.scan_parquet(f)
        cols = lf.collect_schema().names()

        symbols = lf.select("underlying_symbol").unique().collect()["underlying_symbol"].to_list()
        symbol_map = _resolve_symbols(symbols, crsp_tickers)

        joined = (
            lf.collect()
            .join(symbol_map, on="underlying_symbol", how="left")
            .join(
                crsp,
                left_on=["crsp_ticker", "quote_date"],
                right_on=["Ticker", "DlyCalDt"],
                how="left",
            )
        )

        n_rows = joined.height
        n_matched = joined.filter(pl.col("DlyPrc").is_not_null()).height
        total_rows += n_rows
        total_matched += n_matched

        # log moneyness, then classify by contract type
        joined = joined.with_columns(
            (pl.col("strike_price") / pl.col("DlyPrc")).log().alias("log_mny")
        ).with_columns(
            pl.when(pl.col("log_mny").is_null()).then(pl.lit("UNKNOWN"))
            .when(pl.col("log_mny").abs() <= atm_band).then(pl.lit("ATM"))
            .when((pl.col("call_put_flag") == "C") & (pl.col("log_mny") > atm_band)).then(pl.lit("OTM"))
            .when((pl.col("call_put_flag") == "C") & (pl.col("log_mny") < -atm_band)).then(pl.lit("ITM"))
            .when((pl.col("call_put_flag") == "P") & (pl.col("log_mny") < -atm_band)).then(pl.lit("OTM"))
            .when((pl.col("call_put_flag") == "P") & (pl.col("log_mny") > atm_band)).then(pl.lit("ITM"))
            .otherwise(pl.lit("UNKNOWN"))
            .alias("moneyness")
        )

        row_exprs, agg_exprs = [], []
        for group, prefix in [("retail", "cust_"), ("procust", "procust_")]:
            vol_cols = [c for c in cols if c.startswith(prefix) and c.endswith("_vol")]
            row_exprs.append(pl.sum_horizontal(vol_cols).cast(pl.Int64).alias(f"_row_{group}"))
            for mny in ["OTM", "ITM", "ATM", "UNKNOWN"]:
                for cp, cp_label in [("C", "call"), ("P", "put")]:
                    agg_exprs.append(
                        pl.col(f"_row_{group}")
                        .filter((pl.col("moneyness") == mny) & (pl.col("call_put_flag") == cp))
                        .sum()
                        .alias(f"{group}_vol_{mny.lower()}_{cp_label}")
                    )

        daily = (
            joined.with_columns(row_exprs)
            .group_by(["underlying_symbol", "quote_date"])
            .agg(agg_exprs + [pl.col("DlyPrcFlg").first().alias("spot_price_flag")])
            .fill_null(0)
        )

        out_path = out_dir / f"daily_moneyness_{f.stem.split('_')[-1]}.parquet"
        daily.write_parquet(out_path, compression="zstd")
        print(
            f"{f.name}: {n_rows:,} option-rows, {n_matched / n_rows:.1%} priced "
            f"-> {daily.height:,} ticker-days -> {out_path.name}"
        )
        all_daily.append(daily)

    combined = pl.concat(all_daily)
    print(f"\nTotal option-rows processed: {total_rows:,}")
    print(f"Rows with a matched CRSP spot price: {total_matched:,} ({total_matched / total_rows:.1%})")
    print(f"Ticker-days written: {combined.height:,}")
    return combined


if __name__ == "__main__":
    build_moneyness()
