"""
ingest_crsp.py
================
Ingests the CRSP Daily Stock File (CIZ format, from WRDS) into a typed,
compressed Parquet file. The raw export is ~3.7GB of CSV, too large to open
in Excel or load naively into memory, so this streams it via Polars' lazy
CSV reader and sinks straight to Parquet without materialising the whole
thing at once.

Provides the spot prices needed for moneyness (OTM vs ITM classification),
plus market capitalisation for the firm-size control and delisting metadata
for the ~25% of firm-events sitting on securities that stopped trading
before the sample ends.

Notes on the CIZ format specifically:
  - Dates arrive ISO-formatted (YYYY-MM-DD), parsed explicitly rather than
    inferred -- an earlier bug in this project came from assuming a date
    format and letting failures pass silently as nulls.
  - DlyPrcFlg indicates whether DlyPrc is a real trade ("TR") or derived
    from bid-ask ("BA"). CIZ does NOT use the old SIZ convention of
    negative prices for non-traded days, so no abs() is needed -- but the
    flag is retained so non-traded days can be filtered downstream.
  - DlyCap is in THOUSANDS of dollars, not dollars.
  - Delisting fields (DelActionType etc.) are attached to every row for a
    security, not just its final day, so a security's eventual fate is
    known without a separate join.

Usage from Jupyter:
    from pipeline.ingest_crsp import run_crsp_ingestion
    run_crsp_ingestion()

Usage from a terminal (run from the repo root):
    python src/pipeline/ingest_crsp.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import polars as pl

import sys
sys.path.append(str(Path(__file__).parent.parent))
from paths import DATA_DIR

CRSP_DIR = DATA_DIR / "crsp"
DATE_FORMAT = "%Y-%m-%d"

DATE_COLS = {"SecurityEndDt", "DlyCalDt"}
INT_COLS = {"PERMNO", "PERMCO", "DlyVol", "ShrOut", "NAICS"}
FLOAT_COLS = {"DlyPrc", "DlyCap", "DlyFacPrc", "DlyClose", "DlyBid", "DlyAsk"}


def detect_delimiter(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        sample = f.read(8192)
    return csv.Sniffer().sniff(sample, delimiters=",\t;|").delimiter


def build_schema(header: list[str]) -> dict[str, pl.DataType]:
    """Dates are read as Utf8 and parsed explicitly afterwards; everything
    unrecognised falls back to Utf8 rather than erroring, so an unexpected
    column doesn't break the ingestion."""
    schema: dict[str, pl.DataType] = {}
    for col in header:
        if col in DATE_COLS:
            schema[col] = pl.Utf8
        elif col in INT_COLS:
            schema[col] = pl.Int64
        elif col in FLOAT_COLS:
            schema[col] = pl.Float64
        else:
            schema[col] = pl.Utf8
    return schema


def run_crsp_ingestion(
    crsp_dir: Path = CRSP_DIR,
    out_path: Path = None,
) -> Path:
    out_path = out_path or (CRSP_DIR / "crsp_daily.parquet")

    csv_files = sorted(crsp_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No .csv files found in {crsp_dir.resolve()}")
    if len(csv_files) > 1:
        print(f"Found {len(csv_files)} CSV files; using {csv_files[0].name}")
    path = csv_files[0]

    size_gb = path.stat().st_size / 1e9
    delimiter = detect_delimiter(path)
    print(f"Reading {path.name} ({size_gb:.2f} GB), delimiter {delimiter!r}")

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        header = next(csv.reader(f, delimiter=delimiter))
    schema = build_schema(header)

    lf = pl.scan_csv(
        path,
        separator=delimiter,
        schema_overrides=schema,
        encoding="utf8-lossy",
        low_memory=True,
    )

    date_exprs = [
        pl.col(c).str.to_date(DATE_FORMAT, strict=False)
        for c in DATE_COLS if c in header
    ]
    if date_exprs:
        lf = lf.with_columns(date_exprs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print("Streaming to Parquet (this will take a few minutes for a file this size)...")
    lf.sink_parquet(out_path, compression="zstd")

    # validate AFTER writing, reading back lazily so nothing large is held in memory
    check = pl.scan_parquet(out_path)
    stats = check.select(
        pl.len().alias("rows"),
        pl.col("PERMNO").n_unique().alias("permnos"),
        pl.col("Ticker").n_unique().alias("tickers"),
        pl.col("DlyCalDt").min().alias("min_date"),
        pl.col("DlyCalDt").max().alias("max_date"),
        pl.col("DlyCalDt").is_null().sum().alias("null_dates"),
        pl.col("DlyPrc").is_null().sum().alias("null_prices"),
    ).collect()
    s = stats.to_dicts()[0]

    print(f"\nRows written: {s['rows']:,}")
    print(f"Unique PERMNOs: {s['permnos']:,}  |  Unique tickers: {s['tickers']:,}")
    print(f"Date range: {s['min_date']} to {s['max_date']}")
    print(f"Null dates: {s['null_dates']:,}  |  Null prices: {s['null_prices']:,}")

    # a high null-date rate means DATE_FORMAT doesn't match the file -- fail
    # loudly rather than silently writing unusable dates
    if s["rows"] and s["null_dates"] / s["rows"] > 0.01:
        raise ValueError(
            f"{s['null_dates']:,} of {s['rows']:,} DlyCalDt values failed to parse using "
            f"format {DATE_FORMAT!r}. Check the file's actual date format before using this output."
        )

    flag_counts = check.group_by("DlyPrcFlg").agg(pl.len().alias("n")).sort("n", descending=True).collect()
    print("\nDlyPrcFlg distribution (TR = real trade, BA = bid-ask derived):")
    print(flag_counts)

    print(f"\nWritten to {out_path}")
    return out_path


if __name__ == "__main__":
    run_crsp_ingestion()
