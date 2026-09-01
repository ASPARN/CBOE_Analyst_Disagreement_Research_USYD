"""
ingest_ibes.py
================
Discovers whatever IBES Summary History export(s) have been placed in
data/ibes_quarterly_report/, and converts them into a single clean, typed
Parquet file. No filename is hardcoded -- any .csv file dropped into that
folder gets picked up automatically, matching the same "drop files in,
run the script" pattern as extract_zips.py.

This script does NOT yet extract firm-event dispersion values or filter to
a specific fiscal period type (QTR vs ANN) -- that's a separate, later
analytical step, once this raw data is clean and trustworthy. This script's
only job is: raw export in, typed Parquet out, with the data quality
problems that are common in commercial data exports actively checked for
rather than assumed away.

Usage from Jupyter:
    from ingest_ibes import run_ibes_ingestion
    run_ibes_ingestion()

Usage from a terminal (run from the repo root):
    python src/ingest_ibes.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import polars as pl

import sys
sys.path.append(str(Path(__file__).parent.parent))
from paths import IBES_DIR

EXCEL_ROW_CAP = 1_048_576  # Excel's hard row limit (2^20) -- a file landing
                            # exactly here is a strong signal it passed
                            # through Excel and may be padded or truncated

DATE_FORMAT = "%Y-%m-%d"  # ISO format, confirmed against Axel's actual export --
                           # unambiguous by construction, unlike M/D/Y

DATE_COLS = {"STATPERS", "FPEDATS", "ANNDATS_ACT"}
INT_COLS = {"FPI", "NUMEST", "NUMUP", "NUMDOWN", "USFIRM"}
FLOAT_COLS = {"MEDEST", "MEANEST", "STDEV", "HIGHEST", "LOWEST", "ACTUAL"}
STRING_COLS = {
    "CNAME", "MEASURE", "FISCALP", "ESTFLAG", "CURCODE", "ANNTIMS_ACT", "CURR_ACT",
    "TICKER", "CUSIP", "OFTIC",  # OFTIC is the actual ticker symbol -- the join
                                   # key against CBOE's underlying_symbol column
}


def detect_delimiter(path: Path) -> str:
    """Sniff the actual delimiter rather than assume comma just because the
    file is named .csv -- IBES/WRDS exports are inconsistent about this."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        sample = f.read(8192)
    dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
    return dialect.delimiter


def build_schema(header: list[str]) -> dict[str, pl.DataType]:
    schema: dict[str, pl.DataType] = {}
    for col in header:
        if col in DATE_COLS:
            schema[col] = pl.Utf8  # read as string first, parse explicitly after
        elif col in INT_COLS:
            schema[col] = pl.Int32
        elif col in FLOAT_COLS:
            schema[col] = pl.Float64
        elif col in STRING_COLS:
            schema[col] = pl.Utf8
        else:
            schema[col] = pl.Utf8  # unknown columns pass through as text
    return schema


def load_one_file(path: Path) -> pl.DataFrame:
    delimiter = detect_delimiter(path)
    print(f"  {path.name}: detected delimiter {delimiter!r}")

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        header = next(csv.reader(f, delimiter=delimiter))
    schema = build_schema(header)

    df = pl.read_csv(path, separator=delimiter, schema_overrides=schema, encoding="utf8-lossy")

    # parse dates explicitly -- never rely on auto-detection for these
    date_exprs = [
        pl.col(c).str.to_date(DATE_FORMAT, strict=False)
        for c in DATE_COLS if c in df.columns
    ]
    if date_exprs:
        df = df.with_columns(date_exprs)

    # STATPERS should never be legitimately blank (unlike ANNDATS_ACT, which
    # is genuinely blank for not-yet-announced periods) -- a high null rate
    # here means DATE_FORMAT doesn't match this file, not that the data is
    # actually missing. Fail loudly rather than silently write garbage dates.
    if "STATPERS" in df.columns:
        null_frac = df["STATPERS"].is_null().mean()
        if null_frac > 0.01:
            raise ValueError(
                f"{path.name}: {null_frac:.1%} of STATPERS values failed to parse as "
                f"dates using format {DATE_FORMAT!r}. This almost always means the file's "
                f"actual date format doesn't match DATE_FORMAT. Check a few raw rows "
                f"(e.g. via csv.reader, bypassing this script entirely) before re-running."
            )

    return df


def check_data_quality(df: pl.DataFrame, path: Path) -> None:
    raw_rows = df.height
    print(f"\n  Raw rows: {raw_rows:,}")

    if EXCEL_ROW_CAP - 10 <= raw_rows <= EXCEL_ROW_CAP:
        print(
            f"  WARNING: row count is at or near Excel's {EXCEL_ROW_CAP:,}-row limit.\n"
            f"    This file likely passed through Excel at some point. If the true\n"
            f"    dataset is larger than this, rows may have been silently truncated.\n"
            f"    Worth confirming the expected row count with whoever provided {path.name}."
        )

    # rows where every column is null are near-certainly Excel padding, not real data
    all_null_mask = pl.all_horizontal([pl.col(c).is_null() for c in df.columns])
    blank_rows = df.filter(all_null_mask).height
    if blank_rows:
        print(f"  {blank_rows:,} fully-blank row(s) found -- likely Excel padding, will be dropped")

    real_rows = raw_rows - blank_rows
    print(f"  Real (non-blank) rows: {real_rows:,}")


def run_ibes_ingestion(ibes_dir: Path = IBES_DIR, out_path: Path = None) -> pl.DataFrame:
    out_path = out_path or (ibes_dir / "ibes_clean.parquet")

    csv_files = sorted(ibes_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No .csv files found in {ibes_dir.resolve()}")

    print(f"Found {len(csv_files)} file(s) in {ibes_dir}\n")

    frames = []
    for path in csv_files:
        df = load_one_file(path)
        check_data_quality(df, path)

        all_null_mask = pl.all_horizontal([pl.col(c).is_null() for c in df.columns])
        df = df.filter(~all_null_mask)
        frames.append(df)

    combined = pl.concat(frames, how="diagonal_relaxed") if len(frames) > 1 else frames[0]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(out_path, compression="zstd")

    print(f"\nDone. {combined.height:,} clean rows written to {out_path}")
    return combined


if __name__ == "__main__":
    run_ibes_ingestion()