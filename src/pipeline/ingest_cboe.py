"""
inges_cboe.py
==============
Ingest the CBOE Open/Close volume dataset (one CSV per trading day, 2008-2022) into a small number of typed, compressed, yearly Parquet files for fast downstream analysis in Jupyer / Polas / pandas.

Why yearly Parquet files instead of one file per day>
- 3,500 daily CSVs are unweildy to hand around or point pandas/Polars at directly.
-Yearly files keep each output small enough to load indiviudally (e.g. for a single event study year) while cutting file count by >99%.
-Parquet stores an explicity schema combined with a column wise zstd compression. This results in reads downstream being both faster and smaller on disk than raw CSVs.
-This approach was inspired by Tom Roche and his initial ingestion script in his machine learning project.

Usage from Jupyter:
    from ingest_cboe import run_ingestion
    summary = run_ingestion(raw_dir="CBOE_DATA_RAW", out_dir="cboe_parquet")

Usage from a terminal:
    python ingest_cboe.py --raw-dir CBOE_DATA_RAW --out-dir cboe_parquet

Expects filenames of the form C1OpenClose_YYYY-MM-DD.csv (the year is parsed from the filename, not the file contents, so grouping works even if a file fails to parse).
"""

from __future__ import annotations
 
import argparse
import csv
import re
import time
from collections import defaultdict
from pathlib import Path

import polars as pl

import sys
sys.path.append(str(Path(__file__).parent.parent))
from paths import EXTRACTED_DIR, PARQUET_DIR
 
FILENAME_DATE_RE = re.compile(r"(\d{4})-\d{2}-\d{2}")
 
# ---------------------------------------------------------------------------
# Column typing
# ---------------------------------------------------------------------------
# CBOE Open/Close files repeat the same {open,close}_{buy,sell}_{qty,vol}
# pattern across participant types (firm_, bd_, mm_, cust_lt_100/100_199/gt_199,
# procust_lt_100/100_199/gt_199). Rather than hand-type ~80 near-identical
# columns, everything not explicitly listed below is typed as a count column
# (Int32). This also means the schema degrades gracefully if CBOE adds a new
# participant-type breakdown in a later year -- it just picks up the Int32
# default instead of erroring.
#
# NOTE for later analysis: "cust_" is a *prefix substring* of "procust_", so
# any `"cust_" in col` filter to isolate retail customer columns will also
# silently capture the procust_ (professional customer) columns. Match on
# `col.startswith("cust_")` instead.
 
DATE_COLS = {"quote_date", "expiration_date"}
FLOAT_COLS = {
    "strike_price", "first_trade_price", "high_trade_price",
    "low_trade_price", "last_trade_price", "previous_close",
}
STRING_COLS = {"underlying_symbol", "option_symbol", "call_put_flag", "series_type"}
 
 
def build_schema(header: list[str]) -> dict[str, pl.DataType]:
    """Map each column name to a dtype using the naming-convention rules above."""
    schema: dict[str, pl.DataType] = {}
    for col in header:
        if col in DATE_COLS:
            schema[col] = pl.Date
        elif col in FLOAT_COLS:
            schema[col] = pl.Float64
        elif col in STRING_COLS:
            schema[col] = pl.Utf8
        elif col == "security_type":
            schema[col] = pl.Int8
        elif col == "days_to_expire":
            schema[col] = pl.Int16
        else:
            schema[col] = pl.Int32
    return schema
 
 
def read_header(path: Path) -> list[str]:
    with open(path, newline="") as f:
        return next(csv.reader(f))
 
 
# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------
 
def discover_files(raw_dir: Path) -> dict[int, list[Path]]:
    """Group every CSV in raw_dir by year, parsed from the filename."""
    files_by_year: dict[int, list[Path]] = defaultdict(list)
    all_files = sorted(raw_dir.glob("*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir.resolve()}")
 
    for path in all_files:
        match = FILENAME_DATE_RE.search(path.name)
        if not match:
            print(f"  [skip] could not parse a date from filename: {path.name}")
            continue
        files_by_year[int(match.group(1))].append(path)
 
    return dict(sorted(files_by_year.items()))
 
 
# ---------------------------------------------------------------------------
# Per-year ingestion
# ---------------------------------------------------------------------------
 
def ingest_year(
    year: int,
    files: list[Path],
    schema: dict[str, pl.DataType],
    out_dir: Path,
) -> dict:
    """Read every file for one year and write a single yearly Parquet file.
 
    Files are read one at a time (not lazily concatenated) so a single
    corrupt/malformed file can be logged and skipped without losing the rest
    of the year's data.
    """
    frames: list[pl.DataFrame] = []
    failed: list[tuple[str, str]] = []
    t0 = time.time()
 
    for path in files:
        try:
            frames.append(pl.read_csv(path, schema_overrides=schema))
        except Exception as e:
            failed.append((path.name, str(e).splitlines()[0]))
 
    if not frames:
        return {
            "year": year, "n_files": len(files), "n_ok": 0, "n_failed": len(failed),
            "n_rows": 0, "failed": failed, "seconds": time.time() - t0, "out_path": None,
        }
 
    # diagonal_relaxed tolerates columns that appear/disappear across files
    # (e.g. a schema change partway through the dataset) by filling the
    # missing side with nulls rather than raising.
    year_df = pl.concat(frames, how="diagonal_relaxed")
 
    out_path = out_dir / f"cboe_openclose_{year}.parquet"
    year_df.write_parquet(out_path, compression="zstd")
 
    return {
        "year": year, "n_files": len(files), "n_ok": len(frames), "n_failed": len(failed),
        "n_rows": year_df.height, "failed": failed, "seconds": time.time() - t0,
        "out_path": out_path,
    }
 
 
# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
 
def run_ingestion(raw_dir: str = str(EXTRACTED_DIR), out_dir: str = str(PARQUET_DIR)) -> pl.DataFrame:
    raw_path = Path(raw_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
 
    print(f"Scanning {raw_path.resolve()} ...")
    files_by_year = discover_files(raw_path)
    total_files = sum(len(v) for v in files_by_year.values())
    print(
        f"Found {total_files} files spanning {len(files_by_year)} year(s): "
        f"{min(files_by_year)}-{max(files_by_year)}\n"
    )
 
    # Canonical schema is built from the first file encountered. Later files
    # with extra/missing columns are still handled gracefully at concat time
    # (see diagonal_relaxed above) -- this just sets the *dtype* expectation.
    first_file = next(iter(files_by_year.values()))[0]
    schema = build_schema(read_header(first_file))
 
    results = []
    for year, files in files_by_year.items():
        print(f"[{year}] ingesting {len(files)} files ...", end=" ", flush=True)
        result = ingest_year(year, files, schema, out_path)
        results.append(result)
        print(
            f"{result['n_rows']:,} rows | {result['n_ok']}/{result['n_files']} files ok "
            f"| {result['seconds']:.1f}s"
        )
        if result["failed"]:
            print(f"    WARNING: {len(result['failed'])} file(s) failed to parse:")
            for name, err in result["failed"][:5]:
                print(f"      - {name}: {err}")
 
    summary = pl.DataFrame(
        [
            {
                "year": r["year"],
                "files_found": r["n_files"],
                "files_ok": r["n_ok"],
                "files_failed": r["n_failed"],
                "rows": r["n_rows"],
                "seconds": round(r["seconds"], 1),
                "output": str(r["out_path"]) if r["out_path"] else None,
            }
            for r in results
        ]
    )
 
    print("\n=== Ingestion summary ===")
    with pl.Config(tbl_rows=-1):
        print(summary)
 
    total_failed = int(summary["files_failed"].sum())
    if total_failed:
        print(f"\n{total_failed} file(s) failed across all years -- see warnings above.")
 
    return summary
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest CBOE Open/Close daily CSVs into yearly Parquet files."
    )
    parser.add_argument("--raw-dir", default=str(EXTRACTED_DIR), help="Folder of daily CSVs")
    parser.add_argument("--out-dir", default=str(PARQUET_DIR), help="Folder to write yearly Parquet files")
    args = parser.parse_args()
    run_ingestion(args.raw_dir, args.out_dir)