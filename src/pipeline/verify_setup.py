"""
verify_setup.py
================
Sanity checks that the CBOE data pipeline is intact. This script confirms the yearly Parquet files are present and complete, reports current disk usage of each data folder, and spot checks one day's Parquet output against a fresh read of its original source zip (not a cached intermediate copy).

This script should be run any time after moving folders around, cloning the repo fresh, or just to confiurm everything downstream of the raw zips is still trustworthy. 

Usage:
    python src/verify_setup.py
"""

from __future__ import annotations
 
import shutil
import zipfile
from io import BytesIO
from pathlib import Path
 
import polars as pl

import sys
sys.path.append(str(Path(__file__).parent.parent))
from paths import PROJECT_ROOT, ZIP_DIR, DATA_DIR, PARQUET_DIR, IBES_DIR
 
SPOT_CHECK_DATE = (2011, 1, 3)
KEY_COLS = ["option_symbol", "strike_price", "call_put_flag", "total_exchange_vol"]
 
 
def check_row_counts(parquet_dir: Path) -> int:
    print("=== Yearly Parquet row counts ===")
    files = sorted(parquet_dir.glob("*.parquet"))
    if not files:
        print(f"  No Parquet files found in {parquet_dir}")
        return 0
 
    total_rows = 0
    for f in files:
        n = pl.scan_parquet(f).select(pl.len()).collect().item()
        total_rows += n
        print(f"  {f.name}: {n:,} rows")
    print(f"  Total: {total_rows:,} rows across {len(files)} file(s)\n")
    return total_rows
 
 
def folder_size_gb(path: Path) -> float | None:
    if not path.exists():
        return None
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9
 
 
def check_disk_usage(locations: dict[str, Path]) -> None:
    print("=== Disk usage ===")
    for label, path in locations.items():
        size = folder_size_gb(path)
        status = "not found" if size is None else f"{size:.2f} GB"
        print(f"  {label}: {status} -- {path}")
 
    anchor = next((p for p in locations.values() if p.exists()), PROJECT_ROOT)
    free = shutil.disk_usage(anchor.anchor).free / 1e9
    print(f"  Free space: {free:.1f} GB\n")
 
 
def spot_check(zip_dir: Path, parquet_dir: Path, date: tuple[int, int, int]) -> bool:
    print("=== Spot check against a fresh source read ===")
    year, month, day = date
    sample_zip = zip_dir / f"C1OpenClose_{year:04d}-{month:02d}-{day:02d}.zip"
    if not sample_zip.exists():
        print(f"  Sample zip not found: {sample_zip}\n")
        return False
 
    with zipfile.ZipFile(sample_zip) as z:
        member = z.namelist()[0]
        fresh = pl.read_csv(BytesIO(z.read(member)), try_parse_dates=True)
 
    parquet_path = parquet_dir / f"cboe_openclose_{year}.parquet"
    if not parquet_path.exists():
        print(f"  Parquet file not found: {parquet_path}\n")
        return False
 
    from_parquet = pl.scan_parquet(parquet_path).filter(
        pl.col("quote_date") == pl.date(year, month, day)
    ).collect()
 
    print(f"  fresh CSV rows: {fresh.height}, parquet rows for that date: {from_parquet.height}")
    fresh_sorted = fresh.select(KEY_COLS).sort(KEY_COLS)
    pq_sorted = from_parquet.select(KEY_COLS).sort(KEY_COLS)
    match = fresh_sorted.equals(pq_sorted)
    print(f"  values match exactly: {match}\n")
    return match
 
 
def main() -> None:
    check_row_counts(PARQUET_DIR)
    check_disk_usage({
        "Raw zip archives": ZIP_DIR,
        "Parquet output": PARQUET_DIR,
        "IBES data": IBES_DIR,
    })
    spot_check(ZIP_DIR, PARQUET_DIR, SPOT_CHECK_DATE)
 
 
if __name__ == "__main__":
    main()
 