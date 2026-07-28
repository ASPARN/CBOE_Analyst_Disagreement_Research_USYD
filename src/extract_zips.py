"""
extract_zips.py
================
This script extracts the CBOE per-day zip archive (CBOE)Data_2011_2022/) into flat CSV files at data/CBOE/RAW_EXTRACTED/, ready for ingest_cboe.py. Each zip is epexted to hold exactly one CSV. 

Important to note, Excel is also handled as a fallback. Earlier in the project I had some confusion about what format I wanted the data to be stored in, so that branch is defensive rather than load-bearing and is a artifact of a previous design I deicded to keep for the final script.

Usage from Jupyter:
    from extract_zips import run_extraction
    run_extraction()

Usage from a terminal (run from the repo root):
    python src/extract_zips.py
"""

from __future__ import annotations
 
import time
import zipfile
from io import BytesIO
from pathlib import Path
 
import polars as pl
 
from paths import ZIP_DIR, EXTRACTED_DIR
 
EXCEL_EXT = (".xlsx", ".xls", ".xlsm")
 
 
def run_extraction(zip_dir: Path = ZIP_DIR, dest_dir: Path = EXTRACTED_DIR) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
 
    zip_files = sorted(zip_dir.glob("*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"No zip files found in {zip_dir.resolve()}")
 
    print(f"Found {len(zip_files)} per-day zip files to convert\n")
 
    t0 = time.time()
    converted = 0
    skipped: list[tuple[str, str]] = []
 
    for i, zpath in enumerate(zip_files, 1):
        try:
            with zipfile.ZipFile(zpath, "r") as z:
                names = z.namelist()
                data_member, is_excel = None, False
                for n in names:
                    low = n.lower()
                    if low.endswith(".csv"):
                        data_member, is_excel = n, False
                        break
                    if low.endswith(EXCEL_EXT):
                        data_member, is_excel = n, True
                        break
                if data_member is None:
                    skipped.append((zpath.name, f"no CSV/Excel inside (found: {names})"))
                    continue
 
                raw_bytes = z.read(data_member)
                out_path = dest_dir / f"{zpath.stem}.csv"
 
                if is_excel:
                    df = pl.read_excel(BytesIO(raw_bytes))
                    df.write_csv(out_path)
                else:
                    out_path.write_bytes(raw_bytes)
            converted += 1
        except Exception as e:
            skipped.append((zpath.name, str(e)[:120]))
 
        if i % 100 == 0 or i == len(zip_files):
            elapsed = time.time() - t0
            remaining = (len(zip_files) - i) / (i / elapsed) / 60 if i else 0
            print(f"  {i}/{len(zip_files)} done ({elapsed/60:.1f} min elapsed, ~{remaining:.0f} min remaining)")
 
    print(f"\nDone. {converted} files converted to CSV in {dest_dir}")
    if skipped:
        print(f"{len(skipped)} file(s) had issues:")
        for name, reason in skipped[:10]:
            print(f"  - {name}: {reason}")
 
    return dest_dir
 
 
if __name__ == "__main__":
    run_extraction()