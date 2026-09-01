"""
paths.py
========
Single source of truth for every path used across this project's scripts
and notebooks. Resolves relative to this file's own location (PROJECT_ROOT
two levels up from src/), so the whole project works unmodified on any
machine or username -- nothing in any other script or notebook should ever
hardcode a path directly.

Usage from a script in src/ (e.g. extract_zips.py, ingest_cboe.py):
    from paths import PROJECT_ROOT, PARQUET_DIR

Usage from a notebook in notebooks/:
    import sys
    from pathlib import Path
    sys.path.append(str(Path.cwd().parent / "src"))
    from paths import PROJECT_ROOT, PARQUET_DIR
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# NOTE: this folder name is case-sensitive on Mac/Linux, even though Windows
# won't complain either way. Match this exactly to the real folder on disk.
ZIP_DIR = PROJECT_ROOT / "CBOE_Data_2011_2022"

DATA_DIR = PROJECT_ROOT / "data"
EXTRACTED_DIR = DATA_DIR / "CBOE_DATA_RAW_EXTRACTED"
PARQUET_DIR = DATA_DIR / "cboe_parquet"
IBES_DIR = DATA_DIR / "ibes_quarterly_report"
CRSP_DIR = DATA_DIR / "crsp"

SRC_DIR = PROJECT_ROOT / "src"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"