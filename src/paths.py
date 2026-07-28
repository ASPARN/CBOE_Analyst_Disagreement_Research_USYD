"""
paths.py
========
Single source of truth for every path used across this project's scripts and notebooks. Resolves relative to this file's own location (PROJECT_ROOT two levels up from src/). This approach avoids hardcoding any paths specific to the system of origin and should ensure that the analysis conducted in this project will be reproducebale across a range of devices and operating systems.

Usage from a script in src/ (e.g. extract_zips.py, ingest_cboe.py): 
    from paths import PROJECT_ROOT, PARQUET_DIR

Usage from a notebook in notebooks/: 
    import sys
    from pathlib import Path
    sys.path.append(str(Path.cwd().parent/"src"))
    from paths import PROJECT_ROOT, PARQUET_DIR
"""

from pathlib import Path

from pathlib import Path
 
PROJECT_ROOT = Path(__file__).parent.parent
 
# NOTE: this folder name is case-sensitive on Mac/Linux, even though Windows
# won't complain either way. Match this exactly to the real folder on disk.
ZIP_DIR = PROJECT_ROOT / "CBOE_Data_2011_2022"
 
DATA_DIR = PROJECT_ROOT / "data"
EXTRACTED_DIR = DATA_DIR / "CBOE_DATA_RAW_EXTRACTED"
PARQUET_DIR = DATA_DIR / "cboe_parquet"
IBES_DIR = DATA_DIR / "ibes_quarterly_report"
 
SRC_DIR = PROJECT_ROOT / "src"
WORKBOOKS_DIR = PROJECT_ROOT / "workbooks"