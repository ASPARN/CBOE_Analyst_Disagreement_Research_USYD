"""
build_knowledge_intensity.py
==============================
Industry knowledge intensity: the share of each industry's workforce in STEM
occupations, from the BLS Occupational Employment and Wage Statistics (OEWS)
national industry-specific estimates. Used by the sector analysis (B5) to test
whether retail option flow is more informative in knowledge-intensive
industries.

Input
-----
The OEWS national industry-specific release, downloaded from
https://www.bls.gov/oes/tables.htm and extracted into data/external/oews/.
BLS splits the release by industry level; the reader combines
    nat4d_*      4-digit industries
    nat3d_*      3-digit industries
    natsector_*  sectors
and ignores the rest of the zip: the ownership splits (*_owner_*), the 5-
and 6-digit file, and file_descriptions. If none of those names are present,
it falls back to every data file in the folder (for a single all-levels
file). Columns are found by name; needed: NAICS, I_GROUP, OCC_CODE, O_GROUP,
TOT_EMP. If a file also covers states or metro areas (AREA_TYPE), only the
national rows are kept.

STEM definition
---------------
Employment in SOC major groups
    15  Computer and Mathematical
    17  Architecture and Engineering
    19  Life, Physical, and Social Science
plus the three management occupations over those fields
    11-3021 Computer and Information Systems Managers
    11-9041 Architectural and Engineering Managers
    11-9121 Natural Sciences Managers
divided by the industry's total employment (OCC_CODE 00-0000). Summed over
DETAILED occupations only: OEWS lists some occupations at both the broad and
detailed level, and summing both would double-count.

Suppressed employment values ("**", "*", "#") are treated as missing, which
slightly understates the share where BLS suppresses a small STEM occupation.

Output
------
data/external/oews/stem_share.parquet, one row per OEWS industry:
    naics_key   normalised code: 4 digits ("3344"), 3 digits ("334") or a
                2-digit sector ("31"), with sector ranges ("31-33") expanded
    level       4, 3 or 2
    stem_share  STEM employment / total employment
    total_emp, stem_emp, naics_title

The OEWS vintage is a single year (2022 NAICS from May 2022 onward). Industry
STEM intensity changes slowly, so one recent vintage is applied to all events
2016-2025; that simplification is stated in the thesis.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.append(str(Path(__file__).parent.parent))
from paths import DATA_DIR

OEWS_DIR = DATA_DIR / "external" / "oews"
OUT_PATH = OEWS_DIR / "stem_share.parquet"

STEM_MAJOR = ("15-", "17-", "19-")
STEM_MANAGERS = ("11-3021", "11-9041", "11-9121")
NEEDED = ["NAICS", "I_GROUP", "OCC_CODE", "O_GROUP", "TOT_EMP"]


def _read_oews(path: Path) -> pl.DataFrame:
    """Read an OEWS download (xlsx/xls/csv), normalising column names to upper case."""
    import pandas as pd
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path, dtype=str)
    else:
        df = pd.read_csv(path, dtype=str)
    df.columns = [str(c).strip().upper() for c in df.columns]
    missing = [c for c in NEEDED if c not in df.columns]
    if missing:
        raise KeyError(f"{path.name} lacks {missing}; columns found: {list(df.columns)[:20]}")
    out = pl.from_pandas(df)
    if "AREA_TYPE" in out.columns:                       # an all-areas file: keep national rows
        out = out.filter(pl.col("AREA_TYPE").str.strip_chars() == "1")
    return out


def _to_number(col: str) -> pl.Expr:
    return (pl.col(col).cast(pl.Utf8).str.replace_all(",", "").str.strip_chars()
            .cast(pl.Float64, strict=False))


LEVEL_FILES = ("nat4d_", "nat3d_", "natsector_")


def _oews_files(folder: Path) -> list[Path]:
    """The by-level national files from the OEWS release, excluding the
    ownership splits; falls back to any data file for a single-file release."""
    data = [f for f in sorted(folder.iterdir())
            if f.suffix.lower() in (".xlsx", ".xls", ".csv") and f.name != OUT_PATH.name]
    usable = [f for f in data if "owner" not in f.stem.lower() and "description" not in f.stem.lower()]
    by_level = [f for f in usable if f.stem.lower().startswith(LEVEL_FILES)]
    return by_level or usable


def build_stem_share(src: Path | list[Path] | None = None, out_path: Path = OUT_PATH,
                     verbose: bool = True) -> pl.DataFrame:
    files = [src] if isinstance(src, Path) else (src or _oews_files(OEWS_DIR))
    if not files:
        raise FileNotFoundError(f"No OEWS file in {OEWS_DIR} -- see the module docstring for the download.")
    frames = []
    for f in files:
        if verbose:
            print(f"  reading {f.name} ...")
        frames.append(_read_oews(f))
    raw = pl.concat(frames, how="diagonal_relaxed")
    src_names = ", ".join(f.name for f in files)

    raw = raw.with_columns(
        pl.col("NAICS").cast(pl.Utf8).str.strip_chars().alias("NAICS"),
        pl.col("I_GROUP").cast(pl.Utf8).str.strip_chars().str.to_lowercase().alias("I_GROUP"),
        pl.col("OCC_CODE").cast(pl.Utf8).str.strip_chars().alias("OCC_CODE"),
        pl.col("O_GROUP").cast(pl.Utf8).str.strip_chars().str.to_lowercase().alias("O_GROUP"),
        _to_number("TOT_EMP").alias("emp"),
    )
    level = (pl.when(pl.col("I_GROUP") == "4-digit").then(4)
             .when(pl.col("I_GROUP") == "3-digit").then(3)
             .when(pl.col("I_GROUP") == "sector").then(2).otherwise(None))
    raw = raw.with_columns(level.alias("level")).filter(pl.col("level").is_not_null())

    total = (raw.filter(pl.col("OCC_CODE") == "00-0000")
             .group_by("NAICS", "level").agg(pl.col("emp").max().alias("total_emp")))
    is_stem = (pl.any_horizontal([pl.col("OCC_CODE").str.starts_with(p) for p in STEM_MAJOR])
               | pl.col("OCC_CODE").is_in(list(STEM_MANAGERS)))
    stem = (raw.filter((pl.col("O_GROUP") == "detailed") & is_stem)
            .group_by("NAICS", "level").agg(pl.col("emp").sum().alias("stem_emp")))
    titles = raw.group_by("NAICS").agg(pl.col("NAICS_TITLE").first().alias("naics_title")) \
        if "NAICS_TITLE" in raw.columns else None

    ind = (total.join(stem, on=["NAICS", "level"], how="left")
           .with_columns(pl.col("stem_emp").fill_null(0.0))
           .filter(pl.col("total_emp") > 0)
           .with_columns((pl.col("stem_emp") / pl.col("total_emp")).alias("stem_share")))
    if titles is not None:
        ind = ind.join(titles, on="NAICS", how="left")

    # normalised key: 4-digit -> first 4 characters, 3-digit -> first 3, sector -> 2-digit code(s)
    def keys(naics: str, lvl: int) -> list[str]:
        if lvl == 2:
            head = naics.split("-")[0][:2]
            if "-" in naics:                          # "31-33" -> 31, 32, 33
                lo, hi = int(naics[:2]), int(naics.split("-")[1][:2])
                return [str(k) for k in range(lo, hi + 1)]
            return [head]
        return [naics[:lvl]]

    rows = []
    for r in ind.iter_rows(named=True):
        if not r["NAICS"].replace("-", "").isdigit():  # OEWS-specific combination codes (e.g. "3250A1")
            continue
        for k in keys(r["NAICS"], r["level"]):
            if k.isdigit():
                rows.append({"naics_key": k, "level": r["level"], "stem_share": r["stem_share"],
                             "total_emp": r["total_emp"], "stem_emp": r["stem_emp"],
                             "naics_title": r.get("naics_title")})
    out = (pl.DataFrame(rows).sort("level", "total_emp", descending=[True, True])
           .unique(subset=["naics_key", "level"], keep="first"))

    OEWS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_parquet(out_path, compression="zstd")
    if verbose:
        counts = out.group_by("level").len().sort("level", descending=True)
        print(f"STEM shares from {src_names}: " +
              ", ".join(f"{n} industries at {lvl}-digit" if lvl > 2 else f"{n} sectors"
                        for lvl, n in counts.iter_rows()))
        top = out.filter(pl.col("level") == 4).sort("stem_share", descending=True).head(8)
        print("  highest-STEM 4-digit industries:")
        for r in top.iter_rows(named=True):
            print(f"    {r['naics_key']}  {r['stem_share']:.1%}  {r.get('naics_title') or ''}")
    return out


if __name__ == "__main__":
    build_stem_share()
