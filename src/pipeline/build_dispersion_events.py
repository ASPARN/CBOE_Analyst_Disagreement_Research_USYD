"""
build_dispersion_events.py
==========================
Builds the firm-event level dispersion panel this studies projects regression models will use. One row per (ticker, fiscar quarter) earnings event, carrying analyst forecast dispersion from the conensus snapshot closest to, but before, the actual announcement date.

Dispersion is computed as STDEV / max |MEANEST|, DENOMINATOR_FLOOR), ("scaled dispersion), following Diether, Malloy & Shcerbina (2002), the standard approach in this literature. Raw STDEV alone isn't comparable across firms: a $100 strandard deviation means something completely different for a firm with a $4,000 mean EPS estimate than for one with a $0.04 estimate.

Several real data-quality issues were found in the raw IBES export during development and are handled explicitly here, not silently:

1. NUMEST == 1 -- standard deviation is mathematically undefined for a single estimate. IBES correctly returns null STDEV for these; the events are dropped since dispersion cannot be measured for them at all.
2. Implausible |MEANEST| magnitude -- a small number of rows carry EPS means in the hundreds of millions. This is almost certainly a scale/units error upstream of this file. MEANEST_CEILING excludes these rows entirely. MEANEST_FLOOR, by contrast, is set deliberatly tiny. The MEANEST_FLOOR exists to prevent a literal near-zero division, it does not exclude legitimate near breakeven companies. An earlier version used an exlusionary floor of 0.05. This earlier floor iteration dropped ~9% of the sample for this reason alone, discarding exactly the kind of financially distressed firm events this research is attempting to analyse.
3. Denominator flooring in the ratio itself (DENOMINATOR_FLOOR) -- rather than excluding near zero MEANEST rows, the ratio's denominator is floored at a modest value, bounding how extreme a single ratio can get from a small denominator while keeping every observation in the sample. Tested against an alternatie of now denominator floor at all (relying purely on winsorization)L the version here produces a comparable row count but a much better behaved distribution (in testing, a 99th-percentile winsorization cap of ~2.6 versus ~20 with no denominator floor). I decided to keep this version.
4. Winsorixation of the final scaled measure's upper tail -- catches whatever extremity remains after (1)-(3), capping influence without dropping the observation.

Usage from Jupyter:
    from build_dispersion_events import build_dispersion_events
    build_dispersion_events()

Useage from a terminal (run from the repo root):
    python src/build_dispersion_events.py
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

import sys
sys.path.append(str(Path(__file__).parent.parent))
from paths import IBES_DIR, PARQUET_DIR

MEANEST_FLOOR = 0.001 #Numerical safety floor for row exlcusion only. Prevents literal division blowup, not a data quality cut.
DENOMINATOR_FLOOR = 0.05 # Floors the denominator used in the ratio itself, not the row. Bounds how extreme a near xero MEANEST ratio can get without excluding the observation from the sample.
MEANEST_CEILING = 10_000 # Generous upper bound -- comftorbaly above Berkshire's ~$3,800 EPS, catches only clear scale errors
WINSORIZE_PCT = 0.99 # Caps the upper tail of rht scaled measure at this percentile

def get_cboe_date_bounds(parquet_dir: Path = PARQUET_DIR) -> tuple:
    files = sorted(parquet_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No CBOE parquet files found in {parquet_dir.resolve()}")
    bounds = pl.concat([
        pl.scan_parquet(f).select(
            pl.col("quote_date").min().alias("min"),
            pl.col("quote_date").max().alias("max"),
        )
        for f in files
    ]).collect()
    return bounds["min"].min(), bounds["max"].max()

def build_dispersion_events(
    ibes_path: Path = None,
    cboe_dir: Path = PARQUET_DIR,
    out_path: Path = None,
    fiscalp: str = "QTR",
) -> pl.DataFrame:
    ibes_path = ibes_path or (IBES_DIR / "ibes_clean.parquet")
    out_path = out_path or (IBES_DIR / "dispersion_events.parquet")

    lo, hi = get_cboe_date_bounds(cboe_dir)
    print(f"CBOE data window: {lo} to {hi}")

    df = pl.scan_parquet(ibes_path)

    scoped = df.filter(
        (pl.col("MEASURE") == "EPS")
        & (pl.col("FISCALP") == fiscalp)
        & pl.col("ANNDATS_ACT").is_not_null()
        & (pl.col("ANNDATS_ACT") >= lo)
        & (pl.col("ANNDATS_ACT") <= hi)
    )

    pre_event = scoped.filter(pl.col("STATPERS") < pl.col("ANNDATS_ACT"))

    events = (
        pre_event.sort("STATPERS")
        .group_by(["OFTIC", "FPEDATS"], maintain_order=True)
        .agg(pl.all().last())
        .sort(["OFTIC", "FPEDATS"])
        .collect()
    )

    n_before = events.height
    print(f"\n{n_before:,} firm-events before data-quality filtering")

    # 1. dispersion is undefined with only one contributing analyst
    events = events.filter(pl.col("NUMEST") >= 2)
    print(f" after NUMEST >=2: {events.height:,} ({n_before - events.height:,} dropped)")

    # 2. exclude implausible MEANEST magnitudes (scale/units errors)
    n_before_bound = events.height
    events = events.filter(
        pl.col("MEANEST").is_not_null()
        & (pl.col("MEANEST").abs() > MEANEST_FLOOR)
        & (pl.col("MEANEST").abs() < MEANEST_CEILING)
    )
    print(
             f"  after MEANEST bound ({MEANEST_FLOOR}-{MEANEST_CEILING:,}): {events.height:,} "
        f"({n_before_bound - events.height:,} dropped)"
    )

     # 3. scaled dispersion -- comparable across firms of different EPS scale. The denominator is floored (the row is not excluded) so a near zero MEANEST can't produce an artificially extreme ratio, while the observation itself is preserved in the sample.
    events = events.with_columns(
        (pl.col("STDEV") / pl.max_horizontal(pl.col("MEANEST").abs(), pl.lit(DENOMINATOR_FLOOR)))
        .alias("dispersion_scaled")
    )
         
    # 4. winsorize the upper tail -- a thin remaining tail after steps 1-2 is
    #    driven by near-zero denominators rather than further data errors
    cap = events["dispersion_scaled"].quantile(WINSORIZE_PCT)
    n_capped = events.filter(pl.col("dispersion_scaled") > cap).height
    events = events.with_columns(
        pl.when(pl.col("dispersion_scaled") > cap)
        .then(cap)
        .otherwise(pl.col("dispersion_scaled"))
        .alias("dispersion_scaled")
    )
    print(f"  winsorized {n_capped:,} row(s) above the {WINSORIZE_PCT:.0%} percentile (cap={cap:.3f})")
 
    out_path.parent.mkdir(parents=True, exist_ok=True)
    events.write_parquet(out_path, compression="zstd")
 
    print(f"\n{events.height:,} clean firm-events written to {out_path}")
    print(f"Unique tickers: {events['OFTIC'].n_unique():,}")
    print("\nFinal dispersion_scaled distribution:")
    print(events["dispersion_scaled"].describe())
 
    return events
 
 
if __name__ == "__main__":
    build_dispersion_events()
    