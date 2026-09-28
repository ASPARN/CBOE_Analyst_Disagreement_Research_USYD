"""
pull_optionmetrics.py
=======================
Extracts OptionMetrics IvyDB US option prices and underlying prices around
every earnings announcement in the sample, for the option-return tests
(Test 2: holding-period returns; Test 3: delta/vega/theta decomposition).

Why this runs through the WRDS Python library rather than the web query form:
the tests need about four specific trading days per firm-event, scattered
across ten years. The web form only accepts one date range per firm list, so
any range wide enough to cover the events pulls nearly every trading day for
every firm. Here we send the exact (secid, date) pairs instead.

What gets pulled, per firm-event
--------------------------------
Trading days -1, 0, +1 and +2 relative to the IBES announcement date. That
covers both timing cases without a second pull:
  - before-open announcement: effective day 0 = ANNDATS_ACT -> needs -1, +1
  - after-close announcement: effective day 0 = next trading day -> needs 0, +2
The effective day 0 is set later, from ANNTIMS_ACT, in the analysis module.

Linking
-------
Events carry IBES 8-digit CUSIPs; OptionMetrics' security-name table
(secnmd) carries 8-digit CUSIPs too, so events link straight to secid
without routing through CRSP. The match rate is reported; unmatched events
simply have no option data (most are firms without listed options).

Outputs (all under data/optionmetrics/, which data/ in .gitignore covers)
-------------------------------------------------------------------------
  secnmd.parquet                 OptionMetrics security names (link table)
  event_pairs.parquet            every (event, secid, trading date) requested
  opprcd_<year>/part_NNNN.parquet   option prices, one file per query chunk
  secprd_<year>/part_NNNN.parquet   underlying prices, same chunks

Resumable: each chunk writes its own file and is skipped if already present,
so an interrupted pull (dropped connection, Duo timeout) picks up where it
stopped. Read a year back with pl.scan_parquet(dir / "*.parquet").

Things this deliberately does NOT do (handled in the analysis module)
---------------------------------------------------------------------
  - rescale strike_price (stored x1000 by OptionMetrics)
  - map pre-2015 Saturday monthly expiries to the Friday CBOE uses
  - value contracts expiring inside the window at intrinsic value
Raw OptionMetrics fields are stored as delivered so those conventions are
applied in one visible place, not buried in the extract.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import polars as pl

sys.path.append(str(Path(__file__).parent.parent))
from paths import DATA_DIR, IBES_DIR, CRSP_DIR

OM_DIR = DATA_DIR / "optionmetrics"
SCHEMA = "optionm_all"

# Offsets (in trading days) pulled around each announcement date.
OFFSETS = (-1, 0, 1, 2)

# Only keep contracts expiring within this many calendar days of the quote
# date. Earnings trading concentrates in short-dated contracts; this keeps the
# extract to a few GB instead of tens. Raise it if the CBOE join shows
# traded contracts being lost to the filter.
MAX_DAYS_TO_EXPIRY = 120

# Pairs per SQL query. Larger = fewer round trips but bigger queries.
CHUNK_SIZE = 2000

OPTION_COLS = [
    "secid", "date", "optionid", "symbol", "root", "exdate", "cp_flag",
    "strike_price", "best_bid", "best_offer", "volume", "open_interest",
    "impl_volatility", "delta", "gamma", "vega", "theta",
    "contract_size", "ss_flag", "am_settlement", "expiry_indicator",
]
SECURITY_COLS = ["secid", "date", "close", "return", "cfadj", "volume", "shrout"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _q(col: str) -> str:
    """Quote a column name. Needed because 'return' is reserved in Postgres."""
    return f'"{col}"'


def _available_columns(db, table: str, wanted: list[str]) -> list[str]:
    """Intersect the columns we want with those the table actually has,
    reporting any that are missing rather than failing on a SQL error."""
    have = {c.lower() for c in db.describe_table(library=SCHEMA, table=table)["name"]}
    keep = [c for c in wanted if c in have]
    missing = [c for c in wanted if c not in have]
    if missing:
        print(f"  note: {table} has no column(s) {missing} -- skipped")
    return keep


def _trading_calendar() -> pl.Series:
    """Union of the base and extended CRSP trading days, sorted."""
    files = [CRSP_DIR / "crsp_daily.parquet", CRSP_DIR / "crsp_daily_ext.parquet"]
    frames = [
        pl.scan_parquet(f).select(pl.col("DlyCalDt").alias("date")).unique()
        for f in files if f.exists()
    ]
    if not frames:
        raise FileNotFoundError("No CRSP parquet found to build a trading calendar.")
    return pl.concat(frames).unique().sort("date").collect()["date"]


def _load_events(date_from: str, date_to: str) -> pl.DataFrame:
    """Base + extended event panels, restricted to the requested window."""
    files = [IBES_DIR / "dispersion_events.parquet",
             IBES_DIR / "dispersion_events_ext.parquet"]
    frames = []
    for f in files:
        if not f.exists():
            continue
        lf = pl.scan_parquet(f)
        cols = lf.collect_schema().names()
        if "CUSIP" not in cols or "ANNDATS_ACT" not in cols:
            raise KeyError(f"{f.name} needs CUSIP and ANNDATS_ACT; has {cols}")
        keep = [c for c in ["TICKER", "OFTIC", "CUSIP", "ANNDATS_ACT", "ANNTIMS_ACT", "FPEDATS"]
                if c in cols]
        frames.append(lf.select(keep).collect())
    ev = pl.concat(frames, how="diagonal_relaxed").unique()
    return ev.filter(
        (pl.col("ANNDATS_ACT") >= pl.lit(date_from).str.to_date())
        & (pl.col("ANNDATS_ACT") <= pl.lit(date_to).str.to_date())
    )


# ---------------------------------------------------------------------------
# steps
# ---------------------------------------------------------------------------

def pull_secnmd(db, force: bool = False) -> pl.DataFrame:
    """OptionMetrics security names -- the CUSIP -> secid link table."""
    out = OM_DIR / "secnmd.parquet"
    if out.exists() and not force:
        return pl.read_parquet(out)
    print("Pulling secnmd (security names) ...")
    df = pl.from_pandas(db.raw_sql(f"SELECT * FROM {SCHEMA}.secnmd", date_cols=["effect_date"]))
    df = df.with_columns(pl.col("secid").cast(pl.Int64))
    OM_DIR.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out, compression="zstd")
    print(f"  {df.height:,} name records, {df['secid'].n_unique():,} secids")
    return df


def build_event_pairs(secnmd: pl.DataFrame, date_from: str, date_to: str) -> pl.DataFrame:
    """One row per (event, trading-day offset), with the linked secid."""
    events = _load_events(date_from, date_to)
    print(f"Events {date_from} .. {date_to}: {events.height:,}")

    # CUSIP -> secid, as of each announcement date. A CUSIP occasionally maps
    # to more than one secid over time (reorganisations, relistings), so each
    # event takes the secid whose name record was in effect on ANNDATS_ACT.
    # Events dated before a CUSIP's first name record fall back to that first
    # record.
    link = (secnmd.filter(pl.col("cusip").is_not_null())
            .select(pl.col("cusip").alias("CUSIP"), "secid",
                    pl.col("effect_date").cast(pl.Date))
            .unique()
            .sort("effect_date"))
    n_ambig = (link.group_by("CUSIP").agg(pl.col("secid").n_unique().alias("n"))
               .filter(pl.col("n") > 1).height)

    ev = (events.sort("ANNDATS_ACT")
          .join_asof(link, left_on="ANNDATS_ACT", right_on="effect_date",
                     by="CUSIP", strategy="backward", check_sortedness=False))
    first = (link.group_by("CUSIP").agg(pl.col("secid").first().alias("_first_secid")))
    ev = (ev.join(first, on="CUSIP", how="left")
            .with_columns(pl.coalesce("secid", "_first_secid").alias("secid"))
            .drop("_first_secid", "effect_date"))

    n_linked = ev.filter(pl.col("secid").is_not_null()).height
    print(f"  linked to an OptionMetrics secid: {n_linked:,} "
          f"({n_linked / max(events.height, 1):.1%}); "
          f"{n_ambig:,} CUSIPs map to >1 secid over time (matched as of event date)")
    ev = ev.filter(pl.col("secid").is_not_null())

    # Position of each announcement date in the trading calendar. A date that
    # isn't a trading day is mapped to the next trading day (search_sorted
    # 'left' returns that position).
    cal = _trading_calendar()
    pos = cal.search_sorted(ev["ANNDATS_ACT"], side="left")
    ev = ev.with_columns(pl.Series("_pos", pos).cast(pl.Int64))

    rows = []
    for k in OFFSETS:
        rows.append(ev.with_columns((pl.col("_pos") + k).alias("_p"), pl.lit(k).alias("offset")))
    pairs = pl.concat(rows)
    n_before = pairs.height
    pairs = pairs.filter((pl.col("_p") >= 0) & (pl.col("_p") < len(cal)))
    pairs = pairs.with_columns(cal.gather(pairs["_p"]).alias("date"))
    if pairs.height < n_before:
        print(f"  dropped {n_before - pairs.height:,} pairs falling outside the "
              f"trading calendar (events at the very end of the CRSP range)")
    pairs = pairs.drop("_pos", "_p").with_columns(pl.col("date").dt.year().alias("year"))

    out = OM_DIR / "event_pairs.parquet"
    pairs.write_parquet(out, compression="zstd")
    n_unique = pairs.select("secid", "date").unique().height
    print(f"  {pairs.height:,} (event, day) rows -> {n_unique:,} unique (secid, date) pairs")
    return pairs


def _values_clause(chunk: pl.DataFrame) -> str:
    return ", ".join(f"({int(s)}, DATE '{d}')" for s, d in zip(chunk["secid"], chunk["date"]))


def pull_year(db, pairs: pl.DataFrame, year: int, max_chunks: int = None) -> None:
    """Option prices and underlying prices for one year's (secid, date) pairs."""
    yp = (pairs.filter(pl.col("year") == year)
          .select("secid", "date").unique().sort("secid", "date"))
    if yp.height == 0:
        return

    tables = set(db.list_tables(library=SCHEMA))
    opt_table = f"opprcd{year}"
    sec_table = f"secprd{year}" if f"secprd{year}" in tables else "secprd"
    if opt_table not in tables:
        print(f"[{year}] {opt_table} not found in {SCHEMA} -- skipping year")
        return

    opt_cols = _available_columns(db, opt_table, OPTION_COLS)
    sec_cols = _available_columns(db, sec_table, SECURITY_COLS)

    opt_dir = OM_DIR / f"opprcd_{year}"
    sec_dir = OM_DIR / f"secprd_{year}"
    opt_dir.mkdir(parents=True, exist_ok=True)
    sec_dir.mkdir(parents=True, exist_ok=True)

    n_chunks = (yp.height + CHUNK_SIZE - 1) // CHUNK_SIZE
    if max_chunks is not None:
        n_chunks = min(n_chunks, max_chunks)
    print(f"[{year}] {yp.height:,} pairs in {n_chunks} chunk(s) "
          f"(tables: {opt_table}, {sec_table})")

    opt_select = ", ".join(f"o.{_q(c)}" for c in opt_cols)
    sec_select = ", ".join(f"s.{_q(c)}" for c in sec_cols)
    opt_dates = [c for c in ("date", "exdate") if c in opt_cols]

    t_year = time.time()
    rows_year = 0
    for i in range(n_chunks):
        opt_out = opt_dir / f"part_{i:04d}.parquet"
        sec_out = sec_dir / f"part_{i:04d}.parquet"
        if opt_out.exists() and sec_out.exists():
            continue

        chunk = yp.slice(i * CHUNK_SIZE, CHUNK_SIZE)
        values = _values_clause(chunk)

        opt_sql = f"""
            SELECT {opt_select}
            FROM {SCHEMA}.{opt_table} o
            JOIN (VALUES {values}) AS p(secid, date)
              ON o.secid = p.secid AND o.date = p.date
            WHERE o.exdate <= o.date + {MAX_DAYS_TO_EXPIRY}
        """
        sec_sql = f"""
            SELECT {sec_select}
            FROM {SCHEMA}.{sec_table} s
            JOIN (VALUES {values}) AS p(secid, date)
              ON s.secid = p.secid AND s.date = p.date
        """

        for attempt in range(3):
            try:
                opt = db.raw_sql(opt_sql, date_cols=opt_dates)
                sec = db.raw_sql(sec_sql, date_cols=["date"])
                break
            except Exception as e:  # dropped connection, timeout
                if attempt == 2:
                    raise
                print(f"    chunk {i}: {type(e).__name__}, retrying in 10s")
                time.sleep(10)

        pl.from_pandas(opt).write_parquet(opt_out, compression="zstd")
        pl.from_pandas(sec).write_parquet(sec_out, compression="zstd")
        rows_year += len(opt)
        if (i + 1) % 10 == 0 or i + 1 == n_chunks:
            print(f"    chunk {i + 1}/{n_chunks}: {rows_year:,} option rows so far "
                  f"({time.time() - t_year:.0f}s)")


def run_optionmetrics_pull(
    db,
    date_from: str = "2016-01-01",
    date_to: str = "2025-08-27",
    years: list[int] = None,
    max_chunks: int = None,
) -> None:
    """Full pull. Defaults to the thesis's primary sample (2016 onward).

    date_to defaults to 2025-08-27 because the current IvyDB annual update
    ends on 2025-08-29: an after-close announcement on the 27th has its
    effective day 0 on the 28th and day +1 on the 29th, the last day of data.
    Raise it once a later OptionMetrics update lands.

    Use `years=[2021], max_chunks=1` for a small test run first.

    IMPORTANT for resuming: chunks are numbered within each year from the
    sorted (secid, date) pairs, so always call this with the SAME date range.
    If you change date_from/date_to, delete the opprcd_<year> and
    secprd_<year> folders for the affected years first, or the saved chunks
    will no longer line up with the new pairs."""
    OM_DIR.mkdir(parents=True, exist_ok=True)
    secnmd = pull_secnmd(db)
    pairs = build_event_pairs(secnmd, date_from, date_to)
    for year in (years or sorted(pairs["year"].unique().to_list())):
        pull_year(db, pairs, year, max_chunks=max_chunks)
    print("\nDone. Option prices in data/optionmetrics/opprcd_<year>/")


if __name__ == "__main__":
    import wrds
    run_optionmetrics_pull(wrds.Connection())
