"""
compare_retail_proxies.py
===========================
Compares CBOE's exchange-provided participant classification against the
small-trade proxy used in much of the retail options literature.

Background. Bryzgalova, Pavlova & Sikorskaya (2023) identify retail flow
using trade size and wholesaler internalisation, and explicitly criticise
exchange-provided order classifications for false positives. This project
uses the exchange classification (the cust_ columns), so that critique
applies directly to it -- and separately, the professional-customer category
is shown elsewhere in this work to undergo a definitional break in 2015.
This module quantifies how the two approaches relate.

An important data constraint. Only the two CUSTOMER categories (cust_ and
procust_) carry a contract-size breakdown in the CBOE Open/Close file. Firm,
broker-dealer and market-maker flow is reported without size tiers, so a
full small-trade proxy -- small trades regardless of participant type --
cannot be reconstructed from these data. What can be measured exactly:

  recall        share of true retail (cust_) volume that a small-trade
                filter would capture
  contamination share of small CUSTOMER volume that is professional rather
                than retail; a lower bound on the proxy's contamination,
                since firm/bd/mm small trades cannot be observed
  classifiable  share of all participant volume that can be size-classified
                at all, which bounds how informative the proxy can be

Reporting these by year also tests something the levels alone cannot: whether
the small-trade proxy is more temporally stable than the classification. If
the classification breaks in 2015 while recall and contamination do not, the
break is in category membership rather than in trading behaviour.

Usage:
    from analysis.compare_retail_proxies import compare_proxies
    tbl = compare_proxies()
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

import sys
sys.path.append(str(Path(__file__).parent.parent))
from paths import DATA_DIR

DAILY_RETAIL_DIR = DATA_DIR / "cboe_daily_retail"


def _load_daily() -> pl.DataFrame:
    files = sorted(DAILY_RETAIL_DIR.glob("daily_retail_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No daily files in {DAILY_RETAIL_DIR}.")
    return pl.concat([pl.read_parquet(f) for f in files])


def compare_proxies(by_year: bool = True) -> pl.DataFrame:
    daily = _load_daily()

    required = ["retail_vol_lt_100", "procust_vol_lt_100", "retail_vol_total", "procust_vol_total"]
    missing = [c for c in required if c not in daily.columns]
    if missing:
        raise ValueError(
            f"Missing columns {missing}. Rebuild with build_daily_retail_activity.py."
        )
    have_untiered = all(c in daily.columns for c in ["firm_vol_total", "bd_vol_total", "mm_vol_total"])

    if by_year:
        daily = daily.with_columns(pl.col("quote_date").dt.year().alias("year"))
        keys = ["year"]
    else:
        daily = daily.with_columns(pl.lit("all").alias("period"))
        keys = ["period"]

    agg = [
        pl.col("retail_vol_total").sum().alias("retail_vol"),
        pl.col("procust_vol_total").sum().alias("procust_vol"),
        pl.col("retail_vol_lt_100").sum().alias("retail_small"),
        pl.col("procust_vol_lt_100").sum().alias("procust_small"),
    ]
    if have_untiered:
        agg += [
            (pl.col("firm_vol_total") + pl.col("bd_vol_total") + pl.col("mm_vol_total"))
            .sum().alias("untiered_vol")
        ]

    out = daily.group_by(keys).agg(agg).sort(keys)

    out = out.with_columns(
        # of true retail volume, how much a <100-contract filter would capture
        (pl.col("retail_small") / pl.col("retail_vol")).alias("recall"),
        # of small CUSTOMER volume, how much is professional rather than retail.
        # A lower bound: firm/bd/mm small trades are unobservable here.
        (pl.col("procust_small") / (pl.col("retail_small") + pl.col("procust_small")))
        .alias("contamination_lb"),
    )
    if have_untiered:
        out = out.with_columns(
            ((pl.col("retail_vol") + pl.col("procust_vol"))
             / (pl.col("retail_vol") + pl.col("procust_vol") + pl.col("untiered_vol")))
            .alias("classifiable_share")
        )

    keep = keys + ["recall", "contamination_lb"]
    if have_untiered:
        keep.append("classifiable_share")
    keep += ["retail_vol", "procust_vol"]
    return out.select(keep)


if __name__ == "__main__":
    with pl.Config(tbl_rows=-1, float_precision=4):
        print(compare_proxies())
