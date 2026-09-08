"""
analyse_order_size.py
=======================
Tests whether participants altered their order behaviour around the 2015
changes to CBOE's Professional designation rules.

Background. Professional status under CBOE Rule 1.1(ggg) depends on placing
more than 390 orders per day on average during a calendar month. Public
customers receive priority in allocation and fee exemptions that
Professionals do not, so there is a standing incentive to remain below the
threshold. Because the count is mechanical, doing so is a legitimate
response to the rule rather than concealment. A participant wishing to stay
under it would place fewer, larger orders.

SR-CBOE-2015-011 (approved February 2015, mandatory from 1 June 2015)
changed how complex orders are ticketed, and a further 2016 filing revised
the counting methodology. Both alter the arithmetic determining who
qualifies. This project separately documents a permanent halving of
professional-customer coverage in 2015.

What can and cannot be tested. CBOE Open/Close reports volume and
transaction counts, not orders -- so the 390-order threshold cannot be
observed directly and the mechanism cannot be tested head-on. What is
observable is contracts per transaction (vol / qty), the closest available
proxy for average order size. If threshold avoidance occurred, average
trade size should rise around 2015, and rise more for the customer category
than for participants unaffected by the designation.

Market-maker flow is included as a control: market makers are defined by
exchange role rather than order counts, so their trade size should not
respond to a Professional-designation rule change.

Usage:
    from analysis.analyse_order_size import order_size_by_year
    tbl = order_size_by_year()
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

import sys
sys.path.append(str(Path(__file__).parent.parent))
from paths import DATA_DIR

DAILY_RETAIL_DIR = DATA_DIR / "cboe_daily_retail"
GROUPS = ["retail", "procust", "firm", "bd", "mm"]


def order_size_by_year() -> pl.DataFrame:
    """Mean contracts per transaction by participant group and year."""
    files = sorted(DAILY_RETAIL_DIR.glob("daily_retail_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No daily files in {DAILY_RETAIL_DIR}.")
    daily = pl.concat([pl.read_parquet(f) for f in files])

    missing = [g for g in GROUPS if f"{g}_qty_total" not in daily.columns]
    if missing:
        raise ValueError(
            f"No qty columns for {missing}. Rebuild with build_daily_retail_activity.py."
        )

    agg = []
    for g in GROUPS:
        agg += [
            pl.col(f"{g}_vol_total").sum().alias(f"{g}_vol"),
            pl.col(f"{g}_qty_total").sum().alias(f"{g}_qty"),
        ]

    out = (
        daily.with_columns(pl.col("quote_date").dt.year().alias("year"))
        .group_by("year")
        .agg(agg)
        .sort("year")
    )

    # contracts per transaction; null rather than div-by-zero where a group is absent
    return out.select(
        ["year"]
        + [
            pl.when(pl.col(f"{g}_qty") > 0)
            .then(pl.col(f"{g}_vol") / pl.col(f"{g}_qty"))
            .otherwise(None)
            .alias(f"{g}_size")
            for g in GROUPS
        ]
    )


def order_size_change(pre: tuple = (2011, 2014), post: tuple = (2016, 2019)) -> pl.DataFrame:
    """Compares mean trade size before and after the 2015 rule changes.

    2015 itself is excluded from both windows, since the compliance date for
    SR-CBOE-2015-011 falls mid-year (1 June 2015) and the year therefore
    straddles the change. The post window stops at 2019 to avoid confounding
    with the 2020-21 retail boom."""
    tbl = order_size_by_year()
    pre_df = tbl.filter(pl.col("year").is_between(*pre))
    post_df = tbl.filter(pl.col("year").is_between(*post))

    rows = []
    for g in GROUPS:
        a = pre_df[f"{g}_size"].mean()
        b = post_df[f"{g}_size"].mean()
        rows.append({
            "group": g,
            f"pre_{pre[0]}_{pre[1]}": a,
            f"post_{post[0]}_{post[1]}": b,
            "change": (b - a) if (a is not None and b is not None) else None,
            "pct_change": ((b / a - 1) if (a and b) else None),
        })
    return pl.DataFrame(rows)


if __name__ == "__main__":
    with pl.Config(tbl_rows=-1, float_precision=3):
        print(order_size_by_year())
        print()
        print(order_size_change())
