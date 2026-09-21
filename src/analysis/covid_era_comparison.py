"""
covid_era_comparison.py
=========================
Compares retail vs. professional options behaviour around earnings across
three eras spanning both the base (2011-2022) and extended (2022-2026)
samples:

    pre_covid   2016-01-01 .. 2019-12-31   (base sample)
    covid       2020-01-01 .. 2021-12-31   (base sample)
    post_covid  2022-08-01 .. 2025-12-31   (extended sample)

Two design points that this module handles so callers don't have to:

  1. The eras live in DIFFERENT physical datasets. pre_covid and covid come
     from the base parquet files; post_covid from the *_ext files. This
     module flips event_window_profile's sample switch to the correct source
     for each era, so an era is always read from the dataset that actually
     contains it.

  2. The pre-COVID window deliberately starts in 2016, NOT 2011. CBOE's
     professional-customer classification undergoes a documented definitional
     break in 2015 (coverage halves permanently; the control group is not
     consistently defined before then). Beginning pre-COVID at 2016 keeps the
     control group comparable across all three eras.

  3. post_covid ends 2025-12-31, capped by CRSP (moneyness and the size
     control need spot prices, which stop there) rather than by the CBOE
     options data, which runs to May 2026.

The eras are compared as SEPARATE difference-in-differences estimates, never
pooled into one regression. A single regression spanning 2016-2025 would run
across both a classification break and a pandemic, and its coefficients would
not be interpretable. Reporting three self-contained estimates and comparing
them is the defensible design.

Usage:
    from analysis.covid_era_comparison import compare_covid_eras
    tbl = compare_covid_eras()
"""

from __future__ import annotations

import polars as pl

from analysis.event_window_profile import (
    set_sample, current_sample,
    build_diff_in_diff_panel, run_diff_in_diff, build_balanced_panel,
)

# era -> (sample, date_from, date_to)
COVID_ERAS = {
    "pre_covid":  ("base", "2016-01-01", "2019-12-31"),
    "covid":      ("base", "2020-01-01", "2021-12-31"),
    "post_covid": ("ext",  "2022-08-01", "2025-12-31"),
}


def compare_covid_eras(
    outcomes: list[str] = None,
    cluster_by: str = "ticker",
    balanced: bool = True,
    eras: dict = None,
) -> pl.DataFrame:
    """Binary difference-in-differences (treat:post) per outcome, per era,
    each read from the sample that physically contains it.

    balanced=True runs on the balanced panel (all four cells present per
    firm-event), which the project uses as its primary specification because
    the unbalanced panel is biased by professional customers dropping out of
    the short near-event window in small stocks.

    Restores whatever sample was active before the call, so it does not leave
    the switch in an unexpected state for later cells.
    """
    outcomes = outcomes or ["otm", "otm_put", "lt_100", "call", "open"]
    eras = eras or COVID_ERAS
    saved = current_sample()
    rows = []
    try:
        for era_name, (sample, d0, d1) in eras.items():
            set_sample(sample)
            for oc in outcomes:
                panel = build_diff_in_diff_panel(
                    outcome=oc, verbose=False, date_from=d0, date_to=d1
                )
                if balanced:
                    panel = build_balanced_panel(panel, verbose=False)
                if panel.height < 100:
                    print(f"  skipping {era_name}/{oc}: only {panel.height} panel rows")
                    continue
                m = run_diff_in_diff(panel, cluster_by=cluster_by)
                rows.append({
                    "era": era_name,
                    "sample": sample,
                    "outcome": oc,
                    "coef": m.params["treat:post"],
                    "p_value": m.pvalues["treat:post"],
                    "n_obs": int(m.nobs),
                })
    finally:
        set_sample(saved)  # always restore, even if something above raises

    return pl.DataFrame(rows)


if __name__ == "__main__":
    with pl.Config(tbl_rows=-1, float_precision=4):
        print(compare_covid_eras(outcomes=["otm"]))


def compare_covid_eras_controlled(
    outcomes: list[str] = None,
    cluster_by: str = "ticker",
    eras: dict = None,
) -> pl.DataFrame:
    """Size-controlled version of the era comparison.

    Runs the triple-interaction regression with log market cap allowed its
    own event response (dispersion * treat * post + log_mktcap * treat * post)
    per outcome, per era, each on its correct sample. Reports the retail-vs-
    professional near-event coefficient (treat:post) net of the size channel,
    which the project's base analysis showed is the dominant confounder.

    The question this answers: does an era's headline effect survive once
    firm size is controlled, or was size doing the work? An effect that is
    present pre/during COVID and absent post-COVID *even after* controlling
    for size is a genuine behavioural change, not a compositional one.
    """
    from analysis.event_window_profile import add_market_cap, run_dispersion_regression

    outcomes = outcomes or ["otm", "otm_put", "lt_100", "call", "open"]
    eras = eras or COVID_ERAS
    saved = current_sample()
    rows = []
    try:
        for era_name, (sample, d0, d1) in eras.items():
            set_sample(sample)
            for oc in outcomes:
                panel = build_diff_in_diff_panel(
                    outcome=oc, verbose=False, date_from=d0, date_to=d1
                )
                if panel.height < 100:
                    print(f"  skipping {era_name}/{oc}: only {panel.height} panel rows")
                    continue
                pmc = add_market_cap(panel, verbose=False)
                m = run_dispersion_regression(
                    pmc, spec="triple", cluster_by=cluster_by,
                    controls=["log_mktcap"], verbose=False,
                )
                rows.append({
                    "era": era_name,
                    "sample": sample,
                    "outcome": oc,
                    "treat_post": m.params["treat:post"],
                    "p_treat_post": m.pvalues["treat:post"],
                    "size_did": m.params["log_mktcap:treat:post"],
                    "p_size": m.pvalues["log_mktcap:treat:post"],
                    "n_obs": int(m.nobs),
                })
    finally:
        set_sample(saved)

    return pl.DataFrame(rows)


def era_levels(
    outcome: str = "otm",
    eras: dict = None,
) -> pl.DataFrame:
    """Mean outcome share by participant group and period, per era.

    A DiD coefficient of ~0 can mean two very different things: both groups
    were flat, or both moved by the same amount. This shows which. Use it to
    confirm a null (like post-COVID OTM) is 'retail and professionals moved
    together' rather than a data artefact in one group.

    Reports, per era, the four cell means (retail/procust x baseline/near),
    read from the sample that contains that era.
    """
    eras = eras or COVID_ERAS
    saved = current_sample()
    frames = []
    try:
        for era_name, (sample, d0, d1) in eras.items():
            set_sample(sample)
            panel = build_diff_in_diff_panel(
                outcome=outcome, verbose=False, date_from=d0, date_to=d1
            )
            if panel.height < 100:
                continue
            lv = (
                panel.group_by(["participant_group", "is_near_event"])
                .agg(pl.col("share").mean().alias("mean_share"), pl.len().alias("n"))
                .with_columns(pl.lit(era_name).alias("era"), pl.lit(sample).alias("sample"))
            )
            frames.append(lv)
    finally:
        set_sample(saved)

    return (
        pl.concat(frames)
        .select(["era", "sample", "participant_group", "is_near_event", "mean_share", "n"])
        .sort(["era", "participant_group", "is_near_event"])
    )


def era_levels_all(
    outcomes: list[str] = None,
    eras: dict = None,
) -> pl.DataFrame:
    """era_levels for several outcomes at once, stacked into one long table.

    One row per (outcome, era, participant_group, period) with the mean
    share, so every reversal can be eyeballed in a single view rather than
    calling era_levels once per outcome. Read from the sample that contains
    each era, sample restored afterwards."""
    outcomes = outcomes or ["otm", "otm_put", "lt_100", "call", "open"]
    frames = []
    for oc in outcomes:
        lv = era_levels(outcome=oc, eras=eras)
        frames.append(lv.with_columns(pl.lit(oc).alias("outcome")))
    return (
        pl.concat(frames)
        .select(["outcome", "era", "sample", "participant_group", "is_near_event", "mean_share", "n"])
        .sort(["outcome", "era", "participant_group", "is_near_event"])
    )
