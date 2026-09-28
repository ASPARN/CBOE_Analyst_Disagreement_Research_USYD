"""
option_decomposition.py
=========================
Test 3: why do options bought before earnings lose value across the
announcement? Each contract's midpoint price change from day -1 to day +1 is
decomposed with its day -1 Greeks:

    dP = Delta*dS + 1/2*Gamma*dS^2 + Vega*d(sigma) + Theta*(days/365) + residual
         --------   --------------   -------------   -----------------
         direction   convexity        IV crush        time decay

Units follow the IvyDB US manual (v7.0): Delta per $1 of the underlying;
Gamma per $1^2; Vega in $ per 1.00 change in implied volatility (equivalently
cents per volatility point), so it multiplies the change in IV as a decimal;
Theta in $ per year, so it multiplies calendar days / 365. The gamma term is
included because earnings moves are large enough that a delta-only
approximation pushes the curvature into the residual; the residual itself is
reported so the quality of the approximation is visible.

Excluded, with their share of volume reported:
  - contracts that expired inside the window and were valued at their payoff
    (there is no exit IV, so the change cannot be split this way)
  - contracts with a missing implied volatility or Greek at either end

The decomposition works at the midpoint, so it explains the midpoint return
from Test 2 (option_returns); the bid-ask spread is a separate cost on top.

Functions
---------
decomposition_panel(contracts)      contract-level terms and flags
event_decomposition(d, ...)         per firm-event, each term as a share of
                                    the premium the group paid
decomposition_table(ev)             means and medians by era and group
right_direction_summary(d, ...)     Andrew's claim directly: among positions
                                    where the group called the direction
                                    correctly, how many still lost money, and
                                    how often the IV change alone outweighed
                                    the directional gain
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.append(str(Path(__file__).parent.parent))
from analysis.informed_trading import ERAS
from analysis.option_returns import GROUPS, PANEL_PATH

TERMS = ("delta_term", "gamma_term", "vega_term", "theta_term", "residual", "dP")
LABELS = {"delta_term": "direction (delta)", "gamma_term": "convexity (gamma)",
          "vega_term": "IV change (vega)", "theta_term": "time decay (theta)",
          "residual": "residual", "dP": "total midpoint change"}
NEEDED = ("delta_entry", "gamma_entry", "vega_entry", "theta_entry",
          "impl_volatility_entry", "impl_volatility_exit", "spot_entry", "spot_exit")


def _weight(g: str, weight: str) -> pl.Expr:
    ob = pl.col(f"{g}_open_buy").cast(pl.Float64)
    if weight == "open_buy":
        return ob
    if weight == "buy":
        return ob + pl.col(f"{g}_close_buy")
    raise ValueError("weight must be 'buy' or 'open_buy'")


def decomposition_panel(contracts: pl.DataFrame | None = None, verbose: bool = True) -> pl.DataFrame:
    """Contract-level decomposition. `contracts` is the Test 2 contract panel
    (read from disk if not given)."""
    c = contracts if contracts is not None else pl.read_parquet(PANEL_PATH)
    missing = [x for x in NEEDED if x not in c.columns]
    if missing:
        raise KeyError(f"Contract panel lacks {missing}; rebuild it with option_returns.build_option_panel().")
    priced = c.filter(pl.col("ret").is_not_null())

    usable = (~pl.col("valued_at_expiry")
              & pl.all_horizontal(pl.col(x).is_not_null() for x in NEEDED)
              & (pl.col("impl_volatility_entry") > 0) & (pl.col("impl_volatility_exit") > 0))
    d = priced.filter(usable).with_columns(
        (pl.col("spot_exit") - pl.col("spot_entry")).alias("dS"),
        (pl.col("impl_volatility_exit") - pl.col("impl_volatility_entry")).alias("d_sigma"),
        ((pl.col("exit_date") - pl.col("entry_date")).dt.total_days() / 365).alias("dt_years"),
    ).with_columns(
        (pl.col("delta_entry") * pl.col("dS")).alias("delta_term"),
        (0.5 * pl.col("gamma_entry") * pl.col("dS") ** 2).alias("gamma_term"),
        (pl.col("vega_entry") * pl.col("d_sigma")).alias("vega_term"),
        (pl.col("theta_entry") * pl.col("dt_years")).alias("theta_term"),
    ).with_columns(
        (pl.col("dP") - pl.col("delta_term") - pl.col("gamma_term")
         - pl.col("vega_term") - pl.col("theta_term")).alias("residual"),
    ).with_columns(
        (pl.col("delta_term") > 0).alias("right_direction"),
        (pl.col("dP") < 0).alias("lost"),
        (pl.col("vega_term") < -(pl.col("delta_term") + pl.col("gamma_term"))).alias("iv_outweighed_direction"),
    )

    if verbose:
        print(f"Priced contract-events: {priced.height:,}  |  decomposable: {d.height:,} "
              f"({d.height / max(priced.height, 1):.1%})")
        for g in GROUPS.values():
            w = _weight(g, "buy")
            tot = priced.select(w.sum()).item()
            exp = priced.filter(pl.col("valued_at_expiry")).select(w.sum()).item()
            got = d.select(w.sum()).item()
            print(f"  {g} buy volume: {got / max(tot, 1):.1%} decomposable, "
                  f"{exp / max(tot, 1):.1%} expired inside the window, "
                  f"{(tot - got - exp) / max(tot, 1):.1%} missing IV/Greeks")
        pred = d["dP"] - d["residual"]
        r2 = np.corrcoef(d["dP"].to_numpy(), pred.to_numpy())[0, 1] ** 2 if d.height > 2 else float("nan")
        rel = (d["residual"].abs() / d["mid_entry"]).median()
        print(f"  fit: R^2 of actual on approximated change {r2:.3f}; "
              f"median |residual| = {rel:.1%} of premium")
    return d


def event_decomposition(d: pl.DataFrame, weight: str = "buy", moneyness: str | None = None) -> pl.DataFrame:
    """Per firm-event and group: each term as a share of the premium the group
    paid (dollar-weighted by the group's contracts bought). The terms sum to
    the group's midpoint dollar-weighted return on the decomposable
    contracts."""
    x = d if moneyness is None else d.filter(pl.col("moneyness") == moneyness)
    aggs = []
    for g in GROUPS.values():
        w = _weight(g, weight)
        den = (w * pl.col("mid_entry")).sum()
        for t in TERMS:
            aggs.append(pl.when(den > 0).then((w * pl.col(t)).sum() / den).otherwise(None).alias(f"{g}_{t}"))
    return x.group_by("_eid", "OFTIC", "ANNDATS_ACT", "day0", "era", "PERMNO").agg(aggs).sort("day0")


def _stat(df: pl.DataFrame, col: str, winsorise: bool):
    import statsmodels.formula.api as smf
    s = df.select("PERMNO", "day0", col).drop_nulls().to_pandas()
    if len(s) < 30:
        return None
    y = s[col]
    if winsorise:
        y = y.clip(y.quantile(0.01), y.quantile(0.99))
    s["y"] = y
    g_firm = s["PERMNO"].astype("category").cat.codes.to_numpy()
    g_date = s["day0"].astype("category").cat.codes.to_numpy()
    # two-way clustering needs at least two clusters in each dimension; fall
    # back to firm-only clustering for a subsample confined to one date
    groups = np.column_stack([g_firm, g_date]) if len(set(g_date)) > 1 else g_firm
    m = smf.ols("y ~ 1", data=s).fit(cov_type="cluster", cov_kwds={"groups": groups})
    return {"n": int(m.nobs), "mean": m.params["Intercept"], "se": m.bse["Intercept"],
            "median": float(np.median(s[col]))}


def decomposition_table(ev: pl.DataFrame, winsorise: bool = False) -> pl.DataFrame:
    """Mean (firm-and-date clustered SE) and median of each term, by group
    and era. With winsorise=False (default) the component means add up
    exactly to the total; winsorise=True trims each term at 1/99 separately
    as a robustness check, at the cost of exact additivity. Medians never
    add up and should be read one term at a time."""
    rows = []
    for g in GROUPS.values():
        for era in [None] + list(ERAS):
            df = ev if era is None else ev.filter(pl.col("era") == era)
            for t in TERMS:
                st = _stat(df, f"{g}_{t}", winsorise)
                if st is None:
                    continue
                rows.append({"group": g, "era": era or "pooled", "component": LABELS[t], **st})
    return pl.DataFrame(rows)


def right_direction_summary(d: pl.DataFrame, weight: str = "buy", moneyness: str | None = None) -> pl.DataFrame:
    """Contract-level, weighted by each group's contracts bought:
         share_right          share of volume where the position's direction
                              was right (delta term positive)
         right_but_lost       of those, share that still lost money
         iv_outweighed        of those, share where the IV change alone
                              exceeded the directional gain (delta + gamma)
         then, for the right-direction positions only, each term as a share
         of premium (dollar-weighted)."""
    x = d if moneyness is None else d.filter(pl.col("moneyness") == moneyness)
    rows = []
    for g in GROUPS.values():
        w = _weight(g, weight)
        for era in [None] + list(ERAS):
            df = x if era is None else x.filter(pl.col("era") == era)
            tot = df.select(w.sum()).item()
            if not tot:
                continue
            right = df.filter(pl.col("right_direction"))
            rv = right.select(w.sum()).item()
            row = {"group": g, "era": era or "pooled", "contracts_bought": tot,
                   "share_right": rv / tot,
                   "right_but_lost": right.filter(pl.col("lost")).select(w.sum()).item() / rv if rv else None,
                   "iv_outweighed": right.filter(pl.col("iv_outweighed_direction")).select(w.sum()).item() / rv if rv else None}
            den = right.select((w * pl.col("mid_entry")).sum()).item()
            for t in TERMS:
                row[f"right_{t}"] = right.select((w * pl.col(t)).sum()).item() / den if den else None
            rows.append(row)
    return pl.DataFrame(rows)
