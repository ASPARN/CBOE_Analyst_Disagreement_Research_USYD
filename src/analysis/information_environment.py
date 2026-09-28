"""
information_environment.py
============================
Does the information environment around an announcement change how
informative retail option flow is?  Hypotheses stated BEFORE any result was
seen (notebook B6, committed before running):

D1 (primary)  Retail flow is more informative when analyst forecast
    dispersion is high: where analysts disagree, public information is
    weakest and private information most valuable.
    Prediction: flow x dispersion < 0. Tested linearly and with a
    top-quartile indicator, since the thesis's dispersion effects are
    confined to the top quartile.

D2  Retail flow is more informative when analyst coverage is thin.
    Coverage = log(number of analysts). Prediction: flow x coverage > 0
    (the negative flow effect weakens as coverage rises).

D3  (robustness of the uncertainty measure) The same with prior earnings
    volatility. Prediction: flow x earnings volatility < 0.

Design
------
Every specification is

    dv ~ flow_z + flow_z:mod + mod + flow_z:size_z + flow_z:attn_z + attn_z
         + controls + year-quarter FE

- flow x SIZE is always included. Dispersion is strongly correlated with
  firm size (small firms have more dispersed forecasts), so without it a
  flow x dispersion interaction could simply be picking up small firms. The
  version without it is also reported, to show how much size was doing.
- flow x ATTENTION is included because the sector analysis (B5) found
  retail flow more informative among low-attention events.
- Moderators enter as z-scores (after winsorising at 1/99), or as a
  top-quartile indicator; the moderator's own level replaces it in the
  controls, so it is never included twice.
- Standard errors clustered by firm and announcement date; pooled
  2016-2025, eras secondary.

Uses the sector panel from sector_analysis.build_sector_panel, which carries
the Test 1 variables plus attention and abnormal flow.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.append(str(Path(__file__).parent.parent))
from paths import RESULTS_DIR
from analysis.sector_analysis import _fit, _row, _z, CONTROLS

MODERATORS = {
    "dispersion": ("dispersion_scaled", "D1"),
    "coverage": ("_coverage", "D2"),
    "earnings_vol": ("earnings_volatility", "D3"),
}


def _prepare(sp: pl.DataFrame) -> pl.DataFrame:
    return sp.with_columns(
        pl.when(pl.col("NUMEST") > 0).then(pl.col("NUMEST").cast(pl.Float64).log()).otherwise(None).alias("_coverage"))


def moderator_test(sp: pl.DataFrame, moderator: str, dv: str = "car01", flow_col: str = "retail_net_put",
                   form: str = "linear", size_interaction: bool = True, attention: bool = True,
                   hyp: str | None = None, test: str | None = None, group: str = "retail") -> list[dict]:
    """One regression. form='linear' uses the moderator as a z-score;
    form='top_quartile' uses an indicator for the top quartile."""
    col, default_hyp = MODERATORS[moderator]
    hyp = hyp or default_hyp
    controls = [c for c in CONTROLS if c != col]
    need = [flow_col, dv, col, "log_mktcap"] + controls + (["abn_vol"] if attention else [])
    d = _prepare(sp).drop_nulls(need)
    if d.height < 200:
        return []
    d = d.with_columns(_z(d[flow_col]).alias("flow_z"), _z(d["log_mktcap"]).alias("size_z"))
    if form == "linear":
        d = d.with_columns(_z(d[col]).alias("mod"))
    else:
        cut = d[col].quantile(0.75)
        d = d.with_columns((pl.col(col) >= cut).cast(pl.Int8).alias("mod"))
    terms = ["flow_z", "flow_z:mod", "mod"]
    if size_interaction:
        terms.append("flow_z:size_z")
    if attention:
        d = d.with_columns(_z(d["abn_vol"]).alias("attn_z"))
        terms += ["flow_z:attn_z", "attn_z"]
    m, n = _fit(d, dv, terms + controls, [dv] + controls)
    label = test or f"{moderator}, {form}" + ("" if size_interaction else ", NO size interaction")
    keep = ["flow_z", "flow_z:mod"] + (["flow_z:size_z"] if size_interaction else []) \
        + (["flow_z:attn_z"] if attention else [])
    return [_row(hyp, label, dv, t, m, n, group=group, moderator=moderator, form=form) for t in keep]


def test_information_environment(sp: pl.DataFrame, dvs=("car01", "surprise_scaled")) -> pl.DataFrame:
    """The pre-stated set: D1 (linear and top quartile), D2, D3, plus the
    robustness rows -- abnormal flow, professional flow, and D1 without the
    size interaction (descriptive: shows how much size was doing)."""
    rows = []
    for dv in dvs:
        rows += moderator_test(sp, "dispersion", dv)
        rows += moderator_test(sp, "dispersion", dv, form="top_quartile")
        rows += moderator_test(sp, "coverage", dv)
        rows += moderator_test(sp, "earnings_vol", dv)
        rows += moderator_test(sp, "earnings_vol", dv, form="top_quartile")
        rows += moderator_test(sp, "dispersion", dv, flow_col="retail_abn_flow",
                               test="dispersion, linear, abnormal flow")
        rows += moderator_test(sp, "dispersion", dv, flow_col="procust_net_put", hyp="comparison",
                               test="dispersion, linear, professional flow", group="procust")
        rows += moderator_test(sp, "dispersion", dv, size_interaction=False, hyp="descriptive")
    return pl.DataFrame(rows, infer_schema_length=None)


def write_information_results(sp: pl.DataFrame, out_dir: Path | None = None) -> pl.DataFrame:
    out_dir = out_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    t = test_information_environment(sp)
    t.write_csv(out_dir / "table14_information_environment.csv", float_precision=6)
    print("  wrote table14_information_environment.csv")
    return t
