# CBOE Options Analysis — Honours Thesis

Analyst forecast dispersion (IBES) and its relationship to retail options
trading activity (CBOE Open/Close), 2011 – May 2022.

Research questions:

1. To what extent does earnings-related uncertainty measured through
   analyst forecast dispersion and prior earnings volatility predict retail
   investor trading volume in equity options?
2. How does earnings uncertainty influence the type of options contracts
   retail investors select?

The design compares retail customers against professional customers as a
control group, so that shifts around earnings announcements can be attributed
to retail specifically rather than to the information event itself. Retail is
identified using CBOE's own participant classification rather than the
trade-size proxies used in most of this literature.

University of Sydney, Discipline of Finance.
Supervised by Professor Andrew Grant and Professor P. Joakim Westerholm.

## Repository layout

```
CBOE_Data_2011_2022/     Raw CBOE per-day zip files (not included -- see Setup)
data/                    All generated and licensed data (gitignored)
  CBOE_DATA_RAW_EXTRACTED/    extract_zips.py
  cboe_parquet/               ingest_cboe.py
  cboe_daily_retail/          build_daily_retail_activity.py
  cboe_daily_moneyness/       build_moneyness.py
  ibes_quarterly_report/      IBES export goes here; ingestion output lands here
  crsp/                       CRSP export goes here; ingestion output lands here
results/                 Generated tables (CSV) and figures (PNG)
notebooks/
  A1_data_pipeline_setup.ipynb       Builds every intermediate dataset, in order
  B1_core_analysis.ipynb             Event windows, difference-in-differences,
                                      firm-size robustness
  B2_period_and_measurement.ipynb    Period analysis, the 2015 classification
                                      break, prior earnings volatility
  C1_results.ipynb                   Regenerates reported tables and figures
src/
  paths.py                 Single source of truth for every path used project-wide
  pipeline/                Sequential, reproducible data preparation
    extract_zips.py            Raw zip archive -> flat CSVs
    ingest_cboe.py             CSVs -> yearly Parquet (typed, compressed)
    verify_setup.py            Confirms pipeline output is present and correct
    ingest_ibes.py             IBES export -> typed Parquet
    build_dispersion_events.py Firm-event panel: dispersion and earnings volatility
    build_daily_retail_activity.py  Option-level -> daily, by participant type
    ingest_crsp.py             CRSP daily stock file -> typed Parquet
    build_moneyness.py         Joins spot prices; classifies OTM/ITM/ATM
  analysis/
    event_window_profile.py       Event windows, composition, DiD, dispersion
                                   regressions, era splits, balanced panels
    build_results_tables.py       Regenerates every reported table -> results/
    build_results_figures.py      Regenerates every reported figure -> results/
    verify_results.py             Recomputes 17 headline numbers and compares
                                   them against the values reported
    compare_retail_proxies.py     Exchange classification vs. the small-trade proxy
    analyse_order_size.py         Trade size around the 2015 rule changes
    check_moneyness_coverage.py   Spot-price coverage of the event sample
    check_delisting_exposure.py   Survivorship exposure in the ticker universe
    scope_spot_requirements.py    Sizes the spot-price requirement before fetching
extract_prior_outputs.py       Extracts numeric results from notebook outputs, so
                                two runs can be diffed
requirements.txt
```

## Setup

This repo contains no data. CBOE, IBES and CRSP are licensed commercial
datasets, and the raw CBOE archive alone is several GB well over GitHub's
file size limits. All three are available via supervisor access (CBOE and
IBES) and WRDS (CRSP).

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Place the CBOE daily zip files in `CBOE_Data_2011_2022/`
3. Place the IBES Summary History export in `data/ibes_quarterly_report/`
4. Place the CRSP Daily Stock File export in `data/crsp/`

CRSP was requested from WRDS as **Annual Update → Stock Version 2 (CIZ) →
Stock Daily Security Data**, covering 2010-01-01 to 2022-05-16, with
identifiers (PERMNO, Ticker, CUSIP), prices (DlyPrc, DlyPrcFlg, DlyFacPrc,
DlyClose, DlyBid, DlyAsk), capitalisation (DlyCap, ShrOut), volume (DlyVol),
and delisting fields (DelActionType, DelStatusType, DelReasonType).

## Running everything

Run `notebooks/A1_data_pipeline_setup.ipynb` top to bottom from a clean
kernel. It executes every pipeline step in dependency order, skips any stage
whose output already exists, and ends by verifying the headline results.

Then run B1, B2 and C1, each from a clean kernel.

Each pipeline step can also be run directly from the repository root:

```
python src/pipeline/extract_zips.py
python src/pipeline/ingest_cboe.py
python src/pipeline/verify_setup.py
python src/pipeline/ingest_ibes.py
python src/pipeline/build_dispersion_events.py
python src/pipeline/build_daily_retail_activity.py
python src/pipeline/ingest_crsp.py
python src/pipeline/build_moneyness.py
```

**Changing a pipeline script requires re-running its step.** The analysis
notebooks read Parquet files, not code, editing a script has no effect until
the corresponding rebuild is executed, and each step skips itself when its
output is already present. Delete the relevant output directory first.

## Reproducing the results

```
python src/analysis/verify_results.py
python src/analysis/build_results_tables.py
python src/analysis/build_results_figures.py
```

`verify_results.py` recomputes seventeen headline numbers from the current
data and compares them against the values reported in the thesis. Every line
should read MATCH; a mismatch means a result was computed against a
superseded intermediate file. It runs automatically as the last step of A1.

Tables and figures land in `results/` and `results/figures/`. Nothing reported
in the thesis is copied from notebook output.

Both builders default to the 2016-onward primary sample (see below). Pass
`date_from=None` to reproduce pooled estimates.

## Method notes

**Retail identification.** Retail activity is isolated using CBOE's
participant classification (`cust_` columns) rather than contract-size and
single-leg proxies. Professional customers (`procust_`) serve as the control
group: same market role, differing in sophistication and capital rather than
market function.

**Uncertainty measures.** Two, and they behave differently. Analyst forecast
dispersion is `STDEV / max(|MEANEST|, 0.05)`, following Diether, Malloy &
Scherbina (2002); raw standard deviation is not comparable across firms with
different EPS scales. Prior earnings volatility is the rolling standard
deviation of past scaled earnings surprises over eight quarters, strictly
backward-looking. The current event's own surprise is excluded, since it is
not knowable before the announcement. Surprises are winsorised at the 1st and
99th percentiles *before* the rolling standard deviation, because a small
number of firms carry scale-corrupted MEANEST values that pass the ceiling
filter but break a window spanning two incompatible scales.

**Event window.** The near-event window (−1 to +4 trading days) was chosen
empirically, from a volume profile across 97,270 matched firm-events:
activity is flat until day −2, peaks at day 0 and +1 at roughly 2.5x
baseline, and returns to baseline by day +5. Windows are counted in trading
days, aligned to each ticker's own calendar. No two-week pre-announcement
buildup appears in any year of the sample.

**Moneyness.** Classified from log moneyness `ln(strike / spot)` at the
contract level using CRSP daily prices, with a ±2% at-the-money band. About
9% of CBOE option-rows have no CRSP price; these are almost entirely index
and volatility products (^SPX, ^VIX, ^RUT and similar), which have no
earnings announcements and were never in the event sample. Within the event
sample, unpriced volume is 0.2%.

**Inference.** Difference-in-differences with a participant-group ×
event-window interaction, estimated by OLS. The interaction coefficient tests
whether retail's shift differs from professional customers', not merely
whether retail shifts, a distinction that matters since several results are
driven by professionals moving rather than retail. Standard errors are
clustered by firm-event and, as a robustness check, by ticker.

**Firm size.** Market capitalisation enters as `log_mktcap × treat × post`
rather than as a level control, so size is allowed its own event response.
Most apparent dispersion event-effects do not survive it, and size is the
dominant coefficient in every specification where both enter.

**Balanced panel.** A panel row exists only where that group traded in that
period. The near-event window spans ~6 trading days against ~55 for baseline,
and professional customers trade sparsely in small stocks, so `procust` rows
drop out of the near-event window at a rate falling monotonically with firm
size (54% in the smallest size quartile, 14% in the largest). Estimates are
reported on both the full and balanced panels. The balanced panel retains
roughly half the observations and skews toward larger firms; neither sample
is definitive.

**Functional form.** Both uncertainty measures are right-skewed, and both
produce effects confined to their top quartile. Quartile dummies are reported
alongside linear specifications, because a linear coefficient averages a null
across three quartiles with an effect in the fourth.

**Primary sample: 2016 onward.** CBOE's professional-customer classification
exhibits a discontinuity in 2015. Coverage across ticker-days halves
permanently (0.244 → 0.149) and never recovers; contamination of small-trade
volume falls by two-thirds; trade size within the category rises sharply; and
the event response reverses the following year. No other participant category
absorbs the lost volume, and market-maker share, defined by exchange role
rather than order counts is stable throughout, ruling out a market-wide
reporting change. The pattern is consistent with a narrowing of the
population qualifying as Professional under CBOE's order-counting rules,
which were the subject of industry harmonisation through 2015
(SR-CBOE-2015-011, mandatory from 1 June 2015) and a revised counting
methodology in early 2016 (SR-CBOE-2016-005). This study documents the
discontinuity but cannot establish the link directly. Because `procust` is
the control group, comparisons spanning that boundary contrast
differently-composed populations, so results are reported from 2016 onward
with pooled estimates in `table6_period_comparison`.

**Known limitations.** Roughly 40% of firm-events have no same-day CBOE
options activity, concentrated among smaller and less liquid names. About 36%
of tickers in the price-matched universe stopped trading before the sample
ends; CRSP was chosen over free price sources because it retains delisted
securities. Ten mega-cap tickers behave oppositely to the rest of the
universe on position-size measures and carry enough volume to reverse the
sign of volume-weighted aggregates, so firm-event equal weighting is used
throughout. Dispersion and firm size are strongly negatively correlated, so
the independent effect of dispersion is identified off limited variation.
Only the two customer categories carry contract-size breakdowns, so a full
small-trade proxy across all participant types cannot be reconstructed from
these data. Media attention, named as a control in the research proposal, has
no available data source and is not included.

## Notes

- `CBOE_Data_2011_2022` is spelled with this exact capitalisation on disk.
  Windows will not care if it does not match; Mac and Linux will.
- The CBOE dataset runs from 2011-01-03 through 2022-05-16, not through the
  full 2022 calendar year.
- `extract_prior_outputs.py` captures every number a notebook reports, so two
  runs can be diffed mechanically. `reference_before_clean_run.json` holds the
  values from before the notebooks were restructured.
