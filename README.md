# CBOE Options Analysis — Honours Thesis

Analyst forecast dispersion (IBES) and its relationship to retail options
trading activity (CBOE Open/Close), 2011 – May 2022 (base sample) with an
extension to 2022 – 2026 for post-COVID comparison.

Research questions:

1. To what extent does earnings-related uncertainty — measured through
   analyst forecast dispersion and prior earnings volatility — predict retail
   investor trading volume in equity options?
2. How does earnings uncertainty influence the type of options contracts
   retail investors select?

The design compares retail customers against professional customers as a
control group, so that shifts around earnings announcements can be attributed
to retail specifically rather than to the information event itself. Retail is
identified using CBOE's own participant classification rather than the
trade-size proxies used in most of this literature.

Three supplementary tests (B4) ask whether retail investors take the right
side when trading options around earnings: whether their pre-announcement
option flow predicts the announcement return (Test 1), what the options they
trade earn across the announcement before and after trading costs (Test 2),
and how much of the price change comes from direction, the move's size, the
implied-volatility crush and time decay (Test 3).

University of Sydney, Discipline of Finance.
Supervised by Professor Andrew Grant and Professor P. Joakim Westerholm.

## Extend-alongside design

The project maintains **two parallel datasets**:

- **Base sample** (2011-01-03 to 2022-05-16). The thesis's primary results
  are written on this sample. Files live at their original names
  (`cboe_parquet/`, `crsp_daily.parquet`, `dispersion_events.parquet`, and so
  on).
- **Extended sample** (2022-08-01 onward). Built to support the post-COVID
  era comparison, physically separated at every layer under a `*_ext`
  naming convention (`cboe_parquet_ext/`, `crsp_daily_ext.parquet`,
  `dispersion_events_ext.parquet`, etc.).

The analysis layer reads either sample via a single switch in
`event_window_profile`:

```python
from analysis.event_window_profile import set_sample
set_sample("ext")   # every loader now reads *_ext
set_sample("base")  # back to the base sample
```

No analysis code is duplicated between samples. The switch propagates through
DiD panels, market cap, moneyness, CRSP frames, the proxy and order-size
modules, and the results tables/figures. `verify_results` remains base-only
because its checks are calibrated on the base numbers.

The extended sample is capped at **2025-12-31 for anything requiring CRSP**
(moneyness, market cap) and at **~Feb 2026 for events** (IBES actuals). CBOE
options data itself runs to May 2026 but is unusable beyond those ceilings.
There is an ~11-week source-data gap between the base end (2022-05-16) and
the extension start (2022-08-01) — a genuine gap in the CBOE export.

## Repository layout

```
CBOE_Data_2011_2026/     Raw CBOE per-day zip files, full history (not included)
CBOE_Data_2022_2026_new/ New-period zips isolated for staged extraction
data/                    All generated and licensed data (gitignored)
  CBOE_DATA_RAW_EXTRACTED/          base extraction (extract_zips.py)
  CBOE_DATA_RAW_EXTRACTED_EXT/      extension extraction
  cboe_parquet/                     base ingested (ingest_cboe.py)
  cboe_parquet_ext/                 extension ingested
  cboe_daily_retail/                base (build_daily_retail_activity.py)
  cboe_daily_retail_ext/            extension
  cboe_daily_moneyness/             base (build_moneyness.py)
  cboe_daily_moneyness_ext/         extension
  cboe_daily_flow/                  base signed option flow (build_daily_flow.py)
  cboe_daily_flow_ext/              extension
  ibes_quarterly_report/            IBES export + dispersion_events(_ext).parquet
  crsp/                             CRSP export + crsp_daily(_ext).parquet
    ext_source/                     new CRSP CSV isolated for staged ingestion
    returns/                        CRSP daily returns, CRSP market index, and the
                                    constructed market return (build_market_return.py)
  optionmetrics/                    OptionMetrics extract (pull_optionmetrics.py) and
                                    the contract-level option panel (Tests 2-3)
  external/oews/                    BLS OEWS industry file and derived STEM shares (B5)
results/                 Generated tables (CSV) and figures (PNG)
workbooks/
  A1_data_pipeline_setup.ipynb       Builds base sample, in dependency order
  A2_extended_build.ipynb            Builds extension alongside base (*_ext files)
  A3_option_data_build.ipynb         Builds inputs for the option-market tests
                                     (WRDS pulls, signed flow, market return,
                                     OptionMetrics, contract panel); spans both samples
  B1_core_analysis.ipynb             Base: event windows, DiD, firm-size robustness
  B2_period_and_measurement.ipynb    Base: period analysis, 2015 break, earnings vol
  B3_covid_era_analysis.ipynb        Post-COVID era comparison, spans both samples
  B4_option_market_tests.ipynb       Tests 1-3: informed flow, option returns,
                                     Greek decomposition; spans both samples
  B5_sector_analysis.ipynb           Knowledge intensity and informed retail flow
                                     by industry; hypotheses stated before results
  B6_information_environment.ipynb   Dispersion, coverage and informed retail flow;
                                     hypotheses stated before results
  C1_results.ipynb                   Regenerates reported tables and figures (base)
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
                               (takes crsp_path for base-vs-ext routing)
    build_daily_flow.py        Signed daily flow: call/put x open/close x buy/sell
    build_market_return.py     Constructed value-weighted market return, validated
                               against CRSP vwretd
    pull_optionmetrics.py      OptionMetrics IvyDB US extract via the WRDS library
    build_knowledge_intensity.py  Industry STEM employment shares from BLS OEWS
  analysis/
    event_window_profile.py       Event windows, DiD, dispersion regressions,
                                   era splits, balanced panels, sample switch
    build_results_tables.py       Regenerates every reported table -> results/
    build_results_figures.py      Regenerates every reported figure -> results/
    verify_results.py             Recomputes 17 headline numbers (base only)
    compare_retail_proxies.py     Exchange classification vs. small-trade proxy
    analyse_order_size.py         Trade size around the 2015 rule changes
    covid_era_comparison.py       Three-era DiD, size-controlled, era levels
    covid_era_figures.py          Era coefficient and levels figures
    informed_trading.py           Test 1: does pre-announcement flow predict CAR/SUE?
    option_returns.py             Test 2: option holding returns, with trading costs
    option_decomposition.py       Test 3: delta/gamma/vega/theta decomposition
    option_market_results.py      Writes Tests 1-3 tables (7-11) and figures -> results/
    sector_analysis.py            Sectors, knowledge intensity, and the B5 hypothesis
                                  tests; writes tables 12-13 -> results/
    information_environment.py    Flow x dispersion / coverage / earnings-volatility
                                  tests (B6); writes table 14 -> results/
    check_moneyness_coverage.py   Spot-price coverage of the event sample
    check_delisting_exposure.py   Survivorship exposure in the ticker universe
    scope_spot_requirements.py    Sizes the spot-price requirement before fetching
extract_prior_outputs.py       Extracts numeric results from notebook outputs, so
                                two runs can be diffed
reference_before_clean_run.json Recorded results from before the notebooks were
                                restructured
requirements.txt
```

## Setup

This repo contains no data. CBOE, IBES and CRSP are licensed commercial
datasets, and the raw CBOE archive alone is several GB — well over GitHub's
file size limits. CBOE and IBES are available via supervisor access; CRSP
and OptionMetrics IvyDB US via WRDS.

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Place the CBOE daily zip files in `CBOE_Data_2011_2026/` (the folder holds
   both the base and extension zips)
3. Place the IBES Summary History export in `data/ibes_quarterly_report/`
   (one file covers 2011–2026)
4. Place the base CRSP Daily Stock File export in `data/crsp/`; for the
   extension, place the 2022–2025 pull in `data/crsp/ext_source/`
5. For the option-market tests only: a WRDS account with access to CRSP and
   OptionMetrics IvyDB US, and the `wrds` package (`pip install wrds`). A3
   pulls the CRSP returns, the CRSP market index and the OptionMetrics extract
   directly, so no manual download is needed.
6. For the sector analysis only: the BLS OEWS national industry-specific
   estimates (a public download from https://www.bls.gov/oes/tables.htm, May
   2024 or later), with the zip extracted into `data/external/oews/`. The
   national 4-digit, 3-digit and sector files are used; the rest is ignored.

CRSP was requested from WRDS as **Annual Update → Stock Version 2 (CIZ) →
Stock Daily Security Data**, with identifiers (PERMNO, Ticker, CUSIP),
prices (DlyPrc, DlyPrcFlg, DlyFacPrc, DlyClose, DlyBid, DlyAsk),
capitalisation (DlyCap, ShrOut), volume (DlyVol), and delisting fields
(DelActionType, DelStatusType, DelReasonType). Base pull covers 2010-01-01
to 2022-07-31; extension pull covers 2022-01-03 to 2025-12-31.

## Running everything

**Base sample:** run `notebooks/A1_data_pipeline_setup.ipynb` top to bottom
from a clean kernel. It executes every pipeline step in dependency order,
skips any stage whose output already exists, and ends by verifying the
headline results. Then run B1, B2 and C1, each from a clean kernel.

**Extended sample (post-COVID comparison):** after A1, run
`A2_extended_build.ipynb` from a clean kernel — it writes every `*_ext`
counterpart without touching any base file. Then run B3 for the era analysis.

**Option-market tests (Tests 1–3):** after A1 and A2, run
`A3_option_data_build.ipynb`. It prompts for a WRDS login, and the
OptionMetrics pull takes roughly two hours but saves every chunk as it
arrives, so an interrupted run resumes where it stopped. Then run B4.

**Changing a pipeline script requires re-running its step.** The analysis
notebooks read Parquet files, not code — editing a script has no effect until
the corresponding rebuild is executed, and each step skips itself when its
output is already present. Delete the relevant output directory first.

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
python src/pipeline/build_daily_flow.py
python src/pipeline/build_market_return.py
```

`pull_optionmetrics.py` needs a WRDS connection and is run from A3.

## Working with the two samples

**For updated base figures**, do not re-run A1 (the data hasn't changed).
Restart the kernel and re-run C1 (or the relevant cells in B1/B2). Data-build
notebooks (A1, A2) run once; analysis notebooks (B1/B2/B3/C1) are re-run
freely.

**For the sample switch**, note it is stateful within a kernel session — a
loader called after `set_sample("ext")` will read `*_ext` until the switch
is flipped back. B3's functions manage the switch internally and restore to
base afterwards, so a clean top-to-bottom run of any notebook is always safe.
The safe habit for mixed sessions is: restart the kernel before running a
notebook top to bottom, so the switch starts at its `base` default.

The option-market modules (`informed_trading`, `option_returns`,
`option_decomposition`) do not use the switch: they read the base and
extended files together and label each event with its era.

## Reproducing the results

```
python src/analysis/verify_results.py
python src/analysis/build_results_tables.py
python src/analysis/build_results_figures.py
python src/analysis/option_market_results.py
```

`verify_results.py` recomputes seventeen headline numbers from the current
base data and compares them against the values reported in the thesis. Every
line should read MATCH; a mismatch means a result was computed against a
superseded intermediate file. It runs automatically as the last step of A1.
It does not currently have an extended equivalent — the base-sample numbers
are the calibration.

Tables and figures land in `results/` and `results/figures/`. Nothing reported
in the thesis is copied from notebook output. `option_market_results.py`
writes tables 7–11 and the five option-market figures for Tests 1–3; it needs
the A3 outputs and is also run as the last step of B4.

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
backward-looking — the current event's own surprise is excluded, since it is
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
sample, unpriced volume is 0.2%. On the extended sample, 2026 options are
unpriced in full because CRSP ends 2025-12-31 — this is the design ceiling,
not a data error.

**Inference.** Difference-in-differences with a participant-group ×
event-window interaction, estimated by OLS. The interaction coefficient tests
whether retail's shift differs from professional customers', not merely
whether retail shifts — a distinction that matters, since several results are
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
absorbs the lost volume, and market-maker share — defined by exchange role
rather than order counts — is stable throughout, ruling out a market-wide
reporting change. The pattern is consistent with a narrowing of the
population qualifying as Professional under CBOE's order-counting rules,
which were the subject of industry harmonisation through 2015
(SR-CBOE-2015-011, mandatory from 1 June 2015) and a revised counting
methodology in early 2016 (SR-CBOE-2016-005). This study documents the
discontinuity but cannot establish the link directly. Because `procust` is
the control group, comparisons spanning that boundary contrast
differently-composed populations, so results are reported from 2016 onward
with pooled estimates in `table6_period_comparison`.

**Post-COVID era comparison.** B3 compares retail behaviour across three
eras: pre-COVID (2016–2019, base), COVID (2020–2021, base), and post-COVID
(2022-08-01 – 2025-12-31, extended). Eras live in different physical datasets
and are compared as separate DiD estimates, never pooled — a regression
spanning both a classification break and a pandemic would not be
interpretable. `covid_era_comparison` handles the sample switching internally
and restores state afterwards. Pre-COVID starts at 2016 to keep the control
group comparable across all three eras (avoiding the 2015 break). Because
pre-COVID and COVID look alike on most outcomes, the observed change is
best framed as a *post-boom* period rather than a *COVID effect* — too much
changed at once (zero-commission maturity, stimulus, meme-stock episode) to
attribute the shift to any single cause.

**Option-market tests (B4).** Day 0 is the *effective* announcement day:
announcements at or after 16:00 ET move it to the next trading day. IBES
times are not zero-padded (`7:00:00` alongside `16:05:00`), so they are
parsed numerically; compared as text, every before-open announcement would
be misread as after-close. *Test 1* regresses CAR[0,+1] on net put flow
(buys minus sells, scaled by the group's option volume) over days −2 to −1,
with firm-and-date clustered standard errors and year-quarter fixed effects.
Returns are market-adjusted with a constructed value-weighted market return,
because CRSP's legacy index ends at 2024-12-31; the constructed series
matches `vwretd` over 2010–2024 with a correlation of 0.9998 and an average
gap of 1.5 bp a day. Events whose flow window touches the 2022 CBOE gap are
dropped rather than read as zero flow. *Test 2* prices every contract traded
on day −1 at OptionMetrics bid-ask midpoints on days −1 and +1, matched by
expiry, strike (stored ×1000 by OptionMetrics), call/put and secid; contracts
expiring inside the window are valued at their payoff. Trading costs are
applied as a fraction of the quoted half-spread on each trade. *Test 3*
decomposes the midpoint price change with day −1 Greeks, in IvyDB units
(vega per 1.00 change in implied volatility, theta per year); the
decomposition fits with R² 0.974.

**Sector analysis (B5).** The primary hypothesis, stated before any sector
result was seen, is that retail option flow is more informative about
earnings announcements in knowledge-intensive industries, whose employees and
others with industry expertise understand announcements better. Knowledge
intensity is each industry's STEM employment share from the BLS OEWS
national industry-specific estimates (2022 NAICS), matched to firms at the
4-digit NAICS level where available, with the main 2017-to-2022 NAICS changes
translated first. The test is a flow × knowledge-intensity interaction on
announcement returns, estimated alongside a flow × attention interaction,
because media attention brings in uninformed traders who dilute any
informed signal (the secondary hypothesis). Variants use low-attention
events, flow relative to each firm's own previous events, and short-dated
out-of-the-money opening buys, and exclude pharma and biotech; a tail test
asks whether the most extreme flow calls the direction of the move more
often in knowledge-intensive industries. A retained hypothesis is that
earnings are a minor event for pharma and biotech, whose largest news is
regulatory. The data identify informed trading, not insider trading. An
earlier set of three hypotheses was committed first and then revised, still
before any result was seen; the git history records both.

**Information environment (B6).** Hypotheses stated before any result was
seen: retail option flow is more informative about announcements when
analyst dispersion is high (the primary test, linear and top-quartile), when
analyst coverage is thin, and when prior earnings volatility is high. Because
dispersion is strongly correlated with firm size, every specification
includes a flow × size interaction as well as flow × attention; without it,
a dispersion interaction could simply reflect small firms.

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
no direct data source; abnormal pre-announcement trading volume serves as a
proxy. For the option-market tests: OptionMetrics coverage ends at
2025-08-29, so Tests 2 and 3 end there while Test 1 runs to December 2025.
The Greek decomposition covers about 69% of retail buy volume, excluding
contracts that expired inside the window and contracts without an implied
volatility. Each group's execution prices are not observed, so trading costs
are applied as the same fraction of the quoted spread to both groups, which
probably overstates professionals' costs. CBOE's participant data are
aggregate, so changes over time cannot separate individual traders learning
from a change in who is trading.

## Notes

- `CBOE_Data_2011_2026` is spelled with this exact capitalisation on disk.
  Windows will not care if it does not match; Mac and Linux will.
- The base CBOE dataset runs from 2011-01-03 through 2022-05-16; the
  extension runs from 2022-08-01 through 2026-05-29, with the ~11-week gap
  a genuine feature of the source exports.
- The OptionMetrics extract is an annual update ending 2025-08-29. After a
  later update, raise `date_to` in `run_optionmetrics_pull` and delete the
  `opprcd_<year>` and `secprd_<year>` folders for the affected years first,
  since saved chunks are numbered within each year.
- `extract_prior_outputs.py` captures every number a notebook reports, so two
  runs can be diffed mechanically. `reference_before_clean_run.json` holds the
  values from before the notebooks were restructured.
