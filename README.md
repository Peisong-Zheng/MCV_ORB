# MCV_ORB

Minimal workspace for studying whether millennial-scale climate variability
(MCV) depends on orbital precession phase. This workspace was separated from
the much larger `DO_warming` research archive on 2026-09-03.

## Read this first

The retained evidence base contains only:

1. published NGRIP Greenland Interstadial/Greenland Stadial boundaries from
   Rasmussen et al. (2014); and
2. the Barker et al. (2011) synthetic Greenland D-O warming catalogues.

Rousseau/Cheng KS catalogues, Chinese speleothem KS sensitivity experiments,
CH4 experiments, the old manuscript and historical archives are deliberately
absent. They were not forgotten: detector-aware experiments showed that a
strong slow background can interact with a slope-sensitive KS detector and
produce apparent orbital-phase dependence. The new project should therefore
start from published event chronologies rather than from those KS detections.

The present result is promising but not a causal demonstration. NGRIP is a
short direct record; Barker is long but synthetic. Their conditional PI phase
directions agree, but their limitations are complementary rather than absent.

## Scientific question and phase convention

The working question is:

> Does precession phase add predictive information about the occurrence rate
> of MCV transitions after event history and slow climate state are included?

The shared phase convention is:

- precession-index minimum = 0 degrees;
- precession-index maximum = 180 degrees;
- approximately 300--360 degrees is the approach to a precession minimum and
  broadly corresponds to high Northern Hemisphere summer insolation.

The code reports two different tests:

- **Rayleigh** tests unconditional non-uniformity of event phases;
- **conditional PI** compares nested 0.2 ka binned Poisson event-rate models.
  The reduced model contains the catalogue-specific 5 ka event-history term,
  LR04 and CO2; the full model adds sine and cosine of precession phase.

Rayleigh and PI answer different questions. Their p values are not expected to
match automatically.

## Current results

### NGRIP

The retained Rasmussen Table 2 catalogue contains 34 GI starts (warming) and
35 GS starts (cooling). Lettered source labels and parent-selection notes are
preserved in that catalogue, but subevents are not counted independently.
Treating all 69 GI/GS starts as one MCV catalogue is a pooled view of the same
boundaries, not a third independent dataset.

| Catalogue | Rayleigh N | Rayleigh phase | Rayleigh p | PI N | PI peak | LR p | bits/event | max/min rate | Delta AICc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GI warming | 34 | 356.5 deg | 0.224 | 33 | 327.7 deg | 0.0439 | 0.137 | 4.25 | -2.16 |
| GS cooling | 35 | 354.2 deg | 0.407 | 34 | 332.0 deg | 0.0630 | 0.117 | 3.57 | -1.44 |
| All transitions | 69 | 355.5 deg | 0.0935 | 67 | 341.2 deg | 0.0295 | 0.0759 | 2.84 | -2.96 |

None of the Rayleigh tests is significant at 0.05. Warming and the pooled
catalogue show nominal conditional PI support; cooling points in the same
direction but is not nominally significant. The shared direction is more
consistent with a broad MCV activity sector than with opposite warming/cooling
precession phases.

### Barker et al. (2011)

The primary variable-threshold catalogues are overlapping representations of
the same Supplementary Table S3 events. Do not treat them as independent
replications.

| Variant | Rayleigh N | Rayleigh p | PI N | PI peak | LR p | bits/event | max/min rate | Delta AICc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EDC3, 0--640 ka | 104 | 0.149 | 103 | 328.6 deg | 0.0182 | 0.0562 | 2.29 | -4.00 |
| EDC3, 0--800 ka | 126 | 0.212 | 125 | 330.9 deg | 0.0325 | 0.0395 | 1.99 | -2.84 |
| SpeleoAge, 0--400 ka | 70 | 0.0795 | 69 | 333.4 deg | 0.0203 | 0.0814 | 2.86 | -3.76 |

Again, unconditional Rayleigh tests do not reject uniformity, while the
conditional PI comparison points consistently toward approximately 329--333
degrees. Event-definition sensitivity is important: fixed-threshold EDC3
catalogues give weaker nominal PI p values of 0.072 and 0.141; the fixed
SpeleoAge result gives p = 0.043. At a 20 ka event-history window the two EDC3
variable-threshold results weaken to p = 0.058 and 0.098.

### Joint interpretation

The defensible statement is:

> NGRIP and Barker conditional event-rate models independently favour a broad
> sector near precession minimum / high Northern Hemisphere summer insolation.

Do not strengthen that sentence to “precession is an established independent
cause.” The current likelihood-ratio p values are nominal and asymptotic; event
age uncertainty, age-model mismatch and the full model-selection history are
not yet propagated. Barker is a synthetic Greenland reconstruction, and its
SpeleoAge version is tied to Chinese speleothem chronology.

## Directory structure

```text
MCV_ORB/
├── README.md
├── requirements.txt
├── Barker2011_do_predictive_information.py
├── Barker2011_do_predictive_information_audited.py
├── NGRIP/
│   ├── README.md
│   ├── ngrip_event_phase_analysis.py
│   ├── data/processed/
│   └── figures/
├── data/
│   ├── raw/
│   └── processed/
├── figures/
├── toolbox/
├── tests/
├── docs/plans/
└── references/
```

The source-level `toolbox/` is complete; caches are excluded. Generated Barker
tables and figures are retained in both legacy and audited directories. NGRIP
keeps the collapsed event catalogue used by the analysis, its result tables,
and all three figures in PNG and PDF formats.

## Canonical files

- `Barker2011_do_predictive_information_audited.py`: preferred Barker analysis.
- `Barker2011_do_predictive_information.py`: earlier version retained only for
  direct comparison with work completed before the audit.
- `NGRIP/data/processed/ngrip_warming_cooling_starts.csv`: canonical retained
  NGRIP event input; the one-off extraction code and intermediate table are not
  required by the analysis.
- `NGRIP/ngrip_event_phase_analysis.py`: warming, cooling and pooled NGRIP
  Rayleigh/PI analysis.
- `toolbox/orbital_phase.py`: phase construction, sampling, Rayleigh statistics
  and reusable phase plots.
- `toolbox/event_inputs.py`: LR04, CO2 and orbital interpolation plus binned
  event inputs.
- `toolbox/event_process.py`: shared event-history terms and complete-history
  exposure selection.
- `toolbox/poisson.py`: Poisson likelihood and model fitting.
- `toolbox/model_stats.py`: likelihood, AIC/AICc/BIC and information summaries.

## Input data inventory

Root `data/raw/`:

- `Barker et al-2011-SOM.xls`: Barker Supplementary Tables, including event
  catalogue variants;
- `Jouzel-etal-2007-Science-Orbital and Millennial Antarctic Climate Variability over the Past 800,000 Years.txt`:
  EDC series used to construct a local resolution control;
- `Rasmussen2014_GI_GS_starts_no_subevents_wide.xlsx`: independent extraction
  cross-check;
- `lr04.xlsx`: benthic-isotope climate-state covariate;
- `composite_co2.xlsx`: atmospheric CO2 climate-state covariate;
- `pre_1000_60_inter100.txt`: precession-index input;
- `obl_1000_60_inter100.txt`: obliquity input retained for the legacy Barker
  orbital summary.

NGRIP retained event input:

- `NGRIP/data/processed/ngrip_warming_cooling_starts.csv`: 69 published GI/GS
  parent starts with source labels, b2k ages, converted BP ages, and selection
  provenance.

Reference article:

- `references/Barker2011_Science_800kyr_abrupt_climate_variability.pdf`.

Raw inputs are immutable. New transformations must be written to a named
`data/processed/<analysis>/` directory.

## Reproduction

Recommended environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Analyze NGRIP from the retained processed catalogue:

```bash
python NGRIP/ngrip_event_phase_analysis.py
```

Run the preferred Barker analysis:

```bash
python Barker2011_do_predictive_information_audited.py
```

Run focused regression tests:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider \
  tests/test_ngrip_event_phase_analysis.py \
  tests/test_barker2011_predictive_information_audited.py
```

The expected result is 10 passing tests. A Barker warning about linear
extrapolation of 25 early 0.2 ka SpeleoAge bins is known and recorded; it should
not be silently suppressed.

The audited Barker script remains preferred because it exposes assumptions and
sensitivity results more clearly than the legacy comparison script.

## Output guide

NGRIP:

- `NGRIP/data/processed/ngrip_warming_cooling_starts.csv`: canonical 69-event
  parent-start catalogue;
- `NGRIP/data/processed/ngrip_event_phase_analysis/analysis_summary.csv`:
  compact results shown above;
- `NGRIP/figures/ngrip_event_phase_analysis/`: timeline, Rayleigh and PI plots.

Barker audited:

- `data/processed/Barker2011_do_predictive_information_audited/barker2011_analysis_summary.csv`:
  compact primary results;
- `barker2011_event_definition_sensitivity.csv`: variable/fixed threshold
  comparison;
- `barker2011_history_window_sensitivity.csv`: 2, 5, 10 and 20 ka histories;
- `figures/Barker2011_do_predictive_information_audited/fig02_barker_predictive_likelihood_tests.*`:
  Rayleigh and conditional phase-response summary;
- `fig03_barker_inputs_and_fitted_rates.*`: despite its inherited filename,
  this is primarily a catalogue/age-scale audit figure.

## Non-negotiable interpretation rules for future agents

1. Never count the Barker age-scale variants as independent datasets.
2. Never count NGRIP warming, cooling and their union as three independent
   confirmations.
3. Do not interpret an optimizer's preferred angle when the phase comparison
   lacks support.
4. Do not call a nominal asymptotic p value a calibrated causal test.
5. Preserve the distinction between unconditional Rayleigh clustering and
   conditional PI gain.
6. Do not reintroduce Rousseau/Cheng KS events as primary evidence without a
   detector-aware null that the observed result actually exceeds.
7. Do not tune an event detector, chronology shift or phase definition to make
   events align with precession and then use the same alignment as evidence.
8. Keep one row per physical event. Multiple proxy observations of the same
   event improve dating and validation, not the number of independent events.

## Recommended next project: MIS 6 consensus warming chronology

The next data-level advance should use Held et al. (2024), Fohlmeister et al.
(2023), Sofular, Melchsee-Frutt, Huagapo and Sanbao records to create an
**event-level consensus chronology**, not a cross-proxy amplitude stack.

A minimal design is:

1. define every abrupt warming by the same estimand, preferably maximum warming
   slope (`t_mid`);
2. identify candidates independently on each published U-Th age model without
   viewing orbital phase;
3. store age uncertainty, transition-picking uncertainty, local resolution,
   data coverage and clear/ambiguous/missing status;
4. match events by temporal order and uncertainty using neutral event IDs;
5. create a high-confidence catalogue requiring support from at least two
   independent, adequately resolved U-Th archives;
6. use one consensus time distribution per physical warming in PI;
7. propagate age uncertainty and use an exposure mask for coverage gaps;
8. perform per-record and leave-one-record-out checks as robustness tests;
9. use Barker only as a morphology/numbering guide in MIS 6 because its
   SpeleoAge chronology is linked to Sanbao;
10. prefer a confirmatory fixed direction near 330--340 degrees learned from
    NGRIP or Barker outside MIS 6; report a free phase fit as exploratory.

This can improve event reality and timing precision, but cannot turn repeated
measurements of the same 10--20 events into a larger physical sample or add
precession cycles.

The Held/Fohlmeister files have not been copied into this minimal workspace.
Their locations and the reasoning that led to the consensus design are recorded
in the old `DO_warming/README.md`; add them only when the MIS 6 extraction work
begins.

## References

- Barker, S. et al. (2011). *Science* 334, 347--351.
  https://doi.org/10.1126/science.1203580
- Rasmussen, S. O. et al. (2014). *Quaternary Science Reviews* 106, 14--28.
  https://doi.org/10.1016/j.quascirev.2014.09.007
- Fohlmeister, J. et al. (2023). *Communications Earth & Environment* 4, 245.
  https://doi.org/10.1038/s43247-023-00908-0
- Held, F. et al. (2024). *Nature Communications* 15, 1183.
  https://doi.org/10.1038/s41467-024-45507-5
