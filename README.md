# MCV_ORB

Research workspace for testing whether millennial-scale climate variability
(MCV) depends on orbital precession phase. The repository contains three
related lines of evidence:

1. published NGRIP GI/GS boundaries from Rasmussen et al. (2014);
2. the synthetic Greenland D-O warming catalogues of Barker et al. (2011); and
3. a provisional MIS 6 speleothem warming chronology assembled from
   Fohlmeister et al. (2023), Burns et al. (2019), and Held et al. (2024).

The MIS 6 workflow is deliberately event-based. Multiple proxy observations
of one transition improve its identification or age estimate; they do not
increase the number of independent climate events.

## Scientific question

The working question is:

> Does precession phase add predictive information about MCV transition rates
> after event history and slow climate state are included?

The phase convention is shared by all analyses:

- precession-index minimum = 0°;
- precession-index maximum = 180°;
- roughly 300–360° is the approach to a precession minimum and broadly
  corresponds to high Northern Hemisphere summer insolation.

Two statistics answer different questions:

- **Rayleigh** tests unconditional non-uniformity of event phases.
- **Conditional PI** compares nested binned Poisson event-rate models. The
  reduced model contains the catalogue-specific 5 Kyr event-history term,
  LR04, and CO2; the full model adds sine and cosine of precession phase.

NGRIP and MIS 6 do not use a proxy-resolution covariate. NGRIP boundaries are
high-resolution published stratigraphic ages, and resolution was deliberately
excluded from the simplified MIS 6 test. The Barker model retains local EDC
sampling resolution because its event catalogue was detected from a synthetic
record whose resolution varies markedly through time.

## Current results

### NGRIP

The retained catalogue contains 34 GI starts and 35 GS starts. Lettered source
events are collapsed to one parent onset.

| Catalogue | Rayleigh N | Rayleigh phase | Rayleigh p | PI N | PI peak | LR p | bits/event | max/min rate | ΔAICc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GI warming | 34 | 356.5° | 0.224 | 33 | 327.7° | 0.0439 | 0.137 | 4.25 | -2.16 |
| GS cooling | 35 | 354.2° | 0.407 | 34 | 332.0° | 0.0630 | 0.117 | 3.57 | -1.44 |
| All transitions | 69 | 355.5° | 0.0935 | 67 | 341.2° | 0.0295 | 0.0759 | 2.84 | -2.96 |

The pooled catalogue reuses the same GI/GS boundaries and is not a third
independent dataset.

### Barker et al. (2011)

The three primary variants are overlapping representations of Supplementary
Table S3, not independent replications.

| Variant | Rayleigh N | Rayleigh p | PI N | PI peak | LR p | bits/event | max/min rate | ΔAICc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EDC3, 0–640 Kyr | 104 | 0.149 | 103 | 328.6° | 0.0182 | 0.0562 | 2.29 | -4.00 |
| EDC3, 0–800 Kyr | 126 | 0.212 | 125 | 330.9° | 0.0325 | 0.0395 | 1.99 | -2.84 |
| SpeleoAge, 0–400 Kyr | 70 | 0.0795 | 69 | 333.4° | 0.0203 | 0.0814 | 2.86 | -3.76 |

Fixed-threshold catalogues and alternative history windows remain sensitivity
tests. The SpeleoAge chronology is partly tied to Chinese speleothems and
therefore has an additional circularity risk.

### Provisional MIS 6 chronology

The current catalogue contains 21 warming events from 132.5 to 196.5 Kyr BP:
16 MF events labelled by Fohlmeister et al. (2023), followed by selected
Huagapo and Sofular events labelled by Held et al. (2024).

| Rayleigh N | Rayleigh phase | Rayleigh p | PI N | PI peak | LR p | bits/event | max/min rate | ΔAICc |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 21 | 320.6° | 0.206 | 20 | 318.3° | 0.0326 | 0.247 | 7.74 | -2.69 |

The A+ age experiment combines a record-level chronology shift with one of
nine detector settings and rejects proposals that reverse event order. Across
10,000 accepted sequences, the median PI is 0.237 bits/event; 67.47% retain a
nominal LRT p < 0.05 and 96.73% retain ΔAICc < 0. These fractions describe
sensitivity to the adopted age model and are not new Monte Carlo p values.

The MIS 6 chronology remains provisional: its event-definition distribution
is a parameter-sensitivity proxy, Sofular uses a simple local age-error
approximation, and cross-record synchronization error is intentionally absent.

## Project layout

```text
MCV_ORB/
├── README.md
├── Speleothem_published_event_plot.py
├── MIS6_composite_event_record.py
├── MIS6_event_age_uncertainty.py
├── MIS6_event_phase_analysis.py
├── MIS6_event_age_PI_sensitivity.py
├── Barker2011_do_predictive_information_audited.py
├── Barker2011_do_predictive_information.py   # compatibility entry point
├── NGRIP/
├── data/{raw,processed}/
├── figures/
├── experiment_note/
├── toolbox/
└── tests/
```

Scripts remain at the project root so each experiment is easy to open and
read. Repeated numerical operations live in `toolbox/`; paper-specific event
selection and figure composition stay in their research scripts.

## Canonical inputs

- `NGRIP/data/processed/ngrip_warming_cooling_starts.csv`: retained 69-event
  Rasmussen catalogue. The one-off Table 2 extraction code is not required.
- `data/raw/speleothem_data.xlsx`: four published speleothem proxy records.
- `data/processed/Speleothem_published_event_plot/speleothem_mis6_published_event_label_anchors.csv`:
  manually interpreted literature-label anchors; these are label positions,
  not measured transition ages.
- `data/processed/MIS6_event_age_uncertainty/mis6_age_control_points.csv`:
  fixed U-Th control-point transcription with cave, study, source filename,
  and source-table location. Analysis scripts read this file but never
  recreate or overwrite it.
- `data/raw/Fohlmeister J et al-2023-data-mf_d18o_stack.txt`: published MF
  stack and age-envelope input.
- `data/raw/Barker et al-2011-SOM.xls`: Barker Supplementary Table S3.
- `data/raw/Jouzel-etal-2007-Science-Orbital and Millennial Antarctic Climate
  Variability over the Past 800,000 Years.txt`: EDC resolution source.
- `data/raw/lr04.xlsx`, `composite_co2.xlsx`, and
  `pre_1000_60_inter100.txt`: common model covariates.

Raw inputs and manually transcribed canonical tables should not be overwritten
by analysis scripts.

## Reproduction

Recommended environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the MIS 6 workflow in this order:

```bash
python Speleothem_published_event_plot.py
python MIS6_composite_event_record.py
python MIS6_event_age_uncertainty.py
python MIS6_event_phase_analysis.py
python MIS6_event_age_PI_sensitivity.py
```

The fixed control-point table must already be present. The composite script no
longer depends on running the uncertainty script first.

Run the comparison analyses with:

```bash
python NGRIP/ngrip_event_phase_analysis.py
python Barker2011_do_predictive_information_audited.py
```

The old Barker filename remains a thin compatibility entry point and invokes
the audited analysis; it no longer contains or writes a second implementation.

Run the complete regression suite with:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider
```

## Retained outputs

Each analysis keeps a small reader-facing result set. Large deterministic
per-bin tables and repeated orbital extrema are rebuilt in memory rather than
stored under `data/processed/`.

MIS 6 keeps:

- the 21-event composite table;
- the fixed U-Th control table;
- a compact event-age uncertainty summary;
- all 10,000 accepted event-age sequences used by the PI sensitivity test;
- compact point-age phase/PI results and per-realization PI metrics.

NGRIP keeps its canonical event table, compact analysis summary, one physical-
event phase table, model coefficients, and model provenance. Barker keeps the
catalogue used, primary summary, likelihood comparisons, coefficients, and the
two principal sensitivity tables.

Figures are saved as PNG for quick inspection and PDF for publication work.

## Interpretation safeguards

1. Do not count Barker age-scale variants as independent datasets.
2. Do not count NGRIP warming, cooling, and their union as three independent
   confirmations.
3. Do not count multiple speleothem observations of one MIS 6 transition as
   separate physical events.
4. Do not interpret an optimizer's preferred phase when the phase comparison
   lacks support.
5. Keep unconditional Rayleigh clustering distinct from conditional PI gain.
6. Treat likelihood-ratio p values as nominal asymptotic diagnostics.
7. Do not tune event ages, detector settings, or phase definitions using the
   same phase alignment later presented as evidence.
8. The present results support a broad sector near precession minimum; they do
   not establish precession as an independent causal mechanism.

## References

- Barker, S. et al. (2011), *Science* 334, 347–351.
  https://doi.org/10.1126/science.1203580
- Rasmussen, S. O. et al. (2014), *Quaternary Science Reviews* 106, 14–28.
  https://doi.org/10.1016/j.quascirev.2014.09.007
- Fohlmeister, J. et al. (2023), *Communications Earth & Environment* 4, 245.
  https://doi.org/10.1038/s43247-023-00908-0
- Held, F. et al. (2024), *Nature Communications* 15, 1183.
  https://doi.org/10.1038/s41467-024-45507-5
