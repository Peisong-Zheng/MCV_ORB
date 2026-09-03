# NGRIP GI/GS event-phase analysis

This directory tests whether the published NGRIP Greenland Interstadial (GI)
and Greenland Stadial (GS) onsets in Rasmussen et al. (2014) depend on
precession phase. GI starts are labelled **warming** and GS starts **cooling**.
The analysis uses the published event catalogue; it does not rerun KS on an
NGRIP isotope time series.

## Published event-age data

`data/processed/ngrip_warming_cooling_starts.csv` contains the 69 NGRIP event
starts used in the analysis, taken from Table 2 of Rasmussen et al. (2014): 34
warming (GI) starts and 35 cooling (GS) starts. Lettered subevents are not
treated as independent events. If Table 2 has no separate parent row, the
oldest lettered onset represents the start of the parent interval (for example,
GI-1e supplies the GI-1 start).

Table 2 ages are years before AD 2000 (`a b2k`). The analysis keeps both source
ages and converts to the project's conventional ka BP scale (before AD 1950):

```text
age_ka_BP = age_yr_b2k / 1000 - 0.05
```

## Analysis

Run:

```bash
python NGRIP/ngrip_event_phase_analysis.py
```

Rayleigh analysis samples the reusable orbital-phase series at each event age.
The convention is precession-index minimum = 0 degrees and maximum = 180
degrees. The finite-sample Rayleigh approximation in `toolbox/orbital_phase.py`
is applied to warming starts, cooling starts, and all 69 GI/GS transitions
treated as one combined MCV-event catalogue.

The predictive-information (PI) analysis represents each catalogue as 0.2 ka
Poisson event-count bins over 12--120 ka BP. For the directional catalogues,
the history term counts earlier events of the same direction. For the combined
catalogue, it counts any earlier GI or GS transition. Each catalogue compares
the same two nested models:

```text
reduced = 5 ka prior-event history within that catalogue + LR04 + CO2
full    = reduced + sin(precession phase) + cos(precession phase)
```

The Cheng sampling-resolution term is deliberately omitted: these are
published NGRIP stratigraphic boundaries, not events detected from the Cheng
record. Bins at the oldest edge without a complete 5 ka history are excluded,
leaving 33 warming, 34 cooling, and 67 combined events in the PI fits.

## Results

| Catalogue | Rayleigh N | Mean phase | R-bar | Rayleigh p | PI N | PI peak | LR p | bits/event | max/min rate | full - reduced AICc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Warming (GI) | 34 | 356.5° | 0.210 | 0.224 | 33 | 327.7° | 0.0439 | 0.137 | 4.25 | -2.16 |
| Cooling (GS) | 35 | 354.2° | 0.161 | 0.407 | 34 | 332.0° | 0.0630 | 0.117 | 3.57 | -1.44 |
| All transitions | 69 | 355.5° | 0.185 | 0.0935 | 67 | 341.2° | 0.0295 | 0.0759 | 2.84 | -2.96 |

None of the unconditional Rayleigh tests rejects uniform phase occurrence,
although the combined result is closer to the 0.05 threshold. After
conditioning on catalogue history, LR04, and CO2, all three fitted PI curves
peak shortly before the precession minimum. The combined PI comparison is
nominally significant and supports the interpretation of a shared MCV activity
window more directly than an opposite-phase interpretation of warming and
cooling. It reuses the same GI/GS boundaries, however, and is a pooled
sensitivity test rather than a third independent line of evidence.

Rayleigh and PI answer different questions, so their p values need not agree:
Rayleigh asks whether the raw phases are non-uniform, whereas PI asks whether
phase improves conditional prediction after the selected covariates.

Important limitations:

- only about 34 events are available per direction;
- GI and GS boundaries alternate and are not independent replications;
- PI likelihood-ratio p values are nominal asymptotic diagnostics;
- event-age/definition uncertainties are not propagated in this first version;
- converting b2k to BP aligns the zero year but does not reconcile the GICC05,
  LR04, CO2, and orbital age models;
- three catalogue formulations and two statistical approaches create
  multiple-testing risk; the combined catalogue is not independent of the two
  directional catalogues.

All six fitted PI models converged, likelihood nesting held, and no numerical
linear-predictor clipping occurred.

## Outputs

Analysis tables are in `data/processed/ngrip_event_phase_analysis/`.

Figures are in `figures/ngrip_event_phase_analysis/`:

1. `fig01_event_timeline_and_precession`: event ages and the sampled precession
   series;
2. `fig02_rayleigh_precession_phase`: three phase histograms, mean vectors, and
   the p=0.05 Rayleigh threshold;
3. `fig03_predictive_information_phase_response`: three conditional fitted
   phase multipliers and the main PI diagnostics.
