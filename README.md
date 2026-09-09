# MCV_ORB

Paper-stage analysis of whether orbital precession phase adds predictive
information about Dansgaard--Oeschger (D--O) warming-transition rates. The
primary evidence is a pooled catalogue of 55 published transitions drawn from
two high-precision, non-overlapping time intervals:

- 34 NGRIP Greenland Interstadial starts from Rasmussen et al. (2014); and
- 21 MIS 6 speleothem transitions assembled from the MF record of Fohlmeister
  et al. (2023) and the Sofular record of Held et al. (2024).

All event identities come from published studies. This project places the
published MIS 6 events on their source proxy records but does not define new
climate events. A separate [Barker 2011 SpeleoAge supporting analysis](Barker2011/README.md)
uses matched main-model settings and remains outside the pooled catalogue.
The original three-catalogue workflow is preserved in the archive.

## Primary event catalogue

The pooled catalogue is stored in
`data/curated/ngrip_mis6_warming_events.csv`. Its two intervals remain separate
throughout event counting, history construction, and model fitting.

| Segment | Events | Published source | Observation interval | Main response interval |
|---|---:|---|---:|---:|
| NGRIP | 34 | Rasmussen et al. (2014) | 12--123 kyr BP | 12--121.5 kyr BP |
| MIS 6 | 21 | Fohlmeister et al. (2023); Held et al. (2024) | 132.5--204.5 kyr BP | 132.5--203 kyr BP |

The response intervals provide 180 kyr of total exposure. The 123--132.5 kyr
BP gap is unobserved: it is neither assumed to be event-free nor represented
by model bins. Event history is reset between segments. The final 1.5 kyr of
each observation interval supplies complete older-event history but is not
response exposure.

The 123 kyr NGRIP endpoint follows the approximately 123-kyr continuous,
undisturbed climate-record coverage reported by North Greenland Ice Core
Project Members (2004). It denotes the available NGRIP record support;
Rasmussen et al.'s (2014) event framework separately extends to approximately
120 ka b2k.

The 204.5 kyr MIS 6 endpoint lies within the approximately 205-kyr continuous
Sofular record and source-series coverage of Held et al. (2024); the source
series reaches 204.784 kyr BP, and their Figure 3b has no additional published
event label beyond 6.13 in the older direction. This older interval supplies
history support without adding an unpublished event.

The MIS 6 sequence contains MF 6.1--6.16 followed by Sofular 6.9--6.13,
renumbered MIS 6.17--6.21 after concatenation. MF 6.16 corresponds to an
unnumbered Sofular feature preceding Sofular 6.9, so Sofular 6.9 is retained as
the next distinct transition. Sofular was selected for the older extension
because it is high resolution and matches MF well over their overlap, while
limiting the number of independently dated cave records in the composite.

## Statistical design

Precession-index minimum is phase 0 degrees and maximum is 180 degrees. Source
orbital ages are converted from J2000 to kyr BP relative to AD 1950 before
phase interpolation.

Rayleigh's test is an unconditional, descriptive phase-concentration check.
The primary analysis is conditional predictive information (PI), calculated
from two nested binned Poisson models:

```text
reduced = 1.5 kyr prior-event history + LR04 + CO2 + MIS 6 segment indicator
full    = reduced + sin(precession phase) + cos(precession phase)
```

The 1.5-kyr history window represents short-term event dependence based on the
characteristic D--O recurrence timescale reported in previous work. The main
bin width is 0.2 kyr and the bin origin is unshifted. The segment indicator
allows MIS 6 and NGRIP to have different conditional baseline event rates; it
is not an age-offset or synchronization correction. Sampling resolution is
not included as a PI covariate.

PI is the nested log-likelihood gain divided by the number of response events
and converted to bits per event. The asymptotic likelihood-ratio p value is
reported as a nominal diagnostic. The reduced-model parametric bootstrap is
the empirical calibration of the likelihood gain.

The target is the warming-onset rate per unit of whole observation time;
exposure is not restricted to cold states. Out-of-sample validation is deferred
given the small event catalogue.

## Main results

### Point ages

All 55 events enter the main response support. Both Poisson fits converged,
likelihood nesting held, and no numerical linear-predictor clipping occurred.

| Analysis | N | Preferred/mean phase | p | PI (bits/event) | Phase max/min rate ratio | full - reduced AICc |
|---|---:|---:|---:|---:|---:|---:|
| Rayleigh, descriptive | 55 | 339.98 degrees | 0.05796 | -- | -- | -- |
| Conditional PI | 55 | 330.00 degrees | 0.001074 nominal | 0.17933 | 4.94 | -9.61 |

The Rayleigh result is not significant at 0.05. This does not contradict PI:
Rayleigh tests raw phase uniformity, whereas PI tests whether phase improves
prediction after event history, slow climate state, and segment baseline are
included.

### Joint event-age uncertainty

`NGRIP_MIS6_event_uncertainty_sensitivity.py` independently pairs 10,000
NGRIP and MIS 6 combined-error realizations, using each source row once.
NGRIP uses the full annual GICC05 MCE curve and a cumulative Gaussian age
process on 5-kyr knots. The modeled extension adopts a conservative nominal
±4.5%-of-age envelope (approximately 2σ), continuously connected to the counted
endpoint. This uniform ratio is a study assumption, not a pointwise uncertainty
curve published by Moseley et al. (2020).

Sofular now uses raw dated depths and local U–Th errors transferred through
published component age–depth curves. So-4 supplies the main five-event error
model; So-57 supplies an overlap sensitivity. Stack-to-component depth mapping
is an assumed age-coordinate projection, not a known physical event depth or
reconstructed iscam posterior. MF retains its previous shared chronology factor.
See [the upgrade report](docs/reviews/chronology-upgrade-2026-09-05.md) for
covariance diagnostics, component and projection-position sensitivity.

| Quantity | Point ages | MC median | MC 2.5--97.5% |
|---|---:|---:|---:|
| Nominal likelihood-ratio p | 0.001074 | 0.002067 | 0.000251--0.03780 |
| PI (bits/event) | 0.17933 | 0.16215 | 0.08616--0.21746 |
| Preferred phase | 330.00 degrees | 330.04 degrees | 315.89--343.66 degrees |
| Phase max/min rate ratio | 4.94 | 4.53 | 2.92--5.96 |
| full - reduced AICc | -9.61 | -8.30 | -12.52 to -2.49 |

All 10,000 age sequences are retained. Eighteen fall outside fixed observation
support and are flagged without a PI fit; 9,982 fits are valid, including 66
with 54 response events and 9,916 with 55. The table reports valid-fit quantiles.
Nominal p is below 0.05 in 9,851/9,982 valid fits (98.69%); 9,971/9,982 have
negative full-minus-reduced AICc. The below-threshold fraction among all draws
has a known lower bound of 98.51%. These are age-uncertainty robustness
proportions, not empirical p values. All valid fits pass numerical checks.
The So-57 overlap alternative gives 98.53%, median PI 0.16150 and phase 330.13°.
A 2,000-draw screening at each NGRIP knot spacing of 2.5, 5 and 10 kyr gives
98.29%--99.00%, PI medians 0.16120--0.16228 and phases 329.97--330.16°.

### Effect precision: sampling and joint uncertainty

`NGRIP_MIS6_effect_uncertainty.py` adds an independent full-model simulation
workflow. It retains the original null-bootstrap functions and outputs.
Group B uses 5,000 point-age process simulations; group C uses 200 uniformly
selected valid chronologies with 50 simulations each. Event history is generated
dynamically and event counts may vary. All 15,000 fits pass numerical checks.

| Effect | Point | A: age working range | B: conditional joint-region projection | C: joint working range |
|---|---:|---:|---:|---:|
| Preferred phase, degrees | 330.00 | 315.89–343.66 | 286.53–374.08 | 295.92–365.45 |
| Phase maximum/minimum rate ratio | 4.94 | 2.92–5.96 | 1.56–15.58 | 2.10–14.19 |

Angles are unwrapped around the point estimate: 374.08° is 14.08° in the next
cycle. B projects an approximate 95% joint coefficient confidence region;
A and C are central 95% working quantiles. C is not a calibrated combined
confidence interval, and need not enclose B. The zero-effect coefficient pair
is outside the B region, but effect strength is imprecisely estimated.
See [the effect-uncertainty report](docs/reviews/effect-uncertainty-2026-09-05.md)
for interval construction and remaining assumptions.

### Analysis-design sensitivity and pooling

The pre-specified design grid uses the fixed 173-kyr common response core so
that changing the model design does not change exposure. It crosses:

- history windows of 1.0, 1.5, 2.0, 3.0, and 5.0 kyr;
- bin widths of 0.1, 0.2, and 0.5 kyr; and
- unshifted and half-bin-shifted origins.

All 30 designs retain nominal p at or below 0.00702 and negative
full-minus-reduced AICc. PI ranges from 0.13009 to 0.18908 bits/event, preferred
phase from 323.82 to 331.39 degrees, and the phase max/min rate ratio from 4.06
to 5.51. The full-minus-reduced AICc range is -10.39 to -5.76. At the frozen
1.5-kyr/0.2-kyr/unshifted design, the common 173-kyr core (865 bins) gives PI =
0.17249 and nominal p = 0.001394; the main 180-kyr support (901 bins) gives PI =
0.17933 and nominal p = 0.001074.

A segment-by-phase interaction does not improve the pooled model
(`p = 0.8075`, segment-specific-minus-common AICc = 3.65). The fitted
segment-specific phases are 331.20 degrees for NGRIP and 327.43 degrees for MIS
6, with phase max/min rate ratios of 4.07 and 6.95, respectively. This provides
no evidence that the two intervals require different precession-phase
coefficients, while not proving that their responses are identical.

### Climate-shape and event-memory sensitivity

`NGRIP_MIS6_PI_model_sensitivity.py` adds two point-age specification experiments
alongside the existing bin/window design grid. They are post-audit sensitivity
checks; all eight listed variants are reported. The primary model is retained.

On the same 180-kyr, 55-event support, the climate experiment compares linear
LR04/CO₂ with LR04², CO₂², both squares, and both squares plus their interaction.
Each added background term enters both reduced and full models. Across all five
variants, PI is 0.17051–0.17982 bits/event, phase is 322.25–330.00°, and rate ratio
is 4.70–5.01. LR04² lowers full-model AICc by 3.94 relative to the linear full
model; CO₂² alone raises it by 1.91. Nonlinearity shifts the fitted phase by up
to 7.75° while preserving the phase-related likelihood gain.

For event memory, each segment's oldest event bin initializes history but is
excluded from response together with its older exposure. This leaves NGRIP
12–115.2 ka and MIS 6 132.5–194.1 ka: 164.8 kyr, 824 bins and 53 response events.
GI-25 and MIS 6.21 remain in the source catalogue and in subsequent histories.
Elapsed time uses strictly older occupied bin centers, excludes current-bin
events, and resets between segments. All three memory models use this same
support and the original climate scaling:

| History term | PI (bits/event) | Phase | Rate ratio | Nominal p | Full-model ΔAICc vs count |
|---|---:|---:|---:|---:|---:|
| Previous 1.5-kyr event count | 0.19612 | 333.18° | 5.28 | 0.000743 | 0 |
| Time since last event, kyr | 0.18832 | 332.43° | 5.10 | 0.000990 | +0.80 |
| log(1 + time / 1 kyr) | 0.19195 | 332.69° | 5.23 | 0.000866 | +0.47 |

The phase changes by less than 0.8° across the matched memory comparison.
All eight nested fits pass convergence, rank, clipping, coefficient-bound and
likelihood checks. These p values are nominal χ²(2) diagnostics, without a new
bootstrap or chronology ensemble for each alternative. The primary empirical
p = 0.0022 does not calibrate the alternative models. See the
[model-sensitivity report](docs/reviews/model-sensitivity-2026-09-06.md) for the
boundary rationale, full results and reproduction details.

### Other orbital drivers

`NGRIP_MIS6_orbital_driver_sensitivity.py` compares eccentricity, obliquity
and 65°N summer-solstice daily mean insolation with the current phase model.
The matched [Barker experiment](Barker2011/README.md#other-orbital-drivers)
uses the same eight models: B, B+P, and B+X and B+P+X for each scalar X.
B is the existing history/climate baseline and P is the two-term phase block.
Each X is tested after B and after B+P; phase is also tested after B+X.

Each record uses 500 selected saved combined-age chronologies (seed 20260909),
with counts and histories recalculated on fixed response bins. NGRIP–MIS6
has 499 supported draws: one realization moves GI-25 beyond the 123-kyr
observation boundary and is retained as invalid, without replacement. All
Barker draws are supported. Four supported pooled draws have 54 response
events because one event enters the history-only buffer; the remaining 495
have 55. All models within a draw share that response count. Input units and the inferred J2000 epoch of the
old insolation file are documented in the
[orbital input audit](docs/orbital_driver_inputs.md).

At point ages, summer-solstice insolation adds 0.12254 bits/event after B
for NGRIP–MIS6 (nominal p=0.002238; nine-test Holm p=0.01790), but only
0.02191 after B+P. Conversely, phase adds 0.07870 after B+insolation,
compared with 0.17933 after B alone; its nominal p=0.049775 becomes
Holm p=0.29865. No added X is significant after B+P, even before correction.
These results support shared orbital information, not a uniquely identified
phase effect independent of summer-solstice insolation. All new p values
remain nominal rather than event-process-bootstrap calibrated.

The [three-panel figure](figures/NGRIP_MIS6_orbital_driver_sensitivity/NGRIP_MIS6_orbital_driver_sensitivity.pdf)
and [methods/results](experiment_note/NGRIP_MIS6_orbital_driver_sensitivity_Methods_and_results.txt)
report point estimates and age-sensitivity ranges, not full confidence intervals.

The compact figures use BG for the full background baseline, Pre for the
precession-phase terms, and Orb for the row variable. The captions define
these abbreviations, the comparisons, and the age-sensitivity symbols.
To redraw both figures and captions from saved results:

```bash
python -m toolbox.orbital_driver_reporting
```

This command preserves all fitted results. `figure_provenance.csv` records
the rendering inputs separately from the original model-run hashes.

### Reduced-model parametric bootstrap

The formal bootstrap contains 9,999 valid null replicates. Each catalogue is
simulated under the fitted reduced model from oldest to youngest, with event
history updated dynamically and reset between the two observation segments.
Both reduced and full models are refitted to every simulated catalogue.
History before each segment's oldest observed boundary is initialized to zero;
the history-only buffer is simulated under that boundary condition.

- observed likelihood-ratio statistic: 13.6732;
- null replicates at least as large as observed: 21 of 9,999;
- plus-one empirical p: 0.0022;
- exact 95% binomial interval: 0.00130--0.00321; and
- rejected or numerically invalid bootstrap attempts: 0.

The bootstrap event-count distribution has a mean of 54.98 and a 2.5--97.5%
range of 42--69 events, close to the observed count of 55. The empirical result
therefore supports the nominal PI comparison without relying only on the
asymptotic chi-square approximation.

## Barker SpeleoAge supporting analysis

The active [Barker workflow](Barker2011/README.md) uses the 70 variable-threshold
SpeleoAge events over 0–400 kyr. It shares the pooled analysis's 1.5-kyr history,
0.2-kyr bins, climate scaling rule and reduced/full phase comparison. It has
one record, so no segment contrast, and no sampling-resolution covariate.
The complete-history response extends to 398.5 kyr and includes all 70 events.

```bash
python Barker2011/Barker2011_event_phase_analysis.py
python Barker2011/Barker2011_PI_bootstrap.py
python Barker2011/Barker2011_event_age_uncertainty.py
python Barker2011/Barker2011_event_uncertainty_sensitivity.py
OPENBLAS_NUM_THREADS=1 python Barker2011/Barker2011_orbital_driver_sensitivity.py
OPENBLAS_NUM_THREADS=1 python Barker2011/Barker2011_climate_phase_interaction.py
```

The current point-age result is PI = 0.08936 bits/event, nominal p = 0.01309,
preferred phase = 336.91° and maximum/minimum phase rate ratio = 2.885.
The SpeleoAge chronology analysis uses Table S1 combined errors, a 400-ka
auxiliary control and 10,000 ordered realizations. PI is refitted with updated
history for every sequence: median 0.07763 bits/event, 95% MC range
0.05350–0.10610; 89.66% retain nominal p < 0.05. This is a chronology robustness
fraction, not an empirical p-value. Methods, results and figure captions are in
[`Barker2011/experiment_note/`](Barker2011/experiment_note/Barker2011_event_uncertainty_Methods_and_results.txt).
This is a separate supporting analysis; its original three-catalogue code,
sensitivity results, figures and frozen dependencies remain in
[archive/Barker2011_2026-09-05/](archive/Barker2011_2026-09-05/README.md).

The main Barker figure also overlays the 59 published fixed-threshold picks
as a point-age definition sensitivity (PI = 0.10423 bits/event; nominal
p = 0.01409). Main output tables and downstream chronology inputs retain the
70-event primary definition. Only the primary definition receives the separate
9,999-replicate reduced-model bootstrap; fixed threshold has no additional
bootstrap or age MC. The bootstrap p is 0.0138 (137 exceedances, no numerical
replacements; 95% simulation-precision interval 0.01152–0.01618). See the
[Barker workflow](Barker2011/README.md) for outputs.

## Background-climate modulation of precession

The two record-specific scripts compare each current full model with that
model plus LR04 × precession sine and LR04 × precession cosine:

```bash
OPENBLAS_NUM_THREADS=1 python NGRIP_MIS6_climate_phase_interaction.py
OPENBLAS_NUM_THREADS=1 python Barker2011/Barker2011_climate_phase_interaction.py
```

This tests whether the phase response changes with LR04 background, using two
additional parameters and no threshold search. Each experiment reuses its
orbital-driver analysis's 500 selected chronologies. Phase-response curves are
shown at fixed exposure-weighted LR04 quartiles. Nominal interaction p values
are distinct from the primary reduced-model bootstrap; age ranges describe
chronology sensitivity, not complete sampling uncertainty. Outputs and English
manuscript notes follow the respective script names under each record's
`data/processed`, `figures` and `experiment_note` directories. The
[implementation plan](docs/plans/2026-09-09-additional-sensitivity-design.md)
records scope and checks.

Point-age interaction PI is 0.02585 bits/event for NGRIP–MIS6 (nominal
p = 0.37327) and 0.05613 for Barker (p = 0.06566). Valid chronology counts
are 499/500 and 500/500; the unsupported pooled draw is retained in the status
table. Neither comparison establishes background modulation. The
[results review](docs/reviews/additional-sensitivity-2026-09-09.md) brings
together the bootstrap, event-definition and interaction findings.

## Manuscript preparation

The [paper plan](monsoon_paper/paper_plan.md) maps the current evidence to main
and supporting sections, citations, figures and tables. The manuscript
folder follows the previous project's `main.tex`, `SI.tex`, `reference.bib`
and numbered `figures/` structure. Both TeX files now contain complete English
first drafts, with four main figures, fourteen SI figures and five SI table groups.

Run `python paper_figure_export.py` to refresh the mapped PDFs, or
`make -C monsoon_paper pdf` to regenerate manuscript figures/tables from saved
results, synchronize and build both drafts. Selected
research figure exporters also synchronize their PDFs when run. All eighteen
figures are available, including three new main summary compositions.
See the [manuscript README](monsoon_paper/README.md) for the mapping and build
workflow, and the [reference audit](monsoon_paper/reference_audit.md) for
bibliographic sources and verification limits.

## Active project layout

```text
MCV_ORB/
├── README.md
├── NGRIP_MIS6_event_phase_analysis.py
├── NGRIP_MIS6_event_uncertainty_sensitivity.py
├── NGRIP_MIS6_effect_uncertainty.py
├── NGRIP_MIS6_PI_design_sensitivity.py
├── NGRIP_MIS6_PI_model_sensitivity.py
├── NGRIP_MIS6_orbital_driver_sensitivity.py
├── NGRIP_MIS6_climate_phase_interaction.py
├── NGRIP_MIS6_PI_bootstrap.py
├── paper_figure_export.py
├── paper_summary_figures.py
├── paper_tables.py
├── monsoon_paper/
│   ├── paper_plan.md
│   ├── main.tex
│   ├── SI.tex
│   ├── reference.bib
│   ├── figure_manifest.csv
│   └── figures/
├── NGRIP/
│   ├── ngrip_data_preparation.ipynb
│   ├── ngrip_event_age_uncertainty.py
│   └── data/{raw,processed}/
├── MIS6/
│   ├── Speleothem_published_event_plot.ipynb
│   ├── MIS6_composite_event_record.ipynb
│   ├── MIS6_event_age_uncertainty.py
│   ├── event_detection.py
│   ├── sofular_chronology.py
│   ├── data/{raw,curated,processed}/
│   ├── figures/
│   ├── experiment_note/
│   ├── docs/plans/
│   └── tests/
├── Barker2011/
│   ├── Barker2011_event_phase_analysis.py
│   ├── Barker2011_PI_bootstrap.py
│   ├── Barker2011_event_age_uncertainty.py
│   ├── Barker2011_event_uncertainty_sensitivity.py
│   ├── Barker2011_orbital_driver_sensitivity.py
│   ├── Barker2011_climate_phase_interaction.py
│   ├── data/{raw,processed}/
│   ├── references/
│   ├── figures/
│   ├── experiment_note/
│   └── tests/
├── data/
│   ├── raw/
│   ├── curated/
│   └── processed/
├── figures/
├── experiment_note/
├── toolbox/
├── tests/
└── archive/
```

Analysis scripts stay close to the research question. Shared event counting,
history construction, phase interpolation, and Poisson fitting live in
`toolbox/`; `toolbox/combined_pi.py` defines the scientific contract used by
all pooled analyses. Superseded exploratory analyses and their corresponding
outputs are preserved under `archive/` rather than mixed into the active
workflow.

## Canonical inputs

Fixed, reader-facing tables are kept under `data/curated/`:

- `ngrip_mis6_warming_events.csv`: the primary 55-event catalogue, with
  stable IDs, published labels, source records, timing methods, and source
  tables;
- `observation_segments.csv`: observation and common-core limits for the two
  disjoint segments; and
- `age_epoch_audit.csv`: source-specific age-zero evidence and conversions.

The three MIS 6 source tables (published label anchors, `Sofular_MF_visual_peak_match.csv`,
and `mis6_age_control_points.csv`) live under `MIS6/data/curated/`.
The control-point subset supplies figure annotations; full raw dated-depth
tables supply the current chronology uncertainty model.

Additional analysis inputs are:

- `NGRIP/data/raw/Rasmussen2014_GI_GS_starts_no_subevents_wide.xlsx`: the
  retained Table 2 transcription, with definition codes checked against the PDF;
- `NGRIP/data/processed/ngrip_warming_cooling_starts.csv` and
  `ngrip_chronology_grid.csv`: event errors and chronology knots prepared by
  `NGRIP/ngrip_data_preparation.ipynb`;
- `MIS6/data/raw/MIS6_speleothem_records.xlsx`: mixed-source proxy workbook (MF, Sofular, Sanbao and Huagapo);
- `MIS6/data/raw/Held2024/`: complete So-4/So-57 proxy and U–Th tables, source manifest;
- `NGRIP/data/raw/Rasmussen et al-2022-GICC05_time_scale.txt`: annual GICC05 age/MCE
  curve, read by the NGRIP preparation notebook;
- `data/curated/age_epoch_audit.csv`: source-specific age-zero evidence and conversions;
- `MIS6/data/raw/Fohlmeister J et al-2023-data-mf_d18o_stack.txt`: MF stack and
  published age-envelope curves;
- `data/raw/lr04.xlsx`, `composite_co2.xlsx`, and
  `pre_1000_60_inter100.txt`: model covariates and orbital forcing.

Curated tables must not be recreated or overwritten by routine analysis
scripts. Ages ending in `*_kyr_bp` are thousands of years before AD 1950.

## Reproduction order

Recommended environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the primary workflow in this order:

```bash
cd MIS6
jupyter nbconvert --to notebook --execute --inplace Speleothem_published_event_plot.ipynb
jupyter nbconvert --to notebook --execute --inplace MIS6_composite_event_record.ipynb
python MIS6_event_age_uncertainty.py
cd ..
cd NGRIP
jupyter nbconvert --to notebook --execute --inplace ngrip_data_preparation.ipynb
python ngrip_event_age_uncertainty.py
cd ..
python NGRIP_MIS6_event_phase_analysis.py
python NGRIP_MIS6_event_uncertainty_sensitivity.py
python NGRIP_MIS6_effect_uncertainty.py
python NGRIP_MIS6_PI_design_sensitivity.py
python NGRIP_MIS6_PI_model_sensitivity.py
python NGRIP_MIS6_PI_bootstrap.py
OPENBLAS_NUM_THREADS=1 python NGRIP_MIS6_orbital_driver_sensitivity.py
```

The first plotting notebook audits the published label placement and MF--Sofular
matching. The following source workflows construct the speleothem catalogue
and the two uncertainty ensembles. The NGRIP notebook can also be run cell by
cell in the IDE; both it and its sampling script use paths relative to `NGRIP/`.
The remaining scripts run the point
analysis, joint age sensitivity, effect precision, design grid, model
specification sensitivity, formal empirical calibration, and orbital-driver
comparisons. A single BLAS thread avoids threading overhead in the many small
orbital-model fits; this is a performance setting, not a model change.

The 2026-09-07 NGRIP simplification retains only combined-error sampling and
preserves the original 5-kyr model. Its saved 10,000 age realizations are
byte-identical to the previous combined ensemble, so downstream numerical
results remain applicable. See [the NGRIP workflow](NGRIP/README.md) and
[the migration checks](docs/reviews/ngrip-simplification-2026-09-07.md).

Run the regression suite with:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider
```

## Retained outputs

The new `NGRIP_MIS6_orbital_driver_sensitivity/` output directory contains
eight point models, ten comparisons, coefficients, 500-row selected ages,
all MC model/comparison diagnostics, PI and phase summaries, matched binned
inputs, predictor correlations, reference checks, and provenance. Its plot
uses the same layout and axis scale as the separate Barker output directory.

The primary reader-facing results are grouped by analysis name under
`data/processed/` and `figures/`:

- `NGRIP_MIS6_event_phase_analysis/`: point Rayleigh/PI summary, phases,
  coefficients, provenance, and main figure;
- `NGRIP_MIS6_event_uncertainty_sensitivity/`: paired 55-event MC ages,
  realization-level PI metrics, summary, provenance, and figure;
- `NGRIP_MIS6_effect_uncertainty/`: full-model replicates, age-generator IDs,
  effect intervals, joint confidence region, phase-response bands and figure;
- `NGRIP_MIS6_PI_design_sensitivity/`: design grid, support and pooling
  checks, provenance, and figure;
- `NGRIP_MIS6_PI_model_sensitivity/`: climate/history fits and coefficients,
  bin-level predictors and support masks, event roles, initialization boundaries,
  phase curves, provenance and figure; and
- `NGRIP_MIS6_PI_bootstrap/`: formal null replicates, empirical summary,
  provenance, and figure.

NGRIP source uncertainty outputs remain under
`NGRIP/data/processed/ngrip_event_age_uncertainty/` and the matching figure
directory. MIS 6 source catalogues, uncertainty ensembles and figures are
under `MIS6/data/processed/` and `MIS6/figures/`; see the [MIS 6 workflow](MIS6/README.md).
The active Barker SpeleoAge outputs are under `Barker2011/data/processed/`
and `Barker2011/figures/`. Earlier multi-catalogue outputs remain archived.
Most deterministic per-bin tables are rebuilt in memory. The model-sensitivity
output retains one diagnostic table to make the elapsed-time initialization and
two response supports directly inspectable.

## Interpretation safeguards

1. Treat the NGRIP and MIS 6 observations as two disjoint exposure segments,
   not a continuous event-free record.
2. Do not count multiple proxy expressions of one MIS 6 transition as separate
   climate events.
3. Do not treat the Barker age-scale variants as independent datasets.
4. Keep descriptive Rayleigh concentration distinct from conditional PI gain.
5. Call the 9,851/9,982 valid-fit result an age-uncertainty robustness proportion, not
   an empirical p value or the probability that the hypothesis is true.
6. Use the reduced-model parametric bootstrap p = 0.0022 as the empirical
   likelihood-gain calibration for the frozen main design.
7. Do not tune event ages, detector settings, binning, history, or phase
   definitions using the alignment later presented as evidence.
8. Interpret the fitted phase as an association conditional on the included
   model terms, not proof that precession directly triggers D--O events.
9. The NGRIP modelled chronology is not wholly independent of orbital tuning;
   this structural limitation is documented rather than represented as random
   age noise.

## References

- Andersen, K. K. et al. (2006), *Quaternary Science Reviews* 25, 3246--3257.
  https://doi.org/10.1016/j.quascirev.2006.08.002
- Barker, S. et al. (2011), *Science* 334, 347--351.
  https://doi.org/10.1126/science.1203580
- Rasmussen, S. O. et al. (2014), *Quaternary Science Reviews* 106, 14--28.
  https://doi.org/10.1016/j.quascirev.2014.09.007
- Fohlmeister, J. et al. (2023), *Communications Earth & Environment* 4, 245.
  https://doi.org/10.1038/s43247-023-00908-0
- Held, F. et al. (2024), *Nature Communications* 15, 1183.
  https://doi.org/10.1038/s41467-024-45507-5
- Laskar, J. et al. (2004), *Astronomy & Astrophysics* 428, 261--285.
  https://doi.org/10.1051/0004-6361:20041335
- Moseley, G. E. et al. (2020), *Climate of the Past* 16, 29--50.
  https://doi.org/10.5194/cp-16-29-2020
- Myrvoll-Nilsen, E. et al. (2022), *Climate of the Past* 18, 1275--1294.
  https://doi.org/10.5194/cp-18-1275-2022
- North Greenland Ice Core Project Members (2004), *Nature* 431, 147--151.
  https://doi.org/10.1038/nature02805
