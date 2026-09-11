# Barker 2011 SpeleoAge analysis

This supporting analysis uses 70 variable-threshold warming picks from Barker
et al. (2011), Supplementary Table S3, on the speleothem-tuned SpeleoAge axis
between 0 and 400 kyr. It estimates the conditional association between event
rate and precession phase using the same main-model settings as NGRIP–MIS6.
The Barker events are analyzed separately and are not added to the pooled
NGRIP–MIS6 catalogue.

Source: Barker et al. (2011), *Science* 334, 347–351,
[doi:10.1126/science.1203580](https://doi.org/10.1126/science.1203580).
The active code was derived from the archived audited main script. The complete
three-catalogue workflow and its old figures remain in
[`../archive/Barker2011_2026-09-05/`](../archive/Barker2011_2026-09-05/README.md).
The active run has no dependency on that archive.

## Run

From the project root, in the project Python environment:

```bash
python Barker2011/Barker2011_event_phase_analysis.py
python Barker2011/Barker2011_PI_bootstrap.py
python Barker2011/Barker2011_event_age_uncertainty.py
python Barker2011/Barker2011_event_uncertainty_sensitivity.py
OPENBLAS_NUM_THREADS=1 python Barker2011/Barker2011_orbital_driver_sensitivity.py
OPENBLAS_NUM_THREADS=1 python Barker2011/Barker2011_climate_phase_interaction.py
```

Alternatively, enter `Barker2011/` and run `python Barker2011_event_phase_analysis.py`.
Paths resolve relative to the script directory. Each script replaces its own
tables and figure. The uncertainty sampler must run before PI sensitivity;
`Barker2011_event_uncertainty_sensitivity.py` fits the point-age reference
directly and does not require saved main-analysis outputs. The orbital-driver
script additionally reads the saved main summary for its reference check.

## Inputs

- `data/raw/Barker et al-2011-SOM.xls`: unchanged source workbook, restored
  from the archive. Read `Sheet1` with Excel row 9 as the header. Select
  `DO pick variable threshold == 1`, using `SpeloAge (kyr).1` as the event age.
  `source_row` retains the original zero-based data-row index;
  `source_excel_row = source_row + 10` identifies the visible Excel row.
- `../data/raw/lr04.xlsx`: shared LR04 climate covariate.
- `../data/raw/composite_co2.xlsx`: shared atmospheric CO2 covariate.
- `../data/raw/pre_1000_60_inter100.txt`: shared La2004 precession index.
- The orbital-driver sensitivity additionally reads shared `obl_1000_60_inter100.txt`,
  `ecc_1000_60_inter100.txt` and `solstice_insolation_NH.nc` from `../data/raw/`.

## Other orbital drivers

`Barker2011_orbital_driver_sensitivity.py` fits eight models on the unchanged
0–398.5-kyr response support: B, B+P, and B+X and B+P+X for each of
eccentricity, obliquity and 65°N summer-solstice daily mean insolation. B is
the history/LR04/CO2 baseline, P is the sine/cosine phase block, and each X
adds one linear coefficient. No lag or phase-amplitude interaction is fitted.

The script selects 500 saved combined-age chronologies without replacement
(seed 20260909). Every realization is rebinned and its history recalculated;
all eight models share support and fixed predictor scales. All 500 draws are
supported and fitted successfully. Run the main and age-sampler scripts first;
this experiment reads the saved main summary to verify its reference model.
Single-thread BLAS improves speed for these small fits.

At point ages, insolation adds 0.05213 bits/event after B (nominal p=0.02450,
nine-test Holm p=0.17148), and only 0.00108 after B+P. Phase adds 0.03830
after B+insolation (nominal p=0.15590, Holm p=0.93543), compared with
0.08936 after B alone. No new comparison survives the nine-test correction.
The phase/insolation overlap does not establish either predictor as the
unique physical driver. New p values are nominal; age ranges are chronology
sensitivity rather than event-process sampling confidence intervals.

Results and diagnostics are in `data/processed/Barker2011_orbital_driver_sensitivity/`.
The [three-panel figure](figures/Barker2011_orbital_driver_sensitivity/Barker2011_orbital_driver_sensitivity.pdf)
matches the pooled layout. [Methods/results](experiment_note/Barker2011_orbital_driver_sensitivity_Methods_and_results.txt)
and the [caption](experiment_note/Barker2011_orbital_driver_sensitivity_Caption.txt)
are in `experiment_note/`. The shared [input audit](../docs/orbital_driver_inputs.md)
documents units, native resolution, file hashes and the insolation epoch check.

Run `python -m toolbox.orbital_driver_reporting` from the project root to
refresh both orbital figures and captions using saved fits. This does not
resample chronologies or refit models. The compact figures use BG, Pre and
Orb notation; the captions define the full baseline, phase terms, row
variables and age ranges. Rendering hashes are stored separately in
`figure_provenance.csv`.

## Main-input interpretation

The EDC3 event ages, age-mapping columns and Jouzel resolution record are not
used by this main analysis. The published `DO pick` column supplies 59
fixed-threshold picks on the same SpeleoAge axis for a point-age sensitivity.
Input hashes are saved in `parameters_and_provenance.csv`.

SpeleoAge numerical ages are retained and treated as BP1950. Direct verification
of this particular column's reference year remains unresolved in the existing
[epoch audit](../docs/age_epoch_audit.md); no event-age shift is inferred.
The shared orbital tools apply the documented −0.05-kyr J2000-to-BP1950
conversion once. The nominal main script does not perturb event ages.

The two uncertainty entry points additionally use `data/raw/Barker2011_TableS1.csv`,
a transcription of all six numerical columns in the 60-row Table S1. The source
is preserved as `references/Barker2011_SOM.pdf` (printed p. 24, PDF p. 25).
EDC3 age and age-shift columns are source metadata only. The sampler selects
SpeleoAge and the published combined uncertainty; no EDC3 coordinate or error
is introduced. The PDF transcription is checked against the source in the tests.

## Model alignment

| Setting | Barker main analysis | NGRIP–MIS6 main analysis |
|---|---|---|
| Event history | Older bin centers in `(t, t + 1.5 kyr]`; current bin excluded | Same |
| Nominal bin width | 0.2 kyr | Same |
| Origin fraction | 0 | Same |
| Response rule | `maximal_for_history` | Same |
| Climate scaling | Center and divide by range over the response bins | Same method, over its own response bins |
| Reduced predictors | History count, LR04, CO2 | Same, plus MIS6 segment contrast |
| Full predictors | Reduced predictors plus phase sine and cosine | Same phase addition |
| Resolution covariate | Excluded | Excluded |
| Poisson offset | Log of each bin's actual duration | Same |
| Phase definition | Precession minimum = 0°, maximum = 180° | Same |
| LRT | Nominal chi-square, 2 added parameters | Same |

Defaults are read from `toolbox/combined_pi.py`. Its segment-grid and history
functions are reused without changing the pooled implementation. Barker has
one continuous observation segment, so a between-record intercept contrast
is unnecessary.

The 0–400-kyr observation interval gives a 0–398.5-kyr response interval and
398.5 kyr of exposure. The grid is split exactly at 398.5 kyr, preserving a
0.1-kyr final response bin rather than rounding the boundary. There are 2001
observation bins and 1993 response bins. All 70 events fall in the response
interval. The 398.5–400-kyr interval is available for history only.

## Workflow and outputs

`load_barker_source` selects the events → `prepare_bins` builds the history and
climate predictors → `fit_models` fits the reduced/full models → `run_analysis`
adds descriptive phases and checks → `plot_results` and `write_outputs` save
the products.

The following nine CSV files are written under
`data/processed/Barker2011_event_phase_analysis/`:

| File | Contents |
|---|---|
| `analysis_summary.csv` | One-row point-age result and key settings |
| `event_catalogue_used.csv` | 70 events, source rows and response membership |
| `event_precession_phases.csv` | Precession value and phase for each event |
| `binned_inputs_and_fitted_rates.csv` | All bins, history, predictors, response mask and fitted rates |
| `model_summary.csv` | Reduced/full likelihoods, information criteria and numerical checks |
| `predictive_likelihood_tests.csv` | The phase-addition likelihood comparison |
| `predictive_coefficients.csv` | Reduced/full coefficients and unit rate ratios |
| `predictor_scaling.csv` | Response-range LR04/CO2 centering and scaling values |
| `parameters_and_provenance.csv` | Model settings, epoch assumptions, source paths and hashes |

The main figure is saved as `Barker2011_event_phase_analysis.png` (600 dpi) and
`.pdf` (vector) under `figures/Barker2011_event_phase_analysis/`. It uses the
same 180 × 134 mm layout as the pooled figure: timeline above, descriptive
Rayleigh polar panel below left, conditional phase response below right.
The [caption](experiment_note/Barker2011_event_phase_analysis_Caption.txt)
explains the symbols and panel interpretation.

The main figure overlays both definitions with consistent colors and symbols:
variable threshold (70 events, filled rose circles and solid response curve)
and fixed threshold (59 events, open green squares and dashed response curve).
The polar panel overlays sector counts. The primary nine tables remain
primary-only, so existing age and orbital analyses retain their inputs.
`event_definition_sensitivity.csv` adds a two-row comparison, and the
`fixed_threshold/` output subdirectory contains the alternative's nine tables.
Default calls to `load_barker_source()` and `run_analysis()` remain primary-only.
The script entry point additionally runs the fixed-threshold comparison.

Fixed threshold gives PI = 0.104229 bits/event, nominal p = 0.014087,
preferred phase = 350.331° and maximum/minimum rate ratio = 3.10373.
All 59 picks are also present in the primary catalogue. Conditional phase
responses are similar, while descriptive Rayleigh p changes from 0.07949 to
0.04704. These dependent definitions do not constitute independent replication.
Fixed threshold has no separate bootstrap, chronology MC or interaction run.
[Methods/results](experiment_note/Barker2011_event_phase_analysis_Methods_and_results.txt)
document the comparison.

## Current point-age result

With the aligned settings, PI is 0.0893605 bits/event; LR = 8.67159, nominal
p = 0.0130914, and ΔAICc (full minus reduced) = −4.64942. The preferred phase
is 336.907° and the fitted maximum/minimum phase rate ratio is 2.88475.
The descriptive Rayleigh result uses the same 70 events: mean phase 0.830°,
mean resultant length 0.1900 and p = 0.0794888.

These are conditional point-age associations. PI is an in-sample likelihood
gain. `Barker2011_PI_bootstrap.py` separately calibrates the primary point-age
phase test; nominal p values remain explicitly identified in the main figure.
The synthetic Greenland record uses speleothem tuning; restoring it as a
supporting analysis does not establish chronology-independent replication.

## Reduced-model bootstrap

`Barker2011_PI_bootstrap.py` simulates 9,999 catalogues from the fitted reduced
model (history, LR04 and CO2) and refits reduced/full models to each one.
Simulation runs oldest to youngest across the complete 0–400-kyr observation
interval, including the history buffer. Simulated history is updated after
each bin, and bin exposure and forcing scales match the main analysis.
The event total is allowed to vary. The test statistic is LR = twice the
log-likelihood gain; the empirical p uses the plus-one rule.

This experiment calibrates the variable-threshold point-age test only. The
reported binomial interval measures finite-bootstrap precision, not uncertainty
in the phase effect or chronology. It does not nest the chronology MC or test
the fixed-threshold catalogue. Outputs are in `data/processed/Barker2011_PI_bootstrap/`
and `figures/Barker2011_PI_bootstrap/`, with [methods/results](experiment_note/Barker2011_PI_bootstrap_Methods_and_results.txt)
and a [caption](experiment_note/Barker2011_PI_bootstrap_Caption.txt).

The formal run (seed 20260909) gives bootstrap p = **0.0138**, with 137 of
9,999 null LR statistics at least as large as the observed 8.67159. The
95% interval for finite-simulation precision is 0.01152–0.01618. No draws
required numerical replacement. Simulated response counts average 69.57
(central 95% range 55–84), compared with 70 observed events. This agrees
closely with the nominal p = 0.01309.

## Background-climate modulation of precession

`Barker2011_climate_phase_interaction.py` compares the current full model with
that model plus LR04 × precession sine and LR04 × precession cosine. The
two-parameter addition tests whether the phase response changes with background
climate, distinct from fitting a nonlinear background rate. Only LR04 is tested;
no climate-threshold search is performed.

The script uses the same 500 saved age realizations selected for orbital-driver
sensitivity, and fixed exposure-weighted LR04 quartiles for displaying phase
responses. The interaction p is nominal and is not calibrated by the separate
reduced-model bootstrap, whose null hypothesis differs. Chronology ranges do
not provide complete sampling confidence intervals. Outputs and manuscript
notes use the `Barker2011_climate_phase_interaction` experiment name.

The point-age interaction adds 0.056125 bits/event (LR = 5.44641, nominal
p = 0.06566; Delta AICc = −1.41613). All 500 selected chronologies are valid;
the added-PI median is 0.050147, with a 95% age range of 0.027810–0.078006.
The fitted phase response varies across LR04 backgrounds, but the comparison
does not establish background modulation. These age ranges exclude full
event-process sampling uncertainty; a positive range is not a significance test.
See [methods/results](experiment_note/Barker2011_climate_phase_interaction_Methods_and_results.txt)
and the [caption](experiment_note/Barker2011_climate_phase_interaction_Caption.txt).

## Chronology Monte Carlo

`Barker2011_event_age_uncertainty.py` uses each Table S1 combined error as the
half-width of a uniform control-offset proposal. The printed uncertainty's
confidence level is unspecified, so it is not converted to Gaussian sigma.
Controls are independent before whole-map rejection; accepted control ages
must be strictly increasing. This conditioning changes their distributions.
This is a different model from NGRIP's cumulative Gaussian variance increments.

The original SpeleoAge values determine fixed interpolation weights. Every
event inherits the weighted offsets of its two bracketing controls. The
264.24–317.70-ka control gap is bridged linearly, including the seven events in
the approximately 265–315-ka published alignment gap (eight events lie in the
full bracketing interval). Four events older than the 378.15-ka last published
control use an auxiliary control at 400 ka. Its uniform half-width is the larger
of the last-four-control linear extrapolation (1.114818 kyr) and their maximum
published uncertainty, giving 1.30 kyr. The auxiliary offset is drawn independently.
No extra Table S3 event-picking error is added to the combined dating/matching error.

With seed 20260908, 42,764 proposals give 10,037 ordered control maps, of which
the first 10,000 are retained (23.4707% proposal acceptance). All 70 event IDs and
their rank are preserved. Samples are neither sorted nor clipped. The complete
event and control ensembles are saved at double-precision round-trip accuracy.
Order conditioning can shift the accepted mean away from zero: the largest
absolute control-mean offset in this run is approximately −0.558 kyr at the
189.98-ka control. Accepted means, SDs and quantiles are retained in the tables;
the plotted median is calculated from accepted draws rather than forced to zero.

The sampler writes six tables under `data/processed/Barker2011_event_age_uncertainty/`:

| File | Contents |
|---|---|
| `age_control_points.csv` | 60 published and one auxiliary control, adopted errors and accepted-offset statistics |
| `event_catalogue_used.csv` | Stable Table S3 row IDs, nominal ages, bracketing controls, weights and gap/tail flags |
| `control_age_realizations.csv` | 10,000 × 61 accepted control ages, with realization IDs |
| `event_age_realizations.csv` | 10,000 × 70 event ages, with the same realization IDs |
| `event_age_uncertainty_summary.csv` | Interpolated half-widths, pre-rejection SDs and accepted MC age summaries |
| `parameters_and_provenance.csv` | Seed, proposal accounting, assumptions and input/code hashes |

`source_excel_row` identifies the original event in Table S3. The stable ID
`Barker_S3_010`, for example, means Excel row 10, not a published GI event label.
Age columns use `age_ka_bp__<event_id>`. Both the original ages and original
interpolation weights remain fixed across realizations.

The two-panel chronology figure shows the proposal half-widths and accepted
offset curves, with alignment-gap and endpoint shading. It is saved in
`figures/Barker2011_event_age_uncertainty/` as PNG (600 dpi) and vector PDF.

## PI sensitivity to chronology

`Barker2011_event_uncertainty_sensitivity.py` reads the saved event sequences and
uses the main analysis's model terms, grid and fixed forcing values/scaling.
Counts and the 1.5-kyr history predictor are recomputed for every chronology.
The fixed 398.5-kyr response exposure includes all 70 events in all 10,000 saved
realizations. Every reduced/full fit converged, with no likelihood-nesting
failure or linear-predictor clipping.

| Quantity | Point ages | MC median | 2.5th–97.5th percentiles |
|---|---:|---:|---:|
| PI (bits/event) | 0.0893605 | 0.0776334 | 0.0534954–0.106099 |
| Nominal LRT p | 0.0130914 | 0.0231260 | 0.00581141–0.0746001 |
| Preferred phase (°) | 336.907 | 340.463 | 331.274–351.668 |
| Phase rate ratio (maximum/minimum) | 2.88475 | 2.65692 | 2.23378–3.17675 |
| Delta AICc (full minus reduced) | −4.64942 | −3.51142 | −6.27369 to −1.16905 |

Nominal p < 0.05 holds for 8,966/10,000 realizations (89.66%), and Delta AICc < 0
for 9,989/10,000 (99.89%). These are robustness fractions, not empirical p-values.
Intervals describe the specified chronology sensitivity model, not confidence
intervals from event-sampling variability. Phase quantiles are unwrapped about
the point-age preferred phase before summarizing.

The sensitivity script writes six tables under
`data/processed/Barker2011_event_uncertainty_sensitivity/`:
`pi_realizations.csv`, `summary.csv`, `point_age_result.csv`,
`phase_response_summary.csv`, `predictor_scaling.csv`, and
`parameters_and_provenance.csv`. The PI table retains realization IDs, both
likelihoods, phase coefficients, fit diagnostics and response event counts.
Unsupported age draws would remain explicitly marked invalid; this run has none.

The six-panel figure in `figures/Barker2011_event_uncertainty_sensitivity/`
uses the same distribution panels as the pooled sensitivity analysis and adds
a phase-response curve with a pointwise MC band. Both PNG and vector PDF are saved.

Manuscript material is in `experiment_note/`:

- [Methods and results](experiment_note/Barker2011_event_uncertainty_Methods_and_results.txt),
  regenerated by the sensitivity script from its actual output;
- [Chronology figure caption](experiment_note/Barker2011_event_age_uncertainty_Caption.txt);
- [PI sensitivity figure caption](experiment_note/Barker2011_event_uncertainty_sensitivity_Caption.txt).

The interpolation model does not supply extra within-gap age variability or
test alternative event matching. The auxiliary endpoint preserves the recent
published error magnitude but is not an independently dated sample. No forcing
chronology error, event-membership uncertainty or event-sampling bootstrap is
included in this analysis.

Run the focused checks from the project root:

```bash
python -m pytest Barker2011/tests -q
```
