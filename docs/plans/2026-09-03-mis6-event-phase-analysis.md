# MIS6 event-phase quick-test implementation plan

**Goal:** Apply the existing NGRIP Rayleigh and conditional predictive-information
(PI) workflow to the 21-event MIS6 composite warming catalogue, without a proxy
sampling-resolution covariate.

**Architecture:** Treat the 21 rows as one warming-event catalogue. Reuse the
shared orbital-phase, event-input, event-history, Poisson, and likelihood helpers
used by NGRIP. Keep the NGRIP bin width, history window, climate covariates, phase
convention, statistical summaries, and output schema so the two analyses remain
directly comparable. Use an explicit 132.5--196.5 Kyr BP quick-test support,
rounded inward from the continuous proxy coverage (about 132.411--196.584 Kyr
BP), and flag all unpropagated assumptions in the provenance output.

**Tech stack:** Python, pandas, NumPy, SciPy, Matplotlib, pytest, and the existing
`toolbox` modules.

## Step 1: Lock the input and model contract

- Input: `data/processed/MIS6_composite_event_record/mis6_composite_event_record.csv`.
- Require exactly 21 unique `composite_event_id` values and 21 finite, unique
  `event_age_ka_bp` ages within 132.5--196.5 Kyr BP.
- Assign every row to one shared `warming` catalogue for both analyses.
- Use precession-index minimum = 0 degrees and maximum = 180 degrees.
- Use all 21 events, equally weighted, in the Rayleigh test.
- Use 0.2 Kyr count bins and a 5 Kyr older-event history in PI. As in NGRIP,
  bins without a complete older-history window are omitted.
- Define `reduced = history + LR04 + CO2` and
  `full = reduced + sin(precession phase) + cos(precession phase)`.
- Do not read `local_median_resolution_yr` into either model. Record explicitly
  that the resolution covariate and event-age uncertainty are not included.

## Step 2: Implement the root analysis script

**Create:** `MIS6_event_phase_analysis.py`

- Load and validate the catalogue.
- Sample event phases and run the shared finite-sample Rayleigh test.
- Build the PI design table, fit the nested Poisson models, and calculate the
  nominal two-degree-of-freedom likelihood-ratio comparison, bits/event, AICc
  difference, preferred phase, and phase rate ratio.
- Fail clearly on phase extrapolation, failed optimization, broken likelihood
  nesting, or numerical predictor clipping.
- Write audit tables to `data/processed/MIS6_event_phase_analysis/` using the
  NGRIP output names where applicable.
- Write timeline, Rayleigh, and PI diagnostic figures to
  `figures/MIS6_event_phase_analysis/` as PNG and PDF, with ages labelled
  `Kyr BP`.

## Step 3: Verify numerics and assumptions

**Create:** `tests/test_mis6_event_phase_analysis.py`

- Test the 21-event input contract and warming-only catalogue.
- Test that the reduced/full term lists exactly match NGRIP and contain no
  resolution variable.
- Lock the Rayleigh result and the main PI likelihood comparison to numerical
  regression targets.
- Verify the expected difference between Rayleigh N and PI N at the oldest
  history boundary, plus convergence, likelihood nesting, and zero clipping.
- Compile and run the script, run the focused tests, inspect every main CSV, and
  render-check the PNG/PDF figures.

## Interpretation boundary

This is an exploratory quick test on a short, provisional composite chronology.
The PI p value is a nominal asymptotic diagnostic; data resolution, event-age
uncertainty, age-model mismatch, coverage gaps, and chronology-construction
uncertainty are not propagated. A fitted preferred phase is not a causal result.
