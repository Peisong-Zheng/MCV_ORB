# MIS 6 Age-Uncertainty Visualization and PI Sensitivity Plan

**Goal:** Improve the chronology-control display, show how the A+ Monte Carlo ensemble differs from the extracted 21-event sequence, and propagate those age realizations through the existing predictive-information analysis without adding a Rayleigh test.

**Method:** Keep the existing event detector, A+ sampler, forcing series, bin width, history term, and reduced/full Poisson models unchanged. Improve only the display of U-Th controls, summarize all 10,000 age realizations with robust marginal intervals and their joint correlation structure, then refit the same nested PI models to every accepted realization. No proxy-resolution or record-synchronization term is introduced.

## 1. Prevent control-point overlap

- Assign displayed U-Th intervals to the minimum number of vertical lanes with a deterministic greedy interval-partition algorithm.
- Treat each control as its full reported `[age - 2sigma, age + 2sigma]` interval, including a small visual padding.
- Remove the fixed stalagmite lane labels, because the legend and caption already define the marks and fixed labels would become misleading after automatic staggering.
- Regenerate and inspect the composite PNG/PDF.

## 2. Visualize the A+ Monte Carlo sequence ensemble

- Add a static two-panel figure to `MIS6_event_age_uncertainty.py`.
- Panel (a): for each event, show the Monte Carlo median and 16–84% / 2.5–97.5% ranges of `sampled age - original picked age`; the zero line is the original extracted sequence.
- Panel (b): show the correlation matrix of event-age anomalies, preserving the shared chronology shifts within each source record.
- Save PNG/PDF under `figures/MIS6_event_age_uncertainty/` and write a concise caption in `experiment_note/`.

## 3. Add PI-only age-uncertainty sensitivity analysis

- Create `MIS6_event_age_PI_sensitivity.py` at the repository root.
- Read the accepted wide age-realization table and the original composite event catalogue.
- Reuse the exact MIS6/NGRIP PI support, binning, forcing covariates, five-kyr history, reduced model, and phase-augmented full model; never call the Rayleigh workflow and never add data resolution.
- Refit both models for the original sequence and every Monte Carlo realization, using the same bounded Poisson likelihood with an analytic gradient for practical runtime.
- Export realization-level PI results, a compact uncertainty summary, and provenance/settings tables.
- Plot the Monte Carlo distributions of likelihood-ratio p value, information bits per event, preferred phase, and delta AICc, with the original-sequence estimate and decision thresholds identified.

## 4. Verification

- Add focused tests for non-overlapping control lanes, age-ensemble figure inputs, equivalence of the optimized PI fit to the existing implementation, and PI output contracts.
- Compile the scripts; run focused tests; regenerate all affected tables and figures.
- Inspect the generated raster figures and verify that CSV outputs contain finite results and all expected realizations.
