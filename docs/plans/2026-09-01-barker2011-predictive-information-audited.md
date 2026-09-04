# Barker 2011 Predictive-Information Audit and Refactor Implementation Plan

> **Archived plan (superseded).** The current implementation lives in
> `Barker2011_do_predictive_information_audited.py`; the unsuffixed filename is
> now a compatibility wrapper for that script. Legacy output paths listed
> below have been removed. See the root `README.md` for current run commands
> and retained outputs.

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Verify the Barker event-phase analysis, preserve its core numerical results, simplify the script, expose the important catalogue/model limitations, and replace the redundant figures with two auditable figures.

**Architecture:** Keep the existing shared orbital-phase, event-input, Poisson, and likelihood helpers. Represent the three age variants with a small dataclass, load each source once, and run only the three models needed for the scientific question: event-process baseline, climate-state model, and full model. Add compact sensitivity tables for Barker's fixed versus variable event definition and for the history-window choice, while retaining the variable-threshold 0--640 ka catalogue as the stated primary analysis.

**Tech Stack:** Python, pandas, NumPy, SciPy, Matplotlib, pytest, existing `toolbox` modules.

---

### Task 1: Lock down the current numerical behavior

**Files:**
- Create: `tests/test_barker2011_predictive_information.py`
- Modify: none

**Steps:**
1. Test the three catalogue counts, age ranges, monotonicity, and the fact that they overlap rather than form independent replications.
2. Test interpolation/extrapolation and resolution scaling.
3. Test the published-script regression targets for precession Rayleigh statistics and the conditional PI comparison.
4. Run `pytest tests/test_barker2011_predictive_information.py -q`; the new API-oriented tests should fail before the refactor while existing numerical behavior remains documented.

### Task 2: Simplify data preparation and model fitting

**Files:**
- Modify: `Barker2011_do_predictive_information.py`
- Test: `tests/test_barker2011_predictive_information.py`

**Steps:**
1. Replace repeated dictionary lookups with an immutable `CatalogueSpec` dataclass.
2. Read the Barker workbook and Jouzel resolution series once, passing cached tables to helpers.
3. Keep only the baseline, climate-state, and full models and the two nested comparisons that answer the stated question.
4. Return one `AnalysisTables` dataclass instead of several parallel lists and tuples.
5. Add explicit columns for Rayleigh event count versus PI event count/support.
6. Run the Barker test module and compare core metrics against the pre-refactor baseline with numerical tolerances.

### Task 3: Add concise reliability checks

**Files:**
- Modify: `Barker2011_do_predictive_information.py`
- Test: `tests/test_barker2011_predictive_information.py`

**Steps:**
1. Refit the conditional phase comparison for Barker's fixed-threshold event picks and write a small event-definition sensitivity table.
2. Refit the primary variable-threshold catalogues at 2, 5, 10, and 20 ka history windows and write a small history sensitivity table.
3. Label all chi-square LR values as asymptotic/nominal and record that age uncertainty is not propagated.
4. Verify that the primary 5 ka estimates are unchanged.

### Task 4: Replace the redundant figures

**Files:**
- Modify: `Barker2011_do_predictive_information.py`
- Generate: `figures/Barker2011_do_predictive_information/fig01_barker_catalogue_audit.{png,pdf}`
- Generate: `figures/Barker2011_do_predictive_information/fig02_barker_phase_results.{png,pdf}`

**Steps:**
1. Make a catalogue-audit figure showing fixed/variable event rugs, the 640 ka boundary, local EDC sample spacing, and SpeleoAge-minus-EDC3 offsets.
2. Make a statistical figure with three Rayleigh histograms above three conditional PI phase-response curves; annotate N, nominal p, bits/event, preferred phase, and rate ratio.
3. Remove the obsolete duplicate figure files only after the replacements have been generated successfully.
4. Inspect every PNG and render-check every PDF.

### Task 5: End-to-end verification

**Files:**
- Modify: generated CSV and figure outputs under the existing Barker run directories

**Steps:**
1. Run `python Barker2011_do_predictive_information.py`.
2. Run `pytest tests/test_barker2011_predictive_information.py -q` and relevant toolbox tests.
3. Compare core pre/post-refactor Rayleigh and PI values to the saved baseline using `numpy.testing.assert_allclose`.
4. Report the statistical audit separately from code correctness: no obvious likelihood/arithmetic defect, but strong dependence on event definition, age model, and model/history choices.
