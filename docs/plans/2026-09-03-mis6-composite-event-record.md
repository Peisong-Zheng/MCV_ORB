# MIS 6 Composite Event Record Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Build a provisional MIS 6 event chronology from MF (130–175 ka), Huagapo (Held 6.9, 6.12, 6.13), and Sofular (Held 6.10, 6.11), with reproducible transition-age estimates, proxy changes, diagnostics, and composite labels.

**Architecture:** A single root-level Python script reads the existing workbook and published-label anchor table, cleans and segments each record, estimates one direction-constrained transition per selected literature event, exports one compact processed CSV, and renders one static diagnostic figure. Published anchors are used only to delimit local searches; estimated ages are explicitly kept separate from literature labels and age-model uncertainty.

**Tech Stack:** Python, pandas, NumPy, SciPy, Matplotlib, openpyxl.

---

### Task 1: Define the Composite Event Sequence

**Files:**
- Read: `data/processed/Speleothem_published_event_plot/speleothem_mis6_published_event_label_anchors.csv`
- Create: `MIS6_composite_event_record.py`

1. Select all MF/Fohlmeister events whose anchors fall within 130–175 ka.
2. Select Huagapo/Held 6.9, 6.12, and 6.13 and Sofular/Held 6.10 and 6.11.
3. Preserve the original source record, source label, anchor type, and anchor age.
4. Retain MF display labels 6.1–6.16; sort the five post-175 ka events by age and assign display labels 6.17–6.21 plus non-numeric stable IDs `MIS6_DO_01`–`MIS6_DO_21`.
5. Treat Held midpoints and existing MF label positions as search anchors, not final event ages.

### Task 2: Estimate Directional Transition Ages

**Files:**
- Read: `data/raw/speleothem_data.xlsx`
- Modify: `MIS6_composite_event_record.py`

1. Convert age/proxy values to numeric, average duplicate ages, and sort by age.
2. Split records at gaps greater than 0.5 ka so interpolation and smoothing never bridge material hiatuses.
3. Interpolate each continuous segment to a 0.001 ka grid solely to evaluate a time-domain filter and derivative uniformly.
4. Apply record-specific Gaussian smoothing: 0.050 ka for MF, 0.100 ka for Huagapo, and 0.050 ka for Sofular.
5. Search within anchor ±0.400 ka, clipped by continuous-segment limits and by midpoints between adjacent selected anchors.
6. Pick the strongest expected directional gradient: negative d(proxy)/d(age) for MF and Huagapo δ18O, positive for Sofular δ13C.
7. Quantify method sensitivity across three smoothing scales and search half-widths of 0.30, 0.40, and 0.50 ka. Interpret the resulting age range as parameter sensitivity, not chronological uncertainty.
8. Preserve discrete tuning-run candidates, flag multimodal ranges, sparse sampling, segment-edge proximity, and known two-stage MF transitions for visual review.

### Task 3: Calculate Event Changes and Export the Record

**Files:**
- Create: `data/processed/MIS6_composite_event_record/mis6_composite_event_record.csv`
- Modify: `MIS6_composite_event_record.py`

1. For the nominal smoothed curve, summarize young- and old-side proxy levels over 50–150 yr from the selected transition; call this a local flank change rather than a full event amplitude.
2. Store young-minus-old local proxy change, absolute local transition magnitude, flank observation counts and change QC, peak gradient, local median sampling resolution, search bounds, discrete tuning candidates, and local tuning limits.
3. Round exported values only for presentation while retaining millennial ages to 0.001 ka for auditing.
4. Validate 21 ordered, unique composite events; finite ages and changes; selected ages inside their search bounds; and the expected gradient sign.

### Task 4: Draw the Diagnostic Figure

**Files:**
- Create: `figures/MIS6_composite_event_record/MIS6_composite_event_record.png`
- Create: `figures/MIS6_composite_event_record/MIS6_composite_event_record.pdf`
- Modify: `MIS6_composite_event_record.py`

1. Draw a composite timeline showing event source, estimated age, and distinct symbols for stable, moderate, and review estimates.
2. Draw MF, Huagapo, and Sofular proxy panels with raw observations, nominal smoothed curves, source anchors, estimated transition points, and discrete tuning-candidate ticks.
3. Clearly state that points are provisional algorithmic transition estimates and tuning ticks are not age-model uncertainty.
4. Use publication-size typography, vector PDF output, and a 600 dpi PNG.

### Task 5: Verify the Deliverables

**Files:**
- Verify: `MIS6_composite_event_record.py`
- Verify: `data/processed/MIS6_composite_event_record/mis6_composite_event_record.csv`
- Verify: `figures/MIS6_composite_event_record/MIS6_composite_event_record.png`
- Verify: `figures/MIS6_composite_event_record/MIS6_composite_event_record.pdf`

1. Compile and run the script with warnings treated as errors.
2. Re-read the CSV and independently check its schema, row count, ordering, uniqueness, numeric finiteness, and source-to-composite mapping.
3. Inspect the PNG at full scale and in dense-event crops for clipping, overlaps, and misleading symbology.
4. Confirm the PDF is one page, has the intended physical dimensions, contains vector/text content, and renders consistently with the PNG.
