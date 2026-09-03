# Speleothem GRL Figure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Produce a concise GRL-ready static figure in which literature event identifications cannot be mistaken for newly picked transition ages.

**Architecture:** Keep the existing data loading, validation, local-resolution calculation, gap handling, and eight-row layout. Centralize final-size graphics constants, separate panel labels from record titles, and place event text in a visually distinct annotation band. Use subtle dashed leaders as visual guides, but no point symbols that could be mistaken for formal event-age picks.

**Tech Stack:** Python, pandas, NumPy, Matplotlib, openpyxl.

---

### Task 1: Preserve Current User Choices

**Files:**
- Modify: `Speleothem_published_event_plot.py`

1. Keep the global title, subtitle, proxy horizontal grid, and footer disabled.
2. Keep the static PNG/PDF outputs, proxy-axis directions, resolution calculation, and material-gap breaks.
3. Run the existing script once to establish the current 49-anchor baseline.

### Task 2: Set Final GRL Figure Geometry

**Files:**
- Modify: `Speleothem_published_event_plot.py`

1. Set the figure width to 180 mm rather than creating an oversized canvas for later reduction.
2. Set final-size typography explicitly: 8–10 pt for axes/panel labels and at least 7.5–8 pt for dense event labels.
3. Increase proxy and resolution line widths for direct use at final size.
4. Export a vector PDF and a 600 dpi PNG.

### Task 3: Correct the Event-Identification Semantics

**Files:**
- Modify: `Speleothem_published_event_plot.py`

1. Remove event scatter markers.
2. Draw thin, low-contrast dashed leaders from the labels to their approximate proxy-curve anchors, without endpoint symbols.
3. Retain event names in the pale annotation band, with small horizontal nudges for tight clusters.
4. Describe these as event identifications reported in the cited publications, not event ages, picks, midpoints, or onsets.
5. Do not add artificial uncertainty ribbons because the source papers do not define interval bounds for these labels.

### Task 4: Group-Level Panel Labels

**Files:**
- Modify: `Speleothem_published_event_plot.py`

1. Add one bold lowercase `(a)`–`(d)` label to the proxy axis of each record-resolution pair.
2. Do not add a separate label to the resolution axes.
3. Shorten and standardize record titles so they remain legible at 180 mm width.

### Task 5: Render and Verify

**Files:**
- Verify: `figures/Speleothem_published_event_plot/Speleothem_published_event_plot.png`
- Verify: `figures/Speleothem_published_event_plot/Speleothem_published_event_plot.pdf`

1. Run: `/opt/anaconda3/bin/python3 -W error Speleothem_published_event_plot.py`.
2. Expect: 49 labels across four records; PNG and PDF regenerated without warnings.
3. Confirm: PNG width is about 4,250 px at 600 dpi, PDF is one page, and both files remain below 10 MB.
4. Inspect the full figure and dense-label crops for clipping or overlap.
5. Confirm there are 49 dashed leaders, no event markers, and only four group-level panel labels.
