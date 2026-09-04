# MIS 6 Figure and Caption Revision Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Revise the composite MIS 6 diagnostic figure for GRL-scale readability, move figure-level explanation into a standalone caption, and standardize both speleothem figures to `kyr BP`.

**Architecture:** Keep all event ages and processed data unchanged. Modify only plotting semantics and typography in the two existing scripts, regenerate their PNG/PDF outputs, and add one caption text file. Replace full-height event lines with label-to-pick leaders so annotations remain readable, and describe marker shapes by their scientific limitation rather than internal QC names.

**Tech Stack:** Python, Matplotlib, pandas, NumPy, SciPy, plain text, PDF/PNG rendering.

---

### Task 1: Simplify the Composite Figure Frame

**Files:**
- Modify: `MIS6_composite_event_record.py`

1. Remove the figure-level title, subtitle, and bottom explanatory note.
2. Keep the legend, but remove the literature-anchor entry and increase its font size.
3. Move panel (a)'s label and title to the left with `axis.set_title(..., loc="left")`.
4. Increase base, axis-title, tick, legend, and event-label sizes for direct use at 180 mm width.

### Task 2: Make Event Annotations Unambiguous

**Files:**
- Modify: `MIS6_composite_event_record.py`

1. Remove the dotted literature-anchor lines.
2. Remove full-height vertical lines at selected event ages.
3. Connect each event label to its selected curve point with a short leader that terminates at an opaque label background instead of crossing the text.
4. Keep local tuning candidates as short ticks at the bottom of each proxy panel.
5. Map symbols to reader-facing causes: filled circle = stable under tested settings; open diamond = sampling-resolution limited; open triangle = two-stage or record-edge limited.

### Task 3: Standardize the Age Unit

**Files:**
- Modify: `MIS6_composite_event_record.py`
- Modify: `Speleothem_published_event_plot.py`

1. Replace every plotted `Age (ka BP; older →)` label in the composite figure with `Age (kyr BP)`.
2. Replace `Age (ka BP)` in the published-event figure with `Age (kyr BP)`.
3. Regenerate PNG and vector PDF outputs from both scripts.

### Task 4: Write the Figure Caption

**Files:**
- Create: `experiment_note/MIS6_composite_event_record_Caption.txt`

1. Describe panels (a)–(d), source records, source labels, and the renumbering to MIS 6.17–6.21.
2. Move the removed method subtitle and bottom tuning/uncertainty explanation into the caption.
3. Define the smoothing scales, ±400 yr search, direction-constrained gradient picks, symbol meanings, tuning ticks, reversed Sofular axis, and `kyr BP` unit.

### Task 5: Render and Verify

**Files:**
- Verify: `figures/MIS6_composite_event_record/MIS6_composite_event_record.png`
- Verify: `figures/MIS6_composite_event_record/MIS6_composite_event_record.pdf`
- Verify: `figures/Speleothem_published_event_plot/Speleothem_published_event_plot.png`
- Verify: `figures/Speleothem_published_event_plot/Speleothem_published_event_plot.pdf`

1. Compile and run both plotting scripts with warnings treated as errors.
2. Confirm the processed composite CSV is byte-for-byte unchanged.
3. Search the plotting code and extracted PDF text for obsolete `ka BP`, `older`, and `literature search anchor` strings.
4. Render both PDFs to PNG and inspect the full pages plus dense event-label regions for clipping, overlap, and undersized text.
5. Confirm both PDFs are one page with embedded text and the expected 180 mm width.
