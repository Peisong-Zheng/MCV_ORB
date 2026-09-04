# MIS 6 Event-Age Uncertainty (A+) Implementation Plan

**Goal:** Propagate a deliberately simple combination of event-definition and chronology uncertainty for the 21-event MIS 6 composite, while preserving event identities and order.

**Method:** Each proposal first draws one chronology state (`z`) per source record, then one of the nine detector parameter configurations per record. Event ages are re-picked with that shared configuration, chronology uncertainty is evaluated at each picked age, and the shared record-level `z` is applied. A complete 21-event proposal is rejected if its fixed event order is violated; ages are never sorted to repair crossings.

## 1. Add the standalone uncertainty script

- Create `MIS6_event_age_uncertainty.py` in the repository root.
- Reuse the event detector and record definitions from `MIS6_composite_event_record.py` rather than duplicating them.
- Use 10,000 accepted realizations and a fixed seed by default.
- Preserve all nine smoothing-window configurations and sample them as equally weighted method choices, once per record and realization.

## 2. Encode the three chronology approximations

- MF: interpolate the published `age_lower`/`age_upper` age-model envelope at the sampled pick, repair it to enclose the nominal age where needed, use the larger side as an approximately 95% half-width, and divide by 1.95996398454. One shared `z` is used for all 16 MF events.
- Huagapo: linearly interpolate the published adjacent P10-H2 U-Th 2-sigma errors at each sampled pick, then divide by 2. One shared `z` is used for the three Huagapo events.
- Sofular: use the explicitly documented local approximation of 0.375 Kyr (1 sigma), with one shared `z` for both events.
- Do not add record synchronization error, proxy sampling resolution, smoothing scale, or raw U-Th errors already represented by an age model as separate uncertainty terms.

## 3. Export auditable processed data

- Write the required U-Th control-point table, including cave, study, source filename (not path), source locator, uncertainty level, and role.
- Write a 21-row uncertainty summary with nominal ages, definition support, chronology sigma, and Monte Carlo quantiles.
- Write a wide realization table for later Rayleigh/PI propagation, plus compact record-parameter and sampling-diagnostic tables.

## 4. Simplify and update the existing composite figure

- Remove alternative-pick ticks and all QC-dependent event symbols.
- Plot every event estimate as a filled circle.
- Add a narrow U-Th control strip with horizontal 2-sigma error bars to the Huagapo and Sofular panels; identify these controls explicitly as not event-age intervals.
- Update the existing caption to match the revised visual encoding.

## 5. Verify

- Add focused tests for input contracts, duplicate-age envelope handling, local uncertainty interpolation, shared record-level configuration and `z`, fixed-order rejection, reproducibility, and output quantiles.
- Run compilation, tests, the uncertainty script, and the composite-figure script.
- Re-read every generated CSV, inspect the PNG, and render/check the PDF.
