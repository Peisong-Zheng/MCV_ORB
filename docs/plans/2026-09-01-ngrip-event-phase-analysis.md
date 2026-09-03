# NGRIP event-phase analysis plan

## Purpose

Test whether the published Rasmussen et al. (2014) Greenland Interstadial (GI)
and Greenland Stadial (GS) onsets show precession-phase dependence.  GI onsets
are treated as warming starts and GS onsets as cooling starts.  The analysis is
performed on the published event catalogue, not on a newly detected NGRIP
isotope series.

## Data extraction

1. Transcribe the GI/GS rows from Table 2 (PDF page 10; printed page 9), keeping
   the original event label, NGRIP depth, age in years b2k, definition-
   uncertainty code, maximum counting error, and notes.
2. Do not count lettered subevents as separate events.  When Table 2 provides
   only lettered rows for a parent event, use the oldest subevent onset as the
   parent onset (for example, GI-1e becomes the single GI-1 start).  Keep both
   labels and a selection note in the processed catalogue.
3. Preserve ages in years/ka b2k and convert to the project's ka BP convention
   with `age_ka_bp = age_yr_b2k / 1000 - 0.05`.
4. Cross-check labels, ages, depths, and counts against the independent
   workspace reference table.  Expected collapsed counts are 34 GI warming
   starts and 35 GS cooling starts.

## Analysis

1. Sample the reusable precession phase series at each event age and run the
   finite-sample Rayleigh test separately for warming and cooling starts.
2. Build 0.2 ka event-count bins over 12--120 ka BP.
3. Add a 5 ka same-type event-history term.  Do not use the Cheng sampling-
   resolution nuisance term because this is a published event catalogue rather
   than a KS catalogue from the Cheng record.
4. Fit two nested Poisson event-rate models for each event type:

   - reduced: same-type history + LR04 + CO2;
   - full: reduced + sin(precession phase) + cos(precession phase).

5. Report the likelihood-ratio statistic and nominal chi-square p value,
   information gain in bits/event, AICc difference, preferred phase, phase
   rate ratio, convergence, likelihood nesting, and numerical clipping checks.

## Outputs

- `NGRIP/data/processed/rasmussen2014_table2_gi_gs_rows.csv`: traceable Table 2
  GI/GS transcription, including subevent rows and selection flags.
- `NGRIP/data/processed/ngrip_warming_cooling_starts.csv`: one collapsed parent
  start per GI/GS event, used by the analysis.
- `NGRIP/data/processed/ngrip_event_phase_analysis/`: sampled phases, Rayleigh
  results, PI inputs/results, summaries, parameters, and provenance.
- `NGRIP/figures/ngrip_event_phase_analysis/`:
  1. precession series with warming/cooling event rugs;
  2. Rayleigh polar phase distributions;
  3. conditional PI phase-response curves.
- `NGRIP/README.md`: extraction rule, methods, results, and limitations.

## Verification

- Assert 109 source GI/GS rows, 69 collapsed events, 34 warming and 35 cooling
  starts, unique parent labels, finite/ordered ages, and exact b2k-to-BP
  conversion.
- Assert no phase extrapolation, reduced/full convergence, nested likelihoods,
  and no fitted linear-predictor clipping.
- Run focused tests, compile the script, execute the full analysis, inspect all
  CSV summaries, and visually inspect each generated figure.

