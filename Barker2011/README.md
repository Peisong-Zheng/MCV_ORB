# Barker synthetic Greenland events

This directory analyzes warming events from Supplementary Table S3 of
[Barker et al. (2011)](https://doi.org/10.1126/science.1203580).
The variable-threshold catalogue provides the main analysis; the fixed-threshold
catalogue checks sensitivity to event selection. They are alternative
definitions within one reconstruction, fitted separately from NGRIP–MIS6.

## Method

Both catalogues use the [shared continuous-time model](../README.md), which
compares event rates with and without precession phase after accounting for
LR04, CO₂ and exponentially decaying event history. Each catalogue's oldest
event initializes history and sets the start of its fitted interval.

Chronology sensitivity uses the variable-threshold catalogue. Published
combined uncertainties at Table S1 control points are treated as uniform
half-widths; interpolating control-point shifts gives correlated event-age
changes. Sampling preserves event order. No separate onset-picking error is
added. The [supplement, Text S3](../orbital_event_paper/SI.tex) describes the gap
interpolation and auxiliary oldest control.

## Starting points

| Entry | Purpose |
|---|---|
| [data_pre_processing.py](data_pre_processing.py) | Extract both event catalogues into CSVs containing event IDs and ages. |
| [Barker2011_event_phase_analysis.py](Barker2011_event_phase_analysis.py) | Fit both definitions and plot their phase responses. |
| [Barker2011_event_age_uncertainty.py](Barker2011_event_age_uncertainty.py) | Generate chronologies for [age-sensitivity refits](Barker2011_event_uncertainty_sensitivity.py). |
| [Barker2011_likelihood_bootstrap.py](Barker2011_likelihood_bootstrap.py) | Calibrate phase tests with simulated catalogues. |
| [Barker2011_effect_uncertainty.py](Barker2011_effect_uncertainty.py) | Compare chronology and event-sampling uncertainty. |

Run scripts from the project root; settings are near their tops. Shared forcing
inputs come from [forcing_data_pre_processing.py](../forcing_data_pre_processing.py).
Published SpeleoAge values are retained without an epoch shift and used as
working BP ages; their original reference year remains unverified.

Prepared event CSVs are in `data/processed/`. Analysis tables and figures use
`data/processed/<script_name>/` and `figures/<script_name>/`.
