# MCV_ORB

Does precession phase improve the fit to abrupt-warming occurrence after
accounting for background climate and earlier events? This project fits a
continuous-time conditional event rate to Greenland and speleothem event
catalogues. The response is warming onset per unit of total observation time.

## Data and model

The primary catalogue combines 34 NGRIP Greenland Interstadial starts
(Rasmussen et al., 2014) with 21 MIS 6-centered speleothem transitions:
16 from Melchsee–Frutt (Fohlmeister et al., 2023) and five from Sofular
(Held et al., 2024). Barker et al. (2011) SpeleoAge events provide a separate
longer-record comparison. The fixed-threshold Barker catalogue is a nominal-age
definition sensitivity with its own phase-test bootstrap, but no separate
chronology ensemble.

| Segment/catalogue | Inventory events | Observation support, kyr BP | Conditional response support, kyr BP | Response events |
|---|---:|---:|---:|---:|
| NGRIP | 34 | 12–123 | 12–115.320 | 33 |
| MIS 6-centered | 21 | 132.5–204.5 | 132.5–194.238 | 20 |
| Barker varying threshold | 70 | 0–400 | 0–396.464264 | 69 |
| Barker fixed threshold | 59 | 0–400 | 0–392.245696 | 58 |

Each independent segment conditions on its exact oldest event. That event
initializes history and is excluded from the event likelihood sum; all younger
time, including the event-free terminal tail, remains exposed. Primary
response exposure is 165.058 kyr. No history or exposure crosses the record gap.
Unobserved pre-anchor history is fixed to zero as a boundary approximation.
Descriptive phase plots retain every inventory event.

For BP age `a`, larger ages precede smaller ages:

```text
H(a) = sum over strictly older events j in the same segment:
       exp[-(a_j - a) / tau], with tau = 1.5 kyr

log(lambda_BG)  = intercept + beta_H H + beta_L LR04 + beta_C CO2
                 + primary segment contrast
log(lambda_Pre) = the same terms + beta_s sin(phase) + beta_c cos(phase)
beta_H <= 0

log likelihood = sum over response events log(lambda(a_i))
                 - integral of lambda over response time
```

Every accepted event updates history immediately. Both models are refitted
under the same history constraint. LR04 and CO2 use fixed nominal response-time
mean/range scaling. External covariates are interpolated from their source
series; positive quadrature weights integrate intensity between actual events
and forcing breakpoints. Integration precision is a numerical check.

The reported log-likelihood gain is
`G = (loglik_Pre - loglik_BG) / (N_response * log(2))`, in bits/event.
It describes sample fit improvement. Unless stated otherwise, LR comparisons
report nominal p values from the asymptotic chi-square distribution. BG-model
bootstrap simulations check the main phase results against this approximation;
the paper reports both calibrations and Figure 2 displays bootstrap p values.
AIC compares fitted specifications. No integration-node count is
treated as a statistical sample size. Phase zero denotes a precession-index
minimum and 180° a maximum. The fitted phase curve is a conditional rate
multiplier at fixed background and history, not the orbital waveform.

## Uncertainty and model checks

- **Phase test:** 9,999 BG-model simulations for NGRIP–MIS6 and for each
  Barker event definition at nominal ages.
  Exact anchors, forcing and endpoints remain fixed; response events are
  simulated continuously with dynamic history. Both models are refitted.
  Bootstrap p is `(k + 1) / 10,000`, where `k` is the number of simulated LR
  statistics at least as large as the observed LR. These are finite-sample
  calibration sensitivities, not extra chronology realizations. The two Barker
  definitions share source events and are not independent replications.
- **Chronology:** the existing 10,000-member source-age ensembles are reused.
  Each realization updates actual event ages, its anchor, response exposure and
  history, retaining nominal climate scaling. Unsupported draws are recorded
  without clipping or replacement. Their ranges describe chronology sensitivity.
- **Effect precision:** primary age-only ranges are compared with 5,000 nominal
  full-model simulations and 200 chronology generators with 50 simulations
  each. All three use matching 95% coefficient-ellipse projections. Only
  nominal-age sampling gives an approximate conditional confidence region;
  chronology and combined regions describe sensitivity to assumed age errors.
- **History and fit:** the history contribution is tested conditional on phase
  using a separate null bootstrap. Full-model rescaled intervals and adjacent
  residual dependence are calibrated by simulation and refitting, retaining
  fixed termination and segment boundaries. Primary calibration reuses the
  nominal full-model ensemble. These checks produce tables and notes by default.
- **Event deletion:** independent 10% and 20% deletion scenarios remove response
  events, preserve anchors and exposure, and rebuild history from the retained
  catalogue. These are extra-deletion stress tests, separate from age errors;
  they do not infer undetected events or a detection probability.
- **Specifications:** fixed history decay times, pre-anchor history, background
  shape, history alternatives and orbital drivers are checked separately.
  Additional tests do not inherit the main bootstrap p.
  Orbital comparisons retain their specified Holm adjustment of nominal LR
  p values. Rayleigh p describes unadjusted phase concentration and is separate
  from the conditional-model tests.

Barker and the primary records share some chronology dependencies. Conditional
associations and chronology perturbations cannot identify a unique physical
trigger or remove source-age tuning assumptions.

## Project layout and reproduction

| Location | Purpose |
|---|---|
| `NGRIP/`, `MIS6/` | Source preparation, event chronology, data and source figures |
| `Barker2011/` | Separate SpeleoAge analysis and sensitivities |
| `NGRIP_MIS6_*.py` | Combined nominal, uncertainty and sensitivity analyses |
| `climate_precession_contribution.py` | Conditional climate versus precession gains for all three catalogues; figure synchronized to the manuscript, with separate research notes |
| `Figure_orbital_mechanism.py` | Geographic mechanism schematic for Figure 4; no model fitting |
| `toolbox/point_process.py` | Continuous likelihood, strict event history and thinning |
| `toolbox/combined_likelihood.py` | Shared forcing, exact supports and catalogue fits |
| `data/processed/`, `experiment_note/` | Saved research results and explanatory notes |
| `tests/diagnostics/` | Integration and migration checks |
| `orbital_event_paper/` | Manuscript, SI and exported figures |

Use the project's Python environment and `requirements.txt`. From the project
root, after preparing the source event-age ensembles:

```bash
python NGRIP_MIS6_event_phase_analysis.py
python Barker2011/Barker2011_event_phase_analysis.py
python NGRIP_MIS6_event_uncertainty_sensitivity.py
python Barker2011/Barker2011_event_uncertainty_sensitivity.py
python NGRIP_MIS6_likelihood_bootstrap.py
python Barker2011/Barker2011_likelihood_bootstrap.py
python Barker2011/Barker2011_likelihood_bootstrap.py --event-definition fixed_threshold --n-bootstrap 9999 --seed 20260920
python NGRIP/ngrip_transition_phase_sensitivity.py --workers 3
python NGRIP_MIS6_effect_uncertainty.py
python Barker2011/Barker2011_effect_uncertainty.py
python NGRIP_MIS6_model_diagnostics.py
python Barker2011/Barker2011_model_diagnostics.py
python NGRIP_MIS6_event_detection_sensitivity.py
python Barker2011/Barker2011_event_detection_sensitivity.py
python NGRIP_MIS6_likelihood_design_sensitivity.py
python NGRIP_MIS6_likelihood_model_sensitivity.py
python NGRIP_MIS6_orbital_driver_sensitivity.py
python Barker2011/Barker2011_orbital_driver_sensitivity.py
python climate_precession_contribution.py
make -C orbital_event_paper pdf
make -C orbital_event_paper check
```

Main-analysis tables distinguish `event_role` (conditioning/response), actual
ages, fitted rate samples, coefficients, support and model statistics. Fitted
rate samples are plotting coordinates, not event counts. Chronology fits are
saved as `gain_realizations.csv`; bootstrap replicates retain explicit validity.
Scripts supporting `--output-root` can write a separate review tree. Use
`--no-paper-export` where available to defer canonical figure synchronization.

The paper build reads saved results and compiles TeX; it does not refit models
or rewrite manuscript prose. See [the paper README](orbital_event_paper/README.md).
Absolute-age axes put younger ages on the right. Three catalogue colors are
shared in `toolbox/catalogue_colors.py`.

Run checks with `python -m pytest -q`. Source epoch conventions are documented
in [the epoch audit](docs/age_epoch_audit.md); Barker SpeleoAge's exact original
reference year remains an explicit BP1950 working assumption.

The complete pre-migration project is frozen in
[archive/pre_continuous_time_2026-09-12](archive/pre_continuous_time_2026-09-12/README.md).
The retired LR04–phase modulation experiment, including its code, results and
figures, is preserved in
[its archive](archive/LR04_phase_interaction_2026-09-23/README.md).
Current validation and result status are recorded in the
[continuous-time migration audit](docs/reviews/continuous-time-migration-2026-09-12.md).
Earlier review reports retain their historical numbers. Author declarations and
permanent public code/data identifiers remain to be completed before submission.
