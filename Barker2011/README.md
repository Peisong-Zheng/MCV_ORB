# Barker 2011 SpeleoAge analysis

This separate analysis uses the published warming picks in Supplementary
Table S3 of Barker et al. (2011), *Science* 334, 347–351,
[doi:10.1126/science.1203580](https://doi.org/10.1126/science.1203580).
The main catalogue contains 70 varying-threshold events on SpeleoAge within
0–400 kyr BP; 59 fixed-threshold picks provide a nominal-age comparison.
These events are not pooled with NGRIP–MIS6, and their chronology is not
independent of all source-age constraints in the primary study.

## Inputs and age uncertainty

- `data/raw/Barker et al-2011-SOM.xls`: `Sheet1`, Excel row 9 as header.
  Event ages come from `SpeloAge (kyr).1`; retain a pick value of 1 in
  `DO pick variable threshold` or `DO pick`. Stable event IDs retain the
  visible source Excel row as `Barker_S3_XXX`.
- `data/raw/Barker2011_TableS1.csv`: transcription of the 60-row chronology
  table, checked against `references/Barker2011_SOM.pdf`.
- Shared background and orbital sources reside in `../data/raw/`: LR04,
  composite CO2, La2004 precession/obliquity/eccentricity, and 65°N
  summer-solstice insolation.

The age sampler uses SpeleoAge and the published combined uncertainty.
Control-point offsets are interpolated across the data gap; a conservative
auxiliary older control covers events beyond the final dated control.
EDC3 coordinates remain source metadata. The sampler retains event order and
saves combined-age realizations for later model refits. Its construction is
unchanged by the continuous-time migration.

SpeleoAge numerical values are retained and treated as BP1950; the precise
original reference year remains unverified. Shared orbital inputs receive
the documented −0.05-kyr J2000-to-BP1950 conversion once. See
[the epoch audit](../docs/age_epoch_audit.md).

## Continuous model and boundaries

The main fitter is shared with NGRIP–MIS6: history decays exponentially with
fixed tau = 1.5 kyr and has a nonpositive coefficient. LR04 and CO2 enter both
models; full adds precession-phase sine and cosine. Likelihood is the sum of
actual response-event log intensities minus integrated intensity over all
response time. AIC and likelihood gain per response event, G, summarize fits.

Varying threshold conditions on the exact oldest event at 396.464263878 kyr BP,
leaving 69 response events to the fixed endpoint at 0 kyr BP. Fixed threshold
conditions at 392.245695897 kyr BP and uses 58 response events. Unknown
pre-anchor history is fixed to zero; each anchor contributes immediately to
its younger history. The two definitions have different exposure, so their G
values are descriptive comparisons rather than a same-support likelihood test.
Both appear in the main figures; fixed threshold receives no extra bootstrap
or chronology ensemble.

The varying-threshold phase test uses 9,999 continuous reduced-model simulations
with fixed anchor and endpoint, dynamic history and refitting of both models.
The age experiment refits all saved combined-age draws with their own exact
anchors and exposure. History utility, full-model residual checks and extra
10%/20% response-event deletion are separate experiments. Residual tests use
simulation/refitting calibration; deletion scenarios do not estimate missing
source events. The new diagnostics save tables and notes without adding default
manuscript figures.

## Run and outputs

Run from the workspace root using the project Python environment:

```bash
python Barker2011/Barker2011_event_phase_analysis.py
python Barker2011/Barker2011_event_age_uncertainty.py
python Barker2011/Barker2011_event_uncertainty_sensitivity.py
python Barker2011/Barker2011_likelihood_bootstrap.py
python Barker2011/Barker2011_model_diagnostics.py
python Barker2011/Barker2011_event_detection_sensitivity.py
python Barker2011/Barker2011_orbital_driver_sensitivity.py
python Barker2011/Barker2011_climate_phase_interaction.py
```

Existing combined-age realizations can be reused without rerunning the sampler.
`event_phase_analysis` writes nominal main and `fixed_threshold/` results,
including source IDs/roles, event phases, fitted rate samples, coefficients,
likelihood statistics, scaling and support. Other experiments save their own
replicate and summary tables under `data/processed/<script>/`, figures under
`figures/<script>/` and explanatory text under `experiment_note/`.

Bootstrap and LR04-interaction exporters can rebuild the paired research PDF
and synchronize it with the manuscript. Review runs use `--output-root` and,
where provided, `--no-paper-export` to defer this chain. The shared figure
palette assigns rose to varying threshold and green to fixed threshold;
absolute-age axes put younger ages on the right.

Current model validation is in the
[migration audit](../docs/reviews/continuous-time-migration-2026-09-12.md).
The complete previous project, including earlier Barker results, is preserved
in `../archive/pre_continuous_time_2026-09-12/`. The original three-catalogue
Barker workflow remains in `../archive/Barker2011_2026-09-05/`.
