# Barker 2011 SpeleoAge analysis

This separate analysis uses the published warming picks in Supplementary
Table S3 of Barker et al. (2011), *Science* 334, 347–351,
[doi:10.1126/science.1203580](https://doi.org/10.1126/science.1203580).
The main catalogue contains 70 varying-threshold events on SpeleoAge within
0–400 kyr BP; 59 fixed-threshold picks provide a nominal-age comparison.
These events are not pooled with NGRIP–MIS6, and their chronology is not
independent of all source-age constraints in the primary study.

## Inputs and age uncertainty

- `data_pre_processing.py` prepares the event inputs before any analysis:
  `data/processed/barker_events_variable_threshold.csv` (70 rows) and
  `data/processed/barker_events_fixed_threshold.csv` (59 rows). Each contains
  only `event_id,event_age_kyr_bp`, replacing the former 11-column source
  table. The two definitions select different rows, with identical ages for
  shared IDs. Ages are saved with 17 significant digits and read with
  `float_precision="round_trip"`.
- `data/raw/Barker et al-2011-SOM.xls`: `Sheet1`, Excel row 9 as header.
  The preparation script takes `SpeloAge (kyr).1`, retains a pick value of 1
  in `DO pick variable threshold` or `DO pick`, filters to 0–400 kyr and
  sorts by age. Stable IDs retain the visible source Excel row as
  `Barker_S3_XXX`; they are not renumbered within each definition.
- `data/raw/Barker2011_TableS1.csv`: transcription of the 60-row chronology
  table in the published supplement. The age sampler reads this CSV directly;
  a local copy of the supplementary PDF is not required.
- Shared background and orbital sources reside in `../data/raw/`: LR04,
  composite CO2, La2004 precession/obliquity/eccentricity, and 65°N
  summer-solstice insolation.

Analyses read the selected CSV directly, then build the scientific context:

```python
events = pd.read_csv(BARKER_EVENT_CSVS[event_definition], float_precision="round_trip")
context = combined_likelihood.build_barker_context(events, event_definition=event_definition)
fit = combined_likelihood.fit_catalogue(context.events, context)
```

The context adds the required `Barker2011` segment constant; it does not
reconstruct the discarded source, label or duplicate-age columns. Missing
prepared files raise an error rather than triggering Excel parsing. Existing
provenance code may still read raw-file bytes, so the original SOM remains
part of the project.

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
Both appear in the main figures. Each definition has a separate phase-test
bootstrap at nominal ages; only varying threshold has a chronology ensemble.

Unless stated otherwise, LR comparisons report nominal chi-square p values.
Each definition also uses 9,999 continuous BG-model simulations to check the
phase-test calibration, with its own fixed anchor and endpoint, dynamic history
and refitting of both models. The resulting bootstrap p is `(k + 1) / 10,000`,
where `k` counts simulations with LR at least as large as observed. The fixed
threshold run uses seed `20260920`; it does not reuse the varying-threshold null
distribution. Main paper results report both p values, and paper Figure 2
displays bootstrap p for both definitions. The separate Barker research figure
continues to show explicitly labeled nominal p values. These are alternative
definitions of the same reconstruction, not independent replications.

The age experiment refits all saved combined-age draws with their own exact
anchors and exposure. History utility, full-model residual checks and extra
10%/20% response-event deletion are separate experiments. Residual tests use
simulation/refitting calibration; deletion scenarios do not estimate missing
source events. History and residual checks retain their dedicated bootstrap
tests; Rayleigh p remains a descriptive phase-concentration statistic. Orbital
comparisons use their specified Holm adjustment of nominal LR p values. These
diagnostics save tables and notes without adding default manuscript figures.

## Run and outputs

Run from the workspace root using the project Python environment:

```bash
python Barker2011/data_pre_processing.py
python Barker2011/Barker2011_event_phase_analysis.py
python Barker2011/Barker2011_event_age_uncertainty.py
python Barker2011/Barker2011_event_uncertainty_sensitivity.py
python Barker2011/Barker2011_effect_uncertainty.py
python Barker2011/Barker2011_likelihood_bootstrap.py
python Barker2011/Barker2011_likelihood_bootstrap.py --event-definition fixed_threshold --n-bootstrap 9999 --seed 20260920
python Barker2011/Barker2011_model_diagnostics.py
python Barker2011/Barker2011_event_detection_sensitivity.py
python Barker2011/Barker2011_orbital_driver_sensitivity.py
```

For an isolated preparation check, pass `--output-dir` to
`data_pre_processing.py`. It writes only the two event CSVs and checks counts,
IDs, age order and agreement between the definitions.

Existing combined-age realizations can be reused without rerunning the sampler.
Their `age_ka_bp__Barker_S3_*` columns, IDs and saved values are unchanged by
the two-column input preparation; the existing orbital selection is unchanged.
`Barker2011_effect_uncertainty.py` reuses all 10,000 age fits, runs 5,000
full-model simulations at nominal chronology, and simulates 50 sequences
from each of 200 distinct age-specific full models. All three use the same
95% coefficient-ellipse construction and projection as NGRIP–MIS6. Sampling
gives an approximate conditional confidence region; chronology and combined
regions describe sensitivity to the assumed age errors. The figure labels
name these sources directly. Its PDF supplies manuscript Figure S7. The
`--redraw` mode uses saved replicates but requires the saved source hashes to
match. Historical Barker effect runs currently fail that check after shared
code changes; neither the check nor historical hashes are rewritten to bypass
it. Redrawing does not rerun the BG-model significance bootstrap.

`event_phase_analysis` writes nominal main and `fixed_threshold/` results,
including source IDs/roles, event phases, fitted rate samples, coefficients,
likelihood statistics, scaling and support. Other experiments save their own
replicate and summary tables under `data/processed/<script>/`, figures under
`figures/<script>/`. Generated explanatory text is kept locally under
`../agent_work/scratch/experiment_note/Barker2011/`; previous notes are in
`../archive/experiment_notes_2026-09-28/Barker2011/experiment_note/`.

New event-pass-through tables contain fewer source-description columns.
`event_age_kyr_bp` is the single nominal-age field in current calculations and
new chronology summaries; derived quantile names such as `age_q025_ka` remain.
Existing result CSVs and frozen ensembles are not rewritten to adopt this schema.

Local regression checks formerly under `Barker2011/tests/` now reside in
`../agent_work/tests/Barker2011/`. They are not analysis entry points or paper
figure producers. The directory is local, excluded from Git and from default
pytest discovery. To run these checks explicitly from the project root:

```bash
python -m pytest -q agent_work/tests/Barker2011
```

The bootstrap script defaults to varying threshold and keeps its existing
outputs. `--event-definition fixed_threshold` writes into the bootstrap
experiment's `fixed_threshold/` data and figure subdirectories; the note names
also distinguish the definition. This separation prevents a fixed-threshold
run from replacing the varying-threshold calibration.

Bootstrap exporters can rebuild the paired research PDF. Manuscript figure
synchronization follows the active figure manifest. Review runs use
`--output-root` and, where provided, `--no-paper-export` to defer this chain. The shared figure
palette assigns rose to varying threshold and green to fixed threshold;
absolute-age axes put younger ages on the right.

Current model validation is in the
[migration audit](../agent_work/reviews/continuous-time-migration-2026-09-12.md).
The complete previous project, including earlier Barker results, is preserved
in `../archive/pre_continuous_time_2026-09-12/`. The original three-catalogue
Barker workflow remains in `../archive/Barker2011_2026-09-05/`.
