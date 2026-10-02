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
- `../forcing_data_pre_processing.py` extracts the shared raw LR04, CO2 and
  orbital data into five files in `../data/processed/forcings/`: `lr04.csv`,
  `co2.csv`, `orbital.csv`, `insolation_65n.csv`, and
  `precession_phase_anchors.csv`. They retain native time nodes and full
  precision; precession/eccentricity/obliquity share 10,601 nodes, whereas
  insolation retains its separate 1,001 nodes. Analysis performs interpolation
  and model scaling. The prepared ages already use BP1950.

Analyses read the selected CSV directly. The nominal entry now exposes the
response windows, nominal scaling, event/integration features and both fits:

```python
windows = event_model.response_windows(events, observations)
scaling = event_model.nominal_scaling(climate_forcings, windows)
event_x, integral_x = event_model.build_design(
    events, windows, forcings, phase_anchors, scaling, tau=1.5,
)
background = ["intercept", "same_type_exponential_history", "lr04_scaled", "co2_scaled"]
with_phase = background + ["pre_phase_sin", "pre_phase_cos"]
```

`Barker2011_event_phase_analysis.run_analysis()` directly calls
`fit_point_process` for these two column sets; the phase fit starts at the
fitted background coefficients with two zeros appended. Its result dictionary
contains `windows`, `scaling`, `event_features`, `integration_features`,
`reduced` and `full`, alongside the existing summary and descriptive phases.
The nine saved CSVs and figure are unchanged. Sensitivity/uncertainty entries
currently retain their `combined_likelihood` interface, using the same feature
calculations and optimizer while the new nominal flow is reviewed.

The entry adds the required `Barker2011` segment constant; it does not
reconstruct the discarded source, label or duplicate-age columns. Missing
prepared files raise an error rather than triggering Excel parsing. Nominal
analysis and output writing do not read the raw event workbook, including its
bytes for provenance. All Barker analyses now use prepared inputs without
code/input hash tables. The original SOM remains the preparation source.

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
diagnostics save scientific results and print routine checks without generating
Methods/Caption notes or adding default manuscript figures.

## Run and outputs

Run from the workspace root using the project Python environment:

```bash
python forcing_data_pre_processing.py
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
the input/output cleanup. Orbital sensitivity reads the full 10,000-row age
ensemble, sorts by realization ID, and uses `default_rng(20260909).choice`
to select 500 rows without replacement. This reproduces the former 500 IDs
and their order; `selected_realizations.csv` is no longer an input or output.
The model-result tables record the selected IDs. Reproducing a draw requires
the same source ensemble, ordering, seed and selection size.
`Barker2011_effect_uncertainty.py` reuses all 10,000 age fits, runs 5,000
full-model simulations at nominal chronology, and simulates 50 sequences
from each of 200 distinct age-specific full models. All three use the same
95% coefficient-ellipse construction and projection as NGRIP–MIS6. Sampling
gives an approximate conditional confidence region; chronology and combined
regions describe sensitivity to the assumed age errors. The figure labels
name these sources directly. Its PDF supplies manuscript Figure S7. The
`--redraw` mode reads only the saved effect summary, phase-response bands and
coefficient regions. It does not refit the current inputs or compare code
hashes. `selected_age_generators.csv` stores the 200 fitted models, their
chronology IDs and supports; it is a scientific result, not a copied age input.
The nominal and age-specific models share the saved LR04/CO2 scaling table.

`event_phase_analysis` writes nine CSVs in a fresh result directory: five for
the variable-threshold fit and four under `fixed_threshold/`, plus the research
PNG/PDF. Both definitions save the following tables:

| File | Columns |
|---|---|
| `analysis_summary.csv` | 13 fields: model/definition, counts/exposure, LR/p/G, phase/rate ratio, Rayleigh results |
| `event_precession_phases.csv` | `event_id,event_age_kyr_bp,precession_index,pre_phase_rad` |
| `model_coefficients.csv` | `model_id,term,beta` for reduced and full models |
| `predictor_scaling.csv` | `forcing_id,mean,range` for LR04 and CO2 |

Only variable threshold also writes `phase_sector_fit.csv`, with
`model_id,phase_sector_center_deg,fitted_events`. Its 18-sector expected counts
retain the full intensity integral. Saved climate means/ranges retain the
meaning of the scaled coefficients; the event-history and support definitions
remain as described above.

Counts, support, likelihood/AIC, convergence and checks are printed to the
console. The nominal entry no longer writes model-parameter/provenance tables
or autogenerated Methods/Caption notes. The saved nominal directory now follows
this nine-file layout; obsolete nominal tables have been removed and retained
columns preserve their original values. Future output changes should clean up
their superseded files explicitly: the writer does not automatically delete
files. Use an isolated `--output-root` for validation runs.

The remaining result directories now contain the following files:

| Analysis | Saved results |
|---|---|
| Age generation | `age_control_points.csv`, both complete control/event realization tables, `event_age_uncertainty_summary.csv`, `sampling_settings.csv` |
| Age sensitivity | `gain_realizations.csv` (13 columns, including failed fits), `summary.csv` |
| BG bootstrap, each definition | `bootstrap_replicates.csv`, `summary.csv` |
| Effect uncertainty | replicates, summary, bands, nominal/selected generators, `predictor_scaling.csv`, scientific settings, `coefficient_regions.json` |
| Model diagnostics | `history_test.csv`, `history_replicates.csv`, `history_coefficients.csv`, `gof_summary.csv` |
| Event deletion | `replicates.csv`, `scenario_summary.csv`, `reference.csv` |
| Orbital sensitivity | `comparison_summary.csv`, `point_models.csv`, `point_coefficients.csv`, `scaling.csv`, `mc_models.csv`, `mc_comparisons.csv` |

There are 43 processed CSV/JSON files, including the two prepared event tables
and nine nominal tables. The complete saved age ensembles are unchanged;
retained scientific result cells preserve their original values. Old output
copies, hash tables and process-only tables have been removed. Future runs
write the same smaller schemas; they do not automatically delete unrelated files.

GOF directly reuses every `B_sampling` row of the effect ensemble, checking its
model settings, nominal coefficients, replicate IDs and exposure. It does not
copy the 5,000 rows into its own result directory or substitute joint draws.
`--gof-only` updates GOF without touching the independent history experiment.
The latter tests a different null model from the phase-significance bootstrap.
Deletion masks remain under
`../tests/diagnostics/Barker2011_event_detection_sensitivity/retained_event_masks.npz`;
they identify the actual retained events and are required to reproduce subsets.

Chronology `--redraw` uses saved control maps and age intervals; orbital
`--redraw` uses its comparison summary. Neither regenerates the age ensemble.
For small checks, use `--output-root` and `--no-paper-export` where available.
Barker orbital retains its existing convention: `--output-root` is the Barker
project directory itself; the other analysis entries take the workspace root
and create a `Barker2011/` subdirectory. For example:

```bash
python Barker2011/Barker2011_orbital_driver_sensitivity.py --n-realizations 3 --output-root agent_work/scratch/orbital_check/Barker2011
python Barker2011/Barker2011_effect_uncertainty.py --redraw --no-paper-export
```

The four chronology sampling settings preserve the formal ensemble's actual
seed `20260908`, sample/proposal counts and rejected crossed-control maps.
Nominal ages consistently use `event_age_kyr_bp`; derived quantile names such
as `age_q025_ka` and existing realization-column names remain unchanged.
Figures stay under `figures/<script>/`; historical notes and archives are retained.

Local regression checks formerly under `Barker2011/tests/` now reside in
`../agent_work/tests/Barker2011/`. They are not analysis entry points or paper
figure producers. The directory is local, excluded from Git and from default
pytest discovery. To run these checks explicitly from the project root:

```bash
python -m pytest -q agent_work/tests/Barker2011
```

The bootstrap script defaults to varying threshold.
`--event-definition fixed_threshold` writes its two result tables and research
figure into the experiment's `fixed_threshold/` subdirectories. This separation prevents a fixed-threshold
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
