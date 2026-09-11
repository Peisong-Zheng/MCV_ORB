# MIS 6 event-age uncertainty outputs

All ages are **kyr before 1950 CE**. There are two working uncertainty schemes:
the primary scheme transfers So-4 control errors to all five Sofular events;
the `so57_overlap` alternative uses So-57 for MIS 6.17–20 and retains So-4 for
MIS 6.21, which lies outside So-57 support. Both use the same nominal stack
events and the same MF uncertainty method.

## Research files

| Files | Contents |
| --- | --- |
| `mis6_event_age_uncertainty_summary.csv` | Primary event summary: 21 rows, 10 columns. Start here for manual inspection. |
| `mis6_event_age_uncertainty_summary_so57_overlap.csv` | The same summary for the alternative scheme. |
| `mis6_event_age_realizations.csv` | Primary ensemble: 10,000 complete, ordered 21-event sequences. Used by the pooled PI analysis. |
| `mis6_event_age_realizations_so57_overlap.csv` | Alternative ensemble, used by the So-57 PI sensitivity analysis. |
| `parameters_and_provenance.csv` | Primary settings, seed, sources, error assumptions and rejection counts. |
| `parameters_and_provenance_so57_overlap.csv` | The corresponding metadata for the alternative. |

Each parameter table has 25 rows. It records the actual sampling counts and
seed, input-envelope checks, source files, error conversions, component rule
and excluded uncertainties. The notebook Markdown explains the methods in
more detail; event summaries and joint ensembles retain their existing schemas.

Each ensemble has `realization_id` plus one age column per event. Keep the
21 event columns together: a row is a joint realization, and independently
resampling columns would discard the chronology dependence.

## Summary columns

| Column(s) | Meaning |
| --- | --- |
| `composite_event_id`, `composite_event_label`, `source_record` | Stable identifier, display label and source record (MF or Sofular). The original source labels are in the composite-event catalogue. |
| `nominal_event_age_ka_bp` | Age from the nominal event-picking settings. |
| `definition_age_min_ka_bp`, `definition_age_max_ka_bp` | Minimum and maximum picks across the nine detector settings; this is a method-sensitivity range. |
| `chronology_sigma_nominal_ka` | Working chronology standard deviation at the nominal pick, before conditioning on monotone curves and event order. It excludes definition uncertainty. |
| `sampled_age_q025_ka_bp`, `sampled_age_median_ka_bp`, `sampled_age_q975_ka_bp` | 2.5th percentile, median and 97.5th percentile of accepted ages, combining definition and chronology uncertainty. |

The MC ranges describe the specified sensitivity ensemble, not a complete
chronology posterior. In particular, the Sofular stack contains both So-4 and
So-57; transferring individual-stalagmite analytical errors does not recover
the published iscam stack's full age uncertainty.

## Figures and detailed checks

- `MIS6/figures/MIS6_event_age_uncertainty/`: primary age offsets and correlations.
- `MIS6/figures/Sofular_chronology_comparison/`: So-4/So-57 U–Th controls and
  the five Sofular event-age ranges under both schemes.
- `MIS6/tests/diagnostics/MIS6_event_age_uncertainty/`: full summaries, individual
  detector picks, dated-depth diagnostics, projections and covariance checks.

Paths above are relative to the project root. Run
`jupyter nbconvert --to notebook --execute --inplace MIS6/MIS6_event_age_uncertainty.ipynb` to regenerate both schemes and
their figures. To update only the comparison figure from saved results, run
`python MIS6/Sofular_chronology_comparison.py`.
