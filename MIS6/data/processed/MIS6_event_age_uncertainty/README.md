# MIS 6 event-age ensembles

[The uncertainty notebook](../../../MIS6_event_age_uncertainty.ipynb) generates
these files in kyr before 1950 CE:

| Files | Contents |
| --- | --- |
| `mis6_event_age_realizations*.csv` | Joint sequences: one realization per row, with `realization_id` and event-age columns. |
| `mis6_event_age_uncertainty_summary*.csv` | Nominal ages, definition ranges, chronology standard deviations, and accepted-age medians and 2.5th–97.5th percentiles. |
| `parameters_and_provenance*.csv` | Sampling settings, sources, and error assumptions. |

Unsuffixed files use So-4 controls for all Sofular events. The `so57_overlap`
alternative uses So-57 for composite events 6.17–6.20, retaining So-4 for 6.21
beyond So-57 coverage.

Keep each realization's columns together: independent resampling would discard
chronology dependence. Percentiles describe age sensitivity, not a complete
chronology posterior or confidence intervals for the fitted phase effect.
