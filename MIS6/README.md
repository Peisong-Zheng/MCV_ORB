# MIS 6 speleothem events

This subproject combines warming onsets from the Melchsee–Frutt (MF)
oxygen-isotope stack ([Fohlmeister et al., 2023](https://doi.org/10.1038/s43247-023-00908-0))
and an older continuation from the Sofular carbon-isotope record
([Held et al., 2024](https://doi.org/10.1038/s41467-024-45507-5)). Overlapping
features identify corresponding events without retuning either chronology.

Published event identities are retained. Onset ages are assigned from the
steepest proxy change in the expected direction near each published label,
after Gaussian smoothing. Nine smoothing/search-window combinations test onset
placement, with one setting applied to each record within a realization.

Chronology uncertainty uses a shared Gaussian factor for MF, with standard
deviations derived from its published age bounds, and interpolated U–Th control
errors for Sofular. Whole proposals are rejected if chronology or event order reverses.
The MF dependence is assumed; the Sofular procedure transfers analytical errors
through published curves without reconstructing the full stack chronology
posterior. These are age-sensitivity ensembles, not complete uncertainty estimates.

Start with these notebooks, using `MIS6/` as the working directory:

- [Published records](Speleothem_published_event_plot.ipynb): compare the source
  series and event correspondences.
- [Composite catalogue](MIS6_composite_event_record.ipynb): assign onset ages.
- [Age uncertainty](MIS6_event_age_uncertainty.ipynb): generate joint age
  realizations for the pooled analysis.

Results and figures are under `data/processed/` and `figures/`, in directories
named after each notebook. Ages are kyr before 1950 CE. See the
[ensemble guide](data/processed/MIS6_event_age_uncertainty/README.md) for the
saved files and alternative Sofular chronology scheme.
