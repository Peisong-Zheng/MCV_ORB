# NGRIP event ages

This directory prepares Greenland Interstadial (GI, warming) and Greenland
Stadial (GS, cooling) starts from
[Rasmussen et al. (2014), Table 2](https://doi.org/10.1016/j.quascirev.2014.09.007).
Warming starts enter the joint NGRIP–MIS6 analysis described in the
[project README](../README.md). Both transition types constrain the ordering
of sampled chronologies.

## Method

Preparation converts ages from b2k (relative to 2000 CE) to BP1950 by subtracting
0.05 kyr. The age sampler combines independent Gaussian event-definition
errors with correlated chronology shifts. These shifts accumulate on a time
grid and are interpolated to event ages. Proposals that reverse chronology
nodes or event order are rejected.

Chronology standard deviations use half the GICC05 maximum counting error
in the counted section and half an assumed envelope in the model extension.
These are working uncertainty distributions, not hard age bounds. The
[supplement, Text S3](../orbital_event_paper/SI.tex) describes their construction.
A separate analysis compares warming and cooling onsets with the shared
continuous-time event model, bootstrap tests and saved age realizations.

## Files and use

| Entry | Purpose |
|---|---|
| [ngrip_data_preparation.ipynb](ngrip_data_preparation.ipynb) | Prepare event ages, definition errors and chronology grids. |
| [ngrip_event_age_uncertainty.py](ngrip_event_age_uncertainty.py) | Sample ordered event-age realizations. |
| [ngrip_transition_phase_sensitivity.py](ngrip_transition_phase_sensitivity.py) | Compare the two transition directions. |

Run the preparation notebook and age sampler **from `NGRIP/`**, since their
input paths are relative to that directory. The transition script also runs
from the project root. Edit settings near each script’s top.

Prepared inputs are in `data/processed/`. The age ensemble is
[`ngrip_event_age_realizations.csv`](data/processed/ngrip_event_age_uncertainty/ngrip_event_age_realizations.csv);
its event-age columns use kyr BP1950. Analysis tables and figures are stored
under `data/processed/<script_name>/` and `figures/<script_name>/`.
