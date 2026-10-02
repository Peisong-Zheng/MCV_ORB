# MCV_ORB

Research code for testing whether orbital precession phase helps explain abrupt
warming occurrence after accounting for background climate and recent events.

## Method

The primary catalogue combines NGRIP Greenland warming onsets with a MIS 6
speleothem sequence from Melchsee–Frutt and Sofular. The synthetic Greenland
reconstruction of Barker et al. (2011) provides a separate comparison using
its published variable- and fixed-threshold event definitions.

We fit a continuous-time conditional event rate by maximum likelihood:

$$
\log\lambda_r(t)=\beta_0+\beta_L L_r(t)+\beta_C C_r(t)
+\beta_H H_r(t)+\beta_S S_r+\beta_s\sin\phi_r(t)+\beta_c\cos\phi_r(t).
$$

Here, $L$ and $C$ are the LR04 benthic δ¹⁸O stack and atmospheric CO₂, and $S$
allows NGRIP and MIS 6 to have different baseline rates. Climate predictors
are centered and range-scaled over the nominal fitted intervals; these scales
remain fixed in sensitivity analyses. Earlier events contribute to $H$ with
exponentially decreasing weights (decay time 1.5 kyr). Its coefficient is
nonpositive, allowing recent events to suppress the rate. Phase $\phi$ comes
from La2004 precession-index extrema: minima are 0°, maxima 180°, and phase
increases toward older ages.

Each independent record starts at its oldest event, which initializes history
but is excluded from the fitted event count. The fit includes all subsequent
time to the younger observation boundary, including intervals without events.
History does not cross gaps between records; unknown pre-record history is set
to zero. The likelihood sums log rates at observed events and subtracts the
integrated rate, evaluated with Gauss–Legendre quadrature.

The phase test compares models with and without the paired sine/cosine terms,
refitting both. We report likelihood-ratio tests and fit gain in bits per
response event; simulations from the background model calibrate the main
phase test. Separate analyses perturb event chronologies, simulate new event
sequences from the full model for sampling uncertainty, and combine these sources of
uncertainty. Further checks vary event selection, history assumptions, climate
responses, and orbital predictors. These comparisons describe conditional
association, not a causal effect.

## Run the main analyses

Prepared event and climate inputs are included. From the repository root:

```bash
python -m pip install -r requirements.txt
python NGRIP_MIS6_event_phase_analysis.py
python Barker2011/Barker2011_event_phase_analysis.py
```

Edit scientific settings near the top of each script. Analysis tables and
figures are saved under `data/processed/` and `figures/` within the corresponding
project, grouped by script name. The uncertainty and sensitivity scripts reuse
saved ensembles where applicable. Tests can be run with `python -m pytest -q`.

## Find your way around

| Location | Contents |
|---|---|
| [NGRIP](NGRIP/README.md) | Greenland event preparation and chronology uncertainty |
| [MIS6](MIS6/README.md) | Speleothem event timing, composite catalogue and age ensembles |
| [Barker2011](Barker2011/README.md) | Synthetic Greenland event analysis and sensitivities |
| [Curated inputs](data/curated/README.md) | Combined catalogue, observation windows and age conventions |
| [toolbox](toolbox/) | Shared fitting, uncertainty calculations and plotting |
| [Forcing preparation](forcing_data_pre_processing.py) | Preparation of shared climate and orbital CSVs |

The [manuscript](orbital_event_paper/main.tex) and
[Supporting Information](orbital_event_paper/SI.tex) give the full methods and
source references. Ages in the analyses use kyr BP relative to AD 1950; source
conversions and working assumptions are recorded in the
[epoch table](data/curated/age_epoch_audit.csv).
