# MCV_ORB

Research code for testing whether orbital precession phase helps explain abrupt
warming occurrence after accounting for background climate and recent events.

## Method

The primary catalogue combines NGRIP Greenland warming onsets
([Rasmussen et al., 2014](https://doi.org/10.1016/j.quascirev.2014.09.007))
with a MIS 6 speleothem sequence from Melchsee–Frutt
([Fohlmeister et al., 2023](https://doi.org/10.1038/s43247-023-00908-0))
and Sofular ([Held et al., 2024](https://doi.org/10.1038/s41467-024-45507-5)).
The synthetic Greenland reconstruction of
[Barker et al. (2011)](https://doi.org/10.1126/science.1203580) provides a separate
comparison using its published variable- and fixed-threshold event definitions.

We fit a continuous-time conditional event rate by maximum likelihood
([Truccolo et al., 2005](https://doi.org/10.1152/jn.00697.2004)):

$$
\log\lambda_r(t)=\beta_0+\beta_L L_r(t)+\beta_C C_r(t)
+\beta_H H_r(t)+\beta_S S_r+\beta_s\sin\phi_r(t)+\beta_c\cos\phi_r(t).
$$

Here, $L$ and $C$ are the LR04 benthic δ¹⁸O stack
([Lisiecki and Raymo, 2005](https://doi.org/10.1029/2004pa001071)) and atmospheric
CO₂ ([Bereiter et al., 2015](https://doi.org/10.1002/2014gl061957)), and $S$
allows NGRIP and MIS 6 to have different baseline rates. Climate predictors
are centered and range-scaled over the nominal fitted intervals; these scales
remain fixed in sensitivity analyses. Earlier events contribute to $H$ with
exponentially decreasing weights (decay time 1.5 kyr). Its coefficient is
nonpositive, allowing recent events to suppress the rate. We define phase
$\phi$ from the extrema of the La2004 precession index
([Laskar et al., 2004](https://doi.org/10.1051/0004-6361:20041335)): minima are 0°,
maxima 180°, and phase increases toward older ages.

Each independent record starts at its oldest event, which initializes history
but is excluded from the fitted event count. The fit includes all subsequent
time to the younger observation boundary, including intervals without events.
History does not cross gaps between records; unknown pre-record history is set
to zero. The likelihood sums log rates at observed events and subtracts the
integrated rate ([Daley and Vere-Jones, 2003](https://doi.org/10.1007/b97277)).
We evaluate the integral with Gauss–Legendre quadrature.

The phase test compares models with and without the paired sine/cosine terms,
refitting both. We report likelihood-ratio tests and fit gain in bits per
response event; a parametric bootstrap
([Davison and Hinkley, 1997](https://doi.org/10.1017/CBO9780511802843)) simulates
sequences from the background model to calibrate the main phase test.
Separate analyses perturb event chronologies, simulate new event
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

## References

- Rasmussen, S. O., et al. (2014). [A stratigraphic framework for abrupt climatic changes during the Last Glacial period based on three synchronized Greenland ice-core records: refining and extending the INTIMATE event stratigraphy](https://doi.org/10.1016/j.quascirev.2014.09.007). *Quaternary Science Reviews*, 106, 14–28.
- Fohlmeister, J., et al. (2023). [The role of Northern Hemisphere summer insolation for millennial-scale climate variability during the penultimate glacial](https://doi.org/10.1038/s43247-023-00908-0). *Communications Earth & Environment*, 4, 245.
- Held, F., et al. (2024). [Dansgaard-Oeschger cycles of the penultimate and last glacial period recorded in stalagmites from Türkiye](https://doi.org/10.1038/s41467-024-45507-5). *Nature Communications*, 15, 1183.
- Barker, S., et al. (2011). [800,000 years of abrupt climate variability](https://doi.org/10.1126/science.1203580). *Science*, 334, 347–351.
- Truccolo, W., et al. (2005). [A point process framework for relating neural spiking activity to spiking history, neural ensemble, and extrinsic covariate effects](https://doi.org/10.1152/jn.00697.2004). *Journal of Neurophysiology*, 93, 1074–1089.
- Lisiecki, L. E., and Raymo, M. E. (2005). [A Pliocene-Pleistocene stack of 57 globally distributed benthic δ¹⁸O records](https://doi.org/10.1029/2004pa001071). *Paleoceanography*, 20, PA1003.
- Bereiter, B., et al. (2015). [Revision of the EPICA Dome C CO₂ record from 800 to 600 kyr before present](https://doi.org/10.1002/2014gl061957). *Geophysical Research Letters*, 42, 542–549.
- Laskar, J., et al. (2004). [A long-term numerical solution for the insolation quantities of the Earth](https://doi.org/10.1051/0004-6361:20041335). *Astronomy & Astrophysics*, 428, 261–285.
- Daley, D. J., and Vere-Jones, D. (2003). [An Introduction to the Theory of Point Processes: Volume I: Elementary Theory and Methods](https://doi.org/10.1007/b97277) (2nd ed.). Springer.
- Davison, A. C., and Hinkley, D. V. (1997). [Bootstrap Methods and their Application](https://doi.org/10.1017/CBO9780511802843). Cambridge University Press.
