# Held 2024 Sofular source data

These files are unmodified copies of the NOAA individual-stalagmite data
supplied for Held et al. (2024), *Nature Communications* 15, 1183,
[doi:10.1038/s41467-024-45507-5](https://doi.org/10.1038/s41467-024-45507-5).
The dataset is [doi:10.25921/b84y-cm81](https://doi.org/10.25921/b84y-cm81),
NOAA study 39063. Both file headers give a last-modified date of 2025-02-05.
`source_manifest.csv` records download URLs, original local paths, and SHA-256
checksums. The source files must retain their original units and comments.

| File | Proxy rows | Proxy age range (kyr BP) | U–Th dates | U–Th age/error units in source |
|---|---:|---:|---:|---|
| `held2024-so-4.txt` | 2,034 | 53.27028–205.0714 | 24 | kyr |
| `held2024-so-57.txt` | 854 | 170.9862–192.6717 | 7 | years |

The four proxy columns are `depth_mm`, `age_kaBP`, `d13C`, and `d18O`.
For both files, proxy ages are kyr BP and represent published StalAge model
ages. The comment-prefixed chronology table contains dated depths, corrected
U–Th ages relative to AD 1950, and their reported 2σ errors. So-57 chronology
ages and errors must be divided by 1,000; its proxy ages must not. The
analytical working standard deviation is the converted 2σ error divided by
two. No additional calendar-origin shift is applied.

So-4 has a declared hiatus at 781.5 mm. The MIS 6 growth segment contains ten
dated depths from 804 to 1221 mm. One date in the younger growth segment,
So4-M19 at 771 mm, lies 0.2 mm beyond that segment's final proxy measurement.
It remains in the complete control diagnostics but has no interpolated model
age and is not used as a warp knot. No interpolation crosses the hiatus and
no model age is extrapolated beyond a proxy segment. So-57 has 22 repeated
published ages near 172.154 kyr BP at the source's decimal precision; these
values remain unchanged. A projection exactly onto a repeated age is rejected
because its depth is not unique.

`MIS6/sofular_chronology.py` loads the full tables and constructs a working
propagation of U–Th analytical error. At each dated depth, it perturbs the
published model age by an independent normal error with the reported local
standard deviation. A piecewise affine transformation of published age
between dated depths preserves the reference growth-curve shape. An entire
proposal is rejected if the perturbed dated knots are not strictly ordered.
Events sharing dated knots consequently share some chronology error; events
are not all shifted by one common random factor.

The main MIS 6 ensemble uses So-4 for all five Sofular events. The separately
saved component sensitivity uses So-57 for events 6.9–6.12 and So-4 for 6.13,
which is older than the So-57 proxy record. The complete So-57 table includes
dates So57-1 and So57-2, so event 6.9 is no longer outside its dated support.
The older 11-row table in `data/curated/mis6_age_control_points.csv` is retained
for the source-record figure; it is not the computational source of the new
Sofular uncertainty model.

The original event point estimates remain on the published iscam stack.
Projecting a stack age onto an individual StalAge age–depth curve identifies
an assumed coordinate in its error field, not a verified physical event
layer. The saved −1, −0.5, 0, +0.5, and +1 kyr mapping variants move this
projection coordinate to diagnose changes in analytical covariance. They do
not shift the stack point ages or simulate a real inter-record climate lag.

This is analytical-error propagation conditional on the published curves and
the coordinate projection. It does not reconstruct StalAge or iscam posterior
chronologies, unknown growth-rate variation between dates, laboratory-wide
systematic errors, proxy measurement noise, or the original stack alignment
uncertainty. Published model ages need not equal measured U–Th central ages;
the complete control diagnostics retain these residuals explicitly. Analytic
standard deviations and covariance matrices describe proposals before the
monotonicity and event-order conditions; accepted Monte Carlo spreads can
differ.
