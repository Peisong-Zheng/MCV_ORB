"""Unadjusted orbital covariation over 0–200 kyr BP1950; no significance test.

Run from the project root. Insolation is the project's 65N summer-solstice
DAILY mean, not a JJA mean. Precession is the raw index, not its phase angle.
"""
from pathlib import Path
import json

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
pre = np.loadtxt(ROOT / 'data/raw/pre_1000_60_inter100.txt')
# Both source coordinates refer to J2000; harmonize to BP1950.
pre_age = -pre[:, 0] - 0.05
with xr.open_dataset(ROOT / 'data/raw/solstice_insolation_NH.nc') as ds:
    assert np.isclose(float(ds.solar_longitude_deg.item()), 90.0)
    insol = ds.swap_dims({'latitude': 'latitude_degN'}).daily_mean_insolation_Wm2.sel(latitude_degN=65.0).to_numpy()
    insol_age = ds.age_kyr_BP.to_numpy() - 0.05

age = np.linspace(0, 200, 2001)  # Equal temporal weighting at 100-year spacing.
series = []
for t, values in [(pre_age, pre[:, 1]), (insol_age, insol)]:
    order = np.argsort(t)
    t, values = t[order], values[order]
    assert np.all(np.diff(t) > 0)
    assert t[0] <= age[0] and t[-1] >= age[-1]
    assert np.isfinite(values).all()
    series.append(np.interp(age, t, values))
r = float(np.corrcoef(*series)[0, 1])
result = dict(interval_kyr_BP1950=[0, 200], spacing_kyr=0.1, n_time_points=len(age),
              insolation='65N summer-solstice daily mean', precession='raw index',
              pearson_r=r, r_squared=r*r, shared_linear_variance_percent=100*r*r,
              interpretation='Unadjusted bivariate linear association; not mutual information or independent sample inference.')
(OUTPUT / 'correlation_200kyr.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
