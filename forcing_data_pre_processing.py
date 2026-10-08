"""Prepare native climate/orbital samples for the event analyses.

Run once from the project root. All output ages are kyr BP1950; no smoothing,
resampling or model scaling is applied. Analysis scripts interpolate these
native samples onto their own event and integration ages.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import xarray as xr

from toolbox import project_config as config

OUTPUT_DIR = config.FORCING_DIR


def precession_phase_anchors(orbital):
    """Minima are phase 0, maxima pi; successive extrema advance by pi."""
    values = orbital.precession_index.to_numpy(float)
    maxima = orbital.iloc[find_peaks(values)[0]].assign(extremum_type="maximum")
    minima = orbital.iloc[find_peaks(-values)[0]].assign(extremum_type="minimum")
    extrema = pd.concat([minima, maxima], ignore_index=True).sort_values("age_kyr_bp")

    # Collapse adjacent extrema of the same type to the more extreme point.
    # This is the original phase convention, independent of the event catalogue.
    rows = []
    for _, row in extrema.iterrows():
        if not rows or row.extremum_type != rows[-1].extremum_type:
            rows.append(row.copy())
            continue
        previous = rows[-1]
        if ((row.extremum_type == "maximum" and row.precession_index > previous.precession_index)
                or (row.extremum_type == "minimum" and row.precession_index < previous.precession_index)):
            rows[-1] = row.copy()
    if len(rows) < 3:
        raise ValueError("Too few precession extrema to define phase")
    extrema = pd.DataFrame(rows).reset_index(drop=True)
    first_phase = 0.0 if extrema.extremum_type.iloc[0] == "minimum" else np.pi
    return pd.DataFrame({
        "age_kyr_bp": extrema.age_kyr_bp,
        "phase_unwrapped_rad": first_phase + np.arange(len(extrema)) * np.pi,
    })


def prepare_forcings(output_dir=config.FORCING_DIR):
    # LR04 and the CO2 composite retain their BP1950 nodes. Only CO2 age units
    # change, from years to kyr; missing age/value pairs do not define samples.
    lr04 = pd.read_excel(config.LR04_XLSX).rename(columns={
        "Time (ka)": "age_kyr_bp", "Benthic d18O (per mil)": "lr04",
    })[["age_kyr_bp", "lr04"]]
    lr04 = lr04.loc[np.isfinite(lr04).all(axis=1)].sort_values("age_kyr_bp").reset_index(drop=True)
    co2 = pd.read_excel(config.CO2_XLSX, sheet_name="Sheet2")
    co2.columns = co2.columns.str.strip()
    co2 = co2.rename(columns={"Gasage (yr BP)": "age_kyr_bp", "CO2 (ppmv)": "co2_ppm"})[
        ["age_kyr_bp", "co2_ppm"]]
    co2["age_kyr_bp"] = co2["age_kyr_bp"] / 1000
    co2 = co2.loc[np.isfinite(co2).all(axis=1)].sort_values("age_kyr_bp").reset_index(drop=True)
    tables = {"lr04.csv": lr04, "co2.csv": co2}

    # La2004 (Laskar et al., 2004): signed time since J2000 -> positive BP1950.
    # Retain the original decimal parser used for the precession source.
    raw = pd.read_csv(config.PRE_TXT, sep=r"\s+", header=None, names=["time", "value"])
    orbital = pd.DataFrame({
        "age_kyr_bp": -pd.to_numeric(raw.time, errors="coerce") + config.ORBITAL_AGE_OFFSET_TO_BP1950_KA,
        "precession_index": pd.to_numeric(raw.value, errors="coerce"),
    }).dropna().sort_values("age_kyr_bp").reset_index(drop=True)
    anchors = precession_phase_anchors(orbital)
    for name, path in (("eccentricity", config.ECC_TXT), ("obliquity_deg", config.OBL_TXT)):
        raw = np.loadtxt(path)
        ages = -raw[:, 0] + config.ORBITAL_AGE_OFFSET_TO_BP1950_KA
        values = np.rad2deg(raw[:, 1]) if name == "obliquity_deg" else raw[:, 1]
        order = np.argsort(ages)
        if not np.array_equal(ages[order], orbital.age_kyr_bp.to_numpy()):
            raise ValueError(f"{name} and precession have different native age nodes")
        orbital[name] = values[order]
    tables["orbital.csv"] = orbital
    tables["precession_phase_anchors.csv"] = anchors

    # Exact 65 N, solar longitude 90 degrees, W m-2. The NetCDF's age coordinate
    # is J2000 despite its BP name (checked against the original La2004 inputs).
    with xr.open_dataset(config.INSOLATION_NC) as ds:
        if not np.isclose(float(ds.solar_longitude_deg.item()), 90, rtol=0, atol=1e-10):
            raise ValueError("Expected northern summer-solstice insolation")
        if ds.daily_mean_insolation_Wm2.attrs.get("units") != "W m-2":
            raise ValueError("Expected insolation in W m-2")
        values = ds.swap_dims({"latitude": "latitude_degN"}).daily_mean_insolation_Wm2.sel(latitude_degN=65.0).to_numpy()
        ages = ds.age_kyr_BP.to_numpy() + config.ORBITAL_AGE_OFFSET_TO_BP1950_KA
    tables["insolation_65n.csv"] = pd.DataFrame({
        "age_kyr_bp": ages, "insolation_Wm2": values,
    }).sort_values("age_kyr_bp")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        if len(table) < 2 or not np.isfinite(table.to_numpy()).all():
            raise ValueError(f"Missing or non-finite native samples: {name}")
        if np.any(np.diff(table.age_kyr_bp) <= 0):
            raise ValueError(f"Native ages must be unique and increasing: {name}")
        table.to_csv(output_dir / name, index=False, float_format="%.17g")
        print(f"{name}: {len(table)} native nodes, {table.age_kyr_bp.iloc[0]:g}–{table.age_kyr_bp.iloc[-1]:g} kyr BP1950")
    return tables


def main():
    prepare_forcings(OUTPUT_DIR)


if __name__ == "__main__":
    main()
