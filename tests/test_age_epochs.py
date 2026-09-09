"""Check source epochs and prevent a second 50-year shift in shared loaders."""

import numpy as np
import pandas as pd
import pytest

from toolbox import combined_pi, event_inputs, orbital_phase
from toolbox.project_config import (
    AGE_EPOCH,
    B2K_TO_BP1950_KA,
    CO2_XLSX,
    LR04_XLSX,
    OBL_TXT,
    ORBITAL_AGE_OFFSET_TO_BP1950_KA,
    ORBITAL_DRIVER_SETTINGS,
    PRE_TXT,
)


@pytest.mark.parametrize("path,driver,expected", [
    (PRE_TXT, "pre", [-0.033760490135305736, -0.03833572638169867,
                      -0.0006501053038830341, 0.01627964595166013]),
    (OBL_TXT, "obl", [0.4123891471500025, 0.4037251011418818,
                      0.4130265591582125, 0.4090928042223415]),
])
def test_local_raw_orbital_knots_match_official_j2000_solution(path, driver, expected):
    # Independent official INSOLN.LA2004.BTL.ASC values, downloaded 2026-09-05.
    # pre=e*sin(pibar); obl=eps. Local files retain six decimal places.
    raw = pd.read_csv(path, sep=r"\s+", header=None, names=["time", "value"])
    observed = raw.set_index("time").loc[[-1000., -200., -100., 0.], "value"]
    np.testing.assert_allclose(observed, expected, atol=5.01e-7, rtol=0)

    series = orbital_phase.load_orbital_series(path, driver, driver)
    for source_time, value in zip([-1000., -200., -100., 0.], observed):
        # This physical instant is 0.05 kyr younger on BP1950 than on b2k.
        row = series.loc[np.isclose(series.age_ka, -source_time - 0.05)]
        assert len(row) == 1
        assert row.value.iloc[0] == value
    assert series.attrs["age_epoch"] == AGE_EPOCH
    assert series.attrs["source_age_epoch"] == "J2000.0"


def test_binning_and_event_phase_paths_apply_the_same_single_epoch_shift():
    ages = np.array([14.642, 50.0, 150.0, 194.238])
    series = orbital_phase.build_phase_series("pre", ORBITAL_DRIVER_SETTINGS["pre"])
    expected = orbital_phase.evaluate_phase_at_ages(ages, series.extrema)
    binned, _ = event_inputs.build_precession_phase(ages, PRE_TXT)
    np.testing.assert_allclose(binned.pre_phase_unwrapped_rad,
                               expected.phase_unwrapped_rad, atol=1e-12, rtol=0)
    assert ORBITAL_AGE_OFFSET_TO_BP1950_KA == B2K_TO_BP1950_KA == -0.05


def test_bp1950_climate_covariates_are_not_shifted():
    lr04 = pd.read_excel(LR04_XLSX).iloc[[1, 51, 151]]
    _, lr_info = event_inputs.load_lr04(lr04.iloc[:, 0].to_numpy(), LR04_XLSX)
    np.testing.assert_allclose(lr_info["raw"], lr04.iloc[:, 1], atol=1e-12, rtol=0)
    co2 = pd.read_excel(CO2_XLSX, sheet_name="Sheet2").iloc[[2, 501, 1001]]
    _, co_info = event_inputs.load_co2(co2.iloc[:, 0].to_numpy()/1000., CO2_XLSX)
    np.testing.assert_allclose(co_info["raw"], co2.iloc[:, 1], atol=1e-12, rtol=0)
    for info in [lr_info, co_info]:
        assert info["meta"]["source_age_epoch"] == "BP1950"
        assert info["meta"]["age_offset_ka"] == 0.0


def test_curated_ngrip_and_pooled_ages_have_exactly_one_b2k_conversion():
    source = pd.read_csv(combined_pi.PROJECT_ROOT / "NGRIP/data/processed/ngrip_warming_cooling_starts.csv")
    np.testing.assert_allclose(source.age_ka_bp, source.age_ka_b2k-0.05,
                               atol=1e-10, rtol=0)
    pooled = combined_pi.load_event_catalogue().query("segment_id == 'NGRIP'")
    merged = pooled.merge(source[["event_label", "age_ka_bp"]], on="event_label",
                          validate="one_to_one")
    assert len(merged) == 34
    np.testing.assert_allclose(merged.event_age_kyr_bp, merged.age_ka_bp,
                               atol=1e-10, rtol=0)
