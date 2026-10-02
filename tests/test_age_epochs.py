"""Check the actual raw -> prepared CSV -> analysis age/phase path."""

import numpy as np
import pandas as pd
import pytest

import forcing_data_pre_processing as preparation
from toolbox import event_model
from toolbox.project_config import (
    PROJECT_ROOT, EVENT_CATALOGUE_CSV, B2K_TO_BP1950_KA, CO2_XLSX, FORCING_DIR, LR04_XLSX, OBL_TXT,
    ORBITAL_AGE_OFFSET_TO_BP1950_KA, PRE_TXT,
)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    folder = tmp_path_factory.mktemp("prepared_forcings")
    preparation.prepare_forcings(folder)
    return folder


def test_preparation_preserves_every_saved_native_sample(prepared):
    # Every native node, including the unwrapped phase anchors, survives the
    # change of file format. No model grid or hidden resampling belongs here.
    names = ["lr04.csv", "co2.csv", "orbital.csv", "insolation_65n.csv",
             "precession_phase_anchors.csv"]
    for name in names:
        saved = pd.read_csv(FORCING_DIR / name, float_precision="round_trip")
        rebuilt = pd.read_csv(prepared / name, float_precision="round_trip")
        pd.testing.assert_frame_equal(rebuilt, saved, check_exact=True)


@pytest.mark.parametrize("path,column,expected", [
    (PRE_TXT, "precession_index", [-0.033760490135305736, -0.03833572638169867,
                                   -0.0006501053038830341, 0.01627964595166013]),
    (OBL_TXT, "obliquity_deg", [0.4123891471500025, 0.4037251011418818,
                               0.4130265591582125, 0.4090928042223415]),
])
def test_local_raw_orbital_knots_match_official_j2000_solution(prepared, path, column, expected):
    # Independent official INSOLN.LA2004.BTL.ASC values, downloaded 2026-09-05.
    # pre=e*sin(pibar); obl=eps in radians. Local files retain six decimals.
    raw = pd.read_csv(path, sep=r"\s+", header=None, names=["time", "value"])
    observed = raw.set_index("time").loc[[-1000., -200., -100., 0.], "value"]
    np.testing.assert_allclose(observed, expected, atol=5.01e-7, rtol=0)
    orbital = pd.read_csv(prepared / "orbital.csv", float_precision="round_trip")
    for source_time, value in zip([-1000., -200., -100., 0.], observed):
        row = orbital.loc[np.isclose(orbital.age_kyr_bp, -source_time - 0.05)]
        assert len(row) == 1
        expected_value = np.rad2deg(value) if column == "obliquity_deg" else value
        assert row[column].item() == expected_value


def test_prepared_and_event_phase_paths_apply_the_same_single_epoch_shift(prepared):
    anchors = pd.read_csv(prepared / "precession_phase_anchors.csv", float_precision="round_trip")
    orbital = pd.read_csv(prepared / "orbital.csv", float_precision="round_trip")
    phase_anchors = (anchors.age_kyr_bp.to_numpy(), anchors.phase_unwrapped_rad.to_numpy())
    ages = np.array([14.642, 50.0, 150.0, 194.238])
    phase, extrapolated = event_model.interpolate_unwrapped_phase(ages, *phase_anchors)
    expected = np.interp(ages, anchors.age_kyr_bp, anchors.phase_unwrapped_rad)
    np.testing.assert_array_equal(phase, expected)
    assert not extrapolated.any()
    events = pd.DataFrame(dict(event_id=["a", "b", "c", "d"], event_age_kyr_bp=ages))
    actual = event_model.sample_event_phases(events,
        (orbital.age_kyr_bp.to_numpy(), orbital.precession_index.to_numpy()), phase_anchors)
    np.testing.assert_array_equal(actual.pre_phase_unwrapped_rad, phase)
    assert ORBITAL_AGE_OFFSET_TO_BP1950_KA == B2K_TO_BP1950_KA == -0.05


def test_phase_anchors_and_extrapolation_keep_the_minimum_maximum_convention():
    orbital = pd.DataFrame(dict(age_kyr_bp=np.arange(9.),
                                precession_index=[0., 2., 0., -1., 0., 3., 0., -2., 0.]))
    anchors = preparation.precession_phase_anchors(orbital)
    np.testing.assert_array_equal(anchors.age_kyr_bp, [1., 3., 5., 7.])
    np.testing.assert_array_equal(anchors.phase_unwrapped_rad, np.arange(1., 5.) * np.pi)
    phase, extrapolated = event_model.interpolate_unwrapped_phase(
        np.array([-1., 1., 3., 7., 9.]), anchors.age_kyr_bp, anchors.phase_unwrapped_rad)
    np.testing.assert_allclose(phase, np.array([0., 1., 2., 4., 5.]) * np.pi, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(extrapolated, [True, False, False, False, True])


def test_bp1950_climate_covariates_are_not_shifted(prepared):
    lr04 = pd.read_excel(LR04_XLSX).iloc[[1, 51, 151]]
    prepared_lr04 = pd.read_csv(prepared / "lr04.csv", float_precision="round_trip")
    actual = event_model.interpolate_checked(
        lr04.iloc[:, 0].to_numpy(), prepared_lr04.age_kyr_bp.to_numpy(),
        prepared_lr04.lr04.to_numpy(), context="LR04 epoch test")
    np.testing.assert_allclose(actual, lr04.iloc[:, 1], atol=1e-12, rtol=0)
    co2 = pd.read_excel(CO2_XLSX, sheet_name="Sheet2").iloc[[2, 501, 1001]]
    prepared_co2 = pd.read_csv(prepared / "co2.csv", float_precision="round_trip")
    actual = event_model.interpolate_checked(
        co2.iloc[:, 0].to_numpy() / 1000., prepared_co2.age_kyr_bp.to_numpy(),
        prepared_co2.co2_ppm.to_numpy(), context="CO2 epoch test")
    np.testing.assert_allclose(actual, co2.iloc[:, 1], atol=1e-12, rtol=0)


def test_curated_ngrip_and_pooled_ages_have_exactly_one_b2k_conversion():
    source = pd.read_csv(PROJECT_ROOT / "NGRIP/data/processed/ngrip_warming_cooling_starts.csv")
    np.testing.assert_allclose(source.age_ka_bp, source.age_ka_b2k-0.05,
                               atol=1e-10, rtol=0)
    pooled = pd.read_csv(EVENT_CATALOGUE_CSV).query("segment_id == 'NGRIP'")
    merged = pooled.merge(source[["event_label", "age_ka_bp"]], on="event_label",
                          validate="one_to_one")
    assert len(merged) == 34
    np.testing.assert_allclose(merged.event_age_kyr_bp, merged.age_ka_bp,
                               atol=1e-10, rtol=0)
