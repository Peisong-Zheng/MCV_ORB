"""Continuous orbital comparisons, fixed exposure scales and chronology reuse."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from toolbox import event_model
from toolbox.model_stats import fit_summary
from toolbox.point_process import fit_point_process
import NGRIP_MIS6_orbital_driver_sensitivity as root_entry
from toolbox import orbital_driver_sensitivity as orbital
from toolbox.project_config import (
    BARKER_EVENT_CSVS, EVENT_CATALOGUE_CSV, OBSERVATION_SEGMENTS_CSV,
    LR04_CSV, CO2_CSV, ORBITAL_CSV, INSOLATION_65N_CSV, PRECESSION_PHASE_CSV,
    ECC_TXT, OBL_TXT, INSOLATION_NC,
)


@pytest.fixture(scope="module", params=("pooled", "barker"))
def point_analysis(request):
    if request.param == "pooled":
        events = pd.read_csv(EVENT_CATALOGUE_CSV)
        observations = pd.read_csv(OBSERVATION_SEGMENTS_CSV)
        baseline = ("intercept", "same_type_exponential_history", "lr04_scaled", "co2_scaled", "mis6_segment")
    else:
        events = pd.read_csv(BARKER_EVENT_CSVS["variable_threshold"], float_precision="round_trip")
        events["segment_id"] = "Barker2011"
        observations = pd.DataFrame([dict(segment_id="Barker2011",
            observation_start_kyr_bp=0., observation_end_kyr_bp=400.)])
        baseline = ("intercept", "same_type_exponential_history", "lr04_scaled", "co2_scaled")
    lr04 = pd.read_csv(LR04_CSV, float_precision="round_trip")
    co2 = pd.read_csv(CO2_CSV, float_precision="round_trip")
    native = pd.read_csv(ORBITAL_CSV, float_precision="round_trip")
    insolation = pd.read_csv(INSOLATION_65N_CSV, float_precision="round_trip")
    anchors = pd.read_csv(PRECESSION_PHASE_CSV, float_precision="round_trip")
    forcings = {
        "lr04": (lr04.age_kyr_bp.to_numpy(), lr04.lr04.to_numpy()),
        "co2": (co2.age_kyr_bp.to_numpy(), co2.co2_ppm.to_numpy()),
        "precession_index": (native.age_kyr_bp.to_numpy(), native.precession_index.to_numpy()),
        "ecc": (native.age_kyr_bp.to_numpy(), native.eccentricity.to_numpy()),
        "obl": (native.age_kyr_bp.to_numpy(), native.obliquity_deg.to_numpy()),
        "insol65n": (insolation.age_kyr_bp.to_numpy(), insolation.insolation_Wm2.to_numpy()),
    }
    phase_anchors = (anchors.age_kyr_bp.to_numpy(), anchors.phase_unwrapped_rad.to_numpy())
    windows = event_model.response_windows(events, observations)
    scaling = event_model.nominal_scaling(
        {name: source for name, source in forcings.items() if name != "precession_index"}, windows)
    event_x, integral_x = event_model.build_design(events, windows, forcings, phase_anchors, scaling)
    if "mis6_segment" in baseline:
        for frame in (event_x, integral_x):
            frame["mis6_segment"] = frame.segment_id.eq("MIS6").astype(float)
    full_terms = baseline + orbital.PHASE_TERMS
    reduced = fit_point_process(event_x[list(baseline)], integral_x[list(baseline)], integral_x.weight, baseline)
    full = fit_point_process(event_x[list(full_terms)], integral_x[list(full_terms)], integral_x.weight,
                             full_terms, start_beta=np.r_[reduced.beta, 0., 0.])
    reference = fit_summary(reduced, full, event_x, windows, n_source_events=len(events), catalogue_id=request.param)
    result = orbital.fit_models(event_x, integral_x, baseline)
    return dict(events=events, windows=windows, forcings=forcings, phase_anchors=phase_anchors,
                scaling=scaling, baseline=baseline, event_features=event_x,
                integration_features=integral_x, reference=reference, result=result)


def test_native_driver_epoch_units_and_exact_65n_selection():
    native = pd.read_csv(ORBITAL_CSV, float_precision="round_trip")
    insolation = pd.read_csv(INSOLATION_65N_CSV, float_precision="round_trip")
    sources = {
        "ecc": {"age": native.age_kyr_bp, "values": native.eccentricity},
        "obl": {"age": native.age_kyr_bp, "values": native.obliquity_deg},
        "insol65n": {"age": insolation.age_kyr_bp, "values": insolation.insolation_Wm2},
    }
    source_ages = np.array([10.0, 20.0, 30.0])
    target = source_ages - 0.05
    for driver, path, factor in (("ecc", ECC_TXT, 1.0),
                                  ("obl", OBL_TXT, 180 / np.pi)):
        raw = np.loadtxt(path)
        expected = [raw[np.isclose(raw[:, 0], -age), 1].item() * factor for age in source_ages]
        np.testing.assert_allclose(np.interp(target, sources[driver]["age"], sources[driver]["values"]),
                                   expected, rtol=0, atol=1e-10)
    with xr.open_dataset(INSOLATION_NC) as ds:
        latitude = np.flatnonzero(ds.latitude_degN.values == 65).item()
        time = [np.flatnonzero(ds.age_kyr_BP.values == age).item() for age in source_ages]
        expected = ds.daily_mean_insolation_Wm2.isel(latitude=latitude, time=time).values
    actual = np.interp(target, sources["insol65n"]["age"], sources["insol65n"]["values"])
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)


def test_scaling_uses_continuous_exposure_and_stays_fixed(point_analysis):
    data = point_analysis
    columns = [f"{driver}_scaled" for driver in orbital.DRIVER_IDS]
    values = data["integration_features"][columns].to_numpy(float)
    np.testing.assert_allclose(np.average(values, axis=0, weights=data["integration_features"].weight), 0, atol=2e-13)
    original_scaling = data["scaling"].copy(deep=True)
    shifted = data["events"].copy()
    shifted.loc[0, "event_age_kyr_bp"] = shifted.loc[0, "event_age_kyr_bp"] + 0.007
    event_x, _ = event_model.build_design(shifted, data["windows"], data["forcings"],
                                         data["phase_anchors"], data["scaling"])
    pd.testing.assert_frame_equal(data["scaling"], original_scaling, check_exact=True)
    assert len(event_x) == len(data["event_features"])
    assert not np.array_equal(event_x.pre_phase_sin, data["event_features"].pre_phase_sin)
    assert not set(columns).intersection(data["events"].columns)


def test_driver_preparation_rejects_required_extrapolation(point_analysis):
    with pytest.raises(ValueError):
        event_model.nominal_scaling({"ecc": (np.array([20., 40.]), np.array([0.01, 0.02]))},
                                    point_analysis["windows"])


def test_ten_nested_comparisons_use_exact_event_terms_and_same_exposure(point_analysis):
    data = point_analysis
    expected, result = data["reference"], data["result"]
    assert list(orbital.model_specs(data["baseline"])) == [
        "B", "BP", "B_ecc", "BP_ecc", "B_obl", "BP_obl", "B_insol65n", "BP_insol65n"]
    comparisons = result["comparisons"].set_index("comparison_id")
    assert len(result["models"]) == 8 and len(comparisons) == 10
    expected_df = {"phase_reference": 2}
    for driver in orbital.DRIVER_IDS:
        expected_df.update({f"{driver}_after_base": 1, f"{driver}_after_phase": 1,
                            f"phase_after_{driver}": 2})
    assert comparisons.df.to_dict() == expected_df
    assert comparisons.fit_valid.all()
    assert comparisons.n_events.eq(expected["n_response_events"]).all()
    np.testing.assert_allclose(comparisons.exposure_kyr, (data["windows"].response_end_kyr_bp - data["windows"].response_start_kyr_bp).sum())
    np.testing.assert_allclose(comparisons.gain_bits_per_event * len(data["event_features"]) * np.log(2),
                               comparisons.loglik_full - comparisons.loglik_reduced, atol=1e-8)
    np.testing.assert_allclose(comparisons.delta_AIC, 2 * comparisons.df - comparisons.LR_statistic)
    assert not {"n_bins", "AICc", "BIC"}.intersection(result["models"].columns)
    assert all(fit.beta[fit.terms.index("same_type_exponential_history")] <= 0
               for fit in result["fits"].values())


def test_phase_reference_matches_the_continuous_main_model(point_analysis):
    expected, result = point_analysis["reference"], point_analysis["result"]
    comparison = result["comparisons"].set_index("comparison_id").loc["phase_reference"]
    phase = result["models"].set_index("model_id").loc["BP"]
    for key in ("gain_bits_per_event", "LR_statistic"):
        assert comparison[key] == pytest.approx(expected[key], rel=2e-6, abs=2e-6)
    for key in ("pre_phase_preferred_deg", "pre_phase_rate_ratio_max_vs_min"):
        assert phase[key] == pytest.approx(expected[key], rel=2e-6, abs=2e-6)


def test_holm_family_retains_nine_prespecified_tests(point_analysis):
    comparisons = point_analysis["result"]["comparisons"].set_index("comparison_id")
    assert pd.isna(comparisons.loc["phase_reference", "holm_nominal_p"])
    family = comparisons.drop(index="phase_reference")
    p = family.nominal_p.to_numpy()
    order = np.argsort(p)
    expected = np.empty(9)
    expected[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(9, 0, -1)))
    np.testing.assert_allclose(family.holm_nominal_p, expected)
    np.testing.assert_allclose(orbital.holm_adjust([0.01, np.nan, 0.03]),
                               [0.03, np.nan, 0.06], equal_nan=True)


def test_rank_deficient_candidate_is_retained_as_invalid(point_analysis):
    data = point_analysis
    events, integral = data["event_features"].copy(), data["integration_features"].copy()
    for frame in (events, integral):
        frame["ecc_scaled"] = frame.pre_phase_sin
    result = orbital.fit_models(events, integral, data["baseline"])
    models = result["models"].set_index("model_id")
    assert models.loc["BP", "fit_valid"] and not models.loc["BP_ecc", "fit_valid"]
    invalid = result["comparisons"].set_index("comparison_id").loc[["ecc_after_phase", "phase_after_ecc"]]
    assert not invalid.fit_valid.any()
    assert invalid[["gain_bits_per_event", "nominal_p", "holm_nominal_p"]].isna().all().all()


def test_phase_quantiles_cross_zero_and_exclude_failed_draws():
    point = pd.DataFrame({
        "model_id": ["BP", "BP_ecc", "BP_obl", "BP_insol65n"],
        "pre_phase_preferred_deg": [0.0, 350.0, 180.0, 5.0],
        "pre_phase_rate_ratio_max_vs_min": 2.0,
        "fit_valid": True,
        "invalid_reason": "",
    })
    realizations = []
    for realization_id, offset, fit_valid in ((1, -1, True), (2, 1, True), (3, 120, False)):
        draw = point.copy()
        draw["realization_id"] = realization_id
        draw["pre_phase_preferred_deg"] = (draw.pre_phase_preferred_deg + offset) % 360
        draw["fit_valid"] = fit_valid
        realizations.append(draw)
    summary = orbital.summarize_phase(point, pd.concat(realizations, ignore_index=True))
    center = point.pre_phase_preferred_deg.to_numpy()
    np.testing.assert_allclose(summary.pre_phase_preferred_deg_median, center)
    np.testing.assert_allclose(summary.pre_phase_preferred_deg_q025, center - 0.95)
    np.testing.assert_allclose(summary.pre_phase_preferred_deg_q975, center + 0.95)
    assert summary.n_mc_total.eq(3).all()
    assert summary.n_mc_valid.eq(2).all()
    assert summary.n_mc_invalid.eq(1).all()



def test_unsupported_rows_preserve_every_denominator(point_analysis):
    data = point_analysis
    age_columns = [f"age__{event_id}" for event_id in data["events"].event_id]
    selected = pd.DataFrame([data["events"].event_age_kyr_bp.to_numpy()], columns=age_columns)
    selected["realization_id"] = "unsupported_draw"
    selected.loc[0, age_columns[0]] = data["windows"].observation_start_kyr_bp.min() - 1.
    result = orbital.analyze_chronologies(
        data["events"], data["windows"], data["forcings"], data["phase_anchors"], data["scaling"],
        data["baseline"], selected, age_columns, show_progress=False,
    )
    summary = result["comparison_summary"]
    assert len(summary) == 10 and summary.n_mc_total.eq(1).all()
    assert summary.n_mc_valid.eq(0).all() and summary.gain_bits_per_event_median.isna().all()
    assert result["mc_comparisons"].exposure_kyr.isna().all()
    assert result["mc_comparisons"].realization_id.eq("unsupported_draw").all()
    assert not result["realization_status"].within_observation_support.any()


def test_saved_selection_preserves_exact_ages_ids_and_pairing():
    path = Path("data/processed/NGRIP_MIS6_orbital_driver_sensitivity/selected_realizations.csv")
    original = pd.read_csv(path, float_precision="round_trip")
    selected = root_entry.run_analysis(3, show_progress=False)["selected_realizations"]
    assert len(original) == 500
    pd.testing.assert_frame_equal(selected, original.iloc[:3].reset_index(drop=True), check_exact=True)


def test_weighted_correlation_does_not_count_nodes_as_observations():
    frame = pd.DataFrame({"x": [0., 1., 3.], "y": [0., 2., 1.]})
    weights = np.array([1., 4., 2.])
    original = root_entry.weighted_correlation(frame, weights)
    duplicated = pd.concat([frame, frame.iloc[[1]]], ignore_index=True)
    split_weights = [1., 2., 2., 2.]
    pd.testing.assert_frame_equal(original, root_entry.weighted_correlation(duplicated, split_weights))
