"""Age alignment, nested comparisons and validity of the orbital experiments."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Barker2011 import Barker2011_event_phase_analysis as barker
from toolbox import combined_pi
from toolbox import orbital_driver_sensitivity as orbital


@pytest.fixture(scope="module", params=("pooled", "barker"))
def point_analysis(request):
    if request.param == "pooled":
        main = combined_pi.fit_point_catalogue()
        frame = main.response.rename(columns={
            "bin_center_kyr_bp": "bin_center_ka", "dt_kyr": "dt_ka",
        })
        terms, expected = combined_pi.REDUCED_TERMS, main.summary
    else:
        main = barker.run_analysis()
        frame = main["model_frame"]
        terms, expected = barker.REDUCED_TERMS, main["summary"].iloc[0]
    frame, scaling, provenance = orbital.prepare_drivers(frame)
    return frame, terms, expected, orbital.fit_models(frame, terms)


def test_native_driver_points_convert_epoch_once_and_select_65n():
    # These BP1950 ages land exactly on native J2000 ages 10, 20 and 30 kyr.
    source_ages = np.array([10.0, 20.0, 30.0])
    frame = pd.DataFrame({
        "bin_center_ka": source_ages - 0.05,
        "dt_ka": 0.2,
        "event_count": [0, 1, 0],
    })
    actual, _, _ = orbital.prepare_drivers(frame)
    for path, column, unit_factor in (
        (orbital.ECC_TXT, "ecc", 1.0),
        (orbital.OBL_TXT, "obl_deg", 180 / np.pi),
    ):
        raw = np.loadtxt(path)
        expected = [raw[np.isclose(raw[:, 0], -age), 1].item() * unit_factor
                    for age in source_ages]
        np.testing.assert_allclose(actual[column], expected, rtol=0, atol=1e-10)
    with xr.open_dataset(orbital.INSOLATION_NC) as ds:
        latitude = np.flatnonzero(ds.latitude_degN.values == 65).item()
        time = [np.flatnonzero(ds.age_kyr_BP.values == age).item() for age in source_ages]
        expected = ds.daily_mean_insolation_Wm2.isel(latitude=latitude, time=time).values
    np.testing.assert_allclose(actual.insol65n_Wm2, expected, rtol=0, atol=1e-10)


def test_driver_scaling_uses_response_exposure_grid_not_event_locations():
    frame = pd.DataFrame({
        "bin_center_ka": np.arange(0.1, 100, 0.2),
        "dt_ka": 0.2,
        "event_count": 0,
    })
    original, scaling, _ = orbital.prepare_drivers(frame)
    moved_events = frame.copy()
    moved_events.loc[::7, "event_count"] = 1
    changed, changed_scaling, _ = orbital.prepare_drivers(moved_events)
    columns = ["ecc_scaled", "obl_scaled", "insol65n_scaled"]
    np.testing.assert_allclose(original[columns].mean(), 0, atol=2e-14)
    np.testing.assert_allclose(np.ptp(original[columns].to_numpy(), axis=0), 1, atol=2e-14)
    pd.testing.assert_frame_equal(original[columns], changed[columns])
    pd.testing.assert_frame_equal(scaling, changed_scaling)
    assert not set(columns).intersection(frame.columns)


def test_drivers_reject_extrapolation():
    frame = pd.DataFrame({
        "bin_center_ka": [-1.0, 10.0, 20.0], "dt_ka": 0.2, "event_count": 0,
    })
    # Orbital TXT inputs extend into the future; the insolation file does not.
    with pytest.raises(ValueError):
        orbital.prepare_drivers(frame)


def test_model_matrix_uses_only_the_ten_prespecified_nested_comparisons(point_analysis):
    frame, terms, expected, result = point_analysis
    specs = orbital.model_specs(terms)
    assert list(specs) == ["B", "BP", "B_ecc", "BP_ecc", "B_obl", "BP_obl",
                           "B_insol65n", "BP_insol65n"]
    assert len(result["models"]) == 8
    comparisons = result["comparisons"].set_index("comparison_id")
    expected_df = {"phase_reference": 2}
    for driver in ("ecc", "obl", "insol65n"):
        expected_df.update({f"{driver}_after_base": 1, f"{driver}_after_phase": 1,
                            f"phase_after_{driver}": 2})
    assert comparisons.df.to_dict() == expected_df
    assert comparisons.fit_valid.all()
    assert comparisons.info_bits_per_event.ge(-1e-9).all()
    assert comparisons.n_events.eq(expected["n_predictive_events"]).all()
    assert comparisons.n_bins.eq(len(frame)).all()
    np.testing.assert_allclose(comparisons.exposure_kyr, frame.dt_ka.sum())
    np.testing.assert_allclose(
        comparisons.info_bits_per_event * frame.event_count.sum() * np.log(2),
        comparisons.loglik_full - comparisons.loglik_reduced, atol=1e-8,
    )


def test_phase_reference_reproduces_each_current_main_analysis(point_analysis):
    frame, terms, expected, result = point_analysis
    reference = result["comparisons"].set_index("comparison_id").loc["phase_reference"]
    assert reference.info_bits_per_event == pytest.approx(expected["info_bits_per_event"], abs=1e-7)
    assert reference.nominal_p == pytest.approx(expected["nominal_LR_p"], abs=1e-7)
    phase = result["models"].set_index("model_id").loc["BP"]
    for column in ("pre_phase_preferred_deg", "pre_phase_rate_ratio_max_vs_min"):
        assert phase[column] == pytest.approx(expected[column], abs=1e-4)


def test_holm_family_contains_nine_new_tests_and_excludes_main_reference(point_analysis):
    _, _, _, result = point_analysis
    comparisons = result["comparisons"].set_index("comparison_id")
    assert pd.isna(comparisons.loc["phase_reference", "holm_nominal_p"])
    family = comparisons.drop(index="phase_reference")
    p = family.nominal_p.to_numpy()
    order = np.argsort(p)
    expected = np.empty(9)
    expected[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(9, 0, -1)))
    np.testing.assert_allclose(family.holm_nominal_p, expected)
    np.testing.assert_allclose(
        orbital.holm_adjust([0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]),
        [0.009, 0.08, 0.21, 0.6, 1, 1, 1, 1, 1],
    )
    # A failed fit remains in the prespecified family rather than reducing its size.
    np.testing.assert_allclose(orbital.holm_adjust([0.01, np.nan, 0.03]),
                               [0.03, np.nan, 0.06], equal_nan=True)


def test_rank_deficient_models_remain_in_output_as_invalid(point_analysis):
    frame, terms, _, _ = point_analysis
    duplicated = frame.copy()
    duplicated["ecc_scaled"] = duplicated.pre_phase_sin
    result = orbital.fit_models(duplicated, terms)
    models = result["models"].set_index("model_id")
    assert len(models) == 8
    assert not models.loc["BP_ecc", "fit_valid"]
    assert models.loc["BP", "fit_valid"]
    comparisons = result["comparisons"].set_index("comparison_id")
    invalid = comparisons.loc[["ecc_after_phase", "phase_after_ecc"]]
    assert not invalid.fit_valid.any()
    assert invalid[["info_bits_per_event", "nominal_p", "holm_nominal_p"]].isna().all().all()
    assert len(comparisons) == 10


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


def test_shifted_pooled_events_update_history_without_changing_other_segment():
    import NGRIP_MIS6_orbital_driver_sensitivity as experiment

    events = combined_pi.load_event_catalogue()
    context = combined_pi.build_context()
    binned = combined_pi.bin_catalogue(events, context)
    response = binned.loc[binned.in_response_interval].reset_index(drop=True).rename(
        columns={"bin_center_kyr_bp": "bin_center_ka", "dt_kyr": "dt_ka"})
    response, _, _ = orbital.prepare_drivers(response)
    ages = events[combined_pi.EVENT_AGE_COLUMN].to_numpy().copy()
    ages[0] -= 0.4
    shifted = experiment.frame_for_ages(ages, events, context, response)
    assert shifted.event_count.sum() == response.event_count.sum()
    assert not shifted.event_count.equals(response.event_count)
    assert not shifted.same_type_history_count.equals(response.same_type_history_count)
    pd.testing.assert_frame_equal(shifted.loc[shifted.segment_id.eq("MIS6")],
                                  response.loc[response.segment_id.eq("MIS6")])
    predictors = ["ecc_scaled", "obl_scaled", "insol65n_scaled", "lr04_scaled",
                  "co2_scaled", "pre_phase_sin", "pre_phase_cos", "dt_ka"]
    pd.testing.assert_frame_equal(shifted[predictors], response[predictors])


def test_all_unsupported_draws_remain_in_summary_denominators(point_analysis):
    from toolbox import orbital_driver_reporting as reporting

    _, _, _, point = point_analysis
    invalid = reporting.invalid_tables(point, "outside observation support")
    summary = orbital.summarize_comparisons(point["comparisons"], invalid["comparisons"])
    assert len(summary) == 10
    assert summary.n_mc_total.eq(1).all()
    assert summary.n_mc_valid.eq(0).all()
    assert summary.info_bits_per_event_median.isna().all()
    phases = orbital.summarize_phase(point["models"], invalid["models"])
    assert phases.n_mc_invalid.eq(1).all()
    assert phases.pre_phase_preferred_deg_median.isna().all()


def test_saved_chronology_selection_is_reproducible_and_preserves_pairing():
    from toolbox import orbital_driver_reporting as reporting

    table = pd.DataFrame(dict(realization_id=list("abcdef"), ngrip_id=np.arange(6),
                              mis6_id=np.arange(6)[::-1], age_a=np.arange(6), age_b=np.arange(6) + 10))
    selected = reporting.select_realizations(table, ["age_a", "age_b"], 4, 20260909)
    pd.testing.assert_frame_equal(selected,
        reporting.select_realizations(table, ["age_a", "age_b"], 4, 20260909))
    assert selected.realization_id.nunique() == 4
    assert (selected.ngrip_id + selected.mis6_id).eq(5).all()
