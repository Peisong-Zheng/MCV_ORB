"""Continuous orbital comparisons, fixed exposure scales and chronology reuse."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from toolbox import combined_likelihood
from toolbox import orbital_driver_sensitivity as orbital
from toolbox import orbital_driver_reporting as reporting


@pytest.fixture(scope="module", params=("pooled", "barker"))
def point_analysis(request):
    context = (combined_likelihood.build_context() if request.param == "pooled"
               else combined_likelihood.build_barker_context())
    reference = combined_likelihood.fit_catalogue(context.events, context).summary
    context, _, _ = orbital.prepare_drivers(context)
    design = combined_likelihood.prepare_catalogue(context.events, context)
    return context, design, reference, orbital.fit_models(design, context.reduced_terms)


def test_native_driver_epoch_units_and_exact_65n_selection():
    sources, _ = orbital.load_driver_sources()
    source_ages = np.array([10.0, 20.0, 30.0])
    target = source_ages - 0.05
    for driver, path, factor in (("ecc", orbital.ECC_TXT, 1.0),
                                  ("obl", orbital.OBL_TXT, 180 / np.pi)):
        raw = np.loadtxt(path)
        expected = [raw[np.isclose(raw[:, 0], -age), 1].item() * factor for age in source_ages]
        np.testing.assert_allclose(np.interp(target, sources[driver]["age"], sources[driver]["values"]),
                                   expected, rtol=0, atol=1e-10)
    with xr.open_dataset(orbital.INSOLATION_NC) as ds:
        latitude = np.flatnonzero(ds.latitude_degN.values == 65).item()
        time = [np.flatnonzero(ds.age_kyr_BP.values == age).item() for age in source_ages]
        expected = ds.daily_mean_insolation_Wm2.isel(latitude=latitude, time=time).values
    actual = np.interp(target, sources["insol65n"]["age"], sources["insol65n"]["values"])
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)


def test_scaling_uses_continuous_exposure_and_stays_fixed(point_analysis):
    context, design, _, _ = point_analysis
    columns = [f"{driver}_scaled" for driver in orbital.DRIVER_IDS]
    values = design.integration_frame[columns].to_numpy(float)
    np.testing.assert_allclose(np.average(values, axis=0, weights=design.weights), 0, atol=2e-13)
    shifted = context.events.copy()
    shifted.loc[0, combined_likelihood.EVENT_AGE_COLUMN] += 0.007
    changed = combined_likelihood.prepare_catalogue(shifted, context)
    assert changed.context.scaling == context.scaling
    assert len(changed.event_frame) == len(design.event_frame)
    assert not np.array_equal(changed.event_frame.pre_phase_sin, design.event_frame.pre_phase_sin)
    assert not set(columns).intersection(context.events.columns)


def test_driver_preparation_rejects_required_extrapolation():
    context = combined_likelihood.build_context()
    with pytest.raises(ValueError):
        combined_likelihood.add_forcing(context, "ecc", [20., 40.], [0.01, 0.02])


def test_ten_nested_comparisons_use_exact_event_terms_and_same_exposure(point_analysis):
    context, design, expected, result = point_analysis
    assert list(orbital.model_specs(context.reduced_terms)) == [
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
    np.testing.assert_allclose(comparisons.exposure_kyr, context.response_exposure_kyr)
    np.testing.assert_allclose(comparisons.gain_bits_per_event * len(design.event_frame) * np.log(2),
                               comparisons.loglik_full - comparisons.loglik_reduced, atol=1e-8)
    np.testing.assert_allclose(comparisons.delta_AIC, 2 * comparisons.df - comparisons.LR_statistic)
    assert not {"n_bins", "AICc", "BIC"}.intersection(result["models"].columns)
    history = result["coefficients"].query("term == 'same_type_exponential_history'")
    assert history.beta.le(0).all()


def test_phase_reference_matches_the_continuous_main_model(point_analysis):
    _, _, expected, result = point_analysis
    check = reporting.check_reference(result, pd.Series(expected))
    assert check.matches.all()


def test_holm_family_retains_nine_prespecified_tests(point_analysis):
    comparisons = point_analysis[-1]["comparisons"].set_index("comparison_id")
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
    context, design, _, _ = point_analysis
    events, integral = design.event_frame.copy(), design.integration_frame.copy()
    for frame in (events, integral):
        frame["ecc_scaled"] = frame.pre_phase_sin
    duplicated = replace(design, event_frame=events, integration_frame=integral)
    result = orbital.fit_models(duplicated, context.reduced_terms)
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
    point = point_analysis[-1]
    invalid = reporting.invalid_tables(point, "outside observation support")
    summary = orbital.summarize_comparisons(point["comparisons"], invalid["comparisons"])
    assert len(summary) == 10 and summary.n_mc_total.eq(1).all()
    assert summary.n_mc_valid.eq(0).all() and summary.gain_bits_per_event_median.isna().all()
    assert invalid["comparisons"].exposure_kyr.isna().all()


def test_saved_selection_preserves_exact_ages_ids_and_pairing():
    context = combined_likelihood.build_context()
    path = Path("data/processed/NGRIP_MIS6_orbital_driver_sensitivity/selected_realizations.csv")
    columns = [f"age_kyr_bp__{event_id}" for event_id in context.events.event_id]
    original = pd.read_csv(path, float_precision="round_trip")
    selected = reporting.read_selected_realizations(path, columns)
    pd.testing.assert_frame_equal(selected, original, check_exact=True)
    assert len(selected) == 500
    pd.testing.assert_frame_equal(reporting.read_selected_realizations(path, columns, 3),
                                   original.iloc[:3].reset_index(drop=True), check_exact=True)


def test_weighted_correlation_does_not_count_nodes_as_observations():
    frame = pd.DataFrame({"x": [0., 1., 3.], "y": [0., 2., 1.]})
    weights = np.array([1., 4., 2.])
    original = reporting.weighted_correlation(frame, weights)
    duplicated = pd.concat([frame, frame.iloc[[1]]], ignore_index=True)
    split_weights = [1., 2., 2., 2.]
    pd.testing.assert_frame_equal(original, reporting.weighted_correlation(duplicated, split_weights))
