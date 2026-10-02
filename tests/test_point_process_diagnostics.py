"""Checks for conditional interval diagnostics and their refitted calibration."""

import numpy as np
import pandas as pd
import pytest

from toolbox import point_process_diagnostics as diagnostics
from toolbox.project_config import BARKER_EVENT_CSVS


def cumulative_from_uniform(values):
    return np.cumsum(-np.log1p(-np.asarray(values)))


def test_completed_intervals_match_hand_computation_without_connecting_segments():
    events = {"NGRIP": np.array([110., 100.]), "MIS6": np.array([190., 180.])}
    cumulative = {"NGRIP": cumulative_from_uniform([0.2, 0.8]),
                  "MIS6": cumulative_from_uniform([0.9, 0.1])}
    result = diagnostics.residual_statistics(events, cumulative, {"NGRIP": 4., "MIS6": 0.2})
    assert result["statistics"]["ks_uniform"] == pytest.approx(0.3)
    assert result["statistics"]["adjacent_dependence"] == pytest.approx(0.25 / np.sqrt(2))
    assert result["n_adjacent_pairs"] == 2  # Never connect NGRIP's final U to MIS6's first U.
    assert len(result["intervals"]) == 4
    np.testing.assert_allclose(result["intervals"].uniform_residual, [0.2, 0.8, 0.9, 0.1])
    segment = result["segments"].set_index("segment_id").loc["NGRIP"]
    assert segment.residual_at_endpoint == pytest.approx(2 - cumulative["NGRIP"][-1] - 4)
    changed_tail = diagnostics.residual_statistics(events, cumulative, {"NGRIP": 100., "MIS6": 50.})
    assert changed_tail["statistics"] == result["statistics"]


def test_empty_and_single_event_segments_retain_tail_and_boundary_status():
    result = diagnostics.residual_statistics({"empty": []}, {"empty": []}, {"empty": 2.5})
    assert result["statistics"] == {"ks_uniform": 0., "adjacent_dependence": 0.}
    assert result["status"] == "no_response_events"
    assert result["intervals"].empty
    assert result["segments"].iloc[0].residual_at_endpoint == -2.5
    single = diagnostics.residual_statistics({"one": [1.]}, {"one": [-np.log(0.75)]})
    assert single["statistics"]["ks_uniform"] == pytest.approx(0.75)
    assert single["statistics"]["adjacent_dependence"] == 0
    assert np.isnan(single["segments"].iloc[0].residual_at_endpoint)


@pytest.mark.parametrize("events,cumulative,tail", [
    ({"a": [2., 1.]}, {"a": [2., 1.]}, {"a": 0.}),
    ({"a": [2., 2.]}, {"a": [1., 2.]}, {"a": 0.}),
    ({"a": [2., 1.]}, {"a": [1.]}, {"a": 0.}),
    ({"a": [2., 1.]}, {"a": [1., 2.]}, {"a": -0.1}),
])
def test_rescaling_rejects_wrong_order_or_integral_contract(events, cumulative, tail):
    with pytest.raises(ValueError):
        diagnostics.residual_statistics(events, cumulative, tail)


def test_plus_one_and_holm_use_actual_fixed_family_and_exceedance_counts():
    draws = pd.DataFrame({"ks_uniform": [0.1, 0.4, 0.8, 0.9],
                          "adjacent_dependence": [0.01, 0.02, 0.03, 0.04],
                          "fit_valid": True})
    result = diagnostics.bootstrap_summary({"ks_uniform": 0.8, "adjacent_dependence": 0.1}, draws)
    np.testing.assert_allclose(result.bootstrap_p, [3 / 5, 1 / 5])
    np.testing.assert_allclose(result.bootstrap_p_holm, [3 / 5, 2 / 5])
    np.testing.assert_array_equal(result.n_exceeding, [2, 0])
    assert (result.n_bootstrap == 4).all()
    assert result.exceedance_probability_ci95_low.iloc[1] == 0


def test_failed_replicate_is_unknown_not_a_zero_or_valid_only_p_value():
    draws = pd.DataFrame({"ks_uniform": [0.1, np.nan, 0.8],
                          "adjacent_dependence": [0.1, np.nan, 0.3],
                          "fit_valid": [True, False, True]})
    result = diagnostics.bootstrap_summary({"ks_uniform": 0.5, "adjacent_dependence": 0.5}, draws)
    assert result.bootstrap_p.isna().all() and result.bootstrap_p_holm.isna().all()
    assert (result.n_invalid == 1).all()
    assert (result.status == "incomplete_bootstrap").all()
    np.testing.assert_allclose(result.bootstrap_p_lower_bound, [0.5, 0.25])
    np.testing.assert_allclose(result.bootstrap_p_upper_bound, [0.75, 0.5])


@pytest.fixture(scope="module")
def nominal():
    from Barker2011.Barker2011_likelihood_bootstrap import run_analysis
    return run_analysis(n_bootstrap=1, seed=91)


def test_bootstrap_refits_every_catalogue_before_diagnostics_and_does_not_redraw(nominal, monkeypatch):
    from toolbox import sampling, event_model
    from toolbox.point_process import PointProcessFitError
    calls = []
    original_simulate = event_model.simulate_prepared_events
    original_fit = sampling.fit_point_process
    original_statistics = diagnostics.residual_statistics
    def simulate(*args):
        calls.append("simulate")
        return original_simulate(*args)
    def fit(*args, **kwargs):
        calls.append("fit")
        return original_fit(*args, **kwargs)
    def statistics(*args):
        calls.append("statistics")
        return original_statistics(*args)
    monkeypatch.setattr(event_model, "simulate_prepared_events", simulate)
    monkeypatch.setattr(sampling, "fit_point_process", fit)
    monkeypatch.setattr(diagnostics, "residual_statistics", statistics)
    args = [nominal[key] for key in ("events", "windows", "forcings", "phase_anchors", "scaling", "full")]
    first = sampling.gof_bootstrap(*args, full_terms=nominal["full"].terms,
        n_bootstrap=3, seed=83, show_progress=False)
    assert calls == ["simulate", "fit", "statistics"] * 3
    repeated = sampling.gof_bootstrap(*args, full_terms=nominal["full"].terms,
        n_bootstrap=3, seed=83, show_progress=False)
    pd.testing.assert_frame_equal(first, repeated)
    calls.clear()
    def failed_fit(*args, **kwargs):
        calls.append("fit")
        raise PointProcessFitError("known integration failure")
    monkeypatch.setattr(sampling, "fit_point_process", failed_fit)
    failed = sampling.gof_bootstrap(*args, full_terms=nominal["full"].terms,
        n_bootstrap=3, seed=83, show_progress=False)
    assert calls == ["simulate", "fit"] * 3
    assert len(failed) == 3 and not failed.fit_valid.any() and failed.ks_uniform.isna().all()


def test_boundary_lr_and_programming_errors_are_not_silently_reclassified(nominal, monkeypatch):
    from toolbox import sampling
    assert diagnostics.likelihood_ratio(0., 0.) == 0.
    assert diagnostics.likelihood_ratio(-1e-9, 0.) == 0.
    assert diagnostics.likelihood_ratio(-5., -7.) == 4.
    with pytest.raises(diagnostics.DiagnosticFailure):
        diagnostics.likelihood_ratio(-8., -7.)
    def incorrect_fit(*args, **kwargs):
        raise KeyError("misspelled model term")
    monkeypatch.setattr(sampling, "fit_point_process", incorrect_fit)
    args = [nominal[key] for key in ("events", "windows", "forcings", "phase_anchors", "scaling", "full")]
    with pytest.raises(KeyError):
        sampling.gof_bootstrap(*args, full_terms=nominal["full"].terms,
                              n_bootstrap=2, seed=5, show_progress=False)


@pytest.fixture
def sampling_cache(nominal):
    from toolbox.project_config import MODEL_VERSION
    full = nominal["full"]
    saved = pd.DataFrame([dict(zip(full.terms, full.beta))])
    settings = dict(model_version=MODEL_VERSION, history_tau_kyr=1.5, n_point=2, seed=41)
    windows = nominal["windows"]
    exposure = float((windows.response_end_kyr_bp - windows.response_start_kyr_bp).sum())
    draws = pd.DataFrame(dict(
        replicate_id=[1, 2, 1], scenario=["B_sampling", "B_sampling", "C_joint"],
        outer_id=[0, 0, 1], seed=41, fit_valid=[True, False, True],
        invalid_reason=["", "failed fit", ""], response_exposure_kyr=exposure,
        n_response_events=[70, 65, 66], ks_uniform=[0.1, np.nan, 0.9],
        adjacent_dependence=[0.2, np.nan, 0.8], residual_status=["ok", "", "ok"]))
    return draws, settings, saved, full, exposure


def test_sampling_cache_excludes_joint_but_keeps_failed_nominal_refits(sampling_cache):
    draws, settings, saved, full, exposure = sampling_cache
    selected = diagnostics.select_sampling_gof_replicates(draws, settings, saved, full,
                                                         response_exposure_kyr=exposure)
    assert selected.replicate_id.tolist() == [1, 2]
    assert selected.scenario.eq("B_sampling").all() and not selected.fit_valid.iloc[1]
    result = diagnostics.bootstrap_summary({"ks_uniform": 0.05, "adjacent_dependence": 0.1}, selected)
    assert result.bootstrap_p.isna().all()


def test_sampling_cache_rejects_incomplete_ensemble_or_changed_generator(sampling_cache):
    draws, settings, saved, full, exposure = sampling_cache
    with pytest.raises(ValueError, match="incomplete"):
        diagnostics.select_sampling_gof_replicates(draws.iloc[[0, 2]], settings, saved, full,
                                                   response_exposure_kyr=exposure)
    saved["pre_phase_sin"] = saved["pre_phase_sin"] + 0.1
    with pytest.raises(AssertionError, match="generator differs"):
        diagnostics.select_sampling_gof_replicates(draws, settings, saved, full,
                                                   response_exposure_kyr=exposure)
