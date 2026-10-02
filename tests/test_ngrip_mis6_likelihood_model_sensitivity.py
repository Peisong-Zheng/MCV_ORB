"""Continuous climate/history alternatives share the exact response record."""

import numpy as np
import pytest

import NGRIP_MIS6_likelihood_model_sensitivity as sensitivity
from toolbox import event_model
import NGRIP_MIS6_event_phase_analysis as nominal


@pytest.fixture(scope="module")
def analysis():
    return sensitivity.run_analysis()


def test_all_nine_forms_share_exact_support(analysis):
    summary = analysis["summary"]
    assert summary.variant.tolist() == [*sensitivity.CLIMATE_VARIANTS, *sensitivity.HISTORY_VARIANTS]
    assert summary.n_response_events.eq(53).all()
    np.testing.assert_allclose(summary.response_exposure_kyr, 165.058)
    assert summary.all_models_converged.all() and summary.likelihood_nesting_ok.all()
    np.testing.assert_allclose(summary.delta_AIC_full_minus_reduced, 4 - summary.LR_statistic)
    assert not {"n_response_bins", "full_aicc", "bin_width_kyr"}.intersection(summary.columns)
    assert analysis["event_roles"].event_role.value_counts().to_dict() == {"response": 53, "conditioning": 2}


def test_primary_reference_is_reproduced_in_both_experiments(analysis):
    main = nominal.run_analysis()["statistics"]
    for variant in ("frozen_linear", "exponential"):
        row = analysis["summary"].set_index("variant").loc[variant]
        assert row.gain_bits_per_event == pytest.approx(main["gain_bits_per_event"], abs=1e-7)
        assert row.pre_phase_preferred_deg == pytest.approx(main["pre_phase_preferred_deg"], abs=1e-5)
        assert row.delta_AIC_full_vs_same_support_reference == 0


def test_polynomial_products_are_the_same_at_events_and_integral_nodes(analysis):
    for frame in (analysis["event_inputs"], analysis["integration_features"]):
        np.testing.assert_allclose(frame.lr04_squared, frame.lr04_scaled**2)
        np.testing.assert_allclose(frame.co2_squared, frame.co2_scaled**2)
        np.testing.assert_allclose(frame.lr04_co2, frame.lr04_scaled * frame.co2_scaled)


def test_elapsed_and_rectangular_histories_use_actual_events():
    result = nominal.run_analysis()
    events = result["events"]
    ages = events.loc[events.segment_id.eq("NGRIP"), "event_age_kyr_bp"].to_numpy()
    anchor = ages[-1]
    query = np.array([anchor-1.5+1e-7, anchor-1.5-1e-7, ages[-2]])
    window = result["windows"].set_index("segment_id", drop=False).loc["NGRIP"]
    frame = event_model.evaluate_features(query, ages, window, result["forcings"],
        result["phase_anchors"], result["scaling"], history_variants=True)
    np.testing.assert_allclose(frame.rectangular_history_count.iloc[:2], [1., 0.])
    np.testing.assert_allclose(frame[sensitivity.ELAPSED_TERM], anchor-query)
    np.testing.assert_allclose(frame[sensitivity.LOG_ELAPSED_TERM], np.log1p(anchor-query))
    assert frame[sensitivity.ELAPSED_TERM].iloc[-1] > 0  # The event does not reset its own predictor.


def test_elapsed_history_is_not_forced_into_an_inhibition_domain(analysis):
    summary = analysis["summary"].set_index("variant")
    coeffs = analysis["coefficients"]
    for variant in ("exponential", "count_matched_support"):
        row = summary.loc[variant]
        beta = coeffs.loc[coeffs.variant.eq(variant) & coeffs.term.eq(row.history_term), "beta"]
        assert row.history_coefficient_domain == "nonpositive" and beta.le(0).all()
    for variant in ("elapsed_time", "log_elapsed_time"):
        row = summary.loc[variant]
        beta = coeffs.loc[coeffs.variant.eq(variant) & coeffs.term.eq(row.history_term), "beta"]
        assert row.history_coefficient_domain == "unrestricted"
        assert beta.gt(0).all()  # These data express recovery with increasing time after an event.
