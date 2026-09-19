"""Scientific checks for the continuous pooled nominal-age analysis."""
from dataclasses import replace
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import NGRIP_MIS6_event_phase_analysis as analysis
from toolbox import combined_likelihood


@pytest.fixture(scope="module")
def point_result():
    return analysis.run_analysis()


def test_source_inventory_and_exact_conditioning(point_result):
    events = point_result["events"]
    assert len(events) == 55 and events.event_id.is_unique
    assert events.groupby("segment_id").size().to_dict() == {"MIS6": 21, "NGRIP": 34}
    assert events.event_role.value_counts().to_dict() == {"response": 53, "conditioning": 2}
    anchors = events.loc[events.event_role.eq("conditioning")].set_index("segment_id")
    assert anchors.loc["NGRIP", "event_age_kyr_bp"] == pytest.approx(115.320)
    assert anchors.loc["MIS6", "event_age_kyr_bp"] == pytest.approx(194.238)
    assert point_result["context"].response_exposure_kyr == pytest.approx(165.058)
    assert point_result["fit"].design.weights.sum() == pytest.approx(165.058)
    assert not point_result["fit"].design.integration_frame.age_kyr_bp.between(123, 132.5).any()


def test_unchanged_bp1950_phase_convention(point_result):
    phases = point_result["event_phases"].set_index("event_id")
    assert phases.loc["NGRIP:GI-1", "pre_phase_deg"] == pytest.approx(52.7119266, abs=1e-7)
    assert not phases.pre_phase_extrapolated.any()
    assert point_result["rayleigh"]["mean_phase_deg"] == pytest.approx(339.981582, abs=1e-6)
    assert point_result["rayleigh"]["rayleigh_p"] == pytest.approx(0.05795892, abs=1e-8)


def test_actual_age_history_and_continuous_statistics(point_result):
    fit = point_result["fit"]
    context = point_result["context"]
    for segment, frame in fit.design.event_frame.groupby("segment_id"):
        ages = point_result["events"].loc[lambda e: e.segment_id.eq(segment), "event_age_kyr_bp"].to_numpy()
        expected = [np.sum(np.exp(-(ages[ages > a] - a) / context.history_tau_ka)) for a in frame.age_kyr_bp]
        np.testing.assert_allclose(frame.same_type_exponential_history, expected, atol=1e-12)
    for model in (fit.reduced, fit.full):
        assert model.converged and model.identifiable
        assert model.gradient_max <= 1e-6
        assert dict(zip(model.terms, model.beta))[combined_likelihood.HISTORY_TERM] <= 0
    summary = fit.summary
    assert summary["gain_bits_per_event"] == pytest.approx(summary["LR_statistic"] / (2 * 53 * np.log(2)))
    assert summary["delta_AIC_full_minus_reduced"] == pytest.approx(4 - summary["LR_statistic"])
    assert "n_response_bins" not in summary and "delta_AICc_full_minus_reduced" not in summary


def test_nominal_integration_refinement(point_result):
    refined = combined_likelihood.fit_catalogue(point_result["events"],
               replace(point_result["context"], quadrature_order=8))
    fit = point_result["fit"]
    assert abs(refined.reduced.log_likelihood - fit.reduced.log_likelihood) < 1e-6
    assert abs(refined.full.log_likelihood - fit.full.log_likelihood) < 1e-6
    assert abs(refined.summary["LR_statistic"] - fit.summary["LR_statistic"]) < 1e-4
    assert abs(refined.summary["gain_bits_per_event"] - fit.summary["gain_bits_per_event"]) < 1e-6


def test_outputs_are_explicit_continuous_products(point_result, tmp_path):
    analysis.write_outputs(point_result, tmp_path)
    files = {p.name for p in tmp_path.iterdir()}
    assert {"support.csv", "fitted_rates.csv", "event_catalogue_used.csv", "model_coefficients.csv",
            "analysis_summary.csv", "likelihood_tests.csv"} <= files
    assert not any("binned" in name for name in files)
    saved = pd.read_csv(tmp_path / "event_catalogue_used.csv")
    assert saved.event_role.eq("response").sum() == 53
    coefficients = analysis.build_coefficients(point_result["fit"])
    assert coefficients.term_definition.notna().all()
    parameters = analysis.build_parameters(point_result)
    assert parameters.parameter.is_unique
    assert not parameters.parameter.str.contains("bin_width|origin").any()
    with plt.rc_context():
        fig = analysis.plot_results(point_result)
        assert len(fig.axes) == 4 and fig.axes[2].name == "polar"
        assert fig.axes[0].get_xlim()[0] > fig.axes[0].get_xlim()[1]
        assert fig.axes[1].get_xlim()[0] > fig.axes[1].get_xlim()[1]
        plt.close(fig)
