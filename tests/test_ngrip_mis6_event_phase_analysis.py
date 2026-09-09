"""Scientific checks for the pooled NGRIP--MIS 6 point-age analysis."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import NGRIP_MIS6_event_phase_analysis as analysis
from toolbox import combined_pi


@pytest.fixture(scope="module")
def point_result() -> dict[str, object]:
    # run_analysis is intentionally write-free; main() owns all file output.
    return analysis.run_analysis()


def test_default_model_choices_are_frozen():
    assert analysis.HISTORY_WINDOW_KYR == pytest.approx(1.5)
    assert analysis.BIN_WIDTH_KYR == pytest.approx(0.2)
    assert analysis.BIN_ORIGIN_FRACTION == pytest.approx(0.0)
    assert analysis.RESPONSE_MODE == "maximal_for_history"
    assert combined_pi.REDUCED_TERMS == (
        "same_type_history_count",
        "lr04_scaled",
        "co2_scaled",
        "mis6_segment",
    )
    assert combined_pi.FULL_TERMS[-2:] == ("pre_phase_sin", "pre_phase_cos")
    assert analysis.RESOLUTION_COVARIATE_INCLUDED is False


def test_only_published_warming_transitions_enter_the_catalogue(point_result):
    events = point_result["events"]

    assert len(events) == 55
    assert events["event_id"].is_unique
    assert events.groupby("segment_id").size().to_dict() == {
        "MIS6": 21,
        "NGRIP": 34,
    }
    assert events.loc[
        events["segment_id"].eq("NGRIP"), "event_label"
    ].str.startswith("GI-").all()
    assert events["timing_method"].str.contains("published").all()


def test_response_support_is_disjoint_and_history_complete(point_result):
    context = point_result["context"]
    response = context.response_bins

    assert context.response_exposure_kyr == pytest.approx(180.0)
    assert set(response["segment_id"]) == {"NGRIP", "MIS6"}
    assert not response["bin_center_kyr_bp"].between(123.0, 132.5).any()
    assert response["same_type_history_complete"].all()
    assert {
        segment_id: (segment.response_start_kyr_bp, segment.response_end_kyr_bp)
        for segment_id, segment in context.segments.items()
    } == pytest.approx({"NGRIP": (12.0, 121.5), "MIS6": (132.5, 203.0)})


def test_rayleigh_and_pi_results_are_finite(point_result):
    phases = point_result["event_phases"]
    rayleigh = point_result["rayleigh"]
    fit = point_result["fit"]

    assert len(phases) == 55
    assert not phases["pre_phase_extrapolated"].any()
    assert phases["pre_phase_deg"].between(0.0, 360.0).all()
    assert 0.0 <= rayleigh["rayleigh_p"] <= 1.0
    assert 0.0 <= fit.summary["nominal_LR_p"] <= 1.0
    assert np.isfinite(
        [
            rayleigh["mean_phase_deg"],
            rayleigh["mean_resultant_length"],
            fit.summary["LR_statistic"],
            fit.summary["info_bits_per_event"],
            fit.summary["pre_phase_preferred_deg"],
            fit.summary["mis6_vs_ngrip_rate_ratio_full"],
        ]
    ).all()
    assert fit.summary["n_predictive_events"] == 55
    assert fit.summary["all_models_converged"]
    assert fit.summary["likelihood_nesting_ok"]
    assert not fit.summary["eta_clipping_used"]


def test_bp1950_phase_and_main_point_result_are_frozen(point_result):
    phases = point_result["event_phases"].set_index("event_id")
    rayleigh = point_result["rayleigh"]
    fit = point_result["fit"]

    # This published boundary catches a reversed J2000-to-BP1950 offset and a
    # changed definition of zero phase; aggregate checks alone would not.
    assert phases.loc["NGRIP:GI-1", "pre_phase_deg"] == pytest.approx(
        52.7119266, abs=1e-7
    )
    assert rayleigh["mean_phase_deg"] == pytest.approx(339.981582, abs=1e-6)
    assert rayleigh["rayleigh_p"] == pytest.approx(0.05795892, abs=1e-8)

    observed = [
        fit.summary["n_predictive_bins"],
        fit.summary["response_exposure_kyr"],
        fit.summary["LR_statistic"],
        fit.summary["nominal_LR_p"],
        fit.summary["info_bits_per_event"],
        fit.summary["pre_phase_preferred_deg"],
    ]
    expected = [
        901,
        180.0,
        13.6732014732,
        0.001073747153,
        0.1793296360,
        329.9985118,
    ]
    np.testing.assert_allclose(observed, expected, rtol=2e-7, atol=2e-9)


def test_point_analysis_is_reproducible(point_result):
    events = point_result["events"]
    context = point_result["context"]
    repeated_fit = combined_pi.fit_catalogue(events, context)
    repeated_phases = combined_pi.sample_event_phases(events)

    for name in (
        "LR_statistic",
        "nominal_LR_p",
        "info_bits_per_event",
        "pre_phase_preferred_deg",
        "pre_phase_rate_ratio_max_vs_min",
    ):
        assert repeated_fit.summary[name] == pytest.approx(
            point_result["fit"].summary[name], rel=0.0, abs=1e-12
        )
    np.testing.assert_allclose(
        repeated_phases["pre_phase_rad"],
        point_result["event_phases"]["pre_phase_rad"],
        rtol=0.0,
        atol=0.0,
    )


def test_output_tables_are_compact_and_self_describing(point_result):
    summary = analysis.build_analysis_summary(point_result)
    coefficients = analysis.build_coefficients(point_result["fit"])
    parameters = analysis.build_parameters(point_result)

    assert len(summary) == 1
    assert summary.loc[0, "rayleigh_role"] == "descriptive"
    assert summary.loc[0, "pi_role"] == "primary"
    assert not bool(summary.loc[0, "resolution_covariate_included"])
    assert set(coefficients["model_id"]) == {"reduced", "full"}
    assert coefficients["term_definition"].notna().all()
    assert parameters["parameter"].is_unique
    parameter_values = parameters.set_index("parameter")["value"]
    assert parameter_values["segment_indicator"] == "MIS6=1; NGRIP=0"
    assert str(parameter_values["age_unit"]) == "kyr BP"
    parameter_units = parameters.set_index("parameter")["unit"]
    assert parameter_units["lr04_response_mean"] == "per mil"
    assert parameter_units["co2_response_mean"] == "ppm"
