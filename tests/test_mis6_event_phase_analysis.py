"""Focused regression tests for the MIS6 event-phase quick test."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import MIS6_event_phase_analysis as analysis
from NGRIP import ngrip_event_phase_analysis as ngrip


@pytest.fixture(scope="module")
def results():
    events = analysis.load_events()
    event_phases, rayleigh, _ = analysis.run_rayleigh(events)
    pi = analysis.run_predictive_information(events)
    summary = analysis.build_analysis_summary(events, rayleigh, pi)
    return events, event_phases, rayleigh, pi, summary


def test_input_is_one_21_event_warming_catalogue(results):
    events, event_phases, _, _, _ = results

    assert len(events) == 21
    assert events["composite_event_id"].is_unique
    assert events["event_age_ka_bp"].is_unique
    assert events["event_age_ka_bp"].is_monotonic_increasing
    assert set(events["event_type"]) == {"warming"}
    assert set(event_phases["event_label"]) == {analysis.EVENT_LABEL}
    assert not event_phases["phase_extrapolated"].any()


def test_pi_terms_match_ngrip_and_exclude_resolution():
    assert analysis.REDUCED_TERMS == (
        "same_type_history_count",
        "lr04_scaled",
        "co2_scaled",
    )
    assert analysis.FULL_TERMS == analysis.REDUCED_TERMS + (
        "pre_phase_sin",
        "pre_phase_cos",
    )
    assert analysis.REDUCED_TERMS == ngrip.REDUCED_TERMS
    assert analysis.FULL_TERMS == ngrip.FULL_TERMS
    assert not analysis.RESOLUTION_COVARIATE_INCLUDED
    assert not any("resolution" in term.lower() for term in analysis.FULL_TERMS)


def test_rayleigh_regression(results):
    _, _, rayleigh, _, _ = results
    row = rayleigh.query("driver == 'pre' and event_type == 'warming'").iloc[0]

    observed = row[
        [
            "n_phase_events_used",
            "mean_phase_deg",
            "mean_resultant_length",
            "rayleigh_R",
            "rayleigh_z",
            "rayleigh_p",
        ]
    ].to_numpy(dtype=float)
    expected = np.array(
        [21, 320.649058, 0.275030, 5.775630, 1.588472, 0.205904]
    )
    np.testing.assert_allclose(observed, expected, rtol=2e-6, atol=2e-6)


def test_conditional_pi_regression_and_diagnostics(results):
    _, _, _, pi, summary = results
    test = pi["likelihood_tests"].iloc[0]
    full = pi["model_summary"].loc[
        pi["model_summary"]["model_id"].eq(analysis.FULL_MODEL_ID)
    ].iloc[0]

    observed = np.array(
        [
            test["n_events"],
            test["LR_statistic"],
            test["LR_p_value"],
            test["info_bits_per_event"],
            test["delta_AICc_full_minus_reduced"],
            full["pre_phase_preferred_deg"],
            full["pre_phase_rate_ratio_max_vs_min"],
        ],
        dtype=float,
    )
    expected = np.array(
        [
            20,
            6.843932148,
            0.032648183,
            0.246842674,
            -2.690196516,
            318.326056,
            7.743116296,
        ]
    )
    np.testing.assert_allclose(observed, expected, rtol=3e-6, atol=3e-6)
    assert pi["model_summary"]["converged"].all()
    assert test["likelihood_nesting_ok"]
    assert not (
        pi["model_summary"][["n_eta_clipped_low", "n_eta_clipped_high"]] > 0
    ).any().any()

    row = summary.iloc[0]
    assert row["n_rayleigh_events"] == 21
    assert row["n_predictive_events"] == 20
    assert np.isclose(row["predictive_support_start_ka_bp"], 132.5)
    assert np.isclose(row["predictive_support_end_ka_bp"], 191.5)
    assert not row["resolution_covariate_included"]
    assert not row["event_age_uncertainty_propagated"]
