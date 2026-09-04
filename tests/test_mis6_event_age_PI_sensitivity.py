"""Focused tests for PI sensitivity to the MIS 6 A+ age realizations."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import MIS6_event_age_PI_sensitivity as sensitivity
import MIS6_event_phase_analysis as baseline
from toolbox import event_process as predictive


@pytest.fixture(scope="module")
def prepared():
    events = baseline.load_events()
    context, original, reference = sensitivity.build_pi_context(events)
    draws = sensitivity.load_age_realizations().head(8).copy()
    return context, original, reference, draws


def test_fast_original_fit_reproduces_existing_pi(prepared):
    _, original, _, _ = prepared
    observed = np.array(
        [
            original["n_predictive_events"],
            original["LR_statistic"],
            original["nominal_LR_p"],
            original["info_bits_per_event"],
            original["delta_AICc_full_minus_reduced"],
            original["pre_phase_preferred_deg"],
        ],
        dtype=float,
    )
    expected = np.array(
        [20, 6.843932148, 0.032648183, 0.246842674, -2.690196516, 318.326056]
    )
    np.testing.assert_allclose(observed, expected, rtol=3e-6, atol=3e-6)


def test_numpy_history_matches_shared_implementation(prepared):
    context, _, reference, draws = prepared
    ages = draws.iloc[0][sensitivity.event_age_columns()].to_numpy(float)
    counts = sensitivity.counts_from_ages(ages, context)
    observed = sensitivity.history_from_counts(counts, context)

    binned = reference["binned"].sort_values("bin_center_ka").copy()
    binned["event_count"] = counts
    expected = predictive.add_same_type_history(binned, baseline.HISTORY_WINDOW_KA)[
        predictive.HISTORY_TERM
    ].to_numpy(float)
    np.testing.assert_array_equal(observed, expected)


def test_small_fit_is_one_to_one_and_caches_duplicate_patterns(prepared):
    context, _, _, draws = prepared
    duplicated = pd.concat([draws.iloc[:4], draws.iloc[[0]]], ignore_index=True)
    duplicated["realization_id"] = np.arange(1, len(duplicated) + 1)
    results, diagnostics = sensitivity.fit_realizations(duplicated, context)
    sensitivity.validate_results(results, duplicated)

    assert len(results) == len(duplicated)
    assert diagnostics.loc[0, "n_cached_reuses"] >= 1
    metrics = [
        "info_bits_per_event",
        "nominal_LR_p",
        "delta_AICc_full_minus_reduced",
        "pre_phase_preferred_deg",
    ]
    np.testing.assert_allclose(
        results.iloc[0][metrics].to_numpy(float),
        results.iloc[-1][metrics].to_numpy(float),
    )


def test_summary_quantiles_and_decision_fractions_recompute(prepared):
    context, original, _, draws = prepared
    results, _ = sensitivity.fit_realizations(draws, context)
    summary = sensitivity.build_summary(results, original).set_index("metric")

    bits = results["info_bits_per_event"].to_numpy(float)
    np.testing.assert_allclose(
        summary.loc[
            "info_bits_per_event",
            ["mc_q025", "mc_q16", "mc_median", "mc_q84", "mc_q975"],
        ].to_numpy(float),
        np.quantile(bits, [0.025, 0.16, 0.5, 0.84, 0.975]),
    )
    assert summary.loc[
        "nominal_LR_p", "mc_fraction_meeting_decision_rule"
    ] == pytest.approx(results["nominal_LR_p"].lt(0.05).mean())


def test_parameters_explicitly_exclude_rayleigh_and_resolution(prepared):
    context, _, _, draws = prepared
    results, diagnostics = sensitivity.fit_realizations(draws.iloc[:2], context)
    parameters = sensitivity.build_parameters(draws.iloc[:2], context, diagnostics)
    values = parameters.set_index("parameter")["value"].astype(str).str.lower()

    assert values["rayleigh_test_run"] == "false"
    assert values["resolution_covariate_included"] == "false"
    assert values["event_age_uncertainty_propagated"] == "true"
    assert not any("rayleigh" in column.lower() for column in results.columns)
