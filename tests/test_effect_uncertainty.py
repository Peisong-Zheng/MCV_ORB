"""Process, interval geometry and integration checks for full-model effects."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import NGRIP_MIS6_effect_uncertainty as analysis
import NGRIP_MIS6_PI_bootstrap as null_bootstrap
from toolbox import combined_pi
from toolbox import effect_uncertainty as effect


class RecordingRNG:
    def __init__(self, counts):
        self.counts = iter(counts)
        self.means = []

    def poisson(self, mean):
        self.means.append(mean)
        return next(self.counts)


def toy_context():
    frames, segments = [], {}
    for segment_id, start in (("NGRIP", 0.0), ("MIS6", 100.0)):
        edges = np.arange(start, start + 4)
        segments[segment_id] = combined_pi.SegmentContext(
            segment_id, start, start + 3, start, start + 3, edges,
            np.ones(3, dtype=bool), np.array([1, 2, 3]), np.array([3, 3, 3]))
        frames.append(pd.DataFrame({
            "segment_id": segment_id, "dt_kyr": np.ones(3),
            "lr04_scaled": np.zeros(3), "co2_scaled": np.zeros(3),
            "mis6_segment": float(segment_id == "MIS6"),
            "pre_phase_sin": [1, 0, 0], "pre_phase_cos": [0, 0, 1],
        }))
    return combined_pi.PIContext(pd.concat(frames, ignore_index=True), segments,
                                 np.ones(6), 2.0, 1.0, 0.0, "common_core")


def test_full_phase_terms_dynamic_history_and_segment_reset():
    rng = RecordingRNG([1, 2, 0, 1, 0, 0])
    beta = np.array([0, np.log(2), 0, 0, 0, np.log(3), np.log(5)])
    counts = effect.simulate_full_model_counts(toy_context(), beta, rng)
    # Oldest bin cosine multiplies by 5; youngest sine by 3. History is
    # 0, 1, 3 in first segment and resets to 0 at the second segment start.
    np.testing.assert_allclose(rng.means, [5, 2, 24, 5, 2, 6])
    np.testing.assert_array_equal(counts["NGRIP"], [0, 2, 1])


def test_true_bin_exposure_enters_poisson_mean():
    context = toy_context()
    for segment in context.segments.values():
        segment.bin_edges[:] = segment.bin_edges[0] + np.array([0, 0.5, 1, 1.5])
    context.bins["dt_kyr"] = 0.5
    rng = RecordingRNG([0] * 6)
    effect.simulate_full_model_counts(context, np.zeros(7), rng)
    np.testing.assert_allclose(rng.means, 0.5)


def test_zero_phase_special_case_matches_original_null_without_modifying_it():
    context = toy_context()
    reduced = np.array([-0.3, -0.5, 0.1, -0.1, 0.4])
    first = null_bootstrap.simulate_catalogue_counts(context, reduced, np.random.default_rng(15))
    second = effect.simulate_full_model_counts(context, np.r_[reduced, 0, 0], np.random.default_rng(15))
    for segment in combined_pi.SEGMENT_IDS:
        np.testing.assert_array_equal(first[segment], second[segment])


def test_full_beta_length_and_numerical_limits_are_checked():
    with pytest.raises(ValueError, match="full_beta"):
        effect.simulate_full_model_counts(toy_context(), np.zeros(5), np.random.default_rng(1))
    beta = np.zeros(7)
    beta[0] = 21
    with pytest.raises(effect.InvalidEffectSimulation):
        effect.simulate_full_model_counts(toy_context(), beta, np.random.default_rng(1))


def circle_region(center, radius):
    center = np.asarray(center, dtype=float)
    return {"center": center, "covariance": np.eye(2) * radius ** 2,
            "critical_value": 1.0, "origin_in_region": np.linalg.norm(center) <= radius}


def test_ellipse_projection_has_correct_radial_extrema_and_wraparound_arc():
    # Preferred phase is 0 degrees; its interval must cross zero continuously.
    result = effect.project_joint_region(circle_region([0, 2], 0.5))
    width = np.degrees(np.arcsin(0.25))
    assert result["phase_identified"]
    assert result["phase_low_unwrapped_deg"] == pytest.approx(-width, abs=1e-7)
    assert result["phase_high_unwrapped_deg"] == pytest.approx(width, abs=1e-7)
    assert result["ratio_low"] == pytest.approx(np.exp(3), rel=1e-9)
    assert result["ratio_high"] == pytest.approx(np.exp(5), rel=1e-9)


def test_region_containing_zero_cannot_identify_phase_or_exclude_rate_ratio_one():
    result = effect.project_joint_region(circle_region([0.1, 0.2], 0.5))
    assert not result["phase_identified"]
    assert result["ratio_low"] == 1
    assert result["phase_low_unwrapped_deg"] == 0
    assert result["phase_high_unwrapped_deg"] == 360


def test_simultaneous_band_encloses_all_curves_from_joint_region():
    region = circle_region([0.3, -0.5], 0.4)
    angles = np.linspace(0, 360, 91)
    low, high = effect.joint_region_curve_band(region, angles)
    coefficients = effect.ellipse_boundary(region, np.linspace(0, 2 * np.pi, 300))
    direction = np.column_stack((np.sin(np.deg2rad(angles)), np.cos(np.deg2rad(angles))))
    curves = np.exp(coefficients @ direction.T)
    assert np.all(curves >= low - 1e-12)
    assert np.all(curves <= high + 1e-12)


def test_bootstrap_region_uses_errors_about_generator_and_retains_bias():
    point = np.array([0.4, -0.5])
    samples = point + [0.2, 0] + np.random.default_rng(4).normal(size=(200, 2)) * 0.1
    region = effect.bootstrap_joint_region(point, samples)
    errors = samples - point
    expected = np.sum(errors * np.linalg.solve(np.cov(samples.T), errors.T).T, axis=1)
    np.testing.assert_allclose(region["bootstrap_error_quadratic"], expected)
    assert region["critical_value"] == pytest.approx(np.quantile(expected, 0.95))
    np.testing.assert_allclose(region["bootstrap_bias"], errors.mean(axis=0))


@pytest.fixture(scope="module")
def setup():
    events = combined_pi.load_event_catalogue()
    context = combined_pi.build_context()
    fit = combined_pi.fit_catalogue(events, context)
    draws, results = analysis.load_age_inputs(events)
    return events, context, fit, draws, results


def test_outer_selection_is_reproducible_uniform_rule_and_preserves_source_ids(setup):
    events, context, fit, draws, results = setup
    generators, table = analysis.build_generators(*setup, n_outer=3, seed=7)
    _, repeat = analysis.build_generators(*setup, n_outer=3, seed=7)
    pd.testing.assert_frame_equal(table, repeat)
    eligible = np.flatnonzero(results.fit_valid)
    expected = np.random.default_rng(np.random.SeedSequence([7, 100])).choice(eligible, 3, replace=False)
    np.testing.assert_array_equal(table.source_row_index, expected)
    assert table.age_realization_id.is_unique
    assert set(generators) == {0, 1, 2, 3}
    np.testing.assert_array_equal(generators[0], fit.full.beta)
    assert table.age_realization_id.tolist() == draws.iloc[expected].realization_id.tolist()


def test_parallel_replicates_are_reproducible_and_do_not_freeze_event_count(setup):
    _, context, fit, _, _ = setup
    generators = {0: fit.full.beta, 1: fit.full.beta}
    first = analysis.run_simulations(context, generators, 20, 3, 9, workers=1, show_progress=False)
    second = analysis.run_simulations(context, generators, 20, 3, 9, workers=2, show_progress=False)
    pd.testing.assert_frame_equal(first, second)
    assert first.fit_valid.all()
    assert first.n_response_events.nunique() > 1
    assert first.groupby("scenario").size().to_dict() == {"B_sampling": 20, "C_joint": 3}


def test_failed_simulations_are_retained_without_retries(setup):
    _, context, fit, _, _ = setup
    bad = fit.full.beta.copy()
    bad[0] = 100
    table = analysis.run_simulations(context, {0: bad, 1: bad}, 2, 2, 1, show_progress=False)
    assert len(table) == 4
    assert not table.fit_valid.any()
    assert table.invalid_reason.str.len().gt(0).all()


def test_small_analysis_summaries_keep_interval_meanings_and_equal_weights(setup):
    _, context, fit, _, age_results = setup
    generators = {0: fit.full.beta, 1: fit.full.beta, 2: fit.full.beta}
    table = analysis.run_simulations(context, generators, 30, 5, 10, show_progress=False)
    summary, region = analysis.summarize_effects(fit, age_results, table)
    assert len(summary) == 6
    assert summary.loc[summary.scenario.eq("B_sampling"), "interval_type"].str.contains("confidence").all()
    assert summary.loc[summary.scenario.eq("C_joint"), "interval_type"].str.contains("working").all()
    with pytest.raises(ValueError, match="equal weight"):
        analysis.summarize_effects(fit, age_results, table.iloc[:-1])
    curves = analysis.build_curve_table(fit, table, region)
    assert (curves.B_simultaneous_low <= curves.point_multiplier).all()
    assert (curves.B_simultaneous_high >= curves.point_multiplier).all()
    assert (curves.C_pointwise_q025 <= curves.C_pointwise_q975).all()
