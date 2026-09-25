"""Process, interval geometry and integration checks for full-model effects."""

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import NGRIP_MIS6_effect_uncertainty as analysis
from toolbox import combined_likelihood
from toolbox import effect_uncertainty as effect


def test_barker_uses_named_phase_coefficients_and_its_own_age_support():
    from Barker2011 import Barker2011_effect_uncertainty as barker

    context = combined_likelihood.build_barker_context()
    fit = combined_likelihood.fit_catalogue(context.events, context)
    assert "mis6_segment" not in fit.full.terms
    expected = [fit.full.beta[fit.full.terms.index(term)] for term in effect.PHASE_TERMS]
    np.testing.assert_array_equal(effect.phase_coefficients(fit.full), expected)
    reordered = replace(fit.full, terms=fit.full.terms[::-1], beta=fit.full.beta[::-1])
    np.testing.assert_array_equal(effect.phase_coefficients(reordered), expected)

    draws, age_results, columns = barker.load_age_inputs(context.events)
    generators, selected = analysis.build_generators(
        context.events, context, fit, draws, age_results, 2, 25,
        age_columns=columns, source_id_columns=())
    assert selected.age_realization_id.is_unique
    assert "beta__mis6_segment" not in selected
    first = analysis.run_simulations(context, generators, 24, 10, 25, 1, False)
    parallel = analysis.run_simulations(context, generators, 24, 10, 25, 2, False)
    pd.testing.assert_frame_equal(first, parallel)
    assert first.fit_valid.all()
    assert "beta__mis6_segment" not in first
    assert first.n_response_events.nunique() > 1
    for outer_id in (1, 2):
        local = combined_likelihood.condition_context(context, generators[outer_id]["events"])
        assert local.scaling == context.scaling
        segment = local.segments["Barker2011"]
        assert segment.response_start_kyr_bp == 0
        assert segment.anchor_age_kyr_bp == generators[outer_id]["events"][combined_likelihood.EVENT_AGE_COLUMN].max()
        np.testing.assert_allclose(first.loc[first.outer_id.eq(outer_id), "response_exposure_kyr"],
                                   local.response_exposure_kyr)
    summary, regions = analysis.summarize_effects(fit, age_results, first)
    assert set(summary.loc[summary.scenario.eq("A_chronology"), "n"]) == {10000}
    for region in regions.values():
        np.testing.assert_allclose(region["center"], expected)


def test_saved_primary_summaries_reproduce_and_sampling_confidence_is_unchanged():
    context = combined_likelihood.build_context()
    fit = combined_likelihood.fit_catalogue(context.events, context)
    ages = pd.read_csv(analysis.AGE_RESULTS)
    replicates = pd.read_csv(analysis.OUTPUT_DIR / "effect_replicates.csv")
    summary, regions = analysis.summarize_effects(fit, ages, replicates)
    saved = pd.read_csv(analysis.OUTPUT_DIR / "effect_summary.csv")
    # Cached CSVs and fresh optimizer fits differ slightly in numerical precision.
    pd.testing.assert_frame_equal(summary, saved, check_dtype=False, rtol=1e-7, atol=1e-7)
    curves = analysis.build_curve_table(fit, regions)
    pd.testing.assert_frame_equal(curves, pd.read_csv(analysis.OUTPUT_DIR / "phase_response_bands.csv"),
                                  check_dtype=False, rtol=1e-7, atol=1e-7)
    sampling = summary.loc[summary.scenario.eq("B_sampling")].set_index("quantity")
    np.testing.assert_allclose(sampling.loc["preferred_phase_deg", ["low", "high"]].to_numpy(float),
                               [287.83101837, 370.22635637], rtol=1e-7)
    np.testing.assert_allclose(sampling.loc["max_min_rate_ratio", ["low", "high"]].to_numpy(float),
                               [1.72826337, 29.87573929], rtol=1e-7)
    age_coefficients = ages.loc[ages.fit_valid, ["beta_pre_phase_sin", "beta_pre_phase_cos"]]
    assert regions["A_chronology"]["n"] == 9982
    np.testing.assert_allclose(regions["A_chronology"]["covariance"], np.cov(age_coefficients, rowvar=False))
    # All curve envelopes must enclose every curve from the corresponding ellipse.
    radians = np.deg2rad(curves.phase_deg)
    directions = np.column_stack((np.sin(radians), np.cos(radians)))
    for scenario, name in analysis.SCENARIOS.items():
        boundary = effect.ellipse_boundary(regions[scenario], np.linspace(0, 2*np.pi, 1000))
        response = np.exp(boundary @ directions.T)
        assert np.all(response >= curves[f"{name}_low"].to_numpy() - 1e-12)
        assert np.all(response <= curves[f"{name}_high"].to_numpy() + 1e-12)


def test_zero_phase_full_simulator_matches_reduced_continuous_process(setup):
    _, context, fit, _, _ = setup
    model = replace(fit.full, beta=np.r_[fit.reduced.beta, 0., 0.])
    first = combined_likelihood.simulate_reduced_model_events(context, fit.reduced, np.random.default_rng(15))
    second = effect.simulate_full_model_events(context, model, np.random.default_rng(15))
    pd.testing.assert_frame_equal(first, second)


def test_full_generator_checks_terms_and_nonpositive_history(setup):
    _, context, fit, _, _ = setup
    with pytest.raises(ValueError, match="full model"):
        effect.prepare_full_simulation(context, fit.reduced)
    beta = fit.full.beta.copy()
    beta[fit.full.terms.index(combined_likelihood.HISTORY_TERM)] = 0.1
    with pytest.raises(ValueError, match="Positive feedback"):
        effect.prepare_full_simulation(context, replace(fit.full, beta=beta))


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
    events = combined_likelihood.load_event_catalogue()
    context = combined_likelihood.build_context()
    fit = combined_likelihood.fit_catalogue(events, context)
    # Refit a small saved chronology sample, independent of stale output tables.
    saved = pd.read_csv(analysis.AGE_DRAWS)
    columns = [f"age_kyr_bp__{event_id}" for event_id in events.event_id]
    rows, results = [], []
    for _, row in saved.iterrows():
        local_events = events.copy()
        local_events[combined_likelihood.EVENT_AGE_COLUMN] = row[columns].to_numpy(float)
        try:
            local_fit = combined_likelihood.fit_catalogue(local_events, context)
        except ValueError:
            continue
        rows.append(row)
        sine, cosine = effect.phase_coefficients(local_fit.full)
        results.append({"fit_valid": True, "invalid_reason": "", **local_fit.summary,
                        "beta_pre_phase_sin": sine, "beta_pre_phase_cos": cosine})
        if len(rows) == 20:
            break
    return events, context, fit, pd.DataFrame(rows).reset_index(drop=True), pd.DataFrame(results)


def generator(model, context):
    return {"model": model, "events": context.events.copy()}


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
    np.testing.assert_array_equal(generators[0]["model"].beta, fit.full.beta)
    assert table.age_realization_id.tolist() == draws.iloc[expected].realization_id.tolist()


def test_parallel_replicates_are_reproducible_and_do_not_freeze_event_count(setup):
    _, context, fit, _, _ = setup
    generators = {0: generator(fit.full, context), 1: generator(fit.full, context)}
    first = analysis.run_simulations(context, generators, 20, 3, 9, workers=1, show_progress=False)
    second = analysis.run_simulations(context, generators, 20, 3, 9, workers=2, show_progress=False)
    pd.testing.assert_frame_equal(first, second)
    assert first.fit_valid.all()
    assert first.n_response_events.nunique() > 1
    assert first.groupby("scenario").size().to_dict() == {"B_sampling": 20, "C_joint": 3}


def test_failed_simulations_are_retained_without_retries(setup, monkeypatch):
    _, context, fit, _, _ = setup
    def fail(*args, **kwargs):
        raise effect.InvalidEffectSimulation("diagnosed numerical failure")
    monkeypatch.setattr(effect, "simulate_prepared_full_events", fail)
    generators = {i: generator(fit.full, context) for i in (0, 1)}
    table = analysis.run_simulations(context, generators, 2, 2, 1, show_progress=False)
    assert len(table) == 4
    assert not table.fit_valid.any()
    assert table.invalid_reason.str.len().gt(0).all()
    with pytest.raises(RuntimeError, match="failures"):
        analysis.summarize_effects(fit, setup[-1], table)


def test_small_analysis_summaries_keep_interval_meanings_and_equal_weights(setup):
    _, context, fit, _, age_results = setup
    generators = {i: generator(fit.full, context) for i in range(3)}
    table = analysis.run_simulations(context, generators, 30, 15, 10, show_progress=False)
    summary, regions = analysis.summarize_effects(fit, age_results, table)
    assert len(summary) == 6
    assert summary.loc[summary.scenario.eq("B_sampling"), "interval_type"].str.contains("confidence").all()
    assert summary.loc[summary.scenario.eq("C_joint"), "interval_type"].str.contains("working").all()
    with pytest.raises(ValueError, match="equal weight"):
        analysis.summarize_effects(fit, age_results, table.iloc[:-1])
    curves = analysis.build_curve_table(fit, regions)
    for name in analysis.SCENARIOS.values():
        assert (curves[f"{name}_low"] <= curves.point_multiplier).all()
        assert (curves[f"{name}_high"] >= curves.point_multiplier).all()


def test_age_generators_simulate_with_their_own_exact_anchors_and_support(setup):
    _, context, _, _, _ = setup
    generators, selected = analysis.build_generators(*setup, n_outer=3, seed=7)
    analysis._initialize_worker(context, generators)
    for outer_id in range(1, 4):
        row = analysis._simulate_one((2, outer_id, 1, 77))
        local_context, prepared = analysis._PREPARED[outer_id]
        expected = combined_likelihood.condition_context(context, generators[outer_id]["events"])
        assert row["fit_valid"]
        assert row["response_exposure_kyr"] == pytest.approx(expected.response_exposure_kyr)
        assert selected.loc[selected.outer_id.eq(outer_id), "response_exposure_kyr"].iloc[0] == pytest.approx(expected.response_exposure_kyr)
        simulated = effect.simulate_prepared_full_events(prepared, np.random.default_rng(77))
        for name, segment in local_context.segments.items():
            assert segment == expected.segments[name]
            ages = simulated.loc[simulated.segment_id.eq(name), combined_likelihood.EVENT_AGE_COLUMN]
            assert ages.max() == segment.anchor_age_kyr_bp
            assert ages.min() >= segment.response_start_kyr_bp
        assert local_context.scaling == context.scaling


def test_zero_response_draws_keep_gof_sample_but_withhold_effect_region(setup, monkeypatch):
    _, context, fit, _, age_results = setup
    anchors = context.events.sort_values(combined_likelihood.EVENT_AGE_COLUMN).groupby('segment_id').tail(1)
    monkeypatch.setattr(effect, 'simulate_prepared_full_events', lambda *args: anchors.copy())
    generators = {i: generator(fit.full, context) for i in (0, 1)}
    table = analysis.run_simulations(context, generators, 2, 2, 1, show_progress=False)
    assert table.fit_valid.all() and table.n_response_events.eq(0).all()
    assert not table.effect_identified.any()
    assert table[analysis.PHASE_COLUMNS].isna().all().all()
    nominal = table.loc[table.scenario.eq('B_sampling')]
    assert nominal.ks_uniform.eq(0).all() and nominal.adjacent_dependence.eq(0).all()
    assert nominal.residual_status.eq('no_response_events').all()
    with pytest.raises(RuntimeError, match='Unidentified effect'):
        analysis.summarize_effects(fit, age_results, table)


@pytest.mark.parametrize("change", ["generator", "source"])
def test_redraw_rejects_stale_generator_or_changed_input(setup, tmp_path, change):
    import hashlib
    _, context, fit, _, _ = setup
    source = tmp_path / "ages.csv"
    source.write_text("original chronological input")
    saved = pd.DataFrame([dict(zip(analysis.BETA_COLUMNS, fit.full.beta))])
    saved.to_csv(tmp_path / "point_generator.csv", index=False)
    pd.DataFrame([dict(parameter="model_version", value=combined_likelihood.MODEL_VERSION),
                  dict(parameter="history_tau_ka", value=context.history_tau_ka)]).to_csv(
        tmp_path / "parameters_and_provenance.csv", index=False)
    pd.DataFrame([dict(path=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest())]).to_csv(
        tmp_path / "input_code_sha256.csv", index=False)
    analysis.validate_saved_inputs(fit, tmp_path, [source])
    if change == "generator":
        saved.iloc[0, 0] += 0.25
        saved.to_csv(tmp_path / "point_generator.csv", index=False)
    else:
        source.write_text("a different chronology")
    with pytest.raises(ValueError, match="rerun simulations"):
        analysis.validate_saved_inputs(fit, tmp_path, [source])
