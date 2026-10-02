"""Full-model generation, joint-region geometry and saved-product redraw."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

import NGRIP_MIS6_effect_uncertainty as analysis
import NGRIP_MIS6_likelihood_bootstrap as primary
from Barker2011 import Barker2011_likelihood_bootstrap as barker_phase
from Barker2011 import Barker2011_effect_uncertainty as barker
from toolbox import event_model, model_stats as effect, sampling
from toolbox.point_process import fit_point_process

ROOT = Path(__file__).resolve().parents[1]


def make_generators(point, draws, results, *, n_outer=3, seed=7, age_columns=None, source_ids=()):
    columns = age_columns or [f'age_kyr_bp__{x}' for x in point['events'].event_id]
    return sampling.effect_generators(point['events'], point['observations'], point['windows'],
        point['forcings'], point['phase_anchors'], point['scaling'], point['full'], draws, results, columns,
        reduced_terms=point['reduced'].terms, full_terms=point['full'].terms,
        n_outer=n_outer, seed=seed, source_id_columns=source_ids)


def sample(point, generators, n_point, n_inner, seed, workers=1):
    return sampling.sample_effect(generators, point['forcings'], point['phase_anchors'], point['scaling'],
        reduced_terms=point['reduced'].terms, full_terms=point['full'].terms,
        n_point=n_point, n_inner=n_inner, seed=seed, workers=workers, show_progress=False)


def nominal_generator(point, model=None):
    return dict(model=point['full'] if model is None else model,
                events=point['events'].copy(), windows=point['windows'].copy())


@pytest.fixture(scope='module')
def setup():
    point = primary.run_analysis(n_bootstrap=1, seed=91)
    draws = pd.read_csv(analysis.AGE_DRAWS)
    columns = [f'age_kyr_bp__{x}' for x in point['events'].event_id]
    rows, results = [], []
    for _, row in draws.iterrows():
        events = point['events'].copy()
        events['event_age_kyr_bp'] = row[columns].to_numpy(float)
        try:
            windows = event_model.response_windows(events, point['observations'])
        except ValueError:
            continue
        event_x, integral_x = event_model.build_design(events, windows, point['forcings'],
            point['phase_anchors'], point['scaling'])
        for frame in (event_x, integral_x):
            frame['mis6_segment'] = frame.segment_id.eq('MIS6').astype(float)
        reduced_terms, full_terms = point['reduced'].terms, point['full'].terms
        reduced = fit_point_process(event_x[list(reduced_terms)], integral_x[list(reduced_terms)],
                                    integral_x.weight, reduced_terms)
        full = fit_point_process(event_x[list(full_terms)], integral_x[list(full_terms)], integral_x.weight,
                                 full_terms, start_beta=np.r_[reduced.beta, 0., 0.])
        summary = effect.fit_summary(reduced, full, event_x, windows, n_source_events=len(events),
                                     catalogue_id='ngrip_warming_plus_mis6')
        sine, cosine = effect.phase_coefficients(full)
        rows.append(row)
        results.append(dict(fit_valid=True, invalid_reason='', **summary,
                            beta_pre_phase_sin=sine, beta_pre_phase_cos=cosine))
        if len(rows) == 20:
            break
    return point, pd.DataFrame(rows).reset_index(drop=True), pd.DataFrame(results)


def test_barker_uses_named_phase_coefficients_and_its_own_age_support():
    point = barker_phase.run_analysis(n_bootstrap=1, seed=91)
    full = point['full']
    assert 'mis6_segment' not in full.terms
    expected = [full.beta[full.terms.index(term)] for term in effect.PHASE_TERMS]
    np.testing.assert_array_equal(effect.phase_coefficients(full), expected)
    np.testing.assert_array_equal(effect.phase_coefficients(
        replace(full, terms=full.terms[::-1], beta=full.beta[::-1])), expected)
    draws, age_results, columns = barker.load_age_inputs(point['events'])
    generators, selected = make_generators(point, draws, age_results, n_outer=2, seed=25, age_columns=columns)
    assert selected.age_realization_id.is_unique and 'beta__mis6_segment' not in selected
    first = sample(point, generators, 24, 10, 25)
    pd.testing.assert_frame_equal(first, sample(point, generators, 24, 10, 25, 2))
    assert first.fit_valid.all() and 'beta__mis6_segment' not in first
    assert first.n_response_events.nunique() > 1
    for outer in (1, 2):
        windows = generators[outer]['windows']
        assert windows.response_start_kyr_bp.item() == 0
        assert windows.anchor_age_kyr_bp.item() == generators[outer]['events'].event_age_kyr_bp.max()
        np.testing.assert_allclose(first.loc[first.outer_id.eq(outer), 'response_exposure_kyr'],
                                   windows.response_end_kyr_bp.item())
    summary, regions = effect.summarize_effects(full, age_results, first)
    assert set(summary.loc[summary.scenario.eq('A_chronology'), 'n']) == {10000}
    for region in regions.values():
        np.testing.assert_allclose(region['center'], expected)


def test_saved_primary_summaries_reproduce_and_sampling_confidence_is_unchanged(setup):
    full = setup[0]['full']
    ages = pd.read_csv(analysis.AGE_RESULTS)
    output_dir = ROOT / 'data/processed' / analysis.RUN_NAME
    replicates = pd.read_csv(output_dir / 'effect_replicates.csv')
    summary, regions = effect.summarize_effects(full, ages, replicates)
    pd.testing.assert_frame_equal(summary, pd.read_csv(output_dir / 'effect_summary.csv'),
                                  check_dtype=False, rtol=1e-7, atol=1e-7)
    curves = effect.build_curve_table(full, regions)
    pd.testing.assert_frame_equal(curves, pd.read_csv(output_dir / 'phase_response_bands.csv'),
                                  check_dtype=False, rtol=1e-7, atol=1e-7)
    nominal = summary.loc[summary.scenario.eq('B_sampling')].set_index('quantity')
    np.testing.assert_allclose(nominal.loc['preferred_phase_deg', ['low', 'high']].to_numpy(float),
                               [287.83101837, 370.22635637], rtol=1e-7)
    np.testing.assert_allclose(nominal.loc['max_min_rate_ratio', ['low', 'high']].to_numpy(float),
                               [1.72826337, 29.87573929], rtol=1e-7)
    assert regions['A_chronology']['n'] == 9982
    np.testing.assert_allclose(regions['A_chronology']['covariance'], np.cov(
        ages.loc[ages.fit_valid, ['beta_pre_phase_sin', 'beta_pre_phase_cos']], rowvar=False))
    directions = np.column_stack((np.sin(np.deg2rad(curves.phase_deg)), np.cos(np.deg2rad(curves.phase_deg))))
    for scenario, name in effect.SCENARIOS.items():
        coefficients = effect.ellipse_boundary(regions[scenario], np.linspace(0, 2 * np.pi, 1000))
        response = np.exp(coefficients @ directions.T)
        assert np.all(response >= curves[f'{name}_low'].to_numpy() - 1e-12)
        assert np.all(response <= curves[f'{name}_high'].to_numpy() + 1e-12)


def test_zero_phase_full_simulator_matches_reduced_continuous_process(setup):
    point = setup[0]
    model = replace(point['full'], beta=np.r_[point['reduced'].beta, 0., 0.])
    args = [point[key] for key in ('events', 'windows', 'forcings', 'phase_anchors', 'scaling')]
    first = event_model.simulate_prepared_events(event_model.prepare_model_simulation(*args, point['reduced']),
                                              np.random.default_rng(15))
    second = event_model.simulate_prepared_events(event_model.prepare_model_simulation(*args, model),
                                               np.random.default_rng(15))
    pd.testing.assert_frame_equal(first, second)


def test_full_generator_checks_terms_and_nonpositive_history(setup):
    point = setup[0]
    with pytest.raises(ValueError, match='full model'):
        sample(point, {0: nominal_generator(point, point['reduced'])}, 2, 2, 5)
    beta = point['full'].beta.copy()
    beta[point['full'].terms.index(sampling.HISTORY_TERM)] = .1
    with pytest.raises(ValueError, match='Positive feedback'):
        sample(point, {0: nominal_generator(point, replace(point['full'], beta=beta))}, 2, 2, 5)


def test_outer_selection_is_reproducible_uniform_rule_and_preserves_source_ids(setup):
    point, draws, results = setup
    generators, table = make_generators(*setup, source_ids=('ngrip_realization_id', 'mis6_realization_id'))
    _, repeat = make_generators(*setup, source_ids=('ngrip_realization_id', 'mis6_realization_id'))
    pd.testing.assert_frame_equal(table, repeat)
    indices = np.random.default_rng(np.random.SeedSequence([7, 100])).choice(np.flatnonzero(results.fit_valid), 3, replace=False)
    np.testing.assert_array_equal(table.source_row_index, indices)
    assert table.age_realization_id.is_unique and set(generators) == {0, 1, 2, 3}
    np.testing.assert_array_equal(generators[0]['model'].beta, point['full'].beta)
    assert table.age_realization_id.tolist() == draws.iloc[indices].realization_id.tolist()
    for key in ('ngrip_realization_id', 'mis6_realization_id'):
        np.testing.assert_array_equal(table[key], draws.iloc[indices][key])


def test_parallel_replicates_are_reproducible_and_do_not_freeze_event_count(setup):
    point = setup[0]
    generators = {i: nominal_generator(point) for i in (0, 1)}
    first = sample(point, generators, 20, 3, 9)
    pd.testing.assert_frame_equal(first, sample(point, generators, 20, 3, 9, 2))
    assert first.fit_valid.all() and first.n_response_events.nunique() > 1
    assert first.groupby('scenario').size().to_dict() == {'B_sampling': 20, 'C_joint': 3}


def test_failed_simulations_are_retained_without_retries(setup, monkeypatch):
    point = setup[0]
    def fail(*args, **kwargs): raise sampling.InvalidEffectSimulation('diagnosed numerical failure')
    monkeypatch.setattr(event_model, 'simulate_prepared_events', fail)
    table = sample(point, {i: nominal_generator(point) for i in (0, 1)}, 2, 2, 1)
    assert len(table) == 4 and not table.fit_valid.any() and table.invalid_reason.str.len().gt(0).all()
    with pytest.raises(RuntimeError, match='failures'):
        effect.summarize_effects(point['full'], setup[-1], table)


def test_small_analysis_summaries_keep_interval_meanings_and_equal_weights(setup):
    point = setup[0]
    table = sample(point, {i: nominal_generator(point) for i in range(3)}, 30, 15, 10)
    summary, regions = effect.summarize_effects(point['full'], setup[-1], table)
    assert len(summary) == 6
    assert summary.loc[summary.scenario.eq('B_sampling'), 'interval_type'].str.contains('confidence').all()
    assert summary.loc[summary.scenario.eq('C_joint'), 'interval_type'].str.contains('working').all()
    with pytest.raises(ValueError, match='equal weight'):
        effect.summarize_effects(point['full'], setup[-1], table.iloc[:-1])
    curves = effect.build_curve_table(point['full'], regions)
    for name in effect.SCENARIOS.values():
        assert (curves[f'{name}_low'] <= curves.point_multiplier).all()
        assert (curves[f'{name}_high'] >= curves.point_multiplier).all()


def test_age_generators_simulate_with_their_own_exact_anchors_and_support(setup):
    point = setup[0]
    generators, selected = make_generators(*setup)
    table = sample(point, generators, 2, 1, 77)
    for outer in range(1, 4):
        generator = generators[outer]
        windows = generator['windows']
        expected = event_model.response_windows(generator['events'], point['observations'])
        pd.testing.assert_frame_equal(windows, expected)
        exposure = float((windows.response_end_kyr_bp - windows.response_start_kyr_bp).sum())
        assert table.loc[table.outer_id.eq(outer), 'response_exposure_kyr'].item() == pytest.approx(exposure)
        assert selected.loc[selected.outer_id.eq(outer), 'response_exposure_kyr'].item() == pytest.approx(exposure)
        prepared = event_model.prepare_model_simulation(generator['events'], windows, point['forcings'],
            point['phase_anchors'], point['scaling'], generator['model'])
        events = event_model.simulate_prepared_events(prepared, np.random.default_rng(77))
        for window in windows.itertuples(index=False):
            ages = events.loc[events.segment_id.eq(window.segment_id), 'event_age_kyr_bp']
            assert ages.max() == window.anchor_age_kyr_bp and ages.min() >= window.response_start_kyr_bp


def test_zero_response_draws_keep_gof_sample_but_withhold_effect_region(setup, monkeypatch):
    point = setup[0]
    anchors = point['events'].sort_values('event_age_kyr_bp').groupby('segment_id').tail(1)
    monkeypatch.setattr(event_model, 'simulate_prepared_events', lambda *args: anchors.copy())
    table = sample(point, {i: nominal_generator(point) for i in (0, 1)}, 2, 2, 1)
    assert table.fit_valid.all() and table.n_response_events.eq(0).all()
    assert not table.effect_identified.any() and table[effect.PHASE_COLUMNS].isna().all().all()
    nominal = table.loc[table.scenario.eq('B_sampling')]
    assert nominal.ks_uniform.eq(0).all() and nominal.adjacent_dependence.eq(0).all()
    assert nominal.residual_status.eq('no_response_events').all()
    with pytest.raises(RuntimeError, match='Unidentified effect'):
        effect.summarize_effects(point['full'], setup[-1], table)


@pytest.mark.parametrize('module,folder', [(analysis, 'data/processed'), (barker, 'Barker2011/data/processed')])
def test_redraw_uses_only_saved_plot_products(module, folder, tmp_path, monkeypatch):
    import shutil
    target = tmp_path / folder / module.RUN_NAME
    target.mkdir(parents=True)
    for name in ('effect_summary.csv', 'phase_response_bands.csv', 'coefficient_regions.json'):
        shutil.copyfile(ROOT / folder / module.RUN_NAME / name, target / name)
    def fail(*args, **kwargs): raise AssertionError('Redraw must not read/refit current science inputs')
    monkeypatch.setattr(event_model, 'response_windows', fail)
    monkeypatch.setattr(module, 'load_age_inputs', fail)
    captured = []
    monkeypatch.setattr(module, 'save_figure', lambda *args, **kwargs: captured.append(args))
    monkeypatch.setattr(module, 'REDRAW', True)
    monkeypatch.setattr(module, 'EXPORT_PAPER', False)
    monkeypatch.setattr(module, 'OUTPUT_ROOT', tmp_path)
    module.main()
    assert len(captured) == 1 and set(captured[0][2]) == set(effect.SCENARIOS)
    assert len(list(target.iterdir())) == 3


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


def test_barker_joint_ids_detect_a_missing_whole_chronology():
    from Barker2011 import Barker2011_effect_uncertainty as barker

    selected = pd.DataFrame(dict(outer_id=[1, 2], age_realization_id=[31, 47],
                                 response_exposure_kyr=[396., 397.]))
    rows = [dict(scenario=scenario, outer_id=outer, replicate_id=replicate, seed=25,
                 response_exposure_kyr=exposure)
            for scenario, outer, exposure in [("B_sampling", 0, 396.),
                                               ("C_joint", 1, 396.), ("C_joint", 2, 397.)]
            for replicate in (1, 2)]
    replicates = pd.DataFrame(rows)
    barker.validate_simulation_ids(replicates, selected, 2, 2, 2, 25)
    with pytest.raises(ValueError, match="selected chronologies"):
        barker.validate_simulation_ids(replicates.loc[replicates.outer_id.ne(2)],
                                      selected, 2, 2, 2, 25)
    with pytest.raises(ValueError, match="inner draws"):
        barker.validate_simulation_ids(replicates.iloc[:-1], selected, 2, 2, 2, 25)
