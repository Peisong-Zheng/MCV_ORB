"""Continuous null generation, fixed supports and honest bootstrap accounting."""
import numpy as np
import pandas as pd
import pytest
import NGRIP_MIS6_likelihood_bootstrap as bootstrap
from toolbox import event_model, model_stats, sampling
from toolbox.point_process import PointProcessFitError


@pytest.fixture(scope="module")
def point_setup():
    return bootstrap.run_analysis(n_bootstrap=1, seed=91)


def phase_sample(point, **options):
    return sampling.phase_bootstrap(
        point['events'], point['windows'], point['forcings'], point['phase_anchors'], point['scaling'],
        point['reduced'], point['full'], reduced_terms=point['reduced'].terms,
        full_terms=point['full'].terms, **options)


def test_plus_one_p_value_and_exact_binomial_interval():
    p, count = model_stats.empirical_p_value(np.array([0., 1., 2., 3.]), 2)
    assert count == 2 and p == 3 / 5
    assert model_stats.empirical_p_value(np.array([0., 1.]), 2) == (1 / 3, 0)
    low, high = model_stats.clopper_pearson_interval(0, 99)
    assert low == 0 and high == pytest.approx(1 - .025**(1 / 99))
    with pytest.raises(ValueError):
        model_stats.empirical_p_value(np.array([np.nan]), 2)


def test_continuous_null_preserves_exact_anchors_and_rebuilds_history(point_setup):
    point = point_setup
    prepared = event_model.prepare_model_simulation(point['events'], point['windows'], point['forcings'],
        point['phase_anchors'], point['scaling'], point['reduced'])
    generated = event_model.simulate_prepared_events(prepared, np.random.default_rng(192))
    assert not np.array_equal(generated.event_age_kyr_bp, point['events'].event_age_kyr_bp)
    event_x, integral_x = event_model.build_design(generated, point['windows'], point['forcings'],
        point['phase_anchors'], point['scaling'])
    exposure = (point['windows'].response_end_kyr_bp - point['windows'].response_start_kyr_bp).sum()
    assert integral_x.weight.sum() == pytest.approx(exposure)
    for window in point['windows'].itertuples(index=False):
        ages = generated.loc[generated.segment_id.eq(window.segment_id), 'event_age_kyr_bp'].to_numpy()
        assert ages.max() == window.anchor_age_kyr_bp
        anchor = generated.loc[generated.segment_id.eq(window.segment_id) &
                               generated.event_age_kyr_bp.eq(window.anchor_age_kyr_bp)]
        assert anchor.event_id.item() in point['events'].event_id.values
        frame = event_x.loc[event_x.segment_id.eq(window.segment_id)]
        assert (frame.age_kyr_bp < window.anchor_age_kyr_bp).all()
        expected = [sum(np.exp((age - older) / 1.5) for older in ages if older > age) for age in frame.age_kyr_bp]
        np.testing.assert_allclose(frame.same_type_exponential_history, expected)
    fraction = generated.loc[~generated.event_id.isin(point['events'].event_id), 'event_age_kyr_bp'] / .2
    assert np.any(np.abs(fraction - np.round(fraction)) > .01)


def test_replicates_reproduce_across_worker_counts(point_setup):
    serial, errors = phase_sample(point_setup, n_bootstrap=4, seed=87)
    parallel, parallel_errors = phase_sample(point_setup, n_bootstrap=4, seed=87, workers=2)
    pd.testing.assert_frame_equal(serial, parallel)
    assert not errors and not parallel_errors
    assert serial.fit_valid.all() and serial.LR_statistic.ge(0).all()
    assert serial.n_events_response.nunique() > 1
    summary = model_stats.phase_bootstrap_summary(point_setup['summary'].iloc[0], serial, errors).iloc[0]
    assert summary.n_bootstrap == summary.n_valid_replicates == 4
    assert summary.n_resampled_catalogues == 0
    assert summary.empirical_p_plus_one >= 1 / 5


def test_numerical_retry_keeps_the_same_generated_catalogue(point_setup, monkeypatch):
    generated, attempts = [], []
    original_simulate, original_design = event_model.simulate_prepared_events, event_model.build_design
    original_fit = sampling.fit_point_process
    fit_calls = []
    def simulate(prepared, rng):
        events = original_simulate(prepared, rng)
        generated.append(events)
        return events
    def design(events, *args, **kwargs):
        attempts.append((events, kwargs['quadrature_order']))
        return original_design(events, *args, **kwargs)
    def fit(*args, **kwargs):
        fit_calls.append(1)
        if len(fit_calls) == 1:
            raise PointProcessFitError('intentional same-data refinement')
        return original_fit(*args, **kwargs)
    monkeypatch.setattr(event_model, 'simulate_prepared_events', simulate)
    monkeypatch.setattr(event_model, 'build_design', design)
    monkeypatch.setattr(sampling, 'fit_point_process', fit)
    rows, errors = phase_sample(point_setup, n_bootstrap=1, seed=12)
    assert rows.fit_valid.all() and rows.solver_attempts.item() == 2 and not errors
    assert len(generated) == 1 and attempts[0][0] is attempts[1][0] is generated[0]
    assert [order for _, order in attempts] == [4, 8]


def test_failed_replicate_is_saved_and_prevents_calibrated_p(point_setup, monkeypatch):
    def fail(*args, **kwargs): raise PointProcessFitError('unresolved finite optimum')
    monkeypatch.setattr(sampling, 'fit_point_process', fail)
    rows, failures = phase_sample(point_setup, n_bootstrap=2, seed=45)
    assert len(rows) == 2 and not rows.fit_valid.any()
    assert rows.solver_attempts.eq(2).all()
    summary = model_stats.phase_bootstrap_summary(point_setup['summary'].iloc[0], rows, failures).iloc[0]
    assert summary.n_failed_replicates == 2 and np.isnan(summary.empirical_p_plus_one)


def test_all_empty_response_is_lr_zero_and_g_undefined(point_setup, monkeypatch):
    anchors = point_setup['events'].loc[point_setup['events'].event_role.eq('conditioning')].copy()
    monkeypatch.setattr(event_model, 'simulate_prepared_events', lambda *args: anchors)
    rows, failures = phase_sample(point_setup, n_bootstrap=1, seed=9)
    row = rows.iloc[0]
    assert row.status == 'zero_events' and row.fit_valid
    assert row.LR_statistic == 0 and np.isnan(row.gain_bits_per_event)
    summary = model_stats.phase_bootstrap_summary(point_setup['summary'].iloc[0], rows, failures).iloc[0]
    assert summary.n_zero_event_replicates == 1 and summary.empirical_p_plus_one == .5


def test_compact_outputs_keep_no_binned_experiment(point_setup, tmp_path):
    result = bootstrap.run_analysis(n_bootstrap=2, seed=83)
    bootstrap.save_tables(result, tmp_path)
    assert (tmp_path / 'bootstrap_replicates.csv').exists()
    assert (tmp_path / 'failed_replicates.csv').exists()
    assert not list(tmp_path.glob('*bin*'))
    parameters = result['parameters'].set_index('parameter').value
    assert parameters.history_coefficient_domain == 'beta_H <= 0'
    assert parameters.history_tau_kyr == 1.5
    assert bootstrap.N_BOOTSTRAP == 9999 and bootstrap.RANDOM_SEED == 20260905
