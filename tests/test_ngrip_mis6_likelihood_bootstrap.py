"""Continuous null generation, fixed supports and honest bootstrap accounting."""
from collections import Counter
import numpy as np
import pandas as pd
import pytest
import NGRIP_MIS6_likelihood_bootstrap as bootstrap
from toolbox import combined_likelihood as c
from toolbox.point_process import PointProcessFitError


@pytest.fixture(scope="module")
def point_setup():
    context = c.build_context()
    return context, c.fit_catalogue(context.events, context)


def test_plus_one_p_value_and_exact_binomial_interval():
    p, count = bootstrap.empirical_p_value(np.array([0., 1., 2., 3.]), 2)
    assert count == 2 and p == 3 / 5
    assert bootstrap.empirical_p_value(np.array([0., 1.]), 2) == (1 / 3, 0)
    low, high = bootstrap.clopper_pearson_interval(0, 99)
    assert low == 0 and high == pytest.approx(1 - .025**(1 / 99))
    with pytest.raises(ValueError):
        bootstrap.empirical_p_value(np.array([np.nan]), 2)


def test_continuous_null_preserves_exact_anchors_and_rebuilds_history(point_setup):
    context, fit = point_setup
    prepared = c.prepare_model_simulation(context, fit.reduced)
    generated = c.simulate_prepared_events(prepared, np.random.default_rng(192))
    assert not np.array_equal(generated.event_age_kyr_bp, context.events.event_age_kyr_bp)
    design = c.prepare_catalogue(generated, context, fixed_support=True)
    assert design.weights.sum() == pytest.approx(context.response_exposure_kyr)
    for segment_id, segment in context.segments.items():
        ages = generated.loc[generated.segment_id.eq(segment_id), "event_age_kyr_bp"].to_numpy()
        assert ages.max() == segment.anchor_age_kyr_bp
        anchor = generated.loc[(generated.segment_id == segment_id) &
                               (generated.event_age_kyr_bp == segment.anchor_age_kyr_bp)]
        assert anchor.event_id.item() in context.events.event_id.values
        frame = design.event_frame.loc[design.event_frame.segment_id.eq(segment_id)]
        assert (frame.age_kyr_bp < segment.anchor_age_kyr_bp).all()
        expected = [sum(np.exp((age - older) / 1.5) for older in ages if older > age)
                    for age in frame.age_kyr_bp]
        np.testing.assert_allclose(frame.same_type_exponential_history, expected)
    # Actual events are continuous within old 0.2-kyr bins, not their centers.
    fraction = generated.loc[~generated.event_id.isin(context.events.event_id), "event_age_kyr_bp"] / .2
    assert np.any(np.abs(fraction - np.round(fraction)) > .01)


def test_replicates_reproduce_across_worker_counts(point_setup):
    context, fit = point_setup
    serial, errors = bootstrap.run_bootstrap(fit, context, n_bootstrap=4, seed=87)
    parallel, parallel_errors = bootstrap.run_bootstrap(fit, context, n_bootstrap=4, seed=87, n_workers=2)
    pd.testing.assert_frame_equal(serial, parallel)
    assert not errors and not parallel_errors
    assert serial.fit_valid.all() and serial.LR_statistic.ge(0).all()
    assert serial.n_events_response.nunique() > 1
    summary = bootstrap.build_summary(fit, serial, errors).iloc[0]
    assert summary.n_bootstrap == summary.n_valid_replicates == 4
    assert summary.n_resampled_catalogues == 0
    assert summary.empirical_p_plus_one >= 1 / 5


def test_numerical_retry_keeps_the_same_generated_catalogue(point_setup, monkeypatch):
    context, fit = point_setup
    generated = []
    simulated = c.simulate_prepared_events
    fitted = c.fit_catalogue
    attempts = []
    def simulate(prepared, rng):
        events = simulated(prepared, rng)
        generated.append(events)
        return events
    def refit(events, context, **kwargs):
        attempts.append(events)
        if len(attempts) == 1:
            raise PointProcessFitError("intentional same-data refinement")
        return fitted(events, context, **kwargs)
    monkeypatch.setattr(c, "simulate_prepared_events", simulate)
    monkeypatch.setattr(c, "fit_catalogue", refit)
    row = bootstrap._replicate((1, np.random.SeedSequence(12)), context,
                               c.prepare_model_simulation(context, fit.reduced))
    assert row["fit_valid"] and row["solver_attempts"] == 2
    assert len(generated) == 1 and attempts[0] is attempts[1] is generated[0]


def test_failed_replicate_is_saved_and_prevents_calibrated_p(point_setup, monkeypatch):
    context, fit = point_setup
    def fail(*args, **kwargs):
        raise PointProcessFitError("unresolved finite optimum")
    monkeypatch.setattr(c, "fit_catalogue", fail)
    rows, failures = bootstrap.run_bootstrap(fit, context, n_bootstrap=2, seed=45)
    assert len(rows) == 2 and not rows.fit_valid.any()
    assert rows.solver_attempts.eq(2).all()
    summary = bootstrap.build_summary(fit, rows, failures).iloc[0]
    assert summary.n_failed_replicates == 2 and np.isnan(summary.empirical_p_plus_one)


def test_all_empty_response_is_lr_zero_and_g_undefined(point_setup, monkeypatch):
    context, fit = point_setup
    anchors = fit.design.all_events.loc[fit.design.all_events.event_role.eq("conditioning")].copy()
    monkeypatch.setattr(c, "simulate_prepared_events", lambda *args: anchors)
    row = bootstrap._replicate((1, np.random.SeedSequence(9)), context, [])
    assert row["status"] == "zero_events" and row["fit_valid"]
    assert row["LR_statistic"] == 0 and np.isnan(row["gain_bits_per_event"])
    summary = bootstrap.build_summary(fit, pd.DataFrame([row]), Counter()).iloc[0]
    assert summary.n_zero_event_replicates == 1 and summary.empirical_p_plus_one == .5


def test_compact_outputs_keep_no_binned_experiment(point_setup, tmp_path):
    result = bootstrap.run_analysis(n_bootstrap=2, seed=83)
    bootstrap.save_tables(result, tmp_path)
    assert (tmp_path / "bootstrap_replicates.csv").exists()
    assert (tmp_path / "failed_replicates.csv").exists()
    assert not list(tmp_path.glob("*bin*"))
    parameters = result["parameters"].set_index("parameter").value
    assert parameters.history_coefficient_domain == "beta_H <= 0"
    assert parameters.history_tau_kyr == 1.5
    assert bootstrap.DEFAULT_N_BOOTSTRAP == 9999 and bootstrap.DEFAULT_SEED == 20260905
