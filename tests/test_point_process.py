"""Scientific checks for exact-age likelihood, history and event generation."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.optimize._numdiff import approx_derivative

from toolbox.point_process import (
    PointProcessFitError,
    exponential_history,
    fit_point_process,
    gauss_legendre_intervals,
    negative_loglik_gradient,
    negative_loglik_hessian,
    point_process_loglik,
    simulate_segment_events,
)


def test_quadrature_duration_and_polynomial_integral():
    nodes, weights = gauss_legendre_intervals([0, 0.3, 1.1, 3], order=4)
    assert np.all(weights > 0)
    assert_allclose(weights.sum(), 3)
    assert_allclose(weights @ nodes**6, 3**7 / 7, rtol=2e-14)
    assert not np.isin(nodes, [0, 0.3, 1.1, 3]).any()


def test_history_uses_strictly_older_events_and_preserves_query_order():
    events = np.array([6.0, 10.0, 8.0])
    query = np.array([8.0, 9.0, 6.0, 10.0, 7.0])
    expected = np.array([sum(np.exp(-(a - q) / 1.5) for a in events if a > q)
                         for q in query])
    actual = exponential_history(query, events)
    assert_allclose(actual, expected)
    assert actual[3] == 0  # The conditioning event cannot predict itself.
    assert_allclose(exponential_history(query + 250, events + 250), actual)
    assert_allclose(exponential_history(query, events, initial_history=0.5),
                    expected + 0.5 * np.exp((query - 10) / 1.5))
    assert_allclose(exponential_history(np.array([[8, 9], [6, 10]]), events),
                    actual[:4].reshape(2, 2))
    assert exponential_history(10, events) == 0
    with pytest.raises(ValueError, match="Coincident"):
        exponential_history([8], [10, 10])
    with pytest.raises(ValueError, match="precede"):
        exponential_history([11], events)


def test_event_jump_and_segment_reset():
    before = exponential_history([8.0], [10.0, 8.0])[0]
    after = exponential_history([8.0 - 1e-8], [10.0, 8.0])[0]
    assert_allclose(after, before + 1, atol=2e-8)
    assert exponential_history([8], [8])[0] == 0
    assert_allclose(exponential_history([8, 7], [], anchor_age=8, initial_history=1),
                    [1, np.exp(-1 / 1.5)])


def test_history_preserves_event_limits_when_elapsed_times_round_together():
    anchor, event = 396.464, 11.402
    before, after = np.nextafter(event, np.inf), np.nextafter(event, -np.inf)
    # Subtracting the distant anchor collapses these distinct BP coordinates.
    assert anchor - before == anchor - event == anchor - after
    catalogue = np.array([anchor, event + 1, event])
    prior_history = np.exp(-(catalogue[catalogue > event] - event) / 1.5).sum()
    query = np.array([after, before, event])
    actual = exponential_history(query, catalogue, anchor_age=anchor)
    assert_allclose(actual, [prior_history + 1, prior_history, prior_history], atol=1e-14)
    assert exponential_history(event, catalogue, anchor_age=anchor) == pytest.approx(prior_history)


def test_homogeneous_likelihood_and_mle_include_event_free_tail():
    nodes, weights = gauss_legendre_intervals([0, 0.4, 1.5, 2.0, 4.0])
    events = np.ones((3, 1))
    design = np.ones((len(nodes), 1))
    fit = fit_point_process(events, design, weights, ["intercept"])
    assert fit.converged and fit.identifiable
    assert_allclose(fit.beta, [np.log(3 / 4)], atol=1e-10)
    assert_allclose(fit.log_likelihood, 3 * np.log(3 / 4) - 3)
    assert_allclose(fit.aic, 2 - 2 * fit.log_likelihood)
    # No factorial or bin-duration term appears in the continuous likelihood.
    assert_allclose(point_process_loglik([0], events, design, weights), -4)


def test_gradient_hessian_and_concavity():
    nodes, weights = gauss_legendre_intervals([0, 0.5, 2.0, 5.0])
    events = np.column_stack([np.ones(4), [0.2, -0.1, 0.8, 0.3], [0.6, 0.2, 0.1, 0.3]])
    design = np.column_stack([np.ones(len(nodes)), np.sin(nodes), np.exp(-nodes)])
    beta = np.array([-0.4, 0.3, -0.8])
    analytic = negative_loglik_gradient(beta, events, design, weights)
    numeric = approx_derivative(
        lambda b: -point_process_loglik(b, events, design, weights), beta).ravel()
    assert_allclose(analytic, numeric, atol=2e-9)
    hessian = negative_loglik_hessian(beta, design, weights)
    assert_allclose(hessian, approx_derivative(
        lambda b: negative_loglik_gradient(b, events, design, weights), beta), atol=2e-9)
    assert np.linalg.eigvalsh(hessian).min() > 0
    assert point_process_loglik([800, 0, 0], events, design, weights) == -np.inf


def test_continuous_history_integral_matches_independent_quadrature():
    events = np.array([10.0, 8.6, 7.2, 3.7])
    boundaries = np.sort(np.r_[events, 0.0])
    nodes, weights = gauss_legendre_intervals(boundaries, order=16)
    design = np.column_stack([np.ones(len(nodes)), exponential_history(nodes, events)])
    integral = weights @ np.exp(design @ [-0.7, -1.4])
    expected = sum(quad(lambda q: np.exp(-0.7 - 1.4 * sum(
        np.exp((q - a) / 1.5) for a in events if a > q)), lo, hi,
        epsabs=1e-11)[0] for lo, hi in zip(boundaries[:-1], boundaries[1:]))
    assert_allclose(integral, expected, atol=2e-11)


def test_nonpositive_constraint_and_zero_boundary():
    # The unconstrained grouped rates are 1 and 4; inhibition permits equality.
    design = np.array([[1.0, 0.0], [1.0, 1.0]])
    events = np.array([[1.0, 0.0]] + [[1.0, 1.0]] * 4)
    fit = fit_point_process(events, design, [1, 1],
                            ["intercept", "same_type_exponential_history"])
    assert_allclose(fit.beta, [np.log(2.5), 0], atol=1e-7)
    assert fit.gradient_max < 1e-6
    unrestricted = fit_point_process(events, design, [1, 1], ["intercept", "elapsed"])
    assert_allclose(unrestricted.beta, [0, np.log(4)], atol=1e-6)


def test_negative_history_interior_and_nested_likelihood():
    design = np.array([[1.0, 0.0], [1.0, 1.0]])
    events = np.array([[1.0, 0.0]] * 5 + [[1.0, 1.0]])
    fit = fit_point_process(events, design, [2, 2],
                            ["intercept", "same_type_exponential_history"])
    null = fit_point_process(events[:, :1], design[:, :1], [2, 2], ["intercept"])
    assert_allclose(fit.beta, [np.log(2.5), -np.log(5)], atol=1e-6)
    assert fit.log_likelihood > null.log_likelihood


def test_roundoff_line_search_status_requires_independent_kkt(monkeypatch):
    from types import SimpleNamespace
    from toolbox import point_process
    # A failed line search at the exact finite optimum is not a failed MLE.
    result = SimpleNamespace(x=np.array([np.log(3 / 4)]), success=False,
                             message="ABNORMAL line search", nit=10)
    monkeypatch.setattr(point_process, "minimize", lambda *args, **kwargs: result)
    fitted = fit_point_process(np.ones((3, 1)), np.ones((4, 1)),
                               np.ones(4), ["intercept"])
    assert fitted.converged and fitted.gradient_max < 1e-12
    # A materially wrong solution is not accepted merely for being finite.
    result.x = np.array([-100.0])
    with pytest.raises(PointProcessFitError, match="KKT"):
        fit_point_process(np.ones((3, 1)), np.ones((4, 1)),
                          np.ones(4), ["intercept"])


def test_zero_events_and_rank_failure_are_explicit():
    fit = fit_point_process(np.ones((0, 1)), np.ones((3, 1)), [1, 1, 1], ["intercept"])
    assert fit.status == "zero_events" and not fit.converged
    assert fit.log_likelihood == 0 and np.isnan(fit.beta).all() and np.isnan(fit.aic)
    with pytest.raises(PointProcessFitError, match="rank deficient"):
        fit_point_process(np.ones((2, 2)), np.ones((4, 2)), np.ones(4), ["intercept", "other"])
    with pytest.raises(ValueError, match="positive"):
        fit_point_process(np.ones((2, 1)), np.ones((2, 1)), [1, 0], ["intercept"])


def test_same_old_bin_different_exact_times_change_history():
    nodes, weights = gauss_legendre_intervals([0, 2.01, 2.09, 4])
    values = []
    for response_age in [2.01, 2.09]:
        catalogue = [4, response_age]
        event_design = np.array([[1.0, exponential_history(response_age, catalogue).item()]])
        integral_design = np.column_stack([np.ones(len(nodes)), exponential_history(nodes, catalogue)])
        values.append(point_process_loglik([-1, -0.8], event_design, integral_design, weights))
    assert abs(values[0] - values[1]) > 1e-4


def test_poisson_thinning_counts_waiting_times_and_seed():
    rng = np.random.default_rng(918)
    counts, gaps = [], []
    for _ in range(1200):
        ages = simulate_segment_events(20, 0, [0, 7, 20], lambda a: np.log(0.8),
                                       [np.log(0.8)] * 2, 0, 1.5, rng)
        assert ages[0] == 20 and np.all(np.diff(ages) < 0) and ages[-1] > 0
        counts.append(len(ages) - 1)
        if len(ages) > 1:
            gaps.append(20 - ages[1])
    assert abs(np.mean(counts) - 16) < 0.35
    assert abs(np.var(counts) - 16) < 1.7
    assert abs(np.mean(gaps) - 1 / 0.8) < 0.10
    kwargs = dict(anchor_age=20, young_age=0, breakpoints=[0, 20],
                  log_background=lambda a: 0, log_upper_bounds=[0], history_beta=-0.8, tau=1.5)
    assert_allclose(simulate_segment_events(**kwargs, rng=np.random.default_rng(42)),
                    simulate_segment_events(**kwargs, rng=np.random.default_rng(42)))
    translated = dict(kwargs, anchor_age=270, young_age=250, breakpoints=[250, 270])
    assert_allclose(simulate_segment_events(**translated, rng=np.random.default_rng(42)) - 250,
                    simulate_segment_events(**kwargs, rng=np.random.default_rng(42)), atol=1e-12)


def test_thinning_matches_inhibitory_first_waiting_distribution():
    # The fixed anchor suppresses the first waiting-time hazard as well.
    rng = np.random.default_rng(314)
    before_one_kyr = []
    for _ in range(2000):
        ages = simulate_segment_events(2, 0, [0, 2], lambda a: 0, [0], -1.4, 1.5, rng)
        before_one_kyr.append(len(ages) > 1 and ages[1] > 1)
    integrated = quad(lambda t: np.exp(-1.4 * np.exp(-t / 1.5)), 0, 1)[0]
    probability = -np.expm1(-integrated)
    assert abs(np.mean(before_one_kyr) - probability) < 0.035


def test_accepted_event_immediately_changes_the_next_acceptance():
    class CandidateStream:
        waits = iter([0.1, 0.1, 100.0])

        def exponential(self):
            return next(self.waits)

        def uniform(self):
            return 0.1

    ages = simulate_segment_events(2, 0, [0, 2], lambda a: 0, [0], -1.4, 1.5,
                                   CandidateStream())
    # The first candidate is accepted. Its added history rejects the second;
    # omitting the accepted-event update would incorrectly accept both.
    assert_allclose(ages, [2, 1.9])


def test_background_interval_bounds_follow_bp_age_order():
    rng = np.random.default_rng(572)
    younger, older = [], []
    for _ in range(600):
        ages = simulate_segment_events(10, 0, [0, 5, 10],
                                       lambda a: np.log(2 if a < 5 else 0.2),
                                       np.log([2, 0.2]), 0, 1.5, rng)
        younger.append(np.sum(ages[1:] < 5))
        older.append(np.sum(ages[1:] >= 5))
    assert abs(np.mean(younger) - 10) < 0.4
    assert abs(np.mean(older) - 1) < 0.13


def test_thinning_clips_bounds_and_passes_bp_ages_to_background():
    class CandidateStream:
        def __init__(self):
            self.waits = iter([0.5, 100, 0.5, 100, 0.5, 100])

        def exponential(self):
            return next(self.waits)

        def uniform(self):
            return 0.1

    calls = []

    def log_background(age):
        calls.append(age)
        return np.log(2 if age < 4 else 0.5 if age < 6 else 1)

    # Envelopes extend past both ends, with different rates on each interval.
    ages = simulate_segment_events(8, 2, [-10, 0, 4, 6, 10, 20], log_background,
                                   np.log([9, 2, 0.5, 1, 9]), 0, 1.5, CandidateStream())
    assert_allclose(ages, [8, 7.5, 5, 3.75])
    assert_allclose(calls, ages[1:])


def test_envelope_errors_and_zero_rate_are_not_hidden():
    with pytest.raises(ValueError, match="nonpositive"):
        simulate_segment_events(2, 0, [0, 2], lambda a: 0, [0], 0.1, 1.5,
                                np.random.default_rng(1))
    with pytest.raises(ValueError, match="does not bound"):
        simulate_segment_events(100, 0, [0, 100], lambda a: 1, [0], 0, 1.5,
                                np.random.default_rng(1))
    ages = simulate_segment_events(2, 0, [0, 2], lambda a: -np.inf, [-np.inf],
                                   0, 1.5, np.random.default_rng(1))
    assert_allclose(ages, [2])
