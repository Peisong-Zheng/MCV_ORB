"""Continuous conditional event likelihood and inhibitory event simulation.

Rates are events per kyr. Public ages use kyr BP (AD 1950). Internally,
elapsed time u = anchor_age - age increases toward the present within each
segment. Integration weights are positive durations.
"""

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize


NONPOSITIVE_TERMS = ("same_type_exponential_history", "rectangular_history_count")


class PointProcessFitError(RuntimeError):
    """No trustworthy finite optimum was obtained for this event sequence."""


@dataclass
class FittedPointProcess:
    beta: np.ndarray
    terms: tuple
    log_likelihood: float
    aic: float
    converged: bool
    n_events: int
    gradient_max: float
    status: str


def negative_loglik(beta, event_sum, integral_design, weights):
    """Return the negative continuous log likelihood and its analytic gradient.

    ``event_sum`` is the sum of response-event design rows, computed once before
    fitting. Positive duration weights integrate the rate over all exposure.
    """
    # With p coefficients and m quadrature nodes: beta/event_sum are (p,),
    # integral_design is (m, p), and weights is (m,) in kyr.
    # lambda(u) = exp(x(u) @ beta); weight*lambda is an expected event count.
    # The observed event history is a fixed predictor here, just like forcing;
    # optimization changes coefficients, not event ages or history decay time.
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        rate_weights = weights * np.exp(integral_design @ beta)
        # -ell = integral lambda du - sum_i x(u_i) @ beta.
        value = rate_weights.sum() - event_sum @ beta
        # d(-ell)/d beta = sum_q w_q lambda_q x_q - sum_i x(u_i).
        gradient = integral_design.T @ rate_weights - event_sum
    # Reject overflowed trials without clipping the scientific intensity.
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        return np.inf, np.zeros(len(beta))
    return float(value), gradient


def fit_point_process(event_design, integral_design, weights, terms,
                      start_beta=None, nonpositive_terms=NONPOSITIVE_TERMS,
                      maxiter=1000, gradient_tol=1e-5):
    """Fit a log-linear conditional intensity on fixed integration nodes.

    Both matrices explicitly contain a column of ones, named ``intercept``.
    A zero-event catalogue has likelihood supremum zero and no finite MLE:
    its returned ``status`` is ``zero_events`` and its coefficients are NaN.
    Other failures raise ``PointProcessFitError`` rather than returning an
    apparently usable coefficient vector. ``gradient_tol`` checks the final
    projected gradient; the optimizer targets the tighter fixed value 2e-7.
    """
    event_design = np.asarray(event_design, dtype=float)
    integral_design = np.asarray(integral_design, dtype=float)
    weights = np.asarray(weights, dtype=float)
    terms = tuple(terms)
    n_parameters = len(terms)
    # Both designs use the identical ordered terms. Event rows contain only
    # responses; conditioning anchors affect history but are not likelihood events.
    if (weights.shape != (len(integral_design),) or not weights.size
            or not np.isfinite(weights).all() or np.any(weights <= 0)):
        raise ValueError("Each integral node needs one finite positive duration weight")
    if not np.isfinite(event_design).all() or not np.isfinite(integral_design).all():
        raise ValueError("Model predictors must be finite")
    intercept = terms.index("intercept")
    n_events = len(event_design)
    if n_events == 0:
        # With no responses, taking the intercept to -infinity makes the
        # integrated rate tend to zero; there is no finite intercept estimate.
        return FittedPointProcess(np.full(n_parameters, np.nan), terms, 0.0,
                                 np.nan, False, 0, np.nan, "zero_events")
    if np.linalg.matrix_rank(integral_design) < n_parameters:
        raise PointProcessFitError("Integral design is rank deficient")

    # History is nonnegative, so beta_H <= 0 permits inhibition or no history
    # effect. Intercept, climate and orbital coefficients remain unrestricted.
    upper_zero = np.array([term in nonpositive_terms for term in terms])
    bounds = [(None, 0.0) if constrained else (None, None) for constrained in upper_zero]
    beta0 = np.zeros(n_parameters)
    # The constant-rate MLE N/T supplies a scale-appropriate starting intercept.
    beta0[intercept] = np.log(n_events / weights.sum())
    if start_beta is not None:
        beta0 = np.asarray(start_beta, dtype=float).copy()
    # The event contribution is linear in beta, so its predictor sum is fixed
    # throughout optimization; only the integrated intensity must be reevaluated.
    event_sum = event_design.sum(axis=0)

    # jac=True consumes (value, gradient) together, sharing the rate calculation.
    # maxiter caps work; it does not force that many iterations. Even ftol=0
    # can stop at equal floating-point objective values, hence the check below.
    result = minimize(negative_loglik, beta0,
                      args=(event_sum, integral_design, weights),
                      jac=True, bounds=bounds, method="L-BFGS-B",
                      options={"maxiter": int(maxiter), "ftol": 0.0,
                               "gtol": 2e-7, "maxls": 40})
    beta = result.x
    value, gradient = negative_loglik(beta, event_sum, integral_design, weights)
    # At a history upper bound, a negative gradient points outside the domain.
    # Descent would increase beta_H above zero there. That component satisfies
    # the constrained first-order condition and contributes no residual.
    projected = gradient.copy()
    at_upper = upper_zero & (beta >= -1e-10)
    projected[at_upper] = np.maximum(projected[at_upper], 0.0)
    gradient_max = float(np.max(np.abs(projected)))
    # Function-value stopping can occur before the requested gradient tolerance.
    # Check the final solution once; do not refine or restart the optimization.
    # gtol is the optimizer's target; gradient_tol is the independent acceptance
    # limit on the largest feasible first-order residual, not a parameter error.
    if (not np.isfinite(beta).all() or not np.isfinite(value)
            or not np.isfinite(gradient).all() or np.any(beta[upper_zero] > 0)
            or gradient_max > gradient_tol):
        raise PointProcessFitError(
            f"Fit failed KKT/convergence checks: {result.message}; "
            f"projected gradient = {gradient_max:.3g}")
    # AIC = 2p - 2ell, with ell=-value and p the number of fitted coefficients.
    return FittedPointProcess(beta, terms, -value, 2 * n_parameters + 2 * value,
                             True, n_events, gradient_max,
                             "finite_mle")


def gauss_legendre_intervals(breakpoints, order=8):
    """Return interior nodes and positive kyr weights on increasing intervals.

    Standard formula for one interval [a, b]:
        integral_a^b f(t) dt ≈ (b-a)/2 * sum_j w_j * f(t_j)
        t_j = (a+b)/2 + (b-a)/2 * x_j

    Apply this formula to every adjacent pair of breakpoints. The caller
    evaluates f at the returned nodes and sums durations * f(nodes).
    """
    breaks = np.asarray(breakpoints, dtype=float)
    # points and weights are the standard x_j and w_j on [-1, 1].
    # Each has length order: this is the number of nodes PER interval,
    # unrelated to the number of intervals.
    points, weights = leggauss(order)

    # For all intervals: a = breaks[:-1], b = breaks[1:].
    # np.diff(breaks) gives b-a, so this is exactly (b-a)/2.
    half_width = np.diff(breaks) / 2

    # a + (b-a)/2 = (a+b)/2: this is the midpoint in the standard formula.
    middle = breaks[:-1] + half_width

    # Transform each standard node: t_j = (a+b)/2 + (b-a)/2 * x_j.
    # If there are m intervals, [:, None] changes (m,) into a column (m, 1).
    # Broadcasting with points of shape (order,) gives (m, order):
    # nodes[i, j] = middle[i] + half_width[i] * points[j].
    # Each row is one interval; each column is one of its Gaussian nodes.
    nodes = middle[:, None] + half_width[:, None] * points

    # The integral's prefactor (b-a)/2 is included in these weights:
    # durations[i, j] = (b_i-a_i)/2 * w_j, from dt = (b-a)/2 * dx.
    # The caller must not multiply by (b-a)/2 again when summing f(nodes).
    # These are duration weights, not distances between adjacent nodes.
    durations = half_width[:, None] * weights

    # Flatten row by row: all nodes of the first interval, then the next.
    # Flatten weights in the same order to preserve their pairing with nodes.
    return nodes.ravel(), durations.ravel()


def exponential_history(query_ages, event_ages, tau=1.5, anchor_age=None,
                        initial_history=0.0):
    """Evaluate earlier-event history on a 1D array of BP ages, in query order.

    ``event_ages`` includes the observed conditioning event. ``initial_history``
    is the unknown pre-anchor weighted history, fixed to zero in the main model.
    At an event's exact age, that event has not yet entered its own history.
    """
    query = np.asarray(query_ages, dtype=float)
    events = np.asarray(event_ages, dtype=float)
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("History decay time must be positive")
    if not np.isfinite(initial_history) or initial_history < 0:
        raise ValueError("Pre-anchor history must be finite and nonnegative")
    events = np.sort(events)[::-1]  # Chronological order: oldest to youngest.
    if anchor_age is None and initial_history:
        anchor_age = events[0]
    history = np.zeros(len(query))
    # H(u) = sum_{u_j < u} exp(-(u-u_j)/tau). In BP coordinates,
    # u-u_j = event_age-query_age; each earlier event contributes one decay.
    for event_age in events:
        # Smaller BP ages occur later. Compare ages directly so that an event
        # and its floating-point neighbours retain their strict ordering.
        later = query < event_age
        history[later] = history[later] + np.exp((query[later] - event_age) / tau)
    if initial_history:
        # Pre-anchor history decays from the anchor without adding a new event.
        remaining_initial_history = initial_history * np.exp((query - anchor_age) / tau)
        history = history + remaining_initial_history
    return history


def simulate_segment_events(anchor_age, young_age, breakpoints, log_background,
                            log_upper_bounds, history_beta, tau, rng,
                            initial_history=0.0, max_candidates=1_000_000):
    """Thin a background envelope, returning descending ages including anchor.

    ``breakpoints`` increase in BP age. Each bound is for the corresponding
    ascending interval and must bound the *background*, before inhibition.
    Bounds may cover a wider interval than the requested segment. Candidate
    times advance in elapsed kyr; callback inputs and returned ages remain BP.
    Accepted events update history immediately.
    """
    breaks = np.asarray(breakpoints, dtype=float)
    upper = np.asarray(log_upper_bounds, dtype=float)
    if not np.isfinite(history_beta) or history_beta > 0:
        raise ValueError("This thinning envelope requires a nonpositive history coefficient")

    # Clip the BP support before conversion. Reverse the paired interval bounds
    # together with their edges so that simulation advances from u=0 to T.
    elapsed_edges = anchor_age - np.clip(breaks, young_age, anchor_age)[::-1]
    forward_upper = upper[::-1]
    ages = [float(anchor_age)]
    current_elapsed = 0.0
    history = 1.0 + initial_history  # The fixed conditioning event has occurred.
    candidates = 0
    # On each interval, proposals have constant rate M=exp(log_upper).
    # Nonpositive beta_H and nonnegative history ensure lambda <= background <= M.
    for start, end, log_upper in zip(elapsed_edges[:-1], elapsed_edges[1:], forward_upper):
        if end <= start:
            continue
        while current_elapsed < end:
            remaining_kyr = end - current_elapsed
            # Log waiting times remain safe for extremely small envelopes.
            # If E~Exp(1), the proposal waiting time is E/M, in kyr.
            unit_wait = rng.exponential()
            log_wait = np.log(unit_wait) - log_upper if unit_wait > 0 else -np.inf
            if log_wait >= np.log(remaining_kyr):
                # No proposal before this boundary: decay history across the
                # remaining exposure, then use the next interval's envelope.
                decay_factor = np.exp(-remaining_kyr / tau)
                history = history * decay_factor
                current_elapsed = end
                break
            wait = np.exp(log_wait)
            candidate_elapsed = current_elapsed + wait
            candidate_age = anchor_age - candidate_elapsed
            current_age = anchor_age - current_elapsed
            if candidate_elapsed <= current_elapsed or candidate_age >= current_age:
                raise RuntimeError("Simulation waiting time is below floating-point age resolution")
            if candidate_elapsed >= end:
                decay_factor = np.exp(-remaining_kyr / tau)
                history = history * decay_factor
                current_elapsed = end
                break
            decay_factor = np.exp(-wait / tau)
            history = history * decay_factor
            current_elapsed = candidate_elapsed
            candidates = candidates + 1
            if candidates > max_candidates:
                raise RuntimeError("Thinning candidate limit reached; inspect the background envelope")
            log_bg = float(log_background(candidate_age))
            if np.isnan(log_bg) or np.isposinf(log_bg):
                raise ValueError("Background log intensity is invalid at a candidate event")
            if log_bg > log_upper + 1e-12:
                raise ValueError("Simulation envelope does not bound the background intensity")
            # Thinning accepts a proposal with probability lambda/M; using
            # log probabilities avoids exponentiating very small intensities.
            log_accept = log_bg + history_beta * history - log_upper
            uniform = rng.uniform()
            log_uniform = np.log(uniform) if uniform > 0 else -np.inf
            if log_uniform < log_accept:
                ages.append(candidate_age)
                # Only an accepted event adds one; rejected proposals merely
                # advance time and decay the existing history.
                history = history + 1
    return np.asarray(ages)
