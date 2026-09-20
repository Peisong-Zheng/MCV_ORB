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
    identifiable: bool


def gauss_legendre_intervals(breakpoints, order=8):
    """Return interior nodes and positive kyr weights on increasing intervals."""
    breaks = np.asarray(breakpoints, dtype=float)
    if breaks.ndim != 1 or len(breaks) < 2:
        raise ValueError("Integration needs at least two interval boundaries")
    if not np.isfinite(breaks).all() or np.any(np.diff(breaks) <= 0):
        raise ValueError("Integration boundaries must be finite and increasing")
    if not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError("Quadrature order must be a positive integer")
    points, weights = leggauss(order)
    half_width = np.diff(breaks) / 2
    middle = breaks[:-1] + half_width
    nodes = middle[:, None] + half_width[:, None] * points
    durations = half_width[:, None] * weights
    return nodes.ravel(), durations.ravel()


def exponential_history(query_ages, event_ages, tau=1.5, anchor_age=None,
                        initial_history=0.0):
    """Evaluate earlier-event history from BP ages, preserving query order.

    ``event_ages`` includes the observed conditioning event. ``initial_history``
    is the unknown pre-anchor weighted history, fixed to zero in the main model.
    At an event's exact age, that event has not yet entered its own history.
    """
    query = np.asarray(query_ages, dtype=float)
    events = np.asarray(event_ages, dtype=float)
    if events.ndim != 1 or not np.isfinite(events).all() or not np.isfinite(query).all():
        raise ValueError("Event and query ages must be finite")
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("History decay time must be positive")
    if not np.isfinite(initial_history) or initial_history < 0:
        raise ValueError("Pre-anchor history must be finite and nonnegative")
    events = np.sort(events)[::-1]  # Chronological order: oldest to youngest.
    if np.any(np.diff(events) == 0):
        raise ValueError("Coincident events require review of source-age precision")
    if anchor_age is None and len(events):
        anchor_age = events[0]
    if anchor_age is not None:
        if not np.isfinite(anchor_age):
            raise ValueError("Anchor age must be finite")
        if (len(events) and events[0] > anchor_age) or np.any(query > anchor_age):
            raise ValueError("Events and queries cannot precede the conditioning anchor")
    elif initial_history:
        raise ValueError("Nonzero initial history requires an anchor age")

    history = np.zeros(query.size)
    if len(events):
        elapsed_kyr = anchor_age - query.ravel()
        event_elapsed_kyr = anchor_age - events
        # History immediately before each event, advancing in elapsed time.
        at_event = np.zeros(len(events))
        for i in range(1, len(events)):
            gap_kyr = event_elapsed_kyr[i] - event_elapsed_kyr[i - 1]
            at_event[i] = (1 + at_event[i - 1]) * np.exp(-gap_kyr / tau)
        # -age also runs forward, without origin subtraction. Use it for the
        # strict u_j < u comparison: subtracting a distant anchor can otherwise
        # round an event and its nextafter() neighbor to the same elapsed time.
        previous_count = np.searchsorted(-events, -query.ravel(), side="left")
        present = previous_count > 0
        j = previous_count[present] - 1
        since_previous_kyr = elapsed_kyr[present] - event_elapsed_kyr[j]
        history[present] = (1 + at_event[j]) * np.exp(-since_previous_kyr / tau)
    if initial_history:
        elapsed_kyr = anchor_age - query.ravel()
        history += initial_history * np.exp(-elapsed_kyr / tau)
    return history.reshape(query.shape)


def point_process_loglik(beta, event_design, integral_design, weights):
    """Event log intensities minus integrated intensity, without eta clipping."""
    beta = np.asarray(beta, dtype=float)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        integrated = np.asarray(weights) * np.exp(np.asarray(integral_design) @ beta)
    return float(np.asarray(event_design).sum(axis=0) @ beta - integrated.sum())


def negative_loglik_gradient(beta, event_design, integral_design, weights):
    """Gradient of the negative continuous log likelihood."""
    integral_design = np.asarray(integral_design)
    rate_weights = np.asarray(weights) * np.exp(integral_design @ beta)
    return integral_design.T @ rate_weights - np.asarray(event_design).sum(axis=0)


def negative_loglik_hessian(beta, integral_design, weights):
    """Observed information with fixed positive quadrature weights."""
    integral_design = np.asarray(integral_design)
    rate_weights = np.asarray(weights) * np.exp(integral_design @ beta)
    return integral_design.T @ (rate_weights[:, None] * integral_design)


def fit_point_process(event_design, integral_design, weights, terms,
                      start_beta=None, nonpositive_terms=NONPOSITIVE_TERMS,
                      maxiter=1000, gradient_tol=1e-6):
    """Fit a log-linear conditional intensity on fixed integration nodes.

    Both matrices explicitly contain an intercept column, named ``intercept``.
    A zero-event catalogue has likelihood supremum zero and no finite MLE:
    its returned ``status`` is ``zero_events`` and its coefficients are NaN.
    Other failures raise ``PointProcessFitError`` rather than returning an
    apparently usable coefficient vector.
    """
    event_design = np.asarray(event_design, dtype=float)
    integral_design = np.asarray(integral_design, dtype=float)
    weights = np.asarray(weights, dtype=float)
    terms = tuple(terms)
    n_parameters = len(terms)
    if len(set(terms)) != n_parameters or terms.count("intercept") != 1:
        raise ValueError("Terms need unique names and one explicit intercept")
    if event_design.ndim != 2 or integral_design.ndim != 2:
        raise ValueError("Event and integral designs must be matrices")
    if event_design.shape[1] != n_parameters or integral_design.shape[1] != n_parameters:
        raise ValueError("Design columns must match the explicit model terms")
    if weights.ndim != 1 or len(weights) != len(integral_design) or not len(weights):
        raise ValueError("Each integral node needs one duration weight")
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Integration weights must be finite and positive")
    if not np.isfinite(event_design).all() or not np.isfinite(integral_design).all():
        raise ValueError("Model predictors must be finite")
    intercept = terms.index("intercept")
    if not np.all(event_design[:, intercept] == 1) or not np.all(integral_design[:, intercept] == 1):
        raise ValueError("The intercept column must equal one")
    n_events = len(event_design)
    if n_events == 0:
        return FittedPointProcess(np.full(n_parameters, np.nan), terms, 0.0,
                                 np.nan, False, 0, np.nan, "zero_events", False)
    if np.linalg.matrix_rank(integral_design) < n_parameters:
        raise PointProcessFitError("Integral design is rank deficient")

    upper_zero = np.array([term in nonpositive_terms for term in terms])
    bounds = [(None, 0.0) if constrained else (None, None) for constrained in upper_zero]
    beta0 = np.zeros(n_parameters)
    beta0[intercept] = np.log(n_events / weights.sum())
    if start_beta is not None:
        beta0 = np.asarray(start_beta, dtype=float).copy()
        if beta0.shape != (n_parameters,) or not np.isfinite(beta0).all():
            raise ValueError("Initial coefficients must be a finite vector matching terms")
        if np.any(beta0[upper_zero] > 0):
            raise ValueError("Initial history coefficients violate the nonpositive domain")
    event_sum = event_design.sum(axis=0)

    def objective(beta):
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            rate_weights = weights * np.exp(integral_design @ beta)
            value = rate_weights.sum() - event_sum @ beta
            gradient = integral_design.T @ rate_weights - event_sum
        # An overflowed trial is rejected by line search, not clipped into a
        # different scientific intensity model.
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            return np.inf, np.zeros(n_parameters)
        return float(value), gradient

    result = minimize(objective, beta0, jac=True, bounds=bounds, method="L-BFGS-B",
                      options={"maxiter": int(maxiter), "ftol": 1e-14,
                               "gtol": gradient_tol / 5, "maxls": 40})
    beta = result.x
    value, gradient = objective(beta)
    def projected_gradient_max(beta, gradient):
        projected = gradient.copy()
        at_upper = upper_zero & (beta >= -1e-10)
        projected[at_upper] = np.maximum(projected[at_upper], 0.0)
        return float(np.max(np.abs(projected)))

    gradient_max = projected_gradient_max(beta, gradient)
    # Relative likelihood stopping can precede gradient convergence. The small
    # observed-information solve refines the same data and objective.
    if np.isfinite(value):
        for _ in range(8):
            if gradient_max <= gradient_tol:
                break
            active = upper_zero & (beta >= -1e-10) & (gradient <= 0)
            free = ~active
            information = negative_loglik_hessian(beta, integral_design, weights)
            direction = np.zeros(n_parameters)
            try:
                direction[free] = -np.linalg.solve(information[np.ix_(free, free)], gradient[free])
            except np.linalg.LinAlgError:
                break
            for exponent in range(20):
                trial = beta + direction * 0.5**exponent
                trial[upper_zero] = np.minimum(trial[upper_zero], 0)
                trial_value, trial_gradient = objective(trial)
                armijo = trial_value <= value + 1e-4 * gradient @ (trial - beta)
                # Near an optimum the predicted likelihood change can be
                # smaller than floating-point summation error. Accept such a
                # step only when it substantially improves the same strict
                # KKT residual; this does not relax the convergence tolerance.
                roundoff = 8 * np.finfo(float).eps * max(1.0, abs(value))
                improves_kkt = (np.isfinite(trial_value)
                    and abs(trial_value - value) <= roundoff
                    and projected_gradient_max(trial, trial_gradient) < 0.5 * gradient_max)
                if armijo or improves_kkt:
                    beta, value, gradient = trial, trial_value, trial_gradient
                    break
            else:
                break
            gradient_max = projected_gradient_max(beta, gradient)
    # A line-search status alone is not decisive for this concave likelihood:
    # independently verify its finite optimum through KKT and information.
    if not np.isfinite(value) or gradient_max > gradient_tol:
        raise PointProcessFitError(
            f"Fit failed KKT/convergence checks: {result.message}; "
            f"projected gradient = {gradient_max:.3g}")
    information = negative_loglik_hessian(beta, integral_design, weights)
    free = ~(upper_zero & (beta >= -1e-10))
    eigenvalues = np.linalg.eigvalsh(information[np.ix_(free, free)])
    if eigenvalues[0] <= 1e-10 * max(1.0, eigenvalues[-1]):
        raise PointProcessFitError("No identifiable finite optimum; information is singular")
    return FittedPointProcess(beta, terms, -value, 2 * n_parameters + 2 * value,
                             True, n_events, gradient_max,
                             "finite_mle", True)


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
    if not np.isfinite([anchor_age, young_age]).all() or anchor_age <= young_age:
        raise ValueError("Conditioning anchor must be older than the response endpoint")
    if breaks.ndim != 1 or len(breaks) < 2 or not np.isfinite(breaks).all() or np.any(np.diff(breaks) <= 0):
        raise ValueError("Simulation breakpoints must be finite and increasing")
    if upper.shape != (len(breaks) - 1,) or np.isnan(upper).any() or np.isposinf(upper).any():
        raise ValueError("One finite or negative-infinite log bound is required per interval")
    if breaks[0] > young_age or breaks[-1] < anchor_age:
        raise ValueError("Background envelopes must cover the complete response support")
    if not np.isfinite(history_beta) or history_beta > 0:
        raise ValueError("This thinning envelope requires a nonpositive history coefficient")
    if not np.isfinite(tau) or tau <= 0 or not np.isfinite(initial_history) or initial_history < 0:
        raise ValueError("Positive decay time and nonnegative initial history are required")

    # Clip the BP support before conversion. Reverse the paired interval bounds
    # together with their edges so that simulation advances from u=0 to T.
    elapsed_edges = anchor_age - np.clip(breaks, young_age, anchor_age)[::-1]
    forward_upper = upper[::-1]
    ages = [float(anchor_age)]
    current_elapsed = 0.0
    history = 1.0 + initial_history  # The fixed conditioning event has occurred.
    candidates = 0
    for start, end, log_upper in zip(elapsed_edges[:-1], elapsed_edges[1:], forward_upper):
        if end <= start:
            continue
        while current_elapsed < end:
            remaining_kyr = end - current_elapsed
            # Log waiting times remain safe for extremely small envelopes.
            unit_wait = rng.exponential()
            log_wait = np.log(unit_wait) - log_upper if unit_wait > 0 else -np.inf
            if log_wait >= np.log(remaining_kyr):
                history *= np.exp(-remaining_kyr / tau)
                current_elapsed = end
                break
            wait = np.exp(log_wait)
            candidate_elapsed = current_elapsed + wait
            candidate_age = anchor_age - candidate_elapsed
            current_age = anchor_age - current_elapsed
            if candidate_elapsed <= current_elapsed or candidate_age >= current_age:
                raise RuntimeError("Simulation waiting time is below floating-point age resolution")
            if candidate_elapsed >= end:
                history *= np.exp(-remaining_kyr / tau)
                current_elapsed = end
                break
            history *= np.exp(-wait / tau)
            current_elapsed = candidate_elapsed
            candidates += 1
            if candidates > max_candidates:
                raise RuntimeError("Thinning candidate limit reached; inspect the background envelope")
            log_bg = float(log_background(candidate_age))
            if np.isnan(log_bg) or np.isposinf(log_bg):
                raise ValueError("Background log intensity is invalid at a candidate event")
            if log_bg > log_upper + 1e-12:
                raise ValueError("Simulation envelope does not bound the background intensity")
            log_accept = log_bg + history_beta * history - log_upper
            uniform = rng.uniform()
            log_uniform = np.log(uniform) if uniform > 0 else -np.inf
            if log_uniform < log_accept:
                ages.append(candidate_age)
                history += 1
    return np.asarray(ages)
