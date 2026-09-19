"""Full-model simulation and approximate conditional effect intervals.

The full and reduced generators have separate public entry points. Their
continuous thinning implementation and exact event history are shared.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from toolbox import combined_likelihood

PHASE_TERMS = ("pre_phase_sin", "pre_phase_cos")
PHASE_INDICES = [1 + combined_likelihood.FULL_TERMS.index(term) for term in PHASE_TERMS]

class InvalidEffectSimulation(RuntimeError):
    """A numerical failure retained under its original replicate identity."""

def prepare_full_simulation(context, full_model):
    if full_model.terms != ("intercept", *context.full_terms):
        raise ValueError("The effect generator must be the fitted full model")
    return combined_likelihood.prepare_model_simulation(context, full_model)

def simulate_prepared_full_events(prepared, rng):
    return combined_likelihood.simulate_prepared_events(prepared, rng)

def simulate_full_model_events(context, full_model, rng):
    return simulate_prepared_full_events(prepare_full_simulation(context, full_model), rng)

def validate_effect_fit(fit):
    values = np.r_[fit.full.beta, fit.reduced.beta,
                   fit.full.log_likelihood, fit.reduced.log_likelihood]
    if not np.isfinite(values).all():
        raise InvalidEffectSimulation("Non-finite fitted coefficients or likelihood")
    if not fit.full.converged or not fit.reduced.converged:
        raise InvalidEffectSimulation("At least one fitted model did not converge")
    if fit.full.log_likelihood < fit.reduced.log_likelihood - 1e-7:
        raise InvalidEffectSimulation("Full/reduced likelihood nesting failed")


def phase_and_ratio(coefficients):
    """Map (..., 2) sine/cosine coefficients to circular phase and rate ratio."""
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.shape[-1:] != (2,) or not np.isfinite(coefficients).all():
        raise ValueError("Expected finite sine/cosine coefficient pairs")
    sine, cosine = coefficients[..., 0], coefficients[..., 1]
    phase = np.degrees(np.arctan2(sine, cosine)) % 360
    radius = np.hypot(sine, cosine)
    phase = np.where(radius == 0, np.nan, phase)
    return phase, np.exp(2 * radius)


def unwrap_phase(phase, center):
    return center + (np.asarray(phase) - center + 180) % 360 - 180


def bootstrap_joint_region(point_coefficients, bootstrap_coefficients, level=0.95):
    """Calibrate a joint ellipse using bootstrap errors about their generator.

    C is bootstrap covariance. q is the level quantile of
    (beta* - beta_hat)' C^-1 (beta* - beta_hat), retaining bootstrap bias.
    The confidence set is (beta_hat - beta)' C^-1 (beta_hat - beta) <= q.
    Plug-in calibration is approximate; projection gives simultaneous effect
    bounds, not a percentile interval or an exact finite-sample guarantee.
    """
    point = np.asarray(point_coefficients, dtype=float)
    samples = np.asarray(bootstrap_coefficients, dtype=float)
    if point.shape != (2,) or samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError("Joint region requires two phase coefficients")
    if len(samples) < 20 or not np.isfinite(samples).all() or not np.isfinite(point).all():
        raise ValueError("Joint region needs at least 20 finite bootstrap estimates")
    if not 0 < level < 1:
        raise ValueError("Confidence level must be between zero and one")
    covariance = np.cov(samples, rowvar=False)
    if np.linalg.eigvalsh(covariance).min() <= 0:
        raise ValueError("Bootstrap phase covariance must be positive definite")
    errors = samples - point
    quadratic = np.einsum("ij,ji->i", errors, np.linalg.solve(covariance, errors.T))
    critical = float(np.quantile(quadratic, level))
    origin_q = float(point @ np.linalg.solve(covariance, point))
    return {
        "center": point, "covariance": covariance, "critical_value": critical,
        "level": level, "origin_in_region": origin_q <= critical,
        "bootstrap_bias": errors.mean(axis=0), "bootstrap_error_quadratic": quadratic,
    }


def ellipse_boundary(region, angles):
    transform = np.sqrt(region["critical_value"]) * np.linalg.cholesky(region["covariance"])
    angles = np.atleast_1d(angles)
    unit = np.column_stack((np.cos(angles), np.sin(angles)))
    return region["center"] + unit @ transform.T


def _periodic_extreme(function, maximize=False):
    """Locate smooth periodic extrema and refine all sampled local candidates."""
    angles = np.linspace(0, 2 * np.pi, 721)[:-1]
    sign = -1 if maximize else 1
    values = sign * np.asarray(function(angles))
    candidates = np.flatnonzero((values <= np.roll(values, 1)) &
                               (values <= np.roll(values, -1)))
    step = 2 * np.pi / len(angles)
    solutions = [float(values.min())]
    for i in candidates:
        fit = minimize_scalar(lambda t: float(sign * function(np.array([t]))[0]),
                              bounds=(angles[i] - step, angles[i] + step), method="bounded",
                              options={"xatol": 1e-12})
        if not fit.success:
            raise RuntimeError("Could not project the coefficient confidence ellipse")
        solutions.append(float(fit.fun))
    return sign * min(solutions)


def project_joint_region(region):
    """Project a 2D region to radial rate ratio and a continuous circular arc."""
    radius = lambda t: np.linalg.norm(ellipse_boundary(region, t), axis=1)
    low_r = 0.0 if region["origin_in_region"] else _periodic_extreme(radius)
    high_r = _periodic_extreme(radius, maximize=True)
    point_phase, _ = phase_and_ratio(region["center"])
    if region["origin_in_region"]:
        phase_low, phase_high = 0.0, 360.0
    else:
        phase = lambda t: unwrap_phase(phase_and_ratio(ellipse_boundary(region, t))[0], point_phase)
        phase_low = _periodic_extreme(phase)
        phase_high = _periodic_extreme(phase, maximize=True)
    return {"phase_low_unwrapped_deg": phase_low, "phase_high_unwrapped_deg": phase_high,
            "phase_identified": not region["origin_in_region"],
            "ratio_low": float(np.exp(2 * low_r)), "ratio_high": float(np.exp(2 * high_r))}


def joint_region_curve_band(region, phase_deg):
    """Simultaneous curve envelope: exact linear-functional ellipse projection."""
    radians = np.deg2rad(phase_deg)
    direction = np.column_stack((np.sin(radians), np.cos(radians)))
    center = direction @ region["center"]
    half_width = np.sqrt(region["critical_value"] *
                         np.einsum("ij,jk,ik->i", direction, region["covariance"], direction))
    return np.exp(center - half_width), np.exp(center + half_width)
