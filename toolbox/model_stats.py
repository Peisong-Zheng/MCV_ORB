"""Likelihood comparisons and circular statistics for event models."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import chi2, beta as beta_distribution


def unwrap_phase(values, center):
    """Express circular phases within 180 degrees of the reference phase."""
    # Keep phases crossing 0/360 degrees adjacent when taking ordinary quantiles.
    return center + (np.asarray(values) - center + 180) % 360 - 180


def nested_likelihood_metrics(*, loglik_full, loglik_reduced, df, n_events,
                              aic_full, aic_reduced):
    """Compare nested models fitted to the same response events and support."""
    # The gain is in natural-log units; divide by ln(2) and event count for G.
    gain = float(loglik_full - loglik_reduced)
    if gain < -1e-7:
        raise RuntimeError("Full-model likelihood is below the nested reference")
    lr = 2 * gain
    # Chi-square calibration is nominal; negative delta AIC favors the full model.
    return dict(
        df=int(df), n_events=int(n_events), loglik_reduced=float(loglik_reduced),
        loglik_full=float(loglik_full), ll_gain_nats=gain,
        gain_bits_per_event=float(gain / np.log(2) / n_events) if n_events > 0 else np.nan,
        LR_statistic=lr, LR_p_value=float(chi2.sf(max(lr, 0.), df)),
        delta_AIC_full_minus_reduced=float(aic_full - aic_reduced),
    )


def rayleigh_p_value_from_z(z: float, n: int) -> float:
    """Return the finite-sample Rayleigh upper-tail p value."""

    if n <= 0 or not np.isfinite(z):
        return np.nan

    # Finite-n expansion through n^-2; Fisher (1993), Circular Data.
    p = np.exp(-z) * (
        1.0
        + (2.0 * z - z**2) / (4.0 * n)
        - (24.0 * z - 132.0 * z**2 + 76.0 * z**3 - 9.0 * z**4) / (288.0 * n**2)
    )
    return float(np.clip(p, 0.0, 1.0))


def rayleigh_test(phases_rad: np.ndarray) -> dict[str, float]:
    """Test whether circular phases depart from a uniform distribution."""

    theta = np.asarray(phases_rad, dtype=float)
    theta = theta[np.isfinite(theta)]
    n = len(theta)
    if n == 0:
        return {
            "n_phase_events_used": 0,
            "mean_phase_rad": np.nan,
            "mean_phase_deg": np.nan,
            "mean_resultant_length": np.nan,
            "rayleigh_R": np.nan,
            "rayleigh_z": np.nan,
            "rayleigh_p": np.nan,
        }

    # Average unit vectors so the circular mean respects the phase seam.
    cosine_sum = float(np.cos(theta).sum())
    sine_sum = float(np.sin(theta).sum())
    resultant = float(np.hypot(cosine_sum, sine_sum))
    mean_phase = float(np.mod(np.arctan2(sine_sum, cosine_sum), 2 * np.pi))
    mean_resultant_length = resultant / n
    z = n * mean_resultant_length**2
    return {
        "n_phase_events_used": n,
        "mean_phase_rad": mean_phase,
        "mean_phase_deg": float(np.degrees(mean_phase)),
        "mean_resultant_length": mean_resultant_length,
        "rayleigh_R": resultant,
        "rayleigh_z": z,
        "rayleigh_p": rayleigh_p_value_from_z(z, n),
    }


def rayleigh_rbar_threshold(n: int, alpha: float = 0.05) -> float:
    """Return the mean-resultant-length threshold for ``p <= alpha``."""

    if n <= 0:
        return np.nan
    low, high = 0.0, 1.0
    if rayleigh_p_value_from_z(n, n) > alpha:
        return np.nan
    # Invert the same finite-sample p value used by the test, rather than an asymptotic cutoff.
    for _ in range(80):
        middle = (low + high) / 2
        if rayleigh_p_value_from_z(n * middle**2, n) <= alpha:
            high = middle
        else:
            low = middle
    return high


def phase_sector_observed_expected(events, integration, models, n_sectors=12):
    """Observed vs fitted event counts in equal precession-phase sectors.

    For each phase sector, the fitted count under a model is the response-time
    integrated intensity whose precession phase falls in that sector; the
    observed count is the number of response events in the sector. A correctly
    specified model gives a ratio near one in every sector. A phase-organized
    signal appears as a sinusoidal departure for the no-phase (reduced) model
    and flattens once the full model adds the phase terms.
    """
    edges = np.linspace(0.0, 360.0, n_sectors + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    observed, _ = np.histogram(events["pre_phase_deg"].to_numpy(float), bins=edges)
    rows = []
    for model_id, model in models.items():
        rate = np.exp(integration.loc[:, model.terms].to_numpy(float) @ model.beta)
        # Quadrature exposure weights turn rate into expected counts per sector.
        mass = rate * integration["weight"].to_numpy(float)
        fitted, _ = np.histogram(integration["pre_phase_deg"].to_numpy(float),
                                 bins=edges, weights=mass)
        ratio = np.where(fitted > 0.0, observed / fitted, np.nan)
        for center, obs, fit_count, rat in zip(centers, observed, fitted, ratio):
            rows.append(dict(model_id=model_id, phase_sector_center_deg=center,
                             observed_events=int(obs), fitted_events=fit_count,
                             observed_over_fitted=rat))
    return pd.DataFrame(rows)


# Phase coefficients and joint uncertainty regions.

PHASE_TERMS = ("pre_phase_sin", "pre_phase_cos")
def phase_coefficients(model):
    """Read sine/cosine by name; Barker has no MIS6 segment coefficient."""
    return np.asarray([model.beta[model.terms.index(term)] for term in PHASE_TERMS])


def phase_and_ratio(coefficients):
    """Map (..., 2) sine/cosine coefficients to circular phase and rate ratio."""
    coefficients = np.asarray(coefficients, dtype=float)
    sine, cosine = coefficients[..., 0], coefficients[..., 1]
    # b_s sin(phi) + b_c cos(phi) = r cos(phi - phi_peak).
    phase = np.degrees(np.arctan2(sine, cosine)) % 360
    radius = np.hypot(sine, cosine)
    # Zero amplitude has no preferred phase; peak/trough rates differ by exp(2r).
    phase = np.where(radius == 0, np.nan, phase)
    return phase, np.exp(2 * radius)


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
    if len(samples) < 3 or not np.isfinite(samples).all() or not np.isfinite(point).all():
        raise ValueError("Joint region needs at least three finite coefficient pairs")
    # Covariance measures ensemble spread; errors remain centered on the nominal fit.
    covariance = np.cov(samples, rowvar=False)
    errors = samples - point
    quadratic = np.einsum("ij,ji->i", errors, np.linalg.solve(covariance, errors.T))
    # Empirical calibration retains bias; it does not substitute a chi-square cutoff.
    critical = float(np.quantile(quadratic, level))
    # Including the origin permits zero phase amplitude and leaves direction unidentified.
    origin_q = float(point @ np.linalg.solve(covariance, point))
    return {
        "center": point, "covariance": covariance, "critical_value": critical,
        "level": level, "origin_in_region": origin_q <= critical,
        "bootstrap_bias": errors.mean(axis=0), "bootstrap_error_quadratic": quadratic,
    }


def ellipse_boundary(region, angles):
    # Map the unit circle to the joint coefficient ellipse, centered on the point fit.
    transform = np.sqrt(region["critical_value"]) * np.linalg.cholesky(region["covariance"])
    angles = np.atleast_1d(angles)
    unit = np.column_stack((np.cos(angles), np.sin(angles)))
    return region["center"] + unit @ transform.T


def _periodic_extreme(function, maximize=False):
    """Locate smooth periodic extrema and refine all sampled local candidates."""
    angles = np.linspace(0, 2 * np.pi, 721)[:-1]
    sign = -1 if maximize else 1
    values = sign * np.asarray(function(angles))
    # Neighbor comparisons wrap around the seam; refine every candidate extremum.
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
    # Radius controls modulation strength; an enclosed origin sets the minimum ratio to one.
    radius = lambda t: np.linalg.norm(ellipse_boundary(region, t), axis=1)
    low_r = 0.0 if region["origin_in_region"] else _periodic_extreme(radius)
    high_r = _periodic_extreme(radius, maximize=True)
    point_phase, _ = phase_and_ratio(region["center"])
    if region["origin_in_region"]:
        phase_low, phase_high = 0.0, 360.0
    else:
        # Unwrap about the nominal direction before finding arc endpoints across 0 degrees.
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
    # The ellipse support function bounds log-rate modulation at every phase jointly.
    half_width = np.sqrt(region["critical_value"] *
                         np.einsum("ij,jk,ik->i", direction, region["covariance"], direction))
    return np.exp(center - half_width), np.exp(center + half_width)

def empirical_p_value(
    bootstrap_statistics: np.ndarray, observed_statistic: float
) -> tuple[float, int]:
    """Return the plus-one bootstrap p value and exceedance count."""

    values = np.asarray(bootstrap_statistics, dtype=float)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("bootstrap_statistics must be a finite, non-empty vector")
    if not np.isfinite(observed_statistic):
        raise ValueError("observed_statistic must be finite")
    # Include ties in the upper tail; the plus-one correction prevents a zero simulated p.
    exceedances = int(np.count_nonzero(values >= observed_statistic))
    return float((exceedances + 1) / (len(values) + 1)), exceedances

def clopper_pearson_interval(
    successes: int,
    trials: int,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Exact binomial interval for the null exceedance probability."""

    # Invert binomial tails for Monte Carlo uncertainty, not for model-parameter uncertainty.
    alpha = 1.0 - confidence
    low = (
        0.0
        if successes == 0
        else float(
            beta_distribution.ppf(alpha / 2.0, successes, trials - successes + 1)
        )
    )
    high = (
        1.0
        if successes == trials
        else float(
            beta_distribution.ppf(
                1.0 - alpha / 2.0, successes + 1, trials - successes
            )
        )
    )
    return low, high


def phase_bootstrap_summary(observed_summary, replicates, failed_reasons):
    """No calibrated p is published while a requested replicate is unresolved."""
    row = dict(observed_summary)
    failed = int((~replicates.fit_valid).sum())
    row.update(bootstrap_null="continuous fitted reduced model; exact anchors fixed",
               n_bootstrap=len(replicates), n_valid_replicates=int(replicates.fit_valid.sum()),
               n_failed_replicates=failed, n_zero_event_replicates=int(replicates.status.eq("zero_events").sum()),
               n_same_data_refinements=int(replicates.solver_attempts.gt(1).sum()),
               n_resampled_catalogues=0,
               failed_reasons="; ".join(f"{reason} ({n})" for reason,n in failed_reasons.items()))
    p = low = high = mc_se = np.nan
    exceedances = np.nan
    # Dropping failed refits would condition the null distribution on numerical success.
    if not failed:
        p, exceedances = empirical_p_value(replicates.LR_statistic.to_numpy(), row["LR_statistic"])
        low, high = clopper_pearson_interval(exceedances, len(replicates))
        mc_se = np.sqrt(p*(1-p)/(len(replicates)+1))
    row.update(empirical_p_plus_one=p, n_bootstrap_exceeding_or_equal_observed=exceedances,
               empirical_p_mc_se=mc_se, empirical_p_ci95_low=low, empirical_p_ci95_high=high,
               empirical_p_ci_method="Clopper-Pearson interval for the null exceedance probability",
               bootstrap_response_event_count_mean=replicates.n_events_response.mean(),
               bootstrap_response_event_count_q025=replicates.n_events_response.quantile(.025),
               bootstrap_response_event_count_q975=replicates.n_events_response.quantile(.975))
    for name, value in [("mean",replicates.LR_statistic.mean()), ("median",replicates.LR_statistic.median()),
                        ("q95",replicates.LR_statistic.quantile(.95)), ("q99",replicates.LR_statistic.quantile(.99))]:
        row["bootstrap_LR_"+name] = value if not failed else np.nan
    return pd.DataFrame([row])



SCENARIOS = {"A_chronology": "chronology", "B_sampling": "sampling", "C_joint": "combined"}
INTERVAL_TYPES = {"A_chronology": "95_joint_working_region_projection", "B_sampling": "approximate_95_joint_confidence_region_projection", "C_joint": "95_joint_working_region_projection"}
PHASE_COLUMNS = [f"beta__{term}" for term in PHASE_TERMS]

def summarize_effects(full_model, age_results, replicates):
    """Use matched ellipse projections, retaining each ensemble's interpretation."""
    if not replicates.fit_valid.all():
        raise RuntimeError("Cannot summarize effect intervals with failed refits")
    if 'effect_identified' in replicates and not replicates.effect_identified.all():
        raise RuntimeError("Unidentified effect draws retained; confidence region is not reported")
    point = phase_coefficients(full_model)
    point_phase, point_ratio = phase_and_ratio(point)
    b = replicates.loc[replicates.scenario.eq("B_sampling")]
    c = replicates.loc[replicates.scenario.eq("C_joint")]
    # Equal inner sample counts give each outer chronology equal weight in the joint ensemble.
    if c.groupby("outer_id").size().nunique() != 1:
        raise ValueError("Joint groups must have equal weight and complete inner samples")
    # A varies chronology, B simulates under the nominal fit, and C combines both sources.
    samples = {
        "A_chronology": age_results.loc[age_results.fit_valid,
                                        ["beta_pre_phase_sin", "beta_pre_phase_cos"]].to_numpy(float),
        "B_sampling": b[PHASE_COLUMNS].to_numpy(float),
        "C_joint": c[PHASE_COLUMNS].to_numpy(float),
    }
    rows, regions = [], {}
    for scenario, coefficients in samples.items():
        # Reuse the ellipse geometry for all ensembles; only sampling is a bootstrap CI.
        region = bootstrap_joint_region(point, coefficients)
        region["n"] = len(coefficients)
        region["interval_type"] = INTERVAL_TYPES[scenario]
        regions[scenario] = region
        bounds = project_joint_region(region)
        for quantity, estimate, low, high in (
            ("preferred_phase_deg", point_phase, bounds["phase_low_unwrapped_deg"], bounds["phase_high_unwrapped_deg"]),
            ("max_min_rate_ratio", point_ratio, bounds["ratio_low"], bounds["ratio_high"]),
        ):
            rows.append(dict(scenario=scenario, quantity=quantity, point_estimate=float(estimate),
                             center=float(estimate), low=low, high=high, n=len(coefficients),
                             interval_type=INTERVAL_TYPES[scenario],
                             phase_identified=bounds["phase_identified"]))
    summary = pd.DataFrame(rows).sort_values(["quantity", "scenario"]).reset_index(drop=True)
    return summary, regions


def build_curve_table(full_model, regions):
    """Project each complete coefficient region to a phase-curve envelope."""
    phases = np.arange(0.0, 361.0, 1.0)
    radians = np.deg2rad(phases)
    direction = np.column_stack((np.sin(radians), np.cos(radians)))
    point = phase_coefficients(full_model)
    # Hold the other predictors fixed: this is the multiplicative phase component alone.
    table = {"phase_deg": phases, "point_multiplier": np.exp(direction @ point)}
    for scenario, name in SCENARIOS.items():
        table[f"{name}_low"], table[f"{name}_high"] = joint_region_curve_band(regions[scenario], phases)
    return pd.DataFrame(table)




def fit_summary(reduced, full, event_features, windows, *, n_source_events,
                catalogue_id, tau=1.5, initial_history=0.0):
    """Likelihood and phase statistics for a fitted nested pair."""
    from toolbox.project_config import MODEL_VERSION
    metrics = nested_likelihood_metrics(
        loglik_full=full.log_likelihood,
        loglik_reduced=reduced.log_likelihood,
        df=len(full.beta) - len(reduced.beta),
        n_events=len(event_features),
        aic_full=full.aic,
        aic_reduced=reduced.aic,
    )
    betas = dict(zip(full.terms, full.beta))
    beta_sin = betas.get("pre_phase_sin", np.nan)
    beta_cos = betas.get("pre_phase_cos", np.nan)
    amplitude = float(np.hypot(beta_sin, beta_cos))
    return {
        "catalogue_id": catalogue_id,
        "model_version": MODEL_VERSION,
        "n_source_events": n_source_events,
        "n_response_events": len(event_features),
        "n_conditioning_events": len(windows),
        "response_exposure_kyr": float((windows.response_end_kyr_bp - windows.response_start_kyr_bp).sum()),
        "history_kernel": "exponential",
        "history_initialization": "condition_on_exact_oldest_event",
        "history_tau_kyr": tau,
        "history_coefficient_domain": "nonpositive",
        "initial_unobserved_history": initial_history,
        **metrics,
        "nominal_LR_p": metrics["LR_p_value"],
        "pre_phase_preferred_deg": (
            float(np.degrees(np.arctan2(beta_sin, beta_cos)) % 360)
            if amplitude > 1e-10 else np.nan
        ),
        "pre_phase_rate_ratio_max_vs_min": float(np.exp(2 * amplitude)),
        "pre_phase_amplitude": amplitude,
        "beta_history": betas.get("same_type_exponential_history", np.nan),
        "mis6_vs_ngrip_rate_ratio_full": (
            float(np.exp(betas["mis6_segment"])) if "mis6_segment" in betas else np.nan
        ),
        "all_models_converged": bool(reduced.converged and full.converged),
        "likelihood_nesting_ok": metrics["ll_gain_nats"] >= -1e-7,
    }
