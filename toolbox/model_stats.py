"""Likelihood comparisons and circular statistics for event models."""
import numpy as np
import pandas as pd
from scipy.stats import chi2


def unwrap_phase(values, center):
    """Express circular phases within 180 degrees of the reference phase."""
    return center + (np.asarray(values) - center + 180) % 360 - 180


def nested_likelihood_metrics(*, loglik_full, loglik_reduced, df, n_events,
                              aic_full, aic_reduced):
    """Compare nested models fitted to the same response events and support."""
    gain = float(loglik_full - loglik_reduced)
    if gain < -1e-7:
        raise RuntimeError("Full-model likelihood is below the nested reference")
    lr = 2 * gain
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
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie between 0 and 1.")
    low, high = 0.0, 1.0
    if rayleigh_p_value_from_z(n, n) > alpha:
        return np.nan
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
        mass = rate * integration["weight"].to_numpy(float)
        fitted, _ = np.histogram(integration["pre_phase_deg"].to_numpy(float),
                                 bins=edges, weights=mass)
        ratio = np.where(fitted > 0.0, observed / fitted, np.nan)
        for center, obs, fit_count, rat in zip(centers, observed, fitted, ratio):
            rows.append(dict(model_id=model_id, phase_sector_center_deg=center,
                             observed_events=int(obs), fitted_events=fit_count,
                             observed_over_fitted=rat))
    return pd.DataFrame(rows)
