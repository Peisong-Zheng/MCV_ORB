"""Poisson likelihood utilities for binned event-count models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from toolbox.model_stats import information_criteria

ETA_MIN = -50.0
ETA_MAX = 20.0


@dataclass
class FittedPoissonArrays:
    """Result of fitting a binned Poisson model to numeric arrays."""

    beta: np.ndarray
    converged: bool
    optimizer_message: str
    log_likelihood: float
    aic: float
    aicc: float
    bic: float
    fitted_rate_per_kyr: np.ndarray
    fitted_mu_per_bin: np.ndarray
    eta_min: float
    eta_max: float
    n_eta_clipped_low: int
    n_eta_clipped_high: int


@dataclass
class FittedPoissonModel:
    """Container for one fitted binned Poisson hazard model."""

    dataset_id: str
    dataset_label: str
    model_id: str
    model_label: str
    terms: tuple[str, ...]
    beta: np.ndarray
    converged: bool
    optimizer_message: str
    log_likelihood: float
    aic: float
    aicc: float
    bic: float
    fitted_rate_per_kyr: np.ndarray
    fitted_mu_per_bin: np.ndarray
    eta_min: float
    eta_max: float
    n_eta_clipped_low: int
    n_eta_clipped_high: int


def design_matrix(frame: pd.DataFrame, terms: tuple[str, ...]) -> np.ndarray:
    """Return a numeric design matrix for a tuple of predictor columns."""

    if not terms:
        return np.empty((len(frame), 0), dtype=float)
    return frame.loc[:, list(terms)].to_numpy(dtype=float)


def linear_predictor(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Return the unclipped linear predictor eta = beta0 + X beta."""

    return beta[0] + x @ beta[1:]


def clip_linear_predictor(eta: np.ndarray) -> np.ndarray:
    """Clip eta to the finite range used in the likelihood."""

    return np.clip(eta, ETA_MIN, ETA_MAX)


def eta_clip_diagnostics(beta: np.ndarray, x: np.ndarray) -> dict[str, float | int]:
    """Summarize whether fitted eta values hit the numerical clipping limits."""

    eta = linear_predictor(beta, x)
    return {
        "eta_min": float(np.nanmin(eta)) if len(eta) else float("nan"),
        "eta_max": float(np.nanmax(eta)) if len(eta) else float("nan"),
        "n_eta_clipped_low": int(np.sum(eta < ETA_MIN)),
        "n_eta_clipped_high": int(np.sum(eta > ETA_MAX)),
    }


def binned_poisson_loglik(
    beta: np.ndarray, x: np.ndarray, y: np.ndarray, dt: np.ndarray
) -> float:
    """Poisson log likelihood for binned counts with a log-duration offset.

    The fitted model is for a per-kyr event rate,

        log(lambda_i) = beta0 + x_i beta,

    while the observed response is an event count in a finite bin,

        y_i ~ Poisson(mu_i),   mu_i = dt_i * lambda_i.
    """

    eta = clip_linear_predictor(linear_predictor(beta, x))
    log_mu = np.log(dt) + eta
    mu = np.exp(log_mu)
    return float(np.sum(y * log_mu - mu - gammaln(y + 1.0)))


def negative_binned_poisson_loglik(
    beta: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    dt: np.ndarray,
) -> float:
    """Numerically safe objective for scipy.optimize.minimize."""

    value = -binned_poisson_loglik(beta, x, y, dt)
    if not np.isfinite(value):
        return 1e100
    return float(value)


def fitted_rate_and_mu(
    beta: np.ndarray, x: np.ndarray, dt: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert fitted coefficients into rates per kyr and expected bin counts."""

    eta = clip_linear_predictor(linear_predictor(beta, x))
    rate = np.exp(eta)
    return rate, rate * dt


def fit_binned_poisson_arrays(
    x: np.ndarray,
    y: np.ndarray,
    dt: np.ndarray,
    *,
    maxiter: int = 4000,
    ftol: float = 1e-11,
) -> FittedPoissonArrays:
    """Fit a binned Poisson GLM from numeric design, count, and duration arrays."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dt = np.asarray(dt, dtype=float)
    if x.ndim != 2:
        raise ValueError("x must be a two-dimensional design matrix.")
    if len(y) != len(dt) or len(y) != x.shape[0]:
        raise ValueError("x, y, and dt must have the same number of rows.")
    if np.any(dt <= 0.0):
        raise ValueError("All bin durations must be positive.")

    n_terms = int(x.shape[1])
    duration = float(dt.sum())
    n_events = float(y.sum())
    n_obs = len(y)
    beta0 = np.zeros(1 + n_terms, dtype=float)
    beta0[0] = np.log(max(n_events / duration, 1e-12))

    if n_terms == 0:
        beta = beta0
        converged = True
        message = "analytic stationary MLE"
    else:
        bounds = [(-20.0, 5.0)] + [(-20.0, 20.0)] * n_terms
        result = minimize(
            negative_binned_poisson_loglik,
            beta0,
            args=(x, y, dt),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(maxiter), "ftol": float(ftol)},
        )
        beta = result.x
        converged = bool(result.success)
        message = str(result.message)

    rate, mu = fitted_rate_and_mu(beta, x, dt)
    log_likelihood = binned_poisson_loglik(beta, x, y, dt)
    criteria = information_criteria(log_likelihood, len(beta), n_obs)
    diagnostics = eta_clip_diagnostics(beta, x)

    return FittedPoissonArrays(
        beta=beta,
        converged=converged,
        optimizer_message=message,
        log_likelihood=log_likelihood,
        aic=criteria["AIC"],
        aicc=criteria["AICc"],
        bic=criteria["BIC"],
        fitted_rate_per_kyr=rate,
        fitted_mu_per_bin=mu,
        eta_min=float(diagnostics["eta_min"]),
        eta_max=float(diagnostics["eta_max"]),
        n_eta_clipped_low=int(diagnostics["n_eta_clipped_low"]),
        n_eta_clipped_high=int(diagnostics["n_eta_clipped_high"]),
    )


def fit_poisson_model(
    dataset_frame: pd.DataFrame,
    model_id: str,
    terms: tuple[str, ...],
    model_label: str,
    *,
    maxiter: int = 4000,
    ftol: float = 1e-11,
) -> FittedPoissonModel:
    """Fit one binned Poisson hazard GLM for one event dataset.

    The fitted coefficients describe a per-kyr event rate. The finite bin
    duration enters the likelihood through the ``dt_ka`` offset.
    """

    x = design_matrix(dataset_frame, terms)
    y = dataset_frame["event_count"].to_numpy(dtype=float)
    dt = dataset_frame["dt_ka"].to_numpy(dtype=float)
    fit = fit_binned_poisson_arrays(x, y, dt, maxiter=maxiter, ftol=ftol)

    return FittedPoissonModel(
        dataset_id=str(dataset_frame["dataset_id"].iloc[0]),
        dataset_label=str(dataset_frame["dataset_label"].iloc[0]),
        model_id=model_id,
        model_label=model_label,
        terms=terms,
        beta=fit.beta,
        converged=fit.converged,
        optimizer_message=fit.optimizer_message,
        log_likelihood=fit.log_likelihood,
        aic=fit.aic,
        aicc=fit.aicc,
        bic=fit.bic,
        fitted_rate_per_kyr=fit.fitted_rate_per_kyr,
        fitted_mu_per_bin=fit.fitted_mu_per_bin,
        eta_min=fit.eta_min,
        eta_max=fit.eta_max,
        n_eta_clipped_low=fit.n_eta_clipped_low,
        n_eta_clipped_high=fit.n_eta_clipped_high,
    )


def model_lookup(
    models: list[FittedPoissonModel],
) -> dict[tuple[str, str], FittedPoissonModel]:
    """Index fitted models by event dataset and model identifier."""

    return {(model.dataset_id, model.model_id): model for model in models}


def build_model_summary(
    models: list[FittedPoissonModel],
    binned_inputs: pd.DataFrame,
    *,
    bin_width_ka: float | None = None,
) -> pd.DataFrame:
    """Summarize fitted models and derive phase-response quantities."""

    rows = []
    for model in models:
        subset = binned_inputs[binned_inputs["dataset_id"].eq(model.dataset_id)]
        y = subset["event_count"].to_numpy(dtype=float)
        width = (
            float(bin_width_ka)
            if bin_width_ka is not None
            else float(np.nanmedian(subset["dt_ka"]))
        )
        row = {
            "dataset_id": model.dataset_id,
            "dataset_label": model.dataset_label,
            "model_id": model.model_id,
            "model_label": model.model_label,
            "terms": "+".join(model.terms) if model.terms else "none",
            "bin_width_ka": width,
            "n_bins": int(len(subset)),
            "n_events": int(y.sum()),
            "n_parameters": int(len(model.beta)),
            "converged": model.converged,
            "optimizer_message": model.optimizer_message,
            "log_likelihood": model.log_likelihood,
            "AIC": model.aic,
            "AICc": model.aicc,
            "BIC": model.bic,
            "expected_events": float(np.sum(model.fitted_mu_per_bin)),
            "max_fitted_rate_per_kyr": float(np.max(model.fitted_rate_per_kyr)),
            "mean_fitted_rate_per_kyr": float(np.mean(model.fitted_rate_per_kyr)),
            "eta_min": model.eta_min,
            "eta_max": model.eta_max,
            "n_eta_clipped_low": model.n_eta_clipped_low,
            "n_eta_clipped_high": model.n_eta_clipped_high,
        }
        for term, beta in zip(("intercept",) + model.terms, model.beta):
            row[f"beta_{term}"] = float(beta)
        if "pre_phase_sin" in model.terms and "pre_phase_cos" in model.terms:
            beta_map = dict(zip(model.terms, model.beta[1:]))
            b_sin = float(beta_map["pre_phase_sin"])
            b_cos = float(beta_map["pre_phase_cos"])
            amplitude = float(np.hypot(b_sin, b_cos))
            preferred = float(np.mod(np.arctan2(b_sin, b_cos), 2.0 * np.pi))
            row["pre_phase_amplitude"] = amplitude
            row["pre_phase_preferred_rad"] = preferred
            row["pre_phase_preferred_deg"] = float(np.degrees(preferred))
            row["pre_phase_rate_ratio_max_vs_min"] = float(np.exp(2.0 * amplitude))
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary["delta_AICc"] = summary["AICc"] - summary.groupby("dataset_id")[
        "AICc"
    ].transform("min")
    summary["rank_AICc"] = summary.groupby("dataset_id")["AICc"].rank(method="first")
    return summary.sort_values(["dataset_id", "AICc"]).reset_index(drop=True)


def build_coefficient_table(models: list[FittedPoissonModel]) -> pd.DataFrame:
    """Collect fitted coefficients and their one-unit rate ratios."""

    rows = []
    for model in models:
        names = ("intercept",) + model.terms
        for name, beta in zip(names, model.beta):
            rows.append(
                {
                    "dataset_id": model.dataset_id,
                    "dataset_label": model.dataset_label,
                    "model_id": model.model_id,
                    "term": name,
                    "beta": float(beta),
                    "rate_ratio_per_unit": float(np.exp(beta)),
                }
            )
    return pd.DataFrame(rows)
