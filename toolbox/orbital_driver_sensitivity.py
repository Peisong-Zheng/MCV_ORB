"""Shared orbital inputs and nested comparisons for warming-rate sensitivity.

The caller supplies response bins only. Driver scaling is fixed on these bins
and reused when event ages, counts, and histories change across realizations.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import chi2

from toolbox import poisson, project_config


ECC_TXT = project_config.PROJECT_ROOT / "data/raw/ecc_1000_60_inter100.txt"
OBL_TXT = project_config.OBL_TXT
INSOLATION_NC = project_config.PROJECT_ROOT / "data/raw/solstice_insolation_NH.nc"
DRIVER_IDS = ("ecc", "obl", "insol65n")
DRIVER_LABELS = {
    "ecc": "Eccentricity",
    "obl": "Obliquity",
    "insol65n": "65°N summer-solstice insolation",
}
DRIVER_TERMS = {driver: f"{driver}_scaled" for driver in DRIVER_IDS}
DRIVER_RAW_COLUMNS = {"ecc": "ecc", "obl": "obl_deg", "insol65n": "insol65n_Wm2"}
PHASE_TERMS = ("pre_phase_sin", "pre_phase_cos")
NESTING_TOLERANCE = 1e-7
BOUND_TOLERANCE = 1e-5


def _ordered_series(age, values):
    """Sort a finite, unique source age series before interpolation."""
    age = np.asarray(age, dtype=float)
    values = np.asarray(values, dtype=float)
    if age.ndim != 1 or values.shape != age.shape or len(age) < 2:
        raise ValueError("Orbital inputs must contain at least two paired ages and values")
    if not np.isfinite(age).all() or not np.isfinite(values).all():
        raise ValueError("Orbital inputs contain non-finite ages or values")
    order = np.argsort(age)
    age, values = age[order], values[order]
    if np.any(np.diff(age) <= 0):
        raise ValueError("Orbital input ages must be unique")
    return age, values


def prepare_drivers(frame: pd.DataFrame):
    """Interpolate e, obliquity (degrees), and Q65 (W m-2) onto BP1950 bins.

    Return a new response frame, the fixed mean/range scales, and provenance.
    The La2004 source time is relative to J2000; positive past ages therefore
    decrease by 0.05 kyr when expressed relative to 1950.
    """
    response = frame.copy()
    target_age = response["bin_center_ka"].to_numpy(float)
    if not len(target_age) or not np.isfinite(target_age).all():
        raise ValueError("Response-bin ages must be nonempty and finite")
    if "in_response_interval" in response and not response.in_response_interval.all():
        raise ValueError("prepare_drivers requires response bins only")

    # Laskar et al. (2004), doi:10.1051/0004-6361:20041335.
    # TXT time is negative in the past; obliquity source values are radians.
    source_data = {}
    for driver, path in (("ecc", ECC_TXT), ("obl", OBL_TXT)):
        raw = np.loadtxt(path)
        if raw.ndim != 2 or raw.shape[1] != 2:
            raise ValueError(f"Expected two source columns in {path}")
        age_bp = -raw[:, 0] + project_config.ORBITAL_AGE_OFFSET_TO_BP1950_KA
        values = raw[:, 1] if driver == "ecc" else np.rad2deg(raw[:, 1])
        age_bp, values = _ordered_series(age_bp, values)
        source_data[driver] = {
            "path": path, "age": age_bp, "values": values,
            "source_units": "dimensionless" if driver == "ecc" else "radians",
            "units": "dimensionless" if driver == "ecc" else "degrees",
            "age_conversion": "age_BP1950 = -source_time - 0.05 kyr",
            "epoch_evidence": "La2004 source convention; project age-epoch audit",
        }

    with xr.open_dataset(INSOLATION_NC) as ds:
        solar_longitude = float(ds["solar_longitude_deg"].item())
        if not np.isclose(solar_longitude, 90.0, atol=1e-10, rtol=0):
            raise ValueError("Insolation input is not northern summer solstice (90 degrees)")
        if ds["daily_mean_insolation_Wm2"].attrs.get("units") != "W m-2":
            raise ValueError("Expected insolation source units W m-2")
        # Exact latitude selection avoids silently using a neighbouring grid point.
        q65 = ds.swap_dims({"latitude": "latitude_degN"})[
            "daily_mean_insolation_Wm2"
        ].sel(latitude_degN=65.0)
        age_bp = ds["age_kyr_BP"].to_numpy() + project_config.ORBITAL_AGE_OFFSET_TO_BP1950_KA
        age_bp, values = _ordered_series(age_bp, q65.to_numpy())
        source_data["insol65n"] = {
            "path": INSOLATION_NC, "age": age_bp, "values": values,
            "source_units": "W m-2", "units": "W m-2",
            "age_conversion": "age_BP1950 = source_age_kyr_BP - 0.05 kyr",
            "epoch_evidence": (
                "J2000 inferred by numerical reconstruction from unshifted La2004 "
                "inputs; source coordinate name alone does not establish BP1950"
            ),
            "latitude_degN": 65.0, "solar_longitude_deg": solar_longitude,
            "solar_constant": ds.attrs.get("solar_constant", ""),
            "source_history": ds.attrs.get("history", ""),
        }

    scaling_rows, provenance_rows = [], []
    for driver in DRIVER_IDS:
        source = source_data[driver]
        age, values = source["age"], source["values"]
        if target_age.min() < age[0] or target_age.max() > age[-1]:
            raise ValueError(f"Response ages require extrapolation of {driver}")
        interpolated = np.interp(target_age, age, values)
        mean, value_range = float(interpolated.mean()), float(np.ptp(interpolated))
        if value_range <= 0:
            raise ValueError(f"No {driver} variation in the response interval")
        response[DRIVER_RAW_COLUMNS[driver]] = interpolated
        response[DRIVER_TERMS[driver]] = (interpolated - mean) / value_range
        scaling_rows.append({
            "driver_id": driver, "raw_column": DRIVER_RAW_COLUMNS[driver],
            "scaled_column": DRIVER_TERMS[driver], "units": source["units"],
            "response_mean": mean, "response_min": float(interpolated.min()),
            "response_max": float(interpolated.max()), "response_range": value_range,
            "n_response_bins": len(response),
            "scaling": "(value - response-bin arithmetic mean) / response-bin range",
        })
        provenance_rows.append({
            "driver_id": driver, "source_file": str(source["path"]),
            "sha256": hashlib.sha256(source["path"].read_bytes()).hexdigest(),
            "reference": project_config.ORBITAL_REFERENCE,
            "source_units": source["source_units"], "analysis_units": source["units"],
            "source_epoch": "J2000", "analysis_epoch": "BP1950",
            "age_offset_kyr": project_config.ORBITAL_AGE_OFFSET_TO_BP1950_KA,
            "age_conversion": source["age_conversion"],
            "epoch_evidence": source["epoch_evidence"],
            "source_spacing_kyr": float(np.median(np.diff(age))),
            "source_age_min_BP1950_kyr": float(age[0]),
            "source_age_max_BP1950_kyr": float(age[-1]),
            "interpolation": "linear; no extrapolation; native resolution unchanged",
            **{key: source.get(key, np.nan) for key in (
                "latitude_degN", "solar_longitude_deg", "solar_constant", "source_history"
            )},
        })
    return response, pd.DataFrame(scaling_rows), pd.DataFrame(provenance_rows)


def model_specs(baseline_terms):
    """Eight prespecified models; terms exclude the automatically fitted intercept."""
    baseline = tuple(baseline_terms)
    forbidden = set(PHASE_TERMS) | set(DRIVER_TERMS.values()) | {"intercept"}
    if len(set(baseline)) != len(baseline) or set(baseline) & forbidden:
        raise ValueError("Baseline terms must be unique and exclude orbital terms/intercept")
    specs = {"B": baseline, "BP": baseline + PHASE_TERMS}
    for driver in DRIVER_IDS:
        specs[f"B_{driver}"] = baseline + (DRIVER_TERMS[driver],)
        specs[f"BP_{driver}"] = baseline + PHASE_TERMS + (DRIVER_TERMS[driver],)
    return specs


def holm_adjust(p_values):
    """Holm-adjust the complete planned family, retaining missing tests as missing.

    Missing p values still count towards the family size; they are assigned one
    during adjustment and restored to NaN in the returned array.
    """
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or np.any(np.isfinite(p) & ((p < 0) | (p > 1))):
        raise ValueError("Holm adjustment requires one-dimensional p values in [0, 1]")
    finite = np.isfinite(p)
    values = np.where(finite, p, 1.0)
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(len(p), dtype=float)
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate(
        values[order] * np.arange(len(p), 0, -1)
    ))
    adjusted[~finite] = np.nan
    return adjusted


def _comparison_specs():
    comparisons = [("phase_reference", "pre", "reference", "B", "BP")]
    for driver in DRIVER_IDS:
        comparisons.extend([
            (f"{driver}_after_base", driver, "driver_after_base", "B", f"B_{driver}"),
            (f"{driver}_after_phase", driver, "driver_after_phase", "BP", f"BP_{driver}"),
            (f"phase_after_{driver}", driver, "phase_after_driver", f"B_{driver}", f"BP_{driver}"),
        ])
    return comparisons


def fit_models(frame: pd.DataFrame, baseline_terms):
    """Fit eight models on identical bins and return all ten nested comparisons.

    AICc follows the existing project convention n = number of response bins.
    Chi-square p values are nominal; no event-process bootstrap is performed.
    Numerical failures remain in the output with a reason and missing PI.
    """
    specs = model_specs(baseline_terms)
    duration_column = "dt_ka" if "dt_ka" in frame else "dt_kyr"
    dt = frame[duration_column].to_numpy(float)
    y = frame["event_count"].to_numpy(float)
    if "dt_ka" in frame and "dt_kyr" in frame:
        if not np.allclose(frame.dt_ka, frame.dt_kyr, atol=1e-12, rtol=0):
            raise ValueError("dt_ka and dt_kyr disagree")
    if not len(frame) or not np.isfinite(dt).all() or np.any(dt <= 0):
        raise ValueError("Response durations must be nonempty, finite, and positive")
    if not np.isfinite(y).all() or np.any(y < 0) or not np.allclose(y, np.rint(y)):
        raise ValueError("Response event counts must be finite nonnegative integers")
    if y.sum() <= 0:
        raise ValueError("PI per event is undefined for an empty event catalogue")
    if "in_response_interval" in frame and not frame.in_response_interval.all():
        raise ValueError("fit_models requires response bins only")
    support = {"n_events": int(y.sum()), "n_bins": len(frame), "exposure_kyr": float(dt.sum())}
    model_rows, coefficient_rows = [], []
    rates = frame.loc[:, [c for c in ("segment_id", "bin_center_ka") if c in frame]].copy()

    for model_id, terms in specs.items():
        x = frame.loc[:, list(terms)].to_numpy(float)
        design = np.column_stack([np.ones(len(frame)), x])
        n_parameters = design.shape[1]
        finite = bool(np.isfinite(design).all())
        rank = int(np.linalg.matrix_rank(design)) if finite else 0
        condition = float(np.linalg.cond(design)) if finite else np.nan
        reasons = []
        if not finite:
            reasons.append("non-finite predictor values")
        elif rank != n_parameters:
            reasons.append("rank-deficient design")
        row = dict(model_id=model_id, terms=";".join(terms), n_parameters=n_parameters,
                   rank=rank, condition_number=condition, converged=False,
                   optimizer_message="not fitted", log_likelihood=np.nan, AIC=np.nan,
                   AICc=np.nan, eta_min=np.nan, eta_max=np.nan, n_eta_clipped_low=0,
                   n_eta_clipped_high=0, eta_clipping_used=False, n_bound_hits=0,
                   bound_hits="", pre_phase_preferred_deg=np.nan,
                   pre_phase_rate_ratio_max_vs_min=np.nan, **support)
        beta = np.full(n_parameters, np.nan)
        rates[model_id] = np.nan
        if not reasons:
            try:
                fit = poisson.fit_binned_poisson_arrays(x, y, dt)
                beta = fit.beta
                # These are the bounds in the shared main-analysis optimizer.
                lower = np.full(n_parameters, -20.0)
                upper = np.r_[5.0, np.full(n_parameters - 1, 20.0)]
                at_bound = ((np.abs(beta - lower) <= BOUND_TOLERANCE)
                            | (np.abs(beta - upper) <= BOUND_TOLERANCE))
                clipped = fit.n_eta_clipped_low + fit.n_eta_clipped_high > 0
                if not fit.converged:
                    reasons.append("optimizer did not converge")
                if not np.isfinite(beta).all() or not np.isfinite(fit.log_likelihood):
                    reasons.append("non-finite fitted coefficients or likelihood")
                if not np.isfinite(fit.fitted_rate_per_kyr).all():
                    reasons.append("non-finite fitted rates")
                if clipped:
                    reasons.append("linear predictor used numerical clipping")
                if at_bound.any():
                    reasons.append("coefficient reached optimizer bound")
                coefficient_names = ("intercept",) + terms
                row.update(
                    converged=fit.converged, optimizer_message=fit.optimizer_message,
                    log_likelihood=fit.log_likelihood, AIC=fit.aic, AICc=fit.aicc,
                    eta_min=fit.eta_min, eta_max=fit.eta_max,
                    n_eta_clipped_low=fit.n_eta_clipped_low,
                    n_eta_clipped_high=fit.n_eta_clipped_high, eta_clipping_used=clipped,
                    n_bound_hits=int(at_bound.sum()),
                    bound_hits=";".join(np.asarray(coefficient_names)[at_bound]),
                )
                rates[model_id] = fit.fitted_rate_per_kyr
                if all(term in terms for term in PHASE_TERMS) and not reasons:
                    beta_sin, beta_cos = [beta[1 + terms.index(term)] for term in PHASE_TERMS]
                    amplitude = np.hypot(beta_sin, beta_cos)
                    row["pre_phase_rate_ratio_max_vs_min"] = float(np.exp(2 * amplitude))
                    if amplitude > 1e-12:
                        row["pre_phase_preferred_deg"] = float(
                            np.degrees(np.arctan2(beta_sin, beta_cos)) % 360
                        )
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
                reasons.append(f"fit failed: {error}")
        row.update(fit_valid=not reasons, invalid_reason="; ".join(reasons))
        model_rows.append(row)
        for term, value in zip(("intercept",) + terms, beta):
            coefficient_rows.append({"model_id": model_id, "term": term, "beta": value,
                                     "fit_valid": row["fit_valid"], "invalid_reason": row["invalid_reason"]})

    models = pd.DataFrame(model_rows)
    lookup = models.set_index("model_id")
    comparison_rows = []
    for comparison_id, driver, group, reduced_id, full_id in _comparison_specs():
        reduced, full = lookup.loc[reduced_id], lookup.loc[full_id]
        df = int(full.n_parameters - reduced.n_parameters)
        gain = float(full.log_likelihood - reduced.log_likelihood)
        reasons = [f"{model_id}: {lookup.loc[model_id, 'invalid_reason']}"
                   for model_id in (reduced_id, full_id) if not lookup.loc[model_id, "fit_valid"]]
        nesting_ok = bool(np.isfinite(gain) and gain >= -NESTING_TOLERANCE)
        if not reasons and not nesting_ok:
            reasons.append("full likelihood below reduced likelihood")
        valid = not reasons
        # Only round-off-sized negative gains are rounded to zero, after validation.
        lr = 2 * max(gain, 0.0) if valid else np.nan
        comparison_rows.append({
            "comparison_id": comparison_id, "driver_id": driver, "comparison_group": group,
            "reduced_model_id": reduced_id, "full_model_id": full_id, "df": df,
            "loglik_reduced": reduced.log_likelihood, "loglik_full": full.log_likelihood,
            "ll_gain_nats": gain, "info_bits_per_event": lr / (2 * np.log(2) * y.sum()),
            "LR_statistic": lr, "nominal_p": float(chi2.sf(lr, df)) if valid else np.nan,
            "delta_AIC": float(full.AIC - reduced.AIC),
            "delta_AICc": float(full.AICc - reduced.AICc),
            "fit_valid": valid, "invalid_reason": "; ".join(reasons),
            "likelihood_nesting_ok": nesting_ok, **support,
        })
    comparisons = pd.DataFrame(comparison_rows)
    comparisons["holm_nominal_p"] = np.nan
    family = comparisons.comparison_group.ne("reference")
    comparisons.loc[family, "holm_nominal_p"] = holm_adjust(comparisons.loc[family, "nominal_p"])
    return {"models": models, "comparisons": comparisons,
            "coefficients": pd.DataFrame(coefficient_rows), "fitted_rates": rates}


def summarize_comparisons(point_df: pd.DataFrame, mc_df: pd.DataFrame):
    """Point estimates and 95% chronology ranges, retaining invalid fit counts."""
    metrics = ("info_bits_per_event", "LR_statistic", "nominal_p", "delta_AIC", "delta_AICc")
    rows = []
    for _, point in point_df.iterrows():
        sample = mc_df.loc[mc_df.comparison_id.eq(point.comparison_id)]
        valid = sample.loc[sample.fit_valid]
        row = {key: point[key] for key in (
            "comparison_id", "driver_id", "comparison_group", "reduced_model_id",
            "full_model_id", "df", "n_events", "n_bins", "exposure_kyr",
        )}
        row.update(point_fit_valid=bool(point.fit_valid), point_invalid_reason=point.invalid_reason,
                   holm_nominal_p_point=point.holm_nominal_p, n_mc_total=len(sample),
                   n_mc_valid=len(valid), n_mc_invalid=len(sample) - len(valid))
        for metric in metrics:
            row[f"{metric}_point"] = point[metric]
            values = valid[metric].to_numpy(float)
            values = values[np.isfinite(values)]
            quantiles = np.quantile(values, [0.025, 0.5, 0.975]) if len(values) else [np.nan] * 3
            row.update({f"{metric}_{label}": value for label, value in
                        zip(("q025", "median", "q975"), quantiles)})
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_phase(point_models: pd.DataFrame, mc_models: pd.DataFrame):
    """Unwrap preferred phase around each model's own point-age estimate."""
    metrics = ("pre_phase_preferred_deg", "pre_phase_rate_ratio_max_vs_min")
    phase_models = [model_id for model_id in ("BP", *(f"BP_{driver}" for driver in DRIVER_IDS))
                    if model_id in point_models.model_id.to_numpy()]
    rows = []
    for model_id in phase_models:
        point = point_models.loc[point_models.model_id.eq(model_id)].iloc[0]
        sample = mc_models.loc[mc_models.model_id.eq(model_id)]
        valid = sample.loc[sample.fit_valid]
        row = dict(model_id=model_id, point_fit_valid=bool(point.fit_valid),
                   point_invalid_reason=point.get("invalid_reason", ""), n_mc_total=len(sample),
                   n_mc_valid=len(valid), n_mc_invalid=len(sample) - len(valid),
                   phase_quantiles="degrees unwrapped about this model's point phase")
        for metric in metrics:
            row[f"{metric}_point"] = point[metric]
            values = valid[metric].to_numpy(float)
            values = values[np.isfinite(values)]
            if metric == "pre_phase_preferred_deg":
                reference = point[metric]
                values = reference + (values - reference + 180) % 360 - 180
                values = values[np.isfinite(values)]
            quantiles = np.quantile(values, [0.025, 0.5, 0.975]) if len(values) else [np.nan] * 3
            row.update({f"{metric}_{label}": value for label, value in
                        zip(("q025", "median", "q975"), quantiles)})
        rows.append(row)
    return pd.DataFrame(rows)
