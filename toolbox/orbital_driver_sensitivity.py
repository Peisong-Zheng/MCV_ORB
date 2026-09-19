"""Orbital inputs and continuous-time nested warming-rate comparisons."""

from __future__ import annotations

import hashlib
import time

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import chi2

from toolbox import combined_likelihood, project_config


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


def load_driver_sources():
    """Read native e, obliquity (degrees), and Q65 (W m-2) on BP1950 ages.

    Preserve the source knots for exact-age interpolation and integration.
    The La2004 source time is relative to J2000; positive past ages therefore
    decrease by 0.05 kyr when expressed relative to 1950.
    """
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

    provenance_rows = []
    for driver in DRIVER_IDS:
        source = source_data[driver]
        age, values = source["age"], source["values"]
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
    return source_data, pd.DataFrame(provenance_rows)


def prepare_drivers(context):
    """Add native forcing interpolants and fixed nominal time-weighted scales."""
    sources, provenance = load_driver_sources()
    rows = []
    for driver in DRIVER_IDS:
        source = sources[driver]
        context = combined_likelihood.add_forcing(context, driver, source["age"], source["values"])
        scale = context.scaling[driver]
        rows.append(dict(driver_id=driver, raw_column=driver,
            scaled_column=DRIVER_TERMS[driver], units=source["units"],
            response_mean=scale["mean"], response_min=scale["min"],
            response_max=scale["max"], response_range=scale["range"],
            scaling="(value - nominal exposure-time mean) / nominal interpolant range"))
    return context, pd.DataFrame(rows), provenance


def fit_named_models(design, specs):
    """Fit named continuous designs with the common nonpositive history domain.

    Both the event sum and intensity integral use the same forcing interpolants.
    A numerical design/fit failure remains explicit in the candidate table.
    """
    support = dict(n_events=len(design.event_frame), exposure_kyr=float(design.weights.sum()))
    model_rows, coefficient_rows = [], []
    event_rates = design.event_frame.loc[:, [name for name in
        ("event_id", "segment_id", "age_kyr_bp") if name in design.event_frame]].copy()
    for model_id, terms in specs.items():
        terms = tuple(terms)
        beta = np.full(1 + len(terms), np.nan)
        row = dict(model_id=model_id, terms=";".join(terms), n_parameters=len(beta),
            converged=False, fit_valid=False, invalid_reason="", log_likelihood=np.nan,
            AIC=np.nan, pre_phase_preferred_deg=np.nan,
            pre_phase_rate_ratio_max_vs_min=np.nan, **support)
        event_rates[model_id] = np.nan
        try:
            fitted = combined_likelihood.fit_terms(design, terms)
            beta = fitted.beta
            row.update(converged=fitted.converged, log_likelihood=fitted.log_likelihood,
                       AIC=fitted.aic, fit_valid=bool(fitted.converged and fitted.identifiable),
                       fit_status=fitted.status)
            if not fitted.converged or not fitted.identifiable:
                row["invalid_reason"] = "continuous fit failed convergence/KKT checks"
            if not np.isfinite(beta).all() or not np.isfinite(fitted.log_likelihood):
                row.update(fit_valid=False, invalid_reason="non-finite fitted coefficients or likelihood")
            if row["fit_valid"]:
                x = design.event_frame.loc[:, list(terms)].to_numpy(float)
                event_rates[model_id] = np.exp(beta[0] + x @ beta[1:])
                if all(term in terms for term in PHASE_TERMS):
                    b_sin, b_cos = [beta[1 + terms.index(term)] for term in PHASE_TERMS]
                    amplitude = np.hypot(b_sin, b_cos)
                    row["pre_phase_rate_ratio_max_vs_min"] = float(np.exp(2 * amplitude))
                    if amplitude > 1e-12:
                        row["pre_phase_preferred_deg"] = float(np.degrees(np.arctan2(b_sin, b_cos)) % 360)
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
            row["invalid_reason"] = str(error)
        model_rows.append(row)
        coefficient_rows.extend(dict(model_id=model_id, term=term, beta=value,
            fit_valid=row["fit_valid"], invalid_reason=row["invalid_reason"])
            for term, value in zip(("intercept",) + terms, beta))
    return dict(models=pd.DataFrame(model_rows), coefficients=pd.DataFrame(coefficient_rows),
                fitted_rates=event_rates)


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


def fit_models(design, baseline_terms):
    """Fit the eight prespecified continuous models and ten nested comparisons."""
    specs = model_specs(baseline_terms)
    result = fit_named_models(design, specs)
    lookup = result["models"].set_index("model_id")
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
        lr = 2 * max(gain, 0.0) if valid else np.nan
        comparison_rows.append({
            "comparison_id": comparison_id, "driver_id": driver, "comparison_group": group,
            "reduced_model_id": reduced_id, "full_model_id": full_id, "df": df,
            "loglik_reduced": reduced.log_likelihood, "loglik_full": full.log_likelihood,
            "ll_gain_nats": gain, "gain_bits_per_event": lr / (2 * np.log(2) * full.n_events),
            "LR_statistic": lr, "nominal_p": float(chi2.sf(lr, df)) if valid else np.nan,
            "delta_AIC": float(full.AIC - reduced.AIC),
            "fit_valid": valid, "invalid_reason": "; ".join(reasons),
            "likelihood_nesting_ok": nesting_ok, "n_events": int(full.n_events),
            "exposure_kyr": float(full.exposure_kyr),
        })
    comparisons = pd.DataFrame(comparison_rows)
    comparisons["holm_nominal_p"] = np.nan
    family = comparisons.comparison_group.ne("reference")
    comparisons.loc[family, "holm_nominal_p"] = holm_adjust(comparisons.loc[family, "nominal_p"])
    result["comparisons"] = comparisons
    return result


def analyze_chronologies(context, selected, age_columns, show_progress=True):
    """Fit point ages and the existing chronology subset without changing its IDs."""
    from toolbox import orbital_driver_reporting as reporting

    events = context.events
    point_design = combined_likelihood.prepare_catalogue(events, context)
    point = fit_models(point_design, context.reduced_terms)
    if not point["models"].fit_valid.all() or not point["comparisons"].fit_valid.all():
        raise RuntimeError(f"Invalid point-age orbital fits:\n{point['models'].to_string(index=False)}")
    reference = combined_likelihood.fit_catalogue(events, context).summary
    reference_check = reporting.check_reference(point, pd.Series(reference))
    mc_models, mc_comparisons, status = [], [], []
    started = time.perf_counter()
    for index, row in selected.iterrows():
        shifted = events.copy()
        shifted[combined_likelihood.EVENT_AGE_COLUMN] = row[age_columns].to_numpy(float)
        reason = ""
        try:
            design = combined_likelihood.prepare_catalogue(shifted, context)
        except ValueError as error:
            reason = str(error)
            fitted = reporting.invalid_tables(point, reason)
        else:
            fitted = fit_models(design, context.reduced_terms)
        for key, output in (("models", mc_models), ("comparisons", mc_comparisons)):
            output.append(fitted[key].assign(realization_id=row.realization_id))
        valid = bool(fitted["comparisons"].fit_valid.all())
        status.append(dict(realization_id=row.realization_id, within_observation_support=not bool(reason),
            n_response_events=fitted["models"].iloc[0].n_events,
            response_exposure_kyr=fitted["models"].iloc[0].exposure_kyr,
            all_comparisons_valid=valid, invalid_reason=reason or "; ".join(
                fitted["comparisons"].loc[~fitted["comparisons"].fit_valid, "invalid_reason"].unique())))
        if show_progress and (index + 1) % 100 == 0:
            print(f"Orbital drivers: {index + 1}/{len(selected)} chronologies "
                  f"({time.perf_counter() - started:.0f} s)", flush=True)
    mc_models = pd.concat(mc_models, ignore_index=True)
    mc_comparisons = pd.concat(mc_comparisons, ignore_index=True)
    support = pd.DataFrame([dict(segment_id=segment_id,
        observation_start_kyr_bp=segment.observation_start_kyr_bp,
        observation_end_kyr_bp=segment.observation_end_kyr_bp,
        response_start_kyr_bp=segment.response_start_kyr_bp,
        response_end_kyr_bp=segment.response_end_kyr_bp,
        anchor_age_kyr_bp=segment.anchor_age_kyr_bp)
        for segment_id, segment in point_design.context.segments.items()])
    return dict(events=point_design.all_events, response=point_design.event_frame,
        integration=point_design.integration_frame, point=point, support=support,
        selected_realizations=selected, realization_status=pd.DataFrame(status),
        mc_models=mc_models, mc_comparisons=mc_comparisons, reference_check=reference_check,
        comparison_summary=summarize_comparisons(point["comparisons"], mc_comparisons),
        phase_summary=summarize_phase(point["models"], mc_models))


def summarize_comparisons(point_df: pd.DataFrame, mc_df: pd.DataFrame):
    """Point estimates and 95% chronology ranges, retaining invalid fit counts."""
    metrics = ("gain_bits_per_event", "LR_statistic", "nominal_p", "delta_AIC")
    rows = []
    for _, point in point_df.iterrows():
        sample = mc_df.loc[mc_df.comparison_id.eq(point.comparison_id)]
        valid = sample.loc[sample.fit_valid]
        row = {key: point[key] for key in (
            "comparison_id", "driver_id", "comparison_group", "reduced_model_id",
            "full_model_id", "df", "n_events", "exposure_kyr",
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
