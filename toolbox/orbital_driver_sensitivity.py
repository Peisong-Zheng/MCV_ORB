"""Orbital inputs and continuous-time nested warming-rate comparisons."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.stats import chi2

from toolbox import event_model
from toolbox.point_process import fit_point_process


DRIVER_IDS = ("ecc", "obl", "insol65n")
DRIVER_TERMS = {driver: f"{driver}_scaled" for driver in DRIVER_IDS}
PHASE_TERMS = ("pre_phase_sin", "pre_phase_cos")
NESTING_TOLERANCE = 1e-7


def fit_named_models(event_features, integration_features, specs):
    """Fit named continuous designs with the common nonpositive history domain.

    Both the event sum and intensity integral use the same forcing interpolants.
    A numerical design/fit failure remains explicit in the candidate table.
    """
    support = dict(n_events=len(event_features), exposure_kyr=float(integration_features.weight.sum()))
    model_rows, fits = [], {}
    for model_id, terms in specs.items():
        terms = tuple(terms)
        beta = np.full(len(terms), np.nan)
        row = dict(model_id=model_id, terms=";".join(term for term in terms if term != "intercept"), n_parameters=len(beta),
            converged=False, fit_valid=False, invalid_reason="", log_likelihood=np.nan,
            AIC=np.nan, pre_phase_preferred_deg=np.nan,
            pre_phase_rate_ratio_max_vs_min=np.nan, **support)
        fits[model_id] = None
        try:
            fitted = fit_point_process(
                event_features.loc[:, terms].to_numpy(float),
                integration_features.loc[:, terms].to_numpy(float),
                integration_features.weight.to_numpy(float), terms,
            )
            fits[model_id] = fitted
            beta = fitted.beta
            row.update(converged=fitted.converged, log_likelihood=fitted.log_likelihood,
                       AIC=fitted.aic, fit_valid=bool(fitted.converged and fitted.identifiable),
                       fit_status=fitted.status)
            if not fitted.converged or not fitted.identifiable:
                row["invalid_reason"] = "continuous fit failed convergence/KKT checks"
            if not np.isfinite(beta).all() or not np.isfinite(fitted.log_likelihood):
                row.update(fit_valid=False, invalid_reason="non-finite fitted coefficients or likelihood")
            if row["fit_valid"]:
                if all(term in terms for term in PHASE_TERMS):
                    b_sin, b_cos = [beta[terms.index(term)] for term in PHASE_TERMS]
                    amplitude = np.hypot(b_sin, b_cos)
                    row["pre_phase_rate_ratio_max_vs_min"] = float(np.exp(2 * amplitude))
                    if amplitude > 1e-12:
                        row["pre_phase_preferred_deg"] = float(np.degrees(np.arctan2(b_sin, b_cos)) % 360)
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
            row["invalid_reason"] = str(error)
        model_rows.append(row)
    return dict(models=pd.DataFrame(model_rows), fits=fits)


def model_specs(baseline_terms):
    """Eight prespecified models, with the intercept explicit in each term list."""
    baseline = tuple(baseline_terms)
    forbidden = set(PHASE_TERMS) | set(DRIVER_TERMS.values())
    if (not baseline or baseline[0] != "intercept"
            or len(set(baseline)) != len(baseline) or set(baseline) & forbidden):
        raise ValueError("Baseline terms must begin with intercept and exclude orbital additions")
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


def fit_models(event_features, integration_features, baseline_terms):
    """Fit the eight prespecified continuous models and ten nested comparisons."""
    specs = model_specs(baseline_terms)
    result = fit_named_models(event_features, integration_features, specs)
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


def invalid_tables(point, reason):
    """Keep unsupported draws in every comparison's denominator, without estimates."""
    models = point["models"].iloc[:0].reindex(range(len(point["models"])))
    comparisons = point["comparisons"].iloc[:0].reindex(range(len(point["comparisons"])))
    for column in ("model_id", "terms", "n_parameters"):
        models[column] = point["models"][column].to_numpy()
    for column in ("comparison_id", "driver_id", "comparison_group", "reduced_model_id",
                   "full_model_id", "df"):
        comparisons[column] = point["comparisons"][column].to_numpy()
    for frame in (models, comparisons):
        frame["fit_valid"] = False
        frame["invalid_reason"] = reason
    return {"models": models, "comparisons": comparisons}


def analyze_chronologies(events, windows, forcings, phase_anchors, scaling,
                         baseline_terms, selected, age_columns, *, tau=1.5,
                         initial_history=0.0, quadrature_order=4, show_progress=True):
    """Refit the same eight models on each draw, keeping nominal forcing scales."""
    event_x, integral_x = event_model.build_design(
        events, windows, forcings, phase_anchors, scaling, tau=tau,
        initial_history=initial_history, quadrature_order=quadrature_order,
        history_variants=True,
    )
    if "mis6_segment" in baseline_terms:
        for frame in (event_x, integral_x):
            frame["mis6_segment"] = frame.segment_id.eq("MIS6").astype(float)
    point = fit_models(event_x, integral_x, baseline_terms)
    if not point["models"].fit_valid.all() or not point["comparisons"].fit_valid.all():
        raise RuntimeError(f"Invalid point-age orbital fits:\n{point['models'].to_string(index=False)}")
    observations = windows[["segment_id", "observation_start_kyr_bp", "observation_end_kyr_bp"]]
    mc_models, mc_comparisons, status = [], [], []
    started = time.perf_counter()
    for index, row in selected.iterrows():
        shifted = events.copy()
        shifted["event_age_kyr_bp"] = row[age_columns].to_numpy(float)
        reason = ""
        try:
            draw_windows = event_model.response_windows(shifted, observations)
            draw_event_x, draw_integral_x = event_model.build_design(
                shifted, draw_windows, forcings, phase_anchors, scaling, tau=tau,
                initial_history=initial_history, quadrature_order=quadrature_order,
            )
            if "mis6_segment" in baseline_terms:
                for frame in (draw_event_x, draw_integral_x):
                    frame["mis6_segment"] = frame.segment_id.eq("MIS6").astype(float)
        except ValueError as error:
            reason = str(error)
            fitted = invalid_tables(point, reason)
        else:
            fitted = fit_models(draw_event_x, draw_integral_x, baseline_terms)
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
    return dict(events=event_model.mark_event_roles(events, windows), windows=windows,
        event_features=event_x, integration_features=integral_x,
        point_models=point["models"], point_comparisons=point["comparisons"], point_fits=point["fits"],
        selected_realizations=selected, realization_status=pd.DataFrame(status),
        mc_models=mc_models, mc_comparisons=mc_comparisons,
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
