#!/usr/bin/env python3
"""Sensitivity of pooled warming-event PI to climate shape and event memory.

Climate alternatives reproduce the five polynomial-background checks in the
2026-09-05 audit on the main 180-kyr support. Memory alternatives replace the
1.5-kyr count with time since the last event, using identical 164.8-kyr support
for the count and elapsed-time fits. Each segment's oldest event bin supplies
initial history, contributes no response, and is followed only towards younger
ages. No unknown prehistory or NGRIP--MIS 6 gap is imputed.

This is a point-age specification sensitivity, with nominal chi-square LR p
values. It does not reuse the primary model's empirical bootstrap p value.
The new history follows the existing bin-center, strictly older-bin convention;
it is a discrete conditional-count model, not an exact continuous renewal model.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from paper_figure_export import copy_pdf_to_paper
import platform
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mcv_orb_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from toolbox import combined_pi
from toolbox.model_stats import nested_likelihood_metrics
from toolbox.project_config import CO2_XLSX, LR04_XLSX, PRE_TXT, PROJECT_ROOT


RUN_NAME = "NGRIP_MIS6_PI_model_sensitivity"
OUT_DATA_DIR = PROJECT_ROOT / "data/processed" / RUN_NAME
OUT_FIG_DIR = PROJECT_ROOT / "figures" / RUN_NAME
ELAPSED_TERM = "time_since_event_kyr"
LOG_ELAPSED_TERM = "log1p_time_since_event"
ELAPSED_SCALE_KYR = 1.0
PHASE_TERMS = ("pre_phase_sin", "pre_phase_cos")

# Post-audit sensitivity set; report every variant without selecting by p.
CLIMATE_VARIANTS = {
    "frozen_linear": (),
    "quadratic_lr04": ("lr04_squared",),
    "quadratic_co2": ("co2_squared",),
    "quadratic_both": ("lr04_squared", "co2_squared"),
    "quadratic_with_interaction": ("lr04_squared", "co2_squared", "lr04_co2"),
}
HISTORY_VARIANTS = {
    "count_matched_support": combined_pi.REDUCED_TERMS[0],
    "elapsed_time": ELAPSED_TERM,
    "log_elapsed_time": LOG_ELAPSED_TERM,
}
VARIANT_LABELS = {
    "frozen_linear": "Linear climate",
    "quadratic_lr04": "+ LR04²",
    "quadratic_co2": "+ CO₂²",
    "quadratic_both": "+ both squares",
    "quadratic_with_interaction": "+ squares + interaction",
    "count_matched_support": "Count in previous 1.5 kyr",
    "elapsed_time": "Time since last event",
    "log_elapsed_time": "log(1 + time / 1 kyr)",
}
VARIANT_COLORS = {
    "frozen_linear": "#666666",
    "quadratic_lr04": "#0072B2",
    "quadratic_co2": "#D55E00",
    "quadratic_both": "#009E73",
    "quadratic_with_interaction": "#CC79A7",
    "count_matched_support": "#666666",
    "elapsed_time": "#0072B2",
    "log_elapsed_time": "#D55E00",
}


def elapsed_since_event(counts, centers):
    """Return kyr since the nearest strictly older occupied bin, or NaN.

    Centers must increase in BP age. The current bin is evaluated before its
    count updates history, including when it contains multiple events. Only
    within-segment arrays may be passed; no last-event state survives a call.
    """

    counts = np.asarray(counts, dtype=float)
    centers = np.asarray(centers, dtype=float)
    if counts.ndim != 1 or centers.ndim != 1 or counts.shape != centers.shape:
        raise ValueError("Counts and centers must be matching one-dimensional arrays")
    if not np.isfinite(centers).all() or np.any(np.diff(centers) <= 0):
        raise ValueError("Bin centers must be finite and strictly increasing in BP age")
    if not np.isfinite(counts).all() or np.any(counts < 0) or np.any(counts != np.floor(counts)):
        raise ValueError("Counts must be finite non-negative integer values")

    elapsed = np.full(len(counts), np.nan)
    previous_event_center = None
    for index in range(len(counts) - 1, -1, -1):
        if previous_event_center is not None:
            elapsed[index] = previous_event_center - centers[index]
        if counts[index] > 0:
            previous_event_center = centers[index]
    return elapsed


def prepare_predictors(events, context):
    """Keep main-grid predictors and add inspectable memory support and roles."""

    bins = combined_pi.bin_catalogue(events, context)
    bins["lr04_squared"] = bins.lr04_scaled**2
    bins["co2_squared"] = bins.co2_scaled**2
    bins["lr04_co2"] = bins.lr04_scaled * bins.co2_scaled
    bins[ELAPSED_TERM] = np.nan
    bins["is_history_anchor"] = False
    bins["in_history_comparison"] = False
    support_rows = []
    event_frames = []

    for segment_id in combined_pi.SEGMENT_IDS:
        segment = context.segments[segment_id]
        frame = bins.loc[bins.segment_id.eq(segment_id)].copy()
        counts = frame.event_count.to_numpy(float)
        occupied = np.flatnonzero(counts > 0)
        if not len(occupied):
            raise ValueError(f"{segment_id}: no event is available to initialize history")
        anchor_index = int(occupied[-1])
        elapsed = elapsed_since_event(counts, frame.bin_center_kyr_bp.to_numpy(float))
        main_mask = frame.in_response_interval.to_numpy(bool)
        history_mask = main_mask & np.isfinite(elapsed)
        if not history_mask.any() or counts[history_mask].sum() == 0:
            raise ValueError(f"{segment_id}: no response events remain after initialization")
        bins.loc[frame.index, ELAPSED_TERM] = elapsed
        bins.loc[frame.index[anchor_index], "is_history_anchor"] = True
        bins.loc[frame.index, "in_history_comparison"] = history_mask

        # Match numpy.histogram, including its rightmost closed boundary.
        segment_events = events.loc[events.segment_id.eq(segment_id)].copy()
        event_bins = np.searchsorted(
            segment.bin_edges, segment_events[combined_pi.EVENT_AGE_COLUMN], side="right"
        ) - 1
        event_bins = np.minimum(event_bins, len(counts) - 1)
        segment_events["bin_center_kyr_bp"] = frame.bin_center_kyr_bp.to_numpy()[event_bins]
        segment_events["is_history_anchor"] = event_bins == anchor_index
        segment_events["in_main_response"] = main_mask[event_bins]
        segment_events["in_history_comparison"] = history_mask[event_bins]
        event_frames.append(segment_events)

        anchor = frame.iloc[anchor_index]
        response = frame.loc[history_mask]
        anchor_events = segment_events.loc[segment_events.is_history_anchor]
        support_rows.append({
            "segment_id": segment_id,
            "anchor_event_ids": ";".join(anchor_events.event_id),
            "anchor_event_ages_kyr_bp": ";".join(
                f"{age:.6f}" for age in anchor_events[combined_pi.EVENT_AGE_COLUMN]
            ),
            "anchor_bin_start_kyr_bp": anchor.bin_start_kyr_bp,
            "anchor_bin_end_kyr_bp": anchor.bin_end_kyr_bp,
            "anchor_bin_center_kyr_bp": anchor.bin_center_kyr_bp,
            "anchor_event_count": int(anchor.event_count),
            "main_response_start_kyr_bp": segment.response_start_kyr_bp,
            "main_response_end_kyr_bp": segment.response_end_kyr_bp,
            "history_response_start_kyr_bp": response.bin_start_kyr_bp.min(),
            "history_response_end_kyr_bp": response.bin_end_kyr_bp.max(),
            "main_response_exposure_kyr": frame.loc[main_mask, "dt_kyr"].sum(),
            "history_response_exposure_kyr": response.dt_kyr.sum(),
            "removed_response_exposure_kyr": frame.loc[main_mask & ~history_mask, "dt_kyr"].sum(),
            "n_main_response_events": int(counts[main_mask].sum()),
            "n_history_response_events": int(counts[history_mask].sum()),
            "n_removed_response_events": int(counts[main_mask & ~history_mask].sum()),
            "n_history_response_bins": int(history_mask.sum()),
        })

    bins[LOG_ELAPSED_TERM] = np.log1p(bins[ELAPSED_TERM] / ELAPSED_SCALE_KYR)
    # Equivalent to tau/(1 kyr); keeping the native-kyr column makes units visible.
    bins["history_role"] = np.select(
        [bins.in_history_comparison, bins.is_history_anchor, bins[ELAPSED_TERM].isna()],
        ["response", "conditioning_event_bin", "unknown_prior_event"],
        default="outside_main_response",
    )
    return bins, pd.DataFrame(support_rows), pd.concat(event_frames, ignore_index=True)


def fit_variant(response, terms, *, experiment, variant, support_id):
    """Fit a nested pair, requiring identifiable and interior numerical fits."""

    fits = {}
    coefficient_rows = []
    for model_id, model_terms in (("reduced", terms), ("full", terms + PHASE_TERMS)):
        design = np.column_stack([np.ones(len(response)), response.loc[:, model_terms]])
        if not np.isfinite(design).all():
            raise ValueError(f"{variant}/{model_id}: non-finite model predictors")
        if np.linalg.matrix_rank(design) != design.shape[1]:
            raise ValueError(f"{variant}/{model_id}: design matrix is rank deficient")
        model = combined_pi.fit_response_terms(response, model_terms)
        if not model.converged or not np.isfinite(model.beta).all():
            raise RuntimeError(f"{variant}/{model_id}: invalid fit: {model.optimizer_message}")
        if not np.isfinite([model.log_likelihood, model.aicc]).all():
            raise RuntimeError(f"{variant}/{model_id}: non-finite fit summary")
        if model.n_eta_clipped_low or model.n_eta_clipped_high:
            raise RuntimeError(f"{variant}/{model_id}: fitted linear predictor was clipped")
        # These are the optimizer bounds in toolbox.poisson, not scientific priors.
        lower_bounds = np.full(len(model.beta), -20.0)
        upper_bounds = np.array([5.0] + [20.0] * len(model_terms))
        if np.any(model.beta - lower_bounds < 1e-4) or np.any(upper_bounds - model.beta < 1e-4):
            raise RuntimeError(f"{variant}/{model_id}: coefficient reached optimizer bound")
        fits[model_id] = model
        for term, beta in zip(("intercept", *model_terms), model.beta):
            coefficient_rows.append({
                "experiment": experiment, "variant": variant, "support_id": support_id,
                "model_id": model_id, "term": term, "beta": float(beta),
            })

    reduced, full = fits["reduced"], fits["full"]
    metrics = nested_likelihood_metrics(
        loglik_full=full.log_likelihood, loglik_reduced=reduced.log_likelihood,
        df=2, n_bins=len(response), n_events=int(response.event_count.sum()),
        aicc_full=full.aicc, aicc_reduced=reduced.aicc,
    )
    if metrics["ll_gain_nats"] < -1e-7:
        raise RuntimeError(f"{variant}: fitted likelihood nesting failed")
    beta_sin, beta_cos = full.beta[-2:]
    row = {
        "experiment": experiment, "variant": variant,
        "variant_label": VARIANT_LABELS[variant], "support_id": support_id,
        "reduced_terms": ";".join(terms),
        "n_predictive_events": int(response.event_count.sum()),
        "n_predictive_bins": len(response),
        "response_exposure_kyr": float(response.dt_kyr.sum()),
        "n_params_reduced": len(reduced.beta), "n_params_full": len(full.beta),
        "loglik_reduced": reduced.log_likelihood, "loglik_full": full.log_likelihood,
        "LR_statistic": metrics["LR_statistic"],
        "nominal_LR_p": metrics["LR_p_value"],
        "info_bits_per_event": metrics["info_bits_per_event"],
        "pre_phase_preferred_deg": float(np.degrees(np.arctan2(beta_sin, beta_cos)) % 360),
        "pre_phase_rate_ratio_max_vs_min": float(np.exp(2 * np.hypot(beta_sin, beta_cos))),
        "reduced_aic": reduced.aic, "reduced_aicc": reduced.aicc,
        "full_aic": full.aic, "full_aicc": full.aicc,
        "delta_AICc_full_minus_reduced": metrics["delta_AICc_full_minus_reduced"],
        "all_models_converged": True, "likelihood_nesting_ok": True,
        "eta_clipping_used": False, "coefficient_bound_reached": False,
        "full_design_rank": int(design.shape[1]),
        "full_design_condition_number": float(np.linalg.cond(design)),
    }
    return row, coefficient_rows


def run_analysis():
    """Compute both sensitivity experiments without writing or changing inputs."""

    events = combined_pi.load_event_catalogue()
    context = combined_pi.build_context()
    binned, support, roles = prepare_predictors(events, context)
    main_response = binned.loc[binned.in_response_interval].copy()
    history_response = binned.loc[binned.in_history_comparison].copy()
    rows, coefficients = [], []

    for variant, extra_terms in CLIMATE_VARIANTS.items():
        row, model_coefficients = fit_variant(
            main_response, combined_pi.REDUCED_TERMS + extra_terms,
            experiment="climate", variant=variant, support_id="main",
        )
        rows.append(row)
        coefficients.extend(model_coefficients)
    for variant, history_term in HISTORY_VARIANTS.items():
        row, model_coefficients = fit_variant(
            history_response, (history_term, *combined_pi.REDUCED_TERMS[1:]),
            experiment="history", variant=variant, support_id="after_initial_event",
        )
        rows.append(row)
        coefficients.extend(model_coefficients)

    summary = pd.DataFrame(rows)
    for experiment, reference_id in (("climate", "frozen_linear"), ("history", "count_matched_support")):
        mask = summary.experiment.eq(experiment)
        reference_aicc = summary.loc[summary.variant.eq(reference_id), "full_aicc"].iloc[0]
        summary.loc[mask, "reference_variant"] = reference_id
        summary.loc[mask, "delta_AICc_full_vs_same_support_reference"] = (
            summary.loc[mask, "full_aicc"] - reference_aicc
        )
    coefficient_table = pd.DataFrame(coefficients)
    phase = np.arange(361, dtype=float)
    curve_frames = []
    for row in summary.itertuples(index=False):
        beta = coefficient_table.loc[
            coefficient_table.variant.eq(row.variant) & coefficient_table.model_id.eq("full")
        ].set_index("term").beta
        relative_rate = np.exp(
            beta[PHASE_TERMS[0]] * np.sin(np.deg2rad(phase))
            + beta[PHASE_TERMS[1]] * np.cos(np.deg2rad(phase))
        )
        curve_frames.append(pd.DataFrame({
            "experiment": row.experiment, "variant": row.variant,
            "phase_deg": phase, "relative_rate": relative_rate,
        }))
    return {
        "summary": summary, "coefficients": coefficient_table, "binned": binned,
        "segment_support": support, "event_roles": roles,
        "phase_curves": pd.concat(curve_frames, ignore_index=True),
    }


def plot_sensitivity(summary):
    """Compare point estimates within each experiment; no implied intervals."""

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
        "font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.8))
    fields = ("info_bits_per_event", "pre_phase_preferred_deg", "pre_phase_rate_ratio_max_vs_min")
    labels = ("PI (bits / event)", "Preferred phase (degrees)", "Phase max/min rate ratio")

    for row_index, experiment in enumerate(("climate", "history")):
        frame = summary.loc[summary.experiment.eq(experiment)].reset_index(drop=True)
        y = np.arange(len(frame))
        for column, (field, label) in enumerate(zip(fields, labels)):
            ax = axes[row_index, column]
            ax.axvline(frame[field].iloc[0], color="0.65", linewidth=1, linestyle="--", zorder=1)
            for index, row in frame.iterrows():
                ax.scatter(row[field], index, color=VARIANT_COLORS[row.variant], s=45, zorder=3)
            ax.set_yticks(y, frame.variant_label if column == 0 else [])
            ax.set_ylim(len(frame) - 0.5, -0.5)
            ax.set_xlabel(label)
            ax.grid(axis="y", color="0.92", linewidth=0.8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="y", length=0)
            ax.text(-0.04, 1.05, chr(97 + row_index * 3 + column),
                    transform=ax.transAxes, fontweight="bold")
            values = frame[field].to_numpy(float)
            if field == "info_bits_per_event":
                ax.set_xlim(0, max(0.21, values.max() * 1.15))
            elif field == "pre_phase_preferred_deg":
                margin = max(3, np.ptp(values) * 0.25)
                ax.set_xlim(values.min() - margin, values.max() + margin)
            else:
                ax.set_xlim(1, max(6, values.max() * 1.15))

    fig.subplots_adjust(left=0.22, right=0.98, top=0.89, bottom=0.14, hspace=0.65, wspace=0.3)
    for row_index, title in enumerate((
        "Climate shape · 55 response events · 180 kyr",
        "Event memory · 53 response events · 164.8 kyr · common support",
    )):
        box = axes[row_index, 0].get_position()
        fig.text(box.x0, box.y1 + 0.062, title, fontsize=11, fontweight="bold")
    fig.text(0.22, 0.035,
             "Points: fitted effects at point ages. Dashed lines: reference model on the same support.\n"
             "Each row has its own response support; uncertainty intervals are not shown.",
             fontsize=9, color="0.3")
    return fig


def save_results(result, output_dir=OUT_DATA_DIR, figure_dir=OUT_FIG_DIR):
    """Write dedicated results, exact model inputs and reproduction provenance."""

    output_dir, figure_dir = Path(output_dir), Path(figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    filenames = {
        "summary": "model_sensitivity.csv", "coefficients": "model_coefficients.csv",
        "binned": "binned_predictors_and_support.csv", "segment_support": "segment_support.csv",
        "event_roles": "event_roles.csv", "phase_curves": "phase_response_curves.csv",
    }
    for key, filename in filenames.items():
        result[key].to_csv(output_dir / filename, index=False)
    parameters = {
        "analysis": RUN_NAME, "age_epoch": "kyr BP relative to AD 1950",
        "estimand": "warming onset rate per unit of whole observation time",
        "chronology": "curated point ages only; no additional age MC",
        "history_window_kyr": combined_pi.DEFAULT_HISTORY_WINDOW_KA,
        "bin_width_kyr": combined_pi.DEFAULT_BIN_WIDTH_KA,
        "bin_origin_fraction": combined_pi.DEFAULT_ORIGIN_FRACTION,
        "climate_scaling": "original 180-kyr response mean/range, unchanged in all variants",
        "polynomial_transform": "square/product of original scaled climate predictors",
        "elapsed_time_definition": "nearest strictly older occupied bin center minus current bin center, within segment",
        "elapsed_time_scale_kyr": ELAPSED_SCALE_KYR,
        "log_elapsed_transform": "log(1 + time_since_event_kyr / 1 kyr)",
        "oldest_event_handling": "condition on oldest occupied bin; remove its and older response exposure; retain its history",
        "within_bin_history": "current-bin events excluded; event centers approximate timing at 0.2-kyr resolution",
        "phase_convention": "precession minimum=0 degrees; maximum=180 degrees; increases towards older BP age",
        "nominal_p_definition": "chi-square(2) survival of twice nested log-likelihood gain; unadjusted",
        "bootstrap_scope": "none here; primary empirical p=0.0022 does not calibrate these alternative models",
        "selection_scope": "post-audit sensitivity; all eight listed variants reported; no model selection by p",
        "aicc_n": "response bins, following main analysis; compare models only within identical support",
        "randomness": "none; deterministic point fits",
        "method_reference": "Truccolo et al. (2005), https://doi.org/10.1152/jn.00697.2004",
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "pandas_version": pd.__version__, "scipy_version": scipy.__version__,
        "matplotlib_version": matplotlib.__version__,
    }
    pd.DataFrame(parameters.items(), columns=["parameter", "value"]).to_csv(
        output_dir / "parameters_and_provenance.csv", index=False
    )
    sources = [
        Path(__file__).resolve(), combined_pi.EVENT_CATALOGUE_CSV,
        combined_pi.OBSERVATION_SEGMENTS_CSV, LR04_XLSX, CO2_XLSX, PRE_TXT,
        *[PROJECT_ROOT / "toolbox" / name for name in (
            "combined_pi.py", "poisson.py", "model_stats.py", "event_inputs.py",
            "event_process.py", "orbital_phase.py", "project_config.py",
        )],
    ]
    pd.DataFrame([
        {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in sources
    ]).to_csv(output_dir / "input_code_sha256.csv", index=False)
    figure = plot_sensitivity(result["summary"])
    for extension in ("png", "pdf"):
        figure.savefig(figure_dir / f"{RUN_NAME}.{extension}", dpi=300, facecolor="white")
    copy_pdf_to_paper(figure_dir / f"{RUN_NAME}.pdf")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DATA_DIR)
    parser.add_argument("--figure-dir", type=Path, default=OUT_FIG_DIR)
    args = parser.parse_args()
    result = run_analysis()
    save_results(result, args.output_dir, args.figure_dir)
    print(result["summary"].loc[:, [
        "variant", "n_predictive_events", "response_exposure_kyr", "info_bits_per_event",
        "nominal_LR_p", "pre_phase_preferred_deg", "pre_phase_rate_ratio_max_vs_min",
        "delta_AICc_full_vs_same_support_reference",
    ]].to_string(index=False))
    print(f"Saved results to {args.output_dir}")
    print(f"Saved figure to {args.figure_dir}")


if __name__ == "__main__":
    main()
