#!/usr/bin/env python3
"""Continuous-time orbital-driver sensitivity on the primary Barker2011 catalogue.

Draw 500 chronologies without replacement from the complete age ensemble.
Each chronology supplies its own conditioning anchor and shared model support.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from toolbox import combined_likelihood, orbital_driver_sensitivity as orbital
from toolbox import orbital_driver_reporting as reporting
from toolbox.project_config import PROJECT_ROOT, BARKER_EVENT_CSVS

ROOT = PROJECT_ROOT / "Barker2011"
RUN_NAME = "Barker2011_orbital_driver_sensitivity"
CATALOGUE_LABEL = "Barker 2011 SpeleoAge warming events"
N_REALIZATIONS = 500
SELECTION_SEED = 20260909
AGE_INPUT = ROOT / "data/processed/Barker2011_event_age_uncertainty/event_age_realizations.csv"


def run_analysis(n_realizations=N_REALIZATIONS, show_progress=True, quadrature_order=4):
    events = pd.read_csv(BARKER_EVENT_CSVS["variable_threshold"], float_precision="round_trip")
    context = combined_likelihood.build_barker_context(events, quadrature_order=quadrature_order)
    context, _, _ = orbital.prepare_drivers(context)
    age_columns = [f"age_ka_bp__{event_id}" for event_id in context.events.event_id]
    draws = pd.read_csv(AGE_INPUT, float_precision="round_trip").sort_values("realization_id")
    if draws.realization_id.isna().any() or not draws.realization_id.is_unique:
        raise ValueError("Chronology IDs must be present and unique")
    if not isinstance(n_realizations, (int, np.integer)) or not 1 <= n_realizations <= len(draws):
        raise ValueError("Requested chronology count must lie within the complete ensemble")
    # Sorting IDs makes the draw independent of CSV row order. All models use this draw.
    indices = np.random.default_rng(SELECTION_SEED).choice(len(draws), n_realizations, replace=False)
    selected = draws.iloc[indices].reset_index(drop=True)
    ages = selected[age_columns].to_numpy(float)
    if not np.isfinite(ages).all() or np.any(np.diff(ages, axis=1) <= 0):
        raise ValueError("Chronologies must have finite, ordered event ages")
    result = orbital.analyze_chronologies(context, selected, age_columns, show_progress=show_progress)
    result.update(scaling=combined_likelihood.scaling_table(context)[["forcing_id", "mean", "range"]],
                  parameters=dict(
        model_version=combined_likelihood.MODEL_VERSION, selection_seed=SELECTION_SEED,
        history_tau_kyr=context.history_tau_ka, initial_history=context.initial_history,
        quadrature_order=context.quadrature_order))
    return result


def save_results(result, output_root):
    """Save model results and their scales; chronology IDs stay with the fits."""
    data_dir = output_root / "data/processed" / RUN_NAME
    data_dir.mkdir(parents=True, exist_ok=True)
    summary = result["comparison_summary"][[
        "comparison_id", "driver_id", "comparison_group", "reduced_model_id", "full_model_id",
        "df", "n_events", "exposure_kyr", "point_fit_valid", "point_invalid_reason",
        "holm_nominal_p_point", "n_mc_total", "n_mc_valid", "n_mc_invalid",
        "gain_bits_per_event_point", "gain_bits_per_event_q025", "gain_bits_per_event_median",
        "gain_bits_per_event_q975", "LR_statistic_point", "nominal_p_point", "delta_AIC_point",
    ]].assign(**result["parameters"])
    summary.to_csv(data_dir / "comparison_summary.csv", index=False)
    result["point"]["models"][[
        "model_id", "n_parameters", "log_likelihood", "AIC", "n_events", "exposure_kyr",
        "fit_valid", "invalid_reason",
    ]].to_csv(data_dir / "point_models.csv", index=False)
    result["point"]["coefficients"][["model_id", "term", "beta"]].to_csv(
        data_dir / "point_coefficients.csv", index=False)
    result["scaling"].to_csv(data_dir / "scaling.csv", index=False)
    result["mc_models"][[
        "realization_id", "model_id", "n_events", "exposure_kyr", "log_likelihood", "AIC",
        "pre_phase_preferred_deg", "pre_phase_rate_ratio_max_vs_min", "fit_valid", "invalid_reason",
    ]].to_csv(data_dir / "mc_models.csv", index=False)
    result["mc_comparisons"][[
        "realization_id", "comparison_id", "n_events", "exposure_kyr", "gain_bits_per_event",
        "LR_statistic", "nominal_p", "holm_nominal_p", "delta_AIC", "fit_valid", "invalid_reason",
    ]].to_csv(data_dir / "mc_comparisons.csv", index=False)
    reporting.redraw_saved_results(output_root, RUN_NAME, CATALOGUE_LABEL)
    return data_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--n-realizations", type=int, default=N_REALIZATIONS)
    parser.add_argument("--quadrature-order", type=int, default=4)
    parser.add_argument("--redraw", action="store_true", help="Plot the saved comparison summary only.")
    parser.add_argument("--no-paper-export", action="store_true",
                        help="This driver script only writes its own research artifacts.")
    args = parser.parse_args()
    if args.redraw:
        print(reporting.redraw_saved_results(args.output_root, RUN_NAME, CATALOGUE_LABEL))
        return
    result = run_analysis(args.n_realizations, quadrature_order=args.quadrature_order)
    data_dir = save_results(result, args.output_root)
    print(result["comparison_summary"].to_string(index=False))
    print(f"Saved {data_dir}")


if __name__ == "__main__":
    main()
