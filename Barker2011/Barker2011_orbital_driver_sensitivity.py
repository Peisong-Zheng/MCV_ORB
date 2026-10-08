#!/usr/bin/env python3
"""Continuous-time orbital-driver sensitivity on the primary Barker2011 catalogue.

Draw 500 chronologies without replacement from the complete age ensemble.
Each chronology supplies its own conditioning anchor and shared model support.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from toolbox import event_model, orbital_driver_sensitivity as orbital
from toolbox.plotting import plot_orbital_comparisons
from toolbox.project_config import (
    PROJECT_ROOT, LR04_CSV, CO2_CSV, ORBITAL_CSV, INSOLATION_65N_CSV,
    PRECESSION_PHASE_CSV,
    MODEL_VERSION, BARKER_EVENT_CSVS,
)

ROOT = PROJECT_ROOT / "Barker2011"
RUN_NAME = "Barker2011_orbital_driver_sensitivity"
CATALOGUE_LABEL = "Barker 2011 SpeleoAge warming events"
N_REALIZATIONS = 500
SELECTION_SEED = 20260909
AGE_INPUT = ROOT / "data/processed/Barker2011_event_age_uncertainty/event_age_realizations.csv"

# This entry uses the Barker directory itself as the output root.
OUTPUT_ROOT = ROOT
QUADRATURE_ORDER = 4
REDRAW = False


def run_analysis(n_realizations=N_REALIZATIONS, show_progress=True, quadrature_order=4):
    events = pd.read_csv(BARKER_EVENT_CSVS["variable_threshold"], float_precision="round_trip")
    events["segment_id"] = "Barker2011"
    observations = pd.DataFrame([dict(segment_id="Barker2011",
        observation_start_kyr_bp=0., observation_end_kyr_bp=400.)])
    lr04 = pd.read_csv(LR04_CSV, float_precision="round_trip")
    co2 = pd.read_csv(CO2_CSV, float_precision="round_trip")
    orbital_data = pd.read_csv(ORBITAL_CSV, float_precision="round_trip")
    insolation = pd.read_csv(INSOLATION_65N_CSV, float_precision="round_trip")
    anchors = pd.read_csv(PRECESSION_PHASE_CSV, float_precision="round_trip")
    forcings = {
        "lr04": (lr04.age_kyr_bp.to_numpy(), lr04.lr04.to_numpy()),
        "co2": (co2.age_kyr_bp.to_numpy(), co2.co2_ppm.to_numpy()),
        "precession_index": (orbital_data.age_kyr_bp.to_numpy(), orbital_data.precession_index.to_numpy()),
        "ecc": (orbital_data.age_kyr_bp.to_numpy(), orbital_data.eccentricity.to_numpy()),
        "obl": (orbital_data.age_kyr_bp.to_numpy(), orbital_data.obliquity_deg.to_numpy()),
        "insol65n": (insolation.age_kyr_bp.to_numpy(), insolation.insolation_Wm2.to_numpy()),
    }
    phase_anchors = (anchors.age_kyr_bp.to_numpy(), anchors.phase_unwrapped_rad.to_numpy())
    windows = event_model.response_windows(events, observations)
    scaling = event_model.nominal_scaling(
        {name: source for name, source in forcings.items() if name != "precession_index"}, windows)
    baseline = ("intercept", "same_type_exponential_history", "lr04_scaled", "co2_scaled")
    age_columns = [f"age_ka_bp__{event_id}" for event_id in events.event_id]
    draws = pd.read_csv(AGE_INPUT, float_precision="round_trip").sort_values("realization_id")
    if draws.realization_id.isna().any() or not draws.realization_id.is_unique:
        raise ValueError("Chronology IDs must be present and unique")
    if not 1 <= n_realizations <= len(draws):
        raise ValueError("Requested chronology count must lie within the complete ensemble")
    # Sorting IDs makes the draw independent of CSV row order. All models use this draw.
    indices = np.random.default_rng(SELECTION_SEED).choice(len(draws), n_realizations, replace=False)
    selected = draws.iloc[indices].reset_index(drop=True)
    ages = selected[age_columns].to_numpy(float)
    if not np.isfinite(ages).all() or np.any(np.diff(ages, axis=1) <= 0):
        raise ValueError("Chronologies must have finite, ordered event ages")
    result = orbital.analyze_chronologies(
        events, windows, forcings, phase_anchors, scaling, baseline, selected, age_columns,
        quadrature_order=quadrature_order, show_progress=show_progress)
    result.update(scaling=scaling.reset_index()[["forcing_id", "mean", "range"]],
                  parameters=dict(
        model_version=MODEL_VERSION, selection_seed=SELECTION_SEED,
        history_tau_kyr=1.5, initial_history=0.0,
        quadrature_order=quadrature_order))
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
    result["point_models"][[
        "model_id", "n_parameters", "log_likelihood", "AIC", "n_events", "exposure_kyr",
        "fit_valid", "invalid_reason",
    ]].to_csv(data_dir / "point_models.csv", index=False)
    coefficients = pd.DataFrame([
        dict(model_id=model_id, term=term, beta=beta)
        for model_id, fitted in result["point_fits"].items()
        for term, beta in zip(fitted.terms, fitted.beta)
    ])
    coefficients.to_csv(data_dir / "point_coefficients.csv", index=False)
    result["scaling"].to_csv(data_dir / "scaling.csv", index=False)
    result["mc_models"][[
        "realization_id", "model_id", "n_events", "exposure_kyr", "log_likelihood", "AIC",
        "pre_phase_preferred_deg", "pre_phase_rate_ratio_max_vs_min", "fit_valid", "invalid_reason",
    ]].to_csv(data_dir / "mc_models.csv", index=False)
    result["mc_comparisons"][[
        "realization_id", "comparison_id", "n_events", "exposure_kyr", "gain_bits_per_event",
        "LR_statistic", "nominal_p", "holm_nominal_p", "delta_AIC", "fit_valid", "invalid_reason",
    ]].to_csv(data_dir / "mc_comparisons.csv", index=False)
    redraw_saved_results(output_root)
    return data_dir

def redraw_saved_results(output_root):
    """Plot saved comparisons without refitting or reading analysis inputs."""
    run_name, catalogue_label = RUN_NAME, CATALOGUE_LABEL
    data_dir = output_root / "data/processed" / run_name
    figure_dir = output_root / "figures" / run_name
    summary = pd.read_csv(data_dir / "comparison_summary.csv")
    fig, _ = plot_orbital_comparisons(summary, catalogue_label)
    figure_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(figure_dir / f"{run_name}.{extension}", dpi=600)
    plt.close(fig)
    return figure_dir


def main():
    if REDRAW:
        print(redraw_saved_results(OUTPUT_ROOT))
        return
    result = run_analysis(N_REALIZATIONS, quadrature_order=QUADRATURE_ORDER)
    data_dir = save_results(result, OUTPUT_ROOT)
    print(result["comparison_summary"].to_string(index=False))
    print(f"Saved {data_dir}")


if __name__ == "__main__":
    main()
