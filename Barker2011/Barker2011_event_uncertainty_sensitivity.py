#!/usr/bin/env python3
"""Continuous-time refits of Barker's saved SpeleoAge realizations."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from toolbox import event_model, model_stats, age_sensitivity
from toolbox.point_process import fit_point_process
from toolbox.project_config import (
    PROJECT_ROOT, EVENT_CATALOGUE_CSV, OBSERVATION_SEGMENTS_CSV, MODEL_VERSION,
    LR04_CSV, CO2_CSV, ORBITAL_CSV, PRECESSION_PHASE_CSV,
)
from toolbox.plotting import plot_sensitivity
from toolbox.project_config import BARKER_EVENT_CSVS
from paper_figure_export import copy_pdf_to_paper

ROOT = PROJECT_ROOT / "Barker2011"
RUN_NAME = "Barker2011_event_uncertainty_sensitivity"
AGE_INPUT = ROOT / "data/processed/Barker2011_event_age_uncertainty/event_age_realizations.csv"
HISTORY_TAU_KYR = 1.5
HISTORY_TERM = "same_type_exponential_history"
REDUCED_TERMS = ["intercept", HISTORY_TERM, "lr04_scaled", "co2_scaled"]
FULL_TERMS = REDUCED_TERMS + ["pre_phase_sin", "pre_phase_cos"]
CATALOGUE_ID = "barker_variable_threshold_speleo_0_400"
GAIN_COLUMNS = [
    "realization_id", "fit_valid", "invalid_reason", "n_response_events", "response_exposure_kyr",
    "gain_bits_per_event", "LR_statistic", "nominal_LR_p", "delta_AIC_full_minus_reduced",
    "pre_phase_preferred_deg", "pre_phase_rate_ratio_max_vs_min", "beta_pre_phase_sin", "beta_pre_phase_cos",
]
SUMMARY_COLUMNS = [
    "n_realizations", "n_valid", "n_invalid", "n_numerical_failures", "n_outside_observation_support",
    "n_nominal_p_below_0p05", "fraction_nominal_p_below_0p05", "model_version", "history_tau_kyr",
] + [
    name for metric in ("gain_bits_per_event", "nominal_LR_p", "pre_phase_preferred_deg",
                        "pre_phase_rate_ratio_max_vs_min")
    for name in (f"point_{metric}", f"{metric}_q025", f"{metric}_median", f"{metric}_q975")
]

OUTPUT_ROOT = PROJECT_ROOT
N_REALIZATIONS = 10_000
N_WORKERS = 1
REDRAW = False
EXPORT_PAPER = True


def main():
    if N_REALIZATIONS < 1:
        raise ValueError("n_realizations must be positive")
    events = pd.read_csv(BARKER_EVENT_CSVS["variable_threshold"], float_precision="round_trip")
    events["segment_id"] = "Barker2011"
    observations = pd.DataFrame([dict(segment_id="Barker2011", observation_start_kyr_bp=0.,
                                      observation_end_kyr_bp=400.)])
    lr04 = pd.read_csv(LR04_CSV, float_precision="round_trip")
    co2 = pd.read_csv(CO2_CSV, float_precision="round_trip")
    orbital = pd.read_csv(ORBITAL_CSV, float_precision="round_trip")
    anchors = pd.read_csv(PRECESSION_PHASE_CSV, float_precision="round_trip")
    forcings = {"lr04": (lr04.age_kyr_bp.to_numpy(), lr04.lr04.to_numpy()),
                "co2": (co2.age_kyr_bp.to_numpy(), co2.co2_ppm.to_numpy()),
                "precession_index": (orbital.age_kyr_bp.to_numpy(), orbital.precession_index.to_numpy())}
    phase_anchors = (anchors.age_kyr_bp.to_numpy(), anchors.phase_unwrapped_rad.to_numpy())
    windows = event_model.response_windows(events, observations)
    scaling = event_model.scale_forcing_v2({name: forcings[name] for name in ("lr04", "co2")}, windows)
    event_x, integral_x = event_model.build_likelihood_tables(
        events, windows, forcings, phase_anchors, scaling, tau=HISTORY_TAU_KYR,
    )
    reduced = fit_point_process(event_x[REDUCED_TERMS], integral_x[REDUCED_TERMS],
                                integral_x.weight, REDUCED_TERMS, nonpositive_terms=(HISTORY_TERM,))
    full = fit_point_process(event_x[FULL_TERMS], integral_x[FULL_TERMS],
                             integral_x.weight, FULL_TERMS, nonpositive_terms=(HISTORY_TERM,),
                             start_beta=np.r_[reduced.beta, 0., 0.])
    point = model_stats.fit_summary(reduced, full, event_x, windows, n_source_events=len(events),
                                    catalogue_id=CATALOGUE_ID, tau=HISTORY_TAU_KYR)
    root = OUTPUT_ROOT / "Barker2011"
    data = root / "data/processed" / RUN_NAME
    figures = root / "figures" / RUN_NAME
    for directory in (data, figures):
        directory.mkdir(parents=True, exist_ok=True)
    if REDRAW:
        results = pd.read_csv(data / "gain_realizations.csv", float_precision="round_trip")
    else:
        draws = pd.read_csv(AGE_INPUT, float_precision="round_trip").iloc[:N_REALIZATIONS]
        columns = [f"age_ka_bp__{event_id}" for event_id in events.event_id]
        results, diagnostics = age_sensitivity.fit_realizations(
            events, observations, forcings, phase_anchors, scaling, REDUCED_TERMS, FULL_TERMS,
            draws, columns, catalogue_id=CATALOGUE_ID, tau=HISTORY_TAU_KYR,
            n_workers=N_WORKERS, show_progress=True)
        results.reindex(columns=GAIN_COLUMNS).to_csv(data / "gain_realizations.csv", index=False)
        print(pd.Series(diagnostics).to_string())

    # Phase quantiles remain unwrapped about the nominal phase in the shared calculation.
    summary = age_sensitivity.summarize(results, point)
    reasons = results.invalid_reason.fillna("")
    summary["n_numerical_failures"] = int(reasons.str.startswith("numerical_fit").sum())
    summary["n_outside_observation_support"] = int(reasons.str.startswith("outside_").sum())
    summary["history_tau_kyr"] = HISTORY_TAU_KYR
    summary = summary[SUMMARY_COLUMNS]
    summary.to_csv(data / "summary.csv", index=False)
    if summary.n_numerical_failures.item():
        raise RuntimeError("Age ensemble has failed numerical fits; see saved results")
    fig = plot_sensitivity(results, point)
    for ext in ("pdf", "png"):
        fig.savefig(figures / f"{RUN_NAME}.{ext}", dpi=450)
    plt.close(fig)
    if EXPORT_PAPER and OUTPUT_ROOT.resolve() == PROJECT_ROOT.resolve():
        copy_pdf_to_paper(figures / f"{RUN_NAME}.pdf")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
