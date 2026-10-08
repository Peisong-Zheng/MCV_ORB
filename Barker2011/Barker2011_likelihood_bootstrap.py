#!/usr/bin/env python3
"""Continuous phase-null bootstrap for either Barker SpeleoAge event definition.

The common driver generates exact event times, conditions on the observed
oldest event, and refits both models. Each event definition has its own fitted
background model and response interval. Chronology is fixed in this experiment.
"""
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mcv_orb_matplotlib"))
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2
from toolbox.project_config import PROJECT_ROOT, BARKER_EVENT_CSVS, CATALOGUE_COLORS

from toolbox import event_model, model_stats, sampling
from toolbox.point_process import fit_point_process
from toolbox.project_config import (MODEL_VERSION,
    LR04_CSV, CO2_CSV, ORBITAL_CSV, PRECESSION_PHASE_CSV)

RUN_NAME = "Barker2011_likelihood_bootstrap"
N_BOOTSTRAP = 9_999
RANDOM_SEED = 20260909
N_WORKERS = 3
OUTPUT_ROOT = PROJECT_ROOT
EVENT_DEFINITION = "variable_threshold"
EXPORT_PAPER = True

REPLICATE_COLUMNS = [
    "bootstrap_id", "n_events_response", "fit_valid", "status", "solver_attempts", "failure_reason",
    "loglik_reduced", "loglik_full", "LR_statistic", "quadrature_order",
]
SUMMARY_COLUMNS = [
    "model_version", "event_definition", "n_source_events", "n_response_events", "LR_statistic",
    "response_exposure_kyr", "n_bootstrap", "n_valid_replicates", "n_failed_replicates",
    "n_bootstrap_exceeding_or_equal_observed", "empirical_p_plus_one", "empirical_p_ci95_low",
    "empirical_p_ci95_high", "seed", "history_tau_kyr", "initial_unobserved_history", "quadrature_order",
    "event_input_csv", "response_start_kyr_bp", "response_end_kyr_bp", "n_zero_event_replicates",
    "n_same_data_refinements", "loglik_reduced", "loglik_full",
]


HISTORY_TERM = "same_type_exponential_history"

def run_analysis(*, n_bootstrap=N_BOOTSTRAP, seed=RANDOM_SEED,
                 n_workers=1, show_progress=False, event_definition="variable_threshold",
                 quadrature_order=4):
    events = pd.read_csv(BARKER_EVENT_CSVS[event_definition], float_precision="round_trip")
    events["segment_id"] = "Barker2011"
    observations = pd.DataFrame([dict(segment_id="Barker2011", observation_start_kyr_bp=0.,
                                      observation_end_kyr_bp=400.)])
    catalogue_id = f"barker_{event_definition}_speleo_0_400"
    lr04 = pd.read_csv(LR04_CSV, float_precision="round_trip")
    co2 = pd.read_csv(CO2_CSV, float_precision="round_trip")
    orbital = pd.read_csv(ORBITAL_CSV, float_precision="round_trip")
    anchors = pd.read_csv(PRECESSION_PHASE_CSV, float_precision="round_trip")
    forcings = {"lr04": (lr04.age_kyr_bp.to_numpy(), lr04.lr04.to_numpy()),
                "co2": (co2.age_kyr_bp.to_numpy(), co2.co2_ppm.to_numpy()),
                "precession_index": (orbital.age_kyr_bp.to_numpy(), orbital.precession_index.to_numpy())}
    phase_anchors = (anchors.age_kyr_bp.to_numpy(), anchors.phase_unwrapped_rad.to_numpy())
    windows = event_model.response_windows(events, observations)
    scaling = event_model.nominal_scaling({name: forcings[name] for name in ("lr04", "co2")}, windows)
    event_x, integral_x = event_model.build_design(events, windows, forcings, phase_anchors, scaling,
                                                  quadrature_order=quadrature_order)
    reduced_terms = ("intercept", HISTORY_TERM, "lr04_scaled", "co2_scaled")
    full_terms = reduced_terms + ("pre_phase_sin", "pre_phase_cos")
    reduced = fit_point_process(event_x[list(reduced_terms)], integral_x[list(reduced_terms)],
        integral_x.weight, reduced_terms, nonpositive_terms=(HISTORY_TERM,))
    full = fit_point_process(event_x[list(full_terms)], integral_x[list(full_terms)],
        integral_x.weight, full_terms, nonpositive_terms=(HISTORY_TERM,), start_beta=np.r_[reduced.beta, 0., 0.])
    point_summary = model_stats.fit_summary(reduced, full, event_x, windows,
        n_source_events=len(events), catalogue_id=catalogue_id)
    replicates, failures = sampling.phase_bootstrap(events, windows, forcings, phase_anchors,
        scaling, reduced, full, reduced_terms=reduced_terms, full_terms=full_terms,
        quadrature_order=quadrature_order, n_bootstrap=n_bootstrap, seed=seed,
        workers=n_workers, show_progress=show_progress)
    summary = model_stats.phase_bootstrap_summary(point_summary, replicates, failures)
    for name, value in dict(seed=seed, event_definition=event_definition, history_tau_kyr=1.5,
        initial_unobserved_history=0., quadrature_order=quadrature_order,
        event_input_csv=str(BARKER_EVENT_CSVS[event_definition].relative_to(PROJECT_ROOT)),
        response_start_kyr_bp=windows.response_start_kyr_bp.item(),
        response_end_kyr_bp=windows.response_end_kyr_bp.item()).items():
        summary[name] = value
    return dict(events=events, windows=windows, scaling=scaling, reduced=reduced, full=full,
        event_features=event_x, integration_features=integral_x, forcings=forcings,
        phase_anchors=phase_anchors, observations=observations, replicates=replicates,
        summary=summary, rejected_reasons=failures)


def output_directories(output_root, event_definition):
    """Keep the original variable-threshold paths and isolate fixed-threshold runs."""
    root = Path(output_root) / "Barker2011"
    data_dir = root / "data/processed" / RUN_NAME
    figure_dir = root / "figures" / RUN_NAME
    if event_definition == "fixed_threshold":
        data_dir = data_dir / "fixed_threshold"
        figure_dir = figure_dir / "fixed_threshold"
    return data_dir, figure_dir


def save_tables(result, output_dir):
    """Keep the null distribution, failed rows and calibration in two tables."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result["replicates"].reindex(columns=REPLICATE_COLUMNS).to_csv(
        output_dir / "bootstrap_replicates.csv", index=False, float_format="%.12g")
    result["summary"][SUMMARY_COLUMNS].to_csv(
        output_dir / "summary.csv", index=False, float_format="%.12g")


def save_figure(replicates, summary, output_dir, *, paper_export=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    fig = plot_null_distribution(replicates,summary)
    png,pdf = [output_dir/f"{RUN_NAME}.{suffix}" for suffix in ("png","pdf")]
    fig.savefig(png,dpi=450); fig.savefig(pdf)
    plt.close(fig)
    if paper_export:
        from Figure_likelihood_bootstrap import build_figure
        build_figure()
    return png,pdf


def plot_null_distribution(replicates, summary):
    """Show the fitted-reduced-model null and the observed phase improvement."""
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 8, "axes.linewidth": .7, "pdf.fonttype": 42})
    row = summary.iloc[0]
    fig, ax = plt.subplots(figsize=(110 / 25.4, 82 / 25.4))
    xmax = max(replicates.LR_statistic.max(), row.LR_statistic) * 1.06
    ax.hist(replicates.LR_statistic, bins=45, range=(0, xmax), density=True,
            color="#C4C4C4", edgecolor="white", linewidth=.4, label="Reduced-model simulations")
    x = np.linspace(0, xmax, 500)
    ax.plot(x, chi2.pdf(x, df=2), color="#555555", ls="--", lw=1.1,
            label=r"Asymptotic $\chi^2_2$")
    color = CATALOGUE_COLORS["fixed" if row.event_definition == "fixed_threshold" else "variable"]
    ax.axvline(row.LR_statistic, color=color, lw=1.6, label="Observed statistic")
    ax.text(.97, .68, f"Observed LR = {row.LR_statistic:.2f}\nBootstrap $p$ = {row.empirical_p_plus_one:.4f}\n"
            f"95% MC interval: {row.empirical_p_ci95_low:.4f}–{row.empirical_p_ci95_high:.4f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5)
    ax.set(xlabel="Likelihood-ratio statistic", ylabel="Probability density", xlim=(0, xmax))
    ax.legend(frameon=False, loc="upper right", fontsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=1.1)
    return fig

def main():
    result = run_analysis(n_bootstrap=N_BOOTSTRAP, seed=RANDOM_SEED, n_workers=N_WORKERS,
                          show_progress=True, event_definition=EVENT_DEFINITION)
    data_dir, figure_dir = output_directories(OUTPUT_ROOT, EVENT_DEFINITION)
    save_tables(result, data_dir)
    if result["summary"].iloc[0].n_failed_replicates:
        raise RuntimeError("Bootstrap has failed fits; see saved replicate results")
    save_figure(result["replicates"], result["summary"], figure_dir,
                paper_export=EXPORT_PAPER and OUTPUT_ROOT.resolve() == PROJECT_ROOT.resolve())
    print(result["summary"][["LR_statistic", "empirical_p_plus_one", "n_failed_replicates"]].to_string(index=False))


if __name__ == "__main__":
    main()
