#!/usr/bin/env python3
"""Continuous phase-null bootstrap for either Barker SpeleoAge event definition.

The common driver generates exact event times, conditions on the observed
oldest event, and refits both models. Each event definition has its own fitted
background model and response interval. Chronology is fixed in this experiment.
"""
import argparse
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
import NGRIP_MIS6_likelihood_bootstrap as bootstrap
from NGRIP_MIS6_likelihood_bootstrap import empirical_p_value, clopper_pearson_interval
from toolbox import combined_likelihood
from toolbox.project_config import PROJECT_ROOT, BARKER_EVENT_CSVS, CATALOGUE_COLORS

ROOT = PROJECT_ROOT / "Barker2011"
RUN_NAME = "Barker2011_likelihood_bootstrap"
OUT_DATA_DIR = ROOT / "data/processed" / RUN_NAME
OUT_FIG_DIR = ROOT / "figures" / RUN_NAME
DEFAULT_N_BOOTSTRAP = 9_999
DEFAULT_SEED = 20260909
DEFAULT_N_WORKERS = 3
EVENT_DEFINITIONS = ("variable_threshold", "fixed_threshold")

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


def run_analysis(*, n_bootstrap=DEFAULT_N_BOOTSTRAP, seed=DEFAULT_SEED,
                 n_workers=1, show_progress=False, event_definition="variable_threshold"):
    if event_definition not in EVENT_DEFINITIONS:
        raise ValueError(f"Unknown event definition: {event_definition}")
    events = pd.read_csv(BARKER_EVENT_CSVS[event_definition], float_precision="round_trip")
    context = combined_likelihood.build_barker_context(events, event_definition=event_definition)
    result = bootstrap.run_analysis(context=context, n_bootstrap=n_bootstrap, seed=seed,
                                    n_workers=n_workers, show_progress=show_progress)
    result["summary"]["event_definition"] = event_definition
    result["parameters"].loc[len(result["parameters"])] = ["event_definition", event_definition]
    result["parameters"].loc[len(result["parameters"])] = [
        "event_input_csv", str(BARKER_EVENT_CSVS[event_definition].relative_to(PROJECT_ROOT))]
    support = combined_likelihood.support_table(context).iloc[0]
    for name, value in dict(
        history_tau_kyr=context.history_tau_ka,
        initial_unobserved_history=context.initial_history,
        quadrature_order=context.quadrature_order,
        event_input_csv=str(BARKER_EVENT_CSVS[event_definition].relative_to(PROJECT_ROOT)),
        response_start_kyr_bp=support.response_start_kyr_bp,
        response_end_kyr_bp=support.response_end_kyr_bp,
    ).items():
        result["summary"][name] = value
    return result


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


def save_figure(replicates, summary, output_dir=OUT_FIG_DIR, *, paper_export=False):
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

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-bootstrap",type=int,default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--seed",type=int,default=DEFAULT_SEED)
    parser.add_argument("--workers",type=int,default=DEFAULT_N_WORKERS)
    parser.add_argument("--event-definition", choices=EVENT_DEFINITIONS, default="variable_threshold")
    parser.add_argument("--output-root",type=Path,default=PROJECT_ROOT)
    parser.add_argument("--no-paper-export",action="store_true")
    args=parser.parse_args(argv)
    result=run_analysis(n_bootstrap=args.n_bootstrap,seed=args.seed,n_workers=args.workers,
                        show_progress=True,event_definition=args.event_definition)
    data_dir, figure_dir = output_directories(args.output_root, args.event_definition)
    save_tables(result,data_dir)
    if result["summary"].iloc[0].n_failed_replicates:
        raise RuntimeError("Unresolved bootstrap fits saved; p value and figure publication withheld")
    save_figure(result["replicates"],result["summary"],figure_dir,
                paper_export=not args.no_paper_export and args.output_root.resolve()==PROJECT_ROOT.resolve())
    print(result["summary"][["LR_statistic","empirical_p_plus_one","n_failed_replicates"]].to_string(index=False))


if __name__=="__main__":
    main()
