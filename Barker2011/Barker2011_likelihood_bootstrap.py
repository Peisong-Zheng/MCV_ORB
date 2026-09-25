#!/usr/bin/env python3
"""Continuous phase-null bootstrap for either Barker SpeleoAge event definition.

The common driver generates exact event times, conditions on the observed
oldest event, and refits both models. Each event definition has its own fitted
background model and response interval. Chronology is fixed in this experiment.
"""
import argparse
import hashlib
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
from toolbox.project_config import PROJECT_ROOT
from toolbox.catalogue_colors import CATALOGUE_COLORS

ROOT = PROJECT_ROOT / "Barker2011"
RUN_NAME = "Barker2011_likelihood_bootstrap"
OUT_DATA_DIR = ROOT / "data/processed" / RUN_NAME
OUT_FIG_DIR = ROOT / "figures" / RUN_NAME
NOTE_DIR = ROOT / "experiment_note"
DEFAULT_N_BOOTSTRAP = 9_999
DEFAULT_SEED = 20260909
DEFAULT_N_WORKERS = 3
EVENT_DEFINITIONS = ("variable_threshold", "fixed_threshold")


def run_analysis(*, n_bootstrap=DEFAULT_N_BOOTSTRAP, seed=DEFAULT_SEED,
                 n_workers=1, show_progress=False, event_definition="variable_threshold"):
    if event_definition not in EVENT_DEFINITIONS:
        raise ValueError(f"Unknown event definition: {event_definition}")
    context = combined_likelihood.build_barker_context(event_definition)
    result = bootstrap.run_analysis(context=context, n_bootstrap=n_bootstrap, seed=seed,
                                    n_workers=n_workers, show_progress=show_progress)
    result["summary"]["event_definition"] = event_definition
    result["parameters"].loc[len(result["parameters"])] = ["event_definition", event_definition]
    return result


def output_directories(output_root, event_definition):
    """Keep the original variable-threshold paths and isolate fixed-threshold runs."""
    root = Path(output_root) / "Barker2011"
    data_dir = root / "data/processed" / RUN_NAME
    figure_dir = root / "figures" / RUN_NAME
    if event_definition == "fixed_threshold":
        data_dir = data_dir / "fixed_threshold"
        figure_dir = figure_dir / "fixed_threshold"
    return data_dir, figure_dir, root / "experiment_note"


def save_tables(result, output_dir):
    bootstrap.save_tables(result, output_dir)
    path = Path(output_dir) / "input_code_sha256.csv"
    hashes = pd.read_csv(path)
    hashes.loc[len(hashes)] = [str(Path(__file__).relative_to(PROJECT_ROOT)),
                               hashlib.sha256(Path(__file__).read_bytes()).hexdigest()]
    hashes.to_csv(path,index=False)


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


def write_notes(result, notes_dir=NOTE_DIR):
    notes_dir = Path(notes_dir)
    notes_dir.mkdir(parents=True,exist_ok=True)
    s=result["summary"].iloc[0]
    definition = s.event_definition.replace("_", "-")
    note_name = RUN_NAME + ("_fixed_threshold" if s.event_definition == "fixed_threshold" else "")
    caption=f"""Continuous background-model bootstrap calibration of the precession-phase test in Barker et al. (2011) {definition} warming events on SpeleoAge. Gray bars show {s.n_bootstrap} simulated likelihood-ratio (LR) statistics, the colored line marks observed LR={s.LR_statistic:.4f}, and the dashed curve is the chi-square(2) asymptotic reference. Each simulation retains the exact oldest observed event and the fixed younger endpoint, generates continuous response events with dynamic inhibitory exponential history, and refits both models under beta_H <= 0. The plus-one bootstrap p is {s.empirical_p_plus_one:.6g}; the 95% Monte Carlo interval [{s.empirical_p_ci95_low:.6g}, {s.empirical_p_ci95_high:.6g}] is a Clopper-Pearson binomial interval for the null exceedance probability. Chronology is fixed.
"""
    methods=f"""BARKER CONTINUOUS PHASE NULL BOOTSTRAP
The {definition} catalogue contains {s.n_source_events} SpeleoAge warmings, of which the exact oldest event initializes history and {s.n_response_events} enter the event likelihood. The response exposure is {s.response_exposure_kyr:.9f} kyr. Its own background (reduced) model contains LR04, CO2 and inhibitory exponential event history (tau=1.5 kyr), plus an intercept. Full adds phase sine and cosine. Both use actual event log intensities minus integrated intensity. All younger exposure, including the event-free terminal tail, enters the integral. The unknown pre-anchor history is fixed to zero. SpeleoAge follows the main analysis's BP1950 convention.

Simulation uses continuous thinning from the fitted reduced model. Each accepted event immediately updates the exponential history; the total response event count varies. Both models are refitted on each new catalogue, with forcing standardization and support held fixed. Bootstrap calibration uses LR instead of G because response event counts vary. The plus-one p is (1 + count[LR_sim >= LR_observed])/(B+1); its exact binomial interval describes finite simulation precision, not effect uncertainty. Event ages are not binned. Numerical recovery uses the same events with refined integration; no failed catalogue is replaced. An all-empty response has LR=0 and undefined G. Any unresolved replicate prevents p-value and figure publication.

Observed G={s.gain_bits_per_event:.9f} bits/event; LR={s.LR_statistic:.9f}; nominal LR p={s.nominal_LR_p:.9g}; Delta AIC={s.delta_AIC_full_minus_reduced:.9f}. B={s.n_bootstrap}; exceedances={s.n_bootstrap_exceeding_or_equal_observed}; plus-one p={s.empirical_p_plus_one:.9g}; 95% Monte Carlo interval=[{s.empirical_p_ci95_low:.9g}, {s.empirical_p_ci95_high:.9g}]. Seed={s.seed}. Failed replicates={s.n_failed_replicates}; zero-response replicates={s.n_zero_event_replicates}; same-data numerical refinements={s.n_same_data_refinements}. Nominal chi-square LR p remains the standard model-comparison result; bootstrap checks sensitivity to that asymptotic reference for the phase test. This experiment is conditional on the {definition} point-age catalogue and its fitted null model. Age Monte Carlo is not nested, and fixed-threshold chronology uncertainty is not separately assessed.
"""
    (notes_dir/f"{note_name}_Caption.txt").write_text(caption)
    (notes_dir/f"{note_name}_Methods_and_results.txt").write_text(methods)

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
    data_dir, figure_dir, notes_dir = output_directories(args.output_root, args.event_definition)
    save_tables(result,data_dir)
    write_notes(result,notes_dir)
    if result["summary"].iloc[0].n_failed_replicates:
        raise RuntimeError("Unresolved bootstrap fits saved; p value and figure publication withheld")
    save_figure(result["replicates"],result["summary"],figure_dir,
                paper_export=not args.no_paper_export and args.output_root.resolve()==PROJECT_ROOT.resolve())
    print(result["summary"][["LR_statistic","empirical_p_plus_one","n_failed_replicates"]].to_string(index=False))


if __name__=="__main__":
    main()
