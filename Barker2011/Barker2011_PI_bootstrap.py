#!/usr/bin/env python3
"""Reduced-model bootstrap for Barker's variable-threshold SpeleoAge events.

Simulate the fitted history + LR04 + CO2 model from old to young, then refit
it and the model with precession phase. The event total is allowed to vary;
the likelihood-ratio statistic, rather than PI per event, calibrates the test.
This experiment uses point ages and does not nest chronology Monte Carlo.
"""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
import time

# Many small fits run faster with one linear-algebra thread per worker.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mcv_orb_matplotlib"))
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_figure_export import copy_pdf_to_paper
from Barker2011 import Barker2011_event_phase_analysis as main_analysis
from Barker2011 import Barker2011_event_uncertainty_sensitivity as age_sensitivity
from NGRIP_MIS6_PI_bootstrap import empirical_p_value, clopper_pearson_interval
from toolbox import combined_pi, poisson

ROOT = main_analysis.ROOT
RUN_NAME = "Barker2011_PI_bootstrap"
OUT_DATA_DIR = ROOT / "data/processed" / RUN_NAME
OUT_FIG_DIR = ROOT / "figures" / RUN_NAME
NOTE_DIR = ROOT / "experiment_note"
DEFAULT_N_BOOTSTRAP = 9_999
DEFAULT_SEED = 20260909
DEFAULT_MAX_ATTEMPTS = 20
DEFAULT_N_WORKERS = min(6, max(1, (os.cpu_count() or 2) - 1))
NESTING_TOLERANCE = 1e-7
_WORKER_STATE = None


class InvalidBootstrapFit(RuntimeError):
    """A numerical simulation or fit failure that can be logged and redrawn."""


def check_models(models):
    """Require finite, converged and nested fits without numerical clipping."""
    reduced, full = models
    if not np.isfinite(np.r_[reduced.beta, full.beta,
                            reduced.log_likelihood, full.log_likelihood]).all():
        raise InvalidBootstrapFit("non-finite coefficients or likelihood")
    if not all(model.converged for model in models):
        raise InvalidBootstrapFit("optimizer did not converge")
    if any(model.n_eta_clipped_low or model.n_eta_clipped_high for model in models):
        raise InvalidBootstrapFit("fitted linear predictor required clipping")
    if full.log_likelihood < reduced.log_likelihood - NESTING_TOLERANCE:
        raise InvalidBootstrapFit("full likelihood below reduced likelihood")


def prepare_simulation(context, reduced_beta):
    """Calculate the fixed part of the Barker null rate once per experiment."""
    terms = main_analysis.REDUCED_TERMS
    beta = np.asarray(reduced_beta, dtype=float)
    if beta.shape != (1 + len(terms),) or not np.isfinite(beta).all():
        raise ValueError("Expected all finite Barker reduced-model coefficients")
    history_index = terms.index("same_type_history_count")
    fixed_indices = [i for i in range(len(terms)) if i != history_index]
    bins, segment = context["bins"], context["segment"]
    dt = bins.dt_ka.to_numpy(float)
    fixed = bins[[terms[i] for i in fixed_indices]].to_numpy(float)
    if (not np.isfinite(fixed).all() or not np.all(np.isfinite(dt) & (dt > 0))
            or not np.allclose(dt, np.diff(segment.bin_edges), rtol=0, atol=1e-10)):
        raise ValueError("Invalid simulation predictors or bin exposure")
    return dict(segment=segment, dt=dt,
                eta_fixed=beta[0] + fixed @ beta[1 + np.asarray(fixed_indices)],
                history_beta=beta[1 + history_index])


def simulate_counts(prepared, rng):
    """Draw oldest to youngest, including the history-only 398.5–400-kyr buffer."""
    segment = prepared["segment"]
    counts = np.zeros(len(prepared["dt"]), dtype=int)
    history_used = np.zeros(len(counts))
    # Larger BP ages are earlier. Outside the oldest observation edge, history is zero.
    for i in range(len(counts) - 1, -1, -1):
        history = counts[segment.history_left[i]:segment.history_right[i]].sum()
        history_used[i] = history
        eta = prepared["eta_fixed"][i] + prepared["history_beta"] * history
        if not np.isfinite(eta) or eta < poisson.ETA_MIN or eta > poisson.ETA_MAX:
            raise InvalidBootstrapFit("simulated linear predictor outside numerical limits")
        counts[i] = rng.poisson(prepared["dt"][i] * np.exp(eta))
    if not np.array_equal(history_used, combined_pi.history_from_counts(counts, segment)):
        raise RuntimeError("Simulation and fitting history rules disagree")
    return counts


def fit_counts(counts, context):
    """Refit both models after replacing response counts and their event histories."""
    segment = context["segment"]
    counts = np.asarray(counts)
    if (counts.shape != (len(segment.bin_edges) - 1,) or not np.isfinite(counts).all()
            or np.any(counts < 0) or np.any(counts != np.floor(counts))):
        raise ValueError("Expected one non-negative integer count per observation bin")
    frame = context["frame"].copy()
    frame["event_count"] = counts[segment.response_mask]
    frame["same_type_history_count"] = combined_pi.history_from_counts(
        counts, segment)[segment.response_mask]
    if frame.event_count.sum() == 0:
        raise InvalidBootstrapFit("no response events")
    models, tests = main_analysis.fit_models(frame)
    check_models(models)
    reduced, full = models
    return dict(n_events_observation_support=int(counts.sum()),
                n_events_response=int(frame.event_count.sum()),
                loglik_reduced=reduced.log_likelihood, loglik_full=full.log_likelihood,
                ll_gain_nats=full.log_likelihood - reduced.log_likelihood,
                LR_statistic=2 * (full.log_likelihood - reduced.log_likelihood),
                info_bits_per_event=float(tests.iloc[0].info_bits_per_event),
                reduced_converged=reduced.converged, full_converged=full.converged,
                likelihood_nesting_ok=True, eta_clipping_used=False)


def draw_replicate(task, context, prepared, max_attempts):
    """Keep a separate seed stream per replicate and numerical retry."""
    index, replicate_seed = task
    rejections = Counter()
    for attempt, seed in enumerate(replicate_seed.spawn(max_attempts), start=1):
        try:
            counts = simulate_counts(prepared, np.random.default_rng(seed))
            row = fit_counts(counts, context)
        except InvalidBootstrapFit as error:
            rejections[str(error)] += 1
            continue
        return dict(bootstrap_id=index, attempt_count=attempt, **row), rejections
    raise RuntimeError(f"Bootstrap {index} failed after {max_attempts} attempts: {dict(rejections)}")


def initialize_worker(context, prepared, max_attempts):
    global _WORKER_STATE
    _WORKER_STATE = context, prepared, max_attempts


def worker_replicate(task):
    return draw_replicate(task, *_WORKER_STATE)


def run_bootstrap(context, reduced_beta, *, n_bootstrap=DEFAULT_N_BOOTSTRAP,
                  seed=DEFAULT_SEED, max_attempts=DEFAULT_MAX_ATTEMPTS,
                  n_workers=1, show_progress=False):
    """Generate valid null replicates with results independent of worker scheduling."""
    if min(n_bootstrap, max_attempts, n_workers) < 1:
        raise ValueError("Replicate, attempt and worker counts must be positive")
    prepared = prepare_simulation(context, reduced_beta)
    tasks = list(enumerate(np.random.SeedSequence(seed).spawn(n_bootstrap), start=1))
    rows, rejections = [], Counter()
    started = time.perf_counter()
    executor = None
    if n_workers == 1:
        results = (draw_replicate(task, context, prepared, max_attempts) for task in tasks)
    else:
        executor = ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("spawn"),
                                       initializer=initialize_worker,
                                       initargs=(context, prepared, max_attempts))
        results = executor.map(worker_replicate, tasks, chunksize=10)
    try:
        for index, (row, rejected) in enumerate(results, start=1):
            rows.append(row)
            rejections.update(rejected)
            if show_progress and (index % max(1, n_bootstrap // 10) == 0 or index == n_bootstrap):
                print(f"Bootstrap {index:,}/{n_bootstrap:,} ({time.perf_counter() - started:.1f} s)",
                      flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    return pd.DataFrame(rows), rejections


def run_analysis(*, n_bootstrap=DEFAULT_N_BOOTSTRAP, seed=DEFAULT_SEED,
                 max_attempts=DEFAULT_MAX_ATTEMPTS, n_workers=1, show_progress=False):
    """Fit the primary point-age catalogue and calibrate its two-term phase test."""
    events = main_analysis.load_barker_source()
    if not events.event_definition.eq("variable_threshold").all():
        raise ValueError("This bootstrap is restricted to the primary variable-threshold catalogue")
    context = age_sensitivity.prepare_context(events)
    models, tests = main_analysis.fit_models(context["frame"])
    check_models(models)
    observed_counts, _ = np.histogram(events.event_age_ka, bins=context["segment"].bin_edges)
    observed = fit_counts(observed_counts, context)
    replicates, rejections = run_bootstrap(context, models[0].beta,
        n_bootstrap=n_bootstrap, seed=seed, max_attempts=max_attempts,
        n_workers=n_workers, show_progress=show_progress)
    p_value, exceedances = empirical_p_value(replicates.LR_statistic.to_numpy(), observed["LR_statistic"])
    ci_low, ci_high = clopper_pearson_interval(exceedances, n_bootstrap)
    summary = pd.DataFrame([dict(
        dataset_id=main_analysis.DATASET_ID, event_definition="variable_threshold", **observed,
        nominal_LR_p=float(tests.iloc[0].LR_p_value), n_bootstrap=n_bootstrap,
        n_bootstrap_exceeding_or_equal_observed=exceedances, empirical_p_plus_one=p_value,
        empirical_p_mc_se=np.sqrt(p_value * (1 - p_value) / (n_bootstrap + 1)),
        empirical_p_ci95_low=ci_low, empirical_p_ci95_high=ci_high,
        empirical_p_ci_method="Clopper-Pearson interval for null exceedance probability",
        bootstrap_LR_mean=replicates.LR_statistic.mean(),
        bootstrap_LR_q95=replicates.LR_statistic.quantile(.95),
        bootstrap_LR_q99=replicates.LR_statistic.quantile(.99),
        bootstrap_response_event_count_mean=replicates.n_events_response.mean(),
        bootstrap_response_event_count_q025=replicates.n_events_response.quantile(.025),
        bootstrap_response_event_count_q975=replicates.n_events_response.quantile(.975),
        n_rejected_attempts=sum(rejections.values()), seed=seed,
        n_response_bins=len(context["frame"]), response_exposure_kyr=context["frame"].dt_ka.sum(),
        age_uncertainty_propagated=False)])
    parameters = dict(event_catalogue="Table S3 variable-threshold SpeleoAge warming picks",
        event_age_epoch="BP1950 assumed; exact SpeleoAge epoch not directly verified",
        observation_start_kyr=main_analysis.ANALYSIS_START_KA,
        observation_end_kyr=main_analysis.ANALYSIS_END_KA,
        response_start_kyr=context["segment"].response_start_kyr_bp,
        response_end_kyr=context["segment"].response_end_kyr_bp,
        bin_width_kyr=main_analysis.BIN_WIDTH_KA, history_window_kyr=main_analysis.HISTORY_WINDOW_KA,
        bin_origin_fraction=main_analysis.BIN_ORIGIN_FRACTION, response_mode=main_analysis.RESPONSE_MODE,
        reduced_terms="+".join(main_analysis.REDUCED_TERMS), full_terms="+".join(main_analysis.FULL_TERMS),
        simulation_direction="oldest to youngest", oldest_boundary_history="zero outside observation support",
        simulation_support="full observation interval, including history-only buffer",
        refit_history="recalculated from simulated counts", conditioned_event_total=False,
        chronology_MC_nested=False, fixed_threshold_included=False,
        n_bootstrap=n_bootstrap, seed=seed, max_attempts=max_attempts, n_workers=n_workers,
        empirical_p_rule="(1 + number of simulated LR >= observed LR) / (B + 1)",
        numerical_retries="only nonconverged, nonfinite, nonnested, clipped or empty-response draws")
    return dict(events=events, context=context, models=models, replicates=replicates,
                summary=summary, parameters=parameters, rejections=rejections)


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
    ax.axvline(row.LR_statistic, color="#CC6677", lw=1.6, label="Observed statistic")
    ax.text(.97, .68, f"Observed LR = {row.LR_statistic:.2f}\nBootstrap $p$ = {row.empirical_p_plus_one:.4f}\n"
            f"95% simulation interval: {row.empirical_p_ci95_low:.4f}–{row.empirical_p_ci95_high:.4f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5)
    ax.set(xlabel="Likelihood-ratio statistic", ylabel="Probability density", xlim=(0, xmax))
    ax.set_title("Barker 2011 · variable threshold", loc="left", fontsize=9, pad=9)
    ax.legend(frameon=False, loc="upper right", fontsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=1.1)
    return fig


def write_notes(result):
    """Save the simulation assumptions and quantitative results for the manuscript."""
    row = result["summary"].iloc[0]
    text = f"""Barker 2011 reduced-model parametric bootstrap

Methods
The primary catalogue comprises {len(result['events'])} Table S3 variable-threshold warming
events on the SpeleoAge chronology of Barker et al. (2011). The fixed-threshold
catalogue is reserved for the separate point-age event-definition sensitivity.
The observed comparison uses an intercept, prior {main_analysis.HISTORY_WINDOW_KA:g}-kyr warming-event count,
LR04 benthic oxygen isotopes and atmospheric CO2 in the reduced model, and adds
the sine and cosine of precession phase in the full model. Grid width is
{main_analysis.BIN_WIDTH_KA:g} kyr. The observation interval is 0–400 kyr BP; the response interval
is 0–398.5 kyr BP ({int(row.n_response_bins):,} bins and {row.response_exposure_kyr:g} kyr exposure).
The exact 398.5-kyr boundary is inserted in the grid and individual bin durations
enter the Poisson means. Climate standardization is fixed at its observed
response-support values. SpeleoAge numerical values are treated as BP1950,
following the main analysis; its exact original reference year remains unverified.

Each null catalogue is generated under the reduced-model maximum-likelihood fit.
Bins are simulated from oldest to youngest with Poisson means dt * exp(eta).
History counts simulated events in older bin centers within (t, t + 1.5 kyr],
excluding the current bin. The 398.5–400-kyr history buffer is simulated too;
history outside 400 kyr is initialized to zero. Counts and histories are rebuilt
for every response fit, while forcing values and response exposure remain fixed.
The total simulated event count varies. Both models are refitted and compared
using LR = 2 * (log likelihood_full - log likelihood_reduced), rather than PI
per event, because the event-count denominator varies between null catalogues.

The run retains {int(row.n_bootstrap):,} numerically valid catalogues, with base seed
{int(row.seed)} and independent NumPy SeedSequence streams per replicate and retry.
The empirical p value is (1 + count[LR_sim >= LR_observed]) / (B + 1).
The exact Clopper-Pearson interval estimates the null exceedance probability;
it describes finite simulation precision, not an effect confidence interval.
Nonconvergence, nonfinite results, likelihood-nesting failures, numerical
clipping and empty responses trigger logged redraws (at most {result['parameters']['max_attempts']} attempts
per replicate). This calibration is conditional on the selected point-age
catalogue and fitted null model. It does not propagate chronology uncertainty
or replace the separate chronology Monte Carlo sensitivity analysis.

Results
Observed LR = {row.LR_statistic:.9f}; nominal asymptotic p = {row.nominal_LR_p:.9g}.
Observed PI = {row.info_bits_per_event:.9f} bits per event, for
{int(row.n_events_response)} response events. There are
{int(row.n_bootstrap_exceeding_or_equal_observed):,}/{int(row.n_bootstrap):,} null statistics at least as large
as the observed statistic. The plus-one bootstrap p = {row.empirical_p_plus_one:.9g};
95% binomial interval = [{row.empirical_p_ci95_low:.9g}, {row.empirical_p_ci95_high:.9g}],
with approximate Monte Carlo standard error {row.empirical_p_mc_se:.9g}.
The null LR mean is {row.bootstrap_LR_mean:.6f}, its 95th percentile is
{row.bootstrap_LR_q95:.6f}, and its 99th percentile is {row.bootstrap_LR_q99:.6f}.
The mean simulated response count is {row.bootstrap_response_event_count_mean:.3f};
the central 95% count range is [{row.bootstrap_response_event_count_q025:g},
{row.bootstrap_response_event_count_q975:g}]. Rejected numerical attempts: {int(row.n_rejected_attempts)}.
These results assess whether precession improves the fitted event-rate model
under the specified null, and do not establish a causal mechanism or
independence from other reconstructed climate records.
"""
    caption = f"""Reduced-model bootstrap calibration of precession information in Barker et al. (2011) variable-threshold warming events on the SpeleoAge chronology. Gray bars show the density of {int(row.n_bootstrap):,} likelihood-ratio (LR) statistics from simulated catalogues under the fitted reduced model (prior 1.5-kyr event count, LR04 benthic oxygen isotopes and atmospheric CO2, plus an intercept). Both the reduced model and the full model, which adds precession-phase sine and cosine, are refitted to each simulated catalogue. Event history is updated dynamically from older to younger bins. The colored vertical line marks the observed LR = {row.LR_statistic:.2f}; the dashed curve is the nominal chi-square reference with two degrees of freedom. The plus-one bootstrap p is {row.empirical_p_plus_one:.4f}, based on {int(row.n_bootstrap_exceeding_or_equal_observed):,} exceedances. The displayed 95% simulation interval is the exact binomial interval for the null exceedance probability, describing finite bootstrap precision rather than event-age uncertainty or an effect confidence interval. Simulations retain the 0–400-kyr observation interval and 0–398.5-kyr response support, use the primary point-age catalogue, and allow the event total to vary. Fixed-threshold events and chronology Monte Carlo are not included in this calibration.
"""
    NOTE_DIR.mkdir(exist_ok=True)
    (NOTE_DIR / f"{RUN_NAME}_Methods_and_results.txt").write_text(text, encoding="utf-8")
    (NOTE_DIR / f"{RUN_NAME}_Caption.txt").write_text(caption, encoding="utf-8")


def save_results(result):
    OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for key, name in (("replicates", "bootstrap_replicates"), ("summary", "summary")):
        result[key].to_csv(OUT_DATA_DIR / f"{name}.csv", index=False)
    pd.DataFrame(result["parameters"].items(), columns=["parameter", "value"]).to_csv(
        OUT_DATA_DIR / "parameters_and_provenance.csv", index=False)
    pd.DataFrame(result["rejections"].items(), columns=["reason", "n_rejected_attempts"]).to_csv(
        OUT_DATA_DIR / "rejected_attempts.csv", index=False)
    result["events"].to_csv(OUT_DATA_DIR / "event_catalogue_used.csv", index=False)
    result["context"]["scaling"].to_csv(OUT_DATA_DIR / "predictor_scaling.csv", index=False)
    paths = [Path(__file__), Path(main_analysis.__file__), Path(age_sensitivity.__file__),
             Path(combined_pi.__file__), Path(poisson.__file__),
             main_analysis.PROJECT_ROOT / "NGRIP_MIS6_PI_bootstrap.py",
             main_analysis.PROJECT_ROOT / "toolbox/model_stats.py",
             main_analysis.BARKER_XLS, main_analysis.LR04_XLSX, main_analysis.CO2_XLSX,
             main_analysis.PRE_TXT]
    pd.DataFrame([dict(path=str(path.relative_to(main_analysis.PROJECT_ROOT)),
                       sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in paths]).to_csv(
        OUT_DATA_DIR / "input_code_sha256.csv", index=False)
    fig = plot_null_distribution(result["replicates"], result["summary"])
    fig.savefig(OUT_FIG_DIR / f"{RUN_NAME}.png", dpi=600)
    fig.savefig(OUT_FIG_DIR / f"{RUN_NAME}.pdf")
    copy_pdf_to_paper(OUT_FIG_DIR / f"{RUN_NAME}.pdf")
    plt.close(fig)
    write_notes(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=DEFAULT_N_WORKERS)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    args = parser.parse_args()
    started = time.perf_counter()
    result = run_analysis(n_bootstrap=args.n_bootstrap, seed=args.seed,
        max_attempts=args.max_attempts, n_workers=args.workers, show_progress=True)
    result["parameters"]["elapsed_seconds"] = time.perf_counter() - started
    save_results(result)
    print(result["summary"].to_string(index=False), flush=True)
    print(f"Saved {RUN_NAME}; elapsed {time.perf_counter() - started:.1f} s", flush=True)


if __name__ == "__main__":
    main()
