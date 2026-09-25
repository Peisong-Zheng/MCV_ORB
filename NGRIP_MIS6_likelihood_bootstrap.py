#!/usr/bin/env python3
"""Calibrate the continuous NGRIP--MIS6 precession test under its reduced model.

Keep exact conditioning events and physical endpoints fixed. Simulate new
continuous response events with inhibitory exponential history, then refit
both models. Each replicate has one scientific event sequence; numerical
recovery never substitutes another sequence.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import time

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mcv_orb_matplotlib"))
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_distribution, chi2
from toolbox import combined_likelihood
from toolbox.point_process import PointProcessFitError
from toolbox.project_config import PROJECT_ROOT
from toolbox.catalogue_colors import CATALOGUE_COLORS

RUN_NAME = "NGRIP_MIS6_likelihood_bootstrap"
OUT_DATA_DIR = PROJECT_ROOT / "data/processed" / RUN_NAME
OUT_FIG_DIR = PROJECT_ROOT / "figures" / RUN_NAME
DEFAULT_N_BOOTSTRAP = 9_999
DEFAULT_SEED = 20260905
DEFAULT_N_WORKERS = 3
CI_LEVEL = 0.95
NESTING_TOLERANCE = 1e-7
_WORKER_STATE = None


def _replicate(task, context, prepared):
    identifier, seed = task
    events = combined_likelihood.simulate_prepared_events(prepared, np.random.default_rng(seed))
    row = dict(bootstrap_id=identifier, n_events_observation_support=len(events),
               n_events_response=len(events)-len(context.segments), fit_valid=False,
               status="fit_failed", solver_attempts=0, failure_reason="")
    for segment in context.segments:
        row["n_events_response_" + segment] = int(events.segment_id.eq(segment).sum()) - 1
    for order in (context.quadrature_order, 2 * context.quadrature_order):
        row["solver_attempts"] += 1
        try:
            fit = combined_likelihood.fit_catalogue(events, replace(context, quadrature_order=order), fixed_support=True)
            zero = fit.reduced.status == fit.full.status == "zero_events"
            if not zero and not (fit.reduced.converged and fit.full.converged):
                raise PointProcessFitError("A nonzero-event fit lacks finite converged coefficients")
            gain = fit.full.log_likelihood - fit.reduced.log_likelihood
            if not np.isfinite(gain) or gain < -NESTING_TOLERANCE:
                raise PointProcessFitError("Nonfinite or nonnested likelihoods")
            # Only roundoff-size negative gains at a nested optimum become zero.
            gain = max(0.0, gain)
            row.update(loglik_reduced=fit.reduced.log_likelihood, loglik_full=fit.full.log_likelihood,
                       ll_gain_nats=gain, LR_statistic=2*gain,
                       gain_bits_per_event=gain/(row["n_events_response"]*np.log(2)) if row["n_events_response"] else np.nan,
                       reduced_converged=fit.reduced.converged, full_converged=fit.full.converged,
                       likelihood_nesting_ok=True, fit_valid=True,
                       status="zero_events" if zero else "finite_mle", failure_reason="",
                       quadrature_order=order)
            return row
        except (PointProcessFitError, FloatingPointError) as error:
            row["failure_reason"] = str(error)
    row.update(loglik_reduced=np.nan, loglik_full=np.nan, ll_gain_nats=np.nan,
               LR_statistic=np.nan, gain_bits_per_event=np.nan,
               reduced_converged=False, full_converged=False, likelihood_nesting_ok=False)
    return row


def _initialize_worker(context, prepared):
    global _WORKER_STATE
    _WORKER_STATE = context, prepared


def _worker_replicate(task):
    return _replicate(task, *_WORKER_STATE)


def run_bootstrap(observed_fit, context, *, n_bootstrap=DEFAULT_N_BOOTSTRAP,
                  seed=DEFAULT_SEED, n_workers=1, show_progress=False):
    """Return every requested replicate, with explicit same-data failures."""
    if n_bootstrap < 1 or n_workers < 1:
        raise ValueError("Bootstrap repetitions and worker count must be positive")
    if not observed_fit.reduced.converged or not observed_fit.full.converged:
        raise ValueError("The observed catalogue needs finite converged fits")
    prepared = combined_likelihood.prepare_model_simulation(context, observed_fit.reduced)
    tasks = list(enumerate(np.random.SeedSequence(seed).spawn(n_bootstrap), start=1))
    started = time.perf_counter()
    executor = None
    if n_workers == 1:
        results = (_replicate(task, context, prepared) for task in tasks)
    else:
        executor = ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("spawn"),
                                       initializer=_initialize_worker, initargs=(context, prepared))
        results = executor.map(_worker_replicate, tasks, chunksize=10)
    rows = []
    try:
        for index, row in enumerate(results, start=1):
            rows.append(row)
            if show_progress and (index % max(1, n_bootstrap//10) == 0 or index == n_bootstrap):
                print(f"Bootstrap {index:,}/{n_bootstrap:,} ({time.perf_counter()-started:.1f} s)", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    table = pd.DataFrame(rows)
    failures = Counter(table.loc[~table.fit_valid, "failure_reason"])
    return table, failures


def build_summary(observed_fit, replicates, failed_reasons):
    """No calibrated p is published while a requested replicate is unresolved."""
    row = dict(observed_fit.summary)
    failed = int((~replicates.fit_valid).sum())
    row.update(bootstrap_null="continuous fitted reduced model; exact anchors fixed",
               n_bootstrap=len(replicates), n_valid_replicates=int(replicates.fit_valid.sum()),
               n_failed_replicates=failed, n_zero_event_replicates=int(replicates.status.eq("zero_events").sum()),
               n_same_data_refinements=int(replicates.solver_attempts.gt(1).sum()),
               n_resampled_catalogues=0,
               failed_reasons="; ".join(f"{reason} ({n})" for reason,n in failed_reasons.items()))
    p = low = high = mc_se = np.nan
    exceedances = np.nan
    if not failed:
        p, exceedances = empirical_p_value(replicates.LR_statistic.to_numpy(), row["LR_statistic"])
        low, high = clopper_pearson_interval(exceedances, len(replicates))
        mc_se = np.sqrt(p*(1-p)/(len(replicates)+1))
    row.update(empirical_p_plus_one=p, n_bootstrap_exceeding_or_equal_observed=exceedances,
               empirical_p_mc_se=mc_se, empirical_p_ci95_low=low, empirical_p_ci95_high=high,
               empirical_p_ci_method="Clopper-Pearson interval for the null exceedance probability",
               bootstrap_response_event_count_mean=replicates.n_events_response.mean(),
               bootstrap_response_event_count_q025=replicates.n_events_response.quantile(.025),
               bootstrap_response_event_count_q975=replicates.n_events_response.quantile(.975))
    for name, value in [("mean",replicates.LR_statistic.mean()), ("median",replicates.LR_statistic.median()),
                        ("q95",replicates.LR_statistic.quantile(.95)), ("q99",replicates.LR_statistic.quantile(.99))]:
        row["bootstrap_LR_"+name] = value if not failed else np.nan
    return pd.DataFrame([row])


def build_parameters(context, *, n_bootstrap, seed, n_workers):
    parameters = dict(model_version=combined_likelihood.MODEL_VERSION, n_bootstrap=n_bootstrap, seed=seed,
        n_workers=n_workers, history_tau_kyr=context.history_tau_ka, history_coefficient_domain="beta_H <= 0",
        initial_unobserved_history=context.initial_history,
        conditioning="exact oldest event per segment; all younger exposure to the fixed endpoint",
        simulation="continuous thinning with a certified background envelope",
        refit_history="rebuilt from each simulated event sequence", response_exposure_kyr=context.response_exposure_kyr,
        reduced_terms="+".join(context.reduced_terms), full_terms="+".join(context.full_terms),
        conditioned_response_total=False, chronology_nested=False,
        quadrature_order=context.quadrature_order,
        numerical_recovery="same events; one retry with doubled quadrature order; no substitute catalogue",
        zero_events="likelihood supremum 0, LR=0, G undefined",
        empirical_p_rule="(1 + simulated LR >= observed LR) / (B + 1)")
    return pd.DataFrame(parameters.items(), columns=["parameter", "value"])


def run_analysis(*, n_bootstrap=DEFAULT_N_BOOTSTRAP, seed=DEFAULT_SEED,
                 n_workers=1, show_progress=False, context=None):
    context = combined_likelihood.build_context() if context is None else context
    observed = combined_likelihood.fit_catalogue(context.events, context)
    replicates, failures = run_bootstrap(observed, context, n_bootstrap=n_bootstrap,
            seed=seed, n_workers=n_workers, show_progress=show_progress)
    summary = build_summary(observed, replicates, failures)
    summary["seed"] = seed
    return dict(events=observed.design.all_events, context=context, observed_fit=observed,
                replicates=replicates, summary=summary, rejected_reasons=failures,
                parameters=build_parameters(context,n_bootstrap=n_bootstrap,seed=seed,n_workers=n_workers))


def save_tables(result, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, name in [("replicates", "bootstrap_replicates.csv"), ("summary", "summary.csv"),
                      ("parameters", "parameters_and_provenance.csv"), ("events", "event_catalogue_used.csv")]:
        result[key].to_csv(output_dir/name, index=False, float_format="%.12g")
    failed = result["replicates"].loc[lambda frame:~frame.fit_valid, ["bootstrap_id", "solver_attempts", "failure_reason"]]
    failed.to_csv(output_dir/"failed_replicates.csv",index=False)
    combined_likelihood.support_table(result["context"]).to_csv(output_dir/"support.csv",index=False,float_format="%.12g")
    combined_likelihood.scaling_table(result["context"]).to_csv(output_dir/"predictor_scaling.csv",index=False,float_format="%.12g")
    paths = [Path(__file__), Path(combined_likelihood.__file__), PROJECT_ROOT/"toolbox/point_process.py",
             PROJECT_ROOT/"toolbox/model_stats.py"]
    pd.DataFrame([dict(path=str(path.relative_to(PROJECT_ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                  for path in paths]).to_csv(output_dir/"input_code_sha256.csv",index=False)


def save_figure(replicates, summary, output_dir=OUT_FIG_DIR, *, paper_export=False):
    import matplotlib.pyplot as plt
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = plot_null_distribution(replicates, summary)
    png, pdf = [output_dir/f"{RUN_NAME}.{suffix}" for suffix in ("png", "pdf")]
    fig.savefig(png,dpi=450); fig.savefig(pdf)
    plt.close(fig)
    if paper_export:
        from Figure_likelihood_bootstrap import build_figure
        build_figure()
    return png,pdf


def write_notes(result, notes_dir):
    notes_dir = Path(notes_dir)
    notes_dir.mkdir(parents=True,exist_ok=True)
    s = result["summary"].iloc[0]
    caption = f"""Continuous reduced-model bootstrap calibration of the precession-phase test.
Gray bars show the LR distribution from {s.n_bootstrap} simulated catalogues; the colored line marks observed LR={s.LR_statistic:.4f}. The dashed chi-square(2) density is an asymptotic reference. Each simulation retains the exact NGRIP and MIS6 conditioning events and physical younger endpoints, generates continuous response times with dynamic exponential history, and refits both models under beta_H <= 0. The plus-one bootstrap p is {s.empirical_p_plus_one:.6g}. The binomial 95% interval [{s.empirical_p_ci95_low:.6g}, {s.empirical_p_ci95_high:.6g}] measures Monte Carlo uncertainty in the null exceedance probability, not an effect confidence interval. Chronology is fixed.
"""
    methods = f"""NGRIP--MIS6 CONTINUOUS PHASE NULL BOOTSTRAP
The reduced model includes LR04, CO2, an MIS6 segment intercept and inhibitory exponential event history (tau=1.5 kyr). Full adds phase sine and cosine. Likelihood uses actual event log intensities minus integrated intensity. Exact anchors and fixed exposure ({s.response_exposure_kyr:.6f} kyr) are retained in every replicate; no history crosses the record gap. Response event totals vary. Both models are independently refitted to every simulated sequence, with history recomputed from that sequence.

Observed: N={s.n_response_events}, G={s.gain_bits_per_event:.9f} bits/event, LR={s.LR_statistic:.9f}, Delta AIC={s.delta_AIC_full_minus_reduced:.9f}. Bootstrap repetitions={s.n_bootstrap}; exceedances={s.n_bootstrap_exceeding_or_equal_observed}; plus-one p={s.empirical_p_plus_one:.9g}. Seed={s.seed}. Failed replicates={s.n_failed_replicates}; all-empty replicates={s.n_zero_event_replicates}; same-data numerical refinements={s.n_same_data_refinements}. No failed simulation is replaced by a fresh draw. A zero-response catalogue has LR=0 and undefined G. Integration refinement does not change event ages or the simulated catalogue. A p value is withheld if any requested replicate remains unresolved.
"""
    (notes_dir/f"{RUN_NAME}_Caption.txt").write_text(caption)
    (notes_dir/f"{RUN_NAME}_Methods_and_results.txt").write_text(methods)



def empirical_p_value(
    bootstrap_statistics: np.ndarray, observed_statistic: float
) -> tuple[float, int]:
    """Return the plus-one bootstrap p value and exceedance count."""

    values = np.asarray(bootstrap_statistics, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("bootstrap_statistics must be a finite, non-empty vector")
    if not np.isfinite(observed_statistic):
        raise ValueError("observed_statistic must be finite")
    exceedances = int(np.count_nonzero(values >= observed_statistic))
    return float((exceedances + 1) / (len(values) + 1)), exceedances

def clopper_pearson_interval(
    successes: int,
    trials: int,
    *,
    confidence: float = CI_LEVEL,
) -> tuple[float, float]:
    """Exact binomial interval for the null exceedance probability."""

    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("Require 0 <= successes <= trials and trials > 0")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie between zero and one")

    alpha = 1.0 - confidence
    low = (
        0.0
        if successes == 0
        else float(
            beta_distribution.ppf(alpha / 2.0, successes, trials - successes + 1)
        )
    )
    high = (
        1.0
        if successes == trials
        else float(
            beta_distribution.ppf(
                1.0 - alpha / 2.0, successes + 1, trials - successes
            )
        )
    )
    return low, high

def configure_plot_style() -> None:
    """Use compact journal-scale typography and editable PDF fonts."""

    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9.5,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

def plot_null_distribution(
    replicates: pd.DataFrame, summary: pd.DataFrame
) -> object:
    """Plot the empirical reduced-model null and the observed LR gain."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configure_plot_style()
    row = summary.iloc[0]
    values = replicates["LR_statistic"].to_numpy(float)
    observed = float(row["LR_statistic"])
    x_min = min(0.0, float(values.min()))
    x_max = 1.06 * max(float(values.max()), observed)

    fig, axis = plt.subplots(figsize=(4.6, 3.45))
    axis.hist(
        values,
        bins=45,
        range=(x_min, x_max),
        density=True,
        color="#B8B8B8",
        edgecolor="white",
        linewidth=0.45,
        label="Reduced-model simulations",
    )
    x = np.linspace(max(0.0, x_min), x_max, 500)
    axis.plot(
        x,
        chi2.pdf(x, df=2),
        color="#555555",
        lw=1.2,
        ls=(0, (4, 2)),
        label=r"Asymptotic $\chi^2_2$",
    )
    axis.axvline(
        observed,
        color=CATALOGUE_COLORS["primary"],
        lw=1.7,
        label="Observed LR",
        zorder=4,
    )
    statistics_text = (
        f"Observed LR = {observed:.2f}\n"
        f"Bootstrap p = {row['empirical_p_plus_one']:.4g}\n"
        "95% MC interval\n"
        f"{row['empirical_p_ci95_low']:.4g}–"
        f"{row['empirical_p_ci95_high']:.4g}"
    )
    text_box = {
        "boxstyle": "round,pad=0.25",
        "facecolor": "white",
        "edgecolor": "#B8B8B8",
        "alpha": 0.92,
    }
    observed_fraction = (observed - x_min) / (x_max - x_min)
    if observed_fraction < 0.68:
        axis.text(
            0.97,
            0.94,
            statistics_text,
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            bbox=text_box,
        )
    else:
        axis.annotate(
            statistics_text,
            xy=(observed, 0.62),
            xycoords=axis.get_xaxis_transform(),
            xytext=(-8, 0),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=8.5,
            bbox=text_box,
        )
    axis.set_xlabel("Likelihood-ratio statistic")
    axis.set_ylabel("Probability density")
    axis.legend(frameon=False, loc="upper left")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(False)
    fig.tight_layout()
    return fig

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-bootstrap",type=int,default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--seed",type=int,default=DEFAULT_SEED)
    parser.add_argument("--workers",type=int,default=DEFAULT_N_WORKERS)
    parser.add_argument("--output-root",type=Path,default=PROJECT_ROOT)
    parser.add_argument("--no-paper-export",action="store_true")
    args=parser.parse_args(argv)
    result=run_analysis(n_bootstrap=args.n_bootstrap,seed=args.seed,n_workers=args.workers,show_progress=True)
    save_tables(result,args.output_root/"data/processed"/RUN_NAME)
    write_notes(result,args.output_root/"experiment_note")
    if result["summary"].iloc[0].n_failed_replicates:
        raise RuntimeError("Unresolved bootstrap fits saved; p value and figure publication withheld")
    save_figure(result["replicates"],result["summary"],args.output_root/"figures"/RUN_NAME,
                paper_export=not args.no_paper_export and args.output_root.resolve()==PROJECT_ROOT.resolve())
    print(result["summary"][["LR_statistic","empirical_p_plus_one","n_failed_replicates"]].to_string(index=False))


if __name__=="__main__":
    main()
