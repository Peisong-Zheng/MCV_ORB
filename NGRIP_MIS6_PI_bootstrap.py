#!/usr/bin/env python3
"""Calibrate the pooled-event PI test with a parametric bootstrap.

The observed comparison is the same one used by the main pooled analysis:
the reduced model contains 1.5-kyr event history, LR04, CO2, and a MIS 6
segment intercept; the full model adds sine and cosine of precession phase.

Bootstrap catalogues are generated under the fitted reduced model.  Age
increases into the past, so each observation segment is simulated from its
oldest bin towards its youngest bin.  Event history is recomputed from the
simulated counts, and it is reset at the NGRIP--MIS 6 gap.  Both nested models
are then refitted to every simulated catalogue.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
from pathlib import Path
from paper_figure_export import copy_pdf_to_paper
import tempfile
import time

# A stable writable cache avoids rebuilding Matplotlib fonts in every spawned
# worker on systems where ~/.matplotlib is read-only.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mcv_orb_matplotlib")
)
for thread_variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[thread_variable] = "1"

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_distribution
from scipy.stats import chi2

from toolbox import combined_pi, event_process, poisson
from toolbox.project_config import PROJECT_ROOT


RUN_NAME = "NGRIP_MIS6_PI_bootstrap"
OUT_DATA_DIR = PROJECT_ROOT / "data/processed" / RUN_NAME
OUT_FIG_DIR = PROJECT_ROOT / "figures" / RUN_NAME

DEFAULT_N_BOOTSTRAP = 9_999
DEFAULT_SEED = 20260905
DEFAULT_MAX_ATTEMPTS = 20
DEFAULT_N_WORKERS = min(6, max((os.cpu_count() or 2) - 1, 1))
NESTING_TOLERANCE = 1e-7
CI_LEVEL = 0.95

_WORKER_CONTEXT: combined_pi.PIContext | None = None
_WORKER_REDUCED_BETA: np.ndarray | None = None
_WORKER_MAX_ATTEMPTS: int | None = None


class InvalidBootstrapFit(RuntimeError):
    """A simulated catalogue could not be fitted without numerical warnings."""


def _check_fit(fit: combined_pi.CombinedPIFit, *, label: str) -> None:
    """Require finite, converged, nested fits without eta clipping."""

    values = np.concatenate(
        (
            fit.reduced.beta,
            fit.full.beta,
            [
                fit.reduced.log_likelihood,
                fit.full.log_likelihood,
                fit.summary["ll_gain_nats"],
            ],
        )
    )
    if not np.isfinite(values).all():
        raise InvalidBootstrapFit(f"{label}: non-finite fit result")
    if not fit.reduced.converged:
        raise InvalidBootstrapFit(
            f"{label}: reduced fit did not converge: "
            f"{fit.reduced.optimizer_message}"
        )
    if not fit.full.converged:
        raise InvalidBootstrapFit(
            f"{label}: full fit did not converge: {fit.full.optimizer_message}"
        )
    if fit.summary["eta_clipping_used"]:
        raise InvalidBootstrapFit(f"{label}: eta clipping was used")
    if float(fit.summary["ll_gain_nats"]) < -NESTING_TOLERANCE:
        raise InvalidBootstrapFit(
            f"{label}: full log likelihood is below the reduced model"
        )


def simulate_segment_counts(
    segment_bins: pd.DataFrame,
    segment: combined_pi.SegmentContext,
    reduced_beta: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate one segment from oldest to youngest with dynamic history."""

    if combined_pi.REDUCED_TERMS[0] != event_process.HISTORY_TERM:
        raise RuntimeError("The bootstrap expects event history to be the first term")

    beta = np.asarray(reduced_beta, dtype=float)
    expected_size = len(combined_pi.REDUCED_TERMS) + 1
    if beta.shape != (expected_size,) or not np.isfinite(beta).all():
        raise ValueError(f"reduced_beta must contain {expected_size} finite values")

    frame = segment_bins.reset_index(drop=True)
    n_bins = len(segment.bin_edges) - 1
    if len(frame) != n_bins:
        raise ValueError(f"Grid length does not match {segment.segment_id}")

    dt = frame["dt_kyr"].to_numpy(float)
    fixed_terms = combined_pi.REDUCED_TERMS[1:]
    fixed_predictors = frame.loc[:, fixed_terms].to_numpy(float)
    eta_without_history = beta[0] + fixed_predictors @ beta[2:]

    counts = np.zeros(n_bins, dtype=int)
    history_used = np.zeros(n_bins, dtype=float)
    history_beta = float(beta[1])

    # Older event counts are already available when a younger bin is drawn.
    for index in range(n_bins - 1, -1, -1):
        left = int(segment.history_left[index])
        right = int(segment.history_right[index])
        history_used[index] = float(counts[left:right].sum())
        eta = float(eta_without_history[index] + history_beta * history_used[index])
        if eta < poisson.ETA_MIN or eta > poisson.ETA_MAX:
            raise InvalidBootstrapFit(
                f"{segment.segment_id}: simulated eta={eta:.3f} would be clipped"
            )
        counts[index] = int(rng.poisson(float(dt[index] * np.exp(eta))))

    # This catches a reversed simulation direction or a changed history rule.
    recalculated = combined_pi.history_from_counts(counts, segment)
    if not np.array_equal(history_used, recalculated):
        raise RuntimeError("Simulated history does not match combined_pi")
    return counts


def simulate_catalogue_counts(
    context: combined_pi.PIContext,
    reduced_beta: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Simulate the two observation segments independently under the null."""

    counts: dict[str, np.ndarray] = {}
    for segment_id in combined_pi.SEGMENT_IDS:
        segment_bins = context.bins.loc[
            context.bins["segment_id"].eq(segment_id)
        ].copy()
        counts[segment_id] = simulate_segment_counts(
            segment_bins,
            context.segments[segment_id],
            reduced_beta,
            rng,
        )
    return counts


def empirical_p_value(
    bootstrap_statistics: np.ndarray, observed_statistic: float
) -> tuple[float, int]:
    """Return the conservative plus-one bootstrap p value and exceedance count."""

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


def _replicate_row(
    bootstrap_id: int,
    attempt_count: int,
    counts: dict[str, np.ndarray],
    fit: combined_pi.CombinedPIFit,
) -> dict[str, object]:
    response_counts = fit.response.groupby("segment_id", sort=False)[
        "event_count"
    ].sum()
    return {
        "bootstrap_id": int(bootstrap_id),
        "attempt_count": int(attempt_count),
        "n_events_observation_support": int(
            sum(values.sum() for values in counts.values())
        ),
        "n_events_response": int(fit.summary["n_predictive_events"]),
        "n_events_response_NGRIP": int(response_counts.get("NGRIP", 0)),
        "n_events_response_MIS6": int(response_counts.get("MIS6", 0)),
        "loglik_reduced": float(fit.reduced.log_likelihood),
        "loglik_full": float(fit.full.log_likelihood),
        "ll_gain_nats": float(fit.full.log_likelihood - fit.reduced.log_likelihood),
        "LR_statistic": float(
            2.0 * (fit.full.log_likelihood - fit.reduced.log_likelihood)
        ),
        "reduced_converged": bool(fit.reduced.converged),
        "full_converged": bool(fit.full.converged),
        "likelihood_nesting_ok": bool(
            fit.full.log_likelihood - fit.reduced.log_likelihood
            >= -NESTING_TOLERANCE
        ),
        "eta_clipping_used": bool(fit.summary["eta_clipping_used"]),
    }


def _draw_valid_replicate(
    bootstrap_index: int,
    replicate_seed: np.random.SeedSequence,
    context: combined_pi.PIContext,
    reduced_beta: np.ndarray,
    max_attempts: int,
) -> tuple[dict[str, object], Counter[str]]:
    """Draw one valid replicate, using private streams for numerical retries."""

    rejected_reasons: Counter[str] = Counter()
    for attempt_count, attempt_seed in enumerate(
        replicate_seed.spawn(max_attempts), start=1
    ):
        try:
            counts = simulate_catalogue_counts(
                context,
                reduced_beta,
                np.random.default_rng(attempt_seed),
            )
            fit = combined_pi.fit_event_counts(counts, context)
            _check_fit(fit, label=f"bootstrap {bootstrap_index}")
        except (InvalidBootstrapFit, ValueError) as error:
            rejected_reasons[str(error)] += 1
            continue
        return (
            _replicate_row(bootstrap_index, attempt_count, counts, fit),
            rejected_reasons,
        )

    reasons = "; ".join(
        f"{reason} ({count})" for reason, count in rejected_reasons.items()
    )
    raise RuntimeError(
        f"Bootstrap {bootstrap_index} failed after {max_attempts} attempts. "
        f"Rejections: {reasons}"
    )


def _initialize_worker(
    context: combined_pi.PIContext,
    reduced_beta: np.ndarray,
    max_attempts: int,
) -> None:
    """Store read-only bootstrap state once in each worker process."""

    global _WORKER_CONTEXT, _WORKER_REDUCED_BETA, _WORKER_MAX_ATTEMPTS
    _WORKER_CONTEXT = context
    _WORKER_REDUCED_BETA = np.asarray(reduced_beta, dtype=float)
    _WORKER_MAX_ATTEMPTS = int(max_attempts)


def _worker_replicate(
    task: tuple[int, np.random.SeedSequence],
) -> tuple[dict[str, object], Counter[str]]:
    """Process-pool entry point for one deterministic bootstrap replicate."""

    if (
        _WORKER_CONTEXT is None
        or _WORKER_REDUCED_BETA is None
        or _WORKER_MAX_ATTEMPTS is None
    ):
        raise RuntimeError("Bootstrap worker was not initialized")
    bootstrap_index, replicate_seed = task
    return _draw_valid_replicate(
        bootstrap_index,
        replicate_seed,
        _WORKER_CONTEXT,
        _WORKER_REDUCED_BETA,
        _WORKER_MAX_ATTEMPTS,
    )


def run_bootstrap(
    observed_fit: combined_pi.CombinedPIFit,
    context: combined_pi.PIContext,
    *,
    n_bootstrap: int,
    seed: int,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    n_workers: int = 1,
    show_progress: bool = False,
) -> tuple[pd.DataFrame, Counter[str]]:
    """Generate and refit valid null catalogues using independent RNG streams."""

    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    if n_workers <= 0:
        raise ValueError("n_workers must be positive")
    _check_fit(observed_fit, label="observed catalogue")

    replicate_seeds = np.random.SeedSequence(seed).spawn(n_bootstrap)
    rejected_reasons: Counter[str] = Counter()
    rows: list[dict[str, object]] = []
    progress_step = max(n_bootstrap // 10, 1)
    started = time.perf_counter()

    tasks = list(enumerate(replicate_seeds, start=1))
    if n_workers == 1:
        results = (
            _draw_valid_replicate(
                bootstrap_index,
                replicate_seed,
                context,
                observed_fit.reduced.beta,
                max_attempts,
            )
            for bootstrap_index, replicate_seed in tasks
        )
        executor = None
    else:
        # Spawn is explicit because it is the macOS default and avoids relying
        # on inherited optimizer state.  SeedSequence makes results invariant
        # to worker scheduling and worker count.
        executor = ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=mp.get_context("spawn"),
            initializer=_initialize_worker,
            initargs=(context, observed_fit.reduced.beta, max_attempts),
        )
        chunksize = max(1, min(20, n_bootstrap // (n_workers * 20)))
        results = executor.map(_worker_replicate, tasks, chunksize=chunksize)

    try:
        for bootstrap_index, (row, rejections) in enumerate(results, start=1):
            rows.append(row)
            rejected_reasons.update(rejections)
            if show_progress and (
                bootstrap_index % progress_step == 0
                or bootstrap_index == n_bootstrap
            ):
                elapsed = time.perf_counter() - started
                print(
                    f"Bootstrap {bootstrap_index:,}/{n_bootstrap:,} "
                    f"({elapsed:.1f} s)",
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    return pd.DataFrame(rows), rejected_reasons


def build_summary(
    observed_fit: combined_pi.CombinedPIFit,
    replicates: pd.DataFrame,
    rejected_reasons: Counter[str],
) -> pd.DataFrame:
    """Summarize the observed test and its empirical null distribution."""

    observed_lr = float(observed_fit.summary["LR_statistic"])
    bootstrap_lr = replicates["LR_statistic"].to_numpy(float)
    empirical_p, exceedances = empirical_p_value(bootstrap_lr, observed_lr)
    ci_low, ci_high = clopper_pearson_interval(exceedances, len(replicates))
    mc_se = float(np.sqrt(empirical_p * (1.0 - empirical_p) / (len(replicates) + 1)))
    retry_text = (
        "none"
        if not rejected_reasons
        else "; ".join(
            f"{reason} ({count})"
            for reason, count in sorted(rejected_reasons.items())
        )
    )

    row = {
        **observed_fit.summary,
        "bootstrap_null": "fitted reduced model with dynamic event history",
        "n_bootstrap": int(len(replicates)),
        "n_bootstrap_exceeding_or_equal_observed": int(exceedances),
        "empirical_p_plus_one": empirical_p,
        "empirical_p_mc_se": mc_se,
        "empirical_p_ci95_low": ci_low,
        "empirical_p_ci95_high": ci_high,
        "empirical_p_ci_method": "Clopper-Pearson exact binomial",
        "bootstrap_LR_mean": float(np.mean(bootstrap_lr)),
        "bootstrap_LR_median": float(np.median(bootstrap_lr)),
        "bootstrap_LR_q95": float(np.quantile(bootstrap_lr, 0.95)),
        "bootstrap_LR_q99": float(np.quantile(bootstrap_lr, 0.99)),
        "bootstrap_response_event_count_mean": float(
            replicates["n_events_response"].mean()
        ),
        "bootstrap_response_event_count_q025": float(
            replicates["n_events_response"].quantile(0.025)
        ),
        "bootstrap_response_event_count_q975": float(
            replicates["n_events_response"].quantile(0.975)
        ),
        "n_rejected_attempts": int(sum(rejected_reasons.values())),
        "rejected_attempt_reasons": retry_text,
    }
    return pd.DataFrame([row])


def build_parameters(
    context: combined_pi.PIContext,
    *,
    n_bootstrap: int,
    seed: int,
    max_attempts: int,
    n_workers: int,
) -> pd.DataFrame:
    """Record model, simulation, and source choices beside the results."""

    rows = [
        ("age_unit", "kyr BP", "BP1950", "used for events and forcings"),
        (
            "event_catalogue",
            str(combined_pi.EVENT_CATALOGUE_CSV.relative_to(PROJECT_ROOT)),
            "",
            "34 NGRIP warming starts and 21 MIS 6 transitions",
        ),
        (
            "observation_segments",
            str(combined_pi.OBSERVATION_SEGMENTS_CSV.relative_to(PROJECT_ROOT)),
            "",
            "two disjoint observation segments",
        ),
        ("history_window", context.history_window_ka, "kyr", "main setting"),
        ("bin_width", context.bin_width_ka, "kyr", "main setting"),
        (
            "bin_origin_fraction",
            context.origin_fraction,
            "bin width",
            "main setting",
        ),
        ("response_mode", context.response_mode, "", "main setting"),
        (
            "response_exposure",
            context.response_exposure_kyr,
            "kyr",
            "sum of both response segments only",
        ),
        (
            "reduced_model_terms",
            "+".join(combined_pi.REDUCED_TERMS),
            "",
            "null-generating model",
        ),
        (
            "full_model_terms",
            "+".join(combined_pi.FULL_TERMS),
            "",
            "adds precession sine and cosine",
        ),
        (
            "simulation_direction",
            "oldest to youngest",
            "",
            "age increases into the past; older events determine history",
        ),
        (
            "segment_history",
            "reset independently for NGRIP and MIS6",
            "",
            "the unobserved gap is neither exposure nor event history",
        ),
        (
            "oldest_boundary_history",
            "zero outside each observation segment",
            "",
            "the history-only buffer itself is simulated under this boundary condition",
        ),
        (
            "simulation_support",
            "complete observation intervals",
            "",
            "response models are refitted on response bins only",
        ),
        (
            "history_in_refits",
            "recomputed from each simulated catalogue",
            "",
            "the observed history covariate is never held fixed",
        ),
        (
            "resolution_covariate_included",
            False,
            "",
            "not part of reduced or full model",
        ),
        ("n_bootstrap", n_bootstrap, "replicates", "formal run"),
        ("random_seed", seed, "", "NumPy SeedSequence base entropy"),
        (
            "rng_streams",
            "one spawned SeedSequence per replicate and attempt",
            "",
            "reproducible independent streams",
        ),
        (
            "worker_processes",
            n_workers,
            "processes",
            "parallel scheduling does not change replicate streams",
        ),
        (
            "linear_algebra_threads",
            1,
            "thread per worker",
            "prevents nested parallelism during many small fits",
        ),
        ("maximum_attempts", max_attempts, "per replicate", "numerical retries"),
        (
            "valid_fit_rule",
            "both models converge; nested; finite; no eta clipping",
            "",
            "invalid attempts are redrawn and counted",
        ),
        (
            "empirical_p_rule",
            "(1 + count[LR_sim >= LR_obs]) / (B + 1)",
            "",
            "conservative plus-one estimator",
        ),
        (
            "p_interval",
            "Clopper-Pearson exact binomial 95% CI",
            "",
            "interval for the null exceedance probability",
        ),
        (
            "reference_implementation",
            "Predictive_information_bootstrap_diagnostics.py",
            "",
            "algorithm adapted and rewritten for the pooled catalogue",
        ),
    ]
    return pd.DataFrame(rows, columns=["parameter", "value", "unit", "note"])


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
        color="#3973AC",
        lw=1.2,
        ls=(0, (4, 2)),
        label=r"Asymptotic $\chi^2_2$",
    )
    axis.axvline(
        observed,
        color="#B2185B",
        lw=1.7,
        label="Observed LR",
        zorder=4,
    )
    statistics_text = (
        f"Observed LR = {observed:.2f}\n"
        f"Empirical p = {row['empirical_p_plus_one']:.4g}\n"
        "Binomial 95% CI\n"
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


def run_analysis(
    *,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    n_workers: int = 1,
    show_progress: bool = False,
) -> dict[str, object]:
    """Fit the observed test and run the dynamic-history null experiment."""

    context = combined_pi.build_context()
    events = combined_pi.load_event_catalogue()
    observed_fit = combined_pi.fit_catalogue(events, context)
    _check_fit(observed_fit, label="observed catalogue")
    replicates, rejected = run_bootstrap(
        observed_fit,
        context,
        n_bootstrap=n_bootstrap,
        seed=seed,
        max_attempts=max_attempts,
        n_workers=n_workers,
        show_progress=show_progress,
    )
    summary = build_summary(observed_fit, replicates, rejected)
    parameters = build_parameters(
        context,
        n_bootstrap=n_bootstrap,
        seed=seed,
        max_attempts=max_attempts,
        n_workers=n_workers,
    )
    return {
        "events": events,
        "context": context,
        "observed_fit": observed_fit,
        "replicates": replicates,
        "summary": summary,
        "parameters": parameters,
        "rejected_reasons": rejected,
    }


def save_results(result: dict[str, object]) -> None:
    """Write the compact bootstrap tables and publication-ready figure."""

    OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    result["replicates"].to_csv(
        OUT_DATA_DIR / "bootstrap_replicates.csv", index=False
    )
    result["summary"].to_csv(OUT_DATA_DIR / "summary.csv", index=False)
    result["parameters"].to_csv(
        OUT_DATA_DIR / "parameters_and_provenance.csv", index=False
    )

    import matplotlib.pyplot as plt

    figure = plot_null_distribution(result["replicates"], result["summary"])
    figure.savefig(
        OUT_FIG_DIR / f"{RUN_NAME}.png", dpi=400, bbox_inches="tight"
    )
    figure.savefig(OUT_FIG_DIR / f"{RUN_NAME}.pdf", bbox_inches="tight")
    copy_pdf_to_paper(OUT_FIG_DIR / f"{RUN_NAME}.pdf")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    """Parse bootstrap run settings."""

    parser = argparse.ArgumentParser(
        description=(
            "Parametric-bootstrap calibration of precession PI for the pooled "
            "NGRIP-warming and MIS-6-transition catalogue."
        )
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=DEFAULT_N_BOOTSTRAP,
        help=f"Number of valid null catalogues (default: {DEFAULT_N_BOOTSTRAP:,}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"SeedSequence base entropy (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help="Maximum numerical redraws allowed for one valid replicate.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_N_WORKERS,
        help=f"Independent worker processes (default: {DEFAULT_N_WORKERS}).",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Do not print progress updates."
    )
    return parser.parse_args()


def main() -> None:
    """Run, save, and print the formal bootstrap calibration."""

    args = parse_args()
    started = time.perf_counter()
    result = run_analysis(
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        max_attempts=args.max_attempts,
        n_workers=args.workers,
        show_progress=not args.quiet,
    )
    save_results(result)
    elapsed = time.perf_counter() - started

    summary = result["summary"].iloc[0]
    print(
        "\nReduced-model parametric bootstrap\n"
        f"Observed LR: {summary['LR_statistic']:.6f}\n"
        f"Nominal p: {summary['nominal_LR_p']:.6g}\n"
        f"Empirical p: {summary['empirical_p_plus_one']:.6g} "
        f"(95% binomial CI {summary['empirical_p_ci95_low']:.6g}–"
        f"{summary['empirical_p_ci95_high']:.6g})\n"
        f"Valid replicates: {int(summary['n_bootstrap']):,}; "
        f"rejected attempts: {int(summary['n_rejected_attempts']):,}\n"
        f"Elapsed time: {elapsed:.1f} s\n"
        f"Tables: {OUT_DATA_DIR.relative_to(PROJECT_ROOT)}\n"
        f"Figure: {OUT_FIG_DIR.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
