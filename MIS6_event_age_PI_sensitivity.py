#!/usr/bin/env python3
"""Propagate the MIS 6 A+ event-age realizations through PI only.

The forcing series, bins, five-kyr event-history term, and nested Poisson
models are exactly those used by ``MIS6_event_phase_analysis.py``.  This script
does not run a Rayleigh test and does not add a proxy-resolution covariate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from toolbox import event_process as predictive, mis6_pi, poisson
from toolbox.model_stats import nested_likelihood_metrics
from toolbox.project_config import PROJECT_ROOT


RUN_NAME = "MIS6_event_age_PI_sensitivity"
MC_INPUT = (
    PROJECT_ROOT
    / "data/processed/MIS6_event_age_uncertainty/mis6_event_age_realizations.csv"
)
OUT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / RUN_NAME
OUT_FIG_DIR = PROJECT_ROOT / "figures" / RUN_NAME
REALIZATION_OUTPUT = OUT_DATA_DIR / "mis6_event_age_pi_realizations.csv"
SUMMARY_OUTPUT = OUT_DATA_DIR / "mis6_event_age_pi_summary.csv"
PARAMETER_OUTPUT = OUT_DATA_DIR / "parameters_and_provenance.csv"
FIGURE_STEM = RUN_NAME
PNG_DPI = 600
N_EVENTS = mis6_pi.N_EVENTS
P_THRESHOLD = 0.05


def configure_plot_style() -> None:
    """Use the same journal-scale typography as the point-age analysis."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


@dataclass(frozen=True)
class PIContext:
    """Fixed PI inputs plus indices needed to rebuild event history."""

    bin_edges: np.ndarray
    bin_centers: np.ndarray
    history_left: np.ndarray
    history_right: np.ndarray
    complete_history: np.ndarray
    dt_fit: np.ndarray
    fixed_predictors: dict[str, np.ndarray]
    response_support_start_ka: float
    response_support_end_ka: float


def event_age_columns() -> list[str]:
    return [f"MIS6_DO_{number:02d}_age_ka_bp" for number in range(1, N_EVENTS + 1)]


def load_age_realizations(path: Path = MC_INPUT) -> pd.DataFrame:
    """Load the ordered A+ sequences and check the assumptions used below."""
    draws = pd.read_csv(path)
    required = {"realization_id", *event_age_columns()}
    if missing := required.difference(draws.columns):
        raise ValueError(f"Age-realization table is missing columns: {sorted(missing)}")
    if draws.empty or draws["realization_id"].duplicated().any():
        raise ValueError("Realization IDs must be non-empty and unique")

    ages = draws[event_age_columns()].to_numpy(float)
    if not np.isfinite(ages).all():
        raise ValueError("All realized event ages must be finite")
    if not np.all(np.diff(ages, axis=1) > 0):
        raise ValueError("Every realized event sequence must preserve event order")
    if not (
        (ages >= mis6_pi.ANALYSIS_START_KA) & (ages <= mis6_pi.ANALYSIS_END_KA)
    ).all():
        raise ValueError(
            "At least one realized age lies outside the PI binning support"
        )
    return draws


def counts_from_ages(ages: np.ndarray, context: PIContext) -> np.ndarray:
    """Bin all 21 ages; event_count may exceed one in a bin."""
    counts, _ = np.histogram(np.asarray(ages, dtype=float), bins=context.bin_edges)
    if int(counts.sum()) != N_EVENTS:
        raise ValueError("A realized event was lost outside the full analysis grid")
    return counts.astype(float)


def history_from_counts(counts: np.ndarray, context: PIContext) -> np.ndarray:
    """Match the shared `(t, t + 5 Kyr]` older-event history calculation."""
    cumulative = np.concatenate([[0.0], np.cumsum(counts)])
    return cumulative[context.history_right] - cumulative[context.history_left]


def design_matrix_for_counts(
    counts: np.ndarray,
    context: PIContext,
    terms: tuple[str, ...],
) -> np.ndarray:
    """Combine the draw-specific history with fixed climate/phase predictors."""
    history = history_from_counts(counts, context)
    values = {
        predictive.HISTORY_TERM: history,
        **context.fixed_predictors,
    }
    return np.column_stack([values[term] for term in terms])[context.complete_history]


def fit_count_pattern(counts: np.ndarray, context: PIContext) -> dict[str, object]:
    """Fit the reduced and phase-augmented models for one binned catalogue."""
    y = counts[context.complete_history]
    reduced = poisson.fit_binned_poisson_arrays(
        design_matrix_for_counts(counts, context, mis6_pi.REDUCED_TERMS),
        y,
        context.dt_fit,
    )
    full = poisson.fit_binned_poisson_arrays(
        design_matrix_for_counts(counts, context, mis6_pi.FULL_TERMS),
        y,
        context.dt_fit,
    )
    metrics = nested_likelihood_metrics(
        loglik_full=full.log_likelihood,
        loglik_reduced=reduced.log_likelihood,
        df=len(full.beta) - len(reduced.beta),
        n_bins=len(y),
        n_events=int(y.sum()),
        aicc_full=full.aicc,
        aicc_reduced=reduced.aicc,
    )

    beta_sin = float(full.beta[1 + mis6_pi.FULL_TERMS.index("pre_phase_sin")])
    beta_cos = float(full.beta[1 + mis6_pi.FULL_TERMS.index("pre_phase_cos")])
    amplitude = float(np.hypot(beta_sin, beta_cos))
    preferred = float(np.degrees(np.mod(np.arctan2(beta_sin, beta_cos), 2 * np.pi)))
    n_eta_clipped = sum(
        (
            reduced.n_eta_clipped_low,
            reduced.n_eta_clipped_high,
            full.n_eta_clipped_low,
            full.n_eta_clipped_high,
        )
    )
    return {
        "n_predictive_bins": len(y),
        "n_predictive_events": int(y.sum()),
        "max_events_in_single_bin": int(counts.max()),
        "reduced_converged": reduced.converged,
        "full_converged": full.converged,
        "loglik_reduced": metrics["loglik_reduced"],
        "loglik_full": metrics["loglik_full"],
        "info_bits_per_event": metrics["info_bits_per_event"],
        "LR_statistic": metrics["LR_statistic"],
        "nominal_LR_p": metrics["LR_p_value"],
        "delta_AICc_full_minus_reduced": metrics["delta_AICc_full_minus_reduced"],
        "pre_phase_preferred_deg": preferred,
        "pre_phase_rate_ratio_max_vs_min": float(np.exp(2 * amplitude)),
        "likelihood_nesting_ok": metrics["ll_gain_nats"] >= -1e-8,
        "n_eta_clipped": int(n_eta_clipped),
    }


def validate_fast_reference(
    result: dict[str, object], reference: dict[str, object]
) -> None:
    """Check the cached-fit implementation against the point-age PI result."""
    model_summary = reference["model_summary"].set_index("model_id")
    likelihood = reference["likelihood_tests"].iloc[0]
    comparisons = {
        "loglik_reduced": model_summary.loc[mis6_pi.REDUCED_MODEL_ID, "log_likelihood"],
        "loglik_full": model_summary.loc[mis6_pi.FULL_MODEL_ID, "log_likelihood"],
        "LR_statistic": likelihood["LR_statistic"],
        "nominal_LR_p": likelihood["LR_p_value"],
        "info_bits_per_event": likelihood["info_bits_per_event"],
        "delta_AICc_full_minus_reduced": likelihood["delta_AICc_full_minus_reduced"],
        "pre_phase_preferred_deg": model_summary.loc[
            mis6_pi.FULL_MODEL_ID, "pre_phase_preferred_deg"
        ],
    }
    for name, expected in comparisons.items():
        if not np.isclose(float(result[name]), float(expected), atol=2e-7, rtol=2e-8):
            raise RuntimeError(f"Fast PI path does not reproduce baseline {name}")


def build_pi_context(
    original_events: pd.DataFrame,
) -> tuple[PIContext, dict[str, object], dict[str, object]]:
    """Load fixed forcings once and prepare the arrays shared by all draws."""
    reference = mis6_pi.run_predictive_information(original_events)
    binned = reference["binned"].sort_values("bin_center_ka").reset_index(drop=True)
    centers = binned["bin_center_ka"].to_numpy(float)
    edges = np.concatenate(
        [binned["bin_start_ka"].to_numpy(float), [binned["bin_end_ka"].iloc[-1]]]
    )
    left = np.searchsorted(centers, centers + 1e-9, side="right")
    right = np.searchsorted(centers, centers + mis6_pi.HISTORY_WINDOW_KA, side="right")
    complete = binned["same_type_history_complete"].to_numpy(bool)
    fixed_terms = set(mis6_pi.FULL_TERMS).difference({predictive.HISTORY_TERM})
    context = PIContext(
        bin_edges=edges,
        bin_centers=centers,
        history_left=left,
        history_right=right,
        complete_history=complete,
        dt_fit=binned.loc[complete, "dt_ka"].to_numpy(float),
        fixed_predictors={term: binned[term].to_numpy(float) for term in fixed_terms},
        response_support_start_ka=float(binned.loc[complete, "bin_start_ka"].min()),
        response_support_end_ka=float(binned.loc[complete, "bin_end_ka"].max()),
    )
    original_counts = counts_from_ages(
        original_events["event_age_ka_bp"].to_numpy(float), context
    )
    original_result = fit_count_pattern(original_counts, context)
    validate_fast_reference(original_result, reference)
    return context, original_result, reference


def fit_realizations(
    draws: pd.DataFrame,
    context: PIContext,
    *,
    show_progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit every realization, caching identical 0.2-Kyr count patterns."""
    ages = draws[event_age_columns()].to_numpy(float)
    identifiers = draws[
        [column for column in ("realization_id", "proposal_id") if column in draws]
    ].reset_index(drop=True)
    cache: dict[bytes, tuple[int, dict[str, object]]] = {}
    rows: list[dict[str, object]] = []
    started = time.perf_counter()

    for index, realized_ages in enumerate(ages):
        counts = counts_from_ages(realized_ages, context)
        key = counts.astype(np.int8).tobytes()
        if key not in cache:
            cache[key] = (len(cache) + 1, fit_count_pattern(counts, context))
        pattern_id, fit = cache[key]
        row = identifiers.iloc[index].to_dict()
        row.update({"event_count_pattern_id": pattern_id, **fit})
        rows.append(row)
        if show_progress and (index + 1) % 1000 == 0:
            print(
                f"Fitted {index + 1:,}/{len(ages):,} realizations "
                f"({len(cache):,} unique bin patterns)"
            )

    results = pd.DataFrame(rows)
    elapsed = time.perf_counter() - started
    distribution = results["n_predictive_events"].value_counts().sort_index()
    diagnostics = pd.DataFrame(
        [
            {
                "n_realizations": len(results),
                "n_unique_event_count_patterns": len(cache),
                "n_cached_reuses": len(results) - len(cache),
                "cache_reuse_fraction": 1 - len(cache) / len(results),
                "elapsed_seconds": elapsed,
                "predictive_event_count_distribution": ";".join(
                    f"{int(count)}:{int(frequency)}"
                    for count, frequency in distribution.items()
                ),
                "n_realizations_with_multiple_events_in_one_bin": int(
                    results["max_events_in_single_bin"].gt(1).sum()
                ),
                "n_nonconverged_reduced": int(
                    (~results["reduced_converged"].astype(bool)).sum()
                ),
                "n_nonconverged_full": int(
                    (~results["full_converged"].astype(bool)).sum()
                ),
                "n_likelihood_nesting_failures": int(
                    (~results["likelihood_nesting_ok"].astype(bool)).sum()
                ),
                "n_realizations_with_eta_clipping": int(
                    results["n_eta_clipped"].gt(0).sum()
                ),
            }
        ]
    )
    return results, diagnostics


def compact_realization_results(results: pd.DataFrame) -> pd.DataFrame:
    """Keep the per-sequence quantities needed to inspect PI sensitivity."""
    compact = results.assign(
        all_models_converged=(
            results["reduced_converged"].astype(bool)
            & results["full_converged"].astype(bool)
        ),
        eta_clipping_used=results["n_eta_clipped"].gt(0),
    )
    columns = [
        "realization_id",
        "n_predictive_events",
        "max_events_in_single_bin",
        "LR_statistic",
        "nominal_LR_p",
        "info_bits_per_event",
        "delta_AICc_full_minus_reduced",
        "pre_phase_preferred_deg",
        "pre_phase_rate_ratio_max_vs_min",
        "all_models_converged",
        "likelihood_nesting_ok",
        "eta_clipping_used",
    ]
    return compact.loc[:, columns].copy()


def unwrap_around(values_deg: np.ndarray, center_deg: float) -> np.ndarray:
    """Unwrap circular values onto the branch centered on the original result."""
    return center_deg + (np.asarray(values_deg) - center_deg + 180) % 360 - 180


def build_summary(
    results: pd.DataFrame,
    original: dict[str, object],
) -> pd.DataFrame:
    """Summarize PI sensitivity; decision fractions are not calibrated p-values."""
    specs = [
        ("LR_statistic", "LR_statistic", "", "larger favors the phase model"),
        ("nominal_LR_p", "nominal_LR_p", "", "nominal asymptotic LRT p value"),
        (
            "info_bits_per_event",
            "info_bits_per_event",
            "bits/event",
            "in-sample nested log-likelihood gain",
        ),
        (
            "delta_AICc_full_minus_reduced",
            "delta_AICc_full_minus_reduced",
            "AICc",
            "negative favors the phase-augmented model",
        ),
        (
            "pre_phase_preferred_deg_unwrapped_around_original",
            "pre_phase_preferred_deg",
            "degrees",
            "circular phase unwrapped within ±180 degrees of the original result",
        ),
        (
            "pre_phase_rate_ratio_max_vs_min",
            "pre_phase_rate_ratio_max_vs_min",
            "ratio",
            "fitted maximum/minimum phase contribution to event rate",
        ),
        (
            "n_predictive_events",
            "n_predictive_events",
            "events",
            "events inside complete five-kyr-history response support",
        ),
    ]
    rows = []
    for metric, column, unit, note in specs:
        original_value = float(original[column])
        values = results[column].to_numpy(float)
        if column == "pre_phase_preferred_deg":
            values = unwrap_around(values, original_value)
        quantiles = np.quantile(values, [0.025, 0.16, 0.5, 0.84, 0.975])
        decision_rule = ""
        fraction = np.nan
        if column == "nominal_LR_p":
            decision_rule = "nominal_LR_p < 0.05"
            fraction = float(np.mean(values < P_THRESHOLD))
        elif column == "delta_AICc_full_minus_reduced":
            decision_rule = "delta_AICc_full_minus_reduced < 0"
            fraction = float(np.mean(values < 0))
        rows.append(
            {
                "metric": metric,
                "unit": unit,
                "original_sequence_value": original_value,
                "mc_mean": values.mean(),
                "mc_sd": values.std(ddof=1),
                "mc_q025": quantiles[0],
                "mc_q16": quantiles[1],
                "mc_median": quantiles[2],
                "mc_q84": quantiles[3],
                "mc_q975": quantiles[4],
                "original_percentile_in_mc": 100
                * (
                    np.sum(values < original_value)
                    + 0.5 * np.sum(values == original_value)
                )
                / len(values),
                "decision_rule": decision_rule,
                "mc_fraction_meeting_decision_rule": fraction,
                "note": note,
            }
        )
    return pd.DataFrame(rows)


def build_parameters(context: PIContext, diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Record model choices, deliberate omissions, and run diagnostics."""
    rows = [
        (
            "analysis",
            "conditional predictive information only",
            "",
            "Rayleigh is not run",
        ),
        ("rayleigh_test_run", False, "", "explicitly excluded from this script"),
        ("event_age_uncertainty_propagated", True, "", "A+ accepted realizations"),
        (
            "resolution_covariate_included",
            False,
            "",
            "matches the point-age MIS6/NGRIP PI model",
        ),
        (
            "mc_input",
            str(MC_INPUT.relative_to(PROJECT_ROOT)),
            "",
            "wide ordered age sequences",
        ),
        (
            "analysis_start",
            mis6_pi.ANALYSIS_START_KA,
            "Kyr BP",
            "full binning support",
        ),
        ("analysis_end", mis6_pi.ANALYSIS_END_KA, "Kyr BP", "full binning support"),
        ("bin_width", mis6_pi.BIN_WIDTH_KA, "Kyr", "events may share a bin"),
        (
            "history_window",
            mis6_pi.HISTORY_WINDOW_KA,
            "Kyr",
            "older events in (t, t + window]",
        ),
        (
            "response_support_start",
            context.response_support_start_ka,
            "Kyr BP",
            "complete history only",
        ),
        (
            "response_support_end",
            context.response_support_end_ka,
            "Kyr BP",
            "complete history only",
        ),
        (
            "reduced_model_terms",
            "+".join(mis6_pi.REDUCED_TERMS),
            "",
            "same as MIS6/NGRIP PI",
        ),
        (
            "full_model_terms",
            "+".join(mis6_pi.FULL_TERMS),
            "",
            "adds precession sine and cosine",
        ),
        (
            "PI_definition",
            "in-sample nested Poisson log-likelihood gain",
            "",
            "not mutual information estimated by cross-validation",
        ),
        (
            "p_value_method",
            "nominal asymptotic chi-square LRT",
            "",
            "not recalibrated by the age MC",
        ),
        (
            "decision_fraction_interpretation",
            "fraction of age realizations retaining a criterion",
            "",
            "robustness proportion, not a new Monte Carlo p value",
        ),
        (
            "optimizer",
            "toolbox.poisson.fit_binned_poisson_arrays",
            "",
            "same bounded L-BFGS-B likelihood as baseline",
        ),
        (
            "fit_cache",
            "full 320-bin event-count vector",
            "",
            f"{int(diagnostics.loc[0, 'n_unique_event_count_patterns'])} unique patterns",
        ),
        (
            "reference_analysis",
            "MIS6_event_phase_analysis.py",
            "",
            "PI definitions reused; Rayleigh functions not called",
        ),
    ]

    diagnostic_notes = {
        "n_realizations": ("sequences", "all rows in the MC input"),
        "n_unique_event_count_patterns": ("patterns", "unique 0.2-Kyr count vectors"),
        "n_cached_reuses": ("sequences", "fits reused for duplicate count vectors"),
        "cache_reuse_fraction": ("fraction", "share of fits served from the cache"),
        "predictive_event_count_distribution": (
            "events:sequences",
            "events inside complete-history response support",
        ),
        "n_realizations_with_multiple_events_in_one_bin": (
            "sequences",
            "at least one 0.2-Kyr bin contains multiple events",
        ),
        "n_nonconverged_reduced": ("fits", "reduced-model optimizer failures"),
        "n_nonconverged_full": ("fits", "full-model optimizer failures"),
        "n_likelihood_nesting_failures": (
            "fits",
            "full-model likelihood below reduced-model likelihood",
        ),
        "n_realizations_with_eta_clipping": (
            "sequences",
            "at least one fitted linear predictor reached a clipping bound",
        ),
    }
    run = diagnostics.iloc[0]
    for parameter, (unit, note) in diagnostic_notes.items():
        rows.append((parameter, run[parameter], unit, note))
    return pd.DataFrame(rows, columns=["parameter", "value", "unit", "note"])


def validate_results(results: pd.DataFrame, draws: pd.DataFrame) -> None:
    """Reject numerically unreliable sensitivity output."""
    if len(results) != len(draws) or not results["realization_id"].equals(
        draws["realization_id"].reset_index(drop=True)
    ):
        raise ValueError("PI results do not map one-to-one to age realizations")
    numeric = [
        "info_bits_per_event",
        "LR_statistic",
        "nominal_LR_p",
        "delta_AICc_full_minus_reduced",
        "pre_phase_preferred_deg",
        "pre_phase_rate_ratio_max_vs_min",
    ]
    if not np.isfinite(results[numeric].to_numpy(float)).all():
        raise ValueError("PI sensitivity output contains non-finite metrics")
    if not results[["reduced_converged", "full_converged"]].all().all():
        raise RuntimeError("At least one Monte Carlo PI model did not converge")
    if not results["likelihood_nesting_ok"].all():
        raise RuntimeError(
            "At least one full-model likelihood is below its reduced model"
        )
    if results["n_eta_clipped"].gt(0).any():
        raise RuntimeError("At least one Monte Carlo PI fit used eta clipping")


def plot_sensitivity(
    results: pd.DataFrame,
    original: dict[str, object],
) -> tuple[Path, Path]:
    """Show the main conditional-PI distributions against the point-age result."""
    configure_plot_style()
    phase_center = float(original["pre_phase_preferred_deg"])
    panels = [
        (
            "info_bits_per_event",
            results["info_bits_per_event"].to_numpy(float),
            float(original["info_bits_per_event"]),
            "Conditional PI (bits/event)",
            None,
        ),
        (
            "nominal_LR_p",
            results["nominal_LR_p"].to_numpy(float),
            float(original["nominal_LR_p"]),
            "Nominal LRT p value",
            P_THRESHOLD,
        ),
        (
            "delta_AICc",
            results["delta_AICc_full_minus_reduced"].to_numpy(float),
            float(original["delta_AICc_full_minus_reduced"]),
            "ΔAICc (full − reduced)",
            0.0,
        ),
        (
            "preferred_phase",
            unwrap_around(results["pre_phase_preferred_deg"], phase_center),
            phase_center,
            "Preferred precession phase (°)",
            None,
        ),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(180 / 25.4, 125 / 25.4))
    figure.subplots_adjust(
        left=0.10, right=0.985, bottom=0.17, top=0.94, hspace=0.42, wspace=0.28
    )
    titles = (
        "(a) Conditional PI",
        "(b) Nominal LRT p value",
        "(c) AICc comparison",
        "(d) Preferred phase",
    )
    for axis, title, (name, values, original_value, xlabel, threshold) in zip(
        axes.flat, titles, panels
    ):
        q025, q975 = np.quantile(values, [0.025, 0.975])
        axis.axvspan(q025, q975, color="#DCE6F1", alpha=0.75, zorder=0)
        axis.hist(values, bins=36, color="#7A8793", edgecolor="white", linewidth=0.35)
        axis.axvline(original_value, color="#C43C39", linewidth=1.5, zorder=3)
        if threshold is not None:
            axis.axvline(threshold, color="#222222", linestyle="--", linewidth=1.0)
        if name == "nominal_LR_p":
            axis.text(
                0.97,
                0.92,
                f"p < 0.05: {np.mean(values < P_THRESHOLD):.1%}",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=8.5,
            )
            axis.set_xlim(left=0)
        elif name == "delta_AICc":
            axis.text(
                0.97,
                0.92,
                f"ΔAICc < 0: {np.mean(values < 0):.1%}",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=8.5,
            )
        axis.set_title(title, loc="left")
        axis.set_xlabel(xlabel)
        if axis in axes[:, 0]:
            axis.set_ylabel("Realizations")
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)

    handles = [
        Line2D([0], [0], color="#DCE6F1", linewidth=8, label="MC 95% interval"),
        Line2D(
            [0], [0], color="#C43C39", linewidth=1.5, label="Original event sequence"
        ),
        Line2D(
            [0],
            [0],
            color="#222222",
            linestyle="--",
            linewidth=1.0,
            label="Decision threshold",
        ),
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )
    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_FIG_DIR / f"{FIGURE_STEM}.png"
    pdf_path = OUT_FIG_DIR / f"{FIGURE_STEM}.pdf"
    figure.savefig(
        pdf_path,
        metadata={
            "Title": "MIS 6 event-age sensitivity of predictive information",
            "Subject": "Conditional PI across A+ age realizations",
            "Creator": Path(__file__).name,
        },
    )
    figure.savefig(png_path, dpi=PNG_DPI, metadata={"Software": Path(__file__).name})
    plt.close(figure)
    return png_path, pdf_path


def write_outputs(
    realization_results: pd.DataFrame,
    summary: pd.DataFrame,
    parameters: pd.DataFrame,
    output_dir: Path = OUT_DATA_DIR,
) -> None:
    """Write the compact realization metrics, summary, and run metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    realization_results.to_csv(
        output_dir / REALIZATION_OUTPUT.name,
        index=False,
        float_format="%.9g",
    )
    summary.to_csv(
        output_dir / SUMMARY_OUTPUT.name,
        index=False,
        float_format="%.9g",
    )
    parameters.to_csv(output_dir / PARAMETER_OUTPUT.name, index=False)


def main() -> None:
    """Run PI on the original ages and every accepted A+ realization."""
    original_events = mis6_pi.load_events()
    draws = load_age_realizations()
    context, original, _ = build_pi_context(original_events)
    results, diagnostics = fit_realizations(draws, context, show_progress=True)
    validate_results(results, draws)
    realization_output = compact_realization_results(results)
    summary = build_summary(results, original)
    parameters = build_parameters(context, diagnostics)

    write_outputs(realization_output, summary, parameters)
    png_path, pdf_path = plot_sensitivity(results, original)

    p_fraction = results["nominal_LR_p"].lt(P_THRESHOLD).mean()
    aicc_fraction = results["delta_AICc_full_minus_reduced"].lt(0).mean()
    print(
        f"Original PI = {float(original['info_bits_per_event']):.3f} bits/event; "
        f"MC median = {results['info_bits_per_event'].median():.3f} bits/event"
    )
    print(
        f"Robustness fractions: nominal p < 0.05 in {p_fraction:.1%}; "
        f"ΔAICc < 0 in {aicc_fraction:.1%} (neither is a new MC p value)"
    )
    print(f"Wrote data to {OUT_DATA_DIR.relative_to(PROJECT_ROOT)}")
    print(
        f"Wrote figures to {png_path.relative_to(PROJECT_ROOT)} and "
        f"{pdf_path.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
