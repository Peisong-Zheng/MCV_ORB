#!/usr/bin/env python3
"""Structural sensitivity of pooled NGRIP--MIS 6 predictive information.

The 30-cell design grid varies recent-event history, bin width, and the bin
origin while keeping the same 173-kyr common response support.  This isolates
analysis-design choices from changes in exposure.  A separate two-row check
compares the common core with the maximal support available to the main
1.5-kyr-history analysis.

All event counting, history construction, predictor scaling, and Poisson
models are supplied by :mod:`toolbox.combined_pi` so the sensitivity analysis
uses exactly the same scientific contract as the main analysis.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
from paper_figure_export import copy_pdf_to_paper

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from toolbox import combined_pi
from toolbox.model_stats import nested_likelihood_metrics
from toolbox.project_config import CO2_XLSX, LR04_XLSX, PRE_TXT, PROJECT_ROOT


RUN_NAME = "NGRIP_MIS6_PI_design_sensitivity"
OUT_DATA_DIR = PROJECT_ROOT / "data/processed" / RUN_NAME
OUT_FIG_DIR = PROJECT_ROOT / "figures" / RUN_NAME

HISTORY_WINDOWS_KYR = (1.0, 1.5, 2.0, 3.0, 5.0)
BIN_WIDTHS_KYR = (0.1, 0.2, 0.5)
ORIGIN_FRACTIONS = (0.0, 0.5)
DESIGN_RESPONSE_MODE = "common_core"
SUPPORT_RESPONSE_MODES = ("common_core", "maximal_for_history")

PRIMARY_HISTORY_WINDOW_KYR = 1.5
PRIMARY_BIN_WIDTH_KYR = 0.2
PRIMARY_ORIGIN_FRACTION = 0.0
MAIN_ANALYSIS_RESPONSE_MODE = "maximal_for_history"

PHASE_INTERACTION_TERMS = (
    "mis6_x_pre_phase_sin",
    "mis6_x_pre_phase_cos",
)

BIN_COLORS = {0.1: "#0072B2", 0.2: "#D55E00", 0.5: "#009E73"}
ORIGIN_LINESTYLES = {0.0: "-", 0.5: (0, (4, 2))}
ORIGIN_MARKERS = {0.0: "o", 0.5: "s"}


def configure_plot_style() -> None:
    """Use readable journal-scale typography and editable PDF fonts."""

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9.5,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.1,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def _fit_row(
    events: pd.DataFrame,
    *,
    history_window_kyr: float,
    bin_width_kyr: float,
    origin_fraction: float,
    response_mode: str,
) -> dict[str, object]:
    """Fit one design and retain only interpretable diagnostics."""

    context = combined_pi.build_context(
        history_window_ka=history_window_kyr,
        bin_width_ka=bin_width_kyr,
        origin_fraction=origin_fraction,
        response_mode=response_mode,
    )
    fit = combined_pi.fit_catalogue(events, context)
    summary = fit.summary
    intervals = {
        segment_id: context.segments[segment_id]
        for segment_id in combined_pi.SEGMENT_IDS
    }

    # Response exposure is the union of the two segments; their intervening gap
    # is never represented by a bin.
    gap_start = intervals["NGRIP"].response_end_kyr_bp
    gap_end = intervals["MIS6"].response_start_kyr_bp
    gap_counted = (
        context.response_bins["bin_center_kyr_bp"]
        .between(gap_start, gap_end, inclusive="neither")
        .any()
    )

    return {
        "history_window_kyr": float(history_window_kyr),
        "bin_width_kyr": float(bin_width_kyr),
        "origin_fraction": float(origin_fraction),
        "response_mode": response_mode,
        "n_source_events": int(summary["n_source_events"]),
        "n_predictive_events": int(summary["n_predictive_events"]),
        "n_predictive_bins": int(summary["n_predictive_bins"]),
        "response_exposure_kyr": float(summary["response_exposure_kyr"]),
        "LR_statistic": float(summary["LR_statistic"]),
        "nominal_LR_p": float(summary["nominal_LR_p"]),
        "info_bits_per_event": float(summary["info_bits_per_event"]),
        "delta_AICc_full_minus_reduced": float(
            summary["delta_AICc_full_minus_reduced"]
        ),
        "pre_phase_preferred_deg": float(summary["pre_phase_preferred_deg"]),
        "pre_phase_rate_ratio_max_vs_min": float(
            summary["pre_phase_rate_ratio_max_vs_min"]
        ),
        "all_models_converged": bool(summary["all_models_converged"]),
        "likelihood_nesting_ok": bool(summary["likelihood_nesting_ok"]),
        "eta_clipping_used": bool(summary["eta_clipping_used"]),
        "gap_counted_as_exposure": bool(gap_counted),
        "ngrip_response_start_kyr_bp": intervals["NGRIP"].response_start_kyr_bp,
        "ngrip_response_end_kyr_bp": intervals["NGRIP"].response_end_kyr_bp,
        "mis6_response_start_kyr_bp": intervals["MIS6"].response_start_kyr_bp,
        "mis6_response_end_kyr_bp": intervals["MIS6"].response_end_kyr_bp,
    }


def _is_primary_design(frame: pd.DataFrame) -> pd.Series:
    """Identify the pre-specified 1.5-kyr/0.2-kyr/unshifted design."""

    return (
        np.isclose(frame["history_window_kyr"], PRIMARY_HISTORY_WINDOW_KYR)
        & np.isclose(frame["bin_width_kyr"], PRIMARY_BIN_WIDTH_KYR)
        & np.isclose(frame["origin_fraction"], PRIMARY_ORIGIN_FRACTION)
    )


def run_design_sensitivity(events: pd.DataFrame | None = None) -> pd.DataFrame:
    """Fit the pre-specified 5 x 3 x 2 grid on the common 173-kyr core."""

    catalogue = combined_pi.load_event_catalogue() if events is None else events
    rows = [
        _fit_row(
            catalogue,
            history_window_kyr=history,
            bin_width_kyr=bin_width,
            origin_fraction=origin,
            response_mode=DESIGN_RESPONSE_MODE,
        )
        for history, bin_width, origin in product(
            HISTORY_WINDOWS_KYR, BIN_WIDTHS_KYR, ORIGIN_FRACTIONS
        )
    ]
    result = pd.DataFrame(rows)
    result.insert(
        0, "design_id", [f"design_{i:02d}" for i in range(1, len(result) + 1)]
    )
    result.insert(4, "is_primary_design", _is_primary_design(result))
    _validate_design_sensitivity(result)
    return result


def run_support_sensitivity(events: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compare fixed common-core and maximal support for the main design."""

    catalogue = combined_pi.load_event_catalogue() if events is None else events
    result = pd.DataFrame(
        [
            _fit_row(
                catalogue,
                history_window_kyr=PRIMARY_HISTORY_WINDOW_KYR,
                bin_width_kyr=PRIMARY_BIN_WIDTH_KYR,
                origin_fraction=PRIMARY_ORIGIN_FRACTION,
                response_mode=response_mode,
            )
            for response_mode in SUPPORT_RESPONSE_MODES
        ]
    )
    result.insert(
        4,
        "is_main_analysis_support",
        result["response_mode"].eq(MAIN_ANALYSIS_RESPONSE_MODE),
    )
    _validate_support_sensitivity(result)
    return result


def _phase_summary(beta_sin: float, beta_cos: float) -> tuple[float, float]:
    """Return preferred phase and maximum-to-minimum rate ratio."""

    amplitude = float(np.hypot(beta_sin, beta_cos))
    phase = float(np.degrees(np.mod(np.arctan2(beta_sin, beta_cos), 2 * np.pi)))
    return phase, float(np.exp(2 * amplitude))


def run_pooling_diagnostic(events: pd.DataFrame | None = None) -> pd.DataFrame:
    """Test whether NGRIP and MIS 6 require different phase coefficients."""

    catalogue = combined_pi.load_event_catalogue() if events is None else events
    context = combined_pi.build_context()
    common_fit = combined_pi.fit_catalogue(catalogue, context)
    response = common_fit.response.copy()
    response[PHASE_INTERACTION_TERMS[0]] = (
        response[combined_pi.SEGMENT_TERM] * response["pre_phase_sin"]
    )
    response[PHASE_INTERACTION_TERMS[1]] = (
        response[combined_pi.SEGMENT_TERM] * response["pre_phase_cos"]
    )
    heterogeneous_terms = combined_pi.FULL_TERMS + PHASE_INTERACTION_TERMS
    heterogeneous = combined_pi.fit_response_terms(response, heterogeneous_terms)

    metrics = nested_likelihood_metrics(
        loglik_full=heterogeneous.log_likelihood,
        loglik_reduced=common_fit.full.log_likelihood,
        df=2,
        n_bins=len(response),
        n_events=int(response["event_count"].sum()),
        aicc_full=heterogeneous.aicc,
        aicc_reduced=common_fit.full.aicc,
    )
    beta = dict(zip(heterogeneous_terms, heterogeneous.beta[1:]))
    ngrip_phase, ngrip_ratio = _phase_summary(
        float(beta["pre_phase_sin"]), float(beta["pre_phase_cos"])
    )
    mis6_phase, mis6_ratio = _phase_summary(
        float(beta["pre_phase_sin"] + beta[PHASE_INTERACTION_TERMS[0]]),
        float(beta["pre_phase_cos"] + beta[PHASE_INTERACTION_TERMS[1]]),
    )
    clipping = sum(
        (
            heterogeneous.n_eta_clipped_low,
            heterogeneous.n_eta_clipped_high,
            common_fit.full.n_eta_clipped_low,
            common_fit.full.n_eta_clipped_high,
        )
    )
    result = pd.DataFrame(
        [
            {
                "comparison": "segment-specific versus common precession response",
                "n_events": int(response["event_count"].sum()),
                "response_exposure_kyr": context.response_exposure_kyr,
                "common_model_loglik": common_fit.full.log_likelihood,
                "segment_specific_model_loglik": heterogeneous.log_likelihood,
                "LR_statistic": metrics["LR_statistic"],
                "df": metrics["df"],
                "nominal_LR_p": metrics["LR_p_value"],
                "info_bits_per_event_for_interaction": metrics["info_bits_per_event"],
                "delta_AICc_segment_specific_minus_common": metrics[
                    "delta_AICc_full_minus_reduced"
                ],
                "ngrip_preferred_phase_deg": ngrip_phase,
                "mis6_preferred_phase_deg": mis6_phase,
                "ngrip_phase_rate_ratio_max_vs_min": ngrip_ratio,
                "mis6_phase_rate_ratio_max_vs_min": mis6_ratio,
                "both_models_converged": bool(
                    common_fit.full.converged and heterogeneous.converged
                ),
                "likelihood_nesting_ok": bool(metrics["ll_gain_nats"] >= -1e-8),
                "eta_clipping_used": bool(clipping),
            }
        ]
    )
    if not bool(result.loc[0, "both_models_converged"]):
        raise RuntimeError("A pooling-diagnostic model did not converge")
    if not bool(result.loc[0, "likelihood_nesting_ok"]):
        raise RuntimeError("The segment-specific phase model is not nested")
    if bool(result.loc[0, "eta_clipping_used"]):
        raise RuntimeError("The pooling diagnostic reached an eta clip bound")
    return result


def _validate_fit_flags(frame: pd.DataFrame) -> None:
    """Reject failed fits rather than silently plotting them."""

    finite_columns = (
        "LR_statistic",
        "nominal_LR_p",
        "info_bits_per_event",
        "delta_AICc_full_minus_reduced",
        "pre_phase_preferred_deg",
        "pre_phase_rate_ratio_max_vs_min",
    )
    if not np.isfinite(frame.loc[:, finite_columns].to_numpy(float)).all():
        raise RuntimeError("At least one design produced a non-finite PI diagnostic")
    if not frame["all_models_converged"].all():
        raise RuntimeError("At least one design did not converge")
    if not frame["likelihood_nesting_ok"].all():
        raise RuntimeError(
            "At least one full-model likelihood is below its reduced model"
        )
    if frame["eta_clipping_used"].any():
        raise RuntimeError("At least one fitted linear predictor reached a clip bound")
    if frame["gap_counted_as_exposure"].any():
        raise RuntimeError("The NGRIP--MIS 6 gap was incorrectly counted as exposure")


def _validate_design_sensitivity(result: pd.DataFrame) -> None:
    """Check the full factorial grid and its common scientific support."""

    expected = set(product(HISTORY_WINDOWS_KYR, BIN_WIDTHS_KYR, ORIGIN_FRACTIONS))
    observed = set(
        result.loc[
            :, ["history_window_kyr", "bin_width_kyr", "origin_fraction"]
        ].itertuples(index=False, name=None)
    )
    if len(result) != 30 or observed != expected:
        raise RuntimeError("The structural sensitivity must contain all 30 designs")
    if result["is_primary_design"].sum() != 1:
        raise RuntimeError("Exactly one pre-specified primary design is required")
    if not result["response_mode"].eq(DESIGN_RESPONSE_MODE).all():
        raise RuntimeError("Every design-grid fit must use the common response core")
    if not np.allclose(result["response_exposure_kyr"], 173.0, atol=1e-9):
        raise RuntimeError("Every design-grid fit must retain 173 kyr of exposure")
    if (
        not result["n_source_events"].eq(55).all()
        or not result["n_predictive_events"].eq(55).all()
    ):
        raise RuntimeError("Every design-grid fit must retain all 55 events")
    _validate_fit_flags(result)


def _validate_support_sensitivity(result: pd.DataFrame) -> None:
    """Check the two endpoint definitions at the pre-specified design."""

    if len(result) != 2 or set(result["response_mode"]) != set(SUPPORT_RESPONSE_MODES):
        raise RuntimeError("Endpoint sensitivity requires common and maximal support")
    if result["is_main_analysis_support"].sum() != 1:
        raise RuntimeError("Exactly one endpoint row must be marked as main support")
    if (
        not result["n_source_events"].eq(55).all()
        or not result["n_predictive_events"].eq(55).all()
    ):
        raise RuntimeError("Both endpoint fits must retain all 55 events")
    exposure = result.set_index("response_mode")["response_exposure_kyr"]
    if not np.isclose(exposure["common_core"], 173.0) or not np.isclose(
        exposure["maximal_for_history"], 180.0
    ):
        raise RuntimeError("Unexpected response exposure in endpoint sensitivity")
    _validate_fit_flags(result)


def build_parameters(
    design: pd.DataFrame,
    support: pd.DataFrame,
    pooling: pd.DataFrame,
) -> pd.DataFrame:
    """Record design choices, scientific units, and input provenance."""

    rows = [
        (
            "event_catalogue",
            str(combined_pi.EVENT_CATALOGUE_CSV.relative_to(PROJECT_ROOT)),
            "",
            "34 NGRIP warming starts + 21 MIS 6 transitions",
        ),
        (
            "observation_segments",
            str(combined_pi.OBSERVATION_SEGMENTS_CSV.relative_to(PROJECT_ROOT)),
            "",
            "two disjoint observed segments",
        ),
        (
            "history_windows",
            ";".join(map(str, HISTORY_WINDOWS_KYR)),
            "kyr",
            "pre-specified sensitivity grid",
        ),
        (
            "bin_widths",
            ";".join(map(str, BIN_WIDTHS_KYR)),
            "kyr",
            "pre-specified sensitivity grid",
        ),
        (
            "origin_fractions",
            ";".join(map(str, ORIGIN_FRACTIONS)),
            "bin width",
            "unshifted and half-bin-shifted grids",
        ),
        (
            "design_response_mode",
            DESIGN_RESPONSE_MODE,
            "",
            "same support for all 30 design fits",
        ),
        ("design_response_exposure", 173.0, "kyr", "106 kyr NGRIP + 67 kyr MIS 6"),
        (
            "number_of_designs",
            len(design),
            "fits",
            "5 history windows x 3 bin widths x 2 origins",
        ),
        (
            "primary_history_window",
            PRIMARY_HISTORY_WINDOW_KYR,
            "kyr",
            "based on the characteristic D-O recurrence timescale",
        ),
        (
            "primary_bin_width",
            PRIMARY_BIN_WIDTH_KYR,
            "kyr",
            "main-analysis event-count bins",
        ),
        (
            "primary_origin_fraction",
            PRIMARY_ORIGIN_FRACTION,
            "bin width",
            "unshifted main grid",
        ),
        (
            "main_analysis_response_mode",
            MAIN_ANALYSIS_RESPONSE_MODE,
            "",
            "maximal support with complete 1.5-kyr history",
        ),
        (
            "main_analysis_response_exposure",
            float(
                support.loc[
                    support["is_main_analysis_support"], "response_exposure_kyr"
                ].iloc[0]
            ),
            "kyr",
            "endpoint sensitivity comparison",
        ),
        (
            "reduced_model_terms",
            "+".join(combined_pi.REDUCED_TERMS),
            "",
            "fixed PI baseline",
        ),
        (
            "full_model_terms",
            "+".join(combined_pi.FULL_TERMS),
            "",
            "reduced model + precession sine/cosine",
        ),
        (
            "likelihood_p_value",
            "asymptotic chi-square",
            "df=2",
            "nominal; empirical calibration is separate",
        ),
        ("resolution_covariate_included", False, "", "not part of the PI model"),
        (
            "age_uncertainty_propagated",
            False,
            "",
            "point ages; Monte Carlo sensitivity is separate",
        ),
        (
            "lr04_input",
            str(LR04_XLSX.relative_to(PROJECT_ROOT)),
            "",
            "LR04 benthic stack",
        ),
        (
            "co2_input",
            str(CO2_XLSX.relative_to(PROJECT_ROOT)),
            "",
            "composite atmospheric CO2",
        ),
        (
            "precession_input",
            str(PRE_TXT.relative_to(PROJECT_ROOT)),
            "",
            "La2004 precession index",
        ),
        (
            "pooling_diagnostic",
            "common phase coefficients versus MIS6 phase interactions",
            "",
            f"nominal p={pooling.loc[0, 'nominal_LR_p']:.6g}; df=2",
        ),
    ]
    return pd.DataFrame(rows, columns=["parameter", "value", "unit", "note"])


def _add_panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.15,
        1.05,
        label,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )


def _plot_design_metric(
    axis: plt.Axes,
    design: pd.DataFrame,
    column: str,
    ylabel: str,
) -> None:
    """Plot all six bin-width/origin series against history length."""

    for bin_width in BIN_WIDTHS_KYR:
        for origin in ORIGIN_FRACTIONS:
            values = design.loc[
                np.isclose(design["bin_width_kyr"], bin_width)
                & np.isclose(design["origin_fraction"], origin)
            ].sort_values("history_window_kyr")
            axis.plot(
                values["history_window_kyr"],
                values[column],
                color=BIN_COLORS[bin_width],
                linestyle=ORIGIN_LINESTYLES[origin],
                marker=ORIGIN_MARKERS[origin],
                markersize=4.4,
                markerfacecolor="white" if origin else BIN_COLORS[bin_width],
                markeredgewidth=0.9,
                linewidth=1.25,
                label=f"{bin_width:g} kyr; origin {origin:g}",
            )

    primary = design.loc[design["is_primary_design"]].iloc[0]
    axis.scatter(
        [primary["history_window_kyr"]],
        [primary[column]],
        marker="*",
        s=90,
        facecolor="#F0E442",
        edgecolor="#222222",
        linewidth=0.8,
        zorder=5,
        label="Primary design",
    )
    axis.set_xticks(HISTORY_WINDOWS_KYR)
    axis.set_xlabel("Event-history window (kyr)")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.55)
    axis.spines[["top", "right"]].set_visible(False)


def plot_sensitivity(design: pd.DataFrame, support: pd.DataFrame) -> plt.Figure:
    """Compose a compact four-panel structural-sensitivity figure."""

    configure_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(180 / 25.4, 132 / 25.4))
    fig.subplots_adjust(
        left=0.09, right=0.98, bottom=0.11, top=0.88, wspace=0.34, hspace=0.48
    )

    _plot_design_metric(
        axes[0, 0], design, "info_bits_per_event", "PI (bits event$^{-1}$)"
    )
    _plot_design_metric(axes[0, 1], design, "nominal_LR_p", "Nominal p value")
    axes[0, 1].set_yscale("log")
    axes[0, 1].axhline(0.05, color="#666666", linestyle=":", linewidth=0.9)
    axes[0, 1].set_ylim(5e-4, 7e-2)

    _plot_design_metric(
        axes[1, 0], design, "pre_phase_preferred_deg", "Preferred phase (°)"
    )

    support_plot = support.copy()
    support_plot["label"] = support_plot["response_mode"].map(
        {"common_core": "Common core", "maximal_for_history": "Maximal support"}
    )
    colors = ["#B7B7B7", "#3E6C8E"]
    bars = axes[1, 1].bar(
        support_plot["label"],
        support_plot["info_bits_per_event"],
        width=0.62,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
    )
    upper = float(support_plot["info_bits_per_event"].max()) * 1.34
    axes[1, 1].set_ylim(0.0, upper)
    axes[1, 1].set_ylabel("PI (bits event$^{-1}$)")
    axes[1, 1].set_xlabel("Response support")
    axes[1, 1].grid(axis="y", color="#D9D9D9", linewidth=0.55)
    axes[1, 1].spines[["top", "right"]].set_visible(False)
    for bar, row in zip(bars, support_plot.itertuples(index=False)):
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + upper * 0.025,
            f"{row.response_exposure_kyr:.0f} kyr\np = {row.nominal_LR_p:.3g}",
            ha="center",
            va="bottom",
            fontsize=8.2,
        )

    for label, axis in zip("abcd", axes.flat):
        _add_panel_label(axis, label)

    # One shared legend keeps the six design series out of the data panels.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.985),
        frameon=False,
        columnspacing=1.25,
        handletextpad=0.5,
    )
    return fig


def write_outputs(
    design: pd.DataFrame,
    support: pd.DataFrame,
    pooling: pd.DataFrame,
    output_dir: Path = OUT_DATA_DIR,
) -> None:
    """Save compact result and provenance tables."""

    output_dir.mkdir(parents=True, exist_ok=True)
    design.to_csv(
        output_dir / "design_sensitivity.csv", index=False, float_format="%.9g"
    )
    support.to_csv(
        output_dir / "support_sensitivity.csv", index=False, float_format="%.9g"
    )
    pooling.to_csv(
        output_dir / "pooling_diagnostic.csv", index=False, float_format="%.9g"
    )
    build_parameters(design, support, pooling).to_csv(
        output_dir / "parameters_and_provenance.csv",
        index=False,
        float_format="%.9g",
    )


def save_figure(fig: plt.Figure, output_dir: Path = OUT_FIG_DIR) -> tuple[Path, Path]:
    """Save a high-resolution review PNG and editable vector PDF."""

    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{RUN_NAME}.png"
    pdf = output_dir / f"{RUN_NAME}.pdf"
    fig.savefig(png, dpi=600, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    copy_pdf_to_paper(pdf)
    plt.close(fig)
    return png, pdf


def main() -> None:
    events = combined_pi.load_event_catalogue()
    design = run_design_sensitivity(events)
    support = run_support_sensitivity(events)
    pooling = run_pooling_diagnostic(events)
    write_outputs(design, support, pooling)
    png, pdf = save_figure(plot_sensitivity(design, support))

    primary = design.loc[design["is_primary_design"]].iloc[0]
    main_support = support.loc[support["is_main_analysis_support"]].iloc[0]
    print(
        f"Structural sensitivity: {len(design)} designs, "
        f"PI={design.info_bits_per_event.min():.3f}--"
        f"{design.info_bits_per_event.max():.3f} bits/event, "
        f"phase={design.pre_phase_preferred_deg.min():.1f}--"
        f"{design.pre_phase_preferred_deg.max():.1f}°"
    )
    print(
        f"Primary design on common core: PI={primary.info_bits_per_event:.4f}, "
        f"nominal p={primary.nominal_LR_p:.4g}"
    )
    print(
        f"Main maximal support: {main_support.response_exposure_kyr:.0f} kyr, "
        f"PI={main_support.info_bits_per_event:.4f}, "
        f"nominal p={main_support.nominal_LR_p:.4g}"
    )
    print(
        "Segment-specific phase response: "
        f"nominal p={pooling.loc[0, 'nominal_LR_p']:.4g}"
    )
    print(f"Wrote tables to {OUT_DATA_DIR.relative_to(PROJECT_ROOT)}")
    print(
        f"Wrote figures to {png.relative_to(PROJECT_ROOT)} and "
        f"{pdf.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
