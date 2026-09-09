#!/usr/bin/env python3
"""Point-age phase analysis of the pooled NGRIP--MIS 6 event catalogue.

The catalogue contains 34 published NGRIP warming starts and 21 published
MIS 6 speleothem transitions.  NGRIP and MIS 6 are treated as two disjoint
observation segments; the age gap between them is never counted as exposure.

Rayleigh's test is reported as a descriptive phase-concentration check.  The
main analysis is a nested Poisson comparison: the reduced model includes a
1.5-kyr event-history term, LR04, CO2, and a record-segment intercept, while
the full model adds sine and cosine of precession phase.  The segment term
allows MIS 6 and NGRIP to have different conditional baseline event rates.
Sampling resolution and age uncertainty are not included in this point-age
analysis; uncertainty is handled by the separate Monte Carlo experiment.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from toolbox import combined_pi
from toolbox.orbital_phase import rayleigh_rbar_threshold, rayleigh_test
from toolbox.project_config import (
    CO2_XLSX,
    LR04_XLSX,
    ORBITAL_AGE_OFFSET_TO_BP1950_KA,
    ORBITAL_REFERENCE,
    ORBITAL_SOLUTION,
    ORBITAL_SOURCE_EPOCH,
    PRE_TXT,
    PROJECT_ROOT,
)


RUN_NAME = "NGRIP_MIS6_event_phase_analysis"
OUT_DATA_DIR = PROJECT_ROOT / "data/processed" / RUN_NAME
OUT_FIG_DIR = PROJECT_ROOT / "figures" / RUN_NAME

CATALOGUE_LABEL = "NGRIP warming + MIS 6 transitions"
EVENT_TYPE = "warming_transition"
SEGMENT_COLORS = {"NGRIP": "#D55E00", "MIS6": "#0072B2"}

HISTORY_WINDOW_KYR = combined_pi.DEFAULT_HISTORY_WINDOW_KA
BIN_WIDTH_KYR = combined_pi.DEFAULT_BIN_WIDTH_KA
BIN_ORIGIN_FRACTION = combined_pi.DEFAULT_ORIGIN_FRACTION
RESPONSE_MODE = combined_pi.DEFAULT_RESPONSE_MODE
RESOLUTION_COVARIATE_INCLUDED = False


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
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def run_analysis() -> dict[str, object]:
    """Run the point-age Rayleigh and conditional-PI calculations."""

    events = combined_pi.load_event_catalogue()
    context = combined_pi.build_context(
        history_window_ka=HISTORY_WINDOW_KYR,
        bin_width_ka=BIN_WIDTH_KYR,
        origin_fraction=BIN_ORIGIN_FRACTION,
        response_mode=RESPONSE_MODE,
    )
    fit = combined_pi.fit_catalogue(events, context)
    event_phases = combined_pi.sample_event_phases(events)
    rayleigh = rayleigh_test(event_phases["pre_phase_rad"].to_numpy(float))

    _validate_results(events, context, fit, event_phases, rayleigh)
    return {
        "events": events,
        "context": context,
        "fit": fit,
        "event_phases": event_phases,
        "rayleigh": rayleigh,
    }


def _validate_results(
    events: pd.DataFrame,
    context: combined_pi.PIContext,
    fit: combined_pi.CombinedPIFit,
    event_phases: pd.DataFrame,
    rayleigh: dict[str, float],
) -> None:
    """Check the scientific invariants of the main point analysis."""

    if events.groupby("segment_id").size().to_dict() != {"MIS6": 21, "NGRIP": 34}:
        raise RuntimeError("The main catalogue must contain 34 NGRIP and 21 MIS 6 events")
    if len(event_phases) != len(events) or event_phases["pre_phase_extrapolated"].any():
        raise RuntimeError("Every event needs a non-extrapolated precession phase")
    if int(fit.summary["n_predictive_events"]) != len(events):
        raise RuntimeError("Every published event must lie within the response support")
    if not context.response_bins["same_type_history_complete"].all():
        raise RuntimeError("At least one response bin lacks complete event history")

    diagnostics = (
        "LR_statistic",
        "nominal_LR_p",
        "info_bits_per_event",
        "delta_AICc_full_minus_reduced",
        "pre_phase_preferred_deg",
        "pre_phase_rate_ratio_max_vs_min",
        "mis6_vs_ngrip_rate_ratio_full",
    )
    if not np.isfinite([fit.summary[name] for name in diagnostics]).all():
        raise RuntimeError("Conditional-PI results contain a non-finite value")
    if not fit.summary["all_models_converged"]:
        raise RuntimeError("At least one Poisson model did not converge")
    if not fit.summary["likelihood_nesting_ok"]:
        raise RuntimeError("The full-model likelihood is below the reduced model")
    if fit.summary["eta_clipping_used"]:
        raise RuntimeError("A fitted linear predictor reached a numerical clip bound")
    if not np.isfinite([rayleigh["mean_phase_deg"], rayleigh["rayleigh_p"]]).all():
        raise RuntimeError("Rayleigh results contain a non-finite value")


def build_analysis_summary(result: dict[str, object]) -> pd.DataFrame:
    """Collect the point-age results used in the paper workflow."""

    events = result["events"]
    fit = result["fit"]
    rayleigh = result["rayleigh"]
    row = {
        **fit.summary,
        "catalogue_label": CATALOGUE_LABEL,
        "n_ngrip_events": int(events["segment_id"].eq("NGRIP").sum()),
        "n_mis6_events": int(events["segment_id"].eq("MIS6").sum()),
        "rayleigh_role": "descriptive",
        "rayleigh_mean_phase_deg": float(rayleigh["mean_phase_deg"]),
        "rayleigh_mean_resultant_length": float(rayleigh["mean_resultant_length"]),
        "rayleigh_R": float(rayleigh["rayleigh_R"]),
        "rayleigh_z": float(rayleigh["rayleigh_z"]),
        "rayleigh_p": float(rayleigh["rayleigh_p"]),
        "rayleigh_significant_0p05": float(rayleigh["rayleigh_p"]) < 0.05,
        "pi_role": "primary",
        "resolution_covariate_included": RESOLUTION_COVARIATE_INCLUDED,
        "age_uncertainty_propagated": False,
        "event_membership": "published events only",
    }
    return pd.DataFrame([row])


def build_coefficients(fit: combined_pi.CombinedPIFit) -> pd.DataFrame:
    """Add concise scientific definitions to the fitted coefficients."""

    notes = {
        "intercept": "conditional log event rate per kyr",
        "same_type_history_count": "older events in the preceding 1.5 kyr",
        "lr04_scaled": "LR04 anomaly divided by its response-range span",
        "co2_scaled": "CO2 anomaly divided by its response-range span",
        "mis6_segment": "MIS 6 = 1 and NGRIP = 0; conditional segment contrast",
        "pre_phase_sin": "sine of BP1950-corrected La2004 precession phase",
        "pre_phase_cos": "cosine of BP1950-corrected La2004 precession phase",
    }
    table = combined_pi.coefficient_table(fit)
    table["term_definition"] = table["term"].map(notes)
    return table


def build_parameters(result: dict[str, object]) -> pd.DataFrame:
    """Record model choices, observation support, units, and provenance."""

    events = result["events"]
    context = result["context"]
    response = context.response_bins
    rows: list[tuple[str, object, str, str]] = [
        ("age_unit", "kyr BP", "BP1950", "used for every event and forcing"),
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
        (
            "all_event_identities_published",
            True,
            "",
            "MIS 6 operational transition ages are estimated here",
        ),
        ("history_window", HISTORY_WINDOW_KYR, "kyr", "characteristic D-O recurrence timescale"),
        ("bin_width", BIN_WIDTH_KYR, "kyr", "Poisson event-count bins"),
        ("bin_origin_fraction", BIN_ORIGIN_FRACTION, "bin width", "unshifted main grid"),
        ("response_mode", RESPONSE_MODE, "", "maximal support with complete event history"),
        ("response_exposure", context.response_exposure_kyr, "kyr", "sum of both segments only"),
        ("response_gap_counted_as_exposure", False, "", "the NGRIP--MIS 6 gap is omitted"),
        ("reduced_model_terms", "+".join(combined_pi.REDUCED_TERMS), "", "PI baseline"),
        (
            "full_model_terms",
            "+".join(combined_pi.FULL_TERMS),
            "",
            "adds precession sine and cosine",
        ),
        (
            "segment_indicator",
            "MIS6=1; NGRIP=0",
            "",
            "allows different conditional baseline event rates",
        ),
        ("resolution_covariate_included", False, "", "not part of the PI model"),
        ("event_age_uncertainty_propagated", False, "", "point-age analysis"),
        ("rayleigh_role", "descriptive", "", "PI is the primary analysis"),
        ("likelihood_p_value", "asymptotic chi-square", "df=2", "nominal point estimate"),
        ("lr04_input", str(LR04_XLSX.relative_to(PROJECT_ROOT)), "", "LR04 benthic stack"),
        ("co2_input", str(CO2_XLSX.relative_to(PROJECT_ROOT)), "", "composite atmospheric CO2"),
        ("precession_input", str(PRE_TXT.relative_to(PROJECT_ROOT)), "", ORBITAL_REFERENCE),
        ("orbital_solution", ORBITAL_SOLUTION, "", ORBITAL_REFERENCE),
        ("orbital_source_epoch", ORBITAL_SOURCE_EPOCH, "", "source-file age convention"),
        (
            "orbital_age_offset_to_BP1950",
            ORBITAL_AGE_OFFSET_TO_BP1950_KA,
            "kyr",
            "applied before phase interpolation",
        ),
    ]

    for segment_id, segment in context.segments.items():
        n_events = int(events["segment_id"].eq(segment_id).sum())
        rows.extend(
            [
                (f"{segment_id}_event_count", n_events, "events", "published transitions"),
                (
                    f"{segment_id}_observation_interval",
                    f"{segment.observation_start_kyr_bp:g}--{segment.observation_end_kyr_bp:g}",
                    "kyr BP",
                    "including history-only support",
                ),
                (
                    f"{segment_id}_response_interval",
                    f"{segment.response_start_kyr_bp:g}--{segment.response_end_kyr_bp:g}",
                    "kyr BP",
                    "included as event exposure",
                ),
            ]
        )

    forcing_units = {"lr04": "per mil", "co2": "ppm"}
    for forcing in ("lr04", "co2"):
        values = response[forcing].to_numpy(float)
        rows.extend(
            [
                (
                    f"{forcing}_response_mean",
                    float(values.mean()),
                    forcing_units[forcing],
                    "zero point used in scaling",
                ),
                (
                    f"{forcing}_response_range",
                    float(np.ptp(values)),
                    forcing_units[forcing],
                    "divisor used in scaling",
                ),
            ]
        )
    return pd.DataFrame(rows, columns=["parameter", "value", "unit", "note"])


def _add_panel_label(axis: plt.Axes, label: str, *, x: float = -0.12) -> None:
    axis.text(
        x,
        1.08,
        label,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontweight="bold",
        fontsize=11,
    )


def _plot_segment_timeline(
    axis: plt.Axes,
    segment_id: str,
    result: dict[str, object],
) -> None:
    """Plot one observed segment without implying exposure across the gap."""

    context = result["context"]
    event_phases = result["event_phases"]
    segment = context.segments[segment_id]
    bins = context.bins.loc[context.bins["segment_id"].eq(segment_id)]
    phases = event_phases.loc[event_phases["segment_id"].eq(segment_id)]
    color = SEGMENT_COLORS[segment_id]

    axis.axvspan(
        segment.response_end_kyr_bp,
        segment.observation_end_kyr_bp,
        color="#E6E6E6",
        lw=0,
        zorder=0,
    )
    axis.plot(
        bins["bin_center_kyr_bp"],
        bins["precession_index"],
        color="#555555",
        lw=1.05,
        zorder=1,
    )
    axis.scatter(
        phases[combined_pi.EVENT_AGE_COLUMN],
        phases["precession_index"],
        s=21,
        color=color,
        edgecolor="white",
        linewidth=0.35,
        zorder=3,
    )
    n_events = len(phases)
    label = "warming starts" if segment_id == "NGRIP" else "transitions"
    axis.text(
        0.98,
        0.94,
        f"{segment_id} ({n_events} {label})",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )
    axis.set_xlim(segment.observation_start_kyr_bp, segment.observation_end_kyr_bp)
    axis.set_xlabel("Age (kyr BP)")
    axis.grid(axis="y", color="#D9D9D9", lw=0.55)
    axis.spines[["top", "right"]].set_visible(False)


def _mark_discontinuous_axis(left: plt.Axes, right: plt.Axes) -> None:
    """Mark the omitted gap between the two timeline axes."""

    left.spines["right"].set_visible(False)
    right.spines["left"].set_visible(False)
    right.tick_params(axis="y", left=False, labelleft=False)
    size = 0.014
    line = {"color": "#333333", "clip_on": False, "lw": 0.9}
    left.plot((1 - size, 1 + size), (-size, +size), transform=left.transAxes, **line)
    left.plot((1 - size, 1 + size), (1 - size, 1 + size), transform=left.transAxes, **line)
    right.plot((-size, +size), (-size, +size), transform=right.transAxes, **line)
    right.plot((-size, +size), (1 - size, 1 + size), transform=right.transAxes, **line)


def _plot_rayleigh(axis: plt.Axes, result: dict[str, object]) -> None:
    """Draw the descriptive phase histogram and mean resultant vector."""

    phases = result["event_phases"]["pre_phase_rad"].to_numpy(float)
    rayleigh = result["rayleigh"]
    edges = np.linspace(0.0, 2.0 * np.pi, 13)
    counts, _ = np.histogram(phases, bins=edges)
    maximum = max(int(counts.max()), 1)
    threshold = rayleigh_rbar_threshold(len(phases), alpha=0.05)

    axis.bar(
        edges[:-1],
        counts,
        width=np.diff(edges)[0],
        align="edge",
        color="#7A6AA6",
        alpha=0.72,
        edgecolor="white",
        linewidth=0.7,
    )
    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    axis.plot(
        theta,
        np.full_like(theta, threshold * maximum),
        color="#555555",
        lw=1.0,
        ls=(0, (4, 2)),
    )
    axis.annotate(
        "",
        xy=(rayleigh["mean_phase_rad"], rayleigh["mean_resultant_length"] * maximum),
        xytext=(rayleigh["mean_phase_rad"], 0.0),
        arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#202020"},
    )
    axis.set_theta_zero_location("E")
    axis.set_theta_direction(1)
    axis.set_xticks([0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0])
    axis.set_xticklabels(["min", "90°", "max", "270°"])
    axis.tick_params(axis="x", pad=3)
    axis.tick_params(axis="y", labelsize=7.5)
    axis.set_title("Rayleigh test (descriptive)", pad=20, fontsize=10)
    axis.text(
        0.02,
        -0.08,
        (
            rf"$\bar{{R}}$ = {rayleigh['mean_resultant_length']:.2f}; "
            f"p = {rayleigh['rayleigh_p']:.3f}"
        ),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
    )


def _plot_phase_response(axis: plt.Axes, fit: combined_pi.CombinedPIFit) -> None:
    """Plot the fitted multiplicative contribution of precession phase."""

    beta = dict(zip(combined_pi.FULL_TERMS, fit.full.beta[1:]))
    phase_deg = np.linspace(0.0, 360.0, 721)
    multiplier = combined_pi.phase_rate_multiplier(
        phase_deg,
        beta["pre_phase_sin"],
        beta["pre_phase_cos"],
    )
    preferred = float(fit.summary["pre_phase_preferred_deg"])

    axis.plot(phase_deg, multiplier, color="#3E6C8E", lw=2.0)
    axis.axhline(1.0, color="#777777", lw=0.9, ls=":")
    axis.axvline(preferred, color="#333333", lw=1.0, ls=(0, (4, 2)))
    axis.text(
        0.04,
        0.96,
        (
            f"PI = {fit.summary['info_bits_per_event']:.3f} bits event$^{{-1}}$\n"
            f"LR = {fit.summary['LR_statistic']:.2f}; "
            f"nominal p = {fit.summary['nominal_LR_p']:.3g}\n"
            f"Preferred phase = {preferred:.1f}°\n"
            f"Max/min rate = {fit.summary['pre_phase_rate_ratio_max_vs_min']:.2f}"
        ),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2.0},
    )
    axis.set(
        xlim=(0.0, 360.0),
        xticks=[0, 90, 180, 270, 360],
        xlabel="Precession phase (°)",
        ylabel="Multiplicative contribution to event rate",
        title="Conditional predictive information",
    )
    axis.grid(color="#D9D9D9", lw=0.55)
    axis.spines[["top", "right"]].set_visible(False)


def plot_results(result: dict[str, object]) -> plt.Figure:
    """Compose one compact GRL-scale diagnostic and result figure."""

    configure_plot_style()
    fig = plt.figure(figsize=(180 / 25.4, 134 / 25.4))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=(0.88, 1.12),
        width_ratios=(0.82, 1.18),
        hspace=0.58,
        wspace=0.42,
        left=0.09,
        right=0.98,
        bottom=0.11,
        top=0.96,
    )
    timeline = grid[0, :].subgridspec(1, 2, width_ratios=(117, 72), wspace=0.07)
    ngrip_axis = fig.add_subplot(timeline[0, 0])
    mis6_axis = fig.add_subplot(timeline[0, 1], sharey=ngrip_axis)
    rayleigh_axis = fig.add_subplot(grid[1, 0], projection="polar")
    response_axis = fig.add_subplot(grid[1, 1])

    _plot_segment_timeline(ngrip_axis, "NGRIP", result)
    _plot_segment_timeline(mis6_axis, "MIS6", result)
    ngrip_axis.set_ylabel("Precession index")
    _mark_discontinuous_axis(ngrip_axis, mis6_axis)
    _plot_rayleigh(rayleigh_axis, result)
    _plot_phase_response(response_axis, result["fit"])

    _add_panel_label(ngrip_axis, "a", x=-0.08)
    _add_panel_label(rayleigh_axis, "b", x=-0.21)
    _add_panel_label(response_axis, "c", x=-0.13)
    return fig


def write_outputs(result: dict[str, object], output_dir: Path = OUT_DATA_DIR) -> None:
    """Save the compact tables needed to reproduce and interpret the result."""

    output_dir.mkdir(parents=True, exist_ok=True)
    build_analysis_summary(result).to_csv(
        output_dir / "analysis_summary.csv", index=False, float_format="%.9g"
    )
    result["event_phases"].to_csv(
        output_dir / "event_precession_phases.csv", index=False, float_format="%.9g"
    )
    build_coefficients(result["fit"]).to_csv(
        output_dir / "predictive_coefficients.csv", index=False, float_format="%.9g"
    )
    build_parameters(result).to_csv(
        output_dir / "parameters_and_provenance.csv", index=False, float_format="%.9g"
    )


def save_figure(fig: plt.Figure, output_dir: Path = OUT_FIG_DIR) -> tuple[Path, Path]:
    """Save a review PNG and an editable vector PDF."""

    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{RUN_NAME}.png"
    pdf = output_dir / f"{RUN_NAME}.pdf"
    fig.savefig(png, dpi=600, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return png, pdf


def main() -> None:
    result = run_analysis()
    write_outputs(result)
    png, pdf = save_figure(plot_results(result))
    summary = build_analysis_summary(result).iloc[0]

    print(
        f"{CATALOGUE_LABEL}: N={summary.n_predictive_events}, "
        f"PI={summary.info_bits_per_event:.4f} bits/event, "
        f"nominal p={summary.nominal_LR_p:.4g}, "
        f"phase={summary.pre_phase_preferred_deg:.1f}°"
    )
    print(
        f"Rayleigh (descriptive): R-bar={summary.rayleigh_mean_resultant_length:.3f}, "
        f"p={summary.rayleigh_p:.4g}"
    )
    print(f"Wrote tables to {OUT_DATA_DIR.relative_to(PROJECT_ROOT)}")
    print(
        f"Wrote figures to {png.relative_to(PROJECT_ROOT)} and "
        f"{pdf.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
