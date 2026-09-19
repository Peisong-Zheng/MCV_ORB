#!/usr/bin/env python3
"""Nominal-age continuous event analysis for the pooled NGRIP--MIS6 catalogue.

Each record conditions on its exact oldest event. The full conditional
intensity adds precession sine/cosine to LR04, CO2, inhibitory exponential
history (tau = 1.5 kyr), and a segment intercept. Rayleigh statistics use all
55 inventory events; likelihood comparisons use 53 response events.
"""

import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from toolbox import combined_likelihood, event_inputs
from toolbox.catalogue_colors import CATALOGUE_COLORS
from toolbox.orbital_phase import rayleigh_rbar_threshold, rayleigh_test
from toolbox.phase_response_plotting import format_phase_response_axis, mark_preferred_phase
from toolbox.project_config import (
    PROJECT_ROOT, LR04_XLSX, CO2_XLSX, PRE_TXT,
    ORBITAL_SOLUTION, ORBITAL_SOURCE_EPOCH, ORBITAL_AGE_OFFSET_TO_BP1950_KA,
)

RUN_NAME = "NGRIP_MIS6_event_phase_analysis"
OUT_DATA_DIR = PROJECT_ROOT / "data/processed" / RUN_NAME
OUT_FIG_DIR = PROJECT_ROOT / "figures" / RUN_NAME
CATALOGUE_LABEL = "NGRIP warming + MIS 6 transitions"
EVENT_TYPE = "warming_transition"
HISTORY_TAU_KYR = combined_likelihood.DEFAULT_HISTORY_TAU_KA
RESOLUTION_COVARIATE_INCLUDED = False


def run_analysis():
    """Read and fit; writing and paper export belong only to the run entrypoint."""
    context = combined_likelihood.build_context(history_tau_ka=HISTORY_TAU_KYR)
    fit = combined_likelihood.fit_catalogue(context.events, context)
    events = fit.design.all_events
    phases = combined_likelihood.sample_event_phases(events)
    result = dict(events=events, context=fit.context, fit=fit, event_phases=phases,
                  rayleigh=rayleigh_test(phases.pre_phase_rad.to_numpy(float)),
                  fitted_rates=combined_likelihood.fitted_rate_table(fit))
    result["summary"] = build_analysis_summary(result)
    result.update(model_tables(fit))
    _validate_results(result)
    return result


def _validate_results(result):
    events, fit = result["events"], result["fit"]
    if events.groupby("segment_id").size().to_dict() != {"MIS6": 21, "NGRIP": 34}:
        raise RuntimeError("The pooled inventory must contain 34 NGRIP and 21 MIS6 events")
    if events.event_role.value_counts().to_dict() != {"response": 53, "conditioning": 2}:
        raise RuntimeError("Exactly one oldest event per segment must condition the fit")
    if result["event_phases"].pre_phase_extrapolated.any():
        raise RuntimeError("An inventory event has an extrapolated precession phase")
    if not fit.summary["all_models_converged"] or not fit.summary["likelihood_nesting_ok"]:
        raise RuntimeError("Inspect finite-MLE convergence and nested likelihoods")
    for model in (fit.reduced, fit.full):
        if dict(zip(model.terms, model.beta))[combined_likelihood.HISTORY_TERM] > 0:
            raise RuntimeError("The history coefficient violates the inhibitory domain")


def model_tables(fit):
    """Small model-level tables; integration nodes are not statistical samples."""
    rows = []
    for name, model in (("reduced", fit.reduced), ("full", fit.full)):
        rows.append(dict(model_id=name, log_likelihood=model.log_likelihood,
                         aic=model.aic, n_parameters=len(model.terms),
                         n_response_events=model.n_events, converged=model.converged))
    return dict(models={"reduced": fit.reduced, "full": fit.full},
                model_summary=pd.DataFrame(rows),
                likelihood_tests=pd.DataFrame([dict(dataset_id=fit.context.catalogue_id,
                    comparison_id="phase_after_climate",
                    reduced_model_id="reduced", full_model_id="full",
                    **{key:fit.summary[key] for key in ("df", "LR_statistic", "LR_p_value",
                       "gain_bits_per_event", "delta_AIC_full_minus_reduced")})]),
                coefficients=combined_likelihood.coefficient_table(fit),
                scaling=combined_likelihood.scaling_table(fit.context))


def build_coefficients(fit):
    table = combined_likelihood.coefficient_table(fit)
    notes = {
        "intercept": "conditional log rate per kyr",
        "same_type_exponential_history": "strictly older events weighted by exp(-age difference / 1.5 kyr); coefficient <= 0",
        "lr04_scaled": "LR04 anomaly divided by its nominal continuous response-range span",
        "co2_scaled": "CO2 anomaly divided by its nominal continuous response-range span",
        "mis6_segment": "MIS6=1; NGRIP=0; conditional segment contrast",
        "pre_phase_sin": "sine of BP1950 precession phase",
        "pre_phase_cos": "cosine of BP1950 precession phase",
    }
    table["term_definition"] = table.term.map(notes)
    return table


def build_parameters(result):
    context = result["context"]
    rows = [
        ("model_version", combined_likelihood.MODEL_VERSION, "", "continuous conditional intensity"),
        ("age_unit", "kyr BP", "BP1950", "events and astronomical forcing"),
        ("history_tau", context.history_tau_ka, "kyr", "fixed exponential decay time"),
        ("history_coefficient_domain", "beta_H <= 0", "", "inhibition or no history effect"),
        ("initial_unobserved_history", context.initial_history, "weighted events", "boundary approximation"),
        ("conditioning", "exact oldest event per segment", "", "anchor is excluded from the event likelihood term"),
        ("response_exposure", context.response_exposure_kyr, "kyr", "younger event-free tails retained; record gap excluded"),
        ("segment_indicator", "MIS6=1; NGRIP=0", "", "conditional baseline contrast"),
        ("likelihood", "event log intensity minus integrated intensity", "", "actual event ages"),
        ("nominal_p", "chi-square reference, df=2", "", "bootstrap calibration is a separate experiment"),
        ("climate_scaling", "time mean / response range", "", "fixed nominal continuous response support"),
        ("orbital_solution", ORBITAL_SOLUTION, "", "La2004"),
        ("orbital_source_epoch", ORBITAL_SOURCE_EPOCH, "", "source convention"),
        ("orbital_age_offset_to_BP1950", ORBITAL_AGE_OFFSET_TO_BP1950_KA, "kyr", "applied before interpolation"),
    ]
    for name, path in [("event_catalogue", combined_likelihood.EVENT_CATALOGUE_CSV),
                       ("observation_segments", combined_likelihood.OBSERVATION_SEGMENTS_CSV),
                       ("lr04_input", LR04_XLSX), ("co2_input", CO2_XLSX), ("precession_input", PRE_TXT)]:
        rows.append((name, str(path.relative_to(PROJECT_ROOT)), "", "source input"))
    return pd.DataFrame(rows, columns=["parameter", "value", "unit", "note"])


def build_analysis_summary(result: dict) -> pd.DataFrame:
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
        "analysis_role": "primary",
        "resolution_covariate_included": RESOLUTION_COVARIATE_INCLUDED,
        "age_uncertainty_propagated": False,
        "event_membership": "published events only",
    }
    return pd.DataFrame([row])

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
    ages = np.linspace(segment.observation_start_kyr_bp, segment.observation_end_kyr_bp, 1200)
    precession = event_inputs.interpolate_checked(ages, *context.forcings["precession_index"], context="precession timeline")
    phases = event_phases.loc[event_phases["segment_id"].eq(segment_id)]
    color = CATALOGUE_COLORS["primary"]

    axis.axvspan(
        segment.response_end_kyr_bp,
        segment.observation_end_kyr_bp,
        color="#E6E6E6",
        lw=0,
        zorder=0,
    )
    axis.plot(
        ages,
        precession,
        color="#555555",
        lw=1.05,
        zorder=1,
    )
    axis.scatter(
        phases[combined_likelihood.EVENT_AGE_COLUMN],
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
    # Older BP ages are on the left; event ages and phase values are unchanged.
    axis.set_xlim(segment.observation_end_kyr_bp, segment.observation_start_kyr_bp)
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
        color=CATALOGUE_COLORS["primary"],
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
        0.5,
        -0.17,
        (
            rf"$\bar{{R}}$ = {rayleigh['mean_resultant_length']:.2f}; "
            f"p = {rayleigh['rayleigh_p']:.3f}"
        ),
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
    )

def _plot_phase_response(axis: plt.Axes, fit: combined_likelihood.CombinedLikelihoodFit) -> None:
    """Plot the fitted multiplicative contribution of precession phase."""

    beta = dict(zip(fit.full.terms, fit.full.beta))
    phase_deg = np.linspace(0.0, 360.0, 721)
    multiplier = combined_likelihood.phase_rate_multiplier(
        phase_deg,
        beta["pre_phase_sin"],
        beta["pre_phase_cos"],
    )
    preferred = float(fit.summary["pre_phase_preferred_deg"])

    axis.plot(phase_deg, multiplier, color=CATALOGUE_COLORS["primary"], lw=2.0)
    axis.axhline(1.0, color="#777777", lw=0.9, ls=":")
    axis.axvline(preferred, color="#333333", lw=1.0, ls=(0, (4, 2)))
    axis.text(
        0.04,
        0.96,
        (
            f"G = {fit.summary['gain_bits_per_event']:.3f} bits event$^{{-1}}$\n"
            f"LR = {fit.summary['LR_statistic']:.2f}; "
            f"nominal p = {fit.summary['nominal_LR_p']:.3g}\n"
            f"Preferred phase = {preferred:.1f}°\n"
            f"Max/min rate ratio = {fit.summary['pre_phase_rate_ratio_max_vs_min']:.2f}"
        ),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2.0},
    )
    format_phase_response_axis(axis)
    axis.set_ylim(0, max(4.1, np.max(multiplier) * 1.4))  # Leave room for the fitted-summary label above the curve.
    mark_preferred_phase(axis, preferred, fit.summary['pre_phase_rate_ratio_max_vs_min'], CATALOGUE_COLORS["primary"])
    axis.set_title("Fitted warming-event rate")
    axis.grid(False)
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
        top=0.93,
    )
    # The older MIS 6 segment precedes NGRIP in the left-to-right time sequence.
    timeline = grid[0, :].subgridspec(1, 2, width_ratios=(72, 111), wspace=0.07)
    mis6_axis = fig.add_subplot(timeline[0, 0])
    ngrip_axis = fig.add_subplot(timeline[0, 1], sharey=mis6_axis)
    rayleigh_axis = fig.add_subplot(grid[1, 0], projection="polar")
    response_axis = fig.add_subplot(grid[1, 1])

    _plot_segment_timeline(ngrip_axis, "NGRIP", result)
    _plot_segment_timeline(mis6_axis, "MIS6", result)
    mis6_axis.set_ylabel("Precession index")
    _mark_discontinuous_axis(mis6_axis, ngrip_axis)
    _plot_rayleigh(rayleigh_axis, result)
    _plot_phase_response(response_axis, result["fit"])

    _add_panel_label(mis6_axis, "a", x=-0.14)
    _add_panel_label(rayleigh_axis, "b", x=-0.21)
    _add_panel_label(response_axis, "c", x=-0.22)
    return fig

def write_outputs(result, output_dir=OUT_DATA_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "analysis_summary.csv": result["summary"],
        "event_catalogue_used.csv": result["events"],
        "event_precession_phases.csv": result["event_phases"],
        "model_coefficients.csv": build_coefficients(result["fit"]),
        "model_summary.csv": result["model_summary"],
        "likelihood_tests.csv": result["likelihood_tests"],
        "fitted_rates.csv": result["fitted_rates"],
        "phase_sector_fit.csv": combined_likelihood.phase_sector_observed_expected(result["fit"], n_sectors=18),
        "predictor_scaling.csv": result["scaling"],
        "support.csv": combined_likelihood.support_table(result["context"]),
        "parameters_and_provenance.csv": build_parameters(result),
    }
    for name, table in tables.items():
        table.to_csv(output_dir / name, index=False, float_format="%.12g")


def save_figure(fig, output_dir=OUT_FIG_DIR, *, paper_export=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png, pdf = [output_dir / f"{RUN_NAME}.{suffix}" for suffix in ("png", "pdf")]
    fig.savefig(png, dpi=450, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    if paper_export:
        from paper_figure_export import copy_pdf_to_paper
        copy_pdf_to_paper(pdf)
    return png, pdf


def write_notes(result, notes_dir):
    notes_dir = Path(notes_dir)
    notes_dir.mkdir(parents=True, exist_ok=True)
    s = result["summary"].iloc[0]
    caption = f"""NGRIP--MIS6 warming events and their conditional precession-phase association.

(a) Published event identities at their operational ages, plotted on the La2004 precession index. The broken age axis omits the record gap. Blue dots denote all 55 inventory events. Gray older intervals precede each exact conditioning event; they contribute no response exposure. Ages decrease toward the right.
(b) Descriptive counts in twelve 30-degree phase sectors, with the mean direction and nominal Rayleigh p. Phase zero is a precession-index minimum and 180 degrees a maximum; radial labels are event counts. The mean arrow and dashed Rayleigh reference are scaled by the largest sector count.
(c) Conditional phase multiplier exp(beta_sin sin(phi) + beta_cos cos(phi)). Unity denotes zero phase contribution, not the separately fitted reduced-model rate. The model describes warming occurrence over total observation time. The exponential history coefficient is nonpositive, with fixed decay time 1.5 kyr. Both models contain LR04, CO2 and a segment intercept; the full model adds precession sine and cosine. The continuous likelihood conditions on the exact oldest event in each record, using 53 response events over {s.response_exposure_kyr:.3f} kyr. G is in-sample log-likelihood gain per response event. The displayed LR p is nominal; bootstrap calibration and chronology sensitivity are separate experiments.
"""
    methods = f"""CONTINUOUS-TIME NGRIP--MIS6 MAIN ANALYSIS

Source: 34 NGRIP GI starts and 21 MIS6 speleothem transitions. Data and event ages are unchanged. Each record conditions on its exact oldest event (115.320 and 194.238 kyr BP). Earlier unobserved weighted history is set to zero as a boundary approximation; the anchor contributes to all younger history. The young event-free tails remain exposed and histories do not cross the gap.

The intensity is exp(beta0 + beta_history H + beta_L LR04 + beta_C CO2 + beta_S I_MIS6 [+ beta_sin sin(phi) + beta_cos cos(phi)]), where H sums strictly older events with exponential decay time 1.5 kyr and beta_history <= 0. Actual response-event log intensities minus the integrated intensity define the likelihood. Integration splits at source interpolation knots and actual events. Fixed nominal time-weighted climate means and ranges define the scaling.

RESULTS
Response events: {s.n_response_events}; duration: {s.response_exposure_kyr:.6f} kyr.
G = {s.gain_bits_per_event:.9f} bits/event; LR = {s.LR_statistic:.9f}; nominal p = {s.nominal_LR_p:.9g}; Delta AIC (full minus reduced) = {s.delta_AIC_full_minus_reduced:.9f}.
Preferred phase = {s.pre_phase_preferred_deg:.6f} degrees; maximum/minimum conditional phase rate ratio = {s.pre_phase_rate_ratio_max_vs_min:.6f}.
Descriptive Rayleigh p = {s.rayleigh_p:.9g} using all 55 inventory events.

Outputs separate support, event roles, model coefficients and summary, climate scaling, and sampled continuous fitted rates. Plotting ages are not fitting bins. These are nominal-age in-sample associations; no chronology or sampling interval is inferred from this figure.
"""
    (notes_dir / f"{RUN_NAME}_Caption.txt").write_text(caption)
    (notes_dir / f"{RUN_NAME}_Methods_and_results.txt").write_text(methods)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT,
                        help="Project-shaped root for result tables, figures and notes")
    parser.add_argument("--no-paper-export", action="store_true")
    args = parser.parse_args(argv)
    result = run_analysis()
    write_outputs(result, args.output_root / "data/processed" / RUN_NAME)
    save_figure(plot_results(result), args.output_root / "figures" / RUN_NAME,
                paper_export=not args.no_paper_export and args.output_root.resolve() == PROJECT_ROOT.resolve())
    write_notes(result, args.output_root / "experiment_note")
    s = result["summary"].iloc[0]
    print(f"{RUN_NAME}: {s.n_response_events} response events; G={s.gain_bits_per_event:.6f}; "
          f"LR={s.LR_statistic:.6f}; nominal p={s.nominal_LR_p:.6g}; phase={s.pre_phase_preferred_deg:.3f}")


if __name__ == "__main__":
    main()
