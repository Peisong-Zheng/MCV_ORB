#!/usr/bin/env python3
"""Nominal-age continuous event analysis for the pooled NGRIP--MIS6 catalogue.

Each record conditions on its exact oldest event. The full conditional
intensity adds precession sine/cosine to LR04, CO2, inhibitory exponential
history (tau = 1.5 kyr), and a segment intercept. Rayleigh statistics use all
55 inventory events; likelihood comparisons use 53 response events.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from toolbox import model_stats, event_model
from toolbox.point_process import fit_point_process
from toolbox.project_config import EVENT_CATALOGUE_CSV, OBSERVATION_SEGMENTS_CSV, MODEL_VERSION
from toolbox.model_stats import rayleigh_rbar_threshold, rayleigh_test
from toolbox.plotting import format_phase_response_axis, mark_preferred_phase
from toolbox.project_config import (
    PROJECT_ROOT,
    CATALOGUE_COLORS,
    LR04_CSV,
    CO2_CSV,
    ORBITAL_CSV,
    PRECESSION_PHASE_CSV,
    ORBITAL_SOLUTION,
    ORBITAL_SOURCE_EPOCH,
    ORBITAL_AGE_OFFSET_TO_BP1950_KA,
)
from toolbox.project_config import generated_notes_dir

RUN_NAME = "NGRIP_MIS6_event_phase_analysis"
OUTPUT_ROOT = PROJECT_ROOT
EXPORT_PAPER = True
CATALOGUE_LABEL = "NGRIP warming + MIS 6 transitions"
EVENT_TYPE = "warming_transition"
HISTORY_TAU_KYR = 1.5
HISTORY_TERM = "same_type_exponential_history"
CATALOGUE_ID = "ngrip_warming_plus_mis6"
RESOLUTION_COVARIATE_INCLUDED = False


def run_analysis(*, quadrature_order=4):
    """Fit the pooled event catalogue using explicit response and integral matrices."""
    events = pd.read_csv(EVENT_CATALOGUE_CSV)
    if events.event_id.isna().any() or not events.event_id.is_unique:
        raise ValueError("Prepared event identities must be unique and nonempty")
    observations = pd.read_csv(OBSERVATION_SEGMENTS_CSV)
    lr04 = pd.read_csv(LR04_CSV, float_precision="round_trip")
    co2 = pd.read_csv(CO2_CSV, float_precision="round_trip")
    orbital = pd.read_csv(ORBITAL_CSV, float_precision="round_trip")
    anchors = pd.read_csv(PRECESSION_PHASE_CSV, float_precision="round_trip")
    forcings = {
        "lr04": (lr04.age_kyr_bp.to_numpy(), lr04.lr04.to_numpy()),
        "co2": (co2.age_kyr_bp.to_numpy(), co2.co2_ppm.to_numpy()),
        "precession_index": (orbital.age_kyr_bp.to_numpy(), orbital.precession_index.to_numpy()),
    }
    phase_anchors = (anchors.age_kyr_bp.to_numpy(), anchors.phase_unwrapped_rad.to_numpy())
    windows = event_model.response_windows(events, observations)
    scaling = event_model.nominal_scaling({name: forcings[name] for name in ("lr04", "co2")}, windows)
    event_x, integral_x = event_model.build_design(
        events, windows, forcings, phase_anchors, scaling,
        tau=HISTORY_TAU_KYR, quadrature_order=quadrature_order,
    )
    for frame in (event_x, integral_x):
        frame["mis6_segment"] = frame.segment_id.eq("MIS6").astype(float)
    background = ["intercept", HISTORY_TERM, "lr04_scaled", "co2_scaled", "mis6_segment"]
    with_phase = background + ["pre_phase_sin", "pre_phase_cos"]
    reduced = fit_point_process(event_x[background], integral_x[background], integral_x.weight,
                                background, nonpositive_terms=(HISTORY_TERM,))
    full = fit_point_process(event_x[with_phase], integral_x[with_phase], integral_x.weight,
                             with_phase, start_beta=np.r_[reduced.beta, 0., 0.],
                             nonpositive_terms=(HISTORY_TERM,))
    statistics = model_stats.fit_summary(
        reduced, full, event_x, windows, n_source_events=len(events),
        catalogue_id=CATALOGUE_ID, tau=HISTORY_TAU_KYR,
    )
    events = event_model.mark_event_roles(events, windows)
    event_phases = event_model.sample_event_phases(events, forcings["precession_index"], phase_anchors)
    rayleigh = rayleigh_test(event_phases.pre_phase_rad.to_numpy(float))
    fitted_rates = event_model.fitted_rate_table(
        events, windows, forcings, phase_anchors, scaling, {"reduced": reduced, "full": full},
        tau=HISTORY_TAU_KYR,
    )
    result = dict(events=events, observations=observations, windows=windows, forcings=forcings,
                  phase_anchors=phase_anchors, scaling=scaling, event_features=event_x,
                  integration_features=integral_x, reduced=reduced, full=full,
                  statistics=statistics, event_phases=event_phases, rayleigh=rayleigh,
                  fitted_rates=fitted_rates)
    result["summary"] = build_analysis_summary(result)
    result.update(model_tables(result))
    _validate_results(result)
    return result


def _validate_results(result):
    events = result["events"]
    if events.groupby("segment_id").size().to_dict() != {"MIS6": 21, "NGRIP": 34}:
        raise RuntimeError("The pooled inventory must contain 34 NGRIP and 21 MIS6 events")
    if events.event_role.value_counts().to_dict() != {"response": 53, "conditioning": 2}:
        raise RuntimeError("Exactly one oldest event per segment must condition the fit")
    if result["event_phases"].pre_phase_extrapolated.any():
        raise RuntimeError("An inventory event has an extrapolated precession phase")
    if not result["statistics"]["all_models_converged"] or not result["statistics"]["likelihood_nesting_ok"]:
        raise RuntimeError("Inspect finite-MLE convergence and nested likelihoods")
    for model in (result["reduced"], result["full"]):
        coefficients = dict(zip(model.terms, model.beta))
        if coefficients[HISTORY_TERM] > 0:
            raise RuntimeError("The history coefficient violates the inhibitory domain")


def model_tables(fit):
    """Summarize the two fits using response events as the sample size."""
    models = {"reduced": fit["reduced"], "full": fit["full"]}
    rows = []
    for name, model in models.items():
        rows.append({
            "model_id": name,
            "log_likelihood": model.log_likelihood,
            "aic": model.aic,
            "n_parameters": len(model.terms),
            "n_response_events": model.n_events,
            "converged": model.converged,
        })

    comparison = {
        "dataset_id": CATALOGUE_ID,
        "comparison_id": "phase_after_climate",
        "reduced_model_id": "reduced",
        "full_model_id": "full",
    }
    for name in ("df", "LR_statistic", "LR_p_value", "gain_bits_per_event",
                 "delta_AIC_full_minus_reduced"):
        comparison[name] = fit["statistics"][name]

    return {
        "models": models,
        "model_summary": pd.DataFrame(rows),
        "likelihood_tests": pd.DataFrame([comparison]),
    }


def build_coefficients(fit):
    table = pd.DataFrame([
        dict(model_id=name, term=term, beta=beta, rate_ratio_per_unit=np.exp(beta))
        for name in ("reduced", "full") for term, beta in zip(fit[name].terms, fit[name].beta)
    ])
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
    rows = [
        ("model_version", MODEL_VERSION, "", "continuous conditional intensity"),
        ("age_unit", "kyr BP", "BP1950", "events and astronomical forcing"),
        ("history_tau", HISTORY_TAU_KYR, "kyr", "fixed exponential decay time"),
        ("history_coefficient_domain", "beta_H <= 0", "", "inhibition or no history effect"),
        ("initial_unobserved_history", 0.0, "weighted events", "boundary approximation"),
        ("conditioning", "exact oldest event per segment", "", "anchor is excluded from the event likelihood term"),
        ("response_exposure", result["statistics"]["response_exposure_kyr"], "kyr", "younger event-free tails retained; record gap excluded"),
        ("segment_indicator", "MIS6=1; NGRIP=0", "", "conditional baseline contrast"),
        ("likelihood", "event log intensity minus integrated intensity", "", "actual event ages"),
        ("nominal_p", "chi-square reference, df=2", "", "bootstrap calibration is a separate experiment"),
        ("climate_scaling", "time mean / response range", "", "fixed nominal continuous response support"),
        ("orbital_solution", ORBITAL_SOLUTION, "", "La2004"),
        ("orbital_source_epoch", ORBITAL_SOURCE_EPOCH, "", "source convention"),
        ("orbital_age_offset_to_BP1950", ORBITAL_AGE_OFFSET_TO_BP1950_KA, "kyr", "applied before interpolation"),
    ]
    inputs = {
        "event_catalogue": EVENT_CATALOGUE_CSV,
        "observation_segments": OBSERVATION_SEGMENTS_CSV,
        "lr04_input": LR04_CSV,
        "co2_input": CO2_CSV,
        "precession_input": ORBITAL_CSV,
        "phase_anchors_input": PRECESSION_PHASE_CSV,
    }
    for name, path in inputs.items():
        rows.append((name, str(path.relative_to(PROJECT_ROOT)), "", "source input"))
    return pd.DataFrame(rows, columns=["parameter", "value", "unit", "note"])


def build_analysis_summary(result: dict) -> pd.DataFrame:
    """Collect the point-age results used in the paper workflow."""

    events = result["events"]
    rayleigh = result["rayleigh"]
    row = {
        **result["statistics"],
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


def _plot_segment_timeline(axis, segment, event_phases, precession_source):
    """Plot one observed segment without implying exposure across the gap."""
    segment_id = segment.segment_id
    ages = np.linspace(segment.observation_start_kyr_bp, segment.observation_end_kyr_bp, 1200)
    precession = event_model.interpolate_checked(
        ages, *precession_source, context="precession timeline"
    )
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
        phases["event_age_kyr_bp"],
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


def _plot_rayleigh(axis, phases, rayleigh):
    """Draw the descriptive phase histogram and mean resultant vector."""
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


def _plot_phase_response(axis: plt.Axes, fit) -> None:
    """Plot the fitted multiplicative contribution of precession phase."""

    beta = dict(zip(fit["full"].terms, fit["full"].beta))
    phase_deg = np.linspace(0.0, 360.0, 721)
    phase = np.deg2rad(phase_deg)
    multiplier = np.exp(beta["pre_phase_sin"] * np.sin(phase) + beta["pre_phase_cos"] * np.cos(phase))
    preferred = float(fit["statistics"]["pre_phase_preferred_deg"])

    axis.plot(phase_deg, multiplier, color=CATALOGUE_COLORS["primary"], lw=2.0)
    axis.axhline(1.0, color="#777777", lw=0.9, ls=":")
    axis.axvline(preferred, color="#333333", lw=1.0, ls=(0, (4, 2)))
    axis.text(
        0.04,
        0.96,
        (
            f"G = {fit['statistics']['gain_bits_per_event']:.3f} bits event$^{{-1}}$\n"
            f"LR = {fit['statistics']['LR_statistic']:.2f}; "
            f"nominal p = {fit['statistics']['nominal_LR_p']:.3g}\n"
            f"Preferred phase = {preferred:.1f}°\n"
            f"Max/min rate ratio = {fit['statistics']['pre_phase_rate_ratio_max_vs_min']:.2f}"
        ),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2.0},
    )
    format_phase_response_axis(axis)
    # Leave room for the fitted-summary label above the curve.
    axis.set_ylim(0, max(4.1, np.max(multiplier) * 1.4))
    mark_preferred_phase(
        axis, preferred, fit["statistics"]["pre_phase_rate_ratio_max_vs_min"],
        CATALOGUE_COLORS["primary"],
    )
    axis.set_title("Fitted warming-event rate")
    axis.grid(False)
    axis.spines[["top", "right"]].set_visible(False)


def plot_results(result):
    """Show record support, descriptive phases and the fitted phase effect."""

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

    event_phases = result["event_phases"]
    precession_source = result["forcings"]["precession_index"]
    _plot_segment_timeline(ngrip_axis, result["windows"].set_index("segment_id", drop=False).loc["NGRIP"], event_phases, precession_source)
    _plot_segment_timeline(mis6_axis, result["windows"].set_index("segment_id", drop=False).loc["MIS6"], event_phases, precession_source)
    mis6_axis.set_ylabel("Precession index")
    _mark_discontinuous_axis(mis6_axis, ngrip_axis)
    _plot_rayleigh(rayleigh_axis, event_phases.pre_phase_rad.to_numpy(float), result["rayleigh"])
    _plot_phase_response(response_axis, result)

    _add_panel_label(mis6_axis, "a", x=-0.14)
    _add_panel_label(rayleigh_axis, "b", x=-0.21)
    _add_panel_label(response_axis, "c", x=-0.22)
    return fig


def write_outputs(result, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "analysis_summary.csv": result["summary"],
        "event_catalogue_used.csv": result["events"],
        "event_precession_phases.csv": result["event_phases"],
        "model_coefficients.csv": build_coefficients(result),
        "model_summary.csv": result["model_summary"],
        "likelihood_tests.csv": result["likelihood_tests"],
        "fitted_rates.csv": result["fitted_rates"],
        "phase_sector_fit.csv": model_stats.phase_sector_observed_expected(
            result["event_features"], result["integration_features"], result["models"], n_sectors=18,
        ),
        "predictor_scaling.csv": result["scaling"].reset_index().assign(weighting="nominal response time"),
        "support.csv": result["windows"],
        "parameters_and_provenance.csv": build_parameters(result),
    }
    for name, table in tables.items():
        table.to_csv(output_dir / name, index=False, float_format="%.12g")


def save_figure(fig, output_dir, *, paper_export=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{RUN_NAME}.png"
    pdf = output_dir / f"{RUN_NAME}.pdf"
    fig.savefig(png, dpi=450, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    if paper_export:
        from paper_figure_export import copy_pdf_to_paper

        copy_pdf_to_paper(pdf)
    return png, pdf


def write_notes(result, notes_dir):
    """Save a short figure caption and run summary for the full CSV outputs."""
    notes_dir = Path(notes_dir)
    notes_dir.mkdir(parents=True, exist_ok=True)
    summary = result["summary"].iloc[0]
    caption = f"""NGRIP--MIS6 warming events and their conditional precession-phase association.

(a) All {summary.n_source_events} inventory events on the La2004 precession index.
Gray intervals precede the conditioning events and carry no response exposure.
The broken age axis omits the record gap; ages decrease toward the right.
(b) Descriptive event counts in twelve phase sectors and the mean direction.
Phase zero is a precession minimum; 180 degrees is a maximum. The mean arrow
and dashed Rayleigh reference are scaled by the largest sector count.
(c) Conditional phase multiplier exp(beta_sin sin(phi) + beta_cos cos(phi)).
The models use {summary.n_response_events} response events over
{summary.response_exposure_kyr:.3f} kyr, conditional on each record's oldest event.
Both include inhibitory history (tau={HISTORY_TAU_KYR:g} kyr), LR04, CO2
and a segment intercept; the full model adds the two phase terms. Unity denotes
zero phase contribution. G is the in-sample gain per response event; LR p is
nominal. Bootstrap calibration and chronology sensitivity are separate analyses.
"""
    result_columns = [
        "n_response_events", "response_exposure_kyr", "gain_bits_per_event",
        "LR_statistic", "nominal_LR_p", "delta_AIC_full_minus_reduced",
        "pre_phase_preferred_deg", "pre_phase_rate_ratio_max_vs_min", "rayleigh_p",
    ]
    methods = (
        "NGRIP--MIS6 nominal-age conditional event analysis\n\n"
        "Each segment conditions on its oldest event. Younger event-free tails\n"
        "remain exposed; history and exposure do not cross the record gap.\n"
        "Likelihood is the response-event log-intensity sum minus integrated\n"
        "intensity. Climate scaling is fixed on nominal response support.\n\n"
        "Run settings and input paths: parameters_and_provenance.csv\n"
        "Exact supports: support.csv; scaling: predictor_scaling.csv\n"
        "Term definitions and coefficients: model_coefficients.csv\n\n"
        + summary[result_columns].to_string(float_format=lambda value: f"{value:.9g}")
        + "\n"
    )
    (notes_dir / f"{RUN_NAME}_Caption.txt").write_text(caption)
    (notes_dir / f"{RUN_NAME}_Methods_and_results.txt").write_text(methods)


def main():

    data_dir = OUTPUT_ROOT / "data/processed" / RUN_NAME
    figure_dir = OUTPUT_ROOT / "figures" / RUN_NAME
    paper_export = (
        EXPORT_PAPER
        and OUTPUT_ROOT.resolve() == PROJECT_ROOT.resolve()
    )

    result = run_analysis()
    write_outputs(result, data_dir)
    fig = plot_results(result)
    save_figure(fig, figure_dir, paper_export=paper_export)
    write_notes(result, generated_notes_dir(OUTPUT_ROOT))

    summary = result["summary"].iloc[0]
    print(
        f"{RUN_NAME}: {summary.n_response_events} response events; "
        f"G={summary.gain_bits_per_event:.6f}; LR={summary.LR_statistic:.6f}; "
        f"nominal p={summary.nominal_LR_p:.6g}; phase={summary.pre_phase_preferred_deg:.3f}"
    )


if __name__ == "__main__":
    main()
