#!/usr/bin/env python3
"""Barker 2011 SpeleoAge events: continuous-time fit and definition sensitivity.

Variable-threshold events are primary; fixed-threshold events are a nominal-age
sensitivity. Both report nominal LR p here; their BG-model bootstrap calibrations
are separate experiments. Both use inhibitory exponential history with tau =
1.5 kyr and condition on their exact oldest event. SpeleoAge is retained numerically under
the BP1950 working assumption; its precise source epoch remains unverified.
"""

from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from toolbox import event_model, model_stats
from toolbox.point_process import fit_point_process
from toolbox.plotting import (
    configure_barker_style, format_phase_response_axis, mark_preferred_phase,
)
from toolbox.project_config import (
    PROJECT_ROOT, BARKER_EVENT_CSVS, CATALOGUE_COLORS, MODEL_VERSION,
    LR04_CSV, CO2_CSV, ORBITAL_CSV, PRECESSION_PHASE_CSV,
)

RUN_NAME = "Barker2011_event_phase_analysis"
DEFINITION_COLORS = {"variable_threshold": CATALOGUE_COLORS["variable"],
                     "fixed_threshold": CATALOGUE_COLORS["fixed"]}
EVENT_COLOR = DEFINITION_COLORS["variable_threshold"]
ANALYSIS_START_KA, ANALYSIS_END_KA = 0.0, 400.0
HISTORY_TAU_KA = 1.5
HISTORY_TERM = "same_type_exponential_history"

OUTPUT_ROOT = PROJECT_ROOT
EXPORT_PAPER = True


def run_analysis(event_definition="variable_threshold", *, quadrature_order=4):
    events = pd.read_csv(BARKER_EVENT_CSVS[event_definition], float_precision="round_trip")
    events["segment_id"] = "Barker2011"

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
    observations = pd.DataFrame([dict(
        segment_id="Barker2011", observation_start_kyr_bp=ANALYSIS_START_KA,
        observation_end_kyr_bp=ANALYSIS_END_KA,
    )])
    windows = event_model.response_windows(events, observations)
    scaling = event_model.nominal_scaling({name: forcings[name] for name in ("lr04", "co2")}, windows)
    event_x, integral_x = event_model.build_design(
        events, windows, forcings, phase_anchors, scaling,
        tau=HISTORY_TAU_KA, quadrature_order=quadrature_order,
    )

    # The oldest event supplies history; only subsequent events enter the sum.
    background = ["intercept", HISTORY_TERM, "lr04_scaled", "co2_scaled"]
    with_phase = background + ["pre_phase_sin", "pre_phase_cos"]
    reduced = fit_point_process(
        event_x[background], integral_x[background], integral_x.weight, background,
        nonpositive_terms=(HISTORY_TERM,),
    )
    full = fit_point_process(
        event_x[with_phase], integral_x[with_phase], integral_x.weight, with_phase,
        nonpositive_terms=(HISTORY_TERM,), start_beta=np.r_[reduced.beta, 0., 0.],
    )
    comparison = model_stats.nested_likelihood_metrics(
        loglik_full=full.log_likelihood, loglik_reduced=reduced.log_likelihood,
        df=2, n_events=len(event_x), aic_full=full.aic, aic_reduced=reduced.aic,
    )
    beta_sin, beta_cos = full.beta[-2:]
    amplitude = float(np.hypot(beta_sin, beta_cos))
    preferred_phase = float(np.degrees(np.arctan2(beta_sin, beta_cos)) % 360) if amplitude > 1e-10 else np.nan

    events["event_role"] = np.where(events.event_age_kyr_bp == windows.anchor_age_kyr_bp.item(),
                                    "conditioning", "response")
    events["included_in_response"] = events.event_role.eq("response")
    phases = event_model.sample_event_phases(events, forcings["precession_index"], phase_anchors)
    rayleigh = model_stats.rayleigh_test(phases.pre_phase_rad.to_numpy(float))
    summary = pd.DataFrame([dict(
        model_version=MODEL_VERSION, event_definition=event_definition,
        n_source_events=len(events), n_response_events=len(event_x),
        response_exposure_kyr=float((windows.response_end_kyr_bp - windows.response_start_kyr_bp).sum()),
        LR_statistic=comparison["LR_statistic"], nominal_LR_p=comparison["LR_p_value"],
        gain_bits_per_event=comparison["gain_bits_per_event"],
        pre_phase_preferred_deg=preferred_phase,
        pre_phase_rate_ratio_max_vs_min=float(np.exp(2 * amplitude)),
        rayleigh_mean_phase_deg=rayleigh["mean_phase_deg"],
        rayleigh_mean_resultant_length=rayleigh["mean_resultant_length"], rayleigh_p=rayleigh["rayleigh_p"],
    )])
    result = dict(events=events, windows=windows, forcings=forcings, scaling=scaling,
                  event_features=event_x, integration_features=integral_x,
                  reduced=reduced, full=full, summary=summary, event_phases=phases, rayleigh=rayleigh)
    if result["event_phases"].pre_phase_extrapolated.any():
        raise ValueError("An event phase is extrapolated")
    return result


def _add_panel_label(axis: plt.Axes, label: str, *, x: float = -0.12):
    axis.text(
        x,
        1.04,
        label,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontweight="bold",
        fontsize=11,
    )

def _plot_rayleigh(axis, results):
    """Overlay sector counts and mean vectors on one common radial scale."""
    edges = np.linspace(0, 2 * np.pi, 13)
    histograms = [np.histogram(r["event_phases"]["pre_phase_rad"], bins=edges)[0] for r in results]
    maximum = max(max(counts) for counts in histograms)
    for result, counts in zip(results, histograms):
        definition = result["summary"].iloc[0].event_definition
        color = DEFINITION_COLORS[definition]
        fixed = definition == "fixed_threshold"
        # Outlined sectors keep both counts visible, including identical counts.
        axis.bar(edges[:-1], counts, width=np.diff(edges)[0], align="edge",
                 facecolor="none" if fixed else color, edgecolor=color,
                 alpha=1 if fixed else 0.40, linewidth=1.0 if fixed else 0.5,
                 linestyle="--" if fixed else "-", zorder=3 if fixed else 2)
        rayleigh = result["rayleigh"]
        axis.annotate("", xy=(rayleigh["mean_phase_rad"], rayleigh["mean_resultant_length"] * maximum),
                      xytext=(rayleigh["mean_phase_rad"], 0),
                      arrowprops=dict(arrowstyle="->", lw=1.4, color=color,
                                      linestyle="--" if fixed else "-"), zorder=5)
    axis.set_theta_zero_location("E")
    axis.set_theta_direction(1)
    axis.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    axis.set_xticklabels(["min", "90°", "max", "270°"])
    axis.tick_params(axis="x", pad=3)
    axis.tick_params(axis="y", labelsize=7.5)
    axis.set_title("Event phase (descriptive)", pad=20, fontsize=10)
    axis.grid(color="#CCCCCC", lw=0.6)

def phase_response_curve(beta_sin, beta_cos):
    """Multiplicative phase term; other covariates are held fixed."""
    phase_deg = np.linspace(0, 360, 721)
    phase = np.deg2rad(phase_deg)
    multiplier = np.exp(beta_sin * np.sin(phase) + beta_cos * np.cos(phase))
    return phase_deg, multiplier

def plot_results(result, fixed_result=None):
    """Overlay definitions in the pooled figure's existing three-panel layout."""
    configure_barker_style()
    results = [result] if fixed_result is None else [result, fixed_result]
    fig = plt.figure(figsize=(180 / 25.4, 134 / 25.4))
    grid = fig.add_gridspec(2, 2, height_ratios=(0.88, 1.12), width_ratios=(0.82, 1.18),
                           hspace=0.58, wspace=0.42, left=0.09, right=0.98, bottom=0.15, top=0.94)
    timeline = fig.add_subplot(grid[0, :])
    polar = fig.add_subplot(grid[1, 0], projection="polar")
    response = fig.add_subplot(grid[1, 1])
    support = result["windows"].iloc[0]
    ages = np.linspace(ANALYSIS_START_KA, ANALYSIS_END_KA, 2400)
    precession = event_model.interpolate_checked(ages, *result["forcings"]["precession_index"], context="Barker precession timeline")

    # Gray exposure supplies history but is excluded from the response likelihood.
    timeline.axvspan(support.response_end_kyr_bp, ANALYSIS_END_KA, color="#E6E6E6", lw=0, zorder=0)
    timeline.plot(ages, precession, color="#555555", lw=1.05, zorder=1)
    for index, current in enumerate(results):
        s = current["summary"].iloc[0]
        color = DEFINITION_COLORS[s.event_definition]
        fixed = s.event_definition == "fixed_threshold"
        phases = current["event_phases"]
        label = f"{'Fixed' if fixed else 'Variable'} threshold (n = {s.n_source_events})"
        # Large open squares can surround the primary dots at shared event ages.
        timeline.scatter(phases["event_age_kyr_bp"], phases["precession_index"],
                         s=30 if fixed else 15, marker="s" if fixed else "o",
                         facecolor="none" if fixed else color, edgecolor=color,
                         linewidth=0.75 if fixed else 0.4, zorder=4 if fixed else 3, label=label)
        full = current["full"]
        beta = dict(zip(full.terms, full.beta))
        phase, multiplier = phase_response_curve(beta["pre_phase_sin"], beta["pre_phase_cos"])
        response.plot(phase, multiplier, color=color, lw=1.8, ls="--" if fixed else "-")
        response.axvline(s.pre_phase_preferred_deg, color=color, lw=0.8, ls=":", alpha=0.85)
        # Pair each definition's conditional fit with its phase and effect size.
        mark_preferred_phase(response, s.pre_phase_preferred_deg, s.pre_phase_rate_ratio_max_vs_min, color)
        response.text(0.06, 0.97 - 0.15 * index,
                      f"G = {s.gain_bits_per_event:.3f}; nominal p = {s.nominal_LR_p:.3f}\n"
                      f"Preferred phase: {s.pre_phase_preferred_deg:.1f}°; max/min: {s.pre_phase_rate_ratio_max_vs_min:.2f}",
                      color=color, transform=response.transAxes, ha="left", va="top", fontsize=7.2,
                      bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1))
        rayleigh = current["rayleigh"]
        polar.text(0.02, -0.20 - 0.10 * index,
                   rf"$\bar{{R}}$ = {rayleigh['mean_resultant_length']:.2f}; p = {rayleigh['rayleigh_p']:.3f}",
                   color=color, transform=polar.transAxes, ha="left", va="top", fontsize=8)

    timeline.set_title("Barker 2011 · SpeleoAge", loc="left", fontsize=9, pad=12)
    timeline.legend(loc="lower right", bbox_to_anchor=(1.01, 1.02), frameon=False,
                    ncol=2, fontsize=8, handletextpad=0.35, columnspacing=1.1)
    # Reverse only the absolute-age axis: older ages are on the left.
    timeline.set(xlim=(ANALYSIS_END_KA, ANALYSIS_START_KA), xlabel="Age (kyr BP)", ylabel="Precession index")
    timeline.grid(axis="y", color="#D9D9D9", lw=0.55)
    timeline.spines[["top", "right"]].set_visible(False)
    _plot_rayleigh(polar, results)

    response.axhline(1, color="#777777", lw=0.9, ls=":")
    format_phase_response_axis(response)
    response.set_ylim(0, 3.1)
    response.set_title("Fitted warming-event rate")
    response.grid(False)
    response.spines[["top", "right"]].set_visible(False)
    _add_panel_label(timeline, "a", x=-0.047)
    _add_panel_label(polar, "b", x=-0.21)
    _add_panel_label(response, "c", x=-0.06)
    return fig

def write_outputs(result, output_dir):
    """Save scientific summaries, coefficients and the tables used by paper figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "analysis_summary.csv": result["summary"],
        "event_precession_phases.csv": result["event_phases"][[
            "event_id", "event_age_kyr_bp", "precession_index", "pre_phase_rad",
        ]],
        "model_coefficients.csv": pd.DataFrame([
            dict(model_id=name, term=term, beta=beta)
            for name in ("reduced", "full")
            for term, beta in zip(result[name].terms, result[name].beta)
        ]),
        "predictor_scaling.csv": result["scaling"].reset_index()[[
            "forcing_id", "mean", "range",
        ]],
    }
    if result["summary"].event_definition.item() == "variable_threshold":
        tables["phase_sector_fit.csv"] = model_stats.phase_sector_observed_expected(
            result["event_features"], result["integration_features"],
            {name: result[name] for name in ("reduced", "full")}, n_sectors=18,
        )[["model_id", "phase_sector_center_deg", "fitted_events"]]
    for name, table in tables.items():
        table.to_csv(output_dir / name, index=False, float_format="%.12g")

def save_figure(fig, output_dir, *, paper_export=False):
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

def main():
    result = run_analysis()
    fixed = run_analysis("fixed_threshold")
    root = OUTPUT_ROOT / "Barker2011"
    output = root / "data/processed" / RUN_NAME
    write_outputs(result, output)
    write_outputs(fixed, output / "fixed_threshold")
    save_figure(plot_results(result, fixed), root / "figures" / RUN_NAME,
                paper_export=EXPORT_PAPER and OUTPUT_ROOT.resolve() == PROJECT_ROOT.resolve())
    for current in (result, fixed):
        s = current["summary"].iloc[0]
        support = current["windows"].iloc[0]
        print(f"\n{s.event_definition}: {s.n_source_events} events; {s.n_response_events} response events")
        print(f"Input: {BARKER_EVENT_CSVS[s.event_definition].relative_to(PROJECT_ROOT)}")
        print(f"Observation {support.observation_start_kyr_bp:g}–{support.observation_end_kyr_bp:g} kyr BP; "
              f"conditioning age {support.anchor_age_kyr_bp:.9f}; exposure {s.response_exposure_kyr:.9f} kyr")
        print(f"G={s.gain_bits_per_event:.6f}; LR={s.LR_statistic:.6f}; nominal p={s.nominal_LR_p:.6g}; "
              f"phase={s.pre_phase_preferred_deg:.3f}°; max/min={s.pre_phase_rate_ratio_max_vs_min:.6f}")
        print(f"Rayleigh: mean phase={s.rayleigh_mean_phase_deg:.3f}°; "
              f"R={s.rayleigh_mean_resultant_length:.6f}; p={s.rayleigh_p:.6g}")
        models = pd.DataFrame([
            dict(model=name, log_likelihood=model.log_likelihood, AIC=model.aic,
                 converged=model.converged)
            for name, model in (("reduced", current["reduced"]), ("full", current["full"]))
        ])
        print(models.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"Saved 9 CSVs and the research figure under {root}.")


if __name__ == "__main__":
    main()
