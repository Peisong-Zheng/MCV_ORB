#!/usr/bin/env python3
"""Barker 2011 SpeleoAge events: continuous-time fit and definition sensitivity.

Variable-threshold events are primary; fixed-threshold events are a nominal-age
sensitivity. Both use inhibitory exponential history with tau = 1.5 kyr and
condition on their exact oldest event. SpeleoAge is retained numerically under
the BP1950 working assumption; its precise source epoch remains unverified.
"""

import argparse
from pathlib import Path
import hashlib
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from toolbox import combined_likelihood, event_inputs, orbital_phase
from toolbox.catalogue_colors import CATALOGUE_COLORS
from toolbox.phase_response_plotting import format_phase_response_axis, mark_preferred_phase
from toolbox.project_config import (
    PROJECT_ROOT, LR04_XLSX, CO2_XLSX, PRE_TXT,
    ORBITAL_SOLUTION, ORBITAL_SOURCE_EPOCH, ORBITAL_AGE_OFFSET_TO_BP1950_KA,
)

RUN_NAME = "Barker2011_event_phase_analysis"
OUT_DATA_DIR = ROOT / "data/processed" / RUN_NAME
OUT_FIG_DIR = ROOT / "figures" / RUN_NAME
BARKER_XLS = ROOT / "data/raw/Barker et al-2011-SOM.xls"
DATASET_ID = "barker_variable_threshold_speleo_0_400"
EVENT_TYPE = "barker_do_warming_variable_threshold_speleo"
CATALOGUE_LABEL = "Barker variable-threshold D-O warmings, SpeleoAge 0-400 ka"
AGE_COLUMN = "SpeloAge (kyr).1"
PICK_COLUMN = "DO pick variable threshold"
PICK_COLUMNS = {"variable_threshold": PICK_COLUMN, "fixed_threshold": "DO pick"}
EXPECTED_COUNTS = {"variable_threshold": 70, "fixed_threshold": 59}
DEFINITION_COLORS = {"variable_threshold": CATALOGUE_COLORS["variable"],
                     "fixed_threshold": CATALOGUE_COLORS["fixed"]}
EVENT_COLOR = DEFINITION_COLORS["variable_threshold"]
ANALYSIS_START_KA, ANALYSIS_END_KA = 0.0, 400.0
HISTORY_TAU_KA = combined_likelihood.DEFAULT_HISTORY_TAU_KA
REDUCED_TERMS = tuple(t for t in combined_likelihood.REDUCED_TERMS if t != combined_likelihood.SEGMENT_TERM)
FULL_TERMS = REDUCED_TERMS + ("pre_phase_sin", "pre_phase_cos")
MODEL_SPECS = (("reduced", REDUCED_TERMS, "History and climate"),
               ("full", FULL_TERMS, "History, climate and precession"))


def load_barker_source(event_definition="variable_threshold"):
    """Retain the source-table interface used by the upstream chronology script.

    That script creates its own stable Barker_S3 IDs. Main fits use the complete
    ID-bearing catalogue in ``build_barker_context`` instead.
    """
    events = combined_likelihood.load_barker_events(event_definition).drop(columns="event_id")
    events["dataset_id"] = DATASET_ID.replace("variable_threshold", event_definition)
    events["event_type"] = EVENT_TYPE.replace("variable_threshold", event_definition)
    events["age_column"] = AGE_COLUMN
    events["pick_column"] = PICK_COLUMNS[event_definition]
    events["analysis_start_ka"] = ANALYSIS_START_KA
    events["analysis_end_ka"] = ANALYSIS_END_KA
    return events


def run_analysis(event_definition="variable_threshold"):
    context = combined_likelihood.build_barker_context(event_definition, history_tau_ka=HISTORY_TAU_KA)
    fit = combined_likelihood.fit_catalogue(context.events, context)
    events = fit.design.all_events.copy()
    events["dataset_id"] = context.catalogue_id
    events["age_column"] = AGE_COLUMN
    events["pick_column"] = PICK_COLUMNS[event_definition]
    phases = combined_likelihood.sample_event_phases(events)
    # Descriptive-column names remain readable by the chronology figure code.
    for target, source in [("phase_rad", "pre_phase_rad"), ("phase_deg", "pre_phase_deg"),
                           ("phase_extrapolated", "pre_phase_extrapolated"),
                           ("orbital_value_at_event", "precession_index")]:
        phases[target] = phases[source]
    rayleigh = orbital_phase.rayleigh_test(phases.pre_phase_rad.to_numpy(float))
    support = context.segments["Barker2011"]
    summary = pd.DataFrame([dict(**fit.summary, dataset_id=context.catalogue_id,
        catalogue_label=CATALOGUE_LABEL.replace("variable-threshold", event_definition.replace("_", "-")),
        event_definition=event_definition, age_scale="SpeleoAge",
        analysis_start_ka=ANALYSIS_START_KA, analysis_end_ka=ANALYSIS_END_KA,
        n_rayleigh_events=len(events), response_support_start_ka=support.response_start_kyr_bp,
        response_support_end_ka=support.response_end_kyr_bp,
        rayleigh_mean_phase_deg=rayleigh["mean_phase_deg"],
        rayleigh_mean_resultant_length=rayleigh["mean_resultant_length"], rayleigh_p=rayleigh["rayleigh_p"],
        resolution_covariate_included=False, age_uncertainty_propagated=False,
        event_age_epoch="BP1950 assumed; precise source epoch unverified")])
    result = dict(events=events, context=fit.context, fit=fit, summary=summary,
                  event_phases=phases, rayleigh=rayleigh,
                  fitted_rates=combined_likelihood.fitted_rate_table(fit))
    result.update(model_tables(fit))
    validate_results(result)
    return result


def validate_results(result):
    events, fit = result["events"], result["fit"]
    summary = result["summary"].iloc[0]
    expected = EXPECTED_COUNTS[summary.event_definition]
    if len(events) != expected or events.included_in_response.sum() != expected - 1:
        raise ValueError("Exactly one oldest event conditions this definition's fit")
    if not events.event_id.is_unique or np.any(np.diff(events.event_age_ka) <= 0):
        raise ValueError("Barker event identities and strictly ordered ages must be preserved")
    if not np.isclose(summary.response_exposure_kyr, events.event_age_ka.max() - ANALYSIS_START_KA):
        raise ValueError("Continuous support does not match the exact anchor")
    if result["event_phases"].pre_phase_extrapolated.any():
        raise ValueError("An event phase is extrapolated")
    if not summary.all_models_converged or not summary.likelihood_nesting_ok:
        raise ValueError("Inspect finite MLE and likelihood nesting")
    for model in (fit.reduced, fit.full):
        if dict(zip(model.terms, model.beta))[combined_likelihood.HISTORY_TERM] > 0:
            raise ValueError("History must remain nonpositive")


def build_parameters(result):
    s = result["summary"].iloc[0]
    rows = [
        ("model_version", combined_likelihood.MODEL_VERSION, "", "continuous conditional intensity"),
        ("dataset_id", s.dataset_id, "", "SpeleoAge catalogue"),
        ("event_age_column", AGE_COLUMN, "kyr", "Table S3; Sheet1 header row 9"),
        ("event_pick_column", PICK_COLUMNS[s.event_definition], "", "retain published pick value 1"),
        ("event_age_reference", "BP1950 assumed", "", "precise source epoch unverified"),
        ("event_age_offset_applied", 0.0, "kyr", "source numerical ages retained"),
        ("history_tau", HISTORY_TAU_KA, "kyr", "continuous exponential decay time"),
        ("history_coefficient_domain", "beta_H <= 0", "", "inhibition or no history effect"),
        ("initial_unobserved_history", result["context"].initial_history, "weighted events", "boundary approximation"),
        ("conditioning_anchor", s.response_support_end_ka, "kyr BP", "exact oldest event; not a response event"),
        ("response_exposure", s.response_exposure_kyr, "kyr", "includes the younger event-free tail"),
        ("reduced_model_terms", "+".join(REDUCED_TERMS), "", "history, LR04 and CO2"),
        ("full_model_terms", "+".join(FULL_TERMS), "", "adds phase sine and cosine"),
        ("climate_scaling", "time mean / response range", "", "fixed nominal continuous response support"),
        ("orbital_solution", ORBITAL_SOLUTION, "", "shared La2004 solution"),
        ("orbital_source_epoch", ORBITAL_SOURCE_EPOCH, "", "astronomical source convention"),
        ("orbital_age_offset_to_BP1950", ORBITAL_AGE_OFFSET_TO_BP1950_KA, "kyr", "applied once before phase interpolation"),
        ("p_value_method", "nominal chi-square, df=2", "", "bootstrap is a separate experiment"),
        ("chronology_construction", "speleothem-tuned synthetic Greenland record", "", "not chronology-independent replication"),
    ]
    for name, path in [("event_source", BARKER_XLS), ("lr04_source", LR04_XLSX),
                       ("co2_source", CO2_XLSX), ("precession_source", PRE_TXT)]:
        rows.append((name, str(path.relative_to(PROJECT_ROOT)), "", "source input"))
        rows.append((name + "_sha256", hashlib.sha256(path.read_bytes()).hexdigest(), "", "source content"))
    return pd.DataFrame(rows, columns=["parameter", "value", "unit", "note"])


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

def configure_plot_style():
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
    histograms = [np.histogram(r["event_phases"]["phase_rad"], bins=edges)[0] for r in results]
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
    configure_plot_style()
    results = [result] if fixed_result is None else [result, fixed_result]
    fig = plt.figure(figsize=(180 / 25.4, 134 / 25.4))
    grid = fig.add_gridspec(2, 2, height_ratios=(0.88, 1.12), width_ratios=(0.82, 1.18),
                           hspace=0.58, wspace=0.42, left=0.09, right=0.98, bottom=0.15, top=0.94)
    timeline = fig.add_subplot(grid[0, :])
    polar = fig.add_subplot(grid[1, 0], projection="polar")
    response = fig.add_subplot(grid[1, 1])
    summary = result["summary"].iloc[0]
    ages = np.linspace(ANALYSIS_START_KA, ANALYSIS_END_KA, 2400)
    precession = event_inputs.interpolate_checked(ages, *result["context"].forcings["precession_index"], context="Barker precession timeline")

    # Gray exposure supplies history but is excluded from the response likelihood.
    timeline.axvspan(summary.response_support_end_ka, ANALYSIS_END_KA, color="#E6E6E6", lw=0, zorder=0)
    timeline.plot(ages, precession, color="#555555", lw=1.05, zorder=1)
    for index, current in enumerate(results):
        s = current["summary"].iloc[0]
        color = DEFINITION_COLORS[s.event_definition]
        fixed = s.event_definition == "fixed_threshold"
        phases = current["event_phases"]
        label = f"{'Fixed' if fixed else 'Variable'} threshold (n = {s.n_rayleigh_events})"
        # Large open squares can surround the primary dots at shared event ages.
        timeline.scatter(phases["event_age_ka"], phases["orbital_value_at_event"],
                         s=30 if fixed else 15, marker="s" if fixed else "o",
                         facecolor="none" if fixed else color, edgecolor=color,
                         linewidth=0.75 if fixed else 0.4, zorder=4 if fixed else 3, label=label)
        full = current["fit"].full
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

def write_outputs(result, output_dir=OUT_DATA_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "analysis_summary.csv": result["summary"],
        "event_catalogue_used.csv": result["events"],
        "event_precession_phases.csv": result["event_phases"],
        "model_coefficients.csv": result["coefficients"],
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

def write_notes(result, fixed_result, notes_dir=ROOT / "experiment_note"):
    notes_dir = Path(notes_dir)
    notes_dir.mkdir(parents=True, exist_ok=True)
    variable, fixed = [r["summary"].iloc[0] for r in (result, fixed_result)]
    shared = len(set(result["events"].event_id) & set(fixed_result["events"].event_id))
    caption = f"""Barker et al. (2011) SpeleoAge warming events under two published definitions.

(a) Variable-threshold events (rose dots, n=70) and fixed-threshold events (green open squares, n=59) on the La2004 precession index. Older ages are on the left. Gray shading precedes the primary exact conditioning event at {variable.response_support_end_ka:.3f} kyr BP. Each definition conditions on its own oldest event, retaining {variable.n_response_events} and {fixed.n_response_events} response events over {variable.response_exposure_kyr:.3f} and {fixed.response_exposure_kyr:.3f} kyr. Both inventory counts include their conditioning event.
(b) Descriptive event counts in twelve 30-degree sectors: rose fill and green dashed outlines. Arrows denote mean phase; arrow lengths are the mean resultant length times the common largest sector count. Radial ticks give event counts. Phase zero is a precession-index minimum and 180 degrees a maximum. Text gives each definition's Rayleigh statistics.
(c) Conditional phase multipliers exp(beta_sin sin(phi)+beta_cos cos(phi)), at fixed background and event history. Unity means zero phase contribution, not the separately fitted reduced rate. The continuous conditional model contains LR04, CO2 and exponential event history (tau=1.5 kyr, beta_history <= 0); full adds precession sine and cosine. G is the in-sample log-likelihood gain per response event and p is the nominal two-degree-of-freedom LR reference. Both models are refitted for each definition. Bootstrap calibration and chronology sensitivity are separate analyses of the variable-threshold catalogue only.

SpeleoAge numerical values are retained under a BP1950 working assumption; the precise epoch of the published column remains unverified. The two catalogues are correlated definitions of the same reconstruction, not independent samples. The chronology uses speleothem tuning.
"""
    numbers = "\n".join(
        f"{s.event_definition}: N={s.n_response_events}; G={s.gain_bits_per_event:.9f}; "
        f"LR={s.LR_statistic:.9f}; nominal p={s.nominal_LR_p:.9g}; "
        f"Delta AIC={s.delta_AIC_full_minus_reduced:.9f}; phase={s.pre_phase_preferred_deg:.6f}; "
        f"max/min={s.pre_phase_rate_ratio_max_vs_min:.6f}; Rayleigh p={s.rayleigh_p:.9g}."
        for s in (variable, fixed))
    methods = f"""CONTINUOUS-TIME BARKER SPELEOAGE ANALYSIS

Read Supplementary Table S3 in Sheet1 of Barker et al-2011-SOM.xls (header row 9). The second SpeloAge column supplies actual event ages; published DO pick variable threshold and DO pick columns supply the two definitions. Retain pick=1 over 0--400 kyr. Stable event IDs use source Excel row numbers. The two definitions share {shared} rows. No source ages, picks or age realizations are regenerated.

For each catalogue, condition on the exact oldest event and retain all younger exposure to 0 kyr, including the final no-event tail. Unknown pre-anchor history is set to zero as a boundary approximation. History sums strictly older actual events with exp(-age difference / 1.5 kyr) weights; its coefficient is nonpositive. Reduced intensity is exp(intercept + history + LR04 + CO2); full adds precession sine and cosine. Fixed nominal time means and source-interpolated response ranges scale climate. Fitting uses event log intensities minus integrated intensity, with no event bins. Numerical integration splits at events and source interpolation knots.

RESULTS
{numbers}

G is log-likelihood gain per response event, in bits/event. Raw likelihood/AIC values cannot rank the two definitions because their event sets and exact response supports differ. Fixed threshold remains a nominal-age definition sensitivity, with no duplicate bootstrap or chronology pipeline. Main tables contain variable-threshold results; fixed_threshold stores the alternative, and event_definition_sensitivity.csv contains both summary rows.

Source: Barker et al. (2011), Science 334, 347--351, doi:10.1126/science.1203580, Supplementary Table S3. SpeleoAge is treated as BP1950 while its precise source epoch remains unverified. Chronology tuning limits independence from speleothem records.
"""
    (notes_dir / f"{RUN_NAME}_Caption.txt").write_text(caption)
    (notes_dir / f"{RUN_NAME}_Methods_and_results.txt").write_text(methods)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT,
                        help="Project-shaped root; Barker outputs remain below Barker2011")
    parser.add_argument("--no-paper-export", action="store_true")
    args = parser.parse_args(argv)
    result = run_analysis()
    fixed = run_analysis("fixed_threshold")
    root = args.output_root / "Barker2011"
    output = root / "data/processed" / RUN_NAME
    write_outputs(result, output)
    write_outputs(fixed, output / "fixed_threshold")
    pd.concat([result["summary"], fixed["summary"]], ignore_index=True).to_csv(
        output / "event_definition_sensitivity.csv", index=False, float_format="%.12g")
    save_figure(plot_results(result, fixed), root / "figures" / RUN_NAME,
                paper_export=not args.no_paper_export and args.output_root.resolve() == PROJECT_ROOT.resolve())
    write_notes(result, fixed, root / "experiment_note")
    for current in (result, fixed):
        s = current["summary"].iloc[0]
        print(f"{s.event_definition}: {s.n_response_events} response events; G={s.gain_bits_per_event:.6f}; "
              f"LR={s.LR_statistic:.6f}; nominal p={s.nominal_LR_p:.6g}; phase={s.pre_phase_preferred_deg:.3f}")


if __name__ == "__main__":
    main()
