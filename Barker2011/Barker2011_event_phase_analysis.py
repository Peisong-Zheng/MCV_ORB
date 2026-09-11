#!/usr/bin/env python3
"""Barker (2011) SpeleoAge warming events: descriptive phase and conditional PI.

Use Table S3's variable-threshold picks on SpeleoAge over 0–400 kyr as the
primary catalogue; overlay fixed-threshold picks as a point-age sensitivity.
Use the NGRIP–MIS6 main-analysis settings: 0.2-kyr bins, a 1.5-kyr
history window, complete-history response support, and response-range
climate scaling. The reduced model includes history, LR04 and CO2;
the full model adds precession sine and cosine. PI is an in-sample gain.

SpeleoAge numerical values are unchanged and treated as BP1950, with the
column's exact reference year still unverified directly. The single record
needs no segment contrast; sampling resolution is not a model term.
"""

from pathlib import Path
import hashlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from toolbox.phase_response_plotting import format_phase_response_axis, mark_preferred_phase
import numpy as np
import pandas as pd

# Shared tools live one level above this source-specific analysis directory.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from paper_figure_export import copy_pdf_to_paper
from toolbox.catalogue_colors import CATALOGUE_COLORS
from toolbox import combined_pi, event_inputs, orbital_phase, poisson
from toolbox.data_checks import require_unique_values
from toolbox.model_stats import nested_likelihood_metrics
from toolbox.project_config import (
    PROJECT_ROOT, LR04_XLSX, CO2_XLSX, PRE_TXT, ORBITAL_DRIVER_SETTINGS,
    ORBITAL_SOLUTION, ORBITAL_SOURCE_EPOCH, ORBITAL_AGE_OFFSET_TO_BP1950_KA,
)

RUN_NAME = "Barker2011_event_phase_analysis"
OUT_DATA_DIR = ROOT / "data/processed" / RUN_NAME
OUT_FIG_DIR = ROOT / "figures" / RUN_NAME
BARKER_XLS = ROOT / "data/raw/Barker et al-2011-SOM.xls"

DATASET_ID = "barker_variable_threshold_speleo_0_400"
EVENT_TYPE = "barker_do_warming_variable_threshold_speleo"
CATALOGUE_LABEL = "Barker variable-threshold D-O warmings, SpeleoAge 0-400 ka"
AGE_COLUMN = "SpeloAge (kyr).1"  # The second SpeleoAge column belongs to Table S3.
PICK_COLUMN = "DO pick variable threshold"
PICK_COLUMNS = {"variable_threshold": PICK_COLUMN, "fixed_threshold": "DO pick"}
EXPECTED_COUNTS = {"variable_threshold": 70, "fixed_threshold": 59}
DEFINITION_COLORS = {"variable_threshold": CATALOGUE_COLORS["variable"],
                     "fixed_threshold": CATALOGUE_COLORS["fixed"]}
EVENT_COLOR = DEFINITION_COLORS["variable_threshold"]
ANALYSIS_START_KA, ANALYSIS_END_KA = 0.0, 400.0
# Read shared defaults directly so the two main analyses use the same settings.
BIN_WIDTH_KA = combined_pi.DEFAULT_BIN_WIDTH_KA
HISTORY_WINDOW_KA = combined_pi.DEFAULT_HISTORY_WINDOW_KA
BIN_ORIGIN_FRACTION = combined_pi.DEFAULT_ORIGIN_FRACTION
RESPONSE_MODE = combined_pi.DEFAULT_RESPONSE_MODE
# A single record has an intercept, but no between-record contrast.
REDUCED_TERMS = tuple(t for t in combined_pi.REDUCED_TERMS if t != combined_pi.SEGMENT_TERM)
FULL_TERMS = tuple(t for t in combined_pi.FULL_TERMS if t != combined_pi.SEGMENT_TERM)
MODEL_SPECS = (
    ("reduced", REDUCED_TERMS, "History and climate-state model"),
    ("full", FULL_TERMS, "Full model"),
)


def load_barker_source(event_definition="variable_threshold"):
    """Select a published Table S3 pick column on the same SpeleoAge axis."""
    pick_column = PICK_COLUMNS[event_definition]
    raw = pd.read_excel(BARKER_XLS, sheet_name="Sheet1", header=8)
    required = {AGE_COLUMN, pick_column}
    if missing := required.difference(raw.columns):
        raise ValueError(f"Missing Barker columns: {sorted(missing)}")
    events = raw[[AGE_COLUMN, pick_column]].apply(pd.to_numeric, errors="coerce")
    events.columns = ["event_age_ka", "pick_value"]
    events.insert(0, "source_row", raw.index.to_numpy(int))
    events = events.loc[events["pick_value"].eq(1) & events["event_age_ka"].between(
        ANALYSIS_START_KA, ANALYSIS_END_KA
    )].sort_values("event_age_ka").reset_index(drop=True)
    events["source_excel_row"] = events["source_row"] + 10  # Header is Excel row 9.
    events["event_index"] = np.arange(1, len(events) + 1)
    events["dataset_id"] = DATASET_ID.replace("variable_threshold", event_definition)
    events["event_type"] = EVENT_TYPE.replace("variable_threshold", event_definition)
    events["event_label"] = CATALOGUE_LABEL.replace("variable-threshold", event_definition.replace("_", "-"))
    events["event_definition"] = event_definition
    events["age_column"] = AGE_COLUMN
    events["pick_column"] = pick_column
    events["analysis_start_ka"] = ANALYSIS_START_KA
    events["analysis_end_ka"] = ANALYSIS_END_KA
    events["source"] = str(BARKER_XLS.relative_to(PROJECT_ROOT))
    require_unique_values(events, "event_age_ka", context="Barker SpeleoAge events")

    return events


def prepare_bins(events):
    """Use the pooled analysis's segment grid, history rule and climate scaling."""
    if RESPONSE_MODE != "maximal_for_history":
        raise ValueError("Barker has no separately specified common-core support")
    row = pd.Series(dict(segment_id="Barker2011",
                         observation_start_kyr_bp=ANALYSIS_START_KA,
                         observation_end_kyr_bp=ANALYSIS_END_KA))
    # The exact response boundary is inserted even when it cuts a 0.2-kyr bin.
    bins, segment = combined_pi._build_segment_grid(
        row, response_start=ANALYSIS_START_KA, response_end=ANALYSIS_END_KA - HISTORY_WINDOW_KA,
        history_window_ka=HISTORY_WINDOW_KA, bin_width_ka=BIN_WIDTH_KA,
        origin_fraction=BIN_ORIGIN_FRACTION, lr04_path=LR04_XLSX,
        co2_path=CO2_XLSX, precession_path=PRE_TXT, project_root=PROJECT_ROOT,
    )
    bins = bins.drop(columns=combined_pi.SEGMENT_TERM).rename(columns={
        "bin_start_kyr_bp": "bin_start_ka", "bin_end_kyr_bp": "bin_end_ka",
        "bin_center_kyr_bp": "bin_center_ka", "dt_kyr": "dt_ka",
    })
    counts, _ = np.histogram(events["event_age_ka"], bins=segment.bin_edges)
    bins["event_count"] = counts
    bins["same_type_history_count"] = combined_pi.history_from_counts(counts, segment)
    bins["dataset_id"] = events["dataset_id"].iloc[0]
    bins["dataset_label"] = events["event_label"].iloc[0]
    response = bins["in_response_interval"]
    scaling = []
    for forcing in ("lr04", "co2"):
        _, mean, minimum, maximum, span = event_inputs.scale_to_zero_mean_range_one(
            bins.loc[response, forcing].to_numpy(float)
        )
        bins[f"{forcing}_scaled"] = (bins[forcing] - mean) / span
        scaling.append(dict(forcing_id=forcing, mean=mean, min=minimum, max=maximum, range=span,
                            support_start_ka=segment.response_start_kyr_bp,
                            support_end_ka=segment.response_end_kyr_bp))
    return bins, pd.DataFrame(scaling)


def fit_models(frame):
    """Fit the matched reduced/full models and test the two added phase terms."""
    models = [poisson.fit_poisson_model(frame, name, terms, label)
              for name, terms, label in MODEL_SPECS]
    reduced, full = models
    metrics = nested_likelihood_metrics(
        loglik_full=full.log_likelihood, loglik_reduced=reduced.log_likelihood,
        df=len(full.beta) - len(reduced.beta), n_bins=len(frame),
        n_events=int(frame["event_count"].sum()), aicc_full=full.aicc, aicc_reduced=reduced.aicc,
    )
    tests = pd.DataFrame([dict(dataset_id=frame["dataset_id"].iloc[0], comparison_id="phase_after_climate",
                              reduced_model_id="reduced", full_model_id="full", **metrics,
                              p_value_method="nominal asymptotic chi-square LRT")])
    return models, tests


def run_analysis(event_definition="variable_threshold"):
    """Analyze one definition; default calls retain the primary 70-event record."""
    events = load_barker_source(event_definition)
    bins, scaling = prepare_bins(events)
    frame = bins.loc[bins["in_response_interval"]].copy()
    models, likelihood_tests = fit_models(frame)
    model_summary = poisson.build_model_summary(models, frame)

    phase_product = orbital_phase.build_phase_series("pre", ORBITAL_DRIVER_SETTINGS["pre"])
    event_phases = orbital_phase.sample_event_phases(events, {"pre": phase_product})
    rayleigh = orbital_phase.rayleigh_test(event_phases["phase_rad"].to_numpy(float))
    response_start, response_end = frame["bin_start_ka"].min(), frame["bin_end_ka"].max()
    events["included_in_pi_response"] = events["event_age_ka"].between(
        response_start, response_end, inclusive="left"
    )
    event_phases["included_in_pi_response"] = events["included_in_pi_response"].to_numpy()
    event_phases["source_row"] = events["source_row"].to_numpy()
    bins["included_in_pi_response"] = bins["in_response_interval"]
    for model in models:
        bins.loc[frame.index, f"{model.model_id}_rate_per_kyr"] = model.fitted_rate_per_kyr

    phase_test = likelihood_tests.loc[likelihood_tests["comparison_id"].eq("phase_after_climate")].iloc[0]
    full = model_summary.loc[model_summary["model_id"].eq("full")].iloc[0]
    summary = pd.DataFrame([dict(
        dataset_id=events.dataset_id.iloc[0], catalogue_label=events.event_label.iloc[0], age_scale="SpeleoAge",
        event_definition=event_definition, analysis_start_ka=ANALYSIS_START_KA,
        analysis_end_ka=ANALYSIS_END_KA, bin_width_ka=BIN_WIDTH_KA, history_window_ka=HISTORY_WINDOW_KA,
        bin_origin_fraction=BIN_ORIGIN_FRACTION, response_mode=RESPONSE_MODE,
        n_rayleigh_events=len(events), n_predictive_events=int(frame["event_count"].sum()),
        n_pi_bins=len(frame), pi_support_start_ka=response_start, pi_support_end_ka=response_end,
        response_exposure_kyr=frame["dt_ka"].sum(),
        rayleigh_mean_phase_deg=rayleigh["mean_phase_deg"],
        rayleigh_mean_resultant_length=rayleigh["mean_resultant_length"], rayleigh_p=rayleigh["rayleigh_p"],
        LR_statistic=phase_test["LR_statistic"], nominal_LR_p=phase_test["LR_p_value"],
        info_bits_per_event=phase_test["info_bits_per_event"],
        delta_AICc_full_minus_reduced=phase_test["delta_AICc_full_minus_reduced"],
        pre_phase_preferred_deg=full["pre_phase_preferred_deg"],
        pre_phase_rate_ratio_max_vs_min=full["pre_phase_rate_ratio_max_vs_min"],
        all_models_converged=all(m.converged for m in models),
        likelihood_nesting_ok=all(m.log_likelihood >= r.log_likelihood - 1e-8 for r, m in zip(models, models[1:])),
        eta_clipping_used=any(m.n_eta_clipped_low or m.n_eta_clipped_high for m in models),
        resolution_covariate_included=False, age_uncertainty_propagated=False,
        event_age_epoch="BP1950 assumed; exact SpeleoAge epoch not directly verified",
    )])
    result = dict(events=events, bins=bins, model_frame=frame,
                  event_phases=event_phases, rayleigh=rayleigh, models=models,
                  model_summary=model_summary, likelihood_tests=likelihood_tests,
                  coefficients=poisson.build_coefficient_table(models), scaling=scaling, summary=summary)
    validate_results(result)
    return result


def validate_results(result):
    """Check event membership, complete history and numerical fit diagnostics."""
    events, bins, frame = result["events"], result["bins"], result["model_frame"]
    summary = result["summary"].iloc[0]
    expected = EXPECTED_COUNTS[summary.event_definition]
    if len(events) != expected or int(frame["event_count"].sum()) != expected:
        raise ValueError(f"Expected all {expected} {summary.event_definition} events in the response interval")
    if not np.isclose(frame["dt_ka"].sum(), ANALYSIS_END_KA - HISTORY_WINDOW_KA - ANALYSIS_START_KA):
        raise ValueError("Response exposure does not match the complete-history interval")
    if not np.all(np.diff(events["event_age_ka"]) > 0) or bins["event_count"].sum() != len(events):
        raise ValueError("Event ages must be ordered and fully represented in the bins")
    if events["included_in_pi_response"].sum() != frame["event_count"].sum():
        raise ValueError("Point-age and binned response membership disagree")
    if not frame["same_type_history_complete"].all() or result["event_phases"]["phase_extrapolated"].any():
        raise ValueError("Incomplete event history or an extrapolated event phase")
    if not np.isfinite(frame[list(FULL_TERMS)].to_numpy()).all():
        raise ValueError("A predictor contains missing or non-finite values")
    metrics = ["LR_statistic", "nominal_LR_p", "info_bits_per_event",
               "pre_phase_preferred_deg", "pre_phase_rate_ratio_max_vs_min"]
    if not np.isfinite(summary[metrics].to_numpy(float)).all():
        raise ValueError("A phase-result metric is non-finite")
    if not summary.all_models_converged or not summary.likelihood_nesting_ok or summary.eta_clipping_used:
        raise ValueError("Inspect convergence, likelihood nesting or numerical clipping")


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
    bins, summary = result["bins"], result["summary"].iloc[0]

    # Gray exposure supplies history but is excluded from the PI response.
    timeline.axvspan(summary.pi_support_end_ka, ANALYSIS_END_KA, color="#E6E6E6", lw=0, zorder=0)
    timeline.plot(bins["bin_center_ka"], bins["precession_index"], color="#555555", lw=1.05, zorder=1)
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
        full = next(model for model in current["models"] if model.model_id == "full")
        beta = dict(zip(full.terms, full.beta[1:]))
        phase, multiplier = phase_response_curve(beta["pre_phase_sin"], beta["pre_phase_cos"])
        response.plot(phase, multiplier, color=color, lw=1.8, ls="--" if fixed else "-")
        response.axvline(s.pre_phase_preferred_deg, color=color, lw=0.8, ls=":", alpha=0.85)
        # Pair each definition's conditional fit with its phase and effect size.
        mark_preferred_phase(response, s.pre_phase_preferred_deg, s.pre_phase_rate_ratio_max_vs_min, color)
        response.text(0.06, 0.96 - 0.15 * index,
                      f"PI = {s.info_bits_per_event:.3f}; nominal p = {s.nominal_LR_p:.3f}\n"
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
    response.set_ylim(0, 2.5)
    response.set_title("Fitted warming-event rate")
    response.grid(False)
    response.spines[["top", "right"]].set_visible(False)
    _add_panel_label(timeline, "a", x=-0.047)
    _add_panel_label(polar, "b", x=-0.21)
    _add_panel_label(response, "c", x=-0.06)
    return fig


def build_parameters(result):
    """Record source hashes, model choices, scaling and explicit age assumptions."""
    summary = result["summary"].iloc[0]
    frame = result["model_frame"]
    rows = [
        ("dataset_id", summary.dataset_id, "", "one SpeleoAge catalogue"),
        ("event_age_column", AGE_COLUMN, "kyr", "Table S3, Sheet1; Excel header row 9"),
        ("event_pick_column", PICK_COLUMNS[summary.event_definition], "", "retain value 1"),
        ("event_age_reference", "BP1950 assumed", "", "exact SpeleoAge column epoch not directly verified"),
        ("event_age_offset_applied", 0.0, "kyr", "source numerical ages retained"),
        ("observation_start", ANALYSIS_START_KA, "kyr BP", "source-age assumption above"),
        ("observation_end", ANALYSIS_END_KA, "kyr BP", "source-age assumption above"),
        ("pi_support_start", summary.pi_support_start_ka, "kyr BP", "complete-history response"),
        ("pi_support_end", summary.pi_support_end_ka, "kyr BP", "oldest history-window interval is excluded from response"),
        ("bin_width", BIN_WIDTH_KA, "kyr", "event-count bins; log duration is the offset"),
        ("history_window", HISTORY_WINDOW_KA, "kyr", "older bin centers in (t, t + window]; current bin excluded"),
        ("bin_origin_fraction", BIN_ORIGIN_FRACTION, "bin width", "shared main-analysis origin"),
        ("response_mode", RESPONSE_MODE, "", "same complete-history rule as NGRIP–MIS6"),
        ("reduced_model_terms", "+".join(REDUCED_TERMS), "", "history, LR04 and CO2"),
        ("full_model_terms", "+".join(FULL_TERMS), "", "adds precession sine and cosine"),
        ("scaling_support", f"{summary.pi_support_start_ka:g}–{summary.pi_support_end_ka:g}",
         "kyr", "zero-mean/range-one scaling over the actual response bins"),
        ("resolution_covariate_included", False, "", "aligned with NGRIP–MIS6 main analysis"),
        ("segment_contrast_included", False, "", "Barker contains one continuous observation segment"),
        ("phase_extrapolated_response_bins", int(frame["pre_phase_extrapolated"].sum()), "bins", "edge-phase flags retained from the original model"),
        ("phase_extrapolated_events", int(result["event_phases"]["phase_extrapolated"].sum()), "events", "all descriptive events checked"),
        ("orbital_solution", ORBITAL_SOLUTION, "", "shared with NGRIP–MIS6"),
        ("orbital_source_epoch", ORBITAL_SOURCE_EPOCH, "", "astronomical source-file convention"),
        ("orbital_age_offset_to_BP1950", ORBITAL_AGE_OFFSET_TO_BP1950_KA, "kyr", "applied once by the shared orbital tools"),
        ("phase_convention", "minimum=0; maximum=180", "degrees", "phase increases toward older BP ages"),
        ("p_value_method", "nominal asymptotic chi-square LRT", "df=2", "not bootstrap calibrated"),
        ("PI_interpretation", "in-sample conditional likelihood gain", "bits/event", "not validated out-of-sample prediction"),
        ("age_uncertainty_propagated", False, "", "point-age analysis only"),
        ("chronology_construction", "speleothem-tuned synthetic Greenland record", "", "does not establish chronology-independent replication"),
    ]
    for name, path in [("event_source", BARKER_XLS),
                       ("lr04_source", LR04_XLSX), ("co2_source", CO2_XLSX), ("precession_source", PRE_TXT)]:
        rows.append((name, str(path.relative_to(PROJECT_ROOT)), "", "active input"))
        rows.append((f"{name}_sha256", hashlib.sha256(path.read_bytes()).hexdigest(), "", "input content hash"))
    return pd.DataFrame(rows, columns=["parameter", "value", "unit", "note"])


def write_outputs(result, output_dir=OUT_DATA_DIR):
    """Save one catalogue; the alternative uses its own output subdirectory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "analysis_summary.csv": result["summary"],
        "event_catalogue_used.csv": result["events"],
        "event_precession_phases.csv": result["event_phases"],
        "binned_inputs_and_fitted_rates.csv": result["bins"],
        "model_summary.csv": result["model_summary"],
        "predictive_likelihood_tests.csv": result["likelihood_tests"],
        "predictive_coefficients.csv": result["coefficients"],
        "predictor_scaling.csv": result["scaling"],
        "parameters_and_provenance.csv": build_parameters(result),
    }
    for name, frame in tables.items():
        frame.to_csv(output_dir / name, index=False, float_format="%.12g")


def save_figure(fig, output_dir=OUT_FIG_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    png, pdf = [output_dir / f"{RUN_NAME}.{suffix}" for suffix in ("png", "pdf")]
    fig.savefig(png, dpi=600, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    copy_pdf_to_paper(pdf)
    plt.close(fig)
    return png, pdf


def write_notes(result, fixed_result):
    """Keep the manuscript caption and definition comparison tied to saved fits."""
    variable, fixed = [r["summary"].iloc[0] for r in (result, fixed_result)]
    shared = len(set(result["events"].source_row) & set(fixed_result["events"].source_row))
    results_text = "\n".join(
        f"{s.event_definition}: n={s.n_predictive_events}; PI={s.info_bits_per_event:.6f} bits/event; "
        f"LR={s.LR_statistic:.6f}; nominal p={s.nominal_LR_p:.6g}; "
        f"Delta AICc={s.delta_AICc_full_minus_reduced:.6f}; "
        f"preferred phase={s.pre_phase_preferred_deg:.3f} degrees; "
        f"maximum/minimum rate ratio={s.pre_phase_rate_ratio_max_vs_min:.4f}; "
        f"Rayleigh mean={s.rayleigh_mean_phase_deg:.3f} degrees, "
        f"Rbar={s.rayleigh_mean_resultant_length:.4f}, p={s.rayleigh_p:.6g}."
        for s in (variable, fixed)
    )
    caption = f"""Barker et al. (2011) SpeleoAge warming events: sensitivity to the published event definition.

Rose denotes variable threshold (primary catalogue, n = {variable.n_rayleigh_events}); green denotes fixed threshold (sensitivity, n = {fixed.n_rayleigh_events}). Both use Table S3's SpeleoAge column and the same 0–400 kyr observation interval. (a) Events on the La2004 precession index, shown by filled circles and open squares, respectively. Shared picks display both symbols. Older ages are on the left and younger ages on the right. The gray 398.5–400 kyr interval supplies event history but is excluded from the PI response; all events lie within the 0–398.5 kyr response interval. Ages are treated as kyr BP relative to 1950; the precise reference year of the source SpeleoAge column remains unverified. Astronomical ages receive the documented J2000-to-BP1950 correction.

(b) Descriptive event counts in twelve 30° sectors: filled rose bars and dashed green outlines. Phase zero denotes a precession-index minimum, 180° a maximum, and phase increases toward older BP ages. Colored arrows show mean directions; both arrow lengths equal the mean resultant length times the largest sector count across the two catalogues. Radial ticks denote event counts. Text gives mean resultant length (Rbar) and nominal Rayleigh p in matching colors.

(c) Fitted warming-event rate multipliers exp[beta_sin sin(phi) + beta_cos cos(phi)], shown by a solid rose line and a dashed green line. Peak dots and vertical dotted lines mark fitted preferred phases; the horizontal dotted line marks unity. Minimum/Maximum tick labels identify precession-index extrema, not rate extrema. The smooth curves follow the sine/cosine model; unity means zero phase contribution in the full model, not the mean rate or a separately refitted reduced-model rate. The reduced Poisson model contains an intercept, a preceding 1.5-kyr event count, LR04 and atmospheric CO2; the full model adds phase sine and cosine. Both definitions use the same 0.2-kyr nominal bins, actual bin-duration offsets and response-range climate scaling, with history recomputed from each catalogue. PI is the fitted log-likelihood gain in bits per event; panel p values are nominal chi-square likelihood-ratio p values with two degrees of freedom. Preferred phases are {variable.pre_phase_preferred_deg:.1f}° and {fixed.pre_phase_preferred_deg:.1f}°, and maximum/minimum phase rate ratios are {variable.pre_phase_rate_ratio_max_vs_min:.2f} and {fixed.pre_phase_rate_ratio_max_vs_min:.2f}, respectively.

This figure compares point-age estimates. The catalogues are alternative definitions of the same source record, not independent samples. Reduced-model bootstrap and chronology sensitivity are separate analyses of the primary variable-threshold catalogue only; no bootstrap p or age-uncertainty band is attached to the fixed-threshold result.
"""
    methods = f"""BARKER SPELEOAGE: MAIN ANALYSIS AND EVENT-DEFINITION SENSITIVITY

QUESTION AND SOURCE
Compare the association between precession phase and warming-event occurrence
under the two published detection definitions. Read Table S3 from Sheet1 of
Barker et al-2011-SOM.xls (Excel header row 9): SpeloAge (kyr).1 supplies both
age axes; DO pick variable threshold and DO pick supply the two pick columns.
Retain pick value 1 and ages from 0 to 400 kyr. No event re-detection, new
threshold selection or EDC3-age conversion is performed. The definitions share
{shared} source rows; matching uses source-row identity, not an age tolerance.

METHOD
The variable-threshold catalogue remains primary. Both definitions use the
current NGRIP–MIS6 settings: 0.2-kyr nominal bins, origin zero, 1.5-kyr older-bin
history excluding the current bin, and complete-history response support.
Observation is 0–400 kyr; response is 0–398.5 kyr (1993 bins, 398.5 kyr exposure).
The response boundary is inserted exactly, retaining the final 0.1-kyr bin.
Event counts and history are calculated separately for each definition.
The reduced model is intercept + history + LR04 + CO2; the full model adds
precession sine and cosine. Climate predictors use the same centering and
range over the common response support. One record needs no segment contrast.
PI = (logL_full - logL_reduced)/(n_response_events * ln(2)); LR = twice the
log-likelihood gain. The LRT uses two degrees of freedom and is nominal.
The descriptive Rayleigh calculation is separate from the conditional model.

POINT-AGE RESULTS
{results_text}

INTERPRETATION AND SCOPE
Differences concern detection definition, including changed event membership
and its effect on the history covariate. PI is reported per response event;
raw log-likelihoods and p values from the two catalogues do not provide a test
that one definition is better. The two estimates are correlated because they
derive from the same reconstruction, and are not independent replications.
These are conditional associations, not demonstrated out-of-sample prediction.
The fixed-threshold result has no separate age Monte Carlo or bootstrap.
The primary-catalogue reduced-model bootstrap is in Barker2011_PI_bootstrap.py;
the existing two chronology scripts continue to use variable threshold only.
SpeleoAge is retained numerically under the BP1950 working assumption. The
source chronology's speleothem tuning and unresolved precise epoch remain
limitations; changing event definition does not resolve them.

OUTPUTS
The main directory preserves primary-only tables used by downstream scripts.
The fixed_threshold subdirectory contains the same tables for the alternative.
event_definition_sensitivity.csv combines the two summary rows for comparison.
The original three-panel PNG/PDF now overlays both definitions.
All input paths, hashes, model terms and scales are saved with each catalogue.

SOURCE
Barker et al. (2011), Science 334, 347–351, doi:10.1126/science.1203580,
Supplementary Table S3. The published workbook is retained unchanged.
"""
    notes = ROOT / "experiment_note"
    notes.mkdir(parents=True, exist_ok=True)
    (notes / f"{RUN_NAME}_Caption.txt").write_text(caption)
    (notes / f"{RUN_NAME}_Methods_and_results.txt").write_text(methods)


def main():
    result = run_analysis()
    fixed_result = run_analysis("fixed_threshold")
    write_outputs(result)
    write_outputs(fixed_result, OUT_DATA_DIR / "fixed_threshold")
    pd.concat([result["summary"], fixed_result["summary"]], ignore_index=True).to_csv(
        OUT_DATA_DIR / "event_definition_sensitivity.csv", index=False, float_format="%.12g")
    png, pdf = save_figure(plot_results(result, fixed_result))
    write_notes(result, fixed_result)
    summary = result["summary"].iloc[0]
    print(f"Barker SpeleoAge: {summary.n_rayleigh_events} descriptive events, "
          f"{summary.n_predictive_events} PI responses over {summary.pi_support_start_ka:g}–{summary.pi_support_end_ka:g} kyr")
    print(f"PI={summary.info_bits_per_event:.5f} bits/event; nominal p={summary.nominal_LR_p:.5f}; "
          f"preferred phase={summary.pre_phase_preferred_deg:.2f}°; max/min rate={summary.pre_phase_rate_ratio_max_vs_min:.3f}")
    print(f"Rayleigh (descriptive): p={summary.rayleigh_p:.5f}")
    fixed = fixed_result["summary"].iloc[0]
    print(f"Fixed-threshold sensitivity: {fixed.n_predictive_events} events; "
          f"PI={fixed.info_bits_per_event:.5f}; nominal p={fixed.nominal_LR_p:.5f}; "
          f"preferred phase={fixed.pre_phase_preferred_deg:.2f}°; max/min rate={fixed.pre_phase_rate_ratio_max_vs_min:.3f}")
    print(f"Tables: {OUT_DATA_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Figures: {png.relative_to(PROJECT_ROOT)}, {pdf.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
