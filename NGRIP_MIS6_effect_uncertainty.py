#!/usr/bin/env python3
"""Compare chronology, sampling and combined phase-effect uncertainty.

Chronology uses all saved age fits. Sampling simulates 5,000 full-model
catalogues at nominal ages. Combined uses 200 chronologies with 50 refits each.
All displayed ranges project coefficient ellipses constructed in the same way;
only nominal-age sampling gives an approximate confidence region.
Both ensembles condition on observed initial events. The reduced-model null
bootstrap remains a separate experiment and is not rerun by this script.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
from paper_figure_export import copy_pdf_to_paper
import platform
import tempfile
import time

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mcv_orb_matplotlib"))
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from toolbox.phase_response_plotting import format_phase_response_axis, mark_preferred_phase
from toolbox.figure_style import add_panel_label
import numpy as np
import pandas as pd
import scipy

from toolbox import combined_likelihood
from toolbox import effect_uncertainty as effect
from toolbox.point_process import PointProcessFitError
from toolbox.point_process_diagnostics import residual_statistics
from toolbox.project_config import PROJECT_ROOT, CO2_XLSX, LR04_XLSX, PRE_TXT


RUN_NAME = "NGRIP_MIS6_effect_uncertainty"
OUTPUT_DIR = PROJECT_ROOT / "data/processed" / RUN_NAME
FIGURE_DIR = PROJECT_ROOT / "figures" / RUN_NAME
AGE_DIR = PROJECT_ROOT / "data/processed/NGRIP_MIS6_event_uncertainty_sensitivity"
AGE_DRAWS = AGE_DIR / "combined_event_age_realizations.csv"
AGE_RESULTS = AGE_DIR / "gain_realizations.csv"
DEFAULT_SEED = 20260908
DEFAULT_POINT_DRAWS = 5_000
DEFAULT_OUTER_DRAWS = 200
DEFAULT_INNER_DRAWS = 50
BETA_COLUMNS = [f"beta__{term}" for term in ("intercept", *combined_likelihood.FULL_TERMS)]
PHASE_COLUMNS = [f"beta__{term}" for term in effect.PHASE_TERMS]
SCENARIOS = {"A_chronology": "chronology", "B_sampling": "sampling", "C_joint": "combined"}
INTERVAL_TYPES = {
    "A_chronology": "95_joint_working_region_projection",
    "B_sampling": "approximate_95_joint_confidence_region_projection",
    "C_joint": "95_joint_working_region_projection",
}

_CONTEXT = None
_GENERATORS = None
_PREPARED = None


def load_age_inputs(events, age_results_path=AGE_RESULTS):
    """Read the paired ensemble, retaining the original support-failure audit."""
    draws = pd.read_csv(AGE_DRAWS, float_precision="round_trip")
    results = pd.read_csv(age_results_path)
    ids = ["realization_id", "ngrip_realization_id", "mis6_realization_id"]
    if draws.realization_id.duplicated().any() or results.realization_id.duplicated().any():
        raise ValueError("Age input IDs must be unique")
    pd.testing.assert_frame_equal(draws[ids], results[ids])
    if not results.fit_valid.isin([True, False]).all():
        raise ValueError("Every saved age realization needs an explicit support status")
    columns = [f"age_kyr_bp__{event_id}" for event_id in events.event_id]
    ages = draws[columns].to_numpy(float)
    if not np.isfinite(ages).all() or not np.all(np.diff(ages, axis=1) > 0):
        raise ValueError("Saved age sequences must be finite and ordered")
    valid = results.fit_valid.eq(True)
    if results.loc[~valid, "invalid_reason"].fillna("").eq("").any():
        raise ValueError("Outside-support age rows must retain their reason")
    metrics = ["pre_phase_preferred_deg", "pre_phase_rate_ratio_max_vs_min"]
    if not np.isfinite(results.loc[valid, metrics].to_numpy(float)).all():
        raise ValueError("Valid age fits need finite effect estimates")
    return draws, results


def build_generators(events, context, point_fit, draws, results, n_outer, seed,
                     age_columns=None, source_id_columns=("ngrip_realization_id", "mis6_realization_id")):
    """Uniformly sample distinct valid chronologies; fit each full model once."""
    valid_indices = np.flatnonzero(results.fit_valid.to_numpy(bool))
    if not 1 <= n_outer <= len(valid_indices):
        raise ValueError("Requested outer count exceeds valid age realizations")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 100]))
    selected = rng.choice(valid_indices, size=n_outer, replace=False)
    columns = age_columns if age_columns is not None else [
        f"age_kyr_bp__{event_id}" for event_id in events.event_id]
    rows = []
    generators = {0: {"model": point_fit.full, "events": events.copy()}}
    for outer_id, index in enumerate(selected, 1):
        source = draws.iloc[index]
        local_events = events.copy()
        local_events[combined_likelihood.EVENT_AGE_COLUMN] = source[columns].to_numpy(float)
        local_context = combined_likelihood.condition_context(context, local_events)
        fit = combined_likelihood.fit_catalogue(local_events, local_context, fixed_support=True)
        effect.validate_effect_fit(fit)
        # Detect stale or mismatched G and age files before simulation.
        np.testing.assert_allclose(
            [fit.summary[k] for k in ("LR_statistic", "gain_bits_per_event")],
            results.loc[index, ["LR_statistic", "gain_bits_per_event"]].to_numpy(float),
            rtol=2e-6, atol=2e-6,
        )
        generators[outer_id] = {"model": fit.full, "events": local_events}
        row = {"outer_id": outer_id, "source_row_index": int(index),
               "age_realization_id": source.realization_id,
               "n_response_events": fit.summary["n_response_events"],
               "response_exposure_kyr": local_context.response_exposure_kyr}
        for segment_id, segment in local_context.segments.items():
            row[f"response_end_kyr_bp__{segment_id}"] = segment.response_end_kyr_bp
        row.update({column: source[column] for column in source_id_columns})
        row.update({f"beta__{term}": beta for term, beta in zip(fit.full.terms, fit.full.beta)})
        rows.append(row)
    return generators, pd.DataFrame(rows)


def _initialize_worker(context, generators):
    global _CONTEXT, _GENERATORS, _PREPARED
    _CONTEXT, _GENERATORS, _PREPARED = context, generators, {}


def _simulate_one(task):
    scenario, outer_id, inner_id, seed = task
    if outer_id not in _PREPARED:
        generator = _GENERATORS[outer_id]
        local_context = combined_likelihood.condition_context(_CONTEXT, generator["events"])
        prepared = effect.prepare_full_simulation(local_context, generator["model"])
        _PREPARED[outer_id] = (local_context, prepared)
    local_context, prepared = _PREPARED[outer_id]
    # Private streams depend on scientific replicate identity, not worker order.
    rng = np.random.default_rng(np.random.SeedSequence([seed, scenario, outer_id, inner_id]))
    row = {"scenario": "B_sampling" if scenario == 1 else "C_joint",
           "outer_id": outer_id, "inner_id": inner_id, "replicate_id": inner_id, "seed": seed,
           "fit_valid": True, "invalid_reason": "",
           "response_exposure_kyr": local_context.response_exposure_kyr}
    try:
        events = effect.simulate_prepared_full_events(prepared, rng)
        row["n_observation_events"] = len(events)
        fit = combined_likelihood.fit_catalogue(events, local_context, fixed_support=True)
        if not fit.summary["n_response_events"]:
            row.update(n_response_events=0, effect_identified=False, phase_deg=np.nan,
                       rate_ratio=np.nan, full_log_likelihood=0., reduced_log_likelihood=0.)
            row.update({f"beta__{term}": np.nan for term in fit.full.terms})
            if scenario == 1:
                row.update(ks_uniform=0., adjacent_dependence=0., residual_status="no_response_events")
            return row
        effect.validate_effect_fit(fit)
        row["effect_identified"] = True
        row["n_response_events"] = fit.summary["n_response_events"]
        row.update({f"beta__{term}": beta for term, beta in zip(fit.full.terms, fit.full.beta)})
        phase, ratio = effect.phase_and_ratio(effect.phase_coefficients(fit.full))
        row.update(phase_deg=float(phase), rate_ratio=float(ratio),
                   full_log_likelihood=fit.full.log_likelihood,
                   reduced_log_likelihood=fit.reduced.log_likelihood)
        if scenario == 1:
            residuals = residual_statistics(*combined_likelihood.rescaled_event_intervals(fit))
            row.update(residuals["statistics"])
            row["residual_status"] = residuals["status"]
    except (effect.InvalidEffectSimulation, PointProcessFitError) as error:
        row.update(fit_valid=False, invalid_reason=str(error))
    return row


def run_simulations(context, generators, n_point, n_inner, seed, workers=1, show_progress=True):
    if min(n_point, n_inner, workers) < 1:
        raise ValueError("Simulation counts and workers must be positive")
    tasks = [(1, 0, i, seed) for i in range(1, n_point + 1)]
    tasks += [(2, outer, i, seed) for outer in sorted(generators) if outer != 0
              for i in range(1, n_inner + 1)]
    rows = []

    def collect(iterator):
        for i, row in enumerate(iterator, 1):
            rows.append(row)
            if show_progress and (i % 1000 == 0 or i == len(tasks)):
                print(f"Full-model simulation {i:,}/{len(tasks):,}", flush=True)

    if workers == 1:
        _initialize_worker(context, generators)
        collect(map(_simulate_one, tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                 initializer=_initialize_worker,
                                 initargs=(context, generators)) as pool:
            collect(pool.map(_simulate_one, tasks, chunksize=20))
    return pd.DataFrame(rows)


def summarize_effects(point_fit, age_results, replicates):
    """Use matched ellipse projections, retaining each ensemble's interpretation."""
    if not replicates.fit_valid.all():
        raise RuntimeError("Simulation failures saved; review them before publishing intervals")
    if 'effect_identified' in replicates and not replicates.effect_identified.all():
        raise RuntimeError("Unidentified effect draws retained; confidence region is not reported")
    point = effect.phase_coefficients(point_fit.full)
    point_phase, point_ratio = effect.phase_and_ratio(point)
    b = replicates.loc[replicates.scenario.eq("B_sampling")]
    c = replicates.loc[replicates.scenario.eq("C_joint")]
    if c.groupby("outer_id").size().nunique() != 1:
        raise ValueError("Joint groups must have equal weight and complete inner samples")
    samples = {
        "A_chronology": age_results.loc[age_results.fit_valid,
                                        ["beta_pre_phase_sin", "beta_pre_phase_cos"]].to_numpy(float),
        "B_sampling": b[PHASE_COLUMNS].to_numpy(float),
        "C_joint": c[PHASE_COLUMNS].to_numpy(float),
    }
    rows, regions = [], {}
    for scenario, coefficients in samples.items():
        # Reuse the ellipse geometry for all ensembles; only sampling is a bootstrap CI.
        region = effect.bootstrap_joint_region(point, coefficients)
        region["n"] = len(coefficients)
        region["interval_type"] = INTERVAL_TYPES[scenario]
        regions[scenario] = region
        bounds = effect.project_joint_region(region)
        for quantity, estimate, low, high in (
            ("preferred_phase_deg", point_phase, bounds["phase_low_unwrapped_deg"], bounds["phase_high_unwrapped_deg"]),
            ("max_min_rate_ratio", point_ratio, bounds["ratio_low"], bounds["ratio_high"]),
        ):
            rows.append(dict(scenario=scenario, quantity=quantity, point_estimate=float(estimate),
                             center=float(estimate), low=low, high=high, n=len(coefficients),
                             interval_type=INTERVAL_TYPES[scenario],
                             phase_identified=bounds["phase_identified"]))
    summary = pd.DataFrame(rows).sort_values(["quantity", "scenario"]).reset_index(drop=True)
    return summary, regions


def build_curve_table(point_fit, regions):
    """Project each complete coefficient region to a phase-curve envelope."""
    phases = np.arange(0.0, 361.0, 1.0)
    radians = np.deg2rad(phases)
    direction = np.column_stack((np.sin(radians), np.cos(radians)))
    point = effect.phase_coefficients(point_fit.full)
    table = {"phase_deg": phases, "point_multiplier": np.exp(direction @ point)}
    for scenario, name in SCENARIOS.items():
        table[f"{name}_low"], table[f"{name}_high"] = effect.joint_region_curve_band(regions[scenario], phases)
    return pd.DataFrame(table)


def plot_results(summary, curves, regions, figure_dir=FIGURE_DIR, export=True,
                 run_name=RUN_NAME, compact_ratio_ticks=False):
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(180 / 25.4, 145 / 25.4))
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.12, top=0.88, hspace=0.65, wspace=0.85)
    colors = {"A_chronology": "#747474", "B_sampling": "#0072B2", "C_joint": "#D55E00"}
    labels = {"A_chronology": "Chronology", "B_sampling": "Sampling", "C_joint": "Sampling + chronology"}
    styles = {"A_chronology": ":", "B_sampling": "-", "C_joint": "--"}
    fig.legend([Line2D([], [], color=colors[key], linestyle=styles[key], linewidth=1.6) for key in colors],
               list(labels.values()), loc="upper center", bbox_to_anchor=(0.54, 0.985),
               ncol=3, fontsize=8, frameon=False, columnspacing=1.5)
    for ax, quantity, xlabel, title in (
        (axes[0, 0], "preferred_phase_deg", "Preferred phase (°; unwrapped)", "Phase uncertainty"),
        (axes[0, 1], "max_min_rate_ratio", "Maximum/minimum rate ratio", "Effect strength"),
    ):
        for i, scenario in enumerate(colors):
            row = summary.loc[summary.scenario.eq(scenario) & summary.quantity.eq(quantity)].iloc[0]
            ax.hlines(i, row.low, row.high, color=colors[scenario], linewidth=2)
            ax.plot(row.center, i, "o", color=colors[scenario], markersize=4)
        ax.axvline(row.point_estimate, color="black", linestyle="--", linewidth=0.9)
        ax.set(yticks=range(3), yticklabels=["Chronology", "Sampling", "Sampling +\nchronology"],
               xlabel=xlabel, ylim=(2.5, -0.5))
        ax.tick_params(axis="y", labelsize=8)
        ax.set_title(title, fontsize=9, fontweight="normal", pad=9)
        if quantity == "max_min_rate_ratio":
            ax.set_xscale("log")
            ax.axvline(1, color="#999999", linewidth=0.7)
            if compact_ratio_ticks:
                ax.set_xticks([1, 2, 3, 5, 10], labels=["1", "2", "3", "5", "10"])
            else:
                ax.set_xticks([1, 3, 10, 30], labels=["1", "3", "10", "30"])
            ax.minorticks_off()
    ax = axes[1, 0]
    for scenario, region in regions.items():
        boundary = effect.ellipse_boundary(region, np.linspace(0, 2 * np.pi, 721))
        ax.plot(boundary[:, 1], boundary[:, 0], color=colors[scenario],
                linestyle=styles[scenario], linewidth=1.6)
    ax.plot(region["center"][1], region["center"][0], "ko", markersize=4)
    ax.plot(0, 0, "+", color="black", markersize=7, zorder=5)
    ax.axhline(0, color="#dddddd", linewidth=0.6)
    ax.axvline(0, color="#dddddd", linewidth=0.6)
    ax.set(xlabel="Cosine coefficient", ylabel="Sine coefficient")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("95% coefficient regions", fontsize=9, fontweight="normal", pad=9)
    ax = axes[1, 1]
    ax.fill_between(curves.phase_deg, curves.sampling_low, curves.sampling_high,
                    color=colors["B_sampling"], alpha=0.16)
    for scenario, name in SCENARIOS.items():
        for bound in ("low", "high"):
            ax.plot(curves.phase_deg, curves[f"{name}_{bound}"], color=colors[scenario],
                    linestyle=styles[scenario], linewidth=1.1)
    ax.plot(curves.phase_deg, curves.point_multiplier, color="black", linewidth=1.2, label="Point fit")
    ax.axhline(1, color="#999999", linewidth=0.6)
    format_phase_response_axis(ax)
    point_phase = summary.loc[summary.quantity.eq("preferred_phase_deg"), "point_estimate"].iloc[0]
    point_ratio = summary.loc[summary.quantity.eq("max_min_rate_ratio"), "point_estimate"].iloc[0]
    mark_preferred_phase(ax, point_phase, point_ratio)
    ax.set_title("Conditional phase effect", fontsize=9, fontweight="normal", pad=9)
    for letter, ax in zip("abcd", axes.flat):
        add_panel_label(ax, letter, x=-0.16, y=1.04)
        ax.grid(False)
        ax.spines[["top", "right"]].set_visible(False)
    figure_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(figure_dir / f"{run_name}.{suffix}", dpi=450)
    if export:
        copy_pdf_to_paper(figure_dir / f"{run_name}.pdf")
    plt.close(fig)


def save_provenance(output_dir, args, age_results, context, selected, elapsed):
    parameters = vars(args).copy()
    parameters.update(model_version=combined_likelihood.MODEL_VERSION,
        age_reference="BP1950", history_tau_ka=context.history_tau_ka,
        quadrature_order=context.quadrature_order, history_coefficient_domain="beta_H <= 0",
        response_exposure_kyr=context.response_exposure_kyr,
        C_response_exposure_min_kyr=selected.response_exposure_kyr.min(),
        C_response_exposure_max_kyr=selected.response_exposure_kyr.max(),
        age_rows=len(age_results), age_valid=int(age_results.fit_valid.sum()),
        elapsed_seconds=elapsed,
        generator="fitted continuous full model; event-driven exponential history",
        initialization="exact oldest event fixed; earlier history zero; independent segments",
        C_support="selected chronology determines its own anchor; younger endpoints remain fixed",
        B_interval="bootstrap-error-calibrated 2D ellipse; approximate joint confidence projection",
        A_interval="95% coefficient ellipse projection; chronology working region",
        C_interval="95% coefficient ellipse projection; combined working region, not calibrated confidence",
        C_outer="uniform valid chronologies without replacement; equal inner weights",
        failures="retain each original replicate; withhold intervals if failures remain",
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)
    pd.DataFrame([dict(parameter=k, value=v) for k,v in parameters.items()]).to_csv(
        output_dir / "parameters_and_provenance.csv", index=False)
    inputs = [Path(__file__), Path(effect.__file__), Path(combined_likelihood.__file__),
        PROJECT_ROOT / "toolbox/point_process.py", AGE_DRAWS, args.age_results,
        combined_likelihood.EVENT_CATALOGUE_CSV, CO2_XLSX, LR04_XLSX, PRE_TXT]
    pd.DataFrame([dict(path=str(p.relative_to(PROJECT_ROOT)) if p.is_absolute() else str(p),
        sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in inputs]).to_csv(
        output_dir / "input_code_sha256.csv", index=False)


def validate_saved_inputs(point_fit, output_dir, input_paths):
    """Prevent a style-only redraw from mixing old simulations with new inputs."""
    saved = pd.read_csv(output_dir / "point_generator.csv")[BETA_COLUMNS].iloc[0].to_numpy(float)
    # Allow optimizer roundoff, then use the stored generator for an exact redraw.
    if not np.allclose(saved, point_fit.full.beta, rtol=1e-7, atol=1e-8):
        raise ValueError("Saved effect generator differs from the current fit; rerun simulations")
    parameters = pd.read_csv(output_dir / "parameters_and_provenance.csv").set_index("parameter").value
    if (parameters.get("model_version") != combined_likelihood.MODEL_VERSION or
            float(parameters["history_tau_ka"]) != point_fit.context.history_tau_ka):
        raise ValueError("Saved effect model settings differ; rerun simulations")
    manifest = pd.read_csv(output_dir / "input_code_sha256.csv")
    hashes = {(PROJECT_ROOT / str(row.path)).resolve(): row.sha256 for row in manifest.itertuples()}
    for path in input_paths:
        path = Path(path).resolve()
        if hashes.get(path) != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError(f"Saved effect input changed: {path.name}; rerun simulations")
    return saved


def save_effect_outputs(summary, curves, regions, output_dir):
    """Update interval products while leaving simulated sequences and their provenance intact."""
    summary.to_csv(output_dir / "effect_summary.csv", index=False, float_format="%.12g")
    curves.to_csv(output_dir / "phase_response_bands.csv", index=False, float_format="%.12g")
    encoded = {}
    for scenario, region in regions.items():
        encoded[SCENARIOS[scenario]] = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in region.items() if key != "bootstrap_error_quadratic"}
    (output_dir / "coefficient_regions.json").write_text(json.dumps(encoded, indent=2) + "\n")
    # Retain the existing sampling-only audit product for earlier diagnostic consumers.
    (output_dir / "sampling_joint_region.json").write_text(json.dumps(encoded["sampling"], indent=2) + "\n")
    parameters_path = output_dir / "parameters_and_provenance.csv"
    parameters = pd.read_csv(parameters_path).set_index("parameter")
    for scenario, interval in INTERVAL_TYPES.items():
        parameters.loc[f"{scenario[0]}_interval", "value"] = interval
    parameters.to_csv(parameters_path)


def effect_caption(catalogue):
    return (
        f"Uncertainty in the {catalogue} phase effect. "
        "Chronology (gray dotted), sampling (blue solid), and sampling + chronology (orange dashed) "
        "use the same coefficient-ellipse construction. (a,b) Preferred-phase and maximum/minimum "
        "rate-ratio bounds projected from the 95% regions in (c); dots and dashed vertical lines "
        "mark nominal estimates, and phase angles continue across 360 degrees. (c) Coefficient "
        "regions, nominal estimate (black dot), and no phase effect (cross). (d) Curve envelopes "
        "from the complete regions and the nominal phase multiplier (black); blue shading marks "
        "the sampling envelope. Minimum/Maximum refer to the precession index; unity means no "
        "phase contribution. Sampling gives an approximate conditional confidence region; "
        "chronology and combined regions summarize sensitivity to the assumed age errors.\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-point", type=int, default=DEFAULT_POINT_DRAWS)
    parser.add_argument("--n-outer", type=int, default=DEFAULT_OUTER_DRAWS)
    parser.add_argument("--n-inner", type=int, default=DEFAULT_INNER_DRAWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--age-results", type=Path, default=AGE_RESULTS)
    parser.add_argument("--no-paper-export", action="store_true")
    parser.add_argument("--redraw", action="store_true")
    args = parser.parse_args()
    if args.n_point < 20 or min(args.n_outer, args.n_inner, args.workers) < 1 or args.seed < 0:
        parser.error("Require n-point >= 20, positive group/worker counts and nonnegative seed")
    output_dir = args.output_root / "data/processed" / RUN_NAME
    figure_dir = args.output_root / "figures" / RUN_NAME
    notes_dir = args.output_root / "experiment_note"
    for directory in (output_dir, figure_dir, notes_dir):
        directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    context = combined_likelihood.build_context()
    events = context.events
    point_fit = combined_likelihood.fit_catalogue(events, context)
    effect.validate_effect_fit(point_fit)
    draws, age_results = load_age_inputs(events, args.age_results)
    if args.redraw:
        saved_beta = validate_saved_inputs(point_fit, output_dir, [AGE_DRAWS, args.age_results,
            combined_likelihood.EVENT_CATALOGUE_CSV, CO2_XLSX, LR04_XLSX, PRE_TXT])
        point_fit = replace(point_fit, full=replace(point_fit.full, beta=saved_beta))
        replicates = pd.read_csv(output_dir / "effect_replicates.csv")
    else:
        generators, selected = build_generators(events, context, point_fit, draws, age_results,
                                               args.n_outer, args.seed)
        selected.to_csv(output_dir / "selected_age_generators.csv", index=False)
        pd.DataFrame([dict(zip(BETA_COLUMNS, point_fit.full.beta))]).to_csv(
            output_dir / "point_generator.csv", index=False)
        age_results[["realization_id", "fit_valid", "invalid_reason"]].to_csv(
            output_dir / "age_support_status.csv", index=False)
        replicates = run_simulations(context, generators, args.n_point, args.n_inner, args.seed, args.workers)
        replicates.to_csv(output_dir / "effect_replicates.csv", index=False, float_format="%.12g")
        # Reuse the nominal full-model refits for model diagnostics.
        gof_columns = ["replicate_id", "scenario", "fit_valid", "invalid_reason",
                       "n_response_events", "ks_uniform", "adjacent_dependence", "residual_status"]
        replicates.loc[replicates.scenario.eq("B_sampling"), gof_columns].to_csv(
            output_dir / "gof_replicates.csv", index=False, float_format="%.12g")
        save_provenance(output_dir, args, age_results, context, selected, time.perf_counter()-started)
    summary, regions = summarize_effects(point_fit, age_results, replicates)
    curves = build_curve_table(point_fit, regions)
    save_effect_outputs(summary, curves, regions, output_dir)
    calibration = replicates.loc[replicates.scenario.eq("B_sampling"), ["inner_id"]].copy()
    calibration["error_quadratic"] = regions["B_sampling"]["bootstrap_error_quadratic"]
    calibration.to_csv(output_dir / "sampling_region_calibration.csv", index=False)
    plot_results(summary, curves, regions, figure_dir, not args.no_paper_export)
    text = ("Effect uncertainty\n\n"
        "Chronology uses every valid fit in the unchanged age ensemble. Sampling simulates the fitted continuous full model "
        "at nominal ages, holding each oldest event and younger endpoint fixed. Each catalogue is "
        "refitted with the same exact-event likelihood and exponential history (tau=1.5 kyr; beta_H<=0). "
        "Sampling + chronology repeats full-model simulation under selected age realizations, with equal weight per chronology.\n\n"
        "All three sources use coefficient ellipses centered on the nominal fit. Each sample covariance "
        "sets the ellipse shape, and the 95th percentile of squared standardized distances from the "
        "nominal fit sets its size. Projecting each whole ellipse gives phase/rate-ratio bounds "
        "and a curve envelope. Sampling retains its approximate conditional confidence interpretation; "
        "chronology and combined regions are working uncertainty regions, not calibrated confidence intervals. "
        "If a region contains the origin, preferred phase is unidentified.\n\n" + summary.to_string(index=False) + "\n")
    (notes_dir / f"{RUN_NAME}_Methods_and_results.txt").write_text(text)
    (notes_dir / f"{RUN_NAME}_Caption.txt").write_text(effect_caption("NGRIP–MIS6"))
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
