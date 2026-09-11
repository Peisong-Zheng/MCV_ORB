#!/usr/bin/env python3
"""Separate chronology sensitivity, sampling confidence and joint working ranges.

A reads the saved age ensemble. B simulates 5,000 full-model catalogues at
point ages. C uses 200 valid chronologies with 50 full-model refits each.
The existing reduced-model null simulator and its p value are unchanged.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
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
from toolbox.phase_response_plotting import format_phase_response_axis, mark_preferred_phase
import numpy as np
import pandas as pd
import scipy

from toolbox import combined_pi
from toolbox import effect_uncertainty as effect
from toolbox.project_config import PROJECT_ROOT, CO2_XLSX, LR04_XLSX, PRE_TXT


RUN_NAME = "NGRIP_MIS6_effect_uncertainty"
OUTPUT_DIR = PROJECT_ROOT / "data/processed" / RUN_NAME
FIGURE_DIR = PROJECT_ROOT / "figures" / RUN_NAME
AGE_DIR = PROJECT_ROOT / "data/processed/NGRIP_MIS6_event_uncertainty_sensitivity"
AGE_DRAWS = AGE_DIR / "combined_event_age_realizations.csv"
AGE_RESULTS = AGE_DIR / "pi_realizations.csv"
DEFAULT_SEED = 20260908
DEFAULT_POINT_DRAWS = 5_000
DEFAULT_OUTER_DRAWS = 200
DEFAULT_INNER_DRAWS = 50
BETA_COLUMNS = [f"beta__{term}" for term in ("intercept", *combined_pi.FULL_TERMS)]
PHASE_COLUMNS = [f"beta__{term}" for term in effect.PHASE_TERMS]

_CONTEXT = None
_GENERATORS = None
_PREPARED = None


def load_age_inputs(events):
    """Read the paired ensemble, retaining the original support-failure audit."""
    draws = pd.read_csv(AGE_DRAWS)
    results = pd.read_csv(AGE_RESULTS)
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


def build_generators(events, context, point_fit, draws, results, n_outer, seed):
    """Uniformly sample distinct valid chronologies; fit each full model once."""
    valid_indices = np.flatnonzero(results.fit_valid.to_numpy(bool))
    if not 1 <= n_outer <= len(valid_indices):
        raise ValueError("Requested outer count exceeds valid age realizations")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 100]))
    selected = rng.choice(valid_indices, size=n_outer, replace=False)
    columns = [f"age_kyr_bp__{event_id}" for event_id in events.event_id]
    rows = []
    generators = {0: point_fit.full.beta.copy()}
    for outer_id, index in enumerate(selected, 1):
        source = draws.iloc[index]
        local_events = events.copy()
        local_events[combined_pi.EVENT_AGE_COLUMN] = source[columns].to_numpy(float)
        fit = combined_pi.fit_catalogue(local_events, context)
        effect.validate_effect_fit(fit)
        # Detect stale or mismatched PI and age files before simulation.
        np.testing.assert_allclose(
            [fit.summary[k] for k in ("LR_statistic", "info_bits_per_event")],
            results.loc[index, ["LR_statistic", "info_bits_per_event"]].to_numpy(float),
            rtol=2e-6, atol=2e-6,
        )
        generators[outer_id] = fit.full.beta.copy()
        row = {"outer_id": outer_id, "source_row_index": int(index),
               "age_realization_id": source.realization_id,
               "ngrip_realization_id": source.ngrip_realization_id,
               "mis6_realization_id": source.mis6_realization_id,
               "n_response_events": fit.summary["n_predictive_events"]}
        row.update(zip(BETA_COLUMNS, fit.full.beta))
        rows.append(row)
    return generators, pd.DataFrame(rows)


def _initialize_worker(context, generators):
    global _CONTEXT, _GENERATORS, _PREPARED
    _CONTEXT, _GENERATORS, _PREPARED = context, generators, {}


def _simulate_one(task):
    scenario, outer_id, inner_id, seed = task
    if outer_id not in _PREPARED:
        _PREPARED[outer_id] = effect.prepare_full_simulation(_CONTEXT, _GENERATORS[outer_id])
    # Private streams depend on scientific replicate identity, not worker order.
    rng = np.random.default_rng(np.random.SeedSequence([seed, scenario, outer_id, inner_id]))
    row = {"scenario": "B_sampling" if scenario == 1 else "C_joint",
           "outer_id": outer_id, "inner_id": inner_id, "seed": seed,
           "fit_valid": True, "invalid_reason": ""}
    try:
        counts = effect.simulate_prepared_full_counts(_PREPARED[outer_id], rng)
        row["n_observation_events"] = int(sum(values.sum() for values in counts.values()))
        fit = combined_pi.fit_event_counts(counts, _CONTEXT)
        effect.validate_effect_fit(fit)
        row["n_response_events"] = fit.summary["n_predictive_events"]
        row.update(zip(BETA_COLUMNS, fit.full.beta))
        phase, ratio = effect.phase_and_ratio(fit.full.beta[effect.PHASE_INDICES])
        row.update(phase_deg=float(phase), rate_ratio=float(ratio),
                   full_log_likelihood=fit.full.log_likelihood,
                   reduced_log_likelihood=fit.reduced.log_likelihood)
    except effect.InvalidEffectSimulation as error:
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
    """Keep conditional confidence projections distinct from ensemble quantiles."""
    if not replicates.fit_valid.all():
        raise RuntimeError("Simulation failures saved; review them before publishing intervals")
    point = point_fit.full.beta[effect.PHASE_INDICES]
    point_phase, point_ratio = effect.phase_and_ratio(point)
    b = replicates.loc[replicates.scenario.eq("B_sampling")]
    c = replicates.loc[replicates.scenario.eq("C_joint")]
    if c.groupby("outer_id").size().nunique() != 1:
        raise ValueError("Joint groups must have equal weight and complete inner samples")
    region = effect.bootstrap_joint_region(point, b[PHASE_COLUMNS].to_numpy(float))
    bounds = effect.project_joint_region(region)
    rows = []
    for scenario, table, phase_column, ratio_column in (
        ("A_chronology", age_results.loc[age_results.fit_valid],
         "pre_phase_preferred_deg", "pre_phase_rate_ratio_max_vs_min"),
        ("C_joint", c, "phase_deg", "rate_ratio"),
    ):
        for quantity, values, estimate in (
            ("preferred_phase_deg", effect.unwrap_phase(table[phase_column], point_phase), point_phase),
            ("max_min_rate_ratio", table[ratio_column].to_numpy(float), point_ratio),
        ):
            low, median, high = np.quantile(values, [0.025, 0.5, 0.975])
            rows.append(dict(scenario=scenario, quantity=quantity, point_estimate=float(estimate),
                             center=float(median), low=low, high=high, n=len(table),
                             interval_type="central_95_working_range", phase_identified=np.nan))
    for quantity, estimate, low, high in (
        ("preferred_phase_deg", point_phase, bounds["phase_low_unwrapped_deg"], bounds["phase_high_unwrapped_deg"]),
        ("max_min_rate_ratio", point_ratio, bounds["ratio_low"], bounds["ratio_high"]),
    ):
        rows.append(dict(scenario="B_sampling", quantity=quantity, point_estimate=float(estimate),
                         center=float(estimate), low=low, high=high, n=len(b),
                         interval_type="approximate_95_joint_confidence_region_projection",
                         phase_identified=bounds["phase_identified"]))
    summary = pd.DataFrame(rows).sort_values(["quantity", "scenario"]).reset_index(drop=True)
    return summary, region


def build_curve_table(point_fit, replicates, region):
    """B simultaneous confidence envelope and C pointwise raw-refit quantiles."""
    phases = np.arange(0.0, 361.0, 1.0)
    radians = np.deg2rad(phases)
    direction = np.column_stack((np.sin(radians), np.cos(radians)))
    point = point_fit.full.beta[effect.PHASE_INDICES]
    b_low, b_high = effect.joint_region_curve_band(region, phases)
    coefficients = replicates.loc[replicates.scenario.eq("C_joint"), PHASE_COLUMNS].to_numpy(float)
    c_curves = np.exp(coefficients @ direction.T)
    c_low, c_median, c_high = np.quantile(c_curves, [0.025, 0.5, 0.975], axis=0)
    return pd.DataFrame({"phase_deg": phases, "point_multiplier": np.exp(direction @ point),
                         "B_simultaneous_low": b_low, "B_simultaneous_high": b_high,
                         "C_pointwise_q025": c_low, "C_pointwise_median": c_median,
                         "C_pointwise_q975": c_high})


def plot_results(summary, curves, region, replicates, figure_dir=FIGURE_DIR):
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(180 / 25.4, 145 / 25.4))
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.18, top=0.94, hspace=0.65, wspace=0.55)
    colors = {"A_chronology": "#999999", "B_sampling": "#0072B2", "C_joint": "#D55E00"}
    labels = {"A_chronology": "A  Age", "B_sampling": "B  Sampling", "C_joint": "C  Joint"}
    for ax, quantity, xlabel, title in (
        (axes[0, 0], "preferred_phase_deg", "Preferred phase (°; unwrapped)", "a  Phase uncertainty"),
        (axes[0, 1], "max_min_rate_ratio", "Maximum/minimum rate ratio", "b  Effect strength"),
    ):
        for i, scenario in enumerate(colors):
            row = summary.loc[summary.scenario.eq(scenario) & summary.quantity.eq(quantity)].iloc[0]
            ax.hlines(i, row.low, row.high, color=colors[scenario], linewidth=2)
            ax.plot(row.center, i, "o", color=colors[scenario], markersize=4)
        ax.axvline(row.point_estimate, color="black", linestyle="--", linewidth=0.9)
        ax.set(yticks=range(3), yticklabels=list(labels.values()), xlabel=xlabel, ylim=(2.5, -0.5))
        ax.set_title(title, loc="left")
        if quantity == "max_min_rate_ratio":
            ax.set_xscale("log")
            ax.axvline(1, color="#999999", linewidth=0.7)
    ax = axes[1, 0]
    c = replicates.loc[replicates.scenario.eq("C_joint"), PHASE_COLUMNS].to_numpy(float)
    ax.scatter(c[::5, 1], c[::5, 0], s=2, color=colors["C_joint"], alpha=0.15, rasterized=True)
    boundary = effect.ellipse_boundary(region, np.linspace(0, 2 * np.pi, 721))
    ax.plot(boundary[:, 1], boundary[:, 0], color=colors["B_sampling"], label="B  Joint 95% region")
    ax.plot(region["center"][1], region["center"][0], "ko", markersize=4, label="Point estimate")
    ax.plot(0, 0, "+", color="black", markersize=7)
    ax.axhline(0, color="#dddddd", linewidth=0.6)
    ax.axvline(0, color="#dddddd", linewidth=0.6)
    ax.set(xlabel="Cosine coefficient", ylabel="Sine coefficient")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("c  Joint coefficient structure", loc="left")
    ax.legend(fontsize=6.5, frameon=False)
    ax = axes[1, 1]
    ax.fill_between(curves.phase_deg, curves.B_simultaneous_low, curves.B_simultaneous_high,
                    color=colors["B_sampling"], alpha=0.2, label="B  Simultaneous CI")
    ax.plot(curves.phase_deg, curves.C_pointwise_q025, color=colors["C_joint"], linestyle=":", label="C  Pointwise range")
    ax.plot(curves.phase_deg, curves.C_pointwise_q975, color=colors["C_joint"], linestyle=":")
    ax.plot(curves.phase_deg, curves.point_multiplier, color="black", linewidth=1.2, label="Point fit")
    ax.axhline(1, color="#999999", linewidth=0.6)
    format_phase_response_axis(ax)
    point_phase = summary.loc[summary.quantity.eq("preferred_phase_deg"), "point_estimate"].iloc[0]
    point_ratio = summary.loc[summary.quantity.eq("max_min_rate_ratio"), "point_estimate"].iloc[0]
    mark_preferred_phase(ax, point_phase, point_ratio)
    ax.set_title("d  Fitted warming-event rate", loc="left", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    ax.text(0.25, 0.63, f"Preferred phase: {point_phase:.1f}°\n"
            f"Max/min rate ratio: {point_ratio:.2f}", transform=ax.transAxes,
            va="top", fontsize=6.5)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.035, "A, C: central working ranges. B: approximate joint-region projections.\n"
             "Chronology and event-process assumptions remain conditional; these bands have different meanings.",
             ha="center", fontsize=7)
    figure_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(figure_dir / f"{RUN_NAME}.{suffix}", dpi=450)
    copy_pdf_to_paper(figure_dir / f"{RUN_NAME}.pdf")
    plt.close(fig)


def save_provenance(output_dir, args, age_results, elapsed):
    parameters = vars(args).copy()
    parameters.update(age_reference="BP1950", history_window_ka=1.5, bin_width_ka=0.2,
                      response_exposure_kyr=180.0, age_rows=len(age_results),
                      age_valid=int(age_results.fit_valid.sum()),
                      age_outside_support=int((~age_results.fit_valid).sum()), elapsed_seconds=elapsed,
                      generator="fitted full model; dynamic simulated history; zero initial history per segment",
                      B_interval="bootstrap-error-calibrated 2D ellipse; approximate conditional joint confidence",
                      C_interval="raw refit central quantiles; includes finite-sample bias; not a posterior or calibrated CI",
                      C_outer="uniform without replacement among valid age rows; equal inner weights",
                      failures="record without retries; stop interval publication if any occur",
                      phase_convention="degrees unwrapped around point estimate for ranges; circular modulo 360",
                      python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__)
    pd.DataFrame([{"parameter": key, "value": value} for key, value in parameters.items()]).to_csv(
        output_dir / "parameters_and_provenance.csv", index=False)
    paths = [Path(__file__), Path(effect.__file__), Path(combined_pi.__file__),
             PROJECT_ROOT / "toolbox/poisson.py", PROJECT_ROOT / "toolbox/event_inputs.py",
             PROJECT_ROOT / "toolbox/orbital_phase.py", PROJECT_ROOT / "toolbox/project_config.py",
             PROJECT_ROOT / "NGRIP_MIS6_PI_bootstrap.py", AGE_DRAWS, AGE_RESULTS,
             AGE_DIR / "parameters_and_provenance.csv", combined_pi.EVENT_CATALOGUE_CSV,
             combined_pi.OBSERVATION_SEGMENTS_CSV, CO2_XLSX, LR04_XLSX, PRE_TXT]
    pd.DataFrame([{"path": str(path.relative_to(PROJECT_ROOT)),
                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths]).to_csv(
                       output_dir / "input_code_sha256.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-point", type=int, default=DEFAULT_POINT_DRAWS)
    parser.add_argument("--n-outer", type=int, default=DEFAULT_OUTER_DRAWS)
    parser.add_argument("--n-inner", type=int, default=DEFAULT_INNER_DRAWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    args = parser.parse_args()
    if args.n_point < 20 or min(args.n_outer, args.n_inner, args.workers) < 1 or args.seed < 0:
        parser.error("Require n-point >= 20, positive group/worker counts and nonnegative seed")
    started = time.perf_counter()
    events = combined_pi.load_event_catalogue()
    context = combined_pi.build_context()
    point_fit = combined_pi.fit_catalogue(events, context)
    effect.validate_effect_fit(point_fit)
    draws, age_results = load_age_inputs(events)
    generators, selected = build_generators(events, context, point_fit, draws, age_results,
                                           args.n_outer, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output_dir / "selected_age_generators.csv", index=False, float_format="%.12g")
    pd.DataFrame([dict(zip(BETA_COLUMNS, point_fit.full.beta))]).to_csv(
        args.output_dir / "point_generator.csv", index=False, float_format="%.12g")
    age_results[["realization_id", "fit_valid", "invalid_reason"]].to_csv(
        args.output_dir / "age_support_status.csv", index=False)
    replicates = run_simulations(context, generators, args.n_point, args.n_inner, args.seed, args.workers)
    replicates.to_csv(args.output_dir / "effect_replicates.csv", index=False, float_format="%.12g")
    save_provenance(args.output_dir, args, age_results, time.perf_counter() - started)
    summary, region = summarize_effects(point_fit, age_results, replicates)
    curves = build_curve_table(point_fit, replicates, region)
    summary.to_csv(args.output_dir / "effect_summary.csv", index=False, float_format="%.12g")
    curves.to_csv(args.output_dir / "phase_response_bands.csv", index=False, float_format="%.12g")
    encoded = {key: value.tolist() if isinstance(value, np.ndarray) else value
               for key, value in region.items() if key != "bootstrap_error_quadratic"}
    (args.output_dir / "sampling_joint_region.json").write_text(json.dumps(encoded, indent=2) + "\n")
    b = replicates.loc[replicates.scenario.eq("B_sampling"), ["inner_id"]].copy()
    b["error_quadratic"] = region["bootstrap_error_quadratic"]
    b.to_csv(args.output_dir / "sampling_region_calibration.csv", index=False)
    plot_results(summary, curves, region, replicates, args.figure_dir)
    print(summary.to_string(index=False), flush=True)
    print(f"Wrote {args.output_dir}; no null-bootstrap result was changed.", flush=True)


if __name__ == "__main__":
    main()
