#!/usr/bin/env python3
"""Effect precision for Barker's varying-threshold SpeleoAge events.

Chronology reuses all saved age fits. Sampling simulates 5,000 catalogues at nominal ages.
Combined selects 200 age realizations and simulates 50 catalogues per fitted model.
The simulation, interval construction and figure template match NGRIP–MIS6.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import NGRIP_MIS6_effect_uncertainty as shared
from toolbox import combined_likelihood as likelihood
from toolbox import effect_uncertainty as effect
from toolbox.project_config import (PROJECT_ROOT, BARKER_EVENT_CSVS, LR04_CSV, CO2_CSV,
                                    ORBITAL_CSV, PRECESSION_PHASE_CSV)


RUN_NAME = "Barker2011_effect_uncertainty"
BARKER_ROOT = PROJECT_ROOT / "Barker2011"
AGE_DRAWS = BARKER_ROOT / "data/processed/Barker2011_event_age_uncertainty/event_age_realizations.csv"
AGE_RESULTS = BARKER_ROOT / "data/processed/Barker2011_event_uncertainty_sensitivity/gain_realizations.csv"
DEFAULT_SEED = 20260925


def load_age_inputs(events):
    """Match the unchanged age ensemble to its saved continuous-time fits."""
    draws = pd.read_csv(AGE_DRAWS, float_precision="round_trip")
    results = pd.read_csv(AGE_RESULTS)
    if draws.realization_id.duplicated().any() or results.realization_id.duplicated().any():
        raise ValueError("Age realization IDs must be unique")
    if set(draws.realization_id) != set(results.realization_id):
        raise ValueError("Age realizations and fitted results must contain the same IDs")
    results = results.set_index("realization_id").loc[draws.realization_id].reset_index()
    if not results.fit_valid.isin([True, False]).all():
        raise ValueError("Each age realization needs a fit status")
    columns = [f"age_ka_bp__{event_id}" for event_id in events.event_id]
    ages = draws[columns].to_numpy(float)
    if not np.isfinite(ages).all() or not np.all(np.diff(ages, axis=1) > 0):
        raise ValueError("Saved BP ages must be finite and ordered")
    values = results.loc[results.fit_valid, ["pre_phase_preferred_deg",
                                           "pre_phase_rate_ratio_max_vs_min"]]
    if not np.isfinite(values.to_numpy(float)).all():
        raise ValueError("Valid age fits need finite phase-effect estimates")
    return draws, results, columns


def validate_simulation_ids(replicates, selected, n_point, n_outer, n_inner, seed):
    """Require the requested nominal draws and every chronology's inner sample."""
    if set(replicates.scenario) != {"B_sampling", "C_joint"}:
        raise ValueError("Expected nominal sampling and joint scenarios")
    if not replicates.seed.eq(seed).all():
        raise ValueError("Simulation seeds differ from the requested experiment")
    if (not selected.age_realization_id.is_unique or
            not np.array_equal(np.sort(selected.outer_id), np.arange(1, n_outer + 1))):
        raise ValueError("Selected chronology generators are incomplete or duplicated")
    sampling = replicates.loc[replicates.scenario.eq("B_sampling")]
    if (not sampling.outer_id.eq(0).all() or
            not np.array_equal(np.sort(sampling.replicate_id), np.arange(1, n_point + 1))):
        raise ValueError("Nominal sampling ensemble is incomplete or duplicated")
    joint = replicates.loc[replicates.scenario.eq("C_joint")]
    if set(joint.outer_id) != set(selected.outer_id):
        raise ValueError("Joint ensemble does not match the selected chronologies")
    for outer_id, group in joint.groupby("outer_id"):
        if not np.array_equal(np.sort(group.replicate_id), np.arange(1, n_inner + 1)):
            raise ValueError(f"Chronology {outer_id} has incomplete or duplicate inner draws")
        exposure = selected.set_index("outer_id").loc[outer_id, "response_exposure_kyr"]
        np.testing.assert_allclose(group.response_exposure_kyr, exposure, rtol=1e-10, atol=1e-10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-point", type=int, default=5000)
    parser.add_argument("--n-outer", type=int, default=200)
    parser.add_argument("--n-inner", type=int, default=50)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--redraw", action="store_true")
    parser.add_argument("--no-paper-export", action="store_true")
    args = parser.parse_args()
    if args.n_point < 20 or min(args.n_outer, args.n_inner, args.workers) < 1 or args.seed < 0:
        parser.error("Require n-point >= 20, positive counts/workers and a nonnegative seed")
    root = args.output_root / "Barker2011"
    data_dir = root / "data/processed" / RUN_NAME
    figure_dir = root / "figures" / RUN_NAME
    for directory in (data_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)
    export = not args.no_paper_export and args.output_root.resolve() == PROJECT_ROOT.resolve()
    if args.redraw:
        # Saved regions and curves are the result; plotting needs no new model fit.
        summary = pd.read_csv(data_dir / "effect_summary.csv", float_precision="round_trip")
        curves = pd.read_csv(data_dir / "phase_response_bands.csv", float_precision="round_trip")
        encoded = json.loads((data_dir / "coefficient_regions.json").read_text())
        regions = {scenario: encoded[name] for scenario, name in shared.SCENARIOS.items()}
        expected = {(scenario, quantity) for scenario in shared.SCENARIOS
                    for quantity in ("preferred_phase_deg", "max_min_rate_ratio")}
        if len(summary) != 6 or set(zip(summary.scenario, summary.quantity)) != expected:
            raise ValueError("Effect summary must contain both quantities for all three scenarios")
        if (not np.isfinite(summary[["point_estimate", "center", "low", "high"]]).all().all()
                or not np.isfinite(curves.to_numpy(float)).all()
                or not summary.low.le(summary.high).all()):
            raise ValueError("Saved effect bounds must be finite and ordered")
        shared.plot_results(summary, curves, regions, figure_dir, export, run_name=RUN_NAME,
                            compact_ratio_ticks=True)
        print(summary.to_string(index=False), flush=True)
        return

    started = time.perf_counter()
    events = pd.read_csv(BARKER_EVENT_CSVS["variable_threshold"], float_precision="round_trip")
    context = likelihood.build_barker_context(events)
    point_fit = likelihood.fit_catalogue(context.events, context)
    effect.validate_effect_fit(point_fit)
    draws, ages, age_columns = load_age_inputs(context.events)
    generators, selected = shared.build_generators(
        context.events, context, point_fit, draws, ages, args.n_outer, args.seed,
        age_columns=age_columns, source_id_columns=())
    selected.to_csv(data_dir / "selected_age_generators.csv", index=False)
    pd.DataFrame([dict(zip(point_fit.full.terms, point_fit.full.beta))]).to_csv(
        data_dir / "point_generator.csv", index=False)
    likelihood.scaling_table(context)[["forcing_id", "mean", "range"]].to_csv(
        data_dir / "predictor_scaling.csv", index=False, float_format="%.17g")
    replicates = shared.run_simulations(context, generators, args.n_point, args.n_inner,
                                         args.seed, args.workers)
    # Retain failures and all fitted coefficients; effect summaries reject invalid fits.
    replicates.drop(columns=["inner_id", "n_observation_events", "phase_deg", "rate_ratio"],
                    errors="ignore").to_csv(
        data_dir / "effect_replicates.csv", index=False, float_format="%.12g")
    settings = dict(model_version=likelihood.MODEL_VERSION, event_definition="variable_threshold",
                    event_input_csv=str(BARKER_EVENT_CSVS["variable_threshold"].relative_to(PROJECT_ROOT)),
                    age_draws_csv=str(AGE_DRAWS.relative_to(PROJECT_ROOT)),
                    age_results_csv=str(AGE_RESULTS.relative_to(PROJECT_ROOT)),
                    lr04_csv=str(LR04_CSV.relative_to(PROJECT_ROOT)),
                    co2_csv=str(CO2_CSV.relative_to(PROJECT_ROOT)),
                    orbital_csv=str(ORBITAL_CSV.relative_to(PROJECT_ROOT)),
                    phase_anchors_csv=str(PRECESSION_PHASE_CSV.relative_to(PROJECT_ROOT)),
                    n_point=args.n_point, n_outer=args.n_outer, n_inner=args.n_inner,
                    seed=args.seed, n_age_total=len(ages), n_age_valid=int(ages.fit_valid.sum()),
                    history_tau_kyr=context.history_tau_ka, initial_history=context.initial_history,
                    quadrature_order=context.quadrature_order,
                    response_exposure_kyr=context.response_exposure_kyr,
                    response_start_kyr_bp=context.segments["Barker2011"].response_start_kyr_bp,
                    n_failed=int((~replicates.fit_valid).sum()))
    pd.DataFrame(settings.items(), columns=["parameter", "value"]).to_csv(
        data_dir / "parameters_and_provenance.csv", index=False)
    validate_simulation_ids(replicates, selected, args.n_point, args.n_outer, args.n_inner, args.seed)
    summary, regions = shared.summarize_effects(point_fit, ages, replicates)
    curves = shared.build_curve_table(point_fit, regions)
    summary.to_csv(data_dir / "effect_summary.csv", index=False, float_format="%.12g")
    curves.to_csv(data_dir / "phase_response_bands.csv", index=False, float_format="%.12g")
    encoded = {shared.SCENARIOS[scenario]: {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in region.items() if key != "bootstrap_error_quadratic"}
        for scenario, region in regions.items()}
    (data_dir / "coefficient_regions.json").write_text(json.dumps(encoded, indent=2) + "\n")
    shared.plot_results(summary, curves, regions, figure_dir, export, run_name=RUN_NAME,
                        compact_ratio_ticks=True)
    print(summary.to_string(index=False), flush=True)
    print(f"Elapsed: {time.perf_counter() - started:.1f} s; workers: {args.workers}", flush=True)


if __name__ == "__main__":
    main()
