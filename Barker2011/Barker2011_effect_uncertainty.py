#!/usr/bin/env python3
"""Effect precision for Barker's varying-threshold SpeleoAge events.

Chronology reuses all saved age fits. Sampling simulates 5,000 catalogues at nominal ages.
Combined selects 200 age realizations and simulates 50 catalogues per fitted model.
The simulation, interval construction and figure template match NGRIP–MIS6.
"""

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

from toolbox.project_config import (PROJECT_ROOT, BARKER_EVENT_CSVS, LR04_CSV, CO2_CSV,
                                    ORBITAL_CSV, PRECESSION_PHASE_CSV)


from toolbox import event_model, sampling
from toolbox.point_process import fit_point_process
from toolbox.plotting import plot_effect_uncertainty
from toolbox.model_stats import SCENARIOS, INTERVAL_TYPES, summarize_effects, build_curve_table
from toolbox.project_config import MODEL_VERSION
HISTORY_TERM = "same_type_exponential_history"

RUN_NAME = "Barker2011_effect_uncertainty"
BARKER_ROOT = PROJECT_ROOT / "Barker2011"
AGE_DRAWS = BARKER_ROOT / "data/processed/Barker2011_event_age_uncertainty/event_age_realizations.csv"
AGE_RESULTS = BARKER_ROOT / "data/processed/Barker2011_event_uncertainty_sensitivity/gain_realizations.csv"
RANDOM_SEED = 20260925

OUTPUT_ROOT = PROJECT_ROOT
N_POINT_DRAWS = 5_000
N_OUTER_DRAWS = 200
N_INNER_DRAWS = 50
N_WORKERS = 3
REDRAW = False
EXPORT_PAPER = True


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


def save_figure(summary, curves, regions, figure_dir, export=False, *, compact_ratio_ticks=False):
    """Write the common effect figure in this study's output folder."""
    import matplotlib.pyplot as plt
    from paper_figure_export import copy_pdf_to_paper
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig = plot_effect_uncertainty(summary, curves, regions, compact_ratio_ticks=compact_ratio_ticks)
    for suffix in ("png", "pdf"):
        fig.savefig(figure_dir / f"{RUN_NAME}.{suffix}", dpi=450)
    plt.close(fig)
    if export:
        copy_pdf_to_paper(figure_dir / f"{RUN_NAME}.pdf")


def main():
    if N_POINT_DRAWS < 20 or min(N_OUTER_DRAWS, N_INNER_DRAWS, N_WORKERS) < 1 or RANDOM_SEED < 0:
        raise ValueError("Require at least 20 nominal draws, positive counts/workers and a nonnegative seed")
    root = OUTPUT_ROOT / "Barker2011"
    data_dir = root / "data/processed" / RUN_NAME
    figure_dir = root / "figures" / RUN_NAME
    for directory in (data_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)
    export = EXPORT_PAPER and OUTPUT_ROOT.resolve() == PROJECT_ROOT.resolve()
    if REDRAW:
        # Saved regions and curves are the result; plotting needs no new model fit.
        summary = pd.read_csv(data_dir / "effect_summary.csv", float_precision="round_trip")
        curves = pd.read_csv(data_dir / "phase_response_bands.csv", float_precision="round_trip")
        encoded = json.loads((data_dir / "coefficient_regions.json").read_text())
        regions = {scenario: encoded[name] for scenario, name in SCENARIOS.items()}
        expected = {(scenario, quantity) for scenario in SCENARIOS
                    for quantity in ("preferred_phase_deg", "max_min_rate_ratio")}
        if len(summary) != 6 or set(zip(summary.scenario, summary.quantity)) != expected:
            raise ValueError("Effect summary must contain both quantities for all three scenarios")
        if (not np.isfinite(summary[["point_estimate", "center", "low", "high"]]).all().all()
                or not np.isfinite(curves.to_numpy(float)).all()
                or not summary.low.le(summary.high).all()):
            raise ValueError("Saved effect bounds must be finite and ordered")
        save_figure(summary, curves, regions, figure_dir, export,
                            compact_ratio_ticks=True)
        print(summary.to_string(index=False), flush=True)
        return

    started = time.perf_counter()
    event_definition = "variable_threshold"
    quadrature_order = 4
    events = pd.read_csv(BARKER_EVENT_CSVS[event_definition], float_precision="round_trip")
    if (len(events) != {"variable_threshold": 70, "fixed_threshold": 59}[event_definition]
            or events.event_id.isna().any() or not events.event_id.is_unique
            or not np.all(np.diff(events.event_age_kyr_bp) > 0)):
        raise ValueError("Check prepared Barker event count, identities and increasing ages")
    events["segment_id"] = "Barker2011"
    observations = pd.DataFrame([dict(segment_id="Barker2011", observation_start_kyr_bp=0.,
                                      observation_end_kyr_bp=400.)])
    catalogue_id = f"barker_{event_definition}_speleo_0_400"
    lr04 = pd.read_csv(LR04_CSV, float_precision="round_trip")
    co2 = pd.read_csv(CO2_CSV, float_precision="round_trip")
    orbital = pd.read_csv(ORBITAL_CSV, float_precision="round_trip")
    anchors = pd.read_csv(PRECESSION_PHASE_CSV, float_precision="round_trip")
    forcings = {"lr04": (lr04.age_kyr_bp.to_numpy(), lr04.lr04.to_numpy()),
                "co2": (co2.age_kyr_bp.to_numpy(), co2.co2_ppm.to_numpy()),
                "precession_index": (orbital.age_kyr_bp.to_numpy(), orbital.precession_index.to_numpy())}
    phase_anchors = (anchors.age_kyr_bp.to_numpy(), anchors.phase_unwrapped_rad.to_numpy())
    windows = event_model.response_windows(events, observations)
    scaling = event_model.nominal_scaling({name: forcings[name] for name in ("lr04", "co2")}, windows)
    event_x, integral_x = event_model.build_design(events, windows, forcings, phase_anchors, scaling,
                                                  quadrature_order=quadrature_order)
    reduced_terms = ("intercept", HISTORY_TERM, "lr04_scaled", "co2_scaled")
    full_terms = reduced_terms + ("pre_phase_sin", "pre_phase_cos")
    reduced = fit_point_process(event_x[list(reduced_terms)], integral_x[list(reduced_terms)],
        integral_x.weight, reduced_terms, nonpositive_terms=(HISTORY_TERM,))
    full = fit_point_process(event_x[list(full_terms)], integral_x[list(full_terms)],
        integral_x.weight, full_terms, nonpositive_terms=(HISTORY_TERM,), start_beta=np.r_[reduced.beta, 0., 0.])
    sampling.validate_effect_fit(reduced, full)
    draws, ages, age_columns = load_age_inputs(events)
    generators, selected = sampling.effect_generators(events, observations, windows, forcings,
        phase_anchors, scaling, full, draws, ages, age_columns,
        reduced_terms=reduced_terms, full_terms=full_terms, n_outer=N_OUTER_DRAWS, seed=RANDOM_SEED)
    selected.to_csv(data_dir / "selected_age_generators.csv", index=False)
    pd.DataFrame([dict(zip(full.terms, full.beta))]).to_csv(
        data_dir / "point_generator.csv", index=False)
    scaling.reset_index()[["forcing_id", "mean", "range"]].to_csv(
        data_dir / "predictor_scaling.csv", index=False, float_format="%.17g")
    replicates = sampling.sample_effect(generators, forcings, phase_anchors, scaling,
        reduced_terms=reduced_terms, full_terms=full_terms, n_point=N_POINT_DRAWS,
        n_inner=N_INNER_DRAWS, seed=RANDOM_SEED, workers=N_WORKERS)
    # Retain failures and all fitted coefficients; effect summaries reject invalid fits.
    replicates.drop(columns=["inner_id", "n_observation_events", "phase_deg", "rate_ratio"],
                    errors="ignore").to_csv(
        data_dir / "effect_replicates.csv", index=False, float_format="%.12g")
    settings = dict(model_version=MODEL_VERSION, event_definition="variable_threshold",
                    event_input_csv=str(BARKER_EVENT_CSVS["variable_threshold"].relative_to(PROJECT_ROOT)),
                    age_draws_csv=str(AGE_DRAWS.relative_to(PROJECT_ROOT)),
                    age_results_csv=str(AGE_RESULTS.relative_to(PROJECT_ROOT)),
                    lr04_csv=str(LR04_CSV.relative_to(PROJECT_ROOT)),
                    co2_csv=str(CO2_CSV.relative_to(PROJECT_ROOT)),
                    orbital_csv=str(ORBITAL_CSV.relative_to(PROJECT_ROOT)),
                    phase_anchors_csv=str(PRECESSION_PHASE_CSV.relative_to(PROJECT_ROOT)),
                    n_point=N_POINT_DRAWS, n_outer=N_OUTER_DRAWS, n_inner=N_INNER_DRAWS,
                    seed=RANDOM_SEED, n_age_total=len(ages), n_age_valid=int(ages.fit_valid.sum()),
                    history_tau_kyr=1.5, initial_history=0.,
                    quadrature_order=quadrature_order,
                    response_exposure_kyr=float((windows.response_end_kyr_bp-windows.response_start_kyr_bp).sum()),
                    response_start_kyr_bp=windows.response_start_kyr_bp.item(),
                    n_failed=int((~replicates.fit_valid).sum()))
    pd.DataFrame(settings.items(), columns=["parameter", "value"]).to_csv(
        data_dir / "parameters_and_provenance.csv", index=False)
    validate_simulation_ids(replicates, selected, N_POINT_DRAWS, N_OUTER_DRAWS, N_INNER_DRAWS, RANDOM_SEED)
    summary, regions = summarize_effects(full, ages, replicates)
    curves = build_curve_table(full, regions)
    summary.to_csv(data_dir / "effect_summary.csv", index=False, float_format="%.12g")
    curves.to_csv(data_dir / "phase_response_bands.csv", index=False, float_format="%.12g")
    encoded = {SCENARIOS[scenario]: {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in region.items() if key != "bootstrap_error_quadratic"}
        for scenario, region in regions.items()}
    (data_dir / "coefficient_regions.json").write_text(json.dumps(encoded, indent=2) + "\n")
    save_figure(summary, curves, regions, figure_dir, export,
                        compact_ratio_ticks=True)
    print(summary.to_string(index=False), flush=True)
    print(f"Elapsed: {time.perf_counter() - started:.1f} s; workers: {N_WORKERS}", flush=True)


if __name__ == "__main__":
    main()
