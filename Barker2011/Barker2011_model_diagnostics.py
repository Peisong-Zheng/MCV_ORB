#!/usr/bin/env python3
"""History utility and full-model fit checks; reuse Barker S4 sampling refits."""

import os
from pathlib import Path
import sys

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from toolbox import point_process_diagnostics as diagnostics
from toolbox.project_config import PROJECT_ROOT, BARKER_EVENT_CSVS


import numpy as np
import pandas as pd
from toolbox import event_model, sampling
from toolbox.point_process import fit_point_process
from toolbox.project_config import (MODEL_VERSION,
    LR04_CSV, CO2_CSV, ORBITAL_CSV, PRECESSION_PHASE_CSV)
HISTORY_TERM = "same_type_exponential_history"

RUN_NAME = "Barker2011_model_diagnostics"

OUTPUT_ROOT = PROJECT_ROOT
# The saved sampling ensemble is independent of this run's output destination.
GOF_REPLICATES_CSV = PROJECT_ROOT / "Barker2011/data/processed/Barker2011_effect_uncertainty/effect_replicates.csv"
N_HISTORY = 4_999
RANDOM_SEED = 20260916
N_WORKERS = 3
QUADRATURE_ORDER = 4
RUN_HISTORY_TEST = True


def main():
    gof_source = GOF_REPLICATES_CSV
    quadrature_order = QUADRATURE_ORDER
    event_definition = "variable_threshold"
    events = pd.read_csv(BARKER_EVENT_CSVS[event_definition], float_precision="round_trip")
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
    full = fit_point_process(event_x[list(full_terms)], integral_x[list(full_terms)],
                            integral_x.weight, full_terms, nonpositive_terms=(HISTORY_TERM,))
    saved_draws = pd.read_csv(gof_source)
    saved_settings = pd.read_csv(gof_source.parent / "parameters_and_provenance.csv").set_index("parameter").value
    saved_generator = pd.read_csv(gof_source.parent / "point_generator.csv")
    exposure = float((windows.response_end_kyr_bp-windows.response_start_kyr_bp).sum())
    nominal_sampling = diagnostics.select_sampling_gof_replicates(
        saved_draws, saved_settings, saved_generator, full,
        response_exposure_kyr=exposure, quadrature_order=quadrature_order)
    gof_parameters = dict(gof_source=str(gof_source), gof_bootstrap_replicates=len(nominal_sampling),
        gof_seed=int(saved_settings["seed"]), gof_source_role="S4 nominal B_sampling full-model refits only")
    result = {}
    if RUN_HISTORY_TEST:
        result = sampling.history_bootstrap(
            events, windows, forcings, phase_anchors, scaling, full_terms=full_terms,
            catalogue_id=catalogue_id, n_bootstrap=N_HISTORY, seed=RANDOM_SEED,
            workers=N_WORKERS, quadrature_order=quadrature_order)
    result.update(diagnostics.gof_results(event_x, integral_x, windows, full, nominal_sampling,
                                         catalogue_id=catalogue_id))
    data_dir = OUTPUT_ROOT / "Barker2011/data/processed" / RUN_NAME
    data_dir.mkdir(parents=True, exist_ok=True)
    settings = dict(model_version=MODEL_VERSION, history_tau_kyr=1.5,
                    initial_history=0., quadrature_order=quadrature_order,
                    event_input_csv=str(BARKER_EVENT_CSVS["variable_threshold"].relative_to(PROJECT_ROOT)))
    if "history_test" in result:
        scales = {f"{forcing}_{field}": scaling.loc[forcing, field]
                  for forcing in ("lr04", "co2") for field in ("mean", "range")}
        result["history_test"] = result["history_test"].drop(
            columns=["null_model", "parameter_domain"]).assign(
                history_seed=RANDOM_SEED, **settings, **scales)
        for name in ("history_test", "history_replicates", "history_coefficients"):
            result[name].to_csv(data_dir / f"{name}.csv", index=False, float_format="%.17g")
        print(result["history_test"].to_string(index=False))
    result["gof_summary"] = result["gof_summary"].drop(columns="calibration").assign(
        **settings, gof_source=gof_parameters["gof_source"],
        gof_seed=gof_parameters["gof_seed"], gof_scenario="B_sampling")
    result["gof_summary"].to_csv(data_dir / "gof_summary.csv", index=False, float_format="%.17g")
    print(result["gof_summary"].to_string(index=False))
    print(f"Saved {data_dir}")
    if any(not frame.fit_valid.all() for name, frame in result.items() if name.endswith("replicates")):
        raise RuntimeError("Model checks have failed fits; see saved results")


if __name__ == "__main__":
    main()
