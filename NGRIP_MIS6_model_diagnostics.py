#!/usr/bin/env python3
"""History utility and full-model fit checks; reuse primary B_sampling for GOF."""

import os
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

from toolbox import point_process_diagnostics as diagnostics
from toolbox.project_config import PROJECT_ROOT


import numpy as np
import pandas as pd
from toolbox import event_model, sampling
from toolbox.point_process import fit_point_process
from toolbox.project_config import (MODEL_VERSION, EVENT_CATALOGUE_CSV, OBSERVATION_SEGMENTS_CSV,
    LR04_CSV, CO2_CSV, ORBITAL_CSV, PRECESSION_PHASE_CSV, generated_notes_dir)
HISTORY_TERM = "same_type_exponential_history"

RUN_NAME = "NGRIP_MIS6_model_diagnostics"
OUTPUT_ROOT = PROJECT_ROOT
N_HISTORY = 4_999
RANDOM_SEED = 20260914
N_WORKERS = 3
QUADRATURE_ORDER = 4
RUN_HISTORY_TEST = True
# Input is independent of the folder where this script saves diagnostics.
GOF_REPLICATES_CSV = PROJECT_ROOT / "data/processed/NGRIP_MIS6_effect_uncertainty/effect_replicates.csv"


def save_results(result, windows, catalogue_id, quadrature_order, output_root, run_name, parameters):
    """Write model checks as CSV and prose without introducing default figures."""
    parameters = dict(parameters)
    output_root = Path(output_root)
    data_dir = output_root / "data/processed" / run_name
    notes_dir = generated_notes_dir(output_root)
    for directory in (data_dir, notes_dir):
        directory.mkdir(parents=True, exist_ok=True)
    for name, frame in result.items():
        frame.to_csv(data_dir / f"{name}.csv", index=False, float_format="%.12g")
    windows.to_csv(data_dir / "support.csv", index=False)
    metadata = dict(catalogue_id=catalogue_id, model_version=MODEL_VERSION,
                    history_tau_kyr=1.5, initial_history=0.,
                    history_coefficient_domain="beta_H <= 0", quadrature_order=quadrature_order,
                    conditioning="original oldest observed event per segment, excluded from response",
                    fixed_response_exposure_kyr=float((windows.response_end_kyr_bp-windows.response_start_kyr_bp).sum()), **parameters)
    pd.DataFrame([dict(parameter=key, value=value) for key, value in metadata.items()]).to_csv(
        data_dir / "parameters_and_provenance.csv", index=False)
    history_path, gof_path = data_dir / "history_test.csv", data_dir / "gof_summary.csv"
    sections = [f"Conditional model checks: {catalogue_id}",
        "\nThe history test compares background+phase with background+phase+history,",
        "fixing the exponential decay time at 1.5 kyr.",
        "The null has beta_H=0; the alternative constrains beta_H<=0. The likelihood-ratio",
        "statistic is calibrated by simulating the null and refitting both models.",
        "An ordinary chi-square_1 reference is inappropriate at this parameter boundary.",
        "Every simulation retains the original oldest event and response interval in each segment."]
    if history_path.exists():
        history = pd.read_csv(history_path)
        columns = ["beta_history", "history_rate_multiplier", "observed", "bootstrap_p",
                   "n_bootstrap", "n_invalid"]
        sections.extend(["\nHistory result (observed = likelihood-ratio statistic):",
                         history.loc[:, columns].to_string(index=False, float_format=lambda x: f"{x:.6g}")])
    sections.extend(["\nFull-model goodness of fit uses two prespecified discrepancies:",
        "the KS distance of completed rescaled intervals to Uniform(0,1), and the absolute",
        "sum of adjacent (U-0.5) products divided by sqrt(max(1, number of within-segment pairs)).",
        "No adjacent pairs cross record gaps. A zero-response simulation has both statistics zero",
        "and remains in the reference distribution. Terminal no-event intervals are censored:",
        "their intensity integrals enter endpoint residuals but not the completed-interval CDF.",
        "Calibration simulates the nominal full model, refits it, then recomputes each statistic.",
        "Holm adjustment covers these two diagnostics within the catalogue. This is approximate",
        "parametric bootstrap calibration, not proof that the entire event model is correct."])
    if "gof_source" in parameters:
        sections.extend([
            f"\nGOF reuses all {parameters['gof_bootstrap_replicates']:,} nominal-age sampling refits "
            f"from S4 (seed {parameters['gof_seed']}); combined chronology/sampling draws are excluded.",
            f"Source: {parameters['gof_source']}"])
    if gof_path.exists():
        gof = pd.read_csv(gof_path)
        columns = ["statistic", "observed", "bootstrap_p", "bootstrap_p_holm", "n_bootstrap", "n_invalid"]
        sections.extend(["\nFull-model diagnostic results:",
                         gof.loc[:, columns].to_string(index=False, float_format=lambda x: f"{x:.6g}")])
    else:
        sections.append("\nFull-model diagnostics are pending the nominal full-model refit ensemble.")
    sections.extend(["\nPlus-one p=(1+exceedances)/(B+1). Failed simulations/refits are not replaced",
        "or treated as zero discrepancies. If failures remain, p is unresolved and its bounds",
        "show the range of unknown exceedance contributions. Finite-bootstrap intervals quantify",
        "simulation precision of the exceedance probability, not uncertainty of effect estimates.",
        "No chronology Monte Carlo or mixed chronology generator is included in this calibration.",
        "Results are supplied as CSV and prose; no figure is generated by this workflow."])
    (notes_dir / f"{run_name}_Methods_and_results.txt").write_text("\n".join(sections) + "\n", encoding="utf-8")
    return data_dir


def main():
    gof_source = GOF_REPLICATES_CSV
    quadrature_order = QUADRATURE_ORDER
    events = pd.read_csv(EVENT_CATALOGUE_CSV)
    observations = pd.read_csv(OBSERVATION_SEGMENTS_CSV)
    catalogue_id = "ngrip_warming_plus_mis6"
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
    for frame in (event_x, integral_x):
        frame["mis6_segment"] = frame.segment_id.eq("MIS6").astype(float)
    reduced_terms = ("intercept", HISTORY_TERM, "lr04_scaled", "co2_scaled", "mis6_segment")
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
    data_dir = save_results(result, windows, catalogue_id, quadrature_order, OUTPUT_ROOT, RUN_NAME,
        dict(history_bootstrap_replicates=N_HISTORY, history_seed=RANDOM_SEED,
             **gof_parameters))
    print(f"Saved {data_dir}")
    if any(not frame.fit_valid.all() for name, frame in result.items() if name.endswith("replicates")):
        raise RuntimeError("Model checks have failed fits; see saved results")


if __name__ == "__main__":
    main()
