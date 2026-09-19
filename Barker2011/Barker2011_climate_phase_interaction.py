#!/usr/bin/env python3
"""Continuous LR04 modulation of precession in the primary Barker2011 catalogue."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from toolbox import combined_likelihood
from toolbox import climate_phase_interaction as interaction
from toolbox import orbital_driver_reporting as reporting
from toolbox.project_config import PROJECT_ROOT, LR04_XLSX, CO2_XLSX, PRE_TXT

ROOT = PROJECT_ROOT / "Barker2011"
RUN_NAME = "Barker2011_climate_phase_interaction"
CATALOGUE_LABEL = "Barker 2011 SpeleoAge warming events"
AGE_INPUT = ROOT / "data/processed/Barker2011_orbital_driver_sensitivity/selected_realizations.csv"


def run_analysis(n_realizations=None, show_progress=True, quadrature_order=4):
    context = combined_likelihood.build_barker_context(quadrature_order=quadrature_order)
    context = interaction.add_interactions(context)
    age_columns = [f"age_ka_bp__{event_id}" for event_id in context.events.event_id]
    selected = reporting.read_selected_realizations(AGE_INPUT, age_columns, n_realizations)
    result = interaction.analyze_realizations(context, context.full_terms, selected, age_columns,
                                             show_progress=show_progress)
    reference = combined_likelihood.fit_catalogue(context.events, context)
    full = result["point_models"].set_index("model_id").loc["full"]
    if not np.isclose(full.log_likelihood, reference.full.log_likelihood, atol=1e-6, rtol=0):
        raise RuntimeError("The continuous full fit differs from the current main model")
    result["reference_check"] = pd.DataFrame([dict(main_loglik_full=reference.full.log_likelihood,
        current_loglik_full=full.log_likelihood, absolute_tolerance=1e-6,
        n_events_match=full.n_events == reference.summary["n_response_events"],
        exposure_match=np.isclose(full.exposure_kyr, reference.summary["response_exposure_kyr"]))])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--n-realizations", type=int, default=500)
    parser.add_argument("--quadrature-order", type=int, default=4)
    parser.add_argument("--no-paper-export", action="store_true")
    args = parser.parse_args()
    result = run_analysis(args.n_realizations, quadrature_order=args.quadrature_order)
    parameters = dict(catalogue=CATALOGUE_LABEL, model_version=combined_likelihood.MODEL_VERSION,
        method="continuous conditional point process", history_tau_kyr=combined_likelihood.DEFAULT_HISTORY_TAU_KA,
        initial_history=0, history_coefficient_domain="beta_H <= 0", quadrature_order=args.quadrature_order,
        age_epoch="SpeleoAge is treated as kyr BP1950; its published reference year is unverified", source_realizations=str(AGE_INPUT.relative_to(PROJECT_ROOT)),
        realization_selection="saved orbital-experiment rows, original order and exact ages; no new draw",
        selection_seed=20260909, n_realizations=len(result["selected_realizations"]),
        interaction_terms=";".join(interaction.INTERACTION_TERMS), added_parameters=2,
        p_method="nominal chi-square, df=2; no interaction bootstrap",
        interval="2.5–97.5% chronology sensitivity; not full sampling confidence")
    inputs = [Path(__file__).resolve(), Path(interaction.__file__),
        PROJECT_ROOT / "toolbox/orbital_driver_sensitivity.py", Path(reporting.__file__),
        Path(combined_likelihood.__file__), PROJECT_ROOT / "toolbox/point_process.py",
        PROJECT_ROOT / "toolbox/project_config.py", PROJECT_ROOT / "toolbox/event_inputs.py",
        PROJECT_ROOT / "toolbox/event_process.py", PROJECT_ROOT / "toolbox/orbital_phase.py",
        ROOT / "data/raw/Barker et al-2011-SOM.xls",
        AGE_INPUT, LR04_XLSX, CO2_XLSX, PRE_TXT]
    data_dir = interaction.save_results(result, args.output_root, RUN_NAME, CATALOGUE_LABEL,
        parameters, inputs, paper_export=not args.no_paper_export)
    print(result["comparison_summary"].to_string(index=False))
    print(f"Saved {data_dir}")


if __name__ == "__main__":
    main()
