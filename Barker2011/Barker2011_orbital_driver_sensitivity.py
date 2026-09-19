#!/usr/bin/env python3
"""Continuous-time orbital-driver sensitivity on the primary Barker2011 catalogue.

Reuse the frozen 500 chronology IDs and exact ages. Each chronology supplies
its own conditioning anchor; all candidate models share that exact support.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from toolbox import combined_likelihood, orbital_driver_sensitivity as orbital
from toolbox import orbital_driver_reporting as reporting
from toolbox.project_config import PROJECT_ROOT, LR04_XLSX, CO2_XLSX, PRE_TXT, OBL_TXT

ROOT = PROJECT_ROOT / "Barker2011"
RUN_NAME = "Barker2011_orbital_driver_sensitivity"
CATALOGUE_LABEL = "Barker 2011 SpeleoAge warming events"
N_REALIZATIONS = 500
AGE_INPUT = ROOT / "data/processed" / RUN_NAME / "selected_realizations.csv"


def run_analysis(n_realizations=N_REALIZATIONS, show_progress=True, quadrature_order=4):
    context = combined_likelihood.build_barker_context(quadrature_order=quadrature_order)
    context, scaling, provenance = orbital.prepare_drivers(context)
    age_columns = [f"age_ka_bp__{event_id}" for event_id in context.events.event_id]
    selected = reporting.read_selected_realizations(AGE_INPUT, age_columns, n_realizations)
    result = orbital.analyze_chronologies(context, selected, age_columns, show_progress=show_progress)
    result.update(scaling=scaling, provenance=provenance, parameters=dict(
        catalogue=CATALOGUE_LABEL, method="continuous conditional point process",
        n_realizations=len(selected), selection_seed=20260909,
        history_tau_kyr=context.history_tau_ka, initial_history=context.initial_history,
        history_coefficient_domain="beta_H <= 0", quadrature_order=context.quadrature_order,
        response_exposure_kyr=result["point"]["models"].iloc[0].exposure_kyr,
        full_terms=";".join(context.full_terms), age_epoch="SpeleoAge treated as BP1950; reference year unverified",
        source_realizations=str(AGE_INPUT.relative_to(PROJECT_ROOT)),
        selection="saved orbital-experiment rows, original order and exact ages; no new draw",
        n_new_comparisons=9, p_method="nominal chi-square; within-catalogue Holm family of 9",
        interval="2.5–97.5% chronology sensitivity; not full sampling confidence"))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--n-realizations", type=int, default=N_REALIZATIONS)
    parser.add_argument("--quadrature-order", type=int, default=4)
    parser.add_argument("--no-paper-export", action="store_true",
                        help="This driver script only writes its own research artifacts.")
    args = parser.parse_args()
    result = run_analysis(args.n_realizations, quadrature_order=args.quadrature_order)
    inputs = [Path(__file__).resolve(), Path(orbital.__file__), Path(reporting.__file__),
        Path(combined_likelihood.__file__), PROJECT_ROOT / "toolbox/point_process.py",
        PROJECT_ROOT / "toolbox/project_config.py", PROJECT_ROOT / "toolbox/event_inputs.py",
        PROJECT_ROOT / "toolbox/orbital_phase.py", PROJECT_ROOT / "toolbox/event_process.py",
        PROJECT_ROOT / "Barker2011/Barker2011_event_phase_analysis.py",
        ROOT / "data/raw/Barker et al-2011-SOM.xls",
        AGE_INPUT, LR04_XLSX, CO2_XLSX, PRE_TXT, OBL_TXT,
        PROJECT_ROOT / "data/raw/ecc_1000_60_inter100.txt",
        PROJECT_ROOT / "data/raw/solstice_insolation_NH.nc"]
    data_dir, _ = reporting.save_results(result, args.output_root, RUN_NAME, CATALOGUE_LABEL, inputs)
    print(result["comparison_summary"].to_string(index=False))
    print(f"Saved {data_dir}")


if __name__ == "__main__":
    main()
