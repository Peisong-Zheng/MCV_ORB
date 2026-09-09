#!/usr/bin/env python3
"""Test LR04 modulation of precession in primary Barker SpeleoAge events."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Barker2011 import Barker2011_event_phase_analysis as main_analysis
from Barker2011 import Barker2011_event_uncertainty_sensitivity as age_sensitivity
from toolbox import combined_pi
from toolbox import climate_phase_interaction as interaction
from toolbox.project_config import PROJECT_ROOT, LR04_XLSX, CO2_XLSX, PRE_TXT

ROOT = Path(__file__).resolve().parent
RUN_NAME = "Barker2011_climate_phase_interaction"
CATALOGUE_LABEL = "Barker 2011 SpeleoAge warming events"
AGE_INPUT = ROOT / "data/processed/Barker2011_orbital_driver_sensitivity/selected_realizations.csv"
MAIN_SUMMARY = ROOT / "data/processed/Barker2011_event_phase_analysis/model_summary.csv"


def run_analysis(n_realizations=None, show_progress=True):
    # Keep the primary variable-threshold catalogue used by chronology sensitivity.
    events = main_analysis.load_barker_source()
    events.insert(0, "event_id", [f"Barker_S3_{row:03d}" for row in events.source_excel_row])
    context = age_sensitivity.prepare_context(events)
    response = context["frame"].reset_index(drop=True)
    selected = pd.read_csv(AGE_INPUT, float_precision="round_trip")
    if n_realizations is not None:
        selected = selected.iloc[:n_realizations].copy()
    age_columns = [f"age_ka_bp__{event_id}" for event_id in events.event_id]
    result = interaction.analyze_realizations(response, main_analysis.FULL_TERMS, selected,
        age_columns, lambda ages: age_sensitivity.frame_for_ages(ages, context), show_progress=show_progress)
    reference = pd.read_csv(MAIN_SUMMARY).set_index("model_id").loc["full"]
    full = result["point_models"].set_index("model_id").loc["full"]
    if not np.isclose(full.log_likelihood, reference.log_likelihood, atol=5.1e-7, rtol=0):
        raise RuntimeError("The current full fit differs from the saved main analysis")
    result["reference_check"] = pd.DataFrame([dict(saved_loglik_full=reference.log_likelihood,
        current_loglik_full=full.log_likelihood, absolute_tolerance=5.1e-7,
        n_events_match=full.n_events == reference.n_events, n_bins_match=full.n_bins == reference.n_bins)])
    result["events_used"] = events
    result["forcing_scaling"] = context["scaling"]
    return result


def main():
    result = run_analysis()
    parameters = dict(catalogue=CATALOGUE_LABEL, bin_width_kyr=main_analysis.BIN_WIDTH_KA,
        history_window_kyr=main_analysis.HISTORY_WINDOW_KA,
        origin_fraction=main_analysis.BIN_ORIGIN_FRACTION, response_mode=main_analysis.RESPONSE_MODE,
        age_epoch="SpeleoAge is treated as kyr BP1950; its published reference year is unverified",
        source_realizations=str(AGE_INPUT.relative_to(PROJECT_ROOT)),
        realization_selection="all 500 saved orbital-experiment rows, unchanged order; no new draw",
        selection_seed=20260909, full_terms=";".join(main_analysis.FULL_TERMS),
        interaction_terms=";".join(interaction.INTERACTION_TERMS), added_parameters=2,
        p_method="nominal chi-square, df=2; no interaction bootstrap",
        interval="2.5–97.5% chronology sensitivity; not full sampling confidence")
    inputs = [Path(__file__).resolve(), Path(interaction.__file__), Path(main_analysis.__file__),
        Path(age_sensitivity.__file__), Path(combined_pi.__file__), PROJECT_ROOT / "toolbox/poisson.py",
        PROJECT_ROOT / "toolbox/project_config.py", PROJECT_ROOT / "toolbox/event_inputs.py",
        PROJECT_ROOT / "toolbox/orbital_phase.py", main_analysis.BARKER_XLS,
        AGE_INPUT, MAIN_SUMMARY, LR04_XLSX, CO2_XLSX, PRE_TXT]
    data_dir = interaction.save_results(result, ROOT, RUN_NAME, CATALOGUE_LABEL, parameters, inputs)
    print(result["comparison_summary"].to_string(index=False))
    print(f"Saved {data_dir}")


if __name__ == "__main__":
    main()
