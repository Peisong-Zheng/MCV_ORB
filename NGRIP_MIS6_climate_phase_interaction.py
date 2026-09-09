#!/usr/bin/env python3
"""Test LR04 modulation of precession in the primary NGRIP–MIS6 catalogue."""

from pathlib import Path

import numpy as np
import pandas as pd

from toolbox import combined_pi
from toolbox import climate_phase_interaction as interaction
from toolbox.project_config import PROJECT_ROOT, LR04_XLSX, CO2_XLSX, PRE_TXT

RUN_NAME = "NGRIP_MIS6_climate_phase_interaction"
CATALOGUE_LABEL = "NGRIP–MIS6 warming events"
AGE_INPUT = PROJECT_ROOT / "data/processed/NGRIP_MIS6_orbital_driver_sensitivity/selected_realizations.csv"
MAIN_SUMMARY = PROJECT_ROOT / "data/processed/NGRIP_MIS6_event_phase_analysis/analysis_summary.csv"


def run_analysis(n_realizations=None, show_progress=True):
    events = combined_pi.load_event_catalogue()
    context = combined_pi.build_context()
    bins = combined_pi.bin_catalogue(events, context)
    response = bins.loc[bins.in_response_interval].reset_index(drop=True)
    selected = pd.read_csv(AGE_INPUT, float_precision="round_trip")
    if n_realizations is not None:
        selected = selected.iloc[:n_realizations].copy()
    age_columns = [f"age_kyr_bp__{event_id}" for event_id in events.event_id]

    def frame_for_ages(ages):
        # Check each observation segment before binning; preserve event membership.
        for segment_id, segment in context.segments.items():
            values = ages[events.segment_id.eq(segment_id).to_numpy()]
            if np.any(values < segment.bin_edges[0]) or np.any(values > segment.bin_edges[-1]):
                raise ValueError(f"outside {segment_id} observation support")
        shifted = events.copy()
        shifted[combined_pi.EVENT_AGE_COLUMN] = ages
        current = combined_pi.bin_catalogue(shifted, context)
        return current.loc[current.in_response_interval].reset_index(drop=True)

    result = interaction.analyze_realizations(response, combined_pi.FULL_TERMS, selected,
        age_columns, frame_for_ages, show_progress=show_progress)
    reference = pd.read_csv(MAIN_SUMMARY).iloc[0]
    full = result["point_models"].set_index("model_id").loc["full"]
    # The main summary stores six decimal places; allow only that rounding.
    if not np.isclose(full.log_likelihood, reference.loglik_full, atol=5.1e-7, rtol=0):
        raise RuntimeError("The current full fit differs from the saved main analysis")
    result["reference_check"] = pd.DataFrame([dict(saved_loglik_full=reference.loglik_full,
        current_loglik_full=full.log_likelihood, absolute_tolerance=5.1e-7,
        n_events_match=full.n_events == reference.n_predictive_events,
        n_bins_match=full.n_bins == reference.n_predictive_bins)])
    result["events_used"] = events
    result["observation_segments"] = combined_pi.load_observation_segments()
    return result


def main():
    result = run_analysis()
    parameters = dict(catalogue=CATALOGUE_LABEL, bin_width_kyr=combined_pi.DEFAULT_BIN_WIDTH_KA,
        history_window_kyr=combined_pi.DEFAULT_HISTORY_WINDOW_KA,
        origin_fraction=combined_pi.DEFAULT_ORIGIN_FRACTION, response_mode=combined_pi.DEFAULT_RESPONSE_MODE,
        age_epoch="All ages are kyr BP relative to 1950", source_realizations=str(AGE_INPUT.relative_to(PROJECT_ROOT)),
        realization_selection="all 500 saved orbital-experiment rows, unchanged order; no new draw",
        selection_seed=20260909, full_terms=";".join(combined_pi.FULL_TERMS),
        interaction_terms=";".join(interaction.INTERACTION_TERMS), added_parameters=2,
        p_method="nominal chi-square, df=2; no interaction bootstrap",
        interval="2.5–97.5% chronology sensitivity; not full sampling confidence")
    inputs = [Path(__file__).resolve(), Path(interaction.__file__), Path(combined_pi.__file__),
        PROJECT_ROOT / "toolbox/poisson.py", PROJECT_ROOT / "toolbox/project_config.py",
        PROJECT_ROOT / "toolbox/event_inputs.py", PROJECT_ROOT / "toolbox/orbital_phase.py",
        combined_pi.EVENT_CATALOGUE_CSV, combined_pi.OBSERVATION_SEGMENTS_CSV,
        AGE_INPUT, MAIN_SUMMARY, LR04_XLSX, CO2_XLSX, PRE_TXT]
    data_dir = interaction.save_results(result, PROJECT_ROOT, RUN_NAME, CATALOGUE_LABEL, parameters, inputs)
    print(result["comparison_summary"].to_string(index=False))
    print(f"Saved {data_dir}")


if __name__ == "__main__":
    main()
