#!/usr/bin/env python3
"""Compare scalar orbital drivers with phase in the pooled warming catalogue.

Run after the main and combined-age analyses. The eight models share the
current response support and baseline; only the added orbital terms differ.
Saved combined chronologies supply age sensitivity, not a null bootstrap.
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd

from toolbox import combined_pi, orbital_driver_sensitivity as orbital
from toolbox import orbital_driver_reporting as reporting
from toolbox.project_config import PROJECT_ROOT, LR04_XLSX, CO2_XLSX, PRE_TXT, OBL_TXT

RUN_NAME = "NGRIP_MIS6_orbital_driver_sensitivity"
CATALOGUE_LABEL = "NGRIP–MIS6 warming events"
N_REALIZATIONS = 500
SEED = 20260909
AGE_INPUT = PROJECT_ROOT / "data/processed/NGRIP_MIS6_event_uncertainty_sensitivity/combined_event_age_realizations.csv"
MAIN_SUMMARY = PROJECT_ROOT / "data/processed/NGRIP_MIS6_event_phase_analysis/analysis_summary.csv"


def frame_for_ages(ages, events, context, response):
    """Recount each segment independently, retaining the fixed response predictors."""
    values = np.asarray(ages, dtype=float)
    if values.shape != (len(events),) or not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
        raise ValueError("Event ages must match the finite, ordered catalogue")
    shifted = events.copy()
    shifted[combined_pi.EVENT_AGE_COLUMN] = values
    binned = combined_pi.bin_catalogue(shifted, context)
    current = binned.loc[binned.in_response_interval]
    frame = response.copy()
    for column in ("event_count", "same_type_history_count"):
        frame[column] = current[column].to_numpy()
    return frame


def outside_support(ages, events, context):
    for segment_id, segment in context.segments.items():
        values = ages[events.segment_id.eq(segment_id).to_numpy()]
        if np.any(values < segment.bin_edges[0]) or np.any(values > segment.bin_edges[-1]):
            return f"outside {segment_id} observation support"
    return ""


def run_analysis(n_realizations=N_REALIZATIONS, seed=SEED, show_progress=True):
    events = combined_pi.load_event_catalogue()
    context = combined_pi.build_context()
    binned = combined_pi.bin_catalogue(events, context)
    response = binned.loc[binned.in_response_interval].reset_index(drop=True).rename(
        columns={"bin_center_kyr_bp": "bin_center_ka", "dt_kyr": "dt_ka"})
    response, scaling, provenance = orbital.prepare_drivers(response)
    point = orbital.fit_models(response, combined_pi.REDUCED_TERMS)
    if not point["models"].fit_valid.all() or not point["comparisons"].fit_valid.all():
        raise RuntimeError(f"Inspect invalid point-age fits:\n{point['models'].to_string(index=False)}")
    reference_check = reporting.check_reference(point, pd.read_csv(MAIN_SUMMARY).iloc[0])

    # Keep the original NGRIP/MIS6 pairing; sample existing combined rows once.
    age_columns = [f"age_kyr_bp__{event_id}" for event_id in events.event_id]
    selected = reporting.select_realizations(pd.read_csv(AGE_INPUT), age_columns, n_realizations, seed)
    mc_models, mc_comparisons, status = [], [], []
    started = time.perf_counter()
    for index, row in selected.iterrows():
        ages = row[age_columns].to_numpy(float)
        reason = outside_support(ages, events, context)
        if reason:
            fitted = reporting.invalid_tables(point, reason)
        else:
            frame = frame_for_ages(ages, events, context, response)
            fitted = orbital.fit_models(frame, combined_pi.REDUCED_TERMS)
        for key, output in (("models", mc_models), ("comparisons", mc_comparisons)):
            output.append(fitted[key].assign(realization_id=row.realization_id))
        valid = bool(fitted["comparisons"].fit_valid.all())
        status.append(dict(realization_id=row.realization_id, within_observation_support=not bool(reason),
                           n_response_events=fitted["models"].iloc[0].n_events,
                           all_comparisons_valid=valid, invalid_reason=reason or "; ".join(
                               fitted["comparisons"].loc[~fitted["comparisons"].fit_valid, "invalid_reason"].unique())))
        if show_progress and (index + 1) % 100 == 0:
            print(f"NGRIP–MIS6: {index + 1}/{n_realizations} chronologies "
                  f"({time.perf_counter() - started:.0f} s)", flush=True)
    mc_models = pd.concat(mc_models, ignore_index=True)
    mc_comparisons = pd.concat(mc_comparisons, ignore_index=True)
    support = combined_pi.load_observation_segments()
    for segment_id, segment in context.segments.items():
        mask = support.segment_id.eq(segment_id)
        support.loc[mask, "response_start_kyr_bp"] = segment.response_start_kyr_bp
        support.loc[mask, "response_end_kyr_bp"] = segment.response_end_kyr_bp
    return dict(events=events, response=response, point=point, scaling=scaling, provenance=provenance,
        support=support, selected_realizations=selected, realization_status=pd.DataFrame(status),
        mc_models=mc_models, mc_comparisons=mc_comparisons, reference_check=reference_check,
        comparison_summary=orbital.summarize_comparisons(point["comparisons"], mc_comparisons),
        phase_summary=orbital.summarize_phase(point["models"], mc_models),
        parameters=dict(catalogue=CATALOGUE_LABEL, n_realizations=n_realizations, seed=seed,
            bin_width_kyr=context.bin_width_ka, history_window_kyr=context.history_window_ka,
            origin_fraction=context.origin_fraction, response_mode=context.response_mode,
            response_exposure_kyr=context.response_exposure_kyr, age_epoch="BP1950",
            source_realizations=str(AGE_INPUT.relative_to(PROJECT_ROOT)),
            n_new_comparisons=9, p_method="nominal chi-square; within-catalogue Holm family of 9",
            interval="2.5–97.5% age sensitivity; not sampling confidence interval"))


def main():
    result = run_analysis()
    inputs = [Path(__file__).resolve(), Path(orbital.__file__), Path(reporting.__file__),
        Path(combined_pi.__file__), PROJECT_ROOT / "toolbox/project_config.py",
        PROJECT_ROOT / "toolbox/poisson.py", PROJECT_ROOT / "toolbox/model_stats.py",
        PROJECT_ROOT / "toolbox/event_inputs.py", PROJECT_ROOT / "toolbox/orbital_phase.py",
        combined_pi.EVENT_CATALOGUE_CSV, combined_pi.OBSERVATION_SEGMENTS_CSV, AGE_INPUT,
        MAIN_SUMMARY, LR04_XLSX, CO2_XLSX, PRE_TXT, OBL_TXT,
        PROJECT_ROOT / "data/raw/ecc_1000_60_inter100.txt",
        PROJECT_ROOT / "data/raw/solstice_insolation_NH.nc",
        PROJECT_ROOT / "data/curated/orbital_driver_input_audit.csv"]
    data_dir, _ = reporting.save_results(result, PROJECT_ROOT, RUN_NAME, CATALOGUE_LABEL, inputs)
    print(result["comparison_summary"].to_string(index=False))
    print(f"Saved {data_dir}")


if __name__ == "__main__":
    main()
