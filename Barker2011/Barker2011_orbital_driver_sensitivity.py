#!/usr/bin/env python3
"""Scalar orbital drivers and phase on Barker Table S3 SpeleoAge, 0–400 kyr.

Use the same eight comparisons and plot layout as NGRIP–MIS6. Existing
combined-age realizations are rebinned on the unchanged main-analysis grid.
"""

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Barker2011 import Barker2011_event_phase_analysis as main_analysis
from Barker2011 import Barker2011_event_uncertainty_sensitivity as age_sensitivity
from toolbox import combined_pi, orbital_driver_sensitivity as orbital
from toolbox import orbital_driver_reporting as reporting
from toolbox.project_config import PROJECT_ROOT, LR04_XLSX, CO2_XLSX, PRE_TXT, OBL_TXT

ROOT = Path(__file__).resolve().parent
RUN_NAME = "Barker2011_orbital_driver_sensitivity"
CATALOGUE_LABEL = "Barker 2011 SpeleoAge warming events"
N_REALIZATIONS = 500
SEED = 20260909
AGE_INPUT = ROOT / "data/processed/Barker2011_event_age_uncertainty/event_age_realizations.csv"
MAIN_SUMMARY = ROOT / "data/processed/Barker2011_event_phase_analysis/analysis_summary.csv"


def run_analysis(n_realizations=N_REALIZATIONS, seed=SEED, show_progress=True):
    events = main_analysis.load_barker_source()
    # Source row IDs connect the fixed published event membership to saved ages.
    events.insert(0, "event_id", [f"Barker_S3_{row:03d}" for row in events.source_excel_row])
    context = age_sensitivity.prepare_context(events)
    response, scaling, provenance = orbital.prepare_drivers(context["frame"].reset_index(drop=True))
    context["frame"] = response
    point = orbital.fit_models(response, main_analysis.REDUCED_TERMS)
    if not point["models"].fit_valid.all() or not point["comparisons"].fit_valid.all():
        raise RuntimeError(f"Inspect invalid point-age fits:\n{point['models'].to_string(index=False)}")
    reference_check = reporting.check_reference(point, pd.read_csv(MAIN_SUMMARY).iloc[0])

    age_columns = [f"age_ka_bp__{event_id}" for event_id in events.event_id]
    selected = reporting.select_realizations(pd.read_csv(AGE_INPUT), age_columns, n_realizations, seed)
    mc_models, mc_comparisons, status = [], [], []
    segment = context["segment"]
    started = time.perf_counter()
    for index, row in selected.iterrows():
        ages = row[age_columns].to_numpy(float)
        outside = ages[0] < segment.bin_edges[0] or ages[-1] > segment.bin_edges[-1]
        reason = "outside Barker observation support" if outside else ""
        if outside:
            fitted = reporting.invalid_tables(point, reason)
        else:
            frame = age_sensitivity.frame_for_ages(ages, context)
            fitted = orbital.fit_models(frame, main_analysis.REDUCED_TERMS)
        for key, output in (("models", mc_models), ("comparisons", mc_comparisons)):
            output.append(fitted[key].assign(realization_id=row.realization_id))
        valid = bool(fitted["comparisons"].fit_valid.all())
        status.append(dict(realization_id=row.realization_id, within_observation_support=not outside,
                           n_response_events=fitted["models"].iloc[0].n_events,
                           all_comparisons_valid=valid, invalid_reason=reason or "; ".join(
                               fitted["comparisons"].loc[~fitted["comparisons"].fit_valid, "invalid_reason"].unique())))
        if show_progress and (index + 1) % 100 == 0:
            print(f"Barker: {index + 1}/{n_realizations} chronologies "
                  f"({time.perf_counter() - started:.0f} s)", flush=True)
    mc_models = pd.concat(mc_models, ignore_index=True)
    mc_comparisons = pd.concat(mc_comparisons, ignore_index=True)
    support = pd.DataFrame([dict(segment_id="Barker2011", observation_start_kyr_bp=0.0,
        observation_end_kyr_bp=400.0, response_start_kyr_bp=segment.response_start_kyr_bp,
        response_end_kyr_bp=segment.response_end_kyr_bp)])
    return dict(events=events, response=response, point=point, scaling=scaling, provenance=provenance,
        support=support, selected_realizations=selected, realization_status=pd.DataFrame(status),
        mc_models=mc_models, mc_comparisons=mc_comparisons, reference_check=reference_check,
        comparison_summary=orbital.summarize_comparisons(point["comparisons"], mc_comparisons),
        phase_summary=orbital.summarize_phase(point["models"], mc_models),
        parameters=dict(catalogue=CATALOGUE_LABEL, n_realizations=n_realizations, seed=seed,
            bin_width_kyr=main_analysis.BIN_WIDTH_KA, history_window_kyr=main_analysis.HISTORY_WINDOW_KA,
            origin_fraction=main_analysis.BIN_ORIGIN_FRACTION, response_mode=main_analysis.RESPONSE_MODE,
            response_exposure_kyr=response.dt_ka.sum(),
            age_epoch="SpeleoAge treated as BP1950; reference year unverified",
            source_realizations=str(AGE_INPUT.relative_to(PROJECT_ROOT)),
            n_new_comparisons=9, p_method="nominal chi-square; within-catalogue Holm family of 9",
            interval="2.5–97.5% age sensitivity; not sampling confidence interval"))


def main():
    result = run_analysis()
    inputs = [Path(__file__).resolve(), Path(orbital.__file__), Path(reporting.__file__),
        Path(main_analysis.__file__), Path(age_sensitivity.__file__), Path(combined_pi.__file__),
        PROJECT_ROOT / "toolbox/project_config.py", PROJECT_ROOT / "toolbox/poisson.py",
        PROJECT_ROOT / "toolbox/model_stats.py", PROJECT_ROOT / "toolbox/event_inputs.py",
        PROJECT_ROOT / "toolbox/orbital_phase.py", main_analysis.BARKER_XLS, AGE_INPUT,
        MAIN_SUMMARY, LR04_XLSX, CO2_XLSX, PRE_TXT, OBL_TXT,
        PROJECT_ROOT / "data/raw/ecc_1000_60_inter100.txt",
        PROJECT_ROOT / "data/raw/solstice_insolation_NH.nc",
        PROJECT_ROOT / "data/curated/orbital_driver_input_audit.csv"]
    data_dir, _ = reporting.save_results(result, ROOT, RUN_NAME, CATALOGUE_LABEL, inputs)
    print(result["comparison_summary"].to_string(index=False))
    print(f"Saved {data_dir}")


if __name__ == "__main__":
    main()
