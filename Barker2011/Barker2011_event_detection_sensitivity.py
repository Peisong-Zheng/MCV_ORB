#!/usr/bin/env python3
"""Extra random event-deletion sensitivity for Barker varying-threshold events."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import chi2

from toolbox import event_model
from toolbox import event_detection_sensitivity as detection
from toolbox.project_config import (
    PROJECT_ROOT, BARKER_EVENT_CSVS,
    LR04_CSV, CO2_CSV, ORBITAL_CSV, PRECESSION_PHASE_CSV,
)


RUN_NAME = "Barker2011_event_detection_sensitivity"

OUTPUT_ROOT = PROJECT_ROOT
N_REPLICATES = 500
RANDOM_SEED = 20260913
QUADRATURE_ORDER = 4


def save_results(result, output_root):
    """Keep deletion outcomes, their reference and the actual retained members."""
    data_dir = output_root / "Barker2011/data/processed" / RUN_NAME
    diagnostics_dir = output_root / "tests/diagnostics" / RUN_NAME
    data_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    rows = result["replicates"]
    rows[[
        "drop_probability", "replicate_id", "seed", "n_deleted", "n_response_events",
        "fit_valid", "invalid_reason", "gain_bits_per_event", "LR_statistic", "beta_history",
        "beta_pre_phase_sin", "beta_pre_phase_cos", "phase_amplitude", "pre_phase_preferred_deg",
        "pre_phase_rate_ratio_max_vs_min", "phase_offset_deg",
    ]].to_csv(data_dir / "replicates.csv", index=False, float_format="%.12g")
    summary = result["scenario_summary"][[
        "drop_probability", "metric", "n_replicates", "n_valid", "n_finite", "q025", "median", "q975",
    ]].copy()
    significant = rows.fit_valid & (chi2.sf(rows.LR_statistic, 2) < 0.05)
    counts = rows.assign(significant=significant).groupby("drop_probability").significant.sum()
    summary["n_significant"] = summary.drop_probability.map(counts)
    # The denominator includes all requested masks, including any failed fit.
    summary["significant_fraction"] = summary.n_significant / summary.n_replicates
    summary.to_csv(data_dir / "scenario_summary.csv", index=False, float_format="%.12g")
    reference = result["reference"][[
        "n_response_events", "response_exposure_kyr", "gain_bits_per_event", "LR_statistic",
        "beta_history", "beta_pre_phase_sin", "beta_pre_phase_cos",
        "pre_phase_preferred_deg", "pre_phase_rate_ratio_max_vs_min",
    ]].copy()
    for name in ("model_version", "seed", "n_replicates_per_scenario", "history_tau_kyr",
                 "initial_history", "quadrature_order"):
        reference[name] = result["parameters"][name]
    support = result["windows"].iloc[0]
    for name in ("observation_start_kyr_bp", "observation_end_kyr_bp", "response_start_kyr_bp",
                 "response_end_kyr_bp", "anchor_age_kyr_bp"):
        reference[name] = support[name]
    reference.to_csv(data_dir / "reference.csv", index=False, float_format="%.12g")
    np.savez_compressed(diagnostics_dir / "retained_event_masks.npz",
        event_ids=np.asarray(result["event_ids"], dtype=str),
        event_ages_kyr_bp=result["events"]["event_age_kyr_bp"].to_numpy(),
        retained=result["retained_masks"], replicate_id=rows.replicate_id.to_numpy(),
        scope=rows.scope.to_numpy(str), drop_probability=rows.drop_probability.to_numpy())
    print(summary.loc[summary.metric.isin(["gain_bits_per_event", "phase_offset_deg"])]
          .to_string(index=False))
    return data_dir


def run_analysis(n_replicates=500, seed=20260913, *, quadrature_order=4, show_progress=True):
    events = pd.read_csv(BARKER_EVENT_CSVS["variable_threshold"], float_precision="round_trip")
    events["segment_id"] = "Barker2011"
    if len(events) != 70 or events.event_id.isna().any() or not events.event_id.is_unique:
        raise ValueError("Check the prepared Barker event count and identities")
    if not np.all(np.diff(events.event_age_kyr_bp) > 0):
        raise ValueError("Barker event ages must be strictly increasing")
    observations = pd.DataFrame([dict(segment_id="Barker2011",
        observation_start_kyr_bp=0., observation_end_kyr_bp=400.)])
    lr04 = pd.read_csv(LR04_CSV, float_precision="round_trip")
    co2 = pd.read_csv(CO2_CSV, float_precision="round_trip")
    orbital_data = pd.read_csv(ORBITAL_CSV, float_precision="round_trip")
    anchors = pd.read_csv(PRECESSION_PHASE_CSV, float_precision="round_trip")
    forcings = {
        "lr04": (lr04.age_kyr_bp.to_numpy(), lr04.lr04.to_numpy()),
        "co2": (co2.age_kyr_bp.to_numpy(), co2.co2_ppm.to_numpy()),
        "precession_index": (orbital_data.age_kyr_bp.to_numpy(), orbital_data.precession_index.to_numpy()),
    }
    phase_anchors = (anchors.age_kyr_bp.to_numpy(), anchors.phase_unwrapped_rad.to_numpy())
    windows = event_model.response_windows(events, observations)
    scaling = event_model.nominal_scaling({name: forcings[name] for name in ("lr04", "co2")}, windows)
    background = ("intercept", "same_type_exponential_history", "lr04_scaled", "co2_scaled")
    catalogue_id = "barker_variable_threshold_speleo_0_400"
    scopes = {"all": None}
    with_phase = background + ("pre_phase_sin", "pre_phase_cos")
    return detection.analyze_deletions(
        events, windows, forcings, phase_anchors, scaling, background, with_phase,
        catalogue_id=catalogue_id, scopes=scopes, n_replicates=n_replicates, seed=seed,
        quadrature_order=quadrature_order, show_progress=show_progress,
    )


def main():
    result = run_analysis(N_REPLICATES, RANDOM_SEED, quadrature_order=QUADRATURE_ORDER)
    data_dir = save_results(result, OUTPUT_ROOT)
    print(f"Saved {data_dir}; valid fits {result['replicates'].fit_valid.sum()}/{len(result['replicates'])}.")
    if not result["replicates"].fit_valid.all():
        raise RuntimeError("Deletion failures are saved; resolve or report them before publication")


if __name__ == "__main__":
    main()
