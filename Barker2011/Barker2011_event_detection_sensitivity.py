#!/usr/bin/env python3
"""Extra random event-deletion sensitivity for Barker varying-threshold events."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import chi2

from toolbox import combined_likelihood as likelihood
from toolbox import event_detection_sensitivity as detection
from toolbox.project_config import PROJECT_ROOT, BARKER_EVENT_CSVS


RUN_NAME = "Barker2011_event_detection_sensitivity"


def save_results(result, context, output_root):
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
    support = likelihood.support_table(context).iloc[0]
    for name in ("observation_start_kyr_bp", "observation_end_kyr_bp", "response_start_kyr_bp",
                 "response_end_kyr_bp", "anchor_age_kyr_bp"):
        reference[name] = support[name]
    reference.to_csv(data_dir / "reference.csv", index=False, float_format="%.12g")
    np.savez_compressed(diagnostics_dir / "retained_event_masks.npz",
        event_ids=np.asarray(result["event_ids"], dtype=str),
        event_ages_kyr_bp=context.events[likelihood.EVENT_AGE_COLUMN].to_numpy(),
        retained=result["retained_masks"], replicate_id=rows.replicate_id.to_numpy(),
        scope=rows.scope.to_numpy(str), drop_probability=rows.drop_probability.to_numpy())
    print(summary.loc[summary.metric.isin(["gain_bits_per_event", "phase_offset_deg"])]
          .to_string(index=False))
    return data_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--n-replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--quadrature-order", type=int, default=4)
    args = parser.parse_args()
    events = pd.read_csv(BARKER_EVENT_CSVS["variable_threshold"], float_precision="round_trip")
    context = likelihood.build_barker_context(events, quadrature_order=args.quadrature_order)
    result = detection.run_catalogue_analysis(context, scopes={"all": None},
        n_replicates=args.n_replicates, seed=args.seed)
    data_dir = save_results(result, context, args.output_root)
    print(f"Saved {data_dir}; valid fits {result['replicates'].fit_valid.sum()}/{len(result['replicates'])}.")
    if not result["replicates"].fit_valid.all():
        raise RuntimeError("Deletion failures are saved; resolve or report them before publication")


if __name__ == "__main__":
    main()
