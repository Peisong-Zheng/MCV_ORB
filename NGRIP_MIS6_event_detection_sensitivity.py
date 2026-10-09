#!/usr/bin/env python3
"""Extra random event-deletion sensitivity for NGRIP--MIS6; tables and notes."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from toolbox import event_model
from toolbox import event_detection_sensitivity as detection
from toolbox.project_config import (
    PROJECT_ROOT, EVENT_CATALOGUE_CSV, OBSERVATION_SEGMENTS_CSV,
    LR04_CSV, CO2_CSV, ORBITAL_CSV, PRECESSION_PHASE_CSV, generated_notes_dir,
)


RUN_NAME = "NGRIP_MIS6_event_detection_sensitivity"
OUTPUT_ROOT = PROJECT_ROOT
N_REPLICATES = 500
RANDOM_SEED = 20260912
QUADRATURE_ORDER = 4


def run_analysis(n_replicates=500, seed=20260912, *, quadrature_order=4, show_progress=True):
    events = pd.read_csv(EVENT_CATALOGUE_CSV)
    observations = pd.read_csv(OBSERVATION_SEGMENTS_CSV)
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
    scaling = event_model.scale_forcing_v2({name: forcings[name] for name in ("lr04", "co2")}, windows)
    background = ("intercept", "same_type_exponential_history", "lr04_scaled", "co2_scaled", "mis6_segment")
    catalogue_id = "ngrip_warming_plus_mis6"
    scopes = {"both": None, "MIS6_only": ("MIS6",)}
    with_phase = background + ("pre_phase_sin", "pre_phase_cos")
    return detection.analyze_deletions(
        events, windows, forcings, phase_anchors, scaling, background, with_phase,
        catalogue_id=catalogue_id, scopes=scopes, n_replicates=n_replicates, seed=seed,
        quadrature_order=quadrature_order, show_progress=show_progress,
    )


def save_results(result, output_root, *, diagnostics_root=None):
    """Write compact research tables, masks for reproducibility and English notes."""
    run_name = RUN_NAME
    output_root = Path(output_root)
    data_dir = output_root / "data/processed" / run_name
    notes_dir = generated_notes_dir(output_root)
    diagnostics_dir = (output_root / "tests/diagnostics" / run_name if diagnostics_root is None
                       else Path(diagnostics_root) / run_name)
    for directory in (data_dir, notes_dir, diagnostics_dir):
        directory.mkdir(parents=True, exist_ok=True)
    for name in ("scenario_summary", "replicates", "reference"):
        result[name].to_csv(data_dir / f"{name}.csv", index=False, float_format="%.12g")
    result["windows"].to_csv(data_dir / "support.csv", index=False)
    pd.DataFrame([dict(parameter=name, value=value) for name, value in result["parameters"].items()]).to_csv(
        data_dir / "parameters_and_provenance.csv", index=False)
    np.savez_compressed(diagnostics_dir / "retained_event_masks.npz",
                        event_ids=np.asarray(result["event_ids"], dtype=str),
                        event_ages_kyr_bp=result["events"]["event_age_kyr_bp"].to_numpy(),
                        retained=result["retained_masks"],
                        replicate_id=result["replicates"].replicate_id.to_numpy(),
                        scope=result["replicates"].scope.to_numpy(str),
                        drop_probability=result["replicates"].drop_probability.to_numpy())
    input_paths = [Path(__file__), Path(detection.__file__), PROJECT_ROOT / "toolbox/event_model.py",
                   PROJECT_ROOT / "toolbox/model_stats.py", PROJECT_ROOT / "toolbox/point_process.py",
                   LR04_CSV, CO2_CSV, ORBITAL_CSV, PRECESSION_PHASE_CSV]
    input_paths.append(EVENT_CATALOGUE_CSV)
    pd.DataFrame([dict(path=str(path.relative_to(PROJECT_ROOT)),
                       sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                  for path in input_paths]).to_csv(diagnostics_dir / "input_code_sha256.csv", index=False)
    rows = result["replicates"]
    summary = result["scenario_summary"]
    report_metrics = ["gain_bits_per_event", "phase_offset_deg", "pre_phase_rate_ratio_max_vs_min"]
    values = summary.loc[summary.metric.isin(report_metrics),
                         ["scope", "drop_probability", "metric", "median", "q025", "q975", "n_finite"]]
    note = f"""Extra event-deletion sensitivity: {result["parameters"]["catalogue_id"]}

Each noninitial response event is independently deleted with probability 0.1 or 0.2.
There are {result['parameters']['n_replicates_per_scenario']} replicate masks per scope/probability.
Every original conditioning event, response endpoint and forcing scale remains fixed.
Both continuous nested models are refitted, rebuilding history only from retained events.
Replicate seeds pair the deletion masks across probabilities and eligible segment scopes.
This is a stress test of additional loss, not a reconstruction or correction of pre-existing
missing events. It cannot rule out phase-dependent or climate-dependent detection bias.
No nested null bootstrap or chronology Monte Carlo is included in this experiment.

Valid fits: {int(rows.fit_valid.sum())}/{len(rows)}. Failed fits are retained with their masks
and explicit reasons; effect quantiles use finite supported estimates and show their denominator.
Quantiles describe the specified deletion scenarios, not sampling confidence intervals.
Phase offsets are circular differences in [-180, 180) degrees from the original estimate.
Phase coefficients and amplitude remain available because a weak effect has an uncertain peak.

{values.to_string(index=False, float_format=lambda x: f'{x:.6g}')}

Research tables: scenario_summary.csv, replicates.csv, reference.csv, support.csv and parameters_and_provenance.csv.
The compressed membership masks and input hashes are stored under tests/diagnostics.
No figure is generated by this experiment.
"""
    (notes_dir / f"{run_name}_Methods_and_results.txt").write_text(note, encoding="utf-8")
    return data_dir


def main():
    result = run_analysis(N_REPLICATES, RANDOM_SEED, quadrature_order=QUADRATURE_ORDER)
    data_dir = save_results(result, OUTPUT_ROOT)
    print(f"Saved {data_dir}; valid fits {result['replicates'].fit_valid.sum()}/{len(result['replicates'])}.")
    if not result["replicates"].fit_valid.all():
        raise RuntimeError("Deletion experiment has failed fits; see saved results")


if __name__ == "__main__":
    main()
