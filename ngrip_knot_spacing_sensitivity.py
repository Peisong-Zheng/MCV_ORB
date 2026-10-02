#!/usr/bin/env python3
"""Refit the three saved NGRIP chronology-correlation ensembles.

Each knot spacing retains its original 2,000 paired 55-event age sequences.
The same MIS 6 ages enter all three spacings. Original chronology diagnostics
and the complete age archive are copied unchanged; no ages are resampled.
Continuous likelihoods condition on the exact oldest event of each segment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys

for thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[thread_variable] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mcv_orb_matplotlib")

import numpy as np
import pandas as pd
import scipy

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import NGRIP_MIS6_event_uncertainty_sensitivity as pooled
from toolbox import age_sensitivity, combined_likelihood, point_process
from toolbox.project_config import CO2_CSV, LR04_CSV, ORBITAL_CSV, PRECESSION_PHASE_CSV

OUT_DATA_DIR = pooled.OUT_DATA_DIR / "ngrip_knot_spacing"
KNOT_SPACINGS_KA = (2.5, 5.0, 10.0)
N_REALIZATIONS = 2_000
ID_COLUMNS = ["realization_id", "ngrip_realization_id", "mis6_realization_id"]
CHRONOLOGY_FILES = (
    "age_realizations.npz", "sampling_diagnostics.csv", "analytic_basis_summary.csv",
    "analytic_event_sigma.csv", "analytic_adjacent_age_difference_sigma.csv",
    "chronology_knots.csv",
)


def input_manifest(input_dir) -> pd.DataFrame:
    """Hash the actual saved ages, paired IDs, forcings, and fitting code."""
    paths = [
        *[input_dir / filename for filename in CHRONOLOGY_FILES],
        input_dir / "gain_realizations.csv", input_dir / "parameters_and_provenance.csv",
        combined_likelihood.EVENT_CATALOGUE_CSV, combined_likelihood.OBSERVATION_SEGMENTS_CSV,
        CO2_CSV, LR04_CSV, ORBITAL_CSV, PRECESSION_PHASE_CSV, Path(__file__), Path(pooled.__file__),
        Path(age_sensitivity.__file__), Path(combined_likelihood.__file__), Path(point_process.__file__),
        *[PROJECT_ROOT / "toolbox" / filename for filename in (
            "project_config.py", "event_model.py", "model_stats.py",
        )],
    ]
    rows = []
    for path in paths:
        label = str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)
        rows.append({"path": label, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "bytes": path.stat().st_size})
    return pd.DataFrame(rows)


def load_saved_pairings(events, input_dir, n_realizations):
    """Read paired age arrays directly; verify their event and realization IDs."""
    saved_ids = pd.read_csv(input_dir / "gain_realizations.csv", usecols=["knot_spacing_ka", *ID_COLUMNS])
    paired = {}
    mis6_reference = None
    with np.load(input_dir / "age_realizations.npz", allow_pickle=False) as archive:
        if not np.array_equal(archive["pooled_event_ids"], events.event_id.to_numpy(str)):
            raise ValueError("Saved age columns do not match the curated event order")
        for spacing in KNOT_SPACINGS_KA:
            tag = f"knots_{spacing:g}".replace(".", "p")
            ages = archive[f"{tag}__paired_55_ages_ka_bp"]
            identifiers = saved_ids.loc[saved_ids.knot_spacing_ka.eq(spacing), ID_COLUMNS].reset_index(drop=True)
            if len(ages) != len(identifiers) or not 1 <= n_realizations <= len(ages):
                raise ValueError("Requested count or saved pairing dimensions are inconsistent")
            if identifiers.isna().any().any() or any(identifiers[column].duplicated().any() for column in ID_COLUMNS):
                raise ValueError("Saved pairing IDs must be complete and unique within each spacing")
            for column, key in (
                ("ngrip_realization_id", f"{tag}__paired_ngrip_realization_id"),
                ("mis6_realization_id", "paired_mis6_realization_id"),
            ):
                if not np.array_equal(archive[key], identifiers[column].to_numpy(str)):
                    raise ValueError(f"Saved {column} differs between the age archive and result table")
            if ages.shape[1] != len(events) or not np.isfinite(ages).all() or not np.all(np.diff(ages, axis=1) > 0):
                raise ValueError("Saved age arrays must contain 55 finite ordered event ages")
            draws = pd.concat([
                identifiers.iloc[:n_realizations].copy(),
                pd.DataFrame(ages[:n_realizations], columns=pooled.combined_age_columns(events)),
            ], axis=1)
            mis6_columns = [column for column in pooled.combined_age_columns(events) if "MIS6:" in column]
            mis6_used = draws[["mis6_realization_id", *mis6_columns]]
            if mis6_reference is None:
                mis6_reference = mis6_used.copy()
            else:
                pd.testing.assert_frame_equal(mis6_reference, mis6_used, check_exact=True)
            paired[spacing] = draws
    return paired


def run(input_dir, output_dir, n_realizations=N_REALIZATIONS, workers=1):
    manifest_before = input_manifest(input_dir)
    original_parameters = pd.read_csv(input_dir / "parameters_and_provenance.csv").set_index("parameter").value
    context = combined_likelihood.build_context(history_tau_ka=pooled.HISTORY_TAU_KYR)
    events = context.events
    point_fit = combined_likelihood.fit_catalogue(events, context)
    pairings = load_saved_pairings(events, input_dir, n_realizations)
    summaries, gain_tables, fitting_rows = [], [], []
    for spacing, draws in pairings.items():
        print(f"Refitting {spacing:g} kyr knots: {len(draws):,} saved paired chronologies", flush=True)
        results, fitting = pooled.fit_realizations(events, draws, context, show_progress=True, n_workers=workers)
        summary = pooled.build_summary(results, point_fit)
        summary.insert(0, "knot_spacing_ka", spacing)
        summary["experiment"] = f"{n_realizations}-draw correlation-scale screening"
        compact = pooled.compact_gain_results(results)
        compact.insert(0, "knot_spacing_ka", spacing)
        summaries.append(summary)
        gain_tables.append(compact)
        fitting_rows.append({"knot_spacing_ka": spacing, **fitting})
        row = summary.iloc[0]
        print(f"{spacing:g} kyr: nominal p < 0.05 in {int(row.n_nominal_p_below_0p05)}/{int(row.n_valid)} valid fits", flush=True)

    pd.testing.assert_frame_equal(manifest_before, input_manifest(input_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.concat(summaries, ignore_index=True)
    outputs = {
        "summary.csv": summary,
        "gain_realizations.csv": pd.concat(gain_tables, ignore_index=True),
        "fitting_diagnostics.csv": pd.DataFrame(fitting_rows),
        "input_code_sha256.csv": manifest_before,
    }
    for filename, table in outputs.items():
        table.to_csv(output_dir / filename, index=False)
    if input_dir.resolve() != output_dir.resolve():
        for filename in CHRONOLOGY_FILES:
            shutil.copy2(input_dir / filename, output_dir / filename)

    parameters = {
        "experiment": "NGRIP correlation-scale screening; does not replace primary 10000-draw result",
        "model_version": combined_likelihood.MODEL_VERSION,
        "n_realizations_per_spacing": n_realizations,
        "knot_spacings_ka": json.dumps(KNOT_SPACINGS_KA),
        "ngrip_seed": original_parameters["ngrip_seed"],
        "pairing_seed": original_parameters["pairing_seed"],
        "age_input_policy": "reuse original paired_55_ages arrays and saved source IDs; no sampling",
        "age_archive_policy": "complete original NPZ copied byte-for-byte, including when fitting a subset",
        "source_directory": str(input_dir),
        "chronology_diagnostics": "original sampling, knot, and analytical basis diagnostics copied unchanged",
        "ngrip_joint_boundary_count": 69, "ngrip_warming_source_count": 34,
        "mis6_source_count": 21, "n_response_events": point_fit.summary["n_response_events"],
        "n_conditioning_events": point_fit.summary["n_conditioning_events"],
        "mis6_pairing": "identical source IDs and ages across spacing; checked exactly",
        "ngrip_pairing": "original same seed, different innovation dimensions; rows are not the same latent chronology",
        "age_epoch": "kyr BP1950; stored ages already converted, no new epoch shift",
        "working_covariance": "independent cumulative Gaussian variance increments and linear error interpolation",
        "counted_working_sigma": "MCE/2 at knots; actual interpolated sigma recorded separately",
        "modelext_nominal_envelope": "+/-4.5% of age, user-selected working 2-sigma scale, not a hard bound",
        "order_conditioning": "original rejection of nonmonotonic maps or crossed complete 69-boundary sequences",
        "basis_diagnostics": "analytical unconditioned chronology only, before definition error or order conditioning",
        "observation_support": "outside-support draws retained invalid; no clipping or resampling",
        "response_support": "condition on exact oldest event of each segment in each realization",
        "forcing_scaling": "fixed nominal time-weighted scaling",
        "robustness_fraction": "denominator is valid fits; nominal significance fraction is not an empirical p value",
        "history_tau_ka": context.history_tau_ka,
        "history_coefficient_domain": "nonpositive",
        "initial_history": context.initial_history,
        "quadrature_order": context.quadrature_order,
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "scipy": scipy.__version__,
    }
    pd.DataFrame(parameters.items(), columns=["parameter", "value"]).to_csv(
        output_dir / "parameters_and_provenance.csv", index=False,
    )
    if any(row["n_numerical_failures"] for row in fitting_rows):
        raise RuntimeError("Unresolved numerical age fits; inspect fitting_diagnostics.csv")
    print(summary[["knot_spacing_ka", "n_valid", "n_invalid", "fraction_nominal_p_below_0p05",
                   "gain_bits_per_event_median", "pre_phase_preferred_deg_median"]].to_string(index=False))
    print(f"Saved screening products to {output_dir}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=OUT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUT_DATA_DIR)
    parser.add_argument("--n-realizations", type=int, default=N_REALIZATIONS)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    run(args.input_dir, args.output_dir, args.n_realizations, args.workers)


if __name__ == "__main__":
    main()
