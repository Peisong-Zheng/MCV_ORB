#!/usr/bin/env python3
"""Refit the saved So-57 overlap pairings with the continuous event likelihood.

The previous run saved source realization IDs but no joint age table. Ages are
recovered by those IDs, without drawing new pairings or chronologies. The
reconstructed wide table is retained for exact checks on subsequent runs.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys
import time

for thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[thread_variable] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mcv_orb_matplotlib")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

import NGRIP_MIS6_event_uncertainty_sensitivity as joint
from toolbox import age_sensitivity, combined_likelihood, point_process
from toolbox.project_config import CO2_CSV, LR04_CSV, ORBITAL_CSV, PRECESSION_PHASE_CSV

DEFAULT_MIS6_SOURCE = (
    PROJECT_ROOT / "MIS6/data/processed/MIS6_event_age_uncertainty"
    / "mis6_event_age_realizations_so57_overlap.csv"
)
DEFAULT_OUTPUT = joint.OUT_DATA_DIR / "sofular_so57_overlap"
DEFAULT_PAIRINGS = DEFAULT_OUTPUT / "gain_realizations.csv"
ID_COLUMNS = ["realization_id", "ngrip_realization_id", "mis6_realization_id"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_label(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def load_saved_pairings(events, pairing_path, ngrip_path, mis6_path, n_realizations):
    """Recover the original 55 ages from the three saved realization IDs."""
    pairs = pd.read_csv(pairing_path, usecols=ID_COLUMNS)
    if not 1 <= n_realizations <= len(pairs):
        raise ValueError("Requested count must lie within the saved pairings")
    pairs = pairs.loc[:n_realizations - 1, ID_COLUMNS].copy()
    if pairs.isna().any().any() or any(pairs[column].duplicated().any() for column in ID_COLUMNS):
        raise ValueError("Saved pairing IDs must be complete and unique")

    ages = np.empty((len(pairs), len(events)))
    columns = joint.source_age_columns(events)
    for segment, path, id_column in (
        ("NGRIP", ngrip_path, "ngrip_realization_id"),
        ("MIS6", mis6_path, "mis6_realization_id"),
    ):
        source = pd.read_csv(path, dtype={"realization_id": str}, float_precision="round_trip")
        joint._validate_source_ensemble(source, columns[segment], segment)
        selected = source.set_index("realization_id").loc[pairs[id_column].astype(str), columns[segment]]
        ages[:, events.segment_id.eq(segment).to_numpy()] = selected.to_numpy(float)
    if not np.all(np.diff(ages, axis=1) > 0):
        raise ValueError("Recovered event ages must retain the original rank")
    return pd.concat([pairs, pd.DataFrame(ages, columns=joint.combined_age_columns(events))], axis=1)


def run(n_realizations: int, ngrip_source: Path, mis6_source: Path,
        pairing_source: Path, output_dir: Path, workers: int = 1) -> pd.DataFrame:
    context = combined_likelihood.build_context(history_tau_ka=joint.HISTORY_TAU_KYR)
    events = context.events
    sources = {
        "saved_pairings": pairing_source,
        "ngrip_source": ngrip_source,
        "mis6_source": mis6_source,
        "mis6_source_parameters": mis6_source.parent / "parameters_and_provenance_so57_overlap.csv",
        "event_catalogue": combined_likelihood.EVENT_CATALOGUE_CSV,
        "observation_segments": combined_likelihood.OBSERVATION_SEGMENTS_CSV,
        "lr04": LR04_CSV, "co2": CO2_CSV, "precession": ORBITAL_CSV, "phase_anchors": PRECESSION_PHASE_CSV,
        "joint_analysis_code": Path(joint.__file__),
        "age_fitting_code": Path(age_sensitivity.__file__),
        "pooled_model_code": Path(combined_likelihood.__file__),
        "continuous_likelihood_code": Path(point_process.__file__),
        "runner_code": Path(__file__),
    }
    fingerprints = {name: sha256(path) for name, path in sources.items()}
    point_fit = combined_likelihood.fit_catalogue(events, context)
    draws = load_saved_pairings(events, pairing_source, ngrip_source, mis6_source, n_realizations)
    results, diagnostics = joint.fit_realizations(
        events, draws, context, show_progress=True, n_workers=workers,
    )
    summary = joint.build_summary(results, point_fit)
    summary.insert(0, "sensitivity_case", "sofular_so57_overlap")
    parameters = [
        ("sensitivity_case", "sofular_so57_overlap", "So-57 for MIS6.17--20; So-4 for MIS6.21 outside So-57 support"),
        ("model_version", combined_likelihood.MODEL_VERSION, "continuous conditional point-process likelihood"),
        ("n_realizations", n_realizations, "all requested saved rows retained"),
        ("pairing_seed", joint.PAIRING_SEED, "original pairing provenance; no new pairing performed"),
        ("pairing_scheme", "reuse saved joint and source realization IDs", "ages recovered directly from source CSVs with round-trip float parsing"),
        ("source_draw_comparison", "comparison of accepted source ensembles", "upstream rejection can change latent-draw correspondence; not a paired causal effect"),
        ("age_epoch", "BP1950", "sources already converted; no additional epoch shift"),
        ("history_tau_kyr", context.history_tau_ka, "exponentially decaying history rebuilt for each chronology"),
        ("history_coefficient_domain", "nonpositive", "beta_H <= 0 in reduced and full models"),
        ("initial_history", context.initial_history, "pre-anchor history; the conditioning event contributes one"),
        ("quadrature_order", context.quadrature_order, "Gauss-Legendre integration split at events and forcing breakpoints"),
        ("nominal_response_exposure_kyr", context.response_exposure_kyr, "gap and older conditioning portions excluded"),
        ("reduced_terms", "+".join(context.reduced_terms), "continuous log intensity plus intercept"),
        ("full_terms", "+".join(context.full_terms), "adds precession sine and cosine"),
        ("response_support", "condition on each segment's exact oldest event", "53 response events, two conditioning events; exposure changes with sampled anchors"),
        ("forcing_scaling", "fixed nominal time-weighted scaling", "held constant across age realizations"),
        ("observation_boundary_policy", "retain invalid draws with missing G", "no resampling or clipping"),
        ("robustness_denominator", "valid fits", "nominal p < 0.05 proportion is chronology sensitivity, not an empirical p value"),
        ("uncertainty_interpretation", "component sensitivity of analytical error transfer", "not a chronology posterior or full confidence interval; omitted synchronization and dating-systematic covariance remain"),
    ]
    for name, path in sources.items():
        parameters.append((name, source_label(path), "actual file used"))
        parameters.append((name + "_sha256", fingerprints[name], "content fingerprint at run start"))
        if sha256(path) != fingerprints[name]:
            raise RuntimeError(f"Input changed during the fit: {path}")
    for name, value in diagnostics.items():
        parameters.append(("diagnostic_" + name, value, "run diagnostic"))

    output_dir.mkdir(parents=True, exist_ok=True)
    # Keep full age precision; fitted summaries may be rounded for presentation.
    draws.to_csv(output_dir / "combined_event_age_realizations.csv", index=False)
    joint.compact_gain_results(results).to_csv(output_dir / "gain_realizations.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    pd.DataFrame([diagnostics]).to_csv(output_dir / "fitting_diagnostics.csv", index=False)
    pd.DataFrame(parameters, columns=["parameter", "value", "note"]).to_csv(
        output_dir / "parameters_and_provenance.csv", index=False)
    if diagnostics["n_numerical_failures"]:
        raise RuntimeError("Unresolved numerical age fits; inspect fitting_diagnostics.csv")
    print(summary.to_string(index=False), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-realizations", type=int, default=joint.N_REALIZATIONS)
    parser.add_argument("--ngrip-source", type=Path, default=joint.NGRIP_MC_INPUT)
    parser.add_argument("--mis6-source", type=Path, default=DEFAULT_MIS6_SOURCE)
    parser.add_argument("--pairing-source", type=Path, default=DEFAULT_PAIRINGS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    started = time.perf_counter()
    run(args.n_realizations, args.ngrip_source, args.mis6_source,
        args.pairing_source, args.output_dir, args.workers)
    print(f"Completed in {time.perf_counter() - started:.1f} s; outputs: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
