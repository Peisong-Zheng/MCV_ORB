#!/usr/bin/env python3
"""Refit the saved So-57 overlap pairings with the continuous event likelihood.

The previous run saved source realization IDs but no joint age table. Ages are
recovered by those IDs, without drawing new pairings or chronologies. The
reconstructed wide table is retained for exact checks on subsequent runs.
"""

from __future__ import annotations

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

from toolbox import event_model, model_stats, age_sensitivity
from toolbox.point_process import fit_point_process
from toolbox.project_config import (
    PROJECT_ROOT, EVENT_CATALOGUE_CSV, OBSERVATION_SEGMENTS_CSV, MODEL_VERSION,
    LR04_CSV, CO2_CSV, ORBITAL_CSV, PRECESSION_PHASE_CSV,
)
from toolbox import point_process

MIS6_MC_INPUT = (
    PROJECT_ROOT / "MIS6/data/processed/MIS6_event_age_uncertainty"
    / "mis6_event_age_realizations_so57_overlap.csv"
)
OUT_DATA_DIR = PROJECT_ROOT / "data/processed/NGRIP_MIS6_event_uncertainty_sensitivity/sofular_so57_overlap"
NGRIP_MC_INPUT = PROJECT_ROOT / "NGRIP/data/processed/ngrip_event_age_uncertainty/ngrip_event_age_realizations.csv"
N_REALIZATIONS = 10000
N_WORKERS = 1
PAIRING_SEED = 20260906
HISTORY_TAU_KYR = 1.5
HISTORY_TERM = "same_type_exponential_history"
REDUCED_TERMS = ["intercept", HISTORY_TERM, "lr04_scaled", "co2_scaled", "mis6_segment"]
FULL_TERMS = REDUCED_TERMS + ["pre_phase_sin", "pre_phase_cos"]
CATALOGUE_ID = "ngrip_warming_plus_mis6"

PAIRINGS_CSV = (PROJECT_ROOT / "data/processed/NGRIP_MIS6_event_uncertainty_sensitivity"
                / "sofular_so57_overlap/gain_realizations.csv")
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


def restore_pairings(events, pairs, ngrip_table, mis6_table, n_realizations):
    """Recover the original 55 ages from the three saved realization IDs."""
    if not 1 <= n_realizations <= len(pairs):
        raise ValueError("Requested count must lie within the saved pairings")
    pairs = pairs.loc[:n_realizations - 1, ID_COLUMNS].copy()
    if pairs.isna().any().any() or any(pairs[column].duplicated().any() for column in ID_COLUMNS):
        raise ValueError("Saved pairing IDs must be complete and unique")

    ages = np.empty((len(pairs), len(events)))
    columns = {
        "NGRIP": [f"age_ka_bp__{label}" for label in events.loc[events.segment_id.eq("NGRIP"), "event_label"]],
        "MIS6": [f"{event_id.removeprefix('MIS6:')}_age_ka_bp"
                 for event_id in events.loc[events.segment_id.eq("MIS6"), "event_id"]],
    }
    for segment, source, id_column in (
        ("NGRIP", ngrip_table, "ngrip_realization_id"),
        ("MIS6", mis6_table, "mis6_realization_id"),
    ):
        source_ages = source[columns[segment]].to_numpy(float)
        if (source.realization_id.isna().any() or not source.realization_id.is_unique
                or not np.isfinite(source_ages).all() or not np.all(np.diff(source_ages, axis=1) > 0)):
            raise ValueError(f"{segment} source IDs and ordered ages must be valid")
        selected = source.set_index("realization_id").loc[pairs[id_column].astype(str), columns[segment]]
        ages[:, events.segment_id.eq(segment).to_numpy()] = selected.to_numpy(float)
    if not np.all(np.diff(ages, axis=1) > 0):
        raise ValueError("Recovered event ages must retain the original rank")
    return pd.concat([pairs, pd.DataFrame(ages, columns=[f"age_kyr_bp__{event_id}" for event_id in events.event_id])], axis=1)


def run(n_realizations: int, ngrip_source: Path, mis6_source: Path,
        pairing_source: Path, output_dir: Path, workers: int = 1) -> pd.DataFrame:
    events = pd.read_csv(EVENT_CATALOGUE_CSV)
    observations = pd.read_csv(OBSERVATION_SEGMENTS_CSV)
    lr04 = pd.read_csv(LR04_CSV, float_precision="round_trip")
    co2 = pd.read_csv(CO2_CSV, float_precision="round_trip")
    orbital = pd.read_csv(ORBITAL_CSV, float_precision="round_trip")
    anchors = pd.read_csv(PRECESSION_PHASE_CSV, float_precision="round_trip")
    forcings = {"lr04": (lr04.age_kyr_bp.to_numpy(), lr04.lr04.to_numpy()),
                "co2": (co2.age_kyr_bp.to_numpy(), co2.co2_ppm.to_numpy()),
                "precession_index": (orbital.age_kyr_bp.to_numpy(), orbital.precession_index.to_numpy())}
    phase_anchors = (anchors.age_kyr_bp.to_numpy(), anchors.phase_unwrapped_rad.to_numpy())
    windows = event_model.response_windows(events, observations)
    scaling = event_model.scale_forcing_v2({name: forcings[name] for name in ("lr04", "co2")}, windows)
    event_x, integral_x = event_model.build_likelihood_tables(
        events, windows, forcings, phase_anchors, scaling, tau=HISTORY_TAU_KYR,
    )
    for frame in (event_x, integral_x):
        frame["mis6_segment"] = frame.segment_id.eq("MIS6").astype(float)
    reduced = fit_point_process(event_x[REDUCED_TERMS], integral_x[REDUCED_TERMS],
                                integral_x.weight, REDUCED_TERMS, nonpositive_terms=(HISTORY_TERM,))
    full = fit_point_process(event_x[FULL_TERMS], integral_x[FULL_TERMS],
                             integral_x.weight, FULL_TERMS, nonpositive_terms=(HISTORY_TERM,),
                             start_beta=np.r_[reduced.beta, 0., 0.])
    point = model_stats.fit_summary(reduced, full, event_x, windows, n_source_events=len(events),
                                    catalogue_id=CATALOGUE_ID, tau=HISTORY_TAU_KYR)
    sources = {
        "saved_pairings": pairing_source,
        "ngrip_source": ngrip_source,
        "mis6_source": mis6_source,
        "mis6_source_parameters": mis6_source.parent / "parameters_and_provenance_so57_overlap.csv",
        "event_catalogue": EVENT_CATALOGUE_CSV,
        "observation_segments": OBSERVATION_SEGMENTS_CSV,
        "lr04": LR04_CSV, "co2": CO2_CSV, "precession": ORBITAL_CSV, "phase_anchors": PRECESSION_PHASE_CSV,
        "age_fitting_code": Path(age_sensitivity.__file__),
        "event_model_code": Path(event_model.__file__),
        "continuous_likelihood_code": Path(point_process.__file__),
        "runner_code": Path(__file__),
    }
    fingerprints = {name: sha256(path) for name, path in sources.items()}
    pairs = pd.read_csv(pairing_source, usecols=ID_COLUMNS)
    ngrip_table = pd.read_csv(ngrip_source, dtype={"realization_id": str}, float_precision="round_trip")
    mis6_table = pd.read_csv(mis6_source, dtype={"realization_id": str}, float_precision="round_trip")
    draws = restore_pairings(events, pairs, ngrip_table, mis6_table, n_realizations)
    results, diagnostics = age_sensitivity.fit_realizations(
            events, observations, forcings, phase_anchors, scaling, REDUCED_TERMS, FULL_TERMS,
            draws, [f"age_kyr_bp__{event_id}" for event_id in events.event_id],
            catalogue_id=CATALOGUE_ID, tau=HISTORY_TAU_KYR,
            show_progress=True, n_workers=workers)
    summary = age_sensitivity.summarize(results, point)
    summary.insert(0, "sensitivity_case", "sofular_so57_overlap")
    parameters = [
        ("sensitivity_case", "sofular_so57_overlap", "So-57 for MIS6.17--20; So-4 for MIS6.21 outside So-57 support"),
        ("model_version", MODEL_VERSION, "continuous conditional point-process likelihood"),
        ("n_realizations", n_realizations, "all requested saved rows retained"),
        ("pairing_seed", PAIRING_SEED, "original pairing provenance; no new pairing performed"),
        ("pairing_scheme", "reuse saved joint and source realization IDs", "ages recovered directly from source CSVs with round-trip float parsing"),
        ("source_draw_comparison", "comparison of accepted source ensembles", "upstream rejection can change latent-draw correspondence; not a paired causal effect"),
        ("age_epoch", "BP1950", "sources already converted; no additional epoch shift"),
        ("history_tau_kyr", HISTORY_TAU_KYR, "exponentially decaying history rebuilt for each chronology"),
        ("history_coefficient_domain", "nonpositive", "beta_H <= 0 in reduced and full models"),
        ("initial_history", 0.0, "pre-anchor history; the conditioning event contributes one"),
        ("quadrature_order", 4, "Gauss-Legendre integration split at events and forcing breakpoints"),
        ("nominal_response_exposure_kyr", point["response_exposure_kyr"], "gap and older conditioning portions excluded"),
        ("reduced_terms", "+".join(REDUCED_TERMS[1:]), "continuous log intensity plus intercept"),
        ("full_terms", "+".join(FULL_TERMS[1:]), "adds precession sine and cosine"),
        ("response_support", "condition on each segment's exact oldest event", "53 response events, two conditioning events; exposure changes with sampled anchors"),
        ("forcing_scaling", "fixed nominal time-weighted scaling", "held constant across age realizations"),
        ("observation_boundary_policy", "retain invalid draws with missing G", "no resampling or clipping"),
        ("robustness_denominator", "valid fits", "nominal p < 0.05 proportion is chronology sensitivity, not an empirical p value"),
        ("uncertainty_interpretation", "component sensitivity of analytical error transfer", "not a chronology posterior or full confidence interval; omitted synchronization and dating-systematic covariance remain"),
    ]
    for name, path in sources.items():
        parameters.append((name, source_label(path), "actual file used"))
        parameters.append((name + "_sha256", fingerprints[name], "content fingerprint at run start"))
    for name, value in diagnostics.items():
        parameters.append(("diagnostic_" + name, value, "run diagnostic"))

    output_dir.mkdir(parents=True, exist_ok=True)
    # Keep full age precision; fitted summaries may be rounded for presentation.
    draws.to_csv(output_dir / "combined_event_age_realizations.csv", index=False)
    age_sensitivity.compact_results(results).to_csv(output_dir / "gain_realizations.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    pd.DataFrame([diagnostics]).to_csv(output_dir / "fitting_diagnostics.csv", index=False)
    pd.DataFrame(parameters, columns=["parameter", "value", "note"]).to_csv(
        output_dir / "parameters_and_provenance.csv", index=False)
    if diagnostics["n_numerical_failures"]:
        raise RuntimeError("Unresolved numerical age fits; inspect fitting_diagnostics.csv")
    print(summary.to_string(index=False), flush=True)
    return summary


def main() -> None:
    started = time.perf_counter()
    run(N_REALIZATIONS, NGRIP_MC_INPUT, MIS6_MC_INPUT,
        PAIRINGS_CSV, OUT_DATA_DIR, N_WORKERS)
    print(f"Completed in {time.perf_counter() - started:.1f} s; outputs: {OUT_DATA_DIR}", flush=True)


if __name__ == "__main__":
    main()
