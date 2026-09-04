"""Shared event checks and conditional-PI fit for the MIS 6 sequence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from toolbox import event_inputs, event_process, poisson
from toolbox.model_stats import nested_likelihood_metrics
from toolbox.project_config import CO2_XLSX, LR04_XLSX, PRE_TXT, PROJECT_ROOT


EVENTS_CSV = (
    PROJECT_ROOT
    / "data/processed/MIS6_composite_event_record/mis6_composite_event_record.csv"
)

# The interval is rounded inward from continuous MF/Huagapo coverage
# (~132.411--196.584 Kyr BP), so unobserved time is not counted as exposure.
ANALYSIS_START_KA = 132.5
ANALYSIS_END_KA = 196.5
BIN_WIDTH_KA = 0.2
HISTORY_WINDOW_KA = 5.0
N_EVENTS = 21

DATASET_ID = "mis6_warming"
EVENT_TYPE = "warming"
EVENT_LABEL = "MIS 6 composite warming events"
EVENT_COLOR = "#C43C39"

# These terms match the NGRIP experiment. Proxy resolution is not a covariate.
REDUCED_TERMS = (event_process.HISTORY_TERM, *event_process.CLIMATE_TERMS)
FULL_TERMS = REDUCED_TERMS + event_process.PHASE_TERMS
REDUCED_MODEL_ID = "history_climate"
FULL_MODEL_ID = "history_climate_precession"
COMPARISON_ID = "precession_after_history_climate"
RESOLUTION_COVARIATE_INCLUDED = False


def load_events(path: Path = EVENTS_CSV) -> pd.DataFrame:
    """Load the 21 ordered event ages used by the MIS 6 analyses."""
    events = pd.read_csv(path)
    required = {
        "composite_event_id",
        "composite_event_label",
        "event_age_ka_bp",
        "event_age_status",
        "source_record",
    }
    if missing := required.difference(events.columns):
        raise ValueError(f"Event catalogue is missing columns: {sorted(missing)}")

    events = events.copy()
    events["event_age_ka_bp"] = pd.to_numeric(
        events["event_age_ka_bp"], errors="coerce"
    )
    if len(events) != N_EVENTS:
        raise ValueError(f"Expected {N_EVENTS} MIS 6 events, found {len(events)}")
    if events["event_age_ka_bp"].isna().any():
        raise ValueError("Every MIS 6 event must have a finite age")
    for column in ("composite_event_id", "composite_event_label", "event_age_ka_bp"):
        if events[column].duplicated().any():
            raise ValueError(f"MIS 6 catalogue contains duplicate {column}")
    if (
        not events["event_age_ka_bp"]
        .between(ANALYSIS_START_KA, ANALYSIS_END_KA, inclusive="both")
        .all()
    ):
        raise ValueError("At least one MIS 6 event lies outside the analysis interval")

    events = events.sort_values("event_age_ka_bp").reset_index(drop=True)
    events["event_type"] = EVENT_TYPE
    return events


def build_event_dataset(events: pd.DataFrame) -> event_inputs.EventDataset:
    """Represent the composite ages as one warming-event catalogue."""
    return event_inputs.EventDataset(
        dataset_id=DATASET_ID,
        label=EVENT_LABEL,
        color=EVENT_COLOR,
        ages_ka=events["event_age_ka_bp"].to_numpy(float),
        source=str(EVENTS_CSV.relative_to(PROJECT_ROOT)),
    )


def run_predictive_information(events: pd.DataFrame) -> dict[str, object]:
    """Fit the reduced and phase-augmented Poisson models used for MIS 6."""
    if RESOLUTION_COVARIATE_INCLUDED or any(
        "resolution" in term.lower() for term in FULL_TERMS
    ):
        raise RuntimeError("The MIS 6 PI model must not include proxy resolution")

    binned, scale_summary, phase_extrema = event_inputs.build_binned_inputs(
        [build_event_dataset(events)],
        analysis_start_ka=ANALYSIS_START_KA,
        analysis_end_ka=ANALYSIS_END_KA,
        bin_width_ka=BIN_WIDTH_KA,
        lr04_path=LR04_XLSX,
        co2_path=CO2_XLSX,
        precession_path=PRE_TXT,
        project_root=PROJECT_ROOT,
    )
    binned = event_process.add_same_type_history(binned, HISTORY_WINDOW_KA)
    fit_frame = event_process.model_frame(binned)

    reduced = poisson.fit_poisson_model(
        fit_frame,
        REDUCED_MODEL_ID,
        REDUCED_TERMS,
        "Catalogue history + LR04 + CO2",
    )
    full = poisson.fit_poisson_model(
        fit_frame,
        FULL_MODEL_ID,
        FULL_TERMS,
        "Catalogue history + LR04 + CO2 + precession phase",
    )
    models = [reduced, full]
    model_summary = poisson.build_model_summary(models, fit_frame)
    coefficients = poisson.build_coefficient_table(models)
    metrics = nested_likelihood_metrics(
        loglik_full=full.log_likelihood,
        loglik_reduced=reduced.log_likelihood,
        df=len(full.beta) - len(reduced.beta),
        n_bins=len(fit_frame),
        n_events=int(fit_frame["event_count"].sum()),
        aicc_full=full.aicc,
        aicc_reduced=reduced.aicc,
    )
    likelihood_tests = pd.DataFrame(
        [
            {
                "dataset_id": DATASET_ID,
                "dataset_label": EVENT_LABEL,
                "comparison_id": COMPARISON_ID,
                "question": (
                    "Does precession phase add information after history, "
                    "LR04, and CO2?"
                ),
                "reduced_model_id": REDUCED_MODEL_ID,
                "full_model_id": FULL_MODEL_ID,
                **metrics,
                "p_value_method": "nominal asymptotic chi-square LRT",
                "reject_LR_at_0p05": metrics["LR_p_value"] < 0.05,
                "likelihood_nesting_ok": metrics["ll_gain_nats"] >= -1e-8,
                "resolution_covariate_included": False,
                "event_age_uncertainty_propagated": False,
            }
        ]
    )

    if not model_summary["converged"].all():
        raise RuntimeError("At least one MIS 6 PI model did not converge")
    if not likelihood_tests["likelihood_nesting_ok"].all():
        raise RuntimeError("The full-model likelihood is below the reduced model")
    if model_summary[["n_eta_clipped_low", "n_eta_clipped_high"]].gt(0).any().any():
        raise RuntimeError("At least one MIS 6 PI model used eta clipping")

    return {
        "binned": binned,
        "fit_frame": fit_frame,
        "scale_summary": scale_summary,
        "phase_extrema": phase_extrema,
        "model_summary": model_summary,
        "coefficients": coefficients,
        "likelihood_tests": likelihood_tests,
    }


def phase_rate_multiplier(
    phase_deg: np.ndarray, beta_sin: float, beta_cos: float
) -> np.ndarray:
    """Return the fitted multiplicative contribution of precession phase."""
    phase_rad = np.deg2rad(np.asarray(phase_deg, dtype=float))
    return np.exp(beta_sin * np.sin(phase_rad) + beta_cos * np.cos(phase_rad))
