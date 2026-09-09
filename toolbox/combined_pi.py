"""Shared calculations for the pooled NGRIP--MIS 6 event analysis.

The catalogue contains 34 NGRIP warming starts and 21 published MIS 6
transitions.  These are two separate observation segments: the age gap between
them has no exposure, and recent-event history is reset at each segment edge.

The reduced Poisson model contains recent-event history, LR04, CO2, and a MIS 6
segment indicator.  The full model adds sine and cosine of precession phase.
Sampling resolution is deliberately not a model term.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from toolbox import event_inputs, event_process, orbital_phase, poisson
from toolbox.model_stats import nested_likelihood_metrics
from toolbox.project_config import (
    CO2_XLSX,
    LR04_XLSX,
    ORBITAL_DRIVER_SETTINGS,
    PRE_TXT,
    PROJECT_ROOT,
)


EVENT_CATALOGUE_CSV = PROJECT_ROOT / "data/curated/ngrip_mis6_warming_events.csv"
OBSERVATION_SEGMENTS_CSV = PROJECT_ROOT / "data/curated/observation_segments.csv"

EVENT_AGE_COLUMN = "event_age_kyr_bp"
SEGMENT_IDS = ("NGRIP", "MIS6")
SEGMENT_TERM = "mis6_segment"
CATALOGUE_ID = "ngrip_warming_plus_mis6"

DEFAULT_HISTORY_WINDOW_KA = 1.5
DEFAULT_BIN_WIDTH_KA = 0.2
DEFAULT_ORIGIN_FRACTION = 0.0
DEFAULT_RESPONSE_MODE = "maximal_for_history"
RESPONSE_MODES = ("common_core", "maximal_for_history")

REDUCED_TERMS = (
    event_process.HISTORY_TERM,
    *event_process.CLIMATE_TERMS,
    SEGMENT_TERM,
)
FULL_TERMS = REDUCED_TERMS + event_process.PHASE_TERMS
RESOLUTION_COVARIATE_INCLUDED = False

EVENT_COLUMNS = (
    "event_id",
    "event_label",
    EVENT_AGE_COLUMN,
    "segment_id",
    "source_record",
    "source_event_label",
    "data_source",
    "label_source",
    "timing_method",
    "source_table",
)
SEGMENT_COLUMNS = (
    "segment_id",
    "observation_start_kyr_bp",
    "observation_end_kyr_bp",
    "common_response_start_kyr_bp",
    "common_response_end_kyr_bp",
    "support_basis",
)


@dataclass(frozen=True)
class SegmentContext:
    """Grid and history indices for one independently observed time segment."""

    segment_id: str
    observation_start_kyr_bp: float
    observation_end_kyr_bp: float
    response_start_kyr_bp: float
    response_end_kyr_bp: float
    bin_edges: np.ndarray
    response_mask: np.ndarray
    history_left: np.ndarray
    history_right: np.ndarray


@dataclass(frozen=True)
class PIContext:
    """Fixed grids and predictors shared by point and Monte Carlo fits."""

    bins: pd.DataFrame
    segments: dict[str, SegmentContext]
    response_dt: np.ndarray
    history_window_ka: float
    bin_width_ka: float
    origin_fraction: float
    response_mode: str

    @property
    def response_bins(self) -> pd.DataFrame:
        """Return the two exposed intervals, excluding history buffers and gap."""

        return self.bins.loc[self.bins["in_response_interval"]].reset_index(drop=True)

    @property
    def response_exposure_kyr(self) -> float:
        """Total duration represented by the response bins."""

        return float(self.response_dt.sum())


@dataclass(frozen=True)
class CombinedPIFit:
    """Nested Poisson fits and their inspectable binned event table."""

    binned: pd.DataFrame
    response: pd.DataFrame
    reduced: poisson.FittedPoissonArrays
    full: poisson.FittedPoissonArrays
    summary: dict[str, object]


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def validate_event_catalogue(events: pd.DataFrame) -> pd.DataFrame:
    """Validate and return the frozen 55-event warming catalogue."""

    _require_columns(events, EVENT_COLUMNS, "Event catalogue")
    out = events.loc[:, EVENT_COLUMNS].copy().reset_index(drop=True)

    if len(out) != 55:
        raise ValueError(f"Expected 55 pooled warming events, found {len(out)}")
    if out["event_id"].duplicated().any():
        raise ValueError("Event IDs must be unique")
    if out.groupby("segment_id", sort=False).size().to_dict() != {
        "NGRIP": 34,
        "MIS6": 21,
    }:
        raise ValueError("Expected 34 NGRIP and 21 MIS 6 events")
    if set(out["segment_id"]) != set(SEGMENT_IDS):
        raise ValueError(f"Segments must be {SEGMENT_IDS}")

    ages = pd.to_numeric(out[EVENT_AGE_COLUMN], errors="coerce")
    if not np.isfinite(ages).all():
        raise ValueError("Every event needs a finite age in kyr BP")
    out[EVENT_AGE_COLUMN] = ages.astype(float)

    text_columns = [column for column in EVENT_COLUMNS if column != EVENT_AGE_COLUMN]
    if out[text_columns].isna().any().any():
        raise ValueError("Event provenance fields cannot be empty")
    if (
        (out[text_columns].astype(str).apply(lambda column: column.str.strip()) == "")
        .any()
        .any()
    ):
        raise ValueError("Event provenance fields cannot be blank")

    ngrip_labels = out.loc[out["segment_id"].eq("NGRIP"), "event_label"].astype(str)
    if not ngrip_labels.str.startswith("GI-").all():
        raise ValueError("The pooled catalogue may contain NGRIP warming starts only")

    for segment_id in SEGMENT_IDS:
        segment_ages = out.loc[out["segment_id"].eq(segment_id), EVENT_AGE_COLUMN]
        if not segment_ages.is_monotonic_increasing or segment_ages.duplicated().any():
            raise ValueError(f"{segment_id} event ages must be strictly increasing")
    return out


def load_event_catalogue(path: Path = EVENT_CATALOGUE_CSV) -> pd.DataFrame:
    """Load the curated pooled catalogue; ages are in kyr BP (AD 1950)."""

    return validate_event_catalogue(pd.read_csv(path))


def validate_observation_segments(segments: pd.DataFrame) -> pd.DataFrame:
    """Validate the two disjoint observation windows and common response core."""

    _require_columns(segments, SEGMENT_COLUMNS, "Observation-segment table")
    out = segments.copy().reset_index(drop=True)
    if out["segment_id"].tolist() != list(SEGMENT_IDS):
        raise ValueError(f"Observation segments must be ordered as {SEGMENT_IDS}")

    numeric_columns = [
        column for column in SEGMENT_COLUMNS if column.endswith("kyr_bp")
    ]
    out[numeric_columns] = out[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(out[numeric_columns].to_numpy(float)).all():
        raise ValueError("Observation limits must be finite kyr BP values")
    missing_basis = out["support_basis"].isna() | (
        out["support_basis"].astype(str).str.strip() == ""
    )
    if missing_basis.any():
        raise ValueError("Each observation segment needs a support basis")

    for row in out.itertuples(index=False):
        limits = (
            row.observation_start_kyr_bp,
            row.common_response_start_kyr_bp,
            row.common_response_end_kyr_bp,
            row.observation_end_kyr_bp,
        )
        if not limits[0] <= limits[1] < limits[2] <= limits[3]:
            raise ValueError(f"Invalid observation limits for {row.segment_id}")

    expected_core = {
        "NGRIP": (12.0, 118.0),
        "MIS6": (132.5, 199.5),
    }
    for row in out.itertuples(index=False):
        observed = (row.common_response_start_kyr_bp, row.common_response_end_kyr_bp)
        if not np.allclose(observed, expected_core[row.segment_id], atol=1e-10):
            raise ValueError(f"Unexpected common response core for {row.segment_id}")

    previous_end = out["observation_end_kyr_bp"].iloc[0]
    next_start = out["observation_start_kyr_bp"].iloc[1]
    if previous_end >= next_start:
        raise ValueError("NGRIP and MIS 6 observation segments must not overlap")
    return out


def load_observation_segments(path: Path = OBSERVATION_SEGMENTS_CSV) -> pd.DataFrame:
    """Load observation limits and their documented support."""

    return validate_observation_segments(pd.read_csv(path))


def _response_limits(
    row: pd.Series, history_window_ka: float, response_mode: str
) -> tuple[float, float]:
    if response_mode == "common_core":
        return (
            float(row["common_response_start_kyr_bp"]),
            float(row["common_response_end_kyr_bp"]),
        )
    return (
        float(row["observation_start_kyr_bp"]),
        float(row["observation_end_kyr_bp"] - history_window_ka),
    )


def _shifted_bin_edges(
    start: float,
    end: float,
    bin_width_ka: float,
    origin_fraction: float,
    *,
    fixed_edges: tuple[float, ...] = (),
) -> np.ndarray:
    """Build a shifted grid while retaining exact interval boundaries."""

    shift = origin_fraction * bin_width_ka
    first_internal = start + (bin_width_ka if np.isclose(shift, 0.0) else shift)
    regular = np.arange(first_internal, end, bin_width_ka)
    candidates = np.concatenate(([start, end], regular, np.asarray(fixed_edges, float)))
    inside = candidates[(candidates >= start) & (candidates <= end)]
    edges = np.unique(np.round(inside, 10))
    if len(edges) < 2 or np.any(np.diff(edges) <= 0.0):
        raise RuntimeError("Could not construct a positive-duration bin grid")
    return edges


def _build_segment_grid(
    row: pd.Series,
    *,
    response_start: float,
    response_end: float,
    history_window_ka: float,
    bin_width_ka: float,
    origin_fraction: float,
    lr04_path: Path,
    co2_path: Path,
    precession_path: Path,
    project_root: Path,
) -> tuple[pd.DataFrame, SegmentContext]:
    segment_id = str(row["segment_id"])
    observation_start = float(row["observation_start_kyr_bp"])
    observation_end = float(row["observation_end_kyr_bp"])
    edges = _shifted_bin_edges(
        observation_start,
        observation_end,
        bin_width_ka,
        origin_fraction,
        fixed_edges=(response_start, response_end),
    )
    centers = 0.5 * (edges[:-1] + edges[1:])

    _, lr04_info = event_inputs.load_lr04(centers, lr04_path, project_root=project_root)
    _, co2_info = event_inputs.load_co2(centers, co2_path, project_root=project_root)
    phases, _ = event_inputs.build_precession_phase(
        centers, precession_path, project_root=project_root
    )

    response_mask = (edges[:-1] >= response_start - 1e-10) & (
        edges[1:] <= response_end + 1e-10
    )
    history_left = np.arange(len(centers), dtype=int) + 1
    history_right = np.searchsorted(centers, centers + history_window_ka, side="right")
    history_complete = centers + history_window_ka <= observation_end + 1e-10

    bins = pd.DataFrame(
        {
            "segment_id": segment_id,
            "bin_start_kyr_bp": edges[:-1],
            "bin_end_kyr_bp": edges[1:],
            "bin_center_kyr_bp": centers,
            "dt_kyr": np.diff(edges),
            "lr04": lr04_info["raw"],
            "co2": co2_info["raw"],
            SEGMENT_TERM: float(segment_id == "MIS6"),
            "in_response_interval": response_mask,
            "same_type_history_complete": history_complete,
        }
    )
    phase_columns = phases.drop(columns="age_ka").reset_index(drop=True)
    bins = pd.concat([bins, phase_columns], axis=1)

    segment = SegmentContext(
        segment_id=segment_id,
        observation_start_kyr_bp=observation_start,
        observation_end_kyr_bp=observation_end,
        response_start_kyr_bp=response_start,
        response_end_kyr_bp=response_end,
        bin_edges=edges,
        response_mask=response_mask,
        history_left=history_left,
        history_right=history_right,
    )
    return bins, segment


def build_context(
    observation_segments: pd.DataFrame | None = None,
    *,
    history_window_ka: float = DEFAULT_HISTORY_WINDOW_KA,
    bin_width_ka: float = DEFAULT_BIN_WIDTH_KA,
    origin_fraction: float = DEFAULT_ORIGIN_FRACTION,
    response_mode: str = DEFAULT_RESPONSE_MODE,
    lr04_path: Path = LR04_XLSX,
    co2_path: Path = CO2_XLSX,
    precession_path: Path = PRE_TXT,
    project_root: Path = PROJECT_ROOT,
) -> PIContext:
    """Build disjoint response grids and fixed climate/orbital predictors.

    ``common_core`` always uses 12--120 and 132.5--199.5 kyr BP.
    ``maximal_for_history`` ends each response interval one history window
    before the documented observation end.  An origin shift may create edge
    bins shorter than ``bin_width_ka``; their true duration is retained.
    """

    if not np.isfinite(history_window_ka) or history_window_ka <= 0.0:
        raise ValueError("history_window_ka must be positive")
    if not np.isfinite(bin_width_ka) or bin_width_ka <= 0.0:
        raise ValueError("bin_width_ka must be positive")
    if not 0.0 <= origin_fraction < 1.0:
        raise ValueError("origin_fraction must lie in [0, 1)")
    if response_mode not in RESPONSE_MODES:
        raise ValueError(f"response_mode must be one of {RESPONSE_MODES}")

    segment_table = (
        load_observation_segments()
        if observation_segments is None
        else validate_observation_segments(observation_segments)
    )
    frames: list[pd.DataFrame] = []
    contexts: dict[str, SegmentContext] = {}

    for _, row in segment_table.iterrows():
        response_start, response_end = _response_limits(
            row, history_window_ka, response_mode
        )
        if response_end <= response_start:
            raise ValueError(
                f"History window leaves no response interval for {row['segment_id']}"
            )
        bins, segment = _build_segment_grid(
            row,
            response_start=response_start,
            response_end=response_end,
            history_window_ka=history_window_ka,
            bin_width_ka=bin_width_ka,
            origin_fraction=origin_fraction,
            lr04_path=lr04_path,
            co2_path=co2_path,
            precession_path=precession_path,
            project_root=project_root,
        )
        frames.append(bins)
        contexts[segment.segment_id] = segment

    all_bins = pd.concat(frames, ignore_index=True)
    response = all_bins["in_response_interval"].astype(bool)
    if not all_bins.loc[response, "same_type_history_complete"].all():
        raise RuntimeError("At least one response bin lacks a complete event history")

    # One common climate scale is used across both exposed intervals.
    for forcing in ("lr04", "co2"):
        _, mean, _, _, value_range = event_inputs.scale_to_zero_mean_range_one(
            all_bins.loc[response, forcing].to_numpy(float)
        )
        all_bins[f"{forcing}_scaled"] = (all_bins[forcing] - mean) / value_range

    response_dt = all_bins.loc[response, "dt_kyr"].to_numpy(float)
    return PIContext(
        bins=all_bins,
        segments=contexts,
        response_dt=response_dt,
        history_window_ka=float(history_window_ka),
        bin_width_ka=float(bin_width_ka),
        origin_fraction=float(origin_fraction),
        response_mode=response_mode,
    )


def count_events(
    events: pd.DataFrame,
    context: PIContext,
    *,
    age_column: str = EVENT_AGE_COLUMN,
) -> dict[str, np.ndarray]:
    """Count event ages on each observation grid without joining the gap."""

    _require_columns(events, ("segment_id", age_column), "Event-age table")
    unknown = set(events["segment_id"]).difference(context.segments)
    if unknown:
        raise ValueError(f"Unknown event segments: {sorted(unknown)}")

    counts: dict[str, np.ndarray] = {}
    for segment_id in SEGMENT_IDS:
        segment = context.segments[segment_id]
        ages = pd.to_numeric(
            events.loc[events["segment_id"].eq(segment_id), age_column],
            errors="coerce",
        ).to_numpy(float)
        if not np.isfinite(ages).all():
            raise ValueError(f"{segment_id} event ages must be finite")
        if np.any(ages < segment.bin_edges[0]) or np.any(ages > segment.bin_edges[-1]):
            raise ValueError(
                f"An event lies outside the {segment_id} observation window"
            )
        segment_counts, _ = np.histogram(ages, bins=segment.bin_edges)
        if int(segment_counts.sum()) != len(ages):
            raise RuntimeError(f"Failed to bin every {segment_id} event")
        counts[segment_id] = segment_counts.astype(float)
    return counts


def history_from_counts(counts: np.ndarray, segment: SegmentContext) -> np.ndarray:
    """Count older event bins within the window, excluding the current bin.

    Events are represented at the center of their assigned bin.  The history
    interval is therefore ``(current center, current center + window]`` on the
    binned grid, not an exact calculation from the unbinned event ages.  Point,
    Monte Carlo, and bootstrap fits must all use this same definition.
    """

    values = np.asarray(counts, dtype=float)
    if values.shape != (len(segment.bin_edges) - 1,):
        raise ValueError(f"Count vector has the wrong length for {segment.segment_id}")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("Event counts must be finite and non-negative")
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    return cumulative[segment.history_right] - cumulative[segment.history_left]


def bin_event_counts(counts: dict[str, np.ndarray], context: PIContext) -> pd.DataFrame:
    """Attach event counts and segment-local histories to the fixed grids."""

    frames: list[pd.DataFrame] = []
    for segment_id in SEGMENT_IDS:
        if segment_id not in counts:
            raise ValueError(f"Missing count vector for {segment_id}")
        frame = context.bins.loc[context.bins["segment_id"].eq(segment_id)].copy()
        segment = context.segments[segment_id]
        values = np.asarray(counts[segment_id], dtype=float)
        frame["event_count"] = values
        frame[event_process.HISTORY_TERM] = history_from_counts(values, segment)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def bin_catalogue(
    events: pd.DataFrame,
    context: PIContext,
    *,
    age_column: str = EVENT_AGE_COLUMN,
) -> pd.DataFrame:
    """Return the diagnostic count/history table for one event catalogue."""

    return bin_event_counts(
        count_events(events, context, age_column=age_column), context
    )


def fit_response_terms(
    response: pd.DataFrame,
    terms: tuple[str, ...],
) -> poisson.FittedPoissonArrays:
    """Fit one named predictor set to an already prepared response table."""

    required = ("event_count", "dt_kyr", *terms)
    _require_columns(response, required, "PI response table")
    return poisson.fit_binned_poisson_arrays(
        response.loc[:, terms].to_numpy(float),
        response["event_count"].to_numpy(float),
        response["dt_kyr"].to_numpy(float),
    )


def _fit_summary(
    binned: pd.DataFrame,
    reduced: poisson.FittedPoissonArrays,
    full: poisson.FittedPoissonArrays,
    context: PIContext,
) -> dict[str, object]:
    response = binned.loc[binned["in_response_interval"]].reset_index(drop=True)
    metrics = nested_likelihood_metrics(
        loglik_full=full.log_likelihood,
        loglik_reduced=reduced.log_likelihood,
        df=len(full.beta) - len(reduced.beta),
        n_bins=len(response),
        n_events=int(response["event_count"].sum()),
        aicc_full=full.aicc,
        aicc_reduced=reduced.aicc,
    )
    beta_map = dict(zip(FULL_TERMS, full.beta[1:]))
    beta_sin = float(beta_map["pre_phase_sin"])
    beta_cos = float(beta_map["pre_phase_cos"])
    amplitude = float(np.hypot(beta_sin, beta_cos))
    n_eta_clipped = sum(
        (
            reduced.n_eta_clipped_low,
            reduced.n_eta_clipped_high,
            full.n_eta_clipped_low,
            full.n_eta_clipped_high,
        )
    )
    return {
        "catalogue_id": CATALOGUE_ID,
        "n_source_events": int(binned["event_count"].sum()),
        "n_predictive_events": int(response["event_count"].sum()),
        "n_predictive_bins": int(len(response)),
        "response_exposure_kyr": context.response_exposure_kyr,
        "history_window_kyr": context.history_window_ka,
        "bin_width_kyr": context.bin_width_ka,
        "origin_fraction": context.origin_fraction,
        "response_mode": context.response_mode,
        "loglik_reduced": metrics["loglik_reduced"],
        "loglik_full": metrics["loglik_full"],
        "ll_gain_nats": metrics["ll_gain_nats"],
        "LR_statistic": metrics["LR_statistic"],
        "nominal_LR_p": metrics["LR_p_value"],
        "info_bits_per_event": metrics["info_bits_per_event"],
        "delta_AICc_full_minus_reduced": metrics["delta_AICc_full_minus_reduced"],
        "pre_phase_preferred_deg": float(
            np.degrees(np.mod(np.arctan2(beta_sin, beta_cos), 2.0 * np.pi))
        ),
        "pre_phase_rate_ratio_max_vs_min": float(np.exp(2.0 * amplitude)),
        "mis6_vs_ngrip_rate_ratio_full": float(np.exp(beta_map[SEGMENT_TERM])),
        "all_models_converged": bool(reduced.converged and full.converged),
        "likelihood_nesting_ok": bool(metrics["ll_gain_nats"] >= -1e-8),
        "eta_clipping_used": bool(n_eta_clipped > 0),
    }


def fit_event_counts(
    counts: dict[str, np.ndarray], context: PIContext
) -> CombinedPIFit:
    """Fit the reduced and full models to already binned segment counts."""

    binned = bin_event_counts(counts, context)
    response = binned.loc[binned["in_response_interval"]].reset_index(drop=True)
    reduced = fit_response_terms(response, REDUCED_TERMS)
    full = fit_response_terms(response, FULL_TERMS)
    summary = _fit_summary(binned, reduced, full, context)
    return CombinedPIFit(binned, response, reduced, full, summary)


def fit_catalogue(
    events: pd.DataFrame,
    context: PIContext,
    *,
    age_column: str = EVENT_AGE_COLUMN,
) -> CombinedPIFit:
    """Bin and fit one point or Monte Carlo realization of the catalogue."""

    return fit_event_counts(
        count_events(events, context, age_column=age_column), context
    )


def fit_point_catalogue(
    events: pd.DataFrame | None = None,
    context: PIContext | None = None,
) -> CombinedPIFit:
    """Fit the curated point ages with the default context when omitted."""

    point_events = (
        load_event_catalogue() if events is None else validate_event_catalogue(events)
    )
    point_context = build_context() if context is None else context
    return fit_catalogue(point_events, point_context)


def coefficient_table(fit: CombinedPIFit) -> pd.DataFrame:
    """Return model coefficients and their one-unit multiplicative effects."""

    rows: list[dict[str, object]] = []
    for model_id, model, terms in (
        ("reduced", fit.reduced, REDUCED_TERMS),
        ("full", fit.full, FULL_TERMS),
    ):
        for term, beta in zip(("intercept", *terms), model.beta):
            rows.append(
                {
                    "model_id": model_id,
                    "term": term,
                    "beta": float(beta),
                    "rate_ratio_per_unit": float(np.exp(beta)),
                }
            )
    return pd.DataFrame(rows)


def sample_event_phases(
    events: pd.DataFrame,
    *,
    age_column: str = EVENT_AGE_COLUMN,
) -> pd.DataFrame:
    """Evaluate BP1950-corrected La2004 precession phase at event ages."""

    _require_columns(events, ("event_id", "event_label", age_column), "Event table")
    ages = pd.to_numeric(events[age_column], errors="coerce").to_numpy(float)
    if not np.isfinite(ages).all():
        raise ValueError("Event ages must be finite before phase interpolation")

    settings = dict(ORBITAL_DRIVER_SETTINGS["pre"])
    phase_product = orbital_phase.build_phase_series("pre", settings)
    phase = orbital_phase.evaluate_phase_at_ages(ages, phase_product.extrema, True)
    precession_index = event_inputs.interpolate_checked(
        ages,
        phase_product.series["age_ka"].to_numpy(float),
        phase_product.series["value"].to_numpy(float),
        context="precession index at pooled events",
    )
    out = events.reset_index(drop=True).copy()
    out["precession_index"] = precession_index
    for source, target in (
        ("phase_unwrapped_rad", "pre_phase_unwrapped_rad"),
        ("phase_rad", "pre_phase_rad"),
        ("phase_deg", "pre_phase_deg"),
        ("phase_fraction", "pre_phase_fraction"),
        ("phase_extrapolated", "pre_phase_extrapolated"),
    ):
        out[target] = phase[source].to_numpy()
    return out


def phase_rate_multiplier(
    phase_deg: np.ndarray, beta_sin: float, beta_cos: float
) -> np.ndarray:
    """Return the fitted multiplicative contribution of precession phase."""

    phase_rad = np.deg2rad(np.asarray(phase_deg, dtype=float))
    return np.exp(beta_sin * np.sin(phase_rad) + beta_cos * np.cos(phase_rad))
