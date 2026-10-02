"""Event covariates and continuous integration on explicit BP1950 windows.

Input tables are read by each analysis. This module computes the quantities
used by the likelihood; it does not choose model columns or run the optimizer.
"""
import warnings

import numpy as np
import pandas as pd

from toolbox.point_process import exponential_history, gauss_legendre_intervals


def response_windows(events, observations):
    """Condition each independent record on its oldest observed event."""
    if observations.segment_id.duplicated().any():
        raise ValueError("Observation segment IDs must be unique")
    if set(events.segment_id) != set(observations.segment_id):
        raise ValueError("Every observation segment must have conditioning events")
    rows = []
    for row in observations.itertuples(index=False):
        ages = events.loc[events.segment_id.eq(row.segment_id), "event_age_kyr_bp"].to_numpy(float)
        if not np.isfinite(ages).all() or len(np.unique(ages)) != len(ages):
            raise ValueError("Event ages must be finite and unique within each segment")
        young = float(row.observation_start_kyr_bp)
        old = float(row.observation_end_kyr_bp)
        if not np.isfinite([young, old]).all() or young >= old:
            raise ValueError("Observation support must be finite and positive")
        if np.any(ages < young) or np.any(ages > old):
            raise ValueError(f"An event lies outside the {row.segment_id} observation window")
        anchor = float(ages.max())
        if anchor <= young:
            raise ValueError("No response exposure after the conditioning event")
        rows.append(dict(
            segment_id=row.segment_id,
            observation_start_kyr_bp=young, observation_end_kyr_bp=old,
            response_start_kyr_bp=young, response_end_kyr_bp=anchor,
            anchor_age_kyr_bp=anchor,
        ))
    return pd.DataFrame(rows)


def scale_forcing(source, windows):
    """Time-weighted mean and range on the nominal response windows."""
    source_age, source_value = source
    exposure_kyr = 0.0
    forcing_integral = 0.0
    extrema = []
    for window in windows.itertuples(index=False):
        young, old = window.response_start_kyr_bp, window.response_end_kyr_bp
        inside = (source_age > young) & (source_age < old)
        knots = np.r_[young, source_age[inside], old]
        values = interpolate_checked(knots, source_age, source_value,
                                     context="forcing response support")
        forcing_integral += np.trapezoid(values, knots)
        exposure_kyr += old - young
        extrema.extend([values.min(), values.max()])
    minimum, maximum = float(min(extrema)), float(max(extrema))
    return dict(mean=forcing_integral / exposure_kyr, min=minimum,
                max=maximum, range=maximum - minimum or 1.0)


def nominal_scaling(forcings, windows):
    """Return scaling for the explicitly supplied climate predictors."""
    return pd.DataFrame([
        dict(forcing_id=name, **scale_forcing(source, windows))
        for name, source in forcings.items()
    ]).set_index("forcing_id")


def integration_breakpoints(window, forcings, phase_anchors, event_ages, tau):
    """Split at source knots, phase anchors and event-history boundaries."""
    young, old = window.response_start_kyr_bp, window.response_end_kyr_bp
    events = np.asarray(event_ages, float)
    # Retain the existing integration partition, including rectangular exits,
    # while changing the analysis interface. No quadrature approximation changes.
    candidates = np.concatenate([
        np.array([young, old]), *[source[0] for source in forcings.values()],
        phase_anchors[0], events, events - tau,
    ])
    return np.unique(candidates[(candidates >= young) & (candidates <= old)])


def evaluate_features(ages, event_ages, window, forcings, phase_anchors, scaling,
                      *, tau=1.5, initial_history=0.0, history_variants=False):
    """Evaluate climate, phase and strictly earlier-event history at BP ages."""
    ages = np.asarray(ages, float)
    elapsed_kyr = window.anchor_age_kyr_bp - ages
    frame = dict(age_kyr_bp=ages, elapsed_kyr=elapsed_kyr,
                 segment_id=np.repeat(window.segment_id, len(ages)),
                 intercept=np.ones(len(ages)))
    for name, (source_age, source_value) in forcings.items():
        values = (interpolate_checked(ages, source_age, source_value, context=name)
                  if len(ages) else np.array([]))
        frame[name] = values
        if name in scaling.index:
            scale = scaling.loc[name]
            frame[name + "_scaled"] = (values - scale["mean"]) / scale["range"]
    phase, extrapolated = interpolate_unwrapped_phase(ages, *phase_anchors)
    frame.update(
        pre_phase_unwrapped_rad=phase,
        pre_phase_rad=np.mod(phase, 2 * np.pi),
        pre_phase_deg=np.mod(np.degrees(phase), 360),
        pre_phase_sin=np.sin(phase), pre_phase_cos=np.cos(phase),
        pre_phase_extrapolated=extrapolated,
    )
    event_ages = np.sort(np.asarray(event_ages, float))[::-1]
    frame["same_type_exponential_history"] = exponential_history(
        ages, event_ages, tau, anchor_age=window.anchor_age_kyr_bp,
        initial_history=initial_history,
    )
    if history_variants:
        # Compare on -BP before subtracting the anchor: adjacent floating ages
        # must remain distinct at event and rectangular-window boundaries.
        event_elapsed_kyr = window.anchor_age_kyr_bp - event_ages
        comparison_time, event_comparison_time = -ages, -event_ages
        previous_count = np.searchsorted(event_comparison_time, comparison_time, side="left")
        since_last_kyr = np.zeros(len(ages))
        has_previous = previous_count > 0
        since_last_kyr[has_previous] = (
            elapsed_kyr[has_previous] - event_elapsed_kyr[previous_count[has_previous] - 1]
        )
        frame["time_since_last_event_kyr"] = since_last_kyr
        frame["log_time_since_last_event"] = np.log1p(since_last_kyr)
        first_recent = np.searchsorted(event_comparison_time, comparison_time - tau, side="right")
        frame["rectangular_history_count"] = (previous_count - first_recent).astype(float)
    return pd.DataFrame(frame)


def build_design(events, windows, forcings, phase_anchors, scaling, *, tau=1.5,
                 initial_history=0.0, quadrature_order=4, history_variants=False):
    """Return response-event and integration features on explicit windows.

    Recompute windows for chronology draws; retain nominal windows for deletion
    and simulation. Scaling is supplied separately and is never recomputed here.
    """
    if set(events.segment_id) != set(windows.segment_id) or windows.segment_id.duplicated().any():
        raise ValueError("Events need one response window per segment")
    for name, (age, values) in {**forcings, "phase anchors": phase_anchors}.items():
        if len(age) < 2 or not np.isfinite([age, values]).all() or np.any(np.diff(age) <= 0):
            raise ValueError(f"{name} requires finite values at unique increasing ages")
    event_frames, integral_frames = [], []
    for window in windows.itertuples(index=False):
        selected = events.loc[events.segment_id.eq(window.segment_id)].sort_values("event_age_kyr_bp")
        ages = selected.event_age_kyr_bp.to_numpy(float)
        if not np.isfinite(ages).all() or len(np.unique(ages)) != len(ages):
            raise ValueError("Event ages must be finite and unique within each segment")
        if (np.any(ages < window.observation_start_kyr_bp)
                or np.any(ages > window.observation_end_kyr_bp)):
            raise ValueError(f"An event lies outside the {window.segment_id} observation window")
        if not len(ages) or ages.max() != window.anchor_age_kyr_bp:
            raise ValueError("Response window requires the original conditioning event")
        response = (ages < window.response_end_kyr_bp) & (ages >= window.response_start_kyr_bp)
        features = evaluate_features(
            ages[response], ages, window, forcings, phase_anchors, scaling,
            tau=tau, initial_history=initial_history, history_variants=history_variants,
        )
        if "event_id" in selected:
            features["event_id"] = selected.loc[response, "event_id"].to_numpy()
        event_frames.append(features)

        knots = integration_breakpoints(window, forcings, phase_anchors, ages, tau)
        nodes, weights = gauss_legendre_intervals(knots, order=quadrature_order)
        integration = evaluate_features(
            nodes, ages, window, forcings, phase_anchors, scaling,
            tau=tau, initial_history=initial_history, history_variants=history_variants,
        )
        integration["weight"] = weights  # Positive kyr durations also integrate du.
        integral_frames.append(integration)
    return (pd.concat(event_frames, ignore_index=True),
            pd.concat(integral_frames, ignore_index=True))


def sample_event_phases(events, precession, phase_anchors, *, age_column="event_age_kyr_bp"):
    """Attach descriptive phase to all events, including conditioning events."""
    ages = pd.to_numeric(events[age_column], errors="coerce").to_numpy(float)
    if not np.isfinite(ages).all():
        raise ValueError("Event ages must be finite before phase interpolation")
    extrema = pd.DataFrame({"age_ka": phase_anchors[0], "anchor_phase_unwrapped_rad": phase_anchors[1]})
    phase = evaluate_phase_at_ages(ages, extrema, True)
    out = events.reset_index(drop=True).copy()
    out["precession_index"] = interpolate_checked(ages, *precession, context="precession at events")
    for name in ("phase_unwrapped_rad", "phase_rad", "phase_deg", "phase_fraction", "phase_extrapolated"):
        out["pre_" + name] = phase[name].to_numpy()
    return out


def clean_series(
    age_ka: np.ndarray, value: np.ndarray, *, context: str = "input series"
) -> tuple[np.ndarray, np.ndarray]:
    """Drop non-finite pairs, require unique ages, and return age-sorted arrays."""

    age_ka = np.asarray(age_ka, dtype=float)
    value = np.asarray(value, dtype=float)
    ok = np.isfinite(age_ka) & np.isfinite(value)
    frame = pd.DataFrame({"age_ka": age_ka[ok], "value": value[ok]})
    if frame.age_ka.duplicated().any():
        raise ValueError(f"{context} has duplicate ages")
    frame = frame.sort_values("age_ka").reset_index(drop=True)
    return frame["age_ka"].to_numpy(dtype=float), frame["value"].to_numpy(dtype=float)


def require_interpolation_coverage(
    target_age_ka: np.ndarray, source_age_ka: np.ndarray, *, context: str
) -> None:
    """Raise if interpolation targets extend beyond the source age range."""

    target = np.asarray(target_age_ka, dtype=float)
    source = np.asarray(source_age_ka, dtype=float)
    if len(source) < 2:
        raise ValueError(f"{context} needs at least two source ages for interpolation.")
    target_min = float(np.nanmin(target))
    target_max = float(np.nanmax(target))
    source_min = float(np.nanmin(source))
    source_max = float(np.nanmax(source))
    if target_min < source_min or target_max > source_max:
        raise ValueError(
            f"{context} does not cover requested ages: targets {target_min:.3f}-{target_max:.3f} kyr, "
            f"source {source_min:.3f}-{source_max:.3f} kyr."
        )


def interpolate_checked(
    target_age_ka: np.ndarray,
    source_age_ka: np.ndarray,
    source_value: np.ndarray,
    *,
    context: str,
) -> np.ndarray:
    """Interpolate only when source ages fully cover the requested targets."""

    require_interpolation_coverage(target_age_ka, source_age_ka, context=context)
    return np.interp(target_age_ka, source_age_ka, source_value)


def interpolate_unwrapped_phase(
    ages_ka: np.ndarray,
    extrema_age_ka: np.ndarray,
    extrema_phase_unwrapped_rad: np.ndarray,
    warn_on_extrapolation: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate phase and mark ages outside the extrema range.

    ``numpy.interp`` normally clamps values outside its source range.  Phase is
    instead linearly extrapolated from the first or last two anchors so that it
    continues advancing at the local half-cycle rate.  The Boolean return value
    makes those extrapolated phases explicit to callers.
    """

    ages = np.asarray(ages_ka, dtype=float)
    anchor_ages = np.asarray(extrema_age_ka, dtype=float)
    anchor_phases = np.asarray(extrema_phase_unwrapped_rad, dtype=float)
    if len(anchor_ages) < 2 or len(anchor_ages) != len(anchor_phases):
        raise ValueError("Phase interpolation needs at least two paired anchors.")
    if np.any(~np.isfinite(anchor_ages)) or np.any(np.diff(anchor_ages) <= 0):
        raise ValueError("Phase-anchor ages must be finite and strictly increasing.")

    phase = np.interp(ages, anchor_ages, anchor_phases)
    left = ages < anchor_ages[0]
    right = ages > anchor_ages[-1]
    extrapolated = left | right

    if np.any(left):
        slope = (anchor_phases[1] - anchor_phases[0]) / (
            anchor_ages[1] - anchor_ages[0]
        )
        phase[left] = anchor_phases[0] + slope * (ages[left] - anchor_ages[0])
    if np.any(right):
        slope = (anchor_phases[-1] - anchor_phases[-2]) / (
            anchor_ages[-1] - anchor_ages[-2]
        )
        phase[right] = anchor_phases[-1] + slope * (ages[right] - anchor_ages[-1])

    if warn_on_extrapolation and np.any(extrapolated):
        warnings.warn(
            "The phase extrema do not fully cover the requested ages; "
            "extrapolated phase values are used.",
            stacklevel=2,
        )
    return phase, extrapolated


def evaluate_phase_at_ages(
    ages_ka: np.ndarray,
    extrema: pd.DataFrame,
    warn_on_extrapolation: bool = False,
) -> pd.DataFrame:
    """Evaluate wrapped and unwrapped phase at arbitrary ages."""

    ages = np.asarray(ages_ka, dtype=float)
    unwrapped, extrapolated = interpolate_unwrapped_phase(
        ages,
        extrema["age_ka"].to_numpy(dtype=float),
        extrema["anchor_phase_unwrapped_rad"].to_numpy(dtype=float),
        warn_on_extrapolation,
    )
    wrapped = np.mod(unwrapped, 2 * np.pi)
    return pd.DataFrame(
        {
            "age_ka": ages,
            "phase_unwrapped_rad": unwrapped,
            "phase_rad": wrapped,
            "phase_deg": np.degrees(wrapped),
            "phase_fraction": wrapped / (2 * np.pi),
            "phase_extrapolated": extrapolated,
        }
    )
