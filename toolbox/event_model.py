"""Event covariates and continuous integration on explicit BP1950 windows.

Input tables are read by each analysis. This module computes the quantities
used by the likelihood; it does not choose model columns or run the optimizer.
"""
import numpy as np
import pandas as pd

from toolbox.point_process import exponential_history, gauss_legendre_intervals, simulate_segment_events


def response_windows(events, observations):
    """Condition each independent record on its oldest observed event."""
    if observations.segment_id.duplicated().any():
        raise ValueError("Observation segment IDs must be unique")
    if set(events.segment_id) != set(observations.segment_id):
        raise ValueError("Events may not in this segment")
    rows = []
    for row in observations.itertuples(index=False):
        ages = events.loc[events.segment_id.eq(row.segment_id), "event_age_kyr_bp"].to_numpy(float)
        if not np.isfinite(ages).all() or len(np.unique(ages)) != len(ages):
            raise ValueError("Event ages must be finite and unique within each segment")
        young = float(row.observation_start_kyr_bp)
        old = float(row.observation_end_kyr_bp)
        if not np.isfinite([young, old]).all() or young >= old:
            raise ValueError("Series maybe flipped or missing observation start/end ages")
        if np.any(ages < young) or np.any(ages > old):
            raise ValueError(f"An event lies outside the {row.segment_id} observation window")
        # BP ages decrease forward in time. The oldest event fixes the initial
        # history; likelihood exposure begins just after it and ends at young.
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
        # Including every source knot makes this exact for the piecewise-linear
        # forcing. Pool by record exposure (kyr), not by source sampling density.
        window_integral = np.trapezoid(values, knots)
        forcing_integral = forcing_integral + window_integral
        window_duration = old - young
        exposure_kyr = exposure_kyr + window_duration
        extrema.extend([values.min(), values.max()])
    # The range is shared across all response windows; a constant forcing is
    # assigned unit range to avoid dividing its centered predictor by zero.
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
    # Each interval must avoid changes in forcing slope and history jumps so
    # that Gaussian quadrature acts on a smooth part of the intensity.
    candidates = np.concatenate([
        np.array([young, old]), *[source[0] for source in forcings.values()],
        phase_anchors[0], events, events - tau,
    ])
    return np.unique(candidates[(candidates >= young) & (candidates <= old)])


def evaluate_features(ages, event_ages, window, forcings, phase_anchors, scaling,
                      *, tau=1.5, initial_history=0.0, history_variants=False):
    """Evaluate climate, phase and strictly earlier-event history at BP ages."""
    ages = np.asarray(ages, float)
    # Elapsed time increases from zero at the conditioning event, unlike BP age.
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
            # Use the supplied nominal scale for every chronology and refit;
            # coefficients then retain the same units across realizations.
            frame[name + "_scaled"] = (values - scale["mean"]) / scale["range"]
    phase, extrapolated = interpolate_unwrapped_phase(ages, *phase_anchors)
    frame.update(
        pre_phase_unwrapped_rad=phase,
        pre_phase_rad=np.mod(phase, 2 * np.pi),
        pre_phase_deg=np.mod(np.degrees(phase), 360),
        pre_phase_sin=np.sin(phase), pre_phase_cos=np.cos(phase),
        pre_phase_extrapolated=extrapolated,
    )
    # The anchor contributes to later history but an event never counts itself.
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
    for name, (age, values) in forcings.items():
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
        # Event rows supply sum_i log(lambda(t_i)); the conditioning event is
        # excluded from this sum while remaining in ages for history evaluation.
        response = (ages < window.response_end_kyr_bp) & (ages >= window.response_start_kyr_bp)
        features = evaluate_features(
            ages[response], ages, window, forcings, phase_anchors, scaling,
            tau=tau, initial_history=initial_history, history_variants=history_variants,
        )
        if "event_id" in selected:
            features["event_id"] = selected.loc[response, "event_id"].to_numpy()
        event_frames.append(features)

        # Integration rows supply sum_j weight_j * lambda(node_j), the integrated
        # conditional intensity over the exposure, including event-free gaps.
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
    ages = events[age_column].to_numpy(float)
    extrema = pd.DataFrame({"age_ka": phase_anchors[0], "anchor_phase_unwrapped_rad": phase_anchors[1]})
    phase = evaluate_phase_at_ages(ages, extrema)
    out = events.reset_index(drop=True).copy()
    out["precession_index"] = interpolate_checked(ages, *precession, context="precession at events")
    for name in ("phase_unwrapped_rad", "phase_rad", "phase_deg", "phase_fraction", "phase_extrapolated"):
        out["pre_" + name] = phase[name].to_numpy()
    return out


def mark_event_roles(events, windows):
    """Label conditioning, response and history-only events on explicit windows."""
    frames = []
    for window in windows.itertuples(index=False):
        selected = events.loc[events.segment_id.eq(window.segment_id)].sort_values("event_age_kyr_bp").copy()
        ages = selected.event_age_kyr_bp.to_numpy(float)
        response = (ages < window.response_end_kyr_bp) & (ages >= window.response_start_kyr_bp)
        selected["event_role"] = np.where(ages == window.anchor_age_kyr_bp, "conditioning",
                                          np.where(response, "response", "history_only"))
        selected["included_in_response"] = response
        frames.append(selected)
    return pd.concat(frames, ignore_index=True)


def fitted_rate_table(events, windows, forcings, phase_anchors, scaling, models,
                      *, tau=1.5, initial_history=0.0, step_kyr=0.1):
    """Evaluate fitted rates, retaining each discontinuous event-history jump."""
    frames = []
    terms = set().union(*(model.terms for model in models.values()))
    for window in windows.itertuples(index=False):
        event_ages = events.loc[events.segment_id.eq(window.segment_id), "event_age_kyr_bp"].to_numpy(float)
        # The next smaller BP float is immediately after the event. Retaining
        # both sides shows the rate jump without shifting its plotted age.
        after_event = np.nextafter(event_ages, -np.inf)
        after_event = after_event[after_event >= window.response_start_kyr_bp]
        query = np.unique(np.r_[
            np.arange(window.response_start_kyr_bp, window.response_end_kyr_bp, step_kyr),
            event_ages, after_event, window.response_end_kyr_bp,
        ])
        features = evaluate_features(query, event_ages, window, forcings, phase_anchors,
                                     scaling, tau=tau, initial_history=initial_history)
        if "mis6_segment" in terms:
            features["mis6_segment"] = features.segment_id.eq("MIS6").astype(float)
        rates = features[["segment_id", "age_kyr_bp", "lr04", "co2", "precession_index", "pre_phase_deg"]].copy()
        for name, model in models.items():
            rates[name + "_rate"] = np.exp(features.loc[:, model.terms].to_numpy(float) @ model.beta)
        frames.append(rates)
    return pd.concat(frames, ignore_index=True)


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
    if len(anchor_ages) < 2:
        raise ValueError("Phase interpolation needs at least two paired anchors.")
    if (not np.isfinite(anchor_ages).all() or not np.isfinite(anchor_phases).all()
            or np.any(np.diff(anchor_ages) <= 0)):
        raise ValueError("Phase anchors need finite values at strictly increasing ages.")

    # Interpolate continuous radians on increasing BP age; wrapping first would
    # interpolate across an artificial 0/2pi discontinuity at each cycle.
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

    return phase, extrapolated


def evaluate_phase_at_ages(
    ages_ka: np.ndarray,
    extrema: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate wrapped and unwrapped phase at arbitrary ages."""

    ages = np.asarray(ages_ka, dtype=float)
    unwrapped, extrapolated = interpolate_unwrapped_phase(
        ages,
        extrema["age_ka"].to_numpy(dtype=float),
        extrema["anchor_phase_unwrapped_rad"].to_numpy(dtype=float),
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


def _background_values(ages, forcings, phase_anchors, scaling, segment_id, terms, beta):
    """Exogenous log rate on the BP axis, without allocating feature tables."""
    age = np.asarray(ages, float)
    values = {"intercept": np.ones_like(age),
              "mis6_segment": np.full_like(age, float(segment_id == "MIS6"))}
    for name, scale in scaling.items():
        values[name + "_scaled"] = (np.interp(age, *forcings[name]) - scale["mean"]) / scale["range"]
    phase, _ = interpolate_unwrapped_phase(np.atleast_1d(age), *phase_anchors)
    phase = phase.reshape(age.shape)
    values.update(pre_phase_sin=np.sin(phase), pre_phase_cos=np.cos(phase))
    return sum(coef * values[term] for term, coef in zip(terms, beta)
               if term != "same_type_exponential_history")


def prepare_model_simulation(events, windows, forcings, phase_anchors, scaling, model,
                             *, tau=1.5, initial_history=0.0, proposal_interval_kyr=1.0):
    """Certify background envelopes on fixed generator windows for thinning.

    The envelope uses scaled linear forcings, a segment offset and phase terms.
    New nonlinear predictors need their own continuous evaluation and bounds.
    """
    from functools import partial
    if not model.converged or not np.isfinite(model.beta).all():
        raise ValueError("A finite fitted generator is required")
    coefficients = dict(zip(model.terms, model.beta))
    # Nonnegative history with beta_H <= 0 can only suppress the rate, allowing
    # an envelope built from the exogenous background alone.
    history_beta = coefficients.get("same_type_exponential_history", 0.)
    scales = scaling.to_dict("index")
    allowed = {"intercept", "mis6_segment", "same_type_exponential_history",
               "pre_phase_sin", "pre_phase_cos", *[name + "_scaled" for name in scales]}
    if set(model.terms) - allowed:
        raise ValueError("Simulator requires explicit continuous bounds for these model terms")
    prepared = []
    for window in windows.itertuples(index=False):
        name = window.segment_id
        edges = np.r_[np.arange(window.response_start_kyr_bp, window.anchor_age_kyr_bp,
                                proposal_interval_kyr), window.anchor_age_kyr_bp]
        upper = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            bounds = {"intercept": (1., 1.), "mis6_segment": (float(name == "MIS6"),) * 2}
            for forcing, scale in scales.items():
                age, value = forcings[forcing]
                # A linear interpolant attains its extrema at interval ends or
                # source knots, even if those knots fall between proposal edges.
                knots = np.r_[lo, age[(age > lo) & (age < hi)], hi]
                vals = (np.interp(knots, age, value) - scale["mean"]) / scale["range"]
                bounds[forcing + "_scaled"] = (float(vals.min()), float(vals.max()))
            # Global trigonometric bounds cover extrema inside each interval.
            bounds.update(pre_phase_sin=(-1., 1.), pre_phase_cos=(-1., 1.))
            # Maximize each signed contribution separately: the sum remains an
            # upper bound even when predictor extrema occur at different ages.
            upper.append(sum(max(coef * bounds[term][0], coef * bounds[term][1])
                             for term, coef in coefficients.items()
                             if term != "same_type_exponential_history") + 1e-12)
        anchor = events.loc[events.segment_id.eq(name)].sort_values("event_age_kyr_bp").iloc[-1]
        prepared.append(dict(
            segment_id=name, anchor_age=window.anchor_age_kyr_bp,
            young_age=window.response_start_kyr_bp, anchor_id=anchor.event_id,
            breakpoints=edges, log_upper_bounds=np.array(upper),
            log_background=partial(_background_values, forcings=forcings,
                phase_anchors=phase_anchors, scaling=scales, segment_id=name,
                terms=model.terms, beta=model.beta),
            history_beta=history_beta, tau=tau, initial_history=initial_history,
        ))
    return prepared


def simulate_prepared_events(prepared, rng):
    """Simulate independent response segments while retaining exact anchors."""
    frames = []
    for part in prepared:
        age = simulate_segment_events(
            part["anchor_age"], part["young_age"], part["breakpoints"],
            part["log_background"], part["log_upper_bounds"], part["history_beta"],
            part["tau"], rng, initial_history=part["initial_history"],
        )
        # The anchor is conditioned on, so only subsequent events are random;
        # preserving its identity lets the same design builder refit each draw.
        frame = pd.DataFrame({
            "event_age_kyr_bp": age, "segment_id": part["segment_id"],
            "event_id": [part["anchor_id"]] + [f"sim_{part['segment_id']}_{i}" for i in range(1, len(age))],
        })
        frame["event_label"] = frame.event_id
        frames.append(frame.sort_values("event_age_kyr_bp"))
    return pd.concat(frames, ignore_index=True)


def rescaled_event_intervals(event_features, integration_features, windows, model):
    """Cumulative fitted intensities at responses plus terminal censored waits."""
    events, cumulative, tails = {}, {}, {}
    frame = integration_features
    rate = np.exp(frame.loc[:, model.terms].to_numpy(float) @ model.beta)
    if model.status == "zero_events":
        rate = np.zeros(len(frame))
    for window in windows.itertuples(index=False):
        name = window.segment_id
        response_ages = event_features.loc[event_features.segment_id.eq(name), "age_kyr_bp"].to_numpy(float)
        response_ages = np.sort(response_ages)[::-1]
        response_elapsed = window.anchor_age_kyr_bp - response_ages
        mask = frame.segment_id.eq(name).to_numpy()
        node_ages = frame.loc[mask, "age_kyr_bp"].to_numpy(float)
        # Accumulate from the anchor toward the present, keeping nodes and
        # masses aligned. Incoming table row order has no time meaning.
        order = np.argsort(-node_ages)
        node_ages = node_ages[order]
        mass = rate[mask] * frame.loc[mask, "weight"].to_numpy(float)
        mass = mass[order]
        # Integrated rate is dimensionless. Differences of cumulative intensity
        # at successive response events give time-rescaled inter-event waits.
        cumulative_mass = np.r_[0.0, np.cumsum(mass)]
        # Compare on -BP before subtracting the origin to retain precision.
        indices = np.searchsorted(-node_ages, -response_ages, side="left")
        events[name] = response_elapsed
        cumulative[name] = cumulative_mass[indices]
        # Exposure after the last event is a censored wait, not another event;
        # keep it separate from the completed waits used for residual tests.
        tails[name] = float(mass[indices[-1]:].sum() if len(indices) else mass.sum())
    return events, cumulative, tails
