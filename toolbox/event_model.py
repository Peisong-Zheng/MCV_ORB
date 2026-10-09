"""Event covariates and continuous integration on explicit BP1950 windows.

Input tables are read by each analysis. This module computes the quantities
used by the likelihood; it does not choose model columns or run the optimizer.
Native forcing and phase nodes are validated during preprocessing; changing
event catalogues and requested interpolation support are checked here.
"""
from functools import partial

import numpy as np
import pandas as pd

from toolbox.point_process import exponential_history, gauss_legendre_intervals, simulate_segment_events


def response_windows(events, observations):
    """Condition each independent record on its oldest observed event."""
    rows = []
    for row in observations.itertuples(index=False):
        ages = events.loc[events.segment_id.eq(row.segment_id), "event_age_kyr_bp"].to_numpy(float)
        young = float(row.observation_start_kyr_bp)
        old = float(row.observation_end_kyr_bp)
        if not np.isfinite([young, old]).all() or young >= old:
            raise ValueError("Series maybe flipped or missing observation start/end ages")
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


def scale_forcing_v2(forcings, windows):
    """Time-weighted means and ranges on the nominal response windows."""
    support_young = windows.response_start_kyr_bp.min()
    support_old = windows.response_end_kyr_bp.max()
    rows = []
    for name, (source_age, source_value) in forcings.items():
        if support_young < source_age[0] or support_old > source_age[-1]:
            raise ValueError(f"{name} does not cover the nominal response windows")
        exposure_kyr = 0.0
        forcing_integral = 0.0
        extrema = []
        for window in windows.itertuples(index=False):
            young, old = window.response_start_kyr_bp, window.response_end_kyr_bp
            inside = (source_age > young) & (source_age < old)
            knots = np.r_[young, source_age[inside], old]
            values = np.interp(knots, source_age, source_value)
            # Every native knot makes this exact for the linear interpolant.
            # Pool by record exposure, not by source sampling density.
            window_integral = np.trapezoid(values, knots)
            forcing_integral = forcing_integral + window_integral
            window_duration = old - young
            exposure_kyr = exposure_kyr + window_duration
            extrema.extend([values.min(), values.max()])
        minimum, maximum = float(min(extrema)), float(max(extrema))
        # A constant forcing has unit range and a zero centered predictor.
        rows.append(dict(forcing_id=name, mean=forcing_integral / exposure_kyr,
                         min=minimum, max=maximum, range=maximum - minimum or 1.0))
    return pd.DataFrame(rows).set_index("forcing_id")


def build_likelihood_tables(events, windows, forcings, phase_anchors, scaling, *, tau=1.5,
                            initial_history=0.0, quadrature_order=4, history_variants=False):
    """Return response-event and integration features on explicit windows.

    Recompute windows for chronology draws; retain nominal windows for deletion
    and simulation. Scaling is supplied separately and is never recomputed here.
    """
    if set(events.segment_id) != set(windows.segment_id) or windows.segment_id.duplicated().any():
        raise ValueError("Events need one response window per segment")
    # Native source nodes are validated in forcing_data_pre_processing.py.
    # Check new response support once before interpolating either query grid.
    support_young = windows.response_start_kyr_bp.min()
    support_old = windows.response_end_kyr_bp.max()
    for name, (source_age, _) in forcings.items():
        if support_young < source_age[0] or support_old > source_age[-1]:
            raise ValueError(f"{name} does not cover the response windows")
    event_frames, integral_frames = [], []
    for window in windows.itertuples(index=False):
        # Select this independent record, keeping event IDs aligned with ages.
        selected = events.loc[events.segment_id.eq(window.segment_id)].sort_values("event_age_kyr_bp")
        ages = selected.event_age_kyr_bp.to_numpy(float)
        if not np.isfinite(ages).all() or len(np.unique(ages)) != len(ages):
            raise ValueError("Event ages must be finite and unique within each segment")
        if (np.any(ages < window.observation_start_kyr_bp)
                or np.any(ages > window.observation_end_kyr_bp)):
            raise ValueError(f"An event lies outside the {window.segment_id} observation window")
        if not len(ages) or ages.max() != window.anchor_age_kyr_bp:
            raise ValueError("Response window requires the original conditioning event")
        # Responses enter sum_i log(lambda(t_i)); retain the anchor for history.
        response = (ages < window.response_end_kyr_bp) & (ages >= window.response_start_kyr_bp)

        # Split at every forcing slope change and history boundary, retaining
        # the existing rectangular-history exits even in the main model.
        young, old = window.response_start_kyr_bp, window.response_end_kyr_bp
        candidates = np.concatenate([
            np.array([young, old]), *[source[0] for source in forcings.values()],
            phase_anchors[0], ages, ages - tau,
        ])
        knots = np.unique(candidates[(candidates >= young) & (candidates <= old)])
        # Map the Gaussian rule into every smooth integration interval.
        nodes, weights = gauss_legendre_intervals(knots, order=quadrature_order)

        # Evaluate the same predictors once, keeping response rows first.
        n_response = int(response.sum())
        query_ages = np.r_[ages[response], nodes]
        features = evaluate_features(
            query_ages, ages, window, forcings, phase_anchors, scaling,
            tau=tau, initial_history=initial_history, history_variants=history_variants,
        )
        # Split the likelihood's event contribution from its exposure integral.
        responses = features.iloc[:n_response].copy()
        integration = features.iloc[n_response:].copy()
        if "event_id" in selected:
            responses["event_id"] = selected.loc[response, "event_id"].to_numpy()
        # Positive kyr weights integrate the rate through event-free gaps too.
        integration["weight"] = weights
        event_frames.append(responses)
        integral_frames.append(integration)
    return (pd.concat(event_frames, ignore_index=True),
            pd.concat(integral_frames, ignore_index=True))


def evaluate_features(ages, event_ages, window, forcings, phase_anchors, scaling,
                      *, tau=1.5, initial_history=0.0, history_variants=False):
    """Evaluate predictors at BP ages within the caller's checked source support."""
    ages = np.asarray(ages, float)
    # Elapsed time increases from zero at the conditioning event, unlike BP age.
    elapsed_kyr = window.anchor_age_kyr_bp - ages
    frame = dict(age_kyr_bp=ages, elapsed_kyr=elapsed_kyr,
                 segment_id=np.repeat(window.segment_id, len(ages)),
                 intercept=np.ones(len(ages)))
    for name, (source_age, source_value) in forcings.items():
        values = np.interp(ages, source_age, source_value)
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


def sample_event_phases(events, precession, phase_anchors, *, age_column="event_age_kyr_bp"):
    """Attach descriptive phase to all events, including conditioning events."""
    ages = events[age_column].to_numpy(float)
    if ages.min() < precession[0][0] or ages.max() > precession[0][-1]:
        raise ValueError("Precession does not cover the event ages")
    unwrapped, extrapolated = interpolate_unwrapped_phase(ages, *phase_anchors)
    wrapped = np.mod(unwrapped, 2 * np.pi)
    out = events.reset_index(drop=True).copy()
    out["precession_index"] = np.interp(ages, *precession)
    out["pre_phase_unwrapped_rad"] = unwrapped
    out["pre_phase_rad"] = wrapped
    out["pre_phase_deg"] = np.degrees(wrapped)
    out["pre_phase_fraction"] = wrapped / (2 * np.pi)
    out["pre_phase_extrapolated"] = extrapolated
    return out


def fitted_rate_table(events, windows, forcings, phase_anchors, scaling, models,
                      *, tau=1.5, initial_history=0.0, step_kyr=0.1):
    """Evaluate fitted rates, retaining each discontinuous event-history jump."""
    support_young = windows.response_start_kyr_bp.min()
    support_old = windows.response_end_kyr_bp.max()
    for name, (source_age, _) in forcings.items():
        if support_young < source_age[0] or support_old > source_age[-1]:
            raise ValueError(f"{name} does not cover the fitted-rate windows")
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
    if not model.converged or not np.isfinite(model.beta).all():
        raise ValueError("A finite fitted generator is required")
    if not np.isfinite(tau) or tau <= 0 or not np.isfinite(initial_history) or initial_history < 0:
        raise ValueError("Positive decay time and nonnegative initial history are required")
    if not np.isfinite(proposal_interval_kyr) or proposal_interval_kyr <= 0:
        raise ValueError("Proposal interval must be finite and positive")
    support_young = windows.response_start_kyr_bp.min()
    support_old = windows.anchor_age_kyr_bp.max()
    for name, (source_age, _) in forcings.items():
        if support_young < source_age[0] or support_old > source_age[-1]:
            raise ValueError(f"{name} does not cover the generator windows")
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
        young, anchor_age = window.response_start_kyr_bp, window.anchor_age_kyr_bp
        if not np.isfinite([anchor_age, young]).all() or anchor_age <= young:
            raise ValueError("Conditioning anchor must be older than the response endpoint")
        edges = np.r_[np.arange(young, anchor_age, proposal_interval_kyr), anchor_age]
        if not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
            raise ValueError("Simulation breakpoints must be finite and increasing")
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
        log_upper_bounds = np.array(upper)
        if np.isnan(log_upper_bounds).any() or np.isposinf(log_upper_bounds).any():
            raise ValueError("One finite or negative-infinite log bound is required per interval")
        anchor = events.loc[events.segment_id.eq(name)].sort_values("event_age_kyr_bp").iloc[-1]
        prepared.append(dict(
            segment_id=name, anchor_age=window.anchor_age_kyr_bp,
            young_age=window.response_start_kyr_bp, anchor_id=anchor.event_id,
            breakpoints=edges, log_upper_bounds=log_upper_bounds,
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
