"""Conditional point-process diagnostics and refitted bootstrap calibration.

Completed intervals are transformed by their fitted compensator. Fixed-time
termination and fitted parameters are reproduced by simulation and refitting;
ordinary KS reference p values are deliberately not used here. See Gerhard &
Gerstner (2010), NeurIPS, on point-process model checking by time rescaling.
"""


import numpy as np
import pandas as pd
from scipy.stats import beta as beta_distribution


GOF_STATISTICS = ("ks_uniform", "adjacent_dependence")


class DiagnosticFailure(RuntimeError):
    """A diagnosed numerical failure; retain its replicate without redrawing."""


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


def residual_statistics(events_by_segment, cumulative_at_events_by_segment,
                        tail_integrals=None):
    """Return completed-interval statistics and inspectable residual tables.

    Arrays contain response events in chronological order (old to young),
    without the fixed conditioning event. Event coordinates may be decreasing
    BP ages or increasing forward times. The compensator starts at zero at the
    conditioning event. Tail integrals cover the last response event to the
    observation endpoint, or the whole response interval when no event occurs.
    Tails affect cumulative residuals, not the completed waiting-time sample.
    """
    interval_frames, segment_rows, uniform_samples = [], [], []
    adjacent_sum, n_pairs = 0.0, 0
    for segment_id, coordinates in events_by_segment.items():
        events = np.asarray(coordinates, dtype=float)
        cumulative = np.asarray(cumulative_at_events_by_segment[segment_id], dtype=float)
        # Compensator increments are Exp(1) under the model; 1-exp(-z) makes them uniform.
        transformed = np.diff(np.r_[0.0, cumulative])
        tail = np.nan if tail_integrals is None else float(tail_integrals[segment_id])
        if (not np.isfinite(transformed).all() or np.any(transformed < 0)
                or (tail_integrals is not None and (not np.isfinite(tail) or tail < 0))):
            raise ValueError("Invalid cumulative intensity")
        uniform = -np.expm1(-transformed)
        uniform_samples.append(uniform)
        # Only consecutive intervals within the same segment form dependence pairs.
        local_sum = float(np.sum((uniform[:-1] - 0.5) * (uniform[1:] - 0.5)))
        local_pairs = max(len(uniform) - 1, 0)
        adjacent_sum = adjacent_sum + local_sum
        n_pairs = n_pairs + local_pairs

        # The final waiting time is right-censored; include its exposure in N-Lambda only.
        at_last_event = float(cumulative[-1]) if len(cumulative) else 0.0
        endpoint_integral = at_last_event + tail
        segment_rows.append(dict(
            segment_id=segment_id, n_intervals=len(events), n_adjacent_pairs=local_pairs,
            adjacent_product_sum=local_sum, cumulative_at_last_event=at_last_event,
            tail_integral=tail, cumulative_at_endpoint=endpoint_integral,
            residual_at_endpoint=len(events) - endpoint_integral,
            status="ok" if len(events) else "no_response_events"))
        interval_frames.append(pd.DataFrame(dict(
            segment_id=[segment_id] * len(events), event_index=np.arange(1, len(events) + 1),
            event_coordinate=events, cumulative_intensity=cumulative,
            rescaled_interval=transformed, uniform_residual=uniform,
            cumulative_residual=np.arange(1, len(events) + 1) - cumulative)))

    uniform = np.sort(np.concatenate(uniform_samples))
    n = len(uniform)
    ks = 0.0
    if n:
        # Two sides of the empirical CDF; no asymptotic KS p value is used.
        ranks = np.arange(1, n + 1)
        ks = float(max(np.max(ranks / n - uniform), np.max(uniform - (ranks - 1) / n)))
    statistics = dict(ks_uniform=ks,
                      adjacent_dependence=abs(adjacent_sum) / np.sqrt(max(n_pairs, 1)))
    return dict(statistics=statistics, n_intervals=n, n_adjacent_pairs=n_pairs,
                status="ok" if n else "no_response_events",
                intervals=pd.concat(interval_frames, ignore_index=True),
                segments=pd.DataFrame(segment_rows))


def likelihood_ratio(loglik_full, loglik_null, tolerance=1e-7):
    """Nested LR, allowing only negligible negative optimization roundoff."""
    if not np.isfinite([loglik_full, loglik_null]).all():
        raise DiagnosticFailure("Non-finite likelihood in nested comparison")
    gain = float(loglik_full - loglik_null)
    if gain < -tolerance:
        raise DiagnosticFailure("Full likelihood is below the nested null likelihood")
    return 2 * max(gain, 0.0)


def holm_adjust(p_values):
    """Holm adjustment; unresolved tests remain in the prespecified family."""
    p = np.asarray(p_values, dtype=float)
    finite = np.isfinite(p)
    # Missing tests occupy their original family slots but remain unresolved in the output.
    order = np.argsort(np.where(finite, p, 1.0), kind="stable")
    adjusted = np.empty(len(p))
    # Step-down multipliers and a cumulative maximum preserve ordered adjusted p values.
    adjusted[order] = np.minimum(1, np.maximum.accumulate(
        np.where(finite, p, 1.0)[order] * np.arange(len(p), 0, -1)))
    adjusted[~finite] = np.nan
    return adjusted


def bootstrap_summary(observed, replicates, statistics=GOF_STATISTICS, adjust_holm=True):
    """Summarize upper-tail bootstrap tests without discarding failed draws.

    Plus-one p and binomial simulation precision are reported only when every
    requested replicate has a valid statistic. Otherwise p is unresolved and
    its bounds treat invalid replicates as unknown exceedances. The confidence
    interval concerns Monte Carlo exceedance probability, not model parameters.
    """
    if replicates.empty:
        raise ValueError("Bootstrap summary needs simulated draws")
    if not replicates.fit_valid.isin([True, False]).all():
        raise ValueError("Every replicate needs an explicit fit-valid status")
    valid_fit = replicates.fit_valid.to_numpy(bool)
    rows = []
    for name in statistics:
        point = float(observed[name])
        values = replicates[name].to_numpy(float)
        if not np.isfinite(point):
            raise ValueError("Observed statistic must be finite")
        valid = valid_fit & np.isfinite(values)
        total, accepted = len(values), int(valid.sum())
        missing = total - accepted
        exceeding = int(np.sum(values[valid] >= point))
        # Bound unresolved p values by treating every missing draw as below, then above, the observation.
        p_low = (1 + exceeding) / (total + 1)
        p_high = (1 + exceeding + missing) / (total + 1)
        p = p_low if not missing else np.nan
        ci_low = ci_high = mc_se = np.nan
        if not missing:
            # Clopper-Pearson bounds quantify the finite bootstrap's exceedance uncertainty.
            ci_low = 0.0 if exceeding == 0 else beta_distribution.ppf(
                0.025, exceeding, total - exceeding + 1)
            ci_high = 1.0 if exceeding == total else beta_distribution.ppf(
                0.975, exceeding + 1, total - exceeding)
            mc_se = np.sqrt(p * (1 - p) / (total + 1))
        rows.append(dict(statistic=name, observed=point, n_bootstrap=total,
                         n_valid=accepted, n_invalid=missing, n_exceeding=exceeding,
                         bootstrap_p=p, bootstrap_p_lower_bound=p_low,
                         bootstrap_p_upper_bound=p_high, bootstrap_p_mc_se=mc_se,
                         exceedance_probability_ci95_low=ci_low,
                         exceedance_probability_ci95_high=ci_high,
                         status="ok" if not missing else "incomplete_bootstrap"))
    summary = pd.DataFrame(rows)
    if adjust_holm:
        summary["bootstrap_p_holm"] = holm_adjust(summary.bootstrap_p.to_numpy())
    return summary


def select_sampling_gof_replicates(draws, settings, saved_generator, full_model, *,
                                   response_exposure_kyr, tau=1.5, quadrature_order=4):
    """Check the saved nominal generator and keep all B_sampling rows, including failures."""
    # Reused residuals must come from the same fitted generator and integration settings.
    saved_tau = settings.get("history_tau_ka", settings.get("history_tau_kyr"))
    if float(saved_tau) != tau or int(settings.get("quadrature_order", 4)) != quadrature_order:
        raise ValueError("S4 model settings differ from the diagnostic model")
    saved = saved_generator.rename(columns=lambda name: name.removeprefix("beta__"))
    if len(saved) != 1 or set(saved.columns) != set(full_model.terms):
        raise ValueError("GOF generator must match the full-model terms")
    if not np.allclose(saved.loc[0, list(full_model.terms)].to_numpy(float), full_model.beta,
                       rtol=1e-7, atol=1e-8):
        raise ValueError("GOF generator differs from the nominal fit")
    columns = ["replicate_id", "scenario", "outer_id", "seed", "fit_valid", "invalid_reason",
               "response_exposure_kyr", "n_response_events", *GOF_STATISTICS, "residual_status"]
    # GOF uses nominal full-model sampling; chronology-mixture draws target a different ensemble.
    sampling = draws.loc[draws.scenario.eq("B_sampling"), columns].reset_index(drop=True)
    # Check all requested IDs, including failures, so reuse cannot silently shrink the denominator.
    expected_ids = np.arange(1, int(settings["n_point"]) + 1)
    if not np.array_equal(np.sort(sampling.replicate_id), expected_ids):
        raise ValueError("S4 nominal sampling ensemble is incomplete or has duplicate IDs")
    if not sampling.outer_id.eq(0).all():
        raise ValueError("GOF draws must use the nominal generator")
    if not np.allclose(sampling.response_exposure_kyr, response_exposure_kyr,
                       rtol=1e-10, atol=1e-10):
        raise ValueError("GOF draws use different response exposure")
    return sampling


def gof_results(event_features, integration_features, windows, full_model, replicates, *, catalogue_id):
    """Summarize observed residuals against nominal full-model refits."""
    # Observed and simulated statistics both use fitted intensities, accounting for parameter fitting.
    observed = residual_statistics(*rescaled_event_intervals(
        event_features, integration_features, windows, full_model))
    summary = bootstrap_summary(observed["statistics"], replicates)
    summary = summary.assign(catalogue_id=catalogue_id, n_response_events=len(event_features),
        response_exposure_kyr=float((windows.response_end_kyr_bp-windows.response_start_kyr_bp).sum()),
        calibration="simulate nominal full model, refit full, recompute statistics")
    return dict(gof_summary=summary, gof_replicates=replicates,
                rescaled_intervals=observed["intervals"], residual_segments=observed["segments"])
