"""Extra event-deletion stress tests, conditional on observed oldest events.

These scenarios remove members of the current catalogue. They neither recover
previously missed events nor estimate a detection probability. Fitting uses
only retained events, with the original conditioning anchors, support and
forcing scale held fixed by the caller. No latent detection model is implied.
"""

import time

import numpy as np
import pandas as pd
from toolbox import event_model
from toolbox.model_stats import fit_summary
from toolbox.point_process import fit_point_process, PointProcessFitError
from toolbox.project_config import MODEL_VERSION


EFFECT_METRICS = (
    "gain_bits_per_event", "LR_statistic", "beta_history",
    "beta_pre_phase_sin", "beta_pre_phase_cos", "phase_amplitude",
    "pre_phase_preferred_deg", "pre_phase_rate_ratio_max_vs_min",
)


class DeletionFitFailure(RuntimeError):
    """A diagnosed fit failure; retain its deletion mask and do not redraw."""


def _catalogue_masks(events, eligible_segments, anchor_ids,
                     segment_column, age_column, id_column):
    required = [segment_column, age_column, id_column]
    if events.empty or events[required].isna().any().any() or not events[id_column].is_unique:
        raise ValueError("Event IDs must be unique and required fields complete")
    ages = events[age_column].to_numpy(float)
    if not np.isfinite(ages).all():
        raise ValueError("Event ages must be finite kyr BP values")
    anchors = np.zeros(len(events), dtype=bool)
    # Removing an anchor would redefine the response interval, conflating lost
    # events with a changed exposure. Each segment's original oldest event stays.
    for segment in events[segment_column].unique():
        indices = np.flatnonzero(events[segment_column].eq(segment).to_numpy())
        local_ages = ages[indices]
        if len(np.unique(local_ages)) != len(local_ages):
            raise ValueError("Duplicate within-segment ages need a scientific resolution")
        anchors[indices[np.argmax(local_ages)]] = True
    if anchor_ids is not None:
        specified = set(anchor_ids.values() if isinstance(anchor_ids, dict) else anchor_ids)
        actual = set(events.loc[anchors, id_column])
        if specified != actual:
            raise ValueError("Anchors must be the original oldest event in every segment")
    if eligible_segments is None:
        selected = np.ones(len(events), dtype=bool)
    else:
        eligible_segments = set(eligible_segments)
        if not eligible_segments.issubset(set(events[segment_column])):
            raise ValueError("An eligible segment is not present in the catalogue")
        selected = events[segment_column].isin(eligible_segments).to_numpy()
    return anchors, selected & ~anchors


def drop_events(events, drop_probability, rng, *, eligible_segments=None,
                anchor_ids=None, segment_column="segment_id",
                age_column="event_age_kyr_bp", id_column="event_id"):
    """Independently remove eligible response events and retain original columns.

    Return (retained_events, membership_table), both in the original row order.
    Every segment's largest BP age is an immutable conditioning anchor. No age,
    event ID, order, or noneligible event is changed. The caller must fit this
    subset from scratch; deleted events must not remain in its history term.
    """
    if not np.isfinite(drop_probability) or not 0 <= drop_probability <= 1:
        raise ValueError("Deletion probability must lie in [0, 1]")
    anchors, eligible = _catalogue_masks(events, eligible_segments, anchor_ids,
                                         segment_column, age_column, id_column)
    # Drawing once per original member permits paired scopes/probabilities
    # when the caller supplies the same private seed for those comparisons.
    deleted = eligible & (rng.random(len(events)) < drop_probability)
    membership = events.loc[:, [id_column, segment_column, age_column]].copy()
    membership["is_conditioning_event"] = anchors
    membership["eligible_for_deletion"] = eligible
    membership["retained"] = ~deleted
    return events.loc[~deleted].copy(), membership


def paired_metrics(metrics, reference):
    """Keep phase coefficients and circular offsets alongside scalar changes.

    The fitted pair supplies G, LR and history/phase coefficients using the
    EFFECT_METRICS names. Phase, amplitude and max/min ratio are derived when
    both phase coefficients are finite. At zero amplitude the phase is NaN;
    a finite phase at small amplitude does not imply statistical identification.
    """
    normalized = []
    for source in (metrics, reference):
        row = {name: float(source[name]) for name in EFFECT_METRICS[:5]}
        row.update({name: float(source.get(name, np.nan)) for name in EFFECT_METRICS[5:]})
        if np.isinf(list(row.values())).any():
            raise DeletionFitFailure("An effect estimate is infinite")
        sine, cosine = row["beta_pre_phase_sin"], row["beta_pre_phase_cos"]
        if np.isfinite([sine, cosine]).all():
            amplitude = float(np.hypot(sine, cosine))
            row["phase_amplitude"] = amplitude
            row["pre_phase_preferred_deg"] = (
                float(np.degrees(np.arctan2(sine, cosine)) % 360) if amplitude else np.nan)
            with np.errstate(over="ignore"):
                row["pre_phase_rate_ratio_max_vs_min"] = float(np.exp(2 * amplitude))
            if not np.isfinite(row["pre_phase_rate_ratio_max_vs_min"]):
                raise DeletionFitFailure("Phase rate ratio overflow")
        normalized.append(row)
    result, point = normalized
    for name in EFFECT_METRICS:
        if name == "pre_phase_preferred_deg":
            # Compare directions by the shortest signed offset, so 359 and 1
            # degrees differ by two degrees rather than almost a full cycle.
            phase_difference = result[name] - point[name]
            result["phase_offset_deg"] = (phase_difference + 180) % 360 - 180
        else:
            result[f"change__{name}"] = result[name] - point[name]
    return result


def summarize_scenarios(replicates):
    """Return per-scenario quantiles, explicitly conditional on supported fits.

    Phase is summarized as an offset around the undeleted reference, never as
    a linear quantile of angles across 0/360 degrees. Finite values and failed
    fits are counted for every metric; these are stress-test ranges, not CIs.
    """
    if replicates.empty:
        raise ValueError("No deletion replicates")
    if not replicates.fit_valid.isin([True, False]).all():
        raise ValueError("Every deletion fit requires explicit validity")
    metrics = ["n_deleted", "n_response_events", *EFFECT_METRICS,
               *[f"change__{name}" for name in EFFECT_METRICS
                 if name != "pre_phase_preferred_deg"], "phase_offset_deg"]
    metrics = [name for name in metrics if name in replicates and name != "pre_phase_preferred_deg"]
    rows = []
    for (scope, probability), group in replicates.groupby(["scope", "drop_probability"], sort=False):
        fit_valid = group.fit_valid.to_numpy(bool)
        for metric in metrics:
            values = group[metric].to_numpy(float)
            # Counts describe all masks, including masks whose fits failed.
            usable = np.isfinite(values)
            if metric not in ("n_deleted", "n_response_events"):
                usable = usable & fit_valid
            low, median, high = np.quantile(values[usable], [0.025, 0.5, 0.975]) if usable.any() else (np.nan,) * 3
            n_invalid = int((~fit_valid).sum())
            rows.append(dict(scope=scope, drop_probability=probability, metric=metric,
                             n_replicates=len(group), n_valid=int(fit_valid.sum()),
                             n_invalid=n_invalid, n_finite=int(usable.sum()),
                             q025=low, median=median, q975=high,
                             status="ok" if not n_invalid else "incomplete_fits",
                             range_type="extra_deletion_scenario_range"))
    return pd.DataFrame(rows)


def analyze_deletions(events, windows, forcings, phase_anchors, scaling,
                      reduced_terms, full_terms, *, catalogue_id,
                      probabilities=(0.1, 0.2), scopes=None, n_replicates=500,
                      seed=20260912, tau=1.5, initial_history=0.0,
                      quadrature_order=4, show_progress=True):
    """Refit retained events on the original windows and nominal forcing scales.

    Each replicate uses the same uniforms across deletion probabilities and
    segment scopes. Failed fits retain their masks and explicit reasons.
    """
    probabilities = tuple(probabilities)
    scopes = {"both": None} if scopes is None else dict(scopes)
    if n_replicates < 1 or not probabilities or not scopes:
        raise ValueError("Deletion experiments need scenarios and replicates")
    if (len(set(probabilities)) != len(probabilities) or not np.isfinite(probabilities).all()
            or not np.all((np.asarray(probabilities) >= 0) & (np.asarray(probabilities) <= 1))):
        raise ValueError("Deletion probabilities must be distinct and lie in [0, 1]")
    started = time.perf_counter()
    n_fits = 0

    def fit_retained(subset):
        # Rebuild history from retained events only. Exposure and predictor
        # scales remain those of the undeleted catalogue for a paired comparison.
        event_x, integral_x = event_model.build_design(
            subset, windows, forcings, phase_anchors, scaling, tau=tau,
            initial_history=initial_history, quadrature_order=quadrature_order,
        )
        if "mis6_segment" in full_terms or "mis6_segment" in reduced_terms:
            for frame in (event_x, integral_x):
                frame["mis6_segment"] = frame.segment_id.eq("MIS6").astype(float)
        reduced = fit_point_process(event_x[list(reduced_terms)], integral_x[list(reduced_terms)],
                                    integral_x.weight, reduced_terms)
        # Embed the reduced fit by term name; setting added terms to zero gives
        # exactly its intensity as the full model's starting point.
        start_beta = np.zeros(len(full_terms))
        if np.isfinite(reduced.beta).all():
            for term, beta in zip(reduced.terms, reduced.beta):
                if term in full_terms:
                    start_beta[full_terms.index(term)] = beta
        full = fit_point_process(event_x[list(full_terms)], integral_x[list(full_terms)],
                                 integral_x.weight, full_terms, start_beta=start_beta)
        summary = fit_summary(reduced, full, event_x, windows, n_source_events=len(subset),
                              catalogue_id=catalogue_id, tau=tau, initial_history=initial_history)
        if not len(event_x):
            raise DeletionFitFailure("No response events; effect is unidentified")
        beta = dict(zip(full.terms, full.beta))
        return dict(summary, beta_pre_phase_sin=beta["pre_phase_sin"],
                    beta_pre_phase_cos=beta["pre_phase_cos"])

    point_metrics = fit_retained(events)
    reference = paired_metrics(point_metrics, point_metrics)
    rows, masks = [], []
    for scope, eligible_segments in scopes.items():
        anchors, eligible = _catalogue_masks(events, eligible_segments, None,
                                             "segment_id", "event_age_kyr_bp", "event_id")
        for probability in probabilities:
            for replicate in range(n_replicates):
                # Excluding scope/probability from the seed pairs the random
                # uniforms across scenarios. For a fixed scope, higher deletion
                # rates remove supersets of the lower-rate masks.
                rng = np.random.default_rng(np.random.SeedSequence([seed, replicate]))
                deleted = eligible & (rng.random(len(events)) < probability)
                subset = events.loc[~deleted].copy()
                masks.append(~deleted)
                row = dict(scope=scope, drop_probability=probability,
                           replicate_id=replicate + 1, seed=int(seed),
                           n_source_events=len(events), n_conditioning_events=int(anchors.sum()),
                           n_eligible_events=int(eligible.sum()), n_deleted=len(events) - len(subset),
                           n_response_events=len(subset) - int(anchors.sum()),
                           fit_valid=True, invalid_reason="")
                row["response_status"] = "ok" if row["n_response_events"] else "no_response_events"
                try:
                    n_fits = n_fits + 1
                    if show_progress and n_fits % 100 == 0:
                        print(f"{catalogue_id}: deletion fit {n_fits:,} "
                              f"({time.perf_counter() - started:.0f} s)", flush=True)
                    metrics = fit_retained(subset)
                    row["response_exposure_kyr"] = float(metrics["response_exposure_kyr"])
                    row.update(paired_metrics(metrics, point_metrics))
                except (DeletionFitFailure, PointProcessFitError) as error:
                    # Keep this mask and its failure in the ensemble; redrawing
                    # until a fit succeeds would select a different deletion law.
                    row.update(fit_valid=False, invalid_reason=str(error))
                    row.update({name: np.nan for name in reference})
                rows.append(row)
    replicates = pd.DataFrame(rows)
    return dict(events=events, windows=windows, replicates=replicates,
        scenario_summary=summarize_scenarios(replicates),
        retained_masks=np.asarray(masks, dtype=bool), event_ids=events.event_id.to_numpy(copy=True),
        reference=pd.DataFrame([point_metrics]), parameters=dict(
            catalogue_id=catalogue_id, model_version=MODEL_VERSION, seed=seed,
            n_replicates_per_scenario=n_replicates,
            deletion_probabilities=";".join(str(value) for value in probabilities),
            scopes=";".join(scopes), history_tau_kyr=tau, initial_history=initial_history,
            history_coefficient_domain="beta_H <= 0", quadrature_order=quadrature_order,
            response_exposure_kyr=float((windows.response_end_kyr_bp - windows.response_start_kyr_bp).sum()),
            initialization="original oldest event in each segment fixed and retained",
            history="recomputed using only retained events", support_scaling="fixed to nominal undeleted catalogue",
            interpretation="extra independent deletion stress test; not inferred missingness or chronology uncertainty",
            scenario_pairing="same replicate uniforms across probabilities and eligible scopes",
            nested_bootstrap=False, elapsed_seconds=time.perf_counter() - started))
