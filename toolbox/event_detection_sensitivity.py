"""Extra event-deletion stress tests, conditional on observed oldest events.

These scenarios remove members of the current catalogue. They neither recover
previously missed events nor estimate a detection probability. Fitting uses
only retained events, with the original conditioning anchors, support and
forcing scale held fixed by the caller. No latent detection model is implied.
"""

import hashlib
from pathlib import Path
import time

import numpy as np
import pandas as pd


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
    if events.empty or not set(required).issubset(events):
        raise ValueError("Need a nonempty event table with segment, age and event ID")
    if events[required].isna().any().any() or not events[id_column].is_unique:
        raise ValueError("Event IDs must be unique and required fields complete")
    ages = events[age_column].to_numpy(float)
    if not np.isfinite(ages).all():
        raise ValueError("Event ages must be finite kyr BP values")
    anchors = np.zeros(len(events), dtype=bool)
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

    The fit callback supplies G, LR and history/phase coefficients using the
    EFFECT_METRICS names. Phase, amplitude and max/min ratio are derived when
    both phase coefficients are finite. At zero amplitude the phase is NaN;
    a finite phase at small amplitude does not imply statistical identification.
    """
    normalized = []
    for source in (metrics, reference):
        required = {"gain_bits_per_event", "LR_statistic", "beta_history",
                    "beta_pre_phase_sin", "beta_pre_phase_cos"}
        if not required.issubset(source):
            raise ValueError(f"Fit metrics are missing: {sorted(required.difference(source))}")
        row = {name: float(source.get(name, np.nan)) for name in EFFECT_METRICS}
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
    required = {"scope", "drop_probability", "fit_valid", "n_deleted"}
    if not required.issubset(replicates) or replicates.empty:
        raise ValueError("Need nonempty replicate results and scenario identifiers")
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
                usable &= fit_valid
            low, median, high = np.quantile(values[usable], [0.025, 0.5, 0.975]) if usable.any() else (np.nan,) * 3
            n_invalid = int((~fit_valid).sum())
            rows.append(dict(scope=scope, drop_probability=probability, metric=metric,
                             n_replicates=len(group), n_valid=int(fit_valid.sum()),
                             n_invalid=n_invalid, n_finite=int(usable.sum()),
                             q025=low, median=median, q975=high,
                             status="ok" if not n_invalid else "incomplete_fits",
                             range_type="extra_deletion_scenario_range"))
    return pd.DataFrame(rows)


def run_deletion_sensitivity(events, fit, point_metrics, *, probabilities=(0.1, 0.2),
                             scopes=None, n_replicates=500, seed=20260912,
                             anchor_ids=None, segment_column="segment_id",
                             age_column="event_age_kyr_bp", id_column="event_id"):
    """Fit retained subsets, while a callback holds support and scaling fixed.

    fit(subset) returns a metrics dict. It must recompute history using only
    that subset. A fit_valid=False dict with an invalid_reason, or an explicit
    DeletionFitFailure, records a failure without replacing the mask. A legal
    zero-response case may have LR=0 and unidentified effects/G left NaN.

    scopes maps labels to eligible segment IDs; None means every segment.
    Default {"both": None} can be extended with {"MIS6_only": ("MIS6",)}.
    Private replicate seeds pair masks across probabilities and scopes. The
    returned boolean mask array follows the original events and replicate rows
    and can be stored once as a compressed diagnostic, without per-draw CSVs.
    """
    if not isinstance(n_replicates, (int, np.integer)) or n_replicates < 1:
        raise ValueError("n_replicates must be a positive integer")
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    probabilities = tuple(probabilities)
    if not probabilities or len(set(probabilities)) != len(probabilities):
        raise ValueError("Supply distinct deletion probabilities")
    if not np.isfinite(probabilities).all() or not np.all((np.asarray(probabilities) >= 0) & (np.asarray(probabilities) <= 1)):
        raise ValueError("Deletion probabilities must lie in [0, 1]")
    scopes = {"both": None} if scopes is None else dict(scopes)
    if not scopes:
        raise ValueError("At least one deletion scope is required")
    if not point_metrics.get("fit_valid", True):
        raise ValueError("The undeleted reference fit must be valid")
    reference = paired_metrics(point_metrics, point_metrics)
    rows, masks = [], []
    for scope, eligible_segments in scopes.items():
        anchors, eligible = _catalogue_masks(events, eligible_segments, anchor_ids,
                                             segment_column, age_column, id_column)
        for probability in probabilities:
            for replicate in range(n_replicates):
                rng = np.random.default_rng(np.random.SeedSequence([seed, replicate]))
                subset, membership = drop_events(
                    events, probability, rng, eligible_segments=eligible_segments,
                    anchor_ids=anchor_ids, segment_column=segment_column,
                    age_column=age_column, id_column=id_column)
                masks.append(membership.retained.to_numpy(bool))
                row = dict(scope=scope, drop_probability=probability,
                           replicate_id=replicate + 1, seed=int(seed),
                           n_source_events=len(events), n_conditioning_events=int(anchors.sum()),
                           n_eligible_events=int(eligible.sum()), n_deleted=len(events) - len(subset),
                           n_response_events=len(subset) - int(anchors.sum()),
                           fit_valid=True, invalid_reason="")
                row["response_status"] = "ok" if row["n_response_events"] else "no_response_events"
                try:
                    metrics = fit(subset)
                    if not metrics.get("fit_valid", True):
                        reason = metrics.get("invalid_reason", "")
                        if not str(reason).strip():
                            raise ValueError("An invalid fit must supply its failure reason")
                        raise DeletionFitFailure(str(reason))
                    if "response_exposure_kyr" in point_metrics:
                        exposure = float(metrics["response_exposure_kyr"])
                        if not np.isclose(exposure, point_metrics["response_exposure_kyr"], rtol=0, atol=1e-10):
                            raise ValueError("Deleting response events changed the fixed response support")
                        row["response_exposure_kyr"] = exposure
                    if "n_response_events" in metrics and metrics["n_response_events"] != row["n_response_events"]:
                        raise ValueError("Fitted response count disagrees with retained catalogue")
                    row.update(paired_metrics(metrics, point_metrics))
                except DeletionFitFailure as error:
                    row.update(fit_valid=False, invalid_reason=str(error))
                    row.update({name: np.nan for name in reference})
                rows.append(row)
    replicates = pd.DataFrame(rows)
    return dict(replicates=replicates, scenario_summary=summarize_scenarios(replicates),
                retained_masks=np.asarray(masks, dtype=bool),
                event_ids=events[id_column].to_numpy(copy=True),
                reference=pd.DataFrame([{name: reference[name] for name in EFFECT_METRICS}]))


def run_catalogue_analysis(context, *, scopes, n_replicates=500, seed=20260912,
                           show_progress=True):
    """Apply the extra-deletion scenarios to one continuous main-analysis context."""
    from toolbox import combined_likelihood as likelihood
    from toolbox.point_process import PointProcessFitError

    point = likelihood.fit_catalogue(context.events, context, fixed_support=True)
    started = time.perf_counter()
    n_fits = 0

    def metrics_for_fit(fitted):
        if not fitted.summary["all_models_converged"] or not fitted.summary["likelihood_nesting_ok"]:
            raise DeletionFitFailure("Nonconverged or nonnested deletion fit")
        beta = dict(zip(fitted.full.terms, fitted.full.beta))
        return dict(fitted.summary, beta_pre_phase_sin=beta["pre_phase_sin"],
                    beta_pre_phase_cos=beta["pre_phase_cos"])

    reference = metrics_for_fit(point)

    def fit_subset(subset):
        nonlocal n_fits
        n_fits += 1
        if show_progress and n_fits % 100 == 0:
            print(f"{context.catalogue_id}: deletion fit {n_fits:,} "
                  f"({time.perf_counter() - started:.0f} s)", flush=True)
        try:
            fitted = likelihood.fit_catalogue(subset, context, fixed_support=True)
        except PointProcessFitError as error:
            raise DeletionFitFailure(str(error)) from error
        return metrics_for_fit(fitted)

    result = run_deletion_sensitivity(context.events, fit_subset, reference, scopes=scopes,
                                      n_replicates=n_replicates, seed=seed)
    result["reference"] = pd.DataFrame([reference])
    result["parameters"] = dict(catalogue_id=context.catalogue_id,
        model_version=likelihood.MODEL_VERSION, seed=seed,
        n_replicates_per_scenario=n_replicates, deletion_probabilities="0.1;0.2",
        scopes=";".join(scopes), history_tau_kyr=context.history_tau_ka,
        initial_history=context.initial_history, history_coefficient_domain="beta_H <= 0",
        quadrature_order=context.quadrature_order,
        response_exposure_kyr=context.response_exposure_kyr,
        initialization="original oldest event in each segment fixed and retained",
        history="recomputed using only retained events",
        support_scaling="fixed to nominal undeleted catalogue",
        interpretation="extra independent deletion stress test; not inferred missingness or chronology uncertainty",
        scenario_pairing="same replicate uniforms across probabilities and eligible scopes",
        nested_bootstrap=False, elapsed_seconds=time.perf_counter() - started)
    return result


def save_results(result, context, output_root, run_name, *, diagnostics_root=None):
    """Write compact research tables, masks for reproducibility and English notes."""
    from toolbox import combined_likelihood as likelihood
    from toolbox.project_config import PROJECT_ROOT, LR04_XLSX, CO2_XLSX, PRE_TXT

    output_root = Path(output_root)
    data_dir = output_root / "data/processed" / run_name
    notes_dir = output_root / "experiment_note"
    diagnostics_dir = (output_root / "tests/diagnostics" / run_name if diagnostics_root is None
                       else Path(diagnostics_root) / run_name)
    for directory in (data_dir, notes_dir, diagnostics_dir):
        directory.mkdir(parents=True, exist_ok=True)
    for name in ("scenario_summary", "replicates", "reference"):
        result[name].to_csv(data_dir / f"{name}.csv", index=False, float_format="%.12g")
    likelihood.support_table(context).to_csv(data_dir / "support.csv", index=False)
    pd.DataFrame([dict(parameter=name, value=value) for name, value in result["parameters"].items()]).to_csv(
        data_dir / "parameters_and_provenance.csv", index=False)
    np.savez_compressed(diagnostics_dir / "retained_event_masks.npz",
                        event_ids=np.asarray(result["event_ids"], dtype=str),
                        event_ages_kyr_bp=context.events[likelihood.EVENT_AGE_COLUMN].to_numpy(),
                        retained=result["retained_masks"],
                        replicate_id=result["replicates"].replicate_id.to_numpy(),
                        scope=result["replicates"].scope.to_numpy(str),
                        drop_probability=result["replicates"].drop_probability.to_numpy())
    input_paths = [Path(__file__), Path(likelihood.__file__), PROJECT_ROOT / "toolbox/point_process.py",
                   LR04_XLSX, CO2_XLSX, PRE_TXT]
    input_paths.append(PROJECT_ROOT / "Barker2011/data/raw/Barker et al-2011-SOM.xls"
                       if "Barker2011" in context.segments else likelihood.EVENT_CATALOGUE_CSV)
    pd.DataFrame([dict(path=str(path.relative_to(PROJECT_ROOT)),
                       sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                  for path in input_paths]).to_csv(diagnostics_dir / "input_code_sha256.csv", index=False)
    rows = result["replicates"]
    summary = result["scenario_summary"]
    report_metrics = ["gain_bits_per_event", "phase_offset_deg", "pre_phase_rate_ratio_max_vs_min"]
    values = summary.loc[summary.metric.isin(report_metrics),
                         ["scope", "drop_probability", "metric", "median", "q025", "q975", "n_finite"]]
    note = f"""Extra event-deletion sensitivity: {context.catalogue_id}

Each noninitial response event is independently deleted with probability 0.1 or 0.2.
There are {result['parameters']['n_replicates_per_scenario']} replicate masks per scope/probability.
Every original conditioning event, response endpoint and forcing scale remains fixed.
Both continuous nested models are refitted, rebuilding history only from retained events.
Replicate seeds pair the deletion masks across probabilities and eligible segment scopes.
This is a stress test of additional loss, not a reconstruction or correction of pre-existing
missing events. It cannot rule out phase-dependent or climate-dependent detection bias.
No nested null bootstrap or chronology Monte Carlo is included in this experiment.

Valid fits: {int(rows.fit_valid.sum())}/{len(rows)}. Failed fits are retained with their masks
and explicit reasons; effect quantiles use finite supported estimates and show their denominator.
Quantiles describe the specified deletion scenarios, not sampling confidence intervals.
Phase offsets are circular differences in [-180, 180) degrees from the original estimate.
Phase coefficients and amplitude remain available because a weak effect has an uncertain peak.

{values.to_string(index=False, float_format=lambda x: f'{x:.6g}')}

Research tables: scenario_summary.csv, replicates.csv, reference.csv, support.csv and parameters_and_provenance.csv.
The compressed membership masks and input hashes are stored under tests/diagnostics.
No figure is generated by this experiment.
"""
    (notes_dir / f"{run_name}_Methods_and_results.txt").write_text(note, encoding="utf-8")
    return data_dir
