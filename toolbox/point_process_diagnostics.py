"""Conditional point-process diagnostics and refitted bootstrap calibration.

Completed intervals are transformed by their fitted compensator. Fixed-time
termination and fitted parameters are reproduced by simulation and refitting;
ordinary KS reference p values are deliberately not used here. See Gerhard &
Gerstner (2010), NeurIPS, on point-process model checking by time rescaling.
"""

from concurrent.futures import ProcessPoolExecutor
import hashlib
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_distribution


GOF_STATISTICS = ("ks_uniform", "adjacent_dependence")


class DiagnosticFailure(RuntimeError):
    """A diagnosed numerical failure; retain its replicate without redrawing."""


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
    if not events_by_segment:
        raise ValueError("At least one observed segment is required")
    if set(events_by_segment) != set(cumulative_at_events_by_segment):
        raise ValueError("Events and cumulative integrals must have matching segments")
    if tail_integrals is not None and set(tail_integrals) != set(events_by_segment):
        raise ValueError("Tail integrals must cover every segment")

    interval_frames, segment_rows, uniform_samples = [], [], []
    adjacent_sum, n_pairs = 0.0, 0
    for segment_id, coordinates in events_by_segment.items():
        events = np.asarray(coordinates, dtype=float)
        cumulative = np.asarray(cumulative_at_events_by_segment[segment_id], dtype=float)
        if events.ndim != 1 or cumulative.shape != events.shape:
            raise ValueError("Each event needs one cumulative integral")
        if not np.isfinite(events).all() or not np.isfinite(cumulative).all():
            raise ValueError("Event coordinates and integrals must be finite")
        differences = np.diff(events)
        if len(differences) and not (np.all(differences > 0) or np.all(differences < 0)):
            raise ValueError("Event coordinates must be strictly ordered within a segment")
        transformed = np.diff(np.r_[0.0, cumulative])
        if np.any(transformed < 0):
            raise ValueError("Cumulative integrals must be nonnegative and nondecreasing")
        uniform = -np.expm1(-transformed)
        uniform_samples.append(uniform)
        local_sum = float(np.sum((uniform[:-1] - 0.5) * (uniform[1:] - 0.5)))
        local_pairs = max(len(uniform) - 1, 0)
        adjacent_sum += local_sum
        n_pairs += local_pairs

        tail = np.nan if tail_integrals is None else float(tail_integrals[segment_id])
        if tail_integrals is not None and (not np.isfinite(tail) or tail < 0):
            raise ValueError("Tail integrals must be finite and nonnegative")
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
    if p.ndim != 1 or np.any(np.isfinite(p) & ((p < 0) | (p > 1))):
        raise ValueError("Expected p values in [0, 1], or NaN for unresolved tests")
    if np.isinf(p).any():
        raise ValueError("Infinite p values are invalid")
    finite = np.isfinite(p)
    order = np.argsort(np.where(finite, p, 1.0), kind="stable")
    adjusted = np.empty(len(p))
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
    if replicates.empty or not statistics or len(set(statistics)) != len(statistics):
        raise ValueError("Bootstrap summary needs draws and distinct statistic names")
    valid_fit = np.ones(len(replicates), dtype=bool)
    if "fit_valid" in replicates:
        if not replicates.fit_valid.isin([True, False]).all():
            raise ValueError("Every replicate needs an explicit fit-valid status")
        valid_fit = replicates.fit_valid.to_numpy(bool)
    rows = []
    for name in statistics:
        point = float(observed[name])
        values = replicates[name].to_numpy(float)
        if not np.isfinite(point) or point < 0 or np.any(np.isfinite(values) & (values < 0)):
            raise ValueError("These discrepancy statistics must be nonnegative")
        valid = valid_fit & np.isfinite(values)
        total, accepted = len(values), int(valid.sum())
        missing = total - accepted
        exceeding = int(np.sum(values[valid] >= point))
        p_low = (1 + exceeding) / (total + 1)
        p_high = (1 + exceeding + missing) / (total + 1)
        p = p_low if not missing else np.nan
        ci_low = ci_high = mc_se = np.nan
        if not missing:
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


def run_refit_bootstrap(simulate, fit, statistics, *, n_replicates, seed,
                        statistic_names=GOF_STATISTICS):
    """Run simulate(rng), fit(events), statistics(events, fit) for every draw.

    Callbacks close over a fixed generator, conditioning events and support.
    Known numerical failures may raise DiagnosticFailure; programming/input
    errors propagate. Each replicate has its own seed and is never replaced.
    A legal empty catalogue must be handled by fit/statistics, not raised as a
    numerical failure. Use statistic_names=("LR_history",) for a history test.
    """
    if not isinstance(n_replicates, (int, np.integer)) or n_replicates < 1:
        raise ValueError("n_replicates must be a positive integer")
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rows = []
    for replicate in range(n_replicates):
        rng = np.random.default_rng(np.random.SeedSequence([seed, replicate]))
        row = dict(replicate_id=replicate + 1, seed=int(seed), fit_valid=True, invalid_reason="")
        try:
            events = simulate(rng)
            fitted = fit(events)
            values = statistics(events, fitted)
            for name in statistic_names:
                value = float(values[name])
                if not np.isfinite(value) or value < 0:
                    raise DiagnosticFailure(f"Invalid diagnostic statistic: {name}")
                row[name] = value
            if "status" in values:
                row["residual_status"] = str(values["status"])
        except DiagnosticFailure as error:
            row.update(fit_valid=False, invalid_reason=str(error))
            row.update({name: np.nan for name in statistic_names})
        rows.append(row)
    return pd.DataFrame(rows)


def _history_pair(design):
    from toolbox import combined_likelihood as likelihood

    full_terms = design.context.full_terms
    no_history_terms = tuple(term for term in full_terms if term != likelihood.HISTORY_TERM)
    if len(no_history_terms) != len(full_terms) - 1:
        raise ValueError("History comparison must remove exactly one fixed-tau term")
    no_history = likelihood.fit_terms(design, no_history_terms)
    start = np.zeros(len(full_terms) + 1)
    if np.isfinite(no_history.beta).all():
        for term, coefficient in zip(no_history.terms, no_history.beta):
            start[("intercept", *full_terms).index(term)] = coefficient
    full = likelihood.fit_terms(design, full_terms, start_beta=start)
    return no_history, full


_CONTEXT = _PREPARED = _KIND = None


def _initialize_model_bootstrap(context, generator, kind):
    from toolbox import combined_likelihood as likelihood

    global _CONTEXT, _PREPARED, _KIND
    _CONTEXT, _KIND = context, kind
    _PREPARED = likelihood.prepare_model_simulation(context, generator)


def _model_bootstrap_replicate(task):
    from toolbox import combined_likelihood as likelihood
    from toolbox.point_process import PointProcessFitError

    replicate, seed = task
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(_KIND == "gof"), replicate]))
    row = dict(replicate_id=replicate + 1, seed=seed, fit_valid=True, invalid_reason="")
    names = ("LR_history",) if _KIND == "history" else GOF_STATISTICS
    try:
        events = likelihood.simulate_prepared_events(_PREPARED, rng)
        design = likelihood.prepare_catalogue(events, _CONTEXT, fixed_support=True)
        row["n_response_events"] = len(design.event_frame)
        if _KIND == "history":
            no_history, full = _history_pair(design)
            row["LR_history"] = likelihood_ratio(full.log_likelihood, no_history.log_likelihood)
            row["beta_history"] = dict(zip(full.terms, full.beta))[likelihood.HISTORY_TERM]
            row["response_status"] = full.status
        else:
            full = likelihood.fit_terms(design, _CONTEXT.full_terms)
            result = residual_statistics(*likelihood.rescaled_event_intervals(design, full))
            row.update(result["statistics"])
            row.update(n_intervals=result["n_intervals"], n_adjacent_pairs=result["n_adjacent_pairs"],
                       residual_status=result["status"])
    except (PointProcessFitError, DiagnosticFailure) as error:
        row.update(fit_valid=False, invalid_reason=str(error))
        row.update({name: np.nan for name in names})
    return row


def run_model_bootstrap(context, generator, kind, *, n_replicates, seed, workers=1,
                         show_progress=True):
    """Calibrate the history LR or two full-model diagnostics with refits.

    The history generator has no history term; the GOF generator is the full
    point fit. Context, anchors and support are immutable during simulations.
    Failed seeds stay in the output; changing worker count does not change the
    data generated for any replicate. No full-model GOF is simulated implicitly.
    """
    from toolbox import combined_likelihood as likelihood

    if kind not in ("history", "gof"):
        raise ValueError("Bootstrap kind must be history or gof")
    if not isinstance(n_replicates, (int, np.integer)) or n_replicates < 1:
        raise ValueError("n_replicates must be a positive integer")
    if not isinstance(workers, (int, np.integer)) or workers < 1 or seed < 0:
        raise ValueError("Require a positive worker count and nonnegative seed")
    expected = set(("intercept", *context.full_terms))
    if kind == "history":
        expected.remove(likelihood.HISTORY_TERM)
    if set(generator.terms) != expected:
        raise ValueError("Bootstrap generator does not represent the requested null")
    tasks = [(replicate, seed) for replicate in range(n_replicates)]
    started = time.perf_counter()
    rows = []

    def collect(iterator):
        for row in iterator:
            rows.append(row)
            if show_progress and (len(rows) % 500 == 0 or len(rows) == n_replicates):
                print(f"{context.catalogue_id}: {kind} bootstrap {len(rows):,}/{n_replicates:,} "
                      f"({time.perf_counter() - started:.0f} s)", flush=True)

    if workers == 1:
        _initialize_model_bootstrap(context, generator, kind)
        collect(map(_model_bootstrap_replicate, tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                 initializer=_initialize_model_bootstrap,
                                 initargs=(context, generator, kind)) as pool:
            collect(pool.map(_model_bootstrap_replicate, tasks, chunksize=20))
    return pd.DataFrame(rows)


def run_history_test(context, *, n_bootstrap=4999, seed=20260914, workers=1,
                     show_progress=True):
    """Test beta_H=0 after retaining phase and background on identical support."""
    from toolbox import combined_likelihood as likelihood

    design = likelihood.prepare_catalogue(context.events, context, fixed_support=True)
    no_history, full = _history_pair(design)
    observed = {"LR_history": likelihood_ratio(full.log_likelihood, no_history.log_likelihood)}
    replicates = run_model_bootstrap(context, no_history, "history", n_replicates=n_bootstrap,
                                     seed=seed, workers=workers, show_progress=show_progress)
    summary = bootstrap_summary(observed, replicates, statistics=("LR_history",), adjust_holm=False)
    beta = dict(zip(full.terms, full.beta))[likelihood.HISTORY_TERM]
    summary = summary.assign(catalogue_id=context.catalogue_id, beta_history=beta,
                             history_rate_multiplier=np.exp(beta),
                             loglik_no_history=no_history.log_likelihood,
                             loglik_full=full.log_likelihood, n_response_events=len(design.event_frame),
                             response_exposure_kyr=context.response_exposure_kyr,
                             null_model="background + phase, no history",
                             parameter_domain="beta_H <= 0; zero is a boundary")
    coefficients = pd.DataFrame([dict(model_id=name, term=term, beta=coefficient)
        for name, model in (("no_history", no_history), ("full", full))
        for term, coefficient in zip(model.terms, model.beta)])
    return dict(history_test=summary, history_replicates=replicates, history_coefficients=coefficients)


def run_gof(context, *, replicates=None, n_bootstrap=1999, seed=20260915, workers=1,
            show_progress=True):
    """Assess full-model fit, or summarize diagnostics saved during full refits.

    Supplying replicates reuses a single nominal full-model experiment. Age
    ensembles, reduced-model simulations and mixed chronology generators must
    not be passed as that calibration sample.
    """
    from toolbox import combined_likelihood as likelihood

    design = likelihood.prepare_catalogue(context.events, context, fixed_support=True)
    full = likelihood.fit_terms(design, context.full_terms)
    observed = residual_statistics(*likelihood.rescaled_event_intervals(design, full))
    if replicates is None:
        replicates = run_model_bootstrap(context, full, "gof", n_replicates=n_bootstrap,
                                         seed=seed, workers=workers, show_progress=show_progress)
    if "scenario" in replicates and not replicates.scenario.eq("B_sampling").all():
        raise ValueError("GOF calibration must contain only nominal B_sampling full refits")
    summary = bootstrap_summary(observed["statistics"], replicates)
    summary = summary.assign(catalogue_id=context.catalogue_id,
                             n_response_events=len(design.event_frame),
                             response_exposure_kyr=context.response_exposure_kyr,
                             calibration="simulate nominal full model, refit full, recompute statistics")
    return dict(gof_summary=summary, gof_replicates=replicates,
                rescaled_intervals=observed["intervals"], residual_segments=observed["segments"])


def save_results(result, context, output_root, run_name, parameters, *, diagnostics_root=None):
    """Write model checks as CSV and prose without introducing default figures."""
    from toolbox import combined_likelihood as likelihood
    from toolbox.project_config import PROJECT_ROOT, LR04_XLSX, CO2_XLSX, PRE_TXT

    output_root = Path(output_root)
    data_dir = output_root / "data/processed" / run_name
    notes_dir = output_root / "experiment_note"
    diagnostic_dir = (output_root / "tests/diagnostics" / run_name if diagnostics_root is None
                      else Path(diagnostics_root) / run_name)
    for directory in (data_dir, notes_dir, diagnostic_dir):
        directory.mkdir(parents=True, exist_ok=True)
    for name, frame in result.items():
        frame.to_csv(data_dir / f"{name}.csv", index=False, float_format="%.12g")
    likelihood.support_table(context).to_csv(data_dir / "support.csv", index=False)
    metadata = dict(catalogue_id=context.catalogue_id, model_version=likelihood.MODEL_VERSION,
                    history_tau_kyr=context.history_tau_ka, initial_history=context.initial_history,
                    history_coefficient_domain="beta_H <= 0", quadrature_order=context.quadrature_order,
                    conditioning="original oldest observed event per segment, excluded from response",
                    fixed_response_exposure_kyr=context.response_exposure_kyr, **parameters)
    pd.DataFrame([dict(parameter=key, value=value) for key, value in metadata.items()]).to_csv(
        data_dir / "parameters_and_provenance.csv", index=False)
    inputs = [Path(__file__), Path(likelihood.__file__), PROJECT_ROOT / "toolbox/point_process.py",
              LR04_XLSX, CO2_XLSX, PRE_TXT]
    inputs.append(PROJECT_ROOT / "Barker2011/data/raw/Barker et al-2011-SOM.xls"
                  if "Barker2011" in context.segments else likelihood.EVENT_CATALOGUE_CSV)
    pd.DataFrame([dict(path=str(path.relative_to(PROJECT_ROOT)),
                       sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in inputs]).to_csv(
        diagnostic_dir / "input_code_sha256.csv", index=False)
    history_path, gof_path = data_dir / "history_test.csv", data_dir / "gof_summary.csv"
    sections = [f"Conditional model checks: {context.catalogue_id}",
        "\nThe history test compares background+phase with background+phase+history,",
        f"fixing the exponential decay time at {context.history_tau_ka:g} kyr.",
        "The null has beta_H=0; the alternative constrains beta_H<=0. The likelihood-ratio",
        "statistic is calibrated by simulating the null and refitting both models.",
        "An ordinary chi-square_1 reference is inappropriate at this parameter boundary.",
        "Every simulation retains the original oldest event and response interval in each segment."]
    if history_path.exists():
        history = pd.read_csv(history_path)
        columns = ["beta_history", "history_rate_multiplier", "observed", "bootstrap_p",
                   "n_bootstrap", "n_invalid"]
        sections.extend(["\nHistory result (observed = likelihood-ratio statistic):",
                         history.loc[:, columns].to_string(index=False, float_format=lambda x: f"{x:.6g}")])
    sections.extend(["\nFull-model goodness of fit uses two prespecified discrepancies:",
        "the KS distance of completed rescaled intervals to Uniform(0,1), and the absolute",
        "sum of adjacent (U-0.5) products divided by sqrt(max(1, number of within-segment pairs)).",
        "No adjacent pairs cross record gaps. A zero-response simulation has both statistics zero",
        "and remains in the reference distribution. Terminal no-event intervals are censored:",
        "their intensity integrals enter endpoint residuals but not the completed-interval CDF.",
        "Calibration simulates the nominal full model, refits it, then recomputes each statistic.",
        "Holm adjustment covers these two diagnostics within the catalogue. This is approximate",
        "parametric bootstrap calibration, not proof that the entire event model is correct."])
    if gof_path.exists():
        gof = pd.read_csv(gof_path)
        columns = ["statistic", "observed", "bootstrap_p", "bootstrap_p_holm", "n_bootstrap", "n_invalid"]
        sections.extend(["\nFull-model diagnostic results:",
                         gof.loc[:, columns].to_string(index=False, float_format=lambda x: f"{x:.6g}")])
    else:
        sections.append("\nFull-model diagnostics are pending the nominal full-model refit ensemble.")
    sections.extend(["\nPlus-one p=(1+exceedances)/(B+1). Failed simulations/refits are not replaced",
        "or treated as zero discrepancies. If failures remain, p is unresolved and its bounds",
        "show the range of unknown exceedance contributions. Finite-bootstrap intervals quantify",
        "simulation precision of the exceedance probability, not uncertainty of effect estimates.",
        "No chronology Monte Carlo or mixed chronology generator is included in this calibration.",
        "Results are supplied as CSV and prose; no figure is generated by this workflow."])
    (notes_dir / f"{run_name}_Methods_and_results.txt").write_text("\n".join(sections) + "\n", encoding="utf-8")
    return data_dir
