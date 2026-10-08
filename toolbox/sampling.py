"""Refitted phase, history and full-model simulations with fixed random streams."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import time

import numpy as np
import pandas as pd

from toolbox import event_model, model_stats
from toolbox.point_process import fit_point_process, PointProcessFitError

HISTORY_TERM = "same_type_exponential_history"
_PHASE = _EFFECT = _HISTORY = _GOF = None


class InvalidEffectSimulation(RuntimeError):
    """Retain a failed full-model replicate under its original identity."""


def validate_effect_fit(reduced, full):
    # Setting the two phase coefficients to zero embeds the reduced model in
    # the full model, whose optimized likelihood therefore cannot be smaller.
    if full.log_likelihood < reduced.log_likelihood - 1e-7:
        raise InvalidEffectSimulation("Full/reduced likelihood nesting failed")


def _initialize_phase(arguments, prepared):
    global _PHASE
    _PHASE = arguments, prepared


def _phase_replicate(task):
    arguments, prepared = _PHASE
    windows = arguments['windows']
    reduced_terms, full_terms = arguments['reduced_terms'], arguments['full_terms']
    identifier, seed = task
    # Generate once per replicate. A numerical retry below must retain this
    # catalogue, its conditioning anchors and its original random-stream ID.
    events = event_model.simulate_prepared_events(prepared, np.random.default_rng(seed))
    row = dict(bootstrap_id=identifier, n_events_observation_support=len(events),
               n_events_response=len(events) - len(windows), fit_valid=False,
               status="fit_failed", solver_attempts=0, failure_reason="")
    for segment in windows.segment_id:
        row['n_events_response_' + segment] = int(events.segment_id.eq(segment).sum()) - 1
    # If needed, refine only the integral approximation on the same events;
    # drawing a replacement catalogue would select against difficult samples.
    for order in (arguments['quadrature_order'], 2 * arguments['quadrature_order']):
        row['solver_attempts'] = row['solver_attempts'] + 1
        try:
            event_x, integral_x = event_model.build_design(events, windows, arguments['forcings'],
                arguments['phase_anchors'], arguments['scaling'], tau=arguments['tau'],
                initial_history=arguments['initial_history'], quadrature_order=order)
            if 'mis6_segment' in full_terms:
                for frame in (event_x, integral_x):
                    frame['mis6_segment'] = frame.segment_id.eq('MIS6').astype(float)
            reduced = fit_point_process(event_x[list(reduced_terms)], integral_x[list(reduced_terms)],
                integral_x.weight, reduced_terms, nonpositive_terms=(HISTORY_TERM,))
            # Embed the reduced optimum in full-model coordinates, initially
            # assigning zero effect to the additional phase sine/cosine terms.
            start = np.zeros(len(full_terms))
            if np.isfinite(reduced.beta).all():
                for term, beta in zip(reduced.terms, reduced.beta):
                    start[full_terms.index(term)] = beta
            full = fit_point_process(event_x[list(full_terms)], integral_x[list(full_terms)],
                integral_x.weight, full_terms, nonpositive_terms=(HISTORY_TERM,), start_beta=start)
            # With no response events, both likelihood suprema are zero: the
            # LR remains defined even though finite coefficients are unavailable.
            zero = reduced.status == full.status == 'zero_events'
            gain = full.log_likelihood - reduced.log_likelihood
            if not np.isfinite(gain) or gain < -1e-7:
                raise PointProcessFitError("Nonfinite or nonnested likelihoods")
            gain = max(0., gain)
            row.update(loglik_reduced=reduced.log_likelihood, loglik_full=full.log_likelihood,
                       ll_gain_nats=gain, LR_statistic=2 * gain,
                       gain_bits_per_event=gain / (row['n_events_response'] * np.log(2)) if row['n_events_response'] else np.nan,
                       reduced_converged=reduced.converged, full_converged=full.converged,
                       likelihood_nesting_ok=True, fit_valid=True,
                       status='zero_events' if zero else 'finite_mle', failure_reason='', quadrature_order=order)
            return row
        except PointProcessFitError as error:
            row['failure_reason'] = str(error)
    row.update(loglik_reduced=np.nan, loglik_full=np.nan, ll_gain_nats=np.nan,
               LR_statistic=np.nan, gain_bits_per_event=np.nan,
               reduced_converged=False, full_converged=False, likelihood_nesting_ok=False)
    return row


def phase_bootstrap(events, windows, forcings, phase_anchors, scaling, reduced, full, *,
                    reduced_terms, full_terms, tau=1.5, initial_history=0., quadrature_order=4,
                    n_bootstrap=9999, seed=20260905, workers=1, show_progress=False):
    """Generate under BG and refit BG/full; retry only the same event sequence."""
    if tuple(reduced.terms) != tuple(reduced_terms) or tuple(full.terms) != tuple(full_terms):
        raise ValueError('Observed models must use the supplied BG/full columns')
    # The phase null retains climate and event history but omits phase forcing.
    # Both models are refitted so the LR includes parameter-estimation variation.
    prepared = event_model.prepare_model_simulation(events, windows, forcings, phase_anchors,
        scaling, reduced, tau=tau, initial_history=initial_history)
    arguments = dict(windows=windows, forcings=forcings, phase_anchors=phase_anchors, scaling=scaling,
        reduced_terms=tuple(reduced_terms), full_terms=tuple(full_terms), tau=tau,
        initial_history=initial_history, quadrature_order=quadrature_order)
    # Assign streams before dispatch; worker count and completion order then
    # cannot change the catalogue associated with a bootstrap ID.
    tasks = list(enumerate(np.random.SeedSequence(seed).spawn(n_bootstrap), start=1))
    started = time.perf_counter()
    pool = None
    if workers == 1:
        _initialize_phase(arguments, prepared)
        results = map(_phase_replicate, tasks)
    else:
        pool = ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn'),
            initializer=_initialize_phase, initargs=(arguments, prepared))
        results = pool.map(_phase_replicate, tasks, chunksize=10)
    rows = []
    try:
        for index, row in enumerate(results, 1):
            rows.append(row)
            if show_progress and (index % max(1, n_bootstrap // 10) == 0 or index == n_bootstrap):
                print(f'Bootstrap {index:,}/{n_bootstrap:,} ({time.perf_counter() - started:.1f} s)', flush=True)
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    table = pd.DataFrame(rows)
    return table, Counter(table.loc[~table.fit_valid, 'failure_reason'])


def effect_generators(events, observations, windows, forcings, phase_anchors, scaling, full,
                      draws, results, age_columns, *, reduced_terms, full_terms, n_outer, seed,
                      source_id_columns=(), tau=1.5, initial_history=0., quadrature_order=4):
    """Fit selected exact chronologies, retaining nominal predictor scaling."""
    valid_indices = np.flatnonzero(results.fit_valid.to_numpy(bool))
    # Select valid age realizations once, without replacement. Stream tag 100
    # keeps this outer selection separate from subsequent event simulations.
    selected = np.random.default_rng(np.random.SeedSequence([seed, 100])).choice(
        valid_indices, size=n_outer, replace=False)
    # Outer ID 0 is the nominal full model; positive IDs combine chronology
    # variation with event-sampling variation under that chronology's full fit.
    generators = {0: dict(model=full, events=events.copy(), windows=windows)}
    rows = []
    for outer_id, index in enumerate(selected, 1):
        source = draws.iloc[index]
        local_events = events.copy()
        local_events['event_age_kyr_bp'] = source[age_columns].to_numpy(float)
        # Moving ages also moves the oldest conditioning event and exposure;
        # the supplied nominal forcing scale stays fixed for comparability.
        local_windows = event_model.response_windows(local_events, observations)
        event_x, integral_x = event_model.build_design(local_events, local_windows, forcings, phase_anchors,
            scaling, tau=tau, initial_history=initial_history, quadrature_order=quadrature_order)
        if 'mis6_segment' in full_terms:
            for frame in (event_x, integral_x):
                frame['mis6_segment'] = frame.segment_id.eq('MIS6').astype(float)
        reduced = fit_point_process(event_x[list(reduced_terms)], integral_x[list(reduced_terms)],
            integral_x.weight, reduced_terms, nonpositive_terms=(HISTORY_TERM,))
        start = np.zeros(len(full_terms))
        if np.isfinite(reduced.beta).all():
            for term, beta in zip(reduced.terms, reduced.beta):
                start[full_terms.index(term)] = beta
        fitted = fit_point_process(event_x[list(full_terms)], integral_x[list(full_terms)],
            integral_x.weight, full_terms, nonpositive_terms=(HISTORY_TERM,), start_beta=start)
        validate_effect_fit(reduced, fitted)
        generators[outer_id] = dict(model=fitted, events=local_events, windows=local_windows)
        row = dict(outer_id=outer_id, source_row_index=int(index), age_realization_id=source.realization_id,
            n_response_events=len(event_x), response_exposure_kyr=float(
                (local_windows.response_end_kyr_bp - local_windows.response_start_kyr_bp).sum()))
        for window in local_windows.itertuples(index=False):
            row[f'response_end_kyr_bp__{window.segment_id}'] = window.response_end_kyr_bp
        row.update({column: source[column] for column in source_id_columns})
        row.update({f'beta__{term}': beta for term, beta in zip(fitted.terms, fitted.beta)})
        rows.append(row)
    return generators, pd.DataFrame(rows)


def _initialize_effect(arguments, generators):
    global _EFFECT
    _EFFECT = arguments, generators, {}


def _effect_replicate(task):
    from toolbox.point_process_diagnostics import residual_statistics
    arguments, generators, prepared_by_outer = _EFFECT
    scenario, outer_id, inner_id, seed = task
    generator = generators[outer_id]
    windows = generator['windows']
    if outer_id not in prepared_by_outer:
        prepared_by_outer[outer_id] = event_model.prepare_model_simulation(generator['events'], windows,
            arguments['forcings'], arguments['phase_anchors'], arguments['scaling'], generator['model'],
            tau=arguments['tau'], initial_history=arguments['initial_history'])
    # Each scenario/chronology/replicate owns a fixed stream independently of
    # process scheduling and of how many events earlier replicates generated.
    rng = np.random.default_rng(np.random.SeedSequence([seed, scenario, outer_id, inner_id]))
    row = dict(scenario='B_sampling' if scenario == 1 else 'C_joint', outer_id=outer_id,
        inner_id=inner_id, replicate_id=inner_id, seed=seed, fit_valid=True, invalid_reason='',
        response_exposure_kyr=float((windows.response_end_kyr_bp - windows.response_start_kyr_bp).sum()))
    reduced_terms, full_terms = arguments['reduced_terms'], arguments['full_terms']
    try:
        events = event_model.simulate_prepared_events(prepared_by_outer[outer_id], rng)
        row['n_observation_events'] = len(events)
        event_x, integral_x = event_model.build_design(events, windows, arguments['forcings'],
            arguments['phase_anchors'], arguments['scaling'], tau=arguments['tau'],
            initial_history=arguments['initial_history'], quadrature_order=arguments['quadrature_order'])
        if 'mis6_segment' in full_terms:
            for frame in (event_x, integral_x):
                frame['mis6_segment'] = frame.segment_id.eq('MIS6').astype(float)
        reduced = fit_point_process(event_x[list(reduced_terms)], integral_x[list(reduced_terms)],
            integral_x.weight, reduced_terms, nonpositive_terms=(HISTORY_TERM,))
        start = np.zeros(len(full_terms))
        if np.isfinite(reduced.beta).all():
            for term, beta in zip(reduced.terms, reduced.beta):
                start[full_terms.index(term)] = beta
        full = fit_point_process(event_x[list(full_terms)], integral_x[list(full_terms)], integral_x.weight,
            full_terms, nonpositive_terms=(HISTORY_TERM,), start_beta=start)
        # Keep event-free realizations in the ensemble, but do not assign an
        # arbitrary phase or rate ratio when no response identifies the effect.
        if not len(event_x):
            row.update(n_response_events=0, effect_identified=False, phase_deg=np.nan, rate_ratio=np.nan,
                full_log_likelihood=0., reduced_log_likelihood=0.)
            row.update({f'beta__{term}': np.nan for term in full.terms})
            if scenario == 1:
                row.update(ks_uniform=0., adjacent_dependence=0., residual_status='no_response_events')
            return row
        validate_effect_fit(reduced, full)
        row['effect_identified'] = True
        row['n_response_events'] = len(event_x)
        row.update({f'beta__{term}': beta for term, beta in zip(full.terms, full.beta)})
        phase, ratio = model_stats.phase_and_ratio(model_stats.phase_coefficients(full))
        row.update(phase_deg=float(phase), rate_ratio=float(ratio), full_log_likelihood=full.log_likelihood,
                   reduced_log_likelihood=reduced.log_likelihood)
        # Nominal full-model refits also supply the fitted-residual null for GOF;
        # joint chronology draws describe a different uncertainty experiment.
        if scenario == 1:
            residuals = residual_statistics(*event_model.rescaled_event_intervals(event_x, integral_x, windows, full))
            row.update(residuals['statistics'])
            row['residual_status'] = residuals['status']
    # Failed draws retain their original IDs and are reported, never redrawn.
    except (InvalidEffectSimulation, PointProcessFitError) as error:
        row.update(fit_valid=False, invalid_reason=str(error))
    return row


def sample_effect(generators, forcings, phase_anchors, scaling, *, reduced_terms, full_terms,
                  n_point, n_inner, seed, workers=1, show_progress=True, tau=1.5,
                  initial_history=0., quadrature_order=4):
    """Sample nominal and selected-chronology full models with equal inner weights."""
    if any(tuple(g['model'].terms) != tuple(full_terms) for g in generators.values()):
        raise ValueError('The effect generator must be the fitted full model')
    arguments = dict(forcings=forcings, phase_anchors=phase_anchors, scaling=scaling,
        reduced_terms=tuple(reduced_terms), full_terms=tuple(full_terms), tau=tau,
        initial_history=initial_history, quadrature_order=quadrature_order)
    # B varies event sampling at the nominal chronology. C uses the same number
    # of inner samples for every selected chronology, giving equal outer weight.
    tasks = [(1, 0, i, seed) for i in range(1, n_point + 1)]
    joint_tasks = [(2, outer, i, seed) for outer in sorted(generators) if outer != 0 for i in range(1, n_inner + 1)]
    tasks = tasks + joint_tasks
    pool = None
    if workers == 1:
        _initialize_effect(arguments, generators)
        results = map(_effect_replicate, tasks)
    else:
        pool = ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn'),
            initializer=_initialize_effect, initargs=(arguments, generators))
        results = pool.map(_effect_replicate, tasks, chunksize=20)
    rows = []
    try:
        for i, row in enumerate(results, 1):
            rows.append(row)
            if show_progress and (i % 1000 == 0 or i == len(tasks)):
                print(f'Full-model simulation {i:,}/{len(tasks):,}', flush=True)
    finally:
        if pool is not None:
            pool.shutdown()
    return pd.DataFrame(rows)


def history_models(event_x, integral_x, full_terms):
    """Fit phase/background first, then its history alternative from that optimum."""
    full_terms = tuple(full_terms)
    no_history_terms = tuple(term for term in full_terms if term != HISTORY_TERM)
    if len(no_history_terms) != len(full_terms) - 1:
        raise ValueError('History comparison must remove exactly one fixed-tau term')
    no_history = fit_point_process(event_x[list(no_history_terms)], integral_x[list(no_history_terms)],
        integral_x.weight, no_history_terms, nonpositive_terms=())
    # The alternative starts exactly at beta_H = 0, the boundary null, while
    # retaining the fitted climate and phase effects as its initial values.
    start = np.zeros(len(full_terms))
    if np.isfinite(no_history.beta).all():
        for term, beta in zip(no_history.terms, no_history.beta):
            start[full_terms.index(term)] = beta
    full = fit_point_process(event_x[list(full_terms)], integral_x[list(full_terms)], integral_x.weight,
        full_terms, nonpositive_terms=(HISTORY_TERM,), start_beta=start)
    return no_history, full


def _initialize_history(arguments, prepared):
    global _HISTORY
    _HISTORY = arguments, prepared


def _history_replicate(task):
    from toolbox.point_process_diagnostics import likelihood_ratio, DiagnosticFailure
    arguments, prepared = _HISTORY
    replicate, seed = task
    # Stream tag 0 distinguishes history calibration from GOF (tag 1).
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0, replicate]))
    row = dict(replicate_id=replicate + 1, seed=seed, fit_valid=True, invalid_reason='')
    try:
        events = event_model.simulate_prepared_events(prepared, rng)
        event_x, integral_x = event_model.build_design(events, arguments['windows'], arguments['forcings'],
            arguments['phase_anchors'], arguments['scaling'], tau=arguments['tau'],
            initial_history=arguments['initial_history'], quadrature_order=arguments['quadrature_order'])
        if 'mis6_segment' in arguments['full_terms']:
            for frame in (event_x, integral_x):
                frame['mis6_segment'] = frame.segment_id.eq('MIS6').astype(float)
        row['n_response_events'] = len(event_x)
        no_history, full = history_models(event_x, integral_x, arguments['full_terms'])
        row['LR_history'] = likelihood_ratio(full.log_likelihood, no_history.log_likelihood)
        row['beta_history'] = full.beta[full.terms.index(HISTORY_TERM)]
        row['response_status'] = full.status
    except (PointProcessFitError, DiagnosticFailure) as error:
        row.update(fit_valid=False, invalid_reason=str(error), LR_history=np.nan)
    return row


def history_bootstrap(events, windows, forcings, phase_anchors, scaling, *, full_terms, catalogue_id,
                      n_bootstrap=4999, seed=20260914, workers=1, show_progress=True,
                      tau=1.5, initial_history=0., quadrature_order=4):
    """Calibrate the beta_H=0 boundary while retaining phase and background."""
    from toolbox.point_process_diagnostics import likelihood_ratio, bootstrap_summary
    event_x, integral_x = event_model.build_design(events, windows, forcings, phase_anchors, scaling,
        tau=tau, initial_history=initial_history, quadrature_order=quadrature_order)
    if 'mis6_segment' in full_terms:
        for frame in (event_x, integral_x):
            frame['mis6_segment'] = frame.segment_id.eq('MIS6').astype(float)
    no_history, full = history_models(event_x, integral_x, full_terms)
    # Generate at beta_H = 0 while retaining phase and climate. The null lies
    # on the beta_H <= 0 boundary, so calibrate LR by simulation rather than
    # assuming an unconstrained one-parameter chi-squared reference.
    prepared = event_model.prepare_model_simulation(events, windows, forcings, phase_anchors, scaling,
        no_history, tau=tau, initial_history=initial_history)
    arguments = dict(windows=windows, forcings=forcings, phase_anchors=phase_anchors, scaling=scaling,
        full_terms=tuple(full_terms), tau=tau, initial_history=initial_history, quadrature_order=quadrature_order)
    tasks = [(i, seed) for i in range(n_bootstrap)]
    pool = None
    if workers == 1:
        _initialize_history(arguments, prepared)
        results = map(_history_replicate, tasks)
    else:
        pool = ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn'),
            initializer=_initialize_history, initargs=(arguments, prepared))
        results = pool.map(_history_replicate, tasks, chunksize=20)
    rows = []
    try:
        for row in results:
            rows.append(row)
            if show_progress and (len(rows) % 500 == 0 or len(rows) == n_bootstrap):
                print(f'{catalogue_id}: history bootstrap {len(rows):,}/{n_bootstrap:,}', flush=True)
    finally:
        if pool is not None:
            pool.shutdown()
    replicates = pd.DataFrame(rows)
    observed = {'LR_history': likelihood_ratio(full.log_likelihood, no_history.log_likelihood)}
    summary = bootstrap_summary(observed, replicates, statistics=('LR_history',), adjust_holm=False)
    beta = full.beta[full.terms.index(HISTORY_TERM)]
    summary = summary.assign(catalogue_id=catalogue_id, beta_history=beta, history_rate_multiplier=np.exp(beta),
        loglik_no_history=no_history.log_likelihood, loglik_full=full.log_likelihood,
        n_response_events=len(event_x), response_exposure_kyr=float(
            (windows.response_end_kyr_bp - windows.response_start_kyr_bp).sum()),
        null_model='background + phase, no history', parameter_domain='beta_H <= 0; zero is a boundary')
    coefficients = pd.DataFrame([dict(model_id=name, term=term, beta=beta)
        for name, model in (('no_history', no_history), ('full', full)) for term, beta in zip(model.terms, model.beta)])
    return dict(history_test=summary, history_replicates=replicates, history_coefficients=coefficients)


def _initialize_gof(arguments, prepared):
    global _GOF
    _GOF = arguments, prepared


def _gof_replicate(task):
    from toolbox.point_process_diagnostics import residual_statistics, GOF_STATISTICS
    arguments, prepared = _GOF
    replicate, seed = task
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1, replicate]))
    row = dict(replicate_id=replicate + 1, seed=seed, fit_valid=True, invalid_reason='')
    try:
        events = event_model.simulate_prepared_events(prepared, rng)
        event_x, integral_x = event_model.build_design(events, arguments['windows'], arguments['forcings'],
            arguments['phase_anchors'], arguments['scaling'], tau=arguments['tau'],
            initial_history=arguments['initial_history'], quadrature_order=arguments['quadrature_order'])
        terms = arguments['full_terms']
        if 'mis6_segment' in terms:
            for frame in (event_x, integral_x):
                frame['mis6_segment'] = frame.segment_id.eq('MIS6').astype(float)
        row['n_response_events'] = len(event_x)
        # Refit before rescaling: observed residuals use estimated parameters,
        # so calibration must reproduce that estimation step in every replicate.
        full = fit_point_process(event_x[list(terms)], integral_x[list(terms)], integral_x.weight,
            terms, nonpositive_terms=(HISTORY_TERM,))
        residuals = residual_statistics(*event_model.rescaled_event_intervals(event_x, integral_x, arguments['windows'], full))
        row.update(residuals['statistics'])
        row.update(n_intervals=residuals['n_intervals'], n_adjacent_pairs=residuals['n_adjacent_pairs'],
            residual_status=residuals['status'])
    except PointProcessFitError as error:
        row.update(fit_valid=False, invalid_reason=str(error))
        row.update({name: np.nan for name in GOF_STATISTICS})
    return row


def gof_bootstrap(events, windows, forcings, phase_anchors, scaling, full, *, full_terms,
                  n_bootstrap=1999, seed=20260915, workers=1, show_progress=True,
                  tau=1.5, initial_history=0., quadrature_order=4):
    """Generate under the full model, then refit it for residual calibration."""
    if tuple(full.terms) != tuple(full_terms):
        raise ValueError('GOF generator does not represent the full model')
    # GOF asks whether the fitted full process can reproduce its residuals;
    # unlike the phase/history tests, no predictor is removed from the generator.
    prepared = event_model.prepare_model_simulation(events, windows, forcings, phase_anchors, scaling,
        full, tau=tau, initial_history=initial_history)
    arguments = dict(windows=windows, forcings=forcings, phase_anchors=phase_anchors, scaling=scaling,
        full_terms=tuple(full_terms), tau=tau, initial_history=initial_history, quadrature_order=quadrature_order)
    tasks = [(i, seed) for i in range(n_bootstrap)]
    pool = None
    if workers == 1:
        _initialize_gof(arguments, prepared)
        results = map(_gof_replicate, tasks)
    else:
        pool = ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn'),
            initializer=_initialize_gof, initargs=(arguments, prepared))
        results = pool.map(_gof_replicate, tasks, chunksize=20)
    rows = []
    try:
        for row in results:
            rows.append(row)
            if show_progress and (len(rows) % 500 == 0 or len(rows) == n_bootstrap):
                print(f'Full-model GOF bootstrap {len(rows):,}/{n_bootstrap:,}', flush=True)
    finally:
        if pool is not None:
            pool.shutdown()
    return pd.DataFrame(rows)
