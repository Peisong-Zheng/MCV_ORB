"""Refit saved chronological realizations without changing their ages or IDs."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
import pandas as pd
from toolbox import combined_likelihood as likelihood
from toolbox.point_process import PointProcessFitError

_CONTEXT = None


def _initialize(context):
    global _CONTEXT
    _CONTEXT=context


def _fit_ages(ages):
    events=_CONTEXT.events.copy()
    events[likelihood.EVENT_AGE_COLUMN]=ages
    if "event_age_ka" in events:
        events["event_age_ka"]=ages
    for name,segment in _CONTEXT.segments.items():
        x=ages[events.segment_id.eq(name).to_numpy()]
        if x.min()<segment.observation_start_kyr_bp or x.max()>segment.observation_end_kyr_bp:
            return {"fit_valid":False,"invalid_reason":f"outside_{name}_observation_support"}
    try:
        fit=likelihood.fit_catalogue(events,_CONTEXT)
    except PointProcessFitError as error:
        return {"fit_valid":False,"invalid_reason":f"numerical_fit: {error}"}
    beta=dict(zip(fit.full.terms,fit.full.beta))
    return {**fit.summary,"fit_valid":True,"invalid_reason":"",
            "beta_pre_phase_sin":beta["pre_phase_sin"],"beta_pre_phase_cos":beta["pre_phase_cos"],
            **{f"beta__{term}":value for term,value in beta.items()}}


def fit_realizations(context,realizations,age_columns,n_workers=1,show_progress=False):
    age=realizations.loc[:,age_columns].to_numpy(float)
    if not np.isfinite(age).all():
        raise ValueError("Chronology realizations must be finite")
    for name in context.segments:
        mask=context.events.segment_id.eq(name).to_numpy()
        if not np.all(np.diff(age[:,mask],axis=1)>0):
            raise ValueError("Event rank must be preserved; no sorting repair is applied")
    identifiers=[c for c in realizations if not c.startswith(('age_kyr_bp__','age_ka_bp__'))]
    if 'realization_id' not in identifiers or realizations.realization_id.duplicated().any():
        raise ValueError("Realizations require unique IDs")
    rows=[]
    if n_workers>1:
        pool=ProcessPoolExecutor(n_workers,mp_context=mp.get_context('spawn'),initializer=_initialize,initargs=(context,))
        outputs=pool.map(_fit_ages,age,chunksize=25)
    else:
        _initialize(context);pool=None;outputs=map(_fit_ages,age)
    try:
        for i,result in enumerate(outputs):
            rows.append({**realizations.iloc[i][identifiers].to_dict(),**result})
            if show_progress and (i+1)%1000==0:
                print(f"Refitted {i+1:,}/{len(age):,} exact-age realizations",flush=True)
    finally:
        if pool is not None:
            pool.shutdown()
    out=pd.DataFrame(rows)
    numeric_fail=out.invalid_reason.fillna('').str.startswith('numerical_fit').sum()
    diagnostics=dict(n_realizations=len(out),n_valid=int(out.fit_valid.sum()),n_invalid=int((~out.fit_valid).sum()),
                     n_numerical_failures=int(numeric_fail),age_input_reused=True,event_count_pattern_cache=False,
                     model_version=likelihood.MODEL_VERSION)
    return out,diagnostics


def unwrap_phase(values,center):
    return center+(np.asarray(values)-center+180)%360-180


def summarize(results,point):
    metrics=('gain_bits_per_event','LR_statistic','nominal_LR_p','delta_AIC_full_minus_reduced',
             'pre_phase_rate_ratio_max_vs_min','pre_phase_preferred_deg',
             'response_exposure_kyr','n_response_events')
    results=results.copy()
    for name in metrics:
        if name not in results: results[name]=np.nan
    valid=results.loc[results.fit_valid]
    row=dict(n_realizations=len(results),n_valid=len(valid),n_invalid=len(results)-len(valid),
             n_events=point['n_response_events'],catalogue_id=point['catalogue_id'],
             n_nominal_p_below_0p05=int(valid.nominal_LR_p.lt(.05).sum()),
             fraction_nominal_p_below_0p05=float(valid.nominal_LR_p.lt(.05).mean()),
             n_delta_AIC_below_zero=int(valid.delta_AIC_full_minus_reduced.lt(0).sum()),
             fraction_delta_AIC_below_zero=float(valid.delta_AIC_full_minus_reduced.lt(0).mean()),
             model_version=likelihood.MODEL_VERSION)
    for key in ('gain_bits_per_event','LR_statistic','nominal_LR_p','delta_AIC_full_minus_reduced',
                'pre_phase_rate_ratio_max_vs_min','pre_phase_preferred_deg'):
        value=valid[key].to_numpy(float)
        if key=='pre_phase_preferred_deg':
            value=unwrap_phase(value,point[key])
        for label,q in zip(('q025','median','q975'),(np.nanquantile(value,[.025,.5,.975]) if len(value) else [np.nan]*3)):
            row[f'{key}_{label}']=float(q)
        row[f'point_{key}']=point[key]
    row['response_exposure_kyr_min']=valid.response_exposure_kyr.min()
    row['response_exposure_kyr_max']=valid.response_exposure_kyr.max()
    row['n_response_events_min']=valid.n_response_events.min()
    row['n_response_events_max']=valid.n_response_events.max()
    row['phase_quantiles_unwrapped_about_point']=True
    return pd.DataFrame([row])


def compact_results(results):
    ids=[c for c in results if c.endswith('realization_id')]
    metrics=['fit_valid','invalid_reason','n_response_events','n_conditioning_events','response_exposure_kyr',
             'gain_bits_per_event','LR_statistic','nominal_LR_p','delta_AIC_full_minus_reduced',
             'pre_phase_preferred_deg','pre_phase_rate_ratio_max_vs_min','pre_phase_amplitude',
             'beta_history','beta_pre_phase_sin','beta_pre_phase_cos','all_models_converged','likelihood_nesting_ok']
    return results.loc[:,ids+[c for c in metrics if c in results]]
