"""Continuous conditional event likelihood shared by the climate studies.

BP ages increase into the past. Each record conditions on its exact oldest
observed event; there is no exposure or event history across record gaps.
Within a segment, elapsed_kyr = anchor_age_kyr_bp - age_kyr_bp runs forward.
Public data and forcing interpolants retain BP1950 coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd

from toolbox import event_inputs, orbital_phase, point_process
from toolbox.model_stats import nested_likelihood_metrics
from toolbox.project_config import PROJECT_ROOT, LR04_XLSX, CO2_XLSX, PRE_TXT, ORBITAL_DRIVER_SETTINGS

EVENT_CATALOGUE_CSV = PROJECT_ROOT / "data/curated/ngrip_mis6_warming_events.csv"
OBSERVATION_SEGMENTS_CSV = PROJECT_ROOT / "data/curated/observation_segments.csv"
EVENT_AGE_COLUMN = "event_age_kyr_bp"
SEGMENT_IDS = ("NGRIP", "MIS6")
SEGMENT_TERM = "mis6_segment"
HISTORY_TERM = "same_type_exponential_history"
CATALOGUE_ID = "ngrip_warming_plus_mis6"
DEFAULT_HISTORY_TAU_KA = 1.5
MODEL_VERSION = "continuous_exponential_inhibition_2026-09-12"
REDUCED_TERMS = (HISTORY_TERM, "lr04_scaled", "co2_scaled", SEGMENT_TERM)
FULL_TERMS = REDUCED_TERMS + ("pre_phase_sin", "pre_phase_cos")
RESOLUTION_COVARIATE_INCLUDED = False
EVENT_COLUMNS = ("event_id", "event_label", EVENT_AGE_COLUMN, "segment_id", "source_record",
                 "source_event_label", "data_source", "label_source", "timing_method", "source_table")

@dataclass(frozen=True)
class SegmentContext:
    segment_id: str
    observation_start_kyr_bp: float
    observation_end_kyr_bp: float
    response_start_kyr_bp: float
    response_end_kyr_bp: float
    anchor_age_kyr_bp: float

@dataclass(frozen=True)
class LikelihoodContext:
    observation_segments: pd.DataFrame
    events: pd.DataFrame
    segments: dict
    forcings: dict
    scaling: dict
    phase_extrema: tuple
    history_tau_ka: float = 1.5
    initial_history: float = 0.0
    quadrature_order: int = 4
    catalogue_id: str = CATALOGUE_ID
    reduced_terms: tuple = REDUCED_TERMS
    full_terms: tuple = FULL_TERMS
    derived_terms: dict | None = None

    @property
    def response_exposure_kyr(self):
        return sum(s.response_end_kyr_bp-s.response_start_kyr_bp for s in self.segments.values())

@dataclass
class CatalogueDesign:
    event_frame: pd.DataFrame
    integration_frame: pd.DataFrame
    all_events: pd.DataFrame
    context: LikelihoodContext

    @property
    def weights(self):
        return self.integration_frame.weight.to_numpy(float)

@dataclass
class CombinedLikelihoodFit:
    design: CatalogueDesign
    reduced: object
    full: object
    summary: dict

    @property
    def context(self):
        return self.design.context


def _require_columns(frame, columns, name):
    missing=set(columns).difference(frame.columns)
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



def load_observation_segments(path=OBSERVATION_SEGMENTS_CSV):
    frame = pd.read_csv(path)
    cols = ["segment_id", "observation_start_kyr_bp", "observation_end_kyr_bp"]
    _require_columns(frame, cols, "Observation segments")
    if frame.segment_id.duplicated().any():
        raise ValueError("Observation segment IDs must be unique")
    for row in frame.itertuples():
        if not np.isfinite([row.observation_start_kyr_bp,row.observation_end_kyr_bp]).all() or row.observation_start_kyr_bp>=row.observation_end_kyr_bp:
            raise ValueError("Observation support must be finite and positive")
    return frame.drop(columns=[c for c in frame if c.startswith("common_response_")])


@lru_cache(maxsize=8)
def _source_forcings(lr04_path, co2_path, precession_path):
    # Read native samples once; interpolation and scaling have separate roles.
    lr = pd.read_excel(lr04_path)
    co = pd.read_excel(co2_path, sheet_name="Sheet2")
    lr_age, lr_value = event_inputs.clean_series(lr[event_inputs.find_column(lr.columns,"time")].to_numpy(float),
                                                lr[event_inputs.find_column(lr.columns,"d18o")].to_numpy(float))
    co_age, co_value = event_inputs.clean_series(co[event_inputs.find_column(co.columns,"gasage")].to_numpy(float)/1000,
                                                co[event_inputs.find_column(co.columns,"co2")].to_numpy(float))
    settings = dict(ORBITAL_DRIVER_SETTINGS["pre"], path=precession_path)
    phase = orbital_phase.build_phase_series("pre",settings)
    forcings={"lr04":(lr_age,lr_value), "co2":(co_age,co_value),
              "precession_index":(phase.series.age_ka.to_numpy(float), phase.series.value.to_numpy(float))}
    anchors=(phase.extrema.age_ka.to_numpy(float),phase.extrema.anchor_phase_unwrapped_rad.to_numpy(float))
    return forcings,anchors


def _segments_for_events(events, observation_segments):
    segments={}
    if set(events.segment_id)!=set(observation_segments.segment_id):
        raise ValueError("Every observation segment must have conditioning events")
    for row in observation_segments.itertuples():
        ages=events.loc[events.segment_id.eq(row.segment_id), EVENT_AGE_COLUMN].to_numpy(float)
        if not np.isfinite(ages).all() or len(np.unique(ages))!=len(ages):
            raise ValueError("Event ages must be finite and unique within each segment")
        young=float(row.observation_start_kyr_bp); old=float(row.observation_end_kyr_bp)
        if np.any(ages<young) or np.any(ages>old):
            raise ValueError(f"An event lies outside the {row.segment_id} observation window")
        anchor=float(ages.max())
        if anchor<=young:
            raise ValueError("No response exposure after the conditioning event")
        segments[row.segment_id]=SegmentContext(row.segment_id,young,old,young,anchor,anchor)
    return segments


def _scale_forcing(source, segments):
    age,value=source
    total=integral=0.0
    extrema=[]
    for s in segments.values():
        lo,hi=s.response_start_kyr_bp,s.response_end_kyr_bp
        knots=np.r_[lo,age[(age>lo)&(age<hi)],hi]
        values=event_inputs.interpolate_checked(knots,age,value,context="forcing response support")
        integral+=np.trapezoid(values,knots)
        total+=hi-lo
        extrema.extend([values.min(),values.max()])
    minimum,maximum=float(min(extrema)),float(max(extrema))
    return dict(mean=integral/total,min=minimum,max=maximum,range=maximum-minimum or 1.0)


def build_context(observation_segments=None, *, events=None, history_tau_ka=DEFAULT_HISTORY_TAU_KA,
                  initial_history=0.0,quadrature_order=4,catalogue_id=CATALOGUE_ID,
                  lr04_path=LR04_XLSX,co2_path=CO2_XLSX,precession_path=PRE_TXT):
    if history_tau_ka<=0 or initial_history<0:
        raise ValueError("History decay must be positive; initial history must be nonnegative")
    events=load_event_catalogue() if events is None else events.copy()
    observations=load_observation_segments() if observation_segments is None else observation_segments.copy()
    segments=_segments_for_events(events,observations)
    sources,phase=_source_forcings(Path(lr04_path),Path(co2_path),Path(precession_path))
    scaling={name:_scale_forcing(source,segments) for name,source in sources.items() if name!="precession_index"}
    reduced=REDUCED_TERMS if "MIS6" in segments and len(segments)>1 else tuple(t for t in REDUCED_TERMS if t!=SEGMENT_TERM)
    return LikelihoodContext(observations,events,segments,dict(sources),scaling,phase,
                             history_tau_ka,initial_history,quadrature_order,catalogue_id,
                             reduced,reduced+("pre_phase_sin","pre_phase_cos"),{})


def load_barker_events(event_definition="variable_threshold"):
    path=PROJECT_ROOT/"Barker2011/data/raw/Barker et al-2011-SOM.xls"
    raw=pd.read_excel(path,sheet_name="Sheet1",header=8)
    pick={"variable_threshold":"DO pick variable threshold","fixed_threshold":"DO pick"}[event_definition]
    frame=raw[["SpeloAge (kyr).1",pick]].apply(pd.to_numeric,errors="coerce")
    frame.columns=["event_age_ka","pick_value"]
    frame["source_row"]=raw.index
    frame=frame.loc[frame.pick_value.eq(1)&frame.event_age_ka.between(0,400)].sort_values("event_age_ka").reset_index(drop=True)
    frame["source_excel_row"]=frame.source_row+10
    frame["event_index"]=np.arange(1,len(frame)+1)
    frame["event_id"]=[f"Barker_S3_{row:03d}" for row in frame.source_excel_row]
    frame["event_label"]=[f"Barker event {i}" for i in frame.event_index]
    frame["segment_id"]="Barker2011"
    frame[EVENT_AGE_COLUMN]=frame.event_age_ka
    frame["event_definition"]=event_definition
    frame["source"]=str(path.relative_to(PROJECT_ROOT))
    expected=70 if event_definition=="variable_threshold" else 59
    if len(frame)!=expected:
        raise ValueError("Unexpected Barker source event count")
    return frame


def build_barker_context(event_definition="variable_threshold",**kwargs):
    events=load_barker_events(event_definition)
    support=pd.DataFrame([dict(segment_id="Barker2011",observation_start_kyr_bp=0.,observation_end_kyr_bp=400.)])
    return build_context(support,events=events,catalogue_id=f"barker_{event_definition}_speleo_0_400",**kwargs)


def add_forcing(context,name,ages,values):
    ages,values=event_inputs.clean_series(ages,values,context=name)
    return replace(context,forcings={**context.forcings,name:(ages,values)},
                   scaling={**context.scaling,name:_scale_forcing((ages,values),context.segments)})


def condition_context(context,events):
    """Update exact anchors/support for an age draw, retaining nominal scaling."""
    return replace(context,events=events.copy(),segments=_segments_for_events(events,context.observation_segments))


def integration_breakpoints(context,segment,event_ages=()):
    lo,hi=segment.response_start_kyr_bp,segment.response_end_kyr_bp
    sources=[source[0] for source in context.forcings.values()]
    # Include rectangular-history exits too; exponential and elapsed variants
    # are continuous between actual events and external interpolation knots.
    events=np.asarray(event_ages,float)
    candidates=np.concatenate([np.array([lo,hi]),*sources,context.phase_extrema[0],events,events-context.history_tau_ka])
    return np.unique(candidates[(candidates>=lo)&(candidates<=hi)])


def evaluate_features(context,ages,segment_id,event_ages):
    """Evaluate BP forcing samples and forward-time history within one segment."""
    ages=np.asarray(ages,float)
    segment=context.segments[segment_id]
    elapsed_kyr=segment.anchor_age_kyr_bp-ages
    frame={"age_kyr_bp":ages,"elapsed_kyr":elapsed_kyr,
           "segment_id":np.repeat(segment_id,len(ages)),"intercept":np.ones(len(ages))}
    # Interpolate on the original BP axis, including the published phase convention.
    for name,(source_age,source_value) in context.forcings.items():
        values=event_inputs.interpolate_checked(ages,source_age,source_value,context=name) if len(ages) else np.array([])
        frame[name]=values
        if name in context.scaling:
            scale=context.scaling[name]
            frame[name+"_scaled"]=(values-scale["mean"])/scale["range"]
    phase,extra=orbital_phase.interpolate_unwrapped_phase(ages,*context.phase_extrema)
    frame.update(pre_phase_unwrapped_rad=phase,pre_phase_rad=np.mod(phase,2*np.pi),
                 pre_phase_deg=np.mod(np.degrees(phase),360),pre_phase_sin=np.sin(phase),
                 pre_phase_cos=np.cos(phase),pre_phase_extrapolated=extra)
    event_ages=np.sort(np.asarray(event_ages,float))[::-1]
    event_elapsed_kyr=segment.anchor_age_kyr_bp-event_ages
    frame[HISTORY_TERM]=point_process.exponential_history(ages,event_ages,context.history_tau_ka,
                           anchor_age=segment.anchor_age_kyr_bp,initial_history=context.initial_history)
    # Compare on the unshifted forward axis (-BP) to preserve exact event and
    # window endpoints; origin subtraction can merge adjacent floating values.
    comparison_time=-ages
    event_comparison_time=-event_ages
    previous_count=np.searchsorted(event_comparison_time,comparison_time,side="left")
    since_last_kyr=np.zeros(len(ages))
    present=previous_count>0
    since_last_kyr[present]=elapsed_kyr[present]-event_elapsed_kyr[previous_count[present]-1]
    frame["time_since_last_event_kyr"]=since_last_kyr
    frame["log_time_since_last_event"]=np.log1p(since_last_kyr)
    window_start=comparison_time-context.history_tau_ka
    first_recent=np.searchsorted(event_comparison_time,window_start,side="right")
    frame["rectangular_history_count"]=(previous_count-first_recent).astype(float)
    frame[SEGMENT_TERM]=np.full(len(ages),float(segment_id=="MIS6"))
    for name,factors in (context.derived_terms or {}).items():
        frame[name]=np.prod([frame[f] for f in factors],axis=0)
    return pd.DataFrame(frame)


def prepare_catalogue(events,context,*,fixed_support=False):
    events=events.copy()
    _require_columns(events,(EVENT_AGE_COLUMN,"segment_id"),"Event catalogue")
    checked=_segments_for_events(events,context.observation_segments)
    if fixed_support:
        for name,s in context.segments.items():
            if checked[name].anchor_age_kyr_bp!=s.anchor_age_kyr_bp:
                raise ValueError("Fixed support requires the original conditioning event")
        active=replace(context,events=events)
    else:
        active=replace(context,events=events,segments=checked)
    event_frames=[];integral_frames=[];role_frames=[]
    for name,s in active.segments.items():
        part=events.loc[events.segment_id.eq(name)].sort_values(EVENT_AGE_COLUMN).copy()
        age=part[EVENT_AGE_COLUMN].to_numpy(float)
        response=(age<s.response_end_kyr_bp)&(age>=s.response_start_kyr_bp)
        part["event_role"]=np.where(age==s.anchor_age_kyr_bp,"conditioning",np.where(response,"response","history_only"))
        part["included_in_response"]=response
        role_frames.append(part)
        event_frame=evaluate_features(active,age[response],name,age)
        if "event_id" in part:
            event_frame["event_id"]=part.loc[response,"event_id"].to_numpy()
        event_frames.append(event_frame)
        knots=integration_breakpoints(active,s,age)
        # BP-ordered nodes retain their original interpolation samples. The
        # positive kyr weights also integrate du, since u=anchor-age and |du/da|=1.
        nodes,weights=point_process.gauss_legendre_intervals(knots,order=active.quadrature_order)
        integration=evaluate_features(active,nodes,name,age)
        integration["weight"]=weights
        integral_frames.append(integration)
    return CatalogueDesign(pd.concat(event_frames,ignore_index=True),pd.concat(integral_frames,ignore_index=True),
                           pd.concat(role_frames,ignore_index=True),active)


def fit_terms(design,terms,start_beta=None):
    names=("intercept",*terms)
    return point_process.fit_point_process(design.event_frame.loc[:,names].to_numpy(float),
              design.integration_frame.loc[:,names].to_numpy(float),design.weights,names,start_beta=start_beta)


def fit_summary(design,reduced,full):
    metrics=nested_likelihood_metrics(loglik_full=full.log_likelihood,loglik_reduced=reduced.log_likelihood,
                df=len(full.beta)-len(reduced.beta),n_events=len(design.event_frame),aic_full=full.aic,aic_reduced=reduced.aic)
    betas=dict(zip(full.terms,full.beta))
    sine,cosine=betas.get("pre_phase_sin",np.nan),betas.get("pre_phase_cos",np.nan)
    amp=float(np.hypot(sine,cosine))
    context=design.context
    return {"catalogue_id":context.catalogue_id,"model_version":MODEL_VERSION,
            "n_source_events":len(design.all_events),"n_response_events":len(design.event_frame),
            "n_conditioning_events":len(context.segments),"response_exposure_kyr":context.response_exposure_kyr,
            "history_kernel":"exponential","history_initialization":"condition_on_exact_oldest_event",
            "history_tau_kyr":context.history_tau_ka,"history_coefficient_domain":"nonpositive",
            "initial_unobserved_history":context.initial_history,**metrics,"nominal_LR_p":metrics["LR_p_value"],
            "pre_phase_preferred_deg":float(np.degrees(np.arctan2(sine,cosine))%360) if amp>1e-10 else np.nan,
            "pre_phase_rate_ratio_max_vs_min":float(np.exp(2*amp)),"pre_phase_amplitude":amp,
            "beta_history":betas.get(HISTORY_TERM,np.nan),
            "mis6_vs_ngrip_rate_ratio_full":float(np.exp(betas[SEGMENT_TERM])) if SEGMENT_TERM in betas else np.nan,
            "all_models_converged":bool(reduced.converged and full.converged),
            "likelihood_nesting_ok":metrics["ll_gain_nats"]>=-1e-7}


def fit_catalogue(events,context,*,fixed_support=False):
    design=prepare_catalogue(events,context,fixed_support=fixed_support)
    reduced=fit_terms(design,context.reduced_terms)
    start=np.zeros(len(context.full_terms)+1)
    if np.isfinite(reduced.beta).all():
        for term,beta in zip(reduced.terms,reduced.beta):
            start[("intercept",*context.full_terms).index(term)]=beta
    full=fit_terms(design,context.full_terms,start_beta=start)
    return CombinedLikelihoodFit(design,reduced,full,fit_summary(design,reduced,full))


def fit_point_catalogue(events=None,context=None):
    context=build_context() if context is None else context
    return fit_catalogue(context.events if events is None else events,context)


def coefficient_table(fit):
    return pd.DataFrame([dict(model_id=name,term=term,beta=beta,rate_ratio_per_unit=np.exp(beta))
                        for name,model in (("reduced",fit.reduced),("full",fit.full))
                        for term,beta in zip(model.terms,model.beta)])


def support_table(context):
    return pd.DataFrame([vars(segment) for segment in context.segments.values()])


def scaling_table(context):
    return pd.DataFrame([dict(forcing_id=name,**scale,weighting="nominal response time") for name,scale in context.scaling.items()])


def fitted_rate_table(fit,step_kyr=0.1):
    frames=[]
    for name,s in fit.context.segments.items():
        ages=fit.design.all_events.loc[fit.design.all_events.segment_id.eq(name),EVENT_AGE_COLUMN].to_numpy(float)
        # Paired BP neighbors show the event jump. History membership compares
        # their unshifted coordinates, even if anchor-age rounds them to one u.
        after=np.nextafter(ages,-np.inf)
        after=after[after>=s.response_start_kyr_bp]
        query=np.unique(np.r_[np.arange(s.response_start_kyr_bp,s.response_end_kyr_bp,step_kyr),ages,after,s.response_end_kyr_bp])
        features=evaluate_features(fit.context,query,name,ages)
        out=features[["segment_id","age_kyr_bp","lr04","co2","precession_index","pre_phase_deg"]].copy()
        for model_id,model in (("reduced",fit.reduced),("full",fit.full)):
            out[model_id+"_rate"]=np.exp(features.loc[:,model.terms].to_numpy(float)@model.beta)
        frames.append(out)
    return pd.concat(frames,ignore_index=True)

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


def _background_values(ages, context, segment_id, terms, beta):
    """Evaluate the exogenous log rate without constructing data frames."""
    age=np.asarray(ages,float)
    values={"intercept":np.ones_like(age),SEGMENT_TERM:np.full_like(age,float(segment_id=="MIS6"))}
    for name in context.scaling:
        scale=context.scaling[name]
        values[name+"_scaled"]=(np.interp(age,*context.forcings[name])-scale["mean"])/scale["range"]
    phase,_=orbital_phase.interpolate_unwrapped_phase(np.atleast_1d(age),*context.phase_extrema)
    phase=phase.reshape(age.shape)
    values.update(pre_phase_sin=np.sin(phase),pre_phase_cos=np.cos(phase))
    for name,factors in (context.derived_terms or {}).items():
        values[name]=np.prod([values[f] for f in factors],axis=0)
    return sum(coef*values[term] for term,coef in zip(terms,beta) if term!=HISTORY_TERM)


def prepare_model_simulation(context, model, proposal_interval_kyr=1.0):
    """Certify piecewise background envelopes; their width is not a fit bin."""
    from functools import partial
    if not model.converged or not np.isfinite(model.beta).all():
        raise ValueError("A finite fitted generator is required")
    coefficients=dict(zip(model.terms,model.beta))
    if coefficients.get(HISTORY_TERM,0)>0:
        raise ValueError("Positive feedback is outside the chosen model")
    prepared=[]
    for name,s in context.segments.items():
        edges=np.r_[np.arange(s.response_start_kyr_bp,s.anchor_age_kyr_bp,proposal_interval_kyr),s.anchor_age_kyr_bp]
        upper=[]
        for lo,hi in zip(edges[:-1],edges[1:]):
            bounds={"intercept":(1.,1.),SEGMENT_TERM:(float(name=="MIS6"),)*2}
            for forcing,scale in context.scaling.items():
                age,value=context.forcings[forcing]
                knots=np.r_[lo,age[(age>lo)&(age<hi)],hi]
                vals=(np.interp(knots,age,value)-scale["mean"])/scale["range"]
                bounds[forcing+"_scaled"]=(float(vals.min()),float(vals.max()))
            # A global [-1,1] bound is conservative and covers interior extrema.
            bounds.update(pre_phase_sin=(-1.,1.),pre_phase_cos=(-1.,1.))
            for term,factors in (context.derived_terms or {}).items():
                low=high=1.
                for factor in factors:
                    candidates=np.array([low,high])[:,None]*np.array(bounds[factor])[None,:]
                    low,high=float(candidates.min()),float(candidates.max())
                bounds[term]=(low,high)
            upper.append(sum(max(coef*bounds[term][0],coef*bounds[term][1])
                             for term,coef in coefficients.items() if term!=HISTORY_TERM)+1e-12)
        prepared.append(dict(segment_id=name,segment=s,breakpoints=edges,log_upper_bounds=np.array(upper),
                      log_background=partial(_background_values,context=context,segment_id=name,terms=model.terms,beta=model.beta),
                      history_beta=coefficients.get(HISTORY_TERM,0.),tau=context.history_tau_ka,
                      initial_history=context.initial_history,
                      anchor=context.events.loc[context.events.segment_id.eq(name)].sort_values(EVENT_AGE_COLUMN).iloc[-1].to_dict()))
    return prepared


def simulate_prepared_events(prepared,rng):
    """Simulate independent response segments while retaining exact anchors."""
    frames=[]
    for part in prepared:
        s=part["segment"]
        age=point_process.simulate_segment_events(s.anchor_age_kyr_bp,s.response_start_kyr_bp,
              part["breakpoints"],part["log_background"],part["log_upper_bounds"],part["history_beta"],
              part["tau"],rng,initial_history=part["initial_history"])
        frame=pd.DataFrame({EVENT_AGE_COLUMN:age,"segment_id":part["segment_id"],
                            "event_id":[part["anchor"]["event_id"]]+[f"sim_{part['segment_id']}_{i}" for i in range(1,len(age))]})
        frame["event_label"]=frame.event_id
        frames.append(frame.sort_values(EVENT_AGE_COLUMN))
    return pd.concat(frames,ignore_index=True)


def simulate_reduced_model_events(context,model,rng):
    """Public simulator for the no-precession null model."""
    if set(model.terms)!={"intercept",*context.reduced_terms}:
        raise ValueError("Reduced simulator requires the no-precession terms")
    return simulate_prepared_events(prepare_model_simulation(context,model),rng)


def simulate_full_model_events(context,model,rng):
    """Separate public simulator for the full effect model."""
    if set(model.terms)!={"intercept",*context.full_terms}:
        raise ValueError("Full simulator requires the full model terms")
    return simulate_prepared_events(prepare_model_simulation(context,model),rng)


def rescaled_event_intervals(fit, model=None):
    """Cumulative fitted intensities at responses plus terminal censored waits."""
    if model is not None:
        from types import SimpleNamespace
        fit=SimpleNamespace(design=fit,full=model,context=fit.context)
    events={};cumulative={};tails={}
    frame=fit.design.integration_frame
    rate=np.exp(frame.loc[:,fit.full.terms].to_numpy(float)@fit.full.beta)
    if fit.full.status=="zero_events":
        rate=np.zeros(len(frame))
    for name,s in fit.context.segments.items():
        response_ages=fit.design.event_frame.loc[fit.design.event_frame.segment_id.eq(name),"age_kyr_bp"].to_numpy(float)
        response_ages=np.sort(response_ages)[::-1]
        response_elapsed=s.anchor_age_kyr_bp-response_ages
        mask=frame.segment_id.eq(name).to_numpy()
        node_ages=frame.loc[mask,"age_kyr_bp"].to_numpy(float)
        # Accumulate intensity from the anchor toward the present. Sort nodes
        # and their masses together; incoming table row order has no time meaning.
        order=np.argsort(-node_ages)
        node_ages=node_ages[order]
        mass=rate[mask]*frame.loc[mask,"weight"].to_numpy(float)
        mass=mass[order]
        cumulative_mass=np.r_[0.0,np.cumsum(mass)]
        # u_node < u_event, compared before subtracting the origin for precision.
        indices=np.searchsorted(-node_ages,-response_ages,side="left")
        events[name]=response_elapsed
        cumulative[name]=cumulative_mass[indices]
        tails[name]=float(mass[indices[-1]:].sum() if len(indices) else mass.sum())
    return events,cumulative,tails


def phase_sector_observed_expected(fit, n_sectors=12):
    """Observed vs fitted event counts in equal precession-phase sectors.

    For each phase sector, the fitted count under a model is the response-time
    integrated intensity whose precession phase falls in that sector; the
    observed count is the number of response events in the sector. A correctly
    specified model gives a ratio near one in every sector. A phase-organized
    signal appears as a sinusoidal departure for the no-phase (reduced) model
    and flattens once the full model adds the phase terms.
    """
    integration = fit.design.integration_frame
    events = fit.design.event_frame
    edges = np.linspace(0.0, 360.0, n_sectors + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    observed, _ = np.histogram(events["pre_phase_deg"].to_numpy(float), bins=edges)
    rows = []
    for model_id, model in (("reduced", fit.reduced), ("full", fit.full)):
        rate = np.exp(integration.loc[:, model.terms].to_numpy(float) @ model.beta)
        mass = rate * integration["weight"].to_numpy(float)
        fitted, _ = np.histogram(integration["pre_phase_deg"].to_numpy(float),
                                 bins=edges, weights=mass)
        ratio = np.where(fitted > 0.0, observed / fitted, np.nan)
        for center, obs, fit_count, rat in zip(centers, observed, fitted, ratio):
            rows.append(dict(model_id=model_id, phase_sector_center_deg=center,
                             observed_events=int(obs), fitted_events=fit_count,
                             observed_over_fitted=rat))
    return pd.DataFrame(rows)
