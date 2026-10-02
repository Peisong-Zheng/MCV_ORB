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

from toolbox import event_model, model_stats, point_process
from toolbox.model_stats import nested_likelihood_metrics
from toolbox.project_config import (
    PROJECT_ROOT, LR04_CSV, CO2_CSV, ORBITAL_CSV, PRECESSION_PHASE_CSV, MODEL_VERSION,
)

EVENT_CATALOGUE_CSV = PROJECT_ROOT / "data/curated/ngrip_mis6_warming_events.csv"
OBSERVATION_SEGMENTS_CSV = PROJECT_ROOT / "data/curated/observation_segments.csv"
EVENT_AGE_COLUMN = "event_age_kyr_bp"
SEGMENT_IDS = ("NGRIP", "MIS6")
SEGMENT_TERM = "mis6_segment"
HISTORY_TERM = "same_type_exponential_history"
CATALOGUE_ID = "ngrip_warming_plus_mis6"
DEFAULT_HISTORY_TAU_KA = 1.5
REDUCED_TERMS = (HISTORY_TERM, "lr04_scaled", "co2_scaled", SEGMENT_TERM)
FULL_TERMS = REDUCED_TERMS + ("pre_phase_sin", "pre_phase_cos")
RESOLUTION_COVARIATE_INCLUDED = False
EVENT_COLUMNS = (
    "event_id", "event_label", EVENT_AGE_COLUMN, "segment_id", "source_record",
    "source_event_label", "data_source", "label_source", "timing_method", "source_table",
)


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
        return sum(
            segment.response_end_kyr_bp - segment.response_start_kyr_bp
            for segment in self.segments.values()
        )


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
    missing = set(columns).difference(frame.columns)
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
        if (
            not np.isfinite([row.observation_start_kyr_bp, row.observation_end_kyr_bp]).all()
            or row.observation_start_kyr_bp >= row.observation_end_kyr_bp
        ):
            raise ValueError("Observation support must be finite and positive")
    return frame.drop(columns=[column for column in frame if column.startswith("common_response_")])


@lru_cache(maxsize=8)
def _source_forcings(lr04_path, co2_path, precession_path, phase_path):
    # Native BP1950 samples; model interpolation/scaling remain below.
    lr04 = pd.read_csv(lr04_path, float_precision="round_trip")
    co2 = pd.read_csv(co2_path, float_precision="round_trip")
    orbital = pd.read_csv(precession_path, float_precision="round_trip")
    anchors = pd.read_csv(phase_path, float_precision="round_trip")
    forcings = {
        "lr04": (lr04.age_kyr_bp.to_numpy(), lr04.lr04.to_numpy()),
        "co2": (co2.age_kyr_bp.to_numpy(), co2.co2_ppm.to_numpy()),
        "precession_index": (orbital.age_kyr_bp.to_numpy(), orbital.precession_index.to_numpy()),
    }
    phase_extrema = (anchors.age_kyr_bp.to_numpy(), anchors.phase_unwrapped_rad.to_numpy())
    for name, (age, values) in {**forcings, "phase anchors": phase_extrema}.items():
        if len(age) < 2 or not np.isfinite([age, values]).all() or np.any(np.diff(age) <= 0):
            raise ValueError(f"{name} requires finite values at unique increasing ages")
    return forcings, phase_extrema


def _segments_for_events(events, observation_segments):
    windows = event_model.response_windows(events, observation_segments)
    return {row["segment_id"]: SegmentContext(**row) for row in windows.to_dict("records")}


def _scale_forcing(source, segments):
    windows = pd.DataFrame([vars(segment) for segment in segments.values()])
    return event_model.scale_forcing(source, windows)


def build_context(observation_segments=None, *, events=None, history_tau_ka=DEFAULT_HISTORY_TAU_KA,
                  initial_history=0.0, quadrature_order=4, catalogue_id=CATALOGUE_ID,
                  lr04_path=LR04_CSV, co2_path=CO2_CSV, precession_path=ORBITAL_CSV,
                  phase_path=PRECESSION_PHASE_CSV):
    """Set nominal support and time-weighted scaling from prepared BP1950 CSVs.

    Custom precession_path and phase_path must supply the same orbital series
    and its corresponding phase anchors; paths no longer accept raw workbooks.
    """
    if history_tau_ka <= 0 or initial_history < 0:
        raise ValueError("History decay must be positive; initial history must be nonnegative")
    events = load_event_catalogue() if events is None else events.copy()
    observations = (
        load_observation_segments()
        if observation_segments is None else observation_segments.copy()
    )
    segments = _segments_for_events(events, observations)
    sources, phase_extrema = _source_forcings(
        Path(lr04_path), Path(co2_path), Path(precession_path), Path(phase_path),
    )
    scaling = {
        name: _scale_forcing(source, segments)
        for name, source in sources.items() if name != "precession_index"
    }
    if "MIS6" in segments and len(segments) > 1:
        reduced_terms = REDUCED_TERMS
    else:
        reduced_terms = tuple(term for term in REDUCED_TERMS if term != SEGMENT_TERM)
    return LikelihoodContext(
        observation_segments=observations,
        events=events,
        segments=segments,
        forcings=dict(sources),
        scaling=scaling,
        phase_extrema=phase_extrema,
        history_tau_ka=history_tau_ka,
        initial_history=initial_history,
        quadrature_order=quadrature_order,
        catalogue_id=catalogue_id,
        reduced_terms=reduced_terms,
        full_terms=reduced_terms + ("pre_phase_sin", "pre_phase_cos"),
        derived_terms={},
    )


def build_barker_context(events, event_definition="variable_threshold", **kwargs):
    """Set Barker's observation window from an explicitly supplied event table."""
    counts = {"variable_threshold": 70, "fixed_threshold": 59}
    if event_definition not in counts:
        raise ValueError(f"Unknown Barker event definition: {event_definition}")
    _require_columns(events, ("event_id", EVENT_AGE_COLUMN), "Barker events")
    if len(events) != counts[event_definition]:
        raise ValueError(f"Expected {counts[event_definition]} {event_definition} events")
    if events.event_id.isna().any() or not events.event_id.is_unique:
        raise ValueError("Barker event IDs must be present and unique")
    ages = events[EVENT_AGE_COLUMN].to_numpy(float)
    if not np.isfinite(ages).all() or np.any(np.diff(ages) <= 0):
        raise ValueError("Barker event ages must be finite and strictly increasing")
    events = events.assign(segment_id="Barker2011")
    support = pd.DataFrame([dict(
        segment_id="Barker2011", observation_start_kyr_bp=0., observation_end_kyr_bp=400.,
    )])
    return build_context(
        support, events=events, catalogue_id=f"barker_{event_definition}_speleo_0_400", **kwargs,
    )


def add_forcing(context, name, ages, values):
    ages, values = event_model.clean_series(ages, values, context=name)
    return replace(
        context,
        forcings={**context.forcings, name: (ages, values)},
        scaling={**context.scaling, name: _scale_forcing((ages, values), context.segments)},
    )


def condition_context(context, events):
    """Update exact anchors/support for an age draw, retaining nominal scaling."""
    return replace(
        context,
        events=events.copy(),
        segments=_segments_for_events(events, context.observation_segments),
    )


def integration_breakpoints(context, segment, event_ages=()):
    return event_model.integration_breakpoints(
        segment, context.forcings, context.phase_extrema, event_ages, context.history_tau_ka,
    )


def evaluate_features(context, ages, segment_id, event_ages):
    """Supply the existing sensitivity interface from the shared feature maths."""
    frame = event_model.evaluate_features(
        ages, event_ages, context.segments[segment_id], context.forcings,
        context.phase_extrema, pd.DataFrame.from_dict(context.scaling, orient="index"),
        tau=context.history_tau_ka, initial_history=context.initial_history,
        history_variants=True,
    )
    frame[SEGMENT_TERM] = float(segment_id == "MIS6")
    for name, factors in (context.derived_terms or {}).items():
        frame[name] = np.prod([frame[factor] for factor in factors], axis=0)
    return frame


def prepare_catalogue(events, context, *, fixed_support=False):
    """Separate conditioning events, response events and integration samples."""
    events = events.copy()
    _require_columns(events, (EVENT_AGE_COLUMN, "segment_id"), "Event catalogue")
    event_segments = _segments_for_events(events, context.observation_segments)
    if fixed_support:
        for segment_id, segment in context.segments.items():
            if event_segments[segment_id].anchor_age_kyr_bp != segment.anchor_age_kyr_bp:
                raise ValueError("Fixed support requires the original conditioning event")
        active_context = replace(context, events=events)
    else:
        active_context = replace(context, events=events, segments=event_segments)

    event_frame, integration_frame = event_model.build_design(
        events, support_table(active_context), active_context.forcings,
        active_context.phase_extrema,
        pd.DataFrame.from_dict(active_context.scaling, orient="index"),
        tau=active_context.history_tau_ka, initial_history=active_context.initial_history,
        quadrature_order=active_context.quadrature_order, history_variants=True,
    )
    # Existing consumers still declare pooled indicators/derived terms on context.
    # Keep their table layout while the explicit-matrix entry is tried in Barker.
    for frame, final_column in ((event_frame, "event_id"), (integration_frame, "weight")):
        last = frame.pop(final_column) if final_column in frame else None
        frame[SEGMENT_TERM] = frame.segment_id.eq("MIS6").astype(float)
        for name, factors in (active_context.derived_terms or {}).items():
            frame[name] = np.prod([frame[factor] for factor in factors], axis=0)
        if last is not None:
            frame[final_column] = last

    catalogue_frames = []
    for segment_id, segment in active_context.segments.items():
        selected = events.loc[events.segment_id.eq(segment_id)].sort_values(EVENT_AGE_COLUMN).copy()
        ages = selected[EVENT_AGE_COLUMN].to_numpy(float)
        response = (ages < segment.response_end_kyr_bp) & (ages >= segment.response_start_kyr_bp)
        selected["event_role"] = np.where(
            ages == segment.anchor_age_kyr_bp, "conditioning",
            np.where(response, "response", "history_only"),
        )
        selected["included_in_response"] = response
        catalogue_frames.append(selected)
    return CatalogueDesign(
        event_frame=event_frame, integration_frame=integration_frame,
        all_events=pd.concat(catalogue_frames, ignore_index=True), context=active_context,
    )


def fit_terms(design, terms, start_beta=None):
    names = ("intercept", *terms)
    event_design = design.event_frame.loc[:, names].to_numpy(float)
    integral_design = design.integration_frame.loc[:, names].to_numpy(float)
    return point_process.fit_point_process(
        event_design, integral_design, design.weights, names, start_beta=start_beta,
    )


def fit_summary(design, reduced, full):
    metrics = nested_likelihood_metrics(
        loglik_full=full.log_likelihood,
        loglik_reduced=reduced.log_likelihood,
        df=len(full.beta) - len(reduced.beta),
        n_events=len(design.event_frame),
        aic_full=full.aic,
        aic_reduced=reduced.aic,
    )
    betas = dict(zip(full.terms, full.beta))
    beta_sin = betas.get("pre_phase_sin", np.nan)
    beta_cos = betas.get("pre_phase_cos", np.nan)
    amplitude = float(np.hypot(beta_sin, beta_cos))
    context = design.context
    return {
        "catalogue_id": context.catalogue_id,
        "model_version": MODEL_VERSION,
        "n_source_events": len(design.all_events),
        "n_response_events": len(design.event_frame),
        "n_conditioning_events": len(context.segments),
        "response_exposure_kyr": context.response_exposure_kyr,
        "history_kernel": "exponential",
        "history_initialization": "condition_on_exact_oldest_event",
        "history_tau_kyr": context.history_tau_ka,
        "history_coefficient_domain": "nonpositive",
        "initial_unobserved_history": context.initial_history,
        **metrics,
        "nominal_LR_p": metrics["LR_p_value"],
        "pre_phase_preferred_deg": (
            float(np.degrees(np.arctan2(beta_sin, beta_cos)) % 360)
            if amplitude > 1e-10 else np.nan
        ),
        "pre_phase_rate_ratio_max_vs_min": float(np.exp(2 * amplitude)),
        "pre_phase_amplitude": amplitude,
        "beta_history": betas.get(HISTORY_TERM, np.nan),
        "mis6_vs_ngrip_rate_ratio_full": (
            float(np.exp(betas[SEGMENT_TERM])) if SEGMENT_TERM in betas else np.nan
        ),
        "all_models_converged": bool(reduced.converged and full.converged),
        "likelihood_nesting_ok": metrics["ll_gain_nats"] >= -1e-7,
    }


def fit_catalogue(events, context, *, fixed_support=False):
    """Fit the reduced model, then add phase terms from the same starting fit."""
    design = prepare_catalogue(events, context, fixed_support=fixed_support)
    reduced = fit_terms(design, context.reduced_terms)
    start_beta = np.zeros(len(context.full_terms) + 1)
    if np.isfinite(reduced.beta).all():
        for term, beta in zip(reduced.terms, reduced.beta):
            start_beta[("intercept", *context.full_terms).index(term)] = beta
    full = fit_terms(design, context.full_terms, start_beta=start_beta)
    return CombinedLikelihoodFit(
        design=design,
        reduced=reduced,
        full=full,
        summary=fit_summary(design, reduced, full),
    )


def fit_point_catalogue(events=None, context=None):
    context = build_context() if context is None else context
    return fit_catalogue(context.events if events is None else events, context)


def coefficient_table(fit):
    rows = []
    for model_id, model in (("reduced", fit.reduced), ("full", fit.full)):
        for term, beta in zip(model.terms, model.beta):
            rows.append(dict(
                model_id=model_id,
                term=term,
                beta=beta,
                rate_ratio_per_unit=np.exp(beta),
            ))
    return pd.DataFrame(rows)


def support_table(context):
    return pd.DataFrame([vars(segment) for segment in context.segments.values()])


def scaling_table(context):
    return pd.DataFrame([
        dict(forcing_id=name, **scale, weighting="nominal response time")
        for name, scale in context.scaling.items()
    ])


def fitted_rate_table(fit, step_kyr=0.1):
    frames = []
    for segment_id, segment in fit.context.segments.items():
        event_ages = fit.design.all_events.loc[
            fit.design.all_events.segment_id.eq(segment_id), EVENT_AGE_COLUMN,
        ].to_numpy(float)
        # Paired BP neighbors show the event jump. History membership compares
        # their unshifted coordinates, even if anchor-age rounds them to one u.
        after_event = np.nextafter(event_ages, -np.inf)
        after_event = after_event[after_event >= segment.response_start_kyr_bp]
        query_ages = np.unique(np.r_[
            np.arange(segment.response_start_kyr_bp, segment.response_end_kyr_bp, step_kyr),
            event_ages,
            after_event,
            segment.response_end_kyr_bp,
        ])
        features = evaluate_features(fit.context, query_ages, segment_id, event_ages)
        rates = features[[
            "segment_id", "age_kyr_bp", "lr04", "co2", "precession_index", "pre_phase_deg",
        ]].copy()
        for model_id, model in (("reduced", fit.reduced), ("full", fit.full)):
            rate_design = features.loc[:, model.terms].to_numpy(float)
            rates[model_id + "_rate"] = np.exp(rate_design @ model.beta)
        frames.append(rates)
    return pd.concat(frames, ignore_index=True)


def sample_event_phases(events, *, age_column=EVENT_AGE_COLUMN):
    """Evaluate descriptive precession phase from its two prepared inputs."""
    _require_columns(events, ("event_id", age_column), "Event table")
    orbital = pd.read_csv(ORBITAL_CSV, float_precision="round_trip")
    anchors = pd.read_csv(PRECESSION_PHASE_CSV, float_precision="round_trip")
    return event_model.sample_event_phases(
        events, (orbital.age_kyr_bp.to_numpy(), orbital.precession_index.to_numpy()),
        (anchors.age_kyr_bp.to_numpy(), anchors.phase_unwrapped_rad.to_numpy()),
        age_column=age_column,
    )


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
    phase,_=event_model.interpolate_unwrapped_phase(np.atleast_1d(age),*context.phase_extrema)
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
    return model_stats.phase_sector_observed_expected(
        fit.design.event_frame, fit.design.integration_frame,
        {"reduced": fit.reduced, "full": fit.full}, n_sectors,
    )
