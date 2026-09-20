"""Scientific invariants for the exact-age continuous NGRIP--MIS6 model."""
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from toolbox import combined_likelihood as c


@pytest.fixture(scope="module")
def context():
    return c.build_context()


@pytest.fixture(scope="module")
def fitted(context):
    return c.fit_catalogue(context.events, context)


def test_frozen_catalogue_and_chosen_model(context):
    assert len(context.events) == 55
    assert context.events.event_id.is_unique
    assert context.events.groupby("segment_id").size().to_dict() == {"MIS6": 21, "NGRIP": 34}
    assert c.REDUCED_TERMS == (c.HISTORY_TERM, "lr04_scaled", "co2_scaled", "mis6_segment")
    assert c.FULL_TERMS == c.REDUCED_TERMS + ("pre_phase_sin", "pre_phase_cos")
    assert context.history_tau_ka == 1.5
    assert context.initial_history == 0


def test_exact_anchors_define_exposure_without_bins_or_gap(fitted):
    context, design = fitted.context, fitted.design
    assert len(design.event_frame) == 53
    assert design.all_events.event_role.eq("conditioning").sum() == 2
    for name, segment in context.segments.items():
        ages = context.events.loc[context.events.segment_id.eq(name), c.EVENT_AGE_COLUMN]
        assert segment.anchor_age_kyr_bp == ages.max()
        assert segment.response_end_kyr_bp == ages.max()
        assert segment.response_start_kyr_bp == segment.observation_start_kyr_bp
        nodes = design.integration_frame.loc[design.integration_frame.segment_id.eq(name)]
        assert nodes.age_kyr_bp.between(segment.response_start_kyr_bp, segment.anchor_age_kyr_bp).all()
        assert nodes.weight.sum() == pytest.approx(segment.anchor_age_kyr_bp - segment.response_start_kyr_bp)
        for frame in (design.event_frame, design.integration_frame):
            part = frame.loc[frame.segment_id.eq(name)]
            np.testing.assert_allclose(part.elapsed_kyr, segment.anchor_age_kyr_bp - part.age_kyr_bp)
            assert (part.elapsed_kyr > 0).all()
    assert design.weights.sum() == pytest.approx(context.response_exposure_kyr)
    assert (design.weights > 0).all()


def test_forcing_scaling_uses_nominal_time_exposure(fitted):
    nodes = fitted.design.integration_frame
    for forcing in ("lr04", "co2"):
        assert np.average(nodes[forcing + "_scaled"], weights=nodes.weight) == pytest.approx(0, abs=2e-13)
        scale = fitted.context.scaling[forcing]
        assert scale["range"] == pytest.approx(scale["max"] - scale["min"])


def test_strictly_older_exact_history_and_pre_anchor_initialization(context):
    for name, segment in context.segments.items():
        ages = context.events.loc[context.events.segment_id.eq(name), c.EVENT_AGE_COLUMN].to_numpy()
        frame = c.evaluate_features(context, ages, name, ages)
        expected = [np.exp(-(ages[ages > age] - age) / context.history_tau_ka).sum() for age in ages]
        np.testing.assert_allclose(frame[c.HISTORY_TERM], expected, atol=1e-14)
        assert frame[c.HISTORY_TERM].iloc[-1] == 0
        assert frame.elapsed_kyr.iloc[-1] == 0
        initialized = c.evaluate_features(replace(context, initial_history=0.5), ages, name, ages)
        np.testing.assert_allclose(initialized[c.HISTORY_TERM] - frame[c.HISTORY_TERM],
                                   0.5 * np.exp((ages - segment.anchor_age_kyr_bp) / 1.5), atol=1e-15)


def test_small_age_changes_are_not_coalesced_into_count_patterns(context):
    events = context.events.copy()
    index = events.index[0]
    event_id = events.loc[index, "event_id"]
    events.loc[index, c.EVENT_AGE_COLUMN] += 0.001
    old = c.prepare_catalogue(context.events, context, fixed_support=True).event_frame.set_index("event_id")
    new = c.prepare_catalogue(events, context, fixed_support=True).event_frame.set_index("event_id")
    assert new.loc[event_id, "age_kyr_bp"] - old.loc[event_id, "age_kyr_bp"] == pytest.approx(0.001)
    assert new.loc[event_id, c.HISTORY_TERM] != old.loc[event_id, c.HISTORY_TERM]
    assert new.loc[event_id, "pre_phase_unwrapped_rad"] != old.loc[event_id, "pre_phase_unwrapped_rad"]


def test_alternative_history_terms_keep_event_and_window_boundaries(context):
    for name, segment in context.segments.items():
        ages = context.events.loc[context.events.segment_id.eq(name), c.EVENT_AGE_COLUMN].to_numpy()
        exits = ages - context.history_tau_ka
        query = np.r_[ages, exits, np.nextafter(exits, -np.inf), np.nextafter(exits, np.inf)]
        query = query[(query >= segment.response_start_kyr_bp) & (query <= segment.anchor_age_kyr_bp)]
        frame = c.evaluate_features(context, query, name, ages)
        gaps, counts = [], []
        for age in query:
            earlier = ages[ages > age]
            gaps.append(earlier.min() - age if len(earlier) else 0)
            counts.append(np.sum(earlier < age + context.history_tau_ka))
        np.testing.assert_allclose(frame.time_since_last_event_kyr, gaps, atol=3e-14)
        np.testing.assert_allclose(frame.log_time_since_last_event, np.log1p(gaps), atol=3e-14)
        np.testing.assert_array_equal(frame.rectangular_history_count, counts)


def test_age_draw_reconditions_anchor_but_preserves_nominal_climate_scales(context):
    events = context.events.copy()
    index = events.loc[events.segment_id.eq("NGRIP"), c.EVENT_AGE_COLUMN].idxmax()
    events.loc[index, c.EVENT_AGE_COLUMN] += 0.1
    with pytest.raises(ValueError, match="original conditioning"):
        c.prepare_catalogue(events, context, fixed_support=True)
    local = c.condition_context(context, events)
    assert local.response_exposure_kyr == pytest.approx(context.response_exposure_kyr + 0.1)
    assert local.scaling == context.scaling
    result = c.fit_catalogue(events, local, fixed_support=True)
    assert result.summary["n_response_events"] == 53
    assert result.summary["response_exposure_kyr"] == pytest.approx(local.response_exposure_kyr)
    new = result.design.event_frame.set_index("event_id")
    old = c.prepare_catalogue(context.events, context).event_frame.set_index("event_id")
    for name, shift in (("NGRIP", 0.1), ("MIS6", 0)):
        mask = new.segment_id.eq(name)
        np.testing.assert_allclose(new.loc[mask, "elapsed_kyr"] - old.loc[mask, "elapsed_kyr"],
                                   shift, atol=3e-14)


def test_likelihood_event_sum_integral_and_no_bin_sample_size(fitted):
    model, design = fitted.full, fitted.design
    event_lograte = design.event_frame.loc[:, model.terms].to_numpy() @ model.beta
    node_rate = np.exp(design.integration_frame.loc[:, model.terms].to_numpy() @ model.beta)
    assert model.log_likelihood == pytest.approx(event_lograte.sum() - design.weights @ node_rate)
    assert fitted.summary["gain_bits_per_event"] == pytest.approx(
        (model.log_likelihood - fitted.reduced.log_likelihood) / (53 * np.log(2)))
    assert fitted.summary["delta_AIC_full_minus_reduced"] == pytest.approx(4 - fitted.summary["LR_statistic"])
    assert not any("AICc" in name or "bin" in name for name in fitted.summary)
    assert fitted.summary["beta_history"] <= 0


def test_rescaling_keeps_terminal_censoring_and_separate_segments(fitted):
    events, cumulative, tails = c.rescaled_event_intervals(fitted)
    frame = fitted.design.integration_frame
    rate = np.exp(frame.loc[:, fitted.full.terms].to_numpy() @ fitted.full.beta)
    for name in fitted.context.segments:
        mask = frame.segment_id.eq(name).to_numpy()
        assert np.all(np.diff(events[name]) > 0)
        assert np.all(np.diff(cumulative[name]) > 0)
        assert tails[name] >= 0
        assert cumulative[name][-1] + tails[name] == pytest.approx(rate[mask] @ frame.loc[mask, "weight"])
        response_ages = fitted.context.segments[name].anchor_age_kyr_bp - events[name]
        node_ages = frame.loc[mask, "age_kyr_bp"].to_numpy()
        mass = rate[mask] * frame.loc[mask, "weight"].to_numpy()
        # Independent BP selections check every cumulative interval and the tail.
        expected = [mass[node_ages > age].sum() for age in response_ages]
        np.testing.assert_allclose(cumulative[name], expected, rtol=2e-13)
        assert tails[name] == pytest.approx(mass[node_ages < response_ages[-1]].sum())

    shuffled_design = replace(fitted.design,
        integration_frame=frame.sample(frac=1, random_state=246).reset_index(drop=True),
        event_frame=fitted.design.event_frame.sample(frac=1, random_state=975).reset_index(drop=True))
    shuffled = c.rescaled_event_intervals(replace(fitted, design=shuffled_design))
    for expected_group, actual_group in zip((events, cumulative, tails), shuffled):
        for name in fitted.context.segments:
            np.testing.assert_allclose(actual_group[name], expected_group[name], rtol=2e-13)


def test_fitted_rate_curves_retain_history_jump_at_every_event(fitted):
    rates = c.fitted_rate_table(fitted).set_index(["segment_id", "age_kyr_bp"])
    for name, part in fitted.design.all_events.groupby("segment_id"):
        for age in part[c.EVENT_AGE_COLUMN]:
            after = np.nextafter(age, -np.inf)
            if after < fitted.context.segments[name].response_start_kyr_bp:
                continue
            for model_id, model in (("reduced", fitted.reduced), ("full", fitted.full)):
                history_beta = model.beta[model.terms.index(c.HISTORY_TERM)]
                ratio = rates.loc[(name, after), model_id + "_rate"] / rates.loc[(name, age), model_id + "_rate"]
                assert ratio == pytest.approx(np.exp(history_beta), rel=2e-11)


def test_simulation_envelopes_bound_background_inside_intervals(fitted):
    for part in c.prepare_model_simulation(fitted.context, fitted.full):
        for lo, hi, upper in zip(part["breakpoints"][:-1], part["breakpoints"][1:], part["log_upper_bounds"]):
            assert np.max(part["log_background"](np.linspace(lo, hi, 11))) <= upper + 1e-12


def test_event_phase_sampling_uses_bp1950_phase_convention(context):
    phases = c.sample_event_phases(context.events)
    assert len(phases) == 55
    assert phases.pre_phase_deg.between(0, 360).all()
    assert not phases.pre_phase_extrapolated.any()
    np.testing.assert_allclose(np.sin(phases.pre_phase_rad), np.sin(np.deg2rad(phases.pre_phase_deg)), atol=1e-12)
