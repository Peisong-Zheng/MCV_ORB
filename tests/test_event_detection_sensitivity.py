"""Deletion changes membership and visible history, never ages or anchors."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from toolbox import event_detection_sensitivity as detection
from toolbox import event_model
from toolbox.point_process import fit_point_process, PointProcessFitError


def catalogue():
    return pd.DataFrame({"event_id": ["n1", "n2", "n0", "m1", "m0", "m2"],
                         "segment_id": ["NGRIP"] * 3 + ["MIS6"] * 3,
                         "event_age_kyr_bp": [20., 30., 40., 150., 190., 175.],
                         "source_note": ["published"] * 6}, index=[10, 11, 15, 17, 18, 20])


def fit_metrics(phase_deg=350., amplitude=0.5, gain=0.2):
    angle = np.deg2rad(phase_deg)
    return dict(gain_bits_per_event=gain, LR_statistic=8., beta_history=-0.5,
                beta_pre_phase_sin=amplitude * np.sin(angle),
                beta_pre_phase_cos=amplitude * np.cos(angle))


def test_zero_deletion_reproduces_original_table_and_full_deletion_keeps_anchors():
    events = catalogue()
    original = events.copy(deep=True)
    unchanged, membership = detection.drop_events(events, 0., np.random.default_rng(17))
    pd.testing.assert_frame_equal(unchanged, events)
    assert membership.retained.all()
    anchors, membership = detection.drop_events(events, 1., np.random.default_rng(17))
    assert anchors.event_id.tolist() == ["n0", "m0"]
    assert set(membership.loc[membership.is_conditioning_event, "event_id"]) == {"n0", "m0"}
    pd.testing.assert_frame_equal(events, original)


def test_mis6_only_mask_retains_ngrip_and_original_id_age_columns():
    events = catalogue()
    retained, _ = detection.drop_events(events, 1., np.random.default_rng(18),
                                       eligible_segments=("MIS6",),
                                       anchor_ids={"NGRIP": "n0", "MIS6": "m0"})
    pd.testing.assert_frame_equal(retained.loc[retained.segment_id.eq("NGRIP")], events.iloc[:3])
    assert retained.event_id.tolist() == ["n1", "n2", "n0", "m0"]
    with pytest.raises(ValueError):
        detection.drop_events(events, 0.1, np.random.default_rng(18), anchor_ids=["n1", "m0"])


def test_phase_offsets_cross_zero_circularly_and_zero_amplitude_has_no_phase():
    result = detection.paired_metrics(fit_metrics(10.), fit_metrics(350.))
    assert result["phase_offset_deg"] == pytest.approx(20.)
    assert result["phase_amplitude"] == pytest.approx(0.5)
    assert result["pre_phase_rate_ratio_max_vs_min"] == pytest.approx(np.e)
    assert result["change__beta_pre_phase_sin"] > 0
    zero = detection.paired_metrics(fit_metrics(amplitude=0.), fit_metrics())
    assert np.isnan(zero["pre_phase_preferred_deg"]) and np.isnan(zero["phase_offset_deg"])
    assert zero["pre_phase_rate_ratio_max_vs_min"] == 1.


@pytest.fixture
def deletion_inputs():
    events = catalogue()
    observations = pd.DataFrame([
        dict(segment_id="NGRIP", observation_start_kyr_bp=15., observation_end_kyr_bp=50.),
        dict(segment_id="MIS6", observation_start_kyr_bp=140., observation_end_kyr_bp=200.),
    ])
    windows = event_model.response_windows(events, observations)
    age = np.arange(0., 221.)
    forcings = {"lr04": (age, np.sin(age / 17.)), "co2": (age, np.cos(age / 23.)),
                "precession_index": (age, np.sin(age / 10.))}
    phase = (age, age * np.pi / 10.)
    scaling = event_model.nominal_scaling({k: forcings[k] for k in ("lr04", "co2")}, windows)
    baseline = ("intercept", "same_type_exponential_history", "lr04_scaled", "co2_scaled", "mis6_segment")
    return events, windows, forcings, phase, scaling, baseline, baseline + ("pre_phase_sin", "pre_phase_cos")


@pytest.fixture
def fit_spy(monkeypatch):
    calls = []

    def fitted(event_x, integral_x, weights, terms, **kwargs):
        calls.append((event_x.copy(), integral_x.copy(), np.asarray(weights).copy(), terms, kwargs))
        if not len(event_x):
            return fit_point_process(event_x, integral_x, weights, terms, **kwargs)
        beta = np.zeros(len(terms))
        beta[terms.index("same_type_exponential_history")] = -0.5
        if "pre_phase_sin" in terms:
            beta[terms.index("pre_phase_sin")] = -0.1
            beta[terms.index("pre_phase_cos")] = 0.5
        loglik = -10. + (2. if "pre_phase_sin" in terms else 0.)
        return SimpleNamespace(beta=beta, terms=tuple(terms), log_likelihood=loglik,
                               aic=2 * len(terms) - 2 * loglik, converged=True)

    monkeypatch.setattr(detection, "fit_point_process", fitted)
    return calls


def test_each_retained_subset_is_fitted_and_matching_masks_are_reproducible(deletion_inputs, fit_spy, monkeypatch):
    events = deletion_inputs[0]
    seen = []
    original_build = event_model.build_design

    def inspect_design(subset, *args, **kwargs):
        seen.append(subset.event_id.to_numpy())
        assert {"n0", "m0"}.issubset(set(subset.event_id))
        return original_build(subset, *args, **kwargs)

    monkeypatch.setattr(event_model, "build_design", inspect_design)
    arguments = dict(probabilities=(0., 0.1, 0.2), catalogue_id="test", show_progress=False,
                     scopes={"both": None, "MIS6_only": ("MIS6",)}, n_replicates=20, seed=93)
    result = detection.analyze_deletions(*deletion_inputs, **arguments)
    assert len(seen) == 121  # Nominal pair plus every retained catalogue.
    assert len(fit_spy) == 242
    for mask, visible_ids in zip(result["retained_masks"], seen[1:]):
        np.testing.assert_array_equal(events.event_id.to_numpy()[mask], visible_ids)
    for reduced, full in zip(fit_spy[::2], fit_spy[1::2]):
        np.testing.assert_array_equal(reduced[2], full[2])
        np.testing.assert_array_equal(full[4]["start_beta"], [0., -0.5, 0., 0., 0., 0., 0.])
    repeated = detection.analyze_deletions(*deletion_inputs, **arguments)
    pd.testing.assert_frame_equal(result["replicates"], repeated["replicates"])
    np.testing.assert_array_equal(result["retained_masks"], repeated["retained_masks"])
    assert np.all(result["retained_masks"][40:60] <= result["retained_masks"][20:40])
    assert np.all(result["retained_masks"][20:40] <= result["retained_masks"][80:100])
    assert "pre_phase_preferred_deg" not in result["scenario_summary"].metric.tolist()
    assert "phase_offset_deg" in result["scenario_summary"].metric.tolist()


def test_deletion_count_matches_bernoulli_rate_and_is_not_a_fixed_count():
    events = catalogue()
    counts = np.array([len(events) - len(detection.drop_events(
        events, 0.2, np.random.default_rng(np.random.SeedSequence([913, replicate])))[0])
        for replicate in range(1000)])
    # Four eligible events, independently removed with probability 0.2.
    assert abs(counts.mean() - 0.8) < 0.08
    assert 0 in counts and np.any(counts >= 2)


def test_fit_failures_keep_masks_and_are_visible_in_summary_denominators(deletion_inputs, fit_spy, monkeypatch):
    original_fit = detection.fit_point_process

    def fail_after_nominal(*args, **kwargs):
        if len(fit_spy) >= 2:
            raise PointProcessFitError("known optimization failure")
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(detection, "fit_point_process", fail_after_nominal)
    result = detection.analyze_deletions(*deletion_inputs, catalogue_id="test", show_progress=False,
                                         probabilities=(0.2,), n_replicates=4, seed=29)
    assert result["retained_masks"].shape == (4, 6)
    assert not result["replicates"].fit_valid.any()
    summary = result["scenario_summary"]
    assert (summary.n_invalid == 4).all() and (summary.status == "incomplete_fits").all()
    assert summary.loc[summary.metric.eq("gain_bits_per_event"), "n_finite"].iloc[0] == 0
    assert summary.loc[summary.metric.eq("n_deleted"), "n_finite"].iloc[0] == 4


def test_empty_response_retains_masks_and_reports_unidentified_fit(deletion_inputs, fit_spy):
    result = detection.analyze_deletions(*deletion_inputs, catalogue_id="test", show_progress=False,
                                         probabilities=(1.,), n_replicates=2)
    rows = result["replicates"]
    assert not rows.fit_valid.any()
    assert (rows.response_status == "no_response_events").all()
    assert rows.invalid_reason.str.contains("Nonconverged").all()
    assert rows.gain_bits_per_event.isna().all()
    assert (rows.n_deleted == 4).all()
    assert np.all(result["retained_masks"] == [False, False, True, False, True, False])


def test_invalid_membership_and_programming_errors_do_not_become_retryable_draws(deletion_inputs, fit_spy, monkeypatch):
    with pytest.raises(ValueError):
        detection.drop_events(catalogue(), 1.1, np.random.default_rng(3))
    with pytest.raises(ValueError):
        detection.drop_events(catalogue(), 0.1, np.random.default_rng(3), eligible_segments=("unknown",))
    with pytest.raises(ValueError):
        detection.analyze_deletions(*deletion_inputs, catalogue_id="test", n_replicates=0)

    def programming_error(*args, **kwargs):
        raise KeyError("incorrect predictor")

    monkeypatch.setattr(detection, "fit_point_process", programming_error)
    with pytest.raises(KeyError):
        detection.analyze_deletions(*deletion_inputs, catalogue_id="test", n_replicates=1)
