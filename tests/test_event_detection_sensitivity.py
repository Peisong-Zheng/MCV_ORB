"""Deletion changes membership and visible history, never ages or anchors."""

import numpy as np
import pandas as pd
import pytest

from toolbox import event_detection_sensitivity as detection


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


def test_each_retained_subset_is_fitted_and_matching_masks_are_reproducible():
    events = catalogue()
    seen = []

    def fit(subset):
        seen.append(subset.event_id.to_numpy())
        # Stand-in for the scientific fit: its history uses only visible events.
        history_events = subset.loc[subset.event_age_kyr_bp > 25, "event_id"].tolist()
        assert set(history_events).issubset(set(subset.event_id))
        assert {"n0", "m0"}.issubset(set(subset.event_id))
        return fit_metrics(gain=0.1 + len(subset) * 0.01)

    arguments = dict(probabilities=(0., 0.1, 0.2),
                     scopes={"both": None, "MIS6_only": ("MIS6",)}, n_replicates=20, seed=93)
    result = detection.run_deletion_sensitivity(events, fit, fit_metrics(), **arguments)
    assert len(seen) == 120
    for mask, visible_ids in zip(result["retained_masks"], seen):
        np.testing.assert_array_equal(events.event_id.to_numpy()[mask], visible_ids)
    repeated = detection.run_deletion_sensitivity(events, fit, fit_metrics(), **arguments)
    pd.testing.assert_frame_equal(result["replicates"], repeated["replicates"])
    np.testing.assert_array_equal(result["retained_masks"], repeated["retained_masks"])
    # Shared random uniforms pair loss scenarios without ever altering event ages.
    assert np.all(result["retained_masks"][40:60] <= result["retained_masks"][20:40])
    assert np.all(result["retained_masks"][20:40] <= result["retained_masks"][80:100])
    assert "pre_phase_preferred_deg" not in result["scenario_summary"].metric.tolist()
    assert "phase_offset_deg" in result["scenario_summary"].metric.tolist()


def test_deletion_count_matches_bernoulli_rate_and_is_not_a_fixed_count():
    result = detection.run_deletion_sensitivity(
        catalogue(), lambda events: fit_metrics(), fit_metrics(),
        probabilities=(0.2,), n_replicates=1000, seed=913)
    counts = result["replicates"].n_deleted.to_numpy()
    # Four eligible events, independently removed with probability 0.2.
    assert abs(counts.mean() - 0.8) < 0.08
    assert 0 in counts and np.any(counts >= 2)


def test_fit_failures_keep_masks_and_are_visible_in_summary_denominators():
    def fail(subset):
        raise detection.DeletionFitFailure("known optimization failure")

    result = detection.run_deletion_sensitivity(catalogue(), fail, fit_metrics(),
                                                probabilities=(0.2,), n_replicates=4, seed=29)
    assert result["retained_masks"].shape == (4, 6)
    assert not result["replicates"].fit_valid.any()
    summary = result["scenario_summary"]
    assert (summary.n_invalid == 4).all() and (summary.status == "incomplete_fits").all()
    assert summary.loc[summary.metric.eq("gain_bits_per_event"), "n_finite"].iloc[0] == 0
    assert summary.loc[summary.metric.eq("n_deleted"), "n_finite"].iloc[0] == 4


def test_legal_empty_response_is_retained_with_zero_lr_and_unidentified_effects():
    def empty_fit(events):
        assert set(events.event_id) == {"n0", "m0"}
        return dict(gain_bits_per_event=np.nan, LR_statistic=0., beta_history=np.nan,
                    beta_pre_phase_sin=np.nan, beta_pre_phase_cos=np.nan)

    result = detection.run_deletion_sensitivity(catalogue(), empty_fit, fit_metrics(),
                                                probabilities=(1.,), n_replicates=2)
    rows = result["replicates"]
    assert rows.fit_valid.all() and (rows.response_status == "no_response_events").all()
    assert (rows.LR_statistic == 0.).all() and rows.gain_bits_per_event.isna().all()
    assert (rows.n_deleted == 4).all()


def test_invalid_membership_and_callback_contract_do_not_become_retryable_draws():
    with pytest.raises(ValueError):
        detection.drop_events(catalogue(), 1.1, np.random.default_rng(3))
    with pytest.raises(ValueError):
        detection.drop_events(catalogue(), 0.1, np.random.default_rng(3), eligible_segments=("unknown",))
    with pytest.raises(ValueError):
        detection.run_deletion_sensitivity(catalogue(), lambda events: {}, fit_metrics(),
                                           n_replicates=1)

    def programming_error(events):
        raise KeyError("incorrect predictor")

    with pytest.raises(KeyError):
        detection.run_deletion_sensitivity(catalogue(), programming_error, fit_metrics(), n_replicates=1)
