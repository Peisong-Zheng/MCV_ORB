"""Scientific checks for the control interpolation and chronological G refits."""

import re

import numpy as np
import pandas as pd
import pytest
from pypdf import PdfReader
from toolbox import combined_likelihood as c, age_sensitivity

from Barker2011 import Barker2011_event_age_uncertainty as age_mc
from Barker2011 import Barker2011_event_phase_analysis as main_analysis
from Barker2011 import Barker2011_event_uncertainty_sensitivity as sensitivity


@pytest.fixture(scope="module")
def prepared():
    controls = age_mc.prepare_controls()
    events = age_mc.prepare_events(controls)
    context = sensitivity.prepare_context()
    return controls, events, context


def test_table_s1_transcription_matches_source_pdf():
    text = PdfReader(age_mc.SOURCE_PDF).pages[-1].extract_text()
    number = r"-?\d+\.\d+"
    rows = re.findall(rf"^\s*({number}(?:\s+{number}){{5}})\s*$", text, re.MULTILINE)
    published = np.array([[float(value) for value in row.split()] for row in rows])
    raw = pd.read_csv(age_mc.CONTROL_CSV)
    assert published.shape == raw.shape == (60, 6)
    np.testing.assert_array_equal(raw.to_numpy(), published)


def test_auxiliary_control_and_gap_membership(prepared):
    controls, events, _ = prepared
    assert len(controls) == 61 and len(events) == 70
    auxiliary = controls.iloc[-1]
    assert auxiliary.control_type == "auxiliary" and auxiliary.speleo_age_ka == 400
    assert auxiliary.combined_uncertainty_ka == 1.30
    assert auxiliary.unfloored_extrapolation_ka == pytest.approx(1.114818, abs=1e-6)
    assert np.isnan(auxiliary.source_table_row) and np.isnan(auxiliary.absolute_speleo_error_ka)
    assert events.in_published_alignment_gap.sum() == 7
    assert events.in_long_control_interval.sum() == 8
    assert events.beyond_last_published_control.sum() == 4
    gap = events.loc[events.in_long_control_interval]
    assert gap.left_control_id.eq("S1_51").all() and gap.right_control_id.eq("S1_52").all()
    tail = events.loc[events.beyond_last_published_control]
    assert tail.left_control_id.eq("S1_60").all() and tail.right_control_id.eq("AUX_400").all()
    assert events.event_id.is_unique


def test_fixed_speleo_interpolation_and_uniform_proposal_variance():
    controls = pd.DataFrame(dict(speleo_age_ka=[10., 20.], combined_uncertainty_ka=[1., 2.]))
    offsets = np.array([[1., 2.], [-1., -2.]])
    np.testing.assert_allclose(age_mc.interpolate_offsets([10., 15., 20.], controls, offsets),
                               [[1., 1.5, 2.], [-1., -1.5, -2.]])
    events = pd.DataFrame(dict(event_age_ka=[15.]))
    draws, offsets, stats = age_mc.sample_realizations(events, controls, 8000, seed=24)
    assert stats["n_rejected_crossed_controls"] == 0
    # The midpoint is a weighted sum of independent uniforms, not Uniform(-1.5, 1.5).
    assert draws.std(ddof=1) == pytest.approx(np.sqrt(5 / 12), abs=0.02)
    assert abs(np.corrcoef(offsets.T)[0, 1]) < 0.04
    np.testing.assert_allclose(offsets.std(axis=0, ddof=1), np.array([1, 2]) / np.sqrt(3), atol=0.02)
    with pytest.raises(ValueError, match="within"):
        age_mc.interpolation_weights([20.1], controls)


def test_sampling_reproducibility_order_and_bounds(prepared):
    controls, events, _ = prepared
    first = age_mc.sample_realizations(events, controls, 1000, seed=13)
    second = age_mc.sample_realizations(events, controls, 1000, seed=13)
    for left, right in zip(first[:2], second[:2]):
        np.testing.assert_array_equal(left, right)
    draws, offsets, stats = first
    assert np.all(np.diff(draws, axis=1) > 0)
    assert np.all(np.diff(controls.speleo_age_ka.to_numpy() + offsets, axis=1) > 0)
    assert np.all(np.abs(offsets) <= controls.combined_uncertainty_ka.to_numpy())
    np.testing.assert_allclose(draws - events.event_age_ka.to_numpy(),
                               age_mc.interpolate_offsets(events.event_age_ka, controls, offsets))
    assert stats["n_proposals"] == stats["n_rejected_crossed_controls"] + stats["n_accepted_proposals"]
    assert stats["n_accepted_proposals"] == 1000 + stats["n_unused_accepted_tail"]
    assert stats["n_rejected_crossed_controls"] > 0


def test_crossed_control_proposal_is_rejected_as_a_whole(monkeypatch):
    class FixedGenerator:
        def uniform(self, low, high, size):
            offsets = np.zeros(size)
            offsets[0] = [1., -1.]
            return offsets
    monkeypatch.setattr(age_mc.np.random, "default_rng", lambda seed: FixedGenerator())
    controls = pd.DataFrame(dict(speleo_age_ka=[10., 11.], combined_uncertainty_ka=[2., 2.]))
    events = pd.DataFrame(dict(event_age_ka=[10.2, 10.8]))
    draws, offsets, stats = age_mc.sample_realizations(events, controls, 3)
    assert stats["n_rejected_crossed_controls"] == 1
    np.testing.assert_array_equal(offsets, np.zeros((3, 2)))
    np.testing.assert_allclose(draws, np.tile(events.event_age_ka, (3, 1)))


def test_nominal_gain_is_identical_to_main_and_exact_history_updates(prepared):
    controls, events, context = prepared
    point = c.fit_catalogue(context.events, context)
    reference = main_analysis.run_analysis()["summary"].iloc[0]
    for key in ("gain_bits_per_event", "LR_statistic", "pre_phase_preferred_deg"):
        assert point.summary[key] == pytest.approx(reference[key], abs=1e-9)
    assert point.summary["n_response_events"] == 69
    assert point.summary["response_exposure_kyr"] == pytest.approx(context.events[c.EVENT_AGE_COLUMN].max())
    draws, _, _ = age_mc.sample_realizations(events, controls, 3, seed=100)
    table = pd.DataFrame(draws, columns=age_mc.age_columns(events))
    table.insert(0, "realization_id", ["d1", "d2", "d3"])
    results = sensitivity.fit_realizations(table, context)
    moved = context.events.copy()
    moved[c.EVENT_AGE_COLUMN] = draws[0]
    local = c.condition_context(context, moved)
    fitted = c.fit_catalogue(moved, local, fixed_support=True)
    assert results.gain_bits_per_event.iloc[0] == pytest.approx(fitted.summary["gain_bits_per_event"])
    assert local.scaling == context.scaling
    np.testing.assert_allclose(fitted.design.event_frame.age_kyr_bp, draws[0, :-1])
    assert not np.array_equal(fitted.design.event_frame[c.HISTORY_TERM], point.design.event_frame[c.HISTORY_TERM])


def test_unsupported_draws_remain_visible_and_phase_wraps(prepared):
    _, events, context = prepared
    nominal = events.event_age_ka.to_numpy()
    ages = np.tile(nominal, (2, 1))
    ages[1, -1] = 401.0
    table = pd.DataFrame(ages, columns=age_mc.age_columns(events))
    table.insert(0, "realization_id", ["nominal", "outside"])
    results = sensitivity.fit_realizations(table, context)
    assert results.fit_valid.tolist() == [True, False]
    assert results.invalid_reason.iloc[1] == "outside_Barker2011_observation_support"
    assert np.isnan(results.gain_bits_per_event.iloc[1])
    point = c.fit_catalogue(context.events, context)
    summary = age_sensitivity.summarize(results, point.summary).iloc[0]
    assert summary.n_realizations == 2 and summary.n_valid == 1
    assert summary.fraction_nominal_p_below_0p05 == float(results.nominal_LR_p.iloc[0] < .05)
    np.testing.assert_allclose(age_sensitivity.unwrap_phase([355., 5.], 350.), [355., 365.])


def test_saved_chronologies_reconstruct_from_saved_control_ages():
    folder = age_mc.OUT_DATA_DIR
    controls = pd.read_csv(folder / "age_control_points.csv")
    events = pd.read_csv(folder / "event_catalogue_used.csv")
    draws = pd.read_csv(folder / "event_age_realizations.csv")
    control_ages = pd.read_csv(folder / "control_age_realizations.csv")
    assert len(draws) == len(control_ages) == 10000
    assert draws.realization_id.equals(control_ages.realization_id)
    offsets = control_ages.iloc[:, 1:].to_numpy() - controls.speleo_age_ka.to_numpy()
    reconstructed = events.event_age_ka.to_numpy() + age_mc.interpolate_offsets(events.event_age_ka, controls, offsets)
    np.testing.assert_allclose(draws.iloc[:, 1:], reconstructed, rtol=0, atol=2e-9)
    assert np.all(np.diff(draws.iloc[:, 1:], axis=1) > 0)


def test_selected_saved_age_sequences_are_refitted_without_changed_ids(prepared):
    _, events, context = prepared
    saved = pd.read_csv(sensitivity.AGE_INPUT, float_precision="round_trip")
    draws = saved.iloc[[0, 5000, 9999]].copy()
    results = sensitivity.fit_realizations(draws, context)
    assert results.realization_id.tolist() == draws.realization_id.tolist()
    assert results.fit_valid.all() and results.n_response_events.eq(69).all()
    assert results.all_models_converged.all() and results.likelihood_nesting_ok.all()
    np.testing.assert_allclose(results.gain_bits_per_event,
                               results.LR_statistic / (2 * results.n_response_events * np.log(2)), atol=1e-11)
    for i, (_, row) in enumerate(draws.iterrows()):
        moved = context.events.copy()
        moved[c.EVENT_AGE_COLUMN] = row[age_mc.age_columns(events)].to_numpy(float)
        fresh = c.fit_catalogue(moved, context)
        assert results.gain_bits_per_event.iloc[i] == pytest.approx(fresh.summary["gain_bits_per_event"])
