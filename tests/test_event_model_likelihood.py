"""Scientific invariants of the explicit continuous NGRIP--MIS6 model."""
import numpy as np
import pandas as pd
import pytest
import NGRIP_MIS6_event_phase_analysis as nominal
import Barker2011.Barker2011_event_phase_analysis as barker
from toolbox import event_model as model
from toolbox.project_config import BARKER_EVENT_CSVS

AGE = "event_age_kyr_bp"
HISTORY = "same_type_exponential_history"


@pytest.fixture(scope="module")
def fitted():
    return nominal.run_analysis()


def test_frozen_catalogue_and_chosen_model(fitted):
    events = fitted["events"]
    assert len(events) == 55 and events.event_id.is_unique
    assert events.groupby("segment_id").size().to_dict() == {"MIS6": 21, "NGRIP": 34}
    assert fitted["reduced"].terms == ("intercept", HISTORY, "lr04_scaled", "co2_scaled", "mis6_segment")
    assert fitted["full"].terms == fitted["reduced"].terms + ("pre_phase_sin", "pre_phase_cos")
    assert fitted["statistics"]["history_tau_kyr"] == 1.5
    assert fitted["statistics"]["initial_unobserved_history"] == 0


@pytest.mark.parametrize("definition", ("variable_threshold", "fixed_threshold"))
def test_barker_uses_two_column_input_without_modifying_ages(definition):
    source = pd.read_csv(BARKER_EVENT_CSVS[definition], float_precision="round_trip")
    result = barker.run_analysis(definition)
    assert list(source.columns) == ["event_id", AGE]
    pd.testing.assert_frame_equal(result["events"][source.columns], source, check_exact=True)
    assert result["windows"].anchor_age_kyr_bp.item() == source[AGE].max()


def test_exact_anchors_define_exposure_without_bins_or_gap(fitted):
    assert len(fitted["event_features"]) == 53
    assert fitted["events"].event_role.eq("conditioning").sum() == 2
    for window in fitted["windows"].itertuples(index=False):
        ages = fitted["events"].loc[lambda e: e.segment_id.eq(window.segment_id), AGE]
        assert window.anchor_age_kyr_bp == window.response_end_kyr_bp == ages.max()
        assert window.response_start_kyr_bp == window.observation_start_kyr_bp
        nodes = fitted["integration_features"].loc[lambda x: x.segment_id.eq(window.segment_id)]
        assert nodes.age_kyr_bp.between(window.response_start_kyr_bp, window.anchor_age_kyr_bp).all()
        assert nodes.weight.sum() == pytest.approx(window.anchor_age_kyr_bp - window.response_start_kyr_bp)
        for frame in (fitted["event_features"], nodes):
            part = frame.loc[frame.segment_id.eq(window.segment_id)]
            np.testing.assert_allclose(part.elapsed_kyr, window.anchor_age_kyr_bp - part.age_kyr_bp)
            assert (part.elapsed_kyr > 0).all()
    assert fitted["integration_features"].weight.gt(0).all()


def test_forcing_scaling_uses_nominal_time_exposure(fitted):
    nodes = fitted["integration_features"]
    for forcing in ("lr04", "co2"):
        assert np.average(nodes[forcing + "_scaled"], weights=nodes.weight) == pytest.approx(0, abs=2e-13)
        scale = fitted["scaling"].loc[forcing]
        assert scale["range"] == pytest.approx(scale["max"] - scale["min"])


def test_strict_history_initialization_and_alternative_boundaries(fitted):
    for window in fitted["windows"].itertuples(index=False):
        ages = fitted["events"].loc[lambda e: e.segment_id.eq(window.segment_id), AGE].to_numpy()
        args = (ages, ages, window, fitted["forcings"], fitted["phase_anchors"], fitted["scaling"])
        frame = model.evaluate_features(*args)
        expected = [np.exp(-(ages[ages > age] - age) / 1.5).sum() for age in ages]
        np.testing.assert_allclose(frame[HISTORY], expected, atol=1e-14)
        assert frame[HISTORY].iloc[-1] == frame.elapsed_kyr.iloc[-1] == 0
        initialized = model.evaluate_features(*args, initial_history=0.5)
        np.testing.assert_allclose(initialized[HISTORY] - frame[HISTORY],
            0.5 * np.exp((ages - window.anchor_age_kyr_bp) / 1.5), atol=1e-15)
        exits = ages - 1.5
        query = np.r_[ages, exits, np.nextafter(exits, -np.inf), np.nextafter(exits, np.inf)]
        query = query[(query >= window.response_start_kyr_bp) & (query <= window.anchor_age_kyr_bp)]
        frame = model.evaluate_features(query, ages, window, fitted["forcings"],
            fitted["phase_anchors"], fitted["scaling"], history_variants=True)
        gaps, counts = [], []
        for age in query:
            earlier = ages[ages > age]
            gaps.append(earlier.min() - age if len(earlier) else 0)
            counts.append(np.sum(earlier < age + 1.5))
        np.testing.assert_allclose(frame.time_since_last_event_kyr, gaps, atol=3e-14)
        np.testing.assert_allclose(frame.log_time_since_last_event, np.log1p(gaps), atol=3e-14)
        np.testing.assert_array_equal(frame.rectangular_history_count, counts)


def test_small_age_changes_are_not_coalesced_into_count_patterns(fitted):
    events = fitted["events"].copy()
    event_id = events.loc[0, "event_id"]
    events.loc[0, AGE] = events.loc[0, AGE] + .001
    new, _ = model.build_design(events, fitted["windows"], fitted["forcings"],
                                fitted["phase_anchors"], fitted["scaling"])
    new, old = new.set_index("event_id"), fitted["event_features"].set_index("event_id")
    assert new.loc[event_id, "age_kyr_bp"] - old.loc[event_id, "age_kyr_bp"] == pytest.approx(.001)
    assert new.loc[event_id, HISTORY] != old.loc[event_id, HISTORY]
    assert new.loc[event_id, "pre_phase_unwrapped_rad"] != old.loc[event_id, "pre_phase_unwrapped_rad"]


def test_age_draw_reconditions_anchor_but_preserves_nominal_scales(fitted):
    events = fitted["events"].copy()
    idx = events.loc[events.segment_id.eq("NGRIP"), AGE].idxmax()
    events.loc[idx, AGE] = events.loc[idx, AGE] + .1
    args = (fitted["forcings"], fitted["phase_anchors"], fitted["scaling"])
    original_scaling = fitted["scaling"].copy(deep=True)
    with pytest.raises(ValueError, match="original conditioning"):
        model.build_design(events, fitted["windows"], *args)
    windows = model.response_windows(events, fitted["observations"])
    new, integral = model.build_design(events, windows, *args)
    assert integral.weight.sum() == pytest.approx(165.158)
    assert len(new) == 53
    pd.testing.assert_frame_equal(fitted["scaling"], original_scaling, check_exact=True)
    old = fitted["event_features"].set_index("event_id")
    new = new.set_index("event_id")
    for name, shift in (("NGRIP", .1), ("MIS6", 0)):
        mask = new.segment_id.eq(name)
        np.testing.assert_allclose(new.loc[mask, "elapsed_kyr"] - old.loc[mask, "elapsed_kyr"], shift, atol=3e-14)


def test_likelihood_event_sum_integral_and_no_bin_sample_size(fitted):
    full, reduced = fitted["full"], fitted["reduced"]
    events, nodes = fitted["event_features"], fitted["integration_features"]
    assert full.log_likelihood == pytest.approx(
        (events.loc[:, full.terms].to_numpy() @ full.beta).sum()
        - nodes.weight @ np.exp(nodes.loc[:, full.terms].to_numpy() @ full.beta))
    stats = fitted["statistics"]
    assert stats["gain_bits_per_event"] == pytest.approx((full.log_likelihood-reduced.log_likelihood)/(53*np.log(2)))
    assert stats["delta_AIC_full_minus_reduced"] == pytest.approx(4-stats["LR_statistic"])
    assert not any("AICc" in name or "bin" in name for name in stats)
    assert stats["beta_history"] <= 0


def test_rescaling_keeps_terminal_censoring_and_separate_segments(fitted):
    frame, event_x = fitted["integration_features"], fitted["event_features"]
    full, windows = fitted["full"], fitted["windows"]
    events, cumulative, tails = model.rescaled_event_intervals(event_x, frame, windows, full)
    rate = np.exp(frame.loc[:, full.terms].to_numpy() @ full.beta)
    for window in windows.itertuples(index=False):
        name = window.segment_id
        mask = frame.segment_id.eq(name).to_numpy()
        assert np.all(np.diff(events[name]) > 0) and np.all(np.diff(cumulative[name]) > 0)
        assert tails[name] >= 0
        assert cumulative[name][-1] + tails[name] == pytest.approx(rate[mask] @ frame.loc[mask, "weight"])
        ages = window.anchor_age_kyr_bp - events[name]
        node_ages = frame.loc[mask, "age_kyr_bp"].to_numpy()
        mass = rate[mask] * frame.loc[mask, "weight"].to_numpy()
        np.testing.assert_allclose(cumulative[name], [mass[node_ages > a].sum() for a in ages], rtol=2e-13)
        assert tails[name] == pytest.approx(mass[node_ages < ages[-1]].sum())
    shuffled = model.rescaled_event_intervals(event_x.sample(frac=1, random_state=975).reset_index(drop=True),
        frame.sample(frac=1, random_state=246).reset_index(drop=True), windows, full)
    for expected, actual in zip((events, cumulative, tails), shuffled):
        for name in windows.segment_id:
            np.testing.assert_allclose(actual[name], expected[name], rtol=2e-13)


def test_fitted_rate_curves_retain_history_jump_at_every_event(fitted):
    rates = fitted["fitted_rates"].set_index(["segment_id", "age_kyr_bp"])
    for window in fitted["windows"].itertuples(index=False):
        for age in fitted["events"].loc[lambda e: e.segment_id.eq(window.segment_id), AGE]:
            after = np.nextafter(age, -np.inf)
            if after < window.response_start_kyr_bp:
                continue
            for name in ("reduced", "full"):
                fit = fitted[name]
                ratio = rates.loc[(window.segment_id, after), name+"_rate"] / rates.loc[(window.segment_id, age), name+"_rate"]
                assert ratio == pytest.approx(np.exp(fit.beta[fit.terms.index(HISTORY)]), rel=2e-11)


def test_simulation_envelopes_bound_background_inside_intervals(fitted):
    prepared = model.prepare_model_simulation(fitted["events"], fitted["windows"], fitted["forcings"],
        fitted["phase_anchors"], fitted["scaling"], fitted["full"])
    for part in prepared:
        for lo, hi, upper in zip(part["breakpoints"][:-1], part["breakpoints"][1:], part["log_upper_bounds"]):
            assert np.max(part["log_background"](np.linspace(lo, hi, 11))) <= upper + 1e-12


def test_event_phase_sampling_uses_bp1950_phase_convention(fitted):
    args = (fitted["forcings"]["precession_index"], fitted["phase_anchors"])
    phases = model.sample_event_phases(fitted["events"], *args)
    assert len(phases) == 55 and phases.pre_phase_deg.between(0, 360).all()
    assert not phases.pre_phase_extrapolated.any()
    np.testing.assert_allclose(np.sin(phases.pre_phase_rad), np.sin(np.deg2rad(phases.pre_phase_deg)), atol=1e-12)
    minimal = fitted["events"][["event_id", AGE]].copy()
    original = minimal.copy(deep=True)
    minimal_phases = model.sample_event_phases(minimal, *args)
    pd.testing.assert_frame_equal(minimal_phases, phases[minimal_phases.columns], check_exact=True)
    pd.testing.assert_frame_equal(minimal, original, check_exact=True)


def test_analysis_reads_prepared_forcings_without_raw_parsers(monkeypatch):
    import xarray as xr
    def reject_raw(*args, **kwargs):
        raise AssertionError("Raw forcing parsing belongs in preprocessing")
    monkeypatch.setattr(pd, "read_excel", reject_raw)
    monkeypatch.setattr(np, "loadtxt", reject_raw)
    monkeypatch.setattr(xr, "open_dataset", reject_raw)
    fitted = nominal.run_analysis()
    assert len(fitted["event_phases"]) == 55
    assert len(fitted["forcings"]["precession_index"][0]) == 10601
    assert fitted["forcings"]["precession_index"][0][0] == pytest.approx(-60.05)
