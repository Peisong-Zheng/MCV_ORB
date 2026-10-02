"""Scientific boundaries of the explicit event and integration matrices."""
import numpy as np
import pandas as pd
import pytest

from toolbox import event_model as model
from toolbox.point_process import fit_point_process


@pytest.fixture
def inputs():
    events = pd.DataFrame({
        "event_id": ["a1", "a2", "b1", "b2"],
        "segment_id": ["A", "A", "B", "B"],
        "event_age_kyr_bp": [2., 8., 22., 28.],
    })
    observations = pd.DataFrame({
        "segment_id": ["A", "B"],
        "observation_start_kyr_bp": [0., 20.],
        "observation_end_kyr_bp": [10., 30.],
    })
    forcings = {"climate": (np.array([0., 10., 20., 30.]), np.array([0., 2., 1., 3.]))}
    anchors = (np.array([0., 30.]), np.array([0., 2 * np.pi]))
    windows = model.response_windows(events, observations)
    scaling = model.nominal_scaling(forcings, windows)
    return events, observations, windows, forcings, anchors, scaling


def test_segments_condition_separately_and_keep_event_free_tails(inputs):
    events, _, windows, forcings, anchors, scaling = inputs
    event_x, integral_x = model.build_design(events, windows, forcings, anchors, scaling)
    assert event_x.event_id.tolist() == ["a1", "b1"]
    # Each response sees just its own anchor six kyr earlier.
    np.testing.assert_allclose(event_x.same_type_exponential_history, np.exp(-6 / 1.5))
    assert integral_x.groupby("segment_id").weight.sum().to_dict() == pytest.approx({"A": 8., "B": 8.})
    for name, response_age in (("A", 2.), ("B", 22.)):
        tail = integral_x.segment_id.eq(name) & (integral_x.age_kyr_bp < response_age)
        assert integral_x.loc[tail, "weight"].sum() == pytest.approx(2.)


def test_anchor_only_realizations_retain_exposure_and_zero_event_likelihood(inputs):
    events, _, windows, forcings, anchors, scaling = inputs
    anchors_only = events.iloc[[1, 3]]
    event_x, integral_x = model.build_design(anchors_only, windows, forcings, anchors, scaling)
    fit = fit_point_process(event_x[["intercept"]], integral_x[["intercept"]],
                            integral_x.weight, ["intercept"])
    assert len(event_x) == 0 and integral_x.weight.sum() == pytest.approx(16.)
    assert fit.status == "zero_events" and fit.log_likelihood == 0
    assert not fit.identifiable and np.isnan(fit.beta).all()


def test_chronology_changes_anchor_but_keeps_supplied_nominal_scaling(inputs):
    events, observations, windows, forcings, anchors, scaling = inputs
    original_scaling = scaling.copy(deep=True)
    shifted = events.copy()
    shifted.loc[1, "event_age_kyr_bp"] = shifted.loc[1, "event_age_kyr_bp"] + 1
    with pytest.raises(ValueError, match="conditioning event"):
        model.build_design(shifted, windows, forcings, anchors, scaling)
    new_windows = model.response_windows(shifted, observations)
    event_x, integral_x = model.build_design(shifted, new_windows, forcings, anchors, scaling)
    assert integral_x.weight.sum() == pytest.approx(17.)
    np.testing.assert_allclose(event_x.same_type_exponential_history,
                               [np.exp(-7 / 1.5), np.exp(-6 / 1.5)])
    pd.testing.assert_frame_equal(scaling, original_scaling, check_exact=True)
    # These ages moved only the conditioning event; response climate is unchanged.
    original_events, _ = model.build_design(events, windows, forcings, anchors, scaling)
    np.testing.assert_array_equal(event_x.climate_scaled, original_events.climate_scaled)


@pytest.mark.parametrize("age", [-1., 8., np.nan])
def test_fixed_windows_reject_outside_duplicate_or_nonfinite_events(inputs, age):
    events, _, windows, forcings, anchors, scaling = inputs
    changed = events.copy()
    changed.loc[0, "event_age_kyr_bp"] = age
    with pytest.raises(ValueError):
        model.build_design(changed, windows, forcings, anchors, scaling)


@pytest.mark.parametrize("bad_ages,bad_values", [
    ([0., 20., 10., 30.], [0., 2., 1., 3.]),
    ([0., 10., 10., 30.], [0., 2., 1., 3.]),
    ([0., 10., 20., 30.], [0., 2., np.nan, 3.]),
])
def test_design_rejects_invalid_native_forcing_nodes(inputs, bad_ages, bad_values):
    events, _, windows, _, anchors, scaling = inputs
    forcing = {"climate": (np.array(bad_ages), np.array(bad_values))}
    with pytest.raises(ValueError, match="finite values at unique increasing ages"):
        model.build_design(events, windows, forcing, anchors, scaling)
