"""Check the time direction, exposure and reproducibility of Barker's null test."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Barker2011 import Barker2011_PI_bootstrap as bootstrap
from toolbox import combined_pi


class RecordingPoissonRNG:
    def __init__(self, values):
        self.values = iter(values)
        self.means = []

    def poisson(self, mean):
        self.means.append(float(mean))
        return next(self.values)


def test_oldest_first_dynamic_history_and_short_bin_exposure():
    segment = combined_pi.SegmentContext(
        segment_id="Barker2011", observation_start_kyr_bp=0,
        observation_end_kyr_bp=2.5, response_start_kyr_bp=0,
        response_end_kyr_bp=1.5, bin_edges=np.array([0, 1, 1.5, 2.5]),
        response_mask=np.array([True, True, False]),
        history_left=np.array([1, 2, 3]), history_right=np.array([3, 3, 3]))
    context = dict(segment=segment, bins=pd.DataFrame(dict(
        dt_ka=[1, .5, 1], lr04_scaled=np.zeros(3), co2_scaled=np.zeros(3))))
    prepared = bootstrap.prepare_simulation(context, np.array([0, np.log(2), 0, 0]))
    rng = RecordingPoissonRNG([1, 2, 0])
    counts = bootstrap.simulate_counts(prepared, rng)
    np.testing.assert_array_equal(counts, [0, 2, 1])
    # The oldest history-only bin seeds later response history; middle dt is 0.5.
    np.testing.assert_allclose(rng.means, [1, 1, 8])
    np.testing.assert_array_equal(combined_pi.history_from_counts(counts, segment), [3, 1, 0])


@pytest.fixture(scope="module")
def point_setup():
    events = bootstrap.main_analysis.load_barker_source()
    context = bootstrap.age_sensitivity.prepare_context(events)
    models, tests = bootstrap.main_analysis.fit_models(context["frame"])
    return events, context, models, tests


def test_primary_catalogue_support_and_observed_statistic(point_setup):
    events, context, models, tests = point_setup
    assert len(events) == 70
    assert events.event_definition.eq("variable_threshold").all()
    assert len(context["frame"]) == 1993
    assert context["frame"].dt_ka.sum() == pytest.approx(398.5)
    assert context["segment"].bin_edges[-1] == 400
    assert bootstrap.main_analysis.HISTORY_WINDOW_KA == 1.5
    assert bootstrap.main_analysis.BIN_WIDTH_KA == .2
    assert "mis6_segment" not in bootstrap.main_analysis.REDUCED_TERMS
    counts, _ = np.histogram(events.event_age_ka, bins=context["segment"].bin_edges)
    result = bootstrap.fit_counts(counts, context)
    assert result["n_events_response"] == 70
    assert result["LR_statistic"] == pytest.approx(tests.iloc[0].LR_statistic, abs=1e-9)
    assert result["info_bits_per_event"] == pytest.approx(.0893604569525, abs=1e-8)
    assert result["LR_statistic"] == pytest.approx(
        2 * result["info_bits_per_event"] * 70 * np.log(2))
    assert len(models[0].beta) == 4


def test_replicates_reproduce_across_worker_counts(point_setup):
    _, context, models, _ = point_setup
    first, rejects_first = bootstrap.run_bootstrap(context, models[0].beta,
        n_bootstrap=3, seed=14601, n_workers=1)
    second, rejects_second = bootstrap.run_bootstrap(context, models[0].beta,
        n_bootstrap=3, seed=14601, n_workers=2)
    pd.testing.assert_frame_equal(first, second)
    assert rejects_first == rejects_second
    assert first.reduced_converged.all() and first.full_converged.all()
    assert first.likelihood_nesting_ok.all() and not first.eta_clipping_used.any()
    np.testing.assert_allclose(first.LR_statistic, 2 * first.ll_gain_nats)
    np.testing.assert_allclose(first.info_bits_per_event,
                               first.ll_gain_nats / (first.n_events_response * np.log(2)))


def test_reject_bad_count_vectors(point_setup):
    _, context, _, _ = point_setup
    counts = np.zeros(len(context["bins"]))
    counts[0] = -.5
    with pytest.raises(ValueError, match="integer count"):
        bootstrap.fit_counts(counts, context)
    with pytest.raises(ValueError, match="coefficients"):
        bootstrap.prepare_simulation(context, np.zeros(5))
