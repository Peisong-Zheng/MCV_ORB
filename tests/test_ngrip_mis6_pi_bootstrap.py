"""Scientific and numerical checks for the pooled-event PI bootstrap."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import NGRIP_MIS6_PI_bootstrap as bootstrap
from toolbox import combined_pi


class RecordingPoissonRNG:
    """Return fixed counts while retaining the Poisson means passed in."""

    def __init__(self, values):
        self.values = iter(values)
        self.means = []

    def poisson(self, mean):
        self.means.append(float(mean))
        return next(self.values)


def toy_segment(segment_id: str, start: float) -> tuple[pd.DataFrame, combined_pi.SegmentContext]:
    edges = np.array([start, start + 1.0, start + 2.0, start + 3.0])
    frame = pd.DataFrame(
        {
            "segment_id": segment_id,
            "dt_kyr": np.ones(3),
            "lr04_scaled": np.zeros(3),
            "co2_scaled": np.zeros(3),
            "mis6_segment": float(segment_id == "MIS6"),
        }
    )
    segment = combined_pi.SegmentContext(
        segment_id=segment_id,
        observation_start_kyr_bp=float(edges[0]),
        observation_end_kyr_bp=float(edges[-1]),
        response_start_kyr_bp=float(edges[0]),
        response_end_kyr_bp=float(edges[-1]),
        bin_edges=edges,
        response_mask=np.ones(3, dtype=bool),
        history_left=np.array([1, 2, 3]),
        history_right=np.array([3, 3, 3]),
    )
    return frame, segment


def test_simulation_runs_oldest_to_youngest_with_dynamic_history():
    frame, segment = toy_segment("NGRIP", 0.0)
    rng = RecordingPoissonRNG([1, 2, 0])
    beta = np.array([0.0, np.log(2.0), 0.0, 0.0, 0.0])

    counts = bootstrap.simulate_segment_counts(frame, segment, beta, rng)

    # Draw order is oldest -> youngest.  The history counts are then 0, 1, 3.
    assert counts.tolist() == [0, 2, 1]
    np.testing.assert_allclose(rng.means, [1.0, 2.0, 8.0])
    np.testing.assert_array_equal(
        combined_pi.history_from_counts(counts, segment), [3.0, 1.0, 0.0]
    )


def test_segments_reset_history_across_the_unobserved_gap():
    ngrip_frame, ngrip_segment = toy_segment("NGRIP", 0.0)
    mis6_frame, mis6_segment = toy_segment("MIS6", 100.0)
    bins = pd.concat([ngrip_frame, mis6_frame], ignore_index=True)
    context = combined_pi.PIContext(
        bins=bins,
        segments={"NGRIP": ngrip_segment, "MIS6": mis6_segment},
        response_dt=np.ones(6),
        history_window_ka=2.0,
        bin_width_ka=1.0,
        origin_fraction=0.0,
        response_mode="common_core",
    )
    rng = RecordingPoissonRNG([1, 0, 0, 1, 0, 0])
    beta = np.array([0.0, np.log(2.0), 0.0, 0.0, 0.0])

    counts = bootstrap.simulate_catalogue_counts(context, beta, rng)

    # The oldest bin of each segment is drawn with zero prior history.
    assert rng.means[0] == pytest.approx(1.0)
    assert rng.means[3] == pytest.approx(1.0)
    assert counts["NGRIP"].tolist() == [0, 0, 1]
    assert counts["MIS6"].tolist() == [0, 0, 1]


def test_dynamic_history_is_not_the_fixed_observed_history():
    frame, segment = toy_segment("NGRIP", 0.0)
    rng = RecordingPoissonRNG([1, 1, 0])
    beta = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    counts = bootstrap.simulate_segment_counts(frame, segment, beta, rng)
    simulated_history = combined_pi.history_from_counts(counts, segment)

    np.testing.assert_array_equal(simulated_history, [2.0, 1.0, 0.0])
    assert not np.array_equal(simulated_history, np.zeros(3))


def test_plus_one_p_value_and_exact_binomial_interval():
    p_value, exceedances = bootstrap.empirical_p_value(
        np.array([0.5, 2.0, 3.0, 4.0]), 3.0
    )
    assert exceedances == 2
    assert p_value == pytest.approx(3.0 / 5.0)

    low, high = bootstrap.clopper_pearson_interval(0, 99)
    assert low == 0.0
    assert 0.0 < high < 0.05


@pytest.fixture(scope="module")
def point_setup():
    context = combined_pi.build_context()
    fit = combined_pi.fit_point_catalogue(context=context)
    return context, fit


def test_small_bootstrap_refits_are_reproducible_and_numerically_valid(point_setup):
    context, observed_fit = point_setup
    first, first_rejections = bootstrap.run_bootstrap(
        observed_fit,
        context,
        n_bootstrap=3,
        seed=13579,
        show_progress=False,
    )
    second, second_rejections = bootstrap.run_bootstrap(
        observed_fit,
        context,
        n_bootstrap=3,
        seed=13579,
        n_workers=2,
        show_progress=False,
    )

    pd.testing.assert_frame_equal(first, second)
    assert first_rejections == second_rejections
    assert first["reduced_converged"].all()
    assert first["full_converged"].all()
    assert first["likelihood_nesting_ok"].all()
    assert not first["eta_clipping_used"].any()
    assert np.isfinite(first[["ll_gain_nats", "LR_statistic"]]).all().all()
    np.testing.assert_allclose(first["LR_statistic"], 2.0 * first["ll_gain_nats"])


def test_script_uses_the_frozen_main_settings(point_setup):
    context, observed_fit = point_setup
    assert context.history_window_ka == pytest.approx(1.5)
    assert context.bin_width_ka == pytest.approx(0.2)
    assert context.origin_fraction == pytest.approx(0.0)
    assert context.response_mode == "maximal_for_history"
    assert observed_fit.summary["n_source_events"] == 55
    assert observed_fit.summary["n_predictive_events"] == 55
    assert combined_pi.REDUCED_TERMS == (
        "same_type_history_count",
        "lr04_scaled",
        "co2_scaled",
        "mis6_segment",
    )
    assert combined_pi.FULL_TERMS[-2:] == (
        "pre_phase_sin",
        "pre_phase_cos",
    )
