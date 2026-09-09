"""Scientific invariants for the pooled NGRIP--MIS 6 PI core."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from toolbox import combined_pi


@pytest.fixture(scope="module")
def point_events() -> pd.DataFrame:
    return combined_pi.load_event_catalogue()


@pytest.fixture(scope="module")
def legacy_context() -> combined_pi.PIContext:
    return combined_pi.build_context(
        history_window_ka=5.0,
        bin_width_ka=0.2,
        origin_fraction=0.0,
        response_mode="common_core",
    )


def test_curated_catalogue_is_the_55_event_warming_catalogue(point_events):
    assert len(point_events) == 55
    assert point_events["event_id"].is_unique
    assert point_events.groupby("segment_id").size().to_dict() == {
        "MIS6": 21,
        "NGRIP": 34,
    }
    assert (
        point_events.loc[point_events["segment_id"].eq("NGRIP"), "event_label"]
        .str.startswith("GI-")
        .all()
    )
    assert combined_pi.REDUCED_TERMS == (
        "same_type_history_count",
        "lr04_scaled",
        "co2_scaled",
        "mis6_segment",
    )
    assert combined_pi.FULL_TERMS[-2:] == ("pre_phase_sin", "pre_phase_cos")
    assert combined_pi.RESOLUTION_COVARIATE_INCLUDED is False


def test_common_core_is_disjoint_and_has_173_kyr_exposure(legacy_context):
    response = legacy_context.response_bins
    assert response.groupby("segment_id").size().to_dict() == {
        "MIS6": 335,
        "NGRIP": 530,
    }
    assert legacy_context.response_exposure_kyr == pytest.approx(173.0)
    assert not response["bin_center_kyr_bp"].between(118.0, 132.5).any()

    support = response.groupby("segment_id").agg(
        start=("bin_start_kyr_bp", "min"), end=("bin_end_kyr_bp", "max")
    )
    np.testing.assert_allclose(support.loc["NGRIP"], [12.0, 118.0])
    np.testing.assert_allclose(support.loc["MIS6"], [132.5, 199.5])

    for predictor in ("lr04_scaled", "co2_scaled"):
        values = response[predictor].to_numpy(float)
        assert values.mean() == pytest.approx(0.0, abs=2e-14)
        assert np.ptp(values) == pytest.approx(1.0, abs=2e-14)


def test_history_resets_between_segments_and_excludes_current_bin(legacy_context):
    synthetic = pd.DataFrame(
        {
            "segment_id": ["NGRIP", "MIS6"],
            # Events near the two observation edges remain on separate grids.
            "age": [122.0, 132.6],
        }
    )
    binned = combined_pi.bin_catalogue(synthetic, legacy_context, age_column="age")

    ngrip_event_bin = binned.loc[
        binned["segment_id"].eq("NGRIP") & binned["event_count"].eq(1)
    ].iloc[0]
    mis6_event_bin = binned.loc[
        binned["segment_id"].eq("MIS6") & binned["event_count"].eq(1)
    ].iloc[0]

    assert ngrip_event_bin["same_type_history_count"] == 0
    assert mis6_event_bin["same_type_history_count"] == 0

    # An event never contributes to the history predictor of its own bin.
    one_event = pd.DataFrame({"segment_id": ["NGRIP"], "age": [20.05]})
    one_binned = combined_pi.bin_catalogue(one_event, legacy_context, age_column="age")
    event_bin = one_binned.loc[one_binned["event_count"].eq(1)].iloc[0]
    assert event_bin["same_type_history_count"] == 0


def test_history_direction_and_window_edges_follow_the_age_axis(legacy_context):
    def event_histories(*ages):
        events = pd.DataFrame({"segment_id": ["NGRIP"] * len(ages), "age": ages})
        binned = combined_pi.bin_catalogue(events, legacy_context, age_column="age")
        return binned.loc[
            binned["segment_id"].eq("NGRIP") & binned["event_count"].gt(0),
            ["bin_center_kyr_bp", "same_type_history_count"],
        ].reset_index(drop=True)

    direction = event_histories(20.05, 21.05)
    assert direction["same_type_history_count"].tolist() == [1.0, 0.0]

    # A bin centered exactly five kyr older is included; one beyond it is not.
    on_boundary = event_histories(20.05, 25.05)
    outside = event_histories(20.05, 25.25)
    assert on_boundary.loc[0, "same_type_history_count"] == 1
    assert outside.loc[0, "same_type_history_count"] == 0


def test_shifted_origin_keeps_partial_bin_durations_and_exact_exposure():
    shifted = combined_pi.build_context(
        history_window_ka=5.0,
        bin_width_ka=0.2,
        origin_fraction=0.5,
        response_mode="common_core",
    )
    dt = shifted.response_bins["dt_kyr"].to_numpy(float)
    assert np.all(dt > 0.0)
    assert np.any(dt < 0.2 - 1e-10)
    assert shifted.response_exposure_kyr == pytest.approx(173.0)
    assert shifted.response_bins.groupby("segment_id")[
        "dt_kyr"
    ].sum().to_dict() == pytest.approx({"NGRIP": 106.0, "MIS6": 67.0})


def test_default_maximal_support_is_derived_from_history_window():
    context = combined_pi.build_context()
    windows = {
        segment_id: (
            segment.response_start_kyr_bp,
            segment.response_end_kyr_bp,
        )
        for segment_id, segment in context.segments.items()
    }
    assert windows == pytest.approx({"NGRIP": (12.0, 121.5), "MIS6": (132.5, 203.0)})
    assert context.response_exposure_kyr == pytest.approx(180.0)
    assert context.response_bins["same_type_history_complete"].all()


def test_five_kyr_common_core_reproduces_frozen_point_result(
    point_events, legacy_context
):
    fit = combined_pi.fit_catalogue(point_events, legacy_context)
    observed = np.array(
        [
            fit.summary["n_predictive_events"],
            fit.summary["n_predictive_bins"],
            fit.summary["LR_statistic"],
            fit.summary["nominal_LR_p"],
            fit.summary["info_bits_per_event"],
            fit.summary["delta_AICc_full_minus_reduced"],
            fit.summary["pre_phase_preferred_deg"],
            fit.summary["pre_phase_rate_ratio_max_vs_min"],
        ]
    )
    expected = np.array(
        [
            55,
            865,
            11.5389825643,
            0.003121344994,
            0.1513384811,
            -7.478142777,
            324.9116934,
            4.673603068,
        ]
    )
    np.testing.assert_allclose(observed, expected, rtol=1e-5, atol=3e-6)
    assert fit.summary["all_models_converged"]
    assert fit.summary["likelihood_nesting_ok"]
    assert not fit.summary["eta_clipping_used"]


def test_event_phase_sampling_uses_bp1950_phase_convention(point_events):
    phases = combined_pi.sample_event_phases(point_events)
    assert len(phases) == 55
    assert phases["pre_phase_deg"].between(0.0, 360.0).all()
    assert not phases["pre_phase_extrapolated"].any()
    np.testing.assert_allclose(
        np.sin(phases["pre_phase_rad"]),
        np.sin(np.deg2rad(phases["pre_phase_deg"])),
        atol=1e-12,
    )
