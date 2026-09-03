"""Regression tests for the Barker et al. (2011) event-phase analysis."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import Barker2011_do_predictive_information_audited as barker


@pytest.fixture(scope="module")
def core_analysis():
    """Run the three primary variable-threshold catalogues once per test file."""

    return barker.run_core_analysis()


def test_catalogue_counts_ranges_and_shared_source_rows():
    source = barker.load_barker_source()
    events, _ = barker.build_event_catalogues(source.table)

    expected = {
        "barker_variable_threshold_edc3_0_640": (104, 11.5, 637.98),
        "barker_variable_threshold_edc3_0_800": (126, 11.5, 799.0),
        "barker_variable_threshold_speleo_0_400": (
            70,
            11.402407613184174,
            396.4642638777152,
        ),
    }
    for dataset_id, (count, youngest, oldest) in expected.items():
        subset = events[events["dataset_id"].eq(dataset_id)]
        assert len(subset) == count
        np.testing.assert_allclose(
            [subset["event_age_ka"].min(), subset["event_age_ka"].max()],
            [youngest, oldest],
        )
        assert subset["event_age_ka"].is_monotonic_increasing

    rows_640 = set(
        events.loc[
            events["dataset_id"].eq("barker_variable_threshold_edc3_0_640"),
            "source_row",
        ]
    )
    rows_800 = set(
        events.loc[
            events["dataset_id"].eq("barker_variable_threshold_edc3_0_800"),
            "source_row",
        ]
    )
    rows_speleo = set(
        events.loc[
            events["dataset_id"].eq("barker_variable_threshold_speleo_0_400"),
            "source_row",
        ]
    )
    assert rows_640 < rows_800
    assert rows_speleo < rows_640


def test_phase_response_curve_has_expected_rate_ratio():
    phase_deg, relative_rate = barker.phase_response_curve(0.3, -0.4)
    assert len(phase_deg) == len(relative_rate)
    assert phase_deg[0] == 0.0
    assert phase_deg[-1] == 360.0
    np.testing.assert_allclose(
        relative_rate.max() / relative_rate.min(),
        np.exp(2.0 * np.hypot(0.3, -0.4)),
        rtol=2e-4,
    )


def test_primary_rayleigh_regression(core_analysis):
    rows = core_analysis.rayleigh_results.query("driver == 'pre'").set_index(
        "event_type"
    )
    values = rows[
        ["n_phase_events_used", "mean_phase_deg", "mean_resultant_length", "rayleigh_p"]
    ].to_numpy(dtype=float)
    expected = np.array(
        [
            [104, 343.545148, 0.135310331, 0.149019686],
            [126, 350.392603, 0.110957239, 0.212277858],
            [70, 0.033293, 0.190027146, 0.079455941],
        ]
    )
    np.testing.assert_allclose(values, expected, rtol=1e-6, atol=1e-6)


def test_conditional_predictive_information_regression(core_analysis):
    lrt = core_analysis.likelihood_tests.query(
        "comparison_id == 'phase_after_climate'"
    ).set_index("dataset_id")
    summary = core_analysis.model_summary.query("model_id == 'full'").set_index(
        "dataset_id"
    )
    order = [spec.dataset_id for spec in barker.CATALOGUE_SPECS]

    observed = np.column_stack(
        [
            lrt.loc[order, "n_events"],
            lrt.loc[order, "LR_statistic"],
            lrt.loc[order, "LR_p_value"],
            lrt.loc[order, "info_bits_per_event"],
            summary.loc[order, "pre_phase_preferred_deg"],
            summary.loc[order, "pre_phase_rate_ratio_max_vs_min"],
        ]
    ).astype(float)
    expected = np.array(
        [
            [103, 8.017718824, 0.018154090, 0.056151083, 328.595821, 2.291814891],
            [125, 6.850847206, 0.032535496, 0.039534733, 330.856088, 1.992454135],
            [69, 7.790004540, 0.020343328, 0.081439137, 333.390700, 2.861778914],
        ]
    )
    np.testing.assert_allclose(observed, expected, rtol=2e-6, atol=2e-6)
    assert summary.loc[order, "converged"].all()


def test_pi_support_is_explicitly_shorter_than_rayleigh_support(core_analysis):
    summary = core_analysis.analysis_summary.set_index("dataset_id")
    assert summary["n_rayleigh_events"].tolist() == [104, 126, 70]
    assert summary["n_pi_events"].tolist() == [103, 125, 69]
    np.testing.assert_allclose(summary["pi_support_end_ka"], [635.0, 795.0, 395.0])
