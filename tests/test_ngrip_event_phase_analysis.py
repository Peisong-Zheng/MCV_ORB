"""Focused contracts for the retained NGRIP catalogue and phase analysis."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


NGRIP_DIR = Path(__file__).resolve().parents[1] / "NGRIP"
sys.path.insert(0, str(NGRIP_DIR))

import ngrip_event_phase_analysis as analysis  # noqa: E402


def test_processed_catalogue_preserves_collapsed_parent_provenance():
    events = analysis.load_events().set_index("event_label")
    collapsed = events.loc[events["is_lettered_subevent"].astype(bool)]

    assert len(collapsed) == 14
    assert collapsed["source_event_label"].ne(collapsed.index.to_series()).all()
    assert (
        collapsed["selection_note"]
        .eq("lettered subevents collapsed; oldest onset used")
        .all()
    )
    assert events.loc["GI-1", "source_event_label"] == "GI-1e"
    assert events.loc["GS-2.1", "source_event_label"] == "GS-2.1c"
    assert bool(events.loc["GS-14", "is_quasi_stadial"])
    assert events.loc["GI-1", "event_type"] == "warming"
    assert events.loc["GS-2.1", "event_type"] == "cooling"


def test_processed_catalogue_age_scales_are_internally_consistent():
    events = analysis.load_events()

    np.testing.assert_allclose(events["age_ka_b2k"], events["age_yr_b2k"] / 1000)
    np.testing.assert_allclose(events["age_ka_bp"], events["age_ka_b2k"] - 0.05)


def test_processed_catalogue_counts_and_anchor_events():
    events = analysis.load_events().set_index("event_label")

    assert events.groupby("event_type").size().to_dict() == {
        "cooling": 35,
        "warming": 34,
    }
    assert events.loc["GI-1", "source_event_label"] == "GI-1e"
    assert events.loc["GS-2.1", "source_event_label"] == "GS-2.1c"
    assert np.isclose(events.loc["GI-1", "age_ka_bp"], 14.642)
    assert np.isclose(events.loc["GS-26", "age_ka_bp"], 119.09)


def test_event_datasets_include_separate_and_combined_catalogues():
    events = pd.DataFrame(
        {
            "event_type": ["warming", "cooling", "warming"],
            "age_ka_bp": [20.0, 30.0, 40.0],
        }
    )

    datasets = analysis.build_event_datasets(events)

    assert [dataset.dataset_id for dataset in datasets] == list(analysis.EVENT_TYPES)
    np.testing.assert_array_equal(datasets[0].ages_ka, [20.0, 40.0])
    np.testing.assert_array_equal(datasets[1].ages_ka, [30.0])
    np.testing.assert_array_equal(datasets[2].ages_ka, [20.0, 30.0, 40.0])


def test_phase_multiplier_has_expected_peak_and_rate_ratio():
    phase_deg = np.arange(360.0)
    multiplier = analysis.phase_rate_multiplier(
        phase_deg,
        beta_sin=1.0,
        beta_cos=0.0,
    )

    assert int(phase_deg[np.argmax(multiplier)]) == 90
    assert np.isclose(multiplier[90] / multiplier[270], np.exp(2.0))
