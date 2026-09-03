"""Focused contracts for the Rasmussen Table 2 / NGRIP phase analysis."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


NGRIP_DIR = Path(__file__).resolve().parents[1] / "NGRIP"
sys.path.insert(0, str(NGRIP_DIR))

import extract_rasmussen2014_table2 as extraction  # noqa: E402
import ngrip_event_phase_analysis as analysis  # noqa: E402


def test_lettered_subevents_are_collapsed_to_oldest_parent_onset():
    text = "\n".join(
        [
            "Start of GI-1d 1574.8 14,075 a 169 3",
            "Start of GI-1e 1604.64 14,692 ± 4 186 3, 5",
            "Start of GI-2.1 1786.28 23,020 a 583 8, 13",
            "Start of GS-2.1a 1669.09 17,480 c 330 8, 13",
            "Start of GS-2.1c 1783.62 22,900 a 573 7, 8",
            "Start of GS-14 QS 2261.46 49,600 d 2051 8, 13",
        ]
    )

    rows = extraction.parse_table2_gi_gs_rows(text)
    events = extraction.collapse_lettered_subevents(rows).set_index("event_label")

    assert set(events.index) == {"GI-1", "GI-2.1", "GS-2.1", "GS-14"}
    assert events.loc["GI-1", "source_event_label"] == "GI-1e"
    assert events.loc["GS-2.1", "source_event_label"] == "GS-2.1c"
    assert bool(events.loc["GS-14", "is_quasi_stadial"])
    assert events.loc["GI-1", "event_type"] == "warming"
    assert events.loc["GS-2.1", "event_type"] == "cooling"


def test_b2k_to_bp_conversion_is_minus_fifty_years():
    assert np.isclose(extraction.age_yr_b2k_to_ka_bp(14_692), 14.642)
    assert np.isclose(extraction.age_yr_b2k_to_ka_bp(12_896), 12.846)


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
