"""Focused contracts for the retained NGRIP catalogue and phase analysis."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


NGRIP_DIR = Path(__file__).resolve().parents[1] / "NGRIP"
sys.path.insert(0, str(NGRIP_DIR))

import ngrip_event_phase_analysis as analysis  # noqa: E402


@pytest.fixture(scope="module")
def phase_results():
    events = analysis.load_events()
    event_phases, rayleigh, _ = analysis.run_rayleigh(events)
    pi = analysis.run_predictive_information(events)
    summary = analysis.build_analysis_summary(
        events,
        rayleigh,
        pi["model_summary"],
        pi["likelihood_tests"],
        pi["fit_frame"],
    )
    return events, event_phases, rayleigh, pi, summary


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


def test_each_physical_boundary_is_sampled_once(phase_results):
    events, event_phases, _, _, _ = phase_results

    assert len(event_phases) == len(events) == 69
    assert event_phases["event_index"].is_unique
    assert not event_phases["phase_extrapolated"].any()
    assert len(analysis._phase_catalogues(event_phases)) == 138


def test_rayleigh_and_predictive_information_regression(phase_results):
    _, _, _, _, summary = phase_results
    summary = summary.set_index("event_type")

    columns = [
        "n_rayleigh_events",
        "rayleigh_mean_phase_deg",
        "rayleigh_p",
        "n_predictive_events",
        "predictive_preferred_phase_deg",
        "predictive_LR_p",
        "predictive_bits_per_event",
    ]
    observed = summary.loc[list(analysis.EVENT_TYPES), columns].to_numpy(float)
    expected = np.array(
        [
            [34, 356.4976127, 0.223885818, 33, 327.6608472, 0.043910670, 0.136644989],
            [35, 354.2041151, 0.406554046, 34, 332.0253519, 0.063046975, 0.117277324],
            [69, 355.4865273, 0.093486146, 67, 341.1998264, 0.029501479, 0.075866700],
        ]
    )
    np.testing.assert_allclose(observed, expected, rtol=3e-6, atol=3e-6)
    assert summary["all_models_converged"].all()
    assert summary["likelihood_nesting_ok"].all()
    assert not summary["eta_clipping_used"].any()
