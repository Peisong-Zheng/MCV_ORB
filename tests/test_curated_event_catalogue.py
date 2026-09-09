"""Contracts for the fixed event catalogue used by the pooled analysis."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURATED_DIR = PROJECT_ROOT / "data" / "curated"

NGRIP_SOURCE = PROJECT_ROOT / "NGRIP/data/processed/ngrip_warming_cooling_starts.csv"
MIS6_SOURCE = (
    PROJECT_ROOT
    / "MIS6"
    / "data"
    / "processed"
    / "MIS6_composite_event_record"
    / "mis6_composite_event_record.csv"
)

EVENT_COLUMNS = [
    "event_id",
    "event_label",
    "event_age_kyr_bp",
    "segment_id",
    "source_record",
    "source_event_label",
    "data_source",
    "label_source",
    "timing_method",
    "source_table",
]


@pytest.fixture(scope="module")
def events():
    return pd.read_csv(CURATED_DIR / "ngrip_mis6_warming_events.csv")


def test_curated_source_tables_have_stable_contracts():
    required_columns = {
        "ngrip_warming_cooling_starts.csv": {
            "event_label",
            "event_type",
            "age_ka_bp",
            "definition_uncertainty_code",
        },
        "speleothem_mis6_published_event_label_anchors.csv": {
            "record_id",
            "event_label",
            "anchor_age_ka_bp",
            "label_scheme",
        },
        "Sofular_MF_visual_peak_match.csv": {"Sofular", "MF"},
        "mis6_age_control_points.csv": {
            "control_id",
            "record_id",
            "age_ka_bp",
            "age_error_2sigma_ka",
            "source_filename",
        },
    }

    for filename, columns in required_columns.items():
        table = pd.read_csv(NGRIP_SOURCE if filename == NGRIP_SOURCE.name else PROJECT_ROOT / "MIS6/data/curated" / filename)
        assert columns.issubset(table.columns)
        assert not table.empty


def test_catalogue_contains_the_55_selected_warming_transitions(events):
    assert events.columns.tolist() == EVENT_COLUMNS
    assert len(events) == 55
    assert events.groupby("segment_id", sort=False).size().to_dict() == {
        "NGRIP": 34,
        "MIS6": 21,
    }
    mis6_record_counts = events.loc[
        events["segment_id"].eq("MIS6"), "source_record"
    ].value_counts()
    assert mis6_record_counts.to_dict() == {"MF": 16, "Sofular": 5}


def test_catalogue_ids_ages_and_provenance_are_valid(events):
    assert events["event_id"].is_unique
    assert np.isfinite(events["event_age_kyr_bp"]).all()

    for _, segment in events.groupby("segment_id", sort=False):
        assert segment["event_age_kyr_bp"].is_monotonic_increasing

    provenance_columns = [
        "source_record",
        "source_event_label",
        "data_source",
        "label_source",
        "timing_method",
        "source_table",
    ]
    assert events[provenance_columns].notna().all().all()
    assert events[provenance_columns].apply(
        lambda column: column.astype(str).str.strip().ne("").all()
    ).all()


def test_ngrip_rows_match_the_published_start_table(events):
    source = pd.read_csv(NGRIP_SOURCE)
    source = source.loc[source["event_type"].eq("warming")].reset_index(drop=True)
    curated = events.loc[events["segment_id"].eq("NGRIP")].reset_index(drop=True)

    assert curated["event_id"].tolist() == ("NGRIP:" + source["event_label"]).tolist()
    assert curated["event_label"].tolist() == source["event_label"].tolist()
    assert curated["source_event_label"].tolist() == source["source_event_label"].tolist()
    np.testing.assert_allclose(curated["event_age_kyr_bp"], source["age_ka_bp"])


def test_mis6_rows_match_the_composite_event_table(events):
    source = pd.read_csv(MIS6_SOURCE)
    curated = events.loc[events["segment_id"].eq("MIS6")].reset_index(drop=True)

    assert curated["event_id"].tolist() == (
        "MIS6:" + source["composite_event_id"]
    ).tolist()
    assert curated["event_label"].tolist() == source["composite_event_label"].tolist()
    assert curated["source_record"].tolist() == source["source_record"].tolist()
    assert curated["source_event_label"].tolist() == source["source_event_label"].tolist()
    assert curated["data_source"].tolist() == source["data_source"].tolist()
    assert curated["label_source"].tolist() == source["label_source"].tolist()
    np.testing.assert_allclose(curated["event_age_kyr_bp"], source["event_age_ka_bp"])


def test_observation_segments_keep_the_record_gap_out_of_exposure():
    segments = pd.read_csv(CURATED_DIR / "observation_segments.csv")
    assert segments.columns[:6].tolist() == [
        "segment_id",
        "observation_start_kyr_bp",
        "observation_end_kyr_bp",
        "common_response_start_kyr_bp",
        "common_response_end_kyr_bp",
        "support_basis",
    ]
    segments = segments.set_index("segment_id")

    assert segments.index.tolist() == ["NGRIP", "MIS6"]
    columns = [
        "observation_start_kyr_bp",
        "observation_end_kyr_bp",
        "common_response_start_kyr_bp",
        "common_response_end_kyr_bp",
    ]
    np.testing.assert_allclose(
        segments.loc["NGRIP", columns].to_numpy(float),
        [12.0, 123.0, 12.0, 118.0],
    )
    np.testing.assert_allclose(
        segments.loc["MIS6", columns].to_numpy(float),
        [132.5, 204.5, 132.5, 199.5],
    )

    assert (
        segments["observation_start_kyr_bp"]
        <= segments["common_response_start_kyr_bp"]
    ).all()
    assert (
        segments["common_response_end_kyr_bp"]
        <= segments["observation_end_kyr_bp"]
    ).all()
    assert segments.loc["NGRIP", "common_response_end_kyr_bp"] < segments.loc[
        "MIS6", "common_response_start_kyr_bp"
    ]
    assert segments["support_basis"].str.contains(
        "gap is excluded", regex=False
    ).all()
    assert (
        segments["common_response_end_kyr_bp"]
        - segments["common_response_start_kyr_bp"]
    ).sum() == pytest.approx(173.0)
