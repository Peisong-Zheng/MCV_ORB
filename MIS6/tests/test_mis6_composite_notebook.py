"""Scientific and output checks for the composite notebook."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

MIS6_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK = MIS6_DIR / "MIS6_composite_event_record.ipynb"


@pytest.fixture
def prepared(monkeypatch):
    cells = ["".join(c["source"]) for c in json.loads(NOTEBOOK.read_text())["cells"]
             if c["cell_type"] == "code"]
    monkeypatch.chdir(MIS6_DIR)
    monkeypatch.syspath_prepend(str(MIS6_DIR))
    monkeypatch.setattr(plt, "show", lambda: None)
    namespace = {}
    with plt.rc_context():
        for cell in cells[:5]:
            exec(cell.replace("%matplotlib inline\n", ""), namespace)
        yield namespace, cells
        plt.close("all")


def test_nominal_ages_qc_and_all_nine_settings(prepared):
    namespace, _ = prepared
    events = namespace["events"]
    expected_ages = [
        134.976, 138.393, 146.251, 146.847, 148.439, 149.458, 150.462,
        151.115, 159.522, 160.879, 165.006, 168.566, 169.675, 171.506,
        172.364, 174.752, 176.779, 179.831, 180.309, 190.972, 194.238,
    ]
    np.testing.assert_array_equal(events["event_age_ka_bp"].round(3), expected_ages)
    assert events.loc[events["event_age_qc"].eq("review"), "composite_event_display_label"].tolist() == [
        "6.1", "6.10", "6.14", "6.16",
    ]
    assert events.loc[events["event_age_qc"].eq("moderate"), "composite_event_display_label"].tolist() == ["6.13"]
    picks = namespace["tuning_picks"]
    assert len(picks) == 189
    assert picks.groupby("composite_event_id").size().eq(9).all()
    assert set(picks["smoothing_sigma_yr"]) == {35, 50, 75}
    assert set(picks["search_half_width_yr"]) == {300, 400, 500}
    ranges = picks.groupby("composite_event_id")["event_age_ka_bp"].agg(["min", "max"])
    np.testing.assert_array_equal(ranges["min"], events["local_tuning_age_min_ka_bp"])
    np.testing.assert_array_equal(ranges["max"], events["local_tuning_age_max_ka_bp"])


def test_displayed_control_intervals_are_staggered_without_overlap(prepared):
    namespace, _ = prepared
    controls = namespace["age_controls"]
    local = controls.loc[controls["record_id"].eq("Sofular") & controls["shown_in_composite_figure"].astype(bool)]
    local = local.sort_values("age_ka_bp")
    padding = 0.01 * np.ptp(namespace["RECORD_BY_ID"]["Sofular"].plot_range_ka)
    centers = local["age_ka_bp"].to_numpy(float)
    errors = local["age_error_2sigma_ka"].to_numpy(float)
    lanes = namespace["assign_interval_lanes"](centers, errors, padding)
    assert len(np.unique(lanes)) == 3
    for lane in np.unique(lanes):
        indices = np.flatnonzero(lanes == lane)
        order = indices[np.argsort(centers[indices] - errors[indices])]
        assert np.all((centers[order] - errors[order])[1:] >
                      (centers[order] + errors[order])[:-1] + padding)


def test_catalogue_schema_and_figure_outputs(prepared, tmp_path):
    namespace, cells = prepared
    reference_csv = namespace["output_csv"].read_bytes()
    exec(cells[5], namespace)
    fig = namespace["fig"]
    assert len(fig.axes) == 3
    assert all(ax.get_title(loc="left").startswith(f"({label})")
               for ax, label in zip(fig.axes, "abc"))
    assert fig.axes[2].yaxis_inverted()
    namespace["output_csv"] = tmp_path / "mis6_composite_event_record.csv"
    namespace["figure_dir"] = tmp_path / "figures"
    exec(cells[6], namespace)
    assert namespace["output_csv"].read_bytes() == reference_csv
    assert pd.read_csv(namespace["output_csv"]).shape == (21, 18)
    assert {p.name for p in namespace["figure_dir"].iterdir()} == {
        "MIS6_composite_event_record.png", "MIS6_composite_event_record.pdf",
    }
