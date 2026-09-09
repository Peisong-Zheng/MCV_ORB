"""Checks for the sequential speleothem plotting notebook."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pytest

MIS6_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK = MIS6_DIR / "Speleothem_published_event_plot.ipynb"


@pytest.fixture
def prepared(monkeypatch):
    cells = ["".join(c["source"]) for c in json.loads(NOTEBOOK.read_text())["cells"]
             if c["cell_type"] == "code"]
    monkeypatch.chdir(MIS6_DIR)
    monkeypatch.setattr(plt, "show", lambda: None)
    namespace = {}
    with plt.rc_context():
        for cell in cells[:3]:
            exec(cell.replace("%matplotlib inline\n", ""), namespace)
        yield namespace, cells
        plt.close("all")


def test_smoothing_parameters_match_record_resolution(prepared):
    namespace, _ = prepared
    assert {name: spec["sigma"] * 1000 for name, spec in namespace["specs"].items()} == {
        "Sanbao": 100, "Huagapo": 100, "Sofular": 50, "MF": 50,
    }


def test_smoothing_does_not_cross_material_gaps(prepared):
    namespace, _ = prepared
    ages = np.array([130.0, 130.1, 131.0, 131.1])
    segments = namespace["smooth_record_segments"](ages, np.array([0.0, 1.0, 3.0, 4.0]), 0.05)
    assert len(segments) == 2
    assert np.isclose(segments[0][0][-1], 130.1)
    assert np.isclose(segments[1][0][0], 131.0)
    assert all(np.isfinite(curve).all() for _, curve in segments)


def test_both_figures_and_original_output_names(prepared, tmp_path):
    namespace, cells = prepared
    for cell in cells[3:5]:
        exec(cell, namespace)
    overview = namespace["fig_overview"]
    comparison = namespace["fig_comparison"]
    assert len(overview.axes) == 8 and len(comparison.axes) == 2
    assert [axis.texts[0].get_text() for axis in comparison.axes] == ["(a)", "(b)"]
    assert "Sofular" in comparison.axes[0].texts[1].get_text()
    assert "MF" in comparison.axes[1].texts[1].get_text()
    assert len(comparison.artists) == 6
    assert not any(line.get_visible() for line in comparison.axes[0].get_xgridlines())
    assert all(not axis.collections for axis in comparison.axes)
    namespace["figure_dir"] = tmp_path / "figures/Speleothem_published_event_plot"
    exec(cells[5], namespace)
    assert {p.name for p in namespace["figure_dir"].iterdir()} == {
        "Speleothem_published_event_plot.png", "Speleothem_published_event_plot.pdf",
        "Sofular_MF_comparison.png", "Sofular_MF_comparison.pdf",
    }


def test_visual_match_table_has_six_unique_pairs(prepared):
    namespace, _ = prepared
    matches = namespace["matches"]
    assert len(matches) == 6 and not matches.duplicated().any()
    assert matches["Sofular"].is_monotonic_increasing
