"""Preparation, grid covariance and unchanged combined NGRIP sampling."""

from pathlib import Path
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from NGRIP import ngrip_event_age_uncertainty as uncertainty

ROOT = Path(__file__).resolve().parents[1]
NGRIP = ROOT / "NGRIP"


@pytest.fixture(scope="module")
def events():
    return pd.read_csv(NGRIP / uncertainty.EVENTS_CSV)


@pytest.fixture(scope="module")
def grid():
    return pd.read_csv(NGRIP / uncertainty.GRID_CSV, float_precision="round_trip")


def test_paper_definition_codes_and_prepared_scales(events):
    raw = pd.read_excel(
        NGRIP / "data/raw/Rasmussen2014_GI_GS_starts_no_subevents_wide.xlsx",
        sheet_name="Long_no_subevents",
    ).sort_values("age_ka_b2k").reset_index(drop=True)
    assert events.event_label.equals(raw.event_label)
    assert events.source_event_label.equals(raw.source_event_label)
    assert events.definition_uncertainty_code.equals(raw.definition_uncertainty_code)
    assert events.definition_uncertainty_code.value_counts().to_dict() == {
        "a": 39, "b": 17, "f": 7, "d": 3, "±4": 2, "c": 1,
    }
    named = events.set_index("event_label")
    assert named.loc["GI-1", "source_event_label"] == "GI-1e"
    assert named.loc["GI-1", "definition_sigma_yr"] == 4
    assert named.loc["GS-19.2", "definition_sigma_yr"] == 200
    assert named.loc["GS-21.1", "definition_sigma_yr"] == 100
    assert named.loc["GI-25", "definition_sigma_yr"] == 30
    counted = events.chronology_source.eq("MCE")
    assert counted.sum() == 44
    np.testing.assert_array_equal(events.loc[counted, "chronology_envelope_yr"],
                                  raw.loc[counted, "maximum_counting_error_yr"])
    np.testing.assert_allclose(events.loc[~counted, "chronology_envelope_yr"],
                               45 * events.loc[~counted, "age_ka_b2k"])


def test_notebook_recreates_both_inputs_in_order(tmp_path, monkeypatch, events, grid):
    # Execute the four plain cells with no existing globals or processed inputs.
    work = tmp_path / "NGRIP"
    (work / "data/raw").mkdir(parents=True)
    workbook = "Rasmussen2014_GI_GS_starts_no_subevents_wide.xlsx"
    annual = "Rasmussen et al-2022-GICC05_time_scale.txt"
    shutil.copy2(NGRIP / "data/raw" / workbook, work / "data/raw" / workbook)
    shutil.copy2(NGRIP / "data/raw" / annual, work / "data/raw" / annual)
    notebook = json.loads((NGRIP / "ngrip_data_preparation.ipynb").read_text())
    monkeypatch.chdir(work)
    namespace = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            exec("".join(cell["source"]), namespace)
    pd.testing.assert_frame_equal(pd.read_csv(uncertainty.EVENTS_CSV), events)
    pd.testing.assert_frame_equal(pd.read_csv(uncertainty.GRID_CSV, float_precision="round_trip"), grid)


def test_grid_preserves_counted_endpoint_and_working_extension(events, grid):
    basis, knots, knot_basis = uncertainty.chronology_process_basis(events, grid)
    assert len(knots) == 25
    assert 60 not in knots and 60.202 in knots
    selected = grid.loc[grid.knot_spacing_ka.eq(5)]
    assert selected.loc[selected.knot_age_ka_b2k.eq(60.202), "chronology_envelope_ka"].iloc[0] == 2.611
    old = selected.knot_age_ka_b2k.gt(60.202)
    np.testing.assert_allclose(selected.loc[old, "chronology_envelope_ka"],
                               0.045 * selected.loc[old, "knot_age_ka_b2k"])
    variance = (selected.chronology_envelope_ka.to_numpy() / 2)**2
    np.testing.assert_allclose(knot_basis @ knot_basis.T, np.minimum.outer(variance, variance), atol=1e-12)
    assert np.linalg.matrix_rank(basis) > 15
    # Continuous mapping at the actual counted/modelled join.
    near = pd.DataFrame({"age_ka_b2k": [60.202-1e-7, 60.202, 60.202+1e-7]})
    b, _, _ = uncertainty.chronology_process_basis(near, grid)
    np.testing.assert_allclose(b[0], b[1], atol=1e-7)
    np.testing.assert_allclose(b[2], b[1], atol=1e-7)


def test_prepared_alternative_grids_remain_available(events, grid):
    fine, _, _ = uncertainty.chronology_process_basis(events, grid, 2.5)
    coarse, _, _ = uncertainty.chronology_process_basis(events, grid, 10)
    assert np.linalg.matrix_rank(fine) > np.linalg.matrix_rank(coarse)
    with pytest.raises(ValueError, match="Prepare"):
        uncertainty.chronology_process_basis(events, grid, 0)
    wrong = grid.copy()
    wrong.loc[wrong.knot_age_ka_b2k.eq(65), "chronology_envelope_ka"] = 0
    with pytest.raises(ValueError, match="nondecreasing"):
        uncertainty.chronology_process_basis(events, wrong)


def test_combined_sampler_reproduces_all_published_realizations(events, grid):
    draws, stats = uncertainty.sample_age_realizations(events, grid)
    saved = pd.read_csv(NGRIP / uncertainty.OUT_DATA_DIR / "ngrip_event_age_realizations.csv")
    np.testing.assert_allclose(np.round(draws, 6), saved.iloc[:, 1:].to_numpy(), atol=1e-10, rtol=0)
    assert draws.shape == (10000, 69)
    assert np.all(np.diff(draws, axis=1) > 0)
    assert stats["random_seed"] == 20260907
    assert stats["n_proposals"] == 11000
    assert stats["n_rejected_for_event_crossing"] == 427
    assert stats["n_rejected_nonmonotonic_chronology"] == 0
    assert stats["n_unused_accepted_tail"] == 573
    assert stats["acceptance_fraction"] == pytest.approx(10573 / 11000)


def test_sampler_rejects_maps_and_picks_separately_without_sorting(events, grid, monkeypatch):
    class FixedRng:
        def normal(self, *args, size):
            values = np.zeros(size)
            if not args:
                values[0, 1] = -1000  # First proposal: reversed chronology grid.
            else:
                values[1, 1] = -100  # Second proposal: crossed transition pick.
            return values
    monkeypatch.setattr(uncertainty.np.random, "default_rng", lambda seed: FixedRng())
    draws, stats = uncertainty.sample_age_realizations(events, grid, n_realizations=3)
    np.testing.assert_array_equal(draws, np.tile(events.age_ka_bp, (3, 1)))
    assert stats["n_rejected_nonmonotonic_chronology"] == 1
    assert stats["n_rejected_for_event_crossing"] == 1
    assert stats["n_unused_accepted_tail"] == 995


def test_small_draws_have_stable_ids_and_independent_rows(events, grid):
    draws, stats = uncertainty.sample_age_realizations(events, grid, n_realizations=31, seed=541)
    table = uncertainty.realization_table(events, draws)
    assert table.realization_id.iloc[0] == "NGRIP_00001"
    assert table.realization_id.iloc[-1] == "NGRIP_00031"
    assert table.columns.tolist() == ["realization_id", *[f"age_ka_bp__{s}" for s in events.event_label]]
    assert len(np.unique(draws, axis=0)) == 31
    accepted = stats["n_realizations"] + stats["n_unused_accepted_tail"]
    assert accepted + stats["n_rejected_for_event_crossing"] + stats["n_rejected_nonmonotonic_chronology"] == stats["n_proposals"]
    with pytest.raises(ValueError, match="positive"):
        uncertainty.sample_age_realizations(events, grid, n_realizations=0)


def test_grid_screening_caller_uses_prepared_inputs(events, grid):
    from docs.reviews import ngrip_knot_spacing_sensitivity as screening
    assert screening.NGRIP_EVENTS == NGRIP / uncertainty.EVENTS_CSV
    assert screening.NGRIP_GRID == NGRIP / uncertainty.GRID_CSV
    summary, detail, gaps, knots = screening.analytic_basis_diagnostics(events, grid, 5.0)
    assert len(summary) == 2 and len(detail) == 69 and len(knots) == 25
    assert np.isfinite(gaps.unconditioned_gap_sigma_ka).all()


def test_old_active_input_paths_are_gone():
    assert not (ROOT / "data/curated/ngrip_warming_cooling_starts.csv").exists()
    assert not (ROOT / "data/raw/Rasmussen2014_GI_GS_starts_no_subevents_wide.xlsx").exists()
