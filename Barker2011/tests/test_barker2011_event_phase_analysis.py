"""Checks for the restored, parameter-aligned SpeleoAge main analysis."""

from pathlib import Path
import sys
import hashlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from Barker2011 import Barker2011_event_phase_analysis as barker
import NGRIP_MIS6_event_phase_analysis as pooled
from toolbox import combined_pi


@pytest.fixture(scope="module")
def result():
    return barker.run_analysis()


def test_source_events_and_descriptive_phase_are_preserved(result):
    assert hashlib.sha256(barker.BARKER_XLS.read_bytes()).hexdigest() == (
        "5b028eb1d9edc5250abbe403efeb0a8fd80171ed0eab42a8bd46c338619258cd"
    )
    events = result["events"]
    assert len(events) == 70 and events["source_row"].is_unique
    assert events["dataset_id"].unique().tolist() == [barker.DATASET_ID]
    assert events["age_column"].unique().tolist() == ["SpeloAge (kyr).1"]
    assert events["pick_column"].unique().tolist() == ["DO pick variable threshold"]
    np.testing.assert_allclose(events["event_age_ka"].iloc[[0, -1]],
                               [11.402407613184174, 396.4642638777152], rtol=0, atol=1e-12)
    ray = result["rayleigh"]
    np.testing.assert_allclose([ray["mean_phase_deg"], ray["mean_resultant_length"], ray["rayleigh_p"]],
                               [0.8300241493611098, 0.19001179622527256, 0.07948875469940513], rtol=1e-10)


def test_shared_parameters_history_and_exact_response_boundary(result):
    assert barker.HISTORY_WINDOW_KA == pooled.HISTORY_WINDOW_KYR == 1.5
    assert barker.BIN_WIDTH_KA == pooled.BIN_WIDTH_KYR == 0.2
    assert barker.BIN_ORIGIN_FRACTION == pooled.BIN_ORIGIN_FRACTION == 0.0
    assert barker.RESPONSE_MODE == pooled.RESPONSE_MODE == "maximal_for_history"
    assert barker.REDUCED_TERMS == tuple(t for t in combined_pi.REDUCED_TERMS if t != combined_pi.SEGMENT_TERM)
    assert barker.FULL_TERMS == tuple(t for t in combined_pi.FULL_TERMS if t != combined_pi.SEGMENT_TERM)
    assert not result["summary"].iloc[0].resolution_covariate_included

    bins, frame = result["bins"], result["model_frame"]
    assert len(bins) == 2001 and len(frame) == 1993
    assert frame["bin_end_ka"].max() == 398.5
    assert frame["dt_ka"].iloc[-1] == pytest.approx(0.1)
    assert frame["dt_ka"].sum() == pytest.approx(398.5)
    assert frame["event_count"].sum() == 70
    assert result["events"]["included_in_pi_response"].all()
    centers, counts = bins["bin_center_ka"].to_numpy(), bins["event_count"].to_numpy()
    # Directly count the older bins for comparison with the cumulative-sum implementation.
    expected = [counts[(centers > age) & (centers <= age + 1.5)].sum() for age in centers]
    np.testing.assert_array_equal(bins["same_type_history_count"], expected)
    assert frame["same_type_history_complete"].all()
    for forcing in ("lr04", "co2"):
        assert frame[f"{forcing}_scaled"].mean() == pytest.approx(0, abs=1e-14)
        assert np.ptp(frame[f"{forcing}_scaled"]) == pytest.approx(1)


def test_aligned_fit_and_phase_response(result):
    summary = result["summary"].iloc[0]
    assert summary.all_models_converged and summary.likelihood_nesting_ok
    assert not summary.eta_clipping_used
    assert [m.model_id for m in result["models"]] == ["reduced", "full"]
    assert result["likelihood_tests"]["df"].tolist() == [2]
    assert summary.info_bits_per_event == pytest.approx(summary.LR_statistic / (2 * 70 * np.log(2)))
    np.testing.assert_allclose(
        [summary.LR_statistic, summary.nominal_LR_p, summary.info_bits_per_event,
         summary.pre_phase_preferred_deg, summary.pre_phase_rate_ratio_max_vs_min],
        [8.67159283062, 0.0130914436939, 0.0893604569525, 336.907088699, 2.88475328064], rtol=2e-6, atol=2e-6,
    )
    _, multiplier = barker.phase_response_curve(0.3, -0.4)
    assert multiplier.max() / multiplier.min() == pytest.approx(np.exp(1), rel=1e-4)


def test_main_outputs_and_three_panel_figure(result, tmp_path):
    with plt.rc_context():
        fig = barker.plot_results(result)
        assert len(fig.axes) == 3 and fig.axes[1].name == "polar"
        assert [a.texts[-1].get_text() for a in fig.axes] == ["a", "b", "c"]
        np.testing.assert_allclose(fig.get_size_inches() * 25.4, [180, 134])
        barker.write_outputs(result, tmp_path / "data")
        png, pdf = barker.save_figure(fig, tmp_path / "figures")
    assert png.exists() and pdf.exists()
    files = {p.name for p in (tmp_path / "data").iterdir()}
    assert len(files) == 9 and not any("sensitivity" in f for f in files)
    assert len(pd.read_csv(tmp_path / "data/analysis_summary.csv")) == 1
    parameters = pd.read_csv(tmp_path / "data/parameters_and_provenance.csv")
    assert not parameters["value"].astype(str).str.contains("archive/").any()
    assert not any("resolution" in term for term in result["coefficients"]["term"])


def test_fixed_definition_uses_published_picks_and_common_exposure(result):
    fixed = barker.run_analysis("fixed_threshold")
    source = pd.read_excel(barker.BARKER_XLS, sheet_name="Sheet1", header=8)
    ages = pd.to_numeric(source[barker.AGE_COLUMN], errors="coerce")
    selected = pd.to_numeric(source["DO pick"], errors="coerce").eq(1) & ages.between(0, 400)
    assert len(fixed["events"]) == 59
    assert set(fixed["events"].source_row) == set(source.index[selected])
    assert fixed["events"].pick_column.unique().tolist() == ["DO pick"]
    assert fixed["summary"].iloc[0].event_definition == "fixed_threshold"
    assert fixed["likelihood_tests"].dataset_id.iloc[0] == "barker_fixed_threshold_speleo_0_400"
    predictors = ["bin_start_ka", "bin_end_ka", "dt_ka", "lr04_scaled", "co2_scaled",
                  "pre_phase_sin", "pre_phase_cos", "in_response_interval"]
    pd.testing.assert_frame_equal(fixed["bins"][predictors], result["bins"][predictors])
    pd.testing.assert_frame_equal(fixed["scaling"], result["scaling"])
    assert fixed["model_frame"].event_count.sum() == 59
    assert fixed["model_frame"].dt_ka.sum() == pytest.approx(398.5)

    # The alternative must rebuild history, not inherit the primary counts.
    centers = fixed["bins"].bin_center_ka.to_numpy()
    counts = fixed["bins"].event_count.to_numpy()
    expected = [counts[(centers > t) & (centers <= t + 1.5)].sum() for t in centers]
    np.testing.assert_array_equal(fixed["bins"].same_type_history_count, expected)
    assert not np.array_equal(fixed["bins"].same_type_history_count,
                              result["bins"].same_type_history_count)
    s = fixed["summary"].iloc[0]
    assert s.info_bits_per_event == pytest.approx(s.LR_statistic / (2 * 59 * np.log(2)))
    # Default consumers, including the chronology sampler, still receive the primary picks.
    pd.testing.assert_frame_equal(barker.load_barker_source(),
                                  barker.load_barker_source("variable_threshold"))
