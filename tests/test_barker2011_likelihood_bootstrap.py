"""Barker reuses the continuous null driver with its own source and seed."""
import numpy as np
import pandas as pd
import pytest
from Barker2011 import Barker2011_likelihood_bootstrap as bootstrap
from toolbox import combined_likelihood as c


@pytest.fixture(scope="module")
def result():
    return bootstrap.run_analysis(n_bootstrap=4, seed=191)


@pytest.fixture(scope="module")
def fixed_result():
    return bootstrap.run_analysis(n_bootstrap=2, seed=193, event_definition="fixed_threshold")


def test_primary_catalogue_support_and_observed_statistic(result):
    assert len(result["events"]) == 70
    assert result["events"].event_id.str.startswith("Barker_S3_").all()
    assert result["summary"].event_definition.item() == "variable_threshold"
    observed = result["observed_fit"]
    assert observed.summary["n_response_events"] == 69
    assert observed.summary["LR_statistic"] == pytest.approx(10.582480594, abs=1e-6)
    assert result["context"].response_exposure_kyr == pytest.approx(396.4642638777)
    assert result["replicates"].fit_valid.all()
    assert bootstrap.DEFAULT_SEED == 20260909 and bootstrap.DEFAULT_N_BOOTSTRAP == 9999


def test_shared_driver_parallel_and_serial_agree(result):
    parallel = bootstrap.run_analysis(n_bootstrap=4, seed=191, n_workers=2)
    pd.testing.assert_frame_equal(result["replicates"], parallel["replicates"])
    np.testing.assert_allclose(result["summary"].empirical_p_plus_one,
                               parallel["summary"].empirical_p_plus_one)


def test_fixed_threshold_uses_own_anchor_background_and_null(fixed_result, result):
    fixed = fixed_result
    assert len(fixed["events"]) == 59
    assert fixed["summary"].event_definition.item() == "fixed_threshold"
    assert fixed["summary"].n_response_events.item() == 58
    assert fixed["summary"].LR_statistic.item() == pytest.approx(8.57672375244, abs=1e-6)
    assert fixed["context"].response_exposure_kyr == pytest.approx(392.245695897)
    assert fixed["context"].response_exposure_kyr != result["context"].response_exposure_kyr
    assert fixed["context"].scaling != result["context"].scaling
    # The two definitions must not share the fitted BG model or reference LR.
    assert fixed["observed_fit"].reduced.log_likelihood != result["observed_fit"].reduced.log_likelihood
    assert fixed["replicates"].fit_valid.all()
    assert fixed["parameters"].set_index("parameter").loc["event_definition", "value"] == "fixed_threshold"
    row = fixed["summary"].iloc[0]
    exceedances = (fixed["replicates"].LR_statistic >= row.LR_statistic).sum()
    assert row.empirical_p_plus_one == (exceedances + 1) / (len(fixed["replicates"]) + 1)


def test_fixed_outputs_do_not_replace_variable_outputs(fixed_result, result, tmp_path):
    data_dir, figure_dir, notes_dir = bootstrap.output_directories(tmp_path, "variable_threshold")
    fixed_data, fixed_figure, fixed_notes = bootstrap.output_directories(tmp_path, "fixed_threshold")
    bootstrap.save_tables(result, data_dir)
    bootstrap.write_notes(result, notes_dir)
    original_summary = (data_dir / "summary.csv").read_bytes()
    original_note = (notes_dir / "Barker2011_likelihood_bootstrap_Caption.txt").read_bytes()
    bootstrap.save_tables(fixed_result, fixed_data)
    bootstrap.write_notes(fixed_result, fixed_notes)
    assert (data_dir / "summary.csv").read_bytes() == original_summary
    assert (notes_dir / "Barker2011_likelihood_bootstrap_Caption.txt").read_bytes() == original_note
    assert fixed_figure == figure_dir / "fixed_threshold"
    assert pd.read_csv(fixed_data / "summary.csv").event_definition.item() == "fixed_threshold"
    caption = (notes_dir / "Barker2011_likelihood_bootstrap_fixed_threshold_Caption.txt").read_text()
    assert "fixed-threshold" in caption and "Chronology is fixed" in caption


def test_barker_staged_outputs_and_figure(result, tmp_path):
    import matplotlib.pyplot as plt
    bootstrap.save_tables(result, tmp_path / "data")
    bootstrap.write_notes(result, tmp_path / "notes")
    hashes = pd.read_csv(tmp_path / "data/input_code_sha256.csv")
    assert hashes.path.str.contains("Barker2011_likelihood_bootstrap.py").any()
    fig = bootstrap.plot_null_distribution(result["replicates"], result["summary"])
    assert len(fig.axes) == 1
    assert fig.axes[0].get_xlabel() == "Likelihood-ratio statistic"
    plt.close(fig)
    assert "continuous" in (tmp_path / "notes/Barker2011_likelihood_bootstrap_Caption.txt").read_text().lower()
