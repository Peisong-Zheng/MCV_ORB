"""Barker reuses the continuous null driver with its own source and seed."""
import numpy as np
import pandas as pd
import pytest
from Barker2011 import Barker2011_likelihood_bootstrap as bootstrap
from toolbox import combined_likelihood as c


@pytest.fixture(scope="module")
def result():
    return bootstrap.run_analysis(n_bootstrap=4, seed=191)


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
