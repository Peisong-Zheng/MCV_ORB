"""Scientific checks for pooled PI structural sensitivity."""

from __future__ import annotations

from itertools import product
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import NGRIP_MIS6_PI_design_sensitivity as sensitivity


@pytest.fixture(scope="module")
def design() -> pd.DataFrame:
    return sensitivity.run_design_sensitivity()


@pytest.fixture(scope="module")
def support() -> pd.DataFrame:
    return sensitivity.run_support_sensitivity()


@pytest.fixture(scope="module")
def pooling() -> pd.DataFrame:
    return sensitivity.run_pooling_diagnostic()


def test_design_grid_contains_all_30_prespecified_cells(design):
    expected = set(
        product(
            sensitivity.HISTORY_WINDOWS_KYR,
            sensitivity.BIN_WIDTHS_KYR,
            sensitivity.ORIGIN_FRACTIONS,
        )
    )
    observed = set(
        design[["history_window_kyr", "bin_width_kyr", "origin_fraction"]].itertuples(
            index=False, name=None
        )
    )

    assert len(design) == 30
    assert design["design_id"].is_unique
    assert observed == expected
    assert design.groupby("origin_fraction").size().to_dict() == {0.0: 15, 0.5: 15}


def test_common_core_holds_exposure_events_and_gap_fixed(design):
    assert design["response_mode"].eq("common_core").all()
    np.testing.assert_allclose(design["response_exposure_kyr"], 173.0, atol=1e-10)
    assert design["n_source_events"].eq(55).all()
    assert design["n_predictive_events"].eq(55).all()
    assert not design["gap_counted_as_exposure"].any()

    np.testing.assert_allclose(design["ngrip_response_start_kyr_bp"], 12.0)
    np.testing.assert_allclose(design["ngrip_response_end_kyr_bp"], 118.0)
    np.testing.assert_allclose(design["mis6_response_start_kyr_bp"], 132.5)
    np.testing.assert_allclose(design["mis6_response_end_kyr_bp"], 199.5)


def test_primary_design_is_explicit_and_reproduces_point_fit(design):
    primary = design.loc[design["is_primary_design"]]
    assert len(primary) == 1
    row = primary.iloc[0]

    assert row["history_window_kyr"] == pytest.approx(1.5)
    assert row["bin_width_kyr"] == pytest.approx(0.2)
    assert row["origin_fraction"] == pytest.approx(0.0)
    observed = [
        row["n_predictive_bins"],
        row["LR_statistic"],
        row["nominal_LR_p"],
        row["info_bits_per_event"],
        row["pre_phase_preferred_deg"],
        row["pre_phase_rate_ratio_max_vs_min"],
    ]
    expected = [
        865,
        13.1514026535,
        0.001393828047,
        0.1724860308,
        330.0335287,
        4.759994653,
    ]
    np.testing.assert_allclose(observed, expected, rtol=2e-7, atol=2e-9)


def test_every_design_fit_is_numerically_valid(design):
    assert design["all_models_converged"].all()
    assert design["likelihood_nesting_ok"].all()
    assert not design["eta_clipping_used"].any()
    assert design["nominal_LR_p"].between(0.0, 1.0).all()
    assert np.isfinite(
        design[
            [
                "LR_statistic",
                "info_bits_per_event",
                "pre_phase_preferred_deg",
                "pre_phase_rate_ratio_max_vs_min",
            ]
        ].to_numpy(float)
    ).all()


def test_endpoint_sensitivity_compares_common_and_maximal_support(support):
    assert support["response_mode"].tolist() == [
        "common_core",
        "maximal_for_history",
    ]
    assert support["is_main_analysis_support"].tolist() == [False, True]
    assert support["n_predictive_events"].eq(55).all()
    assert not support["gap_counted_as_exposure"].any()

    by_mode = support.set_index("response_mode")
    assert by_mode.loc["common_core", "response_exposure_kyr"] == pytest.approx(173.0)
    assert by_mode.loc["maximal_for_history", "response_exposure_kyr"] == pytest.approx(
        180.0
    )
    assert by_mode.loc["common_core", "n_predictive_bins"] == 865
    assert by_mode.loc["maximal_for_history", "n_predictive_bins"] == 901
    np.testing.assert_allclose(
        by_mode.loc[
            "maximal_for_history",
            [
                "ngrip_response_start_kyr_bp",
                "ngrip_response_end_kyr_bp",
                "mis6_response_start_kyr_bp",
                "mis6_response_end_kyr_bp",
            ],
        ].to_numpy(float),
        [12.0, 121.5, 132.5, 203.0],
    )


def test_endpoint_fits_are_valid_and_main_result_is_stable(support):
    assert support["all_models_converged"].all()
    assert support["likelihood_nesting_ok"].all()
    assert not support["eta_clipping_used"].any()

    main = support.loc[support["is_main_analysis_support"]].iloc[0]
    np.testing.assert_allclose(
        [
            main["LR_statistic"],
            main["nominal_LR_p"],
            main["info_bits_per_event"],
            main["pre_phase_preferred_deg"],
        ],
        [13.6732014732, 0.001073747153, 0.1793296360, 329.9985118],
        rtol=2e-7,
        atol=2e-9,
    )


def test_pooling_diagnostic_tests_two_phase_interactions(pooling):
    row = pooling.iloc[0]
    assert row["n_events"] == 55
    assert row["response_exposure_kyr"] == pytest.approx(180.0)
    assert row["df"] == 2
    assert 0.0 <= row["nominal_LR_p"] <= 1.0
    assert row["both_models_converged"]
    assert row["likelihood_nesting_ok"]
    assert not row["eta_clipping_used"]
    assert row["segment_specific_model_loglik"] >= row["common_model_loglik"]


def test_outputs_and_provenance_are_compact_and_self_describing(
    design, support, pooling, tmp_path
):
    sensitivity.write_outputs(design, support, pooling, tmp_path)
    written_design = pd.read_csv(tmp_path / "design_sensitivity.csv")
    written_support = pd.read_csv(tmp_path / "support_sensitivity.csv")
    written_pooling = pd.read_csv(tmp_path / "pooling_diagnostic.csv")
    parameters = pd.read_csv(tmp_path / "parameters_and_provenance.csv")

    assert len(written_design) == 30
    assert len(written_support) == 2
    assert len(written_pooling) == 1
    assert parameters["parameter"].is_unique
    values = parameters.set_index("parameter")["value"].astype(str)
    assert values["number_of_designs"] == "30"
    assert values["design_response_exposure"] == "173.0"
    assert values["resolution_covariate_included"] == "False"
    assert values["pooling_diagnostic"] == (
        "common phase coefficients versus MIS6 phase interactions"
    )


def test_figure_renders_with_four_labeled_panels(design, support):
    figure = sensitivity.plot_sensitivity(design, support)
    assert len(figure.axes) == 4
    for label, axis in zip("abcd", figure.axes):
        assert label in {text.get_text() for text in axis.texts}
    plt.close(figure)
