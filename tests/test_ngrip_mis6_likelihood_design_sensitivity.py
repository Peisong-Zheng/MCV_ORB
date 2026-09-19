"""Prespecified continuous history scenarios keep exact support and scaling."""

import numpy as np
import pandas as pd
import pytest

import NGRIP_MIS6_likelihood_design_sensitivity as sensitivity
from toolbox import combined_likelihood


@pytest.fixture(scope="module")
def results():
    return (sensitivity.run_design_sensitivity(), sensitivity.run_initial_history_sensitivity(),
            sensitivity.run_pooling_diagnostic())


def test_decay_and_initial_history_grids_are_prespecified(results):
    design, initial, _ = results
    assert design.history_tau_kyr.tolist() == [1., 1.5, 2., 3., 5.]
    assert initial.initial_unobserved_history.tolist() == [0., 0.5, 1.]
    assert initial.history_tau_kyr.eq(1.5).all()
    assert design.initial_unobserved_history.eq(0).all()
    assert design.is_primary_design.sum() == initial.is_primary_design.sum() == 1
    for frame in (design, initial):
        assert frame.n_response_events.eq(53).all()
        np.testing.assert_allclose(frame.response_exposure_kyr, 165.058)
        assert frame.beta_history.le(0).all()
        assert frame.all_models_converged.all() and frame.likelihood_nesting_ok.all()
        np.testing.assert_allclose(frame.delta_AIC_full_minus_reduced, 4 - frame.LR_statistic)
        assert not {"bin_width_kyr", "origin_fraction", "n_response_bins"}.intersection(frame.columns)


def test_primary_scenario_matches_continuous_main(results):
    main = combined_likelihood.fit_point_catalogue().summary
    for frame in results[:2]:
        row = frame.loc[frame.is_primary_design].iloc[0]
        for field in ("gain_bits_per_event", "LR_statistic", "pre_phase_preferred_deg",
                      "pre_phase_rate_ratio_max_vs_min"):
            assert row[field] == pytest.approx(main[field], abs=1e-7)


def test_pooling_adds_two_segment_phase_terms(results):
    row = results[2].iloc[0]
    assert row.df == 2 and row.n_events == 53
    assert row.both_models_converged and row.likelihood_nesting_ok
    assert row.response_exposure_kyr == pytest.approx(165.058)
    assert row.LR_statistic == pytest.approx(2 * (row.segment_specific_model_loglik-row.common_model_loglik))
    assert row.delta_AIC_segment_specific_minus_common == pytest.approx(4-row.LR_statistic)
    assert 0 <= row.ngrip_preferred_phase_deg < 360
    assert 0 <= row.mis6_preferred_phase_deg < 360


def test_written_outputs_have_continuous_support(results, tmp_path):
    sensitivity.write_outputs(*results, output_dir=tmp_path)
    design = pd.read_csv(tmp_path / "design_sensitivity.csv")
    initial = pd.read_csv(tmp_path / "initial_history_sensitivity.csv")
    support = pd.read_csv(tmp_path / "support.csv")
    assert len(design) == 5 and len(initial) == 3
    assert support.anchor_age_kyr_bp.tolist() == pytest.approx([115.32, 194.238])
    assert not (tmp_path / "support_sensitivity.csv").exists()
