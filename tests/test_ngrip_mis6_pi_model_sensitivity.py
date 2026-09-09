"""Scientific checks for nonlinear climate and elapsed-event-time sensitivity."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import NGRIP_MIS6_PI_model_sensitivity as sensitivity
from toolbox import combined_pi


@pytest.fixture(scope="module")
def analysis():
    return sensitivity.run_analysis()


def test_elapsed_time_uses_strictly_older_events_and_resets_after_event():
    centers = np.arange(1.0, 7.0)
    # At center 3 there are two events, neither may affect its own predictor.
    elapsed = sensitivity.elapsed_since_event([0, 0, 2, 0, 1, 0], centers)
    np.testing.assert_allclose(elapsed, [2, 1, 2, 1, np.nan, np.nan], equal_nan=True)
    changed = sensitivity.elapsed_since_event([0, 0, 0, 0, 1, 0], centers)
    assert elapsed[2] == changed[2]
    assert elapsed[1] != changed[1]


def test_elapsed_time_preserves_nonuniform_grid_and_missing_prehistory():
    elapsed = sensitivity.elapsed_since_event([0, 1, 0], [0.1, 0.25, 0.5])
    np.testing.assert_allclose(elapsed, [0.15, np.nan, np.nan], equal_nan=True)
    assert np.isnan(sensitivity.elapsed_since_event([0, 0], [1, 2])).all()
    with pytest.raises(ValueError, match="increasing"):
        sensitivity.elapsed_since_event([1, 0], [2, 1])
    with pytest.raises(ValueError, match="integer"):
        sensitivity.elapsed_since_event([0.5, 0], [1, 2])


def test_history_initialization_does_not_cross_the_gap(analysis):
    bins = analysis["binned"]
    for segment_id in combined_pi.SEGMENT_IDS:
        segment = bins.loc[bins.segment_id.eq(segment_id)]
        anchor = segment.loc[segment.is_history_anchor].iloc[0]
        older = segment.bin_center_kyr_bp.ge(anchor.bin_center_kyr_bp)
        assert segment.loc[older, sensitivity.ELAPSED_TERM].isna().all()
        assert segment.loc[~older, sensitivity.ELAPSED_TERM].notna().all()
        assert not segment.loc[older, "in_history_comparison"].any()
        assert anchor.event_count == 1


def test_seed_events_are_retained_as_history_but_not_response(analysis):
    roles = analysis["event_roles"]
    anchors = roles.loc[roles.is_history_anchor]
    assert set(anchors.event_id) == {"NGRIP:GI-25", "MIS6:MIS6_DO_21"}
    assert roles.in_main_response.sum() == 55
    assert roles.in_history_comparison.sum() == 53
    assert not anchors.in_history_comparison.any()
    bins = analysis["binned"]
    for segment_id in combined_pi.SEGMENT_IDS:
        first_response = bins.loc[
            bins.segment_id.eq(segment_id) & bins.in_history_comparison
        ].iloc[-1]
        assert first_response[sensitivity.ELAPSED_TERM] == pytest.approx(0.2)
        assert first_response[combined_pi.REDUCED_TERMS[0]] == 1


def test_history_models_have_identical_exposure_and_keep_zero_event_tails(analysis):
    summary = analysis["summary"]
    history = summary.loc[summary.experiment.eq("history")]
    assert len(history) == 3
    assert history.n_predictive_events.eq(53).all()
    assert history.n_predictive_bins.eq(824).all()
    np.testing.assert_allclose(history.response_exposure_kyr, 164.8)
    support = analysis["segment_support"].set_index("segment_id")
    assert support.loc["NGRIP", "history_response_end_kyr_bp"] == pytest.approx(115.2)
    assert support.loc["MIS6", "history_response_end_kyr_bp"] == pytest.approx(194.1)
    for _, segment in analysis["binned"].groupby("segment_id"):
        youngest = segment.iloc[0]
        assert youngest.event_count == 0
        assert youngest.in_history_comparison
        assert youngest.dt_kyr > 0


def test_formal_climate_variants_reproduce_the_audit(analysis):
    audit = pd.read_csv(
        ROOT / "docs/reviews/research_audit_2026_09_05_outputs/climate_baseline_sensitivity.csv"
    ).set_index("variant")
    summary = analysis["summary"].set_index("variant")
    for variant in sensitivity.CLIMATE_VARIANTS:
        assert summary.loc[variant, "LR_statistic"] == pytest.approx(audit.loc[variant, "lr"], abs=1e-6)
        assert summary.loc[variant, "full_aic"] == pytest.approx(audit.loc[variant, "full_aic"], abs=1e-6)
        assert summary.loc[variant, "pre_phase_preferred_deg"] == pytest.approx(audit.loc[variant, "preferred_phase_deg"], abs=1e-4)


def test_new_baseline_reproduces_main_fit_and_nested_contract(analysis):
    point = combined_pi.fit_point_catalogue()
    base = analysis["summary"].set_index("variant").loc["frozen_linear"]
    for field in ("info_bits_per_event", "nominal_LR_p", "pre_phase_preferred_deg",
                  "pre_phase_rate_ratio_max_vs_min", "delta_AICc_full_minus_reduced"):
        assert base[field] == pytest.approx(point.summary[field], abs=1e-6)
    summary = analysis["summary"]
    assert (summary.n_params_full - summary.n_params_reduced).eq(2).all()
    assert summary.all_models_converged.all()
    assert summary.likelihood_nesting_ok.all()
    assert not summary.eta_clipping_used.any()
    assert not summary.coefficient_bound_reached.any()
    coefficients = analysis["coefficients"]
    for variant in sensitivity.HISTORY_VARIANTS:
        reduced_terms = set(coefficients.loc[
            coefficients.variant.eq(variant) & coefficients.model_id.eq("reduced"), "term"
        ])
        expected = {"intercept", sensitivity.HISTORY_VARIANTS[variant],
                    "lr04_scaled", "co2_scaled", "mis6_segment"}
        assert reduced_terms == expected


def test_aicc_reference_comparisons_stay_within_each_support(analysis):
    for experiment, frame in analysis["summary"].groupby("experiment", sort=False):
        assert frame.support_id.nunique() == 1
        assert frame.reference_variant.nunique() == 1
        reference = frame.loc[frame.variant.eq(frame.reference_variant.iloc[0])].iloc[0]
        np.testing.assert_allclose(
            frame.delta_AICc_full_vs_same_support_reference,
            frame.full_aicc - reference.full_aicc,
        )
    assert analysis["summary"].support_id.nunique() == 2


def test_saved_phase_curves_match_reported_phase_effect(analysis):
    for row in analysis["summary"].itertuples(index=False):
        curve = analysis["phase_curves"].loc[analysis["phase_curves"].variant.eq(row.variant)]
        # Grid is one degree; maxima and minima approximate the analytic extrema.
        ratio = curve.relative_rate.max() / curve.relative_rate.min()
        assert ratio == pytest.approx(row.pre_phase_rate_ratio_max_vs_min, rel=1e-4)
        assert curve.relative_rate.iloc[0] == pytest.approx(curve.relative_rate.iloc[-1])
