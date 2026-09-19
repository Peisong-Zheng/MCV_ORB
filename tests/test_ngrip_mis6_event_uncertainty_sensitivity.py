"""Scientific checks for the joint NGRIP--MIS 6 age-uncertainty G run."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import NGRIP_MIS6_event_uncertainty_sensitivity as analysis
from toolbox import combined_likelihood, age_sensitivity, age_sensitivity_plotting


@pytest.fixture(scope="module")
def events() -> pd.DataFrame:
    return combined_likelihood.load_event_catalogue()


@pytest.fixture(scope="module")
def source_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(analysis.NGRIP_MC_INPUT),
        pd.read_csv(analysis.MIS6_MC_INPUT),
    )


@pytest.fixture(scope="module")
def small_setup(events):
    context = combined_likelihood.build_context()
    draws = analysis.load_joint_realizations(events, n_realizations=5)
    results, diagnostics = analysis.fit_realizations(events, draws, context)
    point_fit = combined_likelihood.fit_catalogue(events, context)
    return context, draws, results, diagnostics, point_fit


def test_main_model_and_uncertainty_scope_are_frozen():
    assert analysis.N_REALIZATIONS == 10_000
    assert analysis.HISTORY_TAU_KYR == pytest.approx(1.5)
    assert combined_likelihood.DEFAULT_HISTORY_TAU_KA == 1.5
    assert combined_likelihood.REDUCED_TERMS == (
        "same_type_exponential_history",
        "lr04_scaled",
        "co2_scaled",
        "mis6_segment",
    )


def test_source_columns_follow_stable_event_ids_not_display_suffixes(events):
    columns = analysis.source_age_columns(events)

    assert len(columns["NGRIP"]) == 34
    assert len(columns["MIS6"]) == 21
    assert columns["NGRIP"][:3] == [
        "age_ka_bp__GI-1",
        "age_ka_bp__GI-2.1",
        "age_ka_bp__GI-2.2",
    ]
    assert "age_ka_bp__GI-1e" not in columns["NGRIP"]
    assert columns["MIS6"][:2] == [
        "MIS6_DO_01_age_ka_bp",
        "MIS6_DO_02_age_ka_bp",
    ]

    combined = analysis.combined_age_columns(events)
    assert len(combined) == 55
    assert len(set(combined)) == 55
    assert combined[0] == "age_kyr_bp__NGRIP:GI-1"
    assert combined[-1] == "age_kyr_bp__MIS6:MIS6_DO_21"


def test_independent_pairing_is_reproducible_and_traceable(events, source_tables):
    ngrip_table, mis6_table = source_tables
    first = analysis.pair_source_ensembles(
        events,
        ngrip_table,
        mis6_table,
        n_realizations=20,
        seed=analysis.PAIRING_SEED,
    )
    second = analysis.pair_source_ensembles(
        events,
        ngrip_table,
        mis6_table,
        n_realizations=20,
        seed=analysis.PAIRING_SEED,
    )
    pd.testing.assert_frame_equal(first, second)

    assert first["realization_id"].is_unique
    assert first["ngrip_realization_id"].is_unique
    assert first["mis6_realization_id"].is_unique
    assert first["ngrip_realization_id"].tolist() != ngrip_table[
        "realization_id"
    ].head(20).tolist()
    assert first["mis6_realization_id"].tolist() != mis6_table[
        "realization_id"
    ].head(20).tolist()

    ngrip_lookup = ngrip_table.set_index("realization_id")
    mis6_lookup = mis6_table.set_index("realization_id")
    first_row = first.iloc[0]
    assert first_row["age_kyr_bp__NGRIP:GI-1"] == pytest.approx(
        ngrip_lookup.loc[first_row["ngrip_realization_id"], "age_ka_bp__GI-1"]
    )
    assert first_row["age_kyr_bp__MIS6:MIS6_DO_21"] == pytest.approx(
        mis6_lookup.loc[
            first_row["mis6_realization_id"], "MIS6_DO_21_age_ka_bp"
        ]
    )

    ages = first[analysis.combined_age_columns(events)].to_numpy(float)
    assert np.all(np.diff(ages[:, :34], axis=1) > 0.0)
    assert np.all(np.diff(ages[:, 34:], axis=1) > 0.0)
    assert np.all(ages[:, 33] < ages[:, 34])


def test_crossed_source_sequence_is_rejected_not_sorted(events, source_tables):
    ngrip_table, mis6_table = (table.head(3).copy() for table in source_tables)
    first = ngrip_table.columns.get_loc("age_ka_bp__GI-1")
    second = ngrip_table.columns.get_loc("age_ka_bp__GI-2.1")
    ngrip_table.iloc[0, [first, second]] = ngrip_table.iloc[0, [second, first]].to_numpy()

    with pytest.raises(ValueError, match="never sorted"):
        analysis.pair_source_ensembles(
            events,
            ngrip_table,
            mis6_table,
            n_realizations=2,
            seed=analysis.PAIRING_SEED,
        )


def test_small_mc_reconditions_exact_anchors_and_reports_finite_results(small_setup):
    context, draws, results, diagnostics, _ = small_setup
    assert len(results) == 5
    assert results.fit_valid.all()
    assert results.n_response_events.eq(53).all()
    assert results.n_conditioning_events.eq(2).all()
    assert results.nominal_LR_p.between(0, 1).all()
    assert results.pre_phase_preferred_deg.between(0, 360).all()
    assert results.pre_phase_rate_ratio_max_vs_min.ge(1).all()
    assert results.all_models_converged.all() and results.likelihood_nesting_ok.all()
    assert diagnostics['age_input_reused']
    assert not diagnostics['event_count_pattern_cache']
    assert diagnostics['n_numerical_failures'] == 0
    for i, row in draws.iterrows():
        expected = sum(row[f'age_kyr_bp__{context.events.loc[context.events.segment_id.eq(name)].iloc[-1].event_id}']
                       - segment.observation_start_kyr_bp for name, segment in context.segments.items())
        assert results.response_exposure_kyr.iloc[i] == pytest.approx(expected)


def test_summary_reports_actual_denominator_and_circular_phase(small_setup):
    _, _, results, _, point_fit = small_setup
    summary = analysis.build_summary(results, point_fit).iloc[0]
    valid = results.loc[results.fit_valid]
    assert summary.n_nominal_p_below_0p05 == valid.nominal_LR_p.lt(.05).sum()
    assert summary.fraction_nominal_p_below_0p05 == pytest.approx(valid.nominal_LR_p.lt(.05).mean())
    assert summary.n_valid == len(valid)
    phase = age_sensitivity.unwrap_phase(valid.pre_phase_preferred_deg, point_fit.summary['pre_phase_preferred_deg'])
    assert summary.pre_phase_preferred_deg_median == pytest.approx(np.median(phase))
    assert summary.phase_quantiles_unwrapped_about_point
    assert not any('AICc' in column for column in summary.index)


def test_compact_results_keep_identifiers_and_scientific_metrics(small_setup):
    compact = analysis.compact_gain_results(small_setup[2])
    assert {'realization_id','ngrip_realization_id','mis6_realization_id','fit_valid','invalid_reason',
            'LR_statistic','nominal_LR_p','gain_bits_per_event','delta_AIC_full_minus_reduced',
            'beta_history','beta_pre_phase_sin','beta_pre_phase_cos'}.issubset(compact.columns)
    assert not any('count_pattern' in name or 'bin' in name or 'AICc' in name for name in compact)


def boundary_draws(events):
    ages = events[combined_likelihood.EVENT_AGE_COLUMN].to_numpy(float)
    draws = pd.DataFrame(np.tile(ages, (3, 1)), columns=analysis.combined_age_columns(events))
    draws.insert(0, 'realization_id', ['nominal', 'older_anchor', 'outside'])
    draws.insert(1, 'ngrip_realization_id', ['n1', 'n2', 'n3'])
    draws.insert(2, 'mis6_realization_id', ['m1', 'm2', 'm3'])
    draws.loc[1, 'age_kyr_bp__NGRIP:GI-25'] = 122.0
    draws.loc[2, 'age_kyr_bp__NGRIP:GI-25'] = 124.0
    return draws


def test_boundary_draws_preserve_all_ages_and_original_failure_ids(events, small_setup):
    context, _, _, _, point_fit = small_setup
    draws = boundary_draws(events)
    original = draws.copy(deep=True)
    results, diagnostics = analysis.fit_realizations(events, draws, context)
    pd.testing.assert_frame_equal(draws, original)
    assert results.realization_id.tolist() == draws.realization_id.tolist()
    assert results.fit_valid.tolist() == [True, True, False]
    assert results.n_response_events.iloc[:2].tolist() == [53, 53]
    assert results.response_exposure_kyr.iloc[1] > results.response_exposure_kyr.iloc[0]
    assert results.invalid_reason.iloc[2] == 'outside_NGRIP_observation_support'
    assert diagnostics['n_valid'] == 2 and diagnostics['n_invalid'] == 1
    assert not diagnostics['event_count_pattern_cache']
    summary = analysis.build_summary(results, point_fit).iloc[0]
    assert summary.n_realizations == 3 and summary.n_valid == 2
    assert summary.fraction_nominal_p_below_0p05 == pytest.approx(results.iloc[:2].nominal_LR_p.lt(.05).mean())


def test_numerical_failures_retained_but_programming_errors_propagate(events, small_setup, monkeypatch):
    from toolbox.point_process import PointProcessFitError
    context = small_setup[0]
    draws = boundary_draws(events).iloc[:1]
    def failed(*args, **kwargs):
        raise PointProcessFitError('diagnosed optimization failure')
    monkeypatch.setattr(combined_likelihood, 'fit_catalogue', failed)
    results, diagnostics = analysis.fit_realizations(events, draws, context)
    assert not results.fit_valid.any()
    assert results.invalid_reason.str.startswith('numerical_fit:').all()
    assert diagnostics['n_numerical_failures'] == 1
    def broken(*args, **kwargs):
        raise ValueError('unrelated implementation error')
    monkeypatch.setattr(combined_likelihood, 'fit_catalogue', broken)
    with pytest.raises(ValueError, match='unrelated implementation'):
        analysis.fit_realizations(events, draws, context)


def test_all_outside_draws_have_explicit_empty_summary_without_redrawing(events, small_setup):
    context, _, _, _, point_fit = small_setup
    draws = boundary_draws(events).iloc[2:]
    results, diagnostics = analysis.fit_realizations(events, draws, context)
    summary = analysis.build_summary(results, point_fit).iloc[0]
    assert len(results) == 1 and not results.fit_valid.any()
    assert summary.n_realizations == summary.n_invalid == 1
    assert summary.n_valid == 0
    assert np.isnan(summary.fraction_nominal_p_below_0p05)
    assert np.isnan(summary.gain_bits_per_event_median)
    assert diagnostics['n_numerical_failures'] == 0


def test_plot_uses_only_valid_realizations(events, small_setup, monkeypatch):
    import matplotlib.pyplot as plt
    context, _, _, _, point_fit = small_setup
    results, _ = analysis.fit_realizations(events, boundary_draws(events), context)
    sizes = []
    original = age_sensitivity_plotting._histogram_panel
    def inspect(ax, values, *args, **kwargs):
        sizes.append(len(values))
        assert np.isfinite(values).all()
        return original(ax, values, *args, **kwargs)
    monkeypatch.setattr(age_sensitivity_plotting, '_histogram_panel', inspect)
    fig = age_sensitivity_plotting.plot_sensitivity(results, point_fit)
    assert sizes == [2] * 5
    plt.close(fig)
