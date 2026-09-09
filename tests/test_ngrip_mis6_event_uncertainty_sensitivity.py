"""Scientific checks for the joint NGRIP--MIS 6 age-uncertainty PI run."""

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
from toolbox import combined_pi


@pytest.fixture(scope="module")
def events() -> pd.DataFrame:
    return combined_pi.load_event_catalogue()


@pytest.fixture(scope="module")
def source_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(analysis.NGRIP_MC_INPUT),
        pd.read_csv(analysis.MIS6_MC_INPUT),
    )


@pytest.fixture(scope="module")
def small_setup(events):
    context = combined_pi.build_context()
    draws = analysis.load_joint_realizations(events, n_realizations=5)
    results, diagnostics = analysis.fit_realizations(events, draws, context)
    point_fit = combined_pi.fit_catalogue(events, context)
    return context, draws, results, diagnostics, point_fit


def test_main_model_and_uncertainty_scope_are_frozen():
    assert analysis.N_REALIZATIONS == 10_000
    assert analysis.HISTORY_WINDOW_KYR == pytest.approx(1.5)
    assert analysis.BIN_WIDTH_KYR == pytest.approx(0.2)
    assert analysis.BIN_ORIGIN_FRACTION == pytest.approx(0.0)
    assert analysis.RESPONSE_MODE == "maximal_for_history"
    assert analysis.RESOLUTION_COVARIATE_INCLUDED is False
    assert combined_pi.REDUCED_TERMS == (
        "same_type_history_count",
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


def test_small_mc_fit_rebuilds_history_and_reports_finite_results(small_setup):
    context, _, results, diagnostics, _ = small_setup

    assert len(results) == 5
    assert context.response_exposure_kyr == pytest.approx(180.0)
    valid = results.loc[results["fit_valid"].eq(True)]
    assert valid["n_predictive_events"].between(0, 55).all()
    assert valid["n_predictive_bins"].eq(901).all()
    assert valid["nominal_LR_p"].between(0.0, 1.0).all()
    assert valid["pre_phase_preferred_deg"].between(0.0, 360.0).all()
    assert valid["pre_phase_rate_ratio_max_vs_min"].ge(1.0).all()
    assert valid["all_models_converged"].all()
    assert valid["likelihood_nesting_ok"].all()
    assert not valid["eta_clipping_used"].any()
    assert diagnostics["n_realizations"] == 5
    assert diagnostics["n_unique_event_count_patterns"] <= 5


def test_summary_uses_exact_nominal_significance_count(small_setup):
    _, _, results, _, point_fit = small_setup
    summary = analysis.build_summary(results, point_fit).iloc[0]

    n_below = int(results["nominal_LR_p"].lt(analysis.P_THRESHOLD).sum())
    assert summary["n_nominal_p_below_0p05"] == n_below
    assert summary["fraction_nominal_p_below_0p05"] == pytest.approx(
        n_below / int(results["fit_valid"].sum())
    )
    assert not bool(summary["significance_fraction_is_empirical_p"])
    for column in (
        "nominal_LR_p",
        "info_bits_per_event",
        "delta_AICc_full_minus_reduced",
        "pre_phase_rate_ratio_max_vs_min",
        "pre_phase_preferred_deg",
    ):
        assert summary[f"{column}_q025"] <= summary[f"{column}_median"]
        assert summary[f"{column}_median"] <= summary[f"{column}_q975"]


def test_outputs_are_compact_and_provenance_is_explicit(events, small_setup):
    context, _, results, diagnostics, _ = small_setup
    compact = analysis.compact_pi_results(results)
    parameters = analysis.build_parameters(events, context, diagnostics)

    assert "event_count_pattern_id" not in compact
    assert "loglik_reduced" not in compact
    assert {
        "LR_statistic",
        "nominal_LR_p",
        "info_bits_per_event",
        "delta_AICc_full_minus_reduced",
        "pre_phase_preferred_deg",
        "pre_phase_rate_ratio_max_vs_min",
    }.issubset(compact.columns)
    assert parameters["parameter"].is_unique
    values = parameters.set_index("parameter")["value"].astype(str)
    assert values["pairing_scheme"] == "independent permutations without replacement"
    assert values["event_order"] == "fixed; no sorting repair"
    assert values["robustness_fraction"] == "count(valid nominal p < 0.05) / n_valid"
    assert {"fit_valid", "invalid_reason"}.issubset(compact.columns)
    assert values["response_mode"] == "maximal_for_history"


def test_saved_full_run_is_complete_reproducible_and_internally_consistent(events):
    draws = pd.read_csv(analysis.COMBINED_AGE_OUTPUT)
    results = pd.read_csv(analysis.PI_OUTPUT)
    summary = pd.read_csv(analysis.SUMMARY_OUTPUT).iloc[0]
    ngrip_source = pd.read_csv(analysis.NGRIP_MC_INPUT)
    mis6_source = pd.read_csv(analysis.MIS6_MC_INPUT)

    assert draws.shape == (10_000, 58)
    assert results.shape == (10_000, 15)
    assert draws["realization_id"].tolist() == results["realization_id"].tolist()
    assert draws["realization_id"].is_unique
    assert set(draws["ngrip_realization_id"]) == set(
        ngrip_source["realization_id"]
    )
    assert set(draws["mis6_realization_id"]) == set(
        mis6_source["realization_id"]
    )
    assert draws.columns[3:].tolist() == analysis.combined_age_columns(events)

    ages = draws.iloc[:, 3:].to_numpy(float)
    assert np.all(np.diff(ages[:, :34], axis=1) > 0.0)
    assert np.all(np.diff(ages[:, 34:], axis=1) > 0.0)
    assert np.all(ages[:, 33] < ages[:, 34])
    analysis._validate_pi_results(results, expected_events=55)
    valid = results.loc[results["fit_valid"].eq(True)]
    assert summary["n_valid"] == len(valid)
    assert summary["n_invalid"] == len(results) - len(valid)
    assert summary["robustness_denominator"] == len(valid)
    n_below = int(valid["nominal_LR_p"].lt(analysis.P_THRESHOLD).sum())
    assert summary["n_nominal_p_below_0p05"] == n_below
    assert summary["fraction_nominal_p_below_0p05"] == pytest.approx(n_below / len(valid))
    assert summary["fraction_nominal_p_below_0p05_all_draws_lower_bound"] == pytest.approx(n_below / len(results))
    assert summary["info_bits_per_event_median"] == pytest.approx(valid["info_bits_per_event"].median(), abs=1e-8)
    expected_phase = analysis.unwrap_around(valid["pre_phase_preferred_deg"].to_numpy(), summary["point_pre_phase_preferred_deg"])
    assert summary["pre_phase_preferred_deg_median"] == pytest.approx(np.median(expected_phase), abs=1e-6)

    # All paired ages must still be the original source values, including
    # realizations outside model observation coverage.
    columns = analysis.source_age_columns(events)
    for segment, source in [("NGRIP", ngrip_source), ("MIS6", mis6_source)]:
        id_column = "ngrip_realization_id" if segment == "NGRIP" else "mis6_realization_id"
        expected = source.set_index("realization_id").loc[draws[id_column], columns[segment]].to_numpy(float)
        target_columns = [col for col in analysis.combined_age_columns(events) if f"__{segment}:" in col]
        np.testing.assert_allclose(draws[target_columns].to_numpy(), expected, atol=5.1e-7, rtol=0)


def boundary_draws(events):
    ages = events[combined_pi.EVENT_AGE_COLUMN].to_numpy(float)
    draws = pd.DataFrame(np.tile(ages, (3, 1)), columns=analysis.combined_age_columns(events))
    draws.insert(0, "realization_id", ["inside", "history_only", "outside_observation"])
    draws.insert(1, "ngrip_realization_id", ["n1", "n2", "n3"])
    draws.insert(2, "mis6_realization_id", ["m1", "m2", "m3"])
    draws.loc[1, "age_kyr_bp__NGRIP:GI-25"] = 122.0
    draws.loc[2, "age_kyr_bp__NGRIP:GI-25"] = 124.0
    return draws


def test_boundary_draws_retain_all_ages_and_distinguish_response_from_observation(events, small_setup):
    context, _, _, _, point_fit = small_setup
    draws = boundary_draws(events)
    original = draws.copy(deep=True)
    results, diagnostics = analysis.fit_realizations(events, draws, context)
    pd.testing.assert_frame_equal(draws, original)
    assert results.realization_id.tolist() == draws.realization_id.tolist()
    assert results.fit_valid.tolist() == [True, True, False]
    assert results.n_predictive_events.iloc[:2].tolist() == [55, 54]
    assert results.n_source_events.iloc[1] == 55
    assert "NGRIP:GI-25=124" in results.invalid_reason.iloc[2]
    assert results.loc[2, list(analysis.PI_NUMERIC_COLUMNS)].isna().all()
    assert diagnostics["n_valid"] == 2
    assert diagnostics["n_invalid"] == 1
    assert diagnostics["n_realizations_with_response_event_loss"] == 1
    assert diagnostics["n_cached_reuses"] == 0
    summary = analysis.build_summary(results, point_fit).iloc[0]
    n_below = results.loc[results.fit_valid, "nominal_LR_p"].lt(0.05).sum()
    assert summary.n_realizations == 3
    assert summary.n_valid == summary.robustness_denominator == 2
    assert summary.n_invalid == 1
    assert summary.fraction_nominal_p_below_0p05 == pytest.approx(n_below / 2)
    assert summary.fraction_nominal_p_below_0p05_all_draws_lower_bound == pytest.approx(n_below / 3)
    assert analysis.compact_pi_results(results).shape == (3, 15)


def test_only_observation_boundary_exception_is_captured(events, small_setup, monkeypatch):
    context = small_setup[0]
    draws = boundary_draws(events)
    ages = draws[analysis.combined_age_columns(events)].iloc[2].to_numpy()
    with pytest.raises(analysis.OutsideObservationSupport, match="observation support"):
        analysis._counts_from_ages(ages, events, context)

    def broken_fit(*args, **kwargs):
        raise ValueError("unrelated optimizer or input failure")
    monkeypatch.setattr(combined_pi, "fit_event_counts", broken_fit)
    with pytest.raises(ValueError, match="unrelated optimizer"):
        analysis.fit_realizations(events, draws.iloc[:1], context)


def test_all_outside_draws_have_explicit_empty_summary_without_redrawing(events, small_setup):
    context, _, _, _, point_fit = small_setup
    draws = boundary_draws(events).iloc[2:].copy()
    results, diagnostics = analysis.fit_realizations(events, draws, context)
    assert len(results) == 1
    assert results.fit_valid.eq(False).all()
    summary = analysis.build_summary(results, point_fit).iloc[0]
    assert summary.n_realizations == summary.n_invalid == 1
    assert summary.n_valid == summary.robustness_denominator == 0
    assert np.isnan(summary.fraction_nominal_p_below_0p05)
    assert np.isnan(summary.info_bits_per_event_median)
    assert diagnostics["n_unique_event_count_patterns"] == 0
    assert analysis.compact_pi_results(results).shape == (1, 15)


def test_plot_uses_only_valid_realizations(events, small_setup, monkeypatch, tmp_path):
    context, _, _, _, point_fit = small_setup
    results, _ = analysis.fit_realizations(events, boundary_draws(events), context)
    observed_sizes = []
    original_panel = analysis._histogram_panel
    def inspect_panel(ax, values, *args, **kwargs):
        observed_sizes.append(len(values))
        assert np.isfinite(values).all()
        return original_panel(ax, values, *args, **kwargs)
    monkeypatch.setattr(analysis, "_histogram_panel", inspect_panel)
    monkeypatch.setattr(analysis, "OUT_FIG_DIR", tmp_path)
    monkeypatch.setattr(analysis, "PNG_DPI", 80)
    png, pdf = analysis.plot_sensitivity(results, point_fit)
    assert observed_sizes == [2] * 5
    assert png.exists() and pdf.exists()
