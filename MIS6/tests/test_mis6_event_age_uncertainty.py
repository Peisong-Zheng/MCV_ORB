"""Focused tests for the depth-informed MIS 6 uncertainty workflow."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MIS6 import MIS6_event_age_uncertainty as uncertainty
from MIS6 import event_detection as detector


@pytest.fixture(scope="module")
def prepared():
    controls = uncertainty.load_age_controls()
    envelope = uncertainty.load_mf_age_envelope(uncertainty.FOHLMEISTER_STACK)
    anchors = detector.load_selected_anchors(uncertainty.ANCHORS)
    segments = {
        spec.record_id: detector.regularize_record(
            detector.load_record(uncertainty.WORKBOOK, spec), spec
        )
        for spec in detector.RECORDS
    }
    picks = uncertainty.build_definition_picks(anchors, segments, controls, envelope)
    return controls, envelope, picks


def test_fixed_control_table_has_explicit_filename_provenance(prepared):
    controls, _, _ = prepared
    assert len(controls) == 11
    assert controls["control_id"].is_unique
    assert set(controls["record_id"]) == {"Sofular"}
    assert set(controls["cave"]) == {"Sofular Cave"}
    assert set(controls["source_study"]) == {"Held et al. (2024)"}
    assert all(Path(name).name == name for name in controls["source_filename"])
    assert "uncertainty_level" not in controls
    assert controls.columns.tolist() == [
        "control_id",
        "record_id",
        "cave",
        "stalagmite_id",
        "dating_sample_id",
        "depth_mm",
        "age_ka_bp",
        "age_error_2sigma_ka",
        "used_for_event_ids",
        "bracket_role",
        "source_study",
        "source_filename",
        "source_locator",
        "shown_in_composite_figure",
    ]
    assert uncertainty.AGE_CONTROLS == (
        PROJECT_ROOT / "MIS6/data/curated/mis6_age_control_points.csv"
    )


def test_fohlmeister_stack_is_a_workspace_input():
    assert uncertainty.FOHLMEISTER_STACK == (
        PROJECT_ROOT / "MIS6/data/raw/Fohlmeister J et al-2023-data-mf_d18o_stack.txt"
    )
    assert uncertainty.FOHLMEISTER_STACK.exists()


def test_mf_duplicate_ages_keep_the_outer_envelope(tmp_path):
    source = tmp_path / "bounds.txt"
    pd.DataFrame(
        {
            "age": [1.0, 1.0, 2.0],
            "age_upper": [1.2, 1.3, 2.2],
            "age_lower": [0.8, 0.7, 1.8],
        }
    ).to_csv(source, sep="\t", index=False)
    envelope = uncertainty.load_mf_age_envelope(source)
    duplicate = envelope.loc[envelope["age"].eq(1.0)].iloc[0]
    assert duplicate["safe_lower"] == pytest.approx(0.7)
    assert duplicate["safe_upper"] == pytest.approx(1.3)


def test_real_mf_envelope_counts_gap_and_sigma_conversion(prepared):
    _, envelope, _ = prepared
    assert envelope.attrs["valid_row_count"] == 4244
    assert envelope.attrs["unique_age_count"] == 4077
    assert envelope.attrs["source_rows_needing_bound_repair"] == 188
    with pytest.raises(ValueError, match="outside a valid segment"):
        uncertainty.mf_chronology_at_age(176.0, envelope)

    result = uncertainty.mf_chronology_at_age(134.976, envelope)
    assert result["chronology_sigma_ka"] == pytest.approx(
        result["chronology_half_width_approx95_ka"] / uncertainty.NORMAL_975
    )


def test_definition_runs_are_coherent_and_mf_bounds_regress(prepared):
    _, _, picks = prepared
    expected_counts = {"MF": 16, "Sofular": 5}
    for record_id, event_count in expected_counts.items():
        part = picks.loc[picks["source_record"].eq(record_id)]
        assert part["algorithm_config_id"].nunique() == 9
        assert part.groupby("algorithm_config_id").size().eq(event_count).all()

    nominal = picks.query(
        "source_record == 'MF' and algorithm_config_id == 's050_w400'"
    ).set_index("composite_event_id")
    expected = {
        "MIS6_DO_01": (133.9687778, 135.9595, 0.5138983),
        "MIS6_DO_11": (164.4004, 165.0060, 0.3089853),
        "MIS6_DO_12": (168.2558455, 171.1784277, 1.3328958),
        "MIS6_DO_16": (172.6707517, 176.9636584, 1.1284179),
    }
    for event_id, values in expected.items():
        observed = nominal.loc[
            event_id,
            [
                "chronology_input_lower_approx95_ka_bp",
                "chronology_input_upper_approx95_ka_bp",
                "chronology_sigma_ka",
            ],
        ].to_numpy(float)
        np.testing.assert_allclose(observed, values, atol=2e-6, rtol=0)


def test_new_sofular_6p9_event_is_stable_and_uses_so4_controls(prepared):
    _, _, picks = prepared
    event = picks.query(
        "composite_event_id == 'MIS6_DO_17' and algorithm_config_id == 's050_w400'"
    ).iloc[0]

    assert event["source_record"] == "Sofular"
    assert event["source_event_label"] == "6.9"
    assert event["definition_pick_age_ka_bp"] == pytest.approx(176.779)
    context = uncertainty.sofular.build_context([176.779])
    assert event["chronology_sigma_ka"] == pytest.approx(context.nominal_sigma_ka[0])
    assert "So-4:" in event["chronology_support_id"]
    assert "assumed_age_coordinate_projection" in event["chronology_support_id"]


def test_sofular_uses_depth_model_weights_and_no_extrapolation():
    age = 190.972
    result = uncertainty.sofular_chronology_at_age(age)
    context = uncertainty.sofular.build_context([age])
    expected_variance = np.sum(
        (context.interpolation_weights[0] * context.control_sigmas_ka) ** 2
    )
    assert result["chronology_sigma_ka"] ** 2 == pytest.approx(expected_variance)
    with pytest.raises(ValueError):
        uncertainty.sofular_chronology_at_age(206.0)


def test_selected_catalogue_is_mf_then_five_sofular_events(prepared):
    _, _, picks = prepared
    nominal = picks.loc[picks["algorithm_config_id"].eq("s050_w400")]
    nominal = nominal.sort_values("composite_event_number")
    assert nominal["composite_event_number"].tolist() == list(range(1, 22))
    assert nominal["source_record"].tolist() == ["MF"] * 16 + ["Sofular"] * 5
    assert nominal.tail(5)["source_event_label"].tolist() == [
        "6.9",
        "6.10",
        "6.11",
        "6.12",
        "6.13",
    ]


def test_proposal_retains_mf_factor_and_adds_local_sofular_errors(prepared):
    _, _, picks = prepared
    states, configs = uncertainty.build_state_lookup(picks)
    proposal = uncertainty.propose_realization(np.random.default_rng(42), states, configs)
    assert proposal["chronology_valid"]
    standardized = {}
    for record in uncertainty.RECORD_ORDER:
        state = states[(record, proposal["selected_configs"][record])]
        standardized[record] = (
            proposal["sampled_ages"][state.event_indices]
            - proposal["definition_ages"][state.event_indices]
        ) / proposal["chronology_sigmas"][state.event_indices]
    np.testing.assert_allclose(standardized["MF"], proposal["record_z"]["MF"], atol=1e-12)
    assert np.ptp(standardized["Sofular"]) > 0.1
    assert set(proposal["record_z"]) == {"MF"}
    crossed = np.array([1.0, 3.0, 2.0])
    assert not uncertainty.is_strictly_ordered(crossed)
    np.testing.assert_array_equal(crossed, [1.0, 3.0, 2.0])


def test_so57_overlap_uses_new_young_controls_and_so4_oldest(prepared):
    _, _, picks = prepared
    states, _ = uncertainty.build_state_lookup(picks, sofular_component="So-57")
    state = states[("Sofular", "s050_w400")]
    components = {context.component: indices.tolist() for indices, context in state.chronology_contexts}
    assert components == {"So-4": [4], "So-57": [0, 1, 2, 3]}
    draws, stats = uncertainty.sample_realizations(picks, 100, 13, sofular_component="So-57")
    assert draws.shape == (100, 22)
    assert stats["rejected_proposals"] == stats["rejected_nonmonotone_chronologies"] + stats["rejected_event_order"]


def test_sampling_is_reproducible_ordered_and_unsorted(prepared):
    _, _, picks = prepared
    first, first_diagnostics = uncertainty.sample_realizations(
        picks, n_realizations=200, seed=7
    )
    second, _ = uncertainty.sample_realizations(picks, n_realizations=200, seed=7)
    pd.testing.assert_frame_equal(first, second)

    event_columns = [
        f"MIS6_DO_{number:02d}_age_ka_bp"
        for number in range(1, uncertainty.N_EVENTS + 1)
    ]
    assert np.all(np.diff(first[event_columns].to_numpy(float), axis=1) > 0)
    assert first["realization_id"].tolist() == list(range(1, 201))
    assert first.columns.tolist() == ["realization_id", *event_columns]
    diagnostics = first_diagnostics
    assert diagnostics["accepted_realizations"] == 200
    assert diagnostics["total_proposals"] == (
        diagnostics["accepted_realizations"] + diagnostics["rejected_proposals"]
    )


def test_all_summary_quantiles_recompute_from_draws(prepared):
    _, _, picks = prepared
    draws, _ = uncertainty.sample_realizations(picks, n_realizations=300, seed=11)
    summary = uncertainty.build_summary(picks, draws)
    assert summary.columns.tolist() == [
        "composite_event_id",
        "composite_event_number",
        "composite_event_label",
        "source_record",
        "source_event_label",
        "nominal_event_age_ka_bp",
        "definition_age_min_ka_bp",
        "definition_age_max_ka_bp",
        "chronology_sigma_nominal_ka",
        "chronology_sigma_min_ka",
        "chronology_sigma_max_ka",
        "sampled_age_q025_ka_bp",
        "sampled_age_q16_ka_bp",
        "sampled_age_median_ka_bp",
        "sampled_age_q84_ka_bp",
        "sampled_age_q975_ka_bp",
    ]
    quantile_columns = [
        "sampled_age_q025_ka_bp",
        "sampled_age_q16_ka_bp",
        "sampled_age_median_ka_bp",
        "sampled_age_q84_ka_bp",
        "sampled_age_q975_ka_bp",
    ]
    for row in summary.itertuples(index=False):
        values = draws[f"{row.composite_event_id}_age_ka_bp"].to_numpy(float)
        expected = np.quantile(values, [0.025, 0.16, 0.5, 0.84, 0.975])
        observed = np.array([getattr(row, column) for column in quantile_columns])
        np.testing.assert_allclose(observed, expected)


def test_age_realization_plot_data_compare_with_rounded_original(prepared):
    _, _, picks = prepared
    draws, _ = uncertainty.sample_realizations(picks, n_realizations=300, seed=19)
    summary = uncertainty.build_summary(picks, draws)
    original = summary[
        ["composite_event_id", "composite_event_number", "nominal_event_age_ka_bp"]
    ].rename(columns={"nominal_event_age_ka_bp": "event_age_ka_bp"})
    original["event_age_ka_bp"] = original["event_age_ka_bp"].round(3)

    plot_data, correlation = uncertainty.prepare_age_realization_plot_data(
        summary, draws, original
    )
    assert plot_data["composite_event_number"].tolist() == list(
        range(1, uncertainty.N_EVENTS + 1)
    )
    assert correlation.shape == (uncertainty.N_EVENTS, uncertainty.N_EVENTS)
    np.testing.assert_allclose(np.diag(correlation), 1.0)
    first_values = (
        draws["MIS6_DO_01_age_ka_bp"].to_numpy(float)
        - original.loc[0, "event_age_ka_bp"]
    )
    assert plot_data.loc[0, "age_offset_median_ka"] == pytest.approx(
        np.median(first_values)
    )


def test_provenance_records_shared_draws_and_excluded_terms(prepared):
    _, envelope, picks = prepared
    _, sampling = uncertainty.sample_realizations(picks, n_realizations=20, seed=5)
    provenance = uncertainty.build_provenance_table(envelope, sampling)
    assert provenance.columns.tolist() == ["scope", "parameter", "value", "note"]

    shared = provenance.loc[provenance["parameter"].eq("shared_chronology_draw")]
    assert set(shared["scope"]) == {"MF"}
    local = provenance.loc[provenance["parameter"].eq("chronology_draw"), "value"].iloc[0]
    assert "independent normal errors at dated-depth knots" in local
    assert shared["value"].str.contains("one standard-normal z").all()

    excluded = " ".join(
        provenance.loc[
            provenance["parameter"].eq("excluded_uncertainty_terms"), "value"
        ].astype(str)
    ).lower()
    assert "resolution" in excluded
    assert "synchronization" in excluded
    conversions = " ".join(
        provenance.loc[
            provenance["parameter"].eq("conversion_to_1sigma"), "value"
        ].astype(str)
    ).lower()
    assert "resolution" not in conversions
