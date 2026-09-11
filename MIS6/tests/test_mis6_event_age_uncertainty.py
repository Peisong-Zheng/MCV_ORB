"""Focused tests for the depth-informed MIS 6 uncertainty workflow."""

from __future__ import annotations

from pathlib import Path
import ast
import json
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

@pytest.fixture(scope="module")
def uncertainty():
    """Exercise notebook preparation; use small ensembles in focused tests."""
    mis6_dir = PROJECT_ROOT / "MIS6"
    notebook = json.loads((mis6_dir / "MIS6_event_age_uncertainty.ipynb").read_text())
    namespace = {}
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(mis6_dir)
        patch.syspath_prepend(str(mis6_dir))
        patch.setattr("IPython.display.display", lambda *args, **kwargs: None)
        patch.setattr(plt, "show", lambda: None)
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            code = "".join(cell["source"]).replace("%matplotlib inline\n", "")
            if cell["id"] in {"settings", "inputs", "definition-picks", "chronology-contexts"}:
                exec(code, namespace)
            else:
                # Remaining helpers can be tested without writing production
                # files or running the two full ensembles in every unit test.
                parsed = ast.parse(code)
                definitions = [node for node in parsed.body if isinstance(node, ast.FunctionDef)]
                exec(compile(ast.Module(body=definitions, type_ignores=[]), "<notebook>", "exec"), namespace)
        namespace["cells"] = {c["id"]: "".join(c["source"]) for c in notebook["cells"]
                              if c["cell_type"] == "code"}
        yield SimpleNamespace(**namespace)
        plt.close("all")


@pytest.fixture(scope="module")
def prepared(uncertainty):
    return uncertainty.mf_envelope, uncertainty.picks


def test_legacy_display_table_is_separate_from_uncertainty_inputs(uncertainty):
    controls = pd.read_csv(PROJECT_ROOT / "MIS6/data/curated/mis6_age_control_points.csv")
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
    assert not hasattr(uncertainty, "load_age_controls")


def test_fohlmeister_stack_is_a_workspace_input(uncertainty):
    assert uncertainty.mf_stack.resolve() == (
        PROJECT_ROOT / "MIS6/data/raw/Fohlmeister J et al-2023-data-mf_d18o_stack.txt"
    )
    assert uncertainty.mf_stack.exists()


def test_mf_duplicate_ages_keep_the_outer_envelope(tmp_path, uncertainty):
    source = tmp_path / "bounds.txt"
    pd.DataFrame(
        {
            "age": [1.0, 1.0, 2.0],
            "age_upper": [1.2, 1.3, 2.2],
            "age_lower": [0.8, 0.7, 1.8],
        }
    ).to_csv(source, sep="\t", index=False)
    scope = vars(uncertainty).copy()
    scope["mf_stack"] = source
    # Run the actual input preparation up to the independent Sofular reader.
    exec(uncertainty.cells["inputs"].split("sources =")[0], scope)
    envelope = scope["mf_envelope"]
    duplicate = envelope.loc[envelope["age"].eq(1.0)].iloc[0]
    assert duplicate["safe_lower"] == pytest.approx(0.7)
    assert duplicate["safe_upper"] == pytest.approx(1.3)


def test_real_mf_envelope_counts_gap_and_sigma_conversion(prepared, uncertainty):
    envelope, _ = prepared
    assert envelope.attrs["valid_row_count"] == 4244
    assert envelope.attrs["unique_age_count"] == 4077
    assert envelope.attrs["source_rows_needing_bound_repair"] == 188
    with pytest.raises(ValueError, match="outside a valid segment"):
        uncertainty.mf_chronology_at_age(176.0, envelope)

    result = uncertainty.mf_chronology_at_age(134.976, envelope)
    assert result["chronology_sigma_ka"] == pytest.approx(
        result["chronology_half_width_approx95_ka"] / uncertainty.NORMAL_975
    )


def test_definition_runs_are_coherent_and_mf_bounds_regress(prepared, uncertainty):
    _, picks = prepared
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


def test_new_sofular_6p9_event_is_stable_and_uses_so4_controls(prepared, uncertainty):
    _, picks = prepared
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


def test_sofular_uses_depth_model_weights_and_no_extrapolation(uncertainty):
    age = 190.972
    result = uncertainty.picks.query(
        "composite_event_id == 'MIS6_DO_20' and algorithm_config_id == 's050_w400'"
    ).iloc[0]
    context = uncertainty.sofular.build_context([age])
    expected_variance = np.sum(
        (context.interpolation_weights[0] * context.control_sigmas_ka) ** 2
    )
    assert result["chronology_sigma_ka"] ** 2 == pytest.approx(expected_variance)
    with pytest.raises(ValueError):
        uncertainty.sofular.build_context([206.0])


def test_selected_catalogue_is_mf_then_five_sofular_events(prepared, uncertainty):
    _, picks = prepared
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


def test_proposal_retains_mf_factor_and_adds_local_sofular_errors(prepared, uncertainty):
    _, picks = prepared
    states, configs = uncertainty.build_state_lookup(picks, sources=uncertainty.sources)
    # Hold detector settings fixed to isolate the sampled chronology offsets.
    configs = {record: ("s050_w400",) for record in uncertainty.RECORD_ORDER}
    draws, _ = uncertainty.sample_realizations(states, configs, 100, 42)
    ages = draws.iloc[:, 1:].to_numpy()
    standardized = {}
    for record in uncertainty.RECORD_ORDER:
        state = states[(record, "s050_w400")]
        standardized[record] = (ages[:, state["event_indices"]] - state["pick_ages"]) / state["chronology_sigmas"]
    assert np.max(np.ptp(standardized["MF"], axis=1)) < 1e-10
    assert np.median(np.ptp(standardized["Sofular"], axis=1)) > 0.1


def test_crossed_sequences_are_rejected_without_sorting(uncertainty, monkeypatch):
    # Force all Sofular proposals to precede the MF events. Sorting or clipping
    # would accept these proposals, whereas the sampler must exhaust its limit.
    monkeypatch.setattr(uncertainty.sofular, "propose_offsets",
                        lambda context, rng: (np.full(len(context.event_ages_ka), -100.0), None))
    with pytest.raises(RuntimeError, match="Only 0 accepted sequences"):
        uncertainty.sample_realizations(uncertainty.states_so4, uncertainty.configs, 1, 9)


def test_so57_overlap_uses_new_young_controls_and_so4_oldest(prepared, uncertainty):
    _, picks = prepared
    states, _ = uncertainty.build_state_lookup(picks, sources=uncertainty.sources, sofular_component="So-57")
    state = states[("Sofular", "s050_w400")]
    components = {context.component: indices.tolist() for indices, context in state["chronology_contexts"]}
    assert components == {"So-4": [4], "So-57": [0, 1, 2, 3]}
    draws, stats = uncertainty.sample_realizations(
        uncertainty.states_so57, uncertainty.configs, 100, 13, sofular_component="So-57")
    assert draws.shape == (100, 22)
    assert stats["rejected_proposals"] == stats["rejected_nonmonotone_chronologies"] + stats["rejected_event_order"]


def test_sampling_is_reproducible_ordered_and_unsorted(prepared, uncertainty):
    _, picks = prepared
    first, first_diagnostics = uncertainty.sample_realizations(
        uncertainty.states_so4, uncertainty.configs, n_realizations=200, seed=7
    )
    second, _ = uncertainty.sample_realizations(
        uncertainty.states_so4, uncertainty.configs, n_realizations=200, seed=7)
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


def test_all_summary_quantiles_recompute_from_draws(prepared, uncertainty):
    _, picks = prepared
    draws, _ = uncertainty.sample_realizations(
        uncertainty.states_so4, uncertainty.configs, n_realizations=300, seed=11)
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


def test_age_plot_uses_unrounded_offsets_and_original_labels(prepared, uncertainty):
    _, picks = prepared
    draws, _ = uncertainty.sample_realizations(
        uncertainty.states_so4, uncertainty.configs, n_realizations=300, seed=19)
    summary = uncertainty.build_summary(picks, draws)
    scope = vars(uncertainty).copy()
    scope.update(summary=summary, draws=draws)
    exec(uncertainty.cells["age-plot"], scope)
    plot_data, correlation = scope["plot_data"], scope["correlation"]
    assert plot_data.composite_event_number.tolist() == list(range(1, 22))
    np.testing.assert_allclose(np.diag(correlation), 1)
    anomalies = draws.iloc[:, 1:].to_numpy() - uncertainty.original_events.event_age_ka_bp.to_numpy()
    np.testing.assert_array_equal(correlation, np.corrcoef(anomalies, rowvar=False))
    np.testing.assert_allclose(plot_data.age_offset_median_ka, np.median(anomalies, axis=0))
    plt.close(scope["fig"])


def test_export_retains_joint_draws_compact_ranges_and_settings(prepared, tmp_path, uncertainty):
    _, picks = prepared
    scope = vars(uncertainty).copy()
    scope["draws"], scope["sampling"] = uncertainty.sample_realizations(
        uncertainty.states_so4, uncertainty.configs, 80, 27)
    scope["alternative"], scope["alternative_sampling"] = uncertainty.sample_realizations(
        uncertainty.states_so57, uncertainty.configs, 80, 27, sofular_component="So-57")
    for cell in ("summaries", "age-plot"):
        exec(uncertainty.cells[cell], scope)
    scope.update(output_dir=tmp_path / "processed", diagnostic_dir=tmp_path / "qc",
                 figure_dir=tmp_path / "figures")
    exec(uncertainty.cells["export"], scope)
    assert len(list(scope["output_dir"].glob("*.csv"))) == 6
    assert len(list(scope["diagnostic_dir"].glob("*.csv"))) == 7
    for suffix, ensemble in (("", scope["draws"]), ("_so57_overlap", scope["alternative"])):
        name = f"mis6_event_age_uncertainty_summary{suffix}.csv"
        compact = pd.read_csv(scope["output_dir"] / name)
        detailed = pd.read_csv(scope["diagnostic_dir"] / name)
        assert compact.shape == (21, 10) and detailed.shape == (21, 16)
        pd.testing.assert_frame_equal(compact, detailed[compact.columns])
        saved = pd.read_csv(scope["output_dir"] / f"mis6_event_age_realizations{suffix}.csv")
        pd.testing.assert_frame_equal(saved, ensemble.round(6))
        expected = np.quantile(ensemble.iloc[:, 1:], [0.025, 0.5, 0.975], axis=0).T
        np.testing.assert_allclose(compact[["sampled_age_q025_ka_bp", "sampled_age_median_ka_bp",
                                           "sampled_age_q975_ka_bp"]], expected, atol=5.1e-7, rtol=0)
        parameters = pd.read_csv(scope["output_dir"] / f"parameters_and_provenance{suffix}.csv")
        assert parameters.columns.tolist() == ["scope", "parameter", "value", "note"]
        settings = parameters.set_index("parameter")["value"]
        assert int(settings["random_seed"]) == 27 and int(settings["accepted_realizations"]) == 80
        assert "synchronization" in settings["excluded_uncertainty_terms"]
        assert "MIS 6.21 always uses So-4" in " ".join(parameters.note.dropna())
    assert {p.suffix for p in scope["figure_dir"].iterdir()} == {".png", ".pdf"}
    plt.close(scope["fig"])
