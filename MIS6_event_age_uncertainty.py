#!/usr/bin/env python3
"""Propagate a simple A+ age uncertainty for the MIS 6 event sequence.

Each proposal draws one chronology state and one event-picking configuration
per source record. Chronology uncertainty is evaluated at the resulting pick.
Proposals that reverse the fixed 21-event order are rejected, never re-sorted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import MIS6_composite_event_record as detector


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "data" / "processed" / "MIS6_event_age_uncertainty"
CONTROL_OUTPUT = OUTPUT_DIR / "mis6_age_control_points.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "mis6_event_age_uncertainty_summary.csv"
REALIZATION_OUTPUT = OUTPUT_DIR / "mis6_event_age_realizations.csv"
DEFINITION_PICK_OUTPUT = OUTPUT_DIR / "mis6_event_definition_picks.csv"
PARAMETER_OUTPUT = OUTPUT_DIR / "record_uncertainty_parameters.csv"
DIAGNOSTIC_OUTPUT = OUTPUT_DIR / "sampling_diagnostics.csv"
FIGURE_DIR = ROOT / "figures" / "MIS6_event_age_uncertainty"
FIGURE_STEM = "MIS6_event_age_uncertainty"
PNG_DPI = 600

FOHLMEISTER_STACK_FILENAME = "Fohlmeister J et al-2023-data-mf_d18o_stack.txt"
FOHLMEISTER_STACK_CANDIDATES = (
    ROOT / "data" / "raw" / FOHLMEISTER_STACK_FILENAME,
    Path("/Users/pz/Nutstore Files/论文-同步-Dell-XPS13-2022-11-03")
    / FOHLMEISTER_STACK_FILENAME,
)

N_REALIZATIONS = 10_000
RANDOM_SEED = 20260904
NORMAL_975 = 1.95996398454
MIN_EVENT_SEPARATION_KA = 0.0
SOFULAR_SIGMA_KA = 0.375
RECORD_ORDER = ("MF", "Huagapo", "Sofular")


@dataclass(frozen=True)
class PickState:
    """All event picks and local chronology sigmas for one record setting."""

    event_indices: np.ndarray
    pick_ages: np.ndarray
    chronology_sigmas: np.ndarray


def build_age_control_points() -> pd.DataFrame:
    """Return only the U-Th controls needed for the selected late events.

    Huagapo controls enter the A+ calculation directly. Sofular controls are
    retained as the evidence behind a transparent local-error approximation;
    they are not misrepresented as controls on the published iscam stack.
    """
    common_hua = {
        "record_id": "Huagapo",
        "cave": "Huagapo Cave",
        "stalagmite_id": "P10-H2",
        "dating_sample_id": "not listed separately",
        "uncertainty_level": "2sigma",
        "age_model_method": "linear interpolation between U-Th dates",
        "a_plus_use": "direct interpolation of the reported 2sigma error",
        "source_study": "Burns et al. (2019)",
        "source_filename": "41598_2018_37854_MOESM1_ESM.pdf",
        "source_locator": "Supplementary Table S1, p. 3",
        "source_unit_note": "age and error converted from yr BP to Kyr BP",
        "shown_in_composite_figure": True,
    }
    common_sof = {
        "record_id": "Sofular",
        "cave": "Sofular Cave",
        "uncertainty_level": "2sigma",
        "age_model_method": (
            "U-Th controls for individual StalAge models; composite stack "
            "developed with iscam"
        ),
        "a_plus_use": (
            "supports this study's fixed local 1sigma approximation of 0.375 Kyr; "
            "not a published stack uncertainty"
        ),
        "source_study": "Held et al. (2024)",
        "source_filename": "41467_2024_45507_MOESM3_ESM.xlsx",
        "source_locator": "assigned below by stalagmite; row 142 states 2sigma",
    }

    rows = [
        # Huagapo brackets for MIS 6.17, 6.20, and 6.21.
        {
            **common_hua,
            "control_id": "HUA_P10H2_D135",
            "depth_mm": 135.0,
            "age_ka_bp": 177.519,
            "age_error_2sigma_ka": 0.567,
            "used_for_event_ids": "MIS6_DO_17",
            "bracket_role": "younger",
        },
        {
            **common_hua,
            "control_id": "HUA_P10H2_D180",
            "depth_mm": 180.0,
            "age_ka_bp": 179.429,
            "age_error_2sigma_ka": 0.819,
            "used_for_event_ids": "MIS6_DO_17",
            "bracket_role": "older",
        },
        {
            **common_hua,
            "control_id": "HUA_P10H2_D503",
            "depth_mm": 503.0,
            "age_ka_bp": 190.419,
            "age_error_2sigma_ka": 0.575,
            "used_for_event_ids": "MIS6_DO_20",
            "bracket_role": "younger",
        },
        {
            **common_hua,
            "control_id": "HUA_P10H2_D595",
            "depth_mm": 595.0,
            "age_ka_bp": 190.909,
            "age_error_2sigma_ka": 0.667,
            "used_for_event_ids": "MIS6_DO_20",
            "bracket_role": "older",
        },
        {
            **common_hua,
            "control_id": "HUA_P10H2_D902",
            "depth_mm": 902.0,
            "age_ka_bp": 194.262,
            "age_error_2sigma_ka": 0.674,
            "used_for_event_ids": "MIS6_DO_21",
            "bracket_role": "younger",
        },
        {
            **common_hua,
            "control_id": "HUA_P10H2_D1005",
            "depth_mm": 1005.0,
            "age_ka_bp": 195.931,
            "age_error_2sigma_ka": 0.783,
            "used_for_event_ids": "MIS6_DO_21",
            "bracket_role": "older",
        },
        # Both Sofular component records bracket the two selected stack events.
        {
            **common_sof,
            "control_id": "SOF_SO4_M9",
            "stalagmite_id": "So-4",
            "dating_sample_id": "So4-M9",
            "depth_mm": 967.0,
            "age_ka_bp": 176.102399,
            "age_error_2sigma_ka": 0.702634,
            "used_for_event_ids": "MIS6_DO_18",
            "bracket_role": "younger",
            "source_unit_note": "reported corrected age and 2sigma error in Kyr",
            "shown_in_composite_figure": False,
        },
        {
            **common_sof,
            "control_id": "SOF_SO4_M10",
            "stalagmite_id": "So-4",
            "dating_sample_id": "So4-M10",
            "depth_mm": 1014.0,
            "age_ka_bp": 179.935830,
            "age_error_2sigma_ka": 0.670912,
            "used_for_event_ids": "MIS6_DO_18;MIS6_DO_19",
            "bracket_role": "older for 18; younger for 19",
            "source_unit_note": "reported corrected age and 2sigma error in Kyr",
            "shown_in_composite_figure": True,
        },
        {
            **common_sof,
            "control_id": "SOF_SO4_M23",
            "stalagmite_id": "So-4",
            "dating_sample_id": "So4-M23",
            "depth_mm": 1045.0,
            "age_ka_bp": 180.513866,
            "age_error_2sigma_ka": 0.775462,
            "used_for_event_ids": "MIS6_DO_19",
            "bracket_role": "older",
            "source_unit_note": "reported corrected age and 2sigma error in Kyr",
            "shown_in_composite_figure": True,
        },
        {
            **common_sof,
            "control_id": "SOF_SO57_3",
            "stalagmite_id": "So-57",
            "dating_sample_id": "So57-3",
            "depth_mm": 377.5,
            "age_ka_bp": 177.120951,
            "age_error_2sigma_ka": 0.698652,
            "used_for_event_ids": "MIS6_DO_18",
            "bracket_role": "younger",
            "source_unit_note": "source cells converted from yr to Kyr",
            "shown_in_composite_figure": False,
        },
        {
            **common_sof,
            "control_id": "SOF_SO57_4",
            "stalagmite_id": "So-57",
            "dating_sample_id": "So57-4",
            "depth_mm": 485.0,
            "age_ka_bp": 180.207207,
            "age_error_2sigma_ka": 0.600960,
            "used_for_event_ids": "MIS6_DO_18;MIS6_DO_19",
            "bracket_role": "older for 18; younger for 19",
            "source_unit_note": "source cells converted from yr to Kyr",
            "shown_in_composite_figure": True,
        },
        {
            **common_sof,
            "control_id": "SOF_SO57_5",
            "stalagmite_id": "So-57",
            "dating_sample_id": "So57-5",
            "depth_mm": 602.0,
            "age_ka_bp": 183.192722,
            "age_error_2sigma_ka": 0.722279,
            "used_for_event_ids": "MIS6_DO_19",
            "bracket_role": "older",
            "source_unit_note": "source cells converted from yr to Kyr",
            "shown_in_composite_figure": False,
        },
    ]
    columns = [
        "control_id",
        "record_id",
        "cave",
        "stalagmite_id",
        "dating_sample_id",
        "depth_mm",
        "age_ka_bp",
        "age_error_2sigma_ka",
        "uncertainty_level",
        "age_model_method",
        "a_plus_use",
        "used_for_event_ids",
        "bracket_role",
        "source_study",
        "source_filename",
        "source_locator",
        "source_unit_note",
        "shown_in_composite_figure",
    ]
    table = pd.DataFrame(rows)[columns]
    table.loc[table["stalagmite_id"].eq("So-4"), "source_locator"] = (
        "U_Th_Dating!O95:P97; row 142 states 2sigma"
    )
    table.loc[table["stalagmite_id"].eq("So-57"), "source_locator"] = (
        "U_Th_Dating!O135:P137; row 142 states 2sigma"
    )
    return table.sort_values(["record_id", "stalagmite_id", "age_ka_bp"]).reset_index(
        drop=True
    )


def resolve_fohlmeister_stack() -> Path:
    """Use a workspace copy when present, otherwise the user-supplied source."""
    for path in FOHLMEISTER_STACK_CANDIDATES:
        if path.exists():
            return path
    locations = "\n".join(str(path) for path in FOHLMEISTER_STACK_CANDIDATES)
    raise FileNotFoundError(
        f"Cannot find {FOHLMEISTER_STACK_FILENAME} in:\n{locations}"
    )


def load_mf_age_envelope(path: Path) -> pd.DataFrame:
    """Load a gap-aware, duplicate-safe MF chronology envelope."""
    raw = pd.read_csv(path, sep="\t", comment="#")
    required = ["age", "age_upper", "age_lower"]
    if missing := set(required).difference(raw.columns):
        raise ValueError(f"MF age-envelope file is missing columns: {sorted(missing)}")

    frame = raw[required].apply(pd.to_numeric, errors="coerce").dropna()
    frame["source_lower"] = frame[["age_lower", "age_upper"]].min(axis=1)
    frame["source_upper"] = frame[["age_lower", "age_upper"]].max(axis=1)
    frame["source_bound_needs_repair"] = ~frame["age"].between(
        frame["source_lower"], frame["source_upper"]
    )
    frame["safe_lower"] = frame[["age", "source_lower"]].min(axis=1)
    frame["safe_upper"] = frame[["age", "source_upper"]].max(axis=1)
    valid_row_count = len(frame)
    repair_row_count = int(frame["source_bound_needs_repair"].sum())
    # Duplicate nominal ages occur in the stack: preserve their outer envelope.
    frame = (
        frame.groupby("age", as_index=False)
        .agg(
            source_lower=("source_lower", "min"),
            source_upper=("source_upper", "max"),
            safe_lower=("safe_lower", "min"),
            safe_upper=("safe_upper", "max"),
        )
        .sort_values("age")
        .reset_index(drop=True)
    )
    frame["segment"] = frame["age"].diff().gt(detector.MATERIAL_GAP_KA).cumsum()
    frame.attrs.update(
        valid_row_count=valid_row_count,
        unique_age_count=len(frame),
        source_rows_needing_bound_repair=repair_row_count,
    )
    return frame


def mf_chronology_at_age(age: float, envelope: pd.DataFrame) -> dict[str, object]:
    """Interpolate MF bounds only inside one continuous chronology segment."""
    matches = [
        (segment_id, part)
        for segment_id, part in envelope.groupby("segment", sort=True)
        if part["age"].iloc[0] <= age <= part["age"].iloc[-1]
    ]
    if len(matches) != 1:
        raise ValueError(f"MF pick {age:.6f} Kyr BP lies outside a valid segment")
    segment_id, part = matches[0]
    source_lower = float(np.interp(age, part["age"], part["source_lower"]))
    source_upper = float(np.interp(age, part["age"], part["source_upper"]))
    safe_lower = float(np.interp(age, part["age"], part["safe_lower"]))
    safe_upper = float(np.interp(age, part["age"], part["safe_upper"]))
    safe_lower, safe_upper = min(age, safe_lower), max(age, safe_upper)
    half_width = max(age - safe_lower, safe_upper - age)
    return {
        "chronology_input_lower_approx95_ka_bp": safe_lower,
        "chronology_input_upper_approx95_ka_bp": safe_upper,
        "chronology_sampling_lower_approx95_ka_bp": age - half_width,
        "chronology_sampling_upper_approx95_ka_bp": age + half_width,
        "chronology_half_width_approx95_ka": half_width,
        "chronology_sigma_ka": half_width / NORMAL_975,
        "chronology_support_id": f"MF_segment_{int(segment_id)}",
        "source_bound_repair_applied": not (source_lower <= age <= source_upper),
    }


def interpolated_huagapo_error(age: float, controls: pd.DataFrame) -> dict[str, object]:
    """Interpolate the reported P10-H2 2-sigma error without extrapolation."""
    points = controls.loc[controls["record_id"].eq("Huagapo")].sort_values("age_ka_bp")
    ages = points["age_ka_bp"].to_numpy(float)
    errors = points["age_error_2sigma_ka"].to_numpy(float)
    if not ages[0] <= age <= ages[-1]:
        raise ValueError(f"Huagapo pick {age:.6f} lies outside its U-Th controls")

    older = min(int(np.searchsorted(ages, age, side="right")), len(ages) - 1)
    younger = max(0, older - 1)
    if younger == older:  # Only possible at the youngest endpoint.
        older += 1
    weight = (age - ages[younger]) / (ages[older] - ages[younger])
    half_width_2sigma = (1 - weight) * errors[younger] + weight * errors[older]
    return {
        "chronology_input_lower_approx95_ka_bp": age - half_width_2sigma,
        "chronology_input_upper_approx95_ka_bp": age + half_width_2sigma,
        "chronology_sampling_lower_approx95_ka_bp": age - half_width_2sigma,
        "chronology_sampling_upper_approx95_ka_bp": age + half_width_2sigma,
        "chronology_half_width_approx95_ka": half_width_2sigma,
        "chronology_sigma_ka": half_width_2sigma / 2,
        "chronology_support_id": (
            f"{points.iloc[younger]['control_id']}--{points.iloc[older]['control_id']}"
        ),
        "source_bound_repair_applied": False,
    }


def chronology_at_age(
    record_id: str,
    age: float,
    controls: pd.DataFrame,
    mf_envelope: pd.DataFrame,
) -> dict[str, object]:
    """Return the local A+ chronology approximation at an algorithmic pick."""
    if record_id == "MF":
        return mf_chronology_at_age(age, mf_envelope)
    if record_id == "Huagapo":
        return interpolated_huagapo_error(age, controls)
    if record_id == "Sofular":
        half_width = 2 * SOFULAR_SIGMA_KA
        return {
            "chronology_input_lower_approx95_ka_bp": age - half_width,
            "chronology_input_upper_approx95_ka_bp": age + half_width,
            "chronology_sampling_lower_approx95_ka_bp": age - half_width,
            "chronology_sampling_upper_approx95_ka_bp": age + half_width,
            "chronology_half_width_approx95_ka": half_width,
            "chronology_sigma_ka": SOFULAR_SIGMA_KA,
            "chronology_support_id": "Sofular_local_fixed_approximation",
            "source_bound_repair_applied": False,
        }
    raise ValueError(f"Unknown record: {record_id}")


def build_definition_picks(
    anchors: pd.DataFrame,
    segments_by_record: dict[str, list[detector.RegularSegment]],
    controls: pd.DataFrame,
    mf_envelope: pd.DataFrame,
) -> pd.DataFrame:
    """Run all 3 x 3 detector settings, keeping their original equal weights."""
    rows: list[dict[str, object]] = []
    for spec in detector.RECORDS:
        local = anchors.loc[anchors["record_id"].eq(spec.record_id)].copy()
        anchor_ages = local["anchor_age_ka_bp"].sort_values().to_numpy(float)
        for sigma in spec.sensitivity_sigmas_ka:
            for half_width in detector.SENSITIVITY_SEARCH_HALF_WIDTHS_KA:
                config_id = f"s{sigma * 1000:03.0f}_w{half_width * 1000:03.0f}"
                for _, anchor in local.iterrows():
                    anchor_age = float(anchor["anchor_age_ka_bp"])
                    segment = detector.segment_containing(
                        segments_by_record[spec.record_id], anchor_age
                    )
                    bounds = detector.search_bounds(
                        anchor_age, anchor_ages, segment, half_width
                    )
                    _, pick_age = detector.pick_gradient_peak(
                        spec, segment, sigma, bounds
                    )
                    chronology = chronology_at_age(
                        spec.record_id, pick_age, controls, mf_envelope
                    )
                    number = int(
                        str(anchor["composite_event_display_label"]).split(".")[-1]
                    )
                    rows.append(
                        {
                            "composite_event_id": f"MIS6_DO_{number:02d}",
                            "composite_event_number": number,
                            "composite_event_label": f"MIS {anchor['composite_event_display_label']}",
                            "source_record": spec.record_id,
                            "source_event_label": anchor["source_event_display_label"],
                            "algorithm_config_id": config_id,
                            "smoothing_sigma_yr": sigma * 1000,
                            "search_half_width_yr": half_width * 1000,
                            "definition_pick_age_ka_bp": pick_age,
                            **chronology,
                        }
                    )
    picks = (
        pd.DataFrame(rows)
        .sort_values(["source_record", "algorithm_config_id", "composite_event_number"])
        .reset_index(drop=True)
    )
    validate_definition_picks(picks)
    return picks


def validate_definition_picks(picks: pd.DataFrame) -> None:
    """Check that each record/configuration represents one coherent detector run."""
    expected_counts = {"MF": 16, "Huagapo": 3, "Sofular": 2}
    if set(picks["source_record"]) != set(expected_counts):
        raise ValueError("Unexpected source records in definition picks")
    for record_id, expected in expected_counts.items():
        part = picks.loc[picks["source_record"].eq(record_id)]
        if part["algorithm_config_id"].nunique() != 9:
            raise ValueError(f"{record_id} does not have nine detector settings")
        if not part.groupby("algorithm_config_id").size().eq(expected).all():
            raise ValueError(f"{record_id} has an incomplete detector setting")
        for _, group in part.groupby("algorithm_config_id"):
            ordered = group.sort_values("composite_event_number")
            if not np.all(np.diff(ordered["definition_pick_age_ka_bp"]) > 0):
                raise ValueError(f"Definition picks cross within {record_id}")
    if (
        not np.isfinite(picks[["definition_pick_age_ka_bp", "chronology_sigma_ka"]])
        .all()
        .all()
        or not picks["chronology_sigma_ka"].gt(0).all()
    ):
        raise ValueError("Definition picks contain invalid chronology values")


def build_state_lookup(
    picks: pd.DataFrame,
) -> tuple[dict[tuple[str, str], PickState], dict[str, tuple[str, ...]]]:
    states: dict[tuple[str, str], PickState] = {}
    configs: dict[str, tuple[str, ...]] = {}
    for record_id in RECORD_ORDER:
        local = picks.loc[picks["source_record"].eq(record_id)]
        configs[record_id] = tuple(sorted(local["algorithm_config_id"].unique()))
        for config_id, group in local.groupby("algorithm_config_id"):
            group = group.sort_values("composite_event_number")
            states[(record_id, str(config_id))] = PickState(
                event_indices=group["composite_event_number"].to_numpy(int) - 1,
                pick_ages=group["definition_pick_age_ka_bp"].to_numpy(float),
                chronology_sigmas=group["chronology_sigma_ka"].to_numpy(float),
            )
    return states, configs


def propose_realization(
    rng: np.random.Generator,
    states: dict[tuple[str, str], PickState],
    configs: dict[str, tuple[str, ...]],
) -> dict[str, object]:
    """Draw chronology first, then one shared detector setting per record."""
    record_z = {record: float(rng.normal()) for record in RECORD_ORDER}
    selected = {record: str(rng.choice(configs[record])) for record in RECORD_ORDER}
    definition = np.full(21, np.nan)
    sigma = np.full(21, np.nan)
    sampled = np.full(21, np.nan)
    for record in RECORD_ORDER:
        state = states[(record, selected[record])]
        definition[state.event_indices] = state.pick_ages
        sigma[state.event_indices] = state.chronology_sigmas
        sampled[state.event_indices] = (
            state.pick_ages + record_z[record] * state.chronology_sigmas
        )
    return {
        "record_z": record_z,
        "selected_configs": selected,
        "definition_ages": definition,
        "chronology_sigmas": sigma,
        "sampled_ages": sampled,
    }


def is_strictly_ordered(ages: np.ndarray) -> bool:
    """Keep fixed event identities; do not sort a chronology proposal."""
    return bool(
        np.isfinite(ages).all() and np.all(np.diff(ages) > MIN_EVENT_SEPARATION_KA)
    )


def sample_realizations(
    picks: pd.DataFrame,
    n_realizations: int = N_REALIZATIONS,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rejection-sample complete, ordered 21-event sequences."""
    if n_realizations < 1:
        raise ValueError("n_realizations must be positive")
    states, configs = build_state_lookup(picks)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    proposal_id = 0
    max_proposals = max(1000, 100 * n_realizations)

    while len(rows) < n_realizations and proposal_id < max_proposals:
        proposal_id += 1
        proposal = propose_realization(rng, states, configs)
        ages = proposal["sampled_ages"]
        if not is_strictly_ordered(ages):
            continue
        z = proposal["record_z"]
        selected = proposal["selected_configs"]
        row: dict[str, object] = {
            "realization_id": len(rows) + 1,
            "proposal_id": proposal_id,
        }
        for record in RECORD_ORDER:
            row[f"{record}_algorithm_config_id"] = selected[record]
            row[f"{record}_chronology_z"] = z[record]
        row.update(
            {
                f"MIS6_DO_{index + 1:02d}_age_ka_bp": age
                for index, age in enumerate(ages)
            }
        )
        rows.append(row)

    if len(rows) != n_realizations:
        raise RuntimeError(
            f"Accepted only {len(rows)} ordered sequences after {proposal_id} proposals"
        )
    rejected = proposal_id - n_realizations
    diagnostics = pd.DataFrame(
        [
            {
                "random_seed": seed,
                "requested_realizations": n_realizations,
                "accepted_realizations": len(rows),
                "total_proposals": proposal_id,
                "rejected_proposals": rejected,
                "rejection_fraction": rejected / proposal_id,
                "minimum_event_separation_ka": MIN_EVENT_SEPARATION_KA,
                "order_rule": "fixed event rank; reject whole proposal if crossed",
                "sorting_used": False,
                "max_proposals": max_proposals,
            }
        ]
    )
    return pd.DataFrame(rows), diagnostics


def build_summary(picks: pd.DataFrame, draws: pd.DataFrame) -> pd.DataFrame:
    """Summarize the non-Gaussian mixture with quantiles as primary results."""
    rows = []
    for event_id, group in picks.groupby("composite_event_id", sort=False):
        group = group.sort_values("algorithm_config_id")
        first = group.iloc[0]
        spec = detector.RECORD_BY_ID[str(first["source_record"])]
        nominal_id = (
            f"s{spec.nominal_sigma_ka * 1000:03.0f}_"
            f"w{detector.NOMINAL_SEARCH_HALF_WIDTH_KA * 1000:03.0f}"
        )
        nominal = group.loc[group["algorithm_config_id"].eq(nominal_id)].iloc[0]
        values = draws[f"{event_id}_age_ka_bp"].to_numpy(float)
        quantiles = np.quantile(values, [0.025, 0.16, 0.5, 0.84, 0.975])
        source_record = str(first["source_record"])
        chronology_method = {
            "MF": "shared-z; symmetric envelope from inferred 95% age-model bounds",
            "Huagapo": "shared-z; linearly interpolated P10-H2 2sigma errors",
            "Sofular": "shared-z; fixed local approximation",
        }[source_record]
        source_filename = {
            "MF": FOHLMEISTER_STACK_FILENAME,
            "Huagapo": "41598_2018_37854_MOESM1_ESM.pdf",
            "Sofular": "41467_2024_45507_MOESM3_ESM.xlsx",
        }[source_record]
        rows.append(
            {
                "composite_event_id": event_id,
                "composite_event_number": int(first["composite_event_number"]),
                "composite_event_label": first["composite_event_label"],
                "source_record": source_record,
                "source_event_label": first["source_event_label"],
                "nominal_event_age_ka_bp": nominal["definition_pick_age_ka_bp"],
                "definition_distribution": (
                    "nine equally weighted proposal configurations; accepted "
                    "distribution conditional on fixed event order"
                ),
                "definition_interpretation": (
                    "method-sensitivity proxy; not a posterior or confidence interval"
                ),
                "definition_config_count": len(group),
                "definition_unique_pick_count": group[
                    "definition_pick_age_ka_bp"
                ].nunique(),
                "definition_picks_by_config_ka_bp": ";".join(
                    f"{row.algorithm_config_id}:{row.definition_pick_age_ka_bp:.6f}"
                    for row in group.itertuples()
                ),
                "definition_age_min_ka_bp": group["definition_pick_age_ka_bp"].min(),
                "definition_age_max_ka_bp": group["definition_pick_age_ka_bp"].max(),
                "chronology_method": chronology_method,
                "chronology_source_filename": source_filename,
                "chronology_support_at_nominal": nominal["chronology_support_id"],
                "chronology_input_lower_at_nominal_approx95_ka_bp": nominal[
                    "chronology_input_lower_approx95_ka_bp"
                ],
                "chronology_input_upper_at_nominal_approx95_ka_bp": nominal[
                    "chronology_input_upper_approx95_ka_bp"
                ],
                "chronology_sampling_lower_at_nominal_approx95_ka_bp": nominal[
                    "chronology_sampling_lower_approx95_ka_bp"
                ],
                "chronology_sampling_upper_at_nominal_approx95_ka_bp": nominal[
                    "chronology_sampling_upper_approx95_ka_bp"
                ],
                "chronology_half_width_at_nominal_approx95_ka": nominal[
                    "chronology_half_width_approx95_ka"
                ],
                "chronology_sigma_at_nominal_ka": nominal["chronology_sigma_ka"],
                "source_bound_repair_at_nominal": nominal[
                    "source_bound_repair_applied"
                ],
                "chronology_sigma_min_over_definition_support_ka": group[
                    "chronology_sigma_ka"
                ].min(),
                "chronology_sigma_max_over_definition_support_ka": group[
                    "chronology_sigma_ka"
                ].max(),
                "shared_z_group": source_record,
                "n_accepted_realizations": len(draws),
                "sampled_age_mean_ka_bp": values.mean(),
                "sampled_age_sd_ka": values.std(ddof=1),
                "sampled_age_q025_ka_bp": quantiles[0],
                "sampled_age_q16_ka_bp": quantiles[1],
                "sampled_age_median_ka_bp": quantiles[2],
                "sampled_age_q84_ka_bp": quantiles[3],
                "sampled_age_q975_ka_bp": quantiles[4],
            }
        )
    return (
        pd.DataFrame(rows).sort_values("composite_event_number").reset_index(drop=True)
    )


def prepare_age_realization_plot_data(
    summary: pd.DataFrame,
    draws: pd.DataFrame,
    original_events: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Summarize MC offsets and their joint movement around the original picks."""
    summary = summary.sort_values("composite_event_number").reset_index(drop=True)
    original = original_events.sort_values("composite_event_number").reset_index(
        drop=True
    )
    expected_ids = summary["composite_event_id"].tolist()
    if expected_ids != original["composite_event_id"].tolist():
        raise ValueError("Original and uncertainty tables contain different events")
    if expected_ids != [f"MIS6_DO_{number:02d}" for number in range(1, 22)]:
        raise ValueError("The age-realization figure requires ordered events 1--21")

    original_ages = original["event_age_ka_bp"].to_numpy(float)
    nominal_ages = summary["nominal_event_age_ka_bp"].to_numpy(float)
    # The composite CSV deliberately rounds exported event ages to 0.001 Kyr.
    if not np.allclose(original_ages, nominal_ages, atol=5e-4, rtol=0):
        raise ValueError("Original event ages do not match the nominal A+ picks")

    event_columns = [f"{event_id}_age_ka_bp" for event_id in expected_ids]
    if missing := set(event_columns).difference(draws.columns):
        raise ValueError(f"Age-realization table is missing columns: {sorted(missing)}")
    sampled = draws[event_columns].to_numpy(float)
    anomalies = sampled - original_ages
    if not np.isfinite(anomalies).all() or (np.std(anomalies, axis=0) == 0).any():
        raise ValueError("Event-age anomalies must be finite and variable")

    quantiles = np.quantile(anomalies, [0.025, 0.16, 0.5, 0.84, 0.975], axis=0).T
    plot_data = summary[
        [
            "composite_event_id",
            "composite_event_number",
            "composite_event_label",
            "source_record",
        ]
    ].copy()
    plot_data["original_event_age_ka_bp"] = original_ages
    for index, name in enumerate(("q025", "q16", "median", "q84", "q975")):
        plot_data[f"age_offset_{name}_ka"] = quantiles[:, index]
    correlation = np.corrcoef(anomalies, rowvar=False)
    if correlation.shape != (21, 21) or not np.isfinite(correlation).all():
        raise ValueError("Could not calculate a complete event-age correlation matrix")
    return plot_data, correlation


def plot_age_realizations(
    summary: pd.DataFrame,
    draws: pd.DataFrame,
    original_events: pd.DataFrame,
    output_dir: Path = FIGURE_DIR,
) -> tuple[Path, Path]:
    """Compare all A+ realizations with the original 21-event sequence."""
    plot_data, correlation = prepare_age_realization_plot_data(
        summary, draws, original_events
    )
    detector.configure_plot_style()
    colors = {spec.record_id: spec.color for spec in detector.RECORDS}
    figure, (forest, matrix) = plt.subplots(
        1,
        2,
        figsize=(180 / 25.4, 126 / 25.4),
        gridspec_kw={"width_ratios": (1.18, 1.0)},
    )
    figure.subplots_adjust(left=0.205, right=0.90, bottom=0.175, top=0.91, wspace=0.38)

    y = np.arange(len(plot_data))
    for index, row in plot_data.iterrows():
        color = colors[str(row["source_record"])]
        forest.hlines(
            y[index],
            row["age_offset_q025_ka"],
            row["age_offset_q975_ka"],
            color=color,
            linewidth=0.9,
            alpha=0.75,
            zorder=1,
        )
        forest.hlines(
            y[index],
            row["age_offset_q16_ka"],
            row["age_offset_q84_ka"],
            color=color,
            linewidth=3.2,
            zorder=2,
        )
        forest.plot(
            row["age_offset_median_ka"],
            y[index],
            "o",
            color=color,
            markersize=3.8,
            markeredgecolor="white",
            markeredgewidth=0.35,
            zorder=3,
        )
    forest.axvline(0, color="#333333", linewidth=0.8, linestyle="--", zorder=0)
    forest.set_yticks(y)
    forest.set_yticklabels(
        [
            f"{row.composite_event_label.replace('MIS ', '')}  ({row.original_event_age_ka_bp:.3f})"
            for row in plot_data.itertuples()
        ],
        fontsize=7.2,
    )
    for label, record in zip(forest.get_yticklabels(), plot_data["source_record"]):
        label.set_color(colors[str(record)])
    forest.invert_yaxis()
    forest.set_xlabel("Sampled age − original event age (Kyr)")
    forest.set_ylabel("Event (original age, Kyr BP)")
    forest.set_title("(a) Marginal event-age uncertainty", loc="left")
    forest.grid(axis="x", color="#E4E4E4", linewidth=0.6)
    forest.spines[["top", "right"]].set_visible(False)

    image = matrix.imshow(
        correlation,
        origin="lower",
        vmin=-1,
        vmax=1,
        cmap="RdBu_r",
        interpolation="nearest",
        rasterized=True,
    )
    tick_indices = np.array([0, 4, 9, 14, 16, 18, 20])
    tick_labels = (tick_indices + 1).astype(str)
    matrix.set_xticks(tick_indices, tick_labels)
    matrix.set_yticks(tick_indices, tick_labels)
    for boundary in (15.5, 16.5, 18.5):
        matrix.axhline(boundary, color="white", linewidth=0.65)
        matrix.axvline(boundary, color="white", linewidth=0.65)
    matrix.set_xlabel("Composite event number")
    matrix.set_ylabel("Composite event number")
    matrix.set_title("(b) Joint age-anomaly structure", loc="left")
    colorbar = figure.colorbar(image, ax=matrix, fraction=0.046, pad=0.04)
    colorbar.set_label("Correlation, r")
    colorbar.set_ticks([-1, -0.5, 0, 0.5, 1])

    legend_handles = [
        Line2D([0], [0], color="#444444", linewidth=0.9, label="95% interval"),
        Line2D([0], [0], color="#444444", linewidth=3.2, label="68% interval"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#444444",
            markeredgecolor="white",
            markersize=4.5,
            label="MC median",
        ),
        Line2D(
            [0],
            [0],
            color="#333333",
            linestyle="--",
            linewidth=0.8,
            label="Original sequence",
        ),
        *[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=colors[record],
                markeredgecolor=colors[record],
                markersize=4.5,
                label=record,
            )
            for record in RECORD_ORDER
        ],
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        frameon=False,
        fontsize=7.3,
        handlelength=2.2,
        columnspacing=1.25,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{FIGURE_STEM}.png"
    pdf_path = output_dir / f"{FIGURE_STEM}.pdf"
    figure.savefig(
        pdf_path,
        metadata={
            "Title": "MIS 6 event-age uncertainty realizations",
            "Subject": "A+ Monte Carlo sequence uncertainty",
            "Creator": Path(__file__).name,
        },
    )
    figure.savefig(png_path, dpi=PNG_DPI, metadata={"Software": Path(__file__).name})
    plt.close(figure)
    return png_path, pdf_path


def build_parameter_table(mf_envelope: pd.DataFrame | None = None) -> pd.DataFrame:
    """Make all conversions and deliberate omissions visible outside the code."""
    rows = [
        {
            "record_id": "MF",
            "event_count": 16,
            "chronology_input": "published age_lower and age_upper curves",
            "reported_scale": "approximately 95% age-model envelope",
            "conversion_to_1sigma": "max distance to repaired envelope / 1.95996398454",
            "shared_chronology_draw": "one standard-normal z for all MF events",
            "source_study": "Fohlmeister et al. (2023)",
            "source_filename": FOHLMEISTER_STACK_FILENAME,
            "uncertainty_scale_source_filename": "Fohlmeister J et al-2023-SOM.pdf",
            "notes": "bounds inferred from SOM Fig. S3; raw U-Th error not re-added",
        },
        {
            "record_id": "Huagapo",
            "event_count": 3,
            "chronology_input": "adjacent P10-H2 U-Th control errors",
            "reported_scale": "2sigma",
            "conversion_to_1sigma": "linear interpolation at pick, then divide by 2",
            "shared_chronology_draw": "one standard-normal z for all Huagapo events",
            "source_study": "Burns et al. (2019)",
            "source_filename": "41598_2018_37854_MOESM1_ESM.pdf",
            "uncertainty_scale_source_filename": "41598_2018_37854_MOESM1_ESM.pdf",
            "notes": "control errors are interpolated, not differenced",
        },
        {
            "record_id": "Sofular",
            "event_count": 2,
            "chronology_input": "simple local approximation informed by component dates",
            "reported_scale": "1sigma",
            "conversion_to_1sigma": "fixed 0.375 Kyr",
            "shared_chronology_draw": "one standard-normal z for both Sofular events",
            "source_study": "Held et al. (2024)",
            "source_filename": "41467_2024_45507_MOESM3_ESM.xlsx",
            "uncertainty_scale_source_filename": "41467_2024_45507_MOESM3_ESM.xlsx",
            "notes": "approximation, not a published iscam-stack event confidence interval",
        },
    ]
    table = pd.DataFrame(rows)
    table["definition_sampling"] = (
        "one of nine equally weighted proposal configurations per record; "
        "accepted draws are conditional on fixed event order"
    )
    table["definition_interpretation"] = (
        "method-sensitivity proxy; not a posterior or confidence interval"
    )
    table["excluded_terms"] = (
        "cross-record synchronization; proxy resolution; separate smoothing error"
    )
    if mf_envelope is None:
        table["source_valid_rows"] = ""
        table["source_unique_ages"] = ""
        table["source_rows_needing_bound_repair"] = ""
    else:
        table["source_valid_rows"] = [mf_envelope.attrs["valid_row_count"], "", ""]
        table["source_unique_ages"] = [mf_envelope.attrs["unique_age_count"], "", ""]
        table["source_rows_needing_bound_repair"] = [
            mf_envelope.attrs["source_rows_needing_bound_repair"],
            "",
            "",
        ]
    return table


def validate_outputs(
    controls: pd.DataFrame,
    summary: pd.DataFrame,
    draws: pd.DataFrame,
) -> None:
    """Fail loudly if a saved result would violate the A+ contract."""
    if len(controls) != 12 or controls["control_id"].duplicated().any():
        raise ValueError("The age-control table must contain 12 unique controls")
    if any(Path(name).name != name for name in controls["source_filename"]):
        raise ValueError("Control provenance must contain filenames, not paths")
    if len(summary) != 21 or summary["composite_event_id"].duplicated().any():
        raise ValueError("The uncertainty summary must contain 21 unique events")
    if summary["composite_event_number"].tolist() != list(range(1, 22)):
        raise ValueError("Event identities are not in their fixed 1--21 order")

    event_columns = [f"MIS6_DO_{number:02d}_age_ka_bp" for number in range(1, 22)]
    ages = draws[event_columns].to_numpy(float)
    if not np.isfinite(ages).all() or not np.all(
        np.diff(ages, axis=1) > MIN_EVENT_SEPARATION_KA
    ):
        raise ValueError("A realization contains invalid or crossed event ages")
    quantiles = summary[
        [
            "sampled_age_q025_ka_bp",
            "sampled_age_q16_ka_bp",
            "sampled_age_median_ka_bp",
            "sampled_age_q84_ka_bp",
            "sampled_age_q975_ka_bp",
        ]
    ].to_numpy(float)
    if not np.all(np.diff(quantiles, axis=1) >= 0):
        raise ValueError("Summary quantiles are not monotonic")


def main() -> None:
    """Build detector states, sample A+, and write compact audit tables."""
    controls = build_age_control_points()
    mf_envelope = load_mf_age_envelope(resolve_fohlmeister_stack())
    anchors = detector.load_selected_anchors(detector.ANCHORS)
    segments = {
        spec.record_id: detector.regularize_record(
            detector.load_record(detector.WORKBOOK, spec), spec
        )
        for spec in detector.RECORDS
    }
    picks = build_definition_picks(anchors, segments, controls, mf_envelope)
    draws, diagnostics = sample_realizations(picks)
    summary = build_summary(picks, draws)
    parameters = build_parameter_table(mf_envelope)
    validate_outputs(controls, summary, draws)
    original_events = pd.read_csv(detector.OUTPUT_CSV)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    controls.to_csv(CONTROL_OUTPUT, index=False, float_format="%.6f")
    picks.to_csv(DEFINITION_PICK_OUTPUT, index=False, float_format="%.6f")
    summary.to_csv(SUMMARY_OUTPUT, index=False, float_format="%.6f")
    draws.to_csv(REALIZATION_OUTPUT, index=False, float_format="%.6f")
    parameters.to_csv(PARAMETER_OUTPUT, index=False)
    diagnostics.to_csv(DIAGNOSTIC_OUTPUT, index=False, float_format="%.8f")
    png_path, pdf_path = plot_age_realizations(summary, draws, original_events)

    print(f"Wrote {len(controls)} age controls to {CONTROL_OUTPUT.relative_to(ROOT)}")
    print(
        f"Wrote {len(picks)} definition picks to "
        f"{DEFINITION_PICK_OUTPUT.relative_to(ROOT)}"
    )
    print(f"Wrote {len(summary)} event summaries to {SUMMARY_OUTPUT.relative_to(ROOT)}")
    print(
        f"Wrote {len(draws):,} realizations to {REALIZATION_OUTPUT.relative_to(ROOT)}"
    )
    print(
        f"Wrote figures to {png_path.relative_to(ROOT)} and "
        f"{pdf_path.relative_to(ROOT)}"
    )
    print(
        "Rejected "
        f"{int(diagnostics.loc[0, 'rejected_proposals']):,} crossed proposals "
        f"({diagnostics.loc[0, 'rejection_fraction']:.2%})"
    )


if __name__ == "__main__":
    main()
