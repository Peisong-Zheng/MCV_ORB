#!/usr/bin/env python3
"""Propagate chronology and picking uncertainty through the MIS 6 sequence.

Sofular uses local U-Th control errors transferred through published component
age-depth curves; MF retains its shared age-envelope factor. One of nine picking
settings is selected per record. Non-monotone curves or crossed event sequences
are rejected as complete proposals, without sorting or clipping ages.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

# The paper exporter lives in the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_figure_export import copy_pdf_to_paper

if __package__:
    from . import event_detection as detector
    from . import sofular_chronology as sofular
else:
    import event_detection as detector
    import sofular_chronology as sofular


ROOT = Path(__file__).resolve().parent
WORKBOOK = ROOT / "data/raw/MIS6_speleothem_records.xlsx"
ANCHORS = ROOT / "data/curated/speleothem_mis6_published_event_label_anchors.csv"
COMPOSITE_CSV = ROOT / "data/processed/MIS6_composite_event_record/mis6_composite_event_record.csv"
OUTPUT_DIR = ROOT / "data" / "processed" / "MIS6_event_age_uncertainty"
AGE_CONTROLS = ROOT / "data" / "curated" / "mis6_age_control_points.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "mis6_event_age_uncertainty_summary.csv"
REALIZATION_OUTPUT = OUTPUT_DIR / "mis6_event_age_realizations.csv"
PROVENANCE_OUTPUT = OUTPUT_DIR / "parameters_and_provenance.csv"
FIGURE_DIR = ROOT / "figures" / "MIS6_event_age_uncertainty"
FIGURE_STEM = "MIS6_event_age_uncertainty"
PNG_DPI = 600

FOHLMEISTER_STACK_FILENAME = "Fohlmeister J et al-2023-data-mf_d18o_stack.txt"
FOHLMEISTER_STACK = ROOT / "data" / "raw" / FOHLMEISTER_STACK_FILENAME

N_REALIZATIONS = 10_000
RANDOM_SEED = 20260904
NORMAL_975 = 1.95996398454
MIN_EVENT_SEPARATION_KA = 0.0
N_EVENTS = detector.N_EVENTS
RECORD_ORDER = tuple(spec.record_id for spec in detector.RECORDS)


@dataclass(frozen=True)
class PickState:
    """Event ages and chronology errors for one record and detector setting."""

    event_indices: np.ndarray
    pick_ages: np.ndarray
    chronology_sigmas: np.ndarray
    chronology_contexts: tuple = ()


def load_age_controls(path: Path = AGE_CONTROLS) -> pd.DataFrame:
    """Load the legacy display subset, not the new chronology model controls."""
    if not path.exists():
        raise FileNotFoundError(f"Missing age-control table: {path}")

    controls = pd.read_csv(path)
    required = {
        "control_id",
        "record_id",
        "cave",
        "stalagmite_id",
        "age_ka_bp",
        "age_error_2sigma_ka",
        "source_study",
        "source_filename",
        "shown_in_composite_figure",
    }
    if missing := required.difference(controls.columns):
        raise ValueError(f"Age-control table is missing columns: {sorted(missing)}")

    if len(controls) != 11 or controls["control_id"].duplicated().any():
        raise ValueError("Expected 11 unique Sofular age controls")
    if set(controls["record_id"]) != {"Sofular"}:
        raise ValueError("Age controls must contain only the selected Sofular record")
    if any(Path(name).name != name for name in controls["source_filename"]):
        raise ValueError("Control provenance must contain filenames, not paths")

    numeric = controls[["age_ka_bp", "age_error_2sigma_ka"]]
    if (
        not np.isfinite(numeric).all().all()
        or not controls["age_error_2sigma_ka"].gt(0).all()
    ):
        raise ValueError("Age controls contain invalid ages or uncertainties")

    return controls.sort_values(
        ["record_id", "stalagmite_id", "age_ka_bp"]
    ).reset_index(drop=True)


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
        raise ValueError(f"MF pick {age:.6f} kyr BP lies outside a valid segment")
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


def sofular_chronology_at_age(age: float, sources=None) -> dict[str, object]:
    """Transfer dated-depth analytical errors to a projected stack event age."""
    context = sofular.build_context([age], sources=sources)
    sigma = float(context.nominal_sigma_ka[0])
    active = np.flatnonzero(context.interpolation_weights[0] > 0)
    support = "--".join(str(context.control_ids[i]) for i in active)
    return {
        "chronology_input_lower_approx95_ka_bp": age - 2 * sigma,
        "chronology_input_upper_approx95_ka_bp": age + 2 * sigma,
        "chronology_sampling_lower_approx95_ka_bp": age - 2 * sigma,
        "chronology_sampling_upper_approx95_ka_bp": age + 2 * sigma,
        "chronology_half_width_approx95_ka": 2 * sigma,
        "chronology_sigma_ka": sigma,
        "chronology_support_id": f"So-4:{support}; assumed_age_coordinate_projection",
        "source_bound_repair_applied": False,
    }


def chronology_at_age(
    record_id: str,
    age: float,
    controls: pd.DataFrame,
    mf_envelope: pd.DataFrame,
    *,
    sofular_sources=None,
) -> dict[str, object]:
    """Return the pre-conditioning working chronology scale at one pick.

    ``controls`` is retained for callers using the legacy figure-control table;
    the Sofular calculation reads the complete raw depth-resolved data instead.
    """
    if record_id == "MF":
        return mf_chronology_at_age(age, mf_envelope)
    if record_id == "Sofular":
        return sofular_chronology_at_age(age, sources=sofular_sources)
    raise ValueError(f"Unknown record: {record_id}")


def build_definition_picks(
    anchors: pd.DataFrame,
    segments_by_record: dict[str, list[detector.RegularSegment]],
    controls: pd.DataFrame,
    mf_envelope: pd.DataFrame,
) -> pd.DataFrame:
    """Run all 3 x 3 detector settings, keeping their original equal weights."""
    rows: list[dict[str, object]] = []
    sofular_sources = sofular.load_sources()
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
                        spec.record_id, pick_age, controls, mf_envelope,
                        sofular_sources=sofular_sources,
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
    expected_counts = {
        "MF": len(detector.MF_EVENT_LABELS),
        "Sofular": len(detector.SOFULAR_EVENT_LABELS),
    }
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
    *,
    sofular_component: str = "So-4",
) -> tuple[dict[tuple[str, str], PickState], dict[str, tuple[str, ...]]]:
    """Precompute local chronology weights for each coherent picking setting."""
    if sofular_component not in {"So-4", "So-57"}:
        raise ValueError("Sofular component must be So-4 or So-57")
    sources = sofular.load_sources()
    states: dict[tuple[str, str], PickState] = {}
    configs: dict[str, tuple[str, ...]] = {}
    for record_id in RECORD_ORDER:
        local = picks.loc[picks["source_record"].eq(record_id)]
        configs[record_id] = tuple(sorted(local["algorithm_config_id"].unique()))
        for config_id, group in local.groupby("algorithm_config_id"):
            group = group.sort_values("composite_event_number")
            ages = group["definition_pick_age_ka_bp"].to_numpy(float)
            sigmas = group["chronology_sigma_ka"].to_numpy(float)
            contexts = []
            if record_id == "Sofular":
                # The final event is older than the entire So-57 record. The
                # sensitivity uses So-57 only for its four overlapping events.
                components = np.full(len(group), sofular_component, dtype=object)
                components[group["composite_event_number"].to_numpy() == 21] = "So-4"
                for component in np.unique(components):
                    indices = np.flatnonzero(components == component)
                    context = sofular.build_context(
                        ages[indices], component=component, sources=sources
                    )
                    contexts.append((indices, context))
                    sigmas[indices] = context.nominal_sigma_ka
            states[(record_id, str(config_id))] = PickState(
                event_indices=group["composite_event_number"].to_numpy(int) - 1,
                pick_ages=ages,
                chronology_sigmas=sigmas,
                chronology_contexts=tuple(contexts),
            )
    return states, configs


def propose_realization(
    rng: np.random.Generator,
    states: dict[tuple[str, str], PickState],
    configs: dict[str, tuple[str, ...]],
) -> dict[str, object]:
    """Draw one detector setting per record and a complete joint chronology."""
    record_z = {"MF": float(rng.normal())}
    selected = {record: str(rng.choice(configs[record])) for record in RECORD_ORDER}
    definition = np.full(N_EVENTS, np.nan)
    sigma = np.full(N_EVENTS, np.nan)
    sampled = np.full(N_EVENTS, np.nan)
    chronology_valid = True
    for record in RECORD_ORDER:
        state = states[(record, selected[record])]
        definition[state.event_indices] = state.pick_ages
        sigma[state.event_indices] = state.chronology_sigmas
        if record == "MF":
            offsets = record_z[record] * state.chronology_sigmas
        else:
            offsets = np.full(len(state.pick_ages), np.nan)
            for indices, context in state.chronology_contexts:
                local_offsets, _ = sofular.propose_offsets(context, rng)
                if local_offsets is None:
                    chronology_valid = False
                else:
                    offsets[indices] = local_offsets
        sampled[state.event_indices] = state.pick_ages + offsets
    return {
        "record_z": record_z,
        "selected_configs": selected,
        "definition_ages": definition,
        "chronology_sigmas": sigma,
        "sampled_ages": sampled,
        "chronology_valid": chronology_valid,
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
    *,
    sofular_component: str = "So-4",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Rejection-sample complete, ordered event sequences."""
    if n_realizations < 1:
        raise ValueError("n_realizations must be positive")
    states, configs = build_state_lookup(picks, sofular_component=sofular_component)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    n_proposals = 0
    chronology_rejections = 0
    event_order_rejections = 0
    max_proposals = max(1000, 100 * n_realizations)

    while len(rows) < n_realizations and n_proposals < max_proposals:
        n_proposals += 1
        proposal = propose_realization(rng, states, configs)
        ages = proposal["sampled_ages"]
        if not proposal["chronology_valid"]:
            chronology_rejections += 1
            continue
        if not is_strictly_ordered(ages):
            event_order_rejections += 1
            continue
        row: dict[str, object] = {
            "realization_id": len(rows) + 1,
        }
        row.update(
            {
                f"MIS6_DO_{index + 1:02d}_age_ka_bp": age
                for index, age in enumerate(ages)
            }
        )
        rows.append(row)

    if len(rows) != n_realizations:
        raise RuntimeError(
            f"Accepted only {len(rows)} ordered sequences after {n_proposals} proposals"
        )
    rejected = n_proposals - n_realizations
    sampling = {
        "random_seed": seed,
        "requested_realizations": n_realizations,
        "accepted_realizations": len(rows),
        "total_proposals": n_proposals,
        "rejected_proposals": rejected,
        "rejected_nonmonotone_chronologies": chronology_rejections,
        "rejected_event_order": event_order_rejections,
        "sofular_component": sofular_component,
        "rejection_fraction": rejected / n_proposals,
        "minimum_event_separation_ka": MIN_EVENT_SEPARATION_KA,
    }
    return pd.DataFrame(rows), sampling


def build_summary(picks: pd.DataFrame, draws: pd.DataFrame) -> pd.DataFrame:
    """Summarize each event with method spread, chronology error, and MC bounds."""
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
        rows.append(
            {
                "composite_event_id": event_id,
                "composite_event_number": int(first["composite_event_number"]),
                "composite_event_label": first["composite_event_label"],
                "source_record": first["source_record"],
                "source_event_label": first["source_event_label"],
                "nominal_event_age_ka_bp": nominal["definition_pick_age_ka_bp"],
                "definition_age_min_ka_bp": group["definition_pick_age_ka_bp"].min(),
                "definition_age_max_ka_bp": group["definition_pick_age_ka_bp"].max(),
                "chronology_sigma_nominal_ka": nominal["chronology_sigma_ka"],
                "chronology_sigma_min_ka": group["chronology_sigma_ka"].min(),
                "chronology_sigma_max_ka": group["chronology_sigma_ka"].max(),
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
    expected = [f"MIS6_DO_{number:02d}" for number in range(1, N_EVENTS + 1)]
    if expected_ids != expected:
        raise ValueError(f"The age-realization figure requires ordered events 1--{N_EVENTS}")

    original_ages = original["event_age_ka_bp"].to_numpy(float)
    nominal_ages = summary["nominal_event_age_ka_bp"].to_numpy(float)
    # The composite CSV deliberately rounds exported event ages to 0.001 kyr.
    if not np.allclose(original_ages, nominal_ages, atol=5e-4, rtol=0):
        raise ValueError("Original event ages do not match the nominal picks")

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
    if correlation.shape != (N_EVENTS, N_EVENTS) or not np.isfinite(correlation).all():
        raise ValueError("Could not calculate a complete event-age correlation matrix")
    return plot_data, correlation


def plot_age_realizations(
    summary: pd.DataFrame,
    draws: pd.DataFrame,
    original_events: pd.DataFrame,
    output_dir: Path = FIGURE_DIR,
) -> tuple[Path, Path]:
    """Compare all A+ realizations with the original event sequence."""
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
    forest.set_xlabel("Sampled age − original event age (kyr)")
    forest.set_ylabel("Event (original age, kyr BP)")
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
    tick_indices = np.unique(np.linspace(0, N_EVENTS - 1, 6, dtype=int))
    tick_labels = (tick_indices + 1).astype(str)
    matrix.set_xticks(tick_indices, tick_labels)
    matrix.set_yticks(tick_indices, tick_labels)
    source_changes = np.flatnonzero(
        plot_data["source_record"].to_numpy()[1:]
        != plot_data["source_record"].to_numpy()[:-1]
    ) + 0.5
    for boundary in source_changes:
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
            "Subject": "Depth-informed Monte Carlo sequence uncertainty",
            "Creator": Path(__file__).name,
        },
    )
    figure.savefig(png_path, dpi=PNG_DPI, metadata={"Software": Path(__file__).name})
    copy_pdf_to_paper(pdf_path)
    plt.close(figure)
    return png_path, pdf_path


def build_provenance_table(
    mf_envelope: pd.DataFrame,
    sampling: dict[str, object],
) -> pd.DataFrame:
    """Collect method choices, source files, and sampling diagnostics."""
    rows = [
        ("analysis", "random_seed", sampling["random_seed"], "NumPy Generator"),
        (
            "analysis",
            "requested_realizations",
            sampling["requested_realizations"],
            f"complete {N_EVENTS}-event sequences",
        ),
        (
            "analysis",
            "accepted_realizations",
            sampling["accepted_realizations"],
            f"complete {N_EVENTS}-event sequences",
        ),
        ("analysis", "total_proposals", sampling["total_proposals"], ""),
        ("analysis", "rejected_proposals", sampling["rejected_proposals"], ""),
        (
            "analysis",
            "rejection_fraction",
            sampling["rejection_fraction"],
            "rejected / total proposals",
        ),
        (
            "analysis",
            "minimum_event_separation_ka",
            sampling["minimum_event_separation_ka"],
            "kyr",
        ),
        (
            "analysis",
            "event_order_rule",
            "fixed event rank; reject a complete proposal if events cross",
            "event ages are never sorted after sampling",
        ),
        (
            "analysis",
            "definition_sampling",
            "one of nine equally weighted detector settings per record",
            "accepted distribution is conditional on fixed event order",
        ),
        (
            "analysis",
            "definition_interpretation",
            "method-sensitivity proxy",
            "not a posterior or confidence interval",
        ),
        (
            "analysis",
            "excluded_uncertainty_terms",
            "cross-record synchronization; proxy resolution; separate smoothing error",
            "working sensitivity ensemble; not a complete chronology posterior",
        ),
        ("MF", "event_count", 16, ""),
        (
            "MF",
            "chronology_input",
            "published age_lower and age_upper curves",
            "Fohlmeister et al. (2023)",
        ),
        ("MF", "reported_scale", "approximately 95% age-model envelope", ""),
        (
            "MF",
            "conversion_to_1sigma",
            "maximum distance to repaired envelope / 1.95996398454",
            "symmetric sampling error at each event pick",
        ),
        (
            "MF",
            "shared_chronology_draw",
            "one standard-normal z for all MF events",
            "remaining one-factor simplification; not an estimated MF covariance",
        ),
        ("MF", "source_study", "Fohlmeister et al. (2023)", ""),
        ("MF", "source_filename", FOHLMEISTER_STACK_FILENAME, "data and age bounds"),
        (
            "MF",
            "uncertainty_scale_source_filename",
            "Fohlmeister J et al-2023-SOM.pdf",
            "bounds interpreted from Fig. S3",
        ),
        (
            "MF",
            "source_valid_rows",
            mf_envelope.attrs["valid_row_count"],
            "rows with finite age bounds",
        ),
        (
            "MF",
            "source_unique_ages",
            mf_envelope.attrs["unique_age_count"],
            "duplicate ages merged using the outer envelope",
        ),
        (
            "MF",
            "source_rows_needing_bound_repair",
            mf_envelope.attrs["source_rows_needing_bound_repair"],
            "nominal age lay outside the reported bound pair",
        ),
        ("Sofular", "event_count", len(detector.SOFULAR_EVENT_LABELS), ""),
        ("Sofular", "chronology_input", "complete raw U-Th dates, depths and published StalAge curve", "Held et al. (2024)"),
        ("Sofular", "reported_scale", "2sigma", "individual corrected U-Th dates"),
        ("Sofular", "conversion_to_1sigma", "each raw corrected age error / 2, then propagate control covariance", "pre-monotonicity-conditioning analytical sigma"),
        ("Sofular", "chronology_draw", "independent normal errors at dated-depth knots; interpolate their displacement", "published age-depth shape preserved between knots; reject nonmonotone segment curves"),
        ("Sofular", "main_component", sampling["sofular_component"], "So-4 covers all five events; So-57 overlap saved separately"),
        ("Sofular", "depth_projection", "assumed_age_coordinate_projection", "stack event age mapped onto individual StalAge curve; physical event depth is not identified"),
        ("Sofular", "model_interpretation", "analytical-error transfer around published age-depth curve", "not a new StalAge/iscam posterior; knots centered on model curve, not measured U-Th means"),
        ("Sofular", "excluded_uncertainty_terms", "unknown U-Th systematic covariance; stack alignment and age-model structural uncertainty", "component and mapping-lag sensitivity do not recover the missing iscam transformations"),
        ("Sofular", "source_filename", "held2024-so-4.txt; held2024-so-57.txt", "data/raw/Held2024/source_manifest.csv contains original paths, DOI and SHA256"),
        ("Sofular", "growth_hiatus_depth_mm", 781.5, "So-4; do not interpolate across hiatus or extrapolate beyond dated/proxy support"),
        ("analysis", "rejected_nonmonotone_chronologies", sampling["rejected_nonmonotone_chronologies"], "complete proposals"),
        ("analysis", "rejected_event_order", sampling["rejected_event_order"], "complete proposals"),
        ("analysis", "age_reference", "ka BP1950", "raw Sofular U-Th fields explicitly use 1950"),
    ]
    return pd.DataFrame(rows, columns=["scope", "parameter", "value", "note"])


def validate_results(summary: pd.DataFrame, draws: pd.DataFrame) -> None:
    """Check the event identities, sampled order, and uncertainty intervals."""
    if len(summary) != N_EVENTS or summary["composite_event_id"].duplicated().any():
        raise ValueError(f"The uncertainty summary must contain {N_EVENTS} unique events")
    if summary["composite_event_number"].tolist() != list(range(1, N_EVENTS + 1)):
        raise ValueError(f"Event identities are not in their fixed 1--{N_EVENTS} order")

    event_columns = [
        f"MIS6_DO_{number:02d}_age_ka_bp" for number in range(1, N_EVENTS + 1)
    ]
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


def save_sofular_diagnostics(picks: pd.DataFrame, output_dir: Path = OUTPUT_DIR) -> None:
    """Expose the dated depths, projection assumptions and alternate components."""
    sources = sofular.load_sources()
    sofular.control_diagnostics(sources).to_csv(
        output_dir / "sofular_raw_control_diagnostics.csv", index=False
    )
    local = picks.loc[picks["source_record"].eq("Sofular")]
    projection_tables = []
    sensitivity_rows = []
    covariance_rows = []
    for config_id, group in local.groupby("algorithm_config_id"):
        group = group.sort_values("composite_event_number")
        ages = group["definition_pick_age_ka_bp"].to_numpy(float)
        labels = group["composite_event_id"].to_numpy()
        for component in ("So-4", "So-57"):
            selected = np.arange(len(group) if component == "So-4" else len(group) - 1)
            for lag in (-1.0, -0.5, 0.0, 0.5, 1.0):
                context = sofular.build_context(
                    ages[selected], component=component, mapping_lag_ka=lag,
                    sources=sources,
                )
                if lag == 0:
                    projection = sofular.projection_diagnostics(context)
                    projection.insert(0, "composite_event_id", labels[selected])
                    projection.insert(0, "algorithm_config_id", config_id)
                    projection_tables.append(projection)
                basis = context.interpolation_weights * context.control_sigmas_ka
                covariance = basis @ basis.T
                for j, event_id in enumerate(labels[selected]):
                    sensitivity_rows.append({
                        "algorithm_config_id": config_id,
                        "component": component,
                        "mapping_lag_ka": lag,
                        "composite_event_id": event_id,
                        "projected_depth_mm": context.event_depth_mm[j],
                        "chronology_sigma_before_conditioning_ka": context.nominal_sigma_ka[j],
                    })
                    for k, other_id in enumerate(labels[selected]):
                        covariance_rows.append({
                            "algorithm_config_id": config_id,
                            "component": component,
                            "mapping_lag_ka": lag,
                            "event_id": event_id,
                            "other_event_id": other_id,
                            "covariance_before_conditioning_ka2": covariance[j, k],
                        })
    pd.concat(projection_tables, ignore_index=True).to_csv(
        output_dir / "sofular_event_depth_projection.csv", index=False
    )
    pd.DataFrame(sensitivity_rows).to_csv(
        output_dir / "sofular_component_mapping_sensitivity.csv", index=False
    )
    pd.DataFrame(covariance_rows).to_csv(
        output_dir / "sofular_component_mapping_covariance.csv", index=False
    )


def main() -> None:
    """Run the depth-informed experiment and save reproducible diagnostics."""
    controls = load_age_controls()
    mf_envelope = load_mf_age_envelope(FOHLMEISTER_STACK)
    anchors = detector.load_selected_anchors(ANCHORS)
    segments = {
        spec.record_id: detector.regularize_record(
            detector.load_record(WORKBOOK, spec), spec
        )
        for spec in detector.RECORDS
    }
    picks = build_definition_picks(anchors, segments, controls, mf_envelope)
    draws, sampling = sample_realizations(picks)
    summary = build_summary(picks, draws)
    provenance = build_provenance_table(mf_envelope, sampling)
    validate_results(summary, draws)
    original_events = pd.read_csv(COMPOSITE_CSV)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_OUTPUT, index=False, float_format="%.6f")
    draws.to_csv(REALIZATION_OUTPUT, index=False, float_format="%.6f")
    provenance.to_csv(PROVENANCE_OUTPUT, index=False)
    picks.to_csv(OUTPUT_DIR / "definition_picks_and_chronology.csv", index=False)
    save_sofular_diagnostics(picks)
    alternative, alternative_sampling = sample_realizations(picks, sofular_component="So-57")
    alternative.to_csv(
        OUTPUT_DIR / "mis6_event_age_realizations_so57_overlap.csv",
        index=False, float_format="%.6f",
    )
    # Summary chronology scales must match the alternative component weights.
    alternate_picks = picks.copy()
    alternate_states, _ = build_state_lookup(picks, sofular_component="So-57")
    for (record, config), state in alternate_states.items():
        mask = alternate_picks["source_record"].eq(record) & alternate_picks["algorithm_config_id"].eq(config)
        alternate_picks.loc[mask, "chronology_sigma_ka"] = state.chronology_sigmas
    build_summary(alternate_picks, alternative).to_csv(
        OUTPUT_DIR / "mis6_event_age_uncertainty_summary_so57_overlap.csv",
        index=False, float_format="%.6f",
    )
    build_provenance_table(mf_envelope, alternative_sampling).to_csv(
        OUTPUT_DIR / "parameters_and_provenance_so57_overlap.csv", index=False
    )
    png_path, pdf_path = plot_age_realizations(summary, draws, original_events)

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
        f"{int(sampling['rejected_proposals']):,} nonmonotone or crossed proposals "
        f"({sampling['rejection_fraction']:.2%})"
    )


if __name__ == "__main__":
    main()
