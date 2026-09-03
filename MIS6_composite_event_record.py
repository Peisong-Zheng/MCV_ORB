#!/usr/bin/env python3
"""Build a provisional MIS 6 composite event chronology.

Literature label positions define local search regions. Final event ages are
direction-constrained maximum gradients on lightly smoothed proxy records.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


ROOT = Path(__file__).resolve().parent
WORKBOOK = ROOT / "data" / "raw" / "speleothem_data.xlsx"
ANCHORS = (
    ROOT
    / "data"
    / "processed"
    / "Speleothem_published_event_plot"
    / "speleothem_mis6_published_event_label_anchors.csv"
)
OUTPUT_DIR = ROOT / "data" / "processed" / "MIS6_composite_event_record"
OUTPUT_CSV = OUTPUT_DIR / "mis6_composite_event_record.csv"
FIGURE_DIR = ROOT / "figures" / "MIS6_composite_event_record"
FIGURE_STEM = "MIS6_composite_event_record"

GRID_STEP_KA = 0.001
MATERIAL_GAP_KA = 0.5
NOMINAL_SEARCH_HALF_WIDTH_KA = 0.4
SENSITIVITY_SEARCH_HALF_WIDTHS_KA = (0.3, 0.4, 0.5)
CHANGE_INNER_KA = 0.05
CHANGE_OUTER_KA = 0.15
PNG_DPI = 600


@dataclass(frozen=True)
class RecordSpec:
    record_id: str
    sheet: str
    proxy: str
    proxy_label: str
    data_source: str
    label_source: str
    color: str
    gradient_direction: int
    nominal_sigma_ka: float
    sensitivity_sigmas_ka: tuple[float, ...]
    plot_range_ka: tuple[float, float]
    invert_proxy_axis: bool = False


@dataclass
class RegularSegment:
    raw: pd.DataFrame
    grid_age: np.ndarray
    interpolated: np.ndarray
    smoothed: dict[float, np.ndarray]
    gradients: dict[float, np.ndarray]


RECORDS = (
    RecordSpec(
        record_id="MF",
        sheet="\ufeffMF_Fohlmeister_etal_2023",
        proxy="d18O",
        proxy_label=r"$\delta^{18}\mathrm{O}$ (‰ VPDB)",
        data_source="Fohlmeister et al. (2023)",
        label_source="Fohlmeister et al. (2023)",
        color="#C43C39",
        gradient_direction=-1,
        nominal_sigma_ka=0.050,
        sensitivity_sigmas_ka=(0.035, 0.050, 0.075),
        plot_range_ka=(130.0, 175.0),
    ),
    RecordSpec(
        record_id="Huagapo",
        sheet="Huagapo_Burn_etal_2019",
        proxy="d18O",
        proxy_label=r"$\delta^{18}\mathrm{O}$ (‰ VPDB)",
        data_source="Burn et al. (2019)",
        label_source="Held et al. (2024)",
        color="#1976D2",
        gradient_direction=-1,
        nominal_sigma_ka=0.100,
        sensitivity_sigmas_ka=(0.075, 0.100, 0.125),
        plot_range_ka=(176.8, 195.2),
    ),
    RecordSpec(
        record_id="Sofular",
        sheet="Sofular_Held_etal_2024",
        proxy="d13C",
        proxy_label=r"$\delta^{13}\mathrm{C}$ (‰ VPDB)",
        data_source="Held et al. (2024)",
        label_source="Held et al. (2024)",
        color="#6A1B9A",
        gradient_direction=1,
        nominal_sigma_ka=0.050,
        sensitivity_sigmas_ka=(0.035, 0.050, 0.075),
        plot_range_ka=(179.2, 180.9),
        invert_proxy_axis=True,
    ),
)
RECORD_BY_ID = {record.record_id: record for record in RECORDS}
HELD_SELECTION = {
    "Huagapo": {"6.9", "6.12", "6.13"},
    "Sofular": {"6.10", "6.11"},
}


def configure_plot_style() -> None:
    """Use legible typography at the final 180 mm figure width."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "axes.titleweight": "semibold",
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def load_selected_anchors(path: Path) -> pd.DataFrame:
    """Select literature events and assign continuous composite labels."""
    anchors = pd.read_csv(
        path,
        dtype={
            "record_id": "string",
            "event_label": "string",
            "display_label": "string",
        },
    )
    required = {
        "record_id",
        "event_label",
        "display_label",
        "anchor_age_ka_bp",
        "anchor_type",
        "label_scheme",
        "anchor_purpose",
    }
    missing = required.difference(anchors.columns)
    if missing:
        raise ValueError(f"Anchor table is missing columns: {sorted(missing)}")

    anchors["anchor_age_ka_bp"] = pd.to_numeric(
        anchors["anchor_age_ka_bp"], errors="coerce"
    )
    if anchors["anchor_age_ka_bp"].isna().any():
        raise ValueError("All selected anchor ages must be numeric")
    if not anchors["anchor_purpose"].eq("label_placement_only").all():
        raise ValueError("Literature coordinates must be marked label_placement_only")
    if anchors.duplicated(["record_id", "display_label"]).any():
        raise ValueError("Source display labels must be unique within each record")

    mf = anchors.loc[
        anchors["record_id"].eq("MF")
        & anchors["anchor_age_ka_bp"].between(130.0, 175.0)
    ].sort_values("anchor_age_ka_bp")
    expected_mf_labels = [f"6.{number}" for number in range(1, 17)]
    if mf["display_label"].tolist() != expected_mf_labels:
        raise ValueError("Expected the complete MF 6.1–6.16 sequence within 130–175 ka")

    held_parts = []
    for record_id, labels in HELD_SELECTION.items():
        part = anchors.loc[
            anchors["record_id"].eq(record_id) & anchors["display_label"].isin(labels)
        ]
        if set(part["display_label"]) != labels:
            raise ValueError(f"Missing selected Held labels for {record_id}")
        held_parts.append(part)
    held = pd.concat(held_parts, ignore_index=True).sort_values("anchor_age_ka_bp")

    mf = mf.copy()
    held = held.copy()
    mf["composite_event_display_label"] = mf["display_label"]
    held["composite_event_display_label"] = [f"6.{number}" for number in range(17, 22)]
    selected = pd.concat([mf, held], ignore_index=True)
    selected = selected.rename(columns={"display_label": "source_event_display_label"})
    selected = selected.sort_values("anchor_age_ka_bp").reset_index(drop=True)

    if (
        len(selected) != 21
        or selected["composite_event_display_label"].duplicated().any()
    ):
        raise ValueError("The composite chronology must contain 21 unique events")
    return selected


def load_record(path: Path, spec: RecordSpec) -> pd.DataFrame:
    """Read a record, average duplicate ages, sort, and identify data gaps."""
    raw = pd.read_excel(path, sheet_name=spec.sheet)
    if raw.shape[1] < 2:
        raise ValueError(f"{spec.sheet!r} must contain age and proxy columns")

    frame = raw.iloc[:, :2].copy()
    frame.columns = ["age_ka_bp", "proxy"]
    frame = frame.apply(pd.to_numeric, errors="coerce").dropna()
    frame = (
        frame.groupby("age_ka_bp", as_index=False)["proxy"]
        .mean()
        .sort_values("age_ka_bp")
        .reset_index(drop=True)
    )
    if len(frame) < 3 or not frame["age_ka_bp"].is_monotonic_increasing:
        raise ValueError(f"{spec.sheet!r} has an invalid age axis")

    gaps = frame["age_ka_bp"].diff().gt(MATERIAL_GAP_KA)
    frame["segment"] = gaps.cumsum().astype(int)
    return frame


def regularize_record(frame: pd.DataFrame, spec: RecordSpec) -> list[RegularSegment]:
    """Interpolate and smooth each continuous record segment independently."""
    sigmas = sorted(set(spec.sensitivity_sigmas_ka + (spec.nominal_sigma_ka,)))
    segments = []
    for _, raw_segment in frame.groupby("segment", sort=True):
        raw_segment = raw_segment.reset_index(drop=True)
        if len(raw_segment) < 2:
            raise ValueError(f"{spec.record_id} contains an isolated one-point segment")
        ages = raw_segment["age_ka_bp"].to_numpy(dtype=float)
        proxy = raw_segment["proxy"].to_numpy(dtype=float)
        grid_age = np.arange(ages[0], ages[-1] + GRID_STEP_KA / 2, GRID_STEP_KA)
        interpolated = np.interp(grid_age, ages, proxy)
        smoothed = {}
        gradients = {}
        for sigma_ka in sigmas:
            curve = gaussian_filter1d(
                interpolated,
                sigma=sigma_ka / GRID_STEP_KA,
                mode="nearest",
            )
            smoothed[sigma_ka] = curve
            gradients[sigma_ka] = np.gradient(curve, grid_age)
        segments.append(
            RegularSegment(
                raw=raw_segment,
                grid_age=grid_age,
                interpolated=interpolated,
                smoothed=smoothed,
                gradients=gradients,
            )
        )
    return segments


def segment_containing(
    segments: list[RegularSegment], anchor_age: float
) -> RegularSegment:
    for segment in segments:
        if segment.grid_age[0] <= anchor_age <= segment.grid_age[-1]:
            return segment
    raise ValueError(f"No continuous data segment contains anchor {anchor_age:.3f} ka")


def search_bounds(
    anchor_age: float,
    record_anchor_ages: np.ndarray,
    segment: RegularSegment,
    half_width_ka: float,
) -> tuple[float, float]:
    """Clip a local window at record gaps and neighboring event midpoints."""
    position = int(np.flatnonzero(np.isclose(record_anchor_ages, anchor_age))[0])
    lower = max(anchor_age - half_width_ka, float(segment.grid_age[0]))
    upper = min(anchor_age + half_width_ka, float(segment.grid_age[-1]))
    if position > 0:
        lower = max(lower, (record_anchor_ages[position - 1] + anchor_age) / 2)
    if position + 1 < len(record_anchor_ages):
        upper = min(upper, (anchor_age + record_anchor_ages[position + 1]) / 2)
    if lower >= upper:
        raise ValueError(f"Empty search interval around {anchor_age:.3f} ka")
    return lower, upper


def pick_gradient_peak(
    spec: RecordSpec,
    segment: RegularSegment,
    sigma_ka: float,
    bounds: tuple[float, float],
) -> tuple[int, float]:
    mask = (segment.grid_age >= bounds[0]) & (segment.grid_age <= bounds[1])
    candidates = np.flatnonzero(mask)
    if len(candidates) < 3:
        raise ValueError(f"Too few grid points in search interval {bounds}")
    score = spec.gradient_direction * segment.gradients[sigma_ka][candidates]
    local_index = int(np.argmax(score))
    if score[local_index] <= 0:
        raise ValueError(
            "No gradient in the expected direction inside the search window"
        )
    index = int(candidates[local_index])
    return index, float(segment.grid_age[index])


def local_resolution_years(segment: RegularSegment, event_age: float) -> float:
    nearby = segment.raw.loc[
        segment.raw["age_ka_bp"].between(event_age - 0.5, event_age + 0.5),
        "age_ka_bp",
    ].to_numpy(dtype=float)
    if len(nearby) < 3:
        distances = (segment.raw["age_ka_bp"] - event_age).abs()
        nearby = np.sort(
            segment.raw.loc[distances.nsmallest(6).index, "age_ka_bp"].to_numpy(
                dtype=float
            )
        )
    return float(np.median(np.diff(nearby)) * 1000)


def classify_result(
    record_id: str,
    source_label: str,
    tuning_span_yr: float,
    resolution_yr: float,
    boundary_proximity: bool,
) -> tuple[str, str]:
    notes = []
    two_stage = record_id == "MF" and source_label in {"6.10", "6.14"}
    if two_stage:
        notes.append("two-stage transition")
    if tuning_span_yr > 150:
        notes.append("parameter-sensitive age")
    elif tuning_span_yr > 50:
        notes.append("moderate parameter sensitivity")
    if resolution_yr > 100:
        notes.append("sparse local sampling")
    elif resolution_yr > 75:
        notes.append("coarse local sampling")
    if boundary_proximity:
        notes.append("near continuous-segment boundary; local change is edge-sensitive")

    if two_stage or tuning_span_yr > 150 or resolution_yr > 100 or boundary_proximity:
        quality = "review"
    elif tuning_span_yr > 50 or resolution_yr > 75:
        quality = "moderate"
    else:
        quality = "stable"
    return quality, "; ".join(notes) if notes else "stable under tested parameters"


def estimate_events(
    anchors: pd.DataFrame,
    segments_by_record: dict[str, list[RegularSegment]],
) -> pd.DataFrame:
    """Estimate one transition age and proxy change for every selected event."""
    results = []
    for _, anchor in anchors.iterrows():
        record_id = str(anchor["record_id"])
        spec = RECORD_BY_ID[record_id]
        anchor_age = float(anchor["anchor_age_ka_bp"])
        record_anchors = (
            anchors.loc[anchors["record_id"].eq(record_id), "anchor_age_ka_bp"]
            .sort_values()
            .to_numpy(dtype=float)
        )
        segment = segment_containing(segments_by_record[record_id], anchor_age)
        nominal_bounds = search_bounds(
            anchor_age,
            record_anchors,
            segment,
            NOMINAL_SEARCH_HALF_WIDTH_KA,
        )
        event_index, event_age = pick_gradient_peak(
            spec,
            segment,
            spec.nominal_sigma_ka,
            nominal_bounds,
        )

        tuning_ages = []
        for sigma_ka in spec.sensitivity_sigmas_ka:
            for half_width_ka in SENSITIVITY_SEARCH_HALF_WIDTHS_KA:
                bounds = search_bounds(
                    anchor_age, record_anchors, segment, half_width_ka
                )
                _, candidate_age = pick_gradient_peak(spec, segment, sigma_ka, bounds)
                tuning_ages.append(candidate_age)
        unique_tuning_ages = np.unique(np.round(tuning_ages, 3))
        tuning_min = float(min(tuning_ages))
        tuning_max = float(max(tuning_ages))
        tuning_span_yr = (tuning_max - tuning_min) * 1000
        tuning_structure = (
            "multimodal"
            if np.any(np.diff(unique_tuning_ages) > 0.1)
            else "single_cluster"
        )

        curve = segment.smoothed[spec.nominal_sigma_ka]
        gradient = segment.gradients[spec.nominal_sigma_ka]
        young_mask = (segment.grid_age >= event_age - CHANGE_OUTER_KA) & (
            segment.grid_age <= event_age - CHANGE_INNER_KA
        )
        old_mask = (segment.grid_age >= event_age + CHANGE_INNER_KA) & (
            segment.grid_age <= event_age + CHANGE_OUTER_KA
        )
        if not young_mask.any() or not old_mask.any():
            raise ValueError(f"Cannot calculate flanking proxy levels for {event_age}")
        young_level = float(np.median(curve[young_mask]))
        old_level = float(np.median(curve[old_mask]))
        proxy_change = young_level - old_level
        resolution_yr = local_resolution_years(segment, event_age)
        observations = int(segment.raw["age_ka_bp"].between(*nominal_bounds).sum())
        young_observations = int(
            segment.raw["age_ka_bp"]
            .between(event_age - CHANGE_OUTER_KA, event_age - CHANGE_INNER_KA)
            .sum()
        )
        old_observations = int(
            segment.raw["age_ka_bp"]
            .between(event_age + CHANGE_INNER_KA, event_age + CHANGE_OUTER_KA)
            .sum()
        )
        boundary_distance_yr = (
            min(event_age - segment.grid_age[0], segment.grid_age[-1] - event_age)
            * 1000
        )
        boundary_proximity = (
            boundary_distance_yr
            < max(CHANGE_OUTER_KA, 4 * spec.nominal_sigma_ka) * 1000
        )
        composite_label = str(anchor["composite_event_display_label"])
        composite_number = int(composite_label.split(".")[-1])
        source_display_label = str(anchor["source_event_display_label"])
        event_age_qc, qc_note = classify_result(
            record_id,
            source_display_label,
            tuning_span_yr,
            resolution_yr,
            boundary_proximity,
        )
        minimum_flank_observations = min(young_observations, old_observations)
        if boundary_proximity or minimum_flank_observations < 2:
            local_change_qc = "review"
        elif minimum_flank_observations < 3:
            local_change_qc = "moderate"
        else:
            local_change_qc = "stable"
        local_change_note = (
            f"minimum {minimum_flank_observations} raw observations in a flank"
            + ("; segment-edge smoothing applies" if boundary_proximity else "")
        )

        results.append(
            {
                "composite_event_id": f"MIS6_DO_{composite_number:02d}",
                "composite_event_number": composite_number,
                "composite_event_display_label": composite_label,
                "event_age_ka_bp": event_age,
                "event_age_status": "provisional_algorithmic_estimate",
                "source_record": record_id,
                "source_event_id": (
                    f"{anchor['label_scheme']}_{record_id}_"
                    f"{source_display_label.replace('.', '_')}"
                ),
                "source_event_label": anchor["event_label"],
                "source_event_display_label": source_display_label,
                "source_anchor_age_ka_bp": anchor_age,
                "source_anchor_type": anchor["anchor_type"],
                "source_anchor_offset_yr": (event_age - anchor_age) * 1000,
                "proxy": spec.proxy,
                "data_source": spec.data_source,
                "label_source": spec.label_source,
                "expected_gradient_direction": (
                    "negative" if spec.gradient_direction < 0 else "positive"
                ),
                "smoothing_sigma_yr": spec.nominal_sigma_ka * 1000,
                "search_half_width_yr": NOMINAL_SEARCH_HALF_WIDTH_KA * 1000,
                "effective_search_min_ka_bp": nominal_bounds[0],
                "effective_search_max_ka_bp": nominal_bounds[1],
                "local_tuning_candidate_ages_ka_bp": ";".join(
                    f"{age:.3f}" for age in unique_tuning_ages
                ),
                "local_tuning_age_min_ka_bp": tuning_min,
                "local_tuning_age_max_ka_bp": tuning_max,
                "local_tuning_span_yr": tuning_span_yr,
                "local_tuning_structure": tuning_structure,
                "local_tuning_range_type": (
                    "algorithm_parameter_sweep_not_age_model_uncertainty"
                ),
                "event_age_qc": event_age_qc,
                "local_median_resolution_yr": resolution_yr,
                "observations_in_search_window": observations,
                "young_flank_observations": young_observations,
                "old_flank_observations": old_observations,
                "distance_to_segment_boundary_yr": boundary_distance_yr,
                "event_proxy_per_mil": float(curve[event_index]),
                "young_flank_proxy_per_mil": young_level,
                "old_flank_proxy_per_mil": old_level,
                "local_flank_change_young_minus_old_per_mil": proxy_change,
                "local_transition_magnitude_per_mil": abs(proxy_change),
                "local_change_qc": local_change_qc,
                "local_change_note": local_change_note,
                "peak_gradient_per_mil_per_ka": float(gradient[event_index]),
                "qc_note": qc_note,
                "method": (
                    "gap-bounded 1 yr interpolation; Gaussian smoothing; "
                    "expected-direction maximum gradient; local tuning sweep"
                ),
            }
        )
    return pd.DataFrame(results).sort_values("event_age_ka_bp").reset_index(drop=True)


def validate_results(events: pd.DataFrame) -> None:
    expected_labels = [f"6.{number}" for number in range(1, 22)]
    expected_ids = [f"MIS6_DO_{number:02d}" for number in range(1, 22)]
    if events["composite_event_display_label"].tolist() != expected_labels:
        raise ValueError("Composite event labels are not the ordered 6.1–6.21 sequence")
    if events["composite_event_id"].tolist() != expected_ids:
        raise ValueError("Stable composite event IDs are incomplete or out of order")
    if not np.all(np.diff(events["event_age_ka_bp"].to_numpy(dtype=float)) > 0):
        raise ValueError("Estimated event ages are not strictly age-ordered")
    if events.duplicated(["composite_event_id"]).any():
        raise ValueError("Composite event labels must be unique")

    numeric_columns = [
        "event_age_ka_bp",
        "source_anchor_age_ka_bp",
        "effective_search_min_ka_bp",
        "effective_search_max_ka_bp",
        "local_tuning_age_min_ka_bp",
        "local_tuning_age_max_ka_bp",
        "local_flank_change_young_minus_old_per_mil",
        "peak_gradient_per_mil_per_ka",
    ]
    if not np.isfinite(events[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("The event table contains non-finite results")
    if not (
        events["event_age_ka_bp"]
        .between(
            events["effective_search_min_ka_bp"],
            events["effective_search_max_ka_bp"],
        )
        .all()
    ):
        raise ValueError("An event estimate falls outside its nominal search interval")
    if not (
        events["event_age_ka_bp"]
        .between(
            events["local_tuning_age_min_ka_bp"],
            events["local_tuning_age_max_ka_bp"],
        )
        .all()
    ):
        raise ValueError("A nominal estimate is absent from its local tuning sweep")
    direction = events["expected_gradient_direction"].map(
        {"negative": -1, "positive": 1}
    )
    if not (events["peak_gradient_per_mil_per_ka"] * direction > 0).all():
        raise ValueError("An estimated event violates its expected gradient direction")


def export_events(events: pd.DataFrame, path: Path) -> None:
    """Write a readable table while keeping 0.001 ka audit precision."""
    output = events.copy()
    output["composite_event_display_label"] = (
        "MIS " + output["composite_event_display_label"]
    )
    output = output.rename(
        columns={"composite_event_display_label": "composite_event_label"}
    ).drop(columns="source_event_display_label")
    numeric = lambda column: pd.api.types.is_numeric_dtype(output[column])
    age_columns = [
        column for column in output if column.endswith("_ka_bp") and numeric(column)
    ]
    year_columns = [
        column for column in output if column.endswith("_yr") and numeric(column)
    ]
    proxy_columns = [
        column for column in output if column.endswith("_per_mil") and numeric(column)
    ]
    output[age_columns] = output[age_columns].round(3)
    output[year_columns] = output[year_columns].round(1)
    output[proxy_columns] = output[proxy_columns].round(3)
    output["peak_gradient_per_mil_per_ka"] = output[
        "peak_gradient_per_mil_per_ka"
    ].round(2)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)


def style_axis(axis: plt.Axes) -> None:
    axis.set_axisbelow(True)
    axis.grid(axis="x", color="#E4E4E4", linewidth=0.55)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(length=3, width=0.65)


def marker_style(quality: str, color: str) -> dict[str, object]:
    """Distinguish stable, moderate, and review estimates without a color scale."""
    marker = {"stable": "o", "moderate": "D", "review": "^"}[quality]
    return {
        "marker": marker,
        "facecolor": color if quality == "stable" else "white",
        "edgecolor": color,
    }


def draw_timeline(axis: plt.Axes, events: pd.DataFrame) -> None:
    lane = {"MF": 2.0, "Huagapo": 1.0, "Sofular": 0.0}
    for record_id, group in events.groupby("source_record", sort=False):
        spec = RECORD_BY_ID[record_id]
        y = lane[record_id]
        ages = group["event_age_ka_bp"].to_numpy(dtype=float)
        label_shifts = np.zeros(len(group))
        for index, separation in enumerate(np.diff(ages)):
            if separation < 0.75:
                label_shifts[index] -= 0.18
                label_shifts[index + 1] += 0.18
        for index, (_, event) in enumerate(group.iterrows()):
            if abs(event["event_age_ka_bp"] - 175) < 0.5:
                label_shifts[index] -= 0.22
            point_style = marker_style(event["event_age_qc"], spec.color)
            axis.scatter(
                event["event_age_ka_bp"],
                y,
                s=20,
                linewidth=0.8,
                zorder=3,
                **point_style,
            )
            axis.annotate(
                event["composite_event_display_label"],
                xy=(event["event_age_ka_bp"], y + 0.04),
                xytext=(
                    event["event_age_ka_bp"] + label_shifts[index],
                    y + 0.14 + 0.27 * (index % 3),
                ),
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=6.5,
                color=spec.color,
                arrowprops={
                    "arrowstyle": "-",
                    "color": spec.color,
                    "linewidth": 0.45,
                    "alpha": 0.65,
                },
            )
    axis.axvline(175, color="#777777", linestyle="--", linewidth=0.8)
    axis.text(
        175.25,
        -0.34,
        "record transition",
        color="#666666",
        fontsize=6.5,
        ha="left",
        va="bottom",
    )
    axis.set(
        xlim=(130, 196),
        ylim=(-0.45, 3.05),
        yticks=[0, 1, 2],
        yticklabels=["Sofular", "Huagapo", "MF"],
        xlabel="Age (ka BP; older →)",
        title="(a) Composite sequence and source records",
    )
    axis.xaxis.set_major_locator(MultipleLocator(5))
    style_axis(axis)


def event_panel_label(event: pd.Series) -> str:
    if event["source_record"] == "MF":
        return str(event["composite_event_display_label"])
    return (
        f"{event['composite_event_display_label']}\n"
        f"[Held {event['source_event_display_label']}]"
    )


def draw_record_panel(
    axis: plt.Axes,
    panel_letter: str,
    spec: RecordSpec,
    segments: list[RegularSegment],
    events: pd.DataFrame,
) -> None:
    for segment in segments:
        axis.plot(
            segment.raw["age_ka_bp"],
            segment.raw["proxy"],
            color="#B5B5B5",
            linewidth=0.55,
            marker=".",
            markersize=1.8,
            label="observations",
            zorder=1,
        )
        axis.plot(
            segment.grid_age,
            segment.smoothed[spec.nominal_sigma_ka],
            color=spec.color,
            linewidth=1.5,
            label="smoothed proxy",
            zorder=2,
        )

    local_events = events.loc[events["source_record"].eq(spec.record_id)]
    for index, (_, event) in enumerate(local_events.iterrows()):
        for candidate_age in str(event["local_tuning_candidate_ages_ka_bp"]).split(";"):
            axis.axvline(
                float(candidate_age),
                ymin=0.02,
                ymax=0.09,
                color=spec.color,
                linewidth=0.75,
                alpha=0.35,
                zorder=3,
            )
        axis.axvline(
            event["source_anchor_age_ka_bp"],
            color="#777777",
            linestyle=":",
            linewidth=0.75,
            zorder=2,
        )
        axis.axvline(
            event["event_age_ka_bp"],
            color=spec.color,
            linewidth=0.75,
            alpha=0.7,
            zorder=2,
        )
        point_style = marker_style(event["event_age_qc"], spec.color)
        axis.scatter(
            event["event_age_ka_bp"],
            event["event_proxy_per_mil"],
            s=24,
            linewidth=0.9,
            zorder=4,
            **point_style,
        )
        if spec.record_id == "MF":
            y_position = 0.97
            rotation = 90
        else:
            y_position = 0.97 if index % 2 == 0 else 0.78
            rotation = 0
        axis.text(
            event["event_age_ka_bp"],
            y_position,
            event_panel_label(event),
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            rotation=rotation,
            fontsize=6.4,
            color=spec.color,
            linespacing=0.9,
            zorder=5,
        )

    axis.set_xlim(*spec.plot_range_ka)
    axis.set_ylabel(spec.proxy_label)
    axis.set_xlabel("Age (ka BP; older →)")
    axis_note = "; reversed y-axis" if spec.invert_proxy_axis else ""
    axis.set_title(
        f"({panel_letter}) {spec.record_id} — {spec.data_source}; "
        f"Gaussian σ = {spec.nominal_sigma_ka * 1000:.0f} yr{axis_note}",
        loc="left",
    )
    if spec.invert_proxy_axis:
        axis.invert_yaxis()
    style_axis(axis)


def plot_results(
    events: pd.DataFrame,
    segments_by_record: dict[str, list[RegularSegment]],
) -> tuple[Path, Path]:
    configure_plot_style()
    figure = plt.figure(figsize=(180 / 25.4, 236 / 25.4))
    grid = figure.add_gridspec(
        4,
        1,
        height_ratios=[1.25, 2.0, 1.65, 1.65],
        left=0.11,
        right=0.985,
        bottom=0.075,
        top=0.855,
        hspace=0.50,
    )
    axes = [figure.add_subplot(grid[index, 0]) for index in range(4)]

    draw_timeline(axes[0], events)
    for axis, letter, spec in zip(axes[1:], "bcd", RECORDS):
        draw_record_panel(
            axis,
            letter,
            spec,
            segments_by_record[spec.record_id],
            events,
        )

    figure.suptitle(
        "Provisional MIS 6 composite event record",
        x=0.11,
        y=0.985,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    figure.text(
        0.11,
        0.957,
        "Direction-constrained maximum gradient after light Gaussian smoothing",
        ha="left",
        fontsize=8,
        color="#444444",
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            color="#B5B5B5",
            marker=".",
            linewidth=0.7,
            markersize=3,
            label="observations",
        ),
        Line2D([0], [0], color="#333333", linewidth=1.5, label="smoothed proxy"),
        Line2D(
            [0],
            [0],
            color="#777777",
            linestyle=":",
            linewidth=0.8,
            label="literature search anchor",
        ),
        Line2D(
            [0],
            [0],
            color="#333333",
            marker="|",
            linestyle="none",
            markersize=7,
            label="local tuning candidates (not age uncertainty)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#333333",
            markeredgecolor="#333333",
            markersize=4,
            label="event-age QC: stable",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="#333333",
            markersize=4,
            label="event-age QC: moderate",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="#333333",
            markersize=4.5,
            label="event-age QC: review",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.105, 0.938),
        ncol=4,
        frameon=False,
        fontsize=6.7,
        handlelength=2.0,
        columnspacing=1.1,
    )
    figure.text(
        0.5,
        0.018,
        "Short vertical ticks show local algorithm-tuning candidates, not age-model "
        "uncertainty. Proxy changes are local, record-specific, and not directly comparable.",
        ha="center",
        va="bottom",
        fontsize=6.6,
        color="#555555",
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGURE_DIR / f"{FIGURE_STEM}.png"
    pdf_path = FIGURE_DIR / f"{FIGURE_STEM}.pdf"
    figure.savefig(
        pdf_path,
        metadata={
            "Title": "Provisional MIS 6 composite event record",
            "Subject": "Directional-gradient event-age diagnostic",
            "Creator": Path(__file__).name,
        },
    )
    figure.savefig(
        png_path,
        dpi=PNG_DPI,
        metadata={"Software": Path(__file__).name},
    )
    plt.close(figure)
    return png_path, pdf_path


def main() -> None:
    if not WORKBOOK.exists() or not ANCHORS.exists():
        raise FileNotFoundError(
            "Required workbook or published-anchor table is missing"
        )

    selected_anchors = load_selected_anchors(ANCHORS)
    segments_by_record = {}
    for spec in RECORDS:
        record = load_record(WORKBOOK, spec)
        segments_by_record[spec.record_id] = regularize_record(record, spec)

    events = estimate_events(selected_anchors, segments_by_record)
    validate_results(events)
    export_events(events, OUTPUT_CSV)
    png_path, pdf_path = plot_results(events, segments_by_record)

    review = events.loc[
        events["event_age_qc"].eq("review"), "composite_event_display_label"
    ].tolist()
    print(f"Wrote {len(events)} events to {OUTPUT_CSV.relative_to(ROOT)}")
    print(
        f"Wrote figures to {png_path.relative_to(ROOT)} and {pdf_path.relative_to(ROOT)}"
    )
    print(f"Events flagged for review: {', '.join(review) if review else 'none'}")


if __name__ == "__main__":
    main()
