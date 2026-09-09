"""Shared MIS 6 record preparation and directional-gradient event picking.

Used by MIS6_composite_event_record.ipynb and MIS6_event_age_uncertainty.py.
Ages are in kyr BP1950; smoothing and search widths are in kyr.
This module writes no files.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

# The 1-year grid is an interpolation grid, not the sampling resolution.
GRID_STEP_KA = 0.001
MATERIAL_GAP_KA = 0.5
NOMINAL_SEARCH_HALF_WIDTH_KA = 0.4
SENSITIVITY_SEARCH_HALF_WIDTHS_KA = (0.3, 0.4, 0.5)
MF_EVENT_LABELS = tuple(f"6.{number}" for number in range(1, 17))
SOFULAR_EVENT_LABELS = ("6.9", "6.10", "6.11", "6.12", "6.13")
N_EVENTS = len(MF_EVENT_LABELS) + len(SOFULAR_EVENT_LABELS)


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


# Derivatives are taken toward increasing BP age (backward in time).
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
        plot_range_ka=(175.2, 195.2),
        invert_proxy_axis=True,
    ),
)
RECORD_BY_ID = {record.record_id: record for record in RECORDS}


def configure_plot_style() -> None:
    """Use legible typography at the final 180 mm figure width."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 9,
            "axes.labelsize": 9.5,
            "axes.titlesize": 9.5,
            "axes.titleweight": "semibold",
            "axes.linewidth": 0.7,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
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
    if mf["display_label"].tolist() != list(MF_EVENT_LABELS):
        raise ValueError("Expected the complete MF 6.1–6.16 sequence within 130–175 ka")

    held = anchors.loc[
        anchors["record_id"].eq("Sofular")
        & anchors["display_label"].isin(SOFULAR_EVENT_LABELS)
    ].sort_values("anchor_age_ka_bp")
    if held["display_label"].tolist() != list(SOFULAR_EVENT_LABELS):
        raise ValueError("Expected Sofular/Held events 6.9–6.13")

    mf = mf.copy()
    held = held.copy()
    mf["composite_event_display_label"] = mf["display_label"]
    held["composite_event_display_label"] = [
        f"6.{number}" for number in range(17, N_EVENTS + 1)
    ]
    selected = pd.concat([mf, held], ignore_index=True)
    selected = selected.rename(columns={"display_label": "source_event_display_label"})
    selected = selected.sort_values("anchor_age_ka_bp").reset_index(drop=True)

    if (
        len(selected) != N_EVENTS
        or selected["composite_event_display_label"].duplicated().any()
    ):
        raise ValueError(f"The composite chronology must contain {N_EVENTS} unique events")
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
