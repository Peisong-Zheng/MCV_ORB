#!/usr/bin/env python3
"""Plot MIS 6 speleothem records, published event labels, and resolution.

Event-label positions are approximate layout anchors, not event-age estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import (
    FuncFormatter,
    LogLocator,
    MaxNLocator,
    MultipleLocator,
    NullFormatter,
)
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
WORKBOOK_PATH = PROJECT_ROOT / "data" / "raw" / "speleothem_data.xlsx"
ANCHOR_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "Speleothem_published_event_plot"
    / "speleothem_mis6_published_event_label_anchors.csv"
)
FIGURE_DIR = PROJECT_ROOT / "figures" / "Speleothem_published_event_plot"
FIGURE_STEM = "Speleothem_published_event_plot"

AGE_RANGE_KA = (125.0, 205.0)
MATERIAL_GAP_KA = 0.5
LABEL_BAND_FRACTION = 0.65
MM_TO_INCH = 1 / 25.4
FIGURE_SIZE_INCH = (180 * MM_TO_INCH, 230 * MM_TO_INCH)
PNG_DPI = 600


@dataclass(frozen=True)
class RecordSpec:
    record_id: str
    sheet: str
    title: str
    proxy_label: str
    color: str
    invert_proxy_axis: bool


RECORD_SPECS = (
    RecordSpec(
        record_id="Sanbao",
        sheet="Sanbao_Wang_etal_2008",
        title="Sanbao · events identified in Held et al. (2024)",
        proxy_label=r"$\delta^{18}\mathrm{O}$ (‰ VPDB)",
        color="#007C91",
        invert_proxy_axis=True,
    ),
    RecordSpec(
        record_id="Huagapo",
        sheet="Huagapo_Burn_etal_2019",
        title="Huagapo · events identified in Held et al. (2024)",
        proxy_label=r"$\delta^{18}\mathrm{O}$ (‰ VPDB)",
        color="#1976D2",
        invert_proxy_axis=False,
    ),
    RecordSpec(
        record_id="Sofular",
        sheet="Sofular_Held_etal_2024",
        title="Sofular · events identified in Held et al. (2024)",
        proxy_label=r"$\delta^{13}\mathrm{C}$ (‰ VPDB)",
        color="#6A1B9A",
        invert_proxy_axis=True,
    ),
    RecordSpec(
        record_id="MF",
        sheet="\ufeffMF_Fohlmeister_etal_2023",
        title="Melchsee-Frutt (MF) · events identified in Fohlmeister et al. (2023)",
        proxy_label=r"$\delta^{18}\mathrm{O}$ (‰ VPDB)",
        color="#C43C39",
        invert_proxy_axis=False,
    ),
)

REQUIRED_ANCHOR_COLUMNS = {
    "record_id",
    "sheet_name",
    "event_label",
    "display_label",
    "anchor_age_ka_bp",
    "anchor_type",
    "label_scheme",
    "source_figure",
    "confidence",
    "anchor_purpose",
}


def configure_plot_style() -> None:
    """Set typography at the final 180 mm publication size."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "axes.titleweight": "semibold",
            "axes.edgecolor": "#4B4B4B",
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def load_anchors(path: Path) -> pd.DataFrame:
    """Load and validate the literature-based label anchors."""
    if not path.exists():
        raise FileNotFoundError(f"Event-anchor table not found: {path}")

    anchors = pd.read_csv(
        path,
        dtype={
            "record_id": "string",
            "sheet_name": "string",
            "event_label": "string",
            "display_label": "string",
        },
    )
    missing = REQUIRED_ANCHOR_COLUMNS.difference(anchors.columns)
    if missing:
        raise ValueError(f"Anchor table is missing columns: {sorted(missing)}")

    anchors["anchor_age_ka_bp"] = pd.to_numeric(
        anchors["anchor_age_ka_bp"], errors="coerce"
    )
    if anchors["anchor_age_ka_bp"].isna().any():
        raise ValueError("Every anchor_age_ka_bp value must be numeric")
    if not anchors["anchor_purpose"].eq("label_placement_only").all():
        raise ValueError(
            "Every event coordinate must be explicitly marked label_placement_only"
        )
    if anchors.duplicated(["record_id", "event_label"]).any():
        raise ValueError("Event labels must be unique within each record")

    expected_ids = {spec.record_id for spec in RECORD_SPECS}
    unexpected_ids = sorted(set(anchors["record_id"].dropna()) - expected_ids)
    if unexpected_ids:
        raise ValueError(f"Unexpected record IDs in anchor table: {unexpected_ids}")

    for spec in RECORD_SPECS:
        local = anchors.loc[anchors["record_id"].eq(spec.record_id)]
        if local.empty:
            raise ValueError(f"No label anchors found for {spec.record_id}")
        sheet_names = set(local["sheet_name"].dropna())
        if sheet_names != {spec.sheet}:
            raise ValueError(
                f"Anchor sheet mismatch for {spec.record_id}: {sorted(sheet_names)!r}"
            )

    return anchors


def load_record(workbook_path: Path, spec: RecordSpec) -> pd.DataFrame:
    """Read, clean, age-sort, and calculate local resolution for one record."""
    raw = pd.read_excel(workbook_path, sheet_name=spec.sheet)
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
    if len(frame) < 2 or not frame["age_ka_bp"].is_monotonic_increasing:
        raise ValueError(f"{spec.sheet!r} has an invalid age axis")

    ages = frame["age_ka_bp"].to_numpy(dtype=float)
    gaps_ka = np.diff(ages)
    resolution_ka = np.empty(len(ages), dtype=float)
    resolution_ka[0], resolution_ka[-1] = gaps_ka[0], gaps_ka[-1]
    if len(ages) > 2:
        resolution_ka[1:-1] = np.median(np.vstack([gaps_ka[:-1], gaps_ka[1:]]), axis=0)
    if np.any(resolution_ka <= 0):
        raise ValueError(f"{spec.sheet!r} produced non-positive resolution")

    frame["resolution_yr"] = resolution_ka * 1000.0
    return frame


def format_log_tick(value: float, _position: int) -> str:
    """Use compact plain-number labels on logarithmic resolution axes."""
    if value <= 0:
        return ""
    if value >= 1000:
        return f"{value / 1000:g}k"
    if value >= 1:
        return f"{value:g}"
    return f"{value:.1g}"


def style_axis(axis: plt.Axes) -> None:
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(length=3.0, width=0.65)


def spread_close_label_positions(event_ages: np.ndarray) -> np.ndarray:
    """Nudge tight label clusters while preserving their approximate order."""
    label_ages = event_ages.astype(float, copy=True)
    close_threshold_ka = 1.3
    edge_shift_ka = 0.45
    for index, event_age in enumerate(event_ages):
        close_to_previous = (
            index > 0 and event_age - event_ages[index - 1] < close_threshold_ka
        )
        close_to_next = (
            index + 1 < len(event_ages)
            and event_ages[index + 1] - event_age < close_threshold_ka
        )
        if close_to_next and not close_to_previous:
            label_ages[index] -= edge_shift_ka
        elif close_to_previous and not close_to_next:
            label_ages[index] += edge_shift_ka
    return label_ages


def plot_with_material_gap_breaks(
    axis: plt.Axes,
    ages: np.ndarray,
    values: np.ndarray,
    **plot_kwargs: object,
) -> None:
    """Plot a series without drawing artificial lines across material gaps."""
    segment_start = 0
    for gap_index in np.flatnonzero(np.diff(ages) > MATERIAL_GAP_KA):
        segment_end = int(gap_index) + 1
        axis.plot(
            ages[segment_start:segment_end],
            values[segment_start:segment_end],
            **plot_kwargs,
        )
        segment_start = segment_end
    axis.plot(ages[segment_start:], values[segment_start:], **plot_kwargs)


def add_material_gap(
    data_axis: plt.Axes,
    resolution_axis: plt.Axes,
    gap_start: float,
    gap_end: float,
) -> None:
    """Shade a material age gap across a record and its resolution panel."""
    for axis in (data_axis, resolution_axis):
        axis.axvspan(
            gap_start,
            gap_end,
            color="#6B7280",
            alpha=0.15,
            linewidth=0,
            zorder=0,
        )
    gap_width = gap_end - gap_start
    data_axis.text(
        (gap_start + gap_end) / 2,
        0.48,
        f"{gap_width:.2f} ka data gap",
        transform=data_axis.get_xaxis_transform(),
        rotation=90,
        ha="center",
        va="center",
        fontsize=7.5,
        color="#4B5563",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.4},
        zorder=6,
    )


def build_figure(records: dict[str, pd.DataFrame], anchors: pd.DataFrame) -> plt.Figure:
    """Build the eight-row static comparison figure."""
    configure_plot_style()
    figure = plt.figure(figsize=FIGURE_SIZE_INCH)
    outer_grid = figure.add_gridspec(
        4,
        1,
        left=0.14,
        right=0.985,
        top=0.975,
        bottom=0.055,
        hspace=0.30,
    )
    axes: list[plt.Axes] = []
    shared_axis: plt.Axes | None = None
    for group_index in range(4):
        pair_grid = outer_grid[group_index].subgridspec(
            2, 1, height_ratios=(3.3, 1.0), hspace=0.06
        )
        data_axis = figure.add_subplot(pair_grid[0], sharex=shared_axis)
        if shared_axis is None:
            shared_axis = data_axis
        resolution_axis = figure.add_subplot(pair_grid[1], sharex=shared_axis)
        axes.extend((data_axis, resolution_axis))
    figure.patch.set_facecolor("white")

    letters = "abcd"
    label_levels = (0.13, 0.33, 0.53)

    for panel_index, spec in enumerate(RECORD_SPECS):
        data_axis = axes[panel_index * 2]
        resolution_axis = axes[panel_index * 2 + 1]
        frame = records[spec.record_id]
        ages = frame["age_ka_bp"].to_numpy(dtype=float)
        proxy = frame["proxy"].to_numpy(dtype=float)
        resolution = frame["resolution_yr"].to_numpy(dtype=float)
        local_anchors = (
            anchors.loc[anchors["record_id"].eq(spec.record_id)]
            .sort_values("anchor_age_ka_bp")
            .reset_index(drop=True)
        )

        if not local_anchors["anchor_age_ka_bp"].between(ages.min(), ages.max()).all():
            raise ValueError(
                f"At least one {spec.record_id} anchor lies outside the record"
            )

        for axis in (data_axis, resolution_axis):
            style_axis(axis)
            axis.set_xlim(*AGE_RANGE_KA)
            axis.xaxis.set_major_locator(MultipleLocator(5))
            axis.grid(
                True,
                axis="x",
                which="major",
                color="#CBD0D6",
                linewidth=0.45,
                alpha=0.7,
            )

        plot_with_material_gap_breaks(
            data_axis,
            ages,
            proxy,
            color=spec.color,
            linewidth=1,
            solid_capstyle="round",
            zorder=2,
        )

        y_min, y_max = float(np.min(proxy)), float(np.max(proxy))
        y_span = y_max - y_min
        if not np.isfinite(y_span) or y_span <= 0:
            raise ValueError(f"{spec.sheet!r} has no usable proxy range")

        # A pale annotation band separates label anchors from the proxy curve.
        if spec.invert_proxy_axis:
            label_band_edge = y_min - LABEL_BAND_FRACTION * y_span
            data_axis.axhspan(
                label_band_edge, y_min, color="#F7F8FA", linewidth=0, zorder=0
            )
            data_axis.axhline(y_min, color="#C9CDD2", linewidth=0.55, zorder=1)
            data_axis.set_ylim(y_max + 0.06 * y_span, label_band_edge)
        else:
            label_band_edge = y_max + LABEL_BAND_FRACTION * y_span
            data_axis.axhspan(
                y_max, label_band_edge, color="#F7F8FA", linewidth=0, zorder=0
            )
            data_axis.axhline(y_max, color="#C9CDD2", linewidth=0.55, zorder=1)
            data_axis.set_ylim(y_min - 0.06 * y_span, label_band_edge)

        y_tick_locator = MaxNLocator(nbins=5)
        y_ticks = y_tick_locator.tick_values(y_min, y_max)
        y_ticks = y_ticks[(y_ticks >= y_min) & (y_ticks <= y_max)]
        data_axis.set_yticks(y_ticks)
        data_axis.set_ylabel(spec.proxy_label, labelpad=5)
        data_axis.set_title(
            spec.title,
            loc="left",
            pad=4,
            color="#202124",
        )
        data_axis.text(
            -0.085,
            1.015,
            f"({letters[panel_index]})",
            transform=data_axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color="#202124",
            clip_on=False,
        )

        event_ages = local_anchors["anchor_age_ka_bp"].to_numpy(dtype=float)
        label_ages = spread_close_label_positions(event_ages)

        for event_index, event in local_anchors.iterrows():
            event_age = float(event["anchor_age_ka_bp"])
            label_age = float(label_ages[event_index])
            curve_y = float(np.interp(event_age, ages, proxy))
            offset = label_levels[event_index % len(label_levels)] * y_span
            label_y = y_min - offset if spec.invert_proxy_axis else y_max + offset
            data_axis.plot(
                [event_age, label_age],
                [curve_y, label_y],
                color="#6F757B",
                linewidth=0.55,
                linestyle=(0, (2, 2.5)),
                alpha=0.6,
                zorder=1.5,
            )
            data_axis.text(
                label_age,
                label_y,
                str(event["display_label"]),
                ha="center",
                va="center",
                fontsize=7.5,
                color="#30343B",
                clip_on=True,
                zorder=4,
            )

        plot_with_material_gap_breaks(
            resolution_axis,
            ages,
            resolution,
            color=spec.color,
            linewidth=0.95,
            alpha=0.96,
            zorder=2,
        )
        resolution_axis.set_yscale("log")
        resolution_axis.set_ylabel("Resolution\n(yr)", labelpad=5)
        resolution_axis.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
        resolution_axis.yaxis.set_major_formatter(FuncFormatter(format_log_tick))
        resolution_axis.yaxis.set_minor_locator(
            LogLocator(base=10, subs=(2, 5), numticks=12)
        )
        resolution_axis.yaxis.set_minor_formatter(NullFormatter())
        resolution_axis.grid(
            True,
            axis="y",
            which="major",
            color="#D0D5DB",
            linewidth=0.45,
            alpha=0.7,
        )
        resolution_axis.grid(
            True,
            axis="y",
            which="minor",
            color="#E5E7EB",
            linewidth=0.35,
            alpha=0.55,
        )

        median_resolution = float(np.median(resolution))
        resolution_axis.axhline(
            median_resolution,
            color="#50555A",
            linewidth=0.65,
            linestyle=(0, (4, 3)),
            alpha=0.72,
            zorder=3,
        )
        resolution_axis.text(
            204.35,
            median_resolution,
            f"median {median_resolution:.1f} yr",
            ha="right",
            va="bottom",
            fontsize=7.2,
            color="#4B5055",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.0},
            zorder=5,
        )

        gaps_ka = np.diff(ages)
        for gap_index in np.flatnonzero(gaps_ka > MATERIAL_GAP_KA):
            add_material_gap(
                data_axis,
                resolution_axis,
                float(ages[gap_index]),
                float(ages[gap_index + 1]),
            )

        data_axis.tick_params(axis="x", labelbottom=False)
        resolution_axis.tick_params(axis="x", labelbottom=True)

    axes[-1].set_xlabel("Age (ka BP)", labelpad=5)
    return figure


def save_figure(figure: plt.Figure) -> tuple[Path, Path]:
    """Write a high-resolution PNG and an editable vector PDF."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGURE_DIR / f"{FIGURE_STEM}.png"
    pdf_path = FIGURE_DIR / f"{FIGURE_STEM}.pdf"
    figure.savefig(png_path, dpi=PNG_DPI)
    figure.savefig(
        pdf_path,
        metadata={
            "Title": "MIS 6 speleothem variability and local sampling resolution",
            "Creator": Path(__file__).name,
        },
    )
    return png_path, pdf_path


def main() -> None:
    if not WORKBOOK_PATH.exists():
        raise FileNotFoundError(f"Workbook not found: {WORKBOOK_PATH}")

    anchors = load_anchors(ANCHOR_PATH)
    records = {
        spec.record_id: load_record(WORKBOOK_PATH, spec) for spec in RECORD_SPECS
    }
    figure = build_figure(records, anchors)
    png_path, pdf_path = save_figure(figure)
    plt.close(figure)

    print(f"Loaded {len(anchors)} label anchors across {len(RECORD_SPECS)} records.")
    print(f"Saved PNG: {png_path.relative_to(PROJECT_ROOT)}")
    print(f"Saved PDF: {pdf_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
