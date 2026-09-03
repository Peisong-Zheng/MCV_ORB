"""Input preparation helpers for binned climate-event analyses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from toolbox.data_checks import require_unique_values

_WARNED_LINEAR_EXTRAPOLATION_CONTEXTS: set[str] = set()
_WARNED_EDGE_FILL_CONTEXTS: set[str] = set()


@dataclass
class EventDataset:
    """One event catalogue represented by event ages in kyr BP."""

    dataset_id: str
    label: str
    color: str
    ages_ka: np.ndarray
    source: str


def source_label(path: Path, project_root: Path | None = None) -> str:
    """Return a readable source path, relative to project_root when possible."""

    path = Path(path)
    if project_root is None:
        return str(path)
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def find_column(columns: pd.Index, *needles: str) -> str:
    """Return the first column whose normalized name contains all needles."""

    normalized = {str(column).strip().lower(): column for column in columns}
    for key, column in normalized.items():
        if all(needle.lower() in key for needle in needles):
            return str(column)
    raise ValueError(f"Cannot find a column containing {needles}.")


def clean_series(age_ka: np.ndarray, value: np.ndarray, *, context: str = "input series") -> tuple[np.ndarray, np.ndarray]:
    """Drop non-finite pairs, require unique ages, and return age-sorted arrays."""

    age_ka = np.asarray(age_ka, dtype=float)
    value = np.asarray(value, dtype=float)
    ok = np.isfinite(age_ka) & np.isfinite(value)
    frame = pd.DataFrame({"age_ka": age_ka[ok], "value": value[ok]})
    require_unique_values(frame, "age_ka", context=context)
    frame = frame.sort_values("age_ka").reset_index(drop=True)
    return frame["age_ka"].to_numpy(dtype=float), frame["value"].to_numpy(dtype=float)


def scale_to_zero_mean_range_one(values: np.ndarray) -> tuple[np.ndarray, float, float, float, float]:
    """Center a predictor and scale it by its observed range."""

    values = np.asarray(values, dtype=float)
    mean = float(np.nanmean(values))
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    value_range = vmax - vmin
    if not np.isfinite(value_range) or value_range <= 0.0:
        return np.zeros_like(values, dtype=float), mean, vmin, vmax, 1.0
    return (values - mean) / value_range, mean, vmin, vmax, value_range


def require_interpolation_coverage(target_age_ka: np.ndarray, source_age_ka: np.ndarray, *, context: str) -> None:
    """Raise if interpolation targets extend beyond the source age range."""

    target = np.asarray(target_age_ka, dtype=float)
    source = np.asarray(source_age_ka, dtype=float)
    if len(source) < 2:
        raise ValueError(f"{context} needs at least two source ages for interpolation.")
    target_min = float(np.nanmin(target))
    target_max = float(np.nanmax(target))
    source_min = float(np.nanmin(source))
    source_max = float(np.nanmax(source))
    if target_min < source_min or target_max > source_max:
        raise ValueError(
            f"{context} does not cover requested ages: targets {target_min:.3f}-{target_max:.3f} kyr, "
            f"source {source_min:.3f}-{source_max:.3f} kyr."
        )


def interpolate_checked(
    target_age_ka: np.ndarray,
    source_age_ka: np.ndarray,
    source_value: np.ndarray,
    *,
    context: str,
) -> np.ndarray:
    """Interpolate only when source ages fully cover the requested targets."""

    require_interpolation_coverage(target_age_ka, source_age_ka, context=context)
    return np.interp(target_age_ka, source_age_ka, source_value)


def interpolate_with_linear_extrapolation(
    target_age_ka: np.ndarray,
    source_age_ka: np.ndarray,
    source_value: np.ndarray,
    *,
    context: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate and explicitly linearly extrapolate edge targets if needed.

    This is used for phase anchors, where a small edge extrapolation is more
    defensible than allowing ``np.interp`` to silently clamp to the first or
    last phase anchor. The returned Boolean mask marks extrapolated targets.
    """

    target = np.asarray(target_age_ka, dtype=float)
    source_age = np.asarray(source_age_ka, dtype=float)
    source_value = np.asarray(source_value, dtype=float)
    if len(source_age) < 2:
        raise ValueError(f"{context} needs at least two source ages for interpolation.")
    out = np.interp(target, source_age, source_value)
    extrapolated = (target < source_age[0]) | (target > source_age[-1])

    left = target < source_age[0]
    if np.any(left):
        slope = (source_value[1] - source_value[0]) / (source_age[1] - source_age[0])
        out[left] = source_value[0] + slope * (target[left] - source_age[0])

    right = target > source_age[-1]
    if np.any(right):
        slope = (source_value[-1] - source_value[-2]) / (source_age[-1] - source_age[-2])
        out[right] = source_value[-1] + slope * (target[right] - source_age[-1])

    if np.any(extrapolated) and context not in _WARNED_LINEAR_EXTRAPOLATION_CONTEXTS:
        _WARNED_LINEAR_EXTRAPOLATION_CONTEXTS.add(context)
        warnings.warn(
            f"{context} linearly extrapolated {int(extrapolated.sum())} target ages beyond "
            f"{source_age[0]:.3f}-{source_age[-1]:.3f} kyr.",
            RuntimeWarning,
            stacklevel=2,
        )
    return out, extrapolated


def interpolate_with_edge_fill(
    target_age_ka: np.ndarray,
    source_age_ka: np.ndarray,
    source_value: np.ndarray,
    *,
    context: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate and explicitly fill edge targets with nearest endpoint values.

    This preserves the behavior that ``np.interp`` would otherwise apply
    silently, but records the operation and warns the caller. It is appropriate
    for nuisance controls such as local sampling resolution when a bin center
    falls just outside the last observed age.
    """

    target = np.asarray(target_age_ka, dtype=float)
    source_age = np.asarray(source_age_ka, dtype=float)
    source_value = np.asarray(source_value, dtype=float)
    if len(source_age) < 2:
        raise ValueError(f"{context} needs at least two source ages for interpolation.")
    out = np.interp(target, source_age, source_value)
    edge_filled = (target < source_age[0]) | (target > source_age[-1])
    if np.any(edge_filled) and context not in _WARNED_EDGE_FILL_CONTEXTS:
        _WARNED_EDGE_FILL_CONTEXTS.add(context)
        warnings.warn(
            f"{context} edge-filled {int(edge_filled.sum())} target ages beyond "
            f"{source_age[0]:.3f}-{source_age[-1]:.3f} kyr.",
            RuntimeWarning,
            stacklevel=2,
        )
    return out, edge_filled


def load_event_catalogues(
    dataset_settings: dict,
    *,
    analysis_start_ka: float,
    analysis_end_ka: float,
    project_root: Path | None = None,
) -> list[EventDataset]:
    """Load all event catalogues described by a dataset-settings dictionary."""

    event_catalogues: list[EventDataset] = []
    for dataset_id, settings in dataset_settings.items():
        path = Path(settings["path"])
        frame = pd.read_csv(path, encoding="utf-8-sig")
        if "start_time_ka_BP" not in frame.columns:
            raise ValueError(f"{path} must contain start_time_ka_BP.")

        ages = pd.to_numeric(frame["start_time_ka_BP"], errors="coerce").dropna().to_numpy(dtype=float)
        ages = np.sort(ages[(ages >= analysis_start_ka) & (ages <= analysis_end_ka)])
        event_catalogues.append(
            EventDataset(
                dataset_id=dataset_id,
                label=settings["label"],
                color=settings["color"],
                ages_ka=ages,
                source=source_label(path, project_root),
            )
        )

    return event_catalogues


def make_bin_edges(analysis_start_ka: float, analysis_end_ka: float, bin_width_ka: float) -> np.ndarray:
    """Return rounded bin edges for a fixed-width age grid."""

    edges = np.arange(analysis_start_ka, analysis_end_ka + bin_width_ka / 2.0, bin_width_ka)
    edges[-1] = analysis_end_ka
    return np.round(edges, 10)


def load_lr04(
    centers_ka: np.ndarray,
    path: Path,
    *,
    project_root: Path | None = None,
) -> tuple[np.ndarray, dict]:
    """Interpolate LR04 to bin centers and return scaled and raw values."""

    raw = pd.read_excel(path)
    age_col = find_column(raw.columns, "time")
    value_col = find_column(raw.columns, "d18o")
    age, value = clean_series(raw[age_col].to_numpy(), raw[value_col].to_numpy(), context="LR04 series")
    interpolated = interpolate_checked(centers_ka, age, value, context="LR04 series")
    scaled, mean, vmin, vmax, value_range = scale_to_zero_mean_range_one(interpolated)
    meta = {
        "forcing_id": "lr04",
        "forcing_label": "LR04 benthic d18O",
        "source": source_label(path, project_root),
        "mean": mean,
        "min": vmin,
        "max": vmax,
        "range": value_range,
    }
    return scaled, {"raw": interpolated, "meta": meta}


def load_co2(
    centers_ka: np.ndarray,
    path: Path,
    *,
    project_root: Path | None = None,
) -> tuple[np.ndarray, dict]:
    """Interpolate atmospheric CO2 to bin centers and return scaled/raw values."""

    raw = pd.read_excel(path, sheet_name="Sheet2")
    age_col = find_column(raw.columns, "gasage")
    value_col = find_column(raw.columns, "co2")
    age, value = clean_series(
        raw[age_col].to_numpy() / 1000.0,
        raw[value_col].to_numpy(),
        context="CO2 series",
    )
    interpolated = interpolate_checked(centers_ka, age, value, context="CO2 series")
    scaled, mean, vmin, vmax, value_range = scale_to_zero_mean_range_one(interpolated)
    meta = {
        "forcing_id": "co2",
        "forcing_label": "CO2",
        "source": source_label(path, project_root),
        "mean": mean,
        "min": vmin,
        "max": vmax,
        "range": value_range,
    }
    return scaled, {"raw": interpolated, "meta": meta}


def load_precession_series(path: Path, *, project_root: Path | None = None) -> pd.DataFrame:
    """Load the raw precession index with ages in kyr BP."""

    raw = pd.read_csv(path, sep=r"\s+", header=None, names=["age_raw_ka", "value"])
    age, value = clean_series(
        -raw["age_raw_ka"].to_numpy(),
        raw["value"].to_numpy(),
        context="precession index series",
    )
    return pd.DataFrame({"age_ka": age, "precession_index": value})


def enforce_alternating_extrema(extrema: pd.DataFrame) -> pd.DataFrame:
    """Keep a clean min/max/min/max sequence for phase anchoring."""

    rows: list[pd.Series] = []
    for _, row in extrema.sort_values("age_ka").iterrows():
        if not rows:
            rows.append(row.copy())
            continue
        prev = rows[-1]
        if row["extremum_type"] != prev["extremum_type"]:
            rows.append(row.copy())
            continue
        if row["extremum_type"] == "maximum":
            if float(row["precession_index"]) > float(prev["precession_index"]):
                rows[-1] = row.copy()
        else:
            if float(row["precession_index"]) < float(prev["precession_index"]):
                rows[-1] = row.copy()
    out = pd.DataFrame(rows).reset_index(drop=True)
    out["half_cycle_index"] = np.arange(len(out))
    return out


def build_precession_phase(
    centers_ka: np.ndarray,
    precession_path: Path,
    *,
    project_root: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert the raw precession index into circular phase at bin centers.

    Minima of the precession index are assigned phase 0 and maxima are assigned
    phase pi. Between extrema, the phase is linearly interpolated in unwrapped
    radians, then wrapped and represented as sine/cosine columns.
    """

    pre = load_precession_series(precession_path, project_root=project_root)
    values = pre["precession_index"].to_numpy(dtype=float)
    max_idx, _ = find_peaks(values)
    min_idx, _ = find_peaks(-values)

    maxima = pre.iloc[max_idx].copy()
    maxima["extremum_type"] = "maximum"
    minima = pre.iloc[min_idx].copy()
    minima["extremum_type"] = "minimum"
    extrema = pd.concat([minima, maxima], ignore_index=True).sort_values("age_ka")
    extrema = enforce_alternating_extrema(extrema)
    if len(extrema) < 3:
        raise ValueError("Too few precession extrema detected.")

    first_phase = 0.0 if extrema.loc[0, "extremum_type"] == "minimum" else np.pi
    extrema["anchor_phase_unwrapped_rad"] = first_phase + np.arange(len(extrema), dtype=float) * np.pi
    extrema["anchor_phase_rad"] = np.mod(extrema["anchor_phase_unwrapped_rad"], 2.0 * np.pi)
    extrema["anchor_phase_deg"] = np.mod(np.degrees(extrema["anchor_phase_rad"]), 360.0)
    extrema.loc[np.isclose(extrema["anchor_phase_deg"], 360.0), "anchor_phase_deg"] = 0.0

    phase_unwrapped, phase_extrapolated = interpolate_with_linear_extrapolation(
        centers_ka,
        extrema["age_ka"].to_numpy(dtype=float),
        extrema["anchor_phase_unwrapped_rad"].to_numpy(dtype=float),
        context="precession phase anchors",
    )
    phase_rad = np.mod(phase_unwrapped, 2.0 * np.pi)
    pre_at_center = interpolate_checked(
        centers_ka,
        pre["age_ka"].to_numpy(dtype=float),
        pre["precession_index"].to_numpy(dtype=float),
        context="precession index series",
    )
    phase_table = pd.DataFrame(
        {
            "age_ka": centers_ka,
            "precession_index": pre_at_center,
            "pre_phase_unwrapped_rad": phase_unwrapped,
            "pre_phase_rad": phase_rad,
            "pre_phase_deg": np.degrees(phase_rad),
            "pre_phase_sin": np.sin(phase_rad),
            "pre_phase_cos": np.cos(phase_rad),
            "pre_phase_extrapolated": phase_extrapolated,
        }
    )
    return phase_table, extrema


def build_binned_inputs(
    events: list[EventDataset],
    *,
    analysis_start_ka: float,
    analysis_end_ka: float,
    bin_width_ka: float,
    lr04_path: Path,
    co2_path: Path,
    precession_path: Path,
    project_root: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the binned event-count table and shared climate/orbital inputs."""

    edges = make_bin_edges(analysis_start_ka, analysis_end_ka, bin_width_ka)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dt = np.diff(edges)

    lr04_scaled, lr04_info = load_lr04(centers, lr04_path, project_root=project_root)
    co2_scaled, co2_info = load_co2(centers, co2_path, project_root=project_root)
    phase_table, phase_extrema = build_precession_phase(
        centers,
        precession_path,
        project_root=project_root,
    )

    base = pd.DataFrame(
        {
            "bin_start_ka": edges[:-1],
            "bin_end_ka": edges[1:],
            "bin_center_ka": centers,
            "dt_ka": dt,
            "lr04": lr04_info["raw"],
            "lr04_scaled": lr04_scaled,
            "co2": co2_info["raw"],
            "co2_scaled": co2_scaled,
        }
    )
    base = pd.concat([base, phase_table.drop(columns=["age_ka"])], axis=1)

    rows = []
    for dataset in events:
        counts, _ = np.histogram(dataset.ages_ka, bins=edges)
        dataset_frame = base.copy()
        dataset_frame.insert(0, "dataset_id", dataset.dataset_id)
        dataset_frame.insert(1, "dataset_label", dataset.label)
        dataset_frame["event_count"] = counts.astype(int)
        dataset_frame["n_events_total"] = int(len(dataset.ages_ka))
        dataset_frame["source"] = dataset.source
        rows.append(dataset_frame)

    scale_summary = pd.DataFrame(
        [
            lr04_info["meta"],
            co2_info["meta"],
            {
                "forcing_id": "pre_phase_sin",
                "forcing_label": "sin(precession phase)",
                "source": source_label(precession_path, project_root),
                "mean": float(np.nanmean(base["pre_phase_sin"])),
                "min": float(np.nanmin(base["pre_phase_sin"])),
                "max": float(np.nanmax(base["pre_phase_sin"])),
                "range": float(np.nanmax(base["pre_phase_sin"]) - np.nanmin(base["pre_phase_sin"])),
            },
            {
                "forcing_id": "pre_phase_cos",
                "forcing_label": "cos(precession phase)",
                "source": source_label(precession_path, project_root),
                "mean": float(np.nanmean(base["pre_phase_cos"])),
                "min": float(np.nanmin(base["pre_phase_cos"])),
                "max": float(np.nanmax(base["pre_phase_cos"])),
                "range": float(np.nanmax(base["pre_phase_cos"]) - np.nanmin(base["pre_phase_cos"])),
            },
        ]
    )
    return pd.concat(rows, ignore_index=True), scale_summary, phase_extrema
