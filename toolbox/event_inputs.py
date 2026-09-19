"""Native climate and orbital input helpers."""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from toolbox.data_checks import require_unique_values
from toolbox.orbital_phase import build_phase_series, evaluate_phase_at_ages
from toolbox.project_config import (
    AGE_EPOCH,
    CO2_EPOCH_SOURCE_URL,
    LR04_EPOCH_SOURCE_URL,
    ORBITAL_AGE_OFFSET_TO_BP1950_KA,
)

_WARNED_LINEAR_EXTRAPOLATION_CONTEXTS: set[str] = set()


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


def clean_series(
    age_ka: np.ndarray, value: np.ndarray, *, context: str = "input series"
) -> tuple[np.ndarray, np.ndarray]:
    """Drop non-finite pairs, require unique ages, and return age-sorted arrays."""

    age_ka = np.asarray(age_ka, dtype=float)
    value = np.asarray(value, dtype=float)
    ok = np.isfinite(age_ka) & np.isfinite(value)
    frame = pd.DataFrame({"age_ka": age_ka[ok], "value": value[ok]})
    require_unique_values(frame, "age_ka", context=context)
    frame = frame.sort_values("age_ka").reset_index(drop=True)
    return frame["age_ka"].to_numpy(dtype=float), frame["value"].to_numpy(dtype=float)


def scale_to_zero_mean_range_one(
    values: np.ndarray,
) -> tuple[np.ndarray, float, float, float, float]:
    """Center a predictor and scale it by its observed range."""

    values = np.asarray(values, dtype=float)
    mean = float(np.nanmean(values))
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    value_range = vmax - vmin
    if not np.isfinite(value_range) or value_range <= 0.0:
        return np.zeros_like(values, dtype=float), mean, vmin, vmax, 1.0
    return (values - mean) / value_range, mean, vmin, vmax, value_range


def require_interpolation_coverage(
    target_age_ka: np.ndarray, source_age_ka: np.ndarray, *, context: str
) -> None:
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
        slope = (source_value[-1] - source_value[-2]) / (
            source_age[-1] - source_age[-2]
        )
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


def load_lr04(
    query_ages_ka: np.ndarray,
    path: Path,
    *,
    project_root: Path | None = None,
) -> tuple[np.ndarray, dict]:
    """Interpolate LR04 to requested ages and return scaled and raw values."""

    raw = pd.read_excel(path)
    age_col = find_column(raw.columns, "time")
    value_col = find_column(raw.columns, "d18o")
    age, value = clean_series(
        raw[age_col].to_numpy(), raw[value_col].to_numpy(), context="LR04 series"
    )
    interpolated = interpolate_checked(query_ages_ka, age, value, context="LR04 series")
    scaled, mean, vmin, vmax, value_range = scale_to_zero_mean_range_one(interpolated)
    meta = {
        "forcing_id": "lr04",
        "forcing_label": "LR04 benthic d18O",
        "source_age_epoch": AGE_EPOCH,
        "age_epoch": AGE_EPOCH,
        "age_offset_ka": 0.0,
        "epoch_source": LR04_EPOCH_SOURCE_URL,
        "source": source_label(path, project_root),
        "mean": mean,
        "min": vmin,
        "max": vmax,
        "range": value_range,
    }
    return scaled, {"raw": interpolated, "meta": meta}


def load_co2(
    query_ages_ka: np.ndarray,
    path: Path,
    *,
    project_root: Path | None = None,
) -> tuple[np.ndarray, dict]:
    """Interpolate atmospheric CO2 to requested ages and return scaled/raw values."""

    raw = pd.read_excel(path, sheet_name="Sheet2")
    age_col = find_column(raw.columns, "gasage")
    value_col = find_column(raw.columns, "co2")
    age, value = clean_series(
        raw[age_col].to_numpy() / 1000.0,
        raw[value_col].to_numpy(),
        context="CO2 series",
    )
    interpolated = interpolate_checked(query_ages_ka, age, value, context="CO2 series")
    scaled, mean, vmin, vmax, value_range = scale_to_zero_mean_range_one(interpolated)
    meta = {
        "forcing_id": "co2",
        "forcing_label": "CO2",
        "source_age_epoch": AGE_EPOCH,
        "age_epoch": AGE_EPOCH,
        "age_offset_ka": 0.0,
        "epoch_source": CO2_EPOCH_SOURCE_URL,
        "source": source_label(path, project_root),
        "mean": mean,
        "min": vmin,
        "max": vmax,
        "range": value_range,
    }
    return scaled, {"raw": interpolated, "meta": meta}


def build_precession_phase(
    query_ages_ka: np.ndarray,
    precession_path: Path,
    *,
    project_root: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate the shared precession phase at requested ages."""

    phase_product = build_phase_series(
        "pre",
        {
            "path": precession_path,
            "label": "Precession index",
            "age_offset_ka": ORBITAL_AGE_OFFSET_TO_BP1950_KA,
            "source": source_label(precession_path, project_root),
        },
    )
    phases = evaluate_phase_at_ages(query_ages_ka, phase_product.extrema)
    pre_at_center = interpolate_checked(
        query_ages_ka,
        phase_product.series["age_ka"].to_numpy(dtype=float),
        phase_product.series["value"].to_numpy(dtype=float),
        context="precession index series",
    )
    phase_table = pd.DataFrame(
        {
            "age_ka": query_ages_ka,
            "precession_index": pre_at_center,
            "pre_phase_unwrapped_rad": phases["phase_unwrapped_rad"],
            "pre_phase_rad": phases["phase_rad"],
            "pre_phase_deg": phases["phase_deg"],
            "pre_phase_sin": np.sin(phases["phase_rad"]),
            "pre_phase_cos": np.cos(phases["phase_rad"]),
            "pre_phase_extrapolated": phases["phase_extrapolated"],
        }
    )
    extrema = phase_product.extrema.rename(columns={"value": "precession_index"})
    extrema = extrema[
        [
            "age_ka",
            "precession_index",
            "extremum_type",
            "half_cycle_index",
            "anchor_phase_unwrapped_rad",
            "anchor_phase_rad",
            "anchor_phase_deg",
        ]
    ]
    return phase_table, extrema
