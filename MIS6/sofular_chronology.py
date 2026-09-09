"""Propagate Sofular U--Th analytical errors on published age--depth curves.

Held et al. (2024), doi:10.1038/s41467-024-45507-5, used StalAge for
individual stalagmites and iscam for their stack. The NOAA source files
(doi:10.25921/b84y-cm81) supply individual curves and dated depths, but not
the iscam transformations or posterior chronology ensembles.

Here independent Gaussian control errors (reported 2 sigma / 2) perturb
the *published model ages* at dated depths. A piecewise affine age warp
preserves the published growth-curve shape. Ordered control knots imply
ordered ages throughout each continuous growth segment. This is working
analytical-error propagation, not a reconstruction of StalAge or iscam.

Stack ages are projected onto the individual published age coordinate.
Such projections locate an error field, not independently established
physical event depths. Mapping lags are explicit sensitivity parameters.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_RAW_DIR = Path(__file__).resolve().parent / "data/raw/Held2024"
SOURCE_FILES = {"So-4": "held2024-so-4.txt", "So-57": "held2024-so-57.txt"}
SOURCE_AGE_FIELDS = {
    "So-4": ("age_corr_kaBP1950", "age_corr_2s_ka", "ka", 1.0),
    "So-57": ("age_corr_BP1950", "age_corr_2s_yr", "yr", 0.001),
}
PROJECTION_METHOD = "assumed_age_coordinate_projection"


@dataclass(frozen=True)
class SofularSource:
    """One original stalagmite: proxy series, full U--Th table, hiatus depths."""

    component: str
    series: pd.DataFrame
    controls: pd.DataFrame
    hiatus: np.ndarray
    source_path: Path


@dataclass(frozen=True)
class SofularChronologyContext:
    """Fixed projection and weights; ages in kyr BP1950 and depth in mm.

    ``nominal_sigma_ka`` and ``nominal_covariance_ka2`` describe proposals
    before conditioning on monotonicity, not the accepted ensemble.
    """

    component: str
    event_ages_ka: np.ndarray
    projected_ages_ka: np.ndarray
    event_depth_mm: np.ndarray
    event_segment_ids: np.ndarray
    interpolation_weights: np.ndarray
    control_sigmas_ka: np.ndarray
    nominal_model_control_ages_ka: np.ndarray
    nominal_sigma_ka: np.ndarray
    nominal_covariance_ka2: np.ndarray
    control_depth_mm: np.ndarray
    control_ids: tuple[str, ...]
    control_segment_ids: np.ndarray
    bracket_left_ids: tuple[str, ...]
    bracket_right_ids: tuple[str, ...]
    mapping_lag_ka: float


def _finite_numeric(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    """Reject missing or nonnumeric inputs rather than dropping source rows."""

    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{label} missing columns: {sorted(missing)}")
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(frame[columns].to_numpy(float)).all():
        raise ValueError(f"{label} contains non-finite values")


def _validate_source(source: SofularSource) -> None:
    """Check the depth and segment contract without sorting or repairing ages."""

    if source.component not in SOURCE_FILES:
        raise ValueError(f"Unknown Sofular component: {source.component}")
    series, controls = source.series, source.controls
    required_series = {"depth_mm", "age_ka_bp", "d13C", "d18O", "segment_id"}
    required_controls = {
        "control_id", "depth_mm", "age_ka_bp", "age_error_2sigma_ka",
        "chronology_sigma_ka", "segment_id",
    }
    if not required_series.issubset(series) or not required_controls.issubset(controls):
        raise ValueError(f"{source.component} source schema is incomplete")
    for label, frame, numeric in (
        ("proxy", series, ["depth_mm", "age_ka_bp", "d13C", "d18O", "segment_id"]),
        ("U--Th", controls, ["depth_mm", "age_ka_bp", "age_error_2sigma_ka",
                             "chronology_sigma_ka", "segment_id"]),
    ):
        if len(frame) < 2 or not np.isfinite(frame[numeric].to_numpy(float)).all():
            raise ValueError(f"{source.component} {label} needs finite data")
        if np.any(np.diff(frame["depth_mm"].to_numpy(float)) <= 0.0):
            raise ValueError(f"{source.component} {label} depths must strictly increase")
    # So-57 contains repeated ages after source rounding near 172.154 ka.
    # Preserve those values; a query exactly on a plateau is not invertible.
    if np.any(np.diff(series["age_ka_bp"].to_numpy(float)) < 0.0):
        raise ValueError(f"{source.component} published proxy ages must not reverse")
    if controls["control_id"].isna().any() or controls["control_id"].duplicated().any():
        raise ValueError(f"{source.component} control IDs must be present and unique")
    if controls["control_id"].astype(str).str.strip().eq("").any():
        raise ValueError(f"{source.component} control IDs cannot be blank")
    if (controls["age_error_2sigma_ka"] <= 0.0).any():
        raise ValueError(f"{source.component} U--Th uncertainties must be positive")
    if not np.allclose(controls["chronology_sigma_ka"], controls["age_error_2sigma_ka"] / 2):
        raise ValueError(f"{source.component} sigma must equal reported 2 sigma / 2")
    hiatus = np.asarray(source.hiatus, dtype=float)
    if hiatus.ndim != 1 or not np.isfinite(hiatus).all() or np.any(np.diff(hiatus) <= 0):
        raise ValueError(f"{source.component} hiatus depths must be finite and ordered")
    if np.any((hiatus <= series.depth_mm.iloc[0]) | (hiatus >= series.depth_mm.iloc[-1])):
        raise ValueError(f"{source.component} hiatus must lie within the proxy depth range")
    for frame in (series, controls):
        depths = frame["depth_mm"].to_numpy(float)
        if np.isin(depths, hiatus).any():
            raise ValueError("A measurement cannot lie exactly on a hiatus boundary")
        if not np.array_equal(frame["segment_id"], np.searchsorted(hiatus, depths)):
            raise ValueError(f"{source.component} segment IDs disagree with hiatus depths")
    # Measured U--Th ages may invert: retain them, and report their residuals.
    # The monotone reference is the published age--depth curve, not those ages.
    if not controls["segment_id"].isin(series["segment_id"].unique()).all():
        raise ValueError(f"{source.component} dated depths have no matching proxy segment")


def _load_source(path: Path, component: str) -> SofularSource:
    """Read the original NOAA proxy and comment-prefixed chronology tables."""

    lines = path.read_text(encoding="utf-8-sig").splitlines()
    header = None
    chronology_rows = []
    for line in lines:
        if line.startswith("# samp_id\t"):
            if header is not None:
                raise ValueError(f"{path.name} contains multiple chronology tables")
            header = next(csv.reader([line[2:]], delimiter="\t"))
        elif header is not None:
            if line.startswith("#---"):
                break
            if line.startswith("# "):
                row = next(csv.reader([line[2:]], delimiter="\t"))
                if len(row) != len(header):
                    raise ValueError(f"{path.name} malformed chronology row")
                chronology_rows.append(row)
    if header is None or not chronology_rows:
        raise ValueError(f"{path.name} missing NOAA chronology table")

    raw_dates = pd.DataFrame(chronology_rows, columns=header)
    age_column, error_column, source_unit, factor = SOURCE_AGE_FIELDS[component]
    required = {"samp_id", "depth_top_mm", age_column, error_column}
    if not required.issubset(raw_dates):
        raise ValueError(f"{path.name} unexpected U--Th schema or age units")
    is_hiatus = raw_dates["samp_id"].eq("Hiatus")
    hiatus = pd.to_numeric(raw_dates.loc[is_hiatus, "depth_top_mm"], errors="raise").to_numpy(float)
    raw_dates = raw_dates.loc[~is_hiatus].copy()
    _finite_numeric(raw_dates, ["depth_top_mm", age_column, error_column], path.name)
    expected_prefix = "So4-" if component == "So-4" else "So57-"
    if not raw_dates["samp_id"].str.startswith(expected_prefix).all():
        raise ValueError(f"{path.name} contains an unexpected sample ID")
    controls = pd.DataFrame({
        "control_id": raw_dates["samp_id"].to_numpy(),
        "component": component,
        "depth_mm": raw_dates["depth_top_mm"].to_numpy(float),
        "age_ka_bp": raw_dates[age_column].to_numpy(float) * factor,
        "age_error_2sigma_ka": raw_dates[error_column].to_numpy(float) * factor,
        "source_age": raw_dates[age_column].to_numpy(float),
        "source_error_2sigma": raw_dates[error_column].to_numpy(float),
        "source_age_unit": source_unit,
        "source_age_column": age_column,
        "source_error_column": error_column,
        "source_filename": path.name,
    })
    controls["chronology_sigma_ka"] = controls["age_error_2sigma_ka"] / 2.0
    controls["segment_id"] = np.searchsorted(hiatus, controls["depth_mm"])

    data_lines = [line for line in lines if line and not line.startswith("#")]
    series = pd.read_csv(StringIO("\n".join(data_lines)), sep="\t")
    if series.columns.tolist() != ["depth_mm", "age_kaBP", "d13C", "d18O"]:
        raise ValueError(f"{path.name} unexpected proxy schema or age units")
    _finite_numeric(series, list(series.columns), path.name)
    series = series.rename(columns={"age_kaBP": "age_ka_bp"})
    series["segment_id"] = np.searchsorted(hiatus, series["depth_mm"])
    source = SofularSource(component, series, controls, hiatus, path)
    _validate_source(source)
    return source


def load_sources(raw_dir: Path = DEFAULT_RAW_DIR) -> dict[str, SofularSource]:
    """Load both NOAA files, explicitly converting So-57 U--Th years to kyr."""

    return {
        component: _load_source(Path(raw_dir) / filename, component)
        for component, filename in SOURCE_FILES.items()
    }


def _model_ages_at_controls(source: SofularSource) -> np.ndarray:
    # So4-M19 lies 0.2 mm beyond the young proxy segment. Retain the date,
    # but do not extrapolate a model age for it or use it as a warp knot.
    ages = np.full(len(source.controls), np.nan, dtype=float)
    for segment_id, part in source.series.groupby("segment_id", sort=False):
        selected = (
            source.controls["segment_id"].eq(segment_id)
            & source.controls["depth_mm"].between(part.depth_mm.iloc[0], part.depth_mm.iloc[-1])
        ).to_numpy()
        ages[selected] = np.interp(
            source.controls.loc[selected, "depth_mm"], part["depth_mm"], part["age_ka_bp"]
        )
    return ages


def build_context(
    event_ages: np.ndarray,
    component: str = "So-4",
    mapping_lag_ka: float = 0.0,
    sources: dict[str, SofularSource] | None = None,
) -> SofularChronologyContext:
    """Project stack ages and prepare a monotone warp within dated support.

    Positive ``mapping_lag_ka`` projects onto an older individual-record
    coordinate; it changes only the sampled error field, not stack point ages.
    No inverse interpolation crosses a hiatus or extrapolates beyond dates.
    """

    ages = np.array(event_ages, dtype=float, copy=True)
    if ages.ndim != 1 or ages.size == 0 or not np.isfinite(ages).all():
        raise ValueError("Event ages must be a nonempty finite one-dimensional array")
    if np.any(np.diff(ages) <= 0.0):
        raise ValueError("Event ages must strictly increase in published order")
    if not np.isscalar(mapping_lag_ka) or not np.isfinite(mapping_lag_ka):
        raise ValueError("Mapping lag must be a finite scalar in kyr")
    source_map = load_sources() if sources is None else sources
    if component not in source_map:
        raise ValueError(f"Unavailable Sofular component: {component}")
    source = source_map[component]
    if source.component != component:
        raise ValueError("Source component does not match its dictionary key")
    _validate_source(source)
    projected = ages + float(mapping_lag_ka)
    depths = np.empty(len(ages))
    event_segments = np.empty(len(ages), dtype=int)
    for index, age in enumerate(projected):
        matches = [
            (segment_id, part)
            for segment_id, part in source.series.groupby("segment_id", sort=False)
            if part.age_ka_bp.iloc[0] <= age <= part.age_ka_bp.iloc[-1]
        ]
        if len(matches) != 1:
            raise ValueError(f"{component} projection {age:.6f} ka is outside proxy support or in a hiatus")
        segment_id, part = matches[0]
        tied_ages = part.loc[part["age_ka_bp"].duplicated(keep=False), "age_ka_bp"].to_numpy(float)
        if np.isclose(tied_ages, age, rtol=0, atol=1e-12).any():
            raise ValueError(f"{component} projection has a non-unique depth at a repeated published age")
        depths[index] = np.interp(age, part["age_ka_bp"], part["depth_mm"])
        event_segments[index] = segment_id

    all_model_ages = _model_ages_at_controls(source)
    selected = (
        source.controls["segment_id"].isin(np.unique(event_segments)).to_numpy()
        & np.isfinite(all_model_ages)
    )
    controls = source.controls.loc[selected].reset_index(drop=True)
    model_ages = all_model_ages[selected]
    if np.any(np.diff(model_ages) <= 0):
        raise ValueError(f"{component} published ages at dated knots must strictly increase")
    weights = np.zeros((len(ages), len(controls)))
    left_ids, right_ids = [], []
    for index, (depth, age, segment_id) in enumerate(zip(depths, projected, event_segments, strict=True)):
        local = np.flatnonzero(controls["segment_id"].eq(segment_id).to_numpy())
        if len(local) < 2:
            raise ValueError(f"{component} growth segment needs two dated depths")
        local_depth = controls.loc[local, "depth_mm"].to_numpy(float)
        if depth < local_depth[0] or depth > local_depth[-1]:
            raise ValueError(f"{component} projection at {depth:.6f} mm is outside dated support")
        position = int(np.clip(np.searchsorted(local_depth, depth, side="right") - 1, 0, len(local) - 2))
        left, right = local[position], local[position + 1]
        fraction = (age - model_ages[left]) / (model_ages[right] - model_ages[left])
        weights[index, left], weights[index, right] = 1.0 - fraction, fraction
        left_ids.append(str(controls.loc[left, "control_id"]))
        right_ids.append(str(controls.loc[right, "control_id"]))
    if np.any(weights < -1e-12) or not np.allclose(weights.sum(axis=1), 1.0):
        raise RuntimeError("Sofular interpolation weights violate dated support")
    sigmas = controls["chronology_sigma_ka"].to_numpy(float)
    covariance = (weights * sigmas**2) @ weights.T
    return SofularChronologyContext(
        component=component, event_ages_ka=ages, projected_ages_ka=projected,
        event_depth_mm=depths, event_segment_ids=event_segments,
        interpolation_weights=weights, control_sigmas_ka=sigmas,
        nominal_model_control_ages_ka=model_ages,
        nominal_sigma_ka=np.sqrt(np.diag(covariance)), nominal_covariance_ka2=covariance,
        control_depth_mm=controls["depth_mm"].to_numpy(float),
        control_ids=tuple(controls["control_id"].astype(str)),
        control_segment_ids=controls["segment_id"].to_numpy(int),
        bracket_left_ids=tuple(left_ids), bracket_right_ids=tuple(right_ids),
        mapping_lag_ka=float(mapping_lag_ka),
    )


def propose_offsets(
    context: SofularChronologyContext, rng: np.random.Generator
) -> tuple[np.ndarray | None, np.ndarray]:
    """Draw once; return ``None`` if ordered dated knots are violated.

    The caller decides whether to reject the entire joint chronology. There
    are no hidden retries, sorting repairs, or independent event-age draws.
    """

    errors = np.asarray(rng.normal(0.0, context.control_sigmas_ka), dtype=float)
    if errors.shape != context.control_sigmas_ka.shape or not np.isfinite(errors).all():
        raise ValueError("Control-error proposal has invalid shape or non-finite values")
    knots = context.nominal_model_control_ages_ka + errors
    if np.any(np.diff(knots) <= 0.0):
        return None, errors
    return context.interpolation_weights @ errors, errors


def control_diagnostics(sources: dict[str, SofularSource]) -> pd.DataFrame:
    """Retain complete controls, units, and departures of model from measured age."""

    frames = []
    for component, source in sources.items():
        _validate_source(source)
        frame = source.controls.copy()
        frame["component"] = component
        frame["nominal_model_age_ka_bp"] = _model_ages_at_controls(source)
        frame["model_age_available"] = np.isfinite(frame["nominal_model_age_ka_bp"])
        frame["model_minus_uth_age_ka"] = frame["nominal_model_age_ka_bp"] - frame["age_ka_bp"]
        frame["model_minus_uth_in_reported_sigma"] = frame["model_minus_uth_age_ka"] / frame["chronology_sigma_ka"]
        frame["source_path"] = str(source.source_path)
        frame["age_epoch"] = "BP1950"
        frames.append(frame)
    if not frames:
        raise ValueError("Control diagnostics require at least one source")
    return pd.concat(frames, ignore_index=True)


def projection_diagnostics(context: SofularChronologyContext) -> pd.DataFrame:
    """Document projected coordinates, bracket identity, and proposal-only sigma."""

    frame = pd.DataFrame({
        "event_index": np.arange(1, len(context.event_ages_ka) + 1),
        "component": context.component,
        "event_age_ka_bp": context.event_ages_ka,
        "projection_method": PROJECTION_METHOD,
        "mapping_lag_ka": context.mapping_lag_ka,
        "projected_age_ka_bp": context.projected_ages_ka,
        "projected_depth_mm": context.event_depth_mm,
        "segment_id": context.event_segment_ids,
        "bracket_left_id": context.bracket_left_ids,
        "bracket_right_id": context.bracket_right_ids,
        "unconditioned_sigma_ka": context.nominal_sigma_ka,
    })
    lookup = {name: index for index, name in enumerate(context.control_ids)}
    for side, ids in (("left", context.bracket_left_ids), ("right", context.bracket_right_ids)):
        indices = np.array([lookup[name] for name in ids])
        frame[f"bracket_{side}_depth_mm"] = context.control_depth_mm[indices]
        frame[f"bracket_{side}_model_age_ka_bp"] = context.nominal_model_control_ages_ka[indices]
        frame[f"weight_{side}"] = context.interpolation_weights[np.arange(len(frame)), indices]
    return frame
