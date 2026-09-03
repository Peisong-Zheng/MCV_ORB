"""Reusable orbital-phase construction, Rayleigh tests, and phase plots.

Phase convention
----------------
Local minima of an orbital series are phase 0, local maxima are phase pi,
and phase increases linearly between successive extrema.  Wrapped phases are
reported on [0, 2*pi); unwrapped phases are retained for interpolation.

This module has no knowledge of the project output directories.  Plotting
functions return Matplotlib figures, leaving file naming and saving to the
calling analysis script.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from toolbox.event_inputs import interpolate_checked


@dataclass
class OrbitalPhase:
    """An orbital series together with the extrema that anchor its phase."""

    driver: str
    label: str
    series: pd.DataFrame
    extrema: pd.DataFrame


def load_orbital_series(
    path: Path,
    driver: str,
    label: str,
    *,
    source: str | None = None,
) -> pd.DataFrame:
    """Read a two-column orbital file and convert its signed ages to ka BP."""

    raw = pd.read_csv(path, sep=r"\s+", header=None, names=["age_raw_ka", "value"])
    orbital = pd.DataFrame(
        {
            "driver": driver,
            "driver_label": label,
            # The Laskar files use negative ages for the past.
            "age_ka": -pd.to_numeric(raw["age_raw_ka"], errors="coerce"),
            "value": pd.to_numeric(raw["value"], errors="coerce"),
            "source": source if source is not None else str(path),
        }
    ).dropna(subset=["age_ka", "value"])
    return orbital.sort_values("age_ka").reset_index(drop=True)


def _enforce_alternating_extrema(extrema: pd.DataFrame) -> pd.DataFrame:
    """Collapse adjacent extrema of the same type to the more extreme point."""

    rows: list[pd.Series] = []
    for _, row in extrema.sort_values("age_ka").iterrows():
        if not rows or row["extremum_type"] != rows[-1]["extremum_type"]:
            rows.append(row.copy())
            continue

        previous = rows[-1]
        is_more_extreme = (
            row["value"] > previous["value"]
            if row["extremum_type"] == "maximum"
            else row["value"] < previous["value"]
        )
        if is_more_extreme:
            rows[-1] = row.copy()

    out = pd.DataFrame(rows).reset_index(drop=True)
    out["half_cycle_index"] = np.arange(len(out))
    return out


def _detect_extrema(orbital: pd.DataFrame) -> pd.DataFrame:
    """Find an alternating sequence of local minima and maxima."""

    values = orbital["value"].to_numpy(dtype=float)
    maxima = orbital.iloc[find_peaks(values)[0]].copy()
    minima = orbital.iloc[find_peaks(-values)[0]].copy()
    maxima["extremum_type"] = "maximum"
    minima["extremum_type"] = "minimum"

    extrema = pd.concat([minima, maxima], ignore_index=True).sort_values("age_ka")
    extrema = _enforce_alternating_extrema(extrema)
    if len(extrema) < 3:
        driver = orbital["driver"].iloc[0] if len(orbital) else "orbital series"
        raise ValueError(f"Too few extrema detected for {driver}.")
    return extrema


def _assign_anchor_phases(extrema: pd.DataFrame) -> pd.DataFrame:
    """Assign 0 to minima, pi to maxima, and unwrap successive half-cycles."""

    out = extrema.reset_index(drop=True).copy()
    first_phase = 0.0 if out.loc[0, "extremum_type"] == "minimum" else np.pi
    out["anchor_phase_unwrapped_rad"] = first_phase + np.arange(len(out)) * np.pi
    out["anchor_phase_rad"] = np.mod(out["anchor_phase_unwrapped_rad"], 2 * np.pi)
    out["anchor_phase_deg"] = np.mod(np.degrees(out["anchor_phase_rad"]), 360.0)
    out.loc[np.isclose(out["anchor_phase_deg"], 360.0), "anchor_phase_deg"] = 0.0
    return out


def interpolate_unwrapped_phase(
    ages_ka: np.ndarray,
    extrema_age_ka: np.ndarray,
    extrema_phase_unwrapped_rad: np.ndarray,
    warn_on_extrapolation: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate phase and mark ages outside the extrema range.

    ``numpy.interp`` normally clamps values outside its source range.  Phase is
    instead linearly extrapolated from the first or last two anchors so that it
    continues advancing at the local half-cycle rate.  The Boolean return value
    makes those extrapolated phases explicit to callers.
    """

    ages = np.asarray(ages_ka, dtype=float)
    anchor_ages = np.asarray(extrema_age_ka, dtype=float)
    anchor_phases = np.asarray(extrema_phase_unwrapped_rad, dtype=float)
    if len(anchor_ages) < 2 or len(anchor_ages) != len(anchor_phases):
        raise ValueError("Phase interpolation needs at least two paired anchors.")
    if np.any(~np.isfinite(anchor_ages)) or np.any(np.diff(anchor_ages) <= 0):
        raise ValueError("Phase-anchor ages must be finite and strictly increasing.")

    phase = np.interp(ages, anchor_ages, anchor_phases)
    left = ages < anchor_ages[0]
    right = ages > anchor_ages[-1]
    extrapolated = left | right

    if np.any(left):
        slope = (anchor_phases[1] - anchor_phases[0]) / (
            anchor_ages[1] - anchor_ages[0]
        )
        phase[left] = anchor_phases[0] + slope * (ages[left] - anchor_ages[0])
    if np.any(right):
        slope = (anchor_phases[-1] - anchor_phases[-2]) / (
            anchor_ages[-1] - anchor_ages[-2]
        )
        phase[right] = anchor_phases[-1] + slope * (ages[right] - anchor_ages[-1])

    if warn_on_extrapolation and np.any(extrapolated):
        warnings.warn(
            "The phase extrema do not fully cover the requested ages; "
            "extrapolated phase values are used.",
            stacklevel=2,
        )
    return phase, extrapolated


def evaluate_phase_at_ages(
    ages_ka: np.ndarray,
    extrema: pd.DataFrame,
    warn_on_extrapolation: bool = False,
) -> pd.DataFrame:
    """Evaluate wrapped and unwrapped phase at arbitrary ages."""

    ages = np.asarray(ages_ka, dtype=float)
    unwrapped, extrapolated = interpolate_unwrapped_phase(
        ages,
        extrema["age_ka"].to_numpy(dtype=float),
        extrema["anchor_phase_unwrapped_rad"].to_numpy(dtype=float),
        warn_on_extrapolation,
    )
    wrapped = np.mod(unwrapped, 2 * np.pi)
    return pd.DataFrame(
        {
            "age_ka": ages,
            "phase_unwrapped_rad": unwrapped,
            "phase_rad": wrapped,
            "phase_deg": np.degrees(wrapped),
            "phase_fraction": wrapped / (2 * np.pi),
            "phase_extrapolated": extrapolated,
        }
    )


def build_phase_series(driver: str, settings: dict) -> OrbitalPhase:
    """Build the phase series described by one settings dictionary.

    Required settings are ``path`` and ``label``.  An optional human-readable
    ``source`` is copied into output tables; this avoids coupling the toolbox to
    a particular repository root.
    """

    orbital = load_orbital_series(
        Path(settings["path"]),
        driver,
        settings["label"],
        source=settings.get("source"),
    )
    extrema = _assign_anchor_phases(_detect_extrema(orbital))
    phases = evaluate_phase_at_ages(orbital["age_ka"].to_numpy(), extrema)

    # Keep only the phase columns needed for the regularly sampled series.
    orbital = orbital.copy()
    for column in [
        "phase_unwrapped_rad",
        "phase_rad",
        "phase_deg",
        "phase_extrapolated",
    ]:
        orbital[column] = phases[column].to_numpy()
    return OrbitalPhase(driver, settings["label"], orbital, extrema)


def _circular_distance(phases_rad: np.ndarray, target_rad: float) -> np.ndarray:
    """Return signed shortest angular distances from a target phase."""

    return np.angle(np.exp(1j * (phases_rad - target_rad)))


def sample_event_phases(
    events: pd.DataFrame,
    phase_products: dict[str, OrbitalPhase],
) -> pd.DataFrame:
    """Interpolate every orbital value and phase at every event age."""

    required = {"event_age_ka", "event_type", "event_label", "event_index"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"Event table is missing columns: {sorted(missing)}")
    if events.empty:
        raise ValueError("Event table is empty.")

    events = events.reset_index(drop=True)
    ages = events["event_age_ka"].to_numpy(dtype=float)
    frames = []
    for product in phase_products.values():
        phases = evaluate_phase_at_ages(ages, product.extrema, True)
        sampled = events.copy()
        sampled["driver"] = product.driver
        sampled["driver_label"] = product.label
        sampled["orbital_value_at_event"] = interpolate_checked(
            ages,
            product.series["age_ka"].to_numpy(dtype=float),
            product.series["value"].to_numpy(dtype=float),
            context=f"{product.label} series",
        )
        for column in [
            "phase_unwrapped_rad",
            "phase_rad",
            "phase_deg",
            "phase_fraction",
            "phase_extrapolated",
        ]:
            sampled[column] = phases[column].to_numpy()

        theta = sampled["phase_rad"].to_numpy(dtype=float)
        sampled["signed_distance_to_min_rad"] = _circular_distance(theta, 0.0)
        sampled["signed_distance_to_max_rad"] = _circular_distance(theta, np.pi)
        sampled["source_event_order"] = sampled.get("order", np.nan)
        frames.append(
            sampled[
                [
                    "driver",
                    "driver_label",
                    "event_type",
                    "event_label",
                    "event_index",
                    "event_age_ka",
                    "orbital_value_at_event",
                    "phase_unwrapped_rad",
                    "phase_rad",
                    "phase_deg",
                    "phase_fraction",
                    "phase_extrapolated",
                    "signed_distance_to_min_rad",
                    "signed_distance_to_max_rad",
                    "source_event_order",
                ]
            ]
        )
    if not frames:
        raise ValueError("At least one orbital phase product is required.")
    return pd.concat(frames, ignore_index=True)


def rayleigh_p_value_from_z(z: float, n: int) -> float:
    """Return the finite-sample Rayleigh upper-tail p value."""

    if n <= 0 or not np.isfinite(z):
        return np.nan

    # Large-sample Rayleigh approximation:
    #     p ≈ exp(-z)
    #
    # This is the probability, under uniform phases, of obtaining a resultant
    # vector at least as concentrated as the observed one. It is accurate when n
    # is large, but our event catalogues have only ~100 events, so we apply the
    # usual expansion in powers of 1/n:
    #
    #     p ≈ exp(-z) * [1 + O(1/n) + O(1/n^2)]
    #
    # The first correction term is (2z - z^2)/(4n), and the second correction
    # term is the polynomial divided by 288 n^2. These terms slightly adjust the
    # p value for finite event counts without changing the Rayleigh statistic z.
    # Reference:
    # Fisher, N. I. (1993). Statistical Analysis of Circular Data. Cambridge University Press.
    # https://metricgate.com/docs/rayleigh-uniformity-test/
    p = np.exp(-z) * (
        1.0
        + (2.0 * z - z**2) / (4.0 * n)
        - (24.0 * z - 132.0 * z**2 + 76.0 * z**3 - 9.0 * z**4)
        / (288.0 * n**2)
    )
    return float(np.clip(p, 0.0, 1.0))


def rayleigh_test(phases_rad: np.ndarray) -> dict[str, float]:
    """Test whether circular phases depart from a uniform distribution."""

    theta = np.asarray(phases_rad, dtype=float)
    theta = theta[np.isfinite(theta)]
    n = len(theta)
    if n == 0:
        return {
            "n_phase_events_used": 0,
            "mean_phase_rad": np.nan,
            "mean_phase_deg": np.nan,
            "mean_resultant_length": np.nan,
            "rayleigh_R": np.nan,
            "rayleigh_z": np.nan,
            "rayleigh_p": np.nan,
        }

    cosine_sum = float(np.cos(theta).sum())
    sine_sum = float(np.sin(theta).sum())
    resultant = float(np.hypot(cosine_sum, sine_sum))
    mean_phase = float(np.mod(np.arctan2(sine_sum, cosine_sum), 2 * np.pi))
    mean_resultant_length = resultant / n
    z = n * mean_resultant_length**2
    return {
        "n_phase_events_used": n,
        "mean_phase_rad": mean_phase,
        "mean_phase_deg": float(np.degrees(mean_phase)),
        "mean_resultant_length": mean_resultant_length,
        "rayleigh_R": resultant,
        "rayleigh_z": z,
        "rayleigh_p": rayleigh_p_value_from_z(z, n),
    }


def rayleigh_rbar_threshold(n: int, alpha: float = 0.05) -> float:
    """Return the mean-resultant-length threshold for ``p <= alpha``."""

    if n <= 0:
        return np.nan
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie between 0 and 1.")
    low, high = 0.0, 1.0
    if rayleigh_p_value_from_z(n, n) > alpha:
        return np.nan
    for _ in range(80):
        middle = (low + high) / 2
        if rayleigh_p_value_from_z(n * middle**2, n) <= alpha:
            high = middle
        else:
            low = middle
    return high


def build_rayleigh_results(event_phases: pd.DataFrame) -> pd.DataFrame:
    """Apply the Rayleigh test to each driver and event-type catalogue."""

    group_columns = ["driver", "driver_label", "event_type", "event_label"]
    rows = []
    for keys, group in event_phases.groupby(group_columns, sort=False):
        used = group.loc[~group["phase_extrapolated"].astype(bool), "phase_rad"]
        result = rayleigh_test(used.to_numpy(dtype=float))
        result.update(dict(zip(group_columns, keys)))
        result["n_events_total"] = len(group)
        result["n_extrapolated_phase_events"] = int(
            group["phase_extrapolated"].sum()
        )
        rows.append(result)

    columns = [
        *group_columns,
        "n_events_total",
        "n_phase_events_used",
        "n_extrapolated_phase_events",
        "mean_phase_rad",
        "mean_phase_deg",
        "mean_resultant_length",
        "rayleigh_R",
        "rayleigh_z",
        "rayleigh_p",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)[columns].sort_values(
        ["driver", "event_type"]
    ).reset_index(drop=True)


def plot_phase_check(
    phase_products: dict[str, OrbitalPhase],
    *,
    age_range: tuple[float, float],
    driver_colors: dict[str, str],
) -> plt.Figure:
    """Plot orbital extrema and their resulting wrapped phase curves."""

    fig, axes = plt.subplots(len(phase_products), 2, figsize=(13, 6.8), sharex="col")
    axes = np.atleast_2d(axes)
    for row, product in enumerate(phase_products.values()):
        series = product.series[product.series["age_ka"].between(*age_range)]
        extrema = product.extrema[product.extrema["age_ka"].between(*age_range)]
        color = driver_colors[product.driver]
        value_axis, phase_axis = axes[row]

        value_axis.plot(series["age_ka"], series["value"], color=color, lw=1.2)
        for kind, marker, face, label in [
            ("minimum", "v", "white", "min = phase 0"),
            ("maximum", "^", color, "max = phase pi"),
        ]:
            points = extrema[extrema["extremum_type"].eq(kind)]
            value_axis.scatter(
                points["age_ka"],
                points["value"],
                s=28,
                marker=marker,
                facecolors=face,
                edgecolors=color,
                linewidths=0.9,
                label=label,
                zorder=3,
            )
        value_axis.set_title(f"{product.label}: detected extrema", loc="left")
        value_axis.set_ylabel("Raw orbital value")
        value_axis.grid(True, color="#e6e6e6", lw=0.7)
        value_axis.legend(frameon=False, loc="best")

        phase_axis.plot(series["age_ka"], series["phase_deg"], color=color, lw=1.1)
        phase_axis.scatter(
            extrema["age_ka"],
            extrema["anchor_phase_deg"],
            s=14,
            color="#202020",
            alpha=0.8,
            label="extremum anchors",
        )
        phase_axis.set(yticks=[0, 90, 180, 270, 360], ylim=(-8, 368))
        phase_axis.set_title(f"{product.label}: wrapped phase", loc="left")
        phase_axis.set_ylabel("Phase (deg)")
        phase_axis.grid(True, color="#e6e6e6", lw=0.7)
        phase_axis.legend(frameon=False, loc="best")

    for axis in axes[-1]:
        axis.set_xlabel("Age (ka BP)")
    fig.suptitle("Orbital extrema and phase-conversion check", y=0.995, fontsize=14)
    fig.subplots_adjust(top=0.90, hspace=0.30, wspace=0.22)
    return fig


def plot_rayleigh_polar(
    event_phases: pd.DataFrame,
    results: pd.DataFrame,
    *,
    driver: str,
    event_colors: dict[str, str],
    alpha: float = 0.05,
    radial_max: float | None = None,
    annotate_mean_phase: bool = False,
    label_threshold_on_circle: bool = False,
) -> plt.Figure:
    """Plot phase histograms, mean vectors, and the Rayleigh threshold."""

    event_types = list(event_colors)
    fig, axes = plt.subplots(
        1,
        len(event_types),
        figsize=(7.2, 3.4),
        subplot_kw={"projection": "polar"},
    )
    axes = np.atleast_1d(axes)
    bins = np.linspace(0, 2 * np.pi, 19)  # 18 equal 20-degree bins
    theta_grid = np.linspace(0, 2 * np.pi, 361)

    for panel, (axis, event_type) in enumerate(zip(axes, event_types)):
        selected = (
            event_phases["driver"].eq(driver)
            & event_phases["event_type"].eq(event_type)
            & ~event_phases["phase_extrapolated"].astype(bool)
        )
        phases = event_phases.loc[selected]
        result = results[
            results["driver"].eq(driver) & results["event_type"].eq(event_type)
        ].iloc[0]
        counts, _ = np.histogram(phases["phase_rad"], bins=bins)
        axis.bar(
            bins[:-1],
            counts,
            width=np.diff(bins)[0],
            align="edge",
            color=event_colors[event_type],
            alpha=0.55,
            edgecolor="white",
            linewidth=0.8,
        )

        # Histogram counts set the radial scale, so multiply Rbar by the same
        # maximum count when drawing the mean vector and p=alpha threshold.
        maximum_count = max(counts.max(), 1)
        n = int(result["n_phase_events_used"])
        mean_phase = float(result["mean_phase_rad"])
        mean_length = float(result["mean_resultant_length"])
        threshold = rayleigh_rbar_threshold(n, alpha)
        threshold_radius = threshold * maximum_count
        if np.isfinite(threshold_radius):
            axis.plot(
                theta_grid,
                np.full_like(theta_grid, threshold_radius),
                color="#303030",
                linestyle=(0, (4, 2)),
                lw=1.15,
                alpha=0.9,
                zorder=4,
            )
        if label_threshold_on_circle and np.isfinite(threshold_radius):
            axis.text(
                np.deg2rad(135),
                threshold_radius * 1.98,
                rf"$\bar{{R}}_{{{alpha:.2f}}}$",
                ha="center",
                va="center",
                fontsize=8,
                color="#303030",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.2},
                zorder=5,
            )
        axis.annotate(
            "",
            xy=(mean_phase, mean_length * maximum_count),
            xytext=(mean_phase, 0),
            arrowprops={"arrowstyle": "->", "lw": 2.0, "color": "#202020"},
        )

        axis.set_theta_zero_location("E")
        axis.set_theta_direction(1)
        axis.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
        axis.set_xticklabels(["min", "90", "max", "270"])
        axis.set_rlabel_position(35)
        axis.tick_params(axis="x", pad=4, labelsize=9)
        axis.tick_params(axis="y", labelsize=8)
        if radial_max is not None:
            axis.set_ylim(0, radial_max)
            axis.set_yticks(np.arange(2, radial_max, 2))
            axis.grid(True, color="#b8b8b8", lw=0.7, alpha=0.42)

        axis.text(
            0.03,
            0.98,
            chr(ord("a") + panel),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.6},
        )
        if not label_threshold_on_circle and np.isfinite(threshold):
            axis.text(
                0.02,
                0.06,
                rf"$\bar{{R}}_{{{alpha:.2f}}}$={threshold:.2f}",
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontsize=8,
                color="#303030",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.4},
            )
        if annotate_mean_phase:
            axis.text(
                mean_phase,
                mean_length * maximum_count + 1.05,
                f"{float(result['mean_phase_deg']):.1f}°",
                ha="left",
                va="center",
                fontsize=8.5,
                color="#202020",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.0},
                zorder=5,
            )
        axis.set_title(
            f"{phases['event_label'].iloc[0]}\n"
            rf"N={n}, $\bar{{R}}$={mean_length:.2f}, p={float(result['rayleigh_p']):.3f}",
            va="bottom",
            fontsize=9.5,
            pad=30,
        )

    fig.subplots_adjust(top=0.84, bottom=0.08, left=0.05, right=0.95, wspace=0.28)
    return fig


def plot_phase_ecdf(
    event_phases: pd.DataFrame,
    *,
    driver_labels: dict[str, str],
    event_colors: dict[str, str],
) -> plt.Figure:
    """Plot phase empirical CDFs against the circular-uniform expectation."""

    fig, axes = plt.subplots(1, len(driver_labels), figsize=(12.5, 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for panel, (axis, (driver, label)) in enumerate(zip(axes, driver_labels.items())):
        selected = event_phases[
            event_phases["driver"].eq(driver)
            & ~event_phases["phase_extrapolated"].astype(bool)
        ]
        for event_type, group in selected.groupby("event_type", sort=False):
            phases = np.sort(group["phase_fraction"].to_numpy(dtype=float))
            empirical_cdf = np.arange(1, len(phases) + 1) / len(phases)
            axis.step(
                phases,
                empirical_cdf,
                where="post",
                color=event_colors[event_type],
                lw=1.5,
                label=group["event_label"].iloc[0],
            )
        axis.plot([0, 1], [0, 1], color="#666666", ls="--", lw=1, label="uniform")
        axis.set_title(f"{label} phase ECDF", loc="left")
        axis.set_xlabel("Phase fraction (0=min, 0.5=max)")
        axis.grid(True, color="#e6e6e6", lw=0.7)
        axis.text(
            -0.04,
            1.04,
            chr(ord("a") + panel),
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            clip_on=False,
        )
    axes[0].set_ylabel("Empirical CDF")
    axes[-1].legend(frameon=False, loc="lower right")
    fig.subplots_adjust(top=0.88, wspace=0.18)
    return fig
