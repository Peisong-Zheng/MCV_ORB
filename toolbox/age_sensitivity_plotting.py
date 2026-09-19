"""Five-panel chronological sensitivity plot; no fitting or file I/O."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from toolbox.figure_style import add_panel_label
from toolbox.age_sensitivity import unwrap_phase as unwrap_around
P_THRESHOLD=.05

def configure_plot_style():
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                         'font.size':9,'axes.labelsize':9,'pdf.fonttype':42,'ps.fonttype':42})

def _histogram_panel(
    ax: plt.Axes,
    values: np.ndarray,
    point_value: float,
    xlabel: str,
    panel_label: str,
    *,
    threshold: float | None = None,
    log_x: bool = False,
) -> None:
    """Draw one compact MC distribution with point and median references."""

    values = np.asarray(values, dtype=float)
    if not len(values):
        ax.text(0.5, 0.5, "No valid fits", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel(xlabel)
        return
    if log_x:
        positive = values[values > 0.0]
        if len(positive) != len(values):
            raise ValueError("A logarithmic histogram requires positive values")
        lower = 10 ** np.floor(np.log10(positive.min()))
        bins = np.geomspace(lower, 1.0, 42)
    else:
        bins = 38

    ax.hist(values, bins=bins, color="#4477AA", alpha=0.82, edgecolor="none")
    if log_x:
        ax.set_xscale("log")
    ax.axvline(point_value, color="black", lw=1.25, zorder=3)
    ax.axvline(np.median(values), color="#CC6677", lw=1.25, ls="--", zorder=3)
    if threshold is not None:
        ax.axvline(threshold, color="#777777", lw=1.0, ls=":", zorder=3)
    ax.set_xlabel(xlabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)
    ax.set_axisbelow(True)
    add_panel_label(ax, panel_label, x=-0.15, y=1.04)

def plot_sensitivity(
    results: pd.DataFrame,
    point_fit,
):
    """Plot the five quantities used to assess age-uncertainty sensitivity."""

    configure_plot_style()
    n_realizations = len(results)
    results = results.loc[results["fit_valid"].eq(True)]
    n_valid = len(results)
    point = point_fit.summary
    phase = unwrap_around(
        results["pre_phase_preferred_deg"].to_numpy(float),
        float(point["pre_phase_preferred_deg"]),
    )

    fig = plt.figure(figsize=(180 / 25.4, 120 / 25.4))
    grid = fig.add_gridspec(2, 6, hspace=0.56, wspace=0.72)
    axes = [
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[0, 4:6]),
        fig.add_subplot(grid[1, 1:3]),
        fig.add_subplot(grid[1, 3:5]),
    ]

    _histogram_panel(
        axes[0],
        results["gain_bits_per_event"].to_numpy(float),
        float(point["gain_bits_per_event"]),
        "Gain (bits event$^{-1}$)",
        "a",
    )
    _histogram_panel(
        axes[1],
        results["nominal_LR_p"].to_numpy(float),
        float(point["nominal_LR_p"]),
        "Nominal likelihood-ratio p",
        "b",
        threshold=P_THRESHOLD,
        log_x=True,
    )
    n_below = int(results["nominal_LR_p"].lt(P_THRESHOLD).sum())
    axes[1].text(
        0.98,
        0.94,
        f"p < 0.05\n{n_below:,}/{len(results):,} "
        f"({n_below / n_valid:.2%})" if n_valid else "No valid fits",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.78,
            "pad": 1.5,
        },
    )
    _histogram_panel(
        axes[2],
        phase,
        float(point["pre_phase_preferred_deg"]),
        "Preferred precession phase (°)",
        "c",
    )
    _histogram_panel(
        axes[3],
        results["pre_phase_rate_ratio_max_vs_min"].to_numpy(float),
        float(point["pre_phase_rate_ratio_max_vs_min"]),
        "Phase rate ratio (maximum/minimum)",
        "d",
    )
    _histogram_panel(
        axes[4],
        results["delta_AIC_full_minus_reduced"].to_numpy(float),
        float(point["delta_AIC_full_minus_reduced"]),
        r"$\Delta$AIC (full $-$ reduced)",
        "e",
        threshold=0.0,
    )
    axes[0].set_ylabel("Monte Carlo realizations")
    axes[3].set_ylabel("Monte Carlo realizations")

    handles = [
        Line2D([0], [0], color="black", lw=1.25, label="Point ages"),
        Line2D([0], [0], color="#CC6677", lw=1.25, ls="--", label="MC median"),
        Line2D([0], [0], color="#777777", lw=1.0, ls=":", label="Decision threshold"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.005),
    )
    fig.text(0.5, 0.025,
             f"Valid fits: {n_valid:,}/{n_realizations:,}; outside observation support: {n_realizations - n_valid:,}",
             ha="center", fontsize=8)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.17, top=0.91)

    return fig
