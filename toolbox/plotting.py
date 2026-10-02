"""Shared panel styling, phase-response labels and chronology-sensitivity plots."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from toolbox.model_stats import unwrap_phase as unwrap_around

PANEL_FONT_SIZE = 9
PANEL_FONT_FAMILY = "DejaVu Sans"
P_THRESHOLD = 0.05


def add_panel_label(axis, label, x=-0.12, y=1.04):
    """Keep letters separate from titles; positions can follow the panel layout."""
    return axis.text(
        x, y, f"({label})", transform=axis.transAxes,
        fontsize=PANEL_FONT_SIZE, fontfamily=PANEL_FONT_FAMILY,
        fontweight="bold", color="black", ha="left", va="bottom",
        clip_on=False, zorder=10,
    )


def format_phase_response_axis(axis, fontsize=7):
    """The extrema named below the ticks belong to the precession index."""
    axis.set_xlim(0, 360)
    axis.set_xticks([0, 90, 180, 270, 360],
                   ["0°\nMinimum", "90°", "180°\nMaximum", "270°", "360°\nMinimum"])
    axis.tick_params(axis="x", labelsize=fontsize)
    # Keep the longer endpoint labels inside narrow publication panels.
    axis.get_xticklabels()[0].set_ha("left")
    axis.get_xticklabels()[-1].set_ha("right")
    axis.set_xlabel("Precession-index phase", labelpad=3)
    axis.set_ylabel("Rate multiplier")


def mark_preferred_phase(axis, phase_deg, max_min_ratio, color="black"):
    """For exp(a sin(phi) + b cos(phi)), the peak multiplier is sqrt(max/min)."""
    axis.plot(phase_deg, max_min_ratio ** 0.5, "o", color=color,
              markersize=3.5, markeredgecolor="white", markeredgewidth=0.4, zorder=5)


def configure_barker_style():
    """Use readable journal-scale typography and editable PDF fonts."""

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9.5,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def configure_plot_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


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
    point,
):
    """Plot the five quantities used to assess age-uncertainty sensitivity."""

    configure_plot_style()
    n_realizations = len(results)
    results = results.loc[results["fit_valid"].eq(True)]
    n_valid = len(results)
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


def plot_effect_uncertainty(summary, curves, regions, *, compact_ratio_ticks=False):
    """Compose the shared uncertainty figure from saved scientific products."""
    from matplotlib.lines import Line2D
    from toolbox.model_stats import ellipse_boundary, SCENARIOS
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(180 / 25.4, 145 / 25.4))
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.12, top=0.88, hspace=0.65, wspace=0.85)
    colors = {"A_chronology": "#747474", "B_sampling": "#0072B2", "C_joint": "#D55E00"}
    labels = {"A_chronology": "Chronology", "B_sampling": "Sampling", "C_joint": "Sampling + chronology"}
    styles = {"A_chronology": ":", "B_sampling": "-", "C_joint": "--"}
    fig.legend([Line2D([], [], color=colors[key], linestyle=styles[key], linewidth=1.6) for key in colors],
               list(labels.values()), loc="upper center", bbox_to_anchor=(0.54, 0.985),
               ncol=3, fontsize=8, frameon=False, columnspacing=1.5)
    for ax, quantity, xlabel, title in (
        (axes[0, 0], "preferred_phase_deg", "Preferred phase (°; unwrapped)", "Phase uncertainty"),
        (axes[0, 1], "max_min_rate_ratio", "Maximum/minimum rate ratio", "Effect strength"),
    ):
        for i, scenario in enumerate(colors):
            row = summary.loc[summary.scenario.eq(scenario) & summary.quantity.eq(quantity)].iloc[0]
            ax.hlines(i, row.low, row.high, color=colors[scenario], linewidth=2)
            ax.plot(row.center, i, "o", color=colors[scenario], markersize=4)
        ax.axvline(row.point_estimate, color="black", linestyle="--", linewidth=0.9)
        ax.set(yticks=range(3), yticklabels=["Chronology", "Sampling", "Sampling +\nchronology"],
               xlabel=xlabel, ylim=(2.5, -0.5))
        ax.tick_params(axis="y", labelsize=8)
        ax.set_title(title, fontsize=9, fontweight="normal", pad=9)
        if quantity == "max_min_rate_ratio":
            ax.set_xscale("log")
            ax.axvline(1, color="#999999", linewidth=0.7)
            if compact_ratio_ticks:
                ax.set_xticks([1, 2, 3, 5, 10], labels=["1", "2", "3", "5", "10"])
            else:
                ax.set_xticks([1, 3, 10, 30], labels=["1", "3", "10", "30"])
            ax.minorticks_off()
    ax = axes[1, 0]
    for scenario, region in regions.items():
        boundary = ellipse_boundary(region, np.linspace(0, 2 * np.pi, 721))
        ax.plot(boundary[:, 1], boundary[:, 0], color=colors[scenario],
                linestyle=styles[scenario], linewidth=1.6)
    ax.plot(region["center"][1], region["center"][0], "ko", markersize=4)
    ax.plot(0, 0, "+", color="black", markersize=7, zorder=5)
    ax.axhline(0, color="#dddddd", linewidth=0.6)
    ax.axvline(0, color="#dddddd", linewidth=0.6)
    ax.set(xlabel="Cosine coefficient", ylabel="Sine coefficient")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("95% coefficient regions", fontsize=9, fontweight="normal", pad=9)
    ax = axes[1, 1]
    ax.fill_between(curves.phase_deg, curves.sampling_low, curves.sampling_high,
                    color=colors["B_sampling"], alpha=0.16)
    for scenario, name in SCENARIOS.items():
        for bound in ("low", "high"):
            ax.plot(curves.phase_deg, curves[f"{name}_{bound}"], color=colors[scenario],
                    linestyle=styles[scenario], linewidth=1.1)
    ax.plot(curves.phase_deg, curves.point_multiplier, color="black", linewidth=1.2, label="Point fit")
    ax.axhline(1, color="#999999", linewidth=0.6)
    format_phase_response_axis(ax)
    point_phase = summary.loc[summary.quantity.eq("preferred_phase_deg"), "point_estimate"].iloc[0]
    point_ratio = summary.loc[summary.quantity.eq("max_min_rate_ratio"), "point_estimate"].iloc[0]
    mark_preferred_phase(ax, point_phase, point_ratio)
    ax.set_title("Conditional phase effect", fontsize=9, fontweight="normal", pad=9)
    for letter, ax in zip("abcd", axes.flat):
        add_panel_label(ax, letter, x=-0.16, y=1.04)
        ax.grid(False)
        ax.spines[["top", "right"]].set_visible(False)
    return fig


def plot_orbital_comparisons(summary, catalogue_label):
    """Compact paper panels; BG, Pre and Orb are defined in the figure caption."""
    from matplotlib.ticker import MultipleLocator

    DRIVER_LABELS = {"ecc": "Eccentricity", "obl": "Obliquity",
                     "insol65n": "65°N summer-solstice\ninsolation"}
    DRIVER_COLORS = {"ecc": "#CC79A7", "obl": "#009E73", "insol65n": "#D55E00"}
    plt.rcParams.update({"font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 8,
        "axes.linewidth": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(180 / 25.4, 82 / 25.4), sharey=True)
    fig.subplots_adjust(left=0.205, right=0.985, bottom=0.29, top=0.75, wspace=0.16)
    specs = [
        ("driver_after_base", r"$\mathrm{BG}+\mathrm{Orb}$" + "\n" + r"vs $\mathrm{BG}$"),
        ("driver_after_phase", r"$\mathrm{BG}+\mathrm{Pre}+\mathrm{Orb}$" + "\n" +
         r"vs $\mathrm{BG}+\mathrm{Pre}$"),
        ("phase_after_driver", r"$\mathrm{BG}+\mathrm{Pre}+\mathrm{Orb}$" + "\n" +
         r"vs $\mathrm{BG}+\mathrm{Orb}$"),
    ]
    reference = summary.set_index("comparison_id").loc["phase_reference"]
    # A common minimum span keeps both catalogues directly comparable on reruns.
    upper = np.nanmax(summary[["gain_bits_per_event_point", "gain_bits_per_event_q975"]])
    xmax = max(0.30, np.ceil(upper / 0.05) * 0.05 + 0.01)
    for panel, (ax, (group, title)) in enumerate(zip(axes, specs)):
        subset = summary.loc[summary.comparison_group.eq(group)].set_index("driver_id")
        for y, driver in enumerate(DRIVER_LABELS):
            row = subset.loc[driver]
            color = DRIVER_COLORS[driver]
            low, high = row.gain_bits_per_event_q025, row.gain_bits_per_event_q975
            if np.isfinite([low, high]).all():
                ax.hlines(y, low, high, color=color, linewidth=2.3, alpha=0.55)
                ax.plot(row.gain_bits_per_event_median, y, "|", color=color,
                        markersize=11, markeredgewidth=1.5)
            if np.isfinite(row.gain_bits_per_event_point):
                ax.plot(row.gain_bits_per_event_point, y, "o", color=color,
                        markersize=5, markeredgecolor="white", markeredgewidth=0.5)
        if panel != 1:
            ax.axvline(reference.gain_bits_per_event_point, color="0.35",
                       linewidth=0.9, linestyle=(0, (3, 3)), zorder=0)
        ax.set(xlim=(-0.009, xmax), ylim=(2.48, -0.48), xlabel="Gain (bits event$^{-1}$)")
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        ax.set_title(title, fontsize=8.5, pad=10)
        ax.text(0, 1.23, f"({chr(97 + panel)})", transform=ax.transAxes,
                fontweight="bold", fontsize=9)
        ax.grid(axis="x", linewidth=0.45, color="0.90", zorder=0)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=8)
    axes[0].set_yticks(range(3), list(DRIVER_LABELS.values()))
    fig.text(0.205, 0.94, catalogue_label, weight="bold", fontsize=10)
    handles = [Line2D([], [], marker="o", linestyle="none", color="0.25", label="Point ages"),
               Line2D([], [], marker="|", linestyle="none", markersize=10, color="0.25", label="MC median"),
               Line2D([], [], linewidth=2.3, alpha=0.55, color="0.25", label="95% MC range"),
               Line2D([], [], linestyle="--", color="0.35",
                      label=r"$\mathrm{BG}+\mathrm{Pre}$ vs $\mathrm{BG}$ (point ages)")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.035),
               frameon=False, ncol=2, columnspacing=2.0, fontsize=7.5)
    return fig, axes
