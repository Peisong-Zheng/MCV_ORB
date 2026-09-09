"""Compose manuscript Figs. 1, 2 and 4 from saved analysis tables.

Run from the project root. This script only draws figures: event ages, phase
coefficients, bootstrap results and age-MC summaries are never refitted.
"""

from pathlib import Path
import hashlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

from paper_figure_export import copy_pdf_to_paper

PROJECT = Path(__file__).resolve().parent
FIGURES = PROJECT / "figures/paper_summary"
PRIMARY = Path("data/processed/NGRIP_MIS6_event_phase_analysis")
BARKER = Path("Barker2011/data/processed/Barker2011_event_phase_analysis")
PRIMARY_ORBITAL = Path("data/processed/NGRIP_MIS6_orbital_driver_sensitivity")
BARKER_ORBITAL = Path("Barker2011/data/processed/Barker2011_orbital_driver_sensitivity")
COLORS = {"NGRIP": "#3E6C8E", "MF": "#7A6AA6", "Sofular": "#D58936",
          "variable": "#CC6677", "fixed": "#4477AA"}
DRIVERS = {"ecc": ("Eccentricity", "#CC79A7"),
           "obl": ("Obliquity", "#009E73"),
           "insol65n": ("65°N summer-solstice\ninsolation", "#D55E00")}
INPUTS = set()


def read_table(relative_path):
    """Track the saved input files used to compose the manuscript figures."""
    path = PROJECT / relative_path
    INPUTS.add(path)
    return pd.read_csv(path)


def style_axes(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(direction="out", width=0.6, length=3)


def panel_label(axis, label, x=-0.12, y=1.06):
    axis.text(x, y, f"({label})", transform=axis.transAxes,
              weight="bold", fontsize=9, ha="left", va="bottom")


def save_figure(figure, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    pdf = FIGURES / f"{name}.pdf"
    figure.savefig(pdf, facecolor="white")
    figure.savefig(FIGURES / f"{name}.png", dpi=400, facecolor="white")
    copy_pdf_to_paper(pdf)
    plt.close(figure)


def plot_data_overview():
    """Show event membership, observation support and fixed model inputs."""
    events = read_table(Path("data/curated/ngrip_mis6_warming_events.csv"))
    barker_events = read_table(BARKER / "event_catalogue_used.csv")
    support = read_table(PRIMARY_ORBITAL / "support.csv").set_index("segment_id")
    barker_support = read_table(BARKER_ORBITAL / "support.csv").iloc[0]
    drivers = read_table(BARKER / "binned_inputs_and_fitted_rates.csv")
    primary_bins = read_table(PRIMARY_ORBITAL / "binned_inputs_and_fitted_rates.csv")
    assert events.groupby("source_record").size().to_dict() == {"MF": 16, "NGRIP": 34, "Sofular": 5}
    assert len(barker_events) == 70
    assert drivers.bin_center_ka.is_monotonic_increasing

    # At identical bin centers, both catalogues must use the same fixed drivers.
    # The speleothem segment starts half a bin off the Barker grid; those centers
    # are intentionally not compared by re-interpolation or re-aggregation.
    overlap = primary_bins.merge(drivers, on="bin_center_ka", suffixes=("_primary", "_barker"))
    assert len(overlap) > 400
    for column in ("precession_index", "lr04", "co2"):
        np.testing.assert_allclose(overlap[column + "_primary"], overlap[column + "_barker"],
                                   rtol=1e-8, atol=1e-8)

    fig, axes = plt.subplots(4, 1, figsize=(165 / 25.4, 137 / 25.4), sharex=True,
                             gridspec_kw={"height_ratios": (1.28, 1, 1, 1)})
    fig.subplots_adjust(left=0.175, right=0.98, top=0.94, bottom=0.105, hspace=0.35)
    timeline = axes[0]
    rows = [(2, support.loc["NGRIP"]), (1, support.loc["MIS6"]), (0, barker_support)]
    for y, row in rows:
        low, high = row.observation_start_kyr_bp, row.observation_end_kyr_bp
        timeline.broken_barh([(low, high - low)], (y - 0.19, 0.38),
                             facecolor="#E6E6E6", edgecolor="none", zorder=0)
        # The thin black baseline is response exposure; the older gray tail is
        # retained only to initialize the 1.5-kyr event-history covariate.
        timeline.hlines(y - 0.25, row.response_start_kyr_bp, row.response_end_kyr_bp,
                        color="#555555", linewidth=0.9)
    for source in ("NGRIP", "MF", "Sofular"):
        ages = events.loc[events.source_record.eq(source), "event_age_kyr_bp"]
        y = 2 if source == "NGRIP" else 1
        timeline.vlines(ages, y - 0.16, y + 0.16, color=COLORS[source], linewidth=1.15)
    timeline.vlines(barker_events.event_age_ka, -0.16, 0.16,
                    color=COLORS["variable"], linewidth=1.0)
    timeline.set(ylim=(-0.45, 2.45), yticks=[2, 1, 0],
                 yticklabels=["NGRIP (34)", "Speleothems (21)", "Barker (70)"])
    timeline.tick_params(axis="y", length=0, pad=6)
    timeline.spines[["top", "right", "left", "bottom"]].set_visible(False)
    timeline.tick_params(axis="x", bottom=False)
    timeline.legend(handles=[Line2D([], [], color=COLORS["MF"], lw=1.7, label="MF (16)"),
                             Line2D([], [], color=COLORS["Sofular"], lw=1.7, label="Sofular (5)")],
                    loc="upper right", frameon=False, ncol=2, fontsize=7.5,
                    bbox_to_anchor=(1.01, 1.34), handlelength=1.2, columnspacing=1.2)

    specifications = [("precession_index", "Precession\nindex", "#555555", [-0.04, 0, 0.04]),
                      ("lr04", "LR04 δ$^{18}$O\n(‰)", "#4C7292", [3, 4, 5]),
                      ("co2", "CO$_2$\n(ppm)", "#9A774F", [180, 230, 280])]
    for ax, (column, label, color, ticks) in zip(axes[1:], specifications):
        ax.plot(drivers.bin_center_ka, drivers[column], color=color, lw=0.9)
        ax.set_ylabel(label, labelpad=8)
        ax.set_yticks(ticks)
        ax.grid(axis="y", color="0.91", linewidth=0.5)
        style_axes(ax)
    axes[1].set_ylim(-0.052, 0.052)
    axes[2].set_ylim(2.9, 5.2)
    axes[3].set_ylim(165, 295)
    for index, ax in enumerate(axes):
        panel_label(ax, chr(97 + index), x=-0.19, y=0.96)
        ax.set_xlim(0, 400)
        ax.xaxis.set_major_locator(MultipleLocator(50))
    axes[-1].set_xlabel("Age (kyr BP)")
    return fig


def phase_inputs(folder, phase_column, color, label, fixed=False):
    phases = read_table(folder / "event_precession_phases.csv")[phase_column].to_numpy()
    summary = read_table(folder / "analysis_summary.csv").iloc[0]
    coefficients = read_table(folder / "predictive_coefficients.csv")
    coefficients = coefficients.loc[coefficients.model_id.eq("full")].set_index("term").beta
    phase_deg = np.linspace(0, 360, 721)
    radians = np.deg2rad(phase_deg)
    beta_sin, beta_cos = coefficients[["pre_phase_sin", "pre_phase_cos"]]
    multiplier = np.exp(beta_sin * np.sin(radians) + beta_cos * np.cos(radians))
    np.testing.assert_allclose(np.exp(2 * np.hypot(beta_sin, beta_cos)),
                               summary.pre_phase_rate_ratio_max_vs_min, rtol=1e-7)
    assert np.isfinite(phases).all() and ((phases >= 0) & (phases < 2 * np.pi)).all()
    return dict(phases=phases, summary=summary, phase=phase_deg, multiplier=multiplier,
                color=color, label=label, fixed=fixed)


def plot_phase_comparison():
    """Separate raw event-phase counts from the fitted conditional rate factor."""
    primary = phase_inputs(PRIMARY, "pre_phase_rad", COLORS["NGRIP"], "NGRIP + speleothems")
    variable = phase_inputs(BARKER, "phase_rad", COLORS["variable"], "Variable threshold")
    fixed = phase_inputs(BARKER / "fixed_threshold", "phase_rad", COLORS["fixed"],
                         "Fixed threshold", fixed=True)
    bootstrap = read_table(Path("data/processed/NGRIP_MIS6_PI_bootstrap/summary.csv")).iloc[0]
    assert [len(d["phases"]) for d in (primary, variable, fixed)] == [55, 70, 59]

    fig = plt.figure(figsize=(165 / 25.4, 148 / 25.4))
    grid = fig.add_gridspec(2, 2, width_ratios=(1, 1.35), hspace=0.67, wspace=0.40,
                           left=0.07, right=0.98, bottom=0.14, top=0.89)
    fig.text(0.07, 0.972, "NGRIP + speleothems (n = 55)", weight="bold", fontsize=9)
    fig.text(0.07, 0.51, "Barker (variable n = 70; fixed n = 59)", weight="bold", fontsize=9)
    bins = np.linspace(0, 2 * np.pi, 13)
    for row, records in enumerate(((primary,), (variable, fixed))):
        polar = fig.add_subplot(grid[row, 0], projection="polar")
        response = fig.add_subplot(grid[row, 1])
        for record in records:
            counts, _ = np.histogram(record["phases"], bins)
            assert counts.sum() == len(record["phases"])
            outlined = record["fixed"]
            polar.bar(bins[:-1], counts, width=np.diff(bins)[0], align="edge",
                      facecolor="none" if outlined else record["color"],
                      edgecolor=record["color"] if outlined else "white",
                      linewidth=1 if outlined else 0.6, alpha=1 if outlined else 0.62,
                      linestyle="--" if outlined else "-", zorder=3 if outlined else 2)
            response.plot(record["phase"], record["multiplier"], color=record["color"],
                          lw=1.8, linestyle="--" if outlined else "-")
            response.axvline(record["summary"].pre_phase_preferred_deg,
                             color=record["color"], lw=0.7, ls=":")
        polar.set_theta_zero_location("E")
        polar.set_theta_direction(1)
        polar.set_xticks(np.deg2rad([0, 90, 180, 270]), ["0°", "90°", "180°", "270°"])
        polar.set(ylim=(0, 15), yticks=[5, 10, 15])
        polar.set_rlabel_position(52)
        polar.tick_params(axis="x", pad=1, labelsize=8)
        polar.tick_params(axis="y", labelsize=6.5, colors="0.38")
        polar.grid(color="0.82", lw=0.5)
        polar.spines["polar"].set_color("0.7")
        polar.spines["polar"].set_linewidth(0.6)
        response.axhline(1, color="0.55", lw=0.8, ls=":")
        response.set(xlim=(0, 360), ylim=(0, 2.5), xticks=[0, 90, 180, 270, 360],
                     yticks=[0, 0.5, 1, 1.5, 2, 2.5], ylabel="Phase rate multiplier")
        response.grid(axis="y", color="0.91", lw=0.5)
        style_axes(response)
        panel_label(polar, "ac"[row], x=-0.16, y=1.04)
        panel_label(response, "bd"[row], x=-0.18, y=1.04)
        if row == 0:
            polar.set_title("Event counts", pad=20, fontsize=8.5)
            response.set_title("Conditional phase response", pad=17, fontsize=8.5)
            response.text(0.04, 0.96,
                          f"PI = {primary['summary'].info_bits_per_event:.3f} bits/event\n"
                          f"Bootstrap p = {bootstrap.empirical_p_plus_one:.4f}",
                          transform=response.transAxes, va="top", fontsize=7.5)
        else:
            response.set_xlabel("Precession phase (°)")
            for index, record in enumerate(records):
                response.text(0.04, 0.96 - 0.10 * index,
                              f"PI = {record['summary'].info_bits_per_event:.3f} bits/event",
                              transform=response.transAxes, va="top", fontsize=7.5,
                              color=record["color"])
    fig.legend(handles=[Patch(facecolor=COLORS["variable"], alpha=0.62,
                              label="Barker: variable threshold"),
                        Patch(facecolor="none", edgecolor=COLORS["fixed"], linestyle="--",
                              label="Barker: fixed threshold")],
               loc="lower center", bbox_to_anchor=(0.53, 0.035), ncol=2,
               frameon=False, columnspacing=1.3, fontsize=7.5)
    return fig


def plot_orbital_comparison():
    """Compare added orbital information on identical axes for both catalogues."""
    summaries = [read_table(folder / "comparison_summary.csv")
                 for folder in (PRIMARY_ORBITAL, BARKER_ORBITAL)]
    fig, axes = plt.subplots(2, 3, figsize=(165 / 25.4, 136 / 25.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.22, right=0.985, bottom=0.19, top=0.83, hspace=0.67, wspace=0.13)
    specs = [("driver_after_base", "BG + Orb\nvs BG"),
             ("driver_after_phase", "BG + Pre + Orb\nvs BG + Pre"),
             ("phase_after_driver", "BG + Pre + Orb\nvs BG + Orb")]
    for row, (summary, title) in enumerate(zip(summaries, ("NGRIP + speleothems", "Barker"))):
        assert len(summary) == 10 and summary.point_fit_valid.all()
        reference = summary.set_index("comparison_id").loc["phase_reference"]
        for column, (group, comparison) in enumerate(specs):
            ax = axes[row, column]
            subset = summary.loc[summary.comparison_group.eq(group)].set_index("driver_id")
            for y, (driver, (_, color)) in enumerate(DRIVERS.items()):
                estimate = subset.loc[driver]
                low, median, high = estimate[["info_bits_per_event_q025", "info_bits_per_event_median",
                                              "info_bits_per_event_q975"]].astype(float)
                assert 0 <= low <= median <= high < 0.30
                ax.hlines(y, low, high, color=color, linewidth=2.4, alpha=0.58)
                ax.plot(median, y, "|", color=color, markersize=10, markeredgewidth=1.2)
                ax.plot(estimate.info_bits_per_event_point, y, "o", color=color,
                        markersize=4.5, markeredgecolor="white", markeredgewidth=0.5)
            if column != 1:
                ax.axvline(reference.info_bits_per_event_point, color="0.40", linewidth=0.8,
                           linestyle=(0, (3, 3)), zorder=0)
            ax.set(xlim=(-0.008, 0.30), ylim=(2.45, -0.45))
            ax.xaxis.set_major_locator(MultipleLocator(0.1))
            ax.grid(axis="x", color="0.90", linewidth=0.5)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.tick_params(axis="y", length=0, pad=6)
            ax.set_yticks(range(3), [item[0] for item in DRIVERS.values()])
            panel_label(ax, chr(97 + row * 3 + column), x=0, y=1.02)
            if row == 0:
                ax.set_title(comparison, fontsize=8, pad=25)
            else:
                ax.set_xlabel("PI (bits event$^{-1}$)")
        fig.text(0.22, 0.969 if row == 0 else 0.514, title, weight="bold", fontsize=9)
    handles = [Line2D([], [], marker="o", linestyle="none", color="0.3", markersize=4,
                      label="Point ages"),
               Line2D([], [], marker="|", linestyle="none", color="0.3", markersize=9,
                      label="MC median"),
               Line2D([], [], linewidth=2.4, color="0.3", alpha=0.58, label="95% age-MC range"),
               Line2D([], [], linestyle="--", linewidth=0.8, color="0.4",
                      label="BG + Pre vs BG (point ages)")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.57, 0.025),
               frameon=False, ncol=2, fontsize=7.2, columnspacing=1.4, handlelength=2.1)
    return fig


def main():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 8, "axes.labelsize": 8.5, "axes.linewidth": 0.65,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white"})
    save_figure(plot_data_overview(), "data_overview")
    save_figure(plot_phase_comparison(), "phase_comparison")
    save_figure(plot_orbital_comparison(), "orbital_comparison")
    inputs = [dict(source=path.relative_to(PROJECT).as_posix(),
                   sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in sorted(INPUTS)]
    inputs.append(dict(source=Path(__file__).name,
                       sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    pd.DataFrame(inputs).to_csv(FIGURES / "input_sha256.csv", index=False)
    print("Saved manuscript overview, phase comparison and orbital comparison; no models refitted.")


if __name__ == "__main__":
    main()
