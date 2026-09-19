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
import argparse
from toolbox import combined_likelihood, orbital_driver_sensitivity, event_inputs, project_config

from paper_figure_export import copy_pdf_to_paper
from toolbox.catalogue_colors import CATALOGUE_COLORS
from toolbox.figure_style import add_panel_label as panel_label
from toolbox.phase_response_plotting import format_phase_response_axis, mark_preferred_phase

PROJECT = Path(__file__).resolve().parent
RESULT_ROOT = PROJECT
FIGURES = PROJECT / "figures/paper_summary"
PRIMARY = Path("data/processed/NGRIP_MIS6_event_phase_analysis")
BARKER = Path("Barker2011/data/processed/Barker2011_event_phase_analysis")
PRIMARY_ORBITAL = Path("data/processed/NGRIP_MIS6_orbital_driver_sensitivity")
BARKER_ORBITAL = Path("Barker2011/data/processed/Barker2011_orbital_driver_sensitivity")
DRIVERS = {"ecc": ("Eccentricity", "#CC79A7"),
           "obl": ("Obliquity", "#009E73"),
           "insol65n": ("65°N summer-solstice\ninsolation", "#D55E00")}
INPUTS = set()


def read_table(relative_path):
    """Track the saved input files used to compose the manuscript figures."""
    path = (RESULT_ROOT if "processed" in Path(relative_path).parts else PROJECT) / relative_path
    INPUTS.add(path)
    return pd.read_csv(path)


def style_axes(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(direction="out", width=0.6, length=3)


def save_figure(figure, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    pdf = FIGURES / f"{name}.pdf"
    figure.savefig(pdf, facecolor="white")
    figure.savefig(FIGURES / f"{name}.png", dpi=400, facecolor="white")
    copy_pdf_to_paper(pdf)
    plt.close(figure)


def plot_data_overview():
    """Compare orbital inputs and event locations, with younger ages at right."""
    events = read_table(Path("data/curated/ngrip_mis6_warming_events.csv"))
    primary_phases = read_table(PRIMARY / "event_precession_phases.csv")
    variable_phases = read_table(BARKER / "event_precession_phases.csv")
    fixed_phases = read_table(BARKER / "fixed_threshold/event_precession_phases.csv")
    support = read_table(PRIMARY_ORBITAL / "support.csv").set_index("segment_id")
    # Overview curves use native source series, independent of any model grid.
    context = combined_likelihood.build_barker_context()
    age = np.linspace(0, 400, 4001)
    drivers = pd.DataFrame({"age_kyr_bp": age})
    for name, (knots, values) in context.forcings.items():
        drivers[name] = event_inputs.interpolate_checked(age, knots, values, context="overview")
    sources, _ = orbital_driver_sensitivity.load_driver_sources()
    orbital = pd.DataFrame({"age_kyr_bp": age})
    for name, column in (("ecc", "ecc"), ("obl", "obl_deg"), ("insol65n", "insol65n_Wm2")):
        source = sources[name]
        orbital[column] = event_inputs.interpolate_checked(age, source["age"], source["values"], context="overview")
        INPUTS.add(source["path"])
    INPUTS.update((project_config.LR04_XLSX, project_config.CO2_XLSX, project_config.PRE_TXT))
    assert events.groupby("source_record").size().to_dict() == {"MF": 16, "NGRIP": 34, "Sofular": 5}
    assert [len(primary_phases), len(variable_phases), len(fixed_phases)] == [55, 70, 59]
    assert np.isin(fixed_phases.event_age_ka, variable_phases.event_age_ka).all()
    np.testing.assert_allclose(primary_phases.event_age_kyr_bp, events.event_age_kyr_bp)

    # Reserve the right margin for two explicitly colored orbital scales.
    fig, axes = plt.subplots(4, 1, figsize=(180 / 25.4, 158 / 25.4), sharex=True,
                             gridspec_kw={"height_ratios": (1.25, 1.6, 1, 1)})
    fig.subplots_adjust(left=0.12, right=0.81, top=0.945, bottom=0.085, hspace=0.40)
    eccentricity = axes[0]
    obliquity = eccentricity.twinx()
    insolation = eccentricity.twinx()
    insolation.spines["right"].set_position(("outward", 42))
    orbital_specs = [
        (eccentricity, "ecc", "Eccentricity", DRIVERS["ecc"][1], [0, 0.03, 0.06], (-0.003, 0.065)),
        (obliquity, "obl_deg", "Obliquity (°)", DRIVERS["obl"][1], [22, 23, 24], (21.4, 24.9)),
        (insolation, "insol65n_Wm2", "65°N insolation\n(W m$^{-2}$)",
         DRIVERS["insol65n"][1], [440, 490, 540], (420, 565)),
    ]
    for ax, column, label, color, ticks, limits in orbital_specs:
        ax.plot(orbital.age_kyr_bp, orbital[column], color=color, lw=0.9)
        ax.set(ylabel=label, yticks=ticks, ylim=limits)
        ax.yaxis.label.set_color(color)
        ax.tick_params(axis="y", colors=color, labelsize=7, length=3, width=0.6)
        ax.spines[["top", "bottom"]].set_visible(False)
        side = "left" if ax is eccentricity else "right"
        ax.spines[side].set_color(color)
        ax.spines[side].set_linewidth(0.6)
        ax.spines["right" if side == "left" else "left"].set_visible(False)
        ax.grid(False, which="both")
        ax.tick_params(axis="x", bottom=False, labelbottom=False)

    phase = axes[1]
    phase.plot(drivers.age_kyr_bp, drivers.precession_index, color="0.40", lw=0.9, zorder=1)
    # Use the stored forcing value at each event, without age jitter. Larger
    # open squares leave both symbols visible for shared Barker event picks.
    phase.scatter(variable_phases.event_age_ka, variable_phases.orbital_value_at_event,
                  s=13, marker="o", c=CATALOGUE_COLORS["variable"], linewidths=0, zorder=3,
                  label="Barker 2011: varying threshold (n = 70)")
    phase.scatter(fixed_phases.event_age_ka, fixed_phases.orbital_value_at_event,
                  s=24, marker="s", facecolors="none", edgecolors=CATALOGUE_COLORS["fixed"],
                  linewidths=0.7, zorder=4, label="Barker 2011: fixed threshold (n = 59)")
    phase.scatter(primary_phases.event_age_kyr_bp, primary_phases.precession_index,
                  s=21, marker="^", c=CATALOGUE_COLORS["primary"], edgecolors="white",
                  linewidths=0.3, zorder=5, label="NGRIP–MIS6 (n = 55)")
    # Thin strips retain the two primary observation intervals; their gap is
    # unobserved. Barker spans the entire displayed interval (0–400 kyr BP).
    for row in support.itertuples():
        phase.axvspan(row.observation_start_kyr_bp, row.observation_end_kyr_bp,
                      ymin=0.01, ymax=0.035, color=CATALOGUE_COLORS["primary"], alpha=0.35, lw=0)
    handles, labels = phase.get_legend_handles_labels()
    phase.legend([handles[i] for i in (2, 0, 1)], [labels[i] for i in (2, 0, 1)],
                 loc="lower left", bbox_to_anchor=(-0.02, 1.02), frameon=False,
                 ncol=1, fontsize=6.9, handletextpad=0.35, borderaxespad=0,
                 labelspacing=0.25)
    phase.set(ylabel="Precession index", ylim=(-0.061, 0.058), yticks=[-0.04, 0, 0.04])

    specifications = [("lr04", "LR04 δ$^{18}$O (‰)", "0.30", [3, 4, 5]),
                      ("co2", "CO$_2$ (ppm)", "0.40", [180, 230, 280])]
    for ax, (column, label, color, ticks) in zip(axes[2:], specifications):
        ax.plot(drivers.age_kyr_bp, drivers[column], color=color, lw=1)
        ax.set_ylabel(label, labelpad=8)
        ax.set_yticks(ticks)
    axes[2].set_ylim(5.2, 2.9)  # Larger benthic isotope values plot lower.
    axes[3].set_ylim(165, 295)
    for index, ax in enumerate(axes):
        if index:
            style_axes(ax)
        panel_label(ax, chr(97 + index), x=-0.135, y=1.015)
        # BP increases into the past: use descending display limits only.
        ax.set_xlim(400, 0)
        ax.grid(False, which="both")
        ax.xaxis.set_major_locator(MultipleLocator(50))
        if index < 3:
            ax.tick_params(axis="x", bottom=False, labelbottom=False)
            ax.spines["bottom"].set_visible(False)
    axes[-1].set_xlabel("Age (kyr BP)")
    return fig


def phase_inputs(folder, phase_column, color, label, fixed=False):
    phases = read_table(folder / "event_precession_phases.csv")[phase_column].to_numpy()
    summary = read_table(folder / "analysis_summary.csv").iloc[0]
    saved_coefficients = read_table(folder / "model_coefficients.csv")
    coefficients = saved_coefficients.loc[saved_coefficients.model_id.eq("full")].set_index("term").beta
    phase_deg = np.linspace(0, 360, 721)
    radians = np.deg2rad(phase_deg)
    beta_sin, beta_cos = coefficients[["pre_phase_sin", "pre_phase_cos"]]
    multiplier = np.exp(beta_sin * np.sin(radians) + beta_cos * np.cos(radians))
    np.testing.assert_allclose(np.exp(2 * np.hypot(beta_sin, beta_cos)),
                               summary.pre_phase_rate_ratio_max_vs_min, rtol=1e-7)
    assert np.isfinite(phases).all() and ((phases >= 0) & (phases < 2 * np.pi)).all()
    return dict(phases=phases, summary=summary, phase=phase_deg, multiplier=multiplier,
                color=color, label=label, fixed=fixed, coefficients=saved_coefficients)


def plot_phase_comparison():
    """Compare raw phase counts overlaid with fitted full/reduced expected counts."""
    primary = phase_inputs(PRIMARY, "pre_phase_rad", CATALOGUE_COLORS["primary"], "NGRIP–MIS6")
    variable = phase_inputs(BARKER, "phase_rad", CATALOGUE_COLORS["variable"], "Varying threshold")
    fixed = phase_inputs(BARKER / "fixed_threshold", "phase_rad", CATALOGUE_COLORS["fixed"],
                         "Fixed threshold", fixed=True)
    assert [len(d["phases"]) for d in (primary, variable, fixed)] == [55, 70, 59]
    primary["sectors"] = read_table(PRIMARY / "phase_sector_fit.csv")
    variable["sectors"] = read_table(BARKER / "phase_sector_fit.csv")
    fixed["sectors"] = read_table(BARKER / "fixed_threshold/phase_sector_fit.csv")

    fig = plt.figure(figsize=(180 / 25.4, 132 / 25.4))
    grid = fig.add_gridspec(2, 2, width_ratios=(1, 1.35), height_ratios=(1, 1),
                           hspace=0.55, wspace=0.43,
                           left=0.11, right=0.98, bottom=0.18, top=0.96)
    edges = np.linspace(0, 2 * np.pi, 19)  # 20-degree sectors, as in the earlier paper.
    histograms = [np.histogram(r["phases"], edges)[0] for r in (primary, variable, fixed)]
    radial_max = 2 * np.ceil(max(h.max() for h in histograms) / 2)
    for group, records, fitted, title in [
        (0, (primary,), (primary,), "NGRIP–MIS6 (55 events; 53 fitted)"),
        (1, (variable, fixed), (variable,), "Barker: both thresholds"),
    ]:
        polar = fig.add_subplot(grid[group, 0], projection="polar")
        response = fig.add_subplot(grid[group, 1])
        for index, record in enumerate(records):
            counts, _ = np.histogram(record["phases"], edges)
            assert counts.sum() == len(record["phases"])
            # Narrow sectors, white dividers and a common count scale
            # follow the earlier Rayleigh figure. Both Barker fills stay visible.
            polar.bar(edges[:-1], counts, width=np.diff(edges)[0], align="edge",
                      facecolor=record["color"], edgecolor="white", linewidth=0.55,
                      alpha=0.52 if len(records) == 2 else 0.68)
            response.plot(record["phase"], record["multiplier"], color=record["color"], lw=1.6,
                          ls="-")
            response.axvline(record["summary"].pre_phase_preferred_deg,
                             color=record["color"], lw=0.7, ls=":")
            mark_preferred_phase(response, record["summary"].pre_phase_preferred_deg,
                                 record["summary"].pre_phase_rate_ratio_max_vs_min, record["color"])
            # Inventory-event rug along the bottom of the phase-response panel.
            rug_y = 0.03 + 0.05 * index
            response.plot(np.degrees(record["phases"]),
                          np.full(len(record["phases"]), rug_y), "|",
                          transform=response.get_xaxis_transform(),
                          color=record["color"], markersize=4.0, markeredgewidth=0.7)
        # Overlay fitted expected-event counts for the primary catalogue(s):
        # reduced (solid) and full (dotted) polar curves. Barker overlays only
        # the varying-threshold fit to avoid crowding the two histograms.
        for record in fitted:
            sectors = record["sectors"]
            centers = sectors.loc[sectors.model_id.eq("reduced"), "phase_sector_center_deg"].to_numpy(float)
            angle = np.deg2rad(np.r_[centers, centers[0] + 360.0])
            for model_id, line_style in (("reduced", "--"), ("full", "-")):
                values = sectors.loc[sectors.model_id.eq(model_id), "fitted_events"].to_numpy(float)
                polar.plot(angle, np.r_[values, values[0]], color=record["color"],
                           lw=1.5, ls=line_style, alpha=0.9, zorder=4)
        polar.set_theta_zero_location("E")
        polar.set_theta_direction(1)
        polar.set_xticks(np.deg2rad([0, 90, 180, 270]),
                         ["0°\nMin", "90°", "180°\nMax", "270°"])
        polar.set(ylim=(0, radial_max), yticks=np.arange(2, radial_max + 0.1, 2))
        polar.set_yticklabels(["" if count == 2 else str(int(count))
                               for count in np.arange(2, radial_max + 0.1, 2)])
        polar.set_rlabel_position(45)
        polar.tick_params(axis="x", pad=1, labelsize=7.2)
        polar.tick_params(axis="y", labelsize=6.5, colors="0.30")
        polar.set_axisbelow(True)
        polar.grid(color="0.84", lw=0.55)
        polar.spines["polar"].set_color("0.25")
        polar.spines["polar"].set_linewidth(0.75)
        polar.set_title("Event-phase counts and fitted model", pad=22, fontsize=8)
        response.axhline(1, color="0.55", lw=0.8, ls=":")
        response.set(xlim=(0, 360), ylim=(0, 3.1), xticks=[0, 90, 180, 270, 360],
                     yticks=[0, 0.5, 1, 1.5, 2, 2.5])
        format_phase_response_axis(response)
        response.set_xlabel("Precession-index phase", fontsize=8)
        response.set_title("Conditional phase effect", pad=8, fontsize=8)
        response.grid(False, which="both")
        style_axes(response)
        panel_label(polar, "ac"[group], x=-0.24, y=1.06)
        panel_label(response, "bd"[group], x=-0.17, y=1.045)
        # Reserve space above the response maxima for compact text labels.
        for index, record in enumerate(records):
            stats = record["summary"]
            response.text(0.035, 0.97 - 0.18 * index,
                          f"G = {stats.gain_bits_per_event:.3f}; LR p = {stats.nominal_LR_p:.4f}\n"
                          f"Peak {stats.pre_phase_preferred_deg:.1f}°; max/min {stats.pre_phase_rate_ratio_max_vs_min:.2f}",
                          transform=response.transAxes, va="top", fontsize=7, color=record["color"],
                          linespacing=1.25, bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1))
    fig.legend(handles=[
        Patch(facecolor=CATALOGUE_COLORS["variable"], alpha=0.52, label="Barker: varying threshold (n = 70)"),
        Patch(facecolor=CATALOGUE_COLORS["fixed"], alpha=0.52, label="Barker: fixed threshold (n = 59)"),
        Line2D([], [], color="0.3", lw=1.5, ls="--", label="reduced (no phase)"),
        Line2D([], [], color="0.3", lw=1.5, ls="-", label="full (with phase)"),
    ], loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=2,
       frameon=False, columnspacing=1.4, handlelength=1.8, fontsize=7.0)
    return fig


def plot_orbital_comparison():
    """Compare added orbital information on identical axes for both catalogues."""
    summaries = [read_table(folder / "comparison_summary.csv")
                 for folder in (PRIMARY_ORBITAL, BARKER_ORBITAL)]
    max_gain=max(float(t.gain_bits_per_event_q975.max()) for t in summaries)
    xmax=np.ceil(max_gain/0.1)*0.1
    fig, axes = plt.subplots(2, 3, figsize=(180 / 25.4, 136 / 25.4), sharex=True, sharey=True)
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
                low, median, high = estimate[["gain_bits_per_event_q025", "gain_bits_per_event_median",
                                              "gain_bits_per_event_q975"]].astype(float)
                assert 0 <= low <= median <= high
                ax.hlines(y, low, high, color=color, linewidth=2.4, alpha=0.58)
                ax.plot(median, y, "|", color=color, markersize=10, markeredgewidth=1.2)
                ax.plot(estimate.gain_bits_per_event_point, y, "o", color=color,
                        markersize=4.5, markeredgecolor="white", markeredgewidth=0.5)
            if column != 1:
                ax.axvline(reference.gain_bits_per_event_point, color="0.40", linewidth=0.8,
                           linestyle=(0, (3, 3)), zorder=0)
            ax.set(xlim=(-0.008, xmax), ylim=(2.45, -0.45))
            ax.xaxis.set_major_locator(MultipleLocator(0.1))
            ax.grid(axis="x", color="0.90", linewidth=0.5)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.tick_params(axis="y", length=0, pad=6)
            ax.set_yticks(range(3), [item[0] for item in DRIVERS.values()])
            panel_label(ax, chr(97 + row * 3 + column), x=0, y=1.02)
            if row == 0:
                ax.set_title(comparison, fontsize=8, pad=25)
            else:
                ax.set_xlabel("Gain (bits event$^{-1}$)")
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
    global RESULT_ROOT, FIGURES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=PROJECT)
    args = parser.parse_args()
    RESULT_ROOT = args.output_root.resolve()
    FIGURES = RESULT_ROOT / "figures/paper_summary"
    INPUTS.add(PROJECT / "toolbox/phase_response_plotting.py")
    INPUTS.add(PROJECT / "toolbox/catalogue_colors.py")
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
    style_path = PROJECT / "toolbox/figure_style.py"
    inputs.append(dict(source=style_path.relative_to(PROJECT).as_posix(),
                       sha256=hashlib.sha256(style_path.read_bytes()).hexdigest()))
    pd.DataFrame(inputs).to_csv(FIGURES / "input_sha256.csv", index=False)
    print("Saved manuscript overview, phase comparison and orbital comparison; no models refitted.")


if __name__ == "__main__":
    main()
