#!/usr/bin/env python3
"""Compare So-4/So-57 controls and the saved event-age sensitivity ensembles.

Run from the project root: python MIS6/Sofular_chronology_comparison.py
This figure reads the original Held2024 controls and the two saved summaries;
it does not resample ages or refit PI. The uncertainty notebook also uses its
data preparation and plotting functions.
"""

from pathlib import Path

import matplotlib

if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

if __package__:
    from . import sofular_chronology as sofular
else:
    import sofular_chronology as sofular


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data/processed/MIS6_event_age_uncertainty"
FIGURE_DIR = ROOT / "figures/Sofular_chronology_comparison"
FIGURE_STEM = "Sofular_chronology_comparison"
COLORS = {"So-4": "#B65B32", "So-57": "#277DA6"}
MARKERS = {"So-4": "o", "So-57": "s"}


def load_comparison_data(data_dir=DATA_DIR):
    """Select the actual dated knots and align the five saved Sofular events."""
    summaries = {}
    for component, suffix in (("So-4", ""), ("So-57", "_so57_overlap")):
        table = pd.read_csv(
            data_dir / f"mis6_event_age_uncertainty_summary{suffix}.csv"
        )
        summaries[component] = table.loc[table.source_record.eq("Sofular")].set_index(
            "composite_event_id"
        ).sort_index()
    primary, alternative = summaries["So-4"], summaries["So-57"]
    if primary.index.tolist() != [f"MIS6_DO_{i:02d}" for i in range(17, 22)]:
        raise ValueError("Expected composite events 17--21 from Sofular")
    if not primary.index.equals(alternative.index):
        raise ValueError("The two summaries contain different event identities")
    np.testing.assert_allclose(
        primary.nominal_event_age_ka_bp, alternative.nominal_event_age_ka_bp,
        rtol=0, atol=1e-6,
    )

    # Use the same control selection as the sampler. So-57 covers events
    # 17--20; event 21 stays on So-4 in both uncertainty schemes.
    sources = sofular.load_sources()
    ages = primary.nominal_event_age_ka_bp.to_numpy()
    controls = []
    for component in COLORS:
        selected_ages = ages if component == "So-4" else ages[:-1]
        context = sofular.build_context(selected_ages, component=component, sources=sources)
        dates = sources[component].controls
        controls.append(dates.loc[dates.control_id.isin(context.control_ids)])
    return pd.concat(controls, ignore_index=True), summaries


def plot_comparison(controls, summaries):
    """Show measured U--Th errors and accepted MC intervals on distinct panels."""
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.labelsize": 9, "axes.titlesize": 10,
        "axes.titleweight": "normal", "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.linewidth": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42,
    }):
        fig, (ax_controls, ax_events) = plt.subplots(
            2, 1, figsize=(7.2, 5.7), gridspec_kw={"height_ratios": [1, 1.25]},
        )
        fig.subplots_adjust(left=0.22, right=0.97, bottom=0.10, top=0.94, hspace=0.64)

        for component, level in (("So-4", 1.0), ("So-57", 0.0)):
            part = controls.loc[controls.component.eq(component)]
            # Stagger nearby dates; vertical position within a row has no
            # physical meaning and does not compare depths across stalagmites.
            y = level + 0.28 * (np.arange(len(part)) % 2)
            ax_controls.errorbar(
                part.age_ka_bp, y, xerr=part.age_error_2sigma_ka,
                fmt=MARKERS[component], color=COLORS[component], markersize=4,
                elinewidth=1.2, capsize=2,
            )
            for x, height, control_id in zip(part.age_ka_bp, y, part.control_id):
                ax_controls.annotate(
                    control_id.split("-")[-1], (x, height), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=7,
                    color=COLORS[component],
                )
        ax_controls.set_yticks([1.14, 0.14], ["So-4\n10 controls", "So-57\n7 controls"])
        ax_controls.set_ylim(-0.35, 1.9)
        # All absolute-age axes in the current study get younger to the right.
        ax_controls.set_xlim(207, 158)
        ax_controls.set_xticks(np.arange(205, 159, -5))
        ax_controls.set_xlabel("U–Th age (kyr BP)")
        ax_controls.set_title("(a)  Dated controls and reported ±2σ errors", loc="left", pad=10)
        ax_controls.spines["left"].set_visible(False)
        ax_controls.tick_params(axis="y", length=0, pad=10)

        primary = summaries["So-4"]
        y = np.arange(len(primary))
        for component, shift in (("So-4", -0.14), ("So-57", 0.14)):
            table = summaries[component]
            # An offset axis compares uncertainty widths without compressing
            # neighboring events. Positive offsets mean older sampled ages.
            nominal = table.nominal_event_age_ka_bp.to_numpy()
            lower = table.sampled_age_q025_ka_bp.to_numpy() - nominal
            median = table.sampled_age_median_ka_bp.to_numpy() - nominal
            upper = table.sampled_age_q975_ka_bp.to_numpy() - nominal
            ax_events.errorbar(
                median, y + shift, xerr=[median - lower, upper - median],
                fmt=MARKERS[component], color=COLORS[component], markersize=4,
                elinewidth=1.5, capsize=2,
            )
        labels = [
            f"{row.composite_event_label}  |  {row.nominal_event_age_ka_bp:.3f}"
            for row in primary.itertuples()
        ]
        labels[-1] += " *"
        ax_events.set_yticks(y, labels)
        ax_events.set_ylim(len(primary) - 0.5, -0.6)
        ax_events.set_xlim(-1.04, 1.04)
        ax_events.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax_events.axvline(0, color="0.65", linestyle="--", linewidth=0.8, zorder=0)
        ax_events.set_xlabel("Sampled age − nominal age (kyr; positive = older)")
        ax_events.set_title("(b)  Event-age medians and 95% MC ranges", loc="left", pad=10)
        handles = [
            Line2D([], [], marker=MARKERS[c], color=COLORS[c], markersize=4,
                   linewidth=1.4, label=label)
            for c, label in (("So-4", "So-4 primary"), ("So-57", "So-57 overlap alternative"))
        ]
        ax_events.legend(
            handles=handles, loc="lower left", bbox_to_anchor=(0, 1.16),
            frameon=False, ncol=2, fontsize=8, borderaxespad=0, handlelength=1.7,
        )
        fig.text(0.22, 0.016, "* MIS 6.21 uses So-4 in both schemes. Nominal ages: kyr BP (1950).", fontsize=8)
        for ax in (ax_controls, ax_events):
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(False)
        return fig


def main():
    controls, summaries = load_comparison_data()
    fig = plot_comparison(controls, summaries)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(FIGURE_DIR / f"{FIGURE_STEM}.{extension}", dpi=600)
    plt.close(fig)
    print(f"Wrote So-4/So-57 comparison to {FIGURE_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
