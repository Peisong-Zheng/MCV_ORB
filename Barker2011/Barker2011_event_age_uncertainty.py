#!/usr/bin/env python3
"""Ordered SpeleoAge realizations from Barker (2011), Supplementary Table S1.

Combined uncertainty is used as a uniform proposal half-width, not a sigma.
Control offsets are independent before conditioning on an increasing age map.
Offsets are interpolated in the original SpeleoAge coordinate. The long control
gap is bridged linearly; a computational control at 400 ka covers the oldest
events. No EDC3 coordinate or additional event-picking error enters sampling.

Run from the project root or Barker2011 directory. Sources, processed tables
and figures are kept in this directory, alongside the nominal main analysis.
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_figure_export import copy_pdf_to_paper
from toolbox.plotting import add_panel_label, configure_barker_style
from toolbox.project_config import PROJECT_ROOT, BARKER_EVENT_CSVS, CATALOGUE_COLORS

ROOT = Path(__file__).resolve().parent
RUN_NAME = "Barker2011_event_age_uncertainty"
CONTROL_CSV = ROOT / "data/raw/Barker2011_TableS1.csv"
N_REALIZATIONS = 10_000
RANDOM_SEED = 20260908
AUXILIARY_AGE_KA = 400.0
N_TAIL_CONTROLS = 4
PUBLISHED_GAP_KA = (265.0, 315.0)
CONTROL_GAP_KA = (264.24, 317.70)
BLUE = "#4477AA"
ORANGE = "#CC9933"

OUTPUT_ROOT = PROJECT_ROOT
REDRAW = False
EXPORT_PAPER = True


def prepare_controls(raw):
    """Check published uncertainties and append the synthetic 400 ka endpoint."""
    columns = ["speleo_age_ka", "tuning_error_ka", "absolute_speleo_error_ka",
               "combined_uncertainty_ka"]
    controls = raw[columns].copy()
    values = controls.to_numpy(float)
    if (len(controls) != 60 or not np.isfinite(values).all()
            or np.any(np.diff(values[:, 0]) <= 0) or np.any(values[:, 1:] < 0)):
        raise ValueError("Expected 60 ordered, finite Table S1 controls")
    # Keep the printed combined values; quadrature agrees within table rounding.
    combined = np.hypot(controls.tuning_error_ka, controls.absolute_speleo_error_ka)
    if not np.allclose(combined, controls.combined_uncertainty_ka, atol=0.008, rtol=0):
        raise ValueError("Check the Table S1 uncertainty transcription")
    controls.insert(0, "control_id", [f"S1_{i:02d}" for i in range(1, 61)])
    controls["control_type"] = "published"
    controls["source_table_row"] = np.arange(1, 61)
    controls["unfloored_extrapolation_ka"] = np.nan

    tail = controls.tail(N_TAIL_CONTROLS)
    slope, intercept = np.polyfit(tail.speleo_age_ka, tail.combined_uncertainty_ka, 1)
    extrapolated = slope * AUXILIARY_AGE_KA + intercept
    half_width = max(extrapolated, tail.combined_uncertainty_ka.max())
    if AUXILIARY_AGE_KA <= controls.speleo_age_ka.iloc[-1]:
        raise ValueError("The auxiliary control must be older than Table S1")
    auxiliary = dict(control_id="AUX_400", speleo_age_ka=AUXILIARY_AGE_KA,
                     tuning_error_ka=np.nan, absolute_speleo_error_ka=np.nan,
                     combined_uncertainty_ka=half_width, control_type="auxiliary",
                     source_table_row=np.nan, unfloored_extrapolation_ka=extrapolated)
    return pd.concat([controls, pd.DataFrame([auxiliary])], ignore_index=True)


def interpolation_weights(ages, controls):
    """Fixed bracketing indices and weights on the published SpeleoAge axis."""
    ages = np.asarray(ages, dtype=float)
    knots = controls.speleo_age_ka.to_numpy(float)
    if (not np.isfinite(ages).all() or not np.isfinite(knots).all()
            or np.any(np.diff(knots) <= 0)
            or np.any(ages < knots[0]) or np.any(ages > knots[-1])):
        raise ValueError("Ages must lie within an ordered SpeleoAge control sequence")
    right = np.clip(np.searchsorted(knots, ages, side="right"), 1, len(knots) - 1)
    left = right - 1
    weight = (ages - knots[left]) / (knots[right] - knots[left])
    return left, right, weight


def interpolate_offsets(ages, controls, control_offsets):
    left, right, weight = interpolation_weights(ages, controls)
    return ((1 - weight) * control_offsets[:, left]
            + weight * control_offsets[:, right])


def prepare_events(events, controls):
    """Attach chronology controls to the prepared event ages."""
    events = events.copy()
    age = events.event_age_kyr_bp.to_numpy(float)
    left, right, weight = interpolation_weights(age, controls)
    events["left_control_id"] = controls.control_id.to_numpy()[left]
    events["right_control_id"] = controls.control_id.to_numpy()[right]
    events["right_control_weight"] = weight
    events["in_published_alignment_gap"] = (age >= PUBLISHED_GAP_KA[0]) & (age <= PUBLISHED_GAP_KA[1])
    events["in_long_control_interval"] = (age > CONTROL_GAP_KA[0]) & (age < CONTROL_GAP_KA[1])
    last_published = controls.loc[controls.control_type.eq("published"), "speleo_age_ka"].max()
    events["beyond_last_published_control"] = age > last_published
    return events


def sample_realizations(events, controls, n_realizations=N_REALIZATIONS, seed=RANDOM_SEED):
    """Reject whole crossed control maps; never repair a draw by sorting ages."""
    if n_realizations <= 0:
        raise ValueError("n_realizations must be positive")
    ages = events.event_age_kyr_bp.to_numpy(float)
    knots = controls.speleo_age_ka.to_numpy(float)
    half_width = controls.combined_uncertainty_ka.to_numpy(float)
    if np.any(np.diff(ages) <= 0) or not np.isfinite(half_width).all() or np.any(half_width < 0):
        raise ValueError("Expected ordered events and finite nonnegative half-widths")
    interpolation_weights(ages, controls)
    rng = np.random.default_rng(seed)
    accepted = []
    n_accepted = n_proposed = n_rejected = 0
    while n_accepted < n_realizations:
        batch_size = max(1000, n_realizations - n_accepted)
        offsets = rng.uniform(-half_width, half_width, size=(batch_size, len(knots)))
        ordered = np.all(np.diff(knots + offsets, axis=1) > 0, axis=1)
        accepted.append(offsets[ordered])
        n_accepted = n_accepted + int(ordered.sum())
        n_proposed = n_proposed + batch_size
        n_rejected = n_rejected + int((~ordered).sum())
        if n_proposed > max(100_000, 200 * n_realizations):
            raise RuntimeError("Too few ordered control maps; inspect proposal widths")
    control_offsets = np.vstack(accepted)[:n_realizations]
    draws = ages + interpolate_offsets(ages, controls, control_offsets)
    if not np.all(np.diff(draws, axis=1) > 0):
        raise RuntimeError("A monotone control map failed to preserve event order")
    diagnostics = dict(n_realizations=n_realizations, random_seed=seed, n_proposals=n_proposed,
                       n_rejected_crossed_controls=n_rejected, n_accepted_proposals=n_accepted,
                       n_unused_accepted_tail=n_accepted - n_realizations,
                       acceptance_fraction=n_accepted / n_proposed)
    return draws, control_offsets, diagnostics


def age_columns(events):
    return [f"age_ka_bp__{event_id}" for event_id in events.event_id]


def summarize_ages(events, draws):
    """Accepted-ensemble age quantiles, not confidence intervals."""
    summary = events[["event_id", "event_age_kyr_bp"]].copy()
    offsets = draws - events.event_age_kyr_bp.to_numpy(float)
    for label, q in (("q025", 0.025), ("median", 0.5), ("q975", 0.975)):
        summary[f"age_{label}_ka"] = summary.event_age_kyr_bp + np.quantile(offsets, q, axis=0)
    return summary


def plot_uncertainty(events, controls, control_offsets):
    """Show proposal half-widths and accepted joint age-map perturbations."""
    configure_barker_style()
    fig, axes = plt.subplots(2, 1, figsize=(180 / 25.4, 125 / 25.4), sharex=True)
    fig.subplots_adjust(left=0.105, right=0.98, bottom=0.11, top=0.94, hspace=0.35)
    knots = controls.speleo_age_ka.to_numpy(float)
    width = controls.combined_uncertainty_ka.to_numpy(float)
    for label, ax in zip("ab", axes):
        ax.axvspan(*PUBLISHED_GAP_KA, color=ORANGE, alpha=0.13, lw=0)
        ax.axvspan(knots[-2], knots[-1], color=BLUE, alpha=0.07, lw=0)
        ax.axvline(knots[-2], color=BLUE, ls=":", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        add_panel_label(ax, label, x=-0.095, y=1.035)
        # BP age decreases to the right, following the project-wide convention.
        ax.set_xlim(405, 0)
        ax.set_xticks(np.arange(0, 401, 50))
    ax = axes[0]
    ax.plot(knots, width, color="0.2", lw=1.2)
    ax.scatter(knots[:-1], width[:-1], s=12, color="0.2", label="Table S1 combined uncertainty", zorder=3)
    ax.scatter(knots[-1], width[-1], marker="D", s=30, color=BLUE, label="Auxiliary control (400 ka)", zorder=4)
    ax.set_ylabel("Proposal half-width (kyr)")
    ax.set_ylim(0, 3.85)
    ax.text(290, 3.6, "No internal\nalignment", ha="center", va="top", fontsize=8)
    ax.legend(loc="upper right", frameon=False, fontsize=8)

    age = np.unique(np.r_[np.linspace(knots[0], knots[-1], 1001), knots])
    offsets = interpolate_offsets(age, controls, control_offsets)
    q025, median, q975 = np.quantile(offsets, [0.025, 0.5, 0.975], axis=0)
    ax = axes[1]
    ax.fill_between(age, q025, q975, color=BLUE, alpha=0.20, lw=0, label="95% MC range")
    for index in np.linspace(0, len(control_offsets) - 1, 25, dtype=int):
        ax.plot(knots, control_offsets[index], color="0.45", lw=0.45, alpha=0.4)
    ax.plot(age, median, color=BLUE, lw=1.4, label="MC median")
    ax.axhline(0, color="0.2", lw=0.65)
    ax.set_ylim(-3.35, 3.35)
    event_age = events.event_age_kyr_bp.to_numpy(float)
    ax.plot(event_age, np.full(len(events), -3.12), "|", color=CATALOGUE_COLORS["variable"],
            ms=5, label="Warming events", clip_on=False)
    ax.set_xlabel("Age (kyr BP)")
    ax.set_ylabel("Age offset (kyr)")
    ax.legend(loc="upper right", ncol=3, frameon=False, fontsize=8)
    return fig


def save_figure(fig, directory, stem, *, paper_export=True):
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.png", dpi=600)
    fig.savefig(directory / f"{stem}.pdf")
    if paper_export:
        copy_pdf_to_paper(directory / f"{stem}.pdf")
    plt.close(fig)


def main():
    root = OUTPUT_ROOT / "Barker2011"
    data = root / "data/processed" / RUN_NAME
    figures = root / "figures" / RUN_NAME
    events = pd.read_csv(BARKER_EVENT_CSVS["variable_threshold"], float_precision="round_trip")
    if REDRAW:
        controls = pd.read_csv(data / "age_control_points.csv", float_precision="round_trip")
        control_ages = pd.read_csv(data / "control_age_realizations.csv", float_precision="round_trip")
        columns = [f"age_ka_bp__{name}" for name in controls.control_id]
        values = control_ages[columns].to_numpy(float)
        if (control_ages.empty or not control_ages.realization_id.is_unique
                or not np.isfinite(values).all() or not np.all(np.diff(values, axis=1) > 0)):
            raise ValueError("Saved control ages must be finite, ordered and uniquely identified")
        control_offsets = values - controls.speleo_age_ka.to_numpy(float)
    else:
        raw_controls = pd.read_csv(CONTROL_CSV)
        controls = prepare_controls(raw_controls)
        events = prepare_events(events, controls)
        draws, control_offsets, diagnostics = sample_realizations(
            events, controls, N_REALIZATIONS, RANDOM_SEED)
        summary = summarize_ages(events, draws)
        ids = [f"Barker_MC_{i:05d}" for i in range(1, len(draws) + 1)]
        ages_table = pd.DataFrame(draws, columns=age_columns(events))
        ages_table.insert(0, "realization_id", ids)
        control_table = pd.DataFrame(control_offsets + controls.speleo_age_ka.to_numpy(float),
                                     columns=[f"age_ka_bp__{name}" for name in controls.control_id])
        control_table.insert(0, "realization_id", ids)
        settings = pd.DataFrame([(name, diagnostics[name]) for name in (
            "random_seed", "n_realizations", "n_proposals", "n_rejected_crossed_controls")],
            columns=["parameter", "value"])
        data.mkdir(parents=True, exist_ok=True)
        for name, table in (
            ("age_control_points", controls[["control_id", "speleo_age_ka", "combined_uncertainty_ka"]]),
            ("event_age_realizations", ages_table), ("control_age_realizations", control_table),
            ("event_age_uncertainty_summary", summary), ("sampling_settings", settings),
        ):
            # Preserve the interpolation coordinate and draws at round-trip precision.
            table.to_csv(data / f"{name}.csv", index=False, float_format="%.17g")
        print(f"Saved {len(draws):,} ordered chronologies for {len(events)} events; "
              f"proposal acceptance {diagnostics['acceptance_fraction']:.2%}")
    save_figure(plot_uncertainty(events, controls, control_offsets), figures, RUN_NAME,
                paper_export=EXPORT_PAPER and OUTPUT_ROOT.resolve() == PROJECT_ROOT.resolve())


if __name__ == "__main__":
    main()
