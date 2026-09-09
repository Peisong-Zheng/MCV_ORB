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
import hashlib
import platform
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_figure_export import copy_pdf_to_paper
from Barker2011 import Barker2011_event_phase_analysis as main_analysis

ROOT = main_analysis.ROOT
RUN_NAME = "Barker2011_event_age_uncertainty"
CONTROL_CSV = ROOT / "data/raw/Barker2011_TableS1.csv"
SOURCE_PDF = ROOT / "references/Barker2011_SOM.pdf"
OUT_DATA_DIR = ROOT / "data/processed" / RUN_NAME
OUT_FIG_DIR = ROOT / "figures" / RUN_NAME
N_REALIZATIONS = 10_000
RANDOM_SEED = 20260908
AUXILIARY_AGE_KA = 400.0
N_TAIL_CONTROLS = 4
PUBLISHED_GAP_KA = (265.0, 315.0)
CONTROL_GAP_KA = (264.24, 317.70)
BLUE = "#4477AA"
ORANGE = "#CC9933"


def prepare_controls(path=CONTROL_CSV):
    """Read published uncertainties and append the explicitly synthetic endpoint."""
    raw = pd.read_csv(path)
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


def prepare_events(controls):
    events = main_analysis.load_barker_source()
    events.insert(0, "event_id", [f"Barker_S3_{row:03d}" for row in events.source_excel_row])
    age = events.event_age_ka.to_numpy(float)
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
    ages = events.event_age_ka.to_numpy(float)
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
        n_accepted += int(ordered.sum())
        n_proposed += batch_size
        n_rejected += int((~ordered).sum())
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


def summarize_ages(events, controls, draws, control_offsets):
    summary = events[["event_id", "event_age_ka", "left_control_id", "right_control_id",
                      "right_control_weight", "in_published_alignment_gap",
                      "in_long_control_interval", "beyond_last_published_control"]].copy()
    left, right, weight = interpolation_weights(events.event_age_ka, controls)
    half_width = controls.combined_uncertainty_ka.to_numpy(float)
    summary["interpolated_half_width_ka"] = (1 - weight) * half_width[left] + weight * half_width[right]
    # This is the pre-conditioning variance of a weighted sum of independent uniforms.
    summary["proposal_sd_ka"] = np.sqrt(((1 - weight) * half_width[left])**2
                                       + (weight * half_width[right])**2) / np.sqrt(3)
    offsets = draws - events.event_age_ka.to_numpy(float)
    for frame, values in ((summary, offsets), (controls, control_offsets)):
        frame["accepted_offset_mean_ka"] = values.mean(axis=0)
        frame["accepted_offset_sd_ka"] = values.std(axis=0, ddof=1)
        for label, q in (("q025", 0.025), ("median", 0.5), ("q975", 0.975)):
            frame[f"accepted_offset_{label}_ka"] = np.quantile(values, q, axis=0)
    for label in ("q025", "median", "q975"):
        summary[f"age_{label}_ka"] = summary.event_age_ka + summary[f"accepted_offset_{label}_ka"]
    return summary


def parameters_table(settings, sources):
    """Save the scientific settings and hashes of the actual run inputs."""
    values = dict(settings, python_version=platform.python_version(), numpy_version=np.__version__,
                  pandas_version=pd.__version__, scipy_version=scipy.__version__)
    for label, path in sources.items():
        values[f"source_{label}"] = str(path.relative_to(ROOT.parent))
        values[f"sha256_{label}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.DataFrame(values.items(), columns=["parameter", "value"])


def plot_uncertainty(events, controls, control_offsets):
    """Show proposal half-widths and accepted joint age-map perturbations."""
    main_analysis.configure_plot_style()
    fig, axes = plt.subplots(2, 1, figsize=(180 / 25.4, 125 / 25.4), sharex=True)
    fig.subplots_adjust(left=0.105, right=0.98, bottom=0.11, top=0.94, hspace=0.35)
    knots = controls.speleo_age_ka.to_numpy(float)
    width = controls.combined_uncertainty_ka.to_numpy(float)
    for label, ax in zip("ab", axes):
        ax.axvspan(*PUBLISHED_GAP_KA, color=ORANGE, alpha=0.13, lw=0)
        ax.axvspan(knots[-2], knots[-1], color=BLUE, alpha=0.07, lw=0)
        ax.axvline(knots[-2], color=BLUE, ls=":", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(-0.095, 1.035, label, transform=ax.transAxes, fontweight="bold", fontsize=11)
        ax.set_xlim(0, 405)
        ax.set_xticks(np.arange(0, 401, 50))
    ax = axes[0]
    ax.plot(knots, width, color="0.2", lw=1.2)
    ax.scatter(knots[:-1], width[:-1], s=12, color="0.2", label="Table S1 combined uncertainty", zorder=3)
    ax.scatter(knots[-1], width[-1], marker="D", s=30, color=BLUE, label="Auxiliary control (400 ka)", zorder=4)
    ax.set_ylabel("Proposal half-width (kyr)")
    ax.set_ylim(0, 3.85)
    ax.text(290, 3.6, "No internal\nalignment", ha="center", va="top", fontsize=8)
    ax.legend(loc="upper left", frameon=False, fontsize=8)

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
    event_age = events.event_age_ka.to_numpy(float)
    ax.plot(event_age, np.full(len(events), -3.12), "|", color=main_analysis.EVENT_COLOR,
            ms=5, label="Warming events", clip_on=False)
    ax.set_xlabel("Age (kyr BP)")
    ax.set_ylabel("Age offset (kyr)")
    ax.legend(loc="upper left", ncol=3, frameon=False, fontsize=8)
    return fig


def save_figure(fig, directory, stem):
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.png", dpi=600)
    fig.savefig(directory / f"{stem}.pdf")
    copy_pdf_to_paper(directory / f"{stem}.pdf")
    plt.close(fig)


def main():
    controls = prepare_controls()
    events = prepare_events(controls)
    draws, control_offsets, diagnostics = sample_realizations(events, controls)
    controls = controls.copy()
    summary = summarize_ages(events, controls, draws, control_offsets)
    ids = [f"Barker_MC_{i:05d}" for i in range(1, len(draws) + 1)]
    ages_table = pd.DataFrame(draws, columns=age_columns(events))
    ages_table.insert(0, "realization_id", ids)
    control_table = pd.DataFrame(control_offsets + controls.speleo_age_ka.to_numpy(float),
                                 columns=[f"age_ka_bp__{name}" for name in controls.control_id])
    control_table.insert(0, "realization_id", ids)
    settings = dict(diagnostics, n_events=len(events), n_published_controls=60,
                    n_auxiliary_controls=1, auxiliary_age_ka=AUXILIARY_AGE_KA,
                    auxiliary_half_width_ka=controls.combined_uncertainty_ka.iloc[-1],
                    auxiliary_unfloored_extrapolation_ka=controls.unfloored_extrapolation_ka.iloc[-1],
                    tail_control_count=N_TAIL_CONTROLS,
                    age_coordinate="published SpeleoAge; BP1950 assumed, exact epoch unverified",
                    proposal="independent Uniform(-combined uncertainty, +combined uncertainty)",
                    uncertainty_convention="half-width assumed; source sigma/coverage level unspecified",
                    interpolation="linear offsets on fixed original SpeleoAge coordinates",
                    order_condition="reject entire nonmonotone control maps; never sort sampled ages",
                    gap_interpolation="264.24-317.70 ka; no additional internal perturbation",
                    n_events_in_published_gap=int(events.in_published_alignment_gap.sum()),
                    n_events_in_long_control_interval=int(events.in_long_control_interval.sum()),
                    n_events_beyond_last_control=int(events.beyond_last_published_control.sum()),
                    extra_event_picking_error=False, forcing_chronology_perturbed=False,
                    edc3_used_in_sampling=False, intervals="accepted ensemble quantiles, not confidence intervals")
    parameters = parameters_table(settings, dict(table_s1=CONTROL_CSV, som_pdf=SOURCE_PDF,
                                  events=main_analysis.BARKER_XLS, sampler=Path(__file__).resolve(),
                                  main_analysis=Path(main_analysis.__file__).resolve()))
    OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in (("age_control_points", controls), ("event_catalogue_used", events),
                        ("event_age_realizations", ages_table), ("control_age_realizations", control_table),
                        ("event_age_uncertainty_summary", summary), ("parameters_and_provenance", parameters)):
        # Preserve the interpolation coordinate and draws at round-trip precision.
        table.to_csv(OUT_DATA_DIR / f"{name}.csv", index=False, float_format="%.17g")
    save_figure(plot_uncertainty(events, controls, control_offsets), OUT_FIG_DIR, RUN_NAME)
    print(f"Saved {len(draws):,} ordered chronologies for {len(events)} events; "
          f"proposal acceptance {diagnostics['acceptance_fraction']:.2%}")


if __name__ == "__main__":
    main()
