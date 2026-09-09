"""Sample combined NGRIP age uncertainty. Run from the NGRIP directory.

Run ngrip_data_preparation.ipynb first. The 5-kyr cumulative chronology model,
definition sigmas, proposal order and seed match the previous combined run.
MCE/2 and the modeled ±4.5% envelope are working Gaussian scales, not bounds.
"""

from pathlib import Path
import sys
import hashlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# The paper exporter lives in the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_figure_export import copy_pdf_to_paper

EVENTS_CSV = Path("data/processed/ngrip_warming_cooling_starts.csv")
GRID_CSV = Path("data/processed/ngrip_chronology_grid.csv")
OUT_DATA_DIR = Path("data/processed/ngrip_event_age_uncertainty")
OUT_FIG_DIR = Path("figures/ngrip_event_age_uncertainty")
N_REALIZATIONS = 10_000
RANDOM_SEED = 20260907  # Previous combined seed, not the old base seed 20260905.
KNOT_SPACING_KA = 5.0


def chronology_process_basis(events, grid, knot_spacing_ka=KNOT_SPACING_KA):
    """Map shared Gaussian increments at prepared knots to event ages."""
    selected = grid.loc[grid["knot_spacing_ka"].eq(knot_spacing_ka)]
    if selected.empty:
        raise ValueError("Prepare the requested knot spacing in the notebook first")
    knots = selected["knot_age_ka_b2k"].to_numpy(float)
    variance = (selected["chronology_envelope_ka"].to_numpy(float) / 2) ** 2
    increments = np.diff(np.r_[0.0, variance])
    ages = events["age_ka_b2k"].to_numpy(float)
    if (not np.isfinite(knots).all() or not np.isfinite(variance).all()
            or np.any(np.diff(knots) <= 0) or np.any(increments < -1e-12)):
        raise ValueError("Chronology knots must increase with nondecreasing variance")
    if not np.isfinite(ages).all() or np.any(ages < knots[0]) or np.any(ages > knots[-1]):
        raise ValueError("Event ages must lie within the prepared chronology grid")

    # Cumulative variance increments retain correlation between nearby events.
    knot_basis = np.tril(np.ones((len(knots), len(knots)))) * np.sqrt(
        np.maximum(increments, 0)
    )[None, :]
    upper = np.minimum(np.searchsorted(knots, ages, side="right"), len(knots) - 1)
    lower = np.maximum(upper - 1, 0)
    weight = (ages - knots[lower]) / (knots[upper] - knots[lower])
    basis = (1 - weight[:, None]) * knot_basis[lower] + weight[:, None] * knot_basis[upper]
    return basis, knots, knot_basis


def sample_age_realizations(events, grid, n_realizations=N_REALIZATIONS,
                            seed=RANDOM_SEED, knot_spacing_ka=KNOT_SPACING_KA):
    """Draw combined errors and reject crossed age maps or transition picks."""
    ages = events["age_ka_bp"].to_numpy(float)
    sigma = events["definition_sigma_yr"].to_numpy(float) / 1000
    if n_realizations <= 0:
        raise ValueError("n_realizations must be positive")
    if not np.isfinite(ages).all() or np.any(np.diff(ages) <= 0):
        raise ValueError("Event ages must increase from younger to older")
    if not np.isfinite(sigma).all() or np.any(sigma < 0):
        raise ValueError("Definition sigmas must be finite and nonnegative")
    basis, knots, knot_basis = chronology_process_basis(events, grid, knot_spacing_ka)
    rng = np.random.default_rng(seed)
    accepted = []
    n_accepted = n_proposed = n_map_rejected = n_crossed = 0
    while n_accepted < n_realizations:
        # Keep the old batch sizes and RNG call order to reproduce saved ages.
        batch_size = max(1000, n_realizations - n_accepted)
        innovations = rng.normal(size=(batch_size, len(knots)))
        offsets = innovations @ basis.T
        knot_ages = knots + innovations @ knot_basis.T
        map_ok = np.all(np.diff(knot_ages, axis=1) > 0, axis=1)
        proposals = np.broadcast_to(ages, (batch_size, len(ages))).copy()
        proposals += offsets
        proposals += rng.normal(0.0, sigma, size=proposals.shape)
        ordered = np.all(np.diff(proposals, axis=1) > 0, axis=1)
        keep = map_ok & ordered
        accepted.append(proposals[keep])
        n_accepted += int(keep.sum())
        n_proposed += batch_size
        n_map_rejected += int((~map_ok).sum())
        n_crossed += int((map_ok & ~ordered).sum())
        if n_proposed > max(100_000, 100 * n_realizations):
            raise RuntimeError("Too few ordered age proposals were accepted")
    diagnostics = {
        "n_realizations": n_realizations, "random_seed": seed,
        "chronology_knot_spacing_ka": knot_spacing_ka,
        "n_proposals": n_proposed,
        "n_rejected_nonmonotonic_chronology": n_map_rejected,
        "n_rejected_for_event_crossing": n_crossed,
        "n_unused_accepted_tail": n_accepted - n_realizations,
        "acceptance_fraction": n_accepted / n_proposed,
    }
    return np.vstack(accepted)[:n_realizations], diagnostics


def realization_table(events, draws):
    """One row is one jointly ordered chronology, in kyr BP1950."""
    table = pd.DataFrame(draws, columns=[f"age_ka_bp__{label}" for label in events.event_label])
    table.insert(0, "realization_id", [f"NGRIP_{i:05d}" for i in range(1, len(table) + 1)])
    return table


def plot_uncertainty(events, draws):
    """Plot event-level input scales and combined ages at publication size."""
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 9, "axes.linewidth": 0.7,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    age = events["age_ka_bp"].to_numpy(float)
    envelope = events["chronology_envelope_yr"].to_numpy(float) / 1000
    definition = events["definition_sigma_yr"].to_numpy(float) / 1000
    low, median, high = np.quantile(draws - age, [0.025, 0.5, 0.975], axis=0)
    join_age = 60.202 - 0.05
    blue, orange = "#0072B2", "#D55E00"
    # A 180-mm figure preserves readable event labels at double-column width.
    fig, axes = plt.subplots(2, 1, figsize=(180 / 25.4, 166 / 25.4),
                             sharex=True, gridspec_kw={"height_ratios": [1.2, 1]})
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.085, top=0.93, hspace=0.15)

    ax = axes[0]
    ax.plot(age, envelope, color="0.2", lw=1.4, zorder=4,
            label="Chronology envelope (≈2σ)")
    for event_type, marker, color in (("warming", "o", orange),
                                       ("cooling", "s", blue)):
        selected = events["event_type"].eq(event_type).to_numpy()
        x, y = age[selected], definition[selected]
        ax.scatter(x, y, s=16, marker=marker, color=color, edgecolor="white",
                   linewidth=0.35, zorder=5, label=f"{event_type.capitalize()} (1σ)")
        # Spread crowded names horizontally; only text moves, never event ages.
        label_x = x.copy()
        for i in range(1, len(label_x)):
            label_x[i] = max(label_x[i], label_x[i - 1] + 1.7)
        label_x -= np.mean(label_x - x)
        for point_x, point_y, text_x, label in zip(
                x, y, label_x, events.loc[selected, "event_label"]):
            # Separate the two label bands where definition scales overlap.
            if event_type == "warming":
                text_y, alignment = min(point_y * 0.67, 0.012), "top"
            else:
                text_y = point_y * 1.5 if point_y < 0.01 else max(point_y * 1.5, 0.08)
                alignment = "bottom"
            ax.annotate(label, xy=(point_x, point_y), xytext=(text_x, text_y),
                        fontsize=7, color=color, rotation=90, ha="center", va=alignment,
                        arrowprops={"arrowstyle": "-", "color": color, "lw": 0.4,
                                    "alpha": 0.6, "shrinkA": 1.5, "shrinkB": 2}, zorder=6)
    ax.set(yscale="log", ylim=(0.0007, 12), ylabel="Input uncertainty (kyr)")
    ax.set_yticks([0.001, 0.01, 0.1, 1, 10], labels=["0.001", "0.01", "0.1", "1", "10"])
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower left",
              bbox_to_anchor=(0, 1.025), borderaxespad=0, handlelength=2,
              columnspacing=1.6, handletextpad=0.5)

    ax = axes[1]
    ax.fill_between(age, low, high, color=blue, alpha=0.19, lw=0,
                    label="2.5–97.5% range", zorder=1)
    sample_index = np.linspace(0, len(draws) - 1, 30, dtype=int)
    for index in sample_index:
        ax.plot(age, draws[index] - age, color="0.45", lw=0.45, alpha=0.28,
                label="Combined realizations" if index == 0 else None, zorder=2)
    ax.plot(age, median, color=blue, lw=1.4, label="Median", zorder=4)
    ax.axhline(0, color="0.4", lw=0.6, zorder=0)
    # Keep every displayed trajectory visible, with room above for regime labels.
    extent = max(np.max(np.abs(draws[sample_index] - age)),
                 np.max(np.abs(low)), np.max(np.abs(high)))
    limit = np.ceil(extent) + 2
    ax.set(xlabel="Age (kyr BP)", ylabel="Age offset (kyr)", ylim=(-limit, limit))
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5, integer=True))
    ax.legend(frameon=False, fontsize=8, loc="lower left", borderaxespad=0.6,
              handlelength=2.2, labelspacing=0.35)
    # Label both sides of the exact counted/model-extension boundary.
    for x, label in (((10 + join_age) / 2, "Layer counted (GICC05)"),
                      ((join_age + 123) / 2, "Model extension (GICC05modelext)")):
        ax.text(x, 0.955, label, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8, color="0.2")
    for letter, axis in zip("ab", axes):
        axis.axvline(join_age, color="0.45", lw=0.75, ls=(0, (4, 3)), zorder=0)
        axis.text(-0.085, 1.01, f"({letter})", transform=axis.transAxes,
                  ha="left", va="bottom", fontsize=10, fontweight="bold")
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(which="major", direction="out", length=3, width=0.7)
        axis.tick_params(which="minor", direction="out", length=1.5, width=0.5)
    axes[1].set(xlim=(10, 123), xticks=np.arange(20, 121, 20))
    return fig


def main():
    events = pd.read_csv(EVENTS_CSV)
    grid = pd.read_csv(GRID_CSV, float_precision="round_trip")
    draws, diagnostics = sample_age_realizations(events, grid)
    OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    realization_table(events, draws).to_csv(
        OUT_DATA_DIR / "ngrip_event_age_realizations.csv", index=False, float_format="%.6f"
    )

    # Distinguish nominal envelope/2 from the interpolated, pre-rejection SD.
    basis, _, _ = chronology_process_basis(events, grid)
    low, median, high = np.quantile(draws, [0.025, 0.5, 0.975], axis=0)
    summary = events[["event_label", "event_type", "age_ka_bp",
                      "definition_uncertainty_code", "definition_sigma_yr"]].rename(
        columns={"age_ka_bp": "point_age_ka_bp"}
    )
    summary["chronology_regime"] = np.where(events.chronology_source.eq("MCE"), "GICC05", "GICC05modelext")
    summary["chronology_envelope_yr"] = events.chronology_envelope_yr
    summary["chronology_mc_sd_yr"] = np.sqrt((basis**2).sum(axis=1)) * 1000
    summary["chronology_nominal_target_sigma_yr"] = events.chronology_envelope_yr / 2
    summary["combined_mc_sd_yr"] = draws.std(axis=0, ddof=1) * 1000
    summary["combined_mc_age_2p5_ka_bp"] = low
    summary["combined_mc_age_median_ka_bp"] = median
    summary["combined_mc_age_97p5_ka_bp"] = high
    summary.to_csv(OUT_DATA_DIR / "ngrip_event_age_uncertainty_summary.csv", index=False)

    parameters = {
        "scenario": "combined", **diagnostics, "age_epoch": "kyr BP1950",
        "chronology_model": "cumulative Gaussian variance increments; linear offset interpolation",
        "knot_sigma": "MCE-equivalent envelope / 2; not a hard bound",
        "modelext_envelope": "0.045 * age_b2k; exact 60.202 ka counted endpoint retained",
        "order_condition": "reject crossed knot maps and combined event ages; never sort draws",
        "observation_support": "not truncated here; checked by downstream PI",
        "omitted_uncertainties": "systematic counting bias; event membership; forcing chronology; alternate model extension",
    }
    for name, path in (("events", EVENTS_CSV), ("grid", GRID_CSV),
                       ("sampler", Path("ngrip_event_age_uncertainty.py")),
                       ("preparation", Path("ngrip_data_preparation.ipynb"))):
        parameters[f"{name}_path"] = str(path)
        parameters[f"{name}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    pd.DataFrame(parameters.items(), columns=["parameter", "value"]).to_csv(
        OUT_DATA_DIR / "parameters_and_provenance.csv", index=False
    )
    fig = plot_uncertainty(events, draws)
    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(OUT_FIG_DIR / f"ngrip_event_age_uncertainty.{extension}", dpi=600)
    copy_pdf_to_paper(OUT_FIG_DIR / "ngrip_event_age_uncertainty.pdf")
    plt.close(fig)
    print(f"Saved {len(draws):,} combined chronologies; acceptance {diagnostics['acceptance_fraction']:.2%}")


if __name__ == "__main__":
    main()
