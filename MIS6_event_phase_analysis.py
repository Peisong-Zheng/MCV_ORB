#!/usr/bin/env python3
"""Rayleigh and conditional-PI analysis of the 21 MIS 6 warming events.

The model definition follows ``NGRIP/ngrip_event_phase_analysis.py``.  All
events are treated as one warming catalogue.  Proxy sampling resolution and
event-age uncertainty are deliberately not included in this exploratory test.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from toolbox.mis6_pi import (
    ANALYSIS_END_KA,
    ANALYSIS_START_KA,
    BIN_WIDTH_KA,
    DATASET_ID,
    EVENTS_CSV,
    EVENT_COLOR,
    EVENT_LABEL,
    EVENT_TYPE,
    FULL_MODEL_ID,
    FULL_TERMS,
    HISTORY_WINDOW_KA,
    REDUCED_MODEL_ID,
    REDUCED_TERMS,
    RESOLUTION_COVARIATE_INCLUDED,
    load_events,
    phase_rate_multiplier,
    run_predictive_information,
)
from toolbox.orbital_phase import (
    build_phase_series,
    build_rayleigh_results,
    plot_rayleigh_polar,
    sample_event_phases,
)
from toolbox.project_config import (
    CO2_XLSX,
    LR04_XLSX,
    ORBITAL_DRIVER_SETTINGS,
    PRE_TXT,
    PROJECT_ROOT,
)


RUN_NAME = "MIS6_event_phase_analysis"
OUT_DATA_DIR = PROJECT_ROOT / "data/processed" / RUN_NAME
OUT_FIG_DIR = PROJECT_ROOT / "figures" / RUN_NAME


def configure_plot_style() -> None:
    """Use clear journal-scale typography and editable PDF fonts."""

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def run_rayleigh(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    """Sample precession phase and run the shared finite-sample Rayleigh test."""

    phase_product = build_phase_series("pre", ORBITAL_DRIVER_SETTINGS["pre"])
    phase_input = pd.DataFrame(
        {
            "event_index": events["composite_event_id"],
            "event_age_ka": events["event_age_ka_bp"],
            "event_type": EVENT_TYPE,
            # This is the catalogue label; physical IDs belong in event_index.
            "event_label": EVENT_LABEL,
        }
    )
    event_phases = sample_event_phases(phase_input, {"pre": phase_product})
    rayleigh = build_rayleigh_results(event_phases)
    if event_phases["phase_extrapolated"].astype(bool).any():
        raise RuntimeError("At least one event phase was extrapolated.")
    return event_phases, rayleigh, phase_product


def build_analysis_summary(
    events: pd.DataFrame,
    rayleigh: pd.DataFrame,
    pi: dict[str, object],
) -> pd.DataFrame:
    """Collect the main unconditional and conditional results in one row."""

    ray = rayleigh.query("driver == 'pre' and event_type == @EVENT_TYPE").iloc[0]
    model_summary = pi["model_summary"]
    likelihood_tests = pi["likelihood_tests"]
    fit_frame = pi["fit_frame"]
    full = model_summary.query("model_id == @FULL_MODEL_ID").iloc[0]
    test = likelihood_tests.iloc[0]
    n_eta_clipped = int(
        model_summary[["n_eta_clipped_low", "n_eta_clipped_high"]].to_numpy(int).sum()
    )
    return pd.DataFrame(
        [
            {
                "event_type": EVENT_TYPE,
                "event_label": EVENT_LABEL,
                "analysis_start_ka_bp": ANALYSIS_START_KA,
                "analysis_end_ka_bp": ANALYSIS_END_KA,
                "n_catalogue_events": len(events),
                "n_rayleigh_events": int(ray["n_phase_events_used"]),
                "rayleigh_mean_phase_deg": float(ray["mean_phase_deg"]),
                "rayleigh_mean_resultant_length": float(ray["mean_resultant_length"]),
                "rayleigh_R": float(ray["rayleigh_R"]),
                "rayleigh_z": float(ray["rayleigh_z"]),
                "rayleigh_p": float(ray["rayleigh_p"]),
                "rayleigh_significant_0p05": float(ray["rayleigh_p"]) < 0.05,
                "n_rayleigh_extrapolated_events": int(
                    ray["n_extrapolated_phase_events"]
                ),
                "n_predictive_bins": int(len(fit_frame)),
                "n_predictive_events": int(test["n_events"]),
                "predictive_support_start_ka_bp": float(
                    fit_frame["bin_start_ka"].min()
                ),
                "predictive_support_end_ka_bp": float(fit_frame["bin_end_ka"].max()),
                "predictive_preferred_phase_deg": float(
                    full["pre_phase_preferred_deg"]
                ),
                "predictive_rate_ratio_max_vs_min": float(
                    full["pre_phase_rate_ratio_max_vs_min"]
                ),
                "predictive_LR": float(test["LR_statistic"]),
                "predictive_nominal_LR_p": float(test["LR_p_value"]),
                "predictive_significant_0p05": float(test["LR_p_value"]) < 0.05,
                "predictive_log_likelihood_reduced": float(test["loglik_reduced"]),
                "predictive_log_likelihood_full": float(test["loglik_full"]),
                "predictive_bits_per_event": float(test["info_bits_per_event"]),
                "predictive_delta_AICc_full_minus_reduced": float(
                    test["delta_AICc_full_minus_reduced"]
                ),
                "predictive_likelihood_nesting_ok": bool(test["likelihood_nesting_ok"]),
                "predictive_eta_clipping_count": n_eta_clipped,
                "resolution_covariate_included": False,
                "event_age_uncertainty_propagated": False,
                "all_models_converged": bool(model_summary["converged"].all()),
            }
        ]
    )


def plot_timeline(
    events: pd.DataFrame, event_phases: pd.DataFrame, phase_product: object
) -> plt.Figure:
    """Plot the events against the precession series used in both tests."""

    series = phase_product.series.loc[
        phase_product.series["age_ka"].between(ANALYSIS_START_KA, ANALYSIS_END_KA)
    ]
    fig, axes = plt.subplots(
        2, 1, figsize=(7.1, 4.0), sharex=True, height_ratios=(3.0, 0.85)
    )
    axes[0].plot(series["age_ka"], series["value"], color="#555555", lw=1.25)
    axes[0].scatter(
        event_phases["event_age_ka"],
        event_phases["orbital_value_at_event"],
        s=31,
        color=EVENT_COLOR,
        edgecolor="white",
        linewidth=0.45,
        label=EVENT_LABEL,
        zorder=3,
    )
    axes[1].vlines(events["event_age_ka_bp"], 0.0, 1.0, color=EVENT_COLOR, lw=1.15)
    axes[0].set_ylabel("Precession index")
    axes[0].set_title("MIS 6 composite warming events and precession", loc="left")
    axes[0].legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.88,
        loc="upper right",
    )
    axes[1].set(yticks=[], ylim=(0, 1), xlabel="Age (Kyr BP)")
    for axis in axes:
        axis.set_xlim(ANALYSIS_START_KA, ANALYSIS_END_KA)
        axis.grid(True, color="#e5e5e5", lw=0.65)
    fig.tight_layout()
    return fig


def plot_predictive_phase_response(
    model_summary: pd.DataFrame, likelihood_tests: pd.DataFrame
) -> plt.Figure:
    """Plot the fitted phase multiplier from the full conditional model."""

    full = model_summary.query("model_id == @FULL_MODEL_ID").iloc[0]
    test = likelihood_tests.iloc[0]
    phase_deg = np.linspace(0.0, 360.0, 721)
    response = phase_rate_multiplier(
        phase_deg,
        float(full["beta_pre_phase_sin"]),
        float(full["beta_pre_phase_cos"]),
    )
    preferred = float(full["pre_phase_preferred_deg"])

    fig, axis = plt.subplots(figsize=(6.2, 4.0))
    axis.plot(phase_deg, response, color=EVENT_COLOR, lw=2.0)
    axis.axhline(1.0, color="#777777", lw=0.9, ls=":")
    axis.axvline(preferred, color="#333333", lw=1.0, ls=(0, (3, 2)))
    axis.text(
        0.03,
        0.96,
        (
            f"N = {int(test['n_events'])}\n"
            f"LR = {float(test['LR_statistic']):.2f}, "
            f"nominal p = {float(test['LR_p_value']):.3g}\n"
            f"bits/event = {float(test['info_bits_per_event']):.3f}\n"
            f"full - reduced AICc = "
            f"{float(test['delta_AICc_full_minus_reduced']):+.2f}\n"
            f"peak = {preferred:.1f} degrees\n"
            f"max/min rate = "
            f"{float(full['pre_phase_rate_ratio_max_vs_min']):.2f}"
        ),
        transform=axis.transAxes,
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.92},
    )
    axis.set(
        xlim=(0, 360),
        xticks=[0, 90, 180, 270, 360],
        xlabel="Precession phase (degrees)",
        ylabel="Multiplicative phase term in event rate",
        title="Conditional PI phase response (exploratory)",
    )
    axis.grid(True, color="#e5e5e5", lw=0.65)
    fig.text(
        0.5,
        0.01,
        "Reduced: 5 Kyr event history + LR04 + CO2; 0 degrees = precession minimum",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    return fig


def save_figure(fig: plt.Figure, stem: str) -> None:
    """Save a review PNG and vector PDF."""

    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG_DIR / f"{stem}.png", dpi=400, bbox_inches="tight")
    fig.savefig(OUT_FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def build_parameters(events: pd.DataFrame, pi: dict[str, object]) -> pd.DataFrame:
    """Record model choices, source files, and predictor scaling."""
    fit_frame = pi["fit_frame"]
    rows = [
        (
            "analysis_start",
            ANALYSIS_START_KA,
            "Kyr BP",
            "rounded inward from continuous proxy coverage",
        ),
        (
            "analysis_end",
            ANALYSIS_END_KA,
            "Kyr BP",
            "rounded inward from continuous proxy coverage",
        ),
        ("bin_width", BIN_WIDTH_KA, "Kyr", "Poisson event-count bins"),
        ("history_window", HISTORY_WINDOW_KA, "Kyr", "older warming events"),
        ("catalogue_event_count", len(events), "events", "all treated as warming"),
        (
            "predictive_event_count",
            int(fit_frame["event_count"].sum()),
            "events",
            "oldest-edge incomplete-history bins excluded",
        ),
        ("reduced_model_terms", "+".join(REDUCED_TERMS), "", "matches NGRIP"),
        ("full_model_terms", "+".join(FULL_TERMS), "", "matches NGRIP"),
        (
            "reference_analysis",
            "NGRIP/ngrip_event_phase_analysis.py",
            "",
            "source of the Rayleigh and conditional-PI model definition",
        ),
        (
            "NGRIP_resolution_covariate_included",
            False,
            "",
            "confirmed from the NGRIP reduced/full model terms",
        ),
        (
            "resolution_covariate_included",
            False,
            "",
            "deliberately omitted; local proxy resolution is not a model term",
        ),
        (
            "event_age_uncertainty_propagated",
            False,
            "",
            "point ages are treated as exact in this analysis",
        ),
        (
            "p_value_method",
            "nominal asymptotic chi-square LRT",
            "",
            "two added phase coefficients; no resampling calibration",
        ),
        (
            "PI_definition",
            "in-sample nested Poisson log-likelihood gain",
            "",
            "not cross-validated mutual information",
        ),
        (
            "precession_phase_zero",
            "precession-index minimum",
            "",
            "toolbox phase convention",
        ),
        (
            "event_catalogue",
            str(EVENTS_CSV.relative_to(PROJECT_ROOT)),
            "",
            "provisional MIS6 composite chronology",
        ),
        (
            "precession_source",
            str(PRE_TXT.relative_to(PROJECT_ROOT)),
            "",
            "orbital input",
        ),
        (
            "lr04_source",
            str(LR04_XLSX.relative_to(PROJECT_ROOT)),
            "",
            "climate covariate",
        ),
        (
            "co2_source",
            str(CO2_XLSX.relative_to(PROJECT_ROOT)),
            "",
            "climate covariate",
        ),
    ]

    # Scaling depends on the analysis interval, so keep it beside the model setup.
    forcing_units = {
        "lr04": "per mil",
        "co2": "ppm",
        "pre_phase_sin": "dimensionless",
        "pre_phase_cos": "dimensionless",
    }
    for forcing in pi["scale_summary"].itertuples(index=False):
        values = (
            f"mean={forcing.mean:.12g}; min={forcing.min:.12g}; "
            f"max={forcing.max:.12g}; range={forcing.range:.12g}"
        )
        rows.append(
            (
                f"forcing_scale_{forcing.forcing_id}",
                values,
                forcing_units[str(forcing.forcing_id)],
                str(forcing.source),
            )
        )
    return pd.DataFrame(rows, columns=["parameter", "value", "unit", "note"])


def compact_event_phases(event_phases: pd.DataFrame) -> pd.DataFrame:
    """Keep one readable phase record for each composite event."""
    columns = {
        "event_index": "composite_event_id",
        "event_age_ka": "event_age_ka_bp",
        "orbital_value_at_event": "precession_index_at_event",
        "phase_deg": "precession_phase_deg",
        "phase_fraction": "precession_phase_fraction",
        "phase_extrapolated": "phase_extrapolated",
    }
    return event_phases.loc[:, list(columns)].rename(columns=columns)


def compact_coefficients(coefficients: pd.DataFrame) -> pd.DataFrame:
    """Keep model terms and their fitted effects without repeated labels."""
    columns = ["model_id", "term", "beta", "rate_ratio_per_unit"]
    return coefficients.loc[:, columns].copy()


def write_outputs(
    events: pd.DataFrame,
    event_phases: pd.DataFrame,
    pi: dict[str, object],
    summary: pd.DataFrame,
    output_dir: Path = OUT_DATA_DIR,
) -> None:
    """Write the four tables retained for the point-age experiment."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "analysis_summary.csv": summary,
        "event_precession_phases.csv": compact_event_phases(event_phases),
        "predictive_coefficients.csv": compact_coefficients(pi["coefficients"]),
        "parameters_and_provenance.csv": build_parameters(events, pi),
    }
    for filename, table in tables.items():
        table.to_csv(output_dir / filename, index=False)


def main() -> None:
    """Run the point-age analysis and write its retained outputs."""

    configure_plot_style()
    events = load_events()
    event_phases, rayleigh, phase_product = run_rayleigh(events)
    pi = run_predictive_information(events)
    summary = build_analysis_summary(events, rayleigh, pi)

    write_outputs(events, event_phases, pi, summary)

    save_figure(
        plot_timeline(events, event_phases, phase_product),
        "fig01_event_timeline_and_precession",
    )
    polar = plot_rayleigh_polar(
        event_phases,
        rayleigh,
        driver="pre",
        event_colors={EVENT_TYPE: EVENT_COLOR},
        annotate_mean_phase=True,
        show_panel_labels=False,
    )
    polar.set_size_inches(5.4, 4.8)
    polar.text(
        0.5,
        0.015,
        "0 degrees = precession minimum; 180 degrees = maximum",
        ha="center",
        fontsize=8.5,
    )
    polar.subplots_adjust(top=0.79, bottom=0.12, left=0.08, right=0.92)
    save_figure(polar, "fig02_rayleigh_precession_phase")
    save_figure(
        plot_predictive_phase_response(pi["model_summary"], pi["likelihood_tests"]),
        "fig03_predictive_information_phase_response",
    )

    print(
        summary[
            [
                "n_catalogue_events",
                "n_rayleigh_events",
                "rayleigh_mean_phase_deg",
                "rayleigh_p",
                "n_predictive_events",
                "predictive_preferred_phase_deg",
                "predictive_nominal_LR_p",
                "predictive_bits_per_event",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4g}")
    )
    print(f"\nData: {OUT_DATA_DIR}")
    print(f"Figures: {OUT_FIG_DIR}")


if __name__ == "__main__":
    main()
