"""Test the precession phase of published NGRIP GI and GS starts.

The 69 ages are the retained Rasmussen et al. (2014) Table 2 boundaries.  GI
starts are warming events and GS starts are cooling events; no event detector
is rerun here.
"""

from __future__ import annotations

from pathlib import Path
import sys


# Allow direct execution as ``python NGRIP/ngrip_event_phase_analysis.py``.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from toolbox import event_inputs, event_process as predictive, poisson
from toolbox.model_stats import nested_likelihood_metrics
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


RUN_NAME = "ngrip_event_phase_analysis"
NGRIP_DIR = Path(__file__).resolve().parent
EVENTS_CSV = NGRIP_DIR / "data/processed/ngrip_warming_cooling_starts.csv"
OUT_DATA_DIR = NGRIP_DIR / "data/processed" / RUN_NAME
OUT_FIG_DIR = NGRIP_DIR / "figures" / RUN_NAME

# Keep the settings visible: these are the only analysis choices in this first
# version.  The interval covers all 69 collapsed Table 2 starts.
ANALYSIS_START_KA = 12.0
ANALYSIS_END_KA = 120.0
BIN_WIDTH_KA = 0.2
HISTORY_WINDOW_KA = 5.0

DIRECTION_TYPES = ("warming", "cooling")
EVENT_TYPES = (*DIRECTION_TYPES, "all_transitions")
EVENT_LABELS = {
    "warming": "NGRIP warming starts (GI)",
    "cooling": "NGRIP cooling starts (GS)",
    "all_transitions": "All NGRIP transitions (GI + GS)",
}
EVENT_COLORS = {
    "warming": "#D55E00",
    "cooling": "#0072B2",
    "all_transitions": "#6A3D9A",
}

# The PI comparison asks whether precession phase adds information after a
# small event-process/climate-state baseline.  History always refers to the
# analyzed catalogue: prior GI starts for warming, prior GS starts for cooling,
# and prior GI or GS starts for the combined catalogue.  There is no Cheng-
# resolution term because these are published boundaries, not Cheng detections.
REDUCED_TERMS = (
    predictive.HISTORY_TERM,
    *predictive.CLIMATE_TERMS,
)
FULL_TERMS = REDUCED_TERMS + predictive.PHASE_TERMS
REDUCED_MODEL_ID = "history_climate"
FULL_MODEL_ID = "history_climate_precession"
COMPARISON_ID = "precession_after_history_climate"
RESOLUTION_COVARIATE_INCLUDED = False


def load_events(path: Path = EVENTS_CSV) -> pd.DataFrame:
    """Load the collapsed catalogue and check its defining assumptions."""

    events = pd.read_csv(path)
    required = {
        "event_label",
        "source_event_label",
        "event_type",
        "age_yr_b2k",
        "age_ka_b2k",
        "age_ka_bp",
    }
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"Event catalogue is missing columns: {sorted(missing)}")

    events = events.sort_values("age_ka_bp").reset_index(drop=True)
    counts = events.groupby("event_type").size().to_dict()
    if counts != {"cooling": 35, "warming": 34}:
        raise ValueError(f"Unexpected event counts: {counts}")
    if events["event_label"].duplicated().any():
        raise ValueError("Event labels must be unique within the catalogue.")
    if not np.allclose(events["age_ka_b2k"], events["age_yr_b2k"] / 1000.0):
        raise ValueError("The ka b2k ages must equal the year b2k ages / 1000.")
    if not np.allclose(events["age_ka_bp"], events["age_ka_b2k"] - 0.05):
        raise ValueError("The b2k-to-BP conversion must subtract 0.05 ka.")
    if (
        not events["age_ka_bp"]
        .between(ANALYSIS_START_KA, ANALYSIS_END_KA, inclusive="both")
        .all()
    ):
        raise ValueError("At least one event lies outside the analysis interval.")
    return events


def select_catalogue_events(events: pd.DataFrame, event_type: str) -> pd.DataFrame:
    """Return one directional catalogue or all transitions combined."""

    if event_type == "all_transitions":
        return events
    if event_type not in DIRECTION_TYPES:
        raise ValueError(f"Unknown event catalogue: {event_type}")
    return events.loc[events["event_type"].eq(event_type)]


def build_event_datasets(events: pd.DataFrame) -> list[event_inputs.EventDataset]:
    """Build separate warming/cooling and combined transition catalogues."""

    datasets = []
    for event_type in EVENT_TYPES:
        ages = select_catalogue_events(events, event_type)["age_ka_bp"]
        datasets.append(
            event_inputs.EventDataset(
                dataset_id=event_type,
                label=EVENT_LABELS[event_type],
                color=EVENT_COLORS[event_type],
                ages_ka=np.sort(ages.to_numpy(dtype=float)),
                source=str(EVENTS_CSV.relative_to(PROJECT_ROOT)),
            )
        )
    return datasets


def _phase_catalogues(event_phases: pd.DataFrame) -> pd.DataFrame:
    """Add a pooled view without resampling the same physical boundaries."""

    pooled = event_phases.copy()
    pooled["event_type"] = "all_transitions"
    pooled["event_label"] = EVENT_LABELS["all_transitions"]
    return pd.concat([event_phases, pooled], ignore_index=True)


def run_rayleigh(events: pd.DataFrame):
    """Sample every physical boundary once, then test three catalogue views."""

    phase_product = build_phase_series("pre", ORBITAL_DRIVER_SETTINGS["pre"])
    phase_events = pd.DataFrame(
        {
            "event_index": events["event_label"],
            "event_age_ka": events["age_ka_bp"],
            "event_type": events["event_type"],
            "event_label": events["event_type"].map(EVENT_LABELS),
        }
    )
    event_phases = sample_event_phases(phase_events, {"pre": phase_product})
    rayleigh = build_rayleigh_results(_phase_catalogues(event_phases))
    if event_phases["phase_extrapolated"].astype(bool).any():
        raise RuntimeError("At least one event phase was extrapolated.")
    return event_phases, rayleigh, phase_product


def run_predictive_information(events: pd.DataFrame) -> dict:
    """Fit reduced/full models to directional and combined catalogues."""

    binned, scale_summary, phase_extrema = event_inputs.build_binned_inputs(
        build_event_datasets(events),
        analysis_start_ka=ANALYSIS_START_KA,
        analysis_end_ka=ANALYSIS_END_KA,
        bin_width_ka=BIN_WIDTH_KA,
        lr04_path=LR04_XLSX,
        co2_path=CO2_XLSX,
        precession_path=PRE_TXT,
        project_root=PROJECT_ROOT,
    )
    binned = predictive.add_same_type_history(binned, HISTORY_WINDOW_KA)
    fit_frame = predictive.model_frame(binned)

    models = []
    for _, group in fit_frame.groupby("dataset_id", sort=False):
        models.extend(
            [
                poisson.fit_poisson_model(
                    group,
                    REDUCED_MODEL_ID,
                    REDUCED_TERMS,
                    "Catalogue history + LR04 + CO2",
                ),
                poisson.fit_poisson_model(
                    group,
                    FULL_MODEL_ID,
                    FULL_TERMS,
                    "Catalogue history + LR04 + CO2 + precession phase",
                ),
            ]
        )

    model_summary = poisson.build_model_summary(models, fit_frame)
    coefficients = poisson.build_coefficient_table(models)
    lookup = poisson.model_lookup(models)
    tests = []
    for event_type, group in fit_frame.groupby("dataset_id", sort=False):
        reduced = lookup[(event_type, REDUCED_MODEL_ID)]
        full = lookup[(event_type, FULL_MODEL_ID)]
        metrics = nested_likelihood_metrics(
            loglik_full=full.log_likelihood,
            loglik_reduced=reduced.log_likelihood,
            df=len(full.beta) - len(reduced.beta),
            n_bins=len(group),
            n_events=int(group["event_count"].sum()),
            aicc_full=full.aicc,
            aicc_reduced=reduced.aicc,
        )
        tests.append(
            {
                "dataset_id": event_type,
                "dataset_label": EVENT_LABELS[event_type],
                "comparison_id": COMPARISON_ID,
                "reduced_model_id": REDUCED_MODEL_ID,
                "full_model_id": FULL_MODEL_ID,
                **metrics,
                "reject_LR_at_0p05": metrics["LR_p_value"] < 0.05,
                "likelihood_nesting_ok": metrics["ll_gain_nats"] >= -1e-8,
            }
        )
    likelihood_tests = pd.DataFrame(tests)

    if not model_summary["converged"].all():
        raise RuntimeError("At least one predictive model did not converge.")
    if not likelihood_tests["likelihood_nesting_ok"].all():
        raise RuntimeError("A full-model likelihood is below its reduced model.")
    if (model_summary[["n_eta_clipped_low", "n_eta_clipped_high"]] > 0).any().any():
        raise RuntimeError("At least one predictive model used eta clipping.")

    return {
        "binned": binned,
        "fit_frame": fit_frame,
        "scale_summary": scale_summary,
        "phase_extrema": phase_extrema,
        "model_summary": model_summary,
        "coefficients": coefficients,
        "likelihood_tests": likelihood_tests,
    }


def phase_rate_multiplier(
    phase_deg: np.ndarray, beta_sin: float, beta_cos: float
) -> np.ndarray:
    """Return exp(beta_sin sin(theta) + beta_cos cos(theta))."""

    theta = np.deg2rad(np.asarray(phase_deg, dtype=float))
    return np.exp(beta_sin * np.sin(theta) + beta_cos * np.cos(theta))


def build_analysis_summary(
    events: pd.DataFrame,
    rayleigh: pd.DataFrame,
    model_summary: pd.DataFrame,
    likelihood_tests: pd.DataFrame,
    fit_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Put the results needed for interpretation in one compact table."""

    rows = []
    for event_type in EVENT_TYPES:
        ray = rayleigh.loc[
            rayleigh["driver"].eq("pre") & rayleigh["event_type"].eq(event_type)
        ].iloc[0]
        full = model_summary.loc[
            model_summary["dataset_id"].eq(event_type)
            & model_summary["model_id"].eq(FULL_MODEL_ID)
        ].iloc[0]
        test = likelihood_tests.loc[likelihood_tests["dataset_id"].eq(event_type)].iloc[
            0
        ]
        fitted_bins = (
            fit_frame.loc[fit_frame["dataset_id"].eq(event_type)]
            if fit_frame is not None
            else None
        )
        rows.append(
            {
                "event_type": event_type,
                "event_label": EVENT_LABELS[event_type],
                "analysis_start_ka_bp": ANALYSIS_START_KA,
                "analysis_end_ka_bp": ANALYSIS_END_KA,
                "n_catalogue_events": len(select_catalogue_events(events, event_type)),
                "n_rayleigh_events": int(ray["n_phase_events_used"]),
                "rayleigh_mean_phase_deg": float(ray["mean_phase_deg"]),
                "rayleigh_mean_resultant_length": float(ray["mean_resultant_length"]),
                "rayleigh_p": float(ray["rayleigh_p"]),
                "rayleigh_significant_0p05": float(ray["rayleigh_p"]) < 0.05,
                "n_predictive_bins": int(test["n_bins"]),
                "n_predictive_events": int(test["n_events"]),
                "predictive_support_start_ka_bp": (
                    float(fitted_bins["bin_start_ka"].min())
                    if fitted_bins is not None
                    else np.nan
                ),
                "predictive_support_end_ka_bp": (
                    float(fitted_bins["bin_end_ka"].max())
                    if fitted_bins is not None
                    else np.nan
                ),
                "predictive_preferred_phase_deg": float(
                    full["pre_phase_preferred_deg"]
                ),
                "predictive_rate_ratio_max_vs_min": float(
                    full["pre_phase_rate_ratio_max_vs_min"]
                ),
                "predictive_LR": float(test["LR_statistic"]),
                "predictive_LR_p": float(test["LR_p_value"]),
                "predictive_significant_0p05": float(test["LR_p_value"]) < 0.05,
                "predictive_bits_per_event": float(test["info_bits_per_event"]),
                "predictive_delta_AICc_full_minus_reduced": float(
                    test["delta_AICc_full_minus_reduced"]
                ),
                "likelihood_nesting_ok": bool(test["likelihood_nesting_ok"]),
                "all_models_converged": bool(
                    model_summary.loc[
                        model_summary["dataset_id"].eq(event_type), "converged"
                    ].all()
                ),
                "eta_clipping_used": bool(
                    model_summary.loc[
                        model_summary["dataset_id"].eq(event_type),
                        ["n_eta_clipped_low", "n_eta_clipped_high"],
                    ]
                    .gt(0)
                    .any()
                    .any()
                ),
                "resolution_covariate_included": RESOLUTION_COVARIATE_INCLUDED,
                "event_age_uncertainty_propagated": False,
            }
        )
    return pd.DataFrame(rows)


def event_phase_output(event_phases: pd.DataFrame) -> pd.DataFrame:
    """Keep one concise phase record for each physical GI or GS boundary."""

    columns = [
        "event_index",
        "event_type",
        "event_age_ka",
        "driver",
        "orbital_value_at_event",
        "phase_rad",
        "phase_deg",
        "phase_extrapolated",
    ]
    return event_phases.loc[:, columns].rename(columns={"event_index": "event_id"})


def plot_timeline(
    events: pd.DataFrame, event_phases: pd.DataFrame, phase_product
) -> plt.Figure:
    """Show event ages against the same precession series used in the tests."""

    series = phase_product.series.loc[
        phase_product.series["age_ka"].between(ANALYSIS_START_KA, ANALYSIS_END_KA)
    ]
    fig, axes = plt.subplots(
        2, 1, figsize=(11.5, 5.1), sharex=True, height_ratios=(3.0, 1.15)
    )
    axes[0].plot(series["age_ka"], series["value"], color="#555555", lw=1.15)
    # The combined catalogue contains exactly these same points, so plotting it
    # again here would add no information and would obscure the two directions.
    for event_type, marker in zip(DIRECTION_TYPES, ("^", "v")):
        group = event_phases[event_phases["event_type"].eq(event_type)]
        axes[0].scatter(
            group["event_age_ka"],
            group["orbital_value_at_event"],
            s=31,
            marker=marker,
            color=EVENT_COLORS[event_type],
            edgecolor="white",
            linewidth=0.45,
            label=EVENT_LABELS[event_type],
            zorder=3,
        )
        rug = events[events["event_type"].eq(event_type)]
        y = 1.0 if event_type == "warming" else 0.0
        axes[1].scatter(
            rug["age_ka_bp"],
            np.full(len(rug), y),
            s=30,
            marker=marker,
            color=EVENT_COLORS[event_type],
        )

    axes[0].set_ylabel("Precession index")
    axes[0].legend(frameon=False, ncol=2, loc="upper center")
    axes[1].set(
        yticks=[0, 1],
        yticklabels=["GS / cooling", "GI / warming"],
        ylim=(-0.5, 1.5),
        xlabel="Age (Kyr BP)",
    )
    axes[0].set_title(
        "Rasmussen et al. (2014) main GI/GS starts and precession",
        loc="left",
    )
    axes[1].text(
        0.01,
        0.06,
        "Lettered subevents are collapsed to one parent-event onset",
        transform=axes[1].transAxes,
        fontsize=8.8,
    )
    for axis in axes:
        axis.grid(True, color="#e5e5e5", lw=0.65)
        axis.set_xlim(ANALYSIS_START_KA, ANALYSIS_END_KA)
    fig.tight_layout()
    return fig


def plot_predictive_phase_response(
    model_summary: pd.DataFrame, likelihood_tests: pd.DataFrame
) -> plt.Figure:
    """Plot the fitted precession term from each full conditional model."""

    phase_deg = np.linspace(0.0, 360.0, 721)
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.1), sharey=True)
    for panel, (axis, event_type) in enumerate(zip(axes, EVENT_TYPES)):
        full = model_summary.loc[
            model_summary["dataset_id"].eq(event_type)
            & model_summary["model_id"].eq(FULL_MODEL_ID)
        ].iloc[0]
        test = likelihood_tests.loc[likelihood_tests["dataset_id"].eq(event_type)].iloc[
            0
        ]
        response = phase_rate_multiplier(
            phase_deg,
            float(full["beta_pre_phase_sin"]),
            float(full["beta_pre_phase_cos"]),
        )
        preferred = float(full["pre_phase_preferred_deg"])
        supported = float(test["LR_p_value"]) < 0.05
        axis.plot(
            phase_deg,
            response,
            color=EVENT_COLORS[event_type],
            lw=2.0,
            ls="-" if supported else "--",
        )
        axis.axhline(1.0, color="#777777", lw=0.8, ls=":")
        axis.axvline(preferred, color="#333333", lw=1.0, ls=(0, (3, 2)))
        axis.text(
            0.04,
            0.96,
            (
                f"N = {int(test['n_events'])}\n"
                f"LR = {float(test['LR_statistic']):.2f}, "
                f"p = {float(test['LR_p_value']):.3g}\n"
                f"bits/event = {float(test['info_bits_per_event']):.3f}\n"
                f"Delta AICc = "
                f"{float(test['delta_AICc_full_minus_reduced']):+.2f}\n"
                f"peak = {preferred:.1f} degrees\n"
                f"max/min rate = "
                f"{float(full['pre_phase_rate_ratio_max_vs_min']):.2f}"
            ),
            transform=axis.transAxes,
            va="top",
            fontsize=8.8,
            bbox={"facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.9},
        )
        axis.set(
            xlim=(0, 360),
            xticks=[0, 90, 180, 270, 360],
            xlabel="Precession phase (degrees)",
            title=f"{chr(ord('a') + panel)}  {EVENT_LABELS[event_type]}",
        )
        axis.grid(True, color="#e5e5e5", lw=0.65)

    axes[0].set_ylabel("Multiplicative phase term in event rate")
    fig.suptitle(
        "Conditional PI phase response\n"
        "full model versus catalogue history + LR04 + CO2; "
        "0 degrees = precession minimum",
        y=1.04,
    )
    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG_DIR / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT_FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def build_parameters(events: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("analysis_start", ANALYSIS_START_KA, "Kyr BP", "fixed event-process support"),
        ("analysis_end", ANALYSIS_END_KA, "Kyr BP", "fixed event-process support"),
        ("bin_width", BIN_WIDTH_KA, "Kyr", "Poisson event-count bins"),
        (
            "history_window",
            HISTORY_WINDOW_KA,
            "Kyr",
            "older events within each analyzed catalogue",
        ),
        ("b2k_to_bp_offset", -0.05, "Kyr", "BP is relative to AD 1950"),
        (
            "warming_event_count",
            int(events["event_type"].eq("warming").sum()),
            "events",
            "GI starts",
        ),
        (
            "cooling_event_count",
            int(events["event_type"].eq("cooling").sum()),
            "events",
            "GS starts",
        ),
        (
            "combined_transition_count",
            len(events),
            "events",
            "GI and GS starts combined",
        ),
        (
            "reduced_model_terms",
            "+".join(REDUCED_TERMS),
            "",
            "conditional PI reduced model",
        ),
        ("full_model_terms", "+".join(FULL_TERMS), "", "conditional PI full model"),
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
            "processed Table 2 catalogue",
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
            "climate-state covariate",
        ),
        (
            "co2_source",
            str(CO2_XLSX.relative_to(PROJECT_ROOT)),
            "",
            "climate-state covariate",
        ),
    ]
    return pd.DataFrame(rows, columns=["parameter", "value", "unit", "note"])


def main() -> None:
    events = load_events()
    event_phases, rayleigh, phase_product = run_rayleigh(events)
    pi = run_predictive_information(events)
    summary = build_analysis_summary(
        events,
        rayleigh,
        pi["model_summary"],
        pi["likelihood_tests"],
        pi["fit_frame"],
    )

    OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "event_precession_phases.csv": event_phase_output(event_phases),
        "predictive_coefficients.csv": pi["coefficients"],
        "analysis_summary.csv": summary,
        "parameters_and_provenance.csv": build_parameters(events),
    }
    for filename, frame in outputs.items():
        frame.to_csv(OUT_DATA_DIR / filename, index=False)

    save_figure(
        plot_timeline(events, event_phases, phase_product),
        "fig01_event_timeline_and_precession",
    )
    polar = plot_rayleigh_polar(
        _phase_catalogues(event_phases),
        rayleigh,
        driver="pre",
        event_colors=EVENT_COLORS,
        annotate_mean_phase=True,
    )
    polar.set_size_inches(12.5, 4.8)
    polar.subplots_adjust(top=0.68, bottom=0.08, left=0.04, right=0.96, wspace=0.42)
    polar.suptitle(
        "NGRIP transitions by precession phase",
        y=0.99,
        fontsize=12.5,
    )
    polar.text(
        0.5,
        0.935,
        "0 degrees = precession minimum; 180 degrees = maximum",
        ha="center",
        va="top",
        fontsize=9.5,
    )
    save_figure(polar, "fig02_rayleigh_precession_phase")
    save_figure(
        plot_predictive_phase_response(pi["model_summary"], pi["likelihood_tests"]),
        "fig03_predictive_information_phase_response",
    )

    columns = [
        "event_type",
        "n_catalogue_events",
        "rayleigh_mean_phase_deg",
        "rayleigh_p",
        "n_predictive_events",
        "predictive_preferred_phase_deg",
        "predictive_LR_p",
        "predictive_bits_per_event",
    ]
    print(summary[columns].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print(f"\nData: {OUT_DATA_DIR}")
    print(f"Figures: {OUT_FIG_DIR}")


if __name__ == "__main__":
    main()
