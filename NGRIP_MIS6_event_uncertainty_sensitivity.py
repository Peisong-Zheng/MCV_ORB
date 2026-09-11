#!/usr/bin/env python3
"""Propagate joint event-age uncertainty through the pooled PI analysis.

The two source ensembles already combine chronology and transition-timing
uncertainty.  This script pairs their rows independently, without replacement,
and refits the pooled NGRIP-warming + MIS 6-transition model for each of 10,000
paired sequences.  Event membership and rank are fixed; crossed sequences are
rejected by the upstream uncertainty analyses rather than sorted here.

The reported fraction with nominal likelihood-ratio p < 0.05 is a robustness
proportion among realizations with valid observation support. Draws leaving
observation support remain in the output with an explicit invalid-fit status;
their ages are not clipped or resampled. It is not an empirical p value; the latter
is estimated separately by reduced-model parametric bootstrap.
"""

from __future__ import annotations

from pathlib import Path
from paper_figure_export import copy_pdf_to_paper
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from toolbox import combined_pi
from toolbox.project_config import (
    ORBITAL_AGE_OFFSET_TO_BP1950_KA,
    ORBITAL_REFERENCE,
    PROJECT_ROOT,
)


RUN_NAME = "NGRIP_MIS6_event_uncertainty_sensitivity"
OUT_DATA_DIR = PROJECT_ROOT / "data/processed" / RUN_NAME
OUT_FIG_DIR = PROJECT_ROOT / "figures" / RUN_NAME

NGRIP_MC_INPUT = (
    PROJECT_ROOT
    / "NGRIP/data/processed/ngrip_event_age_uncertainty"
    / "ngrip_event_age_realizations.csv"
)
MIS6_MC_INPUT = (
    PROJECT_ROOT
    / "MIS6/data/processed/MIS6_event_age_uncertainty"
    / "mis6_event_age_realizations.csv"
)

COMBINED_AGE_OUTPUT = OUT_DATA_DIR / "combined_event_age_realizations.csv"
PI_OUTPUT = OUT_DATA_DIR / "pi_realizations.csv"
SUMMARY_OUTPUT = OUT_DATA_DIR / "summary.csv"
PARAMETER_OUTPUT = OUT_DATA_DIR / "parameters_and_provenance.csv"
FIGURE_STEM = RUN_NAME

N_REALIZATIONS = 10_000
PAIRING_SEED = 20260906
P_THRESHOLD = 0.05
PNG_DPI = 600

HISTORY_WINDOW_KYR = combined_pi.DEFAULT_HISTORY_WINDOW_KA
BIN_WIDTH_KYR = combined_pi.DEFAULT_BIN_WIDTH_KA
BIN_ORIGIN_FRACTION = combined_pi.DEFAULT_ORIGIN_FRACTION
RESPONSE_MODE = combined_pi.DEFAULT_RESPONSE_MODE
RESOLUTION_COVARIATE_INCLUDED = False


PI_NUMERIC_COLUMNS = (
    "LR_statistic", "nominal_LR_p", "info_bits_per_event",
    "delta_AICc_full_minus_reduced", "pre_phase_preferred_deg",
    "pre_phase_rate_ratio_max_vs_min",
)
FIT_FLAG_COLUMNS = ("all_models_converged", "likelihood_nesting_ok", "eta_clipping_used")


class OutsideObservationSupport(ValueError):
    """An ordered age draw extends beyond documented observation coverage."""


def configure_plot_style() -> None:
    """Use compact, readable typography for a two-column GRL figure."""

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def combined_age_columns(events: pd.DataFrame) -> list[str]:
    """Name one wide-table age column per stable curated event ID."""

    return [f"age_kyr_bp__{event_id}" for event_id in events["event_id"]]


def source_age_columns(events: pd.DataFrame) -> dict[str, list[str]]:
    """Map curated event IDs to their columns in the two source ensembles."""

    ngrip = events.loc[events["segment_id"].eq("NGRIP")]
    mis6 = events.loc[events["segment_id"].eq("MIS6")]

    # NGRIP uncertainty columns use the simple GI label, not Table 2 suffixes
    # such as GI-1e that are retained only in source_event_label.
    ngrip_columns = [f"age_ka_bp__{label}" for label in ngrip["event_label"]]
    mis6_columns = [
        f"{event_id.removeprefix('MIS6:')}_age_ka_bp"
        for event_id in mis6["event_id"]
    ]
    if len(ngrip_columns) != 34 or len(mis6_columns) != 21:
        raise ValueError("Expected 34 NGRIP and 21 MIS 6 uncertainty columns")
    return {"NGRIP": ngrip_columns, "MIS6": mis6_columns}


def _validate_source_ensemble(
    table: pd.DataFrame,
    columns: list[str],
    source: str,
) -> np.ndarray:
    required = {"realization_id", *columns}
    if missing := required.difference(table.columns):
        raise ValueError(f"{source} age ensemble is missing columns: {sorted(missing)}")
    if (
        table["realization_id"].isna().any()
        or table["realization_id"].duplicated().any()
    ):
        raise ValueError(f"{source} realization IDs must be complete and unique")

    ages = table.loc[:, columns].to_numpy(float)
    if not np.isfinite(ages).all():
        raise ValueError(f"{source} age ensemble contains non-finite values")
    if not np.all(np.diff(ages, axis=1) > 0.0):
        raise ValueError(
            f"{source} realizations must preserve event rank; ages are never sorted"
        )
    return ages


def pair_source_ensembles(
    events: pd.DataFrame,
    ngrip_table: pd.DataFrame,
    mis6_table: pd.DataFrame,
    *,
    n_realizations: int,
    seed: int,
) -> pd.DataFrame:
    """Pair independent source rows and return 55 ordered ages per realization."""

    if n_realizations < 1:
        raise ValueError("n_realizations must be positive")
    if n_realizations > min(len(ngrip_table), len(mis6_table)):
        raise ValueError("Requested more pairs than available source realizations")

    columns = source_age_columns(events)
    ngrip_ages = _validate_source_ensemble(ngrip_table, columns["NGRIP"], "NGRIP")
    mis6_ages = _validate_source_ensemble(mis6_table, columns["MIS6"], "MIS 6")

    # Independent permutations avoid imposing row-wise correspondence between
    # separately generated chronology ensembles.  With 10,000 source rows,
    # every row enters the joint experiment exactly once.
    rng = np.random.default_rng(seed)
    ngrip_rows = rng.permutation(len(ngrip_table))[:n_realizations]
    mis6_rows = rng.permutation(len(mis6_table))[:n_realizations]
    paired_ngrip = ngrip_ages[ngrip_rows]
    paired_mis6 = mis6_ages[mis6_rows]

    if not np.all(paired_ngrip[:, -1] < paired_mis6[:, 0]):
        raise ValueError("At least one paired realization interleaves the two segments")

    paired_ages = np.column_stack((paired_ngrip, paired_mis6))
    if not np.all(np.diff(paired_ages, axis=1) > 0.0):
        raise RuntimeError("Joint event rank changed during source pairing")

    draws = pd.DataFrame(paired_ages, columns=combined_age_columns(events))
    draws.insert(
        0,
        "mis6_realization_id",
        mis6_table["realization_id"].to_numpy()[mis6_rows],
    )
    draws.insert(
        0,
        "ngrip_realization_id",
        ngrip_table["realization_id"].to_numpy()[ngrip_rows],
    )
    draws.insert(
        0,
        "realization_id",
        [f"joint_{index:05d}" for index in range(1, n_realizations + 1)],
    )
    return draws


def load_joint_realizations(
    events: pd.DataFrame,
    *,
    n_realizations: int = N_REALIZATIONS,
    seed: int = PAIRING_SEED,
    ngrip_path: Path = NGRIP_MC_INPUT,
    mis6_path: Path = MIS6_MC_INPUT,
) -> pd.DataFrame:
    """Load the existing combined-error ensembles and form fixed random pairs."""

    if not ngrip_path.exists():
        raise FileNotFoundError(f"Missing NGRIP age ensemble: {ngrip_path}")
    if not mis6_path.exists():
        raise FileNotFoundError(f"Missing MIS 6 age ensemble: {mis6_path}")
    return pair_source_ensembles(
        events,
        pd.read_csv(ngrip_path),
        pd.read_csv(mis6_path),
        n_realizations=n_realizations,
        seed=seed,
    )


def _counts_from_ages(
    ages: np.ndarray,
    events: pd.DataFrame,
    context: combined_pi.PIContext,
) -> dict[str, np.ndarray]:
    """Bin one ordered 55-event sequence on the two independent grids."""

    values = np.asarray(ages, dtype=float)
    if values.shape != (len(events),) or not np.isfinite(values).all():
        raise ValueError("Each joint realization must contain 55 finite event ages")
    if not np.all(np.diff(values) > 0.0):
        raise ValueError("Joint event ages must retain their fixed published rank")

    counts: dict[str, np.ndarray] = {}
    for segment_id in combined_pi.SEGMENT_IDS:
        selected = events["segment_id"].eq(segment_id).to_numpy()
        segment = context.segments[segment_id]
        local_ages = values[selected]
        if np.any(local_ages < segment.bin_edges[0]) or np.any(
            local_ages > segment.bin_edges[-1]
        ):
            outside = (local_ages < segment.bin_edges[0]) | (local_ages > segment.bin_edges[-1])
            event_ids = events.loc[selected, "event_id"].to_numpy()[outside]
            detail = ", ".join(f"{event_id}={age:g}" for event_id, age in zip(event_ids, local_ages[outside]))
            raise OutsideObservationSupport(
                f"{segment_id} observation support [{segment.bin_edges[0]:g}, "
                f"{segment.bin_edges[-1]:g}] kyr BP: {detail}"
            )
        local_counts, _ = np.histogram(local_ages, bins=segment.bin_edges)
        if int(local_counts.sum()) != len(local_ages):
            raise RuntimeError(f"Failed to bin every {segment_id} event")
        counts[segment_id] = local_counts.astype(float)
    return counts


def _count_pattern_key(counts: dict[str, np.ndarray]) -> tuple[bytes, ...]:
    """Create a compact cache key without joining the two observation grids."""

    return tuple(
        np.asarray(counts[segment_id], dtype=np.int8).tobytes()
        for segment_id in combined_pi.SEGMENT_IDS
    )


def _validate_pi_results(results: pd.DataFrame, expected_events: int) -> None:
    if not results["fit_valid"].isin([True, False]).all():
        raise RuntimeError("Every realization needs an explicit fit-valid status")
    valid = results.loc[results["fit_valid"].eq(True)]
    invalid = results.loc[results["fit_valid"].eq(False)]
    if valid["invalid_reason"].fillna("").ne("").any():
        raise RuntimeError("Valid fits cannot have an invalid reason")
    if invalid["invalid_reason"].fillna("").astype(str).str.strip().eq("").any():
        raise RuntimeError("Invalid fits must retain an explicit reason")
    if invalid.loc[:, list(PI_NUMERIC_COLUMNS)].notna().any().any():
        raise RuntimeError("Invalid fits must not contain PI estimates")
    if not np.isfinite(valid.loc[:, list(PI_NUMERIC_COLUMNS)].to_numpy(float)).all():
        raise RuntimeError("Valid PI realizations contain non-finite results")
    event_counts = valid["n_predictive_events"].to_numpy(float)
    if not (np.isfinite(event_counts).all()
            and np.all((event_counts >= 0) & (event_counts <= expected_events))
            and np.all(event_counts == np.floor(event_counts))):
        raise RuntimeError("Response event counts must be integers within source membership")
    if not valid["nominal_LR_p"].between(0.0, 1.0).all():
        raise RuntimeError("Nominal p values must lie between zero and one")
    if not valid["pre_phase_preferred_deg"].between(0.0, 360.0).all():
        raise RuntimeError("Preferred phases must lie between 0 and 360 degrees")
    if not valid["pre_phase_rate_ratio_max_vs_min"].ge(1.0).all():
        raise RuntimeError("Phase maximum-to-minimum rate ratios cannot be below one")
    if not valid["all_models_converged"].eq(True).all():
        raise RuntimeError("At least one Poisson fit did not converge")
    if not valid["likelihood_nesting_ok"].eq(True).all():
        raise RuntimeError("At least one full-model likelihood is below the reduced fit")
    if not valid["eta_clipping_used"].eq(False).all():
        raise RuntimeError("At least one fit reached a numerical eta clip bound")


def fit_realizations(
    events: pd.DataFrame,
    draws: pd.DataFrame,
    context: combined_pi.PIContext,
    *,
    show_progress: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Refit PI for each sequence, caching repeated 0.2-kyr count patterns."""

    age_columns = combined_age_columns(events)
    required = {
        "realization_id",
        "ngrip_realization_id",
        "mis6_realization_id",
        *age_columns,
    }
    if missing := required.difference(draws.columns):
        raise ValueError(f"Joint age table is missing columns: {sorted(missing)}")
    if len(draws) == 0:
        raise ValueError("At least one joint age realization is required")
    if draws["realization_id"].duplicated().any():
        raise ValueError("Joint realization IDs must be unique")

    identifiers = draws[
        ["realization_id", "ngrip_realization_id", "mis6_realization_id"]
    ].reset_index(drop=True)
    age_matrix = draws.loc[:, age_columns].to_numpy(float)
    cache: dict[tuple[bytes, ...], tuple[int, dict[str, object]]] = {}
    rows: list[dict[str, object]] = []
    started = time.perf_counter()

    for index, ages in enumerate(age_matrix):
        row = identifiers.iloc[index].to_dict()
        try:
            counts = _counts_from_ages(ages, events, context)
        except OutsideObservationSupport as error:
            # This is the only expected non-fit outcome. Other input or
            # optimizer errors propagate rather than becoming missing draws.
            row.update({column: np.nan for column in PI_NUMERIC_COLUMNS})
            row.update({column: pd.NA for column in FIT_FLAG_COLUMNS})
            row.update(
                fit_valid=False, invalid_reason=str(error),
                event_count_pattern_id=np.nan, max_events_in_single_bin=np.nan,
                n_source_events=len(events), n_predictive_events=np.nan,
                n_predictive_bins=len(context.response_dt),
                response_exposure_kyr=context.response_exposure_kyr,
            )
        else:
            key = _count_pattern_key(counts)
            if key not in cache:
                fit = combined_pi.fit_event_counts(counts, context)
                cache[key] = (len(cache) + 1, dict(fit.summary))
            pattern_id, fit_summary = cache[key]
            row.update(fit_summary)
            row.update(
                fit_valid=True, invalid_reason="", event_count_pattern_id=pattern_id,
                max_events_in_single_bin=int(max(np.max(values) for values in counts.values())),
            )
        rows.append(row)
        if show_progress and (index + 1) % 1_000 == 0:
            print(
                f"Fitted {index + 1:,}/{len(age_matrix):,} realizations "
                f"({len(cache):,} unique count patterns)"
            )

    results = pd.DataFrame(rows)
    _validate_pi_results(results, expected_events=len(events))
    valid = results.loc[results["fit_valid"]]
    n_valid = len(valid)
    diagnostics = {
        "n_realizations": len(results),
        "n_valid": n_valid,
        "n_invalid": len(results) - n_valid,
        "n_realizations_with_response_event_loss": int(valid["n_predictive_events"].lt(len(events)).sum()),
        "n_unique_event_count_patterns": len(cache),
        "n_cached_reuses": n_valid - len(cache),
        "cache_reuse_fraction": (1.0 - len(cache) / n_valid) if n_valid else np.nan,
        "elapsed_seconds": time.perf_counter() - started,
        "max_events_in_single_bin": int(valid["max_events_in_single_bin"].max()) if n_valid else np.nan,
        "n_nonconverged": int(valid["all_models_converged"].eq(False).sum()),
        "n_nesting_failures": int(valid["likelihood_nesting_ok"].eq(False).sum()),
        "n_eta_clipping": int(valid["eta_clipping_used"].eq(True).sum()),
    }
    return results, diagnostics


def compact_pi_results(results: pd.DataFrame) -> pd.DataFrame:
    """Keep only realization-level quantities needed to audit PI sensitivity."""

    columns = [
        "realization_id",
        "ngrip_realization_id",
        "mis6_realization_id",
        "fit_valid",
        "invalid_reason",
        "n_predictive_events",
        "LR_statistic",
        "nominal_LR_p",
        "info_bits_per_event",
        "delta_AICc_full_minus_reduced",
        "pre_phase_preferred_deg",
        "pre_phase_rate_ratio_max_vs_min",
        "all_models_converged",
        "likelihood_nesting_ok",
        "eta_clipping_used",
    ]
    return results.loc[:, columns].copy()


def unwrap_around(values_deg: np.ndarray, center_deg: float) -> np.ndarray:
    """Place circular phase estimates on the branch around the point estimate."""

    values = np.asarray(values_deg, dtype=float)
    return center_deg + (values - center_deg + 180.0) % 360.0 - 180.0


def build_summary(
    results: pd.DataFrame,
    point_fit: combined_pi.CombinedPIFit,
) -> pd.DataFrame:
    """Summarize the joint MC distributions and exact robustness count."""

    n_realizations = len(results)
    results = results.loc[results["fit_valid"].eq(True)]
    n_valid = len(results)
    point = point_fit.summary
    phase = unwrap_around(
        results["pre_phase_preferred_deg"].to_numpy(float),
        float(point["pre_phase_preferred_deg"]),
    )
    n_below = int(results["nominal_LR_p"].lt(P_THRESHOLD).sum())
    n_aicc = int(results["delta_AICc_full_minus_reduced"].lt(0.0).sum())
    row: dict[str, object] = {
        "catalogue_id": combined_pi.CATALOGUE_ID,
        "catalogue_label": "NGRIP warming + MIS 6 transitions",
        "n_events": int(point["n_predictive_events"]),
        "n_ngrip_events": 34,
        "n_mis6_events": 21,
        "n_realizations": n_realizations,
        "n_valid": n_valid,
        "n_invalid": n_realizations - n_valid,
        "robustness_denominator": n_valid,
        "robustness_denominator_definition": "valid fits within observation support",
        "n_realizations_with_response_event_loss": int(results["n_predictive_events"].lt(point["n_predictive_events"]).sum()),
        "n_predictive_events_min": results["n_predictive_events"].min(),
        "n_predictive_events_max": results["n_predictive_events"].max(),
        "point_LR_statistic": float(point["LR_statistic"]),
        "point_nominal_LR_p": float(point["nominal_LR_p"]),
        "point_info_bits_per_event": float(point["info_bits_per_event"]),
        "point_delta_AICc_full_minus_reduced": float(
            point["delta_AICc_full_minus_reduced"]
        ),
        "point_pre_phase_preferred_deg": float(point["pre_phase_preferred_deg"]),
        "point_pre_phase_rate_ratio_max_vs_min": float(
            point["pre_phase_rate_ratio_max_vs_min"]
        ),
        "n_nominal_p_below_0p05": n_below,
        "fraction_nominal_p_below_0p05": n_below / n_valid if n_valid else np.nan,
        "n_delta_AICc_below_zero": n_aicc,
        "fraction_delta_AICc_below_zero": n_aicc / n_valid if n_valid else np.nan,
        "fraction_nominal_p_below_0p05_all_draws_lower_bound": n_below / n_realizations if n_realizations else np.nan,
        "fraction_delta_AICc_below_zero_all_draws_lower_bound": n_aicc / n_realizations if n_realizations else np.nan,
        "significance_fraction_is_empirical_p": False,
    }
    for column in (
        "LR_statistic",
        "nominal_LR_p",
        "info_bits_per_event",
        "delta_AICc_full_minus_reduced",
        "pre_phase_rate_ratio_max_vs_min",
    ):
        q025, median, q975 = np.quantile(results[column], [0.025, 0.5, 0.975]) if n_valid else (np.nan,) * 3
        row[f"{column}_q025"] = float(q025)
        row[f"{column}_median"] = float(median)
        row[f"{column}_q975"] = float(q975)

    q025, median, q975 = np.quantile(phase, [0.025, 0.5, 0.975]) if n_valid else (np.nan,) * 3
    row["pre_phase_preferred_deg_q025"] = float(q025)
    row["pre_phase_preferred_deg_median"] = float(median)
    row["pre_phase_preferred_deg_q975"] = float(q975)
    row["phase_quantiles_unwrapped_about_point"] = True
    return pd.DataFrame([row])


def build_parameters(
    events: pd.DataFrame,
    context: combined_pi.PIContext,
    diagnostics: dict[str, object],
) -> pd.DataFrame:
    """Record model settings, uncertainty provenance, and run diagnostics."""

    response_ranges = {
        segment_id: (
            segment.response_start_kyr_bp,
            segment.response_end_kyr_bp,
        )
        for segment_id, segment in context.segments.items()
    }
    rows = [
        ("analysis", "joint event-age uncertainty sensitivity", "", "conditional PI only"),
        (
            "catalogue",
            combined_pi.CATALOGUE_ID,
            "",
            "34 NGRIP warming starts + 21 MIS 6 transitions",
        ),
        ("age_unit", "kyr BP", "BP1950", "all event and driver ages"),
        (
            "n_realizations",
            N_REALIZATIONS,
            "paired sequences",
            "one combined-error ensemble",
        ),
        (
            "pairing_seed",
            PAIRING_SEED,
            "",
            "reproducible independent source-row permutations",
        ),
        (
            "pairing_scheme",
            "independent permutations without replacement",
            "",
            "every source row is used once in the 10,000-pair run",
        ),
        ("event_membership", "fixed", "", "published events only; no event additions or removals"),
        (
            "event_order",
            "fixed; no sorting repair",
            "",
            "upstream crossed proposals were rejected",
        ),
        (
            "ngrip_mc_input",
            str(NGRIP_MC_INPUT.relative_to(PROJECT_ROOT)),
            "",
            "combined chronology + published-boundary timing uncertainty",
        ),
        (
            "mis6_mc_input",
            str(MIS6_MC_INPUT.relative_to(PROJECT_ROOT)),
            "",
            "combined chronology + algorithmic transition-timing uncertainty",
        ),
        ("inter_record_synchronization_error", False, "", "no extra term is added"),
        ("history_window", HISTORY_WINDOW_KYR, "kyr", "short-term event dependence"),
        ("bin_width", BIN_WIDTH_KYR, "kyr", "fixed main setting"),
        (
            "bin_origin_fraction",
            BIN_ORIGIN_FRACTION,
            "fraction of bin width",
            "fixed main setting",
        ),
        (
            "response_mode",
            RESPONSE_MODE,
            "",
            "maximal support with complete event history",
        ),
        (
            "ngrip_response",
            f"{response_ranges['NGRIP'][0]:g}--{response_ranges['NGRIP'][1]:g}",
            "kyr BP",
            "disjoint exposure segment",
        ),
        (
            "mis6_response",
            f"{response_ranges['MIS6'][0]:g}--{response_ranges['MIS6'][1]:g}",
            "kyr BP",
            "disjoint exposure segment",
        ),
        ("response_exposure", context.response_exposure_kyr, "kyr", "gap excluded"),
        (
            "reduced_model_terms",
            "+".join(combined_pi.REDUCED_TERMS),
            "",
            "includes a MIS 6 segment intercept",
        ),
        (
            "full_model_terms",
            "+".join(combined_pi.FULL_TERMS),
            "",
            "adds sine and cosine of precession phase",
        ),
        ("resolution_covariate_included", RESOLUTION_COVARIATE_INCLUDED, "", "not included in PI"),
        (
            "forcing_age_uncertainty_propagated",
            False,
            "",
            "LR04, CO2, and La2004 remain fixed",
        ),
        (
            "nominal_p_method",
            "asymptotic chi-square likelihood-ratio test",
            "",
            "two added phase terms",
        ),
        (
            "robustness_fraction",
            "count(valid nominal p < 0.05) / n_valid",
            "",
            "conditional on observation support; not an empirical p value",
        ),
        (
            "outside_observation_support",
            "retain draw and identifiers; fit_valid=False; PI estimates missing",
            "",
            "no clipping, source-ensemble redraw, or replacement",
        ),
        (
            "outside_response_inside_observation",
            "fit with actual response event count; retain event for history",
            "",
            "observation support and response exposure are distinct",
        ),
        (
            "orbital_reference",
            ORBITAL_REFERENCE,
            "",
            f"J2000 to BP1950 offset {ORBITAL_AGE_OFFSET_TO_BP1950_KA} kyr",
        ),
    ]
    for name, value in diagnostics.items():
        rows.append((f"diagnostic_{name}", value, "", "run diagnostic"))
    return pd.DataFrame(rows, columns=["parameter", "value", "unit", "note"])


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
    ax.text(
        -0.15,
        1.04,
        panel_label,
        transform=ax.transAxes,
        fontweight="bold",
        va="bottom",
    )


def plot_sensitivity(
    results: pd.DataFrame,
    point_fit: combined_pi.CombinedPIFit,
) -> tuple[Path, Path]:
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
        results["info_bits_per_event"].to_numpy(float),
        float(point["info_bits_per_event"]),
        "PI (bits event$^{-1}$)",
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
        results["delta_AICc_full_minus_reduced"].to_numpy(float),
        float(point["delta_AICc_full_minus_reduced"]),
        r"$\Delta$AICc (full $-$ reduced)",
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

    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_FIG_DIR / f"{FIGURE_STEM}.png"
    pdf = OUT_FIG_DIR / f"{FIGURE_STEM}.pdf"
    fig.savefig(png, dpi=PNG_DPI)
    fig.savefig(pdf)
    copy_pdf_to_paper(pdf)
    plt.close(fig)
    return png, pdf


def write_outputs(
    draws: pd.DataFrame,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    parameters: pd.DataFrame,
) -> None:
    """Save the frozen joint ages and compact uncertainty-analysis products."""

    OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    draws.to_csv(COMBINED_AGE_OUTPUT, index=False, float_format="%.6f")
    compact_pi_results(results).to_csv(PI_OUTPUT, index=False, float_format="%.9g")
    summary.to_csv(SUMMARY_OUTPUT, index=False, float_format="%.9g")
    parameters.to_csv(PARAMETER_OUTPUT, index=False)


def main() -> None:
    events = combined_pi.load_event_catalogue()
    context = combined_pi.build_context(
        history_window_ka=HISTORY_WINDOW_KYR,
        bin_width_ka=BIN_WIDTH_KYR,
        origin_fraction=BIN_ORIGIN_FRACTION,
        response_mode=RESPONSE_MODE,
    )
    point_fit = combined_pi.fit_catalogue(events, context)
    draws = load_joint_realizations(events)
    results, diagnostics = fit_realizations(
        events, draws, context, show_progress=True
    )
    summary = build_summary(results, point_fit)
    parameters = build_parameters(events, context, diagnostics)
    write_outputs(draws, results, summary, parameters)
    png, pdf = plot_sensitivity(results, point_fit)

    row = summary.iloc[0]
    print(
        f"Point PI={row['point_info_bits_per_event']:.4f} bits/event, "
        f"nominal p={row['point_nominal_LR_p']:.4g}; "
        f"MC nominal p<0.05 in {int(row['n_nominal_p_below_0p05']):,}/"
        f"{int(row['n_valid']):,} valid fits "
        f"({row['fraction_nominal_p_below_0p05']:.2%})."
    )
    print(f"Retained {int(row['n_realizations']):,} paired draws; "
          f"{int(row['n_invalid']):,} outside observation support have no PI fit.")
    print(
        "This fraction is an age-uncertainty robustness proportion, "
        "not an empirical p value."
    )
    print(f"Wrote tables to {OUT_DATA_DIR.relative_to(PROJECT_ROOT)}")
    print(
        f"Wrote figures to {png.relative_to(PROJECT_ROOT)} and "
        f"{pdf.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
