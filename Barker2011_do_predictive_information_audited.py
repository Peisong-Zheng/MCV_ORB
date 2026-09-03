"""Audit Barker et al. (2011) predicted D-O warming-event phase dependence.

Barker et al. inferred abrupt Greenland warming candidates from a synthetic
Greenland-temperature record reconstructed from Antarctic ice. These are not
independently observed Greenland events. This script asks two questions:

1. Are event phases non-uniform by a Rayleigh test?
2. Does precession phase improve a binned Poisson event-rate model after event
   history, EDC sampling resolution, LR04, and CO2 have been included?

The variable-threshold catalogue over EDC3 0--640 ka is the primary analysis.
EDC3 0--800 ka and SpeleoAge 0--400 ka are sensitivity variants of the same
Table S3 events, not independent replications. SpeleoAge is tuned partly to
Chinese speleothems, so its orbital-phase result has an additional circularity
risk. Likelihood-ratio p values are asymptotic and age uncertainty is not
propagated; the script therefore calls them nominal p values.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from toolbox import event_inputs, event_process as predictive, orbital_phase, poisson
from toolbox.data_checks import require_unique_values
from toolbox.model_stats import nested_likelihood_metrics
from toolbox.project_config import (
    BIN_WIDTH_KA,
    CO2_XLSX,
    LR04_XLSX,
    ORBITAL_DRIVER_SETTINGS,
    PRE_TXT,
    PROJECT_ROOT,
)


# ---------------------------------------------------------------------------
# Settings: keep the scientific choices together and easy to inspect.
# ---------------------------------------------------------------------------

RUN_NAME = "Barker2011_do_predictive_information_audited"
OUT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / RUN_NAME
OUT_FIG_DIR = PROJECT_ROOT / "figures" / RUN_NAME

BARKER_XLS = PROJECT_ROOT / "data/raw/Barker et al-2011-SOM.xls"
JOUZEL_TXT = (
    PROJECT_ROOT
    / "data/raw/Jouzel-etal-2007-Science-Orbital and Millennial Antarctic Climate Variability over the Past 800,000 Years.txt"
)

VARIABLE_PICK = "DO pick variable threshold"
FIXED_PICK = "DO pick"
MAIN_HISTORY_WINDOW_KA = 5.0
HISTORY_SENSITIVITY_KA = (2.0, 5.0, 10.0, 20.0)

HISTORY_TERM = predictive.HISTORY_TERM
RESOLUTION_TERM = "edc_log_resolution_scaled"
BASELINE_TERMS = (HISTORY_TERM, RESOLUTION_TERM)
CLIMATE_TERMS = predictive.CLIMATE_TERMS
PHASE_TERMS = predictive.PHASE_TERMS

# Only models needed for the two nested scientific questions are fitted.
MODEL_SPECS = (
    ("baseline", BASELINE_TERMS, "Event-process baseline"),
    ("climate", BASELINE_TERMS + CLIMATE_TERMS, "Climate-state model"),
    ("full", BASELINE_TERMS + CLIMATE_TERMS + PHASE_TERMS, "Full model"),
)
LR_TEST_SPECS = (
    (
        "climate_after_baseline",
        "baseline",
        "climate",
        "Do LR04 and CO2 improve after event history and EDC resolution?",
    ),
    (
        "phase_after_climate",
        "climate",
        "full",
        "Does precession phase add information after the climate-state model?",
    ),
)


@dataclass(frozen=True)
class CatalogueSpec:
    """One age/support choice for the same Barker Table S3 event candidates."""

    dataset_id: str
    event_type: str
    label: str
    short_label: str
    age_column: str
    end_ka: float
    resolution_age_axis: str
    color: str


CATALOGUE_SPECS = (
    CatalogueSpec(
        dataset_id="barker_variable_threshold_edc3_0_640",
        event_type="barker_do_warming_variable_threshold_edc3",
        label="Barker variable-threshold D-O warmings, EDC3 0-640 ka",
        short_label="EDC3 0-640 kyr",
        age_column="Age kyr (EDC3)",
        end_ka=640.0,
        resolution_age_axis="edc3",
        color="#d95f02",
    ),
    CatalogueSpec(
        dataset_id="barker_variable_threshold_edc3_0_800",
        event_type="barker_do_warming_variable_threshold_edc3_0_800",
        label="Barker variable-threshold D-O warmings, EDC3 0-800 ka",
        short_label="EDC3 0-800 kyr",
        age_column="Age kyr (EDC3)",
        end_ka=800.0,
        resolution_age_axis="edc3",
        color="#117733",
    ),
    CatalogueSpec(
        dataset_id="barker_variable_threshold_speleo_0_400",
        event_type="barker_do_warming_variable_threshold_speleo",
        label="Barker variable-threshold D-O warmings, SpeleoAge 0-400 ka",
        short_label="SpeleoAge 0-400 kyr",
        age_column="SpeloAge (kyr).1",
        end_ka=400.0,
        resolution_age_axis="speleo_to_edc3",
        color="#cc6677",
    ),
)


@dataclass
class BarkerSource:
    """The event table and SpeleoAge-to-EDC3 mapping read from one workbook."""

    table: pd.DataFrame
    mapping: pd.DataFrame


@dataclass
class PredictiveTables:
    """Tables produced by the Poisson predictive-information analysis."""

    binned_inputs: pd.DataFrame
    model_summary: pd.DataFrame
    likelihood_tests: pd.DataFrame
    coefficients: pd.DataFrame
    resolution_metadata: pd.DataFrame


@dataclass
class AnalysisTables:
    """All core outputs for one Barker event-definition choice."""

    events: pd.DataFrame
    event_phases: pd.DataFrame
    rayleigh_results: pd.DataFrame
    binned_inputs: pd.DataFrame
    model_summary: pd.DataFrame
    likelihood_tests: pd.DataFrame
    coefficients: pd.DataFrame
    resolution_metadata: pd.DataFrame
    analysis_summary: pd.DataFrame


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "axes.linewidth": 0.9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------


def _definition_name(pick_column: str) -> str:
    """Return the short identifier used in output IDs."""

    if pick_column == VARIABLE_PICK:
        return "variable_threshold"
    if pick_column == FIXED_PICK:
        return "fixed_threshold"
    raise ValueError(f"Unknown Barker event-pick column: {pick_column}")


def _catalogue_identity(spec: CatalogueSpec, pick_column: str) -> tuple[str, str, str]:
    """Return dataset ID, event type, and label for an event definition."""

    if pick_column == VARIABLE_PICK:
        return spec.dataset_id, spec.event_type, spec.label
    return (
        spec.dataset_id.replace("variable_threshold", "fixed_threshold"),
        spec.event_type.replace("variable_threshold", "fixed_threshold"),
        spec.label.replace("variable-threshold", "fixed-threshold"),
    )


def load_barker_source() -> BarkerSource:
    """Read Barker Table S3 and its age mapping with one Excel-file access."""

    raw = pd.read_excel(BARKER_XLS, sheet_name="Sheet1", header=8)
    event_columns = [
        "Age kyr (EDC3)",
        "SpeloAge (kyr).1",
        FIXED_PICK,
        VARIABLE_PICK,
    ]
    mapping_columns = ["EDC3 Age (kyr)", "SpeloAge (kyr)"]
    missing = [
        column for column in event_columns + mapping_columns if column not in raw
    ]
    if missing:
        raise ValueError(f"Missing expected Barker workbook columns: {missing}")

    table = raw[event_columns].copy()
    table.insert(0, "source_row", raw.index.to_numpy(dtype=int))
    table[event_columns] = table[event_columns].apply(pd.to_numeric, errors="coerce")
    table = table.dropna(subset=["Age kyr (EDC3)"]).reset_index(drop=True)

    mapping = raw[mapping_columns].copy()
    mapping.columns = ["edc3_age_ka", "speleo_age_ka"]
    mapping = mapping.apply(pd.to_numeric, errors="coerce").dropna()
    mapping = mapping.sort_values("speleo_age_ka").reset_index(drop=True)
    require_unique_values(mapping, "speleo_age_ka", context="Barker SpeleoAge mapping")
    require_unique_values(mapping, "edc3_age_ka", context="Barker EDC3 mapping")
    return BarkerSource(table=table, mapping=mapping)


def load_jouzel_edc_resolution_source() -> pd.DataFrame:
    """Load the EDC record and estimate local EDC3 age spacing per sample."""

    lines = JOUZEL_TXT.read_text(encoding="latin1").splitlines()
    start = next(
        (
            index + 1
            for index, line in enumerate(lines)
            if line.strip().startswith("Bag") and "ztop" in line and "Age" in line
        ),
        None,
    )
    if start is None:
        raise ValueError(f"Cannot locate the data table in {JOUZEL_TXT}.")

    raw = pd.read_csv(
        JOUZEL_TXT,
        sep=r"\s+",
        skiprows=start,
        header=None,
        names=["bag", "ztop_m", "age_yr_bp", "deuterium", "temperature"],
        encoding="latin1",
    )
    frame = pd.DataFrame(
        {
            "edc3_age_ka": pd.to_numeric(raw["age_yr_bp"], errors="coerce") / 1000.0,
            "deuterium": pd.to_numeric(raw["deuterium"], errors="coerce"),
        }
    ).dropna(subset=["edc3_age_ka"])
    frame = frame.sort_values("edc3_age_ka").reset_index(drop=True)
    require_unique_values(frame, "edc3_age_ka", context="Jouzel EDC3 record")

    age = frame["edc3_age_ka"].to_numpy(dtype=float)
    previous_gap = np.r_[np.nan, np.diff(age)]
    next_gap = np.r_[np.diff(age), np.nan]
    spacing = np.nanmedian(np.vstack([previous_gap, next_gap]), axis=0)
    fallback = float(np.nanmedian(np.diff(age)))
    spacing = np.where(np.isfinite(spacing) & (spacing > 0.0), spacing, fallback)
    frame["edc_local_resolution_ka"] = spacing
    return frame


def build_event_catalogues(
    table: pd.DataFrame | None = None,
    *,
    pick_column: str = VARIABLE_PICK,
    specs: tuple[CatalogueSpec, ...] = CATALOGUE_SPECS,
) -> tuple[pd.DataFrame, list[event_inputs.EventDataset]]:
    """Build the three overlapping Barker catalogue variants."""

    if table is None:
        table = load_barker_source().table
    definition = _definition_name(pick_column)
    event_frames: list[pd.DataFrame] = []
    datasets: list[event_inputs.EventDataset] = []

    for spec in specs:
        dataset_id, event_type, label = _catalogue_identity(spec, pick_column)
        frame = pd.DataFrame(
            {
                "source_row": table["source_row"],
                "event_age_ka": pd.to_numeric(table[spec.age_column], errors="coerce"),
                "pick_value": pd.to_numeric(table[pick_column], errors="coerce"),
            }
        ).dropna()
        frame = frame[frame["pick_value"].eq(1.0)]
        frame = frame[frame["event_age_ka"].between(0.0, spec.end_ka, inclusive="both")]
        frame = frame.sort_values("event_age_ka").reset_index(drop=True)
        frame["event_index"] = np.arange(1, len(frame) + 1)
        frame["dataset_id"] = dataset_id
        frame["event_type"] = event_type
        frame["event_label"] = label
        frame["event_definition"] = definition
        frame["age_column"] = spec.age_column
        frame["pick_column"] = pick_column
        frame["analysis_start_ka"] = 0.0
        frame["analysis_end_ka"] = spec.end_ka
        frame["source"] = str(BARKER_XLS.relative_to(PROJECT_ROOT))
        event_frames.append(frame)
        datasets.append(
            event_inputs.EventDataset(
                dataset_id=dataset_id,
                label=label,
                color=spec.color,
                ages_ka=frame["event_age_ka"].to_numpy(dtype=float),
                source=str(BARKER_XLS.relative_to(PROJECT_ROOT)),
            )
        )

    return pd.concat(event_frames, ignore_index=True), datasets


def build_rayleigh_tables(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample orbital phases at event ages and run Rayleigh tests."""

    phase_products = {
        driver: orbital_phase.build_phase_series(driver, settings)
        for driver, settings in ORBITAL_DRIVER_SETTINGS.items()
    }
    event_phases = orbital_phase.sample_event_phases(events, phase_products)
    return event_phases, orbital_phase.build_rayleigh_results(event_phases)


def add_edc_resolution_control(
    binned: pd.DataFrame,
    spec: CatalogueSpec,
    mapping: pd.DataFrame,
    resolution_source: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add log local EDC sample spacing as a detection-resolution control."""

    centers = binned["bin_center_ka"].to_numpy(dtype=float)
    if spec.resolution_age_axis == "edc3":
        edc3_centers = centers
        mapping_extrapolated = np.zeros(len(centers), dtype=bool)
    else:
        edc3_centers, mapping_extrapolated = (
            event_inputs.interpolate_with_linear_extrapolation(
                centers,
                mapping["speleo_age_ka"].to_numpy(dtype=float),
                mapping["edc3_age_ka"].to_numpy(dtype=float),
                context=f"{spec.short_label} SpeleoAge-to-EDC3 mapping",
            )
        )

    spacing, resolution_extrapolated = (
        event_inputs.interpolate_with_linear_extrapolation(
            edc3_centers,
            resolution_source["edc3_age_ka"].to_numpy(dtype=float),
            resolution_source["edc_local_resolution_ka"].to_numpy(dtype=float),
            context=f"{spec.short_label} EDC local resolution",
        )
    )
    spacing = np.clip(spacing, 1e-6, None)
    log_spacing = np.log(spacing)
    scaled, mean, minimum, maximum, value_range = (
        event_inputs.scale_to_zero_mean_range_one(log_spacing)
    )

    out = binned.copy()
    out["resolution_edc3_age_ka"] = edc3_centers
    out["resolution_age_mapping_extrapolated"] = mapping_extrapolated
    out["resolution_source_extrapolated"] = resolution_extrapolated
    out["edc_local_resolution_ka"] = spacing
    out["edc_log_resolution"] = log_spacing
    out[RESOLUTION_TERM] = scaled

    metadata = pd.DataFrame(
        [
            {
                "dataset_id": str(out["dataset_id"].iloc[0]),
                "forcing_id": RESOLUTION_TERM,
                "source": str(JOUZEL_TXT.relative_to(PROJECT_ROOT)),
                "resolution_age_axis": spec.resolution_age_axis,
                "mean": mean,
                "min": minimum,
                "max": maximum,
                "range": value_range,
                "n_mapping_extrapolated_bins": int(mapping_extrapolated.sum()),
                "n_resolution_extrapolated_bins": int(resolution_extrapolated.sum()),
            }
        ]
    )
    return out, metadata


# ---------------------------------------------------------------------------
# Predictive-information analysis
# ---------------------------------------------------------------------------


def _build_design_table(
    dataset: event_inputs.EventDataset,
    spec: CatalogueSpec,
    source: BarkerSource,
    resolution_source: pd.DataFrame,
    history_window_ka: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one catalogue's binned event and predictor table."""

    binned, _, _ = event_inputs.build_binned_inputs(
        [dataset],
        analysis_start_ka=0.0,
        analysis_end_ka=spec.end_ka,
        bin_width_ka=BIN_WIDTH_KA,
        lr04_path=LR04_XLSX,
        co2_path=CO2_XLSX,
        precession_path=PRE_TXT,
        project_root=PROJECT_ROOT,
    )
    binned, metadata = add_edc_resolution_control(
        binned, spec, source.mapping, resolution_source
    )
    binned = predictive.add_same_type_history(binned, history_window_ka)
    return predictive.model_frame(binned), metadata


def _fit_models(frame: pd.DataFrame) -> list[poisson.FittedPoissonModel]:
    """Fit the baseline, climate-state, and full Poisson models."""

    return [
        poisson.fit_poisson_model(frame, model_id, terms, label)
        for model_id, terms, label in MODEL_SPECS
    ]


def _build_likelihood_tests(
    models: list[poisson.FittedPoissonModel],
    frame: pd.DataFrame,
    history_window_ka: float,
) -> pd.DataFrame:
    """Compare the two scientifically relevant pairs of nested models."""

    lookup = poisson.model_lookup(models)
    dataset_id = str(frame["dataset_id"].iloc[0])
    rows = []
    for comparison_id, reduced_id, full_id, question in LR_TEST_SPECS:
        reduced = lookup[(dataset_id, reduced_id)]
        full = lookup[(dataset_id, full_id)]
        metrics = nested_likelihood_metrics(
            loglik_full=full.log_likelihood,
            loglik_reduced=reduced.log_likelihood,
            df=len(full.beta) - len(reduced.beta),
            n_bins=len(frame),
            n_events=int(frame["event_count"].sum()),
            aicc_full=full.aicc,
            aicc_reduced=reduced.aicc,
        )
        rows.append(
            {
                "dataset_id": dataset_id,
                "dataset_label": full.dataset_label,
                "history_window_ka": float(history_window_ka),
                "comparison_id": comparison_id,
                "question": question,
                "reduced_model_id": reduced_id,
                "full_model_id": full_id,
                **metrics,
                "p_value_method": "nominal asymptotic chi-square LRT",
                "reject_LR_at_0p05": metrics["LR_p_value"] < 0.05,
            }
        )
    return pd.DataFrame(rows)


def build_predictive_tables(
    datasets: list[event_inputs.EventDataset],
    source: BarkerSource,
    resolution_source: pd.DataFrame,
    *,
    history_window_ka: float = MAIN_HISTORY_WINDOW_KA,
    specs: tuple[CatalogueSpec, ...] = CATALOGUE_SPECS,
) -> PredictiveTables:
    """Run the same predictive model for each age/support variant."""

    binned, summaries, tests, coefficients, metadata = [], [], [], [], []
    for dataset, spec in zip(datasets, specs, strict=True):
        frame, resolution_meta = _build_design_table(
            dataset, spec, source, resolution_source, history_window_ka
        )
        models = _fit_models(frame)
        binned.append(frame)
        summaries.append(poisson.build_model_summary(models, frame))
        tests.append(_build_likelihood_tests(models, frame, history_window_ka))
        coefficients.append(poisson.build_coefficient_table(models))
        metadata.append(resolution_meta)

    return PredictiveTables(
        binned_inputs=pd.concat(binned, ignore_index=True),
        model_summary=pd.concat(summaries, ignore_index=True),
        likelihood_tests=pd.concat(tests, ignore_index=True),
        coefficients=pd.concat(coefficients, ignore_index=True),
        resolution_metadata=pd.concat(metadata, ignore_index=True),
    )


def build_analysis_summary(
    events: pd.DataFrame,
    rayleigh_results: pd.DataFrame,
    predictive_tables: PredictiveTables,
    *,
    pick_column: str,
    specs: tuple[CatalogueSpec, ...] = CATALOGUE_SPECS,
) -> pd.DataFrame:
    """Collect the core Rayleigh and conditional-PI quantities in one table."""

    rows = []
    for spec in specs:
        dataset_id, event_type, label = _catalogue_identity(spec, pick_column)
        ray = rayleigh_results.query(
            "driver == 'pre' and event_type == @event_type"
        ).iloc[0]
        test = predictive_tables.likelihood_tests.query(
            "dataset_id == @dataset_id and comparison_id == 'phase_after_climate'"
        ).iloc[0]
        full = predictive_tables.model_summary.query(
            "dataset_id == @dataset_id and model_id == 'full'"
        ).iloc[0]
        fit = predictive_tables.binned_inputs.query("dataset_id == @dataset_id")
        rows.append(
            {
                "dataset_id": dataset_id,
                "dataset_label": label,
                "event_definition": _definition_name(pick_column),
                "age_scale": "SpeleoAge" if "speleo" in dataset_id else "EDC3",
                "analysis_end_ka": spec.end_ka,
                "n_rayleigh_events": int(ray["n_phase_events_used"]),
                "rayleigh_mean_phase_deg": float(ray["mean_phase_deg"]),
                "rayleigh_Rbar": float(ray["mean_resultant_length"]),
                "rayleigh_p": float(ray["rayleigh_p"]),
                "n_pi_bins": int(len(fit)),
                "n_pi_events": int(test["n_events"]),
                "pi_support_start_ka": float(fit["bin_start_ka"].min()),
                "pi_support_end_ka": float(fit["bin_end_ka"].max()),
                "history_window_ka": float(test["history_window_ka"]),
                "phase_LR": float(test["LR_statistic"]),
                "phase_nominal_p": float(test["LR_p_value"]),
                "phase_delta_AICc": float(test["delta_AICc_full_minus_reduced"]),
                "phase_bits_per_event": float(test["info_bits_per_event"]),
                "phase_preferred_deg": float(full["pre_phase_preferred_deg"]),
                "phase_rate_ratio_max_min": float(
                    full["pre_phase_rate_ratio_max_vs_min"]
                ),
                "catalogues_are_independent": False,
                "age_uncertainty_propagated": False,
            }
        )
    return pd.DataFrame(rows)


def run_core_analysis(
    *,
    pick_column: str = VARIABLE_PICK,
    history_window_ka: float = MAIN_HISTORY_WINDOW_KA,
    source: BarkerSource | None = None,
    resolution_source: pd.DataFrame | None = None,
) -> AnalysisTables:
    """Run Rayleigh and PI analyses for one Barker event definition."""

    source = load_barker_source() if source is None else source
    resolution_source = (
        load_jouzel_edc_resolution_source()
        if resolution_source is None
        else resolution_source
    )
    events, datasets = build_event_catalogues(source.table, pick_column=pick_column)
    event_phases, rayleigh_results = build_rayleigh_tables(events)
    predictive_tables = build_predictive_tables(
        datasets,
        source,
        resolution_source,
        history_window_ka=history_window_ka,
    )
    analysis_summary = build_analysis_summary(
        events,
        rayleigh_results,
        predictive_tables,
        pick_column=pick_column,
    )
    return AnalysisTables(
        events=events,
        event_phases=event_phases,
        rayleigh_results=rayleigh_results,
        binned_inputs=predictive_tables.binned_inputs,
        model_summary=predictive_tables.model_summary,
        likelihood_tests=predictive_tables.likelihood_tests,
        coefficients=predictive_tables.coefficients,
        resolution_metadata=predictive_tables.resolution_metadata,
        analysis_summary=analysis_summary,
    )


def build_history_sensitivity(
    primary: AnalysisTables,
    source: BarkerSource,
    resolution_source: pd.DataFrame,
) -> pd.DataFrame:
    """Check whether conditional phase results depend on the 5 ka history choice."""

    rows = []
    for window in HISTORY_SENSITIVITY_KA:
        analysis = (
            primary
            if np.isclose(window, MAIN_HISTORY_WINDOW_KA)
            else run_core_analysis(
                history_window_ka=window,
                source=source,
                resolution_source=resolution_source,
            )
        )
        rows.append(
            analysis.analysis_summary[
                [
                    "dataset_id",
                    "history_window_ka",
                    "n_pi_events",
                    "phase_LR",
                    "phase_nominal_p",
                    "phase_delta_AICc",
                    "phase_bits_per_event",
                    "phase_preferred_deg",
                    "phase_rate_ratio_max_min",
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True)


def build_event_definition_sensitivity(
    variable: AnalysisTables,
    fixed: AnalysisTables,
) -> pd.DataFrame:
    """Compare Barker's variable- and fixed-threshold event columns."""

    columns = [
        "dataset_id",
        "event_definition",
        "age_scale",
        "analysis_end_ka",
        "n_rayleigh_events",
        "n_pi_events",
        "rayleigh_p",
        "phase_LR",
        "phase_nominal_p",
        "phase_delta_AICc",
        "phase_bits_per_event",
        "phase_preferred_deg",
        "phase_rate_ratio_max_min",
    ]
    return pd.concat(
        [variable.analysis_summary[columns], fixed.analysis_summary[columns]],
        ignore_index=True,
    )


def phase_response_curve(
    beta_sin: float,
    beta_cos: float,
    *,
    n_points: int = 361,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the phase term's relative event-rate response over one cycle."""

    phase_deg = np.linspace(0.0, 360.0, n_points)
    phase_rad = np.deg2rad(phase_deg)
    relative_rate = np.exp(beta_sin * np.sin(phase_rad) + beta_cos * np.cos(phase_rad))
    return phase_deg, relative_rate


# ---------------------------------------------------------------------------
# Figures: one input/catalogue audit and one direct statistical summary.
# ---------------------------------------------------------------------------


def _save_figure(fig: plt.Figure, stem: str) -> None:
    """Save local PNG/PDF outputs without mutating the manuscript directory."""

    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _format_p(value: float) -> str:
    """Format a p value compactly without implying excess precision."""

    return f"{value:.3f}" if value >= 0.001 else f"{value:.1e}"


def plot_catalogue_audit(
    source: BarkerSource,
    resolution_source: pd.DataFrame,
    variable: AnalysisTables,
    fixed: AnalysisTables,
) -> None:
    """Show catalogue definition, declining resolution, and age-model offsets."""

    fig, axes = plt.subplots(3, 1, figsize=(10.2, 7.6), sharex=False)
    ax_record, ax_resolution, ax_age = axes

    # (a) A two-row rug makes the catalogue definition immediately visible.
    edc = resolution_source.query("0 <= edc3_age_ka <= 800")
    variable_800 = variable.events.query(
        "dataset_id == 'barker_variable_threshold_edc3_0_800'"
    )["event_age_ka"]
    fixed_800 = fixed.events.query("dataset_id == 'barker_fixed_threshold_edc3_0_800'")[
        "event_age_ka"
    ]
    ax_record.eventplot(
        [variable_800, fixed_800],
        colors=["#d95f02", "#333333"],
        lineoffsets=[1.0, 0.0],
        linelengths=0.72,
        linewidths=0.75,
    )
    ax_record.set_yticks([1.0, 0.0])
    ax_record.set_yticklabels(
        [
            f"Variable threshold\nN={len(variable_800)}",
            f"Fixed threshold\nN={len(fixed_800)}",
        ]
    )
    ax_record.set_ylim(-0.65, 1.65)
    ax_record.set_xlabel("Age (kyr, EDC3)")
    ax_record.set_title(
        "a  Barker Table S3 event definitions", loc="left", fontweight="bold"
    )

    # (b) Barker warns that old events can be merged or lost as resolution declines.
    ax_resolution.plot(
        edc["edc3_age_ka"],
        edc["edc_local_resolution_ka"],
        color="#3f7f93",
        lw=0.8,
    )
    ax_resolution.axhline(1.0, color="0.25", ls="--", lw=0.8, label="1 kyr per sample")
    ax_resolution.set_yscale("log")
    ax_resolution.set_ylabel("Local spacing\n(kyr/sample)")
    ax_resolution.set_title(
        "b  EDC sampling-resolution proxy", loc="left", fontweight="bold"
    )
    ax_resolution.legend(frameon=False, loc="upper left")

    for axis in (ax_record, ax_resolution):
        axis.axvspan(600.0, 800.0, color="#efe5c7", alpha=0.65, zorder=-5)
        axis.axvline(640.0, color="#8c6d31", lw=0.9)
        axis.set_xlim(0.0, 800.0)
        axis.grid(False)
    ax_resolution.text(
        720.0,
        ax_resolution.get_ylim()[1] / 1.7,
        "lower-confidence\n600-800 ka interval",
        ha="center",
        va="top",
        color="#735c24",
        fontsize=8.5,
    )
    ax_resolution.set_xlabel("Age (kyr, EDC3)")

    # (c) SpeleoAge is a re-dating of the same events, not another catalogue.
    mapping = source.mapping.sort_values("edc3_age_ka")
    offset = mapping["speleo_age_ka"] - mapping["edc3_age_ka"]
    ax_age.plot(mapping["edc3_age_ka"], offset, color="#8b3f71", lw=1.0)
    ax_age.axhline(0.0, color="0.25", lw=0.8)
    ax_age.axvspan(
        265.0,
        315.0,
        color="0.88",
        alpha=0.7,
        label="limited alignment (265-315 ka)",
    )
    ax_age.set(
        xlim=(0.0, 400.0),
        xlabel="Age (kyr, EDC3)",
        ylabel="SpeleoAge - EDC3\n(kyr)",
    )
    ax_age.set_title("c  Barker SpeleoAge re-dating", loc="left", fontweight="bold")
    ax_age.legend(frameon=False, loc="best")
    ax_age.grid(False)

    fig.subplots_adjust(left=0.10, right=0.98, top=0.96, bottom=0.08, hspace=0.48)
    # Keep the established filename so it matches the retained output archive.
    _save_figure(fig, "fig03_barker_inputs_and_fitted_rates")


def _draw_rayleigh_panel(
    axis: plt.Axes,
    spec: CatalogueSpec,
    analysis: AnalysisTables,
) -> None:
    """Draw a compact phase histogram with a descriptive mean vector."""

    phases = analysis.event_phases.query(
        "driver == 'pre' and event_type == @spec.event_type and not phase_extrapolated"
    )["phase_rad"].to_numpy(dtype=float)
    result = analysis.rayleigh_results.query(
        "driver == 'pre' and event_type == @spec.event_type"
    ).iloc[0]

    width = 2.0 * np.pi / 18.0
    shifted = np.mod(phases + width / 2.0, 2.0 * np.pi)
    counts, edges = np.histogram(shifted, bins=np.linspace(0.0, 2.0 * np.pi, 19))
    centers = edges[:-1] - width / 2.0
    axis.bar(
        centers,
        counts,
        width=width,
        align="center",
        color=spec.color,
        alpha=0.58,
        edgecolor="white",
        linewidth=0.6,
    )
    max_count = max(int(counts.max()), 1)
    axis.annotate(
        "",
        xy=(
            float(result["mean_phase_rad"]),
            float(result["mean_resultant_length"]) * max_count,
        ),
        xytext=(float(result["mean_phase_rad"]), 0.0),
        arrowprops={"arrowstyle": "-|>", "lw": 1.6, "color": "#202020"},
    )
    axis.set_theta_zero_location("E")
    axis.set_theta_direction(1)
    axis.set_xticks([0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0])
    axis.set_xticklabels(["min", "90°", "max", "270°"])
    axis.set_yticklabels([])
    axis.grid(color="0.82", alpha=0.45, lw=0.6)
    axis.set_title(
        f"{spec.short_label}\n"
        rf"Rayleigh N={int(result['n_phase_events_used'])}, $\bar{{R}}$={result['mean_resultant_length']:.2f}, "
        f"p={_format_p(float(result['rayleigh_p']))}\n"
        f"descriptive mean={result['mean_phase_deg']:.1f}°",
        fontsize=9.2,
        pad=24,
    )


def _phase_coefficients(
    analysis: AnalysisTables, dataset_id: str
) -> tuple[float, float]:
    """Return full-model sine and cosine coefficients for one catalogue."""

    rows = analysis.coefficients.query(
        "dataset_id == @dataset_id and model_id == 'full'"
    )
    values = rows.set_index("term")["beta"]
    return float(values["pre_phase_sin"]), float(values["pre_phase_cos"])


def plot_phase_results(variable: AnalysisTables, fixed: AnalysisTables) -> None:
    """Combine unconditional Rayleigh and conditional PI phase results."""

    fig = plt.figure(figsize=(10.6, 7.4))
    grid = fig.add_gridspec(2, 3, height_ratios=(1.05, 0.95), hspace=0.62, wspace=0.30)
    for column, spec in enumerate(CATALOGUE_SPECS):
        polar_axis = fig.add_subplot(grid[0, column], projection="polar")
        _draw_rayleigh_panel(polar_axis, spec, variable)

        axis = fig.add_subplot(grid[1, column])
        variable_id = spec.dataset_id
        fixed_id = spec.dataset_id.replace("variable_threshold", "fixed_threshold")
        for analysis, dataset_id, linestyle, label, alpha in (
            (variable, variable_id, "-", "variable threshold", 1.0),
            (fixed, fixed_id, "--", "fixed threshold", 0.72),
        ):
            beta_sin, beta_cos = _phase_coefficients(analysis, dataset_id)
            phase_deg, relative_rate = phase_response_curve(beta_sin, beta_cos)
            axis.plot(
                phase_deg,
                relative_rate,
                color=spec.color,
                ls=linestyle,
                lw=1.8 if linestyle == "-" else 1.3,
                alpha=alpha,
                label=label,
            )

        main = variable.analysis_summary.query("dataset_id == @variable_id").iloc[0]
        sensitivity = fixed.analysis_summary.query("dataset_id == @fixed_id").iloc[0]
        axis.axvline(main["phase_preferred_deg"], color=spec.color, lw=0.8, alpha=0.6)
        axis.axhline(1.0, color="0.65", lw=0.7)
        axis.set_xlim(0.0, 360.0)
        axis.set_xticks([0.0, 90.0, 180.0, 270.0, 360.0])
        axis.set_xticklabels(["min", "90°", "max", "270°", "min"])
        axis.set_xlabel("Precession phase")
        if column == 0:
            axis.set_ylabel("Phase-term relative rate")
        axis.grid(False)
        axis.text(
            0.02,
            0.98,
            "Variable: "
            f"N={int(main['n_pi_events'])}, nominal p={_format_p(main['phase_nominal_p'])}\n"
            f"{main['phase_bits_per_event']:.3f} bits/event; "
            f"φ={main['phase_preferred_deg']:.1f}°; max/min={main['phase_rate_ratio_max_min']:.2f}\n"
            f"Fixed-threshold sensitivity: p={_format_p(sensitivity['phase_nominal_p'])}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.0,
            bbox={"facecolor": "white", "edgecolor": "0.83", "alpha": 0.9, "pad": 2.0},
        )
        if column == 0:
            axis.legend(frameon=False, loc="lower left")

    fig.text(
        0.5,
        0.495,
        "Conditional PI response: full model versus climate-state model",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.08)
    # Keep the established filename so it matches the retained output archive.
    _save_figure(fig, "fig02_barker_predictive_likelihood_tests")


# ---------------------------------------------------------------------------
# Output and command-line workflow
# ---------------------------------------------------------------------------


def write_outputs(
    primary: AnalysisTables,
    event_definition_sensitivity: pd.DataFrame,
    history_sensitivity: pd.DataFrame,
) -> None:
    """Write compact, audit-friendly tables for the primary analysis."""

    OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "barker2011_event_catalogues_used.csv": primary.events,
        "barker2011_event_orbital_phases.csv": primary.event_phases,
        "barker2011_rayleigh_phase_results.csv": primary.rayleigh_results,
        "barker2011_binned_predictive_inputs.csv": primary.binned_inputs,
        "barker2011_predictive_model_summary.csv": primary.model_summary,
        "barker2011_predictive_likelihood_tests.csv": primary.likelihood_tests,
        "barker2011_predictive_coefficients.csv": primary.coefficients,
        "barker2011_edc_resolution_scale_summary.csv": primary.resolution_metadata,
        "barker2011_analysis_summary.csv": primary.analysis_summary,
        "barker2011_event_definition_sensitivity.csv": event_definition_sensitivity,
        "barker2011_history_window_sensitivity.csv": history_sensitivity,
    }
    for filename, frame in outputs.items():
        frame.to_csv(OUT_DATA_DIR / filename, index=False)

    pd.DataFrame(
        [
            {
                "run_name": RUN_NAME,
                "source": str(BARKER_XLS.relative_to(PROJECT_ROOT)),
                "resolution_source": str(JOUZEL_TXT.relative_to(PROJECT_ROOT)),
                "primary_catalogue": CATALOGUE_SPECS[0].dataset_id,
                "bin_width_ka": BIN_WIDTH_KA,
                "history_window_ka": MAIN_HISTORY_WINDOW_KA,
                "event_definition": VARIABLE_PICK,
                "lr_p_value_method": "nominal asymptotic chi-square",
                "age_uncertainty_propagated": False,
                "catalogue_variants_independent": False,
                "note": (
                    "Baseline controls same-type history and local EDC age spacing. "
                    "Barker variable threshold already adjusts partly for resolution."
                ),
            }
        ]
    ).to_csv(OUT_DATA_DIR / "parameters.csv", index=False)

    # These large/obsolete files belonged to the previous, redundant rate plot.
    stale_files = [
        OUT_DATA_DIR / "barker2011_fitted_rates.csv",
        OUT_FIG_DIR / "fig01_barker_precession_rayleigh_polar.png",
        OUT_FIG_DIR / "fig01_barker_precession_rayleigh_polar.pdf",
        OUT_FIG_DIR / "fig01_barker_catalogue_audit.png",
        OUT_FIG_DIR / "fig01_barker_catalogue_audit.pdf",
        OUT_FIG_DIR / "fig02_barker_phase_results.png",
        OUT_FIG_DIR / "fig02_barker_phase_results.pdf",
    ]
    for path in stale_files:
        if path.exists():
            path.unlink()


def print_summary(
    primary: AnalysisTables,
    event_definition_sensitivity: pd.DataFrame,
    history_sensitivity: pd.DataFrame,
) -> None:
    """Print the core result and the two most informative robustness checks."""

    columns = [
        "dataset_id",
        "n_rayleigh_events",
        "rayleigh_p",
        "n_pi_events",
        "phase_nominal_p",
        "phase_bits_per_event",
        "phase_preferred_deg",
        "phase_rate_ratio_max_min",
    ]
    print("\nPrimary variable-threshold results:")
    print(primary.analysis_summary[columns].to_string(index=False))

    print("\nEvent-definition sensitivity (conditional phase nominal p):")
    print(
        event_definition_sensitivity[
            ["dataset_id", "event_definition", "n_pi_events", "phase_nominal_p"]
        ].to_string(index=False)
    )
    print("\nHistory-window sensitivity (conditional phase nominal p):")
    print(
        history_sensitivity[
            [
                "dataset_id",
                "history_window_ka",
                "phase_nominal_p",
                "phase_preferred_deg",
            ]
        ].to_string(index=False)
    )
    print(
        "\nInterpret cautiously: the three age/support variants overlap; LR p values "
        "are nominal and age uncertainty is not propagated."
    )
    print(f"Wrote tables to {OUT_DATA_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Wrote figures to {OUT_FIG_DIR.relative_to(PROJECT_ROOT)}")


def main() -> None:
    """Run the primary analysis, robustness checks, outputs, and two figures."""

    source = load_barker_source()
    resolution_source = load_jouzel_edc_resolution_source()
    primary = run_core_analysis(source=source, resolution_source=resolution_source)
    fixed = run_core_analysis(
        pick_column=FIXED_PICK,
        source=source,
        resolution_source=resolution_source,
    )
    event_definition_sensitivity = build_event_definition_sensitivity(primary, fixed)
    history_sensitivity = build_history_sensitivity(primary, source, resolution_source)

    write_outputs(primary, event_definition_sensitivity, history_sensitivity)
    plot_catalogue_audit(source, resolution_source, primary, fixed)
    plot_phase_results(primary, fixed)
    print_summary(primary, event_definition_sensitivity, history_sensitivity)


if __name__ == "__main__":
    main()
