"""Tables, paired figure style and manuscript notes for orbital comparisons."""

from pathlib import Path
import hashlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

from toolbox import combined_pi
from toolbox.project_config import PROJECT_ROOT

DRIVER_LABELS = {"ecc": "Eccentricity", "obl": "Obliquity",
                 "insol65n": "65°N summer-solstice\ninsolation"}
DRIVER_COLORS = {"ecc": "#CC79A7", "obl": "#009E73", "insol65n": "#D55E00"}


def select_realizations(table, age_columns, n_realizations, seed):
    """Select saved chronologies once; retain original IDs and pairing columns."""
    if table.realization_id.isna().any() or table.realization_id.duplicated().any():
        raise ValueError("Saved chronology IDs must be present and unique")
    if not 1 <= n_realizations <= len(table):
        raise ValueError("Requested chronology sample exceeds the saved ensemble")
    values = table.loc[:, age_columns].to_numpy(float)
    if not np.isfinite(values).all() or np.any(np.diff(values, axis=1) <= 0):
        raise ValueError("Chronologies must have finite, ordered event ages")
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(table), n_realizations, replace=False)
    return table.iloc[selected].reset_index(drop=True)


def invalid_tables(point, reason):
    """Keep unsupported draws in every comparison's denominator, without estimates."""
    models = point["models"].iloc[:0].reindex(range(len(point["models"])))
    comparisons = point["comparisons"].iloc[:0].reindex(range(len(point["comparisons"])))
    for column in ("model_id", "terms", "n_parameters", "n_bins", "exposure_kyr"):
        models[column] = point["models"][column].to_numpy()
    for column in ("comparison_id", "driver_id", "comparison_group", "reduced_model_id",
                   "full_model_id", "df", "n_bins", "exposure_kyr"):
        comparisons[column] = point["comparisons"][column].to_numpy()
    for frame in (models, comparisons):
        frame["fit_valid"] = False
        frame["invalid_reason"] = reason
    return {"models": models, "comparisons": comparisons}


def check_reference(point, reference):
    """Compare against the saved main result, allowing its CSV rounding only."""
    comparison = point["comparisons"].set_index("comparison_id").loc["phase_reference"]
    phase = point["models"].set_index("model_id").loc["BP"]
    checks = {
        "info_bits_per_event": (comparison.info_bits_per_event, reference.info_bits_per_event),
        "LR_statistic": (comparison.LR_statistic, reference.LR_statistic),
        "n_events": (comparison.n_events, reference.n_predictive_events),
        "exposure_kyr": (comparison.exposure_kyr, reference.response_exposure_kyr),
        "phase_preferred_deg": (phase.pre_phase_preferred_deg, reference.pre_phase_preferred_deg),
        "phase_rate_ratio": (phase.pre_phase_rate_ratio_max_vs_min,
                             reference.pre_phase_rate_ratio_max_vs_min),
    }
    rows = [dict(metric=key, refitted=value, saved_main=expected,
                 matches=bool(np.isclose(value, expected, rtol=2e-6, atol=2e-6)))
            for key, (value, expected) in checks.items()]
    audit = pd.DataFrame(rows)
    if not audit.matches.all():
        raise RuntimeError(f"Main-model reference changed:\n{audit.to_string(index=False)}")
    return audit


def plot_comparisons(summary, catalogue_label):
    """Compact paper panels; BG, Pre and Orb are defined in the figure caption."""
    plt.rcParams.update({"font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 8,
        "axes.linewidth": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(180 / 25.4, 82 / 25.4), sharey=True)
    fig.subplots_adjust(left=0.205, right=0.985, bottom=0.29, top=0.75, wspace=0.16)
    specs = [
        ("driver_after_base", r"$\mathrm{BG}+\mathrm{Orb}$" + "\n" + r"vs $\mathrm{BG}$"),
        ("driver_after_phase", r"$\mathrm{BG}+\mathrm{Pre}+\mathrm{Orb}$" + "\n" +
         r"vs $\mathrm{BG}+\mathrm{Pre}$"),
        ("phase_after_driver", r"$\mathrm{BG}+\mathrm{Pre}+\mathrm{Orb}$" + "\n" +
         r"vs $\mathrm{BG}+\mathrm{Orb}$"),
    ]
    reference = summary.set_index("comparison_id").loc["phase_reference"]
    # A common minimum span keeps both catalogues directly comparable on reruns.
    upper = np.nanmax(summary[["info_bits_per_event_point", "info_bits_per_event_q975"]])
    xmax = max(0.30, np.ceil(upper / 0.05) * 0.05 + 0.01)
    for panel, (ax, (group, title)) in enumerate(zip(axes, specs)):
        subset = summary.loc[summary.comparison_group.eq(group)].set_index("driver_id")
        for y, driver in enumerate(DRIVER_LABELS):
            row = subset.loc[driver]
            color = DRIVER_COLORS[driver]
            low, high = row.info_bits_per_event_q025, row.info_bits_per_event_q975
            if np.isfinite([low, high]).all():
                ax.hlines(y, low, high, color=color, linewidth=2.3, alpha=0.55)
                ax.plot(row.info_bits_per_event_median, y, "|", color=color,
                        markersize=11, markeredgewidth=1.5)
            if np.isfinite(row.info_bits_per_event_point):
                ax.plot(row.info_bits_per_event_point, y, "o", color=color,
                        markersize=5, markeredgecolor="white", markeredgewidth=0.5)
        if panel != 1:
            ax.axvline(reference.info_bits_per_event_point, color="0.35",
                       linewidth=0.9, linestyle=(0, (3, 3)), zorder=0)
        ax.set(xlim=(-0.009, xmax), ylim=(2.48, -0.48), xlabel="PI (bits event$^{-1}$)")
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        ax.set_title(title, fontsize=8.5, pad=10)
        ax.text(0, 1.23, f"({chr(97 + panel)})", transform=ax.transAxes,
                fontweight="bold", fontsize=9)
        ax.grid(axis="x", linewidth=0.45, color="0.90", zorder=0)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=8)
    axes[0].set_yticks(range(3), list(DRIVER_LABELS.values()))
    fig.text(0.205, 0.94, catalogue_label, weight="bold", fontsize=10)
    handles = [Line2D([], [], marker="o", linestyle="none", color="0.25", label="Point ages"),
               Line2D([], [], marker="|", linestyle="none", markersize=10, color="0.25", label="MC median"),
               Line2D([], [], linewidth=2.3, alpha=0.55, color="0.25", label="95% MC range"),
               Line2D([], [], linestyle="--", color="0.35",
                      label=r"$\mathrm{BG}+\mathrm{Pre}$ vs $\mathrm{BG}$ (point ages)")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.035),
               frameon=False, ncol=2, columnspacing=2.0, fontsize=7.5)
    return fig, axes


def write_notes(result, note_dir, run_name, catalogue_label):
    """Generate paper-ready methods and exact results from the saved analysis tables."""
    summary = result["comparison_summary"]
    point = result["point"]["comparisons"]
    reference = point.set_index("comparison_id").loc["phase_reference"]
    phase = result["point"]["models"].set_index("model_id").loc["BP"]
    params = result["parameters"]
    status = result["realization_status"]
    n_supported = int(status.within_observation_support.sum())
    n_valid = int(status.all_comparisons_valid.sum())
    counts = result["mc_models"].loc[
        result["mc_models"].model_id.eq("B") & result["mc_models"].fit_valid, "n_events"]
    count_description = "; ".join(f"{int(count)} events in {int(number)} draws"
                                  for count, number in counts.value_counts().sort_index().items())
    lines = [f"{catalogue_label}: orbital-driver sensitivity", "", "Methods",
        "We model warming-onset rates over the whole observed time interval with the "
        "same binned conditional Poisson likelihood as the main analysis. B contains "
        "an intercept, prior-event count, LR04 and CO2. The pooled analysis additionally "
        "retains its MIS6 segment contrast. P contains sine and cosine of precession phase.",
        f"The bin width is {params['bin_width_kyr']:g} kyr and the history window is "
        f"{params['history_window_kyr']:g} kyr. There are {int(reference.n_events)} point-age "
        f"response events, {int(reference.n_bins)} response bins and {reference.exposure_kyr:g} kyr "
        "of exposure. Exact partial-bin durations enter the likelihood as offsets. "
        "Event history uses strictly older bin centers and resets at each observation segment.",
        "Precession-index minima define 0 degrees and maxima 180 degrees; unwrapped phase "
        "increases toward older BP ages. The scalar orbital variables are used as values, "
        "not transformed into phase.",
        "We examine three scalar linear covariates X: eccentricity, obliquity and 65°N "
        "summer-solstice daily mean top-of-atmosphere insolation (solar longitude 90°; "
        "S0=1365 W m-2). Eight models are fitted: B, B+P, and B+X and B+P+X for each X. "
        "No lag, quadratic term or phase-amplitude interaction is fitted. Eccentricity "
        "represents the climatic-precession amplitude envelope, but its additive main "
        "effect does not make the fitted phase rate ratio depend on eccentricity.",
        "All orbital inputs use La2004. Signed orbital source times are converted to "
        "positive BP1950 ages by -source_time-0.05 kyr. The insolation NetCDF has positive "
        "ages and is shifted by -0.05 kyr; its J2000 origin is inferred from numerical "
        "agreement with the raw orbital solution, not explicitly stated by its BP label. "
        "See docs/orbital_driver_inputs.md. Insolation has a native 1-kyr sampling interval; "
        "linear interpolation to event-bin centers does not increase native resolution. "
        "Obliquity is converted from radians to degrees. Each new scalar is centered and "
        "divided by its range over the fixed response bins of its own catalogue. Existing "
        "climate scaling and phase conventions are retained.",
        f"We select {params['n_realizations']} distinct saved combined-error chronologies "
        f"without replacement (NumPy default_rng, seed {params['seed']}). Selected IDs are "
        "saved. All eight models within a catalogue use the same chronologies, response "
        "support and forcing scales. Counts and history are recalculated for every draw. "
        "Cross-catalogue selection does not synchronize the two age models. Reported "
        "2.5th–97.5th percentiles measure sensitivity to the adopted event-age model, "
        "not event-process sampling uncertainty or a complete confidence interval. "
        "Phase quantiles are unwrapped about each model's point-age peak.",
        "For nested comparisons, PI=(logL_full-logL_reduced)/(N_events ln2). Scalar "
        "additions have one degree of freedom; phase additions have two. Point-age "
        "chi-square p values are nominal. Holm correction uses the nine new comparisons "
        "within each catalogue; the original phase reference is outside that family. "
        "These tests have not been calibrated by a new event-process bootstrap. "
        "AIC is 2k-2logL; AICc follows the existing project convention using the number "
        "of response bins. Raw PI alone does not establish the best nonnested model.",
        "", "Results",
        f"The main reference is reproduced: PI={reference.info_bits_per_event:.6f} bits/event, "
        f"LR={reference.LR_statistic:.6f}, nominal p={reference.nominal_p:.6g}; "
        f"preferred phase={phase.pre_phase_preferred_deg:.3f} degrees and phase max/min "
        f"rate ratio={phase.pre_phase_rate_ratio_max_vs_min:.4f}.", "",
        f"Of {len(status)} selected chronologies, {n_supported} lie within observation "
        f"support and {n_valid} have all ten comparisons valid. Unsupported draws retain "
        "their original IDs and invalid status; they are neither clipped nor replaced. "
        "Each row below reports the valid denominator used for its quantiles.", "",
        f"Response-event counts: {count_description}. Fixed source membership does not "
        "force every event into the response interval: an age may enter a history-only "
        "buffer. Such events still supply history. PI uses each realization's actual "
        "response count, shared by all eight models in that realization.", "",
        "Comparison | PI point | MC median [2.5%,97.5%] | df | nominal p | Holm nominal p | Delta AIC | valid MC"]
    for _, row in summary.iterrows():
        original = point.set_index("comparison_id").loc[row.comparison_id]
        lines.append(f"{row.comparison_id} | {row.info_bits_per_event_point:.6f} | "
                     f"{row.info_bits_per_event_median:.6f} "
                     f"[{row.info_bits_per_event_q025:.6f}, {row.info_bits_per_event_q975:.6f}] | "
                     f"{int(original.df)} | {original.nominal_p:.6g} | "
                     f"{original.holm_nominal_p:.6g} | {original.delta_AIC:.6f} | "
                     f"{int(row.n_mc_valid)}/{int(row.n_mc_total)}")
    lookup = point.set_index("comparison_id")
    q_base, q_added, phase_q = [lookup.loc[key] for key in (
        "insol65n_after_base", "insol65n_after_phase", "phase_after_insol65n")]
    best = result["point"]["models"].sort_values("AIC").iloc[0]
    lines += ["", "Observed pattern",
        f"Insolation added to B gives PI={q_base.info_bits_per_event:.6f} bits/event "
        f"(nominal p={q_base.nominal_p:.6g}; Holm p={q_base.holm_nominal_p:.6g}). "
        f"Its incremental PI after phase is {q_added.info_bits_per_event:.6f} "
        f"(nominal p={q_added.nominal_p:.6g}; Holm p={q_added.holm_nominal_p:.6g}). "
        f"Conversely, phase added after insolation gives PI={phase_q.info_bits_per_event:.6f}, "
        f"nominal p={phase_q.nominal_p:.6g} and Holm p={phase_q.holm_nominal_p:.6g}. "
        "Thus, these data do not establish a phase contribution independent of insolation "
        "at the 0.05 level after the planned nine-test correction. This does not negate "
        "the original phase-versus-background comparison.",
        f"The lowest point-age AIC among the eight candidates is for {best.model_id} "
        f"(AIC={best.AIC:.6f}). These rankings and the shared information between phase "
        "and insolation do not uniquely identify a physical driver.",
        "", "Interpretation and limitations",
        "These are conditional associations after accounting for LR04, CO2 and event "
        "history. A predictor that adds little after phase may contain shared orbital "
        "information or act through background climate. Its lack of incremental "
        "information does not exclude those pathways. Insolation and orbital components "
        "are not independent forcings. Correlations and design-matrix diagnostics are saved.",
        "Barker SpeleoAge continues to be treated as BP1950 without direct verification "
        "of that column's reference year. Barker and the pooled catalogue are separate "
        "analyses, not independent physical replications; no p values are pooled.",
        "", "References",
        "Laskar et al. (2004), A&A 428, 261–285, doi:10.1051/0004-6361:20041335.",
        "Event and chronology provenance follows the current main and combined-age "
        "analyses; exact files and hashes are listed in input_code_sha256.csv."]
    note_dir.mkdir(parents=True, exist_ok=True)
    (note_dir / f"{run_name}_Methods_and_results.txt").write_text("\n".join(lines) + "\n")
    write_caption(result, note_dir, run_name, catalogue_label)


def write_caption(result, note_dir, run_name, catalogue_label):
    """Define the compact figure notation and the age-sensitivity symbols."""
    is_barker = "Barker" in catalogue_label
    source = ("Orbital-driver sensitivity of reconstructed warming events from Barker et al. "
              "(2011) on the speleothem-based chronology (SpeleoAge)." if is_barker else
              "Orbital-driver sensitivity of the combined North Greenland Ice Core Project "
              "(NGRIP) and Marine Isotope Stage 6 (MIS 6) speleothem warming-event record.")
    baseline = ("BG (background) denotes the full baseline model: an intercept, the warming-event "
                "count during the preceding 1.5 kyr, the LR04 benthic oxygen-isotope stack, "
                "and atmospheric CO2.")
    if not is_barker:
        baseline += " Separate baseline rates are fitted for NGRIP and MIS 6."
    summary = result["comparison_summary"]
    minimum_used, maximum_used = int(summary.n_mc_valid.min()), int(summary.n_mc_valid.max())
    used = str(minimum_used) if minimum_used == maximum_used else f"{minimum_used}–{maximum_used}"
    selected = int(result["parameters"]["n_realizations"])
    caption = (source + "\n\n" + baseline +
        " Pre denotes precession phase, represented by sine and cosine terms. Orb denotes "
        "the orbital variable named in each row: eccentricity, obliquity, or 65°N "
        "summer-solstice daily mean top-of-atmosphere insolation. Panels compare "
        "(a) BG + Orb versus BG, (b) BG + Pre + Orb versus BG + Pre, and "
        "(c) BG + Pre + Orb versus BG + Orb. Thus Orb is added in (a,b), whereas "
        "Pre is added in (c). Orb adds one coefficient; Pre adds two.\n\n"
        "PI (predictive information) is the increase in fitted log likelihood divided "
        "by the response-event count and ln 2, expressed in bits per event. Filled circles "
        "show point-age estimates; vertical ticks show Monte Carlo (MC) medians, and "
        "horizontal bars show the 2.5th–97.5th percentile range across "
        f"{used} valid chronologies from {selected} selected realizations. These ranges "
        "describe age sensitivity, not complete confidence intervals. Dashed lines in "
        "(a,c) show the point-age PI for BG + Pre versus BG; they are references, not "
        "significance thresholds. All models use identical events, bins and observation "
        "exposure within each realization.\n")
    note_dir.mkdir(parents=True, exist_ok=True)
    (note_dir / f"{run_name}_Caption.txt").write_text(caption)


def save_results(result, output_root, run_name, catalogue_label, input_paths):
    """Write inspectable inputs, all fits, denominator audits and paper artifacts."""
    data_dir = output_root / "data/processed" / run_name
    figure_dir = output_root / "figures" / run_name
    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    for key in ("models", "comparisons", "coefficients"):
        result["point"][key].to_csv(data_dir / f"point_{key}.csv", index=False)
    rates = result["point"]["fitted_rates"]
    rate_columns = rates.columns.difference(result["response"].columns, sort=False)
    binned = pd.concat([result["response"].reset_index(drop=True),
                       rates[rate_columns].reset_index(drop=True)], axis=1)
    binned.to_csv(data_dir / "binned_inputs_and_fitted_rates.csv", index=False)
    for key in ("comparison_summary", "phase_summary", "mc_models", "mc_comparisons",
                "selected_realizations", "realization_status", "reference_check",
                "scaling", "provenance", "events", "support"):
        result[key].to_csv(data_dir / f"{key}.csv", index=False)
    predictors = [*combined_pi.FULL_TERMS, "ecc_scaled", "obl_scaled", "insol65n_scaled"]
    predictors = [term for term in predictors if term in result["response"]]
    result["response"][predictors].corr().to_csv(data_dir / "predictor_correlation.csv")
    pd.DataFrame([dict(parameter=k, value=v) for k, v in result["parameters"].items()]).to_csv(
        data_dir / "parameters.csv", index=False)
    paths = list(dict.fromkeys([Path(path) for path in input_paths]))
    hashes = [dict(path=str(path.relative_to(PROJECT_ROOT)),
                   sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in paths]
    pd.DataFrame(hashes).to_csv(data_dir / "input_code_sha256.csv", index=False)
    write_notes(result, output_root / "experiment_note", run_name, catalogue_label)
    redraw_saved_results(output_root, run_name, catalogue_label)
    return data_dir, figure_dir


def redraw_saved_results(output_root, run_name, catalogue_label):
    """Refresh only figures and caption from saved fits; record the rendering inputs."""
    data_dir = output_root / "data/processed" / run_name
    figure_dir = output_root / "figures" / run_name
    summary_path = data_dir / "comparison_summary.csv"
    parameter_path = data_dir / "parameters.csv"
    summary = pd.read_csv(summary_path)
    parameters = pd.read_csv(parameter_path).set_index("parameter").value.to_dict()
    fig, _ = plot_comparisons(summary, catalogue_label)
    figure_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(figure_dir / f"{run_name}.{extension}", dpi=600)
    plt.close(fig)
    write_caption(dict(comparison_summary=summary, parameters=parameters),
                  output_root / "experiment_note", run_name, catalogue_label)
    # Rendering provenance is separate from the hashes of the original model run.
    files = [(summary_path, "fitted comparison summary"), (parameter_path, "analysis settings"),
             (Path(__file__), "figure and caption code")]
    rows = []
    for path, role in files:
        resolved = path.resolve()
        label = resolved.relative_to(PROJECT_ROOT) if resolved.is_relative_to(PROJECT_ROOT) else resolved
        rows.append(dict(path=str(label), role=role, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    pd.DataFrame(rows).to_csv(data_dir / "figure_provenance.csv", index=False)
    return figure_dir


if __name__ == "__main__":
    # Run from the project root: python -m toolbox.orbital_driver_reporting
    for output_root, run_name, label in (
        (PROJECT_ROOT, "NGRIP_MIS6_orbital_driver_sensitivity", "NGRIP–MIS6 warming events"),
        (PROJECT_ROOT / "Barker2011", "Barker2011_orbital_driver_sensitivity",
         "Barker 2011 SpeleoAge warming events"),
    ):
        print(redraw_saved_results(output_root, run_name, label))
