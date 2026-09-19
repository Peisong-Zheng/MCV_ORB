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

from toolbox import combined_likelihood
from toolbox.project_config import PROJECT_ROOT

DRIVER_LABELS = {"ecc": "Eccentricity", "obl": "Obliquity",
                 "insol65n": "65°N summer-solstice\ninsolation"}
DRIVER_COLORS = {"ecc": "#CC79A7", "obl": "#009E73", "insol65n": "#D55E00"}


def read_selected_realizations(path, age_columns, n_realizations=None):
    """Read the frozen 500-row chronology selection without drawing new indices."""
    table = pd.read_csv(path, float_precision="round_trip")
    if table.realization_id.isna().any() or table.realization_id.duplicated().any():
        raise ValueError("Saved chronology IDs must be present and unique")
    if n_realizations is not None:
        if not 1 <= n_realizations <= len(table):
            raise ValueError("Requested chronology subset exceeds the saved selection")
        table = table.iloc[:n_realizations].copy()
    values = table.loc[:, age_columns].to_numpy(float)
    if not np.isfinite(values).all() or np.any(np.diff(values, axis=1) <= 0):
        raise ValueError("Chronologies must have finite, ordered event ages")
    return table.reset_index(drop=True)


def invalid_tables(point, reason):
    """Keep unsupported draws in every comparison's denominator, without estimates."""
    models = point["models"].iloc[:0].reindex(range(len(point["models"])))
    comparisons = point["comparisons"].iloc[:0].reindex(range(len(point["comparisons"])))
    for column in ("model_id", "terms", "n_parameters"):
        models[column] = point["models"][column].to_numpy()
    for column in ("comparison_id", "driver_id", "comparison_group", "reduced_model_id",
                   "full_model_id", "df"):
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
        "gain_bits_per_event": (comparison.gain_bits_per_event, reference.gain_bits_per_event),
        "LR_statistic": (comparison.LR_statistic, reference.LR_statistic),
        "n_events": (comparison.n_events, reference.n_response_events),
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
    upper = np.nanmax(summary[["gain_bits_per_event_point", "gain_bits_per_event_q975"]])
    xmax = max(0.30, np.ceil(upper / 0.05) * 0.05 + 0.01)
    for panel, (ax, (group, title)) in enumerate(zip(axes, specs)):
        subset = summary.loc[summary.comparison_group.eq(group)].set_index("driver_id")
        for y, driver in enumerate(DRIVER_LABELS):
            row = subset.loc[driver]
            color = DRIVER_COLORS[driver]
            low, high = row.gain_bits_per_event_q025, row.gain_bits_per_event_q975
            if np.isfinite([low, high]).all():
                ax.hlines(y, low, high, color=color, linewidth=2.3, alpha=0.55)
                ax.plot(row.gain_bits_per_event_median, y, "|", color=color,
                        markersize=11, markeredgewidth=1.5)
            if np.isfinite(row.gain_bits_per_event_point):
                ax.plot(row.gain_bits_per_event_point, y, "o", color=color,
                        markersize=5, markeredgecolor="white", markeredgewidth=0.5)
        if panel != 1:
            ax.axvline(reference.gain_bits_per_event_point, color="0.35",
                       linewidth=0.9, linestyle=(0, (3, 3)), zorder=0)
        ax.set(xlim=(-0.009, xmax), ylim=(2.48, -0.48), xlabel="Gain (bits event$^{-1}$)")
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
    """Generate continuous-time methods and results without fixed interpretation."""
    summary = result["comparison_summary"]
    point = result["point"]["comparisons"].set_index("comparison_id")
    reference = point.loc["phase_reference"]
    phase = result["point"]["models"].set_index("model_id").loc["BP"]
    params = result["parameters"]
    status = result["realization_status"]
    counts = result["mc_models"].loc[
        result["mc_models"].model_id.eq("B") & result["mc_models"].fit_valid, "n_events"]
    count_description = "; ".join(f"{int(count)} events in {int(number)} draws"
                                  for count, number in counts.value_counts().sort_index().items())
    lines = [f"{catalogue_label}: orbital-driver sensitivity", "", "Methods",
        "We fit the same continuous-time conditional point process as the main analysis. "
        "The baseline (BG) includes an intercept, exponential prior-event history, LR04 and CO2; "
        "NGRIP–MIS6 additionally includes a segment intercept contrast. Pre adds sine and cosine "
        "of precession phase. The history coefficient is constrained to be nonpositive. "
        f"Its decay time is {params['history_tau_kyr']:g} kyr, with unobserved prehistory set to zero.",
        f"There are {int(reference.n_events)} point-age response events and "
        f"{reference.exposure_kyr:.6f} kyr of exposure. Each segment conditions on its exact oldest "
        "event, which initializes subsequent history but contributes no response event term. "
        "The younger event-free tail remains exposed. BP ages decrease in forward process time.",
        "The continuous log likelihood is the sum of event log intensities minus the time integral "
        "of conditional intensity. Event ages are never rounded to a grid. The integral uses positive "
        "Gauss–Legendre weights between actual events and all forcing/phase interpolation knots. "
        "Integral nodes are numerical evaluation locations, not observations or sample size.",
        "Three scalar orbital variables (Orb) are considered: eccentricity, obliquity and 65°N "
        "summer-solstice daily-mean top-of-atmosphere insolation (solar longitude 90°, S0=1365 W m-2). "
        "Eight models are fitted: BG, BG+Pre, and BG+Orb and BG+Pre+Orb for each scalar. "
        "No lag search, quadratic driver or phase-amplitude interaction is included. Eccentricity "
        "describes a climatic-precession amplitude envelope, but its additive term does not make "
        "the phase rate ratio depend on eccentricity.",
        "La2004 signed source ages are converted to BP1950 by -source_time-0.05 kyr. The insolation "
        "NetCDF age is shifted by -0.05 kyr; its J2000 origin is inferred from numerical agreement "
        "with the source solution. See docs/orbital_driver_inputs.md. Obliquity is converted from "
        "radians to degrees. Native source knots are retained; interpolation does not increase "
        "the source temporal resolution. Phase zero is at precession-index minima and 180° at "
        "maxima, with unwrapped phase increasing toward older BP ages before sine/cosine evaluation.",
        "All forcing values are centered by their exposure-time mean and divided by their true "
        "piecewise-linear interpolant range over the catalogue's nominal continuous response support. "
        "Event points and integral nodes use the same interpolant and scaling constants. Nominal "
        "scales remain fixed in chronology refits. Predictor correlations use quadrature time weights, "
        "so irregular integration-node density is not treated as exposure.",
        f"The {params['n_realizations']} saved chronology rows are reused in their original order, "
        "without resampling IDs or ages. Their initial selection seed was 20260909. Each row has "
        "its own exact conditioning event and exposure; all eight models share that row's support, "
        "response events and history. Unsupported draws retain their IDs and are not clipped or replaced. "
        "Chronology quantiles describe age sensitivity, not process-sampling confidence intervals.",
        "Nested gain G=(logL_full-logL_reduced)/(N_response ln2) is in bits/event. Scalar additions "
        "have one parameter and phase additions two. Chi-square p values are nominal. Holm correction "
        "uses the nine new comparisons within each catalogue; the original phase reference is outside "
        "the family. These extra comparisons have no new null-bootstrap calibration. AIC=2k-2logL and "
        "delta AIC=2*delta k-LR; no node-count AICc or BIC is used. Usual AIC penalties are approximate "
        "when the history coefficient lies on its constraint boundary.", "", "Results",
        f"Main phase reference: G={reference.gain_bits_per_event:.8f} bits/event, "
        f"LR={reference.LR_statistic:.8f}, nominal p={reference.nominal_p:.8g}; "
        f"preferred phase={phase.pre_phase_preferred_deg:.4f} degrees, "
        f"phase max/min rate ratio={phase.pre_phase_rate_ratio_max_vs_min:.6f}.",
        f"Of {len(status)} selected chronologies, {int(status.within_observation_support.sum())} are "
        f"within observation support and {int(status.all_comparisons_valid.sum())} have all ten "
        "comparisons valid. Each comparison reports its own valid denominator.",
        f"Response-event counts among valid baseline fits: {count_description}.", "",
        "Comparison | G point | MC median [2.5%,97.5%] | df | nominal p | Holm nominal p | Delta AIC | valid MC"]
    for row in summary.itertuples(index=False):
        original = point.loc[row.comparison_id]
        lines.append(f"{row.comparison_id} | {row.gain_bits_per_event_point:.6f} | "
            f"{row.gain_bits_per_event_median:.6f} "
            f"[{row.gain_bits_per_event_q025:.6f}, {row.gain_bits_per_event_q975:.6f}] | "
            f"{int(original.df)} | {original.nominal_p:.6g} | {original.holm_nominal_p:.6g} | "
            f"{original.delta_AIC:.6f} | {int(row.n_mc_valid)}/{int(row.n_mc_total)}")
    best = result["point"]["models"].loc[result["point"]["models"].fit_valid].sort_values("AIC").iloc[0]
    lines += ["", f"Lowest point-age AIC among the eight candidates: {best.model_id} (AIC={best.AIC:.8f}).",
        "", "Interpretation and limitations",
        "These are conditional associations given LR04, CO2 and history. An orbital driver may "
        "share information with precession or act through background climate, so a small incremental "
        "gain does not exclude indirect effects. Nonnegative gain is expected by model nesting and "
        "does not itself establish significance. Age-range endpoints are not sampling confidence limits. "
        "No combined p value is inferred across these related records. Barker SpeleoAge remains treated "
        "as BP1950 although the column reference year has not been independently verified.", "", "References",
        "Laskar et al. (2004), A&A 428, 261–285, doi:10.1051/0004-6361:20041335.",
        "Event and chronology provenance follows the main analyses; input_code_sha256.csv records sources."]
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
                "weights decaying exponentially with a 1.5-kyr time constant, the LR04 benthic oxygen-isotope stack, "
                "and atmospheric CO2. It is fitted as a continuous-time conditional point process; "
                "the history coefficient is constrained to be nonpositive.")
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
        "G is the log-likelihood gain per response event, expressed in bits/event. Filled circles "
        "show point-age estimates; vertical ticks show Monte Carlo (MC) medians, and "
        "horizontal bars show the 2.5th–97.5th percentile range across "
        f"{used} valid chronologies from {selected} selected realizations. These ranges "
        "describe age sensitivity, not complete confidence intervals. Dashed lines in "
        "(a,c) show the point-age G for BG + Pre versus BG; they are references, not "
        "significance thresholds. All models use identical exact response events and observation "
        "exposure within each realization.\n")
    note_dir.mkdir(parents=True, exist_ok=True)
    (note_dir / f"{run_name}_Caption.txt").write_text(caption)


def weighted_correlation(frame, weights):
    """Exposure-time correlation, not correlation of the irregular node counts."""
    values = frame.to_numpy(float)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(frame),) or np.any(weights <= 0) or not np.isfinite(weights).all():
        raise ValueError("Correlation requires finite positive integration weights")
    weights = weights / weights.sum()
    centered = values - np.sum(values * weights[:, None], axis=0)
    covariance = centered.T @ (centered * weights[:, None])
    scale = np.sqrt(np.diag(covariance))
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = covariance / np.outer(scale, scale)
    return pd.DataFrame(correlation, index=frame.columns, columns=frame.columns)


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
    event_inputs = pd.concat([result["response"].reset_index(drop=True),
                              rates[rate_columns].reset_index(drop=True)], axis=1)
    event_inputs.to_csv(data_dir / "event_inputs_and_fitted_rates.csv", index=False)
    for key in ("comparison_summary", "phase_summary", "mc_models", "mc_comparisons",
                "selected_realizations", "realization_status", "reference_check",
                "scaling", "provenance", "events", "support"):
        result[key].to_csv(data_dir / f"{key}.csv", index=False)
    quadrature = result["integration"]
    predictors = [*result["parameters"]["full_terms"].split(";"), "ecc_scaled", "obl_scaled", "insol65n_scaled"]
    predictors = [term for term in predictors if term in quadrature]
    weighted_correlation(quadrature[predictors], quadrature.weight.to_numpy(float)).to_csv(
        data_dir / "predictor_correlation.csv")
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
