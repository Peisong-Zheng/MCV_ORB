#!/usr/bin/env python3
"""Continuous-time orbital-driver sensitivity on the primary NGRIP_MIS6 catalogue.

Reuse the frozen 500 chronology IDs and exact ages. Each chronology supplies
its own conditioning anchor; all candidate models share that exact support.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from toolbox import event_model, orbital_driver_sensitivity as orbital
from toolbox.plotting import plot_orbital_comparisons
from toolbox.project_config import (
    PROJECT_ROOT, LR04_CSV, CO2_CSV, ORBITAL_CSV, INSOLATION_65N_CSV,
    PRECESSION_PHASE_CSV, EVENT_CATALOGUE_CSV, OBSERVATION_SEGMENTS_CSV,
    ORBITAL_REFERENCE, generated_notes_dir,
)

ROOT = PROJECT_ROOT
RUN_NAME = "NGRIP_MIS6_orbital_driver_sensitivity"
CATALOGUE_LABEL = "NGRIP–MIS6 warming events"
N_REALIZATIONS = 500
OUTPUT_ROOT = ROOT
QUADRATURE_ORDER = 4
REDRAW = False
AGE_INPUT = ROOT / "data/processed" / RUN_NAME / "selected_realizations.csv"


def run_analysis(n_realizations=N_REALIZATIONS, show_progress=True, quadrature_order=4):
    events = pd.read_csv(EVENT_CATALOGUE_CSV)
    observations = pd.read_csv(OBSERVATION_SEGMENTS_CSV)
    lr04 = pd.read_csv(LR04_CSV, float_precision="round_trip")
    co2 = pd.read_csv(CO2_CSV, float_precision="round_trip")
    orbital_data = pd.read_csv(ORBITAL_CSV, float_precision="round_trip")
    insolation = pd.read_csv(INSOLATION_65N_CSV, float_precision="round_trip")
    anchors = pd.read_csv(PRECESSION_PHASE_CSV, float_precision="round_trip")
    forcings = {
        "lr04": (lr04.age_kyr_bp.to_numpy(), lr04.lr04.to_numpy()),
        "co2": (co2.age_kyr_bp.to_numpy(), co2.co2_ppm.to_numpy()),
        "precession_index": (orbital_data.age_kyr_bp.to_numpy(), orbital_data.precession_index.to_numpy()),
        "ecc": (orbital_data.age_kyr_bp.to_numpy(), orbital_data.eccentricity.to_numpy()),
        "obl": (orbital_data.age_kyr_bp.to_numpy(), orbital_data.obliquity_deg.to_numpy()),
        "insol65n": (insolation.age_kyr_bp.to_numpy(), insolation.insolation_Wm2.to_numpy()),
    }
    phase_anchors = (anchors.age_kyr_bp.to_numpy(), anchors.phase_unwrapped_rad.to_numpy())
    windows = event_model.response_windows(events, observations)
    scaling = event_model.scale_forcing_v2(
        {name: source for name, source in forcings.items() if name != "precession_index"}, windows)
    baseline = ("intercept", "same_type_exponential_history", "lr04_scaled", "co2_scaled", "mis6_segment")
    age_columns = [f"age_kyr_bp__{event_id}" for event_id in events.event_id]
    selected = pd.read_csv(AGE_INPUT, float_precision="round_trip")
    if selected.realization_id.isna().any() or not selected.realization_id.is_unique:
        raise ValueError("Saved chronology IDs must be present and unique")
    if not 1 <= n_realizations <= len(selected):
        raise ValueError("Requested chronology subset exceeds the saved selection")
    selected = selected.iloc[:n_realizations].reset_index(drop=True)
    ages = selected[age_columns].to_numpy(float)
    if not np.isfinite(ages).all() or np.any(np.diff(ages, axis=1) <= 0):
        raise ValueError("Chronologies must have finite, ordered event ages")
    result = orbital.analyze_chronologies(
        events, windows, forcings, phase_anchors, scaling, baseline, selected, age_columns,
        quadrature_order=quadrature_order, show_progress=show_progress)
    scale_rows, source_rows = [], []
    for driver, path, units in (("ecc", ORBITAL_CSV, "dimensionless"),
                                ("obl", ORBITAL_CSV, "degrees"),
                                ("insol65n", INSOLATION_65N_CSV, "W m-2")):
        scale = scaling.loc[driver]
        age, values = forcings[driver]
        scale_rows.append(dict(driver_id=driver, raw_column=driver, scaled_column=driver + "_scaled",
            units=units, response_mean=scale["mean"], response_min=scale["min"],
            response_max=scale["max"], response_range=scale["range"],
            scaling="(value - nominal exposure-time mean) / nominal interpolant range"))
        source_rows.append(dict(driver_id=driver, source_file=str(path), reference=ORBITAL_REFERENCE,
            source_units=units, analysis_units=units, source_epoch="BP1950", analysis_epoch="BP1950",
            age_offset_kyr=0., source_spacing_kyr=float(np.median(np.diff(age))),
            source_age_min_BP1950_kyr=float(age[0]), source_age_max_BP1950_kyr=float(age[-1]),
            interpolation="linear; no extrapolation; native resolution unchanged"))
    result.update(scaling=pd.DataFrame(scale_rows), provenance=pd.DataFrame(source_rows), parameters=dict(
        catalogue=CATALOGUE_LABEL, method="continuous conditional point process",
        n_realizations=len(selected), selection_seed=20260909,
        history_tau_kyr=1.5, initial_history=0.0,
        history_coefficient_domain="beta_H <= 0", quadrature_order=quadrature_order,
        response_exposure_kyr=result["point_models"].iloc[0].exposure_kyr,
        full_terms=";".join(baseline[1:] + orbital.PHASE_TERMS), age_epoch="BP1950",
        source_realizations=str(AGE_INPUT.relative_to(PROJECT_ROOT)),
        selection="saved orbital-experiment rows, original order and exact ages; no new draw",
        n_new_comparisons=9, p_method="nominal chi-square; within-catalogue Holm family of 9",
        interval="2.5–97.5% chronology sensitivity; not full sampling confidence"))
    return result


def write_notes(result, note_dir, run_name, catalogue_label):
    """Generate continuous-time methods and results without fixed interpretation."""
    summary = result["comparison_summary"]
    point = result["point_comparisons"].set_index("comparison_id")
    reference = point.loc["phase_reference"]
    phase = result["point_models"].set_index("model_id").loc["BP"]
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
    best = result["point_models"].loc[result["point_models"].fit_valid].sort_values("AIC").iloc[0]
    interpretation = ["", f"Lowest point-age AIC among the eight candidates: {best.model_id} (AIC={best.AIC:.8f}).",
        "", "Interpretation and limitations",
        "These are conditional associations given LR04, CO2 and history. An orbital driver may "
        "share information with precession or act through background climate, so a small incremental "
        "gain does not exclude indirect effects. Nonnegative gain is expected by model nesting and "
        "does not itself establish significance. Age-range endpoints are not sampling confidence limits. "
        "No combined p value is inferred across these related records. Barker SpeleoAge remains treated "
        "as BP1950 although the column reference year has not been independently verified.", "", "References",
        "Laskar et al. (2004), A&A 428, 261–285, doi:10.1051/0004-6361:20041335.",
        "Event and chronology provenance follows the main analyses; input_code_sha256.csv records sources."]
    lines = lines + interpretation
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
        baseline = baseline + " Separate baseline rates are fitted for NGRIP and MIS 6."
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


def save_results(result, output_root, input_paths=()):
    """Write inspectable inputs, all fits, denominator audits and paper artifacts."""
    run_name, catalogue_label = RUN_NAME, CATALOGUE_LABEL
    data_dir = output_root / "data/processed" / run_name
    figure_dir = output_root / "figures" / run_name
    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    for key in ("models", "comparisons"):
        result["point_" + key].to_csv(data_dir / f"point_{key}.csv", index=False)
    coefficient_rows = []
    event_inputs = result["event_features"].copy()
    # Preserve the established root table's column order; matrices used above are explicit.
    event_ids = event_inputs.pop("event_id")
    indicator = event_inputs.pop("mis6_segment")
    event_inputs["mis6_segment"] = indicator
    event_inputs["event_id"] = event_ids
    for model_id, fitted in result["point_fits"].items():
        coefficient_rows.extend(dict(model_id=model_id, term=term, beta=value,
            fit_valid=True, invalid_reason="") for term, value in zip(fitted.terms, fitted.beta))
        terms = list(fitted.terms)
        event_inputs[model_id] = np.exp(fitted.beta[0] +
            event_inputs[terms[1:]].to_numpy(float) @ fitted.beta[1:])
    pd.DataFrame(coefficient_rows).to_csv(data_dir / "point_coefficients.csv", index=False)
    event_inputs.to_csv(data_dir / "event_inputs_and_fitted_rates.csv", index=False)
    for key in ("comparison_summary", "phase_summary", "mc_models", "mc_comparisons",
                "realization_status",
                "scaling", "provenance", "events"):
        result[key].to_csv(data_dir / f"{key}.csv", index=False)
    result["windows"].to_csv(data_dir / "support.csv", index=False)
    quadrature = result["integration_features"]
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
    write_notes(result, generated_notes_dir(output_root), run_name, catalogue_label)
    redraw_saved_results(output_root)
    return data_dir, figure_dir


def redraw_saved_results(output_root):
    """Plot saved comparisons without refitting or reading analysis inputs."""
    run_name, catalogue_label = RUN_NAME, CATALOGUE_LABEL
    data_dir = output_root / "data/processed" / run_name
    figure_dir = output_root / "figures" / run_name
    summary = pd.read_csv(data_dir / "comparison_summary.csv")
    fig, _ = plot_orbital_comparisons(summary, catalogue_label)
    figure_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(figure_dir / f"{run_name}.{extension}", dpi=600)
    plt.close(fig)
    return figure_dir


def main():
    if REDRAW:
        print(redraw_saved_results(OUTPUT_ROOT))
        return
    result = run_analysis(N_REALIZATIONS, quadrature_order=QUADRATURE_ORDER)
    inputs = [Path(__file__).resolve(), Path(orbital.__file__), PROJECT_ROOT / "toolbox/plotting.py",
        PROJECT_ROOT / "toolbox/model_stats.py", PROJECT_ROOT / "toolbox/point_process.py",
        PROJECT_ROOT / "toolbox/project_config.py", PROJECT_ROOT / "toolbox/event_model.py",
        EVENT_CATALOGUE_CSV, OBSERVATION_SEGMENTS_CSV,
        AGE_INPUT, LR04_CSV, CO2_CSV, ORBITAL_CSV, PRECESSION_PHASE_CSV, INSOLATION_65N_CSV]
    data_dir, _ = save_results(result, OUTPUT_ROOT, inputs)
    print(result["comparison_summary"].to_string(index=False))
    print(f"Saved {data_dir}")


if __name__ == "__main__":
    main()
