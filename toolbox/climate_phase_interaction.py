"""Does the precession-phase response vary with the LR04 background?

The existing full model is nested in a model with two LR04-by-phase terms.
Response support and climate scaling are supplied by each main analysis.
"""

from pathlib import Path
from dataclasses import replace
import hashlib
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from toolbox.phase_response_plotting import format_phase_response_axis, mark_preferred_phase
from toolbox.figure_style import add_panel_label
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import chi2

from toolbox import combined_likelihood
from toolbox import orbital_driver_sensitivity as orbital

PHASE_TERMS = ("pre_phase_sin", "pre_phase_cos")
INTERACTION_TERMS = ("lr04_phase_sin", "lr04_phase_cos")
METRICS = ("gain_bits_per_event", "LR_statistic", "nominal_p", "delta_AIC")
PHASE_METRICS = ("phase_amplitude", "preferred_phase_deg", "rate_ratio_max_vs_min")
PHASE_GRID = np.linspace(0, 360, 361)
COLORS = ("#D55E00", "#666666", "#0072B2")


def add_interactions(context):
    """Evaluate LR04-by-phase products at exact event and integration ages."""
    derived = dict(context.derived_terms or {})
    for phase, interaction in zip(PHASE_TERMS, INTERACTION_TERMS):
        derived[interaction] = ("lr04_scaled", phase)
    return replace(context, derived_terms=derived)


def background_quantiles(frame):
    """Nominal LR04 time percentiles from positive quadrature weights.

    The midpoint cumulative-weight approximation is independent of event counts.
    All reported backgrounds retain the fixed nominal time-weighted climate scale.
    """
    duration = frame.weight.to_numpy(float)
    raw = frame.lr04.to_numpy(float)
    if not np.isfinite(raw).all() or not np.isfinite(duration).all() or np.any(duration <= 0):
        raise ValueError("LR04 and exposure must be finite, with positive durations")
    order = np.argsort(raw, kind="stable")
    weights = duration[order]
    cumulative = (np.cumsum(weights) - 0.5 * weights) / weights.sum()
    quantiles = np.array([0.25, 0.5, 0.75])
    raw_quantiles = np.interp(quantiles, cumulative, raw[order])
    # Interpolation preserves the exact source scale; no MC-specific rescaling.
    scaled_quantiles = np.interp(raw_quantiles, raw[order], frame.lr04_scaled.to_numpy(float)[order])
    return pd.DataFrame({"background_id": ("q25", "q50", "q75"),
                         "exposure_quantile": quantiles, "lr04_permil": raw_quantiles,
                         "lr04_scaled": scaled_quantiles})


def fit_models(design, full_terms):
    """Compare the current continuous full model with two LR04 modulation terms."""
    full_terms = tuple(full_terms)
    if not {"lr04_scaled", *PHASE_TERMS}.issubset(full_terms):
        raise ValueError("The reference model must retain LR04 and both phase main effects")
    if set(full_terms) & set(INTERACTION_TERMS) or len(set(full_terms)) != len(full_terms):
        raise ValueError("Reference terms must be unique and exclude the interaction")
    result = orbital.fit_named_models(design,
        {"full": full_terms, "interaction": full_terms + INTERACTION_TERMS})
    reference, extended = result["models"].to_dict("records")
    gain = extended["log_likelihood"] - reference["log_likelihood"]
    nesting_ok = np.isfinite(gain) and gain >= -1e-7
    reasons = [f"{row['model_id']}: {row['invalid_reason']}" for row in (reference, extended)
               if not row["fit_valid"]]
    if not nesting_ok:
        reasons.append("interaction likelihood below reference likelihood")
    lr = 2 * max(gain, 0) if not reasons else np.nan
    count = len(design.event_frame)
    comparison = dict(comparison_id="lr04_phase_interaction", df=2, LR_statistic=lr,
        gain_bits_per_event=lr / (2 * count * np.log(2)),
        nominal_p=float(chi2.sf(lr, 2)), delta_AIC=extended["AIC"] - reference["AIC"],
        loglik_full=reference["log_likelihood"], loglik_interaction=extended["log_likelihood"],
        fit_valid=not reasons, invalid_reason="; ".join(reasons),
        likelihood_nesting_ok=bool(nesting_ok), n_events=count, exposure_kyr=float(design.weights.sum()))
    return dict(models=result["models"], coefficients=result["coefficients"], comparison=comparison)


def conditional_phase(coefficients, backgrounds):
    """Phase-only multiplier relative to the same model with its phase term zero.

    At background z, the phase coefficients are b_sin + z*g_sin and
    b_cos + z*g_cos. The peak phase can change together with the amplitude.
    """
    beta = coefficients.loc[coefficients.model_id.eq("interaction")].set_index("term").beta
    phase = np.deg2rad(PHASE_GRID)
    estimates, curves = [], []
    for row in backgrounds.itertuples(index=False):
        b_sin = beta.pre_phase_sin + row.lr04_scaled * beta.lr04_phase_sin
        b_cos = beta.pre_phase_cos + row.lr04_scaled * beta.lr04_phase_cos
        amplitude = np.hypot(b_sin, b_cos)
        preferred = np.degrees(np.arctan2(b_sin, b_cos)) % 360 if amplitude > 1e-12 else np.nan
        estimates.append(dict(background_id=row.background_id, beta_sin=b_sin, beta_cos=b_cos,
            phase_amplitude=amplitude, preferred_phase_deg=preferred,
            rate_ratio_max_vs_min=np.exp(2 * amplitude)))
        curves.append(pd.DataFrame(dict(background_id=row.background_id, phase_deg=PHASE_GRID,
            rate_multiplier=np.exp(b_sin * np.sin(phase) + b_cos * np.cos(phase)))))
    return pd.DataFrame(estimates), pd.concat(curves, ignore_index=True)


def analyze_realizations(context, full_terms, selected, age_columns, show_progress=True):
    """Reuse selected chronology rows; keep unsupported draws in the status table.

    Each chronology defines its exact conditional anchor and response exposure.
    A support error is recorded without clipping or replacing the source row.
    """
    if (selected.empty or selected.realization_id.isna().any()
            or selected.realization_id.duplicated().any()):
        raise ValueError("Selected chronology IDs must be nonempty and unique")
    age_matrix = selected.loc[:, age_columns].to_numpy(float)
    if not np.isfinite(age_matrix).all() or np.any(np.diff(age_matrix, axis=1) <= 0):
        raise ValueError("Selected chronologies must have finite, strictly ordered events")
    point_design = combined_likelihood.prepare_catalogue(context.events, context)
    backgrounds = background_quantiles(point_design.integration_frame)
    point = fit_models(point_design, full_terms)
    if not point["comparison"]["fit_valid"]:
        raise RuntimeError(f"Invalid point-age comparison: {point['comparison']}")
    point_phase, point_curves = conditional_phase(point["coefficients"], backgrounds)
    comparisons, models, coefficients, phases, curves = [], [], [], [], []
    started = time.perf_counter()
    for index, (realization_id, ages) in enumerate(zip(selected.realization_id, age_matrix)):
        shifted = context.events.copy()
        shifted[combined_likelihood.EVENT_AGE_COLUMN] = ages
        if "event_age_ka" in shifted:
            shifted["event_age_ka"] = ages
        try:
            design = combined_likelihood.prepare_catalogue(shifted, context)
        except ValueError as error:
            comparisons.append(dict(realization_id=realization_id, fit_valid=False,
                                    comparison_id=point["comparison"]["comparison_id"],
                                    df=point["comparison"]["df"],
                                    invalid_reason=str(error), within_observation_support=False))
            continue
        fitted = fit_models(design, full_terms)
        comparisons.append(dict(realization_id=realization_id, within_observation_support=True,
                                **fitted["comparison"]))
        models.append(fitted["models"].assign(realization_id=realization_id))
        coefficients.append(fitted["coefficients"].assign(realization_id=realization_id))
        if fitted["comparison"]["fit_valid"]:
            phase, curve = conditional_phase(fitted["coefficients"], backgrounds)
            phases.append(phase.assign(realization_id=realization_id))
            curves.append(curve.rate_multiplier.to_numpy())
        if show_progress and (index + 1) % 100 == 0:
            print(f"LR04 interaction: {index + 1}/{len(selected)} chronologies "
                  f"({time.perf_counter() - started:.0f} s)", flush=True)
    mc_comparisons = pd.DataFrame(comparisons)
    valid = mc_comparisons.loc[mc_comparisons.fit_valid]
    if valid.empty:
        raise RuntimeError("No valid chronology fits")
    summary = dict(n_mc_total=len(selected), n_mc_valid=len(valid),
                   n_mc_invalid=len(selected) - len(valid), **point["comparison"])
    for metric in METRICS:
        summary[f"{metric}_point"] = summary.pop(metric)
        for label, value in zip(("q025", "median", "q975"), np.quantile(valid[metric], [0.025, 0.5, 0.975])):
            summary[f"{metric}_{label}"] = value
    summary["mc_nominal_p_below_0p05_fraction"] = float((valid.nominal_p < 0.05).mean())
    mc_phase = pd.concat(phases, ignore_index=True)
    phase_summary = backgrounds.merge(point_phase, on="background_id")
    for metric in PHASE_METRICS:
        phase_summary = phase_summary.rename(columns={metric: f"{metric}_point"})
    for index, row in phase_summary.iterrows():
        sample = mc_phase.loc[mc_phase.background_id.eq(row.background_id)]
        for metric in PHASE_METRICS:
            values = sample[metric].to_numpy(float)
            if metric == "preferred_phase_deg":
                center = row.preferred_phase_deg_point
                values = center + (values - center + 180) % 360 - 180
            quantiles = np.nanquantile(values, [0.025, 0.5, 0.975])
            for label, value in zip(("q025", "median", "q975"), quantiles):
                phase_summary.loc[index, f"{metric}_{label}"] = value
        phase_summary.loc[index, "phase_mc_resultant_length"] = abs(np.mean(np.exp(
            1j * np.deg2rad(sample.preferred_phase_deg))))
    curve_summary = point_curves.rename(columns={"rate_multiplier": "rate_multiplier_point"})
    for label, values in zip(("q025", "median", "q975"), np.quantile(curves, [0.025, 0.5, 0.975], axis=0)):
        curve_summary[f"rate_multiplier_{label}"] = values
    return dict(event_inputs=point_design.event_frame,
        support=combined_likelihood.support_table(point_design.context),
        forcing_scaling=combined_likelihood.scaling_table(context),
        events_used=point_design.all_events, backgrounds=backgrounds,
        point_models=point["models"], point_coefficients=point["coefficients"],
        point_comparison=pd.DataFrame([point["comparison"]]), selected_realizations=selected,
        mc_comparisons=mc_comparisons, mc_models=pd.concat(models, ignore_index=True),
        mc_coefficients=pd.concat(coefficients, ignore_index=True), mc_phase_estimates=mc_phase,
        comparison_summary=pd.DataFrame([summary]), phase_summary=phase_summary,
        phase_curves=curve_summary)


def plot_results(result, catalogue_label):
    """Matched compact figures: conditional phase curves and the interaction G."""
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8, "axes.linewidth": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(180 / 25.4, 90 / 25.4),
                             gridspec_kw={"width_ratios": [1.55, 1]})
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.38, top=0.77, wspace=0.32)
    curves = result["phase_curves"]
    for background, color in zip(result["backgrounds"].itertuples(index=False), COLORS):
        curve = curves.loc[curves.background_id.eq(background.background_id)]
        estimate = result["phase_summary"].set_index("background_id").loc[background.background_id]
        axes[0].plot(curve.phase_deg, curve.rate_multiplier_point, color=color, lw=1.4,
                     label=f"LR04 {background.exposure_quantile:.0%}: {background.lr04_permil:.2f}‰\n"
                     f"Preferred: {estimate.preferred_phase_deg_point:.1f}°\n"
                     f"Max/min: {estimate.rate_ratio_max_vs_min_point:.2f}")
        mark_preferred_phase(axes[0], estimate.preferred_phase_deg_point,
                             estimate.rate_ratio_max_vs_min_point, color)
        axes[0].fill_between(curve.phase_deg, curve.rate_multiplier_q025,
                             curve.rate_multiplier_q975, color=color, alpha=0.09, linewidth=0)
    axes[0].axhline(1, color="0.65", lw=0.7, ls="--", zorder=0)
    format_phase_response_axis(axes[0], fontsize=6.5)
    axes[0].set_title("Fitted warming-event rate", fontsize=8, pad=16)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.49), frameon=False,
                   fontsize=6.3, ncol=3, columnspacing=0.8, handlelength=1.3)
    summary = result["comparison_summary"].iloc[0]
    valid = result["mc_comparisons"].loc[result["mc_comparisons"].fit_valid, "gain_bits_per_event"]
    axes[1].hist(valid, bins=24, color="#C3CDD3", edgecolor="white", linewidth=0.3)
    axes[1].axvspan(summary.gain_bits_per_event_q025, summary.gain_bits_per_event_q975,
                    color="#0072B2", alpha=0.06, zorder=0)
    axes[1].axvline(summary.gain_bits_per_event_point, color="#222222", lw=1.3)
    axes[1].axvline(summary.gain_bits_per_event_median, color="#0072B2", lw=1.2, ls="--")
    axes[1].set(xlabel="Interaction gain (bits event$^{-1}$)", ylabel="MC realizations", xlim=(0, None))
    axes[1].legend(handles=[Line2D([], [], color="#222222", lw=1.3, label="Point ages"),
        Line2D([], [], color="#0072B2", lw=1.2, ls="--", label="MC median")],
        loc="upper center", bbox_to_anchor=(0.5, -0.34), frameon=False, fontsize=7, ncol=2)
    # The Barker panels follow NGRIP–MIS6 in the combined manuscript figure.
    panel_labels = "cd" if catalogue_label.startswith("Barker") else "ab"
    for label, ax in zip(panel_labels, axes):
        add_panel_label(ax, label, x=0, y=1.08)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(direction="out", length=3, width=0.7)
    fig.text(0.09, 0.94, catalogue_label, fontsize=10, weight="bold", va="top")
    return fig, axes


def save_figure(result, output_root, run_name, catalogue_label, *, paper_export=True):
    """Save this catalogue's panels and refresh the combined manuscript figure."""
    output_root = Path(output_root)
    fig_dir = output_root / "figures" / run_name
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, _ = plot_results(result, catalogue_label)
    fig.savefig(fig_dir / f"{run_name}.png", dpi=600)
    fig.savefig(fig_dir / f"{run_name}.pdf")
    plt.close(fig)
    if not paper_export:
        return
    from Figure_climate_phase_interaction import build_figure, SOURCES

    project_root = Path(__file__).resolve().parents[1]
    # Diagnostic exports to another directory do not refresh the manuscript.
    source_pdf = (fig_dir / f"{run_name}.pdf").resolve()
    if source_pdf not in [(project_root / source).resolve() for source in SOURCES]:
        return
    missing = [source for source in SOURCES if not (project_root / source).is_file()]
    if missing:
        print("Combined interaction figure awaits: " + ", ".join(missing))
    else:
        build_figure(project_root)


def save_results(result, output_root, run_name, catalogue_label, parameters, inputs, *, paper_export=True):
    """Save numerical evidence, publication figures and self-contained English notes."""
    data_dir = Path(output_root) / "data/processed" / run_name
    fig_dir = Path(output_root) / "figures" / run_name
    note_dir = Path(output_root) / "experiment_note"
    for directory in (data_dir, fig_dir, note_dir):
        directory.mkdir(parents=True, exist_ok=True)
    for name, table in result.items():
        table.to_csv(data_dir / f"{name}.csv", index=False)
    pd.DataFrame([{"parameter": name, "value": value} for name, value in parameters.items()]).to_csv(
        data_dir / "parameters.csv", index=False)
    pd.DataFrame([{"path": str(path), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
                  for path in inputs]).to_csv(data_dir / "input_code_sha256.csv", index=False)
    save_figure(result, output_root, run_name, catalogue_label, paper_export=paper_export)
    write_notes(result, note_dir, run_name, catalogue_label, parameters)
    return data_dir


def write_notes(result, note_dir, run_name, catalogue_label, parameters):
    s = result["comparison_summary"].iloc[0]
    pooled = "NGRIP" in run_name
    baseline = ("an intercept, a MIS 6 segment indicator, recent warming-event history, LR04 and CO2"
                if pooled else "an intercept, recent warming-event history, LR04 and CO2")
    source = ("the primary NGRIP warming plus MIS 6 speleothem event catalogue" if pooled else
              "the primary 70-event variable-threshold Barker Table S3 SpeleoAge catalogue")
    rows = []
    for p in result["phase_summary"].itertuples(index=False):
        rows.append(f"LR04 {p.exposure_quantile:.0%} ({p.lr04_permil:.6f} permil; "
            f"scaled {p.lr04_scaled:.6f}): preferred phase {p.preferred_phase_deg_point:.3f} degrees "
            f"[age MC {p.preferred_phase_deg_q025:.3f}, {p.preferred_phase_deg_q975:.3f}]; "
            f"maximum/minimum phase rate ratio {p.rate_ratio_max_vs_min_point:.4f} "
            f"[age MC {p.rate_ratio_max_vs_min_q025:.4f}, {p.rate_ratio_max_vs_min_q975:.4f}].")
    invalid = result["mc_comparisons"].loc[~result["mc_comparisons"].fit_valid,
                                         ["realization_id", "invalid_reason"]]
    text = f"""LR04 modulation of the precession-phase response: {catalogue_label}

METHODS
Use {source}. The model describes warming-event occurrence over the full
observation interval, using a continuous-time conditional point process.
Each segment conditions on its exact oldest event and retains exposure through
the younger observation boundary. The anchor initializes subsequent history
but is excluded from response-event terms. This is not a cold-state triggering model. The existing full model includes {baseline},
and the sine and cosine of precession phase. Recent history is the number of
exponentially weighted older warming events (decay time {parameters['history_tau_kyr']} kyr),
using the main analysis's oldest-to-youngest time convention. The history
coefficient is constrained to be nonpositive, and unknown pre-anchor history
is set to zero. The point-age response contains {int(s.n_events)} events
with {s.exposure_kyr:.6f} kyr of exposure. {parameters['age_epoch']}.

The continuous log likelihood is the sum of exact-event log intensities minus
the time integral of intensity. Positive Gauss–Legendre quadrature partitions
at actual events and forcing/phase interpolation knots. Node counts do not
represent observations or statistical sample size.

Let z be LR04 centered with the main analysis's nominal exposure-time mean
and divided by its piecewise-linear interpolant range over the same support. Compare the current full model to the same model plus
g_sin*z*sin(phase) + g_cos*z*cos(phase). Both main effects remain present.
The null is g_sin = g_cos = 0 (two added parameters); no LR04 threshold or
alternative climate modulator is searched. The additional log-likelihood gain per response event (G) equals (logL_interaction - logL_full)/(N_events*ln(2)),
in bits per event. Likelihood-ratio p values use a nominal chi-square
distribution with two degrees of freedom. They are not event-process bootstrap
calibrated; the separate reduced-model bootstrap tests a different hypothesis.
AIC=2k-2logL. Delta AIC is interaction minus full, equal to 4-LR;
negative values favor the interaction model. No node-count finite-sample
correction is applied. Usual AIC penalties are approximate at parameter-domain
boundaries. Both nested models use the same history constraint and core
convergence, identifiability and projected-gradient checks.

At a fixed background z, the conditional phase-only multiplier is
exp((b_sin + z*g_sin)*sin(phase) + (b_cos + z*g_cos)*cos(phase)).
It is relative to the same background and event history with the phase term
set to zero, not to the arithmetic phase-averaged rate. Its maximum/minimum
ratio is exp(2*hypot(b_sin + z*g_sin, b_cos + z*g_cos)); preferred phase is
atan2(b_sin + z*g_sin, b_cos + z*g_cos) modulo 360 degrees. Phase is zero at
precession-index minima and 180 degrees at maxima, increasing linearly between
successive extrema toward older BP ages; phase is not elapsed lag. Display backgrounds
are the exposure-weighted LR04 25th, 50th and 75th percentiles, using linear
interpolation on midpoint cumulative quadrature time weights, fixed before chronology refits.
Higher LR04 reflects combined benthic temperature and ice-volume variation;
it does not identify a cold-state exposure interval.

Chronology sensitivity reuses the exact {int(s.n_mc_total)} selected age rows
from {parameters['source_realizations']}; selection seed was 20260909. No new
chronology draws are made. Each draw is fitted at its actual ages, recomputing
the exponential history and intensity integration. Forcing interpolants and
nominal scaling remain fixed. Each draw has its own exact oldest-event anchor
and response exposure. Unsupported draws
are recorded, not clipped or replaced. Both models use identical response
events within each draw. Only pairs of valid nested fits contribute to the
MC ranges. MC is Monte Carlo. Ranges are 2.5th-97.5th percentiles across valid
chronologies, not full sampling confidence intervals. Curve bands are pointwise
ranges and are not simultaneous bands. Phase quantiles are unwrapped within
180 degrees of the corresponding point-age phase; they can extend outside
0-360 degrees. The saved circular resultant length describes phase concentration.

RESULTS
Point-age LR = {s.LR_statistic_point:.8f}; nominal p = {s.nominal_p_point:.8g};
additional G = {s.gain_bits_per_event_point:.8f} bits per event;
delta AIC = {s.delta_AIC_point:.8f}.
Valid chronology comparisons: {int(s.n_mc_valid)}/{int(s.n_mc_total)}.
Additional G MC median = {s.gain_bits_per_event_median:.8f},
95% age range [{s.gain_bits_per_event_q025:.8f}, {s.gain_bits_per_event_q975:.8f}].
Nominal p MC median = {s.nominal_p_median:.8g},
95% age range [{s.nominal_p_q025:.8g}, {s.nominal_p_q975:.8g}].
Fraction of valid chronology fits with nominal p < 0.05 =
{s.mc_nominal_p_below_0p05_fraction:.6f}; this is a robustness fraction, not a p value.
{chr(10).join(rows)}
Invalid chronology rows:
{invalid.to_string(index=False) if len(invalid) else 'None.'}

INTERPRETATION
This preselected test concerns conditional modification of the fitted phase
response by LR04. Both amplitude and peak phase may change; this is not an
eccentricity-amplitude experiment. Associations are conditional on the other
baseline terms, including CO2 and history, and do not establish causation or
exclude indirect orbital effects absorbed by background climate. A weak
interaction result means the data do not resolve the extra response variation
under this model; it does not establish climate invariance. The modest event
counts, correlated forcings and broad phase/strength chronology ranges limit
mechanistic interpretation. A positive G or separated point curves alone
does not establish modulation; nominal p is not a bootstrap-calibrated probability.

OUTPUTS
event_inputs.csv, support.csv, forcing_scaling.csv and backgrounds.csv preserve
the exact-event design, continuous support, nominal scales and display values.
point_models.csv, point_coefficients.csv and point_comparison.csv preserve the
two point fits; mc_models.csv and mc_coefficients.csv preserve supported fits.
mc_comparisons.csv retains every selected ID with validity and rejection reason.
comparison_summary.csv contains point and chronology comparison summaries.
phase_summary.csv and mc_phase_estimates.csv contain phase/strength estimates;
phase_curves.csv contains the point curve, MC median and pointwise 95% range.
selected_realizations.csv preserves the exact input ages. parameters.csv and
input_code_sha256.csv record settings and source/code hashes. PNG and vector
PDF files are saved in figures/{run_name}; English methods/results and captions
are saved separately in experiment_note.
"""
    (note_dir / f"{run_name}_Methods_and_results.txt").write_text(text, encoding="utf-8")
    first_panel, second_panel = ("a", "b") if pooled else ("c", "d")
    caption = f"""LR04 modulation of the precession response in {catalogue_label}.
({first_panel}) Phase-only warming-rate multipliers from the model with LR04-by-sine and
LR04-by-cosine interactions, evaluated at the exposure-weighted 25th, 50th
and 75th percentiles of the LR04 benthic oxygen-isotope stack (legend values
in permil). Phase is zero at precession-index minima and 180 degrees at maxima,
increasing toward older BP ages. Lines use the point ages; shading gives pointwise 2.5th-97.5th
percentiles from age Monte Carlo (MC) realizations. The multiplier is relative
to the same background and history with the phase contribution set to zero;
it is not the total warming rate. The horizontal dashed line denotes unity,
not the mean rate or a separately refitted reduced-model rate. Minimum/Maximum
tick labels name precession-index extrema. Peak dots identify fitted preferred
phases; legends give point-age preferred phases and maximum/minimum rate ratios
for each LR04 condition. The smooth curves follow the sine/cosine model.
({second_panel}) Distribution of the additional log-likelihood gain per response event (G) supplied by
the two interaction terms, relative to the existing full model. The full
model includes {baseline}, plus precession-phase sine and cosine. G equals
the log-likelihood improvement divided by the response-event count and ln(2),
in bits per event. The solid vertical line gives the point-age result; the
dashed line gives the MC median, and the pale vertical band spans the central
95% of valid chronology results. {int(s.n_mc_valid)} of {int(s.n_mc_total)} selected
chronologies yielded valid comparisons. These ranges describe age sensitivity,
not complete sampling uncertainty. The two continuous models share exact
response events and exposure within each realization; their exponential-history
coefficient is constrained to be nonpositive (decay time 1.5 kyr). Point-age LR = {s.LR_statistic_point:.3f}
(two degrees of freedom; nominal chi-square p = {s.nominal_p_point:.4g}); this
interaction test has not been calibrated by a null bootstrap.
"""
    (note_dir / f"{run_name}_Caption.txt").write_text(caption, encoding="utf-8")
