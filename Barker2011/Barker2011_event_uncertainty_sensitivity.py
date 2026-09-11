#!/usr/bin/env python3
"""Refit Barker's conditional PI for saved SpeleoAge Monte Carlo realizations.

Run Barker2011_event_age_uncertainty.py first. The main analysis supplies the
event definition, observation/response grids, climate scaling and model terms.
Every realization is rebinned and its event history recalculated. Ensemble
quantiles describe chronology sensitivity, not event-sampling confidence limits.
The fraction with nominal LRT p < 0.05 is a robustness fraction, not a p-value.
"""

from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from toolbox.phase_response_plotting import format_phase_response_axis, mark_preferred_phase
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Barker2011 import Barker2011_event_phase_analysis as main_analysis
from Barker2011 import Barker2011_event_age_uncertainty as chronology
from toolbox import combined_pi

ROOT = main_analysis.ROOT
RUN_NAME = "Barker2011_event_uncertainty_sensitivity"
AGE_INPUT = chronology.OUT_DATA_DIR / "event_age_realizations.csv"
EVENT_INPUT = chronology.OUT_DATA_DIR / "event_catalogue_used.csv"
OUT_DATA_DIR = ROOT / "data/processed" / RUN_NAME
OUT_FIG_DIR = ROOT / "figures" / RUN_NAME
NOTE_DIR = ROOT / "experiment_note"
P_THRESHOLD = 0.05
METRICS = ("info_bits_per_event", "nominal_LR_p", "LR_statistic",
           "pre_phase_preferred_deg", "pre_phase_rate_ratio_max_vs_min",
           "delta_AICc_full_minus_reduced")


def prepare_context(events):
    """Read forcings once and retain exactly the nominal grid and history indices."""
    bins, scaling = main_analysis.prepare_bins(events)
    centers = bins.bin_center_ka.to_numpy(float)
    edges = np.r_[bins.bin_start_ka.to_numpy(float), bins.bin_end_ka.iloc[-1]]
    mask = bins.in_response_interval.to_numpy(bool)
    # Reuse the shared history-count implementation with the existing grid.
    segment = combined_pi.SegmentContext(
        segment_id="Barker2011", observation_start_kyr_bp=edges[0], observation_end_kyr_bp=edges[-1],
        response_start_kyr_bp=bins.loc[mask, "bin_start_ka"].min(),
        response_end_kyr_bp=bins.loc[mask, "bin_end_ka"].max(), bin_edges=edges, response_mask=mask,
        history_left=np.arange(len(centers)) + 1,
        history_right=np.searchsorted(centers, centers + main_analysis.HISTORY_WINDOW_KA, side="right"))
    return dict(bins=bins, frame=bins.loc[mask].copy(), segment=segment, scaling=scaling)


def frame_for_ages(ages, context):
    """Rebin a sequence and recalculate history, keeping exposure and forcings fixed."""
    segment = context["segment"]
    ages = np.asarray(ages, dtype=float)
    if not np.isfinite(ages).all() or np.any(np.diff(ages) <= 0):
        raise ValueError("MC event ages must be finite and strictly ordered")
    if ages[0] < segment.bin_edges[0] or ages[-1] > segment.bin_edges[-1]:
        raise ValueError("Event ages lie outside observation support")
    counts, _ = np.histogram(ages, bins=segment.bin_edges)
    if counts.sum() != len(ages):
        raise ValueError("Binning changed source event membership")
    frame = context["frame"].copy()
    frame["event_count"] = counts[segment.response_mask]
    frame["same_type_history_count"] = combined_pi.history_from_counts(counts, segment)[segment.response_mask]
    return frame


def fit_ages(ages, context):
    frame = frame_for_ages(ages, context)
    models, tests = main_analysis.fit_models(frame)
    reduced, full = models
    metrics = tests.iloc[0]
    coefficients = dict(zip(full.terms, full.beta[1:]))
    b_sin, b_cos = coefficients["pre_phase_sin"], coefficients["pre_phase_cos"]
    row = dict(n_predictive_events=int(frame.event_count.sum()),
               n_predictive_bins=len(frame), response_exposure_kyr=frame.dt_ka.sum(),
               loglik_reduced=reduced.log_likelihood, loglik_full=full.log_likelihood,
               LR_statistic=metrics.LR_statistic, nominal_LR_p=metrics.LR_p_value,
               info_bits_per_event=metrics.info_bits_per_event,
               delta_AICc_full_minus_reduced=metrics.delta_AICc_full_minus_reduced,
               beta_pre_phase_sin=b_sin, beta_pre_phase_cos=b_cos,
               pre_phase_preferred_deg=np.degrees(np.arctan2(b_sin, b_cos)) % 360,
               pre_phase_rate_ratio_max_vs_min=np.exp(2 * np.hypot(b_sin, b_cos)),
               all_models_converged=all(model.converged for model in models),
               likelihood_nesting_ok=full.log_likelihood >= reduced.log_likelihood - 1e-8,
               eta_clipping_used=any(model.n_eta_clipped_low or model.n_eta_clipped_high for model in models))
    if (not np.isfinite([row[key] for key in METRICS]).all()
            or not row["all_models_converged"] or not row["likelihood_nesting_ok"] or row["eta_clipping_used"]):
        raise RuntimeError("Inspect a non-finite, nonconverged, nonnested or clipped PI fit")
    return row


def fit_realizations(events, realizations, context, show_progress=False):
    columns = chronology.age_columns(events)
    if (realizations.empty or realizations.realization_id.isna().any()
            or realizations.realization_id.duplicated().any()
            or realizations.columns.tolist() != ["realization_id", *columns]):
        raise ValueError("Expected unique realization IDs and the exact ordered event columns")
    ages = realizations[columns].to_numpy(float)
    if not np.isfinite(ages).all() or not np.all(np.diff(ages, axis=1) > 0):
        raise ValueError("Realizations contain missing ages or crossed events")
    rows = []
    segment = context["segment"]
    started = time.perf_counter()
    for index, values in enumerate(ages):
        row = dict(realization_id=realizations.realization_id.iloc[index])
        if values[0] < segment.bin_edges[0] or values[-1] > segment.bin_edges[-1]:
            # Keep unsupported draws in the denominator audit; never clip or redraw.
            row.update(fit_valid=False, invalid_reason="outside observation support")
        else:
            row.update(fit_ages(values, context), fit_valid=True, invalid_reason="")
        rows.append(row)
        if show_progress and (index + 1) % 500 == 0:
            print(f"Fitted {index + 1:,}/{len(ages):,} chronologies "
                  f"({time.perf_counter() - started:.0f} s)", flush=True)
    return pd.DataFrame(rows)


def unwrap_phase(values, center):
    return center + (np.asarray(values) - center + 180) % 360 - 180


def build_summary(results, point):
    valid = results.loc[results.fit_valid]
    if valid.empty:
        raise RuntimeError("No supported PI fits to summarize")
    n_below = int(valid.nominal_LR_p.lt(P_THRESHOLD).sum())
    n_aicc = int(valid.delta_AICc_full_minus_reduced.lt(0).sum())
    row = dict(n_realizations=len(results), n_valid=len(valid), n_invalid=len(results) - len(valid),
               n_events=point["n_predictive_events"], robustness_denominator=len(valid),
               n_nominal_p_below_0p05=n_below, fraction_nominal_p_below_0p05=n_below / len(valid),
               n_delta_AICc_below_zero=n_aicc, fraction_delta_AICc_below_zero=n_aicc / len(valid),
               fraction_nominal_p_below_0p05_all_draws_lower_bound=n_below / len(results),
               n_realizations_with_response_event_loss=int(valid.n_predictive_events.lt(point["n_predictive_events"]).sum()),
               n_predictive_events_min=valid.n_predictive_events.min(),
               n_predictive_events_max=valid.n_predictive_events.max(),
               phase_quantiles_unwrapped_about_point=True, significance_fraction_is_empirical_p=False)
    for key in METRICS:
        values = valid[key].to_numpy(float)
        if key == "pre_phase_preferred_deg":
            values = unwrap_phase(values, point[key])
        row[f"point_{key}"] = point[key]
        for label, q in (("q025", 0.025), ("median", 0.5), ("q975", 0.975)):
            row[f"{key}_{label}"] = np.quantile(values, q)
    phase = np.deg2rad(valid.pre_phase_preferred_deg.to_numpy(float))
    row["preferred_phase_mc_resultant_length"] = abs(np.mean(np.exp(1j * phase)))
    return pd.DataFrame([row])


def phase_response_summary(results, point):
    phase_deg = np.linspace(0, 360, 721)
    radians = np.deg2rad(phase_deg)
    valid = results.loc[results.fit_valid]
    curves = np.exp(valid.beta_pre_phase_sin.to_numpy()[:, None] * np.sin(radians)
                    + valid.beta_pre_phase_cos.to_numpy()[:, None] * np.cos(radians))
    q025, median, q975 = np.quantile(curves, [0.025, 0.5, 0.975], axis=0)
    nominal = np.exp(point["beta_pre_phase_sin"] * np.sin(radians)
                     + point["beta_pre_phase_cos"] * np.cos(radians))
    return pd.DataFrame(dict(phase_deg=phase_deg, point_multiplier=nominal,
                            mc_q025=q025, mc_median=median, mc_q975=q975))


def plot_sensitivity(results, point, curves):
    """Match the pooled sensitivity panels, with an additional phase-response band."""
    main_analysis.configure_plot_style()
    valid = results.loc[results.fit_valid]
    fig, axes = plt.subplots(2, 3, figsize=(180 / 25.4, 137 / 25.4))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.16, top=0.895, hspace=0.60, wspace=0.42)
    specs = [("info_bits_per_event", "PI (bits event$^{-1}$)"),
             ("nominal_LR_p", "Nominal likelihood-ratio p"),
             ("pre_phase_preferred_deg", "Preferred phase (°)"),
             ("pre_phase_rate_ratio_max_vs_min", "Phase rate ratio\n(maximum/minimum)"),
             ("delta_AICc_full_minus_reduced", r"$\Delta$AICc (full $-$ reduced)")]
    for ax, (key, label) in zip(axes.flat, specs):
        values = valid[key].to_numpy(float)
        if key == "pre_phase_preferred_deg":
            values = unwrap_phase(values, point[key])
            ax.xaxis.set_major_locator(MultipleLocator(10))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value % 360:g}"))
        edges = 32
        if key == "nominal_LR_p":
            edges = np.geomspace(min(values.min(), point[key]) * 0.8, max(0.1, values.max() * 1.1), 33)
            ax.set_xscale("log")
            ax.axvline(P_THRESHOLD, color="0.45", lw=1, ls=":")
        if key == "delta_AICc_full_minus_reduced":
            ax.axvline(0, color="0.45", lw=1, ls=":")
        ax.hist(values, bins=edges, color=chronology.BLUE, alpha=0.82, edgecolor="none")
        ax.axvline(point[key], color="black", lw=1.2)
        ax.axvline(np.median(values), color=main_analysis.EVENT_COLOR, lw=1.2, ls="--")
        ax.set_xlabel(label)
    ax = axes[0, 1]
    fraction = valid.nominal_LR_p.lt(P_THRESHOLD).mean()
    ax.text(0.98, 1.035, f"p < 0.05: {fraction:.2%}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8)
    ax = axes[1, 2]
    ax.fill_between(curves.phase_deg, curves.mc_q025, curves.mc_q975, color=chronology.BLUE, alpha=0.20)
    ax.plot(curves.phase_deg, curves.point_multiplier, color="black", lw=1.2)
    ax.plot(curves.phase_deg, curves.mc_median, color=main_analysis.EVENT_COLOR, lw=1.2, ls="--")
    ax.axhline(1, color="0.6", lw=0.65, ls=":")
    format_phase_response_axis(ax, fontsize=6)
    mark_preferred_phase(ax, point["pre_phase_preferred_deg"], point["pre_phase_rate_ratio_max_vs_min"])
    ax.set_title("Fitted warming-event rate", fontsize=8, pad=10)
    ax.text(0.04, 0.96, "Point ages:\n"
            f"Preferred phase: {point['pre_phase_preferred_deg']:.1f}°\n"
            f"Max/min rate ratio: {point['pre_phase_rate_ratio_max_vs_min']:.2f}",
            transform=ax.transAxes, va="top", fontsize=6.3,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1))
    for label, ax in zip("abcdef", axes.flat):
        ax.grid(False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(-0.16, 1.04, label, transform=ax.transAxes, fontweight="bold", fontsize=11)
        ax.tick_params(labelsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("MC realizations")
    handles = [Line2D([], [], color="black", lw=1.2, label="Point ages"),
               Line2D([], [], color=main_analysis.EVENT_COLOR, lw=1.2, ls="--", label="MC median"),
               Line2D([], [], color="0.45", lw=1, ls=":", label="Reference threshold")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.52, 0.995))
    fig.text(0.5, 0.02, f"Barker SpeleoAge · {len(valid):,}/{len(results):,} valid fits · chronology sensitivity",
             ha="center", fontsize=8)
    return fig


def write_methods_results(summary, context, results):
    """Keep the manuscript-facing numerical account synchronized with the tables."""
    s = summary.iloc[0]
    controls = pd.read_csv(chronology.OUT_DATA_DIR / "age_control_points.csv")
    diagnostics = pd.read_csv(chronology.OUT_DATA_DIR / "parameters_and_provenance.csv").set_index("parameter").value
    segment = context["segment"]
    text = f"""Barker 2011 SpeleoAge chronology sensitivity: methods and results

METHODS
The analysis retains the 70 variable-threshold warming events in Barker et al.
(2011), Supplementary Table S3, on the published SpeleoAge axis. Events originate
from the synthetic Greenland reconstruction based on the EDC record. Their
SpeleoAge chronology is constrained by matching cold events to weak-monsoon
features in independently dated Chinese speleothem records. Table S1 supplies
60 control ages and matching, absolute speleothem dating, and combined errors.
The combined errors already include the first two components; neither component
is added a second time. EDC3 ages are retained only in the raw transcription and
are not used in the sampling coordinate or the PI analysis.

Let t_j denote a published SpeleoAge control and U_j its printed combined error.
Independent proposals delta_j ~ Uniform(-U_j, +U_j) are generated jointly and
retained only if every t_j + delta_j is strictly increasing. The printed errors
are treated as specified proposal half-widths. Their sigma/coverage convention
is not stated in Table S1; no conversion to Gaussian sigma is assumed. The
unconditioned marginal SD is U_j/sqrt(3). Order conditioning changes both the
marginal distributions and their dependence; retained offsets need not have
zero empirical means. This is not NGRIP's cumulative-Gaussian-increment model.

For an event between t_j and t_(j+1), w = (t_i-t_j)/(t_(j+1)-t_j) is computed
once from the original SpeleoAge values, and t_i* = t_i + (1-w)delta_j + w delta_(j+1).
Adjacent events share control offsets. Interpolation covers the long interval
264.24-317.70 ka without an extra internal perturbation. Eight events lie between
these controls; seven are inside the paper's approximately 265-315 ka alignment
gap. All remain in the analysis.

Four events predate the last published control at 378.15 ka. A computational
control is added at 400 ka. Linear regression of combined uncertainty against
age over the last four published controls predicts {controls.unfloored_extrapolation_ka.iloc[-1]:.6f} kyr there.
The adopted half-width is the larger of that extrapolation and the last four
published errors' maximum, giving {controls.combined_uncertainty_ka.iloc[-1]:.2f} kyr. Its offset is drawn independently
from the corresponding uniform distribution. Its perturbation is not obtained
by extrapolating random offsets. This auxiliary control is not a dated sample.

The run uses NumPy's default_rng with seed {diagnostics['random_seed']} and saves
{int(s.n_realizations):,} chronologies. {int(diagnostics['n_proposals']):,} proposals were evaluated;
{int(diagnostics['n_rejected_crossed_controls']):,} crossed control maps were rejected, for an acceptance fraction of
{float(diagnostics['acceptance_fraction']):.6%}. There were {int(diagnostics['n_unused_accepted_tail']):,} unused accepted proposals in the final batch.
Event identities and ordering are preserved without sorting or clipping ages.

For every saved chronology, event counts and prior-event history are recomputed
on the nominal main-analysis grid. The nominal bin width is {main_analysis.BIN_WIDTH_KA:g} kyr, with origin
fraction {main_analysis.BIN_ORIGIN_FRACTION:g}. History counts older bin centers in (t, t + {main_analysis.HISTORY_WINDOW_KA:g} kyr], excluding
the current bin. Observation support is {segment.observation_start_kyr_bp:g}-{segment.observation_end_kyr_bp:g} kyr; response support is
{segment.response_start_kyr_bp:g}-{segment.response_end_kyr_bp:g} kyr ({context['frame'].dt_ka.sum():g} kyr exposure, {len(context['frame']):,} bins).
The exact response boundary splits a nominal bin when necessary. Forcing values,
their response-range scaling and exposure are held fixed. The reduced Poisson
model contains an intercept, event history, LR04 and CO2. The full model adds
precession-phase sine and cosine. The log of actual bin duration is the offset.
No between-record contrast or sampling-resolution covariate is fitted.

PI = (log L_full - log L_reduced)/(N_events*ln(2)), in bits per event. The nominal
likelihood-ratio test compares 2*(log L_full - log L_reduced) with chi-square(2).
Preferred phase = atan2(beta_sin, beta_cos), modulo 360 degrees. The phase rate
ratio (maximum/minimum) is exp(2*sqrt(beta_sin^2 + beta_cos^2)). Precession minimum
is 0 degrees and maximum is 180 degrees. Phase-response curves show
exp(beta_sin*sin(phi) + beta_cos*cos(phi)), holding other covariates fixed.

RESULTS
All numerical intervals below are the 2.5th-97.5th percentiles of valid chronology
realizations. They are not sampling confidence intervals or age-model posteriors.
Valid PI fits: {int(s.n_valid):,}/{int(s.n_realizations):,}; outside observation support: {int(s.n_invalid):,}.
Response events per valid fit: {int(s.n_predictive_events_min)}-{int(s.n_predictive_events_max)}.
The saved realization table includes convergence, nesting and clipping flags.
All valid fits converged: {bool(results.loc[results.fit_valid, 'all_models_converged'].all())}.
All valid fits respected likelihood nesting: {bool(results.loc[results.fit_valid, 'likelihood_nesting_ok'].all())}.
Any valid fit used eta clipping: {bool(results.loc[results.fit_valid, 'eta_clipping_used'].any())}.

Quantity                                  Point       MC median     MC 95% range
"""
    for key, label in (("info_bits_per_event", "PI (bits/event)"), ("LR_statistic", "Likelihood-ratio statistic"),
                       ("nominal_LR_p", "Nominal LRT p"), ("pre_phase_preferred_deg", "Preferred phase (degrees)"),
                       ("pre_phase_rate_ratio_max_vs_min", "Phase rate ratio (max/min)"),
                       ("delta_AICc_full_minus_reduced", "Delta AICc (full - reduced)")):
        text += (f"{label:40s} {s['point_' + key]:10.6g} {s[key + '_median']:12.6g} "
                 f"[{s[key + '_q025']:.6g}, {s[key + '_q975']:.6g}]\n")
    text += f"""
Nominal p < 0.05: {int(s.n_nominal_p_below_0p05):,}/{int(s.n_valid):,} valid realizations ({s.fraction_nominal_p_below_0p05:.4%}).
Delta AICc < 0: {int(s.n_delta_AICc_below_zero):,}/{int(s.n_valid):,} ({s.fraction_delta_AICc_below_zero:.4%}).
These fractions describe robustness under the specified chronology perturbation;
they are not empirical p-values. Per-realization p-values remain nominal;
Barker2011_PI_bootstrap.py separately calibrates the primary point-age test.
Phase quantiles are unwrapped within +/-180 degrees of the point-age preferred
phase ({s.point_pre_phase_preferred_deg:.6f} degrees). Values above 360 degrees wrap through
0 degrees. The MC preferred-phase resultant length is {s.preferred_phase_mc_resultant_length:.6f}.

The median PI is {100 * (1 - s.info_bits_per_event_median / s.point_info_bits_per_event):.1f}% lower than the point-age value, and the
median phase rate ratio is also lower. Preferred phases remain concentrated near
the point-age estimate, with a median change of
{s.pre_phase_preferred_deg_median - s.point_pre_phase_preferred_deg:+.2f} degrees. The nominal p < 0.05 conclusion holds for most,
but not all, sampled chronologies. This describes robustness conditional on the
specified control-error model and should accompany the point-age result.

INTERPRETATION AND SCOPE
The point-age comparison and the chronology ensemble address in-sample conditional
association, not validated out-of-sample prediction. This analysis propagates the
published combined matching/dating error under a specified interpolation model.
It does not add uncertainty in Table S3 event detection or event membership,
event-sampling variability, forcing chronologies, alternative feature matches,
or real climate lead/lag. Linear interpolation through the long gap supplies no
additional internal age-model variability; the auxiliary endpoint is a pragmatic
extension of recent error magnitudes, not a claim of complete error coverage.
The climate and event epochs follow the main analysis: SpeleoAge is numerically
unchanged and treated as BP1950, with its exact reference year not directly
verified. The shared orbital input applies the documented -0.05-kyr correction.

SOURCES AND OUTPUTS
Barker, S., et al. (2011). 800,000 years of abrupt climate variability.
Science 334, 347-351. doi:10.1126/science.1203580.
Source details: SOM printed p. 5, Figs. S11-S12, Table S1 (printed p. 24);
Table S3 in the companion workbook. The source PDF and Table S1 transcription
are retained in references/Barker2011_SOM.pdf and data/raw/Barker2011_TableS1.csv.
Source hashes and complete settings are saved with each analysis stage.

Run Barker2011_event_age_uncertainty.py, then Barker2011_event_uncertainty_sensitivity.py.
The respective data/processed directories contain controls, fixed event mapping,
control/event realizations, age summaries, PI fits, nominal result, MC summary,
phase-response bands and provenance. The figures directories contain PNG/PDF
exports. Companion captions are stored in this experiment_note directory.
"""
    NOTE_DIR.mkdir(parents=True, exist_ok=True)
    (NOTE_DIR / "Barker2011_event_uncertainty_Methods_and_results.txt").write_text(text)


def main():
    events = pd.read_csv(EVENT_INPUT, float_precision="round_trip")
    # Detect a changed source definition rather than fitting an obsolete ensemble.
    nominal_events = main_analysis.load_barker_source()
    if (not np.array_equal(events.source_excel_row, nominal_events.source_excel_row)
            or not np.allclose(events.event_age_ka, nominal_events.event_age_ka, rtol=0, atol=1e-9)):
        raise ValueError("The saved uncertainty catalogue differs from the current main source")
    realizations = pd.read_csv(AGE_INPUT, float_precision="round_trip")
    context = prepare_context(events)
    point = fit_ages(events.event_age_ka.to_numpy(float), context)
    results = fit_realizations(events, realizations, context, show_progress=True)
    summary = build_summary(results, point)
    curves = phase_response_summary(results, point)
    segment = context["segment"]
    settings = dict(analysis="Barker SpeleoAge chronology sensitivity", n_realizations=len(results),
                    history_window_kyr=main_analysis.HISTORY_WINDOW_KA, bin_width_kyr=main_analysis.BIN_WIDTH_KA,
                    origin_fraction=main_analysis.BIN_ORIGIN_FRACTION, response_mode=main_analysis.RESPONSE_MODE,
                    observation_start_ka=segment.observation_start_kyr_bp, observation_end_ka=segment.observation_end_kyr_bp,
                    response_start_ka=segment.response_start_kyr_bp, response_end_ka=segment.response_end_kyr_bp,
                    response_exposure_kyr=context["frame"].dt_ka.sum(),
                    reduced_terms="+".join(main_analysis.REDUCED_TERMS), full_terms="+".join(main_analysis.FULL_TERMS),
                    history_recomputed_per_realization=True, forcing_values_and_scaling="fixed at nominal main values",
                    intervals="2.5-97.5 percentiles; chronology sensitivity, not sampling confidence intervals",
                    phase_quantiles="unwrapped about nominal preferred phase",
                    robustness_denominator="all valid fits within observation support",
                    p_value_method="nominal asymptotic chi-square LRT, df=2; not bootstrap calibrated",
                    age_coordinate="SpeleoAge; BP1950 assumed, exact epoch unverified")
    sources = dict(realizations=AGE_INPUT, events=EVENT_INPUT,
                   controls=chronology.OUT_DATA_DIR / "age_control_points.csv",
                   chronology_parameters=chronology.OUT_DATA_DIR / "parameters_and_provenance.csv",
                   lr04=main_analysis.LR04_XLSX, co2=main_analysis.CO2_XLSX, precession=main_analysis.PRE_TXT,
                   sensitivity=Path(__file__).resolve(), main_analysis=Path(main_analysis.__file__).resolve(),
                   combined_pi=Path(combined_pi.__file__).resolve(),
                   poisson=ROOT.parent / "toolbox/poisson.py", orbital_phase=ROOT.parent / "toolbox/orbital_phase.py",
                   project_config=ROOT.parent / "toolbox/project_config.py", event_inputs=ROOT.parent / "toolbox/event_inputs.py")
    parameters = chronology.parameters_table(settings, sources)
    OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in (("pi_realizations", results), ("summary", summary),
                        ("point_age_result", pd.DataFrame([point])), ("phase_response_summary", curves),
                        ("predictor_scaling", context["scaling"]), ("parameters_and_provenance", parameters)):
        table.to_csv(OUT_DATA_DIR / f"{name}.csv", index=False, float_format="%.12g")
    chronology.save_figure(plot_sensitivity(results, point, curves), OUT_FIG_DIR, RUN_NAME)
    write_methods_results(summary, context, results)
    s = summary.iloc[0]
    print(f"Point PI {point['info_bits_per_event']:.6f}; MC median {s.info_bits_per_event_median:.6f}; "
          f"nominal p < 0.05 in {int(s.n_nominal_p_below_0p05):,}/{int(s.n_valid):,} valid fits "
          f"({s.fraction_nominal_p_below_0p05:.2%})")


if __name__ == "__main__":
    main()
