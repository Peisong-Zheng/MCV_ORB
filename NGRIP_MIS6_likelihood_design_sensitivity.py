#!/usr/bin/env python3
"""Continuous-history decay, initial-history and segment-phase sensitivity.

All scenarios retain the nominal exact conditioning anchors and forcing scales.
The history coefficient remains nonpositive. Integration resolution is checked
numerically and is not a scientific sensitivity axis.
"""

import argparse
from dataclasses import replace
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from toolbox import combined_likelihood
from toolbox.figure_style import add_panel_label
from toolbox.model_stats import nested_likelihood_metrics
from toolbox.project_config import PROJECT_ROOT, LR04_XLSX, CO2_XLSX, PRE_TXT

RUN_NAME = "NGRIP_MIS6_likelihood_design_sensitivity"
OUT_DATA_DIR = PROJECT_ROOT / "data/processed" / RUN_NAME
OUT_FIG_DIR = PROJECT_ROOT / "figures" / RUN_NAME
HISTORY_TAUS_KYR = (1.0, 1.5, 2.0, 3.0, 5.0)
INITIAL_HISTORY_VALUES = (0.0, 0.5, 1.0)
PRIMARY_HISTORY_TAU_KYR = 1.5
PHASE_INTERACTION_TERMS = ("mis6_x_pre_phase_sin", "mis6_x_pre_phase_cos")


def _fit_row(context, experiment):
    fitted = combined_likelihood.fit_catalogue(context.events, context, fixed_support=True)
    row = dict(experiment=experiment, **fitted.summary)
    beta = dict(zip(fitted.full.terms, fitted.full.beta))
    row.update(beta_pre_phase_sin=beta["pre_phase_sin"], beta_pre_phase_cos=beta["pre_phase_cos"],
               reduced_aic=fitted.reduced.aic, full_aic=fitted.full.aic,
               is_primary_design=context.history_tau_ka == PRIMARY_HISTORY_TAU_KYR
               and context.initial_history == 0)
    return row


def run_design_sensitivity(events=None):
    """Five fixed decay times on the same exact event and exposure support."""
    context = combined_likelihood.build_context(events=events)
    return pd.DataFrame([_fit_row(replace(context, history_tau_ka=tau), "history_decay")
                         for tau in HISTORY_TAUS_KYR])


def run_initial_history_sensitivity(events=None):
    """Prespecified unobserved history levels immediately before each anchor."""
    context = combined_likelihood.build_context(events=events)
    return pd.DataFrame([_fit_row(replace(context, initial_history=value), "initial_history")
                         for value in INITIAL_HISTORY_VALUES])


def _phase_summary(beta_sin, beta_cos):
    amplitude = np.hypot(beta_sin, beta_cos)
    phase = float(np.degrees(np.arctan2(beta_sin, beta_cos)) % 360) if amplitude > 1e-10 else np.nan
    return phase, float(np.exp(2 * amplitude))


def run_pooling_diagnostic(events=None):
    """Compare common and segment-specific phase coefficients on identical ages."""
    context = combined_likelihood.build_context(events=events)
    derived = {name: (combined_likelihood.SEGMENT_TERM, phase)
               for name, phase in zip(PHASE_INTERACTION_TERMS, ("pre_phase_sin", "pre_phase_cos"))}
    context = replace(context, derived_terms=derived)
    design = combined_likelihood.prepare_catalogue(context.events, context)
    common = combined_likelihood.fit_terms(design, context.full_terms)
    terms = context.full_terms + PHASE_INTERACTION_TERMS
    heterogeneous = combined_likelihood.fit_terms(design, terms)
    metrics = nested_likelihood_metrics(loglik_full=heterogeneous.log_likelihood,
        loglik_reduced=common.log_likelihood, df=2, n_events=len(design.event_frame),
        aic_full=heterogeneous.aic, aic_reduced=common.aic)
    beta = dict(zip(heterogeneous.terms, heterogeneous.beta))
    ngrip_phase, ngrip_ratio = _phase_summary(beta["pre_phase_sin"], beta["pre_phase_cos"])
    mis6_phase, mis6_ratio = _phase_summary(beta["pre_phase_sin"] + beta[PHASE_INTERACTION_TERMS[0]],
                                          beta["pre_phase_cos"] + beta[PHASE_INTERACTION_TERMS[1]])
    if metrics["ll_gain_nats"] < -1e-7:
        raise RuntimeError("Segment-specific phase likelihood is below the nested common model")
    return pd.DataFrame([dict(comparison="segment-specific versus common precession response",
        n_events=len(design.event_frame), response_exposure_kyr=context.response_exposure_kyr,
        common_model_loglik=common.log_likelihood, segment_specific_model_loglik=heterogeneous.log_likelihood,
        LR_statistic=metrics["LR_statistic"], df=2, nominal_LR_p=metrics["LR_p_value"],
        gain_bits_per_event_for_interaction=metrics["gain_bits_per_event"],
        delta_AIC_segment_specific_minus_common=metrics["delta_AIC_full_minus_reduced"],
        ngrip_preferred_phase_deg=ngrip_phase, mis6_preferred_phase_deg=mis6_phase,
        ngrip_phase_rate_ratio_max_vs_min=ngrip_ratio, mis6_phase_rate_ratio_max_vs_min=mis6_ratio,
        both_models_converged=common.converged and heterogeneous.converged,
        likelihood_nesting_ok=True)])


def plot_sensitivity(design, initial_history):
    """Show decay and initialization scenarios without implying uncertainty bands."""
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8, "axes.linewidth": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 3, figsize=(180 / 25.4, 120 / 25.4), sharey="col")
    metrics = (("gain_bits_per_event", "Gain (bits event$^{-1}$)"),
               ("pre_phase_preferred_deg", "Preferred phase (°)"),
               ("pre_phase_rate_ratio_max_vs_min", "Max/min rate ratio"))
    rows = ((design, "history_tau_kyr", "History decay time, τ (kyr)", "History decay"),
            (initial_history, "initial_unobserved_history", "Pre-anchor weighted history", "Initial history"))
    for r, (frame, xname, xlabel, title) in enumerate(rows):
        baseline = frame.loc[frame.is_primary_design].iloc[0]
        for c, (metric, ylabel) in enumerate(metrics):
            ax = axes[r, c]
            ax.plot(frame[xname], frame[metric], "o-", color="#0072B2", lw=1.1, ms=4)
            ax.axhline(baseline[metric], color="0.55", lw=0.8, ls="--", zorder=0)
            ax.scatter(baseline[xname], baseline[metric], s=30, facecolor="white",
                       edgecolor="#0072B2", linewidth=1.2, zorder=4)
            ax.set(xlabel=xlabel, ylabel=ylabel, xticks=frame[xname])
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(length=3)
            add_panel_label(ax, chr(97 + 3 * r + c), x=-0.04, y=1.07)
        axes[r, 1].set_title(title, y=1.13, fontsize=9, weight="bold")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.86, bottom=0.11, hspace=0.85, wspace=0.45)
    return fig


def write_outputs(design, initial_history, pooling, output_dir=OUT_DATA_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (("design_sensitivity", design), ("initial_history_sensitivity", initial_history),
                        ("pooling_diagnostic", pooling)):
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    context = combined_likelihood.build_context()
    combined_likelihood.support_table(context).to_csv(output_dir / "support.csv", index=False)
    combined_likelihood.scaling_table(context).to_csv(output_dir / "forcing_scaling.csv", index=False)
    parameters = dict(model_version=combined_likelihood.MODEL_VERSION,
        history_taus_kyr=";".join(map(str, HISTORY_TAUS_KYR)),
        initial_history_values=";".join(map(str, INITIAL_HISTORY_VALUES)),
        history_coefficient_domain="nonpositive", quadrature_order=context.quadrature_order,
        nominal_initial_history=0, primary_history_tau_kyr=PRIMARY_HISTORY_TAU_KYR,
        response_support="fixed exact nominal anchors; young observation boundaries retained",
        climate_scaling="fixed nominal exposure-time means and interpolant ranges",
        statistical_comparison="continuous LR and AIC; nominal chi-square p, no bootstrap calibration",
        phase_convention="index minimum 0 degrees; maximum 180; increases toward older BP ages",
        sampling="deterministic point-age scenarios; no Monte Carlo")
    pd.DataFrame(parameters.items(), columns=["parameter", "value"]).to_csv(
        output_dir / "parameters_and_provenance.csv", index=False)
    inputs = [Path(__file__).resolve(), combined_likelihood.EVENT_CATALOGUE_CSV,
        combined_likelihood.OBSERVATION_SEGMENTS_CSV, LR04_XLSX, CO2_XLSX, PRE_TXT,
        *[PROJECT_ROOT / "toolbox" / filename for filename in (
            "combined_likelihood.py", "point_process.py", "event_inputs.py", "event_process.py",
            "orbital_phase.py", "model_stats.py", "project_config.py")]]
    pd.DataFrame([dict(path=str(path.relative_to(PROJECT_ROOT)),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in inputs]).to_csv(
            output_dir / "input_code_sha256.csv", index=False)


def write_notes(design, initial_history, pooling, note_dir):
    note_dir = Path(note_dir)
    note_dir.mkdir(parents=True, exist_ok=True)
    primary = design.loc[design.is_primary_design].iloc[0]
    interaction = pooling.iloc[0]
    text = f"""NGRIP–MIS6 continuous-history design sensitivity

METHODS
Each segment conditions on its exact oldest event, with no unobserved history
before that anchor in the primary model. There are {int(primary.n_response_events)}
response events over {primary.response_exposure_kyr:.6f} kyr. The anchor enters
subsequent exponential history but contributes no event log-intensity term;
the final event-free interval to the younger observation boundary is retained.
All variants use the same actual events, exact support and nominal exposure-time
forcing scales. The event rate describes the whole observation interval.

The conditional log intensity contains LR04, CO2, a MIS6 intercept contrast,
a history term and (in the full model) precession sine and cosine. The history
coefficient is constrained to be nonpositive. History decay times of
1, 1.5, 2, 3 and 5 kyr are examined with initial history zero. Separately,
pre-anchor weighted history h_pre=0, 0.5 or 1 is assigned to each segment at
tau=1.5 kyr, decaying from its anchor age. These are illustrative initial
conditions, not a calibrated range for unknown history. No response interval
is discarded or reinitialized in these comparisons.

Fit the continuous event-sum-minus-intensity-integral likelihood. Positive
Gauss–Legendre integration nodes resolve event and forcing discontinuities;
their number does not enter the statistical sample size. G is the nested
log-likelihood gain per response event in bits/event. The conditional phase
peak and maximum/minimum multiplier ratio are fitted effect summaries.
Nominal chi-square p values accompany LR, with two phase parameters added;
they do not reuse the primary model's bootstrap calibration. Delta AIC is
full minus reduced and equals 4-LR. AIC penalties at a history-constraint
boundary are approximate. All scenarios are reported, without p-based selection.

A separate pooling diagnostic adds MIS6-by-sine and MIS6-by-cosine to the
common-phase full model. It tests two segment differences, while retaining
the common terms and the same continuous support; its p is nominal.

RESULTS
Primary G={primary.gain_bits_per_event:.8f}, nominal p={primary.nominal_LR_p:.8g},
phase={primary.pre_phase_preferred_deg:.4f} degrees, max/min={primary.pre_phase_rate_ratio_max_vs_min:.6f}.
Decay scenarios: G range [{design.gain_bits_per_event.min():.8f}, {design.gain_bits_per_event.max():.8f}].
Initial-history scenarios: G range [{initial_history.gain_bits_per_event.min():.8f}, {initial_history.gain_bits_per_event.max():.8f}].
Pooling diagnostic: LR={interaction.LR_statistic:.8f}, nominal p={interaction.nominal_LR_p:.8g},
additional G={interaction.gain_bits_per_event_for_interaction:.8f},
delta AIC={interaction.delta_AIC_segment_specific_minus_common:.8f}.
Exact scenario estimates, phases, strengths and history coefficients are saved
in design_sensitivity.csv and initial_history_sensitivity.csv; pooling estimates
are in pooling_diagnostic.csv. These scenario ranges are not confidence intervals.
"""
    (note_dir / f"{RUN_NAME}_Methods_and_results.txt").write_text(text)
    caption = """Sensitivity of the continuous NGRIP–MIS6 phase association to event history.
(a–c) Exponential decay time tau=1, 1.5, 2, 3 or 5 kyr, with unobserved pre-anchor
history zero. (d–f) Pre-anchor weighted history h_pre=0, 0.5 or 1 at tau=1.5 kyr.
Columns show the phase log-likelihood gain per response event G (bits/event),
fitted preferred phase, and maximum/minimum phase-only rate ratio. The vertical
scale is shared within each column. Open markers
and horizontal dashed lines identify the primary settings (tau=1.5 kyr,
h_pre=0). Connected points are prespecified scenario estimates, not uncertainty
intervals. All models use the same exact conditioning anchors, response events,
continuous exposure and nominal climate scaling, with a nonpositive history
coefficient. Phase zero is a precession-index minimum, 180 degrees a maximum,
and phase increases toward older BP ages. The primary bootstrap p does not
calibrate these alternative-history fits.
"""
    (note_dir / f"{RUN_NAME}_Caption.txt").write_text(caption)


def save_figure(fig, output_dir=OUT_FIG_DIR, *, paper_export=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png, pdf = [output_dir / f"{RUN_NAME}.{extension}" for extension in ("png", "pdf")]
    fig.savefig(png, dpi=600, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    if paper_export:
        from paper_figure_export import copy_pdf_to_paper
        copy_pdf_to_paper(pdf)
    return png, pdf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--no-paper-export", action="store_true")
    args = parser.parse_args()
    design = run_design_sensitivity()
    initial = run_initial_history_sensitivity()
    pooling = run_pooling_diagnostic()
    output_dir = args.output_root / "data/processed" / RUN_NAME
    write_outputs(design, initial, pooling, output_dir)
    write_notes(design, initial, pooling, args.output_root / "experiment_note")
    save_figure(plot_sensitivity(design, initial), args.output_root / "figures" / RUN_NAME,
                paper_export=not args.no_paper_export)
    print(design[["history_tau_kyr", "gain_bits_per_event", "nominal_LR_p"]].to_string(index=False))
    print(initial[["initial_unobserved_history", "gain_bits_per_event", "nominal_LR_p"]].to_string(index=False))
    print(pooling.to_string(index=False))


if __name__ == "__main__":
    main()
