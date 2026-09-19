#!/usr/bin/env python3
"""Continuous-time sensitivity to background climate shape and event history.

All nine point-age variants retain exact nominal anchors, response exposure and
forcing scales. Exponential/rectangular histories have nonpositive coefficients;
elapsed-time and log-elapsed coefficients are unrestricted on the finite support.
"""

import argparse
from dataclasses import replace
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from toolbox import combined_likelihood
from toolbox.figure_style import add_panel_label
from toolbox.project_config import CO2_XLSX, LR04_XLSX, PRE_TXT, PROJECT_ROOT

RUN_NAME = "NGRIP_MIS6_likelihood_model_sensitivity"
OUT_DATA_DIR = PROJECT_ROOT / "data/processed" / RUN_NAME
OUT_FIG_DIR = PROJECT_ROOT / "figures" / RUN_NAME
ELAPSED_TERM = "time_since_last_event_kyr"
LOG_ELAPSED_TERM = "log_time_since_last_event"
ELAPSED_SCALE_KYR = 1.0
PHASE_TERMS = ("pre_phase_sin", "pre_phase_cos")

# Post-audit sensitivity set; report every variant without selecting by p.
CLIMATE_VARIANTS = {
    "frozen_linear": (),
    "quadratic_lr04": ("lr04_squared",),
    "quadratic_co2": ("co2_squared",),
    "quadratic_both": ("lr04_squared", "co2_squared"),
    "quadratic_with_interaction": ("lr04_squared", "co2_squared", "lr04_co2"),
}
HISTORY_VARIANTS = {
    "exponential": combined_likelihood.REDUCED_TERMS[0],
    "count_matched_support": "rectangular_history_count",
    "elapsed_time": ELAPSED_TERM,
    "log_elapsed_time": LOG_ELAPSED_TERM,
}
VARIANT_LABELS = {
    "frozen_linear": "Linear climate",
    "quadratic_lr04": "+ LR04²",
    "quadratic_co2": "+ CO₂²",
    "quadratic_both": "+ both squares",
    "quadratic_with_interaction": "+ squares + interaction",
    "exponential": "Exponential history (τ = 1.5 kyr)",
    "count_matched_support": "Count in previous 1.5 kyr",
    "elapsed_time": "Time since last event",
    "log_elapsed_time": "log(1 + time / 1 kyr)",
}
VARIANT_COLORS = {
    "frozen_linear": "#666666",
    "quadratic_lr04": "#0072B2",
    "quadratic_co2": "#D55E00",
    "quadratic_both": "#009E73",
    "quadratic_with_interaction": "#CC79A7",
    "exponential": "#666666",
    "count_matched_support": "#CC79A7",
    "elapsed_time": "#0072B2",
    "log_elapsed_time": "#D55E00",
}


def prepare_predictors(events, context):
    """Register climate products before evaluating exact event/integral features."""
    derived = dict(context.derived_terms or {})
    derived.update(lr04_squared=("lr04_scaled", "lr04_scaled"),
                   co2_squared=("co2_scaled", "co2_scaled"),
                   lr04_co2=("lr04_scaled", "co2_scaled"))
    context = replace(context, derived_terms=derived)
    design = combined_likelihood.prepare_catalogue(events, context, fixed_support=True)
    return design


def fit_variant(design, terms, *, experiment, variant, support_id="exact_nominal"):
    """Refit a continuous nested pair with the appropriate history parameter domain."""
    context = replace(design.context, reduced_terms=tuple(terms), full_terms=tuple(terms) + PHASE_TERMS)
    fitted = combined_likelihood.fit_catalogue(context.events, context, fixed_support=True)
    if not fitted.reduced.converged or not fitted.full.converged:
        raise RuntimeError(f"{variant}: continuous fit did not converge")
    count_history = terms[0] in (combined_likelihood.HISTORY_TERM, "rectangular_history_count")
    summary = dict(fitted.summary)
    summary.update(history_coefficient_domain="nonpositive" if count_history else "unrestricted",
        history_kernel=terms[0], history_tau_kyr=context.history_tau_ka if count_history else np.nan,
        beta_history=dict(zip(fitted.full.terms, fitted.full.beta))[terms[0]])
    row = dict(experiment=experiment, variant=variant, variant_label=VARIANT_LABELS[variant],
        support_id=support_id, reduced_terms=";".join(terms), **summary,
        n_params_reduced=len(fitted.reduced.beta), n_params_full=len(fitted.full.beta),
        reduced_aic=fitted.reduced.aic, full_aic=fitted.full.aic, history_term=terms[0])
    coefficients = [dict(experiment=experiment, variant=variant, support_id=support_id,
        model_id=model_id, term=term, beta=float(beta))
        for model_id, model in (("reduced", fitted.reduced), ("full", fitted.full))
        for term, beta in zip(model.terms, model.beta)]
    return row, coefficients


def run_analysis():
    context = combined_likelihood.build_context()
    design = prepare_predictors(context.events, context)
    rows, coefficients = [], []
    for variant, extra in CLIMATE_VARIANTS.items():
        row, coeffs = fit_variant(design, context.reduced_terms + extra,
                                  experiment="climate", variant=variant)
        rows.append(row)
        coefficients.extend(coeffs)
    for variant, term in HISTORY_VARIANTS.items():
        row, coeffs = fit_variant(design, (term, *context.reduced_terms[1:]),
                                  experiment="history", variant=variant)
        rows.append(row)
        coefficients.extend(coeffs)
    summary = pd.DataFrame(rows)
    for experiment, reference_id in (("climate", "frozen_linear"), ("history", "exponential")):
        mask = summary.experiment.eq(experiment)
        reference_aic = summary.loc[summary.variant.eq(reference_id), "full_aic"].iloc[0]
        summary.loc[mask, "reference_variant"] = reference_id
        summary.loc[mask, "delta_AIC_full_vs_same_support_reference"] = summary.loc[mask, "full_aic"] - reference_aic
    coefficient_table = pd.DataFrame(coefficients)
    phase = np.arange(361, dtype=float)
    curves = []
    for row in summary.itertuples(index=False):
        beta = coefficient_table.loc[coefficient_table.variant.eq(row.variant)
            & coefficient_table.model_id.eq("full")].set_index("term").beta
        curves.append(pd.DataFrame(dict(experiment=row.experiment, variant=row.variant, phase_deg=phase,
            relative_rate=np.exp(beta[PHASE_TERMS[0]] * np.sin(np.deg2rad(phase))
                               + beta[PHASE_TERMS[1]] * np.cos(np.deg2rad(phase))))))
    return dict(summary=summary, coefficients=coefficient_table, event_inputs=design.event_frame,
        segment_support=combined_likelihood.support_table(design.context), event_roles=design.all_events,
        forcing_scaling=combined_likelihood.scaling_table(context),
        phase_curves=pd.concat(curves, ignore_index=True))


def plot_sensitivity(summary):
    """Compare point estimates within each experiment; no implied intervals."""

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
        "ytick.labelsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 3, figsize=(180 / 25.4, 132 / 25.4))
    fields = ("gain_bits_per_event", "pre_phase_preferred_deg", "pre_phase_rate_ratio_max_vs_min")
    labels = ("Gain (bits event$^{-1}$)", "Preferred phase (°)", "Max/min rate ratio")
    plot_labels = {**VARIANT_LABELS, "quadratic_with_interaction": "+ interaction",
                   "exponential": "Exponential", "count_matched_support": "Window count",
                   "elapsed_time": "Elapsed time", "log_elapsed_time": "Log elapsed time"}

    for row_index, experiment in enumerate(("climate", "history")):
        frame = summary.loc[summary.experiment.eq(experiment)].reset_index(drop=True)
        y = np.arange(len(frame))
        for column, (field, label) in enumerate(zip(fields, labels)):
            ax = axes[row_index, column]
            ax.axvline(frame[field].iloc[0], color="0.65", linewidth=1, linestyle="--", zorder=1)
            for index, row in frame.iterrows():
                ax.scatter(row[field], index, color=VARIANT_COLORS[row.variant], s=22, zorder=3)
            ax.set_yticks(y, frame.variant.map(plot_labels) if column == 0 else [])
            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.set_ylim(len(frame) - 0.5, -0.5)
            ax.set_xlabel(label)
            ax.grid(axis="y", color="0.92", linewidth=0.8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="y", length=0)
            add_panel_label(ax, chr(97 + row_index * 3 + column), x=-0.04, y=1.05)
            values = frame[field].to_numpy(float)
            if field == "gain_bits_per_event":
                ax.set_xlim(0, max(0.21, values.max() * 1.15))
            elif field == "pre_phase_preferred_deg":
                margin = max(3, np.ptp(values) * 0.25)
                ax.set_xlim(values.min() - margin, values.max() + margin)
            else:
                ax.set_xlim(1, max(6, values.max() * 1.15))

    fig.subplots_adjust(left=0.24, right=0.98, top=0.86, bottom=0.12, hspace=0.78, wspace=0.38)
    for row_index, title in enumerate((
        "Background climate",
        "Event history",
    )):
        box = axes[row_index, 0].get_position()
        fig.text(box.x0, box.y1 + 0.075, title, fontsize=9, fontweight="bold")
    return fig


def save_results(result, output_dir=OUT_DATA_DIR, figure_dir=OUT_FIG_DIR, *, paper_export=True):
    output_dir, figure_dir = Path(output_dir), Path(figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    filenames = dict(summary="model_sensitivity.csv", coefficients="model_coefficients.csv",
        event_inputs="event_inputs.csv", segment_support="segment_support.csv", event_roles="event_roles.csv",
        forcing_scaling="forcing_scaling.csv", phase_curves="phase_response_curves.csv")
    for key, filename in filenames.items():
        result[key].to_csv(output_dir / filename, index=False)
    parameters = dict(model_version=combined_likelihood.MODEL_VERSION,
        method="continuous conditional point process", age_epoch="kyr BP relative to AD1950",
        response_support="fixed exact nominal anchors; 53 response events over 165.058 kyr",
        history_tau_kyr=combined_likelihood.DEFAULT_HISTORY_TAU_KA,
        initial_unobserved_history=0, quadrature_order=4,
        climate_scaling="fixed nominal exposure-time mean / interpolant range",
        polynomial_transform="squares and products of fixed scaled climate predictors",
        elapsed_time_definition="strictly older event age minus current age within the same segment",
        elapsed_time_scale_kyr=ELAPSED_SCALE_KYR, log_elapsed_transform="log(1 + elapsed_time / 1 kyr)",
        history_domains="exponential/rectangular nonpositive; elapsed/log elapsed unrestricted",
        boundary_handling="exact conditioning anchor supplies history but no response event term",
        phase_convention="index minimum 0 degrees; maximum 180; increases toward older BP age",
        nominal_p_definition="chi-square(2) survival of twice nested continuous likelihood gain",
        model_comparison="AIC only; no node-count finite-sample correction",
        bootstrap_scope="none here; primary empirical p does not calibrate these alternative models",
        selection_scope="all five climate and four history forms; no p-based model selection",
        randomness="none; deterministic point-age fits")
    pd.DataFrame(parameters.items(), columns=["parameter", "value"]).to_csv(
        output_dir / "parameters_and_provenance.csv", index=False)
    inputs = [Path(__file__).resolve(), combined_likelihood.EVENT_CATALOGUE_CSV,
        combined_likelihood.OBSERVATION_SEGMENTS_CSV, LR04_XLSX, CO2_XLSX, PRE_TXT,
        *[PROJECT_ROOT / "toolbox" / filename for filename in ("combined_likelihood.py", "point_process.py",
            "model_stats.py", "event_inputs.py", "event_process.py", "orbital_phase.py", "project_config.py")]]
    pd.DataFrame([dict(path=str(path.relative_to(PROJECT_ROOT)),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in inputs]).to_csv(
            output_dir / "input_code_sha256.csv", index=False)
    fig = plot_sensitivity(result["summary"])
    for extension in ("png", "pdf"):
        fig.savefig(figure_dir / f"{RUN_NAME}.{extension}", dpi=600, facecolor="white")
    plt.close(fig)
    if paper_export:
        from paper_figure_export import copy_pdf_to_paper
        copy_pdf_to_paper(figure_dir / f"{RUN_NAME}.pdf")


def write_notes(result, note_dir):
    note_dir = Path(note_dir)
    note_dir.mkdir(parents=True, exist_ok=True)
    summary = result["summary"]
    text = f"""NGRIP–MIS6 continuous model-form sensitivity

METHODS
Compare five background-climate specifications: scaled linear LR04 and CO2;
add LR04 squared; add CO2 squared; add both squares; or add both squares and
LR04-by-CO2. These are products of the original fixed scaled covariates, not
separate event-based standardizations. Compare four event-history functions:
exponential weighted history at tau=1.5 kyr; count in the preceding 1.5 kyr;
elapsed time since the immediately preceding event divided by 1 kyr; and
log(1 + elapsed_time/1 kyr). Both history counts have nonpositive coefficients;
elapsed and log-elapsed coefficients are unrestricted on finite observation
support. A positive elapsed coefficient can describe recovery after an event.

The exact oldest event in each segment conditions the fit, initializes history
and is excluded from the response-event sum. All variants retain the same
53 response events over 165.058 kyr and include the younger event-free tail.
No history crosses the NGRIP–MIS6 gap. Unobserved pre-anchor history is zero.
The response is warming occurrence over whole observed time, not a cold-state
triggering rate. Each nested comparison uses the same exact events, continuous
support, interpolants and exposure-time scaling. Reduced models contain the
chosen climate/history terms and segment intercept; full models add the same
precession sine and cosine.

The continuous log likelihood sums event log intensities and subtracts the
conditional intensity integral. Integration is divided at actual event ages,
forcing/phase interpolation knots and rectangular-window exits when relevant;
positive Gauss–Legendre weights are durations, not observations. No event is
rounded to a bin center. G is the added phase log-likelihood gain divided by
the response-event number and ln(2), in bits/event. LR p values are nominal
chi-square(2), not recalibrated bootstrap p values. Delta AIC is full minus
reduced (4-LR); full-model AIC contrasts within each experiment use its matching
linear-climate/exponential-history baseline. AIC penalties at parameter-domain
boundaries are approximate. Report every variant without selecting by p.

RESULTS
{summary[['variant', 'gain_bits_per_event', 'nominal_LR_p', 'pre_phase_preferred_deg', 'pre_phase_rate_ratio_max_vs_min', 'delta_AIC_full_vs_same_support_reference']].to_string(index=False)}

These estimates address sensitivity to specified model forms, not sampling
confidence. A change in gain may reflect altered conditional attribution among
correlated predictors. Preferred-phase and rate-ratio estimates retain the
same sine/cosine convention in every model. None of these alternative models
inherits the primary model's bootstrap calibration. The exact model coefficients,
event histories, event roles, support and nominal scales are saved with the results.
"""
    (note_dir / f"{RUN_NAME}_Methods_and_results.txt").write_text(text)
    caption = """Sensitivity of continuous NGRIP–MIS6 phase association to model form.
(a–c) Five climate backgrounds: linear LR04 and CO2, either squared term, both
squares, or both squares plus their interaction. (d–f) Four history terms:
exponential history, previous-window count, elapsed time and log(1 + elapsed
time/1 kyr). The exponential decay and rectangular window both equal 1.5 kyr;
count-history coefficients are nonpositive, while elapsed-history coefficients
are unrestricted. Columns show phase log-likelihood gain G (bits per response
event), preferred phase and phase-only maximum/minimum rate ratio. Dots are
point-age estimates; dashed lines identify the linear-climate/exponential-history
reference. All variants share the same exact conditioning anchors, 53 response
events, 165.058 kyr exposure and nominal climate scales. Phase zero is an index
minimum, 180 degrees a maximum, and phase increases toward older BP ages.
These are specification sensitivities, not confidence intervals or separately
bootstrap-calibrated tests.
"""
    (note_dir / f"{RUN_NAME}_Caption.txt").write_text(caption)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--no-paper-export", action="store_true")
    args = parser.parse_args()
    result = run_analysis()
    output_dir = args.output_root / "data/processed" / RUN_NAME
    save_results(result, output_dir, args.output_root / "figures" / RUN_NAME,
                 paper_export=not args.no_paper_export)
    write_notes(result, args.output_root / "experiment_note")
    print(result["summary"][["variant", "gain_bits_per_event", "nominal_LR_p",
        "pre_phase_preferred_deg", "pre_phase_rate_ratio_max_vs_min",
        "delta_AIC_full_vs_same_support_reference"]].to_string(index=False))


if __name__ == "__main__":
    main()
