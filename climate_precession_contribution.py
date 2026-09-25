#!/usr/bin/env python3
"""Compare conditional climate and precession gains at nominal event ages.

Run from the project root: python climate_precession_contribution.py
The three fits share event times, history, exposure and forcing scales.
The figure is synchronized to the manuscript through its figure manifest.
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from toolbox import combined_likelihood as likelihood
from toolbox.catalogue_colors import CATALOGUE_COLORS
from toolbox.model_stats import nested_likelihood_metrics
from toolbox.project_config import PROJECT_ROOT, LR04_XLSX, CO2_XLSX, PRE_TXT
from paper_figure_export import copy_pdf_to_paper

RUN_NAME = "climate_precession_contribution"
DATA_DIR = PROJECT_ROOT / "data/processed" / RUN_NAME
FIGURE_DIR = PROJECT_ROOT / "figures" / RUN_NAME
NOTE_DIR = PROJECT_ROOT / "experiment_note"
CLIMATE_TERMS = ("lr04_scaled", "co2_scaled")
PHASE_TERMS = ("pre_phase_sin", "pre_phase_cos")
CATALOGUE_LABELS = {
    "primary": "NGRIP + MIS6",
    "variable": "Barker varying threshold",
    "fixed": "Barker fixed threshold",
}


def fit_contributions(context):
    """Remove each two-parameter block while retaining the other block."""
    design = likelihood.prepare_catalogue(context.events, context, fixed_support=True)
    specifications = {
        "without_precession": context.reduced_terms,
        "without_climate": tuple(t for t in context.full_terms if t not in CLIMATE_TERMS),
        "full": context.full_terms,
    }
    models = {}
    for name, terms in specifications.items():
        # Embed the fitted background model as a valid starting full model.
        start = None
        if name == "full":
            previous = dict(zip(models["without_precession"].terms,
                                models["without_precession"].beta))
            start = np.array([previous.get(t, 0.0) for t in ("intercept", *terms)])
        model = likelihood.fit_terms(design, terms, start_beta=start)
        if not model.converged or not model.identifiable or not np.isfinite(model.log_likelihood):
            raise RuntimeError(f"{context.catalogue_id}: {name} has no valid finite fit")
        models[name] = model

    full = models["full"]
    comparisons = []
    for block, reference in (("precession", "without_precession"),
                             ("climate", "without_climate")):
        reduced = models[reference]
        removed = set(full.terms) - set(reduced.terms)
        expected = PHASE_TERMS if block == "precession" else CLIMATE_TERMS
        if removed != set(expected) or full.n_events != reduced.n_events:
            raise RuntimeError("The comparison must remove only the specified block")
        metrics = nested_likelihood_metrics(
            loglik_full=full.log_likelihood, loglik_reduced=reduced.log_likelihood,
            df=len(removed), n_events=full.n_events,
            aic_full=full.aic, aic_reduced=reduced.aic,
        )
        comparisons.append(dict(
            added_block=block,
            conditioned_on="climate" if block == "precession" else "precession",
            reference_model=reference, n_response_events=full.n_events,
            response_exposure_kyr=context.response_exposure_kyr,
            loglik_full=full.log_likelihood, loglik_reference=reduced.log_likelihood,
            gain_bits_per_event=metrics["gain_bits_per_event"],
            LR_statistic=metrics["LR_statistic"], df=metrics["df"],
            nominal_LR_p=metrics["LR_p_value"],
            delta_AIC_full_minus_reference=metrics["delta_AIC_full_minus_reduced"],
        ))
    return models, pd.DataFrame(comparisons)


def run_analysis():
    contexts = {
        "primary": likelihood.build_context(),
        "variable": likelihood.build_barker_context("variable_threshold"),
        "fixed": likelihood.build_barker_context("fixed_threshold"),
    }
    comparisons, model_rows, coefficients = [], [], []
    for catalogue, context in contexts.items():
        models, table = fit_contributions(context)
        table.insert(0, "catalogue", catalogue)
        comparisons.append(table)
        for name, model in models.items():
            model_rows.append(dict(
                catalogue=catalogue, model=name, terms=";".join(model.terms),
                log_likelihood=model.log_likelihood, AIC=model.aic,
                n_parameters=len(model.beta), n_response_events=model.n_events,
                converged=model.converged, fit_status=model.status,
            ))
            coefficients.extend(dict(catalogue=catalogue, model=name, term=t, beta=b)
                                for t, b in zip(model.terms, model.beta))
    return (pd.concat(comparisons, ignore_index=True), pd.DataFrame(model_rows),
            pd.DataFrame(coefficients), contexts)


def plot_contributions(comparisons):
    """Paired bars compare gains, not fractions of variance or causal effects."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(180 / 25.4, 100 / 25.4))
    maximum = comparisons.gain_bits_per_event.max()
    for y, (catalogue, label) in enumerate(CATALOGUE_LABELS.items()):
        rows = comparisons.loc[comparisons.catalogue.eq(catalogue)].set_index("added_block")
        color = CATALOGUE_COLORS[catalogue]
        for block, offset in (("precession", -0.18), ("climate", 0.18)):
            value = rows.loc[block, "gain_bits_per_event"]
            ax.barh(y + offset, value, height=0.28, edgecolor=color, linewidth=1.1,
                    facecolor=color if block == "precession" else "white",
                    hatch=None if block == "precession" else "////")
            ax.text(value + maximum * 0.025, y + offset, f"{value:.3f}",
                    ha="left", va="center", fontsize=9)
    ax.set_yticks(range(3), [
        f"{label}\n({int(comparisons.loc[comparisons.catalogue.eq(key), 'n_response_events'].iloc[0])} response events)"
        for key, label in CATALOGUE_LABELS.items()
    ])
    ax.set_ylim(2.65, -0.65)
    ax.set_xlim(0, maximum * 1.25)
    ax.set_xlabel("Conditional log-likelihood gain (bits/event)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.grid(False)
    handles = [Patch(facecolor="0.4", edgecolor="0.4", label="Precession beyond climate"),
               Patch(facecolor="white", edgecolor="0.4", hatch="////",
                     label="Climate beyond precession")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.98),
               ncol=2, frameon=False, fontsize=8.5, handlelength=1.8, columnspacing=1.5)
    fig.subplots_adjust(left=0.32, right=0.97, bottom=0.17, top=0.84)
    return fig


def write_notes(comparisons, contexts):
    """Keep the method and interpretation beside the research outputs."""
    lines = [
        "Conditional contributions of climate background and precession", "",
        "Question: How much does either predictor block improve the nominal-age fit",
        "after the other block has already been included?", "",
        "Method", "------",
        "Reuse toolbox/combined_likelihood.py without changing the main analyses.",
        "Fit three continuous-time models per catalogue: full; without precession;",
        "and without climate. Climate comprises scaled LR04 and CO2 (two coefficients).",
        "Precession comprises sine and cosine of phase (two coefficients). Every model",
        "retains its intercept, inhibitory exponential event history (tau = 1.5 kyr),",
        "and, for the primary catalogue, the NGRIP/MIS6 intercept contrast.",
        "All remaining coefficients are refitted, with beta_history <= 0.",
        "Each segment conditions on its exact oldest event. Shared nominal response",
        "support, younger boundaries, event histories, climate scales and integration",
        "nodes ensure that only the predictor block changes within each comparison.",
        "BP1950 ages are handled by the existing elapsed-time implementation.",
        "G_block = (loglik_full - loglik_without_block) / (N_response * ln(2)).",
        "The table also reports LR = 2 * loglik gain, nominal chi-square p (df = 2),",
        "and AIC differences. No new bootstrap or chronology ensemble is run here.",
        "The nominal p values are exploratory and unadjusted, not bootstrap values.", "",
        "Results at nominal ages", "-----------------------",
    ]
    for catalogue, label in CATALOGUE_LABELS.items():
        rows = comparisons.loc[comparisons.catalogue.eq(catalogue)].set_index("added_block")
        p, c = rows.loc["precession"], rows.loc["climate"]
        lines.append(
            f"{label}: precession beyond climate = {p.gain_bits_per_event:.6f} bits/event "
            f"(LR={p.LR_statistic:.6f}, nominal p={p.nominal_LR_p:.6g}); "
            f"climate beyond precession = {c.gain_bits_per_event:.6f} bits/event "
            f"(LR={c.LR_statistic:.6f}, nominal p={c.nominal_LR_p:.6g})."
        )
    lines += ["", "Interpretation", "--------------",
        "These gains measure conditional in-sample fit improvement, not percentages",
        "of explained variance, mutual information, or causal importance. Shared",
        "information is not assigned exhaustively, so the bars are not additive.",
        "No uncertainty interval or test of the difference between the two gains is",
        "provided. Ordering their point estimates does not establish a robust ranking.",
        "Background here means the specified linear LR04/CO2 predictors, not every",
        "aspect of climate state. Fixed-threshold Barker is a definition sensitivity",
        "sharing source events with the varying-threshold catalogue, not a replicate.",
        "The comparison is within each catalogue; their event counts and time spans differ.",
    ]
    NOTE_DIR.mkdir(parents=True, exist_ok=True)
    (NOTE_DIR / f"{RUN_NAME}_Methods_and_results.txt").write_text("\n".join(lines) + "\n")
    caption = (
        "Conditional fit contributions of precession phase and climate background. "
        "Solid bars show the log-likelihood gain from adding precession sine/cosine "
        "to a model containing LR04 and CO2; hatched bars show the gain from adding "
        "LR04 and CO2 to a model containing precession. All models retain event history "
        "and, for NGRIP + MIS6, a segment intercept contrast. Each comparison adds "
        "two coefficients and refits all remaining coefficients on identical response "
        "support. Gains are normalized by the number of response events (excluding "
        "the oldest conditioning event in each segment). Colors identify the event "
        "catalogues; Barker fixed threshold is an alternative event definition. "
        "Values are nominal-age point estimates without uncertainty intervals; "
        "they are not additive shares of explained variability or causal effects.\n"
    )
    (NOTE_DIR / f"{RUN_NAME}_Caption.txt").write_text(caption)
    sources = [likelihood.EVENT_CATALOGUE_CSV, likelihood.OBSERVATION_SEGMENTS_CSV,
               PROJECT_ROOT / "Barker2011/data/raw/Barker et al-2011-SOM.xls",
               LR04_XLSX, CO2_XLSX, PRE_TXT]
    provenance = dict(model_version=likelihood.MODEL_VERSION, age_units="kyr BP1950",
                      inputs=[str(p.relative_to(PROJECT_ROOT)) for p in sources], catalogues={})
    for key, context in contexts.items():
        provenance["catalogues"][key] = dict(
            catalogue_id=context.catalogue_id, n_source_events=len(context.events),
            history_tau_kyr=context.history_tau_ka, initial_history=context.initial_history,
            quadrature_order=context.quadrature_order, scaling=context.scaling,
            support=likelihood.support_table(context).to_dict("records"),
        )
    (DATA_DIR / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main():
    comparisons, models, coefficients, contexts = run_analysis()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in (("comparisons", comparisons), ("models", models),
                        ("coefficients", coefficients)):
        table.to_csv(DATA_DIR / f"{name}.csv", index=False, float_format="%.12g")
    figure = plot_contributions(comparisons)
    figure.savefig(FIGURE_DIR / f"{RUN_NAME}.pdf")
    figure.savefig(FIGURE_DIR / f"{RUN_NAME}.png", dpi=300)
    plt.close(figure)
    copy_pdf_to_paper(FIGURE_DIR / f"{RUN_NAME}.pdf")
    write_notes(comparisons, contexts)
    print(comparisons[["catalogue", "added_block", "gain_bits_per_event", "nominal_LR_p"]]
          .to_string(index=False))
    print(f"Figure: {FIGURE_DIR / (RUN_NAME + '.pdf')}")


if __name__ == "__main__":
    main()
