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

from toolbox import event_model
from toolbox.point_process import fit_point_process
from toolbox.project_config import EVENT_CATALOGUE_CSV, OBSERVATION_SEGMENTS_CSV, MODEL_VERSION
from toolbox.model_stats import nested_likelihood_metrics
from toolbox.project_config import PROJECT_ROOT, BARKER_EVENT_CSVS, CATALOGUE_COLORS, LR04_CSV, CO2_CSV, ORBITAL_CSV, PRECESSION_PHASE_CSV
from paper_figure_export import copy_pdf_to_paper
from toolbox.project_config import generated_notes_dir

RUN_NAME = "climate_precession_contribution"
OUTPUT_ROOT = PROJECT_ROOT
EXPORT_PAPER = True
CLIMATE_TERMS = ("lr04_scaled", "co2_scaled")
PHASE_TERMS = ("pre_phase_sin", "pre_phase_cos")
CATALOGUE_LABELS = {
    "primary": "NGRIP + MIS6",
    "variable": "Barker varying threshold",
    "fixed": "Barker fixed threshold",
}


def fit_contributions(event_x, integral_x, windows, background):
    """Remove each two-parameter block while retaining the other block."""
    full_terms = background + PHASE_TERMS
    specifications = {
        "without_precession": background,
        "without_climate": tuple(t for t in full_terms if t not in CLIMATE_TERMS),
        "full": full_terms,
    }
    models = {}
    for name, terms in specifications.items():
        # Embed the fitted background model as a valid starting full model.
        start = None
        if name == "full":
            previous = dict(zip(models["without_precession"].terms,
                                models["without_precession"].beta))
            start = np.array([previous.get(t, 0.0) for t in terms])
        model = fit_point_process(
            event_x[list(terms)], integral_x[list(terms)], integral_x.weight, terms,
            nonpositive_terms=("same_type_exponential_history",), start_beta=start,
        )
        models[name] = model

    full = models["full"]
    comparisons = []
    for block, reference in (("precession", "without_precession"),
                             ("climate", "without_climate")):
        reduced = models[reference]
        removed = set(full.terms) - set(reduced.terms)
        metrics = nested_likelihood_metrics(
            loglik_full=full.log_likelihood, loglik_reduced=reduced.log_likelihood,
            df=len(removed), n_events=full.n_events,
            aic_full=full.aic, aic_reduced=reduced.aic,
        )
        comparisons.append(dict(
            added_block=block,
            conditioned_on="climate" if block == "precession" else "precession",
            reference_model=reference, n_response_events=full.n_events,
            response_exposure_kyr=float((windows.response_end_kyr_bp - windows.response_start_kyr_bp).sum()),
            loglik_full=full.log_likelihood, loglik_reference=reduced.log_likelihood,
            gain_bits_per_event=metrics["gain_bits_per_event"],
            LR_statistic=metrics["LR_statistic"], df=metrics["df"],
            nominal_LR_p=metrics["LR_p_value"],
            delta_AIC_full_minus_reference=metrics["delta_AIC_full_minus_reduced"],
        ))
    return models, pd.DataFrame(comparisons)


def run_analysis():
    lr04 = pd.read_csv(LR04_CSV, float_precision="round_trip")
    co2 = pd.read_csv(CO2_CSV, float_precision="round_trip")
    orbital = pd.read_csv(ORBITAL_CSV, float_precision="round_trip")
    anchors = pd.read_csv(PRECESSION_PHASE_CSV, float_precision="round_trip")
    forcings = {
        "lr04": (lr04.age_kyr_bp.to_numpy(), lr04.lr04.to_numpy()),
        "co2": (co2.age_kyr_bp.to_numpy(), co2.co2_ppm.to_numpy()),
        "precession_index": (orbital.age_kyr_bp.to_numpy(), orbital.precession_index.to_numpy()),
    }
    phase_anchors = (anchors.age_kyr_bp.to_numpy(), anchors.phase_unwrapped_rad.to_numpy())
    comparisons, model_rows, coefficients, metadata = [], [], [], {}
    for catalogue in CATALOGUE_LABELS:
        background = ("intercept", "same_type_exponential_history", "lr04_scaled", "co2_scaled")
        if catalogue == "primary":
            events = pd.read_csv(EVENT_CATALOGUE_CSV)
            observations = pd.read_csv(OBSERVATION_SEGMENTS_CSV)
            background = background + ("mis6_segment",)
            catalogue_id = "ngrip_warming_plus_mis6"
        else:
            definition = {"variable": "variable_threshold", "fixed": "fixed_threshold"}[catalogue]
            events = pd.read_csv(BARKER_EVENT_CSVS[definition], float_precision="round_trip")
            events["segment_id"] = "Barker2011"
            observations = pd.DataFrame([dict(segment_id="Barker2011",
                observation_start_kyr_bp=0., observation_end_kyr_bp=400.)])
            catalogue_id = f"barker_{definition}_speleo_0_400"
        windows = event_model.response_windows(events, observations)
        scaling = event_model.scale_forcing_v2({name: forcings[name] for name in ("lr04", "co2")}, windows)
        event_x, integral_x = event_model.build_likelihood_tables(events, windows, forcings, phase_anchors, scaling)
        if catalogue == "primary":
            for frame in (event_x, integral_x):
                frame["mis6_segment"] = frame.segment_id.eq("MIS6").astype(float)
        metadata[catalogue] = dict(
            catalogue_id=catalogue_id, n_source_events=len(events), history_tau_kyr=1.5,
            initial_history=0., quadrature_order=4, scaling=scaling.to_dict("index"),
            support=windows.to_dict("records"),
        )
        models, table = fit_contributions(event_x, integral_x, windows, background)
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
            pd.DataFrame(coefficients), metadata)


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


def write_notes(comparisons, metadata, data_dir, note_dir):
    """Keep the method and interpretation beside the research outputs."""
    lines = [
        "Conditional contributions of climate background and precession", "",
        "Question: How much does either predictor block improve the nominal-age fit",
        "after the other block has already been included?", "",
        "Method", "------",
        "Use the same continuous point-process likelihood as the main analyses.",
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
    interpretation = ["", "Interpretation", "--------------",
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
    lines = lines + interpretation
    note_dir.mkdir(parents=True, exist_ok=True)
    (note_dir / f"{RUN_NAME}_Methods_and_results.txt").write_text("\n".join(lines) + "\n")
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
    (note_dir / f"{RUN_NAME}_Caption.txt").write_text(caption)
    sources = [EVENT_CATALOGUE_CSV, OBSERVATION_SEGMENTS_CSV,
               *BARKER_EVENT_CSVS.values(),
               LR04_CSV, CO2_CSV, ORBITAL_CSV, PRECESSION_PHASE_CSV]
    provenance = dict(model_version=MODEL_VERSION, age_units="kyr BP1950",
                      inputs=[str(p.relative_to(PROJECT_ROOT)) for p in sources], catalogues={})
    provenance["catalogues"] = metadata
    (data_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main():
    data_dir = OUTPUT_ROOT / "data/processed" / RUN_NAME
    figure_dir = OUTPUT_ROOT / "figures" / RUN_NAME
    comparisons, models, coefficients, metadata = run_analysis()
    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    for name, table in (("comparisons", comparisons), ("models", models),
                        ("coefficients", coefficients)):
        table.to_csv(data_dir / f"{name}.csv", index=False, float_format="%.12g")
    figure = plot_contributions(comparisons)
    figure.savefig(figure_dir / f"{RUN_NAME}.pdf")
    figure.savefig(figure_dir / f"{RUN_NAME}.png", dpi=300)
    plt.close(figure)
    if EXPORT_PAPER:
        copy_pdf_to_paper(figure_dir / f"{RUN_NAME}.pdf")
    write_notes(comparisons, metadata, data_dir, generated_notes_dir(OUTPUT_ROOT))
    print(comparisons[["catalogue", "added_block", "gain_bits_per_event", "nominal_LR_p"]]
          .to_string(index=False))
    print(f"Figure: {figure_dir / (RUN_NAME + '.pdf')}")


if __name__ == "__main__":
    main()
