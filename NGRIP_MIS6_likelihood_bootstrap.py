#!/usr/bin/env python3
"""Calibrate the continuous NGRIP--MIS6 precession test under its reduced model.

Keep exact conditioning events and physical endpoints fixed. Simulate new
continuous response events with inhibitory exponential history, then refit
both models. Each replicate has one scientific event sequence; numerical
recovery never substitutes another sequence.
"""
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mcv_orb_matplotlib"))
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"
import numpy as np
import pandas as pd
from scipy.stats import chi2
from toolbox.project_config import PROJECT_ROOT, CATALOGUE_COLORS
from toolbox.project_config import generated_notes_dir

from toolbox import event_model, model_stats, sampling
from toolbox.point_process import fit_point_process
from toolbox.project_config import (MODEL_VERSION, EVENT_CATALOGUE_CSV, OBSERVATION_SEGMENTS_CSV,
    LR04_CSV, CO2_CSV, ORBITAL_CSV, PRECESSION_PHASE_CSV)

RUN_NAME = "NGRIP_MIS6_likelihood_bootstrap"
N_BOOTSTRAP = 9_999
RANDOM_SEED = 20260905
N_WORKERS = 3
OUTPUT_ROOT = PROJECT_ROOT
EXPORT_PAPER = True


HISTORY_TERM = "same_type_exponential_history"


def build_parameters(windows, reduced_terms, full_terms, *, quadrature_order=4, n_bootstrap, seed, n_workers):
    parameters = dict(model_version=MODEL_VERSION, n_bootstrap=n_bootstrap, seed=seed,
        n_workers=n_workers, history_tau_kyr=1.5, history_coefficient_domain="beta_H <= 0",
        initial_unobserved_history=0.,
        conditioning="exact oldest event per segment; all younger exposure to the fixed endpoint",
        simulation="continuous thinning with a certified background envelope",
        refit_history="rebuilt from each simulated event sequence", response_exposure_kyr=float((windows.response_end_kyr_bp-windows.response_start_kyr_bp).sum()),
        reduced_terms="+".join(reduced_terms[1:]), full_terms="+".join(full_terms[1:]),
        conditioned_response_total=False, chronology_nested=False,
        quadrature_order=quadrature_order,
        numerical_recovery="same events; one retry with doubled quadrature order; no substitute catalogue",
        zero_events="likelihood supremum 0, LR=0, G undefined",
        empirical_p_rule="(1 + simulated LR >= observed LR) / (B + 1)")
    return pd.DataFrame(parameters.items(), columns=["parameter", "value"])


def run_analysis(*, n_bootstrap=N_BOOTSTRAP, seed=RANDOM_SEED,
                 n_workers=1, show_progress=False, quadrature_order=4):
    events = pd.read_csv(EVENT_CATALOGUE_CSV)
    observations = pd.read_csv(OBSERVATION_SEGMENTS_CSV)
    catalogue_id = "ngrip_warming_plus_mis6"
    lr04 = pd.read_csv(LR04_CSV, float_precision="round_trip")
    co2 = pd.read_csv(CO2_CSV, float_precision="round_trip")
    orbital = pd.read_csv(ORBITAL_CSV, float_precision="round_trip")
    anchors = pd.read_csv(PRECESSION_PHASE_CSV, float_precision="round_trip")
    forcings = {"lr04": (lr04.age_kyr_bp.to_numpy(), lr04.lr04.to_numpy()),
                "co2": (co2.age_kyr_bp.to_numpy(), co2.co2_ppm.to_numpy()),
                "precession_index": (orbital.age_kyr_bp.to_numpy(), orbital.precession_index.to_numpy())}
    phase_anchors = (anchors.age_kyr_bp.to_numpy(), anchors.phase_unwrapped_rad.to_numpy())
    windows = event_model.response_windows(events, observations)
    scaling = event_model.nominal_scaling({name: forcings[name] for name in ("lr04", "co2")}, windows)
    event_x, integral_x = event_model.build_design(events, windows, forcings, phase_anchors, scaling,
                                                  quadrature_order=quadrature_order)
    for frame in (event_x, integral_x):
        frame["mis6_segment"] = frame.segment_id.eq("MIS6").astype(float)
    reduced_terms = ("intercept", HISTORY_TERM, "lr04_scaled", "co2_scaled", "mis6_segment")
    full_terms = reduced_terms + ("pre_phase_sin", "pre_phase_cos")
    reduced = fit_point_process(event_x[list(reduced_terms)], integral_x[list(reduced_terms)],
        integral_x.weight, reduced_terms, nonpositive_terms=(HISTORY_TERM,))
    full = fit_point_process(event_x[list(full_terms)], integral_x[list(full_terms)],
        integral_x.weight, full_terms, nonpositive_terms=(HISTORY_TERM,), start_beta=np.r_[reduced.beta, 0., 0.])
    point_summary = model_stats.fit_summary(reduced, full, event_x, windows,
        n_source_events=len(events), catalogue_id=catalogue_id)
    replicates, failures = sampling.phase_bootstrap(events, windows, forcings, phase_anchors,
        scaling, reduced, full, reduced_terms=reduced_terms, full_terms=full_terms,
        quadrature_order=quadrature_order, n_bootstrap=n_bootstrap, seed=seed,
        workers=n_workers, show_progress=show_progress)
    summary = model_stats.phase_bootstrap_summary(point_summary, replicates, failures)
    summary["seed"] = seed
    return dict(events=event_model.mark_event_roles(events, windows), windows=windows, scaling=scaling,
        reduced=reduced, full=full, event_features=event_x, integration_features=integral_x,
        forcings=forcings, phase_anchors=phase_anchors, observations=observations,
        replicates=replicates, summary=summary, rejected_reasons=failures,
        parameters=build_parameters(windows, reduced_terms, full_terms, quadrature_order=quadrature_order,
                                    n_bootstrap=n_bootstrap, seed=seed, n_workers=n_workers))


def save_tables(result, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, name in [("replicates", "bootstrap_replicates.csv"), ("summary", "summary.csv"),
                      ("parameters", "parameters_and_provenance.csv"), ("events", "event_catalogue_used.csv")]:
        result[key].to_csv(output_dir/name, index=False, float_format="%.12g")
    failed = result["replicates"].loc[lambda frame:~frame.fit_valid, ["bootstrap_id", "solver_attempts", "failure_reason"]]
    failed.to_csv(output_dir/"failed_replicates.csv",index=False)
    result["windows"].to_csv(output_dir/"support.csv",index=False,float_format="%.12g")
    result["scaling"].reset_index().to_csv(output_dir/"predictor_scaling.csv",index=False,float_format="%.12g")


def save_figure(replicates, summary, output_dir, *, paper_export=False):
    import matplotlib.pyplot as plt
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = plot_null_distribution(replicates, summary)
    png, pdf = [output_dir/f"{RUN_NAME}.{suffix}" for suffix in ("png", "pdf")]
    fig.savefig(png,dpi=450); fig.savefig(pdf)
    plt.close(fig)
    if paper_export:
        from Figure_likelihood_bootstrap import build_figure
        build_figure()
    return png,pdf


def write_notes(result, notes_dir):
    notes_dir = Path(notes_dir)
    notes_dir.mkdir(parents=True,exist_ok=True)
    s = result["summary"].iloc[0]
    caption = f"""Continuous reduced-model bootstrap calibration of the precession-phase test.
Gray bars show the LR distribution from {s.n_bootstrap} simulated catalogues; the colored line marks observed LR={s.LR_statistic:.4f}. The dashed chi-square(2) density is an asymptotic reference. Each simulation retains the exact NGRIP and MIS6 conditioning events and physical younger endpoints, generates continuous response times with dynamic exponential history, and refits both models under beta_H <= 0. The plus-one bootstrap p is {s.empirical_p_plus_one:.6g}. The binomial 95% interval [{s.empirical_p_ci95_low:.6g}, {s.empirical_p_ci95_high:.6g}] measures Monte Carlo uncertainty in the null exceedance probability, not an effect confidence interval. Chronology is fixed.
"""
    methods = f"""NGRIP--MIS6 CONTINUOUS PHASE NULL BOOTSTRAP
The reduced model includes LR04, CO2, an MIS6 segment intercept and inhibitory exponential event history (tau=1.5 kyr). Full adds phase sine and cosine. Likelihood uses actual event log intensities minus integrated intensity. Exact anchors and fixed exposure ({s.response_exposure_kyr:.6f} kyr) are retained in every replicate; no history crosses the record gap. Response event totals vary. Both models are independently refitted to every simulated sequence, with history recomputed from that sequence.

Observed: N={s.n_response_events}, G={s.gain_bits_per_event:.9f} bits/event, LR={s.LR_statistic:.9f}, Delta AIC={s.delta_AIC_full_minus_reduced:.9f}. Bootstrap repetitions={s.n_bootstrap}; exceedances={s.n_bootstrap_exceeding_or_equal_observed}; plus-one p={s.empirical_p_plus_one:.9g}. Seed={s.seed}. Failed replicates={s.n_failed_replicates}; all-empty replicates={s.n_zero_event_replicates}; same-data numerical refinements={s.n_same_data_refinements}. No failed simulation is replaced by a fresh draw. A zero-response catalogue has LR=0 and undefined G. Integration refinement does not change event ages or the simulated catalogue. A p value is withheld if any requested replicate remains unresolved.
"""
    (notes_dir/f"{RUN_NAME}_Caption.txt").write_text(caption)
    (notes_dir/f"{RUN_NAME}_Methods_and_results.txt").write_text(methods)


def configure_plot_style() -> None:
    """Use compact journal-scale typography and editable PDF fonts."""

    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9.5,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

def plot_null_distribution(
    replicates: pd.DataFrame, summary: pd.DataFrame
) -> object:
    """Plot the empirical reduced-model null and the observed LR gain."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configure_plot_style()
    row = summary.iloc[0]
    values = replicates["LR_statistic"].to_numpy(float)
    observed = float(row["LR_statistic"])
    x_min = min(0.0, float(values.min()))
    x_max = 1.06 * max(float(values.max()), observed)

    fig, axis = plt.subplots(figsize=(4.6, 3.45))
    axis.hist(
        values,
        bins=45,
        range=(x_min, x_max),
        density=True,
        color="#B8B8B8",
        edgecolor="white",
        linewidth=0.45,
        label="Reduced-model simulations",
    )
    x = np.linspace(max(0.0, x_min), x_max, 500)
    axis.plot(
        x,
        chi2.pdf(x, df=2),
        color="#555555",
        lw=1.2,
        ls=(0, (4, 2)),
        label=r"Asymptotic $\chi^2_2$",
    )
    axis.axvline(
        observed,
        color=CATALOGUE_COLORS["primary"],
        lw=1.7,
        label="Observed LR",
        zorder=4,
    )
    statistics_text = (
        f"Observed LR = {observed:.2f}\n"
        f"Bootstrap p = {row['empirical_p_plus_one']:.4g}\n"
        "95% MC interval\n"
        f"{row['empirical_p_ci95_low']:.4g}–"
        f"{row['empirical_p_ci95_high']:.4g}"
    )
    text_box = {
        "boxstyle": "round,pad=0.25",
        "facecolor": "white",
        "edgecolor": "#B8B8B8",
        "alpha": 0.92,
    }
    observed_fraction = (observed - x_min) / (x_max - x_min)
    if observed_fraction < 0.68:
        axis.text(
            0.97,
            0.94,
            statistics_text,
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            bbox=text_box,
        )
    else:
        axis.annotate(
            statistics_text,
            xy=(observed, 0.62),
            xycoords=axis.get_xaxis_transform(),
            xytext=(-8, 0),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=8.5,
            bbox=text_box,
        )
    axis.set_xlabel("Likelihood-ratio statistic")
    axis.set_ylabel("Probability density")
    axis.legend(frameon=False, loc="upper left")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(False)
    fig.tight_layout()
    return fig


def main():
    result = run_analysis(n_bootstrap=N_BOOTSTRAP, seed=RANDOM_SEED,
                          n_workers=N_WORKERS, show_progress=True)
    save_tables(result, OUTPUT_ROOT / "data/processed" / RUN_NAME)
    write_notes(result, generated_notes_dir(OUTPUT_ROOT))
    if result["summary"].iloc[0].n_failed_replicates:
        raise RuntimeError("Bootstrap has failed fits; see saved replicate results")
    save_figure(result["replicates"], result["summary"], OUTPUT_ROOT / "figures" / RUN_NAME,
                paper_export=EXPORT_PAPER and OUTPUT_ROOT.resolve() == PROJECT_ROOT.resolve())
    columns = ["LR_statistic", "empirical_p_plus_one", "n_failed_replicates"]
    print(result["summary"][columns].to_string(index=False))


if __name__ == "__main__":
    main()
